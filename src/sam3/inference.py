import pickle

import numpy as np
import torch
from PIL import Image

import sam3.ml_models as _models
from sam3.config import DEVICE, REDIS_EMBEDDING_KEY_PREFIX, REDIS_TTL
from sam3.image_utils import mask_to_boundary_points
from sam3.redis_client import r


async def encode_and_store_tracker_embeddings(
    image_id: str, pipe_id: str, raw_image: Image.Image
) -> None:
    """Computes and caches tracker embeddings for point-based segmentation."""
    inputs = _models.sam3_tracker_processor(
        images=raw_image, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        embeddings = _models.sam3_tracker_model.get_image_embeddings(
            inputs["pixel_values"]
        )
        if isinstance(embeddings, list):
            bf16_embeddings = [e.to(device="cpu", dtype=torch.bfloat16) for e in embeddings]
        else:
            bf16_embeddings = embeddings.to(device="cpu", dtype=torch.bfloat16)

    data = {
        "embeddings": bf16_embeddings,
        "original_sizes": (raw_image.height, raw_image.width),
    }
    key = f"{REDIS_EMBEDDING_KEY_PREFIX}:{pipe_id}_{image_id}"
    r.set(key, pickle.dumps(data), ex=REDIS_TTL)


def run_text_inference_on_image(
    raw_image: Image.Image,
    prompt: str,
    threshold: float = 0.1,
    mask_threshold: float = 0.1,
) -> list[tuple[int, int, int, int, float, list[list[int]]]]:
    """Runs SAM3 concept model on one image and one prompt.

    Returns list of (x1, y1, x2, y2, score, seg_points).
    """
    inputs = _models.sam3_processor(
        images=raw_image,
        text=prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _models.sam3_model(**inputs)

    target_sizes = inputs.get("original_sizes").tolist()
    results = _models.sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]

    out = []
    for i in range(len(results["masks"])):
        box = results["boxes"][i]
        score = float(results["scores"][i].item())
        x_min, y_min, x_max, y_max = box.tolist()
        mask = results["masks"][i]
        mask_np = (
            mask.cpu().numpy() > 0 if hasattr(mask, "numpy") else np.array(mask) > 0
        )
        out.append(
            (int(x_min), int(y_min), int(x_max), int(y_max), score, mask_to_boundary_points(mask_np))
        )

    del inputs, outputs
    torch.cuda.empty_cache()
    return out


def run_box_exemplar_inference_on_image(
    raw_image: Image.Image,
    exemplar_box: tuple[int, int, int, int],
    threshold: float = 0.1,
    mask_threshold: float = 0.1,
) -> list[tuple[int, int, int, int, float, list[list[int]]]]:
    """Uses a bounding-box crop as a visual exemplar to find all similar objects.

    The exemplar_box region is cropped from raw_image and passed to the concept
    model as a visual prompt (instead of text) so it can detect every instance
    of the same object class across the image.

    Returns list of (x1, y1, x2, y2, score, seg_points).
    """
    ex1, ey1, ex2, ey2 = exemplar_box
    exemplar_crop = raw_image.crop((ex1, ey1, ex2, ey2))

    inputs = _models.sam3_processor(
        images=raw_image,
        exemplar_images=[exemplar_crop],
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _models.sam3_model(**inputs)

    target_sizes = inputs.get("original_sizes").tolist()
    results = _models.sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]

    out = []
    for i in range(len(results["masks"])):
        box = results["boxes"][i]
        score = float(results["scores"][i].item())
        x_min, y_min, x_max, y_max = box.tolist()
        mask = results["masks"][i]
        mask_np = (
            mask.cpu().numpy() > 0 if hasattr(mask, "numpy") else np.array(mask) > 0
        )
        out.append(
            (int(x_min), int(y_min), int(x_max), int(y_max), score, mask_to_boundary_points(mask_np))
        )

    del inputs, outputs
    torch.cuda.empty_cache()
    return out


def run_point_inference_on_image(
    raw_image: Image.Image, x: int, y: int
) -> tuple[int, int, int, int, float, list[list[int]]] | None:
    """Runs tracker model on one image and one point.

    Returns (x1, y1, x2, y2, score, seg_points) or None if no mask found.
    """
    image_inputs = _models.sam3_tracker_processor(
        images=raw_image, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        image_embeddings = _models.sam3_tracker_model.get_image_embeddings(
            image_inputs["pixel_values"]
        )

    prompt_inputs = _models.sam3_tracker_processor(
        input_points=[[[[x, y]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
        original_sizes=[(raw_image.height, raw_image.width)],
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _models.sam3_tracker_model(
            **prompt_inputs, image_embeddings=image_embeddings
        )

    masks = _models.sam3_tracker_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        prompt_inputs["original_sizes"].cpu(),
    )[0]

    scores = outputs.iou_scores.cpu()[0, 0]
    best_idx = int(torch.argmax(scores).item())
    best_mask = masks[0][best_idx].numpy() > 0

    y_indices, x_indices = np.where(best_mask)
    if len(x_indices) == 0:
        return None

    return (
        int(np.min(x_indices)),
        int(np.min(y_indices)),
        int(np.max(x_indices)),
        int(np.max(y_indices)),
        float(scores[best_idx]),
        mask_to_boundary_points(best_mask),
    )
