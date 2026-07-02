import pickle
import time
import traceback

import torch
from fastapi import APIRouter, Header, HTTPException

from sam3.image_utils import (
    build_overlapping_tiles,
    get_center_crop_box,
    mask_to_boundary_points,
    nms_bbox_candidates,
)
from sam3.inference import (
    encode_and_store_tracker_embeddings,
    run_box_exemplar_inference_on_image,
    run_point_inference_on_image,
    run_text_inference_on_image,
)
from sam3.models import (
    BoundingBox,
    ExemplarRequest,
    LegacyPointRequest,
    SegmentationRequest,
    SegmentationResponse,
)
from sam3.redis_client import r
from sam3.config import REDIS_EMBEDDING_KEY_PREFIX
from sam3.tapis import fetch_image_from_tapis

router = APIRouter()


@router.post("/predict", response_model=SegmentationResponse)
async def predict(
    req: SegmentationRequest,
    token: str = Header(None, alias="token"),
):
    start_time = time.time()

    has_text = req.text_prompts is not None and len(req.text_prompts) > 0
    has_point = req.x is not None and req.y is not None

    if not has_text and not has_point:
        raise HTTPException(
            status_code=400,
            detail="Either text_prompts (array) or point coordinates (x,y) must be provided",
        )

    if has_text and has_point:
        raise HTTPException(
            status_code=400,
            detail="Cannot provide both text_prompts and point coordinates in one request. Call separately.",
        )

    try:
        if has_text:
            return await _predict_with_text_batch(req, token, start_time)
        return await _predict_with_point(req, token, start_time)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


@router.post("/predict_box")
async def predict_box_legacy(
    req: LegacyPointRequest,
    token: str = Header(None, alias="token"),
):
    """Legacy endpoint for single point prompts."""
    new_req = SegmentationRequest(
        image_id=req.image_id,
        pipe_id=req.pipe_id,
        system_id=req.system_id,
        image_path=req.image_path,
        x=req.x,
        y=req.y,
        text_prompts=None,
    )
    result = await predict(new_req, token)
    return {
        "bbox": result.bboxes[0].dict() if result.bboxes else None,
        "confidence": result.bboxes[0].confidence if result.bboxes else 0,
        "model": result.model,
        "total_time_seconds": result.total_time_seconds,
    }


async def _predict_with_text_batch(
    req: SegmentationRequest, token: str, start_time: float
) -> SegmentationResponse:
    if req.text_prompts is None:
        raise HTTPException(status_code=400, detail="text_prompts is required")
    if not token:
        raise HTTPException(status_code=400, detail="Token required to fetch image")

    step_start = time.time()
    raw_image = fetch_image_from_tapis(req.system_id, req.image_path, token)
    print(f"Image Load: {time.time() - step_start:.4f}s")

    patch_size = req.get_effective_patch_size()
    if patch_size is not None:
        tile_coords = build_overlapping_tiles(
            raw_image.width, raw_image.height, patch_size, req.overlap_ratio
        )
        print(
            f"Tiled text inference: {len(tile_coords)} tiles, "
            f"patch_size={patch_size}, overlap_ratio={req.overlap_ratio}"
        )
    else:
        tile_coords = [(0, 0, raw_image.width, raw_image.height)]

    all_bboxes: list[BoundingBox] = []
    per_prompt_stats: dict = {}

    for idx, prompt in enumerate(req.text_prompts):
        prompt_start = time.time()
        print(f"Processing prompt {idx + 1}/{len(req.text_prompts)}: '{prompt}'")

        try:
            candidates: list[dict] = []
            for x1, y1, x2, y2 in tile_coords:
                tile_img = raw_image.crop((x1, y1, x2, y2))
                tile_dets = run_text_inference_on_image(
                    tile_img, prompt, req.threshold, req.mask_threshold
                )
                for bx1, by1, bx2, by2, score, seg_pts in tile_dets:
                    candidates.append(
                        {
                            "x_min": int(bx1 + x1),
                            "y_min": int(by1 + y1),
                            "x_max": int(bx2 + x1),
                            "y_max": int(by2 + y1),
                            "confidence": round(score, 4),
                            "label": prompt,
                            "prompt_index": idx,
                            "segmentation": [[px + x1, py + y1] for px, py in seg_pts],
                        }
                    )

            merged = nms_bbox_candidates(candidates, iou_threshold=0.5)
            prompt_bboxes = [BoundingBox(**b) for b in merged]
            all_bboxes.extend(prompt_bboxes)

            prompt_time = time.time() - prompt_start
            per_prompt_stats[prompt] = {
                "detections": len(prompt_bboxes),
                "raw_detections": len(candidates),
                "time_seconds": round(prompt_time, 4),
            }
            print(f"  Found {len(prompt_bboxes)} objects in {prompt_time:.4f}s")
            torch.cuda.empty_cache()

        except Exception as e:
            torch.cuda.empty_cache()
            print(f"  Error processing prompt '{prompt}': {e}")
            per_prompt_stats[prompt] = {"error": str(e), "detections": 0}

    return SegmentationResponse(
        bboxes=all_bboxes,
        prompt_type="text",
        prompt_count=len(req.text_prompts),
        total_detections=len(all_bboxes),
        model="SAM3-Concept",
        total_time_seconds=round(time.time() - start_time, 4),
        per_prompt_breakdown=per_prompt_stats,
    )


async def _predict_with_point(
    req: SegmentationRequest, token: str, start_time: float
) -> SegmentationResponse:
    patch_size = req.get_effective_patch_size()

    if patch_size is not None:
        return await _predict_point_cropped(req, token, start_time, patch_size)
    return await _predict_point_cached(req, token, start_time)


async def _predict_point_cropped(
    req: SegmentationRequest, token: str, start_time: float, patch_size: int
) -> SegmentationResponse:
    """Point inference on a center-cropped region (no embedding cache)."""
    if not token:
        raise HTTPException(
            status_code=400, detail="Token required for crop-based point inference"
        )

    raw_image = fetch_image_from_tapis(req.system_id, req.image_path, token)
    if req.x is None or req.y is None:
        raise HTTPException(status_code=400, detail="Point coordinates are required")
    if req.x < 0 or req.y < 0 or req.x >= raw_image.width or req.y >= raw_image.height:
        raise HTTPException(
            status_code=400, detail="Point coordinates are out of image bounds"
        )

    x1, y1, x2, y2 = get_center_crop_box(
        raw_image.width, raw_image.height, req.x, req.y, patch_size
    )
    cropped = raw_image.crop((x1, y1, x2, y2))
    point_result = run_point_inference_on_image(cropped, req.x - x1, req.y - y1)

    if point_result is None:
        return _empty_point_response(start_time)

    bx1, by1, bx2, by2, score, seg_pts = point_result
    bbox = BoundingBox(
        x_min=int(bx1 + x1),
        y_min=int(by1 + y1),
        x_max=int(bx2 + x1),
        y_max=int(by2 + y1),
        confidence=score,
        label=f"point({req.x},{req.y})",
        prompt_index=0,
        segmentation=[[px + x1, py + y1] for px, py in seg_pts],
    )
    return SegmentationResponse(
        bboxes=[bbox],
        prompt_type="point",
        prompt_count=1,
        total_detections=1,
        model="SAM3-Tracker",
        total_time_seconds=round(time.time() - start_time, 4),
    )


async def _predict_point_cached(
    req: SegmentationRequest, token: str, start_time: float
) -> SegmentationResponse:
    """Point inference using cached image embeddings from Redis."""
    cache_key = f"{REDIS_EMBEDDING_KEY_PREFIX}:{req.pipe_id}_{req.image_id}"
    embedding_bytes = _load_valid_embedding(cache_key)

    if embedding_bytes is None:
        if not token:
            raise HTTPException(
                status_code=400,
                detail="Image missing from cache and no Token provided",
            )
        step_start = time.time()
        raw_image = fetch_image_from_tapis(req.system_id, req.image_path, token)
        print(f"Load Image: {time.time() - step_start:.4f}s")

        await encode_and_store_tracker_embeddings(req.image_id, req.pipe_id, raw_image)
        fresh = r.get(cache_key)
        if not isinstance(fresh, bytes):
            raise HTTPException(status_code=500, detail="Failed to cache embeddings")
        embedding_bytes = fresh

    loaded_data = pickle.loads(embedding_bytes)
    loaded_embeddings = loaded_data["embeddings"]
    original_sizes = loaded_data["original_sizes"]

    from sam3 import ml_models as _models
    image_embeddings = (
        [e.to(_models.sam3_tracker_model.device) for e in loaded_embeddings]
        if isinstance(loaded_embeddings, list)
        else loaded_embeddings.to(_models.sam3_tracker_model.device)
    )

    from sam3.config import DEVICE
    inputs = _models.sam3_tracker_processor(
        input_points=[[[[req.x, req.y]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
        original_sizes=[original_sizes],
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _models.sam3_tracker_model(
            **inputs, image_embeddings=image_embeddings
        )

    masks = _models.sam3_tracker_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
    )[0]

    scores = outputs.iou_scores.cpu()[0, 0]
    best_idx = int(torch.argmax(scores).item())
    best_mask = masks[0][best_idx].numpy() > 0

    y_indices, x_indices = torch.tensor(best_mask).nonzero(as_tuple=True)
    if len(x_indices) == 0:
        return _empty_point_response(start_time)

    import numpy as np
    y_np = y_indices.numpy()
    x_np = x_indices.numpy()

    bbox = BoundingBox(
        x_min=int(np.min(x_np)),
        y_min=int(np.min(y_np)),
        x_max=int(np.max(x_np)),
        y_max=int(np.max(y_np)),
        confidence=float(scores[best_idx]),
        label=f"point({req.x},{req.y})",
        prompt_index=0,
        segmentation=mask_to_boundary_points(best_mask),
    )

    return SegmentationResponse(
        bboxes=[bbox],
        prompt_type="point",
        prompt_count=1,
        total_detections=1,
        model="SAM3-Tracker",
        total_time_seconds=round(time.time() - start_time, 4),
    )


@router.post("/predict_exemplar", response_model=SegmentationResponse)
async def predict_exemplar(
    req: ExemplarRequest,
    token: str = Header(None, alias="token"),
):
    """Detect all objects similar to the supplied exemplar bounding box."""
    start_time = time.time()

    if not token:
        raise HTTPException(status_code=400, detail="Token required to fetch image")

    try:
        raw_image = fetch_image_from_tapis(req.system_id, req.image_path, token)

        x1, y1, x2, y2 = req.exemplar_box
        if x2 > raw_image.width or y2 > raw_image.height:
            raise HTTPException(
                status_code=400,
                detail="exemplar_box extends outside image bounds",
            )

        if req.patch_size is not None:
            tile_coords = build_overlapping_tiles(
                raw_image.width, raw_image.height, req.patch_size, req.overlap_ratio
            )
            print(
                f"Tiled exemplar inference: {len(tile_coords)} tiles, "
                f"patch_size={req.patch_size}, overlap_ratio={req.overlap_ratio}"
            )
        else:
            tile_coords = [(0, 0, raw_image.width, raw_image.height)]

        candidates: list[dict] = []
        label = f"exemplar({x1},{y1},{x2},{y2})"

        for tx1, ty1, tx2, ty2 in tile_coords:
            tile_img = raw_image.crop((tx1, ty1, tx2, ty2))

            # Translate the exemplar box into tile-local coordinates and skip
            # tiles that don't contain the exemplar at all.
            local_ex1 = max(x1 - tx1, 0)
            local_ey1 = max(y1 - ty1, 0)
            local_ex2 = min(x2 - tx1, tx2 - tx1)
            local_ey2 = min(y2 - ty1, ty2 - ty1)

            if local_ex2 <= local_ex1 or local_ey2 <= local_ey1:
                # Exemplar falls outside this tile — use full exemplar crop directly
                local_exemplar_box = (x1, y1, x2, y2)
                exemplar_source = raw_image
            else:
                local_exemplar_box = (local_ex1, local_ey1, local_ex2, local_ey2)
                exemplar_source = tile_img

            tile_dets = run_box_exemplar_inference_on_image(
                tile_img,
                exemplar_box=local_exemplar_box,
                threshold=req.threshold,
                mask_threshold=req.mask_threshold,
            )
            for bx1, by1, bx2, by2, score, seg_pts in tile_dets:
                candidates.append(
                    {
                        "x_min": int(bx1 + tx1),
                        "y_min": int(by1 + ty1),
                        "x_max": int(bx2 + tx1),
                        "y_max": int(by2 + ty1),
                        "confidence": round(score, 4),
                        "label": label,
                        "prompt_index": 0,
                        "segmentation": [[px + tx1, py + ty1] for px, py in seg_pts],
                    }
                )

        merged = nms_bbox_candidates(candidates, iou_threshold=0.5)
        bboxes = [BoundingBox(**b) for b in merged]
        torch.cuda.empty_cache()

        return SegmentationResponse(
            bboxes=bboxes,
            prompt_type="exemplar",
            prompt_count=1,
            total_detections=len(bboxes),
            model="SAM3-Concept",
            total_time_seconds=round(time.time() - start_time, 4),
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


def _load_valid_embedding(cache_key: str) -> bytes | None:
    """Returns cached embedding bytes, or None if missing or corrupt."""
    raw = r.get(cache_key)
    if not isinstance(raw, bytes):
        return None
    try:
        data = pickle.loads(raw)
        if isinstance(data, dict) and "original_sizes" in data:
            return raw
    except Exception:
        pass
    return None


def _empty_point_response(start_time: float) -> SegmentationResponse:
    return SegmentationResponse(
        bboxes=[],
        prompt_type="point",
        prompt_count=0,
        total_detections=0,
        model="SAM3-Tracker",
        total_time_seconds=round(time.time() - start_time, 4),
    )
