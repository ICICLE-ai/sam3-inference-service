"""
Integration-style tests for inference helpers.
Models are mocked so no GPU is required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


def _make_rgb_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def _batch_feature(**kwargs):
    """Minimal stand-in for HuggingFace BatchFeature (a dict subclass with .to())."""
    class BF(dict):
        def to(self, device):
            return self
    return BF(kwargs)


# --- run_text_inference_on_image ---

def test_run_text_inference_returns_empty_when_no_masks():
    mock_outputs = MagicMock()
    mock_results = {"masks": [], "boxes": [], "scores": []}

    mock_processor = MagicMock()
    mock_processor.return_value = _batch_feature(original_sizes=torch.tensor([[64, 64]]))
    mock_processor.post_process_instance_segmentation.return_value = [mock_results]

    mock_model = MagicMock()
    mock_model.return_value = mock_outputs

    import sam3.ml_models as _models
    _models.sam3_model = mock_model
    _models.sam3_processor = mock_processor

    from sam3.inference import run_text_inference_on_image
    result = run_text_inference_on_image(_make_rgb_image(), "cat")
    assert result == []


def test_run_text_inference_parses_single_detection():
    mask_np = np.zeros((64, 64), dtype=np.uint8)
    mask_np[10:30, 10:30] = 1
    mock_mask = torch.tensor(mask_np)

    mock_results = {
        "masks": [mock_mask],
        "boxes": [torch.tensor([10.0, 10.0, 30.0, 30.0])],
        "scores": [torch.tensor(0.85)],
    }

    mock_processor = MagicMock()
    mock_processor.return_value = _batch_feature(original_sizes=torch.tensor([[64, 64]]))
    mock_processor.post_process_instance_segmentation.return_value = [mock_results]

    mock_model = MagicMock()

    import sam3.ml_models as _models
    _models.sam3_model = mock_model
    _models.sam3_processor = mock_processor

    from sam3.inference import run_text_inference_on_image
    result = run_text_inference_on_image(_make_rgb_image(), "cat")

    assert len(result) == 1
    x1, y1, x2, y2, score, seg_pts = result[0]
    assert x1 == 10 and y1 == 10 and x2 == 30 and y2 == 30
    assert pytest.approx(score, abs=1e-3) == 0.85


# --- run_point_inference_on_image ---

def test_run_point_inference_returns_none_for_empty_mask():
    # pred_masks shape expected by the model: (batch, num_masks, num_predictions, H, W)
    pred_mask = torch.zeros((1, 1, 3, 64, 64), dtype=torch.bool)
    # post_process_masks returns [[tensor(num_predictions, H, W)]] per batch item
    processed_mask = torch.zeros((3, 64, 64), dtype=torch.bool)

    mock_outputs = MagicMock()
    mock_outputs.pred_masks = pred_mask
    mock_outputs.iou_scores = torch.zeros((1, 1, 3))

    mock_processor = MagicMock()
    mock_processor.return_value = _batch_feature(
        original_sizes=torch.tensor([[64, 64]]),
        pixel_values=torch.zeros((1, 3, 64, 64)),
    )
    mock_processor.post_process_masks.return_value = [[processed_mask]]

    mock_model = MagicMock()
    mock_model.return_value = mock_outputs
    mock_model.get_image_embeddings.return_value = torch.zeros((1, 256, 16, 16))

    import sam3.ml_models as _models
    _models.sam3_tracker_model = mock_model
    _models.sam3_tracker_processor = mock_processor

    from sam3.inference import run_point_inference_on_image
    result = run_point_inference_on_image(_make_rgb_image(), x=32, y=32)
    assert result is None
