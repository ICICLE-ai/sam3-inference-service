"""
Tests for POST /predict/upload — the image arrives as a multipart file instead
of being fetched from Tapis. Models and Redis are mocked so no GPU or Redis
server is required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import io
import json
import pickle
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import sam3.routes.predict as predict_module


def _jpeg_bytes(w: int = 64, h: int = 48) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _post(client: TestClient, request: dict, data: bytes | None = None):
    return client.post(
        "/predict/upload",
        data={"request": json.dumps(request)},
        files={"file": ("frame.jpg", data if data is not None else _jpeg_bytes(), "image/jpeg")},
    )


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(predict_module.router)
    return TestClient(app)


@pytest.fixture
def no_tapis():
    with patch.object(predict_module, "fetch_image_from_tapis") as fetch:
        fetch.side_effect = AssertionError("must not fetch from Tapis")
        yield fetch


POINT_REQ = {"image_id": "up1_frame_000012", "pipe_id": "pipe1", "x": 5, "y": 5}
CACHED = pickle.dumps({"embeddings": [], "original_sizes": (48, 64)})


@pytest.fixture
def tracker():
    """Tracker model/processor stand-ins that yield an empty mask."""
    model, proc = MagicMock(), MagicMock()
    proc.return_value.to.return_value = {"original_sizes": torch.tensor([[48, 64]])}
    proc.post_process_masks.return_value = [torch.zeros((1, 3, 48, 64))]
    model.return_value.iou_scores = torch.tensor([[[0.9, 0.1, 0.2]]])
    with (
        patch("sam3.ml_models.sam3_tracker_model", model, create=True),
        patch("sam3.ml_models.sam3_tracker_processor", proc, create=True),
    ):
        yield model


def test_point_cache_hit_discards_upload(client, no_tapis, tracker):
    """On a hit the file is never decoded — even bytes that are not an image succeed."""
    with (
        patch.object(predict_module, "r") as redis,
        patch.object(predict_module, "encode_and_store_tracker_embeddings", new=AsyncMock()) as encode,
    ):
        redis.get.return_value = CACHED
        resp = _post(client, POINT_REQ, data=b"not an image")

    assert resp.status_code == 200, resp.text
    redis.get.assert_called_once_with("sam3:emb:pipe1_up1_frame_000012")
    encode.assert_not_awaited()
    tracker.assert_called_once()


def test_point_cache_miss_embeds_uploaded_image(client, no_tapis, tracker):
    with (
        patch.object(predict_module, "r") as redis,
        patch.object(predict_module, "encode_and_store_tracker_embeddings", new=AsyncMock()) as encode,
    ):
        redis.get.side_effect = [None, CACHED]
        resp = _post(client, POINT_REQ)

    assert resp.status_code == 200, resp.text
    encode.assert_awaited_once()
    image_id, pipe_id, image = encode.await_args.args
    assert (image_id, pipe_id) == ("up1_frame_000012", "pipe1")
    assert image.size == (64, 48)


def test_text_prompt_uses_uploaded_image(client, no_tapis):
    with patch.object(predict_module, "run_text_inference_on_image", return_value=[]) as run:
        resp = _post(client, {"image_id": "f", "pipe_id": "p", "text_prompts": ["car"]})

    assert resp.status_code == 200
    assert resp.json()["prompt_type"] == "text"
    assert run.call_args.args[0].size == (64, 48)


def test_unreadable_upload_is_400(client, no_tapis):
    with patch.object(predict_module, "r") as redis:
        redis.get.return_value = None
        resp = _post(client, POINT_REQ, data=b"not an image")
    assert resp.status_code == 400
    assert "not a readable image" in resp.json()["detail"]


def test_invalid_request_json_is_422(client, no_tapis):
    resp = _post(client, {"pipe_id": "p", "x": 1, "y": 1})   # image_id missing
    assert resp.status_code == 422


def test_json_predict_without_path_is_400():
    app = FastAPI()
    app.include_router(predict_module.router)
    with patch.object(predict_module, "r") as redis:
        redis.get.return_value = None
        resp = TestClient(app).post(
            "/predict", json=POINT_REQ, headers={"token": "t"},
        )
    assert resp.status_code == 400
    assert "system_id and image_path" in resp.json()["detail"]
