import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from sam3.image_utils import (
    build_overlapping_tiles,
    get_center_crop_box,
    iou_xyxy,
    mask_to_boundary_points,
    nms_bbox_candidates,
)


# --- get_center_crop_box ---

def test_center_crop_exact_fit():
    assert get_center_crop_box(100, 100, 50, 50, 100) == (0, 0, 100, 100)


def test_center_crop_clamps_to_bounds():
    x1, y1, x2, y2 = get_center_crop_box(100, 100, 5, 5, 50)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 100 and y2 <= 100
    assert x2 - x1 == 50 and y2 - y1 == 50


def test_center_crop_near_right_edge():
    x1, y1, x2, y2 = get_center_crop_box(100, 100, 95, 50, 50)
    assert x2 == 100
    assert x2 - x1 == 50


# --- build_overlapping_tiles ---

def test_tiles_cover_full_image():
    w, h, tile, overlap = 200, 200, 100, 0.2
    tiles = build_overlapping_tiles(w, h, tile, overlap)
    covered = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in tiles:
        covered[y1:y2, x1:x2] = True
    assert covered.all(), "Some pixels not covered by any tile"


def test_single_tile_for_small_image():
    tiles = build_overlapping_tiles(50, 50, 100, 0.2)
    assert len(tiles) == 1
    assert tiles[0] == (0, 0, 50, 50)


def test_no_duplicate_tiles():
    tiles = build_overlapping_tiles(300, 300, 100, 0.2)
    assert len(tiles) == len(set(tiles))


# --- iou_xyxy ---

def test_iou_no_overlap():
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_full_overlap():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_partial_overlap():
    val = iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < val < 1.0


# --- nms_bbox_candidates ---

def _make_cand(x1, y1, x2, y2, score):
    return {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2, "confidence": score}


def test_nms_keeps_best_of_overlapping_pair():
    cands = [_make_cand(0, 0, 10, 10, 0.9), _make_cand(1, 1, 11, 11, 0.5)]
    kept = nms_bbox_candidates(cands, iou_threshold=0.3)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.9


def test_nms_keeps_non_overlapping():
    cands = [_make_cand(0, 0, 10, 10, 0.9), _make_cand(50, 50, 60, 60, 0.8)]
    kept = nms_bbox_candidates(cands, iou_threshold=0.5)
    assert len(kept) == 2


def test_nms_empty_input():
    assert nms_bbox_candidates([]) == []


# --- mask_to_boundary_points ---

def test_empty_mask_returns_empty():
    mask = np.zeros((100, 100), dtype=bool)
    assert mask_to_boundary_points(mask) == []


def test_filled_square_has_boundary_points():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    pts = mask_to_boundary_points(mask)
    assert len(pts) > 0
    for p in pts:
        assert len(p) == 2


def test_boundary_points_capped_at_max():
    mask = np.zeros((200, 200), dtype=bool)
    mask[10:190, 10:190] = True
    pts = mask_to_boundary_points(mask, max_points=64)
    assert len(pts) <= 64
