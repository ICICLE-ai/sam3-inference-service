import math

import cv2
import numpy as np


def get_center_crop_box(
    img_w: int, img_h: int, center_x: int, center_y: int, patch_size: int
) -> tuple[int, int, int, int]:
    """Returns (x1, y1, x2, y2) for a center crop clamped to image bounds."""
    crop_w = min(patch_size, img_w)
    crop_h = min(patch_size, img_h)

    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(img_w, x1 + crop_w)
    y2 = min(img_h, y1 + crop_h)

    # Snap back to keep exact crop size when possible
    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)
    return int(x1), int(y1), int(x2), int(y2)


def build_overlapping_tiles(
    img_w: int, img_h: int, tile_size: int, overlap_ratio: float
) -> list[tuple[int, int, int, int]]:
    """Generates overlapping (x1, y1, x2, y2) tile coords covering the full image."""
    tile_size = max(1, int(tile_size))
    stride = max(1, int(tile_size * (1 - overlap_ratio)))

    cols = max(1, math.ceil((img_w - tile_size) / stride) + 1)
    rows = max(1, math.ceil((img_h - tile_size) / stride) + 1)

    tiles = []
    seen: set[tuple[int, int, int, int]] = set()
    for r_i in range(rows):
        for c_i in range(cols):
            x1 = c_i * stride
            y1 = r_i * stride
            x2 = min(x1 + tile_size, img_w)
            y2 = min(y1 + tile_size, img_h)

            if x2 == img_w:
                x1 = max(0, img_w - tile_size)
            if y2 == img_h:
                y1 = max(0, img_h - tile_size)

            box = (int(x1), int(y1), int(x2), int(y2))
            if box not in seen:
                seen.add(box)
                tiles.append(box)
    return tiles


def iou_xyxy(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_to_boundary_points(
    mask: np.ndarray, max_points: int = 256
) -> list[list[int]]:
    """Extracts ordered contour points from a binary mask."""
    if mask is None or not np.any(mask):
        return []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    points = contour.squeeze(axis=1)
    if len(points) > max_points:
        indices = np.round(np.linspace(0, len(points) - 1, max_points)).astype(int)
        points = points[indices]
    return [[int(p[0]), int(p[1])] for p in points]


def nms_bbox_candidates(
    candidates: list[dict], iou_threshold: float = 0.5
) -> list[dict]:
    """Non-maximum suppression over bbox dicts with x_min/y_min/x_max/y_max/confidence."""
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    kept: list[dict] = []

    for cand in sorted_cands:
        cand_box = (cand["x_min"], cand["y_min"], cand["x_max"], cand["y_max"])
        if all(
            iou_xyxy(cand_box, (k["x_min"], k["y_min"], k["x_max"], k["y_max"]))
            <= iou_threshold
            for k in kept
        ):
            kept.append(cand)
    return kept
