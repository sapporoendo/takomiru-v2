from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CenterEstimation:
    center_x: float
    center_y: float
    outer_radius: Optional[float]
    inner_radius: Optional[float]
    center_uncertain: bool
    debug: Dict[str, object]


def _hough_circles(gray_u8: np.ndarray, *, min_radius: int, max_radius: int, dp: float = 1.2) -> Optional[np.ndarray]:
    blurred = cv2.GaussianBlur(gray_u8, (0, 0), 2.0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=max(10, min(gray_u8.shape[:2]) // 8),
        param1=120,
        param2=30,
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if circles is None:
        return None
    circles = circles[0]
    if circles.size == 0:
        return None
    return circles


def _pick_circle_near_center(circles: np.ndarray, shape_hw: Tuple[int, int]) -> Tuple[float, float, float]:
    h, w = shape_hw
    cx0, cy0 = w / 2.0, h / 2.0
    d2 = (circles[:, 0] - cx0) ** 2 + (circles[:, 1] - cy0) ** 2
    i = int(np.argmin(d2))
    x, y, r = circles[i]
    return float(x), float(y), float(r)


def _largest_dark_blob_center(gray_u8: np.ndarray) -> Optional[Tuple[float, float, float]]:
    blur = cv2.GaussianBlur(gray_u8, (0, 0), 2.0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    h, w = gray_u8.shape[:2]
    min_area = (min(h, w) * 0.01) ** 2

    best = None
    best_score = -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue
        circ_area = np.pi * (r**2)
        fill = float(area) / float(circ_area) if circ_area > 0 else 0.0
        score = area * fill
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(r))

    return best


def estimate_center(gray_u8: np.ndarray) -> CenterEstimation:
    h, w = gray_u8.shape[:2]
    min_dim = min(h, w)

    outer_min_r = int(min_dim * 0.30)
    outer_max_r = int(min_dim * 0.60)

    inner_min_r = int(min_dim * 0.02)
    inner_max_r = int(min_dim * 0.12)

    debug: Dict[str, object] = {
        "shape": [int(h), int(w)],
        "outer_radius_range": [outer_min_r, outer_max_r],
        "inner_radius_range": [inner_min_r, inner_max_r],
    }

    outer_circles = _hough_circles(gray_u8, min_radius=outer_min_r, max_radius=outer_max_r, dp=1.2)
    outer = None
    if outer_circles is not None:
        outer = _pick_circle_near_center(outer_circles, (h, w))
        debug["outer_hough_count"] = int(outer_circles.shape[0])
    else:
        debug["outer_hough_count"] = 0

    inv = 255 - gray_u8
    inner_circles = _hough_circles(inv, min_radius=inner_min_r, max_radius=inner_max_r, dp=1.2)
    inner = None
    if inner_circles is not None:
        inner = _pick_circle_near_center(inner_circles, (h, w))
        debug["inner_hough_count"] = int(inner_circles.shape[0])
    else:
        debug["inner_hough_count"] = 0

    if inner is None:
        inner = _largest_dark_blob_center(gray_u8)
        debug["inner_blob_fallback"] = inner is not None

    candidates = []
    if outer is not None:
        candidates.append((outer[0], outer[1]))
    if inner is not None:
        candidates.append((inner[0], inner[1]))

    if not candidates:
        return CenterEstimation(
            center_x=w / 2.0,
            center_y=h / 2.0,
            outer_radius=None,
            inner_radius=None,
            center_uncertain=True,
            debug={**debug, "reason": "no_circle_detected"},
        )

    if len(candidates) == 1:
        cx, cy = candidates[0]
        return CenterEstimation(
            center_x=cx,
            center_y=cy,
            outer_radius=outer[2] if outer is not None else None,
            inner_radius=inner[2] if inner is not None else None,
            center_uncertain=True,
            debug={**debug, "reason": "single_candidate"},
        )

    (ox, oy), (ix, iy) = candidates[0], candidates[1]
    dx = ox - ix
    dy = oy - iy
    dist = float((dx * dx + dy * dy) ** 0.5)

    cx = (ox + ix) / 2.0
    cy = (oy + iy) / 2.0

    uncertain = dist > 10.0
    debug["outer_center"] = [float(ox), float(oy)]
    debug["inner_center"] = [float(ix), float(iy)]
    debug["outer_inner_dist"] = dist

    return CenterEstimation(
        center_x=cx,
        center_y=cy,
        outer_radius=outer[2] if outer is not None else None,
        inner_radius=inner[2] if inner is not None else None,
        center_uncertain=uncertain,
        debug=debug,
    )
