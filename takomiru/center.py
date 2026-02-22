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


def _remove_border_touching_components(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape[:2]
    if h <= 0 or w <= 0:
        return mask_u8

    m = (mask_u8 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask_u8

    keep = np.zeros((h, w), dtype=np.uint8)
    for lab in range(1, int(n)):
        x, y, ww, hh, area = stats[lab]
        if area <= 0:
            continue
        touches = (x <= 0) or (y <= 0) or ((x + ww) >= w) or ((y + hh) >= h)
        if touches:
            continue
        keep[labels == lab] = 255

    if int(np.count_nonzero(keep)) == 0:
        return mask_u8
    return keep


def _disc_circle_contour(gray_u8: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    h, w = gray_u8.shape[:2]
    min_dim = min(h, w)
    blur = cv2.GaussianBlur(gray_u8, (0, 0), 2.0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(mask)) < 127.0:
        mask = 255 - mask

    mask_area_ratio = float(np.count_nonzero(mask)) / float(max(1, h * w))
    if mask_area_ratio > 0.60 or mask_area_ratio < 0.05:
        # Otsu sometimes collapses to almost full-frame on these scans.
        # Fall back to a percentile threshold assuming the disc is brighter than background.
        # Choose a percentile that yields a reasonable area ratio.
        mask = None
        for p in (80.0, 85.0, 90.0, 92.0, 94.0, 96.0):
            thr = float(np.percentile(blur, p))
            m = (blur >= thr).astype(np.uint8) * 255
            ar = float(np.count_nonzero(m)) / float(max(1, h * w))
            if 0.06 <= ar <= 0.35:
                mask = m
                break
        if mask is None:
            thr = float(np.percentile(blur, 92.0))
            mask = (blur >= thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _remove_border_touching_components(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = None
    best_score = -1e18

    r_min = float(min_dim) * 0.10
    r_max = float(min_dim) * 0.48

    areas = [float(cv2.contourArea(c)) for c in cnts]
    order = np.argsort(np.array(areas, dtype=np.float32))[::-1]
    max_check = min(int(len(order)), 12)
    for i in range(max_check):
        c = cnts[int(order[i])]
        area = float(cv2.contourArea(c))
        if area < (h * w) * 0.01:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(c)
        r = float(r)
        if r <= 0.0:
            continue
        if r < r_min or r > r_max:
            continue
        circ_area = float(np.pi * (r**2))
        if circ_area <= 0.0:
            continue
        fill = float(area) / float(circ_area)
        if fill < 0.10:
            continue
        if not (0.0 <= float(cx) < float(w) and 0.0 <= float(cy) < float(h)):
            continue

        score = (r * 1.0) + (fill * 900.0)
        if score > best_score:
            best_score = float(score)
            best = (float(cx), float(cy), float(r), float(fill))

    return best


def _disc_roi(gray_u8: np.ndarray, *, margin_ratio: float = 0.08) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    h, w = gray_u8.shape[:2]
    blur = cv2.GaussianBlur(gray_u8, (0, 0), 2.0)

    # Make disc area white-ish; background often dark for scans.
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(mask)) < 127.0:
        mask = 255 - mask

    mask_area_ratio = float(np.count_nonzero(mask)) / float(max(1, h * w))
    if mask_area_ratio > 0.60 or mask_area_ratio < 0.05:
        # Otsu sometimes collapses to almost full-frame on these scans.
        # Fall back to a percentile threshold assuming the disc is brighter than background.
        mask = None
        for p in (80.0, 85.0, 90.0, 92.0, 94.0, 96.0):
            thr = float(np.percentile(blur, p))
            m = (blur >= thr).astype(np.uint8) * 255
            ar = float(np.count_nonzero(m)) / float(max(1, h * w))
            if 0.06 <= ar <= 0.35:
                mask = m
                break
        if mask is None:
            thr = float(np.percentile(blur, 92.0))
            mask = (blur >= thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _remove_border_touching_components(mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < (h * w) * 0.02:
        return None

    # If the mask became almost full-frame even after fallback, contour ROI is not useful.
    if area > (h * w) * 0.90:
        return None

    (cx, cy), r = cv2.minEnclosingCircle(c)
    if r <= 0:
        return None

    m = float(margin_ratio) * float(r)
    x0 = int(max(0, np.floor(cx - r - m)))
    y0 = int(max(0, np.floor(cy - r - m)))
    x1 = int(min(w, np.ceil(cx + r + m)))
    y1 = int(min(h, np.ceil(cy + r + m)))

    if (x1 - x0) < 10 or (y1 - y0) < 10:
        return None

    roi = gray_u8[y0:y1, x0:x1]
    return roi, (x0, y0)


def _disc_roi_hough(gray_u8: np.ndarray, *, margin_ratio: float = 0.08) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    h, w = gray_u8.shape[:2]
    min_dim = min(h, w)

    blur = cv2.GaussianBlur(gray_u8, (0, 0), 2.0)
    edges = cv2.Canny(blur, 50, 150)

    min_r = int(min_dim * 0.20)
    max_r = int(min_dim * 0.48)
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, min_dim // 4),
        param1=120,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None
    circles = circles[0]
    if circles.size == 0:
        return None

    best = None
    best_score = -1e18
    max_check = min(int(circles.shape[0]), 40)
    for i in range(max_check):
        cx, cy, r = circles[i]
        edge = _circle_edge_score(gray_u8, x=float(cx), y=float(cy), r=float(r))
        # Prefer circles near the image center; avoids picking partial/false circles far away.
        dx = float(cx) - float(w) / 2.0
        dy = float(cy) - float(h) / 2.0
        dist = float(np.hypot(dx, dy))
        # Strongly prefer larger radii (disc outer boundary) and centrality.
        # Edge score is only a small tie-breaker because it can be high on non-disc structures.
        score = (float(r) * 1.0) - (dist * 0.6) + (float(edge) * 0.02)
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy), float(r))
    if best is None:
        cx, cy, r = circles[0]
    else:
        cx, cy, r = best

    m = float(margin_ratio) * float(r)
    x0 = int(max(0, np.floor(cx - r - m)))
    y0 = int(max(0, np.floor(cy - r - m)))
    x1 = int(min(w, np.ceil(cx + r + m)))
    y1 = int(min(h, np.ceil(cy + r + m)))
    if (x1 - x0) < 10 or (y1 - y0) < 10:
        return None
    roi = gray_u8[y0:y1, x0:x1]
    return roi, (x0, y0)


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


def _pick_circle_near_point(circles: np.ndarray, *, target_xy: Tuple[float, float]) -> Tuple[float, float, float]:
    tx, ty = float(target_xy[0]), float(target_xy[1])
    d2 = (circles[:, 0] - tx) ** 2 + (circles[:, 1] - ty) ** 2
    i = int(np.argmin(d2))
    x, y, r = circles[i]
    return float(x), float(y), float(r)


def _circle_edge_score(gray_u8: np.ndarray, *, x: float, y: float, r: float) -> float:
    if r <= 0:
        return -1e18
    h, w = gray_u8.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return -1e18

    g = gray_u8.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    angles = np.deg2rad(np.arange(0.0, 360.0, 2.0, dtype=np.float32))
    xs = x + np.cos(angles) * float(r)
    ys = y + np.sin(angles) * float(r)
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)

    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if int(np.sum(ok)) < int(ok.size * 0.6):
        return -1e18

    vals = mag[yi[ok], xi[ok]]
    if vals.size == 0:
        return -1e18
    return float(np.percentile(vals, 70))


def _pick_best_outer_circle(gray_u8: np.ndarray, circles: np.ndarray) -> Tuple[float, float, float]:
    if circles.size == 0:
        return 0.0, 0.0, 0.0

    rs = circles[:, 2]
    r_max = float(np.max(rs))
    if not (r_max > 0.0):
        x, y, r = circles[0]
        return float(x), float(y), float(r)

    r_keep = r_max * 0.97
    cand = circles[rs >= r_keep]
    if cand.size == 0:
        cand = circles

    best = None
    best_score = -1e18
    max_check = min(int(cand.shape[0]), 24)
    for i in range(max_check):
        x, y, r = cand[i]
        edge = _circle_edge_score(gray_u8, x=float(x), y=float(y), r=float(r))
        score = float(edge) + float(r) * 0.001
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(r))
    if best is None:
        x, y, r = cand[0]
        return float(x), float(y), float(r)
    return best


def _pick_best_inner_circle(gray_u8: np.ndarray, circles: np.ndarray) -> Tuple[float, float, float]:
    h, w = gray_u8.shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0

    best = None
    best_score = -1e18
    for i in range(int(circles.shape[0])):
        x, y, r = circles[i]
        contrast = _circle_dark_contrast(gray_u8, x=float(x), y=float(y), r=float(r))
        delta = float(contrast[2]) if contrast is not None else -1e9
        dx = float(x) - float(cx0)
        dy = float(y) - float(cy0)
        dist = float((dx * dx + dy * dy) ** 0.5)
        score = (delta * 2.0) - (dist * 0.02)
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(r))

    if best is None:
        x, y, r = circles[0]
        return float(x), float(y), float(r)
    return best


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


def _radial_outer_radius(gray_u8: np.ndarray, *, center_xy: Tuple[float, float], min_r: int, max_r: int) -> Optional[Tuple[float, float]]:
    h, w = gray_u8.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    if max_r <= min_r + 5:
        return None

    angles = np.deg2rad(np.arange(0.0, 360.0, 3.0, dtype=np.float32))
    rs = np.arange(int(min_r), int(max_r) + 1, dtype=np.float32)

    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    xs = cx + cos_a * rs[None, :]
    ys = cy + sin_a * rs[None, :]

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    if int(np.sum(valid)) < valid.size * 0.5:
        return None

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    dx = xs - x0.astype(np.float32)
    dy = ys - y0.astype(np.float32)

    g = gray_u8.astype(np.float32)
    Ia = g[y0c, x0c]
    Ib = g[y0c, x1c]
    Ic = g[y1c, x0c]
    Id = g[y1c, x1c]

    vals = (Ia * (1 - dx) * (1 - dy)) + (Ib * dx * (1 - dy)) + (Ic * (1 - dx) * dy) + (Id * dx * dy)
    vals = np.where(valid, vals, np.nan)

    prof = np.nanmedian(vals, axis=0)
    if not np.isfinite(prof).any():
        return None

    k = 7
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    sm = np.convolve(np.nan_to_num(prof, nan=float(np.nanmedian(prof))), kernel, mode="same")
    grad = np.abs(np.diff(sm))
    if grad.size == 0:
        return None
    i = int(np.nanargmax(grad))
    r = float(rs[i])
    score = float(grad[i])
    return r, score


def _circle_dark_contrast(gray_u8: np.ndarray, *, x: float, y: float, r: float) -> Optional[Tuple[float, float, float]]:
    h, w = gray_u8.shape[:2]
    cx, cy, rr = float(x), float(y), float(r)
    if rr <= 2.0:
        return None
    if not (0.0 <= cx < float(w) and 0.0 <= cy < float(h)):
        return None

    yy, xx = np.ogrid[:h, :w]
    dx = xx.astype(np.float32) - np.float32(cx)
    dy = yy.astype(np.float32) - np.float32(cy)
    d2 = dx * dx + dy * dy

    r_in = rr * 0.70
    r0 = rr * 1.15
    r1 = rr * 1.60
    if r1 <= r0 + 1.0:
        return None

    inside = d2 <= (r_in * r_in)
    ring = (d2 >= (r0 * r0)) & (d2 <= (r1 * r1))
    if int(np.sum(inside)) < 20 or int(np.sum(ring)) < 50:
        return None

    g = gray_u8.astype(np.float32)
    inside_mean = float(np.mean(g[inside]))
    ring_mean = float(np.mean(g[ring]))
    return inside_mean, ring_mean, float(ring_mean - inside_mean)


def _spindle_hole_center(gray_u8: np.ndarray, *, min_r: int, max_r: int) -> Optional[Tuple[float, float, float]]:
    h, w = gray_u8.shape[:2]
    min_dim = min(h, w)
    blur = cv2.GaussianBlur(gray_u8, (0, 0), 2.0)

    cx0, cy0 = w / 2.0, h / 2.0
    max_center_dist = float(min_dim) * 0.35
    max_area = float(min_dim * min_dim) * 0.10

    masks = []
    _, m0 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(m0)
    t = float(np.percentile(blur, 20.0))
    _, m1 = cv2.threshold(blur, max(0.0, min(255.0, t)), 255, cv2.THRESH_BINARY_INV)
    masks.append(m1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    best = None
    best_score = -1e18
    for mask in masks:
        mm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, kernel, iterations=1)
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        for c in cnts:
            area = float(cv2.contourArea(c))
            if not (0.0 < area <= max_area):
                continue
            peri = float(cv2.arcLength(c, True))
            if peri <= 0:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            r_raw = float(r)

            circularity = float(4.0 * np.pi * area / (peri * peri))
            if circularity < 0.15:
                continue

            circ_area = float(np.pi * (r_raw**2))
            fill = float(area) / float(circ_area) if circ_area > 0 else 0.0
            if fill < 0.12:
                continue

            r_eff = float(r_raw) * float(fill**0.5)
            if not (float(min_r) <= r_eff <= float(max_r)):
                continue
            dx = float(x) - float(cx0)
            dy = float(y) - float(cy0)
            dist = float((dx * dx + dy * dy) ** 0.5)
            if dist > max_center_dist:
                continue
            score = (circularity * 2.5) + (fill * 1.8) - (dist / max_center_dist) + (area / max_area)
            if score > best_score:
                best_score = score
                best = (float(x), float(y), float(r_eff))

    return best


def estimate_center(gray_u8: np.ndarray) -> CenterEstimation:
    h, w = gray_u8.shape[:2]
    min_dim = min(h, w)

    outer_min_r = int(min_dim * 0.30)
    outer_max_r = int(min_dim * 0.60)

    inner_min_r = int(min_dim * 0.02)
    inner_max_r = int(min_dim * 0.12)
    spindle_max_r = int(min_dim * 0.065)

    debug: Dict[str, object] = {
        "shape": [int(h), int(w)],
        "outer_radius_range": [outer_min_r, outer_max_r],
        "inner_radius_range": [inner_min_r, inner_max_r],
    }

    spindle = _spindle_hole_center(gray_u8, min_r=inner_min_r, max_r=spindle_max_r)
    debug["spindle_detected"] = spindle is not None
    if spindle is not None:
        debug["spindle_center"] = [float(spindle[0]), float(spindle[1])]
        debug["spindle_radius"] = float(spindle[2])

    inv = 255 - gray_u8
    inner_circles = _hough_circles(inv, min_radius=inner_min_r, max_radius=inner_max_r, dp=1.2)
    inner = None
    if inner_circles is not None:
        inner = _pick_best_inner_circle(gray_u8, inner_circles)
        debug["inner_hough_count"] = int(inner_circles.shape[0])
    else:
        debug["inner_hough_count"] = 0

    if spindle is None and inner_circles is not None:
        best = None
        best_delta = -1e18
        checked = 0
        max_check = 40
        for c in inner_circles[:max_check]:
            x, y, r = float(c[0]), float(c[1]), float(c[2])
            if not (r > 0.0 and r <= float(spindle_max_r)):
                continue
            contrast = _circle_dark_contrast(gray_u8, x=x, y=y, r=r)
            checked += 1
            if contrast is None:
                continue
            inside_mean, ring_mean, delta = contrast
            if float(delta) > float(best_delta):
                best_delta = float(delta)
                best = (x, y, r, float(inside_mean), float(ring_mean), float(delta))

        debug["spindle_from_hough_checked"] = int(checked)
        debug["spindle_from_hough_ok"] = best is not None
        if best is not None:
            x, y, r, inside_mean, ring_mean, delta = best
            debug["spindle_inside_mean"] = float(inside_mean)
            debug["spindle_ring_mean"] = float(ring_mean)
            debug["spindle_dark_delta"] = float(delta)
            debug["spindle_from_hough_best"] = [float(x), float(y), float(r), float(delta)]
            if float(delta) >= 8.0:
                spindle = (float(x), float(y), float(r))
                debug["spindle_detected"] = True
                debug["spindle_center"] = [float(spindle[0]), float(spindle[1])]
                debug["spindle_radius"] = float(spindle[2])

    if spindle is not None:
        inner = spindle
        debug["inner_source"] = "spindle"
    elif inner is not None:
        debug["inner_source"] = "hough"
    else:
        inner = _largest_dark_blob_center(gray_u8)
        debug["inner_blob_fallback"] = inner is not None
        debug["inner_source"] = "blob" if inner is not None else "none"

    outer_circles = _hough_circles(gray_u8, min_radius=outer_min_r, max_radius=outer_max_r, dp=1.2)
    outer = None
    if outer_circles is not None:
        outer = _pick_best_outer_circle(gray_u8, outer_circles)
        debug["outer_pick_target"] = "best_edge"
        debug["outer_hough_count"] = int(outer_circles.shape[0])
    else:
        debug["outer_hough_count"] = 0

    outer_radial = None
    radial_center = None
    if outer is not None:
        radial_center = (outer[0], outer[1])
        debug["outer_radial_center"] = "outer"
    elif inner is not None:
        radial_center = (inner[0], inner[1])
        debug["outer_radial_center"] = "inner"

    if radial_center is not None:
        outer_radial = _radial_outer_radius(gray_u8, center_xy=radial_center, min_r=outer_min_r, max_r=outer_max_r)
        debug["outer_radial_ok"] = outer_radial is not None
        if outer_radial is not None:
            debug["outer_radial_radius"] = float(outer_radial[0])
            debug["outer_radial_score"] = float(outer_radial[1])

    if outer is not None and outer_radial is not None:
        outer = (float(outer[0]), float(outer[1]), float(outer_radial[0]))
        debug["outer_source"] = "hough+radial"
    elif outer is not None:
        debug["outer_source"] = "hough"

    inner_radial = None
    if inner is not None:
        inner_radial = _radial_outer_radius(gray_u8, center_xy=(inner[0], inner[1]), min_r=outer_min_r, max_r=outer_max_r)
        debug["inner_radial_ok"] = inner_radial is not None
        if inner_radial is not None:
            debug["inner_radial_radius"] = float(inner_radial[0])
            debug["inner_radial_score"] = float(inner_radial[1])

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

    debug["outer_center"] = [float(ox), float(oy)]
    debug["inner_center"] = [float(ix), float(iy)]
    debug["outer_inner_dist"] = dist

    outer_score = float(debug.get("outer_radial_score", -1e18))
    inner_score = float(debug.get("inner_radial_score", -1e18))
    debug["center_pick_outer_score"] = float(outer_score)
    debug["center_pick_inner_score"] = float(inner_score)

    if outer_score >= inner_score:
        cx = float(ox)
        cy = float(oy)
        debug["center_pick"] = "outer"
    else:
        cx = float(ix)
        cy = float(iy)
        debug["center_pick"] = "inner"

    uncertain = dist > 12.0

    return CenterEstimation(
        center_x=cx,
        center_y=cy,
        outer_radius=outer[2] if outer is not None else None,
        inner_radius=inner[2] if inner is not None else None,
        center_uncertain=uncertain,
        debug=debug,
    )


def estimate_center_auto(gray_u8: np.ndarray) -> CenterEstimation:
    h, w = gray_u8.shape[:2]
    min_dim = min(h, w)

    disc = _disc_circle_contour(gray_u8)
    if disc is not None:
        cx, cy, r, fill = disc
        debug = {
            "shape": [int(h), int(w)],
            "roi_used": False,
            "roi_method": "contour_circle",
            "disc_fill": float(fill),
        }
        return CenterEstimation(
            center_x=float(cx),
            center_y=float(cy),
            outer_radius=float(r),
            inner_radius=None,
            center_uncertain=False,
            debug=debug,
        )

    est_full = estimate_center(gray_u8)
    if isinstance(est_full.debug, dict) and bool(est_full.debug.get("spindle_detected")) and est_full.debug.get("inner_source") == "spindle":
        outer_r = float(est_full.outer_radius) if est_full.outer_radius is not None else None
        outer_inner_dist = float(est_full.debug.get("outer_inner_dist")) if est_full.debug.get("outer_inner_dist") is not None else None
        outer_score = float(est_full.debug.get("outer_radial_score", -1e18))

        ok = True
        if outer_r is None or outer_r <= 0:
            ok = False
        if outer_inner_dist is None:
            ok = False
        if outer_score <= 0.0:
            ok = False
        if ok and outer_inner_dist > max(18.0, outer_r * 0.15):
            ok = False

        if ok:
            debug_full = dict(est_full.debug)
            debug_full.update({"roi_used": False, "roi_method": "full_frame_spindle"})
            return CenterEstimation(
                center_x=float(est_full.center_x),
                center_y=float(est_full.center_y),
                outer_radius=est_full.outer_radius,
                inner_radius=est_full.inner_radius,
                center_uncertain=False,
                debug=debug_full,
            )

    inner_min_r = int(min_dim * 0.02)
    inner_max_r = int(min_dim * 0.12)
    spindle = _spindle_hole_center(gray_u8, min_r=inner_min_r, max_r=inner_max_r)

    roi_method = None
    roi_res = None
    # Prefer disc contour ROI; spindle detection can be a false positive on some scans and
    # can cause ROI to be cut far away from the actual disc.
    roi_res = _disc_roi(gray_u8)
    roi_method = "contour" if roi_res is not None else roi_method
    if roi_res is None:
        roi_res = _disc_roi_hough(gray_u8)
        roi_method = "hough" if roi_res is not None else roi_method
    if roi_res is None and spindle is not None:
        cx, cy = float(spindle[0]), float(spindle[1])
        # Sanity-check spindle position; ignore if too close to borders.
        if (0.15 * w) <= cx <= (0.85 * w) and (0.15 * h) <= cy <= (0.85 * h):
            outer_max_r = int(min_dim * 0.60)
            half = float(outer_max_r) * 1.15
            x0 = int(max(0, np.floor(cx - half)))
            y0 = int(max(0, np.floor(cy - half)))
            x1 = int(min(w, np.ceil(cx + half)))
            y1 = int(min(h, np.ceil(cy + half)))
            if (x1 - x0) >= 10 and (y1 - y0) >= 10:
                roi = gray_u8[y0:y1, x0:x1]
                roi_res = (roi, (x0, y0))
                roi_method = "spindle"
    if roi_res is None:
        est = estimate_center(gray_u8)
        return CenterEstimation(
            center_x=est.center_x,
            center_y=est.center_y,
            outer_radius=est.outer_radius,
            inner_radius=est.inner_radius,
            center_uncertain=est.center_uncertain,
            debug={**(est.debug if isinstance(est.debug, dict) else {}), "roi_used": False, "roi_method": None},
        )

    roi, (x0, y0) = roi_res
    est_roi = estimate_center(roi)

    debug = dict(est_roi.debug) if isinstance(est_roi.debug, dict) else {}

    # Keep original ROI-local centers for inspection
    if "outer_center" in debug:
        debug["outer_center_roi"] = debug["outer_center"]
    if "inner_center" in debug:
        debug["inner_center_roi"] = debug["inner_center"]

    # Shift debug centers into full-frame coordinates so downstream overlays are correct.
    if "outer_center" in debug and isinstance(debug["outer_center"], (list, tuple)) and len(debug["outer_center"]) >= 2:
        debug["outer_center"] = [float(debug["outer_center"][0]) + float(x0), float(debug["outer_center"][1]) + float(y0)]
    if "inner_center" in debug and isinstance(debug["inner_center"], (list, tuple)) and len(debug["inner_center"]) >= 2:
        debug["inner_center"] = [float(debug["inner_center"][0]) + float(x0), float(debug["inner_center"][1]) + float(y0)]

    debug.update({
        "roi_used": True,
        "roi_method": roi_method,
        "roi_offset": [int(x0), int(y0)],
        "roi_shape": [int(roi.shape[0]), int(roi.shape[1])],
    })

    return CenterEstimation(
        center_x=float(est_roi.center_x) + float(x0),
        center_y=float(est_roi.center_y) + float(y0),
        outer_radius=est_roi.outer_radius,
        inner_radius=est_roi.inner_radius,
        center_uncertain=est_roi.center_uncertain,
        debug=debug,
    )
