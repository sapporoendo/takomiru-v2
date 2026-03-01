from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import pillow_heif


@dataclass(frozen=True)
class CircleCandidate:
    cx: float
    cy: float
    r: float
    score: float
    method: str


@dataclass(frozen=True)
class EllipseFit:
    cx: float
    cy: float
    major: float
    minor: float
    angle_deg: float


@dataclass(frozen=True)
class CircleDetectionResult:
    cx: float
    cy: float
    r: float
    method: str
    candidates: Tuple[CircleCandidate, ...]
    ellipse_fit: Optional[EllipseFit]
    normalize_affine_2x3: np.ndarray
    debug: Dict[str, object]
    debug_images: Dict[str, np.ndarray]


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def load_image_bgr(path: Union[str, Path]) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    if ext == ".heic":
        heif = pillow_heif.read_heif(str(p))
        pil = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
        if pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        arr = np.array(pil)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        raise ValueError(f"Unsupported HEIC decoded image shape: {arr.shape}")

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image (unsupported format or decode error): {p}")
    return img


def _preprocess_for_circle(gray: np.ndarray) -> np.ndarray:
    g = gray
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    g = cv2.equalizeHist(g)
    edges = cv2.Canny(g, 80, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges


def _pick_best_circle(
    circles: Sequence[Tuple[float, float, float]],
    *,
    image_shape: Tuple[int, int],
) -> Tuple[Optional[CircleCandidate], Tuple[CircleCandidate, ...]]:
    h, w = image_shape
    if not circles:
        return None, ()

    cands = []
    for (x, y, r) in circles:
        x = float(x)
        y = float(y)
        r = float(r)
        if r <= 1.0:
            continue
        if not (0.0 <= x <= float(w) and 0.0 <= y <= float(h)):
            continue
        m = min(float(w), float(h))
        r_rel = r / max(m, 1.0)
        center_dist = np.hypot(x - w / 2.0, y - h / 2.0) / max(m, 1.0)
        score = (1.0 - abs(r_rel - 0.45)) + (1.0 - center_dist)
        cands.append(CircleCandidate(cx=x, cy=y, r=r, score=float(score), method="hough"))

    if not cands:
        return None, ()
    cands_sorted = sorted(cands, key=lambda c: c.score, reverse=True)
    return cands_sorted[0], tuple(cands_sorted)


def _radius_guard_ok(*, r: float, image_shape: Tuple[int, int], r_rel_min: float, r_rel_max: float) -> bool:
    h, w = image_shape
    m = float(min(h, w))
    if m <= 1.0:
        return False
    rr = float(r) / m
    return float(r_rel_min) <= rr <= float(r_rel_max)


def _sample_circle_mean(img_u8: np.ndarray, *, cx: float, cy: float, r: float, n: int = 360) -> float:
    h, w = img_u8.shape[:2]
    if r <= 1.0:
        return 0.0
    angles = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    xs = np.clip(np.round(cx + r * np.cos(angles)).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(cy + r * np.sin(angles)).astype(np.int32), 0, h - 1)
    return float(np.mean(img_u8[ys, xs]))


def _edge_support_score(edges_u8: np.ndarray, *, cx: float, cy: float, r: float, n: int = 720) -> Dict[str, float]:
    e = edges_u8
    if e.dtype != np.uint8:
        e = e.astype(np.uint8)
    mean = _sample_circle_mean(e, cx=float(cx), cy=float(cy), r=float(r), n=int(n))
    return {"mean": float(mean)}


def _circle_from_largest_contour(gray: np.ndarray) -> Tuple[Optional[CircleCandidate], Dict[str, object]]:
    g = gray
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Simple "bright paper" mask: avoid complex HSV logic; use robust percentile threshold.
    p = float(np.percentile(g, 70.0))
    thr = int(max(120.0, min(245.0, p + 10.0)))
    mask = (g >= thr).astype(np.uint8) * 255
    k = max(3, int(round(0.01 * float(min(g.shape[:2])))))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, {"status": "no_contour", "thr": int(thr), "k": int(k)}

    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    (x, y), r = cv2.minEnclosingCircle(cnt)
    if r <= 1.0:
        return None, {"status": "bad_r", "thr": int(thr), "k": int(k)}

    cand = CircleCandidate(cx=float(x), cy=float(y), r=float(r), score=0.0, method="contour")
    return cand, {"status": "ok", "area": float(area), "cx": float(x), "cy": float(y), "r": float(r), "thr": int(thr), "k": int(k)}


def _circle_from_white_paper_blob(bgr: np.ndarray) -> Tuple[Optional[CircleCandidate], Dict[str, object], Dict[str, np.ndarray]]:
    img = _as_bgr(bgr)
    h, w = img.shape[:2]
    dbg_images: Dict[str, np.ndarray] = {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, sch, vch = cv2.split(hsv)

    v_thr = int(np.clip(np.percentile(vch, 70.0) + 10.0, 160.0, 245.0))
    s_thr = int(np.clip(np.percentile(sch, 35.0) + 10.0, 20.0, 90.0))
    mask = (((vch >= v_thr) & (sch <= s_thr)).astype(np.uint8) * 255)

    k = max(5, int(round(0.015 * float(min(h, w)))))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

    dbg_images["paper_mask.png"] = _as_bgr(mask)
    dbg_images["paper_mask_morph.png"] = _as_bgr(mask2)

    cnts, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, {"status": "no_contour", "v_thr": int(v_thr), "s_thr": int(s_thr), "k": int(k)}, dbg_images

    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 0.0:
        return None, {"status": "bad_area", "area": float(area), "v_thr": int(v_thr), "s_thr": int(s_thr), "k": int(k)}, dbg_images

    # Infer circle from outer boundary only.
    # Use convex hull to reduce bias from the bottom cutout and small dents.
    hull = cv2.convexHull(cnt)
    (x0, y0), r0 = cv2.minEnclosingCircle(hull)
    cx = float(x0)
    cy = float(y0)
    r = float(r0)
    radius_percentile = float("nan")
    method = "paper_blob_hull_min_enclosing"

    if r <= 1.0:
        return None, {"status": "bad_r", "area": float(area), "v_thr": int(v_thr), "s_thr": int(s_thr), "k": int(k)}, dbg_images

    cand = CircleCandidate(cx=float(cx), cy=float(cy), r=float(r), score=0.0, method=str(method))
    dbg = {
        "status": "ok",
        "area": float(area),
        "cx": float(cx),
        "cy": float(cy),
        "r": float(r),
        "radius_percentile": float(radius_percentile),
        "v_thr": int(v_thr),
        "s_thr": int(s_thr),
        "k": int(k),
        "method": str(method),
    }
    return cand, dbg, dbg_images


def _circle_from_red_speed_ring_retry(bgr: np.ndarray) -> Tuple[Optional[CircleCandidate], Dict[str, object], Dict[str, np.ndarray]]:
    """Retry red ring detection without any hint/annulus restriction.

    This is used when the paper hint is unreliable (e.g., paper radius guard fails) or
    when annulus restriction leads to no candidates.
    """

    cand, dbg, imgs = _circle_from_red_speed_ring(bgr, cx_hint=None, cy_hint=None, r_hint=None)
    dbg = dict(dbg)
    dbg["retry"] = True
    return cand, dbg, imgs


def _circle_from_red_speed_ring(
    bgr: np.ndarray,
    *,
    cx_hint: Optional[float] = None,
    cy_hint: Optional[float] = None,
    r_hint: Optional[float] = None,
) -> Tuple[Optional[CircleCandidate], Dict[str, object], Dict[str, np.ndarray]]:
    img = _as_bgr(bgr)
    h, w = img.shape[:2]
    dbg_images: Dict[str, np.ndarray] = {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red in HSV wraps around 0 degrees.
    # Keep thresholds broad; morphological ops + largest contour should stabilize.
    lower1 = np.array([0, 60, 60], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 60, 60], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # If we have a paper circle hint, restrict detection to an annulus to avoid the red pointer.
    # Note: if the hint is inaccurate, this can hide the ring; caller handles retry without hint.
    ring_mask = None
    if (cx_hint is not None) and (cy_hint is not None) and (r_hint is not None) and (r_hint > 1.0):
        ring_mask = np.zeros((h, w), dtype=np.uint8)
        cxi = int(round(float(cx_hint)))
        cyi = int(round(float(cy_hint)))
        r_out = int(round(float(r_hint) * 0.97))
        r_in = int(round(float(r_hint) * 0.55))
        cv2.circle(ring_mask, (cxi, cyi), max(r_out, 1), 255, -1)
        cv2.circle(ring_mask, (cxi, cyi), max(r_in, 1), 0, -1)
        mask = cv2.bitwise_and(mask, ring_mask)

    # Keep morphology gentle; the printed red ring can be thin.
    k = max(3, int(round(0.003 * float(min(h, w)))))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_m = cv2.dilate(mask, kernel, iterations=1)
    mask_m = cv2.morphologyEx(mask_m, cv2.MORPH_CLOSE, kernel)

    dbg_images["red_mask.png"] = _as_bgr(mask)
    if ring_mask is not None:
        dbg_images["red_ring_roi.png"] = _as_bgr(ring_mask)
    dbg_images["red_mask_morph.png"] = _as_bgr(mask_m)

    cnts, _ = cv2.findContours(mask_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        # Fallback: morphology may wipe the thin ring in some lighting; try raw mask.
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, {"status": "no_contour", "k": int(k)}, dbg_images

    best_cnt = None
    best_r = -1.0
    best_area = 0.0
    r_min = None
    r_max = None
    if (r_hint is not None) and (r_hint > 1.0):
        r_min = float(r_hint) * 0.55
        r_max = float(r_hint) * 0.98

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area <= 0.0:
            continue
        (x0, y0), r0 = cv2.minEnclosingCircle(c)
        rr = float(r0)
        if rr <= 1.0:
            continue
        if (r_min is not None) and (rr < float(r_min)):
            continue
        if (r_max is not None) and (rr > float(r_max)):
            continue
        if rr > best_r:
            best_r = rr
            best_cnt = c
            best_area = area

    if best_cnt is None:
        return None, {
            "status": "no_candidate",
            "k": int(k),
            "r_min": (None if r_min is None else float(r_min)),
            "r_max": (None if r_max is None else float(r_max)),
        }, dbg_images

    area = float(best_area)

    pts = best_cnt.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] > 6000:
        step = int(max(1, pts.shape[0] // 6000))
        pts = pts[::step, :]

    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c = float(p[0]), float(p[1]), float(p[2])
    cx0 = 0.5 * a
    cy0 = 0.5 * bb
    r0_sq = c + cx0 * cx0 + cy0 * cy0
    if r0_sq <= 1.0:
        (x0, y0), r0 = cv2.minEnclosingCircle(best_cnt)
        cx = float(x0)
        cy = float(y0)
        r = float(r0)
    else:
        r0 = float(np.sqrt(r0_sq))
        d = np.sqrt((x - cx0) ** 2 + (y - cy0) ** 2)
        med = float(np.median(d))
        if med > 1.0:
            keep = np.abs(d - med) <= (0.06 * med)
            if int(np.count_nonzero(keep)) >= 30:
                x2 = x[keep]
                y2 = y[keep]
                A2 = np.column_stack([x2, y2, np.ones_like(x2)])
                b2 = x2 * x2 + y2 * y2
                p2, *_ = np.linalg.lstsq(A2, b2, rcond=None)
                a2, bb2, c2 = float(p2[0]), float(p2[1]), float(p2[2])
                cx0 = 0.5 * a2
                cy0 = 0.5 * bb2
                r0_sq = c2 + cx0 * cx0 + cy0 * cy0
                if r0_sq > 1.0:
                    r0 = float(np.sqrt(r0_sq))

        cx = float(cx0)
        cy = float(cy0)
        r = float(r0)
    if r <= 1.0:
        return None, {"status": "bad_r", "area": float(area), "k": int(k)}, dbg_images

    cand = CircleCandidate(cx=float(cx), cy=float(cy), r=float(r), score=0.0, method="red_ring_min_enclosing")
    dbg = {
        "status": "ok",
        "area": float(area),
        "cx": float(cx),
        "cy": float(cy),
        "r": float(r),
        "k": int(k),
        "method": str(cand.method),
        "hint": {
            "cx": (None if cx_hint is None else float(cx_hint)),
            "cy": (None if cy_hint is None else float(cy_hint)),
            "r": (None if r_hint is None else float(r_hint)),
        },
        "select": {
            "r_min": (None if r_min is None else float(r_min)),
            "r_max": (None if r_max is None else float(r_max)),
        },
        "hsv": {
            "lower1": [0, 60, 60],
            "upper1": [10, 255, 255],
            "lower2": [170, 60, 60],
            "upper2": [180, 255, 255],
        },
    }
    return cand, dbg, dbg_images


def _concentric_density_metrics(gray: np.ndarray, *, cx: float, cy: float, r: float) -> Dict[str, object]:
    g = gray
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g, 80, 160)

    r = float(r)
    if r <= 1.0:
        return {"status": "bad_r"}

    fracs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    means = []
    for f in fracs:
        means.append(_sample_circle_mean(edges, cx=float(cx), cy=float(cy), r=float(r) * float(f)))
    mmax = float(max(means) if means else 0.0)
    count = int(sum(1 for v in means if float(v) >= 0.60 * mmax and float(v) >= 10.0))
    score = float(sum(means)) + float(40.0 * count)
    return {
        "status": "ok",
        "fracs": list(map(float, fracs)),
        "means": list(map(float, means)),
        "max": float(mmax),
        "count_strong": int(count),
        "score": float(score),
    }


def _concentric_validation_ok(metrics: Dict[str, object]) -> bool:
    if not isinstance(metrics, dict):
        return False
    if metrics.get("status") != "ok":
        return False
    return int(metrics.get("count_strong", 0)) >= 3


def _boundary_contrast_ok(gray: np.ndarray, *, cx: float, cy: float, r: float) -> Tuple[bool, Dict[str, object]]:
    g = gray
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = g.shape[:2]
    r = float(r)
    if r <= 5.0:
        return False, {"status": "bad_r"}

    rin = max(1.0, r - 8.0)
    rout = r + 8.0
    m_in = _sample_circle_mean(g, cx=float(cx), cy=float(cy), r=float(rin), n=360)
    m_out = _sample_circle_mean(g, cx=float(cx), cy=float(cy), r=float(rout), n=360)
    diff = float(abs(m_in - m_out))
    ok = diff >= 10.0
    return bool(ok), {"status": "ok", "m_in": float(m_in), "m_out": float(m_out), "diff": float(diff), "th": 10.0}


def _refine_radius_by_radial_edge(
    gray: np.ndarray,
    *,
    cx: float,
    cy: float,
    r_min: float,
    r_max: float,
    step: float = 2.0,
    n: int = 360,
) -> Tuple[Optional[float], Dict[str, object], Optional[np.ndarray]]:
    g = gray
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = np.clip(mag, 0, 255).astype(np.uint8)

    r_min = float(r_min)
    r_max = float(r_max)
    if r_max <= r_min + 2.0:
        return None, {"status": "bad_range"}, None

    rs = np.arange(r_min, r_max + 1e-6, float(step), dtype=np.float32)
    scores = []
    for rr in rs:
        scores.append(_sample_circle_mean(mag_u8, cx=float(cx), cy=float(cy), r=float(rr), n=int(n)))

    if not scores:
        return None, {"status": "no_scores"}, None

    scores_f = np.array(scores, dtype=np.float32)
    k = 5 if scores_f.size >= 5 else max(1, int(scores_f.size))
    if k > 1:
        kernel = np.ones((k,), dtype=np.float32) / float(k)
        scores_s = np.convolve(scores_f, kernel, mode="same")
    else:
        scores_s = scores_f

    idx = int(np.argmax(scores_s))
    r_best = float(rs[idx])

    hbar = 120
    wbar = int(max(200, 4 * len(rs)))
    bar = np.zeros((hbar, wbar, 3), dtype=np.uint8)
    smin = float(scores_s.min())
    smax = float(scores_s.max())
    denom = max(1e-6, smax - smin)
    for i, s in enumerate(scores_s.tolist()):
        x = int(round((i / max(1, len(rs) - 1)) * (wbar - 1)))
        y = int(round((1.0 - ((float(s) - smin) / denom)) * (hbar - 1)))
        cv2.circle(bar, (x, y), 2, (0, 255, 0), -1)
    xbest = int(round((idx / max(1, len(rs) - 1)) * (wbar - 1)))
    cv2.line(bar, (xbest, 0), (xbest, hbar - 1), (0, 0, 255), 1)

    dbg = {
        "status": "ok",
        "r_min": float(r_min),
        "r_max": float(r_max),
        "step": float(step),
        "r_best": float(r_best),
        "score_best": float(scores_s[idx]),
    }
    return float(r_best), dbg, bar


def _hough_circle(gray: np.ndarray) -> Tuple[Optional[CircleCandidate], Tuple[CircleCandidate, ...], Dict[str, object]]:
    h, w = gray.shape[:2]
    m = min(h, w)
    edges = _preprocess_for_circle(gray)

    dp = 1.2
    min_dist = m * 0.2
    min_r = int(m * 0.25)
    max_r = int(m * 0.49)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=float(min_dist),
        param1=120,
        param2=30,
        minRadius=int(min_r),
        maxRadius=int(max_r),
    )

    raw = []
    if circles is not None and circles.size:
        raw = [tuple(map(float, c)) for c in circles[0]]

    best, cands = _pick_best_circle(raw, image_shape=(h, w))
    dbg = {
        "status": "ok" if best is not None else "no_circle",
        "hough": {
            "dp": float(dp),
            "minDist": float(min_dist),
            "param1": 120,
            "param2": 30,
            "minRadius": int(min_r),
            "maxRadius": int(max_r),
            "candidates": [list(map(float, x)) for x in raw],
        },
    }
    return best, cands, dbg


def _fit_ellipse_from_paper_mask(bgr: np.ndarray) -> Tuple[Optional[EllipseFit], float, Dict[str, object]]:
    hsv = cv2.cvtColor(_as_bgr(bgr), cv2.COLOR_BGR2HSV)
    _, sch, vch = cv2.split(hsv)
    v35 = float(np.percentile(vch, 35))
    s65 = float(np.percentile(sch, 65))
    v_min = int(max(110.0, v35))
    s_max = int(min(120.0, s65))
    mask = (vch >= v_min) & (sch <= s_max)
    mask_u8 = (mask.astype(np.uint8) * 255)

    mask_u8 = cv2.medianBlur(mask_u8, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)

    ell, score, dbg = _fit_ellipse_from_binary_mask(mask_u8)
    dbg = dict(dbg)
    dbg["mask"] = {"v_min": int(v_min), "s_max": int(s_max), "v35": float(v35), "s65": float(s65)}
    return ell, float(score), dbg


def _fit_ellipse_from_inner_mask(mask_u8: np.ndarray) -> Tuple[Optional[EllipseFit], float, Dict[str, object]]:
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    h, w = mask_u8.shape[:2]
    shrink_px = float(max(8.0, 0.02 * float(min(h, w))))

    dist = cv2.distanceTransform((mask_u8 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    inner = (dist >= float(shrink_px)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    inner = cv2.morphologyEx(inner, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, float("-inf"), {"status": "no_contours", "shrink_px": float(shrink_px)}

    c = max(cnts, key=lambda cc: float(cv2.contourArea(cc)))
    if len(c) < 20:
        return None, float("-inf"), {"status": "too_few_points", "shrink_px": float(shrink_px), "n": int(len(c))}

    try:
        (cx, cy), (ma, mi), ang = cv2.fitEllipse(c)
    except cv2.error:
        return None, float("-inf"), {"status": "fit_error", "shrink_px": float(shrink_px)}

    cx = float(cx)
    cy = float(cy)
    ma = float(max(ma, mi))
    mi = float(min(ma, mi))
    if mi <= 1.0:
        return None, float("-inf"), {"status": "degenerate", "shrink_px": float(shrink_px)}

    r_est = 0.25 * (ma + mi)
    aspect = mi / max(ma, 1e-6)
    score = float(2.0 * aspect + 0.5 * (r_est / max(float(min(h, w)), 1.0)))
    ell = EllipseFit(cx=cx, cy=cy, major=ma, minor=mi, angle_deg=float(ang))
    return ell, float(score), {
        "status": "ok",
        "shrink_px": float(shrink_px),
        "best": {
            "cx": float(cx),
            "cy": float(cy),
            "major": float(ma),
            "minor": float(mi),
            "angle_deg": float(ang),
            "score": float(score),
        },
    }


def _refine_center_by_hole(
    gray: np.ndarray,
    *,
    cx: float,
    cy: float,
    r: float,
) -> Tuple[float, float, Dict[str, object]]:
    h, w = gray.shape[:2]
    r = float(r)
    if r <= 1.0:
        return float(cx), float(cy), {"status": "skip"}

    win = int(round(max(40.0, min(float(min(w, h)), 0.65 * r))))
    x0 = int(max(0, round(cx - win)))
    y0 = int(max(0, round(cy - win)))
    x1 = int(min(w, round(cx + win)))
    y1 = int(min(h, round(cy + win)))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return float(cx), float(cy), {"status": "skip"}

    roi = gray[y0:y1, x0:x1]
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    inv = 255 - roi_blur

    min_r = int(max(6.0, 0.04 * r))
    max_r = int(max(min_r + 1, 0.12 * r))

    circles = cv2.HoughCircles(
        inv,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=float(max(10.0, 0.2 * r)),
        param1=120,
        param2=22,
        minRadius=int(min_r),
        maxRadius=int(max_r),
    )

    if circles is None or not circles.size:
        return float(cx), float(cy), {
            "status": "no_hole",
            "roi": [int(x0), int(y0), int(x1), int(y1)],
            "min_r": int(min_r),
            "max_r": int(max_r),
        }

    best = None
    best_d = 1e18
    for (x, y, rr) in circles[0]:
        xg = float(x0) + float(x)
        yg = float(y0) + float(y)
        d = float(np.hypot(xg - float(cx), yg - float(cy)))
        if d < best_d:
            best_d = d
            best = (float(xg), float(yg), float(rr))

    if best is None:
        return float(cx), float(cy), {"status": "no_hole"}

    max_shift = 0.18 * r
    if best_d > max_shift:
        return float(cx), float(cy), {
            "status": "reject_far",
            "roi": [int(x0), int(y0), int(x1), int(y1)],
            "best": [float(best[0]), float(best[1]), float(best[2])],
            "d": float(best_d),
            "max_shift": float(max_shift),
        }

    return float(best[0]), float(best[1]), {
        "status": "ok",
        "roi": [int(x0), int(y0), int(x1), int(y1)],
        "best": [float(best[0]), float(best[1]), float(best[2])],
        "d": float(best_d),
        "min_r": int(min_r),
        "max_r": int(max_r),
    }


def _fit_ellipse_from_binary_mask(mask_u8: np.ndarray) -> Tuple[Optional[EllipseFit], float, Dict[str, object]]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, float("-inf"), {"status": "no_contours"}

    h, w = mask_u8.shape[:2]
    m = float(min(h, w))
    best = None
    best_score = -1e9

    for c in cnts:
        if len(c) < 20:
            continue
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        try:
            (cx, cy), (ma, mi), ang = cv2.fitEllipse(c)
        except cv2.error:
            continue
        cx = float(cx)
        cy = float(cy)
        if not (0.0 <= cx <= float(w) and 0.0 <= cy <= float(h)):
            continue
        ma = float(max(ma, mi))
        mi = float(min(ma, mi))
        if mi <= 1.0:
            continue

        r_est = 0.25 * (ma + mi)
        r_rel = r_est / max(m, 1.0)
        aspect = mi / max(ma, 1e-6)
        center_dist = float(np.hypot(cx - w / 2.0, cy - h / 2.0) / max(m, 1.0))
        fill = float(area) / max(float(w * h), 1.0)

        score = (2.0 * (1.0 - abs(r_rel - 0.45))) + (1.5 * (2.0 * aspect)) + (1.0 - center_dist) + (2.0 * fill)
        if score > best_score:
            best_score = score
            best = EllipseFit(cx=cx, cy=cy, major=ma, minor=mi, angle_deg=float(ang))

    if best is None:
        return None, float("-inf"), {"status": "no_fit"}

    return best, float(best_score), {
        "status": "ok",
        "best": {
            "cx": float(best.cx),
            "cy": float(best.cy),
            "major": float(best.major),
            "minor": float(best.minor),
            "angle_deg": float(best.angle_deg),
            "score": float(best_score),
        },
    }


def _fit_ellipse_from_edges(gray: np.ndarray) -> Tuple[Optional[EllipseFit], float, Dict[str, object]]:
    edges = _preprocess_for_circle(gray)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, float("-inf"), {"status": "no_contours"}

    h, w = gray.shape[:2]
    m = float(min(h, w))
    best = None
    best_score = -1e9

    for c in cnts:
        if len(c) < 20:
            continue
        area = float(cv2.contourArea(c))
        if area <= 0:
            continue
        try:
            (cx, cy), (ma, mi), ang = cv2.fitEllipse(c)
        except cv2.error:
            continue
        cx = float(cx)
        cy = float(cy)
        ma = float(max(ma, mi))
        mi = float(min(ma, mi))
        if mi <= 1.0:
            continue
        r_est = 0.25 * (ma + mi)
        r_rel = r_est / max(m, 1.0)
        aspect = mi / max(ma, 1e-6)
        center_dist = float(np.hypot(cx - w / 2.0, cy - h / 2.0) / max(m, 1.0))
        score = (1.0 - abs(r_rel - 0.45)) + (2.0 * aspect) + (1.0 - center_dist)
        if score > best_score:
            best_score = score
            best = EllipseFit(cx=cx, cy=cy, major=ma, minor=mi, angle_deg=float(ang))

    if best is None:
        return None, float("-inf"), {"status": "no_fit"}
    return best, float(best_score), {
        "status": "ok",
        "best": {
            "cx": float(best.cx),
            "cy": float(best.cy),
            "major": float(best.major),
            "minor": float(best.minor),
            "angle_deg": float(best.angle_deg),
            "score": float(best_score),
        },
    }


def _affine_normalize_from_ellipse(ellipse: EllipseFit) -> np.ndarray:
    cx, cy = float(ellipse.cx), float(ellipse.cy)
    angle = float(ellipse.angle_deg)
    ma = float(ellipse.major)
    mi = float(ellipse.minor)
    s = 1.0 if ma <= 1e-6 else (ma / max(mi, 1e-6))
    m_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    a = float(m_rot[0, 0])
    b = float(m_rot[0, 1])
    c = float(m_rot[1, 0])
    d = float(m_rot[1, 1])
    tx = float(m_rot[0, 2])
    ty = float(m_rot[1, 2])
    m_scale = np.array(
        [
            [a, b * s, tx],
            [c, d * s, ty],
        ],
        dtype=np.float32,
    )
    return m_scale


def detect_tachograph_circle(
    image_bgr: np.ndarray,
    *,
    prefer_hough: bool = True,
    wobble_refine: bool = False,
) -> CircleDetectionResult:
    bgr = _as_bgr(image_bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    dbg_images: Dict[str, np.ndarray] = {}
    # Radius guard: keep it loose enough to avoid unnecessary fallbacks.
    guard = {"r_rel_min": 0.30, "r_rel_max": 0.62}
    h, w = gray.shape[:2]
    m = float(min(h, w))

    paper_cand, paper_dbg, paper_imgs = _circle_from_white_paper_blob(bgr)
    dbg_images.update(paper_imgs)

    red_cand, red_dbg, red_imgs = _circle_from_red_speed_ring(
        bgr,
        cx_hint=(None if paper_cand is None else float(paper_cand.cx)),
        cy_hint=(None if paper_cand is None else float(paper_cand.cy)),
        r_hint=(None if paper_cand is None else float(paper_cand.r)),
    )
    dbg_images.update(red_imgs)

    # If annulus constraint caused miss, retry without hint.
    if (red_cand is None) and isinstance(red_dbg, dict) and (red_dbg.get("status") in {"no_candidate", "no_contour"}):
        red2, red2_dbg, red2_imgs = _circle_from_red_speed_ring_retry(bgr)
        if red2 is not None:
            red_cand, red_dbg = red2, red2_dbg
            dbg_images.update(red2_imgs)

    best = None
    candidates: Tuple[CircleCandidate, ...] = ()
    if (red_cand is not None) and (paper_cand is not None):
        # Prefer red ring for center and paper blob for outer radius.
        merged = CircleCandidate(
            cx=float(red_cand.cx),
            cy=float(red_cand.cy),
            r=float(paper_cand.r),
            score=0.0,
            method="red_center_paper_r",
        )
        rr_ok = _radius_guard_ok(
            r=float(merged.r),
            image_shape=gray.shape[:2],
            r_rel_min=float(guard["r_rel_min"]),
            r_rel_max=float(guard["r_rel_max"]),
        )
        if rr_ok:
            best = merged
            candidates = (merged,)

    if best is None and paper_cand is not None:
        rr_ok = _radius_guard_ok(
            r=float(paper_cand.r),
            image_shape=gray.shape[:2],
            r_rel_min=float(guard["r_rel_min"]),
            r_rel_max=float(guard["r_rel_max"]),
        )
        paper_dbg = dict(paper_dbg)
        paper_dbg["rr_ok"] = bool(rr_ok)
        paper_dbg["r_rel"] = float(paper_cand.r) / float(max(1.0, m))
        # Even if guard fails, prefer paper over hard fallback; it keeps center/radius closer to reality.
        best = paper_cand
        candidates = (paper_cand,)

    if best is None and red_cand is not None:
        # If only red ring is available, expand slightly to approximate outer boundary.
        scale = 1.06
        approx = CircleCandidate(
            cx=float(red_cand.cx),
            cy=float(red_cand.cy),
            r=float(red_cand.r) * float(scale),
            score=0.0,
            method="red_ring_scaled",
        )
        rr_ok = _radius_guard_ok(
            r=float(approx.r),
            image_shape=gray.shape[:2],
            r_rel_min=float(guard["r_rel_min"]),
            r_rel_max=float(guard["r_rel_max"]),
        )
        if rr_ok:
            best = approx
            candidates = (approx,)

    if best is None:
        best = CircleCandidate(cx=float(w) / 2.0, cy=float(h) / 2.0, r=0.45 * m, score=0.0, method="fallback")
        candidates = (best,)

    wobble_dbg: Dict[str, object] = {"status": "skip"}
    if bool(wobble_refine) and str(best.method) != "fallback":
        # Optional: Subpixel-ish refinement to reduce wobble in polar bands.
        cx2, cy2, wobble_dbg = refine_center_by_polar_wobble(
            bgr,
            cx=float(best.cx),
            cy=float(best.cy),
            r=float(best.r),
        )
        best = CircleCandidate(cx=float(cx2), cy=float(cy2), r=float(best.r), score=float(best.score), method=str(best.method))

    ellipse_fit, ellipse_score, ellipse_dbg = _fit_ellipse_from_edges(gray)
    mask_ellipse_fit, mask_ellipse_score, mask_ellipse_dbg = _fit_ellipse_from_paper_mask(bgr)
    hole_dbg = _refine_center_by_hole(gray, cx=float(best.cx), cy=float(best.cy), r=float(best.r))[2]

    if ellipse_fit is not None:
        m_aff = _affine_normalize_from_ellipse(ellipse_fit)
    else:
        m_aff = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    dbg: Dict[str, object] = {
        "status": "ok",
        "paper": dict(paper_dbg),
        "red": dict(red_dbg),
        "wobble_refine": dict(wobble_dbg),
        "ellipse": dict(ellipse_dbg),
        "mask_ellipse": dict(mask_ellipse_dbg),
        "hole": dict(hole_dbg),
        "guard": dict(guard),
        "chosen": {
            "cx": float(best.cx),
            "cy": float(best.cy),
            "r": float(best.r),
            "method": str(best.method),
        },
    }

    return CircleDetectionResult(
        cx=float(best.cx),
        cy=float(best.cy),
        r=float(best.r),
        method=str(best.method),
        candidates=tuple(candidates),
        ellipse_fit=ellipse_fit,
        normalize_affine_2x3=m_aff,
        debug=dbg,
        debug_images=dbg_images,
    )


def normalize_image_by_affine(
    image_bgr: np.ndarray,
    *,
    affine_2x3: np.ndarray,
    out_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    bgr = _as_bgr(image_bgr)
    h, w = bgr.shape[:2]
    if out_size is None:
        m, (ow, oh) = _prepare_safe_affine(affine_2x3.astype(np.float32), w=w, h=h, margin_frac=0.04)
    else:
        m = affine_2x3.astype(np.float32)
        ow, oh = (int(out_size[0]), int(out_size[1]))
    return cv2.warpAffine(bgr, m, (int(ow), int(oh)), flags=cv2.INTER_LINEAR)


def _prepare_safe_affine(
    affine_2x3: np.ndarray,
    *,
    w: int,
    h: int,
    margin_frac: float,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    m = affine_2x3.astype(np.float32)
    corners = np.array(
        [[0.0, 0.0], [float(w), 0.0], [0.0, float(h)], [float(w), float(h)]],
        dtype=np.float32,
    )
    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.concatenate([corners, ones], axis=1)
    t = (m @ pts.T).T
    xs = t[:, 0]
    ys = t[:, 1]
    minx = float(xs.min())
    maxx = float(xs.max())
    miny = float(ys.min())
    maxy = float(ys.max())

    margin = float(max(w, h)) * float(max(0.0, margin_frac))
    tx = -minx + margin
    ty = -miny + margin
    m2 = m.copy()
    m2[0, 2] = float(m2[0, 2] + tx)
    m2[1, 2] = float(m2[1, 2] + ty)
    ow = int(max(1.0, np.ceil((maxx - minx) + 2.0 * margin)))
    oh = int(max(1.0, np.ceil((maxy - miny) + 2.0 * margin)))
    return m2, (ow, oh)


def unwrap_polar_linear(
    image_bgr: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    max_radius_px: float,
    out_width: int = 1440,
    out_height: Optional[int] = None,
) -> np.ndarray:
    """Unwrap full 360 degrees using OpenCV warpPolar.

    Output layout:
      - x axis: angle (0..360)
      - y axis: radius (0..max_radius_px)
    """

    bgr = _as_bgr(image_bgr)
    if out_height is None:
        out_height = int(max(1, round(float(max_radius_px))))

    dst = cv2.warpPolar(
        bgr,
        (int(out_width), int(out_height)),
        (float(center_xy[0]), float(center_xy[1])),
        float(max_radius_px),
        flags=cv2.WARP_POLAR_LINEAR,
    )
    return dst


def extract_ring_band_from_polar(
    polar_bgr: np.ndarray,
    *,
    r_in_px: float,
    r_out_px: float,
) -> np.ndarray:
    if r_out_px <= r_in_px:
        raise ValueError("r_out_px must be > r_in_px")
    h = polar_bgr.shape[0]
    y0 = int(round(max(0.0, min(float(h - 1), float(r_in_px)))))
    y1 = int(round(max(0.0, min(float(h), float(r_out_px)))))
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return polar_bgr[y0:y1, :, :]


def _horizontal_edge_score(band_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(_as_bgr(band_bgr), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Horizontal edges have mostly vertical gradient => |gy| >> |gx|
    denom = cv2.abs(gx) + cv2.abs(gy) + 1e-6
    ratio = cv2.abs(gy) / denom
    # Focus on stronger edges to avoid background noise.
    thr = float(np.percentile(mag, 92.0))
    mask = (mag >= thr).astype(np.float32)
    return float((mag * ratio * mask).sum())


def refine_center_by_polar_wobble(
    image_bgr: np.ndarray,
    *,
    cx: float,
    cy: float,
    r: float,
    speed_r_in_frac: float = 0.75,
    speed_r_out_frac: float = 0.95,
    polar_width: int = 720,
    search_radius_px: float = 3.0,
    step_px: float = 0.5,
    downscale: float = 0.5,
) -> Tuple[float, float, Dict[str, object]]:
    """Refine center by maximizing 'horizontalness' of edges in speed band in polar domain.

    This aims to reduce wobble in band_speed caused by slight center errors.
    """

    if r <= 1.0:
        return float(cx), float(cy), {"status": "bad_r"}

    img = _as_bgr(image_bgr)
    h, w = img.shape[:2]
    ds = float(np.clip(float(downscale), 0.2, 1.0))
    if ds < 1.0:
        img = cv2.resize(img, (int(round(w * ds)), int(round(h * ds))), interpolation=cv2.INTER_AREA)
    cx0 = float(cx) * ds
    cy0 = float(cy) * ds
    r0 = float(r) * ds

    best = (float(cx0), float(cy0))
    best_score = float("-inf")

    offsets = np.arange(-float(search_radius_px), float(search_radius_px) + 1e-9, float(step_px), dtype=np.float32)
    for dx in offsets:
        for dy in offsets:
            cxx = float(cx0 + float(dx))
            cyy = float(cy0 + float(dy))
            polar = unwrap_polar_linear(img, center_xy=(cxx, cyy), max_radius_px=float(r0), out_width=int(polar_width))
            r_in = float(r0) * float(speed_r_in_frac)
            r_out = float(r0) * float(speed_r_out_frac)
            band = extract_ring_band_from_polar(polar, r_in_px=r_in, r_out_px=r_out)
            s = _horizontal_edge_score(band)
            if s > best_score:
                best_score = float(s)
                best = (float(cxx), float(cyy))

    cx1 = float(best[0]) / ds
    cy1 = float(best[1]) / ds
    dbg = {
        "status": "ok",
        "cx_in": float(cx),
        "cy_in": float(cy),
        "cx_out": float(cx1),
        "cy_out": float(cy1),
        "best_score": float(best_score),
        "params": {
            "speed_r_in_frac": float(speed_r_in_frac),
            "speed_r_out_frac": float(speed_r_out_frac),
            "polar_width": int(polar_width),
            "search_radius_px": float(search_radius_px),
            "step_px": float(step_px),
            "downscale": float(ds),
        },
    }
    return float(cx1), float(cy1), dbg


@dataclass(frozen=True)
class RingUnwrapResult:
    normalized_bgr: np.ndarray
    polar_bgr: np.ndarray
    speed_band_bgr: np.ndarray
    distance_band_bgr: np.ndarray
    params: Dict[str, object]


def unwrap_speed_and_distance_rings(
    image_bgr: np.ndarray,
    *,
    circle: CircleDetectionResult,
    normalize: bool,
    polar_width: int,
    polar_theta_start_frac: float = 0.5,
    polar_theta_end_frac: float = 1.0,
    speed_r_in_frac: float,
    speed_r_out_frac: float,
    distance_r_in_frac: float,
    distance_r_out_frac: float,
) -> RingUnwrapResult:
    """Prototype ring unwrapping for both speed and distance bands.

    Fractions are relative to detected circle radius `r`.
    Defaults are intentionally conservative and may be tuned per sheet type.
    """

    bgr = _as_bgr(image_bgr)

    if normalize:
        m0 = circle.normalize_affine_2x3.astype(np.float32)
        h0, w0 = bgr.shape[:2]
        m, (ow, oh) = _prepare_safe_affine(m0, w=int(w0), h=int(h0), margin_frac=0.04)
        norm = cv2.warpAffine(bgr, m, (int(ow), int(oh)), flags=cv2.INTER_LINEAR)

        cx0 = float(circle.cx)
        cy0 = float(circle.cy)
        r0 = float(circle.r)

        def _tf(x: float, y: float) -> Tuple[float, float]:
            xx = float(m[0, 0] * x + m[0, 1] * y + m[0, 2])
            yy = float(m[1, 0] * x + m[1, 1] * y + m[1, 2])
            return xx, yy

        cx1, cy1 = _tf(cx0, cy0)
        px, py = _tf(cx0 + r0, cy0)
        qx, qy = _tf(cx0, cy0 + r0)
        r1x = float(np.hypot(px - cx1, py - cy1))
        r1y = float(np.hypot(qx - cx1, qy - cy1))

        center = (float(cx1), float(cy1))
        max_r = float(0.5 * (r1x + r1y))
    else:
        norm = bgr
        center = (float(circle.cx), float(circle.cy))
        max_r = float(circle.r)

    polar_full = unwrap_polar_linear(norm, center_xy=(float(center[0]), float(center[1])), max_radius_px=float(max_r), out_width=int(polar_width))

    w = int(polar_full.shape[1])
    a0 = float(np.clip(float(polar_theta_start_frac), 0.0, 1.0))
    a1 = float(np.clip(float(polar_theta_end_frac), 0.0, 1.0))
    if a1 <= a0:
        a0, a1 = 0.0, 1.0
    x0 = int(round(a0 * float(w)))
    x1 = int(round(a1 * float(w)))
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    polar_full = polar_full[:, x0:x1].copy()

    r_speed_in = float(max_r) * float(speed_r_in_frac)
    r_speed_out = float(max_r) * float(speed_r_out_frac)
    r_dist_in = float(max_r) * float(distance_r_in_frac)
    r_dist_out = float(max_r) * float(distance_r_out_frac)

    band_speed = extract_ring_band_from_polar(polar_full, r_in_px=float(r_speed_in), r_out_px=float(r_speed_out))
    band_dist = extract_ring_band_from_polar(polar_full, r_in_px=float(r_dist_in), r_out_px=float(r_dist_out))

    polar = polar_full

    return RingUnwrapResult(
        normalized_bgr=norm,
        polar_bgr=polar,
        speed_band_bgr=band_speed,
        distance_band_bgr=band_dist,
        params={
            "normalize": bool(normalize),
            "polar_width": int(polar_width),
            "polar_theta_start_frac": float(a0),
            "polar_theta_end_frac": float(a1),
            "max_radius_px": float(max_r),
            "speed": {
                "r_in_frac": float(speed_r_in_frac),
                "r_out_frac": float(speed_r_out_frac),
                "r_in_px": float(r_speed_in),
                "r_out_px": float(r_speed_out),
            },
            "distance": {
                "r_in_frac": float(distance_r_in_frac),
                "r_out_frac": float(distance_r_out_frac),
                "r_in_px": float(r_dist_in),
                "r_out_px": float(r_dist_out),
            },
            "polar_mode": "full",
        },
    )
