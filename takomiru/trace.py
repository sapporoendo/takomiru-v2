from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TracePoint:
    angle_deg: float
    radius_px: float
    ink_score: float


@dataclass(frozen=True)
class TraceResult:
    points: List[TracePoint]
    debug: Dict[str, object]


@dataclass
class SpeedScaleRadiiEstimation:
    r20: int
    r120: int
    debug: Dict[str, object]


@dataclass(frozen=True)
class ZeroAngleEstimation:
    zero_angle_deg: float
    debug: Dict[str, object]


@dataclass(frozen=True)
class SpeedBandEstimation:
    r_in: int
    r_out: int
    debug: Dict[str, object]


@dataclass(frozen=True)
class NeedleExtraction:
    needle_mask_u8: np.ndarray
    roi_mask_u8: np.ndarray
    binary_mask_u8: np.ndarray
    debug: Dict[str, object]


def fixed_speed_band_from_outer_radius(
    *,
    outer_radius: Optional[float],
    r_in_ratio: float = 0.55,
    r_out_ratio: float = 0.92,
) -> Optional[SpeedBandEstimation]:
    if outer_radius is None:
        return None
    r = float(outer_radius)
    if not (r > 0):
        return None
    r_in = int(round(float(r_in_ratio) * r))
    r_out = int(round(float(r_out_ratio) * r))
    if r_out <= r_in + 10:
        return None
    return SpeedBandEstimation(
        r_in=r_in,
        r_out=r_out,
        debug={
            "method": "fixed_outer_radius_ratio",
            "outer_radius": float(r),
            "r_in": int(r_in),
            "r_out": int(r_out),
            "r_in_ratio": float(r_in_ratio),
            "r_out_ratio": float(r_out_ratio),
        },
    )


def estimate_speed_scale_radii(
    gray_u8: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    disc_outer_radius: float,
    angle_step_deg: float = 0.5,
    canny1: int = 40,
    canny2: int = 120,
) -> SpeedScaleRadiiEstimation:
    """Estimate printed 20km/h and 120km/h concentric rings radii.

    The chart contains many concentric rings; naive peak picking is unstable.
    We compute radial edge coverage and select a (r20, r120) pair jointly with
    constraints on absolute radius ratios and expected gap.
    """

    if gray_u8.ndim != 2 or gray_u8.dtype != np.uint8:
        raise ValueError("gray_u8 must be a 2D uint8 image")
    R = float(disc_outer_radius)
    if not (R > 0):
        raise ValueError("disc_outer_radius must be > 0")

    blur = cv2.GaussianBlur(gray_u8, (0, 0), 1.2)
    ed = cv2.Canny(blur, int(canny1), int(canny2))

    h, w = gray_u8.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])

    # Search where speed scale ring is expected.
    r0 = int(round(R * 0.55))
    r1 = int(round(R * 0.95))
    rs = np.arange(r0, r1 + 1, dtype=np.float32)

    angles = np.deg2rad(np.arange(0.0, 360.0, float(angle_step_deg), dtype=np.float32))
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    dens = np.zeros((len(rs),), dtype=np.float32)
    for i, r in enumerate(rs):
        xs = np.rint(cx + float(r) * cos_t).astype(np.int32)
        ys = np.rint(cy + float(r) * sin_t).astype(np.int32)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(ok):
            dens[i] = 0.0
        else:
            dens[i] = float(ed[ys[ok], xs[ok]].mean()) / 255.0

    # Smooth and find local maxima.
    k = 21
    kernel = np.ones((k,), np.float32) / float(k)
    dens_s = np.convolve(dens, kernel, mode="same")

    peaks: List[int] = []
    for i in range(2, len(dens_s) - 2):
        if dens_s[i] > dens_s[i - 1] and dens_s[i] > dens_s[i + 1]:
            peaks.append(i)

    win = 25
    cand: List[Dict[str, float]] = []
    for i in peaks:
        a = max(0, i - win)
        b = min(len(dens_s), i + win + 1)
        base = float(np.min(dens_s[a:b]))
        prom = float(dens_s[i] - base)
        score = float(dens_s[i]) * (0.3 + prom)
        r = int(round(float(rs[int(i)])))
        cand.append(
            {
                "r": float(r),
                "ratio": float(r / max(1e-6, R)),
                "dens": float(dens_s[i]),
                "score": float(score),
            }
        )

    cand.sort(key=lambda c: float(c["score"]), reverse=True)

    best: Optional[Tuple[int, int]] = None
    best_score = -1e18

    # Joint selection constraints.
    # Empirically, on common 0-140 tachograph sheets the 20km/h ring sits fairly
    # outward (often overlapping inner time tick ring), and 120km/h is close to
    # the outer speed scale.
    for c20 in cand[:80]:
        ratio20 = float(c20["ratio"])
        if not (0.78 <= ratio20 <= 0.88):
            continue
        r20 = float(c20["r"])
        for c120 in cand[:80]:
            ratio120 = float(c120["ratio"])
            if not (0.88 <= ratio120 <= 0.95):
                continue
            r120 = float(c120["r"])
            if not (r120 > r20 + 5):
                continue

            gap = float((r120 - r20) / max(1e-6, R))
            if not (0.07 <= gap <= 0.18):
                continue

            # Preferences (soft): shift slightly outward
            pref_r20 = 1.0 - min(1.0, abs(ratio20 - 0.84) / 0.06)
            pref_outer = 1.0 - min(1.0, abs(ratio120 - 0.92) / 0.05)
            pref_gap = 1.0 - min(1.0, abs(gap - 0.10) / 0.05)
            sc = float(c20["score"]) + float(c120["score"]) + 0.10 * pref_r20 + 0.10 * pref_outer + 0.08 * pref_gap
            if sc > best_score:
                best_score = sc
                best = (int(round(r20)), int(round(r120)))

    if best is None:
        # Fallback to fixed ratios close to our previous heuristic.
        best = (int(round(R * 0.80)), int(round(R * 0.90)))

    r20_i, r120_i = best
    debug: Dict[str, object] = {
        "method": "radial_edge_coverage_pair",
        "disc_outer_radius": float(R),
        "search_r_min": int(r0),
        "search_r_max": int(r1),
        "angle_step_deg": float(angle_step_deg),
        "canny1": int(canny1),
        "canny2": int(canny2),
        "r20": int(r20_i),
        "r120": int(r120_i),
        "r20_ratio": float(r20_i / max(1e-6, R)),
        "r120_ratio": float(r120_i / max(1e-6, R)),
        "gap_ratio": float((r120_i - r20_i) / max(1e-6, R)),
        "top_candidates": cand[:30],
    }

    return SpeedScaleRadiiEstimation(r20=int(r20_i), r120=int(r120_i), debug=debug)


def _annulus_mask(shape_hw: Tuple[int, int], *, center_xy: Tuple[float, float], r_in: int, r_out: int) -> np.ndarray:
    h, w = shape_hw
    cx, cy = float(center_xy[0]), float(center_xy[1])
    yy, xx = np.ogrid[:h, :w]
    rr = (xx - cx) ** 2 + (yy - cy) ** 2
    return ((rr >= float(r_in * r_in)) & (rr <= float(r_out * r_out))).astype(np.uint8) * 255


def _sector_mask(
    shape_hw: Tuple[int, int],
    *,
    center_xy: Tuple[float, float],
    center_angle_deg_img: float,
    half_width_deg: float,
) -> np.ndarray:
    h, w = shape_hw
    cx, cy = float(center_xy[0]), float(center_xy[1])
    yy, xx = np.ogrid[:h, :w]
    ang = (np.rad2deg(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0

    c = float(center_angle_deg_img) % 360.0
    hw = max(0.0, float(half_width_deg))

    diff = ((ang - c + 540.0) % 360.0) - 180.0
    return (np.abs(diff) <= hw).astype(np.uint8) * 255


def extract_needle_mask(
    gray_u8: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    r_in: int,
    r_out: int,
    sector_center_angle_deg_img: Optional[float] = None,
    sector_half_width_deg: Optional[float] = None,
    threshold_offset: int = 15,
    min_area: int = 120,
    min_aspect: float = 3.0,
) -> NeedleExtraction:
    if gray_u8.ndim != 2 or gray_u8.dtype != np.uint8:
        raise ValueError("gray_u8 must be a 2D uint8 image")
    if r_in <= 0 or r_out <= r_in:
        raise ValueError("invalid r_in/r_out")

    h, w = gray_u8.shape[:2]
    ann = _annulus_mask((h, w), center_xy=center_xy, r_in=int(r_in), r_out=int(r_out))

    roi_mask = ann

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray_u8)

    roi = cv2.bitwise_and(eq, eq, mask=roi_mask)
    roi_blur = cv2.GaussianBlur(roi, (0, 0), 1.2)

    otsu_thr, _ = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    strict_thr = max(0, int(round(float(otsu_thr))) - int(threshold_offset))
    binary = (roi_blur < strict_thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.bitwise_and(binary, binary, mask=roi_mask)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cx, cy = float(center_xy[0]), float(center_xy[1])
    best_id = None
    best_score = -1e18
    for k in range(1, int(num)):
        x, y, ww, hh, area = [int(v) for v in stats[k].tolist()]
        if area < int(min_area):
            continue
        aspect = float(max(ww, hh)) / float(max(1, min(ww, hh)))
        if aspect < float(min_aspect):
            continue

        mx, my = float(centroids[k][0]), float(centroids[k][1])
        vx = mx - cx
        vy = my - cy
        vlen = float((vx * vx + vy * vy) ** 0.5) + 1e-6

        ys, xs = np.where(labels == k)
        if xs.size < 20:
            continue
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        vx_l, vy_l, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        llen = float((float(vx_l) ** 2 + float(vy_l) ** 2) ** 0.5) + 1e-6
        lx = float(vx_l) / llen
        ly = float(vy_l) / llen
        ux = float(vx) / vlen
        uy = float(vy) / vlen
        radial = abs(lx * ux + ly * uy)

        score = float(area) * (0.4 + 0.6 * radial) * (0.5 + 0.5 * min(6.0, aspect) / 6.0)
        if score > best_score:
            best_score = score
            best_id = int(k)

    needle = np.zeros((h, w), dtype=np.uint8)
    if best_id is not None:
        needle[labels == best_id] = 255
        needle = cv2.bitwise_and(needle, needle, mask=roi_mask)

    debug: Dict[str, object] = {
        "method": "connected_components_elongated",
        "r_in": int(r_in),
        "r_out": int(r_out),
        "sector_center_angle_deg_img": None if sector_center_angle_deg_img is None else float(sector_center_angle_deg_img),
        "sector_half_width_deg": None if sector_half_width_deg is None else float(sector_half_width_deg),
        "sector_ignored": bool(sector_center_angle_deg_img is not None or sector_half_width_deg is not None),
        "otsu_thr": float(otsu_thr),
        "strict_thr": int(strict_thr),
        "threshold_offset": int(threshold_offset),
        "min_area": int(min_area),
        "min_aspect": float(min_aspect),
        "component_found": bool(best_id is not None),
        "n_components": int(max(0, int(num) - 1)),
    }
    if best_id is not None:
        x, y, ww, hh, area = [int(v) for v in stats[int(best_id)].tolist()]
        debug["best_component"] = {
            "id": int(best_id),
            "bbox": {"x": x, "y": y, "w": ww, "h": hh},
            "area": int(area),
            "centroid": {"x": float(centroids[int(best_id)][0]), "y": float(centroids[int(best_id)][1])},
        }

    return NeedleExtraction(
        needle_mask_u8=needle,
        roi_mask_u8=roi_mask,
        binary_mask_u8=binary,
        debug=debug,
    )


def estimate_zero_angle_deg(
    gray_u8: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    r_min: int,
    r_max: int,
    clockwise: bool = True,
) -> ZeroAngleEstimation:
    """Estimate where 00:00 lies (angle origin).

    MVP2では仮置きで -90deg（画像の12時方向）を返す。
    MVP4での自動推定に備えてインターフェースだけ先に用意する。
    """

    _ = (gray_u8, center_xy, r_min, r_max, clockwise)
    return ZeroAngleEstimation(zero_angle_deg=-90.0, debug={"method": "stub"})


def estimate_speed_band(
    gray_u8: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    disc_outer_radius: Optional[float] = None,
    search_r_min: Optional[int] = None,
    search_r_max: Optional[int] = None,
    angle_step_deg: float = 1.0,
) -> Optional[SpeedBandEstimation]:
    """Estimate speed band ring (r_in, r_out) where printed concentric lines are dense.

    Implements spec 6-4/13 in a scan-robust way:
    - Detect edges
    - For each radius r, sample edge density along the circle
    - Pick the largest contiguous high-density radius interval
    """

    if gray_u8.ndim != 2 or gray_u8.dtype != np.uint8:
        raise ValueError("gray_u8 must be a 2D uint8 image")

    h, w = gray_u8.shape[:2]
    min_dim = int(min(h, w))
    cx, cy = float(center_xy[0]), float(center_xy[1])

    # Use disc outer radius when available (more stable than full image size with margins).
    disc_r = float(disc_outer_radius) if disc_outer_radius is not None and float(disc_outer_radius) > 0 else None
    if disc_r is not None:
        base_r = disc_r
    else:
        base_r = float(min_dim) / 2.0

    if search_r_min is None:
        # Speed scale ring is near the outer region of the disc.
        search_r_min = int(round(base_r * 0.55))
    if search_r_max is None:
        search_r_max = int(round(base_r * 0.92))

    search_r_min = max(5, int(search_r_min))
    # Clip to image bounds.
    search_r_max = min(int(search_r_max), int(round(min_dim * 0.60)))
    if search_r_max <= search_r_min + 10:
        return None

    blur = cv2.GaussianBlur(gray_u8, (0, 0), 1.5)
    edges = cv2.Canny(blur, 60, 160)

    rs = np.arange(search_r_min, search_r_max + 1, dtype=np.float32)
    angles_deg = np.arange(0.0, 360.0, float(angle_step_deg), dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg)

    # Sample edge presence along each circle.
    dens = np.zeros((rs.shape[0],), dtype=np.float32)
    for i, r in enumerate(rs):
        xs = cx + float(r) * np.cos(angles_rad)
        ys = cy + float(r) * np.sin(angles_rad)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        map_x = coords[:, 0].reshape(-1, 1)
        map_y = coords[:, 1].reshape(-1, 1)
        sampled = cv2.remap(
            edges,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).reshape(-1)
        dens[i] = float(np.mean(sampled > 0))

    # Smooth density profile along radius.
    dens_s = cv2.GaussianBlur(dens.reshape(-1, 1), (1, 0), 2.0).reshape(-1)

    med = float(np.median(dens_s))
    mad = float(np.median(np.abs(dens_s - med)))
    thr = med + 3.0 * (1.4826 * mad + 1e-6)

    hi = dens_s > float(thr)
    if not bool(np.any(hi)):
        # fallback: use top-quantile
        q = float(np.quantile(dens_s, 0.92))
        hi = dens_s >= q

    # Find contiguous high-density runs; choose the best-scoring one.
    # Prefer mid-radius bands and avoid runs that touch the search boundary.
    best_score = -1e18
    best_a = None
    best_b = None
    a = None
    hi_list = hi.tolist() + [False]
    for i, ok in enumerate(hi_list):
        if ok and a is None:
            a = i
        if (not ok) and a is not None:
            b = i
            ln = b - a
            if ln > 0:
                r_in_f = float(rs[int(a)])
                r_out_f = float(rs[int(b - 1)])
                r_mid = 0.5 * (r_in_f + r_out_f)

                mean_d = float(np.mean(dens_s[a:b]))
                # Penalize being too close to the search boundaries.
                penalty = 0.0
                if a <= 1 or b >= len(rs) - 1:
                    penalty += 0.10
                # Prefer bands around the outer ring where the speed scale lives.
                target = 0.72
                pref = 1.0 - min(1.0, abs((r_mid / float(max(1.0, base_r))) - target) / 0.18)
                score = (mean_d * float(ln)) * (0.5 + 0.5 * pref) - penalty * float(ln)

                if score > best_score:
                    best_score = score
                    best_a, best_b = a, b
            a = None

    if best_a is None or best_b is None:
        return None

    r_in = int(round(float(rs[int(best_a)])))
    r_out = int(round(float(rs[int(best_b - 1)])))

    rejected_heuristics: List[str] = []
    band_thickness = int(r_out - r_in)
    if band_thickness < int(round(min_dim * 0.06)):
        rejected_heuristics.append("thin_band")

    r_out_ratio = float(r_out) / float(max(1.0, base_r))
    if r_out_ratio > 0.55:
        rejected_heuristics.append("outer_edge_band")

    debug: Dict[str, object] = {
        "method": "radial_edge_density",
        "disc_outer_radius": None if disc_r is None else float(disc_r),
        "base_r": float(base_r),
        "search_r_min": int(search_r_min),
        "search_r_max": int(search_r_max),
        "angle_step_deg": float(angle_step_deg),
        "thr": float(thr),
        "r_in": int(r_in),
        "r_out": int(r_out),
        "band_thickness": int(band_thickness),
        "r_out_ratio": float(r_out_ratio),
        "rejected_heuristics": list(rejected_heuristics),
    }

    return SpeedBandEstimation(r_in=int(r_in), r_out=int(r_out), debug=debug)


def _sample_radial_profile(
    gray_u8: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    angle_rad: float,
    r_min: int,
    r_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (radii, ink_scores) for the given angle.

    ink_scores: higher means more ink-like (darker).
    """

    h, w = gray_u8.shape[:2]
    cx, cy = center_xy

    rs = np.arange(r_min, r_max + 1, dtype=np.float32)
    xs = cx + rs * np.cos(angle_rad)
    ys = cy + rs * np.sin(angle_rad)

    coords = np.stack([xs, ys], axis=1).astype(np.float32)

    # OpenCV remap wants map_x/map_y shaped (H, W); here we sample 1D.
    map_x = coords[:, 0].reshape(-1, 1)
    map_y = coords[:, 1].reshape(-1, 1)

    sampled = cv2.remap(
        gray_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    ).reshape(-1)

    # ink score base: invert grayscale
    ink = (255.0 - sampled.astype(np.float32))

    return rs, ink


def _ink_score_profile(
    gray_u8: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    angle_rad: float,
    r_min: int,
    r_max: int,
    radial_smooth_sigma: float = 1.2,
    baseline_smooth_sigma: float = 8.0,
    edge_boost: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (radii, score) where higher score is more likely to be the trace.

    印刷罫線/影などの“広い暗さ”に引っ張られないよう、
    - 短周期の暗さ（局所的な線）
    - 長周期の暗さ（紙のムラ/影）
    を分離し、(local - baseline) を主成分として使う。
    """

    rs, ink = _sample_radial_profile(
        gray_u8,
        center_xy=center_xy,
        angle_rad=angle_rad,
        r_min=r_min,
        r_max=r_max,
    )

    ink_1d = ink.reshape(-1, 1)
    local = cv2.GaussianBlur(ink_1d, (1, 0), float(radial_smooth_sigma)).reshape(-1)
    baseline = cv2.GaussianBlur(ink_1d, (1, 0), float(baseline_smooth_sigma)).reshape(-1)
    hp = local - baseline

    # 線のエッジ（濃度の立ち上がり）も少しだけ評価に入れる
    grad = np.gradient(local).astype(np.float32)
    grad_abs = np.abs(grad)

    score = hp + float(edge_boost) * grad_abs
    return rs, score


def _viterbi_pick_radii(
    score_by_angle: np.ndarray,
    radii: np.ndarray,
    *,
    smooth_l1: float,
    max_jump_px: float,
) -> np.ndarray:
    """Find best radius index per angle using DP.

    score_by_angle: shape (A, R)
    radii: shape (R,)
    """

    a_count, r_count = score_by_angle.shape
    dp = np.full((a_count, r_count), -1e18, dtype=np.float32)
    prev = np.full((a_count, r_count), -1, dtype=np.int32)

    dp[0] = score_by_angle[0].astype(np.float32)

    # Precompute transition costs for dr between radius indices.
    r_vals = radii.astype(np.float32)
    dr_mat = np.abs(r_vals.reshape(-1, 1) - r_vals.reshape(1, -1))
    cost = float(smooth_l1) * dr_mat
    big = dr_mat > float(max_jump_px)
    # discourage very large jumps harder
    cost = cost + big.astype(np.float32) * (dr_mat - float(max_jump_px)) * float(smooth_l1) * 4.0

    for a in range(1, a_count):
        # dp[a-1, k] - cost[k, j] + score[a, j]
        # We compute best predecessor for each j.
        best_prev = (dp[a - 1].reshape(-1, 1) - cost).astype(np.float32)
        prev_idx = np.argmax(best_prev, axis=0).astype(np.int32)
        dp[a] = best_prev[prev_idx, np.arange(r_count)] + score_by_angle[a]
        prev[a] = prev_idx

    # backtrack
    idxs = np.empty((a_count,), dtype=np.int32)
    idxs[-1] = int(np.argmax(dp[-1]))
    for a in range(a_count - 1, 0, -1):
        idxs[a - 1] = prev[a, idxs[a]]

    return idxs


def extract_speed_trace(
    gray_u8: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    r_min: int,
    r_max: int,
    angle_step_deg: float = 1.0,
    zero_angle_deg: Optional[float] = -90.0,
    clockwise: bool = True,
    continuity_weight: float = 2.0,
    max_jump_px: float = 30.0,
    blur_sigma: float = 1.0,
    method: str = "viterbi",
    radial_smooth_sigma: float = 1.2,
    baseline_smooth_sigma: float = 8.0,
    edge_boost: float = 0.4,
    red_suppress_mask: Optional[np.ndarray] = None,
    red_suppress_weight: float = 60.0,
) -> TraceResult:
    """MVP2: extract one radial trace point per angle.

    Assumption for MVP2 bootstrap:
    - 00:00 (time origin) is at 12 o'clock direction.
    - Angles increase clockwise.

    This will be revisited in MVP4 (time alignment).
    """

    if gray_u8.ndim != 2 or gray_u8.dtype != np.uint8:
        raise ValueError("gray_u8 must be a 2D uint8 image")
    if r_min <= 0 or r_max <= r_min:
        raise ValueError("invalid r_min/r_max")

    # Denoise slightly so the score is less sensitive to scan noise.
    if blur_sigma > 0:
        gray_u8 = cv2.GaussianBlur(gray_u8, (0, 0), float(blur_sigma))

    direction = -1.0 if clockwise else 1.0
    angles_deg = np.arange(0.0, 360.0, float(angle_step_deg), dtype=np.float32)

    if zero_angle_deg is None:
        zero_est = estimate_zero_angle_deg(gray_u8, center_xy=center_xy, r_min=r_min, r_max=r_max, clockwise=clockwise)
        zero_angle_deg_val = float(zero_est.zero_angle_deg)
    else:
        zero_est = None
        zero_angle_deg_val = float(zero_angle_deg)

    # Build ink matrix (A, R)
    ink_rows: List[np.ndarray] = []
    radii_ref: Optional[np.ndarray] = None

    for a in angles_deg:
        angle_deg = float(a)
        theta = np.deg2rad(zero_angle_deg_val + direction * angle_deg)
        rs, ink = _sample_radial_profile(gray_u8, center_xy=center_xy, angle_rad=float(theta), r_min=r_min, r_max=r_max)
        if radii_ref is None:
            radii_ref = rs
        ink_rows.append(ink.astype(np.float32))

    assert radii_ref is not None
    ink_by_angle = np.stack(ink_rows, axis=0)

    ink_lp = cv2.GaussianBlur(ink_by_angle, (1, 0), float(radial_smooth_sigma))
    ink_base = cv2.GaussianBlur(ink_by_angle, (1, 0), float(baseline_smooth_sigma))
    hp = ink_lp - ink_base

    # Suppress printed concentric rings by subtracting per-radius median across angles.
    ring_base = np.percentile(hp, 25, axis=0, keepdims=True).astype(np.float32)
    hp2 = hp - ring_base

    angle_base = np.median(hp2, axis=1, keepdims=True).astype(np.float32)
    hp3 = hp2 - angle_base

    grad = np.gradient(ink_lp, axis=1).astype(np.float32)
    score_by_angle = hp3 + float(edge_boost) * np.abs(grad)

    if red_suppress_mask is not None:
        red_rows: List[np.ndarray] = []
        for a in angles_deg:
            angle_deg = float(a)
            theta = np.deg2rad(float(zero_angle_deg_val) + direction * angle_deg)
            rs2 = radii_ref.astype(np.float32)
            xs = float(center_xy[0]) + rs2 * np.cos(float(theta))
            ys = float(center_xy[1]) + rs2 * np.sin(float(theta))
            map_x = xs.reshape(-1, 1).astype(np.float32)
            map_y = ys.reshape(-1, 1).astype(np.float32)
            sampled = cv2.remap(
                red_suppress_mask,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).reshape(-1).astype(np.float32) / 255.0
            red_rows.append(sampled)
        red_by_angle = np.stack(red_rows, axis=0)
        score_by_angle = score_by_angle - float(red_suppress_weight) * red_by_angle

    method_l = method.strip().lower()
    if method_l not in {"viterbi", "greedy"}:
        raise ValueError("method must be 'viterbi' or 'greedy'")

    if method_l == "viterbi":
        idxs = _viterbi_pick_radii(
            score_by_angle,
            radii_ref,
            smooth_l1=float(continuity_weight),
            max_jump_px=float(max_jump_px),
        )
    else:
        # Fallback: sequential greedy
        idxs = np.empty((score_by_angle.shape[0],), dtype=np.int32)
        prev_r: Optional[float] = None
        for i in range(score_by_angle.shape[0]):
            score = score_by_angle[i]
            if prev_r is None:
                idxs[i] = int(np.argmax(score))
            else:
                dr = np.abs(radii_ref - float(prev_r))
                penalty = float(continuity_weight) * dr
                big = dr > float(max_jump_px)
                penalty[big] += float(continuity_weight) * (dr[big] - float(max_jump_px)) * 4.0
                idxs[i] = int(np.argmax(score - penalty))
            prev_r = float(radii_ref[idxs[i]])

    points: List[TracePoint] = []
    for i, a in enumerate(angles_deg):
        idx = int(idxs[i])
        points.append(
            TracePoint(
                angle_deg=float(a),
                radius_px=float(radii_ref[idx]),
                ink_score=float(score_by_angle[i, idx]),
            )
        )

    debug: Dict[str, object] = {
        "r_min": int(r_min),
        "r_max": int(r_max),
        "angle_step_deg": float(angle_step_deg),
        "zero_angle_deg": float(zero_angle_deg_val),
        "clockwise": bool(clockwise),
        "continuity_weight": float(continuity_weight),
        "max_jump_px": float(max_jump_px),
        "method": method_l,
        "radial_smooth_sigma": float(radial_smooth_sigma),
        "baseline_smooth_sigma": float(baseline_smooth_sigma),
        "edge_boost": float(edge_boost),
        "red_suppress_applied": bool(red_suppress_mask is not None),
        "red_suppress_weight": float(red_suppress_weight),
    }

    if zero_est is not None:
        debug["zero_angle_estimation"] = zero_est.debug

    return TraceResult(points=points, debug=debug)
