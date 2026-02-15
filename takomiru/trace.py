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


@dataclass(frozen=True)
class ZeroAngleEstimation:
    zero_angle_deg: float
    debug: Dict[str, object]


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

    # Build score matrix (A, R)
    score_rows: List[np.ndarray] = []
    radii_ref: Optional[np.ndarray] = None

    for a in angles_deg:
        angle_deg = float(a)
        theta = np.deg2rad(zero_angle_deg_val + direction * angle_deg)
        rs, score = _ink_score_profile(gray_u8, center_xy=center_xy, angle_rad=float(theta), r_min=r_min, r_max=r_max)
        if radii_ref is None:
            radii_ref = rs
        score_rows.append(score.astype(np.float32))

    assert radii_ref is not None
    score_by_angle = np.stack(score_rows, axis=0)

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
    }

    if zero_est is not None:
        debug["zero_angle_estimation"] = zero_est.debug

    return TraceResult(points=points, debug=debug)
