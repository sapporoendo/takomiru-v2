from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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

    # ink score: invert grayscale
    ink = (255.0 - sampled.astype(np.float32))

    return rs, ink


def extract_speed_trace(
    gray_u8: np.ndarray,
    *,
    center_xy: Tuple[float, float],
    r_min: int,
    r_max: int,
    angle_step_deg: float = 1.0,
    zero_angle_deg: float = -90.0,
    clockwise: bool = True,
    continuity_weight: float = 2.0,
    max_jump_px: float = 30.0,
    blur_sigma: float = 1.0,
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

    points: List[TracePoint] = []
    prev_r: Optional[float] = None

    for a in angles_deg:
        angle_deg = float(a)
        theta = np.deg2rad(zero_angle_deg + direction * angle_deg)

        rs, ink = _sample_radial_profile(gray_u8, center_xy=center_xy, angle_rad=float(theta), r_min=r_min, r_max=r_max)

        # Smooth along radius so we prefer consistent peaks (ink line thickness).
        if ink.size >= 5:
            ink_smooth = cv2.GaussianBlur(ink.reshape(-1, 1), (1, 0), 1.0).reshape(-1)
        else:
            ink_smooth = ink

        if prev_r is None:
            idx = int(np.argmax(ink_smooth))
            chosen_r = float(rs[idx])
            chosen_score = float(ink_smooth[idx])
        else:
            # Continuity rule: penalize large jumps from previous radius.
            dr = np.abs(rs - float(prev_r))
            penalty = continuity_weight * dr

            # Additional hard-ish penalty for very large jumps.
            big = dr > float(max_jump_px)
            penalty[big] += continuity_weight * (dr[big] - float(max_jump_px)) * 4.0

            objective = ink_smooth - penalty
            idx = int(np.argmax(objective))
            chosen_r = float(rs[idx])
            chosen_score = float(ink_smooth[idx])

        points.append(TracePoint(angle_deg=angle_deg, radius_px=chosen_r, ink_score=chosen_score))
        prev_r = chosen_r

    debug: Dict[str, object] = {
        "r_min": int(r_min),
        "r_max": int(r_max),
        "angle_step_deg": float(angle_step_deg),
        "zero_angle_deg": float(zero_angle_deg),
        "clockwise": bool(clockwise),
        "continuity_weight": float(continuity_weight),
        "max_jump_px": float(max_jump_px),
    }

    return TraceResult(points=points, debug=debug)
