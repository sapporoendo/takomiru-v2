from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .center import estimate_center
from .io import load_image, rgb_to_gray_u8
from .trace import extract_speed_trace


def _default_r_range(est) -> tuple[int, int]:
    h, w = est.debug.get("shape", [0, 0]) if isinstance(est.debug, dict) else (0, 0)
    min_dim = int(min(h, w)) if h and w else 0

    if est.inner_radius is not None and est.outer_radius is not None:
        r_min = int(round(float(est.inner_radius) * 1.5))
        r_max = int(round(float(est.outer_radius) * 0.92))
        if r_max > r_min + 10:
            return r_min, r_max

    if min_dim:
        return int(min_dim * 0.10), int(min_dim * 0.48)

    return 30, 250


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image/PDF path")
    parser.add_argument("--pdf-dpi", type=int, default=300)
    parser.add_argument("--pdf-page", type=int, default=0)

    parser.add_argument("--angle-step", type=float, default=1.0)
    parser.add_argument("--zero-angle-deg", type=float, default=-90.0, help="00:00 direction. Default is 12 o'clock")
    parser.add_argument("--clockwise", action="store_true")

    parser.add_argument("--r-min", type=int, default=None)
    parser.add_argument("--r-max", type=int, default=None)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-image", default=None)

    args = parser.parse_args()

    loaded = load_image(args.input, pdf_page_index=args.pdf_page, pdf_dpi=args.pdf_dpi)
    gray = rgb_to_gray_u8(loaded.rgb)
    est = estimate_center(gray)

    if args.r_min is None or args.r_max is None:
        r_min, r_max = _default_r_range(est)
    else:
        r_min, r_max = int(args.r_min), int(args.r_max)

    trace = extract_speed_trace(
        gray,
        center_xy=(est.center_x, est.center_y),
        r_min=r_min,
        r_max=r_max,
        angle_step_deg=float(args.angle_step),
        zero_angle_deg=float(args.zero_angle_deg),
        clockwise=bool(args.clockwise),
    )

    out = {
        "source": str(Path(loaded.source_path)),
        "center": {"x": est.center_x, "y": est.center_y},
        "center_uncertain": bool(est.center_uncertain),
        "trace": {
            "r_min": r_min,
            "r_max": r_max,
            "angle_step_deg": float(args.angle_step),
            "zero_angle_deg": float(args.zero_angle_deg),
            "clockwise": bool(args.clockwise),
            "points": [{"angle_deg": p.angle_deg, "radius_px": p.radius_px, "ink_score": p.ink_score} for p in trace.points],
        },
    }

    if args.debug:
        debug_image_path = (
            Path(args.debug_image)
            if args.debug_image
            else Path(loaded.source_path).with_name(Path(loaded.source_path).stem + "_trace_debug.png")
        )

        bgr = cv2.cvtColor(loaded.rgb, cv2.COLOR_RGB2BGR)
        cx = float(est.center_x)
        cy = float(est.center_y)

        # draw center
        cv2.drawMarker(
            bgr,
            (int(round(cx)), int(round(cy))),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=40,
            thickness=3,
        )

        # draw trace points (green)
        direction = -1.0 if bool(args.clockwise) else 1.0
        for p in trace.points[:: max(1, int(round(5.0 / float(args.angle_step))))]:
            theta = np.deg2rad(float(args.zero_angle_deg) + direction * float(p.angle_deg))
            x = cx + float(p.radius_px) * float(np.cos(theta))
            y = cy + float(p.radius_px) * float(np.sin(theta))
            cv2.circle(bgr, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)

        label = f"center_uncertain={bool(est.center_uncertain)} step={float(args.angle_step):g}deg"
        cv2.putText(bgr, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

        debug_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_image_path), bgr)
        out["debug_image"] = str(debug_image_path)

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
