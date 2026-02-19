from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .center import estimate_center_auto
from .io import load_image, rgb_to_gray_u8
from .speed_scale import speed_scale_from_spindle_radius
from .trace import extract_needle_mask, extract_speed_trace, estimate_speed_scale_radii, fixed_speed_band_from_outer_radius


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

    parser.add_argument("--speed-band-r-in-ratio", type=float, default=0.55)
    parser.add_argument("--speed-band-r-out-ratio", type=float, default=0.86)

    parser.add_argument("--speed-vmax-kmh", type=float, default=None)
    parser.add_argument("--needle-speed-vmax-kmh", type=float, default=120.0)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-image", default=None)
    parser.add_argument("--debug-trace-mask", default=None)
    parser.add_argument("--needle-mask-out", default=None)

    parser.add_argument("--needle-sector-center-angle-deg", type=float, default=None)
    parser.add_argument("--needle-sector-half-width-deg", type=float, default=None)
    parser.add_argument("--needle-speed-min-kmh", type=float, default=20.0)
    parser.add_argument("--needle-speed-max-kmh", type=float, default=120.0)
    parser.add_argument("--needle-speed-radius-gamma", type=float, default=1.0)
    parser.add_argument("--speed-scale-mode", type=str, default="spindle", choices=["spindle", "outer", "auto"])
    parser.add_argument("--use-fixed-speed-scale", action="store_true")
    parser.set_defaults(use_fixed_speed_scale=True)
    parser.add_argument("--speed-scale-r20-ratio", type=float, default=0.840705)
    parser.add_argument("--speed-scale-r120-ratio", type=float, default=0.925113)
    parser.add_argument("--spindle-scale-r20-mul", type=float, default=5.15)
    parser.add_argument("--spindle-scale-r120-mul", type=float, default=6.01)
    parser.add_argument("--auto-speed-scale", action="store_true")
    parser.add_argument("--needle-threshold-offset", type=int, default=15)
    parser.add_argument("--needle-min-area", type=int, default=120)
    parser.add_argument("--needle-min-aspect", type=float, default=3.0)

    args = parser.parse_args()

    loaded = load_image(args.input, pdf_page_index=args.pdf_page, pdf_dpi=args.pdf_dpi)
    gray = rgb_to_gray_u8(loaded.rgb)
    est = estimate_center_auto(gray)

    speed_band_uncertain = False
    speed_band_debug = None
    r_range_source = "cli"
    if args.r_min is None or args.r_max is None:
        r_range_source = "fixed_outer_radius"
        r_in_ratio = float(args.speed_band_r_in_ratio)
        r_out_ratio = float(args.speed_band_r_out_ratio)
        if not (0.0 < r_in_ratio < r_out_ratio < 1.2):
            raise ValueError("--speed-band-r-in-ratio/--speed-band-r-out-ratio must satisfy 0 < r_in < r_out")
        band = fixed_speed_band_from_outer_radius(
            outer_radius=est.outer_radius,
            r_in_ratio=r_in_ratio,
            r_out_ratio=r_out_ratio,
        )
        if band is None:
            speed_band_uncertain = True
            raise ValueError("outer_radius is required to compute speed band; provide --r-min/--r-max or fix center estimation")
        r_min, r_max = int(band.r_in), int(band.r_out)
        speed_band_debug = dict(band.debug)
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

    # Needle ROI (20–120km/h band) is a subset of the fixed speed-band annulus.
    # We map speed->radius linearly within [r_min(=0km/h), r_max(=Vmax)].
    vmax_arg = args.needle_speed_vmax_kmh if args.speed_vmax_kmh is None else float(args.speed_vmax_kmh)
    vmax = float(vmax_arg)
    if not (vmax > 0):
        raise ValueError("--needle-speed-vmax-kmh must be > 0")
    v0 = float(args.needle_speed_min_kmh)
    v1 = float(args.needle_speed_max_kmh)
    if v1 < v0:
        raise ValueError("--needle-speed-max-kmh must be >= --needle-speed-min-kmh")
    v0c = max(0.0, min(vmax, v0))
    v1c = max(0.0, min(vmax, v1))

    gamma = float(args.needle_speed_radius_gamma)
    if not (gamma > 0):
        raise ValueError("--needle-speed-radius-gamma must be > 0")

    f0 = (v0c / vmax) ** gamma if vmax > 0 else 0.0
    f1 = (v1c / vmax) ** gamma if vmax > 0 else 0.0
    needle_r_in = int(round(float(r_min) + float(f0) * float(r_max - r_min)))
    needle_r_out = int(round(float(r_min) + float(f1) * float(r_max - r_min)))
    needle_r_in = max(int(r_min), min(int(r_max - 1), int(needle_r_in)))
    needle_r_out = max(int(needle_r_in + 1), min(int(r_max), int(needle_r_out)))

    speed_scale_est = None
    speed_scale_fixed = None
    speed_scale_mode = str(args.speed_scale_mode)
    if bool(args.auto_speed_scale):
        speed_scale_mode = "auto"

    if speed_scale_mode == "spindle":
        spindle_r = None
        if isinstance(est.debug, dict) and est.debug.get("spindle_detected"):
            spindle_r = est.debug.get("spindle_radius")
        if spindle_r is not None:
            sc = speed_scale_from_spindle_radius(
                float(spindle_r),
                r20_mul=float(args.spindle_scale_r20_mul),
                r120_mul=float(args.spindle_scale_r120_mul),
            )
            if sc is not None:
                needle_r_in = int(sc.r20)
                needle_r_out = int(sc.r120)
                speed_scale_fixed = dict(sc.debug)

    if speed_scale_mode == "outer" and est.outer_radius is not None:
        R = float(est.outer_radius)
        r20_ratio = float(args.speed_scale_r20_ratio)
        r120_ratio = float(args.speed_scale_r120_ratio)
        if not (0.0 < r20_ratio < r120_ratio < 1.2):
            raise ValueError("--speed-scale-r20-ratio/--speed-scale-r120-ratio must satisfy 0 < r20 < r120")
        r20 = int(round(R * r20_ratio))
        r120 = int(round(R * r120_ratio))
        needle_r_in = int(r20)
        needle_r_out = int(r120)
        speed_scale_fixed = {
            "method": "outer_fixed_ratio",
            "outer_radius": float(R),
            "r20": int(r20),
            "r120": int(r120),
            "r20_ratio": float(r20_ratio),
            "r120_ratio": float(r120_ratio),
        }

    if speed_scale_mode == "auto" and est.outer_radius is not None:
        speed_scale_est = estimate_speed_scale_radii(
            gray,
            center_xy=(est.center_x, est.center_y),
            disc_outer_radius=float(est.outer_radius),
        )
        needle_r_in = int(speed_scale_est.r20)
        needle_r_out = int(speed_scale_est.r120)

    needle = extract_needle_mask(
        gray,
        center_xy=(est.center_x, est.center_y),
        r_in=int(needle_r_in),
        r_out=int(needle_r_out),
        sector_center_angle_deg_img=args.needle_sector_center_angle_deg,
        sector_half_width_deg=args.needle_sector_half_width_deg,
        threshold_offset=int(args.needle_threshold_offset),
        min_area=int(args.needle_min_area),
        min_aspect=float(args.needle_min_aspect),
    )

    out = {
        "source": str(Path(loaded.source_path)),
        "center": {"x": est.center_x, "y": est.center_y},
        "outer_radius": None if est.outer_radius is None else float(est.outer_radius),
        "center_uncertain": bool(est.center_uncertain),
        "speed_band_uncertain": bool(speed_band_uncertain),
        "r_range_source": str(r_range_source),
        "trace": {
            "r_min": r_min,
            "r_max": r_max,
            "angle_step_deg": float(args.angle_step),
            "zero_angle_deg": float(args.zero_angle_deg),
            "clockwise": bool(args.clockwise),
            "points": [{"angle_deg": p.angle_deg, "radius_px": p.radius_px, "ink_score": p.ink_score} for p in trace.points],
        },
        "needle": {
            "method": str(needle.debug.get("method", "")),
            "debug": dict(needle.debug),
        },
    }

    if bool(args.debug):
        out["center_debug"] = dict(est.debug) if isinstance(est.debug, dict) else {}
        if speed_band_debug is not None:
            out["speed_band_debug"] = dict(speed_band_debug) if isinstance(speed_band_debug, dict) else speed_band_debug
        if speed_scale_est is not None:
            out["speed_scale_est"] = dict(speed_scale_est.debug) if hasattr(speed_scale_est, "debug") else {
                "r20": int(speed_scale_est.r20),
                "r120": int(speed_scale_est.r120),
            }
        if speed_scale_fixed is not None:
            out["speed_scale_fixed"] = dict(speed_scale_fixed)

    out["needle_roi"] = {
        "vmax_kmh": float(vmax),
        "speed_min_kmh": float(v0c),
        "speed_max_kmh": float(v1c),
        "speed_radius_gamma": float(gamma),
        "r_in": int(needle_r_in),
        "r_out": int(needle_r_out),
    }

    if speed_scale_fixed is not None:
        out["speed_scale_fixed"] = dict(speed_scale_fixed)

    if speed_scale_est is not None:
        out["speed_scale_estimation"] = dict(speed_scale_est.debug)

    if speed_band_debug is not None:
        out["speed_band_estimation"] = speed_band_debug

    if est.outer_radius is not None and r_range_source == "fixed_outer_radius":
        R = float(est.outer_radius)
        out["speed_band"] = {
            "r_in": int(r_min),
            "r_out": int(r_max),
            "r_in_ratio": float(r_min) / float(max(1.0, R)),
            "r_out_ratio": float(r_max) / float(max(1.0, R)),
        }

    needle_mask_path = (
        Path(args.needle_mask_out)
        if args.needle_mask_out
        else Path(loaded.source_path).with_name(Path(loaded.source_path).stem + "_needle_mask.png")
    )
    needle_mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(needle_mask_path), needle.needle_mask_u8)
    out["needle_mask"] = str(needle_mask_path)

    if args.debug:
        debug_image_path = (
            Path(args.debug_image)
            if args.debug_image
            else Path(loaded.source_path).with_name(Path(loaded.source_path).stem + "_trace_debug.png")
        )

        debug_outer_path = debug_image_path.with_name(debug_image_path.stem + "_outer_circle.png")
        debug_annulus_path = debug_image_path.with_name(debug_image_path.stem + "_speed_band_annulus.png")
        debug_speed_scale_path = debug_image_path.with_name(debug_image_path.stem + "_speed_scale_check.png")
        debug_needle_path = debug_image_path.with_name(debug_image_path.stem + "_needle_overlay.png")
        debug_needle_roi_path = debug_image_path.with_name(debug_image_path.stem + "_needle_roi.png")
        debug_needle_roi_mask_path = debug_image_path.with_name(debug_image_path.stem + "_needle_roi_mask.png")
        debug_needle_binary_path = debug_image_path.with_name(debug_image_path.stem + "_needle_binary.png")

        bgr_base = cv2.cvtColor(loaded.rgb, cv2.COLOR_RGB2BGR)
        cx = float(est.center_x)
        cy = float(est.center_y)

        def draw_center_and_outer(img: np.ndarray) -> None:
            cv2.drawMarker(
                img,
                (int(round(cx)), int(round(cy))),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=40,
                thickness=3,
            )
            if est.outer_radius is not None:
                cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(float(est.outer_radius))), (255, 0, 255), 2)

        bgr_outer = bgr_base.copy()
        draw_center_and_outer(bgr_outer)

        bgr_ann = bgr_base.copy()
        draw_center_and_outer(bgr_ann)
        cv2.circle(bgr_ann, (int(round(cx)), int(round(cy))), int(round(r_min)), (255, 0, 0), 2)
        cv2.circle(bgr_ann, (int(round(cx)), int(round(cy))), int(round(r_max)), (255, 0, 0), 2)

        bgr = bgr_ann.copy()

        # draw trace points (green)
        direction = -1.0 if bool(args.clockwise) else 1.0

        stride = max(1, int(round(2.0 / float(args.angle_step))))
        poly = []
        for p in trace.points[::stride]:
            theta = np.deg2rad(float(args.zero_angle_deg) + direction * float(p.angle_deg))
            x = cx + float(p.radius_px) * float(np.cos(theta))
            y = cy + float(p.radius_px) * float(np.sin(theta))
            poly.append([int(round(x)), int(round(y))])

        if len(poly) >= 2:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        for x, y in poly[:: max(1, int(round(8.0 / float(args.angle_step))))]:
            cv2.circle(bgr, (int(x), int(y)), 3, (0, 255, 0), -1)

        label = f"center_uncertain={bool(est.center_uncertain)} step={float(args.angle_step):g}deg"
        cv2.putText(bgr, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

        bgr_outer_path_parent = debug_outer_path.parent
        bgr_outer_path_parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_outer_path), bgr_outer)
        cv2.imwrite(str(debug_annulus_path), bgr_ann)

        # speed scale check (20/120 boundary circles from needle_roi)
        bgr_scale = bgr_base.copy()
        draw_center_and_outer(bgr_scale)
        if "needle_roi" in out and isinstance(out["needle_roi"], dict):
            nr_in = int(out["needle_roi"].get("r_in", 0))
            nr_out = int(out["needle_roi"].get("r_out", 0))
            if nr_in > 0:
                cv2.circle(bgr_scale, (int(round(cx)), int(round(cy))), int(round(nr_in)), (0, 255, 255), 2)
            if nr_out > 0:
                cv2.circle(bgr_scale, (int(round(cx)), int(round(cy))), int(round(nr_out)), (0, 255, 255), 2)
        cv2.imwrite(str(debug_speed_scale_path), bgr_scale)

        # needle overlay
        bgr_needle = bgr_base.copy()
        draw_center_and_outer(bgr_needle)
        green = np.zeros_like(bgr_needle)
        green[:, :, 1] = needle.needle_mask_u8
        bgr_needle = cv2.addWeighted(bgr_needle, 1.0, green, 0.7, 0.0)
        cv2.imwrite(str(debug_needle_path), bgr_needle)

        # needle roi (wedge-style) overlay
        bgr_roi = bgr_base.copy()
        draw_center_and_outer(bgr_roi)
        blue = np.zeros_like(bgr_roi)
        blue[:, :, 0] = needle.roi_mask_u8
        bgr_roi = cv2.addWeighted(bgr_roi, 1.0, blue, 0.35, 0.0)

        # draw needle ROI radii for clarity
        if "needle_roi" in out and isinstance(out["needle_roi"], dict):
            nr_in = int(out["needle_roi"].get("r_in", 0))
            nr_out = int(out["needle_roi"].get("r_out", 0))
            if nr_in > 0:
                cv2.circle(bgr_roi, (int(round(cx)), int(round(cy))), int(round(nr_in)), (255, 255, 0), 2)
            if nr_out > 0:
                cv2.circle(bgr_roi, (int(round(cx)), int(round(cy))), int(round(nr_out)), (255, 255, 0), 2)

        cv2.imwrite(str(debug_needle_roi_path), bgr_roi)

        # raw ROI mask
        cv2.imwrite(str(debug_needle_roi_mask_path), needle.roi_mask_u8)

        # needle binary mask visualization
        cv2.imwrite(str(debug_needle_binary_path), needle.binary_mask_u8)

        cv2.imwrite(str(debug_image_path), bgr)
        out["debug_outer_circle"] = str(debug_outer_path)
        out["debug_speed_band_annulus"] = str(debug_annulus_path)
        out["debug_speed_scale_check"] = str(debug_speed_scale_path)
        out["debug_needle_overlay"] = str(debug_needle_path)
        out["debug_needle_roi"] = str(debug_needle_roi_path)
        out["debug_needle_roi_mask"] = str(debug_needle_roi_mask_path)
        out["debug_needle_binary"] = str(debug_needle_binary_path)
        out["debug_image"] = str(debug_image_path)

        if args.debug_trace_mask is not None:
            mask_path = Path(args.debug_trace_mask) if args.debug_trace_mask else debug_image_path.with_name(debug_image_path.stem + "_mask.png")
            mask = np.full((bgr.shape[0], bgr.shape[1], 3), 0, dtype=np.uint8)
            if len(poly) >= 2:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(mask, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.drawMarker(
                mask,
                (int(round(cx)), int(round(cy))),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=40,
                thickness=3,
            )
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r_min)), (255, 0, 0), 2)
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r_max)), (255, 0, 0), 2)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)
            out["debug_trace_mask"] = str(mask_path)

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
