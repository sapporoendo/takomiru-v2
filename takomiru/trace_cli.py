from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .center import detect_center_from_red_aux_rings, estimate_center_auto
from .io import load_image, rgb_to_gray_u8
try:
    from .speed_scale import speed_scale_from_spindle_radius
except ModuleNotFoundError:  # pragma: no cover
    speed_scale_from_spindle_radius = None
try:
    from .trace import detect_noon12_angle_debug
except ImportError:  # pragma: no cover
    detect_noon12_angle_debug = None
from .trace import extract_needle_mask, extract_speed_trace, estimate_speed_scale_radii, fixed_speed_band_from_outer_radius


def make_red_suppress_mask(bgr_u8: np.ndarray) -> np.ndarray:
    """HSVで赤補助線（80/100km/h）のマスクを生成する。"""
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    # 赤は色相が0付近と180付近の2範囲
    mask1 = cv2.inRange(hsv, (0, 60, 60), (35, 255, 255))
    mask2 = cv2.inRange(hsv, (165, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)
    # 少し膨張させて赤線の周辺も抑制
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.dilate(red_mask, kernel, iterations=2)
    return red_mask


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

    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)

    parser.add_argument(
        "--calibration-json",
        type=str,
        default=None,
        help="calibration.jsonのパス。指定するとcenter-x/y/outer-radiusを自動注入する。",
    )

    parser.add_argument("--angle-step", type=float, default=1.0)
    parser.add_argument("--zero-angle-deg", type=float, default=-90.0, help="00:00 direction. Default is 12 o'clock")
    parser.add_argument("--clockwise", action="store_true")

    parser.add_argument("--r-min", type=int, default=None)
    parser.add_argument("--r-max", type=int, default=None)
    parser.add_argument("--outer-radius", type=float, default=None)

    parser.add_argument("--speed-band-r-in-ratio", type=float, default=0.55)
    parser.add_argument("--speed-band-r-out-ratio", type=float, default=0.86)

    parser.add_argument("--speed-vmax-kmh", type=float, default=None)
    parser.add_argument("--needle-speed-vmax-kmh", type=float, default=120.0)

    parser.add_argument("--time-ring-out-margin", type=int, default=10)
    parser.add_argument("--time-ring-thickness", type=int, default=90)
    parser.add_argument("--time-ring-r-out-ratio", type=float, default=None)
    parser.add_argument("--time-ring-thickness-ratio", type=float, default=None)
    parser.add_argument("--time-ring-refine-outer-radius", action="store_true")
    parser.add_argument("--time-ring-polar-width", type=int, default=2400)
    parser.add_argument("--time-ring-angle-flip", action="store_true")
    parser.add_argument("--time-ring-use-green", action="store_true")
    parser.add_argument("--time-ring-template-12", type=str, default=None)
    parser.add_argument("--time-ring-template-12-from-mark", type=str, default=None)
    parser.add_argument("--time-ring-template-12-out", type=str, default=None)
    parser.add_argument("--time-ring-min-score", type=float, default=0.35)
    parser.add_argument("--manual-noon-angle-deg", type=float, default=None)
    parser.add_argument(
        "--noon-from-needle-binary",
        action="store_true",
        help="Fallback: estimate 00:00 direction from the white sector in needle_binary, then set 12:00 at +180deg",
    )

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

    if args.calibration_json is not None:
        with open(args.calibration_json, "r") as f:
            calib = json.load(f)
        args.center_x = calib["center_x"]
        args.center_y = calib["center_y"]
        args.outer_radius = calib["outer_radius"]
    else:
        # ファイル名からキャリブレーション情報を自動抽出
        # 例: tacho_cx540_cy960_r480.jpg
        import re

        m = re.search(r"cx(\d+)_cy(\d+)_r(\d+)", args.input)
        if m and args.center_x is None:
            args.center_x = int(m.group(1))
            args.center_y = int(m.group(2))
            args.outer_radius = int(m.group(3))

    loaded = load_image(args.input, pdf_page_index=args.pdf_page, pdf_dpi=args.pdf_dpi)
    gray = rgb_to_gray_u8(loaded.rgb)
    bgr_image = cv2.cvtColor(loaded.rgb, cv2.COLOR_RGB2BGR)
    time_gray = gray
    if bool(args.time_ring_use_green):
        rgb = loaded.rgb
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)
        # Emphasize green ink relative to background.
        v = (g - 0.55 * r - 0.55 * b)
        v = (v - float(np.percentile(v, 1.0)))
        v = v / float(max(1e-6, np.percentile(v, 99.0)))
        v = (v * 255.0).clip(0, 255).astype(np.uint8)
        time_gray = v
    est = estimate_center_auto(gray)
    red_aux = None
    if (args.center_x is not None) or (args.center_y is not None):
        if args.center_x is None or args.center_y is None:
            raise ValueError("--center-x and --center-y must be provided together")
        dbg = dict(est.debug) if isinstance(est.debug, dict) else {}
        dbg["manual_center"] = {"x": float(args.center_x), "y": float(args.center_y)}
        est = replace(
            est,
            center_x=float(args.center_x),
            center_y=float(args.center_y),
            center_uncertain=False,
            debug=dbg,
        )
    else:
        red_aux = detect_center_from_red_aux_rings(bgr_image)
        if red_aux is not None:
            dbg = dict(est.debug) if isinstance(est.debug, dict) else {}
            dbg["red_aux_rings"] = {
                "center": {"x": float(red_aux.center_x), "y": float(red_aux.center_y)},
                "r80": float(red_aux.r80),
                "r100": float(red_aux.r100),
                "debug": dict(red_aux.debug) if isinstance(red_aux.debug, dict) else {},
            }
            est = replace(
                est,
                center_x=float(red_aux.center_x),
                center_y=float(red_aux.center_y),
                center_uncertain=False,
                debug=dbg,
            )

    if args.outer_radius is not None:
        est = replace(est, outer_radius=float(args.outer_radius))

    noon_det = None
    theta_noon_deg = None
    if args.manual_noon_angle_deg is not None:
        theta_noon_deg = float(args.manual_noon_angle_deg) % 360.0
    else:
        if detect_noon12_angle_debug is None:
            if args.time_ring_template_12_from_mark or args.time_ring_template_12:
                raise ImportError("detect_noon12_angle_debug is unavailable in this workspace")
            if bool(args.debug):
                print(
                    "warning: detect_noon12_angle_debug is unavailable; skipping auto noon detection",
                    file=sys.stderr,
                )
        
        templ_bin = None
        if args.time_ring_template_12_from_mark:
            if est.outer_radius is None:
                raise ValueError("--time-ring-template-12-from-mark requires a valid outer_radius")

            # First, compute polar band/bin without a template.
            if detect_noon12_angle_debug is None:
                raise ImportError("detect_noon12_angle_debug is unavailable in this workspace")
            noon_no_template = detect_noon12_angle_debug(
                time_gray,
                refine_gray_u8=gray,
                center_xy=(est.center_x, est.center_y),
                outer_radius=float(est.outer_radius),
                time_ring_out_margin=int(args.time_ring_out_margin),
                time_ring_thickness=int(args.time_ring_thickness),
                time_ring_r_out_ratio=args.time_ring_r_out_ratio,
                time_ring_thickness_ratio=args.time_ring_thickness_ratio,
                time_ring_refine_outer_radius=bool(args.time_ring_refine_outer_radius),
                polar_width=int(args.time_ring_polar_width),
                template_12_bin_u8=None,
                min_score=float(args.time_ring_min_score),
                angle_flip=bool(args.time_ring_angle_flip),
            )

            dbg0 = noon_no_template.debug if isinstance(noon_no_template.debug, dict) else {}
            r_in = int(dbg0.get("r_in", 0))
            r_out = int(dbg0.get("r_out", 0))
            W = int(dbg0.get("polar_width", 0))
            if r_in <= 0 or r_out <= r_in + 2 or W <= 0:
                raise ValueError("failed to get valid polar band geometry for template extraction")

            mark_path = str(args.time_ring_template_12_from_mark)
            mark_bgr = cv2.imread(mark_path, cv2.IMREAD_COLOR)
            if mark_bgr is None:
                raise ValueError(f"failed to read --time-ring-template-12-from-mark: {mark_path}")

            hsv = cv2.cvtColor(mark_bgr, cv2.COLOR_BGR2HSV)
            # Do not assume the marker color is pure red. In practice, depending on the editor/
            # color profile, the rectangle may appear with a different hue.
            # We instead pick highly saturated pixels.
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            mark_mask = (((s.astype(np.int16) >= 120) & (v.astype(np.int16) >= 120))).astype(np.uint8) * 255
            mark_mask = cv2.morphologyEx(mark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

            # Restrict to the time-ring annulus to avoid picking red printed rings/text.
            mh, mw = mark_mask.shape[:2]
            yy, xx = np.indices((mh, mw), dtype=np.float32)
            rr0 = np.hypot(xx - float(est.center_x), yy - float(est.center_y))
            r_lo = max(0.0, float(r_in) - 40.0)
            r_hi = float(r_out) + 40.0
            ann = ((rr0 >= r_lo) & (rr0 <= r_hi)).astype(np.uint8) * 255
            mark_mask = cv2.bitwise_and(mark_mask, ann)

            # Keep only the largest connected component (expected to be the rectangle).
            nlab, lab, stats, _ = cv2.connectedComponentsWithStats((mark_mask > 0).astype(np.uint8), connectivity=8)
            if nlab <= 1:
                raise ValueError("mark rectangle not found in marked image (no components)")
            areas = stats[1:, cv2.CC_STAT_AREA]
            k = int(np.argmax(areas)) + 1
            if int(stats[k, cv2.CC_STAT_AREA]) < 200:
                raise ValueError("mark rectangle not found in marked image (component too small)")
            mark_mask = (lab == k).astype(np.uint8) * 255
            # Thicken lines so warpPolar keeps a usable bbox even if the rectangle is drawn thin.
            mark_mask = cv2.dilate(mark_mask, np.ones((5, 5), np.uint8), iterations=1)

            cx = float(est.center_x)
            cy = float(est.center_y)
            polar_red = cv2.warpPolar(
                mark_mask,
                (int(W), int(r_out)),
                (cx, cy),
                maxRadius=float(r_out),
                flags=cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
            )
            polar_red_band = polar_red[int(r_in) : int(r_out), :]
            ys, xs = np.where(polar_red_band > 0)
            if xs.size < 50:
                raise ValueError("mark rectangle not found in marked image (too few pixels after warpPolar)")

            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())

            # Expand the bbox a bit to avoid clipping the digit strokes.
            bw = max(0, x1 - x0 + 1)
            bh = max(0, y1 - y0 + 1)
            pad = int(min(14, max(4, min(bw, bh) // 6)))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(int(W) - 1, x1 + pad)
            y1 = min(int(r_out - r_in) - 1, y1 + pad)
            if x1 <= x0 + 4 or y1 <= y0 + 4:
                raise ValueError("mark bbox too small; cannot build template")

            templ_gray = noon_no_template.polar_band[y0 : y1 + 1, x0 : x1 + 1].copy()
            if templ_gray.ndim != 2 or templ_gray.dtype != np.uint8:
                raise ValueError("failed to crop grayscale template region")

            # Save as grayscale template. trace.py will automatically choose grayscale matching
            # (template has >2 unique values) which preserves digit shapes better than binarization.
            p2 = cv2.normalize(templ_gray, None, 0, 255, cv2.NORM_MINMAX)
            templ_bin = p2.astype(np.uint8)
            out_path = str(args.time_ring_template_12_out) if args.time_ring_template_12_out else None
            if not out_path:
                raise ValueError("--time-ring-template-12-from-mark requires --time-ring-template-12-out")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out_path, templ_bin)

        if args.time_ring_template_12:
            tpath = str(args.time_ring_template_12)
            templ = cv2.imread(tpath, cv2.IMREAD_GRAYSCALE)
            if templ is None:
                raise ValueError(f"failed to read --time-ring-template-12: {tpath}")
            templ_bin = templ

        if (detect_noon12_angle_debug is not None) and (est.outer_radius is not None):
            noon_det = detect_noon12_angle_debug(
                time_gray,
                refine_gray_u8=gray,
                center_xy=(est.center_x, est.center_y),
                outer_radius=float(est.outer_radius),
                time_ring_out_margin=int(args.time_ring_out_margin),
                time_ring_thickness=int(args.time_ring_thickness),
                time_ring_r_out_ratio=args.time_ring_r_out_ratio,
                time_ring_thickness_ratio=args.time_ring_thickness_ratio,
                time_ring_refine_outer_radius=bool(args.time_ring_refine_outer_radius),
                polar_width=int(args.time_ring_polar_width),
                template_12_bin_u8=templ_bin,
                min_score=float(args.time_ring_min_score),
                angle_flip=bool(args.time_ring_angle_flip),
            )
            theta_noon_deg = noon_det.theta_noon_deg

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

    red_mask = make_red_suppress_mask(bgr_image)
    trace = extract_speed_trace(
        gray,
        center_xy=(est.center_x, est.center_y),
        r_min=r_min,
        r_max=r_max,
        angle_step_deg=float(args.angle_step),
        zero_angle_deg=float(args.zero_angle_deg),
        clockwise=bool(args.clockwise),
        red_suppress_mask=red_mask,
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

    if not bool(args.use_fixed_speed_scale):
        speed_scale_mode = "skip"

    if speed_scale_mode == "spindle":
        if speed_scale_from_spindle_radius is None:
            if bool(args.debug):
                print(
                    "warning: speed_scale module is missing; skipping spindle scale mode",
                    file=sys.stderr,
                )
            speed_scale_mode = "skip"
        spindle_r = None
        if isinstance(est.debug, dict) and est.debug.get("spindle_detected"):
            spindle_r = est.debug.get("spindle_radius")
        if (speed_scale_mode == "spindle") and (spindle_r is not None):
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

    # Fallback: estimate time anchor from cutout/white sector in needle binary.
    needle_midnight_deg = None
    needle_anchor_dbg = None
    if theta_noon_deg is None and bool(args.noon_from_needle_binary):
        m = needle.binary_mask_u8
        if m is not None and m.ndim == 2 and m.dtype == np.uint8:
            ys, xs = np.where(m > 0)
            if xs.size >= 200:
                cx = float(est.center_x)
                cy = float(est.center_y)
                dx = xs.astype(np.float32) - cx
                dy = ys.astype(np.float32) - cy
                ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
                # Use 1deg histogram, smoothed, pick dominant direction.
                bins = np.floor(ang).astype(np.int32) % 360
                hist = np.bincount(bins, minlength=360).astype(np.float32)
                hist_s = cv2.GaussianBlur(hist.reshape(1, -1), (0, 0), 2.0).reshape(-1)
                k = int(np.argmax(hist_s))
                needle_midnight_deg = float(k)
                theta_noon_deg = float((needle_midnight_deg + 180.0) % 360.0)
                needle_anchor_dbg = {
                    "method": "needle_binary_white_sector",
                    "white_pixel_count": int(xs.size),
                    "midnight_angle_deg": float(needle_midnight_deg),
                    "noon_angle_deg": float(theta_noon_deg),
                    "hist_peak": int(k),
                    "hist_peak_value": float(hist_s[k]),
                }

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

    if theta_noon_deg is not None:
        out["noon12_angle_deg"] = float(theta_noon_deg)
        method = "noon12" if noon_det is not None and noon_det.theta_noon_deg is not None else "needle_binary"
        out["time_angle_correction"] = {"method": str(method), "anchor_time": "12:00:00", "anchor_angle_deg": float(theta_noon_deg)}
    elif args.manual_noon_angle_deg is not None:
        out["noon12_angle_deg"] = float(args.manual_noon_angle_deg) % 360.0
        out["time_angle_correction"] = {"method": "manual", "anchor_time": "12:00:00", "anchor_angle_deg": float(out["noon12_angle_deg"])}
    else:
        out["time_angle_correction"] = {"method": "fallback", "reason": "no_noon12_angle"}

    if bool(args.debug):
        out["center_debug"] = dict(est.debug) if isinstance(est.debug, dict) else {}
        if noon_det is not None:
            out["noon12_debug"] = dict(noon_det.debug) if isinstance(noon_det.debug, dict) else {}
        if needle_anchor_dbg is not None:
            out["needle_time_anchor_debug"] = dict(needle_anchor_dbg)
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
        debug_polar_band_path = debug_image_path.with_name(debug_image_path.stem + "_polar_band.png")
        debug_polar_band_bin_path = debug_image_path.with_name(debug_image_path.stem + "_polar_band_bin.png")
        debug_match_12_path = debug_image_path.with_name(debug_image_path.stem + "_match_12.png")
        debug_noon_overlay_path = debug_image_path.with_name(debug_image_path.stem + "_noon_overlay.png")

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
        ann_r_in = int(r_min)
        ann_r_out = int(r_max)
        if "needle_roi" in out and isinstance(out["needle_roi"], dict):
            nr_in = int(out["needle_roi"].get("r_in", 0))
            nr_out = int(out["needle_roi"].get("r_out", 0))
            if nr_in > 0 and nr_out > nr_in:
                ann_r_in = nr_in
                ann_r_out = nr_out
        ann_color = (0, 255, 255)
        ann_thickness = 2
        cv2.circle(bgr_ann, (int(round(cx)), int(round(cy))), int(round(ann_r_in)), ann_color, int(ann_thickness))
        cv2.circle(bgr_ann, (int(round(cx)), int(round(cy))), int(round(ann_r_out)), ann_color, int(ann_thickness))

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

        if noon_det is not None:
            cv2.imwrite(str(debug_polar_band_path), noon_det.polar_band)
            cv2.imwrite(str(debug_polar_band_bin_path), noon_det.polar_band_bin)
            if noon_det.match_vis is not None:
                cv2.imwrite(str(debug_match_12_path), noon_det.match_vis)

        # noon overlay (even if using manual override)
        if theta_noon_deg is not None:
            bgr_noon = bgr_base.copy()
            draw_center_and_outer(bgr_noon)
            theta = float(theta_noon_deg)
            rad = np.deg2rad(float(theta))
            R = float(est.outer_radius) if est.outer_radius is not None else float(min(bgr_noon.shape[0], bgr_noon.shape[1]) * 0.45)
            x1 = float(cx) + float(R) * float(np.cos(rad))
            y1 = float(cy) + float(R) * float(np.sin(rad))
            cv2.line(bgr_noon, (int(round(cx)), int(round(cy))), (int(round(x1)), int(round(y1))), (0, 0, 255), 3)
            cv2.putText(
                bgr_noon,
                f"NOON12 {theta:.1f}deg",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(debug_noon_overlay_path), bgr_noon)

        cv2.imwrite(str(debug_image_path), bgr)
        out["debug_outer_circle"] = str(debug_outer_path)
        out["debug_speed_band_annulus"] = str(debug_annulus_path)
        out["debug_speed_scale_check"] = str(debug_speed_scale_path)
        out["debug_needle_overlay"] = str(debug_needle_path)
        out["debug_needle_roi"] = str(debug_needle_roi_path)
        out["debug_needle_roi_mask"] = str(debug_needle_roi_mask_path)
        out["debug_needle_binary"] = str(debug_needle_binary_path)
        if noon_det is not None:
            out["debug_polar_band"] = str(debug_polar_band_path)
            out["debug_polar_band_bin"] = str(debug_polar_band_bin_path)
            if noon_det.match_vis is not None:
                out["debug_match_12"] = str(debug_match_12_path)
        if theta_noon_deg is not None:
            out["debug_noon_overlay"] = str(debug_noon_overlay_path)
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
            ann_color = (0, 255, 255)
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r_min)), ann_color, 2)
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r_max)), ann_color, 2)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)
            out["debug_trace_mask"] = str(mask_path)

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
