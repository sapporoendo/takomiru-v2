from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .center import estimate_center_auto
from .io import load_image, rgb_to_gray_u8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Image/PDF path")
    parser.add_argument("--pdf-dpi", type=int, default=300)
    parser.add_argument("--pdf-page", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-image", default=None)
    args = parser.parse_args()

    loaded = load_image(args.input, pdf_page_index=args.pdf_page, pdf_dpi=args.pdf_dpi)
    gray = rgb_to_gray_u8(loaded.rgb)
    bgr = cv2.cvtColor(loaded.rgb, cv2.COLOR_RGB2BGR)
    est = estimate_center_auto(gray, bgr_u8=bgr)

    out = {
        "source": str(Path(loaded.source_path)),
        "center": {"x": est.center_x, "y": est.center_y},
        "outer_radius": est.outer_radius,
        "inner_radius": est.inner_radius,
        "center_uncertain": bool(est.center_uncertain),
    }
    if args.debug:
        out["debug"] = est.debug
        debug_image_path = (
            Path(args.debug_image)
            if args.debug_image
            else Path(loaded.source_path).with_name(Path(loaded.source_path).stem + "_center_debug.png")
        )

        cx = int(round(est.center_x))
        cy = int(round(est.center_y))
        cv2.drawMarker(bgr, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=3)

        outer_center = est.debug.get("outer_center") if isinstance(est.debug, dict) else None
        inner_center = est.debug.get("inner_center") if isinstance(est.debug, dict) else None

        if est.outer_radius is not None:
            cv2.circle(bgr, (cx, cy), int(round(float(est.outer_radius))), (255, 0, 0), 3)

        if inner_center is not None and est.inner_radius is not None:
            ix, iy = int(round(float(inner_center[0]))), int(round(float(inner_center[1])))
            cv2.circle(bgr, (ix, iy), int(round(float(est.inner_radius))), (0, 0, 255), 3)
            cv2.circle(bgr, (ix, iy), 6, (0, 0, 255), -1)

        label = f"center_uncertain={bool(est.center_uncertain)}"
        cv2.putText(bgr, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

        debug_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_image_path), bgr)
        out["debug_image"] = str(debug_image_path)

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
