from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

import fitz

from .analyzer import (
    detect_tachograph_circle,
    load_image_bgr,
    normalize_image_by_affine,
    unwrap_speed_and_distance_rings,
)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic"}


def _render_pdf_pages_bgr(
    pdf_path: Path,
    *,
    dpi: int,
    max_pages: Optional[int],
) -> Iterable[Tuple[str, np.ndarray]]:
    doc = fitz.open(str(pdf_path))
    try:
        n = doc.page_count
        if max_pages is not None:
            n = min(n, int(max_pages))

        zoom = float(dpi) / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            w = int(pix.width)
            h = int(pix.height)
            c = int(pix.n)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            if c == 1:
                img = arr.reshape((h, w))
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = arr.reshape((h, w, c))
                # PyMuPDF returns RGB
                if c >= 3:
                    bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
                else:
                    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            yield f"{pdf_path.stem}_p{i:03d}", bgr
    finally:
        doc.close()


def _iter_inputs(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for f in sorted(pp.rglob("*")):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in _IMAGE_EXTS or ext == ".pdf":
                    out.append(f)
        else:
            out.append(pp)
    return out


def _default_out_dir(first_input: str) -> Path:
    p = Path(first_input)
    if p.is_dir():
        return p / "opencv_out"
    return p.with_suffix("").with_name(p.stem + "_opencv_out")


def _safe_name(p: Path) -> str:
    return p.stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input image/PDF path(s) or a directory")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--prefer-hough", type=int, default=1)
    parser.add_argument("--wobble-refine", type=int, default=0)
    parser.add_argument("--write-normalized", type=int, default=1)
    parser.add_argument("--write-polar", type=int, default=1)
    parser.add_argument("--write-bands", type=int, default=1)
    parser.add_argument("--normalize", type=int, default=1)
    parser.add_argument("--polar-width", type=int, default=1440)

    parser.add_argument("--polar-theta-start-frac", type=float, default=0.5)
    parser.add_argument("--polar-theta-end-frac", type=float, default=1.0)

    parser.add_argument("--speed-r-in-frac", type=float, default=0.80)
    parser.add_argument("--speed-r-out-frac", type=float, default=0.95)
    parser.add_argument("--distance-r-in-frac", type=float, default=0.58)
    parser.add_argument("--distance-r-out-frac", type=float, default=0.76)

    parser.add_argument("--pdf-dpi", type=int, default=200)
    parser.add_argument("--pdf-max-pages", type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.out_dir) if args.out_dir else _default_out_dir(args.inputs[0])
    out_root.mkdir(parents=True, exist_ok=True)

    items = _iter_inputs(list(args.inputs))
    results = []
    summary_rows: List[dict] = []

    for p in items:
        ext = p.suffix.lower()
        if ext == ".pdf":
            for page_name, bgr in _render_pdf_pages_bgr(
                p,
                dpi=int(args.pdf_dpi),
                max_pages=args.pdf_max_pages,
            ):
                out_dir = out_root / _safe_name(p) / page_name
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    payload = _process_one(
                        name=page_name,
                        base_name=str(page_name),
                        img=bgr,
                        out_dir=out_dir,
                        prefer_hough=bool(int(args.prefer_hough)),
                        wobble_refine=bool(int(args.wobble_refine)),
                        write_normalized=bool(int(args.write_normalized)),
                        write_polar=bool(int(args.write_polar)),
                        write_bands=bool(int(args.write_bands)),
                        normalize=bool(int(args.normalize)),
                        polar_width=int(args.polar_width),
                        polar_theta_start_frac=float(args.polar_theta_start_frac),
                        polar_theta_end_frac=float(args.polar_theta_end_frac),
                        speed_r_in_frac=float(args.speed_r_in_frac),
                        speed_r_out_frac=float(args.speed_r_out_frac),
                        distance_r_in_frac=float(args.distance_r_in_frac),
                        distance_r_out_frac=float(args.distance_r_out_frac),
                    )
                    results.append(payload)
                    summary_rows.append(_summary_row(input_path=str(p), name=page_name, payload=payload))
                except Exception as e:
                    summary_rows.append(
                        {
                            "input_path": str(p),
                            "name": page_name,
                            "out_dir": str(out_dir),
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
        else:
            if not p.exists():
                continue
            if ext not in _IMAGE_EXTS:
                continue
            out_dir = out_root / _safe_name(p)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                img = load_image_bgr(str(p))
                payload = _process_one(
                    name=str(p),
                    base_name=str(_safe_name(p)),
                    img=img,
                    out_dir=out_dir,
                    prefer_hough=bool(int(args.prefer_hough)),
                    wobble_refine=bool(int(args.wobble_refine)),
                    write_normalized=bool(int(args.write_normalized)),
                    write_polar=bool(int(args.write_polar)),
                    write_bands=bool(int(args.write_bands)),
                    normalize=bool(int(args.normalize)),
                    polar_width=int(args.polar_width),
                    polar_theta_start_frac=float(args.polar_theta_start_frac),
                    polar_theta_end_frac=float(args.polar_theta_end_frac),
                    speed_r_in_frac=float(args.speed_r_in_frac),
                    speed_r_out_frac=float(args.speed_r_out_frac),
                    distance_r_in_frac=float(args.distance_r_in_frac),
                    distance_r_out_frac=float(args.distance_r_out_frac),
                )
                results.append(payload)
                summary_rows.append(_summary_row(input_path=str(p), name=str(p), payload=payload))
            except Exception as e:
                summary_rows.append(
                    {
                        "input_path": str(p),
                        "name": str(p),
                        "out_dir": str(out_dir),
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

    (out_root / "batch_summary.json").write_text(
        json.dumps({"status": "ok", "count": len(results), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (out_root / "summary.json").write_text(
        json.dumps({"status": "ok", "count": len(summary_rows), "items": summary_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_summary_csv(out_root / "summary.csv", summary_rows)

    print(json.dumps({"out_dir": str(out_root), "count": len(results)}, ensure_ascii=False))
    return 0


def _summary_row(*, input_path: str, name: str, payload: dict) -> dict:
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    ellipse = debug.get("ellipse") if isinstance(debug.get("ellipse"), dict) else {}
    ellipse_status = ellipse.get("status")
    normalized_ok = bool(ellipse_status == "ok")
    return {
        "input_path": str(input_path),
        "name": str(name),
        "out_dir": str(payload.get("out_dir")),
        "polar_path": str(payload.get("polar_path", "")),
        "ok": True,
        "cx": float(payload.get("cx")),
        "cy": float(payload.get("cy")),
        "r": float(payload.get("r")),
        "method": str(payload.get("method")),
        "normalized_ok": normalized_ok,
        "ellipse_status": ellipse_status,
    }


def _write_summary_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "input_path",
        "name",
        "out_dir",
        "polar_path",
        "ok",
        "cx",
        "cy",
        "r",
        "method",
        "normalized_ok",
        "ellipse_status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            w.writerow(out)


def _process_one(
    *,
    name: str,
    base_name: str,
    img: np.ndarray,
    out_dir: Path,
    prefer_hough: bool,
    wobble_refine: bool,
    write_normalized: bool,
    write_polar: bool,
    write_bands: bool,
    normalize: bool,
    polar_width: int,
    polar_theta_start_frac: float,
    polar_theta_end_frac: float,
    speed_r_in_frac: float,
    speed_r_out_frac: float,
    distance_r_in_frac: float,
    distance_r_out_frac: float,
) -> dict:
    res = detect_tachograph_circle(img, prefer_hough=prefer_hough, wobble_refine=bool(wobble_refine))

    annotated = img.copy()
    cv2.circle(
        annotated,
        (int(round(res.cx)), int(round(res.cy))),
        int(round(res.r)),
        (0, 255, 0),
        3,
    )
    cv2.drawMarker(
        annotated,
        (int(round(res.cx)), int(round(res.cy))),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=40,
        thickness=3,
    )
    cv2.imwrite(str(out_dir / "circle_detect.png"), annotated)

    if getattr(res, "debug_images", None):
        dbg_dir = out_dir / "debug_images"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        for dbg_name, im in res.debug_images.items():
            if not isinstance(dbg_name, str):
                continue
            safe = dbg_name.replace("/", "_")
            cv2.imwrite(str(dbg_dir / safe), im)

    if write_normalized:
        norm_img = normalize_image_by_affine(img, affine_2x3=res.normalize_affine_2x3)
        cv2.imwrite(str(out_dir / "normalized.png"), norm_img)

    unwrap = unwrap_speed_and_distance_rings(
        img,
        circle=res,
        normalize=normalize,
        polar_width=int(polar_width),
        polar_theta_start_frac=float(polar_theta_start_frac),
        polar_theta_end_frac=float(polar_theta_end_frac),
        speed_r_in_frac=float(speed_r_in_frac),
        speed_r_out_frac=float(speed_r_out_frac),
        distance_r_in_frac=float(distance_r_in_frac),
        distance_r_out_frac=float(distance_r_out_frac),
    )

    if write_polar:
        polar_filename = f"{str(base_name)}_polar.png"
        cv2.imwrite(str(out_dir / polar_filename), unwrap.polar_bgr)
        polar_path = str(out_dir / polar_filename)
    else:
        polar_path = ""
    if write_bands:
        cv2.imwrite(str(out_dir / "band_speed.png"), unwrap.speed_band_bgr)
        cv2.imwrite(str(out_dir / "band_distance.png"), unwrap.distance_band_bgr)

    payload = {
        "name": name,
        "out_dir": str(out_dir),
        "polar_path": polar_path,
        "cx": res.cx,
        "cy": res.cy,
        "r": res.r,
        "method": res.method,
        "unwrap": unwrap.params,
        "debug": res.debug,
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
