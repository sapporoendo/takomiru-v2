from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LoadedImage:
    rgb: np.ndarray
    source_path: str


def _pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    return arr


def load_image(path: str, *, pdf_page_index: int = 0, pdf_dpi: int = 300) -> LoadedImage:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suffix = p.suffix.lower()

    if suffix == ".pdf":
        import fitz  # PyMuPDF

        doc = fitz.open(str(p))
        if doc.page_count <= pdf_page_index:
            raise ValueError(f"PDF has only {doc.page_count} pages")
        page = doc.load_page(pdf_page_index)
        zoom = pdf_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return LoadedImage(rgb=_pil_to_rgb_array(img), source_path=str(p))

    if suffix in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()

    img = Image.open(str(p))
    return LoadedImage(rgb=_pil_to_rgb_array(img), source_path=str(p))


def rgb_to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    gray = (0.299 * r + 0.587 * g + 0.114 * b).round().clip(0, 255).astype(np.uint8)
    return gray


def crop_square_around(center_xy: Tuple[float, float], side: int, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img.shape[:2]
    cx, cy = center_xy
    half = side // 2

    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + side
    y1 = y0 + side

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)) + (() if img.ndim == 2 else ((0, 0),)),
            mode="constant",
            constant_values=0,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    cropped = img[y0:y1, x0:x1]
    return cropped, (x0, y0)
