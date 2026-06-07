"""Image loading and preview/original coordinate helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def lanczos_resampling() -> int:
    """Return Pillow's LANCZOS enum across Pillow versions."""
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.LANCZOS)
    return int(Image.LANCZOS)


def nearest_resampling() -> int:
    """Return Pillow's NEAREST enum across Pillow versions."""
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.NEAREST)
    return int(Image.NEAREST)


def compute_preview_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Compute a preview size that preserves aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}.")
    if max_side <= 0:
        return width, height
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = float(max_side) / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_rgb_preview(image: np.ndarray, max_side: int) -> np.ndarray:
    """Return a low-resolution RGB preview for ``image``."""
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    preview_size = compute_preview_size(width, height, max_side)
    if preview_size == (width, height):
        return arr.copy()
    return np.asarray(Image.fromarray(arr).resize(preview_size, lanczos_resampling()))


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to ``(width, height)`` with nearest-neighbor sampling."""
    mask_image = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L")
    return np.asarray(mask_image.resize(size, nearest_resampling())) > 127


def relative_path(path: Path, base_dir: Path) -> str:
    """Return a path relative to ``base_dir`` when possible."""
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), base_dir.resolve())


def unique_output_path(directory: Path, stem: str, suffix: str, ext: str) -> Path:
    """Return a non-existing output path in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"{stem}{suffix}{ext}"
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = directory / f"{stem}{suffix}_{idx}{ext}"
        if not candidate.exists():
            return candidate
        idx += 1


@dataclass
class ImageView:
    """Original-resolution image plus its display preview."""

    path: Path
    image: np.ndarray
    preview: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def preview_width(self) -> int:
        return int(self.preview.shape[1])

    @property
    def preview_height(self) -> int:
        return int(self.preview.shape[0])

    @property
    def is_downsampled(self) -> bool:
        return self.preview_width != self.width or self.preview_height != self.height

    def preview_to_image_xy(self, x: float, y: float) -> tuple[float, float]:
        """Map preview pixel coordinates to original image coordinates."""
        if self.preview_width == self.width and self.preview_height == self.height:
            out_x = x
            out_y = y
        else:
            out_x = (x + 0.5) * (self.width / self.preview_width) - 0.5
            out_y = (y + 0.5) * (self.height / self.preview_height) - 0.5
        return (
            float(np.clip(out_x, 0.0, self.width - 1.0)),
            float(np.clip(out_y, 0.0, self.height - 1.0)),
        )

    def image_to_preview_points(self, points: np.ndarray) -> np.ndarray:
        """Map original image points to preview coordinates."""
        arr = np.asarray(points, dtype=np.float64)
        if self.preview_width == self.width and self.preview_height == self.height:
            out = arr.copy()
        else:
            x = (arr[:, 0] + 0.5) * (self.preview_width / self.width) - 0.5
            y = (arr[:, 1] + 0.5) * (self.preview_height / self.height) - 0.5
            out = np.column_stack((x, y))
        return out.astype(np.float32)

    def image_to_preview_box(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Map an original image box to preview coordinates."""
        points = np.asarray([[box[0], box[1]], [box[2], box[3]]], dtype=np.float64)
        mapped = self.image_to_preview_points(points)
        return float(mapped[0, 0]), float(mapped[0, 1]), float(mapped[1, 0]), float(mapped[1, 1])

    def preview_box_to_image(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Map a preview box to original image coordinates."""
        x0, y0 = self.preview_to_image_xy(box[0], box[1])
        x1, y1 = self.preview_to_image_xy(box[2], box[3])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def mask_to_preview(self, mask: np.ndarray) -> np.ndarray:
        """Resize an original-resolution boolean mask into preview space."""
        if mask.shape == (self.preview_height, self.preview_width):
            return mask.astype(bool, copy=True)
        return resize_mask(mask, (self.preview_width, self.preview_height))

    def mask_to_image(self, mask: np.ndarray) -> np.ndarray:
        """Resize a preview-resolution boolean mask into original image space."""
        if mask.shape == (self.height, self.width):
            return mask.astype(bool, copy=True)
        return resize_mask(mask, (self.width, self.height))


def load_image_view(path: str | Path, preview_max_side: int) -> ImageView:
    """Load an RGB image and create its preview."""
    image_path = Path(path).expanduser().resolve()
    with Image.open(image_path) as img:
        image = np.asarray(img.convert("RGB"))
    return ImageView(
        path=image_path,
        image=image,
        preview=resize_rgb_preview(image, preview_max_side),
    )
