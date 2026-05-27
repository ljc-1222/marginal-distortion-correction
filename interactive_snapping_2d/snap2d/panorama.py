"""Equirectangular panorama seam handling for 2D strokes."""

from __future__ import annotations

import numpy as np


def unwrap_panorama_stroke(image: np.ndarray, stroke: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Horizontally unwrap a panorama stroke into a three-tile image."""
    arr = np.asarray(image)
    points = np.asarray(stroke, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"image must have at least two dimensions; got {arr.shape}.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"stroke must have shape (N, 2); got {points.shape}.")

    height, width = arr.shape[:2]
    if width <= 0:
        raise ValueError("Panorama width must be positive.")

    extended = np.concatenate([arr, arr, arr], axis=1)
    unwrapped = points.astype(np.float32, copy=True)
    unwrapped[:, 0] = np.mod(unwrapped[:, 0], float(width))
    unwrapped[0, 0] += float(width)
    for idx in range(1, len(unwrapped)):
        x = float(unwrapped[idx, 0])
        previous = float(unwrapped[idx - 1, 0])
        shift = round((previous - x) / float(width))
        candidates = np.asarray([x + (shift + delta) * width for delta in (-1, 0, 1)], dtype=np.float32)
        best = int(np.argmin(np.abs(candidates - previous)))
        unwrapped[idx, 0] = candidates[best]

    info = {
        "original_width": int(width),
        "original_height": int(height),
        "extended_width": int(width * 3),
        "x_offset": int(width),
    }
    return extended, unwrapped.astype(np.float32), info


def wrap_panorama_points(points: np.ndarray, unwrap_info: dict) -> np.ndarray:
    """Wrap unwrapped panorama points back to original image coordinates."""
    wrapped = np.asarray(points, dtype=np.float32).copy()
    width = int(unwrap_info["original_width"])
    height = int(unwrap_info.get("original_height", 0))
    wrapped[:, 0] = np.mod(wrapped[:, 0], float(width))
    if height > 0:
        wrapped[:, 1] = np.clip(wrapped[:, 1], 0.0, float(height - 1))
    return wrapped
