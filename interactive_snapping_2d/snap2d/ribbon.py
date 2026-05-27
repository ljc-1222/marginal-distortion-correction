"""Ribbon search-space construction for 2D snapping."""

from __future__ import annotations

import numpy as np


_EPS = 1e-9


def build_ribbon_candidates(
    centers: np.ndarray,
    normals: np.ndarray,
    width_px: float,
    offset_step_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build candidate points ``center_i + offset_j * normal_i``."""
    centers = np.asarray(centers, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"centers must have shape (N, 2); got {centers.shape}.")
    if normals.shape != centers.shape:
        raise ValueError(f"normals must have shape {centers.shape}; got {normals.shape}.")
    if width_px < 0:
        raise ValueError(f"width_px must be non-negative; got {width_px}.")
    if offset_step_px <= 0:
        raise ValueError(f"offset_step_px must be positive; got {offset_step_px}.")

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    safe_normals = normals / np.maximum(norm, _EPS)
    offsets = np.arange(-float(width_px), float(width_px) + 0.5 * float(offset_step_px), float(offset_step_px), dtype=np.float32)
    if not np.any(np.isclose(offsets, 0.0)):
        offsets = np.sort(np.concatenate([offsets, np.asarray([0.0], dtype=np.float32)]))
    candidates = centers[:, None, :] + safe_normals[:, None, :] * offsets[None, :, None]
    return candidates.astype(np.float32), offsets.astype(np.float32)
