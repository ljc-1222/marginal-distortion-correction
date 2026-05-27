"""Mode-specific postprocessing for snapped 2D paths."""

from __future__ import annotations

import numpy as np

from .config import SnapConfig
from .stroke import resample_polyline, smooth_polyline


_EPS = 1e-9


def fit_weighted_line_2d(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2D line by weighted total least squares."""
    points = np.asarray(points, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2); got {points.shape}.")
    if len(points) < 2:
        raise ValueError("At least two points are required to fit a line.")
    if weights.shape != (len(points),):
        raise ValueError(f"weights must have shape ({len(points)},); got {weights.shape}.")

    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= _EPS:
        weights = np.ones_like(weights, dtype=np.float32)
    total = float(weights.sum())
    center = (points * weights[:, None]).sum(axis=0) / total
    centered = points - center
    cov = (centered * weights[:, None]).T @ centered / total
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    direction = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= _EPS:
        direction = points[-1] - points[0]
        norm = float(np.linalg.norm(direction))
    if norm <= _EPS:
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    return center.astype(np.float32), direction.astype(np.float32)


def _resample_output(points: np.ndarray, config: SnapConfig) -> np.ndarray:
    if len(points) < 2:
        return points.astype(np.float32, copy=True)
    return resample_polyline(points, max(float(config.output_spacing_px), 0.5))


def postprocess_curve(points: np.ndarray, config: SnapConfig) -> np.ndarray:
    """Lightly smooth and resample a snapped curve."""
    if len(points) < 3:
        return points.astype(np.float32, copy=True)
    smoothed = smooth_polyline(points, window=max(3, int(config.smooth_window)), passes=1)
    return _resample_output(smoothed, config)


def postprocess_pinhole_line(
    points: np.ndarray,
    source_stroke: np.ndarray,
    weights: np.ndarray,
    config: SnapConfig,
) -> np.ndarray:
    """Fit and sample a 2D straight line for pinhole line annotations."""
    line_point, direction = fit_weighted_line_2d(points, weights)
    source = np.asarray(source_stroke, dtype=np.float32)
    if np.dot(direction, source[-1] - source[0]) < 0:
        direction = -direction

    source_t = (source - line_point) @ direction
    path_t = (np.asarray(points, dtype=np.float32) - line_point) @ direction
    start_t = max(float(source_t.min()), float(path_t.min()) - float(config.endpoint_snap_radius_px))
    end_t = min(float(source_t.max()), float(path_t.max()) + float(config.endpoint_snap_radius_px))
    if end_t < start_t:
        start_t, end_t = end_t, start_t
    length = end_t - start_t
    if length <= _EPS:
        return np.vstack([line_point, line_point + direction]).astype(np.float32)

    n_samples = max(2, int(np.ceil(length / max(float(config.output_spacing_px), 0.5))) + 1)
    t = np.linspace(start_t, end_t, n_samples, dtype=np.float32)
    return (line_point[None, :] + t[:, None] * direction[None, :]).astype(np.float32)


def postprocess_annotation(
    points: np.ndarray,
    source_stroke: np.ndarray,
    mode: str,
    camera_type: str,
    config: SnapConfig,
    debug: dict,
) -> np.ndarray:
    """Apply mode-specific output postprocessing."""
    weights = np.asarray(debug.get("edge_scores", np.ones(len(points))), dtype=np.float32)
    if mode == "line" and camera_type == "pinhole":
        return postprocess_pinhole_line(points, source_stroke, weights, config)
    return postprocess_curve(points, config)
