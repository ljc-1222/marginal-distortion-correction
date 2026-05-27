"""Stroke preprocessing utilities for 2D annotation snapping."""

from __future__ import annotations

import numpy as np


_EPS = 1e-9


def _as_points(points: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2); got {arr.shape}.")
    return arr


def clean_stroke(stroke: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
    """Remove invalid and duplicate stroke points."""
    points = _as_points(stroke)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 2:
        raise ValueError("At least two finite stroke points are required.")

    kept = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - kept[-1])) > _EPS:
            kept.append(point)

    cleaned = np.asarray(kept, dtype=np.float32)
    if len(cleaned) < 2:
        raise ValueError("At least two distinct stroke points are required.")
    if _polyline_length(cleaned) <= _EPS:
        raise ValueError("Stroke length is too small.")
    return cleaned


def _polyline_length(points: np.ndarray) -> float:
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline with approximately uniform arc-length spacing."""
    points = clean_stroke(points)
    if step <= 0:
        raise ValueError(f"step must be positive; got {step}.")

    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(segments.sum())
    if total <= _EPS:
        raise ValueError("Cannot resample a zero-length polyline.")

    targets = np.arange(0.0, total, float(step), dtype=np.float32)
    if len(targets) == 0 or abs(float(targets[-1]) - total) > max(1e-4, step * 0.25):
        targets = np.concatenate([targets, np.asarray([total], dtype=np.float32)])
    elif float(targets[-1]) < total:
        targets[-1] = total

    cumulative = np.concatenate([[0.0], np.cumsum(segments)]).astype(np.float32)
    resampled = np.empty((len(targets), 2), dtype=np.float32)
    segment_idx = 0
    for idx, target in enumerate(targets):
        while segment_idx < len(segments) - 1 and target > cumulative[segment_idx + 1]:
            segment_idx += 1
        seg_len = float(segments[segment_idx])
        if seg_len <= _EPS:
            resampled[idx] = points[segment_idx]
            continue
        alpha = (float(target) - float(cumulative[segment_idx])) / seg_len
        resampled[idx] = (1.0 - alpha) * points[segment_idx] + alpha * points[segment_idx + 1]
    return resampled


def smooth_polyline(points: np.ndarray, window: int = 7, passes: int = 1) -> np.ndarray:
    """Smooth a polyline with an endpoint-preserving moving average."""
    points = _as_points(points)
    if len(points) < 3 or window < 3 or passes <= 0:
        return points.astype(np.float32, copy=True)

    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, len(points) if len(points) % 2 == 1 else len(points) - 1)
    if window < 3:
        return points.astype(np.float32, copy=True)

    kernel = np.ones(window, dtype=np.float32) / float(window)
    half = window // 2
    out = points.astype(np.float32, copy=True)
    for _ in range(int(passes)):
        padded = np.pad(out, ((half, half), (0, 0)), mode="edge")
        smoothed = np.empty_like(out)
        for dim in range(2):
            smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
        smoothed[0] = out[0]
        smoothed[-1] = out[-1]
        out = smoothed
    return out


def estimate_tangents(points: np.ndarray) -> np.ndarray:
    """Estimate unit tangent vectors along a polyline."""
    points = clean_stroke(points)
    tangents = np.empty_like(points, dtype=np.float32)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    invalid = norms[:, 0] <= _EPS
    norms[invalid] = 1.0
    tangents = tangents / norms
    if np.any(invalid):
        tangents[invalid] = np.asarray([1.0, 0.0], dtype=np.float32)
    return tangents.astype(np.float32)


def rotate90(vectors: np.ndarray) -> np.ndarray:
    """Rotate 2D vectors counter-clockwise by 90 degrees."""
    vectors = _as_points(vectors)
    rotated = np.empty_like(vectors, dtype=np.float32)
    rotated[:, 0] = -vectors[:, 1]
    rotated[:, 1] = vectors[:, 0]
    return rotated


def estimate_normals(points: np.ndarray) -> np.ndarray:
    """Estimate unit normals by rotating tangents by 90 degrees."""
    return rotate90(estimate_tangents(points))
