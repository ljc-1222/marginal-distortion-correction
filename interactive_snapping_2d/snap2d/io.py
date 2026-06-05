"""MaDCoW JSON export helpers for snapped 2D annotations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import SnapResult
from .stroke import clean_stroke

from MaDCoW.src import CameraConfig
from MaDCoW.src.camera import Camera


MADCOW_LINE_POINTS = 128


def _relative_path(path: Path, base_dir: Path) -> str:
    """Return a path relative to ``base_dir`` when possible."""
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), base_dir.resolve())


def _resample_to_count(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample a polyline to exactly ``n_samples`` points."""
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2; got {n_samples}.")
    clean = clean_stroke(points).astype(np.float64)
    segments = np.linalg.norm(np.diff(clean, axis=0), axis=1)
    total = float(segments.sum())
    if total <= 1e-12:
        raise ValueError("Cannot export a zero-length line annotation.")

    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    targets = np.linspace(0.0, total, n_samples)
    result = np.empty((n_samples, 2), dtype=np.float64)
    segment_idx = 0
    for idx, target in enumerate(targets):
        while segment_idx < len(segments) - 1 and target > cumulative[segment_idx + 1]:
            segment_idx += 1
        seg_len = float(segments[segment_idx])
        if seg_len <= 1e-12:
            result[idx] = clean[segment_idx]
            continue
        alpha = (target - cumulative[segment_idx]) / seg_len
        result[idx] = (1.0 - alpha) * clean[segment_idx] + alpha * clean[segment_idx + 1]
    return result.astype(np.float32)


def _madcow_camera_model(camera_type: str) -> str:
    """Map line-aid camera type names to MaDCoW camera model names."""
    if camera_type == "pinhole":
        return "pinhole"
    if camera_type == "panorama":
        return "360"
    raise ValueError(f"camera_type must be 'pinhole' or 'panorama'; got {camera_type!r}.")


def result_to_madcow_line_json(
    result: SnapResult,
    camera: Camera,
    n_samples: int = MADCOW_LINE_POINTS,
) -> dict[str, list[list[float]]]:
    """Convert one snapped 2D result into MaDCoW ``points_dir``."""
    sampled = _resample_to_count(result.points, n_samples)
    xs = sampled[:, 0].astype(np.float64)
    ys = sampled[:, 1].astype(np.float64)
    lam, phi = camera.pixel_to_direction(xs, ys)
    points_dir = [[float(lam_i), float(phi_i)] for lam_i, phi_i in zip(lam, phi)]
    return {"points_dir": points_dir}


def result_to_madcow_annotation_dict(
    results: SnapResult | list[SnapResult],
    image_path: str,
    output_path: str,
    image_shape: tuple[int, int],
    camera_type: str,
    fov_deg: float | None = None,
) -> dict[str, Any]:
    """Convert snapped line results to a MaDCoW-compatible annotation JSON."""
    result_list = [results] if isinstance(results, SnapResult) else list(results)
    if not result_list:
        raise ValueError("At least one snapped line result is required.")
    height, width = image_shape
    camera_model = _madcow_camera_model(camera_type)
    fov_value = None if fov_deg is None else float(fov_deg)
    if camera_model == "pinhole":
        if fov_value is None:
            raise ValueError("fov_deg is required for pinhole MaDCoW annotations.")
        if fov_value <= 0 or fov_value >= 180:
            raise ValueError(f"fov_deg must lie in (0, 180); got {fov_value}.")

    camera = Camera(
        CameraConfig(
            fov_deg=fov_value,
            width=int(width),
            height=int(height),
            model=camera_model,
        )
    )
    json_out = Path(output_path).resolve()
    payload: dict[str, Any] = {
        "image_path": _relative_path(Path(image_path), json_out.parent),
        "camera_model": camera_model,
        "lines": [result_to_madcow_line_json(result, camera) for result in result_list],
        "regions": [],
    }
    if camera_model == "pinhole":
        payload["fov_deg"] = fov_value
    return payload


def save_madcow_annotation_json(
    results: SnapResult | list[SnapResult],
    image_path: str,
    output_path: str,
    image_shape: tuple[int, int],
    camera_type: str,
    fov_deg: float | None = None,
) -> None:
    """Save a MaDCoW-compatible line annotation JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = result_to_madcow_annotation_dict(
        results=results,
        image_path=image_path,
        output_path=output_path,
        image_shape=image_shape,
        camera_type=camera_type,
        fov_deg=fov_deg,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
