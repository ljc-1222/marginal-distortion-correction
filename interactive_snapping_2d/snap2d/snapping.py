"""Public 2D annotation snapping pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import SnapConfig, SnapResult
from .curve_snapper import snap_curve_2d
from .line_snapper import snap_line_2d
from .panorama import unwrap_panorama_stroke, wrap_panorama_points
from .stroke import clean_stroke


CAMERA_TYPES = {"pinhole", "panorama"}
MODES = {"line", "curve"}


def _validate_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"image must have shape (H, W) or (H, W, C); got {arr.shape}.")
    if arr.shape[0] <= 1 or arr.shape[1] <= 1:
        raise ValueError(f"image is too small: {arr.shape}.")
    return arr


def _clip_points(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    clipped = np.asarray(points, dtype=np.float32).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height - 1))
    return clipped


def _confidence(debug: dict[str, Any], config: SnapConfig) -> float:
    edge = float(debug.get("mean_edge_score", 0.0))
    orient = float(debug.get("mean_orientation_score", 0.0))
    offset = float(debug.get("mean_abs_offset_px", 0.0))
    offset_score = 1.0 - min(1.0, offset / max(float(config.search_width_px), 1.0))
    return float(np.clip(0.55 * edge + 0.35 * orient + 0.10 * offset_score, 0.0, 1.0))


def snap_annotation(
    image: np.ndarray,
    stroke: list[tuple[float, float]] | np.ndarray,
    camera_type: str = "pinhole",
    mode: str = "curve",
    config: SnapConfig | None = None,
) -> SnapResult:
    """Snap a rough 2D stroke to nearby image boundaries.

    The returned ``points`` are always original input-image pixel coordinates.
    """
    if camera_type not in CAMERA_TYPES:
        raise ValueError(f"camera_type must be one of {sorted(CAMERA_TYPES)}; got {camera_type!r}.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}; got {mode!r}.")

    cfg = config or SnapConfig()
    image_arr = _validate_image(image)
    source_stroke = clean_stroke(stroke)
    work_image = image_arr
    work_stroke = source_stroke
    unwrap_info: dict[str, Any] | None = None

    if camera_type == "panorama":
        work_image, work_stroke, unwrap_info = unwrap_panorama_stroke(image_arr, source_stroke)

    if mode == "line":
        post_points_work, debug = snap_line_2d(work_image, work_stroke, cfg)
    else:
        post_points_work, debug = snap_curve_2d(work_image, work_stroke, cfg)

    if camera_type == "panorama":
        if unwrap_info is None:
            raise RuntimeError("Missing panorama unwrap information.")
        points = wrap_panorama_points(post_points_work, unwrap_info)
    else:
        points = _clip_points(post_points_work, image_arr.shape[:2])

    debug.update(
        {
            "source_stroke_work": work_stroke,
            "unwrap_info": unwrap_info,
            "mode": mode,
            "camera_type": camera_type,
        }
    )
    confidence = _confidence(debug, cfg)
    return SnapResult(
        points=points.astype(np.float32),
        source_stroke=source_stroke.astype(np.float32),
        mode=mode,
        camera_type=camera_type,
        confidence=confidence,
        debug=debug,
    )
