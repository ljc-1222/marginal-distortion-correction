"""Configuration and result dataclasses for 2D annotation snapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SnapConfig:
    """Hyperparameters for the 2D ribbon snapper."""

    resample_step_px: float = 2.0
    smooth_window: int = 7
    smooth_passes: int = 1
    crop_margin_px: int = 48
    search_width_px: float = 30.0
    offset_step_px: float = 1.0
    max_offset_jump: int = 8

    gaussian_blur_ksize: int = 5
    canny_low: int = 50
    canny_high: int = 150

    edge_weight: float = 1.0
    dist_weight: float = 0.015
    orient_weight: float = 0.35
    smooth_weight: float = 0.08
    invalid_cost: float = 1e6

    output_spacing_px: float = 3.0
    endpoint_snap_radius_px: float = 10.0


@dataclass
class SnapResult:
    """Result returned by :func:`snap_annotation`.

    Attributes:
        points: Snapped 2D image-space points with shape ``(N, 2)``.
        source_stroke: Original user stroke points with shape ``(M, 2)``.
        mode: Annotation mode, either ``"line"`` or ``"curve"``.
        camera_type: Camera type, either ``"pinhole"`` or ``"panorama"``.
        confidence: Heuristic confidence in ``[0, 1]``.
        debug: Non-serialized diagnostic arrays and summaries.
    """

    points: np.ndarray
    source_stroke: np.ndarray
    mode: str
    camera_type: str
    confidence: float
    debug: dict[str, Any] = field(default_factory=dict)
