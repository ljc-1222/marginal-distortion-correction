"""Configuration and result dataclasses for 2D annotation snapping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "snap_config.json"


@dataclass
class SnapConfig:
    """Hyperparameters for the 2D ribbon snapper."""

    resample_step_px: float = 2.0
    smooth_window: int = 5
    smooth_passes: int = 1
    crop_margin_px: int = 120
    search_width_px: float = 90.0
    offset_step_px: float = 1.0
    max_offset_jump: float = 5.0

    gaussian_blur_ksize: int = 3
    edge_scales: tuple[float, ...] = (0.8, 1.5, 3.0)
    canny_low: int = 30
    canny_high: int = 90

    edge_weight: float = 7.0
    band_center_weight: float = 0.0
    band_center_radii_px: tuple[float, ...] = (8.0, 12.0, 16.0, 24.0, 32.0)
    band_center_min_edge: float = 0.12
    dist_weight: float = 0.00015
    orient_weight: float = 1.5
    color_weight: float = 2.5
    orientation_gate_min: float = 0.35
    orthogonal_penalty_weight: float = 6.0
    jump_gate_min_edge: float = 0.0
    offset_prior_weight: float = 0.004
    offset_prior_clip_px: float = 35.0
    smooth_weight: float = 0.75
    curvature_weight: float = 1.8
    path_curvature_weight: float = 0.05
    path_orientation_weight: float = 12.0
    path_orientation_min: float = 0.55
    turn_weight: float = 20.0
    polarity_flip_weight: float = 8.0
    polarity_min_abs: float = 0.15
    polarity_flip_min_offset_jump: float = 3.0
    normal_gradient_consistency_weight: float = 0.0
    normal_gradient_consistency_min_votes: int = 8
    invalid_cost: float = 1e6

    output_spacing_px: float = 3.0
    offset_smooth_window: int = 13
    offset_smooth_passes: int = 2
    output_smooth_window: int = 15
    output_smooth_passes: int = 3
    endpoint_snap_radius_px: float = 10.0
    distance_clip_px: float = 4.0
    color_contrast_radius_px: float = 2.0
    color_contrast_scale: float = 55.0

    line_angle_search_deg: float = 24.0
    line_angle_step_deg: float = 1.0
    line_rho_step_px: float = 1.0
    line_gap_threshold: float = 0.25
    line_gap_weight: float = 0.45
    line_refit_band_px: float = 3.0
    line_min_refit_score: float = 0.2

    profile_top_k: int = 11
    profile_peak_rel_threshold: float = 0.1
    profile_min_feature_score: float = 0.06
    profile_anchor_step_px: float = 20.0
    profile_anchor_steps_each_side: int = 1

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "SnapConfig":
        """Create a config from a flat mapping of dataclass field names."""
        field_names = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - field_names)
        if unknown:
            raise ValueError(f"Unknown SnapConfig keys: {unknown}")
        coerced = dict(values)
        for key in ("edge_scales", "band_center_radii_px"):
            if key in coerced:
                coerced[key] = tuple(float(item) for item in coerced[key])
        return cls(**coerced)


def load_snap_config(path: str | Path | None = None) -> SnapConfig:
    """Load snap parameters from JSON, or return defaults when no file exists."""
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not config_path.exists():
        if path is None:
            return SnapConfig()
        raise FileNotFoundError(f"Snap config JSON does not exist: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Snap config JSON must contain an object: {config_path}")
    return SnapConfig.from_mapping(data)


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
