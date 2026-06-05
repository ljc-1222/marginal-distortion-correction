"""Edge-guided exact 2D line snapping."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import SnapConfig
from .edge_features import EdgeFeatures, bilinear_sample, compute_edge_features, sample_color_contrast
from .postprocess import fit_weighted_line_2d
from .stroke import clean_stroke, resample_polyline, smooth_polyline


_EPS = 1e-9


def _crop_around_stroke(image: np.ndarray, stroke: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    x0 = max(0, int(np.floor(float(stroke[:, 0].min()) - margin)))
    y0 = max(0, int(np.floor(float(stroke[:, 1].min()) - margin)))
    x1 = min(width, int(np.ceil(float(stroke[:, 0].max()) + margin + 1.0)))
    y1 = min(height, int(np.ceil(float(stroke[:, 1].max()) + margin + 1.0)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Stroke crop is empty.")
    origin = np.asarray([x0, y0], dtype=np.float32)
    return image[y0:y1, x0:x1], (stroke - origin).astype(np.float32), origin


def _valid_points(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= float(width - 1))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= float(height - 1))
    )


def _orient_direction(direction: np.ndarray, stroke: np.ndarray) -> np.ndarray:
    oriented = np.asarray(direction, dtype=np.float32)
    if float(np.dot(oriented, stroke[-1] - stroke[0])) < 0.0:
        oriented = -oriented
    return oriented.astype(np.float32)


def _sample_segment(line_point: np.ndarray, direction: np.ndarray, stroke: np.ndarray, spacing: float) -> np.ndarray:
    t_values = (stroke - line_point[None, :]) @ direction
    t0 = float(t_values.min())
    t1 = float(t_values.max())
    if t1 < t0:
        t0, t1 = t1, t0
    length = max(t1 - t0, 0.0)
    n_samples = max(2, int(np.ceil(length / max(float(spacing), 0.5))) + 1)
    t = np.linspace(t0, t1, n_samples, dtype=np.float32)
    return (line_point[None, :] + t[:, None] * direction[None, :]).astype(np.float32)


def _longest_low_run_fraction(values: np.ndarray, threshold: float, valid: np.ndarray) -> np.ndarray:
    low = (values < float(threshold)) | (~valid)
    if low.ndim == 1:
        low = low[None, :]
    longest = np.zeros(low.shape[0], dtype=np.int32)
    current = np.zeros(low.shape[0], dtype=np.int32)
    for col in range(low.shape[1]):
        current = np.where(low[:, col], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest.astype(np.float32) / max(float(low.shape[1]), 1.0)


def _line_scores_for_angle(
    direction: np.ndarray,
    normal: np.ndarray,
    rhos: np.ndarray,
    stroke: np.ndarray,
    edge_features: EdgeFeatures,
    config: SnapConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t_values = stroke @ direction
    t0 = float(t_values.min())
    t1 = float(t_values.max())
    if t1 < t0:
        t0, t1 = t1, t0
    length = max(t1 - t0, 0.0)
    n_samples = max(2, int(np.ceil(length / max(float(config.resample_step_px), 0.5))) + 1)
    t = np.linspace(t0, t1, n_samples, dtype=np.float32)
    rho_values = np.asarray(rhos, dtype=np.float32).reshape(-1)
    samples = rho_values[:, None, None] * normal[None, None, :] + t[None, :, None] * direction[None, None, :]
    flat = samples.reshape(-1, 2)
    valid = _valid_points(flat, edge_features.strength.shape[:2]).reshape(len(rho_values), n_samples)
    edge = bilinear_sample(edge_features.strength, flat).reshape(len(rho_values), n_samples)
    edge_tx = bilinear_sample(edge_features.tangent_x, flat).reshape(len(rho_values), n_samples)
    edge_ty = bilinear_sample(edge_features.tangent_y, flat).reshape(len(rho_values), n_samples)
    raw_orient = np.abs(edge_tx * direction[0] + edge_ty * direction[1])
    raw_orient = np.clip(raw_orient, 0.0, 1.0).astype(np.float32)
    orientation_floor = np.clip(float(config.orientation_gate_min), 0.0, 0.95)
    orientation_gate = np.clip((raw_orient - orientation_floor) / max(1.0 - orientation_floor, _EPS), 0.0, 1.0)
    flat_normals = np.repeat(normal[None, :], len(flat), axis=0)
    color = sample_color_contrast(
        edge_features,
        flat,
        flat_normals,
        config.color_contrast_radius_px,
        config.color_contrast_scale,
    ).reshape(len(rho_values), n_samples)
    edge = np.where(valid, edge, 0.0)
    raw_orient = np.where(valid, raw_orient, 0.0)
    orientation_gate = np.where(valid, orientation_gate, 0.0)
    orient = np.where(valid, edge * raw_orient, 0.0)
    aligned_edge = edge * orientation_gate
    aligned_color = np.where(valid, color * orientation_gate, 0.0)
    orthogonal_penalty = float(config.orthogonal_penalty_weight) * edge * (1.0 - orientation_gate) ** 2
    valid_fraction = np.mean(valid, axis=1).astype(np.float32)
    stroke_projection = stroke @ normal
    line_dist = np.abs(stroke_projection[None, :] - rho_values[:, None])
    robust_dist = np.minimum(line_dist, float(config.distance_clip_px))
    gap = _longest_low_run_fraction(edge, config.line_gap_threshold, valid)
    mean_edge = np.mean(edge, axis=1)
    mean_aligned_edge = np.mean(aligned_edge, axis=1)
    mean_orient = np.mean(orient, axis=1)
    mean_color = np.mean(aligned_color, axis=1)
    scores = (
        float(config.edge_weight) * mean_aligned_edge
        + float(config.orient_weight) * mean_orient
        + float(config.color_weight) * mean_color
        + valid_fraction
        - float(config.line_gap_weight) * gap
        - float(config.dist_weight) * np.mean(robust_dist * robust_dist, axis=1)
        - np.mean(orthogonal_penalty, axis=1)
    ).astype(np.float32)
    stats = {
        "mean_edge_score": mean_edge.astype(np.float32),
        "mean_orientation_score": mean_orient.astype(np.float32),
        "mean_color_contrast": mean_color.astype(np.float32),
        "gap_penalty": gap,
        "valid_fraction": valid_fraction,
        "mean_abs_offset_px": np.mean(line_dist, axis=1).astype(np.float32),
    }
    return scores, stats


def _refit_line(
    line_point: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    stroke: np.ndarray,
    edge_features: EdgeFeatures,
    config: SnapConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base = _sample_segment(line_point, direction, stroke, max(float(config.resample_step_px), 1.0))
    step = max(float(config.offset_step_px), 0.5)
    offsets = np.arange(-float(config.line_refit_band_px), float(config.line_refit_band_px) + 0.5 * step, step, dtype=np.float32)
    candidates = base[:, None, :] + normal[None, None, :] * offsets[None, :, None]
    flat = candidates.reshape(-1, 2)
    valid = _valid_points(flat, edge_features.strength.shape[:2])
    edge = bilinear_sample(edge_features.strength, flat)
    edge_tx = bilinear_sample(edge_features.tangent_x, flat)
    edge_ty = bilinear_sample(edge_features.tangent_y, flat)
    raw_orient = np.abs(edge_tx * direction[0] + edge_ty * direction[1])
    raw_orient = np.clip(raw_orient, 0.0, 1.0)
    orientation_floor = np.clip(float(config.orientation_gate_min), 0.0, 0.95)
    orientation_gate = np.clip((raw_orient - orientation_floor) / max(1.0 - orientation_floor, _EPS), 0.0, 1.0)
    normals = np.repeat(normal[None, :], len(flat), axis=0)
    color = sample_color_contrast(
        edge_features,
        flat,
        normals,
        config.color_contrast_radius_px,
        config.color_contrast_scale,
    )
    total_weight = max(float(config.edge_weight + config.orient_weight + config.color_weight), _EPS)
    combined = (
        float(config.edge_weight) * edge * orientation_gate
        + float(config.orient_weight) * edge * raw_orient
        + float(config.color_weight) * color * orientation_gate
    ) / total_weight
    combined = np.where(valid, combined, 0.0).astype(np.float32)
    valid_scores = combined[valid]
    if len(valid_scores):
        threshold = max(float(config.line_min_refit_score), float(np.percentile(valid_scores, 70.0)))
    else:
        threshold = float(config.line_min_refit_score)
    keep = valid & (combined >= threshold)
    if int(np.count_nonzero(keep)) < 2:
        return line_point.astype(np.float32), direction.astype(np.float32), {"refit_points": 0}

    refit_point, refit_dir = fit_weighted_line_2d(flat[keep], combined[keep])
    refit_dir = _orient_direction(refit_dir, stroke)
    return refit_point, refit_dir, {
        "refit_points": int(np.count_nonzero(keep)),
        "refit_score_threshold": float(threshold),
    }


def _sample_exact_line(line_point: np.ndarray, direction: np.ndarray, stroke: np.ndarray, config: SnapConfig) -> np.ndarray:
    direction = _orient_direction(direction, stroke)
    start_t = float((stroke[0] - line_point) @ direction)
    end_t = float((stroke[-1] - line_point) @ direction)
    if end_t < start_t:
        start_t, end_t = end_t, start_t
    length = end_t - start_t
    if length <= _EPS:
        end_t = start_t + 1.0
        length = 1.0
    n_samples = max(2, int(np.ceil(length / max(float(config.output_spacing_px), 0.5))) + 1)
    t = np.linspace(start_t, end_t, n_samples, dtype=np.float32)
    return (line_point[None, :] + t[:, None] * direction[None, :]).astype(np.float32)


def snap_line_2d(image: np.ndarray, stroke: np.ndarray, config: SnapConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Snap a rough stroke to a nearby exact 2D line segment."""
    source = clean_stroke(stroke)
    crop_margin = float(config.crop_margin_px) + float(config.search_width_px) + 2.0 * float(config.offset_step_px)
    crop, local_stroke, crop_origin = _crop_around_stroke(image, source, crop_margin)
    centers = resample_polyline(local_stroke, float(config.resample_step_px))
    centers = smooth_polyline(centers, window=int(config.smooth_window), passes=int(config.smooth_passes))
    initial_point, initial_direction = fit_weighted_line_2d(centers, np.ones(len(centers), dtype=np.float32))
    initial_direction = _orient_direction(initial_direction, centers)
    initial_normal = np.asarray([-initial_direction[1], initial_direction[0]], dtype=np.float32)
    initial_rho = float(initial_point @ initial_normal)
    initial_theta = float(np.arctan2(initial_normal[1], initial_normal[0]))

    edge_features = compute_edge_features(crop, config)
    angle_step = max(float(config.line_angle_step_deg), 0.25)
    angle_offsets = np.deg2rad(
        np.arange(
            -float(config.line_angle_search_deg),
            float(config.line_angle_search_deg) + 0.5 * angle_step,
            angle_step,
            dtype=np.float32,
        )
    )
    rho_step = max(float(config.line_rho_step_px), float(config.offset_step_px), 0.5)
    rho_offsets = np.arange(-float(config.search_width_px), float(config.search_width_px) + 0.5 * rho_step, rho_step, dtype=np.float32)

    best_score = -np.inf
    best_point = initial_point
    best_direction = initial_direction
    best_normal = initial_normal
    best_rho = initial_rho
    best_stats: dict[str, float] = {}
    line_search_candidates = 0
    for angle_offset in angle_offsets:
        theta = initial_theta + float(angle_offset)
        normal = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
        direction = np.asarray([-normal[1], normal[0]], dtype=np.float32)
        direction = _orient_direction(direction, centers)
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        base_rho = float(initial_point @ normal)
        rhos = base_rho + rho_offsets
        scores, stats = _line_scores_for_angle(direction, normal, rhos, centers, edge_features, config)
        line_search_candidates += len(rhos)
        rho_idx = int(np.argmax(scores))
        score = float(scores[rho_idx])
        if score > best_score:
            best_score = score
            best_rho = float(rhos[rho_idx])
            best_point = normal * best_rho
            best_direction = direction
            best_normal = normal
            best_stats = {key: float(values[rho_idx]) for key, values in stats.items()}

    refit_point, refit_direction, refit_debug = _refit_line(
        best_point,
        best_direction,
        best_normal,
        centers,
        edge_features,
        config,
    )
    points_local = _sample_exact_line(refit_point, refit_direction, local_stroke, config)
    points = points_local + crop_origin[None, :]
    offsets = np.abs((centers - refit_point[None, :]) @ np.asarray([-refit_direction[1], refit_direction[0]], dtype=np.float32))
    debug: dict[str, Any] = {
        **best_stats,
        **refit_debug,
        "crop_origin": crop_origin,
        "crop_shape": crop.shape[:2],
        "edge_strength": edge_features.strength,
        "path_before_postprocess": points,
        "source_stroke_work": source,
        "selected_offsets": offsets.astype(np.float32),
        "line_search_best_score": float(best_score),
        "line_search_best_rho": float(best_rho),
        "line_search_candidates": int(line_search_candidates),
        "line_direction": refit_direction.astype(np.float32),
        "mode": "line",
    }
    debug["mean_abs_offset_px"] = float(np.mean(offsets)) if len(offsets) else float(best_stats.get("mean_abs_offset_px", 0.0))
    return points.astype(np.float32), debug
