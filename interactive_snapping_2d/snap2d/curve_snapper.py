"""Peak-profile dynamic-programming curve snapping."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import SnapConfig
from .edge_features import EdgeFeatures, bilinear_sample, compute_edge_features, sample_color_contrast
from .line_snapper import _crop_around_stroke, _valid_points
from .postprocess import postprocess_curve
from .ribbon import build_ribbon_candidates
from .stroke import clean_stroke, estimate_normals, estimate_tangents, resample_polyline, smooth_polyline


_EPS = 1e-9


def _profile_scores(
    candidates: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    edge_features: EdgeFeatures,
    config: SnapConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_centers, n_offsets = candidates.shape[:2]
    flat = candidates.reshape(-1, 2)
    valid = _valid_points(flat, edge_features.strength.shape[:2]).reshape(n_centers, n_offsets)
    edge = bilinear_sample(edge_features.strength, flat).reshape(n_centers, n_offsets)
    edge_tx = bilinear_sample(edge_features.tangent_x, flat).reshape(n_centers, n_offsets)
    edge_ty = bilinear_sample(edge_features.tangent_y, flat).reshape(n_centers, n_offsets)
    edge_gx = bilinear_sample(edge_features.gradient_x, flat).reshape(n_centers, n_offsets)
    edge_gy = bilinear_sample(edge_features.gradient_y, flat).reshape(n_centers, n_offsets)
    raw_orient = np.abs(edge_tx * tangents[:, None, 0] + edge_ty * tangents[:, None, 1])
    raw_orient = np.clip(raw_orient, 0.0, 1.0).astype(np.float32)
    orientation_floor = np.clip(float(config.orientation_gate_min), 0.0, 0.95)
    orientation_gate = np.clip((raw_orient - orientation_floor) / max(1.0 - orientation_floor, _EPS), 0.0, 1.0)
    polarity = edge_gx * normals[:, None, 0] + edge_gy * normals[:, None, 1]
    polarity = np.clip(polarity, -1.0, 1.0).astype(np.float32)
    flat_normals = np.repeat(normals[:, None, :], n_offsets, axis=1).reshape(-1, 2)
    color = sample_color_contrast(
        edge_features,
        flat,
        flat_normals,
        config.color_contrast_radius_px,
        config.color_contrast_scale,
    ).reshape(n_centers, n_offsets)
    band_score, band_radius = _band_center_scores(flat, tangents, normals, edge_features, valid, config)
    edge = np.where(valid, edge, 0.0).astype(np.float32)
    orient = np.where(valid, edge * raw_orient, 0.0).astype(np.float32)
    color = np.where(valid, color, 0.0).astype(np.float32)
    raw_orient = np.where(valid, raw_orient, 0.0).astype(np.float32)
    orientation_gate = np.where(valid, orientation_gate, 0.0).astype(np.float32)
    polarity = np.where(valid, polarity, 0.0).astype(np.float32)
    aligned_edge = (edge * orientation_gate).astype(np.float32)
    aligned_color = (color * orientation_gate).astype(np.float32)
    orthogonal_penalty = float(config.orthogonal_penalty_weight) * edge * (1.0 - orientation_gate) ** 2
    score = (
        float(config.edge_weight) * aligned_edge
        + float(config.band_center_weight) * band_score
        + float(config.orient_weight) * orient
        + float(config.color_weight) * aligned_color
        - orthogonal_penalty
    ).astype(np.float32)
    score = np.where(valid, score, -float(config.invalid_cost)).astype(np.float32)
    return score, edge, orient, color, polarity, raw_orient, band_score, band_radius


def _band_center_scores(
    flat_points: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    edge_features: EdgeFeatures,
    valid: np.ndarray,
    config: SnapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_centers, n_offsets = valid.shape
    band_score = np.zeros((n_centers, n_offsets), dtype=np.float32)
    band_radius = np.zeros((n_centers, n_offsets), dtype=np.float32)
    if float(config.band_center_weight) <= 0.0:
        return band_score, band_radius

    radii = tuple(radius for radius in config.band_center_radii_px if float(radius) > 0.0)
    if not radii:
        return band_score, band_radius

    flat_normals = np.repeat(normals[:, None, :], n_offsets, axis=1).reshape(-1, 2)
    flat_tangents = np.repeat(tangents[:, None, :], n_offsets, axis=1).reshape(-1, 2)
    height_width = edge_features.strength.shape[:2]
    min_edge = max(float(config.band_center_min_edge), 0.0)
    best = np.zeros(len(flat_points), dtype=np.float32)
    best_radius = np.zeros(len(flat_points), dtype=np.float32)

    for radius in radii:
        radius_value = float(radius)
        left = flat_points - flat_normals * radius_value
        right = flat_points + flat_normals * radius_value
        valid_pair = _valid_points(left, height_width) & _valid_points(right, height_width)
        if not np.any(valid_pair):
            continue

        left_edge = bilinear_sample(edge_features.strength, left)
        right_edge = bilinear_sample(edge_features.strength, right)
        left_tx = bilinear_sample(edge_features.tangent_x, left)
        left_ty = bilinear_sample(edge_features.tangent_y, left)
        right_tx = bilinear_sample(edge_features.tangent_x, right)
        right_ty = bilinear_sample(edge_features.tangent_y, right)
        left_gx = bilinear_sample(edge_features.gradient_x, left)
        left_gy = bilinear_sample(edge_features.gradient_y, left)
        right_gx = bilinear_sample(edge_features.gradient_x, right)
        right_gy = bilinear_sample(edge_features.gradient_y, right)

        left_orient = np.abs(left_tx * flat_tangents[:, 0] + left_ty * flat_tangents[:, 1])
        right_orient = np.abs(right_tx * flat_tangents[:, 0] + right_ty * flat_tangents[:, 1])
        left_polarity = left_gx * flat_normals[:, 0] + left_gy * flat_normals[:, 1]
        right_polarity = right_gx * flat_normals[:, 0] + right_gy * flat_normals[:, 1]

        pair_edge = np.sqrt(np.maximum(left_edge * right_edge, 0.0))
        pair_orient = np.sqrt(np.maximum(left_orient * right_orient, 0.0))
        opposite_polarity = np.clip(-left_polarity * right_polarity, 0.0, 1.0)
        score = pair_edge * pair_orient * (0.25 + 0.75 * opposite_polarity)
        valid_center = valid.reshape(-1)
        score = np.where((left_edge >= min_edge) & (right_edge >= min_edge) & valid_pair & valid_center, score, 0.0)
        better = score > best
        best[better] = score[better]
        best_radius[better] = radius_value

    return best.reshape(n_centers, n_offsets).astype(np.float32), best_radius.reshape(n_centers, n_offsets)


def _is_local_peak(values: np.ndarray, idx: int) -> bool:
    left = values[idx - 1] if idx > 0 else -np.inf
    right = values[idx + 1] if idx + 1 < len(values) else -np.inf
    return bool(values[idx] >= left and values[idx] >= right)


def _select_profile_peaks(
    scores: np.ndarray,
    candidates: np.ndarray,
    offsets: np.ndarray,
    config: SnapConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_centers, n_offsets = scores.shape
    top_k = max(1, int(config.profile_top_k))
    peak_points = np.zeros((n_centers, top_k, 2), dtype=np.float32)
    peak_offsets = np.zeros((n_centers, top_k), dtype=np.float32)
    peak_scores = np.full((n_centers, top_k), -float(config.invalid_cost), dtype=np.float32)
    peak_indices = np.zeros((n_centers, top_k), dtype=np.int32)
    zero_index = int(np.argmin(np.abs(offsets)))
    mandatory_indices = {zero_index}
    anchor_step = float(config.profile_anchor_step_px)
    anchor_count = max(0, int(config.profile_anchor_steps_each_side))
    if anchor_step > 0.0 and anchor_count > 0:
        for step_idx in range(1, anchor_count + 1):
            for sign in (-1.0, 1.0):
                target_offset = sign * anchor_step * float(step_idx)
                if float(offsets[0]) <= target_offset <= float(offsets[-1]):
                    mandatory_indices.add(int(np.argmin(np.abs(offsets - target_offset))))
    mandatory = sorted(mandatory_indices, key=lambda idx: abs(float(offsets[idx])))

    for row in range(n_centers):
        row_scores = scores[row]
        finite = row_scores > -0.5 * float(config.invalid_cost)
        if not np.any(finite):
            chosen = [zero_index]
        else:
            max_score = float(np.max(row_scores[finite]))
            if max_score < float(config.profile_min_feature_score):
                chosen = [zero_index]
            else:
                threshold = max_score * float(config.profile_peak_rel_threshold)
                local_peaks = [
                    idx
                    for idx in range(n_offsets)
                    if finite[idx] and row_scores[idx] >= threshold and _is_local_peak(row_scores, idx)
                ]
                if not local_peaks:
                    local_peaks = [idx for idx in np.argsort(row_scores)[::-1] if finite[idx]][:top_k]
                chosen = sorted(local_peaks, key=lambda idx: float(row_scores[idx]), reverse=True)
        selected_mandatory = [idx for idx in mandatory if idx < n_offsets]
        budget = max(top_k - len(selected_mandatory), 0)
        chosen = [idx for idx in chosen if idx not in selected_mandatory][:budget] + selected_mandatory
        chosen = chosen[:top_k]
        for col, offset_idx in enumerate(chosen):
            peak_points[row, col] = candidates[row, offset_idx]
            peak_offsets[row, col] = offsets[offset_idx]
            peak_scores[row, col] = row_scores[offset_idx]
            peak_indices[row, col] = offset_idx

    return peak_points, peak_offsets, peak_scores, peak_indices


def _solve_peak_dp(
    peak_points: np.ndarray,
    peak_offsets: np.ndarray,
    peak_scores: np.ndarray,
    peak_edges: np.ndarray,
    peak_polarities: np.ndarray,
    pair_tangents: np.ndarray,
    config: SnapConfig,
) -> np.ndarray:
    n_centers, top_k = peak_offsets.shape
    robust = np.minimum(np.abs(peak_offsets), float(config.distance_clip_px))
    lane_robust = np.minimum(np.abs(peak_offsets), float(config.offset_prior_clip_px))
    unary = (
        -peak_scores
        + float(config.dist_weight) * robust * robust
        + float(config.offset_prior_weight) * lane_robust * lane_robust
    )
    invalid = peak_scores <= -0.5 * float(config.invalid_cost)
    unary = np.where(invalid, float(config.invalid_cost), unary).astype(np.float32)

    if n_centers == 1:
        return np.asarray([int(np.argmin(unary[0]))], dtype=np.int32)
    invalid_cost = float(config.invalid_cost)
    max_jump = max(float(config.max_offset_jump), 0.0)
    polarity_min_abs = max(float(config.polarity_min_abs), 0.0)
    polarity_flip_weight = max(float(config.polarity_flip_weight), 0.0)
    polarity_flip_min_jump = max(float(config.polarity_flip_min_offset_jump), 0.0)
    jump_gate_min_edge = max(float(config.jump_gate_min_edge), 0.0)
    path_orientation_weight = max(float(config.path_orientation_weight), 0.0)
    path_orientation_min = np.clip(float(config.path_orientation_min), 0.0, 1.0)

    def pair_cost(prev_idx: int, curr_idx: int) -> np.ndarray:
        prev_offsets = peak_offsets[prev_idx, :, None]
        curr_offsets = peak_offsets[curr_idx, None, :]
        jump = np.abs(curr_offsets - prev_offsets)
        smooth = float(config.smooth_weight) * (curr_offsets - prev_offsets) ** 2
        prev_reliable = peak_edges[prev_idx, :, None] >= jump_gate_min_edge
        curr_reliable = peak_edges[curr_idx, None, :] >= jump_gate_min_edge
        reliable_pair = prev_reliable & curr_reliable
        gate = np.where((max_jump > 0.0) & reliable_pair & (jump > max_jump), invalid_cost, 0.0)
        prev_polarity = peak_polarities[prev_idx, :, None]
        curr_polarity = peak_polarities[curr_idx, None, :]
        reliable = (np.abs(prev_polarity) >= polarity_min_abs) & (np.abs(curr_polarity) >= polarity_min_abs)
        flips = reliable & (jump >= polarity_flip_min_jump) & (prev_polarity * curr_polarity < 0.0)
        polarity = np.where(flips, polarity_flip_weight, 0.0)
        candidate_vectors = peak_points[curr_idx, None, :, :] - peak_points[prev_idx, :, None, :]
        lengths = np.linalg.norm(candidate_vectors, axis=2)
        tangent = pair_tangents[prev_idx]
        alignment = np.zeros_like(lengths, dtype=np.float32)
        valid_length = lengths > _EPS
        alignment[valid_length] = np.abs(np.sum(candidate_vectors[valid_length] * tangent[None, :], axis=1) / lengths[valid_length])
        path_orientation = path_orientation_weight * np.maximum(0.0, path_orientation_min - alignment) ** 2
        return (smooth + gate + polarity + path_orientation).astype(np.float32)

    if n_centers == 2:
        costs = unary[0, :, None] + unary[1, None, :]
        pair = costs + pair_cost(0, 1)
        prev, curr = np.unravel_index(int(np.argmin(pair)), pair.shape)
        return np.asarray([prev, curr], dtype=np.int32)

    pair = unary[0, :, None] + unary[1, None, :] + pair_cost(0, 1)

    back = np.full((n_centers, top_k, top_k), -1, dtype=np.int32)
    for idx in range(2, n_centers):
        prevprev_offsets = peak_offsets[idx - 2, :, None, None]
        prev_offsets = peak_offsets[idx - 1, None, :, None]
        curr_offsets = peak_offsets[idx, None, None, :]
        offset_curvature = (
            float(config.curvature_weight)
            * (curr_offsets - 2.0 * prev_offsets + prevprev_offsets) ** 2
        )
        costs = pair[:, :, None] + unary[idx, None, None, :] + pair_cost(idx - 1, idx)[None, :, :] + offset_curvature

        if float(config.path_curvature_weight) != 0.0:
            prevprev_points = peak_points[idx - 2, :, None, None, :]
            prev_points = peak_points[idx - 1, None, :, None, :]
            curr_points = peak_points[idx, None, None, :, :]
            path_second_diff = curr_points - 2.0 * prev_points + prevprev_points
            path_curvature = np.sum(path_second_diff * path_second_diff, axis=3)
            costs = costs + float(config.path_curvature_weight) * path_curvature

        if float(config.turn_weight) != 0.0:
            prevprev_points = peak_points[idx - 2, :, None, None, :]
            prev_points = peak_points[idx - 1, None, :, None, :]
            curr_points = peak_points[idx, None, None, :, :]
            prev_vector = prev_points - prevprev_points
            curr_vector = curr_points - prev_points
            numerator = np.sum(prev_vector * curr_vector, axis=3)
            denominator = np.maximum(
                np.linalg.norm(prev_vector, axis=3) * np.linalg.norm(curr_vector, axis=3),
                _EPS,
            )
            turn = 1.0 - np.clip(numerator / denominator, -1.0, 1.0)
            costs = costs + float(config.turn_weight) * turn

        back[idx] = np.argmin(costs, axis=0).astype(np.int32)
        new_pair = np.min(costs, axis=0).astype(np.float32)
        pair = new_pair

    selected = np.empty(n_centers, dtype=np.int32)
    prev, curr = np.unravel_index(int(np.argmin(pair)), pair.shape)
    selected[-2] = int(prev)
    selected[-1] = int(curr)
    for idx in range(n_centers - 1, 1, -1):
        prevprev = back[idx, selected[idx - 1], selected[idx]]
        selected[idx - 2] = int(prevprev)
    return selected


def _apply_normal_gradient_consistency(
    peak_scores: np.ndarray,
    peak_polarities: np.ndarray,
    selected: np.ndarray,
    config: SnapConfig,
) -> tuple[np.ndarray, float, int]:
    weight = max(float(config.normal_gradient_consistency_weight), 0.0)
    if weight <= 0.0 or len(selected) == 0:
        return peak_scores, 0.0, 0

    rows = np.arange(len(selected))
    selected_polarity = peak_polarities[rows, selected]
    min_abs = max(float(config.polarity_min_abs), 0.0)
    reliable = np.abs(selected_polarity) >= min_abs
    if int(np.count_nonzero(reliable)) < max(1, int(config.normal_gradient_consistency_min_votes)):
        return peak_scores, 0.0, 0

    signed_sum = float(np.sum(selected_polarity[reliable]))
    if abs(signed_sum) <= _EPS:
        return peak_scores, 0.0, 0

    dominant_sign = 1.0 if signed_sum > 0.0 else -1.0
    candidate_reliable = np.abs(peak_polarities) >= min_abs
    opposite = candidate_reliable & (peak_polarities * dominant_sign < 0.0)
    if not np.any(opposite):
        return peak_scores, dominant_sign, 0

    adjusted = peak_scores.copy()
    adjusted[opposite] -= weight * np.abs(peak_polarities[opposite])
    return adjusted.astype(np.float32), dominant_sign, int(np.count_nonzero(opposite))


def _smooth_offsets(offsets: np.ndarray, config: SnapConfig) -> np.ndarray:
    values = np.asarray(offsets, dtype=np.float32).reshape(-1)
    window = int(config.offset_smooth_window)
    passes = int(config.offset_smooth_passes)
    if len(values) < 3 or window < 3 or passes <= 0:
        return values.astype(np.float32, copy=True)

    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values.astype(np.float32, copy=True)

    kernel = np.ones(window, dtype=np.float32) / float(window)
    half = window // 2
    out = values.astype(np.float32, copy=True)
    for _ in range(passes):
        padded = np.pad(out, (half, half), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid").astype(np.float32)
        smoothed[0] = out[0]
        smoothed[-1] = out[-1]
        out = smoothed
    return out.astype(np.float32)


def _subpixel_refine_offsets(
    selected_indices: np.ndarray,
    peak_indices: np.ndarray,
    profile_edge: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    selected_profile_indices = peak_indices[np.arange(len(selected_indices)), selected_indices]
    refined = offsets[selected_profile_indices].astype(np.float32, copy=True)
    step = float(np.median(np.diff(offsets))) if len(offsets) > 1 else 1.0
    for idx, profile_idx in enumerate(selected_profile_indices):
        if profile_idx <= 0 or profile_idx + 1 >= len(offsets):
            continue
        left = float(profile_edge[idx, profile_idx - 1])
        center = float(profile_edge[idx, profile_idx])
        right = float(profile_edge[idx, profile_idx + 1])
        denom = left - 2.0 * center + right
        if abs(denom) <= _EPS:
            continue
        delta = 0.5 * (left - right) / denom
        refined[idx] += float(np.clip(delta, -1.0, 1.0)) * step
    return refined.astype(np.float32)


def snap_curve_2d(image: np.ndarray, stroke: np.ndarray, config: SnapConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Snap a rough stroke to a nearby 2D curve boundary."""
    source = clean_stroke(stroke)
    crop_margin = float(config.crop_margin_px) + float(config.search_width_px) + 2.0 * float(config.offset_step_px)
    crop, local_stroke, crop_origin = _crop_around_stroke(image, source, crop_margin)
    centers = resample_polyline(local_stroke, float(config.resample_step_px))
    centers = smooth_polyline(centers, window=int(config.smooth_window), passes=int(config.smooth_passes))
    tangents = estimate_tangents(centers)
    normals = estimate_normals(centers)
    edge_features = compute_edge_features(crop, config)
    candidates, offsets = build_ribbon_candidates(centers, normals, config.search_width_px, config.offset_step_px)
    (
        profile_score,
        edge_score,
        orient_score,
        color_score,
        polarity_score,
        raw_orient_score,
        band_center_score,
        band_center_radius,
    ) = _profile_scores(
        candidates,
        tangents,
        normals,
        edge_features,
        config,
    )
    peak_points, peak_offsets, peak_scores, peak_indices = _select_profile_peaks(profile_score, candidates, offsets, config)
    peak_edges = edge_score[np.arange(len(peak_indices))[:, None], peak_indices]
    peak_polarities = polarity_score[np.arange(len(peak_indices))[:, None], peak_indices]
    pair_vectors = np.diff(centers, axis=0)
    pair_lengths = np.linalg.norm(pair_vectors, axis=1, keepdims=True)
    pair_tangents = pair_vectors / np.maximum(pair_lengths, _EPS)
    selected = _solve_peak_dp(peak_points, peak_offsets, peak_scores, peak_edges, peak_polarities, pair_tangents, config)
    peak_scores, dominant_normal_gradient_sign, normal_gradient_penalty_count = _apply_normal_gradient_consistency(
        peak_scores,
        peak_polarities,
        selected,
        config,
    )
    if normal_gradient_penalty_count > 0:
        selected = _solve_peak_dp(peak_points, peak_offsets, peak_scores, peak_edges, peak_polarities, pair_tangents, config)
    refined_offsets = _subpixel_refine_offsets(selected, peak_indices, edge_score, offsets)
    final_offsets = _smooth_offsets(refined_offsets, config)
    path_local = centers + normals * final_offsets[:, None]
    post_local = postprocess_curve(path_local, config)
    points = post_local + crop_origin[None, :]

    rows = np.arange(len(selected))
    selected_profile_indices = peak_indices[rows, selected]
    selected_edge = edge_score[rows, selected_profile_indices]
    selected_orient = orient_score[rows, selected_profile_indices]
    selected_raw_orient = raw_orient_score[rows, selected_profile_indices]
    selected_color = color_score[rows, selected_profile_indices]
    selected_polarity = polarity_score[rows, selected_profile_indices]
    selected_band = band_center_score[rows, selected_profile_indices]
    selected_band_radius = band_center_radius[rows, selected_profile_indices]
    selected_feature = np.maximum(selected_edge, selected_band)
    offset_jumps = np.abs(np.diff(final_offsets))
    reliable_polarity = np.abs(selected_polarity) >= float(config.polarity_min_abs)
    raw_polarity_flips = (
        reliable_polarity[:-1]
        & reliable_polarity[1:]
        & (selected_polarity[:-1] * selected_polarity[1:] < 0.0)
    )
    polarity_flips = raw_polarity_flips & (offset_jumps >= float(config.polarity_flip_min_offset_jump))
    path_vectors = np.diff(path_local, axis=0)
    path_lengths = np.linalg.norm(path_vectors, axis=1)
    path_alignment = np.zeros(len(path_vectors), dtype=np.float32)
    valid_path = path_lengths > _EPS
    if len(pair_tangents):
        path_alignment[valid_path] = np.abs(np.sum(path_vectors[valid_path] * pair_tangents[valid_path], axis=1) / path_lengths[valid_path])
    debug: dict[str, Any] = {
        "crop_origin": crop_origin,
        "crop_shape": crop.shape[:2],
        "edge_strength": edge_features.strength,
        "path_before_postprocess": path_local + crop_origin[None, :],
        "source_stroke_work": source,
        "selected_indices": selected.astype(np.int32),
        "selected_offsets": final_offsets.astype(np.float32),
        "selected_offsets_before_smooth": refined_offsets.astype(np.float32),
        "peak_offsets": peak_offsets,
        "peak_scores": peak_scores,
        "edge_scores": selected_edge.astype(np.float32),
        "orientation_scores": selected_orient.astype(np.float32),
        "raw_orientation_scores": selected_raw_orient.astype(np.float32),
        "color_scores": selected_color.astype(np.float32),
        "polarity_scores": selected_polarity.astype(np.float32),
        "band_center_scores": selected_band.astype(np.float32),
        "band_center_radii": selected_band_radius.astype(np.float32),
        "mean_edge_score": float(np.mean(selected_edge)) if len(selected_edge) else 0.0,
        "mean_band_center_score": float(np.mean(selected_band)) if len(selected_band) else 0.0,
        "mean_feature_adherence_score": float(np.mean(selected_feature)) if len(selected_feature) else 0.0,
        "mean_orientation_score": float(np.mean(selected_orient)) if len(selected_orient) else 0.0,
        "mean_raw_orientation_score": float(np.mean(selected_raw_orient)) if len(selected_raw_orient) else 0.0,
        "mean_color_contrast": float(np.mean(selected_color)) if len(selected_color) else 0.0,
        "mean_path_alignment": float(np.mean(path_alignment)) if len(path_alignment) else 1.0,
        "min_path_alignment": float(np.min(path_alignment)) if len(path_alignment) else 1.0,
        "low_path_alignment_count": int(np.count_nonzero(path_alignment < float(config.path_orientation_min))),
        "mean_abs_polarity": float(np.mean(np.abs(selected_polarity))) if len(selected_polarity) else 0.0,
        "polarity_flip_count": int(np.count_nonzero(polarity_flips)),
        "raw_polarity_flip_count": int(np.count_nonzero(raw_polarity_flips)),
        "normal_gradient_consistency_sign": float(dominant_normal_gradient_sign),
        "normal_gradient_penalty_count": int(normal_gradient_penalty_count),
        "mean_abs_offset_px": float(np.mean(np.abs(final_offsets))) if len(final_offsets) else 0.0,
        "mode": "curve",
    }
    return points.astype(np.float32), debug
