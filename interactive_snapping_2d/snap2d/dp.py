"""Dynamic-programming solver for ribbon snapping."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import SnapConfig
from .edge_features import EdgeFeatures, bilinear_sample


def solve_ribbon_dp(
    candidates: np.ndarray,
    offsets: np.ndarray,
    tangents: np.ndarray,
    edge_features: EdgeFeatures,
    config: SnapConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the ribbon Viterbi problem and return selected path points."""
    candidates = np.asarray(candidates, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    tangents = np.asarray(tangents, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[2] != 2:
        raise ValueError(f"candidates must have shape (N, M, 2); got {candidates.shape}.")
    n_centers, n_offsets = candidates.shape[:2]
    if offsets.shape != (n_offsets,):
        raise ValueError(f"offsets must have shape ({n_offsets},); got {offsets.shape}.")
    if tangents.shape != (n_centers, 2):
        raise ValueError(f"tangents must have shape ({n_centers}, 2); got {tangents.shape}.")

    height, width = edge_features.strength.shape[:2]
    flat = candidates.reshape(-1, 2)
    valid = (
        (flat[:, 0] >= 0.0)
        & (flat[:, 0] <= float(width - 1))
        & (flat[:, 1] >= 0.0)
        & (flat[:, 1] <= float(height - 1))
    ).reshape(n_centers, n_offsets)

    edge_scores = bilinear_sample(edge_features.strength, flat).reshape(n_centers, n_offsets)
    edge_tx = bilinear_sample(edge_features.tangent_x, flat).reshape(n_centers, n_offsets)
    edge_ty = bilinear_sample(edge_features.tangent_y, flat).reshape(n_centers, n_offsets)
    orientation_scores = np.abs(edge_tx * tangents[:, None, 0] + edge_ty * tangents[:, None, 1])
    orientation_scores = np.clip(orientation_scores, 0.0, 1.0).astype(np.float32)

    edge_scores = np.where(valid, edge_scores, 0.0).astype(np.float32)
    orientation_scores = np.where(valid, orientation_scores, 0.0).astype(np.float32)
    unary = (
        float(config.edge_weight) * (1.0 - edge_scores)
        + float(config.dist_weight) * (offsets[None, :] * offsets[None, :])
        + float(config.orient_weight) * (1.0 - orientation_scores)
    ).astype(np.float32)
    unary = np.where(valid, unary, float(config.invalid_cost)).astype(np.float32)

    dp = np.empty((n_centers, n_offsets), dtype=np.float32)
    back = np.full((n_centers, n_offsets), -1, dtype=np.int32)
    dp[0] = unary[0]
    max_jump = max(0, int(config.max_offset_jump))
    smooth_weight = float(config.smooth_weight)

    for idx in range(1, n_centers):
        previous = dp[idx - 1]
        for current in range(n_offsets):
            lo = max(0, current - max_jump)
            hi = min(n_offsets, current + max_jump + 1)
            prev_offsets = offsets[lo:hi]
            pairwise = smooth_weight * (offsets[current] - prev_offsets) ** 2
            costs = previous[lo:hi] + pairwise
            best_rel = int(np.argmin(costs))
            best_prev = lo + best_rel
            dp[idx, current] = unary[idx, current] + costs[best_rel]
            back[idx, current] = best_prev

    selected = np.empty(n_centers, dtype=np.int32)
    selected[-1] = int(np.argmin(dp[-1]))
    for idx in range(n_centers - 1, 0, -1):
        selected[idx - 1] = back[idx, selected[idx]]

    rows = np.arange(n_centers)
    path = candidates[rows, selected]
    selected_offsets = offsets[selected]
    selected_edge = edge_scores[rows, selected]
    selected_orient = orientation_scores[rows, selected]
    selected_valid = valid[rows, selected]
    debug: dict[str, Any] = {
        "unary_cost": unary,
        "selected_indices": selected,
        "selected_offsets": selected_offsets,
        "edge_scores": selected_edge,
        "orientation_scores": selected_orient,
        "valid_selected": selected_valid,
        "total_cost": float(dp[-1, selected[-1]]),
        "mean_edge_score": float(np.mean(selected_edge)) if len(selected_edge) else 0.0,
        "mean_orientation_score": float(np.mean(selected_orient)) if len(selected_orient) else 0.0,
        "mean_abs_offset_px": float(np.mean(np.abs(selected_offsets))) if len(selected_offsets) else 0.0,
    }
    return path.astype(np.float32), debug
