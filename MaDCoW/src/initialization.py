"""Stage 2 initialization (Section 5, initialization paragraph).

Vertices inside any ROI ``k`` are placed at ``T_k(v_{i,j})``. Vertices
outside every ROI are an exponentially-weighted blend (Eq. 7-8) of the
stereographic projection ``P_sg`` and the per-ROI projections ``T_k``.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import MeshGrid, Tensor
from .projections import stereographic


def _as_ref_tensor(value: object, ref: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    """Convert ``value`` to a tensor on ``ref``'s device."""
    return torch.as_tensor(value, dtype=dtype or ref.dtype, device=ref.device)


def _validate_init_shapes(
    p_stereo: Tensor,
    t_per_region: Tensor,
    region_masks: Tensor,
    mesh: MeshGrid,
) -> None:
    """Validate the tensor shapes used by the Stage 2 initialization."""
    if p_stereo.ndim != 3 or p_stereo.shape[-1] != 2:
        raise ValueError(f"p_stereo must have shape (H, W, 2); got {tuple(p_stereo.shape)}.")
    if t_per_region.ndim != 4 or t_per_region.shape[-1] != 2:
        raise ValueError(
            f"t_per_region must have shape (K, H, W, 2); got {tuple(t_per_region.shape)}."
        )
    if region_masks.ndim != 3:
        raise ValueError(f"region_masks must have shape (K, H, W); got {tuple(region_masks.shape)}.")

    K, H, W, _ = t_per_region.shape
    if mesh.lambda_grid.shape != (H, W) or mesh.phi_grid.shape != (H, W):
        raise ValueError(
            "mesh dimensions must match the initialization tensors; got "
            f"lambda {mesh.lambda_grid.shape}, phi {mesh.phi_grid.shape}, expected {(H, W)}."
        )
    if p_stereo.shape[:2] != (H, W):
        raise ValueError(
            "p_stereo and t_per_region mesh dimensions differ: "
            f"{tuple(p_stereo.shape[:2])} vs {(H, W)}."
        )
    if region_masks.shape != (K, H, W):
        raise ValueError(
            "region_masks must match t_per_region's first three dimensions: "
            f"{tuple(region_masks.shape)} vs {(K, H, W)}."
        )


def _nearest_squared_distances_in_stereo(p_stereo: Tensor, masks: Tensor) -> Tensor:
    """Compute Eq. 7 squared Euclidean distances in stereographic projection space."""
    H, W, _ = p_stereo.shape
    points = p_stereo.detach().cpu().numpy().reshape(-1, 2)
    masks_np = masks.detach().cpu().numpy().reshape(masks.shape[0], -1).astype(bool)

    all_distances_sq: list[np.ndarray] = []
    for mask in masks_np:
        tree = cKDTree(points[mask])
        distances, _ = tree.query(points, k=1)
        all_distances_sq.append((distances * distances).reshape(H, W))

    distances_sq_np = np.stack(all_distances_sq, axis=0)
    return torch.as_tensor(distances_sq_np, dtype=p_stereo.dtype, device=p_stereo.device)


def stereographic_init(mesh: MeshGrid) -> Tensor:
    """Evaluate ``P_sg`` at every mesh vertex.

    Args:
        mesh: The view-sphere mesh.

    Returns:
        Tensor of shape ``(H, W, 2)`` with stereographic ``(u, v)``
        coordinates.
    """
    if mesh.lambda_grid.shape != mesh.phi_grid.shape:
        raise ValueError(
            "mesh.lambda_grid and mesh.phi_grid must have the same shape; got "
            f"{mesh.lambda_grid.shape} and {mesh.phi_grid.shape}."
        )

    lam = torch.as_tensor(mesh.lambda_grid, dtype=torch.float64)
    phi = torch.as_tensor(mesh.phi_grid, dtype=torch.float64)
    return stereographic(lam, phi)


def init_full_mesh(
    p_stereo: Tensor,
    t_per_region: Tensor,
    region_masks: Tensor,
    mesh: MeshGrid,
    blend_c: float,
) -> Tensor:
    """Compose the Stage 2 starting mesh from per-ROI projections and ``P_sg``.

    Args:
        p_stereo: Stereographic positions, shape ``(H, W, 2)``.
        t_per_region: Per-ROI target positions, shape ``(K, H, W, 2)``.
        region_masks: Boolean ROI masks on the mesh, shape ``(K, H, W)``.
        mesh: The view-sphere mesh, used for the exponential distance in
            stereographic space.
        blend_c: Scale ``c`` in ``alpha_k = exp(-c * d_k)``.

    Returns:
        Tensor of shape ``(H, W, 2)`` with the initial ``p_{i,j}``.
    """
    if blend_c < 0:
        raise ValueError(f"blend_c must be non-negative; got {blend_c}.")

    p = torch.as_tensor(p_stereo)
    targets = _as_ref_tensor(t_per_region, p)
    masks = _as_ref_tensor(region_masks, p, dtype=torch.bool)
    _validate_init_shapes(p, targets, masks, mesh)

    active = masks.flatten(1).any(dim=1)
    if not bool(active.any()):
        return p.clone()

    targets_active = targets[active]
    masks_active = masks[active]

    distances_sq = _nearest_squared_distances_in_stereo(p, masks_active)
    alphas = torch.exp(-float(blend_c) * distances_sq)

    alpha_mean = alphas.mean(dim=0)
    blended_targets = (alphas.unsqueeze(-1) * targets_active).mean(dim=0)
    outside_value = blended_targets + (1.0 - alpha_mean).unsqueeze(-1) * p

    inside_any = masks_active.any(dim=0)
    inside_weights = masks_active.to(dtype=p.dtype).unsqueeze(-1)
    inside_count = inside_weights.sum(dim=0).clamp_min(1.0)
    inside_value = (inside_weights * targets_active).sum(dim=0) / inside_count

    # Overlapping ROI masks are not expected, but averaging makes the
    # initialization deterministic when they occur.
    return torch.where(inside_any.unsqueeze(-1), inside_value, outside_value)


if __name__ == "__main__":
    from .mesh import build_mesh

    torch.set_default_dtype(torch.float64)

    mesh = build_mesh(n_lambda=21, n_phi=15, horizontal_span_deg=70.0)
    p_stereo = stereographic_init(mesh)
    print("stereographic init shape:", tuple(p_stereo.shape), "(expected (15, 21, 2))")

    # 1) Empty ROI stacks leave the stereographic initialization unchanged.
    no_targets = torch.empty((0,) + p_stereo.shape, dtype=p_stereo.dtype)
    no_masks = torch.empty((0,) + p_stereo.shape[:2], dtype=torch.bool)
    p_empty = init_full_mesh(p_stereo, no_targets, no_masks, mesh, blend_c=30.0)
    print("empty regions keep stereo?:", bool(torch.equal(p_empty, p_stereo)))

    # 2) Vertices inside an ROI are copied from that region's projection.
    mask = torch.zeros(p_stereo.shape[:2], dtype=torch.bool)
    mask[5:10, 7:13] = True
    shift = torch.tensor([0.2, -0.1], dtype=p_stereo.dtype)
    target = p_stereo + shift
    p_init = init_full_mesh(p_stereo, target.unsqueeze(0), mask.unsqueeze(0), mesh, blend_c=2.0)
    inside_ok = torch.allclose(p_init[mask], target[mask])
    print("inside ROI equals target?:", bool(inside_ok))

    # 3) Outside vertices follow Eq. 7-8 for a single region.
    i, j = 2, 3
    roi_points = p_stereo[mask].reshape(-1, 2)
    distance_sq = torch.linalg.norm(roi_points - p_stereo[i, j], dim=-1).min() ** 2
    alpha = torch.exp(-2.0 * distance_sq)
    expected = alpha * target[i, j] + (1.0 - alpha) * p_stereo[i, j]
    print("outside blend matches Eq. 7-8?:", bool(torch.allclose(p_init[i, j], expected)))

    # 4) Multiple non-overlapping regions use the paper's average-alpha blend.
    mask2 = torch.zeros_like(mask)
    mask2[9:13, 14:18] = True
    target2 = p_stereo + torch.tensor([-0.15, 0.05], dtype=p_stereo.dtype)
    masks = torch.stack((mask, mask2), dim=0)
    targets = torch.stack((target, target2), dim=0)
    p_multi = init_full_mesh(p_stereo, targets, masks, mesh, blend_c=2.0)
    point = p_stereo[2, 18]
    d1 = torch.linalg.norm(p_stereo[mask].reshape(-1, 2) - point, dim=-1).min() ** 2
    d2 = torch.linalg.norm(p_stereo[mask2].reshape(-1, 2) - point, dim=-1).min() ** 2
    a1 = torch.exp(-2.0 * d1)
    a2 = torch.exp(-2.0 * d2)
    expected_multi = 0.5 * (a1 * target[2, 18] + a2 * target2[2, 18])
    expected_multi = expected_multi + (1.0 - 0.5 * (a1 + a2)) * p_stereo[2, 18]
    print("multi-region blend matches Eq. 8?:", bool(torch.allclose(p_multi[2, 18], expected_multi)))

    # 5) Numpy masks/targets are accepted for compatibility with mesh utilities.
    p_numpy = init_full_mesh(p_stereo, targets.numpy(), masks.numpy(), mesh, blend_c=2.0)
    print("numpy inputs supported?:", bool(torch.allclose(p_numpy, p_multi)))
