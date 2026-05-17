"""Stage 2 -- full-image warp optimization (Section 5).

Minimizes the composite objective
``E_total = w_l * E_l + w_c * E_c + w_s * E_s + w_DVC * E_DVC``
over the per-vertex mapping ``p_{i,j}`` using ``torch.optim.LBFGS``.
"""

from __future__ import annotations

import torch

from . import LineAnnotation, MeshGrid, PipelineConfig, Tensor
from .losses import conformal_loss, dvc_loss, line_loss, smoothness_loss


def _validate_warp_inputs(
    p_init: Tensor,
    mesh: MeshGrid,
    weights: Tensor,
    t_targets: Tensor,
    region_masks: Tensor,
    valid_mask: Tensor | None = None,
) -> None:
    """Validate tensor and mesh shapes for the Stage 2 optimization."""
    if p_init.ndim != 3 or p_init.shape[-1] != 2:
        raise ValueError(f"p_init must have shape (H, W, 2); got {tuple(p_init.shape)}.")

    H, W, _ = p_init.shape
    if mesh.lambda_grid.shape != (H, W) or mesh.phi_grid.shape != (H, W):
        raise ValueError(
            "mesh dimensions must match p_init; got "
            f"lambda {mesh.lambda_grid.shape}, phi {mesh.phi_grid.shape}, expected {(H, W)}."
        )
    if weights.shape != (H, W):
        raise ValueError(f"weights must have shape {(H, W)}; got {tuple(weights.shape)}.")
    if t_targets.ndim != 4 or t_targets.shape[-1] != 2:
        raise ValueError(f"t_targets must have shape (K, H, W, 2); got {tuple(t_targets.shape)}.")
    if region_masks.ndim != 3:
        raise ValueError(f"region_masks must have shape (K, H, W); got {tuple(region_masks.shape)}.")

    K = t_targets.shape[0]
    if t_targets.shape[1:3] != (H, W):
        raise ValueError(
            f"t_targets mesh dimensions must be {(H, W)}; got {tuple(t_targets.shape[1:3])}."
        )
    if region_masks.shape != (K, H, W):
        raise ValueError(
            "region_masks must match t_targets's first three dimensions: "
            f"{tuple(region_masks.shape)} vs {(K, H, W)}."
        )
    if valid_mask is not None and valid_mask.shape != (H, W):
        raise ValueError(f"valid_mask must have shape {(H, W)}; got {tuple(valid_mask.shape)}.")


def _zero_like_loss(ref: Tensor) -> Tensor:
    """Return a scalar zero connected to ``ref`` for autograd consistency."""
    return ref.sum() * 0.0


def _stage2_objective(
    p: Tensor,
    mesh: MeshGrid,
    weights: Tensor,
    lines: list[LineAnnotation],
    t_targets: Tensor,
    region_masks: Tensor,
    cfg: PipelineConfig,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate Eq. 6, the full MaDCoW Stage 2 objective."""
    loss = _zero_like_loss(p)
    lw = cfg.loss_weights

    if lw.w_l != 0:
        loss = loss + float(lw.w_l) * line_loss(p, lines, mesh, valid_mask=valid_mask)
    if lw.w_c != 0:
        loss = loss + float(lw.w_c) * conformal_loss(p, mesh, weights, valid_mask=valid_mask)
    if lw.w_s != 0:
        loss = loss + float(lw.w_s) * smoothness_loss(p, mesh, weights, valid_mask=valid_mask)
    if lw.w_dvc != 0:
        loss = loss + float(lw.w_dvc) * dvc_loss(p, t_targets, region_masks, valid_mask=valid_mask)

    return loss


def optimize_warp(
    p_init: Tensor,
    mesh: MeshGrid,
    weights: Tensor,
    lines: list[LineAnnotation],
    t_targets: Tensor,
    region_masks: Tensor,
    cfg: PipelineConfig,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Run the L-BFGS optimization for the full-image warp.

    Args:
        p_init: Initial per-vertex coordinates from
            :func:`initialization.init_full_mesh`, shape ``(H, W, 2)``.
        mesh: The view-sphere mesh.
        weights: Per-vertex weights, shape ``(H, W)``.
        lines: Straight-line annotations.
        t_targets: Per-ROI target projections, shape ``(K, H, W, 2)``.
        region_masks: Per-ROI boolean masks on the mesh, shape
            ``(K, H, W)``.
        cfg: Pipeline configuration (loss weights, max iterations, lr).
        valid_mask: Optional boolean mesh mask for vertices supported by the
            input FOV. All Stage 2 loss terms are restricted to this domain.

    Returns:
        Tensor of shape ``(H, W, 2)`` with the optimized per-vertex
        coordinates ``p_final``.
    """
    if cfg.stage2_max_iter < 0:
        raise ValueError(f"stage2_max_iter must be non-negative; got {cfg.stage2_max_iter}.")
    if cfg.lbfgs_lr <= 0:
        raise ValueError(f"lbfgs_lr must be positive; got {cfg.lbfgs_lr}.")

    p_start = torch.as_tensor(p_init)
    weights_t = torch.as_tensor(weights, dtype=p_start.dtype, device=p_start.device)
    targets_t = torch.as_tensor(t_targets, dtype=p_start.dtype, device=p_start.device)
    masks_t = torch.as_tensor(region_masks, dtype=torch.bool, device=p_start.device)
    valid_t = (
        None
        if valid_mask is None
        else torch.as_tensor(valid_mask, dtype=torch.bool, device=p_start.device)
    )
    _validate_warp_inputs(p_start, mesh, weights_t, targets_t, masks_t, valid_t)
    if valid_t is not None:
        masks_t = masks_t & valid_t.unsqueeze(0)

    has_active_region = bool(masks_t.any()) if masks_t.numel() else False
    if cfg.stage2_max_iter == 0 or (not lines and not has_active_region):
        return p_start.detach().clone()

    p = p_start.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [p],
        lr=float(cfg.lbfgs_lr),
        max_iter=int(cfg.stage2_max_iter),
        line_search_fn="strong_wolfe",
    )

    best_loss = float("inf")
    best_p = p.detach().clone()

    def closure() -> Tensor:
        nonlocal best_loss, best_p
        optimizer.zero_grad()
        objective = _stage2_objective(p, mesh, weights_t, lines, targets_t, masks_t, cfg, valid_t)
        if torch.isfinite(objective):
            loss_value = float(objective.detach())
            if loss_value < best_loss:
                best_loss = loss_value
                best_p = p.detach().clone()
        objective.backward()
        return objective

    try:
        optimizer.step(closure)
    except RuntimeError as exc:
        if best_loss == float("inf"):
            raise exc

    return best_p.detach().clone()


if __name__ == "__main__":
    import numpy as np

    from . import LossWeights, PipelineConfig
    from .initialization import stereographic_init
    from .mesh import build_mesh

    torch.set_default_dtype(torch.float64)

    mesh = build_mesh(n_lambda=17, n_phi=11, horizontal_span_deg=60.0)
    p_stereo = stereographic_init(mesh)
    weights = torch.ones(p_stereo.shape[:2], dtype=p_stereo.dtype)
    mask = torch.zeros(p_stereo.shape[:2], dtype=torch.bool)
    mask[4:8, 6:11] = True
    target = p_stereo.clone()
    p_init = p_stereo.clone()
    p_init[mask] = p_init[mask] + torch.tensor([0.2, -0.12], dtype=p_init.dtype)

    cfg_dvc = PipelineConfig(
        mesh_n_lambda=17,
        mesh_n_phi=11,
        loss_weights=LossWeights(w_l=0.0, w_c=0.0, w_s=0.0, w_dvc=1.0),
        blend_c=30.0,
        stage1_max_iter=5,
        stage2_max_iter=25,
        lbfgs_lr=1.0,
    )

    before = dvc_loss(p_init, target.unsqueeze(0), mask.unsqueeze(0))
    p_opt = optimize_warp(p_init, mesh, weights, [], target.unsqueeze(0), mask.unsqueeze(0), cfg_dvc)
    after = dvc_loss(p_opt, target.unsqueeze(0), mask.unsqueeze(0))
    print("DVC-only optimization lowers loss?:", bool(after < before), float(before), float(after))
    print("DVC-only ROI reaches target?:", bool(torch.allclose(p_opt[mask], target[mask], atol=1e-5)))

    # 2) Eq. 6 equals the explicit weighted sum of the four paper losses.
    line_points = tuple(
        (float(-0.3 + 0.6 * t), 0.0)
        for t in torch.linspace(0.0, 1.0, 128)
    )
    lines = [LineAnnotation(points_dir=line_points)]
    cfg_eq6 = PipelineConfig(
        mesh_n_lambda=17,
        mesh_n_phi=11,
        loss_weights=LossWeights(w_l=25.0, w_c=1.0, w_s=12.0, w_dvc=1.0),
        blend_c=30.0,
        stage1_max_iter=5,
        stage2_max_iter=10,
        lbfgs_lr=1.0,
    )
    objective = _stage2_objective(p_init, mesh, weights, lines, target.unsqueeze(0), mask.unsqueeze(0), cfg_eq6)
    manual = (
        25.0 * line_loss(p_init, lines, mesh)
        + conformal_loss(p_init, mesh, weights)
        + 12.0 * smoothness_loss(p_init, mesh, weights)
        + dvc_loss(p_init, target.unsqueeze(0), mask.unsqueeze(0))
    )
    print("Eq. 6 weighted sum exact?:", bool(torch.allclose(objective, manual)))

    # 3) The composite MaDCoW objective decreases from a perturbed ROI init.
    composite_before = _stage2_objective(
        p_init,
        mesh,
        weights,
        [],
        target.unsqueeze(0),
        mask.unsqueeze(0),
        cfg_eq6,
    )
    p_composite = optimize_warp(
        p_init,
        mesh,
        weights,
        [],
        target.unsqueeze(0),
        mask.unsqueeze(0),
        cfg_eq6,
    )
    composite_after = _stage2_objective(
        p_composite,
        mesh,
        weights,
        [],
        target.unsqueeze(0),
        mask.unsqueeze(0),
        cfg_eq6,
    )
    print("composite objective decreases?:", bool(composite_after < composite_before), float(composite_before), float(composite_after))
    print("stage2 output finite?:", bool(torch.isfinite(p_composite).all()))

    # 4) Zero iterations return the input unchanged.
    cfg_zero = PipelineConfig(
        mesh_n_lambda=17,
        mesh_n_phi=11,
        loss_weights=LossWeights(w_l=0.0, w_c=0.0, w_s=0.0, w_dvc=1.0),
        blend_c=30.0,
        stage1_max_iter=5,
        stage2_max_iter=0,
        lbfgs_lr=1.0,
    )
    p_zero = optimize_warp(p_init, mesh, weights, [], target.unsqueeze(0), mask.unsqueeze(0), cfg_zero)
    print("zero-iteration warp unchanged?:", bool(torch.equal(p_zero, p_init)))

    # 5) No line or active region constraints returns the starting projection
    # to avoid the unanchored conformal/smoothness degeneracy.
    cfg_free = PipelineConfig(
        mesh_n_lambda=17,
        mesh_n_phi=11,
        loss_weights=LossWeights(w_l=0.0, w_c=1.0, w_s=12.0, w_dvc=1.0),
        blend_c=30.0,
        stage1_max_iter=5,
        stage2_max_iter=10,
        lbfgs_lr=1.0,
    )
    empty_targets = torch.empty((0,) + p_stereo.shape, dtype=p_stereo.dtype)
    empty_masks = torch.empty((0,) + p_stereo.shape[:2], dtype=torch.bool)
    p_free = optimize_warp(p_stereo, mesh, weights, [], empty_targets, empty_masks, cfg_free)
    print("unanchored warp unchanged?:", bool(torch.equal(p_free, p_stereo)))

    # 6) Numpy inputs are accepted for compatibility with preprocessing code.
    p_np = optimize_warp(
        p_init.numpy(),
        mesh,
        weights.numpy(),
        [],
        target.unsqueeze(0).numpy(),
        mask.unsqueeze(0).numpy(),
        cfg_dvc,
    )
    print("numpy inputs supported?:", bool(torch.allclose(p_np, p_opt, atol=1e-5)))

    # 7) Invalid shapes are rejected before optimization.
    try:
        optimize_warp(p_init, mesh, np.ones((3, 3)), [], target.unsqueeze(0), mask.unsqueeze(0), cfg_dvc)
    except ValueError as exc:
        print("bad weights rejected?:", "weights" in str(exc))
