"""Stage 1 -- per-ROI linear perspective optimization (Section 4.1).

Given a single ROI mask, fits the six parameters of
``T(v) = S * P(v) + t`` (Eq. 1) such that the boundary of the region under
``T`` matches the boundary of the region under stereographic projection,
measured by the conformal loss ``E_c`` along the boundary vertices. The
six-parameter objective is minimized with damped Newton iterations.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from . import Array, MeshGrid, Tensor
from .losses import conformal_loss
from .mesh import boundary_vertices
from .projections import T_projection


_EPS = 1e-12
_NEWTON_GRAD_TOL = 1e-9
_NEWTON_STEP_TOL = 1e-10
_NEWTON_INITIAL_DAMPING = 1e-8
_NEWTON_DAMPING_GROWTH = 10.0
_NEWTON_MAX_DAMPING_STEPS = 16
_NEWTON_MAX_LINE_SEARCH = 20
_NEWTON_ARMIJO_C = 1e-4


def _validate_region_mask(region_mask: Array, mesh: MeshGrid) -> Array:
    """Return a boolean ROI mask after validating its mesh shape."""
    mask = np.asarray(region_mask).astype(bool)
    if mask.shape != mesh.lambda_grid.shape:
        raise ValueError(
            f"region_mask must have shape {mesh.lambda_grid.shape}; got {mask.shape}."
        )
    if not bool(mask.any()):
        raise ValueError("region_mask must contain at least one mesh vertex.")
    return mask


def _direction_vectors(lam: Array, phi: Array) -> Array:
    """Convert mesh ``(lambda, phi)`` arrays to 3D unit directions."""
    cos_phi = np.cos(phi)
    x = np.sin(lam) * cos_phi
    y = np.sin(phi)
    z = np.cos(lam) * cos_phi
    return np.stack((x, y, z), axis=-1)


def _direction_to_angles(direction: Array) -> tuple[float, float]:
    """Convert a 3D unit direction to ``(lambda, phi)`` angles."""
    vec = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= _EPS:
        return 0.0, 0.0

    vec = vec / norm
    lam = math.atan2(float(vec[0]), float(vec[2]))
    horizontal = math.hypot(float(vec[0]), float(vec[2]))
    phi = math.atan2(float(vec[1]), horizontal)
    return lam, phi


def _mesh_tensors(mesh: MeshGrid, ref: Tensor) -> tuple[Tensor, Tensor]:
    """Return mesh coordinate tensors on ``ref``'s device and dtype."""
    lam = torch.as_tensor(mesh.lambda_grid, dtype=ref.dtype, device=ref.device)
    phi = torch.as_tensor(mesh.phi_grid, dtype=ref.dtype, device=ref.device)
    return lam, phi


def _boundary_mask(region_mask: Array, mesh: MeshGrid, ref: Tensor) -> Tensor:
    """Convert the paper's boundary vertex set ``V`` to a boolean tensor."""
    boundary = boundary_vertices(region_mask)
    mask = torch.zeros(region_mask.shape, dtype=torch.bool, device=ref.device)
    if boundary.size:
        rows = torch.as_tensor(boundary[:, 0], dtype=torch.long, device=ref.device)
        cols = torch.as_tensor(boundary[:, 1], dtype=torch.long, device=ref.device)
        mask[rows, cols] = True
    return mask


def _evaluate_region_projection(mesh: MeshGrid, params: Tensor) -> Tensor:
    """Evaluate ``T(v)`` for every mesh vertex."""
    lam, phi = _mesh_tensors(mesh, params)
    return T_projection(lam, phi, params)


def _compose_boundary_field(
    mesh: MeshGrid,
    params: Tensor,
    region_mask: Tensor,
    p_stereo: Tensor,
) -> Tensor:
    """Use ``T(v)`` inside the ROI and stereographic projection outside."""
    t_eval = _evaluate_region_projection(mesh, params)
    return torch.where(region_mask.unsqueeze(-1), t_eval, p_stereo)


def _stage1_objective(
    params: Tensor,
    mesh: MeshGrid,
    region_mask: Tensor,
    boundary_mask: Tensor,
    weights: Tensor,
    p_stereo: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute the Stage 1 boundary conformal objective from Eq. 2."""
    p_region = _compose_boundary_field(mesh, params, region_mask, p_stereo)
    return conformal_loss(p_region, mesh, weights, boundary_mask, valid_mask=valid_mask)


def _stage1_value_grad_hessian(
    params: Tensor,
    mesh: MeshGrid,
    region_mask: Tensor,
    boundary_mask: Tensor,
    weights: Tensor,
    p_stereo: Tensor,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate Stage 1 objective, gradient, and Hessian for Newton updates."""
    params_var = params.detach().clone().requires_grad_(True)
    objective = _stage1_objective(
        params_var,
        mesh,
        region_mask,
        boundary_mask,
        weights,
        p_stereo,
        valid_mask,
    )
    grad = torch.autograd.grad(objective, params_var, create_graph=True)[0]
    hessian_rows = []
    for grad_component in grad:
        row = torch.autograd.grad(grad_component, params_var, retain_graph=True)[0]
        hessian_rows.append(row)

    hessian = torch.stack(hessian_rows, dim=0)
    hessian = 0.5 * (hessian + hessian.T)
    return objective.detach(), grad.detach(), hessian.detach()


def _stage1_value(
    params: Tensor,
    mesh: MeshGrid,
    region_mask: Tensor,
    boundary_mask: Tensor,
    weights: Tensor,
    p_stereo: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate only the Stage 1 objective for candidate Newton steps."""
    return _stage1_objective(
        params.detach(),
        mesh,
        region_mask,
        boundary_mask,
        weights,
        p_stereo,
        valid_mask,
    ).detach()


def _newton_direction(grad: Tensor, hessian: Tensor) -> Tensor | None:
    """Solve a damped Newton system and return a descent direction."""
    eye = torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)
    damping = 0.0
    for damping_step in range(_NEWTON_MAX_DAMPING_STEPS):
        matrix = hessian if damping == 0.0 else hessian + damping * eye
        try:
            direction = torch.linalg.solve(matrix, -grad)
        except RuntimeError:
            direction = None

        if direction is not None and torch.isfinite(direction).all():
            directional_derivative = torch.dot(grad, direction)
            if bool(directional_derivative < -_EPS):
                return direction

        if damping_step == 0:
            damping = _NEWTON_INITIAL_DAMPING
        else:
            damping *= _NEWTON_DAMPING_GROWTH

    return None


def _backtracking_line_search(
    params: Tensor,
    direction: Tensor,
    objective_value: Tensor,
    grad: Tensor,
    mesh: MeshGrid,
    region_mask: Tensor,
    boundary_mask: Tensor,
    weights: Tensor,
    p_stereo: Tensor,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor] | None:
    """Accept a Newton step using Armijo backtracking."""
    step_scale = 1.0
    directional_derivative = torch.dot(grad, direction)
    for _ in range(_NEWTON_MAX_LINE_SEARCH):
        candidate = params + step_scale * direction
        candidate_value = _stage1_value(
            candidate,
            mesh,
            region_mask,
            boundary_mask,
            weights,
            p_stereo,
            valid_mask,
        )
        armijo_bound = objective_value + _NEWTON_ARMIJO_C * step_scale * directional_derivative
        if bool(torch.isfinite(candidate_value)) and bool(candidate_value <= armijo_bound):
            return candidate.detach(), candidate_value.detach()
        step_scale *= 0.5

    return None


def _optimize_region_newton(
    params_init: Tensor,
    mesh: MeshGrid,
    region_mask: Tensor,
    boundary_mask: Tensor,
    weights: Tensor,
    p_stereo: Tensor,
    max_iter: int,
    valid_mask: Tensor | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> Tensor:
    """Minimize Stage 1 ``E_c`` with damped Newton iterations."""
    params = params_init.detach().clone()
    best_params = params.detach().clone()
    best_loss = _stage1_value(params, mesh, region_mask, boundary_mask, weights, p_stereo, valid_mask)

    for iteration in range(max_iter):
        try:
            objective_value, grad, hessian = _stage1_value_grad_hessian(
                params,
                mesh,
                region_mask,
                boundary_mask,
                weights,
                p_stereo,
                valid_mask,
            )
            if not bool(torch.isfinite(objective_value) and torch.isfinite(grad).all() and torch.isfinite(hessian).all()):
                break

            if bool(objective_value < best_loss):
                best_loss = objective_value.detach()
                best_params = params.detach().clone()

            grad_norm = torch.linalg.norm(grad)
            if bool(grad_norm <= _NEWTON_GRAD_TOL):
                break

            direction = _newton_direction(grad, hessian)
            if direction is None:
                break

            accepted = _backtracking_line_search(
                params,
                direction,
                objective_value,
                grad,
                mesh,
                region_mask,
                boundary_mask,
                weights,
                p_stereo,
                valid_mask,
            )
            if accepted is None:
                break

            next_params, next_loss = accepted
            step_norm = torch.linalg.norm(next_params - params)
            params = next_params
            if bool(next_loss < best_loss):
                best_loss = next_loss.detach()
                best_params = params.detach().clone()
            if bool(step_norm <= _NEWTON_STEP_TOL * (1.0 + torch.linalg.norm(params))):
                break
        finally:
            if progress_callback is not None:
                progress_callback(iteration + 1)

    return best_params.detach().clone()


def init_region_params(region_mask: Array, mesh: MeshGrid) -> Tensor:
    """Produce the initial Stage 1 parameter vector for one ROI.

    Following the paper: ``v_P`` is the midpoint on the view sphere between
    the image centre direction and the centroid direction of the mask;
    ``a = 1`` and ``b = tx = ty = 0``.

    Args:
        region_mask: Boolean array on the mesh, shape ``(H, W)``.
        mesh: The view-sphere mesh.

    Returns:
        1D tensor of shape ``(6,)`` packing
        ``(lambda_P, phi_P, a, b, tx, ty)``.
    """
    mask = _validate_region_mask(region_mask, mesh)

    directions = _direction_vectors(mesh.lambda_grid, mesh.phi_grid)
    centroid = directions[mask].mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm <= _EPS:
        centroid = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        centroid = centroid / centroid_norm

    centre = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    midpoint = centre + centroid
    midpoint_norm = float(np.linalg.norm(midpoint))
    if midpoint_norm <= _EPS:
        midpoint = centre
    else:
        midpoint = midpoint / midpoint_norm

    lam_p, phi_p = _direction_to_angles(midpoint)
    return torch.tensor([lam_p, phi_p, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)


def optimize_region(
    mesh: MeshGrid,
    region_mask: Array,
    weights: Array,
    p_stereo: Tensor,
    max_iter: int,
    valid_mask: Array | Tensor | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize the six-parameter projection ``T_k`` for one ROI.

    The optimizer minimizes ``E_c`` evaluated only on the boundary vertices
    of the ROI. Vertices inside the ROI follow ``T_k(v_{i,j})`` and vertices
    outside the ROI retain their stereographic positions ``p_stereo``.

    Args:
        mesh: The view-sphere mesh.
        region_mask: Boolean array on the mesh, shape ``(H, W)``.
        weights: Per-vertex weights, shape ``(H, W)``.
        p_stereo: Per-vertex stereographic positions, shape ``(H, W, 2)``;
            used to keep vertices outside the ROI fixed.
        max_iter: Maximum optimizer iterations.
        valid_mask: Optional boolean mesh mask for vertices supported by the
            input FOV. The ROI and boundary conformal loss are restricted to
            this valid domain.
        progress_callback: Optional callback receiving completed Newton
            iteration count for GUI progress reporting.

    Returns:
        Tuple ``(params, t_evaluated)``:
            * ``params`` -- best 6-parameter tensor.
            * ``t_evaluated`` -- the projection ``T_k`` evaluated at every
              mesh vertex, shape ``(H, W, 2)``.
    """
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative; got {max_iter}.")

    mask_np = _validate_region_mask(region_mask, mesh)
    valid_np: Array | None = None
    if valid_mask is not None:
        valid_np = np.asarray(valid_mask).astype(bool)
        if valid_np.shape != mesh.lambda_grid.shape:
            raise ValueError(
                f"valid_mask must have shape {mesh.lambda_grid.shape}; got {valid_np.shape}."
            )
        mask_np = mask_np & valid_np
        if not bool(mask_np.any()):
            raise ValueError("region_mask must contain at least one valid mesh vertex.")

    p_ref = torch.as_tensor(p_stereo)
    if p_ref.ndim != 3 or p_ref.shape[:2] != mask_np.shape or p_ref.shape[-1] != 2:
        raise ValueError(
            f"p_stereo must have shape {(*mask_np.shape, 2)}; got {tuple(p_ref.shape)}."
        )

    weights_t = torch.as_tensor(weights, dtype=p_ref.dtype, device=p_ref.device)
    if weights_t.shape != mask_np.shape:
        raise ValueError(f"weights must have shape {mask_np.shape}; got {tuple(weights_t.shape)}.")

    region_t = torch.as_tensor(mask_np, dtype=torch.bool, device=p_ref.device)
    valid_t = (
        None
        if valid_np is None
        else torch.as_tensor(valid_np, dtype=torch.bool, device=p_ref.device)
    )
    boundary_t = _boundary_mask(mask_np, mesh, p_ref)
    params_init = init_region_params(mask_np, mesh).to(dtype=p_ref.dtype, device=p_ref.device)

    if max_iter == 0 or not bool(boundary_t.any()):
        params_final = params_init.detach().clone()
        return params_final, _evaluate_region_projection(mesh, params_final).detach()

    params_final = _optimize_region_newton(
        params_init=params_init,
        mesh=mesh,
        region_mask=region_t,
        boundary_mask=boundary_t,
        weights=weights_t,
        p_stereo=p_ref,
        max_iter=max_iter,
        valid_mask=valid_t,
        progress_callback=progress_callback,
    )
    t_final = _evaluate_region_projection(mesh, params_final).detach()
    return params_final, t_final


if __name__ == "__main__":
    from .initialization import stereographic_init
    from .mesh import build_mesh

    torch.set_default_dtype(torch.float64)

    mesh = build_mesh(n_lambda=25, n_phi=17, horizontal_span_deg=80.0)
    p_stereo = stereographic_init(mesh)
    weights = np.ones(mesh.lambda_grid.shape, dtype=np.float64)

    # 1) A centered ROI initializes to the optical axis.
    mask_center = np.zeros(mesh.lambda_grid.shape, dtype=bool)
    mask_center[7:10, 11:14] = True
    params_center = init_region_params(mask_center, mesh)
    print("center init near optical axis?:", bool(torch.allclose(params_center[:2], torch.zeros(2, dtype=params_center.dtype), atol=1e-2)))
    print("center init affine identity?:", bool(torch.allclose(params_center[2:], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=params_center.dtype))))

    # 2) An off-center ROI initializes halfway toward its spherical centroid.
    mask_right = np.zeros(mesh.lambda_grid.shape, dtype=bool)
    mask_right[7:10, 17:21] = True
    params_right = init_region_params(mask_right, mesh)
    roi_lam_mean = float(mesh.lambda_grid[mask_right].mean())
    print("right ROI has positive half-way yaw?:", bool(0.0 < params_right[0] < roi_lam_mean))

    # 3) Optimizing Stage 1 lowers the paper's boundary conformal objective.
    mask = np.zeros(mesh.lambda_grid.shape, dtype=bool)
    mask[5:12, 14:21] = True
    params_init = init_region_params(mask, mesh).to(dtype=p_stereo.dtype)
    region_t = torch.as_tensor(mask, dtype=torch.bool)
    boundary_t = _boundary_mask(mask, mesh, p_stereo)
    weights_t = torch.as_tensor(weights, dtype=p_stereo.dtype)
    params_opt, t_opt = optimize_region(mesh, mask, weights, p_stereo, max_iter=20)
    loss_init = _stage1_objective(params_init, mesh, region_t, boundary_t, weights_t, p_stereo)
    loss_opt = _stage1_objective(params_opt, mesh, region_t, boundary_t, weights_t, p_stereo)
    print("stage1 objective decreases?:", bool(loss_opt <= loss_init), float(loss_init), float(loss_opt))
    print("stage1 target shape:", tuple(t_opt.shape), "(expected (17, 25, 2))")
    print("stage1 output finite?:", bool(torch.isfinite(params_opt).all() and torch.isfinite(t_opt).all()))

    # 4) Full masks have no boundary set and return the initial projection.
    full_mask = np.ones(mesh.lambda_grid.shape, dtype=bool)
    full_params, full_target = optimize_region(mesh, full_mask, weights, p_stereo, max_iter=10)
    full_expected = _evaluate_region_projection(mesh, init_region_params(full_mask, mesh).to(dtype=p_stereo.dtype))
    print("full mask skips optimization?:", bool(torch.allclose(full_target, full_expected)))

    # 5) Empty masks are rejected early.
    try:
        optimize_region(mesh, np.zeros(mesh.lambda_grid.shape, dtype=bool), weights, p_stereo, max_iter=1)
    except ValueError as exc:
        print("empty mask rejected?:", "at least one" in str(exc))
