"""Loss terms used by the MaDCoW optimizations.

* ``E_c`` -- discrete conformal loss (Eq. 2 of Carroll et al. and the paper);
  used both along ROI boundaries in Stage 1 and over the whole mesh in
  Stage 2.
* ``E_l`` -- straight-line preservation loss (Eq. 4-5).
* ``E_s`` -- smoothness penalty on the differential north vector.
* ``E_DVC`` -- per-ROI DVC matching loss (Eq. 3).

All losses are differentiable functions of the per-vertex output coordinates
``p`` so they can be optimized by ``torch.optim.LBFGS``.
"""

from __future__ import annotations

import math

import torch

from . import LineAnnotation, MeshGrid, Tensor
from .mesh import bilinear_sample, mesh_angular_steps


_EPS = 1e-12


def _mesh_phi_tensor(mesh: MeshGrid, ref: Tensor) -> Tensor:
    """Return ``mesh.phi_grid`` as a tensor on ``ref``'s device and dtype."""
    return torch.as_tensor(mesh.phi_grid, dtype=ref.dtype, device=ref.device)


def _zero_like_loss(ref: Tensor) -> Tensor:
    """Return a scalar zero connected to ``ref`` for autograd consistency."""
    return ref.sum() * 0.0


def _bool_mask_tensor(mask: Tensor | None, ref: Tensor, shape: tuple[int, int], name: str) -> Tensor:
    """Return a boolean mesh mask on ``ref``'s device, defaulting to all true."""
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=ref.device)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=ref.device)
    if mask_t.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(mask_t.shape)}.")
    return mask_t


def _line_points_tensor(line: LineAnnotation, ref: Tensor) -> tuple[Tensor, Tensor]:
    """Return annotated line samples as lambda and phi tensors."""
    points = torch.tensor(line.points_dir, dtype=ref.dtype, device=ref.device)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError(
            "line.points_dir must have shape (N, 2) with at least two samples."
        )
    return points[:, 0], points[:, 1]


def _query_bilinear_validity(
    mesh: MeshGrid,
    query_lam: Tensor,
    query_phi: Tensor,
    ref: Tensor,
    valid_mask: Tensor | None,
) -> Tensor:
    """Check whether query directions have a fully valid bilinear footprint."""
    H, W = mesh.lambda_grid.shape
    lam_min = float(mesh.lambda_grid[0, 0])
    lam_max = float(mesh.lambda_grid[0, -1])
    phi_min = float(mesh.phi_grid[0, 0])
    phi_max = float(mesh.phi_grid[-1, 0])
    delta_lam, delta_phi = mesh_angular_steps(mesh)
    eps = 1e-9

    inside = (
        (query_lam >= min(lam_min, lam_max) - eps)
        & (query_lam <= max(lam_min, lam_max) + eps)
        & (query_phi >= min(phi_min, phi_max) - eps)
        & (query_phi <= max(phi_min, phi_max) + eps)
    )

    if valid_mask is None:
        return inside

    valid = _bool_mask_tensor(valid_mask, ref, (H, W), "valid_mask")
    j_f = (query_lam - lam_min) / delta_lam
    i_f = (query_phi - phi_min) / delta_phi
    j0 = torch.floor(j_f).long().clamp(0, W - 1)
    i0 = torch.floor(i_f).long().clamp(0, H - 1)
    j1 = torch.clamp(j0 + 1, max=W - 1)
    i1 = torch.clamp(i0 + 1, max=H - 1)
    footprint_valid = valid[i0, j0] & valid[i0, j1] & valid[i1, j0] & valid[i1, j1]
    return inside & footprint_valid


def conformal_loss(
    p: Tensor,
    mesh: MeshGrid,
    w: Tensor,
    vertex_mask: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute Carroll et al.'s discrete conformal loss ``E_c`` (Eq. 2).

    Args:
        p: Per-vertex output coordinates, shape ``(H, W, 2)``.
        mesh: Mesh providing ``phi`` for the ``cos(phi)`` factor.
        w: Per-vertex weights, shape ``(H, W)``.
        vertex_mask: Optional boolean mask, shape ``(H, W)``. When supplied,
            only vertices in the mask contribute to the sum (used for the
            Stage 1 boundary set ``V``).
        valid_mask: Optional boolean mask selecting vertices supported by the
            input FOV. Cells contribute only when their anchor, down, and right
            vertices are valid.

    Returns:
        Scalar tensor.
    """
    H, W, _ = p.shape
    u = p[..., 0]
    v = p[..., 1]
    phi = _mesh_phi_tensor(mesh, p)

    weight = torch.as_tensor(w, dtype=p.dtype, device=p.device)
    if weight.shape != (H, W):
        raise ValueError(f"w must have shape {(H, W)}; got {tuple(weight.shape)}.")
    vertex = _bool_mask_tensor(vertex_mask, p, (H, W), "vertex_mask")
    valid_vertices = _bool_mask_tensor(valid_mask, p, (H, W), "valid_mask")

    du_dphi = (u[1:, :-1] - u[:-1, :-1])
    dv_dphi = (v[1:, :-1] - v[:-1, :-1])
    du_dlambda = (u[:-1, 1:] - u[:-1, :-1])
    dv_dlambda = (v[:-1, 1:] - v[:-1, :-1])

    cos_phi = torch.cos(phi[:-1, :-1])
    weight_sq = weight[:-1, :-1] * weight[:-1, :-1]
    valid_cell = valid_vertices[:-1, :-1] & valid_vertices[1:, :-1] & valid_vertices[:-1, 1:]
    # The vertex_mask is an anchor mask: Stage 1 boundary vertices select
    # which finite-difference cells to anchor, while valid_mask guards all
    # vertices touched by each cell.
    valid_cell = valid_cell & vertex[:-1, :-1]
    valid = valid_cell.to(dtype=p.dtype)

    # The paper writes the same discrete Cauchy-Riemann constraints in its
    # latitude/longitude convention. Here rows increase with ``phi`` and
    # image ``v`` points down, so the metric-corrected form below makes
    # stereographic projection nearly zero up to discretization error.
    term_a = cos_phi * dv_dphi - du_dlambda
    term_b = cos_phi * du_dphi + dv_dlambda
    return (weight_sq * valid * (term_a * term_a + term_b * term_b)).sum()


def line_loss(
    p: Tensor,
    lines: list[LineAnnotation],
    mesh: MeshGrid,
    n_samples: int = 32,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute the straight-line preservation loss ``E_l`` (Eq. 4-5).

    Each line contributes its annotated view-sphere samples directly. The
    squared distance from every valid warped sample to the image-plane line
    through the warped first and last samples is summed.

    Args:
        p: Per-vertex output coordinates, shape ``(H, W, 2)``.
        lines: List of straight-line annotations.
        mesh: The view-sphere mesh, used by the internal bilinear sampler.
        n_samples: Ignored for freehand annotations and kept only for API
            compatibility.
        valid_mask: Optional boolean mask selecting vertices supported by the
            input FOV. Lines with invalid endpoints are skipped, and invalid
            annotated samples do not contribute.

    Returns:
        Scalar tensor.
    """
    if not lines:
        return _zero_like_loss(p)
    _ = n_samples

    loss = _zero_like_loss(p)
    for line in lines:
        sample_lam, sample_phi = _line_points_tensor(line, p)
        endpoint_lam = torch.stack((sample_lam[0], sample_lam[-1]))
        endpoint_phi = torch.stack((sample_phi[0], sample_phi[-1]))
        endpoint_valid = _query_bilinear_validity(mesh, endpoint_lam, endpoint_phi, p, valid_mask)
        if not bool(endpoint_valid.all()):
            continue
        endpoint_pos = bilinear_sample(p, mesh, endpoint_lam, endpoint_phi)
        start_pos = endpoint_pos[0]
        end_pos = endpoint_pos[1]

        direction = end_pos - start_pos
        length = torch.linalg.norm(direction).clamp_min(_EPS)
        normal = torch.stack((-direction[1], direction[0])) / length

        sample_valid = _query_bilinear_validity(mesh, sample_lam, sample_phi, p, valid_mask)
        if int(sample_valid.sum()) < 2:
            continue
        sample_lam = sample_lam[sample_valid]
        sample_phi = sample_phi[sample_valid]
        sample_pos = bilinear_sample(p, mesh, sample_lam, sample_phi)
        signed_distance = (sample_pos - start_pos) @ normal
        loss = loss + (signed_distance * signed_distance).sum()

    return loss


def smoothness_loss(
    p: Tensor,
    mesh: MeshGrid,
    w: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute the smoothness penalty ``E_s``.

    Penalizes fast changes of the differential north vector of ``f`` so the
    warp stays as rigid as possible.

    Args:
        p: Per-vertex output coordinates, shape ``(H, W, 2)``.
        mesh: Mesh providing ``phi`` for the ``cos^2(phi)`` factor.
        w: Per-vertex weights, shape ``(H, W)``.
        valid_mask: Optional boolean mask selecting vertices supported by the
            input FOV. A smoothness term contributes only when all vertices
            touched by its finite differences are valid.

    Returns:
        Scalar tensor.
    """
    H, W, _ = p.shape
    if H < 2 or W < 3:
        return _zero_like_loss(p)

    u = p[..., 0]
    v = p[..., 1]
    phi = _mesh_phi_tensor(mesh, p)
    weight = torch.as_tensor(w, dtype=p.dtype, device=p.device)
    if weight.shape != (H, W):
        raise ValueError(f"w must have shape {(H, W)}; got {tuple(weight.shape)}.")
    valid_vertices = _bool_mask_tensor(valid_mask, p, (H, W), "valid_mask")

    u_xx = (u[:-1, 2:] - 2.0 * u[:-1, 1:-1] + u[:-1, :-2])
    v_xx = (v[:-1, 2:] - 2.0 * v[:-1, 1:-1] + v[:-1, :-2])
    u_xy = (u[1:, 2:] - u[1:, 1:-1] - u[:-1, 2:] + u[:-1, 1:-1])
    v_xy = (v[1:, 2:] - v[1:, 1:-1] - v[:-1, 2:] + v[:-1, 1:-1])

    cos_phi = torch.cos(phi[:-1, 1:-1])
    weight_sq = weight[:-1, 1:-1] * weight[:-1, 1:-1]
    valid_xx = valid_vertices[:-1, 2:] & valid_vertices[:-1, 1:-1] & valid_vertices[:-1, :-2]
    valid_xy = valid_vertices[1:, 2:] & valid_vertices[1:, 1:-1] & valid_vertices[:-1, 2:] & valid_vertices[:-1, 1:-1]
    valid = (valid_xx & valid_xy).to(dtype=p.dtype)
    second_order = u_xx * u_xx + v_xx * v_xx + u_xy * u_xy + v_xy * v_xy
    return (weight_sq * cos_phi * cos_phi * valid * second_order).sum()


def dvc_loss(
    p: Tensor,
    t_targets: Tensor,
    region_mask: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute the DVC matching loss ``E_DVC`` (Eq. 3).

    Penalizes the squared deviation of ``f(v_{i,j})`` from each region's
    target projection ``T_k(v_{i,j})`` over the vertices in that region.

    Args:
        p: Per-vertex output coordinates, shape ``(H, W, 2)``.
        t_targets: Stack of per-ROI target coordinates, shape
            ``(K, H, W, 2)``; entries outside region ``k`` may be anything.
        region_mask: Boolean stack of shape ``(K, H, W)`` selecting the
            vertices that belong to each region.
        valid_mask: Optional boolean mask selecting vertices supported by the
            input FOV.

    Returns:
        Scalar tensor.
    """
    targets = torch.as_tensor(t_targets, dtype=p.dtype, device=p.device)
    mask_bool = torch.as_tensor(region_mask, dtype=torch.bool, device=p.device)
    if targets.numel() == 0 or mask_bool.numel() == 0:
        return _zero_like_loss(p)
    if valid_mask is not None:
        valid = _bool_mask_tensor(valid_mask, p, p.shape[:2], "valid_mask")
        mask_bool = mask_bool & valid.unsqueeze(0)

    diff_sq = ((p.unsqueeze(0) - targets) ** 2).sum(dim=-1)
    mask = mask_bool.to(dtype=p.dtype)
    return (diff_sq * mask).sum()


if __name__ == "__main__":
    from . import CameraConfig, LineAnnotation
    from .camera import Camera
    from .mesh import build_input_domain_mesh, build_mesh, compute_valid_mesh_mask
    from .projections import stereographic

    torch.set_default_dtype(torch.float64)

    def _conformal_mean(loss: Tensor, mesh: MeshGrid) -> Tensor:
        H, W = mesh.lambda_grid.shape
        return loss / float((H - 1) * (W - 1))

    def _make_line(
        start: tuple[float, float],
        end: tuple[float, float],
        n: int = 128,
    ) -> LineAnnotation:
        t_values = torch.linspace(0.0, 1.0, n)
        points = tuple(
            (
                float((1.0 - t) * start[0] + t * end[0]),
                float((1.0 - t) * start[1] + t * end[1]),
            )
            for t in t_values
        )
        return LineAnnotation(points_dir=points)

    def _make_curved_line(n: int = 128) -> LineAnnotation:
        t_values = torch.linspace(0.0, 1.0, n)
        points = tuple(
            (
                float(-0.35 + 0.7 * t),
                float(0.08 * torch.sin(torch.as_tensor(math.pi, dtype=t.dtype) * t)),
            )
            for t in t_values
        )
        return LineAnnotation(points_dir=points)

    mesh = build_mesh(n_lambda=41, n_phi=31, horizontal_span_deg=60.0)
    lam = torch.from_numpy(mesh.lambda_grid)
    phi = torch.from_numpy(mesh.phi_grid)
    p = stereographic(lam, phi).requires_grad_(True)
    w = torch.ones_like(lam)

    # 1) Stereographic projection is conformal up to discretization error.
    ec = conformal_loss(p, mesh, w)
    print("stereographic conformal mean small?:", bool(ec.ndim == 0 and _conformal_mean(ec, mesh) < 1e-4))

    # 2) Conformal vertex masks restrict contributions.
    empty_mask = torch.zeros_like(w, dtype=torch.bool)
    ec_empty = conformal_loss(p, mesh, w, empty_mask)
    print("empty conformal mask zero?:", bool(torch.allclose(ec_empty, torch.zeros_like(ec_empty))))

    # 3) A straight annotated line stays straight under stereographic projection.
    line = _make_line((-0.35, 0.0), (0.35, 0.0))
    el = line_loss(p, [line], mesh, n_samples=16)
    print("straight line loss zero?:", bool(torch.allclose(el, torch.zeros_like(el), atol=1e-12)))

    # 4) A curved annotated line has nonzero loss under a linear warp.
    p_linear = torch.stack((lam, phi), dim=-1).requires_grad_(True)
    curved_line = _make_curved_line()
    curved_el = line_loss(p_linear, [curved_line], mesh)
    print("curved line loss nonzero?:", bool(curved_el > 1e-6))

    # 5) Linear fields have zero second-order smoothness loss.
    es = smoothness_loss(p_linear, mesh, w)
    print("linear smoothness zero?:", bool(torch.allclose(es, torch.zeros_like(es), atol=1e-12)))

    # 6) DVC loss is zero when targets match p on the masked region.
    mask = torch.zeros((1,) + lam.shape, dtype=torch.bool)
    mask[:, 10:20, 12:30] = True
    edvc = dvc_loss(p, p.detach().unsqueeze(0), mask)
    print("matching DVC zero?:", bool(torch.allclose(edvc, torch.zeros_like(edvc))))

    # 7) Gradients flow through the combined loss.
    total = ec + el + es + edvc
    total.backward()
    print("combined grad finite?:", bool(p.grad is not None and torch.isfinite(p.grad).all()))

    # 8) Non-square angular spacing remains well scaled by derivative terms.
    cam = Camera(CameraConfig(fov_deg=90.0, width=160, height=90))
    non_square_mesh = build_input_domain_mesh(cam, n_lambda=41, n_phi=31)
    lam_ns = torch.from_numpy(non_square_mesh.lambda_grid)
    phi_ns = torch.from_numpy(non_square_mesh.phi_grid)
    p_ns = stereographic(lam_ns, phi_ns)
    w_ns = torch.ones_like(lam_ns)
    ec_ns = conformal_loss(p_ns, non_square_mesh, w_ns)
    print(
        "non-square angular conformal mean small?:",
        bool(_conformal_mean(ec_ns, non_square_mesh) < 1e-4),
    )

    # 9) Invalid input-FOV vertices do not contribute to core energies.
    wide_mesh = build_mesh(n_lambda=61, n_phi=41, horizontal_span_deg=140.0)
    valid_np = compute_valid_mesh_mask(Camera(CameraConfig(fov_deg=70.0, width=100, height=80)), wide_mesh)
    valid = torch.as_tensor(valid_np, dtype=torch.bool)
    lam_wide = torch.from_numpy(wide_mesh.lambda_grid)
    phi_wide = torch.from_numpy(wide_mesh.phi_grid)
    p_wide = stereographic(lam_wide, phi_wide)
    w_wide = torch.ones_like(lam_wide)
    w_wide[~valid] = 0.0
    p_corrupt = p_wide.clone()
    p_corrupt[~valid] = p_corrupt[~valid] + torch.tensor([1000.0, -1000.0], dtype=p_corrupt.dtype)

    ec_valid = conformal_loss(p_wide, wide_mesh, w_wide, valid_mask=valid)
    ec_corrupt = conformal_loss(p_corrupt, wide_mesh, w_wide, valid_mask=valid)
    es_valid = smoothness_loss(p_wide, wide_mesh, w_wide, valid_mask=valid)
    es_corrupt = smoothness_loss(p_corrupt, wide_mesh, w_wide, valid_mask=valid)
    mask_all = torch.ones((1,) + valid.shape, dtype=torch.bool)
    edvc_valid = dvc_loss(p_wide, p_wide.unsqueeze(0), mask_all, valid_mask=valid)
    print("invalid weights zero after mask?:", bool((~valid).any() and torch.all(w_wide[~valid] == 0.0)))
    print(
        "invalid vertices ignored by conformal/smoothness?:",
        bool(torch.allclose(ec_valid, ec_corrupt) and torch.allclose(es_valid, es_corrupt)),
    )
    print("DVC valid mask keeps matching target zero?:", bool(torch.allclose(edvc_valid, torch.zeros_like(edvc_valid))))

    # 10) Line samples use a full bilinear validity footprint.
    line_valid = torch.ones_like(w, dtype=torch.bool)
    line_valid[14:17, 19:22] = False
    sample_lam, sample_phi = _line_points_tensor(line, p)
    sample_valid = _query_bilinear_validity(mesh, sample_lam, sample_phi, p, line_valid)
    p_line = p.detach().clone()
    p_line_corrupt = p_line.clone()
    p_line_corrupt[~line_valid] = p_line_corrupt[~line_valid] + torch.tensor([1000.0, -1000.0], dtype=p_line.dtype)
    line_clean = line_loss(p_line, [line], mesh, n_samples=32, valid_mask=line_valid)
    line_corrupt = line_loss(p_line_corrupt, [line], mesh, n_samples=32, valid_mask=line_valid)
    print("line footprint filters some samples?:", bool(2 <= int(sample_valid.sum()) < sample_valid.numel()))
    print("line loss ignores corrupt invalid footprint?:", bool(torch.allclose(line_clean, line_corrupt)))
