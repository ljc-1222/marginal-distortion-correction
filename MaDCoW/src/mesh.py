"""View-sphere mesh utilities used by every stage of the pipeline.

The mesh discretizes the input angular domain into a regular ``(H, W)`` grid
of directions ``(lambda, phi)`` in radians. The Stage 1 / Stage 2 optimizers
treat the per-vertex output coordinates ``p_{i,j}`` as the unknowns.

This module also provides utilities to map ROI masks from input-image space
onto the mesh, identify boundary vertices (set ``V`` in Eq. 2 of the paper),
and sample a per-vertex 2D field by bilinear interpolation.

Conventions:
    * Mesh layout uses ``numpy.meshgrid(..., indexing="xy")``: row index
      ``i`` runs over ``phi`` (height) and column index ``j`` runs over
      ``lambda`` (width). Hence shape is ``(n_phi, n_lambda) == (H, W)``.
    * The mesh is regular -- ``delta_lambda`` and ``delta_phi`` are constants
      -- which makes inverse lookups in :func:`bilinear_sample` analytic.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import Array, MeshGrid, Tensor
from .camera import Camera


def _validate_mesh_resolution(n_lambda: int, n_phi: int) -> None:
    """Validate mesh resolution parameters."""
    if n_lambda < 2 or n_phi < 2:
        raise ValueError("n_lambda and n_phi must both be at least 2.")


def build_mesh(n_lambda: int, n_phi: int, horizontal_span_deg: float) -> MeshGrid:
    """Build a regular angular mesh from a symmetric horizontal span.

    The yaw range is ``[-span/2, +span/2]``. The pitch range is chosen so that
    the angular spacing along pitch matches the angular spacing along yaw,
    i.e. mesh cells are roughly square. As a consequence the vertical
    coverage is determined by ``n_phi / n_lambda``.

    This helper is kept for synthetic tests and standalone experiments. The
    MaDCoW pipeline should use :func:`build_input_domain_mesh`, because the
    mesh domain is the source image's input angular domain rather than an
    independently specified output-camera parameter.

    Args:
        n_lambda: Number of vertices along the yaw direction.
        n_phi: Number of vertices along the pitch direction.
        horizontal_span_deg: Symmetric horizontal angular span, in degrees.

    Returns:
        A :class:`MeshGrid` whose ``lambda_grid`` / ``phi_grid`` are both
        arrays of shape ``(n_phi, n_lambda)`` in radians.
    """
    _validate_mesh_resolution(n_lambda, n_phi)
    if horizontal_span_deg <= 0 or horizontal_span_deg >= 180:
        raise ValueError(
            f"horizontal_span_deg must lie in (0, 180); got {horizontal_span_deg}."
        )

    span_rad = math.radians(horizontal_span_deg)
    half_lambda = span_rad / 2.0
    delta = span_rad / (n_lambda - 1)
    half_phi = delta * (n_phi - 1) / 2.0

    lam_1d = np.linspace(-half_lambda, half_lambda, n_lambda, dtype=np.float64)
    phi_1d = np.linspace(-half_phi, half_phi, n_phi, dtype=np.float64)
    lam_grid, phi_grid = np.meshgrid(lam_1d, phi_1d, indexing="xy")

    return MeshGrid(lambda_grid=lam_grid, phi_grid=phi_grid)


def build_input_domain_mesh(camera: Camera, n_lambda: int, n_phi: int) -> MeshGrid:
    """Build a regular mesh covering the input image's angular domain.

    The input perspective camera defines the measured ray domain. We sample
    the image border, convert those pixels to view-sphere directions, and
    use the resulting yaw/pitch bounds as the rectangular angular domain for
    the MaDCoW mesh. This is a discretization of the source rays that can be
    reprojected by the optimized nonlinear mapping.

    Args:
        camera: Input camera whose pixel-to-ray mapping defines the domain.
        n_lambda: Number of vertices along yaw.
        n_phi: Number of vertices along pitch.

    Returns:
        A :class:`MeshGrid` covering the input image's view-sphere domain.
    """
    _validate_mesh_resolution(n_lambda, n_phi)

    width = camera.cfg.width
    height = camera.cfg.height
    x_edge = np.linspace(0.0, width - 1.0, max(width, 2), dtype=np.float64)
    y_edge = np.linspace(0.0, height - 1.0, max(height, 2), dtype=np.float64)
    xs = np.concatenate(
        (
            x_edge,
            x_edge,
            np.zeros_like(y_edge),
            np.full_like(y_edge, width - 1.0),
        )
    )
    ys = np.concatenate(
        (
            np.zeros_like(x_edge),
            np.full_like(x_edge, height - 1.0),
            y_edge,
            y_edge,
        )
    )
    lam_border, phi_border = camera.pixel_to_direction(xs, ys)

    lam_1d = np.linspace(float(lam_border.min()), float(lam_border.max()), n_lambda)
    phi_1d = np.linspace(float(phi_border.min()), float(phi_border.max()), n_phi)
    lam_grid, phi_grid = np.meshgrid(lam_1d, phi_1d, indexing="xy")
    return MeshGrid(lambda_grid=lam_grid, phi_grid=phi_grid)


def mesh_angular_steps(mesh: MeshGrid) -> tuple[float, float]:
    """Return average angular steps ``(delta_lambda, delta_phi)`` in radians."""
    H, W = mesh.lambda_grid.shape
    if H < 2 or W < 2:
        raise ValueError(f"mesh must have at least 2 vertices per axis; got {(H, W)}.")
    if mesh.phi_grid.shape != (H, W):
        raise ValueError(
            f"mesh.lambda_grid and mesh.phi_grid shapes differ: {mesh.lambda_grid.shape} vs {mesh.phi_grid.shape}."
        )

    delta_lambda = (float(mesh.lambda_grid[0, -1]) - float(mesh.lambda_grid[0, 0])) / (W - 1)
    delta_phi = (float(mesh.phi_grid[-1, 0]) - float(mesh.phi_grid[0, 0])) / (H - 1)
    if delta_lambda == 0.0 or delta_phi == 0.0:
        raise ValueError(f"mesh angular steps must be nonzero; got {(delta_lambda, delta_phi)}.")
    return abs(delta_lambda), abs(delta_phi)


def compute_valid_mesh_mask(camera: Camera, mesh: MeshGrid) -> Array:
    """Return boolean mask of mesh vertices whose directions project inside the input FOV."""
    return np.asarray(camera.direction_in_fov(mesh.lambda_grid, mesh.phi_grid), dtype=bool)


def rasterize_mask_to_mesh(mask: Array, camera: Camera, mesh: MeshGrid) -> Array:
    """Project an input-space ROI mask onto the mesh vertices.

    For every mesh vertex, the corresponding ``(lambda, phi)`` direction is
    mapped back to the input image with :meth:`Camera.direction_to_pixel`
    and the mask value sampled there. Vertices whose direction falls outside
    the FOV are marked ``False``.

    Args:
        mask: Boolean / uint8 mask in input-image space, shape
            ``(H_img, W_img)``. Truthy entries are inside the ROI.
        camera: The input :class:`Camera`.
        mesh: The view-sphere mesh.

    Returns:
        Boolean array of shape ``(H_mesh, W_mesh)`` marking mesh vertices
        whose corresponding input pixel is inside the mask.
    """
    mask_f = np.asarray(mask).astype(np.float32)
    H_img, W_img = mask_f.shape

    in_fov = camera.direction_in_fov(mesh.lambda_grid, mesh.phi_grid)
    x, y = camera.direction_to_pixel(mesh.lambda_grid, mesh.phi_grid)

    # Bilinear sample with safe clipping; values outside the FOV will be
    # masked out below regardless of their sampled value.
    x_c = np.clip(x, 0.0, W_img - 1.0)
    y_c = np.clip(y, 0.0, H_img - 1.0)
    x0 = np.floor(x_c).astype(np.int64)
    y0 = np.floor(y_c).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W_img - 1)
    y1 = np.clip(y0 + 1, 0, H_img - 1)
    fx = x_c - x0
    fy = y_c - y0

    sampled = (
        mask_f[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + mask_f[y0, x1] * fx * (1.0 - fy)
        + mask_f[y1, x0] * (1.0 - fx) * fy
        + mask_f[y1, x1] * fx * fy
    )
    return (sampled >= 0.5) & in_fov


def boundary_vertices(mask_mesh: Array) -> Array:
    """Return the indices of mesh vertices on the ROI boundary.

    A vertex ``(i, j)`` belongs to the boundary set ``V`` (Eq. 2 of the
    paper) when at least one but not all of ``{(i, j), (i + 1, j),
    (i, j + 1)}`` lie within the region of interest.

    Args:
        mask_mesh: Boolean array of shape ``(H, W)`` produced by
            :func:`rasterize_mask_to_mesh`.

    Returns:
        Integer array of shape ``(N, 2)``, each row a ``(i, j)`` index of a
        boundary vertex.
    """
    mask_b = np.asarray(mask_mesh).astype(bool)
    # Triangles only exist where (i, j), (i+1, j), (i, j+1) are all valid,
    # i.e. for i in [0, H-1) and j in [0, W-1).
    a = mask_b[:-1, :-1]
    b = mask_b[1:, :-1]
    c = mask_b[:-1, 1:]
    count = a.astype(np.int8) + b.astype(np.int8) + c.astype(np.int8)
    boundary = (count > 0) & (count < 3)

    return np.argwhere(boundary)


def bilinear_sample(
    p: Tensor,
    mesh: MeshGrid,
    query_lam: Tensor,
    query_phi: Tensor,
) -> Tensor:
    """Sample a per-vertex 2D field at arbitrary view-sphere directions.

    The mesh is regular in ``(lambda, phi)``; each query direction is mapped
    to a fractional grid index and bilinearly interpolated. Query points
    outside the mesh are clipped to the boundary.

    Args:
        p: Per-vertex output coordinates of shape ``(H, W, 2)``.
        mesh: The view-sphere mesh.
        query_lam: Query yaw angles, shape ``(M,)``.
        query_phi: Query pitch angles, shape ``(M,)``.

    Returns:
        Tensor of shape ``(M, 2)`` with the interpolated values.
    """
    H, W, _ = p.shape

    lam_min = float(mesh.lambda_grid[0, 0])
    lam_max = float(mesh.lambda_grid[0, -1])
    phi_min = float(mesh.phi_grid[0, 0])
    phi_max = float(mesh.phi_grid[-1, 0])
    delta_lam = (lam_max - lam_min) / (W - 1)
    delta_phi = (phi_max - phi_min) / (H - 1)

    j_f = (query_lam - lam_min) / delta_lam
    i_f = (query_phi - phi_min) / delta_phi
    j_f = torch.clamp(j_f, 0.0, float(W - 1))
    i_f = torch.clamp(i_f, 0.0, float(H - 1))

    j0 = torch.floor(j_f).long()
    i0 = torch.floor(i_f).long()
    j1 = torch.clamp(j0 + 1, max=W - 1)
    i1 = torch.clamp(i0 + 1, max=H - 1)
    fj = (j_f - j0.to(j_f.dtype)).unsqueeze(-1)
    fi = (i_f - i0.to(i_f.dtype)).unsqueeze(-1)

    p00 = p[i0, j0]
    p01 = p[i0, j1]
    p10 = p[i1, j0]
    p11 = p[i1, j1]

    return (
        p00 * (1.0 - fj) * (1.0 - fi)
        + p01 * fj * (1.0 - fi)
        + p10 * (1.0 - fj) * fi
        + p11 * fj * fi
    )


if __name__ == "__main__":
    from . import CameraConfig

    # 1) Mesh has the expected shape, range, and centre.
    mesh = build_mesh(n_lambda=201, n_phi=121, horizontal_span_deg=120.0)
    print("mesh shape:", mesh.lambda_grid.shape, "(expected (121, 201))")
    print("lambda range:", float(mesh.lambda_grid.min()), "to", float(mesh.lambda_grid.max()))
    print("phi range:", float(mesh.phi_grid.min()), "to", float(mesh.phi_grid.max()))
    centre_lam = mesh.lambda_grid[121 // 2, 201 // 2]
    centre_phi = mesh.phi_grid[121 // 2, 201 // 2]
    print(f"mesh centre (lam, phi): ({centre_lam:.3e}, {centre_phi:.3e}) (expected 0, 0)")
    delta_lam, delta_phi = mesh_angular_steps(mesh)
    print("mesh angular steps positive?:", bool(delta_lam > 0.0 and delta_phi > 0.0))

    # 2) Rasterize a centred rectangular mask in input pixel space.
    cam = Camera(CameraConfig(fov_deg=90.0, width=800, height=600))
    mask = np.zeros((600, 800), dtype=bool)
    mask[200:400, 300:500] = True
    mask_mesh = rasterize_mask_to_mesh(mask, cam, mesh)
    print(
        "mask coverage (in / total):",
        int(mask_mesh.sum()),
        "/",
        mask_mesh.size,
        "(should be > 0 and < total)",
    )
    valid = compute_valid_mesh_mask(cam, mesh)
    print("valid mesh mask shape/has invalid?:", valid.shape, bool((~valid).any()))

    # 3) Boundary vertices for the same mask.
    boundary = boundary_vertices(mask_mesh)
    print("boundary vertex count:", boundary.shape[0])
    print("boundary indices dtype/shape:", boundary.dtype, boundary.shape)

    # 4) Boundary edge cases.
    full = np.ones((10, 10), dtype=bool)
    empty = np.zeros((10, 10), dtype=bool)
    print("boundary of full mask:", boundary_vertices(full).shape[0], "(expected 0)")
    print("boundary of empty mask:", boundary_vertices(empty).shape[0], "(expected 0)")

    # 5) bilinear_sample: identity field returns identity at vertices.
    H, W = mesh.lambda_grid.shape
    lam_t = torch.from_numpy(mesh.lambda_grid)
    phi_t = torch.from_numpy(mesh.phi_grid)
    p_identity = torch.stack((lam_t, phi_t), dim=-1)
    q_lam = lam_t[60, 100:103].clone()
    q_phi = phi_t[60, 100:103].clone()
    sampled = bilinear_sample(p_identity, mesh, q_lam, q_phi)
    expected = torch.stack((q_lam, q_phi), dim=-1)
    print(f"bilinear at vertices max err: {(sampled - expected).abs().max().item():.2e}")

    # 6) bilinear_sample at midpoint between two columns -> midpoint values.
    j = 100
    mid_lam = 0.5 * (lam_t[60, j] + lam_t[60, j + 1])
    sampled_mid = bilinear_sample(p_identity, mesh, mid_lam.unsqueeze(0), phi_t[60, j].unsqueeze(0))
    expected_mid = torch.tensor([float(mid_lam), float(phi_t[60, j])])
    print(f"bilinear midpoint err: {(sampled_mid[0] - expected_mid).abs().max().item():.2e}")

    # 7) bilinear_sample preserves gradients through p.
    p_grad = p_identity.clone().requires_grad_(True)
    out = bilinear_sample(p_grad, mesh, q_lam, q_phi)
    out.sum().backward()
    print("bilinear grad nonzero:", bool((p_grad.grad != 0).any()))
