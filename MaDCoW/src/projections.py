"""View-sphere to image-plane projections.

Three projections are needed by the pipeline:

* :func:`stereographic` -- the initial mapping ``P_sg`` that avoids marginal
  distortion (Section 3).
* :func:`perspective` -- linear perspective ``P(v)`` with optical axis
  ``v_P`` (Section 4.1).
* :func:`T_projection` -- the parameterized region projection
  ``T(v) = S * P(v) + t`` (Eq. 1) with similarity ``S`` and translation
  ``t``; this is what Stage 1 fits per ROI.

Conventions (shared with :mod:`camera`):
    * Direction of ``(lambda, phi)`` is
      ``(sin(lambda) * cos(phi), sin(phi), cos(lambda) * cos(phi))``
      in a right-handed frame with ``+X`` right, ``+Y`` down, ``+Z`` forward.
    * Stereographic uses ``u = 2 X / (1 + Z),  v = 2 Y / (1 + Z)`` so that
      near the optical axis it matches the standard perspective ``u = X/Z``
      to first order.
    * The similarity matrix from Eq. 1 is ``S = [[a, b], [-b, a]]``.

All functions accept and return :class:`torch.Tensor` objects to integrate
with autograd.
"""

from __future__ import annotations

import torch

from . import Tensor


def _direction_vector(lam: Tensor, phi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert ``(lambda, phi)`` into 3D unit-direction components."""
    cos_phi = torch.cos(phi)
    X = torch.sin(lam) * cos_phi
    Y = torch.sin(phi)
    Z = torch.cos(lam) * cos_phi
    return X, Y, Z


def stereographic(lam: Tensor, phi: Tensor) -> Tensor:
    """Stereographic projection from the view sphere to the image plane.

    Args:
        lam: Yaw angles in radians, any shape ``S``.
        phi: Pitch angles in radians, same shape as ``lam``.

    Returns:
        Tensor of shape ``S + (2,)`` containing ``(u, v)`` image-plane
        coordinates.
    """
    X, Y, Z = _direction_vector(lam, phi)
    denom = 1.0 + Z
    u = 2.0 * X / denom
    v = 2.0 * Y / denom
    return torch.stack((u, v), dim=-1)


def perspective(lam: Tensor, phi: Tensor, v_p: Tensor) -> Tensor:
    """Linear perspective projection with arbitrary optical axis.

    The view-sphere direction is rotated so that ``v_p`` aligns with the
    camera's forward axis (``+Z``), then projected through a pinhole.

    The alignment rotation is ``R = R_x(phi_P) @ R_y(-lambda_P)``: first
    spin around ``Y`` by ``-lambda_P`` to zero out yaw, then around ``X`` by
    ``+phi_P`` to zero out pitch (see derivation in module docstring).

    Args:
        lam: Yaw angles in radians, any shape ``S``.
        phi: Pitch angles in radians, same shape as ``lam``.
        v_p: Optical axis direction ``(lambda_P, phi_P)`` in radians, shape
            ``(2,)``.

    Returns:
        Tensor of shape ``S + (2,)`` containing ``(u, v)`` image-plane
        coordinates of the perspective projection ``P(v)``.
    """
    lam_p = v_p[0]
    phi_p = v_p[1]

    X, Y, Z = _direction_vector(lam, phi)

    # R_y(-lambda_P): yaw alignment.
    cl = torch.cos(lam_p)
    sl = torch.sin(lam_p)
    X1 = X * cl - Z * sl
    Y1 = Y
    Z1 = X * sl + Z * cl

    # R_x(+phi_P): pitch alignment.
    cp = torch.cos(phi_p)
    sp = torch.sin(phi_p)
    X2 = X1
    Y2 = Y1 * cp - Z1 * sp
    Z2 = Y1 * sp + Z1 * cp

    u = X2 / Z2
    v = Y2 / Z2
    return torch.stack((u, v), dim=-1)


def affine_2d(uv: Tensor, a: Tensor, b: Tensor, tx: Tensor, ty: Tensor) -> Tensor:
    """Apply the similarity ``S = [[a, b], [-b, a]]`` and translation ``t``.

    Args:
        uv: Input image-plane coordinates, shape ``S + (2,)``.
        a: Similarity scalar.
        b: Similarity scalar.
        tx: Translation x.
        ty: Translation y.

    Returns:
        Tensor of shape ``S + (2,)`` containing ``S @ uv + t``.
    """
    u = uv[..., 0]
    v = uv[..., 1]
    u_new = a * u + b * v + tx
    v_new = -b * u + a * v + ty
    return torch.stack((u_new, v_new), dim=-1)


def T_projection(lam: Tensor, phi: Tensor, params: Tensor) -> Tensor:
    """Evaluate the per-ROI projection ``T(v) = S * P(v) + t`` (Eq. 1).

    Args:
        lam: Yaw angles in radians, any shape ``S``.
        phi: Pitch angles in radians, same shape as ``lam``.
        params: 1D tensor of shape ``(6,)`` packing
            ``(lambda_P, phi_P, a, b, tx, ty)``.

    Returns:
        Tensor of shape ``S + (2,)`` containing the ``(u, v)`` outputs of
        ``T``.
    """
    v_p = params[0:2]
    a, b, tx, ty = params[2], params[3], params[4], params[5]
    uv = perspective(lam, phi, v_p)
    return affine_2d(uv, a, b, tx, ty)


if __name__ == "__main__":
    import math

    # 1) Stereographic origin maps to (0, 0).
    z = torch.tensor(0.0)
    p_sg = stereographic(z, z)
    print("stereographic(0, 0):", p_sg.tolist(), "(expected [0, 0])")

    # 2) Stereographic and perspective agree to first order near origin.
    angles = torch.linspace(-0.05, 0.05, 5)
    p_sg = stereographic(angles, torch.zeros_like(angles))
    p_per = perspective(angles, torch.zeros_like(angles), torch.zeros(2))
    max_diff = (p_sg - p_per).abs().max().item()
    print(f"|stereographic - perspective| near origin: max diff = {max_diff:.2e}")

    # 3) Perspective with v_p mapping the principal point to origin.
    v_p = torch.tensor([math.radians(20.0), math.radians(-10.0)])
    p = perspective(v_p[0], v_p[1], v_p)
    print("perspective(v_p, v_p):", p.tolist(), "(expected [0, 0])")

    # 4) Identity affine.
    uv = torch.tensor([[1.0, 2.0], [3.0, -4.0]])
    out = affine_2d(uv, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
    print("identity affine equal to input?:", torch.equal(out, uv))

    # 5) 90-degree rotation via similarity (a=0, b=1).
    out = affine_2d(uv, torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0))
    # [a,b;-b,a] @ [u;v] = [v; -u]
    print("S=[[0,1],[-1,0]] applied:", out.tolist(), "(expected [[2,-1],[-4,-3]])")

    # 6) T_projection reduces to perspective when (a=1, b=tx=ty=0).
    params = torch.tensor([0.1, -0.05, 1.0, 0.0, 0.0, 0.0])
    lam_grid = torch.linspace(-0.3, 0.3, 4)
    phi_grid = torch.linspace(-0.2, 0.2, 4)
    t_out = T_projection(lam_grid, phi_grid, params)
    p_out = perspective(lam_grid, phi_grid, params[0:2])
    print(f"T == perspective with identity affine: max diff = {(t_out - p_out).abs().max().item():.2e}")

    # 7) Differentiability: gradients flow through T_projection.
    params = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.1, -0.2], requires_grad=True)
    out = T_projection(torch.tensor(0.05), torch.tensor(0.05), params)
    out.sum().backward()
    print("grad wrt params (shape, nonzero):", params.grad.shape, bool((params.grad != 0).any()))
