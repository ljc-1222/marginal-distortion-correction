"""Final image rendering from the optimized mesh.

Given the per-vertex mapping ``p_{i,j}`` produced by Stage 2, every output
pixel is located in its enclosing mesh quadrilateral, the inverse bilinear
weights recovered, and the resulting view-sphere direction sampled from the
input image through :class:`Camera`.
"""

from __future__ import annotations

import numpy as np

from . import Array, MeshGrid
from .camera import Camera


_EPS = 1e-12


def _to_numpy(value: object) -> Array:
    """Convert numpy-like or torch-like values to a CPU numpy array."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_inputs(image: Array, mesh: MeshGrid, p_final: Array, out_size: tuple[int, int]) -> None:
    """Validate rendering inputs before rasterization."""
    if image.ndim not in (2, 3):
        raise ValueError(f"image must have shape (H, W) or (H, W, C); got {image.shape}.")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"image height and width must be positive; got {image.shape[:2]}.")
    if len(out_size) != 2 or out_size[0] <= 0 or out_size[1] <= 0:
        raise ValueError(f"out_size must be a positive (H, W) tuple; got {out_size}.")
    if p_final.ndim != 3 or p_final.shape[-1] != 2:
        raise ValueError(f"p_final must have shape (H, W, 2); got {p_final.shape}.")
    if p_final.shape[:2] != mesh.lambda_grid.shape or p_final.shape[:2] != mesh.phi_grid.shape:
        raise ValueError(
            "p_final and mesh dimensions must match; got "
            f"p_final {p_final.shape[:2]}, lambda {mesh.lambda_grid.shape}, phi {mesh.phi_grid.shape}."
        )
    if not np.isfinite(p_final).all():
        raise ValueError("p_final must contain only finite coordinates.")


def _validate_valid_mesh_mask(valid_mesh_mask: Array | None, mesh: MeshGrid) -> Array | None:
    """Return a validated boolean mesh mask, or ``None`` when not supplied."""
    if valid_mesh_mask is None:
        return None
    valid = np.asarray(valid_mesh_mask, dtype=bool)
    if valid.shape != mesh.lambda_grid.shape:
        raise ValueError(f"valid_mesh_mask must have shape {mesh.lambda_grid.shape}; got {valid.shape}.")
    return valid


def _fit_projection_to_output(
    p_final: Array,
    out_size: tuple[int, int],
    valid_mesh_mask: Array | None = None,
) -> tuple[float, Array]:
    """Return a uniform scale and translation from projection plane to pixels."""
    H_out, W_out = out_size
    if valid_mesh_mask is None:
        points = p_final.reshape(-1, 2)
    else:
        points = p_final[valid_mesh_mask]
        if points.size == 0:
            raise ValueError("valid_mesh_mask must contain at least one valid vertex.")

    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    p_range = p_max - p_min
    if np.any(p_range <= _EPS):
        raise ValueError(f"p_final projection bounds are degenerate: min {p_min}, max {p_max}.")

    scale_x = max(W_out - 1, 1) / p_range[0]
    scale_y = max(H_out - 1, 1) / p_range[1]
    scale = float(min(scale_x, scale_y))

    p_center = 0.5 * (p_min + p_max)
    out_center = np.array([(W_out - 1) / 2.0, (H_out - 1) / 2.0], dtype=np.float64)
    offset = out_center - scale * p_center
    return scale, offset


def _projection_to_pixel(p: Array, scale: float, offset: Array) -> Array:
    """Map projection-plane points to output pixel coordinates."""
    return scale * p + offset


def _pixel_to_projection(x: Array, y: Array, scale: float, offset: Array) -> Array:
    """Map output pixel coordinates back to projection-plane points."""
    u = (x - offset[0]) / scale
    v = (y - offset[1]) / scale
    return np.stack((u, v), axis=-1)


def _cross2(a: Array, b: Array) -> Array:
    """2D cross product supporting vectorized arrays."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inverse_bilinear(quad: Array, points: Array) -> tuple[Array, Array, Array]:
    """Recover bilinear ``(s, t)`` coordinates of ``points`` in ``quad``."""
    q00, q01, q10, q11 = quad
    a = q00
    b = q01 - q00
    c = q10 - q00
    d = q00 - q01 - q10 + q11

    rhs = points - a
    det = _cross2(b, c)
    if abs(float(det)) > _EPS:
        s = _cross2(rhs, c) / det
        t = _cross2(b, rhs) / det
    else:
        q_min = quad.min(axis=0)
        q_range = np.maximum(quad.max(axis=0) - q_min, _EPS)
        st = (points - q_min) / q_range
        s = st[:, 0]
        t = st[:, 1]

    valid = np.ones(points.shape[0], dtype=bool)
    for _ in range(8):
        mapped = a + b * s[:, None] + c * t[:, None] + d * (s * t)[:, None]
        residual = mapped - points
        ds = b + d * t[:, None]
        dt = c + d * s[:, None]
        det_j = _cross2(ds, dt)
        step_valid = np.abs(det_j) > _EPS
        valid &= step_valid
        if not bool(step_valid.any()):
            break

        step_s = np.zeros_like(s)
        step_t = np.zeros_like(t)
        step_s[step_valid] = _cross2(residual[step_valid], dt[step_valid]) / det_j[step_valid]
        step_t[step_valid] = _cross2(ds[step_valid], residual[step_valid]) / det_j[step_valid]
        s = np.clip(s - step_s, -0.5, 1.5)
        t = np.clip(t - step_t, -0.5, 1.5)

    mapped = a + b * s[:, None] + c * t[:, None] + d * (s * t)[:, None]
    residual_norm = np.linalg.norm(mapped - points, axis=1)
    return s, t, valid & np.isfinite(residual_norm)


def _bilinear_sample_image(image: Array, x: Array, y: Array) -> tuple[Array, Array]:
    """Bilinearly sample ``image`` at floating source pixel coordinates."""
    H, W, C = image.shape
    in_bounds = (x >= 0.0) & (x <= W - 1.0) & (y >= 0.0) & (y <= H - 1.0)
    if not bool(in_bounds.any()):
        return np.zeros((x.shape[0], C), dtype=np.float64), in_bounds

    x_c = np.clip(x, 0.0, W - 1.0)
    y_c = np.clip(y, 0.0, H - 1.0)
    x0 = np.floor(x_c).astype(np.int64)
    y0 = np.floor(y_c).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (x_c - x0)[:, None]
    fy = (y_c - y0)[:, None]

    sampled = (
        image[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + image[y0, x1] * fx * (1.0 - fy)
        + image[y1, x0] * (1.0 - fx) * fy
        + image[y1, x1] * fx * fy
    )
    sampled[~in_bounds] = 0.0
    return sampled, in_bounds


def _restore_dtype(image: Array, dtype: np.dtype, squeeze: bool) -> Array:
    """Cast the floating render buffer back to the input image dtype."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        image = np.clip(np.rint(image), info.min, info.max).astype(dtype)
    elif dtype == np.bool_:
        image = image > 0.5
    else:
        image = image.astype(dtype, copy=False)
    return image[..., 0] if squeeze else image


def _content_bbox(mask: Array) -> tuple[int, int, int, int] | None:
    """Return ``(top, left, bottom, right)`` bounding all true pixels."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def _window_invalid_counts(integral: Array, height: int, width: int) -> Array:
    """Return invalid-pixel counts for every ``height`` by ``width`` window."""
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def _choose_centered_valid_window(integral: Array, height: int, width: int) -> tuple[int, int] | None:
    """Return the valid window closest to the mask center."""
    invalid_counts = _window_invalid_counts(integral, height, width)
    rows, cols = np.nonzero(invalid_counts == 0)
    if rows.size == 0:
        return None

    target_y = (integral.shape[0] - 2) / 2.0
    target_x = (integral.shape[1] - 2) / 2.0
    centers_y = rows + (height - 1) / 2.0
    centers_x = cols + (width - 1) / 2.0
    best = int(np.argmin((centers_y - target_y) ** 2 + (centers_x - target_x) ** 2))
    return int(rows[best]), int(cols[best])


def _largest_aspect_true_rectangle(mask: Array, aspect_ratio: float) -> tuple[int, int, int, int] | None:
    """Return the largest all-true rectangle matching ``aspect_ratio``."""
    bbox = _content_bbox(mask)
    if bbox is None:
        return None

    base_top, base_left, base_bottom, base_right = bbox
    mask_box = mask[base_top : base_bottom + 1, base_left : base_right + 1]
    H, W = mask_box.shape
    max_height = min(H, int(np.floor((W + 0.5) / aspect_ratio)))
    if max_height <= 0:
        return None

    invalid = np.logical_not(mask_box).astype(np.int32, copy=False)
    integral = np.zeros((H + 1, W + 1), dtype=np.int32)
    integral[1:, 1:] = invalid.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)

    best_height = 0
    lo, hi = 1, max_height
    while lo <= hi:
        height = (lo + hi) // 2
        width = max(1, int(np.floor(height * aspect_ratio + 0.5)))
        if width > W:
            hi = height - 1
            continue

        has_valid_window = bool(np.any(_window_invalid_counts(integral, height, width) == 0))
        if has_valid_window:
            best_height = height
            lo = height + 1
        else:
            hi = height - 1

    if best_height == 0:
        return None

    best_width = max(1, int(np.floor(best_height * aspect_ratio + 0.5)))
    window = _choose_centered_valid_window(integral, best_height, best_width)
    if window is None:
        return None

    top, left = window
    return (
        base_top + top,
        base_left + left,
        base_top + top + best_height - 1,
        base_left + left + best_width - 1,
    )


def warp_image(
    image: Array,
    camera: Camera,
    mesh: MeshGrid,
    p_final: Array,
    out_size: tuple[int, int],
    return_mask: bool = False,
    valid_mesh_mask: Array | None = None,
) -> Array | tuple[Array, Array]:
    """Render the corrected output image.

    Args:
        image: Input image, shape ``(H_img, W_img, 3)``.
        camera: The input camera, used to sample the input image at the
            recovered view-sphere directions.
        mesh: The view-sphere mesh.
        p_final: Per-vertex output coordinates from Stage 2, shape
            ``(H, W, 2)``.
        out_size: Output image size as ``(H_out, W_out)``.
        return_mask: When true, also return a boolean mask of pixels filled by
            the warp.
        valid_mesh_mask: Optional boolean mask of mesh vertices supported by
            the input FOV. Quads are rasterized only when all four vertices are
            valid, and projection fitting uses only valid vertices.

    Returns:
        Output image of shape ``(H_out, W_out, 3)``. If ``return_mask`` is
        true, returns ``(image, mask)``.
    """
    image_arr = np.asarray(image)
    p_arr = _to_numpy(p_final).astype(np.float64, copy=False)
    _validate_inputs(image_arr, mesh, p_arr, out_size)
    valid_mesh = _validate_valid_mesh_mask(valid_mesh_mask, mesh)

    H_out, W_out = out_size
    squeeze = image_arr.ndim == 2
    image_f = image_arr[..., None] if squeeze else image_arr
    image_f = image_f.astype(np.float64, copy=False)
    H_mesh, W_mesh = mesh.lambda_grid.shape

    scale, offset = _fit_projection_to_output(p_arr, out_size, valid_mesh)
    p_pixels = _projection_to_pixel(p_arr, scale, offset)
    output = np.zeros((H_out, W_out, image_f.shape[2]), dtype=np.float64)
    filled = np.zeros((H_out, W_out), dtype=bool)

    lam_grid = np.asarray(mesh.lambda_grid, dtype=np.float64)
    phi_grid = np.asarray(mesh.phi_grid, dtype=np.float64)

    for i in range(H_mesh - 1):
        for j in range(W_mesh - 1):
            if valid_mesh is not None and not bool(
                valid_mesh[i, j]
                and valid_mesh[i, j + 1]
                and valid_mesh[i + 1, j]
                and valid_mesh[i + 1, j + 1]
            ):
                continue

            quad_plane = np.array(
                [p_arr[i, j], p_arr[i, j + 1], p_arr[i + 1, j], p_arr[i + 1, j + 1]],
                dtype=np.float64,
            )
            quad_pixel = np.array(
                [
                    p_pixels[i, j],
                    p_pixels[i, j + 1],
                    p_pixels[i + 1, j],
                    p_pixels[i + 1, j + 1],
                ],
                dtype=np.float64,
            )

            x0 = max(0, int(np.floor(quad_pixel[:, 0].min())) - 1)
            x1 = min(W_out - 1, int(np.ceil(quad_pixel[:, 0].max())) + 1)
            y0 = max(0, int(np.floor(quad_pixel[:, 1].min())) - 1)
            y1 = min(H_out - 1, int(np.ceil(quad_pixel[:, 1].max())) + 1)
            if x0 > x1 or y0 > y1:
                continue

            yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            flat_y = yy.reshape(-1)
            flat_x = xx.reshape(-1)
            not_filled = ~filled[flat_y, flat_x]
            if not bool(not_filled.any()):
                continue

            flat_y = flat_y[not_filled]
            flat_x = flat_x[not_filled]
            points = _pixel_to_projection(flat_x.astype(np.float64), flat_y.astype(np.float64), scale, offset)
            s, t, valid = _inverse_bilinear(quad_plane, points)

            tol = 1e-6
            residual_tol = max(1e-6, 0.25 / scale)
            q_reconstructed = (
                quad_plane[0]
                + (quad_plane[1] - quad_plane[0]) * s[:, None]
                + (quad_plane[2] - quad_plane[0]) * t[:, None]
                + (quad_plane[0] - quad_plane[1] - quad_plane[2] + quad_plane[3]) * (s * t)[:, None]
            )
            residual = np.linalg.norm(q_reconstructed - points, axis=1)
            inside = valid & (s >= -tol) & (s <= 1.0 + tol) & (t >= -tol) & (t <= 1.0 + tol)
            inside &= residual <= residual_tol
            if not bool(inside.any()):
                continue

            s_i = np.clip(s[inside], 0.0, 1.0)
            t_i = np.clip(t[inside], 0.0, 1.0)
            out_x = flat_x[inside]
            out_y = flat_y[inside]

            lam = (
                lam_grid[i, j] * (1.0 - s_i) * (1.0 - t_i)
                + lam_grid[i, j + 1] * s_i * (1.0 - t_i)
                + lam_grid[i + 1, j] * (1.0 - s_i) * t_i
                + lam_grid[i + 1, j + 1] * s_i * t_i
            )
            phi = (
                phi_grid[i, j] * (1.0 - s_i) * (1.0 - t_i)
                + phi_grid[i, j + 1] * s_i * (1.0 - t_i)
                + phi_grid[i + 1, j] * (1.0 - s_i) * t_i
                + phi_grid[i + 1, j + 1] * s_i * t_i
            )
            in_fov = camera.direction_in_fov(lam, phi)
            src_x, src_y = camera.direction_to_pixel(lam, phi)
            sampled, in_bounds = _bilinear_sample_image(image_f, src_x, src_y)
            valid_sample = in_fov & in_bounds
            if not bool(valid_sample.any()):
                continue

            output[out_y[valid_sample], out_x[valid_sample]] = sampled[valid_sample]
            filled[out_y[valid_sample], out_x[valid_sample]] = True

    rendered = _restore_dtype(output, image_arr.dtype, squeeze)
    if return_mask:
        return rendered, filled.copy()
    return rendered


def crop_to_rect(
    image: Array,
    valid_mask: Array | None = None,
    target_aspect: tuple[int, int] | None = None,
) -> Array:
    """Crop the warped output to the maximal content bounding rectangle.

    Args:
        image: Warped output of shape ``(H, W, 3)`` possibly with
            transparent / black borders from the warp.
        valid_mask: Optional boolean mask of valid warped pixels. When
            provided, the crop is based on this mask instead of pixel color.
        target_aspect: Optional ``(width, height)`` aspect to preserve while
            finding the largest all-valid crop rectangle.

    Returns:
        Cropped image of shape ``(H', W', 3)`` with outer black borders
        removed while preserving as many pixels as possible.
    """
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"image must have shape (H, W) or (H, W, C); got {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return arr.copy()

    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != arr.shape[:2]:
            raise ValueError(f"valid_mask must have shape {arr.shape[:2]}; got {valid.shape}.")
    elif arr.ndim == 3 and arr.shape[2] == 4:
        valid = arr[..., 3] != 0
    elif arr.ndim == 3:
        valid = np.any(arr != 0, axis=-1)
    else:
        valid = arr != 0

    if target_aspect is not None:
        try:
            aspect_width, aspect_height = target_aspect
        except (TypeError, ValueError) as exc:
            raise ValueError("target_aspect must be a (width, height) pair.") from exc
        aspect_width = int(aspect_width)
        aspect_height = int(aspect_height)
        if aspect_width <= 0 or aspect_height <= 0:
            raise ValueError(f"target_aspect dimensions must be positive; got {target_aspect}.")
        rect = _largest_aspect_true_rectangle(valid, aspect_width / aspect_height)
    else:
        rect = _content_bbox(valid)

    if rect is None:
        return arr.copy()

    top, left, bottom, right = rect
    return arr[top : bottom + 1, left : right + 1].copy()


if __name__ == "__main__":
    from . import CameraConfig
    from .mesh import build_mesh

    # 1) Rendering a direction-coded image through p=(lambda, phi) recovers
    # the same direction code up to image/grid interpolation.
    width, height = 96, 64
    cam = Camera(CameraConfig(fov_deg=70.0, width=width, height=height))
    xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    lam_img, phi_img = cam.pixel_to_direction(xs, ys)
    image = np.stack(
        (
            (lam_img + 0.8) / 1.6,
            (phi_img + 0.5) / 1.0,
            np.full_like(lam_img, 0.5),
        ),
        axis=-1,
    ).astype(np.float64)

    mesh = build_mesh(n_lambda=23, n_phi=15, horizontal_span_deg=50.0)
    p_identity = np.stack((mesh.lambda_grid, mesh.phi_grid), axis=-1)
    out = warp_image(image, cam, mesh, p_identity, (45, 69))
    scale, offset = _fit_projection_to_output(p_identity, (45, 69))
    cy, cx = 22, 34
    expected_lam_phi = _pixel_to_projection(np.array([cx], dtype=np.float64), np.array([cy], dtype=np.float64), scale, offset)[0]
    expected = np.array([(expected_lam_phi[0] + 0.8) / 1.6, (expected_lam_phi[1] + 0.5) / 1.0, 0.5])
    print("warp output shape:", out.shape, "(expected (45, 69, 3))")
    print("identity-direction sample close?:", bool(np.allclose(out[cy, cx], expected, atol=3e-2)))
    print("warp finite?:", bool(np.isfinite(out).all()))

    # 2) Torch-like inputs are accepted via detach/cpu conversion.
    try:
        import torch

        out_torch = warp_image(image, cam, mesh, torch.from_numpy(p_identity), (45, 69))
        print("torch p_final supported?:", bool(np.allclose(out_torch, out)))
    except ImportError:
        print("torch p_final supported?: skipped")

    # 3) Integer inputs preserve dtype and produce nonzero rendered content.
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    out_u8 = warp_image(image_u8, cam, mesh, p_identity, (45, 69))
    print("uint8 dtype preserved?:", out_u8.dtype == np.uint8)
    print("uint8 has content?:", bool(np.any(out_u8 != 0)))

    # 4) Crop uses the validity mask, so real black content is preserved.
    framed = np.zeros((6, 7, 3), dtype=np.uint8)
    framed[1:5, 1:6] = 10
    valid_framed = np.zeros((6, 7), dtype=bool)
    valid_framed[1:5, 1:6] = True
    framed[1, 1:6] = 0
    framed[2, 3] = 0
    cropped = crop_to_rect(framed, valid_framed)
    print("crop shape maximal bbox?:", cropped.shape[:2], "(expected (4, 5))")
    cropped_square = crop_to_rect(framed, valid_framed, target_aspect=(1, 1))
    print("crop shape aspect-preserving square?:", cropped_square.shape[:2], "(expected (4, 4))")

    # 5) Valid mesh masks skip invalid-domain quads.
    valid_mesh = np.ones(mesh.lambda_grid.shape, dtype=bool)
    valid_mesh[:, -3:] = False
    _, full_filled = warp_image(image, cam, mesh, p_identity, (45, 69), return_mask=True)
    out_valid, valid_filled = warp_image(
        image,
        cam,
        mesh,
        p_identity,
        (45, 69),
        return_mask=True,
        valid_mesh_mask=valid_mesh,
    )
    print("valid mesh mask skips some quads?:", bool(valid_filled.sum() < full_filled.sum()))

    # 6) Invalid p vertices do not affect fitting or rasterization.
    p_corrupt = p_identity.copy()
    p_corrupt[~valid_mesh] += np.array([1000.0, -1000.0], dtype=np.float64)
    out_corrupt, corrupt_filled = warp_image(
        image,
        cam,
        mesh,
        p_corrupt,
        (45, 69),
        return_mask=True,
        valid_mesh_mask=valid_mesh,
    )
    scale_valid, offset_valid = _fit_projection_to_output(p_identity, (45, 69), valid_mesh)
    scale_corrupt, offset_corrupt = _fit_projection_to_output(p_corrupt, (45, 69), valid_mesh)
    print("valid-mask render ignores corrupt invalid vertices?:", bool(np.allclose(out_valid, out_corrupt) and np.array_equal(valid_filled, corrupt_filled)))
    print("valid-mask fit ignores corrupt invalid vertices?:", bool(np.allclose(scale_valid, scale_corrupt) and np.allclose(offset_valid, offset_corrupt)))

    # 7) Invalid shapes are rejected early.
    try:
        warp_image(image, cam, mesh, p_identity[:-1], (45, 69))
    except ValueError as exc:
        print("bad p_final rejected?:", "p_final" in str(exc))

    try:
        warp_image(image, cam, mesh, p_identity, (45, 69), valid_mesh_mask=valid_mesh[:-1])
    except ValueError as exc:
        print("bad valid_mesh_mask rejected?:", "valid_mesh_mask" in str(exc))
