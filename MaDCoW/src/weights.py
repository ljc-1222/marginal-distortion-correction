"""Per-vertex importance weights ``w_{i,j}`` for the conformal/smoothness loss.

Follows Carroll et al. (2009), as used by MaDCoW: vertices in high-frequency
image regions get larger weights, line endpoints receive additional Gaussian
weight to avoid local foldovers, and the face-detection term from Carroll is
omitted. The resulting weight is
``w = 1 + 2 * w_L + 2 * w_S``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from . import Array, LineAnnotation, MeshGrid
from .camera import Camera


_EPS = 1e-12
_BASELINE_WEIGHT = 1.0
_LINE_ENDPOINT_WEIGHT = 2.0
_SALIENCE_WEIGHT = 2.0
_LINE_SIGMA_SCALE = 100.0
_WINDOW_SCALE = 100.0


def _image_to_float_channels(image: Array) -> Array:
    """Convert an image to finite floating-point channels for gradients."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        channels = arr[..., None]
    elif arr.ndim == 3 and arr.shape[2] >= 1:
        channels = arr[..., : min(arr.shape[2], 3)]
    else:
        raise ValueError(
            "image must have shape (H, W) or (H, W, C) with at least one channel."
        )
    if channels.shape[0] == 0 or channels.shape[1] == 0:
        raise ValueError("image height and width must be positive.")

    channels = channels.astype(np.float64, copy=False)
    if arr.dtype == np.bool_:
        pass
    elif np.issubdtype(arr.dtype, np.integer):
        max_value = np.iinfo(arr.dtype).max
        if max_value > 0:
            channels = channels / float(max_value)

    return np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)


def _local_window_size(height: int, width: int) -> int:
    """Choose an odd local-salience window size from the input resolution."""
    radius = max(1, int(round(min(height, width) / _WINDOW_SCALE)))
    return 2 * radius + 1


def _local_color_stddev(image: Array) -> Array:
    """Compute the local color standard deviation used as salience."""
    channels = _image_to_float_channels(image)
    H, W, C = channels.shape
    window_size = _local_window_size(H, W)

    variance_sum = np.zeros((H, W), dtype=np.float64)
    for c in range(C):
        channel = channels[..., c]
        mean = uniform_filter(channel, size=window_size, mode="nearest")
        mean_sq = uniform_filter(channel * channel, size=window_size, mode="nearest")
        variance_sum += np.maximum(mean_sq - mean * mean, 0.0)

    return np.sqrt(variance_sum / float(C))


def _mesh_fractional_index(mesh: MeshGrid, lam: float, phi: float) -> tuple[float, float]:
    """Map a view-sphere direction to fractional mesh indices ``(i, j)``."""
    H, W = mesh.lambda_grid.shape

    lam_min = float(mesh.lambda_grid[0, 0])
    lam_max = float(mesh.lambda_grid[0, -1])
    phi_min = float(mesh.phi_grid[0, 0])
    phi_max = float(mesh.phi_grid[-1, 0])
    delta_lam = (lam_max - lam_min) / (W - 1)
    delta_phi = (phi_max - phi_min) / (H - 1)

    j = (float(lam) - lam_min) / delta_lam
    i = (float(phi) - phi_min) / delta_phi
    return i, j


def _line_endpoint_weights(mesh: MeshGrid, lines: list[LineAnnotation] | None) -> Array:
    """Compute Carroll et al.'s Gaussian weights around line endpoints."""
    H, W = mesh.lambda_grid.shape
    weights = np.zeros((H, W), dtype=np.float64)
    if not lines:
        return weights

    sigma = max(float(W) / _LINE_SIGMA_SCALE, _EPS)
    rows = np.arange(H, dtype=np.float64)[:, None]
    cols = np.arange(W, dtype=np.float64)[None, :]

    for line in lines:
        if len(line.points_dir) < 2:
            continue
        for lam, phi in (line.points_dir[0], line.points_dir[-1]):
            i, j = _mesh_fractional_index(mesh, lam, phi)
            dist_sq = (rows - i) * (rows - i) + (cols - j) * (cols - j)
            weights += np.exp(-0.5 * dist_sq / (sigma * sigma))

    return weights


def _bilinear_sample_field(field: Array, x: Array, y: Array) -> Array:
    """Bilinearly sample a 2D image-space field at floating pixel positions."""
    H, W = field.shape
    x_c = np.clip(x, 0.0, W - 1.0)
    y_c = np.clip(y, 0.0, H - 1.0)

    x0 = np.floor(x_c).astype(np.int64)
    y0 = np.floor(y_c).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    fx = x_c - x0
    fy = y_c - y0

    return (
        field[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + field[y0, x1] * fx * (1.0 - fy)
        + field[y1, x0] * (1.0 - fx) * fy
        + field[y1, x1] * fx * fy
    )


def compute_weights(
    image: Array,
    camera: Camera,
    mesh: MeshGrid,
    lines: list[LineAnnotation] | None = None,
) -> Array:
    """Compute Carroll et al.'s per-vertex weights ``w_{i,j}``.

    The local image-space color standard deviation is sampled at every mesh
    vertex (via the camera) and normalized to ``[0, 1]`` as ``w_S``. Line
    endpoints contribute Gaussian weights ``w_L`` with standard deviation
    equal to the mesh width divided by 100. Following MaDCoW, Carroll's
    face-detection term is omitted, giving ``w = 1 + 2 w_L + 2 w_S``.

    Args:
        image: Input image of shape ``(H_img, W_img, 3)`` or
            ``(H_img, W_img)``.
        camera: Camera used to map mesh vertices back to input pixels.
        mesh: The view-sphere mesh.
        lines: Optional straight-line annotations whose endpoints receive
            additional Gaussian weight.

    Returns:
        Array of shape ``(H_mesh, W_mesh)`` with non-negative weights.
    """
    local_stddev = _local_color_stddev(image)

    in_fov = camera.direction_in_fov(mesh.lambda_grid, mesh.phi_grid)
    x, y = camera.direction_to_pixel(mesh.lambda_grid, mesh.phi_grid)

    sampled = _bilinear_sample_field(local_stddev, x, y)
    sampled = np.where(in_fov, sampled, 0.0)

    max_salience = float(np.max(sampled)) if sampled.size else 0.0
    if max_salience <= _EPS:
        salience = np.zeros_like(sampled, dtype=np.float64)
    else:
        salience = np.clip(sampled / max_salience, 0.0, 1.0)

    line_weights = _line_endpoint_weights(mesh, lines)
    weights = (
        _BASELINE_WEIGHT
        + _LINE_ENDPOINT_WEIGHT * line_weights
        + _SALIENCE_WEIGHT * salience
    )
    return weights.astype(np.float64, copy=False)


if __name__ == "__main__":
    from . import CameraConfig, LineAnnotation
    from .mesh import build_mesh, compute_valid_mesh_mask

    # 1) Constant images have no local salience, but keep the baseline.
    cam = Camera(CameraConfig(fov_deg=90.0, width=100, height=80))
    mesh = build_mesh(n_lambda=41, n_phi=31, horizontal_span_deg=80.0)
    image = np.full((80, 100, 3), 128, dtype=np.uint8)
    weights = compute_weights(image, cam, mesh)
    print("constant weights shape:", weights.shape, "(expected (31, 41))")
    print(
        "constant weights range:",
        float(weights.min()),
        "to",
        float(weights.max()),
        "(expected 1.0 to 1.0)",
    )

    # 2) A high-contrast boundary receives larger weight than smooth sides.
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:, 50:] = 255
    weights = compute_weights(image, cam, mesh)
    centre_band = weights[:, 19:22].mean()
    side_band = np.concatenate((weights[:, :5].ravel(), weights[:, -5:].ravel())).mean()
    print("edge weights range:", float(weights.min()), "to", float(weights.max()))
    print("edge centre > side?:", bool(centre_band > side_band))

    # 3) Line endpoints receive additional Gaussian weight.
    line_points = tuple(
        (float(0.4 * t), 0.0)
        for t in np.linspace(0.0, 1.0, 128)
    )
    line = LineAnnotation(points_dir=line_points)
    weights = compute_weights(np.zeros((80, 100, 3), dtype=np.uint8), cam, mesh, [line])
    centre_weight = weights[31 // 2, 41 // 2]
    corner_weight = weights[0, 0]
    print("line endpoint centre > corner?:", bool(centre_weight > corner_weight))

    # 4) Grayscale inputs are supported and produce finite weights.
    gray = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (80, 1))
    weights = compute_weights(gray, cam, mesh)
    print("grayscale weights finite?:", bool(np.isfinite(weights).all()))

    # 5) The main pipeline zeros invalid-FOV vertices after computing weights.
    wide_mesh = build_mesh(n_lambda=61, n_phi=41, horizontal_span_deg=140.0)
    weights = compute_weights(image, cam, wide_mesh)
    valid = compute_valid_mesh_mask(cam, wide_mesh)
    weights[~valid] = 0.0
    print(
        "outside-FOV weights zero after valid mask?:",
        bool((~valid).any() and np.allclose(weights[~valid], 0.0)),
    )
