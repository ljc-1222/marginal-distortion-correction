"""Perspective input-camera model.

The :class:`Camera` bridges between input image pixels and view-sphere
directions ``(lambda, phi)`` in radians. It is the only module that touches
the input image coordinate system; everything downstream operates on view
sphere directions or mesh vertices.

Conventions:
    * Pixel coordinates ``(x, y)`` follow the standard image convention:
      origin at the top-left corner, ``x`` increases rightward, ``y`` downward.
      The principal point sits at the geometric image centre
      ``(cx, cy) = ((width - 1) / 2, (height - 1) / 2)``.
    * The camera frame uses OpenCV's convention: ``+X`` right, ``+Y`` down,
      ``+Z`` forward along the optical axis.
    * The horizontal FOV ``cfg.fov_deg`` determines the focal length in
      pixels via ``f = (width / 2) / tan(fov_rad / 2)``.
    * View-sphere directions ``(lambda, phi)`` are yaw / pitch in radians:
      ``lambda = atan2(X_cam, Z_cam)`` and
      ``phi = atan2(Y_cam, sqrt(X_cam^2 + Z_cam^2))``. With this convention
      ``lambda > 0`` looks rightwards and ``phi > 0`` looks downwards.
"""

from __future__ import annotations

import math

import numpy as np

from . import Array, CameraConfig


class Camera:
    """Pinhole perspective camera defined by a horizontal field of view.

    Attributes:
        cfg: The :class:`CameraConfig` describing FOV and image size.
        focal_length: Focal length in pixels, derived from ``cfg.fov_deg``
            and ``cfg.width``.
        cx: Principal point ``x`` coordinate in pixels.
        cy: Principal point ``y`` coordinate in pixels.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        """Initialize the camera from a :class:`CameraConfig`.

        Args:
            cfg: Intrinsic configuration of the input image.
        """
        if cfg.fov_deg <= 0 or cfg.fov_deg >= 180:
            raise ValueError(f"fov_deg must lie in (0, 180); got {cfg.fov_deg}.")
        if cfg.width <= 0 or cfg.height <= 0:
            raise ValueError(
                f"width and height must be positive; got ({cfg.width}, {cfg.height})."
            )

        self.cfg = cfg
        fov_rad = math.radians(cfg.fov_deg)
        self.focal_length: float = (cfg.width / 2.0) / math.tan(fov_rad / 2.0)
        self.cx: float = (cfg.width - 1) / 2.0
        self.cy: float = (cfg.height - 1) / 2.0

    def pixel_to_direction(self, x: Array, y: Array) -> tuple[Array, Array]:
        """Convert input image pixel coordinates to view-sphere directions.

        Args:
            x: Pixel x coordinates, any shape.
            y: Pixel y coordinates, same shape as ``x``.

        Returns:
            A pair ``(lambda, phi)`` of arrays with the same shape as the
            inputs, representing yaw and pitch in radians.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # Camera-frame ray direction (unnormalized): (X, Y, f).
        X = x_arr - self.cx
        Y = y_arr - self.cy
        f = self.focal_length

        lam = np.arctan2(X, f)
        phi = np.arctan2(Y, np.sqrt(X * X + f * f))
        return lam, phi

    def direction_to_pixel(self, lam: Array, phi: Array) -> tuple[Array, Array]:
        """Project view-sphere directions back to input pixel coordinates.

        Args:
            lam: Yaw angles in radians, any shape.
            phi: Pitch angles in radians, same shape as ``lam``.

        Returns:
            A pair ``(x, y)`` of arrays with the same shape as the inputs.
            Coordinates outside the input image are still returned but may
            be invalid; combine with :meth:`direction_in_fov` to filter.
        """
        lam_arr = np.asarray(lam, dtype=np.float64)
        phi_arr = np.asarray(phi, dtype=np.float64)

        # Unit direction in camera frame.
        cos_phi = np.cos(phi_arr)
        X = np.sin(lam_arr) * cos_phi
        Y = np.sin(phi_arr)
        Z = np.cos(lam_arr) * cos_phi

        # Use a safe denominator so directions behind the camera (Z <= 0)
        # do not raise warnings; the caller is expected to filter them with
        # :meth:`direction_in_fov`.
        Z_safe = np.where(Z > 0, Z, 1.0)
        x = self.focal_length * X / Z_safe + self.cx
        y = self.focal_length * Y / Z_safe + self.cy
        return x, y

    def direction_in_fov(self, lam: Array, phi: Array) -> Array:
        """Check whether view-sphere directions fall inside the camera FOV.

        Args:
            lam: Yaw angles in radians, any shape.
            phi: Pitch angles in radians, same shape as ``lam``.

        Returns:
            A boolean array with the same shape as the inputs.
        """
        lam_arr = np.asarray(lam, dtype=np.float64)
        phi_arr = np.asarray(phi, dtype=np.float64)

        cos_phi = np.cos(phi_arr)
        X = np.sin(lam_arr) * cos_phi
        Y = np.sin(phi_arr)
        Z = np.cos(lam_arr) * cos_phi
        in_front = Z > 0

        Z_safe = np.where(in_front, Z, 1.0)
        x = self.focal_length * X / Z_safe + self.cx
        y = self.focal_length * Y / Z_safe + self.cy

        # A small tolerance makes the bounds check robust to floating-point
        # round-trip error: any direction obtained from a valid input pixel
        # should still be reported as inside the FOV.
        eps = 1e-6
        in_bounds = (
            in_front
            & (x >= -eps)
            & (x <= self.cfg.width - 1 + eps)
            & (y >= -eps)
            & (y <= self.cfg.height - 1 + eps)
        )
        return in_bounds

if __name__ == "__main__":
    cfg = CameraConfig(fov_deg=90, width=1920, height=1080)
    camera = Camera(cfg)

    # Build a full-image pixel grid; meshgrid is required so x and y share
    # the same 2D shape before passing into pixel_to_direction.
    xs = np.arange(cfg.width)
    ys = np.arange(cfg.height)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")

    lam, phi = camera.pixel_to_direction(x_grid, y_grid)
    print("lam, phi shapes:", lam.shape, phi.shape)

    x_back, y_back = camera.direction_to_pixel(lam, phi)
    print(
        "round-trip max err:",
        float(np.max(np.abs(x_back - x_grid))),
        float(np.max(np.abs(y_back - y_grid))),
    )

    in_fov = camera.direction_in_fov(lam, phi)
    print("in_fov shape, all inside?:", in_fov.shape, bool(in_fov.all()))