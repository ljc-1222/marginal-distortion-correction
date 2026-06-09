"""Input-camera models.

The :class:`Camera` bridges between input image pixels and view-sphere
directions ``(lambda, phi)`` in radians. It is the only module that touches
the input image coordinate system; everything downstream operates on view
sphere directions or mesh vertices.

Conventions:
    * Pixel coordinates ``(x, y)`` follow the standard image convention:
      origin at the top-left corner, ``x`` increases rightward, ``y`` downward.
      The principal point sits at the geometric image centre
      ``(cx, cy) = ((width - 1) / 2, (height - 1) / 2)``.
    * Pinhole mode uses OpenCV's camera-frame convention: ``+X`` right,
      ``+Y`` down, ``+Z`` forward along the optical axis.
    * In pinhole mode, the horizontal FOV ``cfg.fov_deg`` determines the
      focal length in pixels via ``f = (width / 2) / tan(fov_rad / 2)``.
    * Panorama-view mode maps cropped equirectangular view pixels to local
      equirectangular view angles around the crop center. The panorama center
      is used when the annotation view is rendered, so the MaDCoW camera must
      not apply that center rotation again.
    * View-sphere directions ``(lambda, phi)`` are yaw / pitch in radians:
      ``lambda = atan2(X_cam, Z_cam)`` and
      ``phi = atan2(Y_cam, sqrt(X_cam^2 + Z_cam^2))``. With this convention
      ``lambda > 0`` looks rightwards and ``phi > 0`` looks downwards.
"""

from __future__ import annotations

import math

import numpy as np

from . import Array, CameraConfig


_SUPPORTED_MODELS = {"pinhole", "panorama_view"}


def _normalize_model(model: str) -> str:
    """Validate and return the camera model name."""
    if model in _SUPPORTED_MODELS:
        return model
    raise ValueError(f"camera model must be one of {sorted(_SUPPORTED_MODELS)}; got {model!r}.")


class _PinholeCamera:
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
        if cfg.fov_deg is None:
            raise ValueError("fov_deg is required for pinhole camera model.")
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


class _PanoramaViewCamera:
    """Cropped centered equirectangular panorama view from v2 annotations.

    The derived image has already been rendered around ``center_yaw_rad`` and
    ``center_pitch_rad``. Downstream MaDCoW optimization should therefore see
    the crop's own local view sphere, with the crop center as the optical axis,
    not the source panorama's world sphere.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        """Initialize a panorama-derived crop camera from annotation metadata."""
        if cfg.view is None:
            raise ValueError("panorama_view camera requires annotation view metadata.")
        view = cfg.view
        if str(view.get("projection")) != "equirectangular_crop":
            raise ValueError(
                "panorama_view camera requires view.projection == 'equirectangular_crop'; "
                f"got {view.get('projection')!r}."
            )
        source_size = view.get("source_size")
        crop = view.get("crop_original_px")
        if not isinstance(source_size, list) or len(source_size) != 2:
            raise ValueError("view.source_size must be [width, height].")
        if not isinstance(crop, list) or len(crop) != 4:
            raise ValueError("view.crop_original_px must be [x0, y0, x1, y1].")

        self.cfg = cfg
        self.source_width = int(source_size[0])
        self.source_height = int(source_size[1])
        if self.source_width < 2 or self.source_height < 2:
            raise ValueError(
                "panorama_view source_size must be at least 2x2; "
                f"got {self.source_width}x{self.source_height}."
            )
        self.crop_x0 = float(crop[0])
        self.crop_y0 = float(crop[1])
        self.crop_x1 = float(crop[2])
        self.crop_y1 = float(crop[3])
        self.crop_cx = 0.5 * (self.crop_x0 + self.crop_x1 - 1.0)
        self.crop_cy = 0.5 * (self.crop_y0 + self.crop_y1 - 1.0)
        self.center_lam = float(view.get("center_yaw_rad", 0.0))
        self.center_phi = float(view.get("center_pitch_rad", 0.0))
        self.cx: float = (cfg.width - 1) / 2.0
        self.cy: float = (cfg.height - 1) / 2.0
        self.focal_length: None = None
        self._lam_per_px = (2.0 * math.pi) / float(self.source_width - 1)
        self._phi_per_px = math.pi / float(self.source_height - 1)

    def pixel_to_direction(self, x: Array, y: Array) -> tuple[Array, Array]:
        """Convert view-crop pixels to local view yaw/pitch directions."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        setup_x = x_arr + self.crop_x0
        setup_y = y_arr + self.crop_y0
        lam = (setup_x - self.crop_cx) * self._lam_per_px
        phi = (setup_y - self.crop_cy) * self._phi_per_px
        return lam, phi

    def direction_to_pixel(self, lam: Array, phi: Array) -> tuple[Array, Array]:
        """Project local view yaw/pitch directions to view-crop pixels."""
        setup_x = (np.asarray(lam, dtype=np.float64) / self._lam_per_px) + self.crop_cx
        setup_y = (np.asarray(phi, dtype=np.float64) / self._phi_per_px) + self.crop_cy
        return setup_x - self.crop_x0, setup_y - self.crop_y0

    def direction_in_fov(self, lam: Array, phi: Array) -> Array:
        """Check whether directions fall inside the cropped view."""
        x, y = self.direction_to_pixel(lam, phi)
        eps = 1e-6
        return (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= -eps)
            & (x <= self.cfg.width - 1 + eps)
            & (y >= -eps)
            & (y <= self.cfg.height - 1 + eps)
        )


class Camera:
    """Input camera wrapper preserving the downstream MaDCoW camera API."""

    def __init__(self, cfg: CameraConfig) -> None:
        """Initialize the selected input camera model."""
        if cfg.width <= 0 or cfg.height <= 0:
            raise ValueError(
                f"width and height must be positive; got ({cfg.width}, {cfg.height})."
            )

        self.cfg = cfg
        self.model = _normalize_model(cfg.model)
        if self.model == "pinhole":
            self._camera = _PinholeCamera(cfg)
        else:
            self._camera = _PanoramaViewCamera(cfg)

        self.focal_length = self._camera.focal_length
        self.cx = self._camera.cx
        self.cy = self._camera.cy

    def pixel_to_direction(self, x: Array, y: Array) -> tuple[Array, Array]:
        """Convert input image pixel coordinates to view-sphere directions."""
        return self._camera.pixel_to_direction(x, y)

    def direction_to_pixel(self, lam: Array, phi: Array) -> tuple[Array, Array]:
        """Project view-sphere directions back to input pixel coordinates."""
        return self._camera.direction_to_pixel(lam, phi)

    def direction_in_fov(self, lam: Array, phi: Array) -> Array:
        """Check whether view-sphere directions fall inside the input image domain."""
        return self._camera.direction_in_fov(lam, phi)


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

    full_panorama_view = {
        "projection": "equirectangular_crop",
        "source_size": [400, 200],
        "crop_original_px": [0.0, 0.0, 400.0, 200.0],
        "center_yaw_rad": 0.0,
        "center_pitch_rad": 0.0,
    }
    pano_cfg = CameraConfig(
        fov_deg=None,
        width=400,
        height=200,
        model="panorama_view",
        view=full_panorama_view,
    )
    pano_camera = Camera(pano_cfg)
    xs = np.arange(pano_cfg.width)
    ys = np.arange(pano_cfg.height)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    lam, phi = pano_camera.pixel_to_direction(x_grid, y_grid)
    x_back, y_back = pano_camera.direction_to_pixel(lam, phi)
    print(
        "full panorama-view round-trip max err:",
        float(np.max(np.abs(x_back - x_grid))),
        float(np.max(np.abs(y_back - y_grid))),
    )
    print("full panorama-view all inside?:", bool(pano_camera.direction_in_fov(lam, phi).all()))

    shifted_panorama_view = {
        "projection": "equirectangular_crop",
        "source_size": [400, 200],
        "crop_original_px": [40.0, 20.0, 340.0, 160.0],
        "center_yaw_rad": 0.9,
        "center_pitch_rad": -0.35,
    }
    shifted_cfg = CameraConfig(
        fov_deg=None,
        width=300,
        height=140,
        model="panorama_view",
        view=shifted_panorama_view,
    )
    shifted_camera = Camera(shifted_cfg)
    xs = np.arange(shifted_cfg.width)
    ys = np.arange(shifted_cfg.height)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    lam_shifted, phi_shifted = shifted_camera.pixel_to_direction(x_grid, y_grid)
    expected_lam = (x_grid - ((shifted_cfg.width - 1) / 2.0)) * ((2.0 * math.pi) / 399.0)
    expected_phi = (y_grid - ((shifted_cfg.height - 1) / 2.0)) * (math.pi / 199.0)
    x_back, y_back = shifted_camera.direction_to_pixel(lam_shifted, phi_shifted)
    print(
        "shifted panorama-view uses crop-centered local angles?:",
        bool(np.allclose(lam_shifted, expected_lam) and np.allclose(phi_shifted, expected_phi)),
    )
    print(
        "shifted panorama-view round-trip max err:",
        float(np.max(np.abs(x_back - x_grid))),
        float(np.max(np.abs(y_back - y_grid))),
    )
