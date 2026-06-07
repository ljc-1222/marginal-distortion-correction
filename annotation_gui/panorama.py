"""Panorama view transforms and original-resolution rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .image import ImageView, relative_path, resize_rgb_preview, unique_output_path


MAX_FOV_DEG = 179.0
MIN_VIEW_SIZE = 2


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def equirect_pixel_to_angles(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert equirectangular pixels to yaw and pitch angles."""
    if width < 2 or height < 2:
        raise ValueError(f"Panorama view requires width and height of at least 2; got {width}x{height}.")
    lam = (x / float(width - 1)) * (2.0 * math.pi) - math.pi
    phi = (y / float(height - 1)) * math.pi - (math.pi / 2.0)
    return lam, phi


def angles_to_equirect_pixel(
    lam: np.ndarray,
    phi: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project yaw and pitch angles to equirectangular pixels."""
    x = ((lam + math.pi) / (2.0 * math.pi)) * float(width - 1)
    y = ((phi + (math.pi / 2.0)) / math.pi) * float(height - 1)
    return x, y


def angles_to_vectors(lam: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert yaw and pitch to unit-sphere vectors."""
    cos_phi = np.cos(phi)
    x = np.sin(lam) * cos_phi
    y = np.sin(phi)
    z = np.cos(lam) * cos_phi
    return x, y, z


def vectors_to_angles(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert unit-sphere vectors to yaw and pitch."""
    lam = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt((x * x) + (z * z)))
    return lam, phi


def local_to_world_angles(
    local_lam: np.ndarray,
    local_phi: np.ndarray,
    center_lam: float,
    center_phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map local view angles to world panorama angles."""
    x, y, z = angles_to_vectors(local_lam, local_phi)

    cp = math.cos(-center_phi)
    sp = math.sin(-center_phi)
    y_pitch = (y * cp) - (z * sp)
    z_pitch = (y * sp) + (z * cp)

    cy = math.cos(center_lam)
    sy = math.sin(center_lam)
    x_world = (x * cy) + (z_pitch * sy)
    z_world = (-x * sy) + (z_pitch * cy)
    return vectors_to_angles(x_world, y_pitch, z_world)


def world_to_local_angles(
    world_lam: np.ndarray,
    world_phi: np.ndarray,
    center_lam: float,
    center_phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map world panorama angles to local view angles."""
    x, y, z = angles_to_vectors(world_lam, world_phi)

    cy = math.cos(-center_lam)
    sy = math.sin(-center_lam)
    x_yaw = (x * cy) + (z * sy)
    z_yaw = (-x * sy) + (z * cy)

    cp = math.cos(center_phi)
    sp = math.sin(center_phi)
    y_local = (y * cp) - (z_yaw * sp)
    z_local = (y * sp) + (z_yaw * cp)
    return vectors_to_angles(x_yaw, y_local, z_local)


def render_centered_equirectangular(
    source_image: np.ndarray,
    center_lam: float,
    center_phi: float,
    out_width: int | None = None,
    out_height: int | None = None,
) -> np.ndarray:
    """Render a local equirectangular panorama view for setup preview."""
    import cv2

    arr = np.asarray(source_image)
    src_height, src_width = arr.shape[:2]
    dst_width = src_width if out_width is None else int(out_width)
    dst_height = src_height if out_height is None else int(out_height)
    yy, xx = np.indices((dst_height, dst_width), dtype=np.float64)
    local_lam, local_phi = equirect_pixel_to_angles(xx, yy, dst_width, dst_height)
    world_lam, world_phi = local_to_world_angles(local_lam, local_phi, center_lam, center_phi)
    source_x, source_y = angles_to_equirect_pixel(world_lam, world_phi, src_width, src_height)
    return cv2.remap(
        arr,
        source_x.astype(np.float32),
        source_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def _normalized_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    left = float(np.clip(min(x0, x1), 0.0, float(width)))
    right = float(np.clip(max(x0, x1), 0.0, float(width)))
    top = float(np.clip(min(y0, y1), 0.0, float(height)))
    bottom = float(np.clip(max(y0, y1), 0.0, float(height)))
    if right - left < MIN_VIEW_SIZE:
        right = min(float(width), left + MIN_VIEW_SIZE)
        left = max(0.0, right - MIN_VIEW_SIZE)
    if bottom - top < MIN_VIEW_SIZE:
        bottom = min(float(height), top + MIN_VIEW_SIZE)
        top = max(0.0, bottom - MIN_VIEW_SIZE)
    return left, top, right, bottom


def _horizontal_fov_from_crop(crop_original_px: tuple[float, float, float, float], source_width: int) -> float:
    x0, _y0, x1, _y1 = crop_original_px
    width = max(abs(float(x1) - float(x0)), 1.0)
    fov = width / max(float(source_width - 1), 1.0) * 360.0
    return float(np.clip(fov, 1.0, MAX_FOV_DEG))


def _vertical_fov_from_aspect(horizontal_fov_deg: float, view_width: int, view_height: int) -> float:
    h_rad = math.radians(horizontal_fov_deg)
    v_rad = 2.0 * math.atan(math.tan(h_rad / 2.0) * (float(view_height) / float(view_width)))
    return math.degrees(v_rad)


def _vertical_fov_from_crop(crop_original_px: tuple[float, float, float, float], source_height: int) -> float:
    _x0, y0, _x1, y1 = crop_original_px
    height = max(abs(float(y1) - float(y0)), 1.0)
    fov = height / max(float(source_height - 1), 1.0) * 180.0
    return float(np.clip(fov, 1.0, MAX_FOV_DEG))


@dataclass
class PanoramaViewResult:
    """Rendered derived annotation image and its v2 JSON metadata."""

    image_view: ImageView
    source_image_path: Path
    source_size: tuple[int, int]
    crop_original_px: tuple[float, float, float, float]
    crop_preview_px: tuple[float, float, float, float]
    center_yaw_rad: float
    center_pitch_rad: float
    horizontal_fov_deg: float
    vertical_fov_deg: float

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "type": "panorama_view",
            "source_camera_model": "360",
            "projection": "equirectangular_crop",
            "source_size": [int(self.source_size[0]), int(self.source_size[1])],
            "view_size": [int(self.image_view.width), int(self.image_view.height)],
            "preview_size": [int(self.image_view.preview_width), int(self.image_view.preview_height)],
            "center_yaw_rad": float(self.center_yaw_rad),
            "center_pitch_rad": float(self.center_pitch_rad),
            "crop_original_px": [float(v) for v in self.crop_original_px],
            "crop_preview_px": [float(v) for v in self.crop_preview_px],
            "horizontal_fov_deg": float(self.horizontal_fov_deg),
            "vertical_fov_deg": float(self.vertical_fov_deg),
        }


def build_panorama_view(
    source_view: ImageView,
    output_dir: str | Path,
    center_yaw_rad: float,
    center_pitch_rad: float,
    crop_preview_px: tuple[float, float, float, float],
    preview_max_side: int,
    output_suffix: str = "_view",
) -> PanoramaViewResult:
    """Render and save an original-resolution centered equirectangular crop."""
    preview_box = _normalized_box(crop_preview_px, source_view.preview_width, source_view.preview_height)
    crop_original = source_view.preview_box_to_image(preview_box)
    crop_original = _normalized_box(crop_original, source_view.width, source_view.height)
    x0, y0, x1, y1 = crop_original
    ix0 = max(0, min(int(np.ceil(x0)), source_view.width - MIN_VIEW_SIZE))
    iy0 = max(0, min(int(np.ceil(y0)), source_view.height - MIN_VIEW_SIZE))
    ix1 = max(ix0 + MIN_VIEW_SIZE, min(int(np.ceil(x1)), source_view.width))
    iy1 = max(iy0 + MIN_VIEW_SIZE, min(int(np.ceil(y1)), source_view.height))
    crop_original = (float(ix0), float(iy0), float(ix1), float(iy1))
    view_width = ix1 - ix0
    view_height = iy1 - iy0
    horizontal_fov_deg = _horizontal_fov_from_crop(crop_original, source_view.width)
    vertical_fov_deg = _vertical_fov_from_crop(crop_original, source_view.height)
    centered = render_centered_equirectangular(
        source_view.image,
        center_yaw_rad,
        center_pitch_rad,
        source_view.width,
        source_view.height,
    )
    rendered = centered[iy0:iy1, ix0:ix1].copy()
    output_path = unique_output_path(Path(output_dir), source_view.path.stem, output_suffix, ".png")
    Image.fromarray(rendered).save(output_path)
    image_view = ImageView(
        path=output_path,
        image=rendered,
        preview=resize_rgb_preview(rendered, preview_max_side),
    )
    return PanoramaViewResult(
        image_view=image_view,
        source_image_path=source_view.path,
        source_size=(source_view.width, source_view.height),
        crop_original_px=crop_original,
        crop_preview_px=preview_box,
        center_yaw_rad=wrap_angle(center_yaw_rad),
        center_pitch_rad=float(center_pitch_rad),
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
    )


def panorama_view_json_paths(result: PanoramaViewResult, json_dir: Path) -> dict[str, str]:
    """Return relative JSON path fields for a panorama view result."""
    return {
        "image_path": relative_path(result.image_view.path, json_dir),
        "source_image_path": relative_path(result.source_image_path, json_dir),
    }
