"""Shared annotation image session state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .image import ImageView, load_image_view
from .panorama import PanoramaViewResult, build_panorama_view


@dataclass
class AnnotationSession:
    """Original image plus the active original-resolution annotation view."""

    source_view: ImageView
    active_view: ImageView
    camera_model: str
    fov_deg: float | None
    panorama_result: PanoramaViewResult | None = None

    @classmethod
    def from_image(cls, image_path: str | Path, preview_max_side: int, fov_deg: float | None) -> "AnnotationSession":
        """Create a pinhole session from an input image."""
        source_view = load_image_view(image_path, preview_max_side)
        return cls(
            source_view=source_view,
            active_view=source_view,
            camera_model="pinhole",
            fov_deg=fov_deg,
        )

    @property
    def image_path(self) -> str:
        return str(self.active_view.path)

    @property
    def source_image_path(self) -> str:
        return str(self.source_view.path)

    @property
    def image(self) -> np.ndarray:
        return self.active_view.image

    @property
    def preview(self) -> np.ndarray:
        return self.active_view.preview

    @property
    def width(self) -> int:
        return self.active_view.width

    @property
    def height(self) -> int:
        return self.active_view.height

    @property
    def preview_width(self) -> int:
        return self.active_view.preview_width

    @property
    def preview_height(self) -> int:
        return self.active_view.preview_height

    @property
    def view_metadata(self) -> dict[str, object] | None:
        return None if self.panorama_result is None else self.panorama_result.metadata

    def use_pinhole_source(self, fov_deg: float) -> None:
        """Use the original input image as a pinhole annotation view."""
        self.active_view = self.source_view
        self.camera_model = "pinhole"
        self.fov_deg = float(fov_deg)
        self.panorama_result = None

    def use_panorama_view(
        self,
        output_dir: str | Path,
        center_yaw_rad: float,
        center_pitch_rad: float,
        crop_preview_px: tuple[float, float, float, float],
        preview_max_side: int,
    ) -> None:
        """Render and activate an original-resolution panorama-derived view."""
        result = build_panorama_view(
            source_view=self.source_view,
            output_dir=output_dir,
            center_yaw_rad=center_yaw_rad,
            center_pitch_rad=center_pitch_rad,
            crop_preview_px=crop_preview_px,
            preview_max_side=preview_max_side,
        )
        self.active_view = result.image_view
        self.camera_model = "panorama_view"
        self.fov_deg = result.horizontal_fov_deg
        self.panorama_result = result
