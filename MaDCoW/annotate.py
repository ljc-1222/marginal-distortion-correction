"""Matplotlib-based annotation GUI.

Provides a click-and-drag interface for authoring the straight-line and ROI
annotations consumed by :mod:`main`. The GUI saves a JSON file alongside one
PNG mask per non-empty ROI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image

from annotation_gui import AnnotationSession, load_image_view
from annotation_gui.base import (
    EmbeddedViewSetupController,
    add_button_row,
    clear_widget_axes,
    create_image_figure,
    set_image_artist,
)
from annotation_gui.io import build_annotation_payload, write_annotation_json

import piexif
from .src import CameraConfig
from .src.camera import Camera


DEFAULT_IMAGE = Path(__file__).resolve().parent / "data" / "test.jpg"
FALLBACK_FOV_DEG = 90.0
PREVIEW_MAX_SIDE = 1200
LINE_RESAMPLE_POINTS = 128
LINE_MIN_SPACING_PX = 3.0
CAMERA_MODEL_CHOICES = ("pinhole", "360", "panorama_view")
CAMERA_MODEL_LABELS = {
    "pinhole": "Pinhole",
    "360": "360",
    "panorama_view": "Panorama View",
}
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "madcow_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)


def _ratio_to_float(value: Any) -> float | None:
    """Convert EXIF rational-like values to floats."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, tuple) and len(value) == 2 and value[1] != 0:
        return float(value[0]) / float(value[1])
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator:
        return float(numerator) / float(denominator)
    return None


def estimate_fov_from_exif(image_path: str, fallback: float = FALLBACK_FOV_DEG) -> tuple[float, str]:
    """Estimate horizontal FOV from EXIF metadata when available.

    Args:
        image_path: Input image path.
        fallback: FOV used when EXIF data are missing or insufficient.

    Returns:
        Tuple ``(fov_deg, source)`` where ``source`` describes the estimate.
    """
    image_path_obj = Path(image_path).expanduser().resolve()

    try:
        with Image.open(image_path_obj) as img:
            width = int(img.width)
    except OSError:
        return float(fallback), "fallback"

    try:
        exif_dict = piexif.load(str(image_path_obj))
        exif_ifd = exif_dict.get("Exif", {}) or {}

        f35 = _ratio_to_float(exif_ifd.get(piexif.ExifIFD.FocalLengthIn35mmFilm))
        if f35 and f35 > 0:
            fov = math.degrees(2.0 * math.atan(36.0 / (2.0 * f35)))
            return fov, "EXIF 35mm-equivalent focal length via piexif"

        focal = _ratio_to_float(exif_ifd.get(piexif.ExifIFD.FocalLength))
        x_res = _ratio_to_float(exif_ifd.get(piexif.ExifIFD.FocalPlaneXResolution))
        unit_raw = exif_ifd.get(piexif.ExifIFD.FocalPlaneResolutionUnit)

        try:
            unit = int(unit_raw) if unit_raw is not None else None
        except (TypeError, ValueError):
            unit = None

        if focal and focal > 0 and x_res and x_res > 0 and unit in (2, 3, 4, 5):
            unit_to_mm = {
                2: 25.4,   # inch
                3: 10.0,   # centimeter
                4: 1.0,    # millimeter
                5: 0.001,  # micrometer
            }
            sensor_width_mm = width / x_res * unit_to_mm[unit]
            if sensor_width_mm > 0:
                fov = math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal)))
                return fov, "EXIF focal length and focal-plane resolution via piexif"
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        pass

    try:
        with Image.open(image_path_obj) as img:
            exif = img.getexif()
            lookup = {ExifTags.TAGS.get(key, key): value for key, value in exif.items()}
    except OSError:
        return float(fallback), "fallback"

    f35 = _ratio_to_float(lookup.get("FocalLengthIn35mmFilm"))
    if f35 and f35 > 0:
        fov = math.degrees(2.0 * math.atan(36.0 / (2.0 * f35)))
        return fov, "EXIF 35mm-equivalent focal length via PIL fallback"

    focal = _ratio_to_float(lookup.get("FocalLength"))
    x_res = _ratio_to_float(lookup.get("FocalPlaneXResolution"))
    unit = lookup.get("FocalPlaneResolutionUnit")
    if focal and focal > 0 and x_res and x_res > 0 and unit in (2, 3, 4, 5):
        unit_to_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}
        sensor_width_mm = width / x_res * unit_to_mm[int(unit)]
        if sensor_width_mm > 0:
            fov = math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal)))
            return fov, "EXIF focal length and focal-plane resolution via PIL fallback"

    return float(fallback), "fallback"


def _sanitize_name(name: str, fallback: str) -> str:
    """Create a filesystem-safe ROI name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return cleaned or fallback


def _relative_path(path: Path, base_dir: Path) -> str:
    """Portable relative path helper."""
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), base_dir.resolve())


def _normalize_camera_model(model: str) -> str:
    """Validate and return the annotation camera model name."""
    if model in CAMERA_MODEL_CHOICES:
        return model
    choices = ", ".join(CAMERA_MODEL_CHOICES)
    raise ValueError(f"camera_model must be one of: {choices}; got {model!r}.")


def _lanczos_resampling() -> int:
    """Return the Pillow LANCZOS resampling enum across Pillow versions."""
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.LANCZOS)
    return int(Image.LANCZOS)


def _nearest_resampling() -> int:
    """Return the Pillow NEAREST resampling enum across Pillow versions."""
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.NEAREST)
    return int(Image.NEAREST)


def _compute_preview_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Compute a preview size that preserves aspect ratio.

    Args:
        width: Original image width.
        height: Original image height.
        max_side: Maximum allowed preview side length. Values <= 0 disable downsampling.

    Returns:
        The preview ``(width, height)``.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}.")
    if max_side <= 0:
        return width, height

    longest = max(width, height)
    if longest <= max_side:
        return width, height

    scale = float(max_side) / float(longest)
    preview_width = max(1, int(round(width * scale)))
    preview_height = max(1, int(round(height * scale)))
    return preview_width, preview_height


def _polyline_length(points: list[tuple[float, float]]) -> float:
    """Return total arc length of a 2D polyline in preview pixels."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _resample_polyline(
    points: list[tuple[float, float]],
    n_samples: int = LINE_RESAMPLE_POINTS,
) -> list[tuple[float, float]]:
    """Resample a 2D polyline to a fixed number of equally spaced points."""
    if len(points) < 2:
        raise ValueError("At least two points are required to resample a polyline.")
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2; got {n_samples}.")

    lengths = [
        math.hypot(x1 - x0, y1 - y0)
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    ]
    total_length = sum(lengths)
    if total_length <= 1e-9:
        return [points[0]] * n_samples

    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)

    targets = np.linspace(0.0, total_length, n_samples)
    resampled: list[tuple[float, float]] = []
    segment_idx = 0
    for target in targets:
        while segment_idx < len(lengths) - 1 and target > cumulative[segment_idx + 1]:
            segment_idx += 1

        seg_length = lengths[segment_idx]
        x0, y0 = points[segment_idx]
        x1, y1 = points[segment_idx + 1]
        if seg_length <= 1e-12:
            resampled.append((float(x0), float(y0)))
            continue

        alpha = (float(target) - cumulative[segment_idx]) / seg_length
        x = (1.0 - alpha) * x0 + alpha * x1
        y = (1.0 - alpha) * y0 + alpha * y1
        resampled.append((float(x), float(y)))

    resampled[0] = points[0]
    resampled[-1] = points[-1]
    return resampled


class AnnotationGUI:
    """Interactive annotation tool for a single input image.

    Controls:
        * ROI phase left drag: paint current ROI mask.
        * ROI phase ``Redraw``: clear the current ROI draft.
        * ROI phase ``Next ROI``: accept the current ROI and start the next draft.
        * ROI phase ``Done ROI``: lock ROI masks and switch to lines.
        * Lines phase left drag: trace a straight-structure curve.
        * Lines phase ``Redraw``: clear the pending line draft.
        * Lines phase ``Next``: accept the pending line.
        * ``+`` / ``-``: change brush radius.
        * ``s``: save annotations.
        * ``d``: finish ROI setup and switch to line drawing.
        * ``r`` / ``escape``: redraw the current draft.
    """

    def __init__(
        self,
        image_path: str,
        fov_deg: float | None,
        output_dir: str,
        camera_model: str = "pinhole",
        source_image_path: str | None = None,
        view_metadata: dict[str, Any] | None = None,
        start_with_setup: bool = False,
    ) -> None:
        """Initialize the GUI state.

        Args:
            image_path: Path to the input image.
            fov_deg: Horizontal field of view of a pinhole input image, in degrees.
            output_dir: Directory where the JSON file and mask PNGs are saved.
            camera_model: Input camera model to store in the annotation JSON.
            preview_max_side: Maximum side length used for the interactive preview.
        """
        self.image_path = str(Path(image_path).resolve())
        self.camera_model = _normalize_camera_model(camera_model)
        self.fov_deg = None if fov_deg is None else float(fov_deg)
        if self.camera_model == "pinhole" and self.fov_deg is None:
            self.fov_deg = FALLBACK_FOV_DEG
        if self.fov_deg is not None and (self.fov_deg <= 0 or self.fov_deg >= 180):
            raise ValueError(f"fov_deg must lie in (0, 180); got {self.fov_deg}.")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_image_path = source_image_path
        self.view_metadata = view_metadata

        self.preview_max_side = PREVIEW_MAX_SIDE
        self.session = AnnotationSession.from_image(self.image_path, self.preview_max_side, self.fov_deg)
        self.image_view = load_image_view(self.image_path, self.preview_max_side)
        self.original_width = self.image_view.width
        self.original_height = self.image_view.height
        self.image = self.image_view.preview
        self.height, self.width = self.image.shape[:2]
        self.preview_width = self.width
        self.preview_height = self.height
        self.is_downsampled = self.image_view.is_downsampled
        self.camera: Camera | None = None
        if not start_with_setup:
            self._set_camera_model(self.camera_model)

        self.phase = "setup" if start_with_setup else "roi"
        self.mode = "region"
        self.status_extra = "Choose camera mode." if start_with_setup else "Draw the current ROI draft."
        self.brush_radius = max(4, int(round(min(self.height, self.width) * 0.02)))
        self.lines: list[list[tuple[float, float]]] = []
        self.pending_line: list[tuple[float, float]] | None = None
        self.current_line_points: list[tuple[float, float]] = []
        self._drawing_line = False
        self.regions: list[dict[str, Any]] = []
        self.current_region = 0
        self.next_region_idx = 1
        self._drawing = False
        self._stroke_value = True
        self._dynamic_artists: list[Any] = []
        self.setup_controller: EmbeddedViewSetupController | None = None

        self.json_path = self.output_dir / f"{Path(self.image_path).stem}.json"
        self._build_figure()
        if start_with_setup:
            self._start_view_setup()
        else:
            self._add_region()
            self._set_roi_controls()
            self._refresh()

    def _set_camera_model(self, camera_model: str) -> None:
        """Set the input camera model used for saving line directions."""
        model = _normalize_camera_model(camera_model)
        if model == "pinhole" and self.fov_deg is None:
            self.fov_deg = FALLBACK_FOV_DEG
        self.camera = Camera(
            CameraConfig(
                fov_deg=self.fov_deg,
                width=self.original_width,
                height=self.original_height,
                model=model,
                view=self.view_metadata,
            )
        )
        self.camera_model = self.camera.model

    def run(self) -> None:
        """Open the matplotlib window and block until the user exits."""
        import matplotlib.pyplot as plt

        plt.show()

    def _build_figure(self) -> None:
        """Create the Matplotlib figure, artists, widgets, and callbacks."""
        self.fig, self.ax, self.image_artist, self.status, self.help_text = create_image_figure(
            "MaDCoW Annotation Tool",
            self.image,
        )
        self.mask_artist = self.ax.imshow(np.zeros((self.height, self.width, 4)), interpolation="nearest")

        self.widgets: list[Any] = []

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_draw)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _start_view_setup(self) -> None:
        """Start embedded single-window camera/view setup."""
        self.setup_controller = EmbeddedViewSetupController(
            session=self.session,
            output_dir=self.output_dir,
            preview_max_side=self.preview_max_side,
            fig=self.fig,
            ax=self.ax,
            image_artist=self.image_artist,
            status=self.status,
            help_text=self.help_text,
            widgets=self.widgets,
            clear_dynamic_artists=self._clear_dynamic_artists,
            on_done=self._enter_annotation_session,
            fov_deg=self.fov_deg,
            fallback_fov_deg=FALLBACK_FOV_DEG,
        )
        self.setup_controller.start()

    def _enter_annotation_session(self, session: AnnotationSession) -> None:
        """Activate the selected original-resolution annotation view."""
        self.session = session
        self.image_path = session.image_path
        self.source_image_path = session.source_image_path if session.view_metadata is not None else self.source_image_path
        self.view_metadata = session.view_metadata
        self.camera_model = _normalize_camera_model(session.camera_model)
        self.fov_deg = session.fov_deg
        self.image_view = session.active_view
        self.original_width = self.image_view.width
        self.original_height = self.image_view.height
        self.image = self.image_view.preview
        self.height, self.width = self.image.shape[:2]
        self.preview_width = self.width
        self.preview_height = self.height
        self.is_downsampled = self.image_view.is_downsampled
        self.brush_radius = max(4, int(round(min(self.height, self.width) * 0.02)))
        self.json_path = self.output_dir / f"{Path(self.image_path).stem}.json"
        self._set_camera_model(self.camera_model)
        set_image_artist(self.image_artist, self.ax, self.image)
        self.mask_artist.set_data(np.zeros((self.height, self.width, 4), dtype=np.float32))
        self.mask_artist.set_extent((-0.5, self.width - 0.5, self.height - 0.5, -0.5))
        self.lines.clear()
        self.pending_line = None
        self.current_line_points = []
        self.regions.clear()
        self.next_region_idx = 1
        self._add_region()
        self.current_region = len(self.regions) - 1
        self.phase = "roi"
        self.mode = "region"
        self.status_extra = "Draw the current ROI draft."
        self._set_roi_controls()
        self._refresh()

    def _clear_controls(self) -> None:
        """Remove the phase-specific controls."""
        clear_widget_axes(self.widgets)

    def _clear_dynamic_artists(self) -> None:
        """Remove transient overlay artists."""
        for artist in self._dynamic_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._dynamic_artists.clear()

    def _add_centered_button_row(self, buttons: list[tuple[str, float, Any]], y0: float) -> None:
        """Add a centered row of buttons."""
        add_button_row(self.fig, self.widgets, buttons, y0=y0)

    def _set_roi_controls(self) -> None:
        """Show ROI setup controls."""
        self._clear_controls()
        self._add_centered_button_row(
            [
                ("Redraw", 0.09, self._button_redraw_region),
                ("Next ROI", 0.105, self._button_next_region),
                ("Brush -", 0.085, self._button_brush_down),
                ("Brush +", 0.085, self._button_brush_up),
                ("Done ROI", 0.105, self._button_done_regions),
            ],
            0.075,
        )

    def _set_line_controls(self) -> None:
        """Show line annotation controls after ROI setup is locked."""
        self._clear_controls()
        self._add_centered_button_row(
            [
                ("Redraw", 0.09, self._button_redraw_line),
                ("Next", 0.09, self._button_next_line),
                ("Save", 0.075, self._button_save),
                ("Save+Close", 0.120, self._button_save_close),
            ],
            0.075,
        )

    def _add_region(self, name: str | None = None) -> None:
        """Create a new empty ROI mask and select it."""
        if name is None:
            region_name = f"region_{self.next_region_idx}"
            self.next_region_idx += 1
        else:
            region_name = name
        self.regions.append(
            {
                "name": region_name,
                "mask": np.zeros((self.original_height, self.original_width), dtype=bool),
            }
        )
        self.current_region = len(self.regions) - 1

    def _current_region_data(self) -> dict[str, Any]:
        """Return the selected ROI data dictionary."""
        return self.regions[self.current_region]

    def _set_mode(self, mode: str) -> None:
        """Switch interaction mode."""
        if mode == "region" and self.phase != "roi":
            self.status_extra = "ROI setup is locked."
            self._refresh()
            return
        if mode == "line" and self.phase != "line":
            self.status_extra = "Finish ROI setup before drawing lines."
            self._refresh()
            return
        self.mode = mode
        if mode != "line":
            self._drawing_line = False
            self.current_line_points = []
        self._refresh()

    def _button_done_regions(self, _event: object) -> None:
        """Lock ROI masks and switch to line annotation."""
        if self.phase != "roi":
            return
        self._drawing = False
        self.phase = "line"
        self.mode = "line"
        self._drawing_line = False
        self.current_line_points = []
        self.pending_line = None
        self.status_extra = "ROI setup locked. Draw a line draft."
        self._set_line_controls()
        self._refresh()

    def _button_redraw_region(self, _event: object) -> None:
        """Clear the current ROI draft."""
        if self.phase != "roi":
            self.status_extra = "ROI setup is locked."
            self._refresh()
            return
        current = self._current_region_data()
        current["mask"][:] = False
        self._drawing = False
        self.status_extra = "Current ROI draft cleared."
        self._refresh()

    def _button_new_region(self, _event: object) -> None:
        if self.phase != "roi":
            self.status_extra = "ROI setup is locked."
            self._refresh()
            return
        self._button_next_region(_event)

    def _button_next_region(self, _event: object) -> None:
        if self.phase != "roi":
            self.status_extra = "ROI setup is locked."
            self._refresh()
            return
        current = self._current_region_data()
        if not bool(current["mask"].any()):
            self.status_extra = "Current ROI draft is empty."
            self._refresh()
            return
        self._add_region()
        self.mode = "region"
        self._drawing = False
        self.status_extra = f"Accepted ROI {len(self.regions) - 1}. Draw the next ROI draft."
        self._refresh()

    def _button_brush_down(self, _event: object) -> None:
        if self.phase != "roi":
            self.status_extra = "ROI setup is locked."
            self._refresh()
            return
        self.brush_radius = max(1, self.brush_radius - 2)
        self._refresh()

    def _button_brush_up(self, _event: object) -> None:
        if self.phase != "roi":
            self.status_extra = "ROI setup is locked."
            self._refresh()
            return
        self.brush_radius = min(max(self.height, self.width), self.brush_radius + 2)
        self._refresh()

    def _button_redraw_line(self, _event: object) -> None:
        """Clear the current pending line draft."""
        if self.phase != "line":
            return
        self.pending_line = None
        self.current_line_points = []
        self._drawing_line = False
        self.status_extra = "Pending line cleared. Draw again."
        self._refresh()

    def _button_next_line(self, _event: object) -> None:
        """Accept the pending line draft."""
        if self.phase != "line":
            return
        if self._accept_pending_line():
            self.status_extra = f"Accepted line {len(self.lines)}. Draw the next line draft."
        else:
            self.status_extra = "No pending line to accept."
        self._refresh()

    def _button_save(self, _event: object) -> None:
        self._save(str(self.json_path))
        self._refresh()

    def _button_save_close(self, _event: object) -> None:
        if self._save(str(self.json_path)):
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def _clip_xy(self, x: float, y: float) -> tuple[float, float]:
        """Clip floating preview coordinates into the preview image extent."""
        x_clipped = float(np.clip(x, 0.0, self.width - 1.0))
        y_clipped = float(np.clip(y, 0.0, self.height - 1.0))
        return x_clipped, y_clipped

    def _preview_to_original_xy(self, x: float, y: float) -> tuple[float, float]:
        """Map preview pixel coordinates back to original image coordinates."""
        return self.image_view.preview_to_image_xy(x, y)

    def _original_to_preview_points(self, points: list[tuple[float, float]]) -> np.ndarray:
        """Map original image points to preview coordinates."""
        if not points:
            return np.empty((0, 2), dtype=np.float32)
        return self.image_view.image_to_preview_points(np.asarray(points, dtype=np.float64))

    def _line_min_spacing_original(self) -> float:
        """Return the line spacing threshold in original image pixels."""
        scale_x = self.original_width / max(float(self.width), 1.0)
        scale_y = self.original_height / max(float(self.height), 1.0)
        return LINE_MIN_SPACING_PX * max(scale_x, scale_y)

    def _mask_to_original_size(self, mask: np.ndarray) -> np.ndarray:
        """Resize a preview-space boolean mask back to original image resolution."""
        if mask.shape == (self.original_height, self.original_width):
            return mask.astype(bool, copy=True)

        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_image = mask_image.resize(
            (self.original_width, self.original_height),
            _nearest_resampling(),
        )
        return np.asarray(mask_image) > 127

    def _mask_to_preview_size(self, mask: np.ndarray) -> np.ndarray:
        """Resize an original-space boolean mask into preview space."""
        return self.image_view.mask_to_preview(mask)

    def _on_click(self, event: object) -> None:
        """Matplotlib mouse click handler.

        Args:
            event: A ``matplotlib.backend_bases.MouseEvent``.
        """
        if self.phase == "setup" and self.setup_controller is not None:
            if self.setup_controller.on_click(event):
                return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return

        preview_x, preview_y = self._clip_xy(float(event.xdata), float(event.ydata))
        x, y = self._preview_to_original_xy(preview_x, preview_y)
        button = getattr(event, "button", None)
        if self.phase == "line" and self.mode == "line":
            if self.pending_line is not None:
                self.status_extra = "Choose Redraw or Next before drawing another line."
                self._refresh()
                return
            if button == 1:
                self._drawing_line = True
                self.current_line_points = [(x, y)]
            elif button == 3:
                self._drawing_line = False
                self.current_line_points = []
            self._refresh()
            return

        if self.phase == "roi" and self.mode == "region" and button == 1:
            self._drawing = True
            self._stroke_value = True
            self._paint_at(x, y, self._stroke_value)
            self._refresh()

    def _on_draw(self, event: object) -> None:
        """Matplotlib mouse drag handler for line tracing and region painting.

        Args:
            event: A ``matplotlib.backend_bases.MouseEvent``.
        """
        if self.phase == "setup" and self.setup_controller is not None:
            if self.setup_controller.on_motion(event):
                return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return

        if self.phase == "line" and self.mode == "line" and self._drawing_line:
            preview_x, preview_y = self._clip_xy(float(event.xdata), float(event.ydata))
            x, y = self._preview_to_original_xy(preview_x, preview_y)
            if not self.current_line_points:
                self.current_line_points = [(x, y)]
            else:
                last_x, last_y = self.current_line_points[-1]
                if math.hypot(x - last_x, y - last_y) >= self._line_min_spacing_original():
                    self.current_line_points.append((x, y))
            self._refresh()
            return

        if self.phase != "roi" or not self._drawing or self.mode != "region":
            return

        preview_x, preview_y = self._clip_xy(float(event.xdata), float(event.ydata))
        x, y = self._preview_to_original_xy(preview_x, preview_y)
        self._paint_at(x, y, self._stroke_value)
        self._refresh()

    def _on_release(self, event: object) -> None:
        """Finish a line or paint stroke."""
        if self.phase == "setup" and self.setup_controller is not None:
            if self.setup_controller.on_release(event):
                return
        if self.phase == "line" and self.mode == "line" and self._drawing_line:
            if (
                getattr(event, "inaxes", None) is self.ax
                and getattr(event, "xdata", None) is not None
                and getattr(event, "ydata", None) is not None
            ):
                preview_x, preview_y = self._clip_xy(float(event.xdata), float(event.ydata))
                x, y = self._preview_to_original_xy(preview_x, preview_y)
                if self.current_line_points:
                    last_x, last_y = self.current_line_points[-1]
                    if math.hypot(x - last_x, y - last_y) > 0.0:
                        self.current_line_points.append((x, y))

            self._drawing_line = False
            if (
                len(self.current_line_points) >= 2
                and _polyline_length(self.current_line_points) >= self._line_min_spacing_original()
            ):
                self.pending_line = _resample_polyline(self.current_line_points, LINE_RESAMPLE_POINTS)
                self.status_extra = "Pending line ready. Choose Redraw or Next."
            else:
                self.status_extra = "Line draft is too short."
            self.current_line_points = []
            self._refresh()
            return

        if not self._drawing:
            return
        self._drawing = False
        self._refresh()

    def _on_key(self, event: object) -> None:
        """Keyboard shortcut handler."""
        key = (getattr(event, "key", "") or "").lower()
        if self.phase == "setup" and self.setup_controller is not None:
            if self.setup_controller.on_key(event):
                return
        if key in ("d", "enter") and self.phase == "roi":
            self._button_done_regions(event)
        elif key in ("n", "tab") and self.phase == "roi":
            self._button_next_region(event)
        elif key in ("r", "escape") and self.phase == "roi":
            self._button_redraw_region(event)
        elif key in ("+", "=") and self.phase == "roi":
            self._button_brush_up(event)
        elif key in ("-", "_") and self.phase == "roi":
            self._button_brush_down(event)
        elif key in ("r", "escape") and self.phase == "line":
            self._button_redraw_line(event)
        elif key in ("n", "enter", " ", "tab") and self.phase == "line":
            self._button_next_line(event)
        elif key in ("s", "ctrl+s", "cmd+s"):
            self._save(str(self.json_path))
            self._refresh()

    def _paint_at(self, x: float, y: float, value: bool) -> None:
        """Paint or erase the current ROI mask around one image point."""
        mask = self._current_region_data()["mask"]
        cx = int(round(x))
        cy = int(round(y))
        scale_x = self.original_width / max(float(self.width), 1.0)
        scale_y = self.original_height / max(float(self.height), 1.0)
        radius = max(1, int(round(self.brush_radius * max(scale_x, scale_y))))
        x0 = max(0, cx - radius)
        x1 = min(self.original_width - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(self.original_height - 1, cy + radius)
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        disk = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= radius * radius
        patch = mask[y0 : y1 + 1, x0 : x1 + 1]
        patch[disk] = value

    def _mask_overlay(self) -> np.ndarray:
        """Build the RGBA overlay for all ROI masks."""
        overlay = np.zeros((self.height, self.width, 4), dtype=np.float32)
        colors = np.array(
            [
                [0.0, 0.9, 1.0],
                [1.0, 0.8, 0.0],
                [0.3, 1.0, 0.2],
                [1.0, 0.2, 0.7],
                [0.7, 0.4, 1.0],
            ],
            dtype=np.float32,
        )
        for idx, region in enumerate(self.regions):
            mask = region["mask"]
            if not bool(mask.any()):
                continue
            preview_mask = self._mask_to_preview_size(mask)
            color = colors[idx % len(colors)]
            alpha = 0.40 if idx == self.current_region else 0.22
            overlay[preview_mask, :3] = color
            overlay[preview_mask, 3] = alpha
        return overlay

    def _refresh(self) -> None:
        """Redraw overlays, lines, and status text."""
        if self.phase == "setup":
            self.fig.canvas.draw_idle()
            return
        self.mask_artist.set_data(self._mask_overlay())
        self._clear_dynamic_artists()

        for idx, points in enumerate(self.lines, start=1):
            preview_points = self._original_to_preview_points(points)
            xs = preview_points[:, 0]
            ys = preview_points[:, 1]
            line_artist = self.ax.plot(
                xs,
                ys,
                color="lime",
                linewidth=2.0,
            )[0]
            label_x = float(preview_points[-1, 0])
            label_y = float(preview_points[-1, 1])
            label = self.ax.text(
                label_x,
                label_y,
                f" {idx}",
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
            )
            self._dynamic_artists.extend((line_artist, label))

        if self.pending_line is not None:
            preview_points = self._original_to_preview_points(self.pending_line)
            pending_artist = self.ax.plot(
                preview_points[:, 0],
                preview_points[:, 1],
                color="#ff3333",
                linewidth=2.8,
                zorder=6,
            )[0]
            self._dynamic_artists.append(pending_artist)

        if self.current_line_points:
            preview_points = self._original_to_preview_points(self.current_line_points)
            xs = preview_points[:, 0]
            ys = preview_points[:, 1]
            active_line = self.ax.plot(
                xs,
                ys,
                color="yellow",
                linewidth=2.0,
                marker="o",
                markersize=3,
            )[0]
            self._dynamic_artists.append(active_line)

        current = self._current_region_data()
        pixels = int(current["mask"].sum())
        size_text = f"Preview: {self.preview_width}x{self.preview_height}"
        if self.is_downsampled:
            size_text += f" -> Original: {self.original_width}x{self.original_height}"
        camera_text = f"Camera: {CAMERA_MODEL_LABELS[self.camera_model]}"
        if self.camera_model == "pinhole":
            camera_text += f" ({self.fov_deg:.2f} deg)"
        phase_text = "ROI SETUP" if self.phase == "roi" else "LINES"
        if self.phase == "setup":
            phase_text = "VIEW SETUP"
        self.status.set_text(
            f"Phase: {phase_text} | Mode: {self.mode.upper()} | ROI {self.current_region + 1}/{len(self.regions)} "
            f"'{current['name']}' ({pixels} original px) | Lines: {len(self.lines)} | "
            f"Brush: {self.brush_radius}px | {camera_text} | {size_text} | Save: {self.json_path}"
            f" | {self.status_extra}"
        )
        self.fig.canvas.draw_idle()

    def _accept_pending_line(self) -> bool:
        """Move the pending line draft into accepted lines."""
        if self.pending_line is None:
            return False
        self.lines.append(self.pending_line)
        self.pending_line = None
        self.current_line_points = []
        self._drawing_line = False
        return True

    def _line_json(self) -> list[dict[str, list[list[float]]]]:
        """Convert preview-space line curves into original view-sphere directions."""
        if self.camera is None:
            raise RuntimeError("Camera is not initialized.")
        result: list[dict[str, list[list[float]]]] = []
        for points in self.lines:
            if len(points) != LINE_RESAMPLE_POINTS:
                raise ValueError(
                    f"Saved lines must contain exactly {LINE_RESAMPLE_POINTS} points; "
                    f"got {len(points)}."
                )

            xs = np.array([p[0] for p in points], dtype=np.float64)
            ys = np.array([p[1] for p in points], dtype=np.float64)
            lam, phi = self.camera.pixel_to_direction(xs, ys)
            points_dir = [
                [float(lam_i), float(phi_i)]
                for lam_i, phi_i in zip(lam, phi)
            ]
            result.append({"points_dir": points_dir})
        return result

    def _save(self, json_path: str) -> bool:
        """Write masks to disk and emit the annotation JSON.

        Args:
            json_path: Output JSON path.
        """
        if self.phase != "line":
            self.status_extra = "Press Done ROI before saving."
            return False
        self._accept_pending_line()
        json_out = Path(json_path).resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        regions_json: list[dict[str, str]] = []

        for idx, region in enumerate(self.regions, start=1):
            mask = region["mask"]
            if not bool(mask.any()):
                continue
            safe_name = _sanitize_name(str(region["name"]), f"region_{idx}")
            unique_name = safe_name
            suffix = 2
            while unique_name in used_names:
                unique_name = f"{safe_name}_{suffix}"
                suffix += 1
            used_names.add(unique_name)

            mask_path = json_out.parent / f"mask_{unique_name}.png"
            Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
            regions_json.append(
                {
                    "name": str(region["name"]),
                    "mask_path": _relative_path(mask_path, json_out.parent),
                }
            )

        payload = build_annotation_payload(
            image_path=self.image_path,
            output_path=json_out,
            source_image_path=self.source_image_path,
            camera_model=self.camera_model,
            fov_deg=self.fov_deg if self.camera_model == "pinhole" else None,
            view_metadata=self.view_metadata,
            lines=self._line_json(),
            regions=regions_json,
        )
        write_annotation_json(json_out, payload)
        self.status_extra = f"Saved annotations to {json_out}."
        return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the annotation tool.

    Returns:
        Namespace with attributes: ``image`` (str), ``output_dir`` (str | None).
    """
    parser = argparse.ArgumentParser(description="MaDCoW annotation GUI.")
    parser.add_argument(
        "--image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help=f"Path to the input image. Defaults to {DEFAULT_IMAGE}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save JSON and PNG masks. Defaults to the input image directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_path = Path(args.image).resolve()
    fov, _source = estimate_fov_from_exif(str(image_path))
    output_dir = args.output_dir or str(image_path.parent)
    AnnotationGUI(
        str(image_path),
        fov,
        output_dir,
        start_with_setup=True,
    ).run()
