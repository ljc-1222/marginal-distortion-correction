"""Minimal Matplotlib GUI for 2D interactive snapping annotations."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ExifTags, Image

from annotation_gui import AnnotationSession, render_centered_equirectangular
from annotation_gui.base import (
    add_button_row,
    add_styled_button,
    add_text_box,
    clear_widget_axes,
    create_image_figure,
    set_image_artist,
)
from annotation_gui.io import build_annotation_payload, write_annotation_json

from .snap2d import SnapConfig, SnapResult, load_snap_config, snap_annotation
from .snap2d.io import result_to_madcow_annotation_dict, save_madcow_annotation_json


DEFAULT_IMAGE = Path(__file__).resolve().parent / "data" / "test_1.jpg"
FALLBACK_FOV_DEG = 90.0
PREVIEW_MAX_SIDE = 1200
STROKE_MIN_SPACING_PX = 3.0
CAMERA_TYPE_CHOICES = ("pinhole", "panorama")
SNAP_MODE_CHOICES = ("line", "curve")

STATE_CAMERA_SELECT = "camera_select"
STATE_PINHOLE_SETUP = "pinhole_setup"
STATE_PANORAMA_SETUP = "panorama_setup"
STATE_ANNOTATE = "annotate"

MIN_CROP_FRACTION = 0.08
CROP_EDGE_HIT_PX = 10.0
PANORAMA_MAX_ABS_PITCH = (math.pi / 2.0) - 1e-4

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "interactive_snapping_2d_matplotlib"
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
    """Estimate horizontal field-of-view from EXIF metadata when available."""
    image_path_obj = Path(image_path).expanduser().resolve()
    try:
        with Image.open(image_path_obj) as img:
            width = int(img.width)
    except OSError:
        return float(fallback), "fallback"

    try:
        import piexif

        exif_dict = piexif.load(str(image_path_obj))
        exif_ifd = exif_dict.get("Exif", {}) or {}
        f35 = _ratio_to_float(exif_ifd.get(piexif.ExifIFD.FocalLengthIn35mmFilm))
        if f35 and f35 > 0:
            return math.degrees(2.0 * math.atan(36.0 / (2.0 * f35))), "EXIF 35mm-equivalent focal length"

        focal = _ratio_to_float(exif_ifd.get(piexif.ExifIFD.FocalLength))
        x_res = _ratio_to_float(exif_ifd.get(piexif.ExifIFD.FocalPlaneXResolution))
        unit_raw = exif_ifd.get(piexif.ExifIFD.FocalPlaneResolutionUnit)
        unit = int(unit_raw) if unit_raw is not None else None
        if focal and focal > 0 and x_res and x_res > 0 and unit in (2, 3, 4, 5):
            unit_to_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}
            sensor_width_mm = width / x_res * unit_to_mm[unit]
            if sensor_width_mm > 0:
                return math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal))), (
                    "EXIF focal length and focal-plane resolution"
                )
    except (ImportError, OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        pass

    try:
        with Image.open(image_path_obj) as img:
            exif = img.getexif()
            lookup = {ExifTags.TAGS.get(key, key): value for key, value in exif.items()}
    except OSError:
        return float(fallback), "fallback"

    f35 = _ratio_to_float(lookup.get("FocalLengthIn35mmFilm"))
    if f35 and f35 > 0:
        return math.degrees(2.0 * math.atan(36.0 / (2.0 * f35))), "EXIF 35mm-equivalent focal length"
    return float(fallback), "fallback"


def _compute_preview_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Compute a preview size that preserves the input aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}.")
    if max_side <= 0:
        return width, height
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = float(max_side) / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _lanczos_resampling() -> int:
    """Return Pillow's LANCZOS enum across Pillow versions."""
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.LANCZOS)
    return int(Image.LANCZOS)


def _normalize_camera_type(camera_type: str) -> str:
    """Normalize GUI camera type aliases."""
    if camera_type not in CAMERA_TYPE_CHOICES:
        raise ValueError(f"camera_type must be one of {CAMERA_TYPE_CHOICES}; got {camera_type!r}.")
    return camera_type


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _equirect_pixel_to_angles(x: np.ndarray, y: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert equirectangular pixel coordinates to yaw and pitch angles."""
    if width < 2 or height < 2:
        raise ValueError(f"Panorama view requires width and height of at least 2; got {width}x{height}.")
    lam = (x / float(width - 1)) * (2.0 * math.pi) - math.pi
    phi = (y / float(height - 1)) * math.pi - (math.pi / 2.0)
    return lam, phi


def _angles_to_equirect_pixel(
    lam: np.ndarray,
    phi: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project yaw and pitch angles to equirectangular pixel coordinates."""
    x = ((lam + math.pi) / (2.0 * math.pi)) * float(width - 1)
    y = ((phi + (math.pi / 2.0)) / math.pi) * float(height - 1)
    return x, y


def _angles_to_vectors(lam: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert yaw and pitch angles to unit-sphere vectors."""
    cos_phi = np.cos(phi)
    x = np.sin(lam) * cos_phi
    y = np.sin(phi)
    z = np.cos(lam) * cos_phi
    return x, y, z


def _vectors_to_angles(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert unit-sphere vectors to yaw and pitch angles."""
    lam = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt((x * x) + (z * z)))
    return lam, phi


def _local_to_world_angles(
    local_lam: np.ndarray,
    local_phi: np.ndarray,
    center_lam: float,
    center_phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map local equirectangular angles to world panorama angles."""
    x, y, z = _angles_to_vectors(local_lam, local_phi)

    cp = math.cos(-center_phi)
    sp = math.sin(-center_phi)
    y_pitch = (y * cp) - (z * sp)
    z_pitch = (y * sp) + (z * cp)

    cy = math.cos(center_lam)
    sy = math.sin(center_lam)
    x_world = (x * cy) + (z_pitch * sy)
    z_world = (-x * sy) + (z_pitch * cy)
    return _vectors_to_angles(x_world, y_pitch, z_world)


def _world_to_local_angles(
    world_lam: np.ndarray,
    world_phi: np.ndarray,
    center_lam: float,
    center_phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map world panorama angles to local equirectangular view angles."""
    x, y, z = _angles_to_vectors(world_lam, world_phi)

    cy = math.cos(-center_lam)
    sy = math.sin(-center_lam)
    x_yaw = (x * cy) + (z * sy)
    z_yaw = (-x * sy) + (z * cy)

    cp = math.cos(center_phi)
    sp = math.sin(center_phi)
    y_local = (y * cp) - (z_yaw * sp)
    z_local = (y * sp) + (z_yaw * cp)
    return _vectors_to_angles(x_yaw, y_local, z_local)


def _remap_panorama_preview(
    source_image: np.ndarray,
    center_lam: float,
    center_phi: float,
    out_width: int | None = None,
    out_height: int | None = None,
) -> np.ndarray:
    """Render a local equirectangular panorama preview for the selected center."""
    import cv2

    arr = np.asarray(source_image)
    if arr.ndim < 2:
        raise ValueError(f"source_image must have at least two dimensions; got {arr.shape}.")
    src_height, src_width = arr.shape[:2]
    dst_width = src_width if out_width is None else int(out_width)
    dst_height = src_height if out_height is None else int(out_height)
    yy, xx = np.indices((dst_height, dst_width), dtype=np.float64)
    local_lam, local_phi = _equirect_pixel_to_angles(xx, yy, dst_width, dst_height)
    world_lam, world_phi = _local_to_world_angles(local_lam, local_phi, center_lam, center_phi)
    source_x, source_y = _angles_to_equirect_pixel(world_lam, world_phi, src_width, src_height)
    return cv2.remap(
        arr,
        source_x.astype(np.float32),
        source_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


@dataclass
class AnnotationItem:
    """One annotation stored by the GUI."""

    result: SnapResult
    preview_points: np.ndarray
    source_preview_points: np.ndarray
    source_original_points: np.ndarray
    camera_type: str
    snap_mode: str


class LineAidAnnotationGUI:
    """Line annotation GUI with camera selection and panorama view setup."""

    def __init__(
        self,
        image_path: str,
        fov_deg: float | None,
        output_dir: str,
        camera_type: str = "pinhole",
        snap_config: SnapConfig | None = None,
        initial_session: AnnotationSession | None = None,
        json_path: str | Path | None = None,
        fig: Any | None = None,
        ax: Any | None = None,
        image_artist: Any | None = None,
        status: Any | None = None,
        help_text: Any | None = None,
        widgets: list[Any] | None = None,
        on_complete: Any | None = None,
        save_close_label: str = "Save+Close",
        allow_empty_annotations: bool = False,
        control_layout: str = "bottom",
    ) -> None:
        external_items = (fig, ax, image_artist, status, help_text)
        if any(item is not None for item in external_items) and not all(item is not None for item in external_items):
            raise ValueError("fig, ax, image_artist, status, and help_text must be provided together.")
        self.image_path = str(Path(image_path).resolve())
        self.initial_camera_type = _normalize_camera_type(camera_type)
        self.camera_type: str | None = None
        self.fov_deg = None if fov_deg is None else float(fov_deg)
        if self.fov_deg is not None and (self.fov_deg <= 0 or self.fov_deg >= 180):
            raise ValueError(f"fov_deg must lie in (0, 180); got {self.fov_deg}.")
        self.default_fov_deg = float(self.fov_deg if self.fov_deg is not None else FALLBACK_FOV_DEG)
        self.snap_config = snap_config or SnapConfig()

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = initial_session or AnnotationSession.from_image(self.image_path, PREVIEW_MAX_SIDE, self.fov_deg)
        self.image_path = self.session.image_path if initial_session is not None else self.image_path
        self.original_image = self.session.source_view.image
        self.preview_image = self.session.source_view.preview
        self.original_width = self.session.source_view.width
        self.original_height = self.session.source_view.height
        self.preview_width = self.session.source_view.preview_width
        self.preview_height = self.session.source_view.preview_height
        self.is_downsampled = self.session.source_view.is_downsampled
        self.state = STATE_CAMERA_SELECT
        self.snap_mode = "line"
        self.annotations: list[AnnotationItem] = []
        self.pending: AnnotationItem | None = None
        self.current_stroke: list[tuple[float, float]] = []
        self._drawing_stroke = False
        self._drag_action: str | None = None
        self._drag_start_xy: tuple[float, float] | None = None
        self._drag_start_center: tuple[float, float] | None = None
        self._drag_start_crop: tuple[float, float, float, float] | None = None
        self._dynamic_artists: list[Any] = []
        self.status_extra = "Choose a camera mode."

        self.panorama_center_lam = 0.0
        self.panorama_center_phi = 0.0
        self.panorama_setup_image: np.ndarray | None = None
        self.panorama_crop_box = self._default_crop_box()
        self.annotation_image: np.ndarray | None = None
        self.annotation_original_image: np.ndarray | None = None
        self.annotation_setup_image: np.ndarray | None = None
        self.annotation_center_lam = 0.0
        self.annotation_center_phi = 0.0
        self.annotation_crop_box = self.panorama_crop_box
        self.annotation_crop_origin = (0.0, 0.0)
        self.on_complete = on_complete
        self.save_close_label = save_close_label
        self.allow_empty_annotations = bool(allow_empty_annotations)
        self.control_layout = control_layout
        self._connection_ids: list[int] = []

        stem = Path(self.image_path).stem
        self.json_path = Path(json_path).resolve() if json_path is not None else self.output_dir / f"{stem}.json"

        if fig is None:
            self._build_figure()
        else:
            self._attach_figure(fig, ax, image_artist, status, help_text, widgets)
        if initial_session is None:
            self._show_camera_select()
        else:
            self.enter_annotation_session(initial_session)

    def run(self) -> None:
        """Open the Matplotlib GUI and block until it closes."""
        import matplotlib.pyplot as plt

        plt.show()

    @classmethod
    def from_session(
        cls,
        session: AnnotationSession,
        output_dir: str | Path,
        snap_config: SnapConfig | None = None,
        json_path: str | Path | None = None,
        fig: Any | None = None,
        ax: Any | None = None,
        image_artist: Any | None = None,
        status: Any | None = None,
        help_text: Any | None = None,
        widgets: list[Any] | None = None,
        on_complete: Any | None = None,
        save_close_label: str = "Save+Close",
        allow_empty_annotations: bool = False,
        control_layout: str = "bottom",
    ) -> "LineAidAnnotationGUI":
        """Create a line snapping GUI that starts from a prepared annotation session."""
        return cls(
            image_path=session.image_path,
            fov_deg=session.fov_deg,
            output_dir=str(output_dir),
            snap_config=snap_config,
            initial_session=session,
            json_path=json_path,
            fig=fig,
            ax=ax,
            image_artist=image_artist,
            status=status,
            help_text=help_text,
            widgets=widgets,
            on_complete=on_complete,
            save_close_label=save_close_label,
            allow_empty_annotations=allow_empty_annotations,
            control_layout=control_layout,
        )

    def _build_figure(self) -> None:
        """Create the figure, controls, and event callbacks."""
        self.blank_image = np.full((max(1, self.preview_height), max(1, self.preview_width), 3), 245, dtype=np.uint8)
        self.fig, self.ax, self.image_artist, self.status, self.help_text = create_image_figure(
            "2D Interactive Snapping Annotation",
            self.blank_image,
        )

        self.widgets: list[Any] = []
        self._buttons: dict[str, Any] = {}
        self.mode_radio: Any | None = None
        self.fov_box: Any | None = None
        self._syncing_fov_box = False

        self._connect_events()

    def _attach_figure(
        self,
        fig: Any,
        ax: Any,
        image_artist: Any,
        status: Any,
        help_text: Any,
        widgets: list[Any] | None,
    ) -> None:
        """Attach line annotation to an existing Matplotlib figure."""
        self.blank_image = np.full((max(1, self.preview_height), max(1, self.preview_width), 3), 245, dtype=np.uint8)
        self.fig = fig
        self.ax = ax
        self.image_artist = image_artist
        self.status = status
        self.help_text = help_text
        self.widgets = widgets if widgets is not None else []
        self._buttons = {}
        self.mode_radio = None
        self.fov_box = None
        self._syncing_fov_box = False
        self._connect_events()

    def _connect_events(self) -> None:
        """Connect line-aid event handlers and retain ids for integrated GUI reuse."""
        self._connection_ids = [
            self.fig.canvas.mpl_connect("button_press_event", self._on_click),
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.fig.canvas.mpl_connect("button_release_event", self._on_release),
            self.fig.canvas.mpl_connect("key_press_event", self._on_key),
        ]

    def disconnect_events(self) -> None:
        """Disconnect line-aid event handlers from the figure."""
        for connection_id in self._connection_ids:
            self.fig.canvas.mpl_disconnect(connection_id)
        self._connection_ids.clear()

    def _clear_controls(self) -> None:
        """Remove all current-state controls from the figure."""
        clear_widget_axes(self.widgets)
        self._buttons.clear()
        self.mode_radio = None
        self.fov_box = None

    def _set_controls(self, button_specs: list[tuple[str, str, float, float, float, Any]]) -> None:
        """Create only the buttons that are valid for the current GUI state."""
        self._clear_controls()
        for key, label, x0, y0, width, callback in button_specs:
            button = add_button_row(self.fig, self.widgets, [(label, width, callback)], y0=y0, x0=x0)[0]
            self._buttons[key] = button

    def _set_annotation_controls(self) -> None:
        """Create annotation review controls for the active camera mode."""
        from matplotlib.widgets import RadioButtons

        self._clear_controls()
        if self.control_layout == "sidebar":
            self._buttons["line"] = add_styled_button(
                self.fig,
                self.widgets,
                "Line",
                (0.04, 0.36, 0.14, 0.045),
                lambda _event: self._on_snap_mode_changed("line"),
                selected=self.snap_mode == "line",
            )
            self._buttons["curve"] = add_styled_button(
                self.fig,
                self.widgets,
                "Curve",
                (0.20, 0.36, 0.14, 0.045),
                lambda _event: self._on_snap_mode_changed("curve"),
                selected=self.snap_mode == "curve",
            )
            self._buttons["redraw"] = add_styled_button(
                self.fig,
                self.widgets,
                "Redraw",
                (0.04, 0.29, 0.14, 0.045),
                self._button_redraw,
            )
            self._buttons["next"] = add_styled_button(
                self.fig,
                self.widgets,
                "Next",
                (0.20, 0.29, 0.14, 0.045),
                self._button_next,
            )
            self._buttons["save"] = add_styled_button(
                self.fig,
                self.widgets,
                "Save",
                (0.04, 0.235, 0.14, 0.045),
                self._button_save,
            )
            self._buttons["save_close"] = add_styled_button(
                self.fig,
                self.widgets,
                self.save_close_label,
                (0.20, 0.235, 0.14, 0.045),
                self._button_save_close,
                primary=True,
            )
            return
        mode_ax = self.fig.add_axes([0.04, 0.035, 0.14, 0.12])
        mode_ax.set_title("Type", fontsize=9)
        active_mode = SNAP_MODE_CHOICES.index(self.snap_mode)
        self.mode_radio = RadioButtons(mode_ax, SNAP_MODE_CHOICES, active=active_mode)
        self.mode_radio.on_clicked(self._on_snap_mode_changed)
        self.widgets.append(self.mode_radio)

        button_specs = [
            ("redraw", "Redraw", 0.09, self._button_redraw),
            ("next", "Next", 0.09, self._button_next),
            ("save", "Save", 0.09, self._button_save),
            ("save_close", self.save_close_label, 0.13, self._button_save_close),
        ]
        created = add_button_row(
            self.fig,
            self.widgets,
            [(label, width, callback) for _key, label, width, callback in button_specs],
            y0=0.08,
        )
        for (key, _label, _width, _callback), button in zip(button_specs, created):
            self._buttons[key] = button

    def _set_axes_visible(self, visible: bool) -> None:
        """Show or hide the main image axes."""
        self.ax.set_visible(visible)

    def _set_image(self, image: np.ndarray) -> None:
        """Update the image artist and axes limits for a new image."""
        set_image_artist(self.image_artist, self.ax, image)

    def _clear_dynamic_artists(self) -> None:
        """Remove dynamic overlay artists."""
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists.clear()

    def _default_crop_box(self) -> tuple[float, float, float, float]:
        """Return the default panorama crop box: left/right/up/down 90 degrees."""
        x0 = float(self.preview_width) * 0.25
        x1 = float(self.preview_width) * 0.75
        y0 = 0.0
        y1 = float(self.preview_height)
        return x0, y0, x1, y1

    def _min_crop_width(self) -> float:
        """Return the minimum crop width in preview pixels."""
        return max(8.0, float(self.preview_width) * MIN_CROP_FRACTION)

    def _min_crop_height(self) -> float:
        """Return the minimum crop height in preview pixels."""
        return max(8.0, float(self.preview_height) * MIN_CROP_FRACTION)

    def _show_camera_select(self) -> None:
        """Show the camera-mode selection state."""
        self.state = STATE_CAMERA_SELECT
        self.camera_type = None
        self.pending = None
        self.annotations.clear()
        self.current_stroke = []
        self._drawing_stroke = False
        self._drag_action = None
        self.snap_mode = "line"
        self.panorama_center_lam = 0.0
        self.panorama_center_phi = 0.0
        self.panorama_setup_image = None
        self.panorama_crop_box = self._default_crop_box()
        self.annotation_image = None
        self.annotation_original_image = None
        self.annotation_setup_image = None
        self.annotation_center_lam = 0.0
        self.annotation_center_phi = 0.0
        self.annotation_crop_box = self.panorama_crop_box
        self.annotation_crop_origin = (0.0, 0.0)
        self._clear_dynamic_artists()
        self._set_controls(
            [
                ("pinhole", "Pinhole", 0.35, 0.08, 0.12, self._button_choose_pinhole),
                ("panorama", "Panorama", 0.53, 0.08, 0.14, self._button_choose_panorama),
            ]
        )
        self._set_image(self.preview_image)
        self._set_axes_visible(True)
        self.status.set_text("Choose camera mode.")
        self.help_text.set_text("Pinhole opens FOV setup. Panorama opens view/crop setup first.")
        self.fig.canvas.draw_idle()

    def _button_choose_pinhole(self, _event: object) -> None:
        """Enter pinhole line annotation."""
        self._enter_pinhole_setup()

    def _button_choose_panorama(self, _event: object) -> None:
        """Enter panorama view setup."""
        self._enter_panorama_setup()

    def _enter_pinhole_setup(self) -> None:
        """Show pinhole FOV setup before annotation starts."""
        self.state = STATE_PINHOLE_SETUP
        self.camera_type = "pinhole"
        if self.fov_deg is None:
            self.fov_deg = FALLBACK_FOV_DEG
        self.pending = None
        self.annotations.clear()
        self.current_stroke = []
        self._drawing_stroke = False
        self._drag_action = None
        self._clear_dynamic_artists()
        self._clear_controls()
        self._set_image(self.preview_image)
        self._set_axes_visible(True)

        self.fov_box = add_text_box(
            self.fig,
            self.widgets,
            "FOV",
            f"{self.fov_deg:.2f}",
            self._on_fov_submit,
            width=0.18,
            y0=0.08,
            x0=0.38,
        )
        self._buttons["done"] = add_button_row(
            self.fig,
            self.widgets,
            [("Done", 0.10, self._button_done_pinhole)],
            y0=0.08,
            x0=0.60,
        )[0]

        self.status_extra = "Adjust pinhole horizontal FOV."
        self._refresh()

    def _enter_panorama_setup(self) -> None:
        """Show panorama center and crop setup."""
        self.state = STATE_PANORAMA_SETUP
        self.camera_type = "panorama"
        self.pending = None
        self.annotations.clear()
        self.current_stroke = []
        self._drawing_stroke = False
        self._drag_action = None
        self.panorama_center_lam = 0.0
        self.panorama_center_phi = 0.0
        self.panorama_crop_box = self._default_crop_box()
        self._set_controls(
            [
                ("h_reset", "H Reset", 0.32, 0.08, 0.11, self._button_h_reset),
                ("v_reset", "V Reset", 0.45, 0.08, 0.11, self._button_v_reset),
                ("done", "Done", 0.58, 0.08, 0.10, self._button_done_setup),
            ]
        )
        self._set_axes_visible(True)
        self._render_panorama_setup_image()
        self.status_extra = "Adjust panorama view center and crop range."
        self._refresh()

    def _button_done_pinhole(self, _event: object) -> None:
        """Finalize pinhole FOV setup and enter line annotation."""
        if self.state == STATE_PINHOLE_SETUP:
            self._enter_annotation("pinhole")

    def _enter_annotation(self, camera_type: str) -> None:
        """Enter line annotation for the selected camera type."""
        self.state = STATE_ANNOTATE
        selected_camera_type = _normalize_camera_type(camera_type)
        self.camera_type = selected_camera_type
        if selected_camera_type == "pinhole" and self.fov_deg is None:
            self.fov_deg = FALLBACK_FOV_DEG
        self.current_stroke = []
        self.pending = None
        self._drawing_stroke = False
        self._drag_action = None
        self._set_annotation_controls()
        self._set_axes_visible(True)
        if selected_camera_type == "panorama":
            self._build_panorama_annotation_image()
            self.camera_type = self.session.camera_model
        else:
            self.session.use_pinhole_source(float(self.fov_deg))
            self.annotation_original_image = self.session.image
            self.annotation_image = self.session.preview
            self.annotation_setup_image = None
            self.annotation_center_lam = 0.0
            self.annotation_center_phi = 0.0
            self.annotation_crop_box = (0.0, 0.0, float(self.original_width), float(self.original_height))
            self.annotation_crop_origin = (0.0, 0.0)
        self.fov_deg = self.session.fov_deg
        self.status_extra = "Draw a rough stroke. Then choose Redraw or Next."
        self._refresh()

    def enter_annotation_session(self, session: AnnotationSession) -> None:
        """Enter line annotation using an already finalized shared session."""
        if session.camera_model not in ("pinhole", "panorama_view"):
            raise ValueError(f"Unsupported annotation camera model: {session.camera_model!r}.")
        if session.camera_model == "pinhole" and session.fov_deg is None:
            raise ValueError("pinhole annotation session must define fov_deg.")

        self.session = session
        self.image_path = session.image_path
        self.camera_type = str(session.camera_model)
        self.fov_deg = session.fov_deg
        self.original_image = session.source_view.image
        self.preview_image = session.source_view.preview
        self.original_width = session.source_view.width
        self.original_height = session.source_view.height
        self.preview_width = session.source_view.preview_width
        self.preview_height = session.source_view.preview_height
        self.is_downsampled = session.source_view.is_downsampled
        self.state = STATE_ANNOTATE
        self.snap_mode = "line"
        self.annotations.clear()
        self.pending = None
        self.current_stroke = []
        self._drawing_stroke = False
        self._drag_action = None
        self._drag_start_xy = None
        self._drag_start_center = None
        self._drag_start_crop = None
        self._clear_dynamic_artists()
        self._set_annotation_controls()
        self._set_axes_visible(True)

        self.annotation_original_image = session.image
        self.annotation_image = session.preview
        self.annotation_setup_image = None
        if session.panorama_result is not None:
            self.annotation_center_lam = float(session.panorama_result.center_yaw_rad)
            self.annotation_center_phi = float(session.panorama_result.center_pitch_rad)
            self.annotation_crop_box = session.panorama_result.crop_original_px
        else:
            self.annotation_center_lam = 0.0
            self.annotation_center_phi = 0.0
            self.annotation_crop_box = (0.0, 0.0, float(session.width), float(session.height))
        self.annotation_crop_origin = (0.0, 0.0)
        self.status_extra = "Draw a rough stroke. Then choose Redraw or Next."
        self._refresh()

    def _set_fov_value(self, value: float) -> bool:
        """Validate and store the pinhole FOV value."""
        if not math.isfinite(value) or value <= 0.0 or value >= 180.0:
            self.status_extra = "FOV must be in (0, 180) degrees."
            self._refresh()
            return False
        self.fov_deg = float(value)
        if self.fov_box is not None:
            self._syncing_fov_box = True
            try:
                self.fov_box.set_val(f"{self.fov_deg:.2f}")
            finally:
                self._syncing_fov_box = False
        self.status_extra = "Pinhole FOV updated."
        self._refresh()
        return True

    def _on_fov_submit(self, text: str) -> None:
        """Apply a FOV typed into the setup text box."""
        if self._syncing_fov_box:
            return
        try:
            value = float(text)
        except ValueError:
            self.status_extra = "FOV must be numeric."
            self._refresh()
            return
        self._set_fov_value(value)

    def _button_done_setup(self, _event: object) -> None:
        """Finalize panorama setup and enter line annotation."""
        if self.state == STATE_PANORAMA_SETUP:
            self._enter_annotation("panorama")

    def _button_h_reset(self, _event: object) -> None:
        """Reset the panorama horizontal view center."""
        if self.state != STATE_PANORAMA_SETUP:
            return
        self.panorama_center_lam = 0.0
        self._render_panorama_setup_image()
        self.status_extra = "Horizontal view center reset."
        self._refresh()

    def _button_v_reset(self, _event: object) -> None:
        """Reset the panorama vertical view center."""
        if self.state != STATE_PANORAMA_SETUP:
            return
        self.panorama_center_phi = 0.0
        self._render_panorama_setup_image()
        self.status_extra = "Vertical view center reset."
        self._refresh()

    def _on_snap_mode_changed(self, label: str) -> None:
        """Set the annotation type and resnap the pending stroke if present."""
        if label not in SNAP_MODE_CHOICES:
            raise ValueError(f"mode must be one of {SNAP_MODE_CHOICES}; got {label!r}.")
        self.snap_mode = label
        if self.state != STATE_ANNOTATE:
            return
        if self.control_layout == "sidebar":
            self._set_annotation_controls()
        if self.pending is not None:
            self._resnap_pending()
        else:
            self.status_extra = f"Annotation type set to {self.snap_mode}."
            self._refresh()

    def _button_redraw(self, _event: object) -> None:
        """Discard the pending result and allow the user to draw it again."""
        if self.state != STATE_ANNOTATE:
            return
        self.pending = None
        self.current_stroke = []
        self._drawing_stroke = False
        self.status_extra = "Pending result discarded. Draw again."
        self._refresh()

    def _button_next(self, _event: object) -> None:
        """Accept the pending result and prepare for the next annotation."""
        if self.state != STATE_ANNOTATE:
            return
        if self._accept_pending():
            self.status_extra = f"Accepted annotation {len(self.annotations)}. Draw the next stroke."
        else:
            self.status_extra = "No pending result to accept. Draw a rough stroke first."
        self._refresh()

    def _button_save(self, _event: object) -> None:
        self._save()
        self._refresh()

    def _button_save_close(self, _event: object) -> None:
        if self._save():
            if self.on_complete is not None:
                self.on_complete(self)
                return
            import matplotlib.pyplot as plt

            plt.close(self.fig)
        else:
            self._refresh()

    def _on_key(self, event: object) -> None:
        """Handle keyboard aliases for the visible actions."""
        key = (getattr(event, "key", "") or "").lower()
        if self.state == STATE_ANNOTATE:
            if key in ("r", "escape"):
                self._button_redraw(event)
            elif key in ("n", "enter", " "):
                self._button_next(event)
            elif key in ("s", "ctrl+s", "cmd+s"):
                self._button_save(event)
        elif self.state == STATE_PINHOLE_SETUP:
            if key in ("enter", " "):
                self._button_done_pinhole(event)
        elif self.state == STATE_PANORAMA_SETUP:
            if key in ("h",):
                self._button_h_reset(event)
            elif key in ("v",):
                self._button_v_reset(event)
            elif key in ("enter", " "):
                self._button_done_setup(event)

    def _clip_setup_xy(self, x: float, y: float) -> tuple[float, float]:
        """Clip floating setup coordinates into the preview extent."""
        return float(np.clip(x, 0.0, self.preview_width - 1.0)), float(np.clip(y, 0.0, self.preview_height - 1.0))

    def _clip_annotation_xy(self, x: float, y: float) -> tuple[float, float]:
        """Clip floating annotation coordinates into the annotation image extent."""
        image = self._require_annotation_image()
        height, width = image.shape[:2]
        return float(np.clip(x, 0.0, width - 1.0)), float(np.clip(y, 0.0, height - 1.0))

    def _require_annotation_image(self) -> np.ndarray:
        """Return the active annotation preview image."""
        if self.annotation_image is None:
            raise RuntimeError("Annotation image is not initialized.")
        return self.annotation_image

    def _require_annotation_original_image(self) -> np.ndarray:
        """Return the active original-resolution annotation image."""
        if self.annotation_original_image is None:
            raise RuntimeError("Original-resolution annotation image is not initialized.")
        return self.annotation_original_image

    def _render_panorama_setup_image(self) -> None:
        """Render the full preview panorama for the current center."""
        self.panorama_setup_image = render_centered_equirectangular(
            self.session.source_view.preview,
            self.panorama_center_lam,
            self.panorama_center_phi,
            self.preview_width,
            self.preview_height,
        )

    def _build_panorama_annotation_image(self) -> None:
        """Render the selected original-resolution panorama view for annotation."""
        crop_box = self._normalized_crop_box()
        self.session.use_panorama_view(
            self.output_dir,
            self.panorama_center_lam,
            self.panorama_center_phi,
            crop_box,
            PREVIEW_MAX_SIDE,
        )
        if self.session.panorama_result is None:
            raise RuntimeError("Panorama annotation view was not created.")
        self.annotation_center_lam = self.panorama_center_lam
        self.annotation_center_phi = self.panorama_center_phi
        self.annotation_crop_box = self.session.panorama_result.crop_original_px
        self.annotation_setup_image = None
        self.annotation_crop_origin = (0.0, 0.0)
        self.annotation_original_image = self.session.image
        self.annotation_image = self.session.preview
        self.fov_deg = self.session.fov_deg

    def _crop_indices_from_box(self, crop_box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """Convert a setup crop box to array slice indices using pixel-center inclusion."""
        x0, y0, x1, y1 = crop_box
        ix0 = int(np.ceil(x0))
        iy0 = int(np.ceil(y0))
        ix1 = int(np.ceil(x1))
        iy1 = int(np.ceil(y1))
        ix0 = max(0, min(ix0, self.preview_width - 2))
        iy0 = max(0, min(iy0, self.preview_height - 2))
        ix1 = max(ix0 + 2, min(ix1, self.preview_width))
        iy1 = max(iy0 + 2, min(iy1, self.preview_height))
        return ix0, iy0, ix1, iy1

    def _normalized_crop_box(self) -> tuple[float, float, float, float]:
        """Return the crop box ordered and clipped to the setup image extent."""
        x0, y0, x1, y1 = self.panorama_crop_box
        width = float(self.preview_width)
        height = float(self.preview_height)
        left = float(np.clip(min(x0, x1), 0.0, width))
        right = float(np.clip(max(x0, x1), 0.0, width))
        top = float(np.clip(min(y0, y1), 0.0, height))
        bottom = float(np.clip(max(y0, y1), 0.0, height))
        min_width = min(self._min_crop_width(), width)
        min_height = min(self._min_crop_height(), height)
        if right - left < min_width:
            center = (left + right) * 0.5
            half = min_width * 0.5
            center = float(np.clip(center, half, width - half))
            left = center - half
            right = center + half
        if bottom - top < min_height:
            center = (top + bottom) * 0.5
            half = min_height * 0.5
            center = float(np.clip(center, half, height - half))
            top = center - half
            bottom = center + half
        return left, top, right, bottom

    def _hit_crop_edge(self, x: float, y: float) -> str | None:
        """Return the crop edge under the cursor, if any."""
        left, top, right, bottom = self._normalized_crop_box()
        near_left = abs(x - left) <= CROP_EDGE_HIT_PX and top - CROP_EDGE_HIT_PX <= y <= bottom + CROP_EDGE_HIT_PX
        near_right = abs(x - right) <= CROP_EDGE_HIT_PX and top - CROP_EDGE_HIT_PX <= y <= bottom + CROP_EDGE_HIT_PX
        near_top = abs(y - top) <= CROP_EDGE_HIT_PX and left - CROP_EDGE_HIT_PX <= x <= right + CROP_EDGE_HIT_PX
        near_bottom = abs(y - bottom) <= CROP_EDGE_HIT_PX and left - CROP_EDGE_HIT_PX <= x <= right + CROP_EDGE_HIT_PX
        distances = [
            ("left", abs(x - left), near_left),
            ("right", abs(x - right), near_right),
            ("top", abs(y - top), near_top),
            ("bottom", abs(y - bottom), near_bottom),
        ]
        hits = [(name, dist) for name, dist, active in distances if active]
        if not hits:
            return None
        return min(hits, key=lambda item: item[1])[0]

    def _update_crop_edge(self, edge: str, x: float, y: float) -> None:
        """Resize the setup crop box by dragging one edge."""
        left, top, right, bottom = self._drag_start_crop or self._normalized_crop_box()
        if edge == "left":
            left = min(float(np.clip(x, 0.0, float(self.preview_width))), right - self._min_crop_width())
        elif edge == "right":
            right = max(float(np.clip(x, 0.0, float(self.preview_width))), left + self._min_crop_width())
        elif edge == "top":
            top = min(float(np.clip(y, 0.0, float(self.preview_height))), bottom - self._min_crop_height())
        elif edge == "bottom":
            bottom = max(float(np.clip(y, 0.0, float(self.preview_height))), top + self._min_crop_height())
        self.panorama_crop_box = (
            float(np.clip(left, 0.0, float(self.preview_width))),
            float(np.clip(top, 0.0, float(self.preview_height))),
            float(np.clip(right, 0.0, float(self.preview_width))),
            float(np.clip(bottom, 0.0, float(self.preview_height))),
        )

    def _update_panorama_center(self, x: float, y: float) -> None:
        """Pan the panorama setup center based on image-content dragging."""
        if self._drag_start_xy is None or self._drag_start_center is None:
            return
        start_x, start_y = self._drag_start_xy
        start_lam, start_phi = self._drag_start_center
        dx = x - start_x
        dy = y - start_y
        self.panorama_center_lam = _wrap_angle(start_lam - (dx / max(float(self.preview_width - 1), 1.0)) * 2.0 * math.pi)
        self.panorama_center_phi = float(
            np.clip(
                start_phi - (dy / max(float(self.preview_height - 1), 1.0)) * math.pi,
                -PANORAMA_MAX_ABS_PITCH,
                PANORAMA_MAX_ABS_PITCH,
            )
        )
        self._render_panorama_setup_image()

    def _annotation_points_to_original(self, points: np.ndarray) -> np.ndarray:
        """Map annotation-preview coordinates to active original image coordinates."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2); got {arr.shape}.")
        return np.asarray(
            [self.session.active_view.preview_to_image_xy(float(x), float(y)) for x, y in arr],
            dtype=np.float32,
        )

    def _original_points_to_annotation(self, points: np.ndarray) -> np.ndarray:
        """Map active original image coordinates to annotation-preview coordinates."""
        arr = np.asarray(points, dtype=np.float64)
        return self.session.active_view.image_to_preview_points(arr)

    def _on_click(self, event: object) -> None:
        """Handle mouse press by state."""
        if self.state == STATE_ANNOTATE and self.pending is not None:
            self.status_extra = "Choose Redraw or Next before drawing another stroke."
            self._refresh()
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        if getattr(event, "button", None) != 1:
            return

        if self.state == STATE_PANORAMA_SETUP:
            x, y = self._clip_setup_xy(float(event.xdata), float(event.ydata))
            edge = self._hit_crop_edge(x, y)
            self._drag_action = f"edge:{edge}" if edge is not None else "pan"
            self._drag_start_xy = (x, y)
            self._drag_start_center = (self.panorama_center_lam, self.panorama_center_phi)
            self._drag_start_crop = self._normalized_crop_box()
            return

        if self.state == STATE_ANNOTATE:
            x, y = self._clip_annotation_xy(float(event.xdata), float(event.ydata))
            self.current_stroke = [(x, y)]
            self._drawing_stroke = True
            self.status_extra = "Drawing rough stroke."
            self._refresh()

    def _on_motion(self, event: object) -> None:
        """Handle mouse drag by state."""
        if self.state == STATE_ANNOTATE and self.pending is not None:
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return

        if self.state == STATE_PANORAMA_SETUP and self._drag_action is not None:
            x, y = self._clip_setup_xy(float(event.xdata), float(event.ydata))
            if self._drag_action == "pan":
                self._update_panorama_center(x, y)
            elif self._drag_action.startswith("edge:"):
                self._update_crop_edge(self._drag_action.split(":", 1)[1], x, y)
            self._refresh()
            return

        if self.state == STATE_ANNOTATE and self._drawing_stroke:
            x, y = self._clip_annotation_xy(float(event.xdata), float(event.ydata))
            if not self.current_stroke:
                self.current_stroke = [(x, y)]
            else:
                last_x, last_y = self.current_stroke[-1]
                if math.hypot(x - last_x, y - last_y) >= STROKE_MIN_SPACING_PX:
                    self.current_stroke.append((x, y))
            self._refresh()

    def _on_release(self, event: object) -> None:
        """Handle mouse release by state."""
        if self.state == STATE_PANORAMA_SETUP and self._drag_action is not None:
            self._drag_action = None
            self._drag_start_xy = None
            self._drag_start_center = None
            self._drag_start_crop = None
            self._refresh()
            return

        if self.state != STATE_ANNOTATE or not self._drawing_stroke or self.pending is not None:
            return
        self._drawing_stroke = False
        if (
            getattr(event, "inaxes", None) is self.ax
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x, y = self._clip_annotation_xy(float(event.xdata), float(event.ydata))
            if self.current_stroke:
                last_x, last_y = self.current_stroke[-1]
                if math.hypot(x - last_x, y - last_y) > 0.0:
                    self.current_stroke.append((x, y))
        if len(self.current_stroke) < 2:
            self.status_extra = "Stroke is too short. Draw again."
            self.current_stroke = []
            self._refresh()
            return
        stroke = np.asarray(self.current_stroke, dtype=np.float32)
        self.current_stroke = []
        self._set_pending_from_annotation_stroke(stroke)

    def _set_pending_from_annotation_stroke(self, stroke: np.ndarray) -> None:
        """Run the 2D snapper on the original-resolution annotation image."""
        image = self._require_annotation_original_image()
        stroke_original = self._annotation_points_to_original(stroke)
        try:
            snapped = snap_annotation(
                image,
                stroke_original,
                camera_type="pinhole",
                mode=self.snap_mode,
                config=self.snap_config,
            )
            result_points = snapped.points.astype(np.float32, copy=True)
            source_points = snapped.source_stroke.astype(np.float32, copy=True)
            preview_points = self._original_points_to_annotation(result_points)
            source_preview_points = self._original_points_to_annotation(source_points)
            debug = dict(snapped.debug)
            debug.update(
                {
                    "annotation_camera_type": self.camera_type,
                    "annotation_points": preview_points,
                    "annotation_source_stroke": source_preview_points,
                    "annotation_original_points": result_points,
                    "annotation_original_source_stroke": source_points,
                    "panorama_center_lam": self.annotation_center_lam,
                    "panorama_center_phi": self.annotation_center_phi,
                    "panorama_crop_box": self.annotation_crop_box,
                }
            )
            result = SnapResult(
                points=result_points.astype(np.float32),
                source_stroke=source_points.astype(np.float32),
                mode=self.snap_mode,
                camera_type=str(self.camera_type),
                confidence=snapped.confidence,
                debug=debug,
            )
            self.pending = AnnotationItem(
                result=result,
                preview_points=preview_points.astype(np.float32, copy=True),
                source_preview_points=source_preview_points.astype(np.float32, copy=True),
                source_original_points=source_points.astype(np.float32, copy=True),
                camera_type=str(self.camera_type),
                snap_mode=self.snap_mode,
            )
            self.status_extra = (
                f"Snapped {self.snap_mode} result. "
                f"Confidence={result.confidence:.3f}. Choose Redraw or Next."
            )
        except Exception as exc:
            self.pending = None
            self.status_extra = f"Snapping failed: {exc}. Draw again."
        self._refresh()

    def _resnap_pending(self) -> None:
        """Recompute the pending result after the annotation type changes."""
        if self.pending is None:
            return
        stroke = self.pending.source_preview_points.copy()
        self.pending = None
        self._set_pending_from_annotation_stroke(stroke)

    def _accept_pending(self) -> bool:
        """Move the pending result into the accepted annotation list."""
        if self.pending is None:
            return False
        self.annotations.append(self.pending)
        self.pending = None
        self.current_stroke = []
        self._drawing_stroke = False
        return True

    def _results_for_save(self) -> list[SnapResult]:
        """Return accepted snapped results for export."""
        return [item.result for item in self.annotations]

    def export_lines(self, output_path: str | Path) -> list[dict[str, list[list[float]]]]:
        """Return MaDCoW ``lines[]`` entries for accepted snapped results."""
        if self.pending is not None:
            self._accept_pending()
        results = self._results_for_save()
        if not results:
            return []
        payload = result_to_madcow_annotation_dict(
            results,
            self.session.image_path,
            str(output_path),
            image_shape=(self.session.height, self.session.width),
            camera_type=str(self.camera_type),
            fov_deg=self.fov_deg,
            source_image_path=self.session.source_image_path if self.session.view_metadata is not None else None,
            view_metadata=self.session.view_metadata,
        )
        return list(payload["lines"])

    def _save(self) -> bool:
        """Save accepted annotations, accepting the pending result first if needed."""
        if self.state != STATE_ANNOTATE:
            self.status_extra = "Enter annotation mode before saving."
            return False
        if self.pending is not None:
            self._accept_pending()

        results = self._results_for_save()
        if not results:
            if not self.allow_empty_annotations:
                self.status_extra = "Nothing to save. Draw at least one annotation first."
                return False
            madcow_status = "MaDCoW JSON written"
            camera_model = "panorama_view" if str(self.camera_type) == "panorama" else str(self.camera_type)
            try:
                payload = build_annotation_payload(
                    image_path=self.session.image_path,
                    output_path=self.json_path,
                    source_image_path=self.session.source_image_path if self.session.view_metadata is not None else None,
                    camera_model=camera_model,
                    fov_deg=self.fov_deg if camera_model == "pinhole" else None,
                    view_metadata=self.session.view_metadata,
                    lines=[],
                    regions=[],
                )
                write_annotation_json(self.json_path, payload)
            except Exception as exc:
                madcow_status = f"MaDCoW JSON not written: {exc}"
            self.status_extra = f"Saved 0 annotation(s). {madcow_status}: {self.json_path}."
            return True

        madcow_status = "MaDCoW JSON written"
        try:
            save_madcow_annotation_json(
                results,
                self.session.image_path,
                str(self.json_path),
                image_shape=(self.session.height, self.session.width),
                camera_type=str(self.camera_type),
                fov_deg=self.fov_deg,
                source_image_path=self.session.source_image_path if self.session.view_metadata is not None else None,
                view_metadata=self.session.view_metadata,
            )
        except Exception as exc:
            madcow_status = f"MaDCoW JSON not written: {exc}"

        self.status_extra = f"Saved {len(results)} annotation(s). {madcow_status}: {self.json_path}."
        return True

    def _refresh(self) -> None:
        """Redraw the current GUI state."""
        self._clear_dynamic_artists()
        if self.state == STATE_CAMERA_SELECT:
            self.fig.canvas.draw_idle()
            return

        if self.state == STATE_PINHOLE_SETUP:
            self._set_image(self.preview_image)
            fov = float(self.fov_deg or FALLBACK_FOV_DEG)
            self.status.set_text(f"Camera: pinhole | FOV: {fov:.2f} deg | {self.status_extra}")
            self.help_text.set_text("Type horizontal FOV, then press Done.")
            self.fig.canvas.draw_idle()
            return

        if self.state == STATE_PANORAMA_SETUP:
            if self.panorama_setup_image is None:
                self._render_panorama_setup_image()
            if self.panorama_setup_image is None:
                raise RuntimeError("Panorama setup image is not initialized.")
            self._set_image(self.panorama_setup_image)
            self._draw_crop_overlay()
            center_deg = (math.degrees(self.panorama_center_lam), math.degrees(self.panorama_center_phi))
            self.status.set_text(
                f"Camera: panorama | Center: ({center_deg[0]:.1f}, {center_deg[1]:.1f}) | {self.status_extra}"
            )
            self.help_text.set_text(
                "Drag image content to move view center. Drag crop edges to resize. H/V Reset recalibrates center."
            )
            self.fig.canvas.draw_idle()
            return

        image = self._require_annotation_image()
        self._set_image(image)
        for idx, item in enumerate(self.annotations, start=1):
            points = item.preview_points
            line_artist = self.ax.plot(points[:, 0], points[:, 1], color="#00cc66", linewidth=2.2, zorder=4)[0]
            self._dynamic_artists.append(line_artist)
            label = self.ax.text(
                points[-1, 0],
                points[-1, 1],
                f" {idx}",
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
                zorder=5,
            )
            self._dynamic_artists.append(label)

        if self.pending is not None:
            rough = self.pending.source_preview_points
            pending_points = self.pending.preview_points
            rough_artist = self.ax.plot(
                rough[:, 0],
                rough[:, 1],
                color="#ffcc00",
                linewidth=1.5,
                linestyle="--",
                zorder=5,
            )[0]
            result_artist = self.ax.plot(
                pending_points[:, 0],
                pending_points[:, 1],
                color="#ff3333",
                linewidth=2.8,
                zorder=6,
            )[0]
            self._dynamic_artists.extend([rough_artist, result_artist])

        if self.current_stroke:
            stroke = np.asarray(self.current_stroke, dtype=np.float32)
            stroke_artist = self.ax.plot(
                stroke[:, 0],
                stroke[:, 1],
                color="#ffcc00",
                linewidth=2.0,
                marker="o",
                markersize=3,
                zorder=6,
            )[0]
            self._dynamic_artists.append(stroke_artist)

        fov_text = f", fov={self.fov_deg:.2f}" if self.camera_type == "pinhole" and self.fov_deg is not None else ""
        size_text = f"source preview {self.preview_width}x{self.preview_height}"
        if self.is_downsampled:
            size_text += f", source original {self.original_width}x{self.original_height}"
        image_h, image_w = image.shape[:2]
        original_h, original_w = self._require_annotation_original_image().shape[:2]
        pending_text = "yes" if self.pending is not None else "no"
        self.status.set_text(
            f"Camera: {self.camera_type}{fov_text} | Type: {self.snap_mode} | "
            f"Accepted: {len(self.annotations)} | Pending: {pending_text} | "
            f"annotation preview {image_w}x{image_h}, original {original_w}x{original_h} | "
            f"{size_text} | {self.status_extra}"
        )
        self.help_text.set_text(
            "Left-drag: draw rough stroke | Red: pending snapped result | Green: accepted annotations"
        )
        self.fig.canvas.draw_idle()

    def _draw_crop_overlay(self) -> None:
        """Draw dimmed outside-crop overlay and crop edge lines."""
        from matplotlib.patches import Rectangle

        left, top, right, bottom = self._normalized_crop_box()
        width = float(self.preview_width)
        height = float(self.preview_height)
        overlays = [
            (0.0, 0.0, width, top),
            (0.0, bottom, width, height - bottom),
            (0.0, top, left, bottom - top),
            (right, top, width - right, bottom - top),
        ]
        for x, y, w, h in overlays:
            if w <= 0.0 or h <= 0.0:
                continue
            rect = Rectangle((x, y), w, h, facecolor="black", alpha=0.45, edgecolor="none", zorder=4)
            self.ax.add_patch(rect)
            self._dynamic_artists.append(rect)
        crop_rect = Rectangle(
            (left, top),
            right - left,
            bottom - top,
            fill=False,
            edgecolor="#ffcc00",
            linewidth=1.2,
            zorder=5,
        )
        self.ax.add_patch(crop_rect)
        self._dynamic_artists.append(crop_rect)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the simplified annotation tool."""
    parser = argparse.ArgumentParser(description="Minimal 2D interactive snapping annotation GUI.")
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help=f"Path to the input image. Defaults to {DEFAULT_IMAGE}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save the MaDCoW annotation JSON. Defaults to the input image directory.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional snap parameter JSON path. Defaults to interactive_snapping_2d/config/snap_config.json.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    fov, _source = estimate_fov_from_exif(str(image_path))
    output_dir = args.output_dir or str(image_path.parent)
    snap_config = load_snap_config(args.config_json)
    LineAidAnnotationGUI(str(image_path), fov, output_dir, snap_config=snap_config).run()


if __name__ == "__main__":
    main()
