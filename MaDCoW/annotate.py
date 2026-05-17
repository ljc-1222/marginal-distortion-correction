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

import piexif
from .src import CameraConfig
from .src.camera import Camera


DEFAULT_IMAGE = Path(__file__).resolve().parent / "data" / "test.jpg"
FALLBACK_FOV_DEG = 90.0
PREVIEW_MAX_SIDE = 1200
LINE_RESAMPLE_POINTS = 128
LINE_MIN_SPACING_PX = 3.0
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
        * ``l`` / ``r``: switch line / region mode.
        * Left drag in line mode: trace a straight-structure curve.
        * Left drag in region mode: paint current ROI mask.
        * Right drag in region mode: erase current ROI mask.
        * ``+`` / ``-``: change brush radius.
        * ``n``: create a new ROI.
        * ``[`` / ``]``: previous / next ROI.
        * ``u``: undo last line or mask stroke.
        * ``c``: clear current ROI.
        * ``s``: save annotations.
        * ``escape``: cancel the current line or stroke.
    """

    def __init__(
        self,
        image_path: str,
        fov_deg: float,
        output_dir: str,
    ) -> None:
        """Initialize the GUI state.

        Args:
            image_path: Path to the input image.
            fov_deg: Horizontal field of view of the input image, in degrees.
            output_dir: Directory where the JSON file and mask PNGs are saved.
            preview_max_side: Maximum side length used for the interactive preview.
        """
        if fov_deg <= 0 or fov_deg >= 180:
            raise ValueError(f"fov_deg must lie in (0, 180); got {fov_deg}.")

        self.image_path = str(Path(image_path).resolve())
        self.fov_deg = float(fov_deg)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preview_max_side = PREVIEW_MAX_SIDE
        with Image.open(self.image_path) as img:
            original_image = img.convert("RGB")
            self.original_width, self.original_height = original_image.size
            preview_size = _compute_preview_size(
                self.original_width,
                self.original_height,
                self.preview_max_side,
            )
            if preview_size == original_image.size:
                preview_image = original_image
            else:
                preview_image = original_image.resize(preview_size, _lanczos_resampling())
            self.image = np.array(preview_image)

        self.height, self.width = self.image.shape[:2]
        self.preview_width = self.width
        self.preview_height = self.height
        self.is_downsampled = (
            self.preview_width != self.original_width
            or self.preview_height != self.original_height
        )
        self.camera = Camera(
            CameraConfig(
                fov_deg=self.fov_deg,
                width=self.original_width,
                height=self.original_height,
            )
        )

        self.mode = "line"
        self.brush_radius = max(4, int(round(min(self.height, self.width) * 0.02)))
        self.lines: list[list[tuple[float, float]]] = []
        self.current_line_points: list[tuple[float, float]] = []
        self._drawing_line = False
        self.regions: list[dict[str, Any]] = []
        self.current_region = 0
        self.history: list[tuple[str, Any]] = []
        self._drawing = False
        self._stroke_value = True
        self._stroke_snapshot: np.ndarray | None = None
        self._syncing_name = False
        self._dynamic_artists: list[Any] = []

        self._add_region("region_1")
        self.json_path = self.output_dir / f"{Path(self.image_path).stem}.json"
        self._build_figure()
        self._refresh()

    def run(self) -> None:
        """Open the matplotlib window and block until the user exits."""
        import matplotlib.pyplot as plt

        plt.show()

    def _build_figure(self) -> None:
        """Create the Matplotlib figure, artists, widgets, and callbacks."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("MaDCoW Annotation Tool")
        self.fig.subplots_adjust(left=0.03, right=0.99, bottom=0.18, top=0.94)
        self.ax.imshow(self.image)
        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(self.height - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.mask_artist = self.ax.imshow(np.zeros((self.height, self.width, 4)), interpolation="nearest")
        self.status = self.fig.text(0.03, 0.965, "", ha="left", va="center", fontsize=10)

        self.widgets: list[Any] = []
        buttons = [
            ("Line", 0.03, self._button_line),
            ("Region", 0.105, self._button_region),
            ("New ROI", 0.195, self._button_new_region),
            ("Prev", 0.295, self._button_prev_region),
            ("Next", 0.36, self._button_next_region),
            ("Brush -", 0.425, self._button_brush_down),
            ("Brush +", 0.51, self._button_brush_up),
            ("Undo", 0.595, self._button_undo),
            ("Clear", 0.67, self._button_clear_region),
            ("Save", 0.745, self._button_save),
            ("Save+Close", 0.82, self._button_save_close),
        ]
        for label, x0, callback in buttons:
            ax_button = self.fig.add_axes([x0, 0.045, 0.07, 0.045])
            button = Button(ax_button, label)
            button.on_clicked(callback)
            self.widgets.append(button)

        name_ax = self.fig.add_axes([0.12, 0.105, 0.28, 0.045])
        self.name_box = TextBox(name_ax, "ROI name", initial=self._current_region_data()["name"])
        self.name_box.on_submit(self._on_name_submit)
        self.widgets.append(self.name_box)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_draw)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _add_region(self, name: str | None = None) -> None:
        """Create a new empty ROI mask and select it."""
        idx = len(self.regions) + 1
        region_name = name or f"region_{idx}"
        self.regions.append(
            {
                "name": region_name,
                "mask": np.zeros((self.height, self.width), dtype=bool),
            }
        )
        self.current_region = len(self.regions) - 1

    def _current_region_data(self) -> dict[str, Any]:
        """Return the selected ROI data dictionary."""
        return self.regions[self.current_region]

    def _set_mode(self, mode: str) -> None:
        """Switch interaction mode."""
        self.mode = mode
        if mode != "line":
            self._drawing_line = False
            self.current_line_points = []
        self._refresh()

    def _sync_name_box(self) -> None:
        """Synchronize the ROI name text box with the selected region."""
        if not hasattr(self, "name_box"):
            return
        self._syncing_name = True
        self.name_box.set_val(self._current_region_data()["name"])
        self._syncing_name = False

    def _on_name_submit(self, text: str) -> None:
        """Update the current ROI name from the text box."""
        if self._syncing_name:
            return
        name = text.strip() or f"region_{self.current_region + 1}"
        self._current_region_data()["name"] = name
        self._refresh()

    def _button_line(self, _event: object) -> None:
        self._set_mode("line")

    def _button_region(self, _event: object) -> None:
        self._set_mode("region")

    def _button_new_region(self, _event: object) -> None:
        self._add_region()
        self._sync_name_box()
        self._set_mode("region")

    def _button_prev_region(self, _event: object) -> None:
        self.current_region = (self.current_region - 1) % len(self.regions)
        self._sync_name_box()
        self._set_mode("region")

    def _button_next_region(self, _event: object) -> None:
        self.current_region = (self.current_region + 1) % len(self.regions)
        self._sync_name_box()
        self._set_mode("region")

    def _button_brush_down(self, _event: object) -> None:
        self.brush_radius = max(1, self.brush_radius - 2)
        self._refresh()

    def _button_brush_up(self, _event: object) -> None:
        self.brush_radius = min(max(self.height, self.width), self.brush_radius + 2)
        self._refresh()

    def _button_undo(self, _event: object) -> None:
        self._undo()

    def _button_clear_region(self, _event: object) -> None:
        self._clear_current_region()

    def _button_save(self, _event: object) -> None:
        self._save(str(self.json_path))
        self._refresh()

    def _button_save_close(self, _event: object) -> None:
        self._save(str(self.json_path))
        import matplotlib.pyplot as plt

        plt.close(self.fig)

    def _clip_xy(self, x: float, y: float) -> tuple[float, float]:
        """Clip floating preview coordinates into the preview image extent."""
        x_clipped = float(np.clip(x, 0.0, self.width - 1.0))
        y_clipped = float(np.clip(y, 0.0, self.height - 1.0))
        return x_clipped, y_clipped

    def _preview_to_original_xy(self, x: float, y: float) -> tuple[float, float]:
        """Map preview pixel coordinates back to original image coordinates."""
        if self.width == self.original_width and self.height == self.original_height:
            x_original = x
            y_original = y
        else:
            x_original = (x + 0.5) * (self.original_width / self.width) - 0.5
            y_original = (y + 0.5) * (self.original_height / self.height) - 0.5

        x_original = float(np.clip(x_original, 0.0, self.original_width - 1.0))
        y_original = float(np.clip(y_original, 0.0, self.original_height - 1.0))
        return x_original, y_original

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

    def _on_click(self, event: object) -> None:
        """Matplotlib mouse click handler.

        Args:
            event: A ``matplotlib.backend_bases.MouseEvent``.
        """
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return

        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        button = getattr(event, "button", None)
        if self.mode == "line":
            if button == 1:
                self._drawing_line = True
                self.current_line_points = [(x, y)]
            elif button == 3:
                self._drawing_line = False
                self.current_line_points = []
            self._refresh()
            return

        if self.mode == "region" and button in (1, 3):
            self._drawing = True
            self._stroke_value = button == 1
            self._stroke_snapshot = self._current_region_data()["mask"].copy()
            self._paint_at(x, y, self._stroke_value)
            self._refresh()

    def _on_draw(self, event: object) -> None:
        """Matplotlib mouse drag handler for line tracing and region painting.

        Args:
            event: A ``matplotlib.backend_bases.MouseEvent``.
        """
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return

        if self.mode == "line" and self._drawing_line:
            x, y = self._clip_xy(float(event.xdata), float(event.ydata))
            if not self.current_line_points:
                self.current_line_points = [(x, y)]
            else:
                last_x, last_y = self.current_line_points[-1]
                if math.hypot(x - last_x, y - last_y) >= LINE_MIN_SPACING_PX:
                    self.current_line_points.append((x, y))
            self._refresh()
            return

        if not self._drawing or self.mode != "region":
            return

        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        self._paint_at(x, y, self._stroke_value)
        self._refresh()

    def _on_release(self, event: object) -> None:
        """Finish a line or paint stroke and store it in undo history."""
        if self.mode == "line" and self._drawing_line:
            if (
                getattr(event, "inaxes", None) is self.ax
                and getattr(event, "xdata", None) is not None
                and getattr(event, "ydata", None) is not None
            ):
                x, y = self._clip_xy(float(event.xdata), float(event.ydata))
                if self.current_line_points:
                    last_x, last_y = self.current_line_points[-1]
                    if math.hypot(x - last_x, y - last_y) > 0.0:
                        self.current_line_points.append((x, y))

            self._drawing_line = False
            if (
                len(self.current_line_points) >= 2
                and _polyline_length(self.current_line_points) >= LINE_MIN_SPACING_PX
            ):
                resampled = _resample_polyline(self.current_line_points, LINE_RESAMPLE_POINTS)
                self.lines.append(resampled)
                self.history.append(("line", None))
            self.current_line_points = []
            self._refresh()
            return

        if not self._drawing:
            return
        self._drawing = False
        mask = self._current_region_data()["mask"]
        if self._stroke_snapshot is not None and not np.array_equal(mask, self._stroke_snapshot):
            self.history.append(("mask", self.current_region, self._stroke_snapshot))
        self._stroke_snapshot = None
        self._refresh()

    def _on_key(self, event: object) -> None:
        """Keyboard shortcut handler."""
        key = (getattr(event, "key", "") or "").lower()
        if key in ("l",):
            self._set_mode("line")
        elif key in ("r",):
            self._set_mode("region")
        elif key in ("n",):
            self._button_new_region(event)
        elif key in ("[",):
            self._button_prev_region(event)
        elif key in ("]", "tab"):
            self._button_next_region(event)
        elif key in ("+", "="):
            self._button_brush_up(event)
        elif key in ("-", "_"):
            self._button_brush_down(event)
        elif key in ("u", "ctrl+z", "cmd+z"):
            self._undo()
        elif key in ("c",):
            self._clear_current_region()
        elif key in ("s", "ctrl+s", "cmd+s"):
            self._save(str(self.json_path))
            self._refresh()
        elif key == "escape":
            self._drawing_line = False
            self.current_line_points = []
            self._drawing = False
            self._stroke_snapshot = None
            self._refresh()

    def _paint_at(self, x: float, y: float, value: bool) -> None:
        """Paint or erase the current ROI mask around one image point."""
        mask = self._current_region_data()["mask"]
        cx = int(round(x))
        cy = int(round(y))
        radius = int(self.brush_radius)
        x0 = max(0, cx - radius)
        x1 = min(self.width - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(self.height - 1, cy + radius)
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        disk = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= radius * radius
        patch = mask[y0 : y1 + 1, x0 : x1 + 1]
        patch[disk] = value

    def _clear_current_region(self) -> None:
        """Clear the selected ROI mask while preserving undo history."""
        mask = self._current_region_data()["mask"]
        before = mask.copy()
        if bool(mask.any()):
            mask[:] = False
            self.history.append(("mask", self.current_region, before))
        self._refresh()

    def _undo(self) -> None:
        """Undo the most recent line or mask stroke."""
        if not self.history:
            return
        action = self.history.pop()
        if action[0] == "line" and self.lines:
            self.lines.pop()
        elif action[0] == "mask":
            _, region_idx, previous = action
            if 0 <= region_idx < len(self.regions):
                self.regions[region_idx]["mask"] = previous.copy()
                self.current_region = region_idx
                self._sync_name_box()
        self._refresh()

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
            color = colors[idx % len(colors)]
            alpha = 0.40 if idx == self.current_region else 0.22
            overlay[mask, :3] = color
            overlay[mask, 3] = alpha
        return overlay

    def _refresh(self) -> None:
        """Redraw overlays, lines, and status text."""
        self.mask_artist.set_data(self._mask_overlay())
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists.clear()

        for idx, points in enumerate(self.lines, start=1):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            line_artist = self.ax.plot(
                xs,
                ys,
                color="lime",
                linewidth=2.0,
            )[0]
            label_x, label_y = points[-1]
            label = self.ax.text(
                label_x,
                label_y,
                f" {idx}",
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
            )
            self._dynamic_artists.extend((line_artist, label))

        if self.current_line_points:
            xs = [p[0] for p in self.current_line_points]
            ys = [p[1] for p in self.current_line_points]
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
        self.status.set_text(
            f"Mode: {self.mode.upper()} | ROI {self.current_region + 1}/{len(self.regions)} "
            f"'{current['name']}' ({pixels} preview px) | Lines: {len(self.lines)} | "
            f"Brush: {self.brush_radius}px | {size_text} | Save: {self.json_path}"
        )
        self.fig.canvas.draw_idle()

    def _line_json(self) -> list[dict[str, list[list[float]]]]:
        """Convert preview-space line curves into original view-sphere directions."""
        result: list[dict[str, list[list[float]]]] = []
        for points in self.lines:
            if len(points) != LINE_RESAMPLE_POINTS:
                raise ValueError(
                    f"Saved lines must contain exactly {LINE_RESAMPLE_POINTS} points; "
                    f"got {len(points)}."
                )

            original_points = [self._preview_to_original_xy(x, y) for x, y in points]
            xs = np.array([p[0] for p in original_points], dtype=np.float64)
            ys = np.array([p[1] for p in original_points], dtype=np.float64)
            lam, phi = self.camera.pixel_to_direction(xs, ys)
            points_dir = [
                [float(lam_i), float(phi_i)]
                for lam_i, phi_i in zip(lam, phi)
            ]
            result.append({"points_dir": points_dir})
        return result

    def _save(self, json_path: str) -> None:
        """Write masks to disk and emit the annotation JSON.

        Args:
            json_path: Output JSON path.
        """
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
            original_mask = self._mask_to_original_size(mask)
            Image.fromarray((original_mask.astype(np.uint8) * 255)).save(mask_path)
            regions_json.append(
                {
                    "name": str(region["name"]),
                    "mask_path": _relative_path(mask_path, json_out.parent),
                }
            )

        payload = {
            "image_path": _relative_path(Path(self.image_path), json_out.parent),
            "fov_deg": self.fov_deg,
            "lines": self._line_json(),
            "regions": regions_json,
        }
        json_out.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the annotation tool.

    Returns:
        Namespace with attributes: ``image`` (str), ``fov`` (float | None),
        ``output_dir`` (str | None).
    """
    parser = argparse.ArgumentParser(description="MaDCoW annotation GUI.")
    parser.add_argument(
        "--image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help=f"Path to the input image. Defaults to {DEFAULT_IMAGE}.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Horizontal FOV in degrees. If omitted, EXIF is used when possible, otherwise 90.",
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
    if args.fov is None:
        fov, source = estimate_fov_from_exif(str(image_path))
        print(f"Using horizontal FOV {fov:.2f} degrees ({source}).")
    else:
        fov = args.fov
    output_dir = args.output_dir or str(image_path.parent)
    AnnotationGUI(str(image_path), fov, output_dir).run()
