"""Minimal Matplotlib GUI for 2D interactive snapping annotations."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image

from .snap2d import SnapConfig, SnapResult, snap_annotation
from .snap2d.io import save_madcow_annotation_json


DEFAULT_IMAGE = Path(__file__).resolve().parent / "data" / "test_1.jpg"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FALLBACK_FOV_DEG = 90.0
PREVIEW_MAX_SIDE = 1200
STROKE_MIN_SPACING_PX = 3.0
CAMERA_TYPE_CHOICES = ("pinhole", "panorama")
SNAP_MODE_CHOICES = ("line", "curve")

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
    if camera_type == "360":
        camera_type = "panorama"
    if camera_type not in CAMERA_TYPE_CHOICES:
        raise ValueError(f"camera_type must be one of {CAMERA_TYPE_CHOICES}; got {camera_type!r}.")
    return camera_type


def _draw_polyline(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    """Draw one polyline on a BGR image."""
    if len(points) < 2:
        return
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], False, color, thickness, lineType=cv2.LINE_AA)


@dataclass
class AnnotationItem:
    """One annotation stored by the simplified GUI."""

    result: SnapResult
    preview_points: np.ndarray
    source_preview_points: np.ndarray
    source_original_points: np.ndarray
    camera_type: str
    snap_mode: str


class LineAidAnnotationGUI:
    """Minimal annotation GUI with draw, redraw, next, save, and save+close actions."""

    def __init__(
        self,
        image_path: str,
        fov_deg: float | None,
        output_dir: str,
        camera_type: str = "pinhole",
    ) -> None:
        self.image_path = str(Path(image_path).resolve())
        self.camera_type = _normalize_camera_type(camera_type)
        self.fov_deg = None if fov_deg is None else float(fov_deg)
        if self.camera_type == "pinhole" and self.fov_deg is None:
            self.fov_deg = FALLBACK_FOV_DEG
        if self.fov_deg is not None and (self.fov_deg <= 0 or self.fov_deg >= 180):
            raise ValueError(f"fov_deg must lie in (0, 180); got {self.fov_deg}.")

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(self.image_path) as img:
            original_image = img.convert("RGB")
            self.original_width, self.original_height = original_image.size
            preview_size = _compute_preview_size(self.original_width, self.original_height, PREVIEW_MAX_SIDE)
            preview_image = (
                original_image
                if preview_size == original_image.size
                else original_image.resize(preview_size, _lanczos_resampling())
            )
            self.original_image = np.array(original_image)
            self.image = np.array(preview_image)

        self.height, self.width = self.image.shape[:2]
        self.is_downsampled = self.width != self.original_width or self.height != self.original_height
        self.snap_mode = "line"
        self.annotations: list[AnnotationItem] = []
        self.pending: AnnotationItem | None = None
        self.current_stroke: list[tuple[float, float]] = []
        self._drawing_stroke = False
        self._dynamic_artists: list[Any] = []
        self.status_extra = "Draw a rough stroke. Then choose Redraw or Next."

        stem = Path(self.image_path).stem
        self.json_path = self.output_dir / f"{stem}.json"
        self.snap_json_path = self.output_dir / f"{stem}_snapping_2d.json"
        self.preview_path = self.output_dir / f"{stem}_snapping_preview.png"

        self._build_figure()
        self._refresh()

    def run(self) -> None:
        """Open the Matplotlib GUI and block until it closes."""
        import matplotlib.pyplot as plt

        plt.show()

    def _build_figure(self) -> None:
        """Create the simplified figure, controls, and event callbacks."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("2D Interactive Snapping Annotation")
        self.fig.subplots_adjust(left=0.03, right=0.99, bottom=0.20, top=0.92)
        self.image_artist = self.ax.imshow(self.image)
        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(self.height - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.status = self.fig.text(0.03, 0.955, "", ha="left", va="center", fontsize=10)
        self.help_text = self.fig.text(
            0.03,
            0.925,
            "Left-drag: draw rough stroke | Red: current snapped result | Green: accepted annotations",
            ha="left",
            va="center",
            fontsize=9,
        )

        self.widgets: list[Any] = []

        camera_ax = self.fig.add_axes([0.04, 0.035, 0.16, 0.12])
        camera_ax.set_title("Camera", fontsize=9)
        camera_active = 0 if self.camera_type == "pinhole" else 1
        self.camera_radio = RadioButtons(camera_ax, CAMERA_TYPE_CHOICES, active=camera_active)
        self.camera_radio.on_clicked(self._on_camera_changed)
        self.widgets.append(self.camera_radio)

        mode_ax = self.fig.add_axes([0.24, 0.035, 0.14, 0.12])
        mode_ax.set_title("Type", fontsize=9)
        self.mode_radio = RadioButtons(mode_ax, SNAP_MODE_CHOICES, active=0)
        self.mode_radio.on_clicked(self._on_snap_mode_changed)
        self.widgets.append(self.mode_radio)

        button_specs = [
            ("Redraw", 0.46, self._button_redraw),
            ("Next", 0.56, self._button_next),
            ("Save", 0.66, self._button_save),
            ("Save + Close", 0.76, self._button_save_close),
        ]
        for label, x0, callback in button_specs:
            width = 0.09 if label != "Save + Close" else 0.14
            button_ax = self.fig.add_axes([x0, 0.075, width, 0.055])
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.widgets.append(button)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_camera_changed(self, label: str) -> None:
        """Set the global camera type and resnap the pending stroke if present."""
        self.camera_type = _normalize_camera_type(label)
        if self.camera_type == "pinhole" and self.fov_deg is None:
            self.fov_deg = FALLBACK_FOV_DEG
        if self.pending is not None:
            self._resnap_pending()
        else:
            self.status_extra = f"Camera set to {self.camera_type}."
            self._refresh()

    def _on_snap_mode_changed(self, label: str) -> None:
        """Set the annotation type and resnap the pending stroke if present."""
        if label not in SNAP_MODE_CHOICES:
            raise ValueError(f"mode must be one of {SNAP_MODE_CHOICES}; got {label!r}.")
        self.snap_mode = label
        if self.pending is not None:
            self._resnap_pending()
        else:
            self.status_extra = f"Annotation type set to {self.snap_mode}."
            self._refresh()

    def _button_redraw(self, _event: object) -> None:
        """Discard the pending result and allow the user to draw it again."""
        self.pending = None
        self.current_stroke = []
        self._drawing_stroke = False
        self.status_extra = "Pending result discarded. Draw again."
        self._refresh()

    def _button_next(self, _event: object) -> None:
        """Accept the pending result and prepare for the next annotation."""
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
            import matplotlib.pyplot as plt

            plt.close(self.fig)
        else:
            self._refresh()

    def _on_key(self, event: object) -> None:
        """Handle keyboard aliases for the visible actions only."""
        key = (getattr(event, "key", "") or "").lower()
        if key in ("r", "escape"):
            self._button_redraw(event)
        elif key in ("n", "enter", " "):
            self._button_next(event)
        elif key in ("s", "ctrl+s", "cmd+s"):
            self._button_save(event)

    def _clip_xy(self, x: float, y: float) -> tuple[float, float]:
        """Clip floating preview coordinates into the preview image extent."""
        return float(np.clip(x, 0.0, self.width - 1.0)), float(np.clip(y, 0.0, self.height - 1.0))

    def _preview_to_original_xy(self, x: float, y: float) -> tuple[float, float]:
        """Map preview coordinates back to original image coordinates."""
        if self.width == self.original_width and self.height == self.original_height:
            x_original = x
            y_original = y
        else:
            x_original = (x + 0.5) * (self.original_width / self.width) - 0.5
            y_original = (y + 0.5) * (self.original_height / self.height) - 0.5
        return (
            float(np.clip(x_original, 0.0, self.original_width - 1.0)),
            float(np.clip(y_original, 0.0, self.original_height - 1.0)),
        )

    def _preview_points_to_original(self, points: list[tuple[float, float]]) -> np.ndarray:
        """Map preview-space points to original-image coordinates."""
        return np.asarray([self._preview_to_original_xy(x, y) for x, y in points], dtype=np.float32)

    def _original_points_to_preview(self, points: np.ndarray) -> np.ndarray:
        """Map original-image coordinates to preview-space coordinates."""
        arr = np.asarray(points, dtype=np.float32)
        if self.width == self.original_width and self.height == self.original_height:
            return arr
        scale_x = self.width / self.original_width
        scale_y = self.height / self.original_height
        preview = arr.copy()
        preview[:, 0] = (preview[:, 0] + 0.5) * scale_x - 0.5
        preview[:, 1] = (preview[:, 1] + 0.5) * scale_y - 0.5
        return preview

    def _on_click(self, event: object) -> None:
        """Start a rough stroke when no pending result is waiting for review."""
        if self.pending is not None:
            self.status_extra = "Choose Redraw or Next before drawing another stroke."
            self._refresh()
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        if getattr(event, "button", None) != 1:
            return
        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        self.current_stroke = [(x, y)]
        self._drawing_stroke = True
        self.status_extra = "Drawing rough stroke."
        self._refresh()

    def _on_motion(self, event: object) -> None:
        """Track the rough stroke while dragging."""
        if not self._drawing_stroke or self.pending is not None:
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        if not self.current_stroke:
            self.current_stroke = [(x, y)]
        else:
            last_x, last_y = self.current_stroke[-1]
            if math.hypot(x - last_x, y - last_y) >= STROKE_MIN_SPACING_PX:
                self.current_stroke.append((x, y))
        self._refresh()

    def _on_release(self, event: object) -> None:
        """Finish the rough stroke and create a pending snapped result."""
        if not self._drawing_stroke or self.pending is not None:
            return
        self._drawing_stroke = False
        if (
            getattr(event, "inaxes", None) is self.ax
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x, y = self._clip_xy(float(event.xdata), float(event.ydata))
            if self.current_stroke:
                last_x, last_y = self.current_stroke[-1]
                if math.hypot(x - last_x, y - last_y) > 0.0:
                    self.current_stroke.append((x, y))
        if len(self.current_stroke) < 2:
            self.status_extra = "Stroke is too short. Draw again."
            self.current_stroke = []
            self._refresh()
            return

        stroke_original = self._preview_points_to_original(self.current_stroke)
        self.current_stroke = []
        self._set_pending_from_original_stroke(stroke_original)

    def _set_pending_from_original_stroke(self, stroke_original: np.ndarray) -> None:
        """Run the 2D snapper and store the result as pending."""
        try:
            result = snap_annotation(
                self.original_image,
                stroke_original,
                camera_type=self.camera_type,
                mode=self.snap_mode,
                config=SnapConfig(),
            )
            self.pending = AnnotationItem(
                result=result,
                preview_points=self._original_points_to_preview(result.points),
                source_preview_points=self._original_points_to_preview(stroke_original),
                source_original_points=stroke_original.astype(np.float32, copy=True),
                camera_type=self.camera_type,
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
        """Recompute the pending result after camera or type changes."""
        if self.pending is None:
            return
        stroke_original = self.pending.source_original_points.copy()
        self.pending = None
        self._set_pending_from_original_stroke(stroke_original)

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

    def _save(self) -> bool:
        """Save accepted annotations, accepting the pending result first if needed."""
        if self.pending is not None:
            self._accept_pending()

        results = self._results_for_save()
        if not results:
            self.status_extra = "Nothing to save. Draw at least one annotation first."
            return False

        self.snap_json_path.write_text(json.dumps(self._build_2d_payload(), indent=2), encoding="utf-8")
        self._save_preview(self.annotations, self.preview_path)

        madcow_status = "MaDCoW JSON written"
        try:
            save_madcow_annotation_json(
                results,
                self.image_path,
                str(self.json_path),
                image_shape=(self.original_height, self.original_width),
                camera_type=self.camera_type,
                fov_deg=self.fov_deg,
            )
        except Exception as exc:
            madcow_status = f"MaDCoW JSON not written: {exc}"

        self.status_extra = (
            f"Saved {len(results)} annotation(s). 2D JSON: {self.snap_json_path}; "
            f"preview: {self.preview_path}; {madcow_status}."
        )
        return True

    def _build_2d_payload(self) -> dict[str, Any]:
        """Build the authoritative 2D annotation payload."""
        return {
            "version": "0.2.0",
            "tool": "interactive_snapping_2d_simple",
            "image_path": self.image_path,
            "coordinate_space": "input_image_pixel",
            "image_shape": [int(self.original_height), int(self.original_width)],
            "camera_type": self.camera_type,
            "fov_deg": None if self.fov_deg is None else float(self.fov_deg),
            "annotations": [
                {
                    "id": f"anno_{idx:03d}",
                    "mode": item.snap_mode,
                    "camera_type": item.camera_type,
                    "points": item.result.points.astype(float).tolist(),
                    "source_stroke": item.result.source_stroke.astype(float).tolist(),
                    "confidence": float(item.result.confidence),
                }
                for idx, item in enumerate(self.annotations, start=1)
            ],
        }

    def _save_preview(self, items: list[AnnotationItem], path: Path) -> None:
        """Save an RGB overlay preview image."""
        preview = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
        for idx, item in enumerate(items, start=1):
            _draw_polyline(preview, item.result.source_stroke, (0, 255, 255), 2)
            _draw_polyline(preview, item.result.points, (0, 0, 255), 3)
            x, y = np.round(item.result.points[-1]).astype(int)
            label = f"{idx}:{item.snap_mode[0].upper()}"
            cv2.putText(preview, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(preview, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), preview)

    def _refresh(self) -> None:
        """Redraw accepted annotations, pending result, current stroke, and status."""
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists.clear()

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
        size_text = f"preview {self.width}x{self.height}"
        if self.is_downsampled:
            size_text += f", original {self.original_width}x{self.original_height}"
        pending_text = "yes" if self.pending is not None else "no"
        self.status.set_text(
            f"Camera: {self.camera_type}{fov_text} | Type: {self.snap_mode} | "
            f"Accepted: {len(self.annotations)} | Pending: {pending_text} | {size_text} | {self.status_extra}"
        )
        self.fig.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the simplified annotation tool."""
    parser = argparse.ArgumentParser(description="Minimal 2D interactive snapping annotation GUI.")
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help=f"Path to the input image. Defaults to {DEFAULT_IMAGE}.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Horizontal FOV in degrees for pinhole images. If omitted, EXIF is used when possible.",
    )
    parser.add_argument(
        "--camera-type",
        choices=("pinhole", "panorama", "360"),
        default="pinhole",
        help="Camera type for snapping and export.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save JSON and preview PNG files. Defaults to the input image directory.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    camera_type = _normalize_camera_type(args.camera_type)
    if camera_type == "pinhole" and args.fov is None:
        fov, source = estimate_fov_from_exif(str(image_path))
        print(f"Using horizontal FOV {fov:.2f} degrees ({source}).")
    else:
        fov = args.fov
        if camera_type == "panorama":
            print("Using panorama camera type.")

    output_dir = args.output_dir or str(image_path.parent)
    LineAidAnnotationGUI(str(image_path), fov, output_dir, camera_type=camera_type).run()


if __name__ == "__main__":
    main()
