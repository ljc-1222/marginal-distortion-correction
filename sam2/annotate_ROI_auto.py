"""SAM2-assisted ROI annotation GUI.

This tool creates MaDCoW-compatible ROI masks and annotation JSON files using
SAM2 image prompts. It is intentionally ROI-only: saved JSON files contain an
empty ``lines`` list and one PNG mask per saved ROI.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import ExifTags, Image


SAM2_ROOT = Path(__file__).resolve().parent
INNER_PACKAGE_DIR = SAM2_ROOT / "sam2"
DEFAULT_IMAGE = SAM2_ROOT / "data" / "test_1.png"
DEFAULT_CHECKPOINT = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
FALLBACK_FOV_DEG = 90.0
PREVIEW_MAX_SIDE = 1200

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "sam2_roi_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)


def _bootstrap_sam2_package() -> None:
    """Expose the official inner SAM2 package when this outer module runs."""
    if not INNER_PACKAGE_DIR.is_dir():
        raise RuntimeError(f"Cannot find inner SAM2 package at {INNER_PACKAGE_DIR}.")

    import sam2 as sam2_package

    package_paths = [str(INNER_PACKAGE_DIR), str(SAM2_ROOT)]
    existing_paths = [str(Path(path).resolve()) for path in getattr(sam2_package, "__path__", [])]
    for path in existing_paths:
        if path not in package_paths:
            package_paths.append(path)
    sam2_package.__path__ = package_paths

    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(config_dir=str(INNER_PACKAGE_DIR), version_base="1.2")


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
    """Estimate horizontal FOV from EXIF metadata when available."""
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

    focal = _ratio_to_float(lookup.get("FocalLength"))
    x_res = _ratio_to_float(lookup.get("FocalPlaneXResolution"))
    unit = lookup.get("FocalPlaneResolutionUnit")
    if focal and focal > 0 and x_res and x_res > 0 and unit in (2, 3, 4, 5):
        unit_to_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}
        sensor_width_mm = width / x_res * unit_to_mm[int(unit)]
        if sensor_width_mm > 0:
            return math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal))), (
                "EXIF focal length and focal-plane resolution"
            )

    return float(fallback), "fallback"


def _sanitize_name(name: str, fallback: str) -> str:
    """Create a filesystem-safe ROI name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return cleaned or fallback


def _relative_path(path: Path, base_dir: Path) -> str:
    """Return a path relative to ``base_dir`` when possible."""
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), base_dir.resolve())


def _lanczos_resampling() -> int:
    """Return Pillow's LANCZOS enum across Pillow versions."""
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.LANCZOS)
    return int(Image.LANCZOS)


def _nearest_resampling() -> int:
    """Return Pillow's NEAREST enum across Pillow versions."""
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.NEAREST)
    return int(Image.NEAREST)


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


def _normalize_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Return a box as ``(x0, y0, x1, y1)``."""
    x0, y0, x1, y1 = box
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _resolve_device(device: str) -> str:
    """Resolve the requested device string."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@contextlib.contextmanager
def _inference_context(device: str) -> Iterator[None]:
    """Run SAM2 prediction under inference mode and CUDA autocast when useful."""
    import torch

    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            yield


def _resolve_checkpoint(path: str) -> Path:
    """Resolve checkpoint paths relative to the current working directory."""
    checkpoint = Path(path).expanduser()
    if checkpoint.is_absolute():
        return checkpoint
    return (Path.cwd() / checkpoint).resolve()


def _normalize_model_cfg(model_cfg: str) -> str:
    """Convert an existing config path into a Hydra config name when possible."""
    cfg_path = Path(model_cfg).expanduser()
    if cfg_path.is_absolute():
        resolved = cfg_path.resolve()
    else:
        resolved = (Path.cwd() / cfg_path).resolve()
    if resolved.exists():
        try:
            return resolved.relative_to(INNER_PACKAGE_DIR.resolve()).as_posix()
        except ValueError:
            return model_cfg
    return model_cfg


def build_image_predictor(model_cfg: str, checkpoint: Path, device: str) -> Any:
    """Build the SAM2 image predictor."""
    _bootstrap_sam2_package()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(model_cfg, str(checkpoint), device=device)
    return SAM2ImagePredictor(sam2_model)


class SAM2ROIAnnotationGUI:
    """Interactive SAM2 ROI prompt GUI for one input image."""

    def __init__(
        self,
        image_path: str,
        fov_deg: float,
        output_dir: str,
        predictor: Any,
        device: str,
    ) -> None:
        if fov_deg <= 0 or fov_deg >= 180:
            raise ValueError(f"fov_deg must lie in (0, 180); got {fov_deg}.")

        self.image_path = str(Path(image_path).resolve())
        self.fov_deg = float(fov_deg)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = predictor
        self.device = device
        self.preview_max_side = PREVIEW_MAX_SIDE

        with Image.open(self.image_path) as img:
            original_image = img.convert("RGB")
            self.original_width, self.original_height = original_image.size
            preview_size = _compute_preview_size(
                self.original_width,
                self.original_height,
                self.preview_max_side,
            )
            preview_image = (
                original_image
                if preview_size == original_image.size
                else original_image.resize(preview_size, _lanczos_resampling())
            )
            self.original_image = np.array(original_image)
            self.image = np.array(preview_image)

        self.height, self.width = self.image.shape[:2]
        self.is_downsampled = self.width != self.original_width or self.height != self.original_height
        self.mode = "box"
        self.regions: list[dict[str, Any]] = []
        self.current_region = 0
        self.next_region_idx = 1
        self._dragging_box = False
        self._box_start: tuple[float, float] | None = None
        self._box_preview: tuple[float, float, float, float] | None = None
        self._dynamic_artists: list[Any] = []
        self._syncing_name = False
        self.message = ""
        self.json_path = self.output_dir / f"{Path(self.image_path).stem}.json"

        print("Computing SAM2 image embedding...")
        with _inference_context(self.device):
            self.predictor.set_image(self.original_image)
        print("Image embedding is ready.")

        self._add_region()
        self._build_figure()
        self._refresh()

    def run(self) -> None:
        """Open the Matplotlib GUI and block until it closes."""
        import matplotlib.pyplot as plt

        plt.show()

    def _build_figure(self) -> None:
        """Create the figure, widgets, and event callbacks."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("SAM2 ROI Annotation Tool")
        self.fig.subplots_adjust(left=0.03, right=0.99, bottom=0.20, top=0.94)
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
            ("Box", 0.075, self._button_box),
            ("Point", 0.085, self._button_point),
            ("New ROI", 0.095, self._button_new_region),
            ("Prev", 0.075, self._button_prev_region),
            ("Next", 0.075, self._button_next_region),
            ("Reset ROI", 0.105, self._button_reset_region),
            ("Save", 0.075, self._button_save),
            ("Save+Close", 0.12, self._button_save_close),
        ]
        gap = 0.018
        total_width = sum(width for _label, width, _callback in buttons) + gap * (len(buttons) - 1)
        x0 = (1.0 - total_width) * 0.5
        for label, width, callback in buttons:
            ax_button = self.fig.add_axes([x0, 0.045, width, 0.045])
            button = Button(ax_button, label)
            button.on_clicked(callback)
            self.widgets.append(button)
            x0 += width + gap

        name_ax = self.fig.add_axes([0.33, 0.11, 0.34, 0.045])
        self.name_box = TextBox(name_ax, "ROI name", initial=self._current_region_data()["name"])
        self.name_box.on_submit(self._on_name_submit)
        self.widgets.append(self.name_box)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _add_region(self, name: str | None = None) -> None:
        """Create and select a new empty ROI prompt."""
        if name is None:
            region_name = f"region_{self.next_region_idx}"
            self.next_region_idx += 1
        else:
            region_name = name
        self.regions.append(
            {
                "name": region_name,
                "box": None,
                "points": [],
                "mask": None,
                "score": None,
            }
        )
        self.current_region = len(self.regions) - 1

    def _current_region_data(self) -> dict[str, Any]:
        """Return the selected ROI data dictionary."""
        return self.regions[self.current_region]

    def _sync_name_box(self) -> None:
        """Synchronize the ROI name text box with the selected ROI."""
        if not hasattr(self, "name_box"):
            return
        self._syncing_name = True
        self.name_box.set_val(self._current_region_data()["name"])
        self._syncing_name = False

    def _on_name_submit(self, text: str) -> None:
        """Update the current ROI name from the text box."""
        if self._syncing_name:
            return
        self._current_region_data()["name"] = text.strip() or f"region_{self.current_region + 1}"
        self._refresh()

    def _button_box(self, _event: object) -> None:
        self._set_mode("box")

    def _button_point(self, _event: object) -> None:
        self._set_mode("point")

    def _button_new_region(self, _event: object) -> None:
        self._add_region()
        self._select_region(len(self.regions) - 1, self.mode)

    def _button_prev_region(self, _event: object) -> None:
        self._select_region((self.current_region - 1) % len(self.regions), self.mode)

    def _button_next_region(self, _event: object) -> None:
        self._select_region((self.current_region + 1) % len(self.regions), self.mode)

    def _button_reset_region(self, _event: object) -> None:
        self._reset_current_region()

    def _button_save(self, _event: object) -> None:
        self._save(str(self.json_path))
        self._refresh()

    def _button_save_close(self, _event: object) -> None:
        self._save(str(self.json_path))
        import matplotlib.pyplot as plt

        plt.close(self.fig)

    def _set_mode(self, mode: str) -> None:
        """Switch interaction mode."""
        self.mode = mode
        self._dragging_box = False
        self._box_start = None
        self._box_preview = None
        self._refresh()

    def _select_region(self, region_idx: int, mode: str | None = None) -> None:
        """Select an ROI while preserving or explicitly setting the prompt mode."""
        self.current_region = region_idx % len(self.regions)
        if mode is not None:
            self.mode = mode
        self._dragging_box = False
        self._box_start = None
        self._box_preview = None
        self._sync_name_box()
        current = self._current_region_data()
        self.message = f"Selected ROI {self.current_region + 1}: {current['name']}."
        self._refresh()

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

    def _preview_box_to_original(self, box: tuple[float, float, float, float]) -> np.ndarray:
        """Map a preview-space box to original-image XYXY coordinates."""
        x0, y0, x1, y1 = _normalize_box(box)
        ox0, oy0 = self._preview_to_original_xy(x0, y0)
        ox1, oy1 = self._preview_to_original_xy(x1, y1)
        return np.array([min(ox0, ox1), min(oy0, oy1), max(ox0, ox1), max(oy0, oy1)], dtype=np.float32)

    def _mask_to_preview_size(self, mask: np.ndarray) -> np.ndarray:
        """Resize an original-image mask into preview space for display."""
        if mask.shape == (self.height, self.width):
            return mask.astype(bool, copy=True)
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_image = mask_image.resize((self.width, self.height), _nearest_resampling())
        return np.asarray(mask_image) > 127

    def _on_click(self, event: object) -> None:
        """Handle prompt clicks and box starts."""
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        if getattr(event, "button", None) != 1:
            return

        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        current = self._current_region_data()
        if self.mode == "box":
            self._dragging_box = True
            self._box_start = (x, y)
            self._box_preview = (x, y, x, y)
            self._refresh()
            return

        current["points"].append((x, y))
        self._predict_current_region()
        self._refresh()

    def _on_motion(self, event: object) -> None:
        """Update the box preview while dragging."""
        if not self._dragging_box or self._box_start is None:
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return

        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        x0, y0 = self._box_start
        self._box_preview = (x0, y0, x, y)
        self._refresh()

    def _on_release(self, event: object) -> None:
        """Finish a box prompt and run prediction."""
        if not self._dragging_box or self._box_start is None:
            return
        self._dragging_box = False

        if (
            getattr(event, "inaxes", None) is self.ax
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        else:
            _, _, x, y = self._box_preview or (*self._box_start, *self._box_start)

        x0, y0 = self._box_start
        box = _normalize_box((x0, y0, x, y))
        self._box_start = None
        self._box_preview = None

        if box[2] - box[0] >= 2.0 and box[3] - box[1] >= 2.0:
            self._current_region_data()["box"] = box
            self._predict_current_region()
        self._refresh()

    def _on_key(self, event: object) -> None:
        """Keyboard shortcut handler."""
        key = (getattr(event, "key", "") or "").lower()
        if key == "b":
            self._set_mode("box")
        elif key in ("p", "f"):
            self._set_mode("point")
        elif key == "n":
            self._button_new_region(event)
        elif key == "[":
            self._button_prev_region(event)
        elif key in ("]", "tab"):
            self._button_next_region(event)
        elif key in ("r", "c"):
            self._reset_current_region()
        elif key in ("s", "ctrl+s", "cmd+s"):
            self._save(str(self.json_path))
            self._refresh()
        elif key == "escape":
            self._dragging_box = False
            self._box_start = None
            self._box_preview = None
            self._refresh()

    def _reset_current_region(self) -> None:
        """Clear the current ROI prompts and predicted mask."""
        current = self._current_region_data()
        current["box"] = None
        current["points"] = []
        current["mask"] = None
        current["score"] = None
        self._box_start = None
        self._box_preview = None
        self._dragging_box = False
        self.message = "Current ROI reset."
        self._refresh()

    def _predict_current_region(self) -> None:
        """Run SAM2 prediction for the current ROI prompt."""
        current = self._current_region_data()
        points = current["points"]
        box = current["box"]
        if not points and box is None:
            current["mask"] = None
            current["score"] = None
            self.message = "Add a box or point prompt before prediction."
            return

        point_coords = None
        point_labels = None
        if points:
            original_points = [self._preview_to_original_xy(float(x), float(y)) for x, y in points]
            point_coords = np.array(original_points, dtype=np.float32)
            point_labels = np.ones(len(points), dtype=np.int32)

        input_box = self._preview_box_to_original(box) if box is not None else None
        multimask_output = bool(input_box is None and len(points) <= 1)

        self.message = "Predicting ROI mask..."
        self._refresh()
        self.fig.canvas.flush_events()

        try:
            with _inference_context(self.device):
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=input_box,
                    multimask_output=multimask_output,
                )
            best_idx = int(np.argmax(scores))
            current["mask"] = np.asarray(masks[best_idx]) > 0
            current["score"] = float(scores[best_idx])
            pixels = int(current["mask"].sum())
            self.message = f"Predicted ROI mask with {pixels} pixels."
        except Exception as exc:
            current["mask"] = None
            current["score"] = None
            self.message = f"Prediction failed: {exc}"

    def _region_color(self, region_idx: int) -> tuple[np.ndarray, float]:
        """Return the display color and alpha for a region."""
        if region_idx == self.current_region:
            return np.array([0.0, 0.9, 1.0], dtype=np.float32), 0.44
        inactive_colors = np.array(
            [
                [1.0, 0.78, 0.0],
                [0.35, 1.0, 0.25],
                [1.0, 0.25, 0.65],
                [0.75, 0.45, 1.0],
                [1.0, 0.45, 0.2],
            ],
            dtype=np.float32,
        )
        return inactive_colors[region_idx % len(inactive_colors)], 0.28

    def _mask_overlay(self) -> np.ndarray:
        """Build the RGBA overlay for all predicted ROI masks."""
        overlay = np.zeros((self.height, self.width, 4), dtype=np.float32)
        for idx, region in enumerate(self.regions):
            mask = region["mask"]
            if mask is None or not bool(mask.any()):
                continue
            preview_mask = self._mask_to_preview_size(mask)
            color, alpha = self._region_color(idx)
            overlay[preview_mask, :3] = color
            overlay[preview_mask, 3] = alpha
        return overlay

    def _refresh(self) -> None:
        """Redraw mask, prompt overlays, and status text."""
        self.mask_artist.set_data(self._mask_overlay())
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists.clear()

        from matplotlib.patches import Rectangle

        for idx, region in enumerate(self.regions, start=1):
            box = self._box_preview if idx - 1 == self.current_region and self._box_preview else region["box"]
            if box is not None:
                x0, y0, x1, y1 = _normalize_box(box)
                edge_color = tuple(self._region_color(idx - 1)[0].tolist())
                rectangle = Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=2.4 if idx - 1 == self.current_region else 1.6,
                )
                self.ax.add_patch(rectangle)
                self._dynamic_artists.append(rectangle)

            for x, y in region["points"]:
                is_current = idx - 1 == self.current_region
                point_color = tuple(self._region_color(idx - 1)[0].tolist())
                marker = self.ax.scatter(
                    [x],
                    [y],
                    c=[point_color],
                    s=70 if is_current else 38,
                    marker="o",
                    edgecolors="white",
                    linewidths=1.4 if is_current else 0.8,
                    alpha=1.0 if is_current else 0.45,
                )
                self._dynamic_artists.append(marker)

            if region["mask"] is not None and bool(region["mask"].any()):
                label_x, label_y = self._region_label_position(region)
                label = self.ax.text(
                    label_x,
                    label_y,
                    f" {idx}",
                    color="white",
                    fontsize=9,
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
                )
                self._dynamic_artists.append(label)

        current = self._current_region_data()
        score = current["score"]
        score_text = "score: n/a" if score is None else f"score: {score:.3f}"
        mask = current["mask"]
        pixels = 0 if mask is None else int(mask.sum())
        point_count = len(current["points"])
        box_text = "box: yes" if current["box"] is not None else "box: no"
        size_text = f"Preview: {self.width}x{self.height}"
        if self.is_downsampled:
            size_text += f" -> Original: {self.original_width}x{self.original_height}"
        self.status.set_text(
            f"Mode: {self.mode.upper()} | ROI {self.current_region + 1}/{len(self.regions)} "
            f"'{current['name']}' | {box_text}, points: {point_count}, mask px: {pixels}, "
            f"{score_text} | Device: {self.device} | {size_text} | Save: {self.json_path} | {self.message}"
        )
        self.fig.canvas.draw_idle()

    def _region_label_position(self, region: dict[str, Any]) -> tuple[float, float]:
        """Return a stable preview-space label position for a region."""
        if region["box"] is not None:
            x0, y0, _x1, _y1 = _normalize_box(region["box"])
            return x0, y0
        if region["points"]:
            x, y = region["points"][0]
            return float(x), float(y)
        mask = region["mask"]
        if mask is None or not bool(mask.any()):
            return 0.0, 0.0
        preview_mask = self._mask_to_preview_size(mask)
        ys, xs = np.nonzero(preview_mask)
        return float(xs[0]), float(ys[0])

    def _save(self, json_path: str) -> None:
        """Write predicted masks and the MaDCoW-compatible JSON."""
        json_out = Path(json_path).resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        regions_json: list[dict[str, str]] = []

        for idx, region in enumerate(self.regions, start=1):
            mask = region["mask"]
            if mask is None or not bool(mask.any()):
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

        payload = {
            "image_path": _relative_path(Path(self.image_path), json_out.parent),
            "fov_deg": self.fov_deg,
            "lines": [],
            "regions": regions_json,
        }
        json_out.write_text(json.dumps(payload, indent=4), encoding="utf-8")
        self.message = f"Saved {len(regions_json)} ROI mask(s) to {json_out}."


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SAM2 ROI annotation tool."""
    parser = argparse.ArgumentParser(description="SAM2-assisted ROI annotation GUI.")
    parser.add_argument(
        "--image",
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
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help=f"SAM2 checkpoint path. Defaults to {DEFAULT_CHECKPOINT}.",
    )
    parser.add_argument(
        "--model-cfg",
        default=DEFAULT_MODEL_CFG,
        help=f"SAM2 Hydra model config name. Defaults to {DEFAULT_MODEL_CFG}.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device. Use 'auto' to prefer CUDA when available, otherwise CPU.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    if args.fov is None:
        fov, source = estimate_fov_from_exif(str(image_path))
        print(f"Using horizontal FOV {fov:.2f} degrees ({source}).")
    else:
        fov = float(args.fov)

    output_dir = args.output_dir or str(image_path.parent)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")

    device = _resolve_device(str(args.device))
    model_cfg = _normalize_model_cfg(str(args.model_cfg))
    print(f"Loading SAM2 model on {device}: {checkpoint}")
    predictor = build_image_predictor(model_cfg, checkpoint, device)
    SAM2ROIAnnotationGUI(str(image_path), fov, output_dir, predictor, device).run()


if __name__ == "__main__":
    sys.exit(main())
