"""Integrated top-level annotation workflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from annotation_gui import AnnotationSession, load_image_view
from annotation_gui.base import (
    EmbeddedViewSetupController,
    clear_widget_axes,
    create_image_figure,
    set_image_artist,
)
from annotation_gui.io import build_annotation_payload, write_annotation_json
from interactive_snapping_2d.annotate_line_aid import LineAidAnnotationGUI
from interactive_snapping_2d.snap2d import load_snap_config
from pipeline.annotation_launcher import (
    FIXED_SNAP_CONFIG_PATH,
    SAM2_MODEL_OPTIONS,
    build_launcher_defaults,
    default_workspace_for_image,
    is_model_available,
    list_image_browser_entries,
    model_option_by_key,
    output_annotation_for_workspace,
    repo_root,
)
from sam2.annotate_ROI_auto import (
    FALLBACK_FOV_DEG,
    PREVIEW_MAX_SIDE,
    SAM2ROIAnnotationGUI,
    build_image_predictor,
    estimate_fov_from_exif,
    normalize_model_cfg,
    resolve_checkpoint,
    resolve_device,
)


TOTAL_WORKFLOW_STEPS = 7
FILE_ROWS_PER_PAGE = 8
BLANK_PREVIEW = np.full((600, 900, 3), 242, dtype=np.uint8)
STATE_SELECT_IMAGE = "select_image"
STATE_SELECT_OUTPUT = "select_output"
STATE_SELECT_MODEL = "select_model"
STATE_VIEW_SETUP = "view_setup"
STATE_ROI = "roi"
STATE_LINE = "line"
STATE_COMPLETE = "complete"
SIDEBAR_X = 0.04
PREVIEW_RECT = (0.40, 0.23, 0.56, 0.60)
SIDEBAR_BUTTON_H = 0.045


@dataclass(frozen=True)
class IntegratedAnnotationConfig:
    """Configuration defaults for one integrated annotation run."""

    image: str | None = None
    workspace: str | None = None
    output_annotation: str | None = None
    device: str = "auto"
    snap_config: str | None = FIXED_SNAP_CONFIG_PATH
    sam2_checkpoint: str | None = None
    sam2_model_cfg: str | None = None


def _make_workspace(workspace: Path) -> dict[str, Path]:
    """Create and return the integrated annotation workspace directories."""
    paths = {
        "root": workspace,
        "annotation": workspace / "annotation",
        "masks": workspace / "masks",
        "views": workspace / "views",
        "logs": workspace / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _summary_payload(
    session: Any,
    output_annotation: Path,
    workspace: Path,
    lines: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the annotation run summary JSON."""
    summary: dict[str, Any] = {
        "input_image": session.source_image_path,
        "active_annotation_image": session.image_path,
        "camera_model": session.camera_model,
        "number_of_lines": len(lines),
        "number_of_regions": len(regions),
        "output_annotation": str(output_annotation.resolve()),
        "workspace": str(workspace.resolve()),
    }
    if session.view_metadata is not None:
        summary["source_image_path"] = session.source_image_path
        summary["view"] = session.view_metadata
    if session.camera_model == "pinhole":
        summary["fov_deg"] = session.fov_deg
    return summary


class _WorkflowStatusProxy:
    """Prefix downstream status text with the current top-level workflow step."""

    def __init__(self, owner: "IntegratedAnnotationGUI") -> None:
        self.owner = owner

    def set_text(self, text: object) -> None:
        self.owner._set_workflow_status(str(text))


class IntegratedAnnotationGUI:
    """Single-window setup, ROI, line, and final annotation controller."""

    def __init__(self, config: IntegratedAnnotationConfig) -> None:
        self.config = config
        self.completed = False
        self.session: AnnotationSession | None = None
        self.roi_gui: SAM2ROIAnnotationGUI | None = None
        self.line_gui: LineAidAnnotationGUI | None = None
        self.regions: list[dict[str, Any]] = []
        self.lines: list[dict[str, Any]] = []
        self._setup_connection_ids: list[int] = []
        self._input_text_artists: list[Any] = []
        self._input_page = 0
        self._input_entries: list[Path] = []
        self.state = STATE_SELECT_IMAGE

        defaults = build_launcher_defaults(config.image, config.workspace, config.output_annotation)
        self.current_folder = Path(defaults.data_dir).expanduser()
        self.selected_image_path = Path(defaults.image).expanduser() if defaults.image else None
        if self.selected_image_path is not None and not self.selected_image_path.is_file():
            self.selected_image_path = None
        self.workspace_path = Path(defaults.workspace).expanduser()
        self.output_annotation = Path(defaults.output_annotation).expanduser()
        self.output_manual = config.workspace is not None or config.output_annotation is not None
        self.selected_model_key = defaults.model_key
        if config.sam2_checkpoint and config.sam2_model_cfg:
            self.selected_model_key = ""
        self.sam2_checkpoint = config.sam2_checkpoint
        self.sam2_model_cfg = config.sam2_model_cfg
        self.paths: dict[str, Path] = {}
        self._folder_text_box: Any | None = None
        self._output_text_box: Any | None = None

        self.fig, self.ax, self.image_artist, self.status_text, self.help_text = create_image_figure(
            "Integrated Annotation",
            BLANK_PREVIEW,
        )
        self.fig.patch.set_facecolor("#fbfbfb")
        self.status = _WorkflowStatusProxy(self)
        self.widgets: list[Any] = []
        self._show_image_step()

    def run(self) -> Path:
        """Run the integrated GUI until the final save phase is complete."""
        import matplotlib.pyplot as plt

        plt.show()
        if not self.completed:
            raise RuntimeError("Integrated annotation GUI closed before final save.")
        return self.output_annotation

    def _phase_prefix(self) -> str:
        return f"Step {self._current_step} of {TOTAL_WORKFLOW_STEPS}: {self._current_title}"

    def _set_phase(self, step: int, title: str, status: str = "", help_text: str = "") -> None:
        self._current_step = int(step)
        self._current_title = title
        self._set_workflow_status(status)
        self.help_text.set_text(help_text)

    def _set_workflow_status(self, status: str) -> None:
        detail = f" | {status}" if status else ""
        self.status_text.set_text(f"{self._phase_prefix()}{detail}")

    @staticmethod
    def _ellipsize(text: object, max_chars: int = 48) -> str:
        value = str(text)
        if len(value) <= max_chars:
            return value
        return "..." + value[-max_chars + 3 :]

    @staticmethod
    def _display_path(path: object | None) -> str:
        """Return a repo-root-relative path string for GUI display."""
        if path is None:
            return "-"
        value = str(path)
        if not value:
            return "-"
        root = repo_root().resolve()
        raw_path = Path(value).expanduser()
        path_to_show = raw_path if raw_path.is_absolute() else root / raw_path
        try:
            return Path(os.path.relpath(path_to_show.resolve(), root)).as_posix()
        except OSError:
            return Path(os.path.relpath(path_to_show, root)).as_posix()

    @staticmethod
    def _path_from_display_text(text: str) -> Path:
        """Parse a GUI path field; relative input is resolved from repo root."""
        raw = Path(text).expanduser()
        if raw.is_absolute():
            return raw
        return repo_root() / raw

    def _clear_input_artists(self) -> None:
        clear_widget_axes(self.widgets)
        for artist in list(self._input_text_artists):
            try:
                artist.remove()
            except ValueError:
                pass
        self._input_text_artists.clear()

    def _add_text(self, x: float, y: float, text: str, **kwargs: Any) -> Any:
        artist = self.fig.text(x, y, text, **kwargs)
        self._input_text_artists.append(artist)
        return artist

    def _finish_render(self) -> None:
        self.fig.canvas.draw_idle()

    def _add_step_heading(self) -> None:
        self._add_text(
            SIDEBAR_X,
            0.885,
            f"Step {self._current_step} of {TOTAL_WORKFLOW_STEPS}",
            fontsize=9,
            color="#666666",
            ha="left",
            va="center",
        )
        self._add_text(
            SIDEBAR_X,
            0.855,
            self._current_title,
            fontsize=13,
            weight="bold",
            color="#222222",
            ha="left",
            va="center",
        )

    def _add_summary_line(self, y: float, label: str, value: object | None, max_chars: int = 46) -> None:
        self._add_text(SIDEBAR_X, y, label, fontsize=8.5, weight="bold", color="#555555", ha="left", va="center")
        self._add_text(
            SIDEBAR_X,
            y - 0.025,
            self._ellipsize(self._display_path(value), max_chars=max_chars),
            fontsize=8.5,
            color="#333333",
            ha="left",
            va="center",
        )

    def _add_phase_sidebar(self, notes: list[str] | None = None) -> None:
        self._add_step_heading()
        self._add_summary_line(0.79, "Image", self.selected_image_path, max_chars=48)
        self._add_summary_line(0.71, "Output", self.output_annotation, max_chars=48)
        if self.selected_model_key:
            model = model_option_by_key(self.selected_model_key).label
        elif self.sam2_checkpoint:
            model = Path(self.sam2_checkpoint).name
        else:
            model = "-"
        self._add_summary_line(0.63, "SAM2 model", model, max_chars=48)
        if notes:
            y = 0.52
            self._add_text(SIDEBAR_X, y, "Current task", fontsize=8.5, weight="bold", color="#555555", ha="left")
            y -= 0.035
            for note in notes:
                self._add_text(SIDEBAR_X, y, note, fontsize=8.5, color="#333333", ha="left", va="center")
                y -= 0.035

    def _add_button(
        self,
        label: str,
        rect: tuple[float, float, float, float],
        callback: Any | None,
        enabled: bool = True,
    ) -> Any:
        from matplotlib.widgets import Button

        ax_button = self.fig.add_axes(rect)
        color = "#f7f7f7" if enabled else "#eeeeee"
        button = Button(ax_button, label, color=color, hovercolor="#e6eef8")
        if enabled and callback is not None:
            button.on_clicked(callback)
        else:
            button.set_active(False)
        self.widgets.append(button)
        return button

    def _add_text_box(
        self,
        label: str,
        initial: str,
        rect: tuple[float, float, float, float],
        callback: Any,
    ) -> Any:
        from matplotlib.widgets import TextBox

        ax_box = self.fig.add_axes(rect)
        text_box = TextBox(ax_box, label, initial=initial)
        text_box.on_submit(callback)
        self.widgets.append(text_box)
        return text_box

    @staticmethod
    def _text_box_value(text_box: Any | None, fallback: str) -> str:
        """Return the current text in a Matplotlib TextBox."""
        if text_box is None:
            return fallback
        return str(getattr(text_box, "text", fallback))

    def _set_launcher_preview_layout(self) -> None:
        self.ax.set_position(PREVIEW_RECT)
        self.ax.set_visible(True)

    def _set_annotation_layout(self) -> None:
        self.ax.set_position(PREVIEW_RECT)
        self.ax.set_visible(True)

    def _show_blank_preview(self, message: str) -> None:
        set_image_artist(self.image_artist, self.ax, BLANK_PREVIEW)
        self.ax.set_title(message, fontsize=10)
        self.fig.canvas.draw_idle()

    def _render_selected_image_preview(self) -> None:
        if self.selected_image_path is None or not self.selected_image_path.is_file():
            self._show_blank_preview("No image selected")
            return
        try:
            preview = load_image_view(self.selected_image_path, PREVIEW_MAX_SIDE).preview
        except OSError as exc:
            self._show_blank_preview(f"Preview unavailable: {exc}")
            return
        set_image_artist(self.image_artist, self.ax, preview)
        self.ax.set_title(self._display_path(self.selected_image_path), fontsize=9)
        self.fig.canvas.draw_idle()

    def _load_input_entries(self) -> None:
        directories, images = list_image_browser_entries(self.current_folder)
        self._input_entries = directories + images
        max_page = max(0, (len(self._input_entries) - 1) // FILE_ROWS_PER_PAGE)
        self._input_page = max(0, min(self._input_page, max_page))

    def _select_file_entry(self, path: Path) -> None:
        if self.state != STATE_SELECT_IMAGE:
            return
        if path.is_dir():
            self.current_folder = path
            self._input_page = 0
            self._show_image_step()
            return
        self.selected_image_path = path
        if not self.output_manual:
            self.workspace_path = default_workspace_for_image(path)
            self.output_annotation = output_annotation_for_workspace(self.workspace_path, self.config.output_annotation)
        self._render_selected_image_preview()
        self._show_image_step(status=f"Selected image: {path.name}")

    def _open_typed_folder(self, text: str) -> None:
        if self.state != STATE_SELECT_IMAGE:
            return
        if not text.strip():
            self._show_image_step(status="Enter a folder path before opening.")
            return
        folder = self._path_from_display_text(text)
        if not folder.is_dir():
            self._show_image_step(status=f"Folder does not exist: {self._display_path(folder)}")
            return
        self.current_folder = folder
        self._input_page = 0
        self._show_image_step()

    def _show_image_step(self, status: str | None = None) -> None:
        self.state = STATE_SELECT_IMAGE
        self._clear_input_artists()
        self._set_launcher_preview_layout()
        self._load_input_entries()
        self._set_phase(
            1,
            "Select Image",
            status or f"{len([p for p in self._input_entries if p.is_file()])} image(s) found in {self._display_path(self.current_folder)}.",
            "Click an image filename to preview it. Click a folder to enter it.",
        )
        self._add_step_heading()
        self._add_text(SIDEBAR_X, 0.815, "Folder", fontsize=8.5, weight="bold", color="#555555", ha="left")

        self._folder_text_box = self._add_text_box("", self._display_path(self.current_folder), (0.04, 0.775, 0.30, 0.04), self._open_typed_folder)
        self._add_button(
            "Open",
            (0.04, 0.725, 0.14, 0.04),
            lambda _event: self._open_typed_folder(self._text_box_value(self._folder_text_box, self._display_path(self.current_folder))),
        )
        self._add_button("Up", (0.20, 0.725, 0.14, 0.04), lambda _event: self._select_file_entry(self.current_folder.parent))

        start = self._input_page * FILE_ROWS_PER_PAGE
        page_entries = self._input_entries[start : start + FILE_ROWS_PER_PAGE]
        y = 0.665
        for entry in page_entries:
            label = f"[DIR] {entry.name}" if entry.is_dir() else entry.name
            self._add_button(label[:34], (0.04, y, 0.30, 0.035), lambda _event, path=entry: self._select_file_entry(path))
            y -= 0.045

        total_pages = max(1, (len(self._input_entries) + FILE_ROWS_PER_PAGE - 1) // FILE_ROWS_PER_PAGE)
        self._add_text(0.04, 0.285, f"Page {self._input_page + 1} of {total_pages}", fontsize=8.5, color="#555555", ha="left", va="center")
        self._add_button("Prev", (0.16, 0.265, 0.08, 0.04), self._button_prev_page, enabled=self._input_page > 0)
        self._add_button("Next", (0.26, 0.265, 0.08, 0.04), self._button_next_page, enabled=self._input_page + 1 < total_pages)

        selected = self.selected_image_path if self.selected_image_path else None
        self._add_summary_line(0.22, "Selected image", selected, max_chars=44)
        self._add_button("Continue", (0.20, 0.07, 0.14, SIDEBAR_BUTTON_H), self._button_image_done, enabled=self.selected_image_path is not None)
        self._render_selected_image_preview()
        self._finish_render()

    def _button_prev_page(self, _event: object) -> None:
        if self.state != STATE_SELECT_IMAGE:
            return
        self._input_page = max(0, self._input_page - 1)
        self._show_image_step()

    def _button_next_page(self, _event: object) -> None:
        if self.state != STATE_SELECT_IMAGE:
            return
        self._input_page += 1
        self._show_image_step()

    def _button_image_done(self, _event: object) -> None:
        if self.state != STATE_SELECT_IMAGE:
            return
        if self.selected_image_path is None or not self.selected_image_path.is_file():
            self._show_image_step("Select an input image before continuing.")
            return
        self._show_output_step()

    def _update_workspace_from_text(self, text: str) -> bool:
        if not text.strip():
            return False
        self.output_manual = True
        self.workspace_path = self._path_from_display_text(text)
        self.output_annotation = output_annotation_for_workspace(self.workspace_path, self.config.output_annotation)
        return True

    def _on_workspace_submit(self, text: str) -> None:
        if self.state != STATE_SELECT_OUTPUT:
            return
        if not self._update_workspace_from_text(text):
            self._show_output_step("Enter an output directory before continuing.")
            return
        self._show_output_step("Output directory updated.")

    def _show_output_step(self, status: str | None = None) -> None:
        self.state = STATE_SELECT_OUTPUT
        self._clear_input_artists()
        self._set_launcher_preview_layout()
        self._render_selected_image_preview()
        if self.selected_image_path is not None and not self.output_manual:
            self.workspace_path = default_workspace_for_image(self.selected_image_path)
            self.output_annotation = output_annotation_for_workspace(self.workspace_path, self.config.output_annotation)
        self._set_phase(
            2,
            "Select Output Directory",
            status or "Confirm or edit the annotation output directory.",
            "Press Enter in the text field to apply edits. The final JSON path updates below.",
        )
        self._add_step_heading()
        self._add_summary_line(0.79, "Input image", self.selected_image_path, max_chars=46)
        self._add_text(SIDEBAR_X, 0.68, "Output directory", fontsize=8.5, weight="bold", color="#555555", ha="left")
        self._output_text_box = self._add_text_box("", self._display_path(self.workspace_path), (0.04, 0.635, 0.30, 0.045), self._on_workspace_submit)
        self._add_summary_line(0.55, "Final annotation JSON", self.output_annotation, max_chars=46)
        self._add_text(
            SIDEBAR_X,
            0.43,
            "Workspace folders are created when annotation starts.",
            fontsize=8.5,
            color="#555555",
            ha="left",
        )
        self._add_button(
            "Back",
            (0.04, 0.07, 0.12, SIDEBAR_BUTTON_H),
            lambda _event: self._show_image_step() if self.state == STATE_SELECT_OUTPUT else None,
        )
        self._add_button("Continue", (0.20, 0.07, 0.14, SIDEBAR_BUTTON_H), self._button_output_done)
        self._finish_render()

    def _button_output_done(self, _event: object) -> None:
        if self.state != STATE_SELECT_OUTPUT:
            return
        if not self._update_workspace_from_text(self._text_box_value(self._output_text_box, self._display_path(self.workspace_path))):
            self._show_output_step("Enter an output directory before continuing.")
            return
        self._show_model_step()

    def _select_model(self, key: str) -> None:
        if self.state != STATE_SELECT_MODEL:
            return
        option = model_option_by_key(key)
        if not is_model_available(option):
            self._show_model_step(f"Missing checkpoint for {option.label}.")
            return
        self.selected_model_key = key
        self._show_model_step(f"Selected SAM2 model: {option.label}.")

    def _show_model_step(self, status: str | None = None) -> None:
        self.state = STATE_SELECT_MODEL
        self._clear_input_artists()
        self._set_launcher_preview_layout()
        self._render_selected_image_preview()
        self._set_phase(
            3,
            "Select SAM2 Model",
            status or "Choose one SAM2.1 model for ROI annotation.",
            "Missing checkpoints are shown as unavailable. Continue starts the camera/view setup in this same window.",
        )
        self._add_step_heading()
        self._add_summary_line(0.79, "Input image", self.selected_image_path, max_chars=46)
        self._add_summary_line(0.71, "Output", self.output_annotation, max_chars=46)
        y = 0.595
        for option in SAM2_MODEL_OPTIONS:
            available = is_model_available(option)
            selected = option.key == self.selected_model_key
            prefix = "[x]" if selected else "[ ]"
            suffix = "" if available else " (missing checkpoint)"
            label = f"{prefix} {option.label}{suffix}"
            self._add_button(label, (0.04, y, 0.30, 0.04), lambda _event, key=option.key: self._select_model(key), enabled=available)
            self._add_text(SIDEBAR_X, y - 0.020, option.description, fontsize=8.1, color="#555555", ha="left", va="center")
            y -= 0.105
        self._add_button(
            "Back",
            (0.04, 0.07, 0.12, SIDEBAR_BUTTON_H),
            lambda _event: self._show_output_step() if self.state == STATE_SELECT_MODEL else None,
        )
        self._add_button("Start", (0.20, 0.07, 0.14, SIDEBAR_BUTTON_H), self._button_model_done)
        self._finish_render()

    def _button_model_done(self, _event: object) -> None:
        if self.state != STATE_SELECT_MODEL:
            return
        if self.selected_model_key:
            option = model_option_by_key(self.selected_model_key)
            if not is_model_available(option):
                self._show_model_step(f"Missing checkpoint for {option.label}.")
                return
            self.sam2_checkpoint = str((Path(__file__).resolve().parents[1] / option.checkpoint).resolve())
            self.sam2_model_cfg = option.model_cfg
        if self.sam2_checkpoint is None or self.sam2_model_cfg is None:
            self._show_model_step("Select an available SAM2 model before continuing.")
            return
        self._start_view_setup()

    def _start_view_setup(self) -> None:
        if self.state != STATE_SELECT_MODEL:
            return
        if self.selected_image_path is None:
            raise RuntimeError("Input image is not selected.")
        self.state = STATE_VIEW_SETUP
        self._clear_input_artists()
        self._set_annotation_layout()
        self.workspace_path = Path(self.workspace_path).expanduser().resolve()
        self.output_annotation = output_annotation_for_workspace(self.workspace_path, self.config.output_annotation).resolve()
        self.paths = _make_workspace(self.workspace_path)
        self.output_annotation.parent.mkdir(parents=True, exist_ok=True)

        image_path = self.selected_image_path.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")
        fov_deg, _source = estimate_fov_from_exif(str(image_path), fallback=FALLBACK_FOV_DEG)
        self.session = AnnotationSession.from_image(image_path, PREVIEW_MAX_SIDE, fov_deg)
        set_image_artist(self.image_artist, self.ax, self.session.source_view.preview)
        self._set_phase(
            4,
            "Camera/View Setup",
            "Choose camera mode.",
            "Choose Pinhole or Panorama, finalize view setup, then ROI annotation starts.",
        )
        self._add_phase_sidebar(["Choose Pinhole for FOV setup.", "Choose Panorama for center/crop setup."])
        self.setup_controller = EmbeddedViewSetupController(
            session=self.session,
            output_dir=self.paths["views"],
            preview_max_side=PREVIEW_MAX_SIDE,
            fig=self.fig,
            ax=self.ax,
            image_artist=self.image_artist,
            status=self.status,
            help_text=self.help_text,
            widgets=self.widgets,
            clear_dynamic_artists=lambda: None,
            on_done=self._on_setup_done,
            fov_deg=fov_deg,
            fallback_fov_deg=FALLBACK_FOV_DEG,
        )
        self._connect_setup_events()
        self.setup_controller.start()

    def _connect_setup_events(self) -> None:
        """Connect setup event routing for the shared view setup controller."""
        self._setup_connection_ids = [
            self.fig.canvas.mpl_connect("button_press_event", self.setup_controller.on_click),
            self.fig.canvas.mpl_connect("motion_notify_event", self.setup_controller.on_motion),
            self.fig.canvas.mpl_connect("button_release_event", self.setup_controller.on_release),
            self.fig.canvas.mpl_connect("key_press_event", self.setup_controller.on_key),
        ]

    def _disconnect_setup_events(self) -> None:
        """Disconnect setup event routing before entering annotation phases."""
        for connection_id in self._setup_connection_ids:
            self.fig.canvas.mpl_disconnect(connection_id)
        self._setup_connection_ids.clear()

    def _on_setup_done(self, session: AnnotationSession) -> None:
        """Start SAM2 ROI annotation after shared setup is finalized."""
        self._disconnect_setup_events()
        self.state = STATE_ROI
        self._clear_input_artists()
        self._set_annotation_layout()
        self.session = session
        set_image_artist(self.image_artist, self.ax, session.preview)
        self._set_phase(
            5,
            "SAM2 ROI Annotation",
            "Loading SAM2 model...",
            "Use box or point prompts to create ROI masks, then press Done ROI.",
        )
        self._add_phase_sidebar(["Create ROI masks with box or point prompts.", "Press Done ROI to continue to lines."])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        checkpoint = resolve_checkpoint(str(self.sam2_checkpoint))
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
        device = resolve_device(self.config.device)
        model_cfg = normalize_model_cfg(str(self.sam2_model_cfg))
        print(f"Loading SAM2 model on {device}: {checkpoint}")
        predictor = build_image_predictor(model_cfg, checkpoint, device)

        self.roi_gui = SAM2ROIAnnotationGUI.from_session(
            session=session,
            output_dir=self.paths["annotation"],
            predictor=predictor,
            device=device,
            json_path=self.paths["annotation"] / "roi_only.json",
            fig=self.fig,
            ax=self.ax,
            image_artist=self.image_artist,
            status=self.status,
            help_text=self.help_text,
            widgets=self.widgets,
            on_complete=self._on_roi_done,
            save_close_label="Done ROI",
        )

    def _on_roi_done(self, roi_gui: SAM2ROIAnnotationGUI) -> None:
        """Start line snapping after ROI annotation is finished."""
        if self.session is None:
            raise RuntimeError("Annotation session is not initialized.")
        self.state = STATE_LINE
        self.regions = roi_gui.export_regions(mask_dir=self.paths["masks"], json_base_dir=self.output_annotation.parent)
        roi_gui.detach_for_reuse()
        self._clear_input_artists()
        self._set_annotation_layout()

        self._set_phase(
            6,
            "Snapping Line Annotation",
            "Preparing interactive snapping...",
            "Draw rough strokes, accept snapped lines, then press Save Final.",
        )
        self._add_phase_sidebar(["Draw rough strokes near straight structures.", "Press Save Final when lines are done."])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        snap_config = load_snap_config(self.config.snap_config)
        self.line_gui = LineAidAnnotationGUI.from_session(
            session=self.session,
            output_dir=self.paths["annotation"],
            snap_config=snap_config,
            json_path=self.paths["annotation"] / "lines_only.json",
            fig=self.fig,
            ax=self.ax,
            image_artist=self.image_artist,
            status=self.status,
            help_text=self.help_text,
            widgets=self.widgets,
            on_complete=self._on_line_done,
            save_close_label="Save Final",
            allow_empty_annotations=True,
        )

    def _on_line_done(self, line_gui: LineAidAnnotationGUI) -> None:
        """Write the complete annotation JSON and show the completion step."""
        if self.session is None:
            raise RuntimeError("Annotation session is not initialized.")
        self.state = STATE_COMPLETE
        self._set_phase(7, "Save Complete Annotation", "Writing final annotation JSON...", "The final JSON and summary are being written.")
        self.lines = line_gui.export_lines(self.output_annotation)
        line_gui.disconnect_events()
        clear_widget_axes(self.widgets)

        payload = build_annotation_payload(
            image_path=self.session.image_path,
            output_path=self.output_annotation,
            source_image_path=self.session.source_image_path if self.session.view_metadata is not None else None,
            camera_model=self.session.camera_model,
            fov_deg=self.session.fov_deg if self.session.camera_model == "pinhole" else None,
            view_metadata=self.session.view_metadata,
            lines=self.lines,
            regions=self.regions,
        )
        write_annotation_json(self.output_annotation, payload)

        summary_path = self.paths["root"] / "annotation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                _summary_payload(
                    self.session,
                    self.output_annotation,
                    self.paths["root"],
                    self.lines,
                    self.regions,
                ),
                f,
                indent=4,
            )
            f.write("\n")

        self.completed = True
        self._show_complete_step(summary_path)

    def _show_complete_step(self, summary_path: Path) -> None:
        self.state = STATE_COMPLETE
        self._clear_input_artists()
        clear_widget_axes(self.widgets)
        self._set_phase(
            7,
            "Save Complete Annotation",
            f"Saved integrated annotation JSON: {self._display_path(self.output_annotation)}",
            "Close the window to return to the terminal.",
        )
        self._add_phase_sidebar(["Annotation JSON saved.", "Close the window to return to terminal."])
        self._add_summary_line(0.43, "Summary", summary_path, max_chars=46)
        self._add_text(SIDEBAR_X, 0.33, f"Lines: {len(self.lines)}", fontsize=9, color="#333333", ha="left")
        self._add_text(SIDEBAR_X, 0.30, f"Regions: {len(self.regions)}", fontsize=9, color="#333333", ha="left")
        self._add_button("Close", (0.20, 0.07, 0.14, SIDEBAR_BUTTON_H), self._button_close)
        self.fig.canvas.draw_idle()

    def _button_close(self, _event: object) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)


def run_integrated_annotation(config: IntegratedAnnotationConfig) -> Path:
    """Run the unified setup, ROI annotation, snapping, and final save GUI."""
    return IntegratedAnnotationGUI(config).run()
