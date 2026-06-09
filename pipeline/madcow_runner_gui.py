"""Matplotlib GUI for running MaDCoW correction from top-level inputs."""

from __future__ import annotations

import argparse
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.patches import Rectangle

from annotation_gui import load_image_view
from annotation_gui.base import (
    add_styled_button,
    clear_widget_axes,
    create_image_figure,
    set_button_style,
    set_image_artist,
    set_text_box_alignment,
)
from pipeline.annotation_launcher import repo_root


TOTAL_RUN_STEPS = 4
FILE_ROWS_PER_PAGE = 8
PREVIEW_MAX_SIDE = 1200
BLANK_PREVIEW = np.full((600, 900, 3), 242, dtype=np.uint8)
STATE_SELECT_ANNOTATION = "select_annotation"
STATE_SELECT_OUTPUT = "select_output"
STATE_RUNNING = "running"
STATE_COMPLETE = "complete"
SIDEBAR_X = 0.04
PREVIEW_RECT = (0.40, 0.23, 0.56, 0.60)
SIDEBAR_BUTTON_H = 0.045


@dataclass(frozen=True)
class MaDCoWRunnerConfig:
    """Configuration defaults for the MaDCoW runner GUI."""

    config: str
    annotations: str | None = None
    output_dir: str | None = None
    crop: bool = False


def _default_annotation_dir() -> Path:
    annotation_dir = repo_root() / "annotation"
    return annotation_dir if annotation_dir.exists() else repo_root()


def _default_output_root() -> Path:
    return repo_root() / "outputs"


def _list_annotation_entries(folder: str | Path) -> list[Path]:
    folder_path = Path(folder).expanduser()
    if not folder_path.is_dir():
        return []
    entries: list[Path] = []
    for child in sorted(folder_path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if child.is_dir() or (child.is_file() and child.suffix.lower() == ".json"):
            entries.append(child)
    return entries


def _default_output_dir_for_image(image_path: str | Path | None) -> Path:
    if image_path:
        image = Path(image_path).expanduser()
        name = image.parent.name or image.stem or "output"
    else:
        name = "output"
    return _default_output_root() / name


def _output_path_for_image(output_dir: str | Path, image_path: str | Path, crop: bool) -> Path:
    image = Path(image_path).expanduser()
    return Path(output_dir).expanduser() / _default_output_name_for_image(image, crop)


def _default_output_name_for_image(image_path: str | Path, crop: bool) -> str:
    image = Path(image_path).expanduser()
    suffix = "_corrected_crop.png" if crop else "_corrected.png"
    return f"{image.stem}{suffix}"


def _normalize_output_name(name: str) -> str:
    output_name = Path(name.strip()).name
    if not output_name:
        return ""
    if not Path(output_name).suffix:
        output_name += ".png"
    return output_name


class MaDCoWRunnerGUI:
    """Single-window MaDCoW correction launcher and progress viewer."""

    def __init__(self, config: MaDCoWRunnerConfig) -> None:
        self.config = config
        self.state = STATE_SELECT_ANNOTATION
        self.completed = False
        self.error_message: str | None = None
        self.widgets: list[Any] = []
        self._dynamic_artists: list[Any] = []
        self._annotation_page = 0
        self._annotation_entries: list[Path] = []
        self._progress_queue: queue.Queue[dict[str, object]] = queue.Queue()
        self._timer: Any | None = None
        self._worker_thread: threading.Thread | None = None
        self._folder_text_box: Any | None = None
        self._output_text_box: Any | None = None
        self._output_name_text_box: Any | None = None
        self._corrected_preview_button: Any | None = None
        self._input_preview_button: Any | None = None
        self._stage_bars: dict[str, tuple[Any, Any, Any]] = {}

        self.current_folder = Path(config.annotations).expanduser().parent if config.annotations else _default_annotation_dir()
        self.selected_annotation_path = Path(config.annotations).expanduser() if config.annotations else None
        if self.selected_annotation_path is not None and not self.selected_annotation_path.is_file():
            self.selected_annotation_path = None
        self.annotations_data: Any | None = None
        self.annotation_image_path: Path | None = None
        self.output_manual = config.output_dir is not None
        self.output_dir = Path(config.output_dir).expanduser() if config.output_dir else _default_output_root()
        self.output_name_manual = False
        self.output_name = "corrected.png"
        self.output_path = self.output_dir / "corrected.png"
        self.crop = bool(config.crop)
        self.preview_mode = "corrected"

        self.fig, self.ax, self.image_artist, self.status_text, self.help_text = create_image_figure(
            "Run MaDCoW",
            BLANK_PREVIEW,
        )
        self.fig.patch.set_facecolor("#fbfbfb")
        self.ax.set_position(PREVIEW_RECT)
        if self.selected_annotation_path:
            self._select_annotation_file(self.selected_annotation_path, render=False)
        self._show_annotation_step()

    def run(self) -> Path:
        """Run the GUI until correction completes and the window closes."""
        import matplotlib.pyplot as plt

        plt.show()
        if self.error_message is not None:
            raise RuntimeError(self.error_message)
        if not self.completed:
            raise RuntimeError("MaDCoW runner GUI closed before correction completed.")
        return self.output_path

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

    @staticmethod
    def _ellipsize(text: object, max_chars: int = 48) -> str:
        value = str(text)
        if len(value) <= max_chars:
            return value
        return "..." + value[-max_chars + 3 :]

    @staticmethod
    def _text_box_value(text_box: Any | None, fallback: str) -> str:
        if text_box is None:
            return fallback
        return str(getattr(text_box, "text", fallback))

    def _phase_prefix(self) -> str:
        return f"Step {self._current_step} of {TOTAL_RUN_STEPS}: {self._current_title}"

    def _set_phase(self, step: int, title: str, status: str = "", help_text: str = "") -> None:
        self._current_step = int(step)
        self._current_title = title
        detail = f" | {status}" if status else ""
        self.status_text.set_text(f"{self._phase_prefix()}{detail}")
        self.help_text.set_text(help_text)

    def _set_status(self, status: str) -> None:
        detail = f" | {status}" if status else ""
        self.status_text.set_text(f"{self._phase_prefix()}{detail}")

    def _clear_dynamic(self) -> None:
        clear_widget_axes(self.widgets)
        self._corrected_preview_button = None
        self._input_preview_button = None
        for artist in list(self._dynamic_artists):
            try:
                artist.remove()
            except ValueError:
                pass
        self._dynamic_artists.clear()
        self._stage_bars.clear()

    def _add_text(self, x: float, y: float, text: str, **kwargs: Any) -> Any:
        artist = self.fig.text(x, y, text, **kwargs)
        self._dynamic_artists.append(artist)
        return artist

    def _add_step_heading(self) -> None:
        self._add_text(
            SIDEBAR_X,
            0.885,
            f"Step {self._current_step} of {TOTAL_RUN_STEPS}",
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

    def _add_button(
        self,
        label: str,
        rect: tuple[float, float, float, float],
        callback: Any | None,
        enabled: bool = True,
        selected: bool = False,
        primary: bool = False,
        align: str = "center",
    ) -> Any:
        return add_styled_button(
            self.fig,
            self.widgets,
            label,
            rect,
            callback,
            enabled=enabled,
            selected=selected,
            primary=primary,
            align=align,
        )

    def _add_text_box(
        self,
        initial: str,
        rect: tuple[float, float, float, float],
        callback: Any,
    ) -> Any:
        from matplotlib.widgets import TextBox

        ax_box = self.fig.add_axes(rect)
        text_box = TextBox(ax_box, "", initial=initial, textalignment="left")
        set_text_box_alignment(text_box, "left")
        text_box.on_submit(callback)
        self.widgets.append(text_box)
        return text_box

    def _add_progress_bar(self, stage_key: str, label: str, y: float) -> None:
        self._add_text(SIDEBAR_X, y + 0.045, label, fontsize=8.8, weight="bold", color="#444444", ha="left")
        ax_bar = self.fig.add_axes((SIDEBAR_X, y, 0.30, 0.027))
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(0, 1)
        ax_bar.axis("off")
        ax_bar.add_patch(Rectangle((0, 0), 1, 1, facecolor="#eeeeee", edgecolor="#cccccc", linewidth=0.7))
        fill = Rectangle((0, 0), 0, 1, facecolor="#4c78a8", edgecolor="none")
        ax_bar.add_patch(fill)
        text = self._add_text(SIDEBAR_X, y - 0.027, "Waiting", fontsize=8.2, color="#555555", ha="left")
        self._dynamic_artists.append(ax_bar)
        self._stage_bars[stage_key] = (fill, text, ax_bar)

    def _set_progress(self, stage_key: str, current: object, total: object, message: str) -> None:
        if stage_key not in self._stage_bars:
            return
        try:
            total_value = max(1.0, float(total))
            ratio = max(0.0, min(1.0, float(current) / total_value))
        except (TypeError, ValueError):
            ratio = 0.0
        fill, text, _ax_bar = self._stage_bars[stage_key]
        fill.set_width(ratio)
        text.set_text(f"{int(round(ratio * 100.0))}%  {message}")

    def _finish_render(self) -> None:
        self.fig.canvas.draw_idle()

    def _show_blank_preview(self, message: str) -> None:
        self.ax.set_position(PREVIEW_RECT)
        self.ax.set_visible(True)
        set_image_artist(self.image_artist, self.ax, BLANK_PREVIEW)
        self.ax.set_title(message, fontsize=10)
        self.fig.canvas.draw_idle()

    def _render_image_preview(self, image_path: Path | None, title: str | None = None) -> None:
        if image_path is None or not image_path.is_file():
            self._show_blank_preview("Preview unavailable")
            return
        try:
            preview = load_image_view(image_path, PREVIEW_MAX_SIDE).preview
        except OSError as exc:
            self._show_blank_preview(f"Preview unavailable: {exc}")
            return
        self.ax.set_position(PREVIEW_RECT)
        self.ax.set_visible(True)
        set_image_artist(self.image_artist, self.ax, preview)
        self.ax.set_title(title or self._display_path(image_path), fontsize=9)
        self.fig.canvas.draw_idle()

    def _load_annotation_entries(self) -> None:
        self._annotation_entries = _list_annotation_entries(self.current_folder)
        max_page = max(0, (len(self._annotation_entries) - 1) // FILE_ROWS_PER_PAGE)
        self._annotation_page = max(0, min(self._annotation_page, max_page))

    def _open_typed_folder(self, text: str) -> None:
        if self.state != STATE_SELECT_ANNOTATION:
            return
        folder = self._path_from_display_text(text)
        if not folder.is_dir():
            self._show_annotation_step(f"Folder does not exist: {self._display_path(folder)}")
            return
        self.current_folder = folder
        self._annotation_page = 0
        self._show_annotation_step()

    def _select_entry(self, path: Path) -> None:
        if self.state != STATE_SELECT_ANNOTATION:
            return
        if path.is_dir():
            self.current_folder = path
            self._annotation_page = 0
            self._show_annotation_step()
            return
        self._select_annotation_file(path)

    def _select_annotation_file(self, path: Path, render: bool = True) -> None:
        from MaDCoW.main import load_annotations

        try:
            annotations = load_annotations(str(path))
        except Exception as exc:
            self.selected_annotation_path = None
            self.annotations_data = None
            self.annotation_image_path = None
            if render:
                self._show_annotation_step(f"Invalid annotation JSON: {exc}")
            return

        image_path = Path(annotations.image_path).expanduser()
        if not image_path.is_file():
            self.selected_annotation_path = None
            self.annotations_data = None
            self.annotation_image_path = None
            if render:
                self._show_annotation_step(f"Annotation image does not exist: {self._display_path(image_path)}")
            return

        self.selected_annotation_path = path
        self.annotations_data = annotations
        self.annotation_image_path = image_path
        if not self.output_manual:
            self.output_dir = _default_output_dir_for_image(image_path)
        if not self.output_name_manual:
            self.output_name = _default_output_name_for_image(image_path, self.crop)
        self.output_path = self.output_dir / self.output_name
        if render:
            self._show_annotation_step(f"Selected annotation: {path.name}")

    def _show_annotation_step(self, status: str | None = None) -> None:
        self.state = STATE_SELECT_ANNOTATION
        self._clear_dynamic()
        self._load_annotation_entries()
        self._set_phase(
            1,
            "Select Annotation",
            status or f"{len([p for p in self._annotation_entries if p.is_file()])} annotation JSON file(s) found.",
            "Click an annotation JSON to preview its image. Click a folder to enter it.",
        )
        self._add_step_heading()
        self._add_text(SIDEBAR_X, 0.815, "Folder", fontsize=8.5, weight="bold", color="#555555", ha="left")
        self._folder_text_box = self._add_text_box(
            self._display_path(self.current_folder),
            (0.04, 0.775, 0.30, 0.04),
            self._open_typed_folder,
        )
        self._add_button(
            "Open",
            (0.04, 0.725, 0.14, 0.04),
            lambda _event: self._open_typed_folder(self._text_box_value(self._folder_text_box, self._display_path(self.current_folder))),
        )
        self._add_button("Up", (0.20, 0.725, 0.14, 0.04), lambda _event: self._select_entry(self.current_folder.parent))

        start = self._annotation_page * FILE_ROWS_PER_PAGE
        page_entries = self._annotation_entries[start : start + FILE_ROWS_PER_PAGE]
        y = 0.665
        for entry in page_entries:
            label = f"[DIR] {entry.name}" if entry.is_dir() else entry.name
            self._add_button(
                label[:34],
                (0.04, y, 0.30, 0.035),
                lambda _event, path=entry: self._select_entry(path),
                align="left",
            )
            y -= 0.045

        total_pages = max(1, (len(self._annotation_entries) + FILE_ROWS_PER_PAGE - 1) // FILE_ROWS_PER_PAGE)
        self._add_text(0.04, 0.285, f"Page {self._annotation_page + 1} of {total_pages}", fontsize=8.5, color="#555555", ha="left", va="center")
        self._add_button("Prev", (0.16, 0.265, 0.08, 0.04), self._button_prev_page, enabled=self._annotation_page > 0)
        self._add_button("Next", (0.26, 0.265, 0.08, 0.04), self._button_next_page, enabled=self._annotation_page + 1 < total_pages)
        self._add_summary_line(0.22, "Selected annotation", self.selected_annotation_path, max_chars=44)
        self._add_button(
            "Continue",
            (0.20, 0.07, 0.14, SIDEBAR_BUTTON_H),
            self._button_annotation_done,
            enabled=self.annotations_data is not None,
            primary=True,
        )

        if self.annotation_image_path is not None:
            self._render_image_preview(self.annotation_image_path)
        else:
            self._show_blank_preview("No annotation selected")
        self._finish_render()

    def _button_prev_page(self, _event: object) -> None:
        if self.state != STATE_SELECT_ANNOTATION:
            return
        self._annotation_page = max(0, self._annotation_page - 1)
        self._show_annotation_step()

    def _button_next_page(self, _event: object) -> None:
        if self.state != STATE_SELECT_ANNOTATION:
            return
        self._annotation_page += 1
        self._show_annotation_step()

    def _button_annotation_done(self, _event: object) -> None:
        if self.state != STATE_SELECT_ANNOTATION:
            return
        if self.annotations_data is None:
            self._show_annotation_step("Select a valid annotation JSON before continuing.")
            return
        self._show_output_step()

    def _update_output_dir(self, text: str) -> bool:
        if not text.strip() or self.annotation_image_path is None:
            return False
        self.output_manual = True
        self.output_dir = self._path_from_display_text(text)
        if not self.output_name_manual:
            self.output_name = _default_output_name_for_image(self.annotation_image_path, self.crop)
        self.output_path = self.output_dir / self.output_name
        return True

    def _update_output_name(self, text: str) -> bool:
        if self.annotation_image_path is None:
            return False
        output_name = _normalize_output_name(text)
        if not output_name:
            return False
        self.output_name_manual = True
        self.output_name = output_name
        self.output_path = self.output_dir / self.output_name
        return True

    def _on_output_submit(self, text: str) -> None:
        if self.state != STATE_SELECT_OUTPUT:
            return
        if not self._update_output_dir(text):
            self._show_output_step("Enter an output directory before continuing.")
            return
        self._show_output_step("Output directory updated.")

    def _on_output_name_submit(self, text: str) -> None:
        if self.state != STATE_SELECT_OUTPUT:
            return
        if not self._update_output_name(text):
            self._show_output_step("Enter an output image name before continuing.")
            return
        self._show_output_step("Output image name updated.")

    def _show_output_step(self, status: str | None = None) -> None:
        if self.annotation_image_path is None:
            self._show_annotation_step("Select a valid annotation JSON before continuing.")
            return
        self.state = STATE_SELECT_OUTPUT
        self._clear_dynamic()
        if not self.output_manual:
            self.output_dir = _default_output_dir_for_image(self.annotation_image_path)
        if not self.output_name_manual:
            self.output_name = _default_output_name_for_image(self.annotation_image_path, self.crop)
        self.output_path = self.output_dir / self.output_name
        self._render_image_preview(self.annotation_image_path)
        self._set_phase(
            2,
            "Select Output",
            status or "Confirm output directory and crop setting.",
            "Press Enter in the text field to apply edits. Start runs MaDCoW with the fixed config.",
        )
        self._add_step_heading()
        self._add_summary_line(0.79, "Annotation", self.selected_annotation_path, max_chars=46)
        self._add_summary_line(0.71, "Input image", self.annotation_image_path, max_chars=46)
        self._add_text(SIDEBAR_X, 0.62, "Output directory", fontsize=8.5, weight="bold", color="#555555", ha="left")
        self._output_text_box = self._add_text_box(
            self._display_path(self.output_dir),
            (0.04, 0.575, 0.30, 0.045),
            self._on_output_submit,
        )
        self._add_text(SIDEBAR_X, 0.515, "Output image name", fontsize=8.5, weight="bold", color="#555555", ha="left")
        self._output_name_text_box = self._add_text_box(
            self.output_name,
            (0.04, 0.470, 0.30, 0.045),
            self._on_output_name_submit,
        )
        self._add_summary_line(0.390, "Corrected image", self.output_path, max_chars=46)
        self._add_button("Crop", (0.04, 0.285, 0.14, SIDEBAR_BUTTON_H), self._button_toggle_crop, selected=self.crop)
        self._add_text(SIDEBAR_X, 0.250, "Crop uses MaDCoW's existing validity-mask crop.", fontsize=8.2, color="#555555", ha="left")
        self._add_button(
            "Back",
            (0.04, 0.07, 0.12, SIDEBAR_BUTTON_H),
            lambda _event: self._show_annotation_step() if self.state == STATE_SELECT_OUTPUT else None,
        )
        self._add_button("Start", (0.20, 0.07, 0.14, SIDEBAR_BUTTON_H), self._button_start_run, primary=True)
        self._finish_render()

    def _button_toggle_crop(self, _event: object) -> None:
        if self.state != STATE_SELECT_OUTPUT:
            return
        self.crop = not self.crop
        if self.annotation_image_path is not None and not self.output_name_manual:
            self.output_name = _default_output_name_for_image(self.annotation_image_path, self.crop)
            self.output_path = self.output_dir / self.output_name
        self._show_output_step("Crop enabled." if self.crop else "Crop disabled.")

    def _button_start_run(self, _event: object) -> None:
        if self.state != STATE_SELECT_OUTPUT:
            return
        if not self._update_output_dir(self._text_box_value(self._output_text_box, self._display_path(self.output_dir))):
            self._show_output_step("Enter an output directory before continuing.")
            return
        if not self._update_output_name(self._text_box_value(self._output_name_text_box, self.output_name)):
            self._show_output_step("Enter an output image name before continuing.")
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._show_run_step()
        self._start_worker()

    def _show_run_step(self) -> None:
        self.state = STATE_RUNNING
        self._clear_dynamic()
        self._render_image_preview(self.annotation_image_path)
        self._set_phase(
            3,
            "Run MaDCoW",
            "Running correction pipeline...",
            "Stage progress is updated from MaDCoW optimizer callbacks.",
        )
        self._add_step_heading()
        self._add_summary_line(0.79, "Annotation", self.selected_annotation_path, max_chars=46)
        self._add_summary_line(0.71, "Output", self.output_path, max_chars=46)
        self._add_progress_bar("stage1", "Stage 1 ROI optimization", 0.55)
        self._add_progress_bar("stage2", "Stage 2 warp optimization", 0.42)
        self._add_progress_bar("finalize", "Finalize output", 0.29)
        self._set_progress("stage1", 0, 1, "Waiting")
        self._set_progress("stage2", 0, 1, "Waiting")
        self._set_progress("finalize", 0, 1, "Waiting")
        self._finish_render()

    def _start_worker(self) -> None:
        self._timer = self.fig.canvas.new_timer(interval=150)
        self._timer.add_callback(self._poll_progress)
        self._timer.start()
        self._worker_thread = threading.Thread(target=self._run_pipeline_worker, daemon=True)
        self._worker_thread.start()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _run_pipeline_worker(self) -> None:
        from MaDCoW.main import run_pipeline

        try:
            args = argparse.Namespace(
                image=None,
                annotations=str(self.selected_annotation_path),
                config=self.config.config,
                output=str(self.output_path),
                crop=bool(self.crop),
            )
            run_pipeline(args, progress_callback=self._progress_queue.put)
        except Exception as exc:
            self._progress_queue.put({"event": "error", "message": str(exc)})
            return
        self._progress_queue.put({"event": "complete", "output": str(self.output_path)})

    def _poll_progress(self) -> bool:
        handled_terminal_event = False
        while True:
            try:
                event = self._progress_queue.get_nowait()
            except queue.Empty:
                break
            self._process_progress_event(event)
            if self.state == STATE_COMPLETE:
                handled_terminal_event = True
                break
        if (
            not handled_terminal_event
            and self.state == STATE_RUNNING
            and self._worker_thread is not None
            and not self._worker_thread.is_alive()
            and self.output_path.is_file()
        ):
            self.completed = True
            self._stop_timer()
            self._show_complete_step()
        self.fig.canvas.draw_idle()
        return self.state != STATE_COMPLETE

    def _process_progress_event(self, event: dict[str, object]) -> None:
        event_type = event.get("event")
        if event_type == "complete":
            output = event.get("output")
            if output:
                self.output_path = Path(str(output)).expanduser()
            self.completed = True
            self._stop_timer()
            self._show_complete_step()
            return
        if event_type == "error":
            self.error_message = str(event.get("message", "Unknown MaDCoW error."))
            self._stop_timer()
            self._show_error_step(self.error_message)
            return

        stage = str(event.get("stage", ""))
        message = str(event.get("message", ""))
        if stage in {"stage1", "stage2", "finalize"}:
            self._set_progress(stage, event.get("current", 0), event.get("total", 1), message)
            self._set_status(message)

    def _show_complete_step(self) -> None:
        self.state = STATE_COMPLETE
        self._clear_dynamic()
        self.preview_mode = "corrected"
        self.ax.set_position(PREVIEW_RECT)
        self.ax.set_visible(True)
        self._render_complete_preview()
        self._set_phase(
            4,
            "Complete",
            f"Saved corrected image: {self._display_path(self.output_path)}",
            "Close the window to return to the terminal.",
        )
        self._add_step_heading()
        self._add_summary_line(0.76, "Annotation", self.selected_annotation_path, max_chars=46)
        self._add_summary_line(0.68, "Corrected image", self.output_path, max_chars=46)
        self._add_text(SIDEBAR_X, 0.56, f"Crop: {'enabled' if self.crop else 'disabled'}", fontsize=9, color="#333333", ha="left")
        self._corrected_preview_button = self._add_button(
            "Corrected",
            (0.04, 0.45, 0.14, SIDEBAR_BUTTON_H),
            self._button_show_corrected,
            selected=True,
        )
        self._input_preview_button = self._add_button(
            "Input",
            (0.20, 0.45, 0.14, SIDEBAR_BUTTON_H),
            self._button_show_input,
        )
        self._sync_preview_toggle_labels()
        self._add_button("Close", (0.20, 0.07, 0.14, SIDEBAR_BUTTON_H), self._button_close, primary=True)
        self._finish_render()

    def _render_complete_preview(self) -> None:
        if self.preview_mode == "input":
            self._render_image_preview(self.annotation_image_path, title=f"Input: {self._display_path(self.annotation_image_path)}")
        else:
            self._render_image_preview(self.output_path, title=f"Corrected: {self._display_path(self.output_path)}")

    def _sync_preview_toggle_labels(self) -> None:
        if self._corrected_preview_button is not None:
            set_button_style(self._corrected_preview_button, selected=self.preview_mode == "corrected")
        if self._input_preview_button is not None:
            set_button_style(self._input_preview_button, selected=self.preview_mode == "input")

    def _button_show_corrected(self, _event: object) -> None:
        if self.state != STATE_COMPLETE:
            return
        self.preview_mode = "corrected"
        self._render_complete_preview()
        self._sync_preview_toggle_labels()
        self._set_status(f"Showing corrected image: {self._display_path(self.output_path)}")
        self._finish_render()

    def _button_show_input(self, _event: object) -> None:
        if self.state != STATE_COMPLETE:
            return
        self.preview_mode = "input"
        self._render_complete_preview()
        self._sync_preview_toggle_labels()
        self._set_status(f"Showing input image: {self._display_path(self.annotation_image_path)}")
        self._finish_render()

    def _show_error_step(self, message: str) -> None:
        self.state = STATE_COMPLETE
        self._clear_dynamic()
        self._set_phase(4, "Run Failed", message, "Close the window to return to the terminal.")
        self._add_step_heading()
        self._add_summary_line(0.76, "Annotation", self.selected_annotation_path, max_chars=46)
        self._add_summary_line(0.68, "Output", self.output_path, max_chars=46)
        self._add_text(SIDEBAR_X, 0.54, self._ellipsize(message, max_chars=70), fontsize=8.5, color="#9a3412", ha="left")
        self._add_button("Close", (0.20, 0.07, 0.14, SIDEBAR_BUTTON_H), self._button_close, primary=True)
        self._finish_render()

    def _button_close(self, _event: object) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)


def run_madcow_gui(config: MaDCoWRunnerConfig) -> Path:
    """Open the MaDCoW runner GUI and return the corrected output path."""
    return MaDCoWRunnerGUI(config).run()
