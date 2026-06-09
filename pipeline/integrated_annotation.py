"""Integrated top-level annotation workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from annotation_gui import AnnotationSession
from annotation_gui.base import EmbeddedViewSetupController, create_image_figure, set_image_artist
from annotation_gui.io import build_annotation_payload, write_annotation_json
from interactive_snapping_2d.annotate_line_aid import LineAidAnnotationGUI
from interactive_snapping_2d.snap2d import load_snap_config
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


@dataclass(frozen=True)
class IntegratedAnnotationConfig:
    """Configuration for one integrated annotation run."""

    image: str
    workspace: str
    output_annotation: str
    sam2_checkpoint: str
    sam2_model_cfg: str
    device: str
    snap_config: str | None = None


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
    config: IntegratedAnnotationConfig,
    session: Any,
    output_annotation: Path,
    workspace: Path,
    lines: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the annotation run summary JSON."""
    summary: dict[str, Any] = {
        "input_image": str(Path(config.image).expanduser().resolve()),
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


class IntegratedAnnotationGUI:
    """Single-window setup, ROI, line, and final annotation controller."""

    def __init__(
        self,
        config: IntegratedAnnotationConfig,
        image_path: Path,
        workspace_paths: dict[str, Path],
        output_annotation: Path,
    ) -> None:
        self.config = config
        self.image_path = image_path
        self.paths = workspace_paths
        self.output_annotation = output_annotation
        self.completed = False
        self.session: AnnotationSession | None = None
        self.roi_gui: SAM2ROIAnnotationGUI | None = None
        self.line_gui: LineAidAnnotationGUI | None = None
        self.regions: list[dict[str, Any]] = []
        self.lines: list[dict[str, Any]] = []
        self._setup_connection_ids: list[int] = []

        fov_deg, _source = estimate_fov_from_exif(str(image_path), fallback=FALLBACK_FOV_DEG)
        self.session = AnnotationSession.from_image(image_path, PREVIEW_MAX_SIDE, fov_deg)
        self.fig, self.ax, self.image_artist, self.status, self.help_text = create_image_figure(
            "Integrated Annotation",
            self.session.source_view.preview,
        )
        self.widgets: list[Any] = []
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

    def run(self) -> Path:
        """Run the integrated GUI until the final line phase is saved."""
        import matplotlib.pyplot as plt

        plt.show()
        if not self.completed:
            raise RuntimeError("Integrated annotation GUI closed before final save.")
        return self.output_annotation

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
        self.session = session
        set_image_artist(self.image_artist, self.ax, session.preview)
        self.status.set_text("Loading SAM2 model...")
        self.help_text.set_text("Shared view setup is complete. Preparing ROI annotation.")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        checkpoint = resolve_checkpoint(self.config.sam2_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
        device = resolve_device(self.config.device)
        model_cfg = normalize_model_cfg(self.config.sam2_model_cfg)
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
        self.regions = roi_gui.export_regions(mask_dir=self.paths["masks"], json_base_dir=self.output_annotation.parent)
        roi_gui.detach_for_reuse()

        self.status.set_text("Starting interactive snapping...")
        self.help_text.set_text("ROI annotation is complete. Preparing line annotation.")
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
        )

    def _on_line_done(self, line_gui: LineAidAnnotationGUI) -> None:
        """Write the complete annotation JSON and close the integrated GUI."""
        if self.session is None:
            raise RuntimeError("Annotation session is not initialized.")
        self.lines = line_gui.export_lines(self.output_annotation)
        line_gui.disconnect_events()

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
                    self.config,
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
        self.status.set_text(f"Saved integrated annotation JSON: {self.output_annotation}")
        self.help_text.set_text("Integrated annotation complete.")
        self.fig.canvas.draw_idle()
        import matplotlib.pyplot as plt

        plt.close(self.fig)


def run_integrated_annotation(config: IntegratedAnnotationConfig) -> Path:
    """Run shared setup, SAM2 ROI annotation, snapping line annotation, and save JSON."""
    image_path = Path(config.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    workspace = Path(config.workspace).expanduser().resolve()
    output_annotation = Path(config.output_annotation).expanduser().resolve()
    paths = _make_workspace(workspace)
    output_annotation.parent.mkdir(parents=True, exist_ok=True)

    return IntegratedAnnotationGUI(config, image_path, paths, output_annotation).run()
