"""Shared original-space annotation GUI helpers."""

from .image import ImageView, load_image_view, relative_path
from .panorama import PanoramaViewResult, build_panorama_view, render_centered_equirectangular
from .session import AnnotationSession
from .base import (
    EmbeddedViewSetupController,
    add_button_row,
    add_text_box,
    clear_widget_axes,
    configure_image_axes,
    create_image_figure,
    run_view_setup,
)

__all__ = [
    "AnnotationSession",
    "ImageView",
    "PanoramaViewResult",
    "EmbeddedViewSetupController",
    "add_button_row",
    "add_text_box",
    "build_panorama_view",
    "clear_widget_axes",
    "configure_image_axes",
    "create_image_figure",
    "load_image_view",
    "relative_path",
    "render_centered_equirectangular",
    "run_view_setup",
]
