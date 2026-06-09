"""Small Matplotlib helpers shared by annotation GUIs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .panorama import render_centered_equirectangular, wrap_angle
from .session import AnnotationSession


STATE_CAMERA_SELECT = "camera_select"
STATE_PINHOLE_SETUP = "pinhole_setup"
STATE_PANORAMA_SETUP = "panorama_setup"
STATE_DONE = "done"
MIN_CROP_FRACTION = 0.08
CROP_EDGE_HIT_PX = 10.0
PANORAMA_MAX_ABS_PITCH = (math.pi / 2.0) - 1e-4
FIGSIZE = (12, 8)
FIG_LEFT = 0.03
FIG_RIGHT = 0.99
FIG_BOTTOM = 0.18
FIG_TOP = 0.92
STATUS_Y = 0.955
HELP_Y = 0.925
BUTTON_ROW_Y = 0.07
BUTTON_HEIGHT = 0.05
BUTTON_GAP = 0.018
TEXT_BOX_Y = 0.075
TEXT_BOX_HEIGHT = 0.045
BUTTON_COLOR = "#f7f7f7"
BUTTON_HOVER_COLOR = "#e6eef8"
BUTTON_SELECTED_COLOR = "#dbeafe"
BUTTON_SELECTED_HOVER_COLOR = "#c7ddff"
BUTTON_PRIMARY_COLOR = "#e8f0ff"
BUTTON_DISABLED_COLOR = "#eeeeee"
BUTTON_DISABLED_TEXT_COLOR = "#999999"
BUTTON_TEXT_COLOR = "#222222"


def set_image_artist(image_artist: Any, ax: Any, image: np.ndarray) -> None:
    """Update an ``imshow`` artist and axes limits for a new image."""
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    image_artist.set_data(arr)
    image_artist.set_extent((-0.5, width - 0.5, height - 0.5, -0.5))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)


def configure_image_axes(ax: Any) -> None:
    """Apply the shared image-axis styling."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def clear_widget_axes(widgets: list[Any]) -> None:
    """Remove all widget axes in a shared control panel."""
    seen_axes: set[int] = set()
    for widget in list(widgets):
        disconnect = getattr(widget, "disconnect_events", None)
        if callable(disconnect):
            try:
                disconnect()
            except (AttributeError, ValueError):
                pass
        ax = getattr(widget, "ax", None)
        if ax is None:
            continue
        ax_id = id(ax)
        if ax_id in seen_axes:
            continue
        seen_axes.add(ax_id)
        try:
            ax.remove()
        except ValueError:
            pass
    widgets.clear()


def _button_colors(selected: bool = False, primary: bool = False, enabled: bool = True) -> tuple[str, str]:
    """Return Matplotlib button colors for a shared visual state."""
    if not enabled:
        return BUTTON_DISABLED_COLOR, BUTTON_DISABLED_COLOR
    if selected:
        return BUTTON_SELECTED_COLOR, BUTTON_SELECTED_HOVER_COLOR
    if primary:
        return BUTTON_PRIMARY_COLOR, BUTTON_SELECTED_HOVER_COLOR
    return BUTTON_COLOR, BUTTON_HOVER_COLOR


def set_button_style(
    button: Any,
    selected: bool = False,
    primary: bool = False,
    enabled: bool = True,
    align: str | None = None,
) -> None:
    """Apply the shared visual state to an existing Matplotlib button."""
    color, hovercolor = _button_colors(selected=selected, primary=primary, enabled=enabled)
    button.color = color
    button.hovercolor = hovercolor
    button.ax.set_facecolor(color)
    button.label.set_color(BUTTON_TEXT_COLOR if enabled else BUTTON_DISABLED_TEXT_COLOR)
    text_align = align or getattr(button, "_annotation_gui_align", "center")
    button._annotation_gui_align = text_align
    if text_align == "left":
        button.label.set_ha("left")
        button.label.set_position((0.04, 0.5))
    else:
        button.label.set_ha("center")
        button.label.set_position((0.5, 0.5))


def add_styled_button(
    fig: Any,
    widgets: list[Any],
    label: str,
    rect: tuple[float, float, float, float],
    callback: Any | None,
    enabled: bool = True,
    selected: bool = False,
    primary: bool = False,
    align: str = "center",
) -> Any:
    """Add one shared-style Matplotlib button."""
    from matplotlib.widgets import Button

    color, hovercolor = _button_colors(selected=selected, primary=primary, enabled=enabled)
    ax_button = fig.add_axes(rect)
    button = Button(ax_button, label, color=color, hovercolor=hovercolor)
    if enabled and callback is not None:
        button.on_clicked(callback)
    else:
        button.set_active(False)
    set_button_style(button, selected=selected, primary=primary, enabled=enabled, align=align)
    widgets.append(button)
    return button


def set_text_box_alignment(text_box: Any, align: str = "left") -> None:
    """Apply horizontal alignment to the editable text in a Matplotlib TextBox."""
    text_disp = getattr(text_box, "text_disp", None)
    if text_disp is None:
        return
    text_disp.set_ha(align)
    if align == "left":
        text_disp.set_position((0.03, 0.5))
    elif align == "right":
        text_disp.set_position((0.97, 0.5))
    else:
        text_disp.set_position((0.5, 0.5))


def add_button_row(
    fig: Any,
    widgets: list[Any],
    buttons: list[tuple[str, float, Any]],
    y0: float = BUTTON_ROW_Y,
    height: float = BUTTON_HEIGHT,
    gap: float = BUTTON_GAP,
    x0: float | None = None,
) -> list[Any]:
    """Add a centered row of buttons and return the created widgets."""
    if not buttons:
        return []
    total_width = sum(width for _label, width, _callback in buttons) + gap * (len(buttons) - 1)
    x0 = (1.0 - total_width) * 0.5 if x0 is None else float(x0)
    created: list[Any] = []
    for label, width, callback in buttons:
        button = add_styled_button(fig, widgets, label, (x0, y0, width, height), callback)
        created.append(button)
        x0 += width + gap
    return created


def add_text_box(
    fig: Any,
    widgets: list[Any],
    label: str,
    initial: str,
    callback: Any,
    width: float = 0.18,
    y0: float = TEXT_BOX_Y,
    height: float = TEXT_BOX_HEIGHT,
    x0: float | None = None,
) -> Any:
    """Add a centered text box."""
    from matplotlib.widgets import TextBox

    x0 = (1.0 - width) * 0.5 if x0 is None else float(x0)
    ax_box = fig.add_axes([x0, y0, width, height])
    text_box = TextBox(ax_box, label, initial=initial, textalignment="left")
    set_text_box_alignment(text_box, "left")
    text_box.on_submit(callback)
    widgets.append(text_box)
    return text_box


def create_image_figure(title: str, initial_image: np.ndarray) -> tuple[Any, Any, Any, Any, Any]:
    """Create a shared Matplotlib figure layout for annotation tools."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.canvas.manager.set_window_title(title)
    fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, bottom=FIG_BOTTOM, top=FIG_TOP)
    image_artist = ax.imshow(initial_image)
    configure_image_axes(ax)
    status = fig.text(FIG_LEFT, STATUS_Y, "", ha="left", va="center", fontsize=10)
    help_text = fig.text(FIG_LEFT, HELP_Y, "", ha="left", va="center", fontsize=9)
    return fig, ax, image_artist, status, help_text


class EmbeddedViewSetupController:
    """Single-window camera/view setup controller embedded in an annotator GUI."""

    def __init__(
        self,
        session: AnnotationSession,
        output_dir: str | Path,
        preview_max_side: int,
        fig: Any,
        ax: Any,
        image_artist: Any,
        status: Any,
        help_text: Any,
        widgets: list[Any],
        clear_dynamic_artists: Any,
        on_done: Any,
        fov_deg: float | None,
        fallback_fov_deg: float = 90.0,
        control_layout: str = "bottom",
    ) -> None:
        self.session = session
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preview_max_side = int(preview_max_side)
        self.fig = fig
        self.ax = ax
        self.image_artist = image_artist
        self.status = status
        self.help_text = help_text
        self.widgets = widgets
        self.clear_dynamic_artists = clear_dynamic_artists
        self.on_done = on_done
        self.fov_deg = fallback_fov_deg if fov_deg is None else float(fov_deg)
        self.control_layout = control_layout
        self.default_fov_deg = float(self.fov_deg)
        self.fallback_fov_deg = float(fallback_fov_deg)
        self.fov_box: Any | None = None
        self._syncing_fov_box = False
        self.state = STATE_CAMERA_SELECT
        self.status_extra = "Choose camera mode."

        source = self.session.source_view
        self.preview_width = source.preview_width
        self.preview_height = source.preview_height
        self.panorama_setup_image: np.ndarray | None = None
        self.panorama_center_lam = 0.0
        self.panorama_center_phi = 0.0
        self.panorama_crop_box = self._default_crop_box()
        self._drag_action: str | None = None
        self._drag_start_xy: tuple[float, float] | None = None
        self._drag_start_center: tuple[float, float] | None = None
        self._drag_start_crop: tuple[float, float, float, float] | None = None
        self._dynamic_artists: list[Any] = []

    @property
    def active(self) -> bool:
        """Return whether setup still owns GUI interactions."""
        return self.state != STATE_DONE

    def start(self) -> None:
        """Show the initial camera selection state."""
        self._show_camera_select()

    def clear_controls(self) -> None:
        """Clear shared controls and reset setup-specific widget handles."""
        clear_widget_axes(self.widgets)
        self.fov_box = None

    def _show_camera_select(self) -> None:
        """Show source preview and camera-choice controls."""
        self.state = STATE_CAMERA_SELECT
        self._clear_dynamic_artists()
        self.clear_controls()
        if self.control_layout == "sidebar":
            add_styled_button(self.fig, self.widgets, "Pinhole", (0.04, 0.36, 0.14, 0.045), self._button_pinhole)
            add_styled_button(self.fig, self.widgets, "Panorama", (0.20, 0.36, 0.14, 0.045), self._button_panorama)
        else:
            add_button_row(
                self.fig,
                self.widgets,
                [
                    ("Pinhole", 0.12, self._button_pinhole),
                    ("Panorama", 0.14, self._button_panorama),
                ],
            )
        set_image_artist(self.image_artist, self.ax, self.session.source_view.preview)
        self.status.set_text("View setup | Choose camera mode.")
        self.help_text.set_text("Preview only; all saved coordinates use original image space.")
        self.fig.canvas.draw_idle()

    def _enter_pinhole_setup(self) -> None:
        """Show pinhole FOV setup."""
        self.state = STATE_PINHOLE_SETUP
        self._clear_dynamic_artists()
        self.clear_controls()
        set_image_artist(self.image_artist, self.ax, self.session.source_view.preview)
        self.fov_box = add_text_box(
            self.fig,
            self.widgets,
            "FOV",
            f"{self.fov_deg:.2f}",
            self._on_fov_submit,
            width=0.30 if self.control_layout == "sidebar" else 0.18,
            y0=0.36 if self.control_layout == "sidebar" else 0.075,
            x0=0.04 if self.control_layout == "sidebar" else 0.38,
        )
        if self.control_layout == "sidebar":
            add_styled_button(
                self.fig,
                self.widgets,
                "Done",
                (0.20, 0.07, 0.14, 0.045),
                self._button_done_pinhole,
                primary=True,
            )
        else:
            add_button_row(self.fig, self.widgets, [("Done", 0.10, self._button_done_pinhole)], y0=0.075, x0=0.60)
        self.status_extra = "Set pinhole horizontal FOV."
        self._refresh()

    def _enter_panorama_setup(self) -> None:
        """Show panorama center/crop setup."""
        self.state = STATE_PANORAMA_SETUP
        self.panorama_center_lam = 0.0
        self.panorama_center_phi = 0.0
        self.panorama_crop_box = self._default_crop_box()
        self.clear_controls()
        if self.control_layout == "sidebar":
            add_styled_button(self.fig, self.widgets, "H Reset", (0.04, 0.36, 0.14, 0.045), self._button_h_reset)
            add_styled_button(self.fig, self.widgets, "V Reset", (0.20, 0.36, 0.14, 0.045), self._button_v_reset)
            add_styled_button(
                self.fig,
                self.widgets,
                "Done",
                (0.20, 0.07, 0.14, 0.045),
                self._button_done_panorama,
                primary=True,
            )
        else:
            add_button_row(
                self.fig,
                self.widgets,
                [
                    ("H Reset", 0.11, self._button_h_reset),
                    ("V Reset", 0.11, self._button_v_reset),
                    ("Done", 0.10, self._button_done_panorama),
                ],
                y0=BUTTON_ROW_Y,
            )
        self._render_panorama_setup_image()
        self.status_extra = "Adjust panorama center and crop range."
        self._refresh()

    def _button_pinhole(self, _event: object) -> None:
        self._enter_pinhole_setup()

    def _button_panorama(self, _event: object) -> None:
        self._enter_panorama_setup()

    def _button_done_pinhole(self, _event: object) -> None:
        if self.state != STATE_PINHOLE_SETUP:
            return
        self.session.use_pinhole_source(self.fov_deg)
        self.state = STATE_DONE
        self.clear_controls()
        self._clear_dynamic_artists()
        self.on_done(self.session)

    def _button_h_reset(self, _event: object) -> None:
        """Reset panorama yaw without changing the crop box."""
        if self.state != STATE_PANORAMA_SETUP:
            return
        self.panorama_center_lam = 0.0
        self._render_panorama_setup_image()
        self.status_extra = "Horizontal view center reset."
        self._refresh()

    def _button_v_reset(self, _event: object) -> None:
        """Reset panorama pitch without changing the crop box."""
        if self.state != STATE_PANORAMA_SETUP:
            return
        self.panorama_center_phi = 0.0
        self._render_panorama_setup_image()
        self.status_extra = "Vertical view center reset."
        self._refresh()

    def _button_done_panorama(self, _event: object) -> None:
        if self.state != STATE_PANORAMA_SETUP:
            return
        self.session.use_panorama_view(
            self.output_dir,
            self.panorama_center_lam,
            self.panorama_center_phi,
            self._normalized_crop_box(),
            self.preview_max_side,
        )
        self.state = STATE_DONE
        self.clear_controls()
        self._clear_dynamic_artists()
        self.on_done(self.session)

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

    def _default_crop_box(self) -> tuple[float, float, float, float]:
        """Return the default panorama crop box in source-preview pixels."""
        return (
            float(self.preview_width) * 0.25,
            0.0,
            float(self.preview_width) * 0.75,
            float(self.preview_height),
        )

    def _min_crop_width(self) -> float:
        return max(8.0, float(self.preview_width) * MIN_CROP_FRACTION)

    def _min_crop_height(self) -> float:
        return max(8.0, float(self.preview_height) * MIN_CROP_FRACTION)

    def _normalized_crop_box(self) -> tuple[float, float, float, float]:
        """Return ordered crop box clipped to source-preview extent."""
        x0, y0, x1, y1 = self.panorama_crop_box
        width = float(self.preview_width)
        height = float(self.preview_height)
        left = float(np.clip(min(x0, x1), 0.0, width))
        right = float(np.clip(max(x0, x1), 0.0, width))
        top = float(np.clip(min(y0, y1), 0.0, height))
        bottom = float(np.clip(max(y0, y1), 0.0, height))
        if right - left < self._min_crop_width():
            center = (left + right) * 0.5
            half = self._min_crop_width() * 0.5
            center = float(np.clip(center, half, width - half))
            left = center - half
            right = center + half
        if bottom - top < self._min_crop_height():
            center = (top + bottom) * 0.5
            half = self._min_crop_height() * 0.5
            center = float(np.clip(center, half, height - half))
            top = center - half
            bottom = center + half
        return left, top, right, bottom

    def _clip_xy(self, x: float, y: float) -> tuple[float, float]:
        return (
            float(np.clip(x, 0.0, self.preview_width - 1.0)),
            float(np.clip(y, 0.0, self.preview_height - 1.0)),
        )

    def _hit_crop_edge(self, x: float, y: float) -> str | None:
        """Return the crop edge under the cursor."""
        left, top, right, bottom = self._normalized_crop_box()
        hits = [
            ("left", abs(x - left), abs(x - left) <= CROP_EDGE_HIT_PX and top - CROP_EDGE_HIT_PX <= y <= bottom + CROP_EDGE_HIT_PX),
            ("right", abs(x - right), abs(x - right) <= CROP_EDGE_HIT_PX and top - CROP_EDGE_HIT_PX <= y <= bottom + CROP_EDGE_HIT_PX),
            ("top", abs(y - top), abs(y - top) <= CROP_EDGE_HIT_PX and left - CROP_EDGE_HIT_PX <= x <= right + CROP_EDGE_HIT_PX),
            ("bottom", abs(y - bottom), abs(y - bottom) <= CROP_EDGE_HIT_PX and left - CROP_EDGE_HIT_PX <= x <= right + CROP_EDGE_HIT_PX),
        ]
        active = [(name, dist) for name, dist, ok in hits if ok]
        if not active:
            return None
        return min(active, key=lambda item: item[1])[0]

    def _update_crop_edge(self, edge: str, x: float, y: float) -> None:
        """Resize crop box by one edge."""
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
        """Pan view center from a preview drag."""
        if self._drag_start_xy is None or self._drag_start_center is None:
            return
        start_x, start_y = self._drag_start_xy
        start_lam, start_phi = self._drag_start_center
        dx = x - start_x
        dy = y - start_y
        self.panorama_center_lam = wrap_angle(start_lam - (dx / max(float(self.preview_width - 1), 1.0)) * 2.0 * math.pi)
        self.panorama_center_phi = float(
            np.clip(
                start_phi - (dy / max(float(self.preview_height - 1), 1.0)) * math.pi,
                -PANORAMA_MAX_ABS_PITCH,
                PANORAMA_MAX_ABS_PITCH,
            )
        )
        self._render_panorama_setup_image()

    def _render_panorama_setup_image(self) -> None:
        """Render setup preview for current center."""
        self.panorama_setup_image = render_centered_equirectangular(
            self.session.source_view.preview,
            self.panorama_center_lam,
            self.panorama_center_phi,
            self.preview_width,
            self.preview_height,
        )

    def _clear_dynamic_artists(self) -> None:
        """Remove setup overlay artists."""
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists.clear()

    def _draw_crop_overlay(self) -> None:
        """Draw dimmed outside-crop overlay."""
        from matplotlib.patches import Rectangle

        left, top, right, bottom = self._normalized_crop_box()
        width = float(self.preview_width)
        height = float(self.preview_height)
        for x, y, w, h in [
            (0.0, 0.0, width, top),
            (0.0, bottom, width, height - bottom),
            (0.0, top, left, bottom - top),
            (right, top, width - right, bottom - top),
        ]:
            if w <= 0.0 or h <= 0.0:
                continue
            rect = Rectangle((x, y), w, h, facecolor="black", alpha=0.45, edgecolor="none", zorder=4)
            self.ax.add_patch(rect)
            self._dynamic_artists.append(rect)
        rect = Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor="#ffcc00", linewidth=1.2, zorder=5)
        self.ax.add_patch(rect)
        self._dynamic_artists.append(rect)

    def _refresh(self) -> None:
        """Redraw setup state."""
        self._clear_dynamic_artists()
        if self.state == STATE_PINHOLE_SETUP:
            set_image_artist(self.image_artist, self.ax, self.session.source_view.preview)
            self.status.set_text(f"View setup | Pinhole FOV: {self.fov_deg:.2f} deg | {self.status_extra}")
            self.help_text.set_text("Type horizontal FOV, then press Done.")
            self.fig.canvas.draw_idle()
            return
        if self.state == STATE_PANORAMA_SETUP:
            if self.panorama_setup_image is None:
                self._render_panorama_setup_image()
            if self.panorama_setup_image is None:
                raise RuntimeError("Panorama setup image is not initialized.")
            set_image_artist(self.image_artist, self.ax, self.panorama_setup_image)
            self._draw_crop_overlay()
            center_deg = (math.degrees(self.panorama_center_lam), math.degrees(self.panorama_center_phi))
            self.status.set_text(
                f"View setup | Panorama center: ({center_deg[0]:.1f}, {center_deg[1]:.1f}) | {self.status_extra}"
            )
            self.help_text.set_text("Drag image to move center. Drag crop edges to set the derived view. H/V Reset recalibrates center.")
        self.fig.canvas.draw_idle()

    def on_click(self, event: object) -> bool:
        """Handle setup mouse press. Returns true when consumed."""
        if self.state != STATE_PANORAMA_SETUP:
            return False
        if getattr(event, "inaxes", None) is not self.ax:
            return False
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return False
        if getattr(event, "button", None) != 1:
            return False
        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        edge = self._hit_crop_edge(x, y)
        self._drag_action = f"edge:{edge}" if edge is not None else "pan"
        self._drag_start_xy = (x, y)
        self._drag_start_center = (self.panorama_center_lam, self.panorama_center_phi)
        self._drag_start_crop = self._normalized_crop_box()
        return True

    def on_motion(self, event: object) -> bool:
        """Handle setup mouse drag. Returns true when consumed."""
        if self.state != STATE_PANORAMA_SETUP or self._drag_action is None:
            return False
        if getattr(event, "inaxes", None) is not self.ax:
            return False
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return False
        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        if self._drag_action == "pan":
            self._update_panorama_center(x, y)
        elif self._drag_action.startswith("edge:"):
            self._update_crop_edge(self._drag_action.split(":", 1)[1], x, y)
        self._refresh()
        return True

    def on_release(self, _event: object) -> bool:
        """Finish setup drag. Returns true when consumed."""
        if self.state != STATE_PANORAMA_SETUP or self._drag_action is None:
            return False
        self._drag_action = None
        self._drag_start_xy = None
        self._drag_start_center = None
        self._drag_start_crop = None
        self._refresh()
        return True

    def on_key(self, event: object) -> bool:
        """Handle setup keyboard shortcuts. Returns true when consumed."""
        key = (getattr(event, "key", "") or "").lower()
        if self.state == STATE_CAMERA_SELECT:
            if key in ("p", "enter"):
                self._button_pinhole(event)
                return True
        elif self.state == STATE_PINHOLE_SETUP and key in ("enter", " "):
            self._button_done_pinhole(event)
            return True
        elif self.state == STATE_PANORAMA_SETUP:
            if key in ("h",):
                self._button_h_reset(event)
                return True
            if key in ("v",):
                self._button_v_reset(event)
                return True
            if key in ("enter", " "):
                self._button_done_panorama(event)
                return True
        return False


class ViewSetupGUI:
    """Shared camera/view setup GUI that returns an original-resolution session."""

    def __init__(
        self,
        image_path: str | Path,
        fov_deg: float | None,
        output_dir: str | Path,
        preview_max_side: int,
        fallback_fov_deg: float = 90.0,
        title: str = "Annotation View Setup",
    ) -> None:
        self.session = AnnotationSession.from_image(image_path, preview_max_side, fov_deg)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preview_max_side = int(preview_max_side)
        self.fov_deg = fallback_fov_deg if fov_deg is None else float(fov_deg)
        self.default_fov_deg = float(self.fov_deg)
        self.fallback_fov_deg = float(fallback_fov_deg)
        self.fov_box: Any | None = None
        self._syncing_fov_box = False
        self.title = title
        self.state = STATE_CAMERA_SELECT
        self.selected = False
        self.status_extra = "Choose a camera mode."

        source = self.session.source_view
        self.preview_width = source.preview_width
        self.preview_height = source.preview_height
        self.blank_image = np.full((max(1, self.preview_height), max(1, self.preview_width), 3), 245, dtype=np.uint8)
        self.panorama_setup_image: np.ndarray | None = None
        self.panorama_center_lam = 0.0
        self.panorama_center_phi = 0.0
        self.panorama_crop_box = self._default_crop_box()
        self._drag_action: str | None = None
        self._drag_start_xy: tuple[float, float] | None = None
        self._drag_start_center: tuple[float, float] | None = None
        self._drag_start_crop: tuple[float, float, float, float] | None = None
        self._dynamic_artists: list[Any] = []
        self.widgets: list[Any] = []

        self._build_figure()
        self._show_camera_select()

    def run(self) -> AnnotationSession:
        """Run the setup GUI and return the selected annotation session."""
        import matplotlib.pyplot as plt

        plt.show()
        if not self.selected:
            self.session.use_pinhole_source(self.fov_deg)
        return self.session

    def _build_figure(self) -> None:
        """Create the setup figure."""
        self.fig, self.ax, self.image_artist, self.status, self.help_text = create_image_figure(
            self.title,
            self.blank_image,
        )
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _clear_controls(self) -> None:
        """Remove state controls."""
        seen_axes: set[int] = set()
        for widget in list(self.widgets):
            ax = getattr(widget, "ax", None)
            if ax is None:
                continue
            ax_id = id(ax)
            if ax_id in seen_axes:
                continue
            seen_axes.add(ax_id)
            try:
                ax.remove()
            except ValueError:
                pass
        self.widgets.clear()
        self.fov_box = None

    def _set_controls(self, specs: list[tuple[str, float, float, float, Any]]) -> None:
        """Create a button row for the current state."""
        self._clear_controls()
        for label, x0, y0, width, callback in specs:
            add_button_row(self.fig, self.widgets, [(label, width, callback)], y0=y0, height=BUTTON_HEIGHT, x0=x0)

    def _show_camera_select(self) -> None:
        """Show source preview and camera-choice controls."""
        self.state = STATE_CAMERA_SELECT
        self._clear_dynamic_artists()
        self._set_controls(
            [
                ("Pinhole", 0.35, 0.07, 0.12, self._button_pinhole),
                ("Panorama", 0.53, 0.07, 0.14, self._button_panorama),
            ]
        )
        set_image_artist(self.image_artist, self.ax, self.session.source_view.preview)
        self.status.set_text("View setup | Choose camera mode.")
        self.help_text.set_text("Pinhole opens FOV setup. Panorama opens view/crop setup. Stored state uses original pixels.")
        self.fig.canvas.draw_idle()

    def _enter_pinhole_setup(self) -> None:
        """Show pinhole FOV setup before annotation starts."""
        self.state = STATE_PINHOLE_SETUP
        self._clear_dynamic_artists()
        self._clear_controls()
        set_image_artist(self.image_artist, self.ax, self.session.source_view.preview)

        self.fov_box = add_text_box(
            self.fig,
            self.widgets,
            "FOV",
            f"{self.fov_deg:.2f}",
            self._on_fov_submit,
            width=0.18,
            y0=0.075,
            x0=0.38,
        )
        add_button_row(self.fig, self.widgets, [("Done", 0.10, self._button_done_pinhole)], y0=0.075, x0=0.60)

        self.status_extra = "Adjust pinhole horizontal FOV."
        self._refresh()

    def _enter_panorama_setup(self) -> None:
        """Show panorama center/crop setup."""
        self.state = STATE_PANORAMA_SETUP
        self.panorama_center_lam = 0.0
        self.panorama_center_phi = 0.0
        self.panorama_crop_box = self._default_crop_box()
        self._set_controls(
            [
                ("H Reset", 0.32, 0.07, 0.11, self._button_h_reset),
                ("V Reset", 0.45, 0.07, 0.11, self._button_v_reset),
                ("Done", 0.58, 0.07, 0.10, self._button_done),
            ]
        )
        self._render_panorama_setup_image()
        self.status_extra = "Adjust panorama center and crop range."
        self._refresh()

    def _button_pinhole(self, _event: object) -> None:
        """Open pinhole FOV setup."""
        self._enter_pinhole_setup()

    def _button_done_pinhole(self, _event: object) -> None:
        """Select the original pinhole image with the configured FOV."""
        if self.state != STATE_PINHOLE_SETUP:
            return
        self.session.use_pinhole_source(self.fov_deg)
        self.selected = True
        self.state = STATE_DONE
        import matplotlib.pyplot as plt

        plt.close(self.fig.number)

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

    def _button_panorama(self, _event: object) -> None:
        """Open panorama setup."""
        self._enter_panorama_setup()

    def _button_h_reset(self, _event: object) -> None:
        """Reset panorama yaw without changing the crop box."""
        if self.state != STATE_PANORAMA_SETUP:
            return
        self.panorama_center_lam = 0.0
        self._render_panorama_setup_image()
        self.status_extra = "Horizontal view center reset."
        self._refresh()

    def _button_v_reset(self, _event: object) -> None:
        """Reset panorama pitch without changing the crop box."""
        if self.state != STATE_PANORAMA_SETUP:
            return
        self.panorama_center_phi = 0.0
        self._render_panorama_setup_image()
        self.status_extra = "Vertical view center reset."
        self._refresh()

    def _button_done(self, _event: object) -> None:
        """Render the original-resolution panorama-derived view and close."""
        if self.state != STATE_PANORAMA_SETUP:
            return
        self.session.use_panorama_view(
            self.output_dir,
            self.panorama_center_lam,
            self.panorama_center_phi,
            self._normalized_crop_box(),
            self.preview_max_side,
        )
        self.selected = True
        self.state = STATE_DONE
        import matplotlib.pyplot as plt

        plt.close(self.fig.number)

    def _default_crop_box(self) -> tuple[float, float, float, float]:
        """Return the default panorama crop box in source-preview pixels."""
        return (
            float(self.preview_width) * 0.25,
            0.0,
            float(self.preview_width) * 0.75,
            float(self.preview_height),
        )

    def _min_crop_width(self) -> float:
        return max(8.0, float(self.preview_width) * MIN_CROP_FRACTION)

    def _min_crop_height(self) -> float:
        return max(8.0, float(self.preview_height) * MIN_CROP_FRACTION)

    def _normalized_crop_box(self) -> tuple[float, float, float, float]:
        """Return ordered crop box clipped to source-preview extent."""
        x0, y0, x1, y1 = self.panorama_crop_box
        width = float(self.preview_width)
        height = float(self.preview_height)
        left = float(np.clip(min(x0, x1), 0.0, width))
        right = float(np.clip(max(x0, x1), 0.0, width))
        top = float(np.clip(min(y0, y1), 0.0, height))
        bottom = float(np.clip(max(y0, y1), 0.0, height))
        if right - left < self._min_crop_width():
            center = (left + right) * 0.5
            half = self._min_crop_width() * 0.5
            center = float(np.clip(center, half, width - half))
            left = center - half
            right = center + half
        if bottom - top < self._min_crop_height():
            center = (top + bottom) * 0.5
            half = self._min_crop_height() * 0.5
            center = float(np.clip(center, half, height - half))
            top = center - half
            bottom = center + half
        return left, top, right, bottom

    def _clip_xy(self, x: float, y: float) -> tuple[float, float]:
        return (
            float(np.clip(x, 0.0, self.preview_width - 1.0)),
            float(np.clip(y, 0.0, self.preview_height - 1.0)),
        )

    def _hit_crop_edge(self, x: float, y: float) -> str | None:
        """Return the crop edge under the cursor."""
        left, top, right, bottom = self._normalized_crop_box()
        hits = [
            ("left", abs(x - left), abs(x - left) <= CROP_EDGE_HIT_PX and top - CROP_EDGE_HIT_PX <= y <= bottom + CROP_EDGE_HIT_PX),
            ("right", abs(x - right), abs(x - right) <= CROP_EDGE_HIT_PX and top - CROP_EDGE_HIT_PX <= y <= bottom + CROP_EDGE_HIT_PX),
            ("top", abs(y - top), abs(y - top) <= CROP_EDGE_HIT_PX and left - CROP_EDGE_HIT_PX <= x <= right + CROP_EDGE_HIT_PX),
            ("bottom", abs(y - bottom), abs(y - bottom) <= CROP_EDGE_HIT_PX and left - CROP_EDGE_HIT_PX <= x <= right + CROP_EDGE_HIT_PX),
        ]
        active = [(name, dist) for name, dist, ok in hits if ok]
        if not active:
            return None
        return min(active, key=lambda item: item[1])[0]

    def _update_crop_edge(self, edge: str, x: float, y: float) -> None:
        """Resize crop box by one edge."""
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
        """Pan view center from a preview drag."""
        if self._drag_start_xy is None or self._drag_start_center is None:
            return
        start_x, start_y = self._drag_start_xy
        start_lam, start_phi = self._drag_start_center
        dx = x - start_x
        dy = y - start_y
        self.panorama_center_lam = wrap_angle(start_lam - (dx / max(float(self.preview_width - 1), 1.0)) * 2.0 * math.pi)
        self.panorama_center_phi = float(
            np.clip(
                start_phi - (dy / max(float(self.preview_height - 1), 1.0)) * math.pi,
                -PANORAMA_MAX_ABS_PITCH,
                PANORAMA_MAX_ABS_PITCH,
            )
        )
        self._render_panorama_setup_image()

    def _render_panorama_setup_image(self) -> None:
        """Render setup preview for current center."""
        self.panorama_setup_image = render_centered_equirectangular(
            self.session.source_view.preview,
            self.panorama_center_lam,
            self.panorama_center_phi,
            self.preview_width,
            self.preview_height,
        )

    def _clear_dynamic_artists(self) -> None:
        """Remove overlay artists."""
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists.clear()

    def _draw_crop_overlay(self) -> None:
        """Draw dimmed outside-crop overlay."""
        from matplotlib.patches import Rectangle

        left, top, right, bottom = self._normalized_crop_box()
        width = float(self.preview_width)
        height = float(self.preview_height)
        for x, y, w, h in [
            (0.0, 0.0, width, top),
            (0.0, bottom, width, height - bottom),
            (0.0, top, left, bottom - top),
            (right, top, width - right, bottom - top),
        ]:
            if w <= 0.0 or h <= 0.0:
                continue
            rect = Rectangle((x, y), w, h, facecolor="black", alpha=0.45, edgecolor="none", zorder=4)
            self.ax.add_patch(rect)
            self._dynamic_artists.append(rect)
        rect = Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor="#ffcc00", linewidth=1.2, zorder=5)
        self.ax.add_patch(rect)
        self._dynamic_artists.append(rect)

    def _refresh(self) -> None:
        """Redraw current setup state."""
        self._clear_dynamic_artists()
        if self.state == STATE_PINHOLE_SETUP:
            set_image_artist(self.image_artist, self.ax, self.session.source_view.preview)
            self.status.set_text(f"View setup | Pinhole FOV: {self.fov_deg:.2f} deg | {self.status_extra}")
            self.help_text.set_text("Type horizontal FOV, then press Done.")
            self.fig.canvas.draw_idle()
            return
        if self.state == STATE_PANORAMA_SETUP:
            if self.panorama_setup_image is None:
                self._render_panorama_setup_image()
            if self.panorama_setup_image is None:
                raise RuntimeError("Panorama setup image is not initialized.")
            set_image_artist(self.image_artist, self.ax, self.panorama_setup_image)
            self._draw_crop_overlay()
            center_deg = (math.degrees(self.panorama_center_lam), math.degrees(self.panorama_center_phi))
            self.status.set_text(
                f"View setup | Panorama center: ({center_deg[0]:.1f}, {center_deg[1]:.1f}) | {self.status_extra}"
            )
            self.help_text.set_text(
                "Drag image to move center. Drag crop edges to set the original-resolution derived view. H/V Reset recalibrates center."
            )
        self.fig.canvas.draw_idle()

    def _on_click(self, event: object) -> None:
        """Handle setup mouse press."""
        if self.state != STATE_PANORAMA_SETUP:
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        if getattr(event, "button", None) != 1:
            return
        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        edge = self._hit_crop_edge(x, y)
        self._drag_action = f"edge:{edge}" if edge is not None else "pan"
        self._drag_start_xy = (x, y)
        self._drag_start_center = (self.panorama_center_lam, self.panorama_center_phi)
        self._drag_start_crop = self._normalized_crop_box()

    def _on_motion(self, event: object) -> None:
        """Handle setup drag."""
        if self.state != STATE_PANORAMA_SETUP or self._drag_action is None:
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        x, y = self._clip_xy(float(event.xdata), float(event.ydata))
        if self._drag_action == "pan":
            self._update_panorama_center(x, y)
        elif self._drag_action.startswith("edge:"):
            self._update_crop_edge(self._drag_action.split(":", 1)[1], x, y)
        self._refresh()

    def _on_release(self, _event: object) -> None:
        """Finish setup drag."""
        self._drag_action = None
        self._drag_start_xy = None
        self._drag_start_center = None
        self._drag_start_crop = None
        if self.state == STATE_PANORAMA_SETUP:
            self._refresh()

    def _on_key(self, event: object) -> None:
        """Handle setup keyboard shortcuts."""
        key = (getattr(event, "key", "") or "").lower()
        if self.state == STATE_CAMERA_SELECT:
            if key in ("p", "enter"):
                self._button_pinhole(event)
        elif self.state == STATE_PINHOLE_SETUP:
            if key in ("enter", " "):
                self._button_done_pinhole(event)
        elif self.state == STATE_PANORAMA_SETUP:
            if key in ("h",):
                self._button_h_reset(event)
            elif key in ("v",):
                self._button_v_reset(event)
            elif key in ("enter", " "):
                self._button_done(event)


def run_view_setup(
    image_path: str | Path,
    fov_deg: float | None,
    output_dir: str | Path,
    preview_max_side: int,
    fallback_fov_deg: float = 90.0,
    title: str = "Annotation View Setup",
) -> AnnotationSession:
    """Run the shared view setup GUI and return the selected session."""
    return ViewSetupGUI(
        image_path=image_path,
        fov_deg=fov_deg,
        output_dir=output_dir,
        preview_max_side=preview_max_side,
        fallback_fov_deg=fallback_fov_deg,
        title=title,
    ).run()
