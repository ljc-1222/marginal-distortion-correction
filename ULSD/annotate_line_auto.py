import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image, ImageDraw

from annotation_gui.base import add_button_row, create_image_figure
from annotation_gui.io import build_annotation_payload, write_annotation_json
from annotation_gui.panorama import full_panorama_view_metadata


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp") / "ulsd_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

from MaDCoW.src import CameraConfig
from MaDCoW.src.camera import Camera
from network.dataset import Dataset
from network.ulsd import ULSD
from util import bezier as bez


LINE_SAMPLE_POINTS = 128
LINE_MIN_SPACING_PX = 3.0
PREVIEW_MAX_SIDE = 1200
HIT_TOLERANCE_PX = 8.0
DEFAULT_IMAGE = BASE_DIR / "data" / "test_1.jpg"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data"
DEFAULT_SELECTION_CONFIG = BASE_DIR / "config" / "line_selection.json"
PANORAMA_CAMERA_MODEL = "panorama_view"
PANORAMA_MODEL_NAME = "spherical.pkl"

CAMERA_MODEL_LABELS = {
    "panorama_view": "Panorama",
}


@dataclass
class SelectionConfig:
    score_thresh: float
    top_k: int
    duplicate_dist_px: float


@dataclass
class LineCandidate:
    line_heatmap: np.ndarray
    line_original: np.ndarray
    original_points: np.ndarray
    preview_points: np.ndarray
    score: float
    source: str = "auto"
    selected: bool = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ULSD, review detected lines, and export MaDCoW annotations."
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path.")
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        default="",
        help=f"ULSD model filename. Defaults to {PANORAMA_MODEL_NAME}.",
    )
    parser.add_argument("--order", type=int, default=4, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id. Use -1 for CPU.")
    parser.add_argument("--config_path", "--config-path", default="config", help="ULSD config folder.")
    parser.add_argument("--config_file", "--config-file", default="default.yaml", help="ULSD config filename.")
    parser.add_argument("--selection-config", default=str(DEFAULT_SELECTION_CONFIG), help="Line selection JSON path.")
    parser.add_argument("--output_dir", "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--marked_image", "--marked-image", default="", help="Marked image output path.")
    parser.add_argument("--json", default="", help="MaDCoW JSON output path.")

    parser.add_argument("--score_thresh", "--score-thresh", dest="score_thresh", type=float, default=None)
    parser.add_argument("--top_k", "--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--duplicate_dist_px", "--duplicate-dist-px", dest="duplicate_dist_px", type=float, default=None)
    parser.add_argument("--junc_score_thresh", "--junc-score-thresh", dest="junc_score_thresh", type=float, default=None)
    parser.add_argument("--line_score_thresh", "--line-score-thresh", dest="line_score_thresh", type=float, default=None)

    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=0)
    return parser.parse_args()


def _lanczos_resampling() -> int:
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.LANCZOS)
    return int(Image.LANCZOS)


def _compute_preview_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}.")
    if max_side <= 0:
        return width, height
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = float(max_side) / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), base_dir.resolve())


def _resolve_existing_or_base(path: str | Path, base_dir: Path) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    cwd_candidate = path_obj.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / path_obj).resolve()


def _json_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _json_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")
    return int(value)


def load_selection_config(
    path: str | Path,
    score_override: float | None,
    top_k_override: int | None,
    duplicate_dist_override: float | None,
) -> SelectionConfig:
    config_path = _resolve_existing_or_base(path, BASE_DIR)
    if not config_path.is_file():
        raise FileNotFoundError(f"selection config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("selection config must be a JSON object.")

    if "score_thresh" not in data:
        raise ValueError("selection config must contain score_thresh.")
    if "top_k" not in data:
        raise ValueError("selection config must contain top_k.")

    score_thresh = _json_number(data["score_thresh"], "selection_config.score_thresh")
    if score_override is not None:
        score_thresh = _json_number(score_override, "--score_thresh")
    if score_thresh < 0.0 or score_thresh > 1.0:
        raise ValueError(f"score_thresh must be in [0, 1]; got {score_thresh}.")

    top_k = _json_nonnegative_int(data["top_k"], "selection_config.top_k")
    if top_k_override is not None:
        top_k = _json_nonnegative_int(top_k_override, "--top_k")

    duplicate_dist_px = _json_number(data.get("duplicate_dist_px", 6.0), "selection_config.duplicate_dist_px")
    if duplicate_dist_override is not None:
        duplicate_dist_px = _json_number(duplicate_dist_override, "--duplicate_dist_px")
    if duplicate_dist_px < 0.0:
        raise ValueError(f"duplicate_dist_px must be >= 0; got {duplicate_dist_px}.")

    return SelectionConfig(
        score_thresh=score_thresh,
        top_k=top_k,
        duplicate_dist_px=duplicate_dist_px,
    )


def model_name_for_camera(model_override: str = "") -> str:
    if model_override:
        return model_override
    return PANORAMA_MODEL_NAME


def load_cfg(args: argparse.Namespace, model_name: str, selection: SelectionConfig):
    from config.cfg import PATH_KEYS, resolve_path
    from yacs.config import CfgNode

    config_path = _resolve_existing_or_base(args.config_path, BASE_DIR)
    yaml_file = config_path / args.config_file

    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = CfgNode.load_cfg(f)

    cfg.defrost()
    cfg.dataset_name = str(Path(args.image).resolve())
    cfg.order = args.order
    cfg.gpu = args.gpu
    cfg.model_name = model_name
    cfg.version = ".".join(cfg.model_name.split(".")[:-1])
    cfg.config_path = str(config_path)
    cfg.config_file = args.config_file
    cfg.test_dataset_path = str(Path(args.image).resolve())
    cfg.groundtruth_path = str(Path(args.image).resolve().parent)
    cfg.output_path = str(Path(args.output_dir).resolve())
    cfg.figure_path = str(Path(args.output_dir).resolve())
    cfg.log_path = os.path.join(cfg.log_path, cfg.version)
    cfg.image_size = tuple(cfg.image_size)
    cfg.heatmap_size = tuple(cfg.heatmap_size)
    cfg.score_thresh = float(selection.score_thresh)
    if args.junc_score_thresh is not None:
        cfg.junc_score_thresh = float(args.junc_score_thresh)
    if args.line_score_thresh is not None:
        cfg.line_score_thresh = float(args.line_score_thresh)

    for key in PATH_KEYS:
        cfg[key] = resolve_path(cfg[key])
    cfg.freeze()
    return cfg


def rescale_lines_to_image(lines: np.ndarray, image_shape: tuple[int, int], cfg) -> np.ndarray:
    lines = lines.astype(np.float64, copy=True)
    height, width = image_shape
    sx = width / cfg.heatmap_size[0]
    sy = height / cfg.heatmap_size[1]
    lines[:, :, 0] *= sx
    lines[:, :, 1] *= sy
    return lines


def choose_lines(line_pred: np.ndarray, line_score: np.ndarray, score_thresh: float, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    keep = line_score > score_thresh
    lines = line_pred[keep]
    scores = line_score[keep]
    if top_k > 0 and len(scores) > top_k:
        order = np.argsort(scores)[::-1][:top_k]
        lines = lines[order]
        scores = scores[order]
    return lines, scores


def _point_to_polyline_distance(x: float, y: float, points: np.ndarray) -> float:
    if len(points) < 2:
        return float("inf")
    p = np.array([x, y], dtype=np.float64)
    a = points[:-1].astype(np.float64, copy=False)
    b = points[1:].astype(np.float64, copy=False)
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    t = np.sum((p - a) * ab, axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    dists = np.linalg.norm(closest - p, axis=1)
    return float(dists.min())


def _polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _resample_polyline(
    points: list[tuple[float, float]],
    n_samples: int = LINE_SAMPLE_POINTS,
) -> list[tuple[float, float]]:
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


def _matching_line_distance(points_a: np.ndarray, points_b: np.ndarray) -> tuple[float, float]:
    forward = np.linalg.norm(points_a - points_b, axis=1)
    backward = np.linalg.norm(points_a - points_b[::-1], axis=1)
    dists = backward if backward.mean() < forward.mean() else forward
    return float(dists.mean()), float(dists.max())


def suppress_duplicate_candidates(candidates: list[LineCandidate], duplicate_dist_px: float) -> list[LineCandidate]:
    if duplicate_dist_px <= 0.0 or len(candidates) <= 1:
        return candidates

    kept: list[LineCandidate] = []
    max_dist_px = duplicate_dist_px * 3.0
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = False
        for kept_candidate in kept:
            mean_dist, max_dist = _matching_line_distance(candidate.preview_points, kept_candidate.preview_points)
            if mean_dist <= duplicate_dist_px and max_dist <= max_dist_px:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _read_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


class AutoLineReviewGUI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.image_path = Path(args.image).resolve()
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = Path(args.json).resolve() if args.json else self.output_dir / f"{self.image_path.stem}.json"
        self.marked_image_path = (
            Path(args.marked_image).resolve()
            if args.marked_image
            else self.output_dir / f"{self.image_path.stem}_lines.jpg"
        )
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.marked_image_path.parent.mkdir(parents=True, exist_ok=True)

        self.camera_model = PANORAMA_CAMERA_MODEL
        self.detected_camera_model = self.camera_model
        self.selection_config_path = Path(args.selection_config)
        self.score_override = args.score_thresh
        self.top_k_override = args.top_k
        self.duplicate_dist_override = args.duplicate_dist_px
        self.model_override = args.model_name
        self.selection = load_selection_config(
            self.selection_config_path,
            self.score_override,
            self.top_k_override,
            self.duplicate_dist_override,
        )

        self.original_image = _read_rgb_image(self.image_path)
        self.original_height, self.original_width = self.original_image.shape[:2]
        preview_size = _compute_preview_size(self.original_width, self.original_height, PREVIEW_MAX_SIDE)
        if preview_size == (self.original_width, self.original_height):
            self.preview_image = self.original_image
        else:
            preview = Image.fromarray(self.original_image).resize(preview_size, _lanczos_resampling())
            self.preview_image = np.asarray(preview)
        self.preview_height, self.preview_width = self.preview_image.shape[:2]
        self.view_metadata = full_panorama_view_metadata(
            self.original_width,
            self.original_height,
            preview_size=(self.preview_width, self.preview_height),
        )

        self.candidates: list[LineCandidate] = []
        self.history: list[tuple[str, LineCandidate]] = []
        self.mode = "review"
        self._drawing_line = False
        self.current_line_points: list[tuple[float, float]] = []
        self.current_model_name = ""
        self.use_gpu = False
        self.needs_rerun = False
        self.status_extra = ""
        self._dynamic_artists: list[Any] = []
        self._detect_candidates()
        self._build_figure()
        self._refresh()

    def run(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()

    def _build_figure(self) -> None:
        self.fig, self.ax, self.image_artist, self.status, self.help_text = create_image_figure(
            "ULSD Line Review Tool",
            self.preview_image,
        )
        self.ax.set_xlim(-0.5, self.preview_width - 0.5)
        self.ax.set_ylim(self.preview_height - 0.5, -0.5)
        self.help_text.set_text("Review auto candidates or draw additional lines; saved coordinates use original image space.")

        self.widgets: list[Any] = []
        add_button_row(
            self.fig,
            self.widgets,
            [
                ("Review", 0.085, self._button_review),
                ("Draw", 0.075, self._button_draw),
                ("Select All", 0.105, self._button_select_all),
                ("Drop All", 0.095, self._button_drop_all),
                ("Invert", 0.075, self._button_invert),
                ("Rerun", 0.075, self._button_rerun),
            ],
            0.045,
        )
        add_button_row(
            self.fig,
            self.widgets,
            [
                ("Undo", 0.075, self._button_undo),
                ("Save", 0.075, self._button_save),
                ("Save+Close", 0.120, self._button_save_close),
            ],
            0.105,
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_draw)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _detect_candidates(self) -> None:
        manual_candidates = [
            candidate for candidate in getattr(self, "candidates", [])
            if candidate.source == "manual"
        ]
        self.selection = load_selection_config(
            self.selection_config_path,
            self.score_override,
            self.top_k_override,
            self.duplicate_dist_override,
        )
        self.current_model_name = model_name_for_camera(self.model_override)
        cfg = load_cfg(self.args, self.current_model_name, self.selection)
        self.use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
        device = torch.device(f"cuda:{cfg.gpu}" if self.use_gpu else "cpu")

        model = ULSD(cfg).to(device)
        model_filename = Path(cfg.model_path) / cfg.model_name
        checkpoint = torch.load(model_filename, map_location=device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint.keys() else checkpoint
        model.load_state_dict(state_dict)
        model.lpn.junc_score_thresh = float(cfg.junc_score_thresh)
        model.lpn.line_score_thresh = float(cfg.line_score_thresh)
        model.eval()

        dataset = Dataset(str(self.image_path), cfg, with_label=False)
        if len(dataset) == 0:
            raise FileNotFoundError(f"No supported image found at {self.image_path}")
        loader = Data.DataLoader(dataset=dataset, batch_size=1, num_workers=self.args.num_workers, shuffle=False)

        with torch.no_grad():
            images = next(iter(loader)).to(device)
            _, _, line_preds, line_scores = model(images)
            line_pred = line_preds[0].detach().cpu().numpy()
            line_score = line_scores[0].detach().cpu().numpy()

        selected_lines, selected_scores = choose_lines(
            line_pred=line_pred,
            line_score=line_score,
            score_thresh=self.selection.score_thresh,
            top_k=self.selection.top_k,
        )
        image_shape = (self.original_height, self.original_width)
        selected_lines_original = rescale_lines_to_image(selected_lines, image_shape, cfg)

        candidates: list[LineCandidate] = []
        for line_heatmap, line_original, score in zip(selected_lines, selected_lines_original, selected_scores):
            samples_original = bez.interp_line(line_original[None, :, :], num=LINE_SAMPLE_POINTS)[0]
            preview_points = self._original_to_preview_points(samples_original)
            candidates.append(
                LineCandidate(
                    line_heatmap=line_heatmap.astype(np.float64, copy=True),
                    line_original=line_original.astype(np.float64, copy=True),
                    original_points=samples_original.astype(np.float64, copy=True),
                    preview_points=preview_points,
                    score=float(score),
                    source="auto",
                    selected=True,
                )
            )

        raw_count = len(candidates)
        candidates = suppress_duplicate_candidates(candidates, self.selection.duplicate_dist_px)

        removed = raw_count - len(candidates)
        self.candidates = candidates + manual_candidates
        self.detected_camera_model = self.camera_model
        self.needs_rerun = False
        self.status_extra = (
            f"Detected {len(candidates)} auto lines; removed {removed} overlapping lines; "
            f"kept {len(manual_candidates)} hand-drawn lines."
        )

    def _original_to_preview_points(self, points: np.ndarray) -> np.ndarray:
        preview = points.astype(np.float64, copy=True)
        preview[:, 0] = (preview[:, 0] + 0.5) * (self.preview_width / self.original_width) - 0.5
        preview[:, 1] = (preview[:, 1] + 0.5) * (self.preview_height / self.original_height) - 0.5
        preview[:, 0] = np.clip(preview[:, 0], 0.0, self.preview_width - 1.0)
        preview[:, 1] = np.clip(preview[:, 1], 0.0, self.preview_height - 1.0)
        return preview

    def _clip_preview_xy(self, x: float, y: float) -> tuple[float, float]:
        x_clipped = float(np.clip(x, 0.0, self.preview_width - 1.0))
        y_clipped = float(np.clip(y, 0.0, self.preview_height - 1.0))
        return x_clipped, y_clipped

    def _preview_to_original_points(self, points: np.ndarray) -> np.ndarray:
        original = points.astype(np.float64, copy=True)
        original[:, 0] = (original[:, 0] + 0.5) * (self.original_width / self.preview_width) - 0.5
        original[:, 1] = (original[:, 1] + 0.5) * (self.original_height / self.preview_height) - 0.5
        original[:, 0] = np.clip(original[:, 0], 0.0, self.original_width - 1.0)
        original[:, 1] = np.clip(original[:, 1], 0.0, self.original_height - 1.0)
        return original

    def _annotation_camera(self) -> Camera:
        return Camera(
            CameraConfig(
                fov_deg=None,
                width=self.original_width,
                height=self.original_height,
                model=self.camera_model,
                view=self.view_metadata,
            )
        )

    def _button_save(self, _event: object) -> None:
        self._save()
        self._refresh()

    def _button_save_close(self, _event: object) -> None:
        if self._save():
            import matplotlib.pyplot as plt

            plt.close(self.fig)
        else:
            self._refresh()

    def _button_select_all(self, _event: object) -> None:
        for candidate in self.candidates:
            candidate.selected = True
        self.status_extra = "Selected all candidate lines."
        self._refresh()

    def _button_drop_all(self, _event: object) -> None:
        for candidate in self.candidates:
            candidate.selected = False
        self.status_extra = "Dropped all candidate lines."
        self._refresh()

    def _button_invert(self, _event: object) -> None:
        for candidate in self.candidates:
            candidate.selected = not candidate.selected
        self.status_extra = "Inverted line selection."
        self._refresh()

    def _button_rerun(self, _event: object) -> None:
        self.status_extra = "Running ULSD detection..."
        self._refresh()
        self.fig.canvas.flush_events()
        try:
            self._detect_candidates()
        except Exception as exc:
            self.status_extra = f"Rerun failed: {exc}"
            print(self.status_extra)
        self._refresh()

    def _button_review(self, _event: object) -> None:
        self._set_mode("review")

    def _button_draw(self, _event: object) -> None:
        self._set_mode("draw")

    def _button_undo(self, _event: object) -> None:
        self._undo()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode != "draw":
            self._drawing_line = False
            self.current_line_points = []
        self.status_extra = f"Mode changed to {mode}."
        self._refresh()

    def _undo(self) -> None:
        if not self.history:
            self.status_extra = "Nothing to undo."
            self._refresh()
            return
        action, candidate = self.history.pop()
        if action == "manual_line":
            remove_idx = next(
                (idx for idx, item in enumerate(self.candidates) if item is candidate),
                None,
            )
            if remove_idx is not None:
                self.candidates.pop(remove_idx)
                self.status_extra = "Removed last hand-drawn line."
            else:
                self.status_extra = "Last hand-drawn line was already removed."
            self._refresh()
            return
        self._refresh()

    def _on_click(self, event: object) -> None:
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return

        x, y = self._clip_preview_xy(float(event.xdata), float(event.ydata))
        button = getattr(event, "button", None)
        if self.mode == "draw":
            if button == 1:
                self._drawing_line = True
                self.current_line_points = [(x, y)]
            elif button == 3:
                self._drawing_line = False
                self.current_line_points = []
            self._refresh()
            return

        if button != 1:
            return

        best_idx = None
        best_dist = float("inf")
        for idx, candidate in enumerate(self.candidates):
            dist = _point_to_polyline_distance(x, y, candidate.preview_points)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist

        if best_idx is not None and best_dist <= HIT_TOLERANCE_PX:
            candidate = self.candidates[best_idx]
            candidate.selected = not candidate.selected
            state = "selected" if candidate.selected else "dropped"
            self.status_extra = f"Line {best_idx + 1} {state} (score={candidate.score:.3f})."
            self._refresh()

    def _on_draw(self, event: object) -> None:
        if not self._drawing_line or self.mode != "draw":
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return

        x, y = self._clip_preview_xy(float(event.xdata), float(event.ydata))
        if not self.current_line_points:
            self.current_line_points = [(x, y)]
        else:
            last_x, last_y = self.current_line_points[-1]
            if math.hypot(x - last_x, y - last_y) >= LINE_MIN_SPACING_PX:
                self.current_line_points.append((x, y))
        self._refresh()

    def _on_release(self, event: object) -> None:
        if not self._drawing_line or self.mode != "draw":
            return
        if (
            getattr(event, "inaxes", None) is self.ax
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x, y = self._clip_preview_xy(float(event.xdata), float(event.ydata))
            if self.current_line_points:
                last_x, last_y = self.current_line_points[-1]
                if math.hypot(x - last_x, y - last_y) > 0.0:
                    self.current_line_points.append((x, y))

        self._drawing_line = False
        if (
            len(self.current_line_points) >= 2
            and _polyline_length(self.current_line_points) >= LINE_MIN_SPACING_PX
        ):
            preview_points = np.asarray(
                _resample_polyline(self.current_line_points, LINE_SAMPLE_POINTS),
                dtype=np.float64,
            )
            original_points = self._preview_to_original_points(preview_points)
            candidate = LineCandidate(
                line_heatmap=np.empty((0, 2), dtype=np.float64),
                line_original=original_points.copy(),
                original_points=original_points,
                preview_points=preview_points,
                score=1.0,
                source="manual",
                selected=True,
            )
            self.candidates.append(candidate)
            self.history.append(("manual_line", candidate))
            self.status_extra = "Added hand-drawn line."
        self.current_line_points = []
        self._refresh()

    def _on_key(self, event: object) -> None:
        key = (getattr(event, "key", "") or "").lower()
        if key in ("s", "ctrl+s", "cmd+s"):
            self._save()
            self._refresh()
        elif key in ("a",):
            self._button_select_all(event)
        elif key in ("d",):
            self._button_drop_all(event)
        elif key in ("i",):
            self._button_invert(event)
        elif key in ("r",):
            self._button_rerun(event)
        elif key in ("l",):
            self._set_mode("draw")
        elif key in ("v",):
            self._set_mode("review")
        elif key in ("u", "ctrl+z", "cmd+z"):
            self._undo()
        elif key == "escape":
            self._drawing_line = False
            self.current_line_points = []
            self._refresh()

    def _selected_candidates(self) -> list[LineCandidate]:
        return [candidate for candidate in self.candidates if candidate.selected]

    def _line_json(self) -> list[dict[str, list[list[float]]]]:
        result: list[dict[str, list[list[float]]]] = []
        camera = self._annotation_camera()
        for candidate in self._selected_candidates():
            samples = candidate.original_points
            x = np.clip(samples[:, 0], 0.0, self.original_width - 1.0)
            y = np.clip(samples[:, 1], 0.0, self.original_height - 1.0)
            lam, phi = camera.pixel_to_direction(x, y)
            points_dir = [
                [float(lam_i), float(phi_i)]
                for lam_i, phi_i in zip(lam, phi)
            ]
            result.append({"points_dir": points_dir})
        return result

    def _write_selected_lines_image(self) -> None:
        image = Image.fromarray(self.original_image.copy())
        draw = ImageDraw.Draw(image)
        selected = self._selected_candidates()
        for candidate in selected:
            points = [
                (float(x), float(y))
                for x, y in candidate.original_points
            ]
            color = (0, 255, 0) if candidate.source == "auto" else (255, 220, 0)
            draw.line(points, fill=color, width=4)
            for x, y in (points[0], points[-1]):
                radius = 5
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=(255, 255, 0),
                )
        image.save(self.marked_image_path, quality=95)

    def _save(self) -> bool:
        if self.needs_rerun or self.camera_model != self.detected_camera_model:
            self.status_extra = "Camera model changed. Press Rerun before saving."
            return False

        payload = build_annotation_payload(
            image_path=self.image_path,
            output_path=self.json_path,
            source_image_path=self.image_path,
            camera_model=self.camera_model,
            view_metadata=self.view_metadata,
            lines=self._line_json(),
            regions=[],
        )
        write_annotation_json(self.json_path, payload)
        self._write_selected_lines_image()
        self.status_extra = (
            f"Saved {len(payload['lines'])} lines to {self.json_path.name} "
            f"and {self.marked_image_path.name}."
        )
        return True

    def _refresh(self) -> None:
        for artist in self._dynamic_artists:
            artist.remove()
        self._dynamic_artists.clear()

        for selected_state in (False, True):
            for candidate in self.candidates:
                if candidate.selected != selected_state:
                    continue
                points = candidate.preview_points
                color = "#00ff66" if candidate.selected else "#d65a5a"
                if candidate.source == "manual" and candidate.selected:
                    color = "#ffd700"
                alpha = 0.95 if candidate.selected else 0.35
                linewidth = 2.4 if candidate.source == "manual" and candidate.selected else 2.0 if candidate.selected else 1.4
                zorder = 4 if candidate.selected else 3
                line_artist = self.ax.plot(
                    points[:, 0],
                    points[:, 1],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=zorder,
                )[0]
                self._dynamic_artists.append(line_artist)

        if self.current_line_points:
            xs = [point[0] for point in self.current_line_points]
            ys = [point[1] for point in self.current_line_points]
            active_line = self.ax.plot(
                xs,
                ys,
                color="yellow",
                linewidth=2.0,
                marker="o",
                markersize=3,
                zorder=5,
            )[0]
            self._dynamic_artists.append(active_line)

        selected_count = len(self._selected_candidates())
        manual_count = sum(1 for candidate in self.candidates if candidate.source == "manual")
        size_text = f"Preview: {self.preview_width}x{self.preview_height}"
        if (self.preview_width, self.preview_height) != (self.original_width, self.original_height):
            size_text += f" -> Original: {self.original_width}x{self.original_height}"
        camera_text = f"Camera: {CAMERA_MODEL_LABELS[self.camera_model]}"
        pending = " | Rerun required" if self.needs_rerun else ""
        self.status.set_text(
            f"{camera_text} | Model: {self.current_model_name} | use_gpu: {self.use_gpu} | "
            f"Mode: {self.mode.upper()} | Candidates: {len(self.candidates)} | "
            f"Selected: {selected_count} | Hand-drawn: {manual_count} | "
            f"score>{self.selection.score_thresh:.3f}, top_k={self.selection.top_k}, "
            f"dedupe={self.selection.duplicate_dist_px:.1f}px | "
            f"Review: click line to toggle; Draw: left-drag line | "
            f"Save: {self.json_path.name}, {self.marked_image_path.name} | "
            f"{size_text}{pending} | {self.status_extra}"
        )
        self.fig.canvas.draw_idle()


def main() -> None:
    args = parse_args()
    AutoLineReviewGUI(args).run()


if __name__ == "__main__":
    main()
