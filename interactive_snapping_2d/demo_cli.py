"""Headless CLI demo for the 2D interactive snapping algorithm."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .snap2d import SnapConfig, snap_annotation
from .snap2d.io import load_strokes_json, save_annotation_json, save_madcow_annotation_json


DEFAULT_IMAGE = Path(__file__).resolve().parent / "data" / "test_1.jpg"
DEFAULT_STROKES = Path(__file__).resolve().parent / "examples" / "sample_strokes.json"


def _draw_polyline(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    if len(points) < 2:
        return
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], False, color, thickness, lineType=cv2.LINE_AA)


def save_preview(image: np.ndarray, result, output_path: str) -> None:
    """Save an overlay preview with source stroke and snapped result."""
    preview = image.copy()
    _draw_polyline(preview, result.source_stroke, (0, 255, 255), 3)
    _draw_polyline(preview, result.points, (0, 0, 255), 3)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a headless 2D snapping demo.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path.")
    parser.add_argument("--stroke-json", default=str(DEFAULT_STROKES), help="JSON file containing rough strokes.")
    parser.add_argument("--stroke-name", default=None, help="Optional stroke name to select.")
    parser.add_argument("--camera-type", choices=("pinhole", "panorama"), default=None, help="Override camera type.")
    parser.add_argument("--mode", choices=("line", "curve"), default=None, help="Override annotation mode.")
    parser.add_argument("--output-json", default="interactive_snapping_2d/outputs/demo_annotation.json")
    parser.add_argument(
        "--output-madcow-json",
        default=None,
        help="Optional MaDCoW-compatible annotation JSON output path.",
    )
    parser.add_argument("--output-preview", default="interactive_snapping_2d/outputs/demo_preview.png")
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Horizontal FOV in degrees for pinhole MaDCoW annotation export.",
    )
    return parser.parse_args()


def _select_stroke(strokes: list[dict], name: str | None) -> dict:
    if name is None:
        return strokes[0]
    for stroke in strokes:
        if stroke.get("name") == name:
            return stroke
    raise ValueError(f"No stroke named {name!r}.")


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    strokes = load_strokes_json(args.stroke_json)
    stroke_item = _select_stroke(strokes, args.stroke_name)
    camera_type = args.camera_type or str(stroke_item.get("camera_type", "pinhole"))
    mode = args.mode or str(stroke_item.get("mode", "curve"))
    result = snap_annotation(
        image,
        stroke_item["points"],
        camera_type=camera_type,
        mode=mode,
        config=SnapConfig(),
    )
    save_annotation_json(result, args.image, args.output_json)
    if args.output_madcow_json is not None:
        save_madcow_annotation_json(
            result,
            args.image,
            args.output_madcow_json,
            image_shape=image.shape[:2],
            camera_type=camera_type,
            fov_deg=args.fov,
        )
    save_preview(image, result, args.output_preview)
    print(f"Saved annotation: {args.output_json}")
    if args.output_madcow_json is not None:
        print(f"Saved MaDCoW annotation: {args.output_madcow_json}")
    print(f"Saved preview: {args.output_preview}")
    print(f"confidence={result.confidence:.3f}, points={len(result.points)}")


if __name__ == "__main__":
    main()
