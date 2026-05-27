from __future__ import annotations

import json

import cv2
import numpy as np

from interactive_snapping_2d.snap2d import SnapConfig, snap_annotation
from interactive_snapping_2d.snap2d.io import save_annotation_json, save_madcow_annotation_json
from MaDCoW.main import load_annotations
from interactive_snapping_2d.snap2d.panorama import unwrap_panorama_stroke, wrap_panorama_points
from interactive_snapping_2d.snap2d.postprocess import fit_weighted_line_2d


def _config() -> SnapConfig:
    return SnapConfig(
        resample_step_px=2.0,
        search_width_px=18.0,
        offset_step_px=1.0,
        smooth_weight=0.05,
        output_spacing_px=3.0,
        canny_low=30,
        canny_high=90,
    )


def _line_points(x0: float, x1: float, count: int, slope: float, intercept: float) -> np.ndarray:
    x = np.linspace(x0, x1, count, dtype=np.float32)
    y = slope * x + intercept
    return np.stack([x, y], axis=1)


def _mean_line_distance(points: np.ndarray, slope: float, intercept: float) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(np.mean(np.abs(slope * x - y + intercept) / np.sqrt(slope * slope + 1.0)))


def test_synthetic_straight_edge_curve_mode() -> None:
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    slope = 0.35
    intercept = 45.0
    true_points = _line_points(20, 200, 80, slope, intercept)
    cv2.polylines(image, [np.round(true_points).astype(np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), 3)
    normal = np.asarray([-slope, 1.0], dtype=np.float32)
    normal /= np.linalg.norm(normal)
    rough = true_points[::10] + normal[None, :] * 10.0
    result = snap_annotation(image, rough, camera_type="pinhole", mode="curve", config=_config())
    assert _mean_line_distance(result.points, slope, intercept) < 4.0
    assert result.points.shape[1] == 2


def test_synthetic_curve_edge() -> None:
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    x = np.linspace(30, 210, 120, dtype=np.float32)
    y = 0.004 * (x - 120.0) ** 2 + 55.0
    curve = np.stack([x, y], axis=1)
    cv2.polylines(image, [np.round(curve).astype(np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), 3)
    rough = curve[::12].copy()
    rough[:, 1] += 9.0
    result = snap_annotation(image, rough, camera_type="pinhole", mode="curve", config=_config())
    expected_y = 0.004 * (result.points[:, 0] - 120.0) ** 2 + 55.0
    assert float(np.mean(np.abs(result.points[:, 1] - expected_y))) < 5.0


def test_pinhole_line_postprocess_outputs_collinear_points() -> None:
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    slope = -0.25
    intercept = 120.0
    true_points = _line_points(20, 200, 80, slope, intercept)
    cv2.polylines(image, [np.round(true_points).astype(np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), 3)
    rough = true_points[::10].copy()
    rough[:, 1] += 8.0
    result = snap_annotation(image, rough, camera_type="pinhole", mode="line", config=_config())
    line_point, direction = fit_weighted_line_2d(result.points, np.ones(len(result.points), dtype=np.float32))
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    distances = np.abs((result.points - line_point) @ normal)
    assert float(distances.max()) < 1e-3


def test_panorama_line_keeps_polyline_shape() -> None:
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    x = np.linspace(25, 215, 140, dtype=np.float32)
    y = 70.0 + 25.0 * np.sin((x - 25.0) / 190.0 * np.pi)
    curve = np.stack([x, y], axis=1)
    cv2.polylines(image, [np.round(curve).astype(np.int32).reshape(-1, 1, 2)], False, (255, 255, 255), 3)
    rough = curve[::14].copy()
    rough[:, 1] += 8.0
    result = snap_annotation(image, rough, camera_type="panorama", mode="line", config=_config())
    line_point, direction = fit_weighted_line_2d(result.points, np.ones(len(result.points), dtype=np.float32))
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    residual = np.mean(np.abs((result.points - line_point) @ normal))
    assert float(residual) > 1.0


def test_panorama_seam_unwrap_and_wrap() -> None:
    image = np.zeros((20, 100, 3), dtype=np.uint8)
    stroke = np.asarray([[94, 8], [98, 9], [2, 10], [6, 11]], dtype=np.float32)
    _, unwrapped, info = unwrap_panorama_stroke(image, stroke)
    assert float(np.max(np.abs(np.diff(unwrapped[:, 0])))) < 12.0
    wrapped = wrap_panorama_points(unwrapped, info)
    assert np.all(wrapped[:, 0] >= 0.0)
    assert np.all(wrapped[:, 0] < 100.0)


def test_json_schema(tmp_path) -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.line(image, (10, 50), (110, 50), (255, 255, 255), 3)
    rough = np.asarray([[10, 58], [60, 58], [110, 58]], dtype=np.float32)
    result = snap_annotation(image, rough, camera_type="pinhole", mode="curve", config=_config())
    output_path = tmp_path / "annotation.json"
    save_annotation_json(result, "synthetic.png", str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))
    for key in ["points", "source_stroke", "mode", "camera_type", "confidence", "coordinate_space"]:
        assert key in data
    assert data["coordinate_space"] == "input_image_pixel"


def test_madcow_annotation_schema_loads_in_main(tmp_path) -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    cv2.line(image, (10, 50), (110, 50), (255, 255, 255), 3)
    rough = np.asarray([[10, 58], [60, 58], [110, 58]], dtype=np.float32)
    result = snap_annotation(image, rough, camera_type="pinhole", mode="line", config=_config())
    image_path = tmp_path / "synthetic.png"
    cv2.imwrite(str(image_path), image)
    output_path = tmp_path / "madcow_annotation.json"
    save_madcow_annotation_json(
        result,
        str(image_path),
        str(output_path),
        image_shape=image.shape[:2],
        camera_type="pinhole",
        fov_deg=90.0,
    )
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["camera_model"] == "pinhole"
    assert data["fov_deg"] == 90.0
    assert len(data["lines"]) == 1
    assert len(data["lines"][0]["points_dir"]) == 128
    loaded = load_annotations(str(output_path))
    assert len(loaded.lines) == 1
    assert len(loaded.lines[0].points_dir) == 128
