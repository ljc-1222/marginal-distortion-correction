"""Fallback utility for merging ROI-only and line-only annotation JSON files."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


FOV_TOLERANCE = 1e-6


class AnnotationMergeError(ValueError):
    """Raised when two annotation JSON files cannot be safely merged."""


def _read_json(path: str | Path) -> tuple[dict[str, Any], Path]:
    json_path = Path(path).expanduser().resolve()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise AnnotationMergeError(f"Annotation JSON must contain an object: {json_path}")
    return data, json_path.parent


def _relative_path(path: Path, base_dir: Path) -> str:
    """Return a path relative to ``base_dir`` when possible."""
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), base_dir.resolve())


def _resolve_path_field(data: dict[str, Any], base_dir: Path, key: str) -> Path | None:
    value = data.get(key)
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _require_same_path(
    roi_data: dict[str, Any],
    roi_base: Path,
    line_data: dict[str, Any],
    line_base: Path,
    key: str,
) -> Path | None:
    roi_path = _resolve_path_field(roi_data, roi_base, key)
    line_path = _resolve_path_field(line_data, line_base, key)
    if roi_path != line_path:
        raise AnnotationMergeError(f"Annotation metadata mismatch for {key}: {roi_path} != {line_path}")
    return line_path


def _require_same_metadata(roi_data: dict[str, Any], roi_base: Path, line_data: dict[str, Any], line_base: Path) -> None:
    image_path = _require_same_path(roi_data, roi_base, line_data, line_base, "image_path")
    if image_path is None:
        raise AnnotationMergeError("Annotation metadata must include image_path.")

    roi_camera = roi_data.get("camera_model")
    line_camera = line_data.get("camera_model")
    if roi_camera != line_camera:
        raise AnnotationMergeError(f"Annotation metadata mismatch for camera_model: {roi_camera!r} != {line_camera!r}")
    if roi_camera not in ("pinhole", "panorama_view"):
        raise AnnotationMergeError(f"Unsupported camera_model: {roi_camera!r}")

    if roi_camera == "pinhole":
        if "fov_deg" not in roi_data or "fov_deg" not in line_data:
            raise AnnotationMergeError("pinhole annotations must include fov_deg in both inputs.")
        roi_fov = float(roi_data["fov_deg"])
        line_fov = float(line_data["fov_deg"])
        if abs(roi_fov - line_fov) > FOV_TOLERANCE:
            raise AnnotationMergeError(f"Annotation metadata mismatch for fov_deg: {roi_fov} != {line_fov}")

    _require_same_path(roi_data, roi_base, line_data, line_base, "source_image_path")

    roi_view = roi_data.get("view")
    line_view = line_data.get("view")
    if roi_view != line_view:
        raise AnnotationMergeError("Annotation metadata mismatch for view metadata.")
    if roi_camera == "panorama_view":
        if roi_data.get("schema_version") != 2 or line_data.get("schema_version") != 2:
            raise AnnotationMergeError("panorama_view annotations must contain schema_version: 2 in both inputs.")
        if roi_view is None:
            raise AnnotationMergeError("panorama_view annotations must contain view metadata.")


def merge_annotation_payloads(
    roi_data: dict[str, Any],
    roi_base: Path,
    line_data: dict[str, Any],
    line_base: Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Merge ROI and line annotation payloads after metadata validation."""
    _require_same_metadata(roi_data, roi_base, line_data, line_base)

    output_dir = Path(output_path).expanduser().resolve().parent
    image_path = _resolve_path_field(line_data, line_base, "image_path")
    source_image_path = _resolve_path_field(line_data, line_base, "source_image_path")
    if image_path is None:
        raise AnnotationMergeError("Annotation metadata must include image_path.")

    merged: dict[str, Any] = {
        "image_path": _relative_path(image_path, output_dir),
        "camera_model": str(line_data["camera_model"]),
        "lines": list(line_data.get("lines", [])),
        "regions": [],
    }
    if merged["camera_model"] == "pinhole":
        merged["fov_deg"] = float(line_data["fov_deg"])
    if source_image_path is not None:
        merged["source_image_path"] = _relative_path(source_image_path, output_dir)
    if line_data.get("view") is not None:
        merged["schema_version"] = 2
        merged["view"] = line_data["view"]

    for idx, region in enumerate(roi_data.get("regions", [])):
        if not isinstance(region, dict):
            raise AnnotationMergeError(f"regions[{idx}] must be an object.")
        if "mask_path" not in region:
            raise AnnotationMergeError(f"regions[{idx}] must contain mask_path.")
        mask = Path(str(region["mask_path"])).expanduser()
        if not mask.is_absolute():
            mask = (roi_base / mask).resolve()
        merged["regions"].append(
            {
                "name": str(region.get("name", f"region_{idx}")),
                "mask_path": _relative_path(mask, output_dir),
            }
        )

    return merged


def merge_annotation_json(roi_json: str | Path, line_json: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Merge two annotation JSON files and write the combined payload."""
    roi_data, roi_base = _read_json(roi_json)
    line_data, line_base = _read_json(line_json)
    output = Path(output_path).expanduser().resolve()
    payload = merge_annotation_payloads(roi_data, roi_base, line_data, line_base, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
    return payload


def _line_samples() -> list[list[float]]:
    return [[float(idx) / 127.0, 0.0] for idx in range(128)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def _expect_merge_error(roi_json: Path, line_json: Path, output: Path, text: str) -> None:
    try:
        merge_annotation_json(roi_json, line_json, output)
    except AnnotationMergeError as exc:
        if text not in str(exc):
            raise AssertionError(f"Expected error containing {text!r}; got {exc!r}") from exc
        return
    raise AssertionError(f"Expected merge error containing {text!r}.")


def run_self_check() -> None:
    """Run a temporary merge self-check without GUI, SAM2, or real images."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        roi_dir = root / "roi"
        line_dir = root / "line"
        out_dir = root / "out"
        roi_dir.mkdir()
        line_dir.mkdir()
        out_dir.mkdir()

        image_path = root / "image.png"
        image_path.write_bytes(b"image")
        mask_path = roi_dir / "mask_region_1.png"
        mask_path.write_bytes(b"mask")

        roi_payload = {
            "image_path": "../image.png",
            "camera_model": "pinhole",
            "fov_deg": 90.0,
            "lines": [],
            "regions": [{"name": "region_1", "mask_path": "mask_region_1.png"}],
        }
        line_payload = {
            "image_path": "../image.png",
            "camera_model": "pinhole",
            "fov_deg": 90.0,
            "lines": [{"points_dir": _line_samples()}],
            "regions": [],
        }
        roi_json = roi_dir / "roi.json"
        line_json = line_dir / "line.json"
        output = out_dir / "full.json"
        _write_json(roi_json, roi_payload)
        _write_json(line_json, line_payload)

        merged = merge_annotation_json(roi_json, line_json, output)
        if len(merged["lines"]) != 1:
            raise AssertionError("Expected one copied line.")
        if len(merged["regions"]) != 1:
            raise AssertionError("Expected one copied region.")
        if merged["camera_model"] != "pinhole" or merged["fov_deg"] != 90.0:
            raise AssertionError("Expected preserved pinhole camera metadata.")
        merged_mask = (output.parent / merged["regions"][0]["mask_path"]).resolve()
        if merged_mask != mask_path.resolve() or not merged_mask.exists():
            raise AssertionError("Expected merged mask_path to remain valid relative to output JSON.")

        bad_roi = dict(roi_payload)
        bad_roi["image_path"] = "../different.png"
        bad_roi_json = roi_dir / "bad_image.json"
        _write_json(bad_roi_json, bad_roi)
        _expect_merge_error(bad_roi_json, line_json, out_dir / "bad_image_out.json", "image_path")

        bad_roi = dict(roi_payload)
        bad_roi["camera_model"] = "panorama_view"
        bad_roi_json = roi_dir / "bad_camera.json"
        _write_json(bad_roi_json, bad_roi)
        _expect_merge_error(bad_roi_json, line_json, out_dir / "bad_camera_out.json", "camera_model")

        bad_roi = dict(roi_payload)
        bad_roi["fov_deg"] = 91.0
        bad_roi_json = roi_dir / "bad_fov.json"
        _write_json(bad_roi_json, bad_roi)
        _expect_merge_error(bad_roi_json, line_json, out_dir / "bad_fov_out.json", "fov_deg")

    print("annotation_merge self-check passed")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Merge ROI-only and line-only MaDCoW annotation JSON files.")
    parser.add_argument("--roi-json", default=None, help="ROI-only annotation JSON path.")
    parser.add_argument("--line-json", default=None, help="Line-only annotation JSON path.")
    parser.add_argument("--output", default=None, help="Merged full annotation JSON path.")
    parser.add_argument("--self-check", action="store_true", help="Run the built-in non-GUI self-check.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.self_check:
        run_self_check()
        return
    if not args.roi_json or not args.line_json or not args.output:
        raise SystemExit("--roi-json, --line-json, and --output are required unless --self-check is used.")
    merge_annotation_json(args.roi_json, args.line_json, args.output)


if __name__ == "__main__":
    main()

