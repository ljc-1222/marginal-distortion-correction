"""Shared MaDCoW annotation JSON writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .image import relative_path


def build_annotation_payload(
    image_path: str | Path,
    output_path: str | Path,
    camera_model: str,
    lines: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    fov_deg: float | None = None,
    source_image_path: str | Path | None = None,
    view_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a MaDCoW-compatible annotation JSON payload."""
    json_dir = Path(output_path).resolve().parent
    payload: dict[str, Any] = {
        "image_path": relative_path(Path(image_path), json_dir),
        "camera_model": str(camera_model),
    }
    if source_image_path is not None:
        payload["source_image_path"] = relative_path(Path(source_image_path), json_dir)
    if fov_deg is not None:
        payload["fov_deg"] = float(fov_deg)
    if view_metadata is not None:
        payload["schema_version"] = 2
        payload["view"] = view_metadata
    payload["lines"] = lines
    payload["regions"] = regions
    return payload


def write_annotation_json(
    output_path: str | Path,
    payload: dict[str, Any],
) -> None:
    """Write an annotation payload to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
