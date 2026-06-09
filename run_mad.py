"""Top-level MaDCoW runner CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MADCOW_CONFIG_PATH = "MaDCoW/config.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the top-level MaDCoW wrapper."""
    parser = argparse.ArgumentParser(description="Run MaDCoW correction on a full annotation JSON.")
    parser.add_argument("--image", default=None, help="Optional input image override.")
    parser.add_argument("--annotations", required=True, help="Full MaDCoW annotation JSON path.")
    parser.add_argument("--output", required=True, help="Corrected output image path.")
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Use MaDCoW's existing crop behavior.",
    )
    parser.add_argument("--summary", default=None, help="Optional run summary JSON path.")
    return parser.parse_args()


def _write_summary(args: argparse.Namespace) -> None:
    """Write a lightweight MaDCoW run summary."""
    from MaDCoW.main import load_annotations

    annotations = load_annotations(args.annotations)
    payload = {
        "annotations": str(Path(args.annotations).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "output": str(Path(args.output).expanduser().resolve()),
        "crop": bool(args.crop),
        "camera_model": annotations.camera_model,
        "number_of_lines": len(annotations.lines),
        "number_of_regions": len(annotations.regions),
    }
    if args.image is not None:
        payload["image"] = str(Path(args.image).expanduser().resolve())

    summary_path = Path(args.summary).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    args.config = MADCOW_CONFIG_PATH
    from MaDCoW.main import run_pipeline

    run_pipeline(args)
    if args.summary:
        _write_summary(args)


if __name__ == "__main__":
    main()
