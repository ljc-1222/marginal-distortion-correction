"""Top-level integrated annotation CLI."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the integrated annotation workflow."""
    parser = argparse.ArgumentParser(description="Integrated SAM2 ROI and snapping line annotation workflow.")
    parser.add_argument("--image", default=None, help="Optional launcher prefill for the input image path.")
    parser.add_argument("--workspace", default=None, help="Optional launcher prefill for the annotation output directory.")
    parser.add_argument("--output-annotation", default=None, help="Optional launcher prefill for the final annotation JSON path.")
    parser.add_argument(
        "--device",
        default="auto",
        help="SAM2 inference device. Use 'auto' to prefer CUDA when available.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    from pipeline.annotation_launcher import FIXED_SNAP_CONFIG_PATH
    from pipeline.integrated_annotation import IntegratedAnnotationConfig, run_integrated_annotation

    output = run_integrated_annotation(
        IntegratedAnnotationConfig(
            image=args.image,
            workspace=args.workspace,
            output_annotation=args.output_annotation,
            device=args.device,
            snap_config=FIXED_SNAP_CONFIG_PATH,
        )
    )
    print(f"Saved integrated annotation JSON: {output}")


if __name__ == "__main__":
    main()
