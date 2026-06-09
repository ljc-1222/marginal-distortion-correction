"""Top-level integrated annotation CLI."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the integrated annotation workflow."""
    parser = argparse.ArgumentParser(description="Integrated SAM2 ROI and snapping line annotation workflow.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--workspace", required=True, help="Workspace directory for generated annotation assets.")
    parser.add_argument("--output-annotation", required=True, help="Final full MaDCoW annotation JSON path.")
    parser.add_argument("--sam2-checkpoint", required=True, help="SAM2 checkpoint path.")
    parser.add_argument(
        "--sam2-model-cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 model config name or path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SAM2 inference device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--snap-config",
        default="interactive_snapping_2d/config/snap_config.json",
        help="Interactive snapping config JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    from pipeline.integrated_annotation import IntegratedAnnotationConfig, run_integrated_annotation

    output = run_integrated_annotation(
        IntegratedAnnotationConfig(
            image=args.image,
            workspace=args.workspace,
            output_annotation=args.output_annotation,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_cfg=args.sam2_model_cfg,
            device=args.device,
            snap_config=args.snap_config,
        )
    )
    print(f"Saved integrated annotation JSON: {output}")


if __name__ == "__main__":
    main()

