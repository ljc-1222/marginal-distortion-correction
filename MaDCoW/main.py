"""MaDCoW command-line entry point.

Loads a wide-angle photograph together with its annotation JSON, runs the
two-stage MaDCoW pipeline, and writes the corrected image to disk.
"""

from __future__ import annotations

import argparse
import json
import numbers
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .src import (
    AnnotationData,
    CameraConfig,
    LineAnnotation,
    LossWeights,
    PipelineConfig,
    RegionAnnotation,
)
from .src.camera import Camera
from .src.initialization import init_full_mesh, stereographic_init
from .src.mesh import build_input_domain_mesh, compute_valid_mesh_mask, rasterize_mask_to_mesh
from .src.render import crop_to_rect, warp_image
from .src.stage1_region import optimize_region
from .src.stage2_warp import optimize_warp
from .src.weights import compute_weights


LINE_ANNOTATION_POINTS = 128


def _resolve_path(path: str, base_dir: Path) -> str:
    """Resolve ``path`` relative to ``base_dir`` when it is not absolute."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((base_dir / path_obj).resolve())


def _require_mapping(value: object, name: str) -> dict:
    """Return a JSON object as a dict or raise a clear error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _require_json_number(value: object, name: str) -> float:
    """Return a JSON numeric value as float or raise a clear error."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be numeric.")
    return float(value)


def _read_image(path: str) -> np.ndarray:
    """Read an image file as RGB uint8."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _read_mask(path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    """Read an ROI mask as a boolean array in input-image space."""
    with Image.open(path) as img:
        mask = np.asarray(img.convert("L")) >= 128
    if mask.shape != expected_shape:
        raise ValueError(
            f"mask {path} has shape {mask.shape}, expected input image shape {expected_shape}."
        )
    return mask


def _save_image(path: str, image: np.ndarray) -> None:
    """Write an image array, creating the output directory when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with attributes: ``image`` (str), ``annotations`` (str),
        ``config`` (str), ``output`` (str), ``crop`` (bool).
    """
    parser = argparse.ArgumentParser(description="MaDCoW marginal distortion correction.")
    parser.add_argument("--image", default="./MaDCoW/data/test_1.png", help="Path to the input .jpg.")
    parser.add_argument("--annotations", required=True, help="Path to the annotation JSON.")
    parser.add_argument("--config", default="./MaDCoW/config.json", help="Path to the pipeline config JSON.")
    parser.add_argument("--output", required=True, help="Path of the output image.")
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop output to the largest black-border-free rectangle with the input aspect ratio.",
    )
    return parser.parse_args()


def load_config(path: str) -> PipelineConfig:
    """Load and parse the pipeline configuration JSON.

    Args:
        path: Filesystem path to ``config.json``.

    Returns:
        The :class:`PipelineConfig` populated from the JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = _require_mapping(json.load(f), "config")

    mesh = _require_mapping(data.get("mesh"), "config.mesh")
    weights = _require_mapping(data.get("loss_weights"), "config.loss_weights")

    return PipelineConfig(
        mesh_n_lambda=int(mesh["n_lambda"]),
        mesh_n_phi=int(mesh["n_phi"]),
        loss_weights=LossWeights(
            w_l=float(weights["w_l"]),
            w_c=float(weights["w_c"]),
            w_s=float(weights["w_s"]),
            w_dvc=float(weights["w_dvc"]),
        ),
        blend_c=float(data["blend_c"]),
        stage1_max_iter=int(data["stage1_max_iter"]),
        stage2_max_iter=int(data["stage2_max_iter"]),
        lbfgs_lr=float(data["lbfgs_lr"]),
    )


def load_annotations(path: str) -> AnnotationData:
    """Load and parse a user annotation JSON.

    Args:
        path: Filesystem path to the annotation file produced by
            ``annotate.py``.

    Returns:
        The :class:`AnnotationData` populated from the JSON.
    """
    annotation_path = Path(path).resolve()
    base_dir = annotation_path.parent
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = _require_mapping(json.load(f), "annotations")

    lines_raw = data.get("lines", [])
    if not isinstance(lines_raw, list):
        raise ValueError("annotations.lines must be a list.")
    regions_raw = data.get("regions", [])
    if not isinstance(regions_raw, list):
        raise ValueError("annotations.regions must be a list.")
    if "camera_model" not in data:
        raise ValueError("annotations.camera_model must be set by annotate.py.")
    camera_model = str(data["camera_model"])

    lines: list[LineAnnotation] = []
    for idx, item in enumerate(lines_raw):
        line = _require_mapping(item, f"annotations.lines[{idx}]")
        if "points_dir" not in line:
            raise ValueError(f"annotations.lines[{idx}] must contain points_dir.")
        points_raw = line["points_dir"]
        if not isinstance(points_raw, list):
            raise ValueError(f"annotations.lines[{idx}].points_dir must be a list.")
        if len(points_raw) != LINE_ANNOTATION_POINTS:
            raise ValueError(
                f"annotations.lines[{idx}].points_dir must contain exactly "
                f"{LINE_ANNOTATION_POINTS} points; got {len(points_raw)}."
            )

        points: list[tuple[float, float]] = []
        for point_idx, point in enumerate(points_raw):
            point_name = f"annotations.lines[{idx}].points_dir[{point_idx}]"
            if not isinstance(point, list):
                raise ValueError(f"{point_name} must be a list of two numeric values.")
            if len(point) != 2:
                raise ValueError(f"{point_name} must contain exactly two numeric values.")
            lam = _require_json_number(point[0], f"{point_name}[0]")
            phi = _require_json_number(point[1], f"{point_name}[1]")
            points.append((lam, phi))

        lines.append(LineAnnotation(points_dir=tuple(points)))

    regions: list[RegionAnnotation] = []
    for idx, item in enumerate(regions_raw):
        region = _require_mapping(item, f"annotations.regions[{idx}]")
        name = str(region.get("name", f"region_{idx}"))
        mask_path = _resolve_path(str(region["mask_path"]), base_dir)
        regions.append(RegionAnnotation(name=name, mask_path=mask_path))

    image_path = _resolve_path(str(data.get("image_path", "")), base_dir) if data.get("image_path") else ""
    fov_value = data.get("fov_deg")
    fov_deg = None if fov_value is None else float(fov_value)
    return AnnotationData(
        image_path=image_path,
        fov_deg=fov_deg,
        camera_model=camera_model,
        lines=lines,
        regions=regions,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full MaDCoW pipeline end to end.

    Steps:
        1. Load config and annotations; load the input image.
        2. Build a :class:`Camera` from the annotation's input camera model.
        3. Build the view-sphere :class:`MeshGrid` from the input camera's
           angular domain and the mesh resolution in ``cfg``.
        4. For each ROI mask: rasterize to mesh, compute weights, run
           :func:`stage1_region.optimize_region` to get ``T_k`` evaluations.
        5. Stack stereographic init, per-ROI ``T_k``, and run
           :func:`initialization.init_full_mesh`.
        6. Run :func:`stage2_warp.optimize_warp` to obtain ``p_final``.
        7. Render the output with :func:`render.warp_image`, optionally
           :func:`render.crop_to_rect` to remove outer black borders while
           preserving the input aspect ratio, and save to ``args.output``.

    Args:
        args: Parsed CLI arguments from :func:`parse_args`.
    """
    cfg = load_config(args.config)
    annotations = load_annotations(args.annotations)
    image_path = args.image or annotations.image_path
    if not image_path:
        raise ValueError("No input image path provided in arguments or annotations.")
    image = _read_image(image_path)
    H_img, W_img = image.shape[:2]

    camera = Camera(
        CameraConfig(
            fov_deg=annotations.fov_deg,
            width=W_img,
            height=H_img,
            model=annotations.camera_model,
        )
    )
    mesh = build_input_domain_mesh(
        camera=camera,
        n_lambda=cfg.mesh_n_lambda,
        n_phi=cfg.mesh_n_phi,
    )
    valid_mesh_mask_np = compute_valid_mesh_mask(camera, mesh)

    weights_np = compute_weights(image, camera, mesh, annotations.lines)
    weights_np[~valid_mesh_mask_np] = 0.0
    p_stereo = stereographic_init(mesh)

    region_masks_np: list[np.ndarray] = []
    t_per_region: list[torch.Tensor] = []
    for region in annotations.regions:
        mask_img = _read_mask(region.mask_path, (H_img, W_img))
        mask_mesh = rasterize_mask_to_mesh(mask_img, camera, mesh) & valid_mesh_mask_np
        if not bool(mask_mesh.any()):
            raise ValueError(f"region {region.name!r} does not cover any valid mesh vertex.")

        _, t_eval = optimize_region(
            mesh=mesh,
            region_mask=mask_mesh,
            weights=weights_np,
            p_stereo=p_stereo,
            max_iter=cfg.stage1_max_iter,
            valid_mask=valid_mesh_mask_np,
        )
        region_masks_np.append(mask_mesh)
        t_per_region.append(t_eval)

    if t_per_region:
        t_targets = torch.stack(t_per_region, dim=0).to(dtype=p_stereo.dtype, device=p_stereo.device)
        region_masks = torch.as_tensor(np.stack(region_masks_np, axis=0), dtype=torch.bool)
    else:
        t_targets = torch.empty((0,) + p_stereo.shape, dtype=p_stereo.dtype, device=p_stereo.device)
        region_masks = torch.empty((0,) + p_stereo.shape[:2], dtype=torch.bool, device=p_stereo.device)

    p_init = init_full_mesh(
        p_stereo=p_stereo,
        t_per_region=t_targets,
        region_masks=region_masks,
        mesh=mesh,
        blend_c=cfg.blend_c,
    )
    p_final = optimize_warp(
        p_init=p_init,
        mesh=mesh,
        weights=torch.as_tensor(weights_np, dtype=p_init.dtype, device=p_init.device),
        lines=annotations.lines,
        t_targets=t_targets,
        region_masks=region_masks,
        cfg=cfg,
        valid_mask=torch.as_tensor(valid_mesh_mask_np, dtype=torch.bool, device=p_init.device),
    )

    if args.crop:
        output, valid_mask = warp_image(
            image,
            camera,
            mesh,
            p_final,
            out_size=(H_img, W_img),
            return_mask=True,
            valid_mesh_mask=valid_mesh_mask_np,
        )
        output = crop_to_rect(output, valid_mask, target_aspect=(W_img, H_img))
    else:
        output = warp_image(
            image,
            camera,
            mesh,
            p_final,
            out_size=(H_img, W_img),
            valid_mesh_mask=valid_mesh_mask_np,
        )
    _save_image(args.output, output)


if __name__ == "__main__":
    run_pipeline(parse_args())
