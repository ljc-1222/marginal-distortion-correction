# MaDCoW

This directory contains a Python implementation of the MaDCoW pipeline from
*MaDCoW: Marginal Distortion Correction for Wide-Angle Photography with
Arbitrary Objects* (Zhang et al., CVPR 2025).

The implementation corrects marginal distortion by optimizing a nonlinear mesh
warp over the input camera's angular domain. Users provide straight-line
constraints and region-of-interest (ROI) masks, the pipeline fits one local
projection per ROI, blends those projections with a stereographic
initialization, then optimizes the full image warp with differentiable losses.

## Current Pipeline

1. Load the input image, annotation JSON, and pipeline config.
2. Build an input perspective camera from the image size and horizontal FOV.
3. Build the mesh from the input camera rays by sampling the image border and
   covering the resulting `(lambda, phi)` domain.
4. Rasterize each ROI mask from input-image pixels onto the angular mesh.
5. Run Stage 1 per ROI to fit `T_k(v) = S * P(v) + t`.
6. Initialize the full mesh from stereographic projection and the per-ROI
   projections.
7. Run Stage 2 L-BFGS optimization with the weighted objective
   `w_l * E_l + w_c * E_c + w_s * E_s + w_dvc * E_DVC`.
8. Render the corrected image by inverse-bilinear sampling from the input
   image. Optional cropping uses the renderer's validity mask, not RGB values,
   and preserves the input aspect ratio.

## Repository Layout

- `main.py` - command-line entry point for the correction pipeline.
- `annotate.py` - Matplotlib GUI for line and ROI annotation.
- `config.json` - default mesh resolution, loss weights, and optimizer
  settings.
- `data/` - sample image, annotation JSON files, and ROI mask PNGs.
- `outputs/` - generated example outputs.
- `src/camera.py` - input perspective camera and pixel/ray conversion.
- `src/mesh.py` - angular mesh construction, valid-domain masks, ROI mask
  rasterization, and mesh sampling.
- `src/projections.py` - stereographic, perspective, and per-ROI projection
  functions.
- `src/stage1_region.py` - per-ROI projection fitting.
- `src/initialization.py` - full-mesh initialization.
- `src/losses.py` - Stage 1 and Stage 2 loss terms.
- `src/stage2_warp.py` - full-image L-BFGS optimization.
- `src/render.py` - final rasterization and optional cropping.

## Annotate an Image

```bash
./.venv/bin/python -m MaDCoW.annotate \
    --image MaDCoW/data/test_1.png \
    --fov 90 \
    --output-dir MaDCoW/data
```

`--fov` is the horizontal FOV of the input image in degrees. If it is omitted,
the tool tries to estimate it from EXIF metadata and falls back to 90 degrees.
The GUI writes `<image_stem>.json` and one `mask_<roi>.png` file per non-empty
ROI into the output directory.

Useful GUI controls:

- `l` or the `Line` button: switch to straight-line mode.
- Left-drag in line mode along an input-image curve corresponding to a
  real-world straight structure. The stroke is resampled to 128 view-sphere
  points and saved as `points_dir`.
- `r` or the `Region` button: switch to ROI painting mode.
- Left drag in region mode: paint the selected ROI.
- Right drag in region mode: erase the selected ROI.
- `n`: create a new ROI.
- `[` / `]`: switch ROI.
- `+` / `-`: change brush radius.
- `u`: undo the last line or mask stroke.
- `c`: clear the selected ROI.
- `s` or `Save`: save JSON and masks.

## Run Correction

```bash
./.venv/bin/python -m MaDCoW.main \
    --image MaDCoW/data/test_1.png \
    --annotations MaDCoW/data/test_1.json \
    --config MaDCoW/config.json \
    --output MaDCoW/outputs/result_1.jpg \
    --crop
```

`--crop` is optional. When enabled, the renderer returns a validity mask and
the output is cropped to the largest black-border-free rectangle with the
input aspect ratio. Because cropping uses the validity mask, legitimate black
pixels in the input are not treated as borders.

## Annotation JSON

The annotation file contains the input image path, input horizontal FOV,
straight-line constraints, and ROI mask paths:

```json
{
    "image_path": "test_1.png",
    "fov_deg": 90.0,
    "lines": [
        {
            "points_dir": [
                [0.0, 0.0],
                [0.001, 0.0002],
                "... 126 more points ..."
            ]
        }
    ],
    "regions": [
        {
            "name": "region_1",
            "mask_path": "mask_region_1.png"
        }
    ]
}
```

Paths inside the JSON are resolved relative to the JSON file. Line samples are
view-sphere directions in radians: `lambda` is yaw and `phi` is pitch. The
annotated curve may appear curved in the input image, but it should represent
a real-world straight structure. The output line loss forces the warped curve
samples to become collinear.

## Config

`config.json` currently has this structure:

```json
{
    "mesh": {
        "n_lambda": 128,
        "n_phi": 128
    },
    "loss_weights": {
        "w_l": 25,
        "w_c": 1,
        "w_s": 12,
        "w_dvc": 1
    },
    "blend_c": 30,
    "stage1_max_iter": 150,
    "stage2_max_iter": 300,
    "lbfgs_lr": 1.0
}
```

- `mesh.n_lambda`, `mesh.n_phi`: angular mesh resolution.
- `loss_weights.w_l`: straight-line preservation loss weight.
- `loss_weights.w_c`: conformal loss weight.
- `loss_weights.w_s`: smoothness loss weight.
- `loss_weights.w_dvc`: ROI projection matching loss weight.
- `blend_c`: exponential blending scale used during full-mesh initialization.
- `stage1_max_iter`: maximum Newton iterations for each ROI projection fit.
- `stage2_max_iter`: maximum L-BFGS iterations for the full warp.
- `lbfgs_lr`: L-BFGS learning rate.

The mesh extent is derived from the input camera rays; changing the config
mesh values changes sampling density, not a separate output camera model.

## Example Data

The included sample files can be used as quick smoke tests:

```bash
./.venv/bin/python -m MaDCoW.main \
    --image MaDCoW/data/test_1.png \
    --annotations MaDCoW/data/test_1.json \
    --config MaDCoW/config.json \
    --output MaDCoW/outputs/result_1.jpg \
    --crop
```

## Development Checks

Several modules contain small executable self-checks:

```bash
./.venv/bin/python -m compileall MaDCoW
./.venv/bin/python -m MaDCoW.main --help
./.venv/bin/python -m MaDCoW.annotate --help
./.venv/bin/python -m MaDCoW.src.render
./.venv/bin/python -m MaDCoW.src.losses
./.venv/bin/python -m MaDCoW.src.stage2_warp
```
