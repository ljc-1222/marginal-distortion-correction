# MaDCoW

This directory contains a Python implementation of the MaDCoW pipeline from
*MaDCoW: Marginal Distortion Correction for Wide-Angle Photography with
Arbitrary Objects* (Zhang et al., CVPR 2025).

The implementation corrects marginal distortion by optimizing a nonlinear mesh
warp over the input camera's angular domain. Users provide straight-line
constraints and optional region-of-interest (ROI) masks. The pipeline fits one
local projection per ROI, blends those projections with a stereographic
initialization, then optimizes the full image warp with differentiable losses.

## Paper and Citation

MaDCoW addresses marginal distortion in wide-angle images, especially distorted
object appearance near image borders. The original paper formulates the task as
an annotation-guided warp: users mark straight lines and regions of interest,
then the method estimates local perspective-like projections and solves a
global warp that balances straight-line preservation, conformality,
smoothness, and ROI projection consistency.

- Paper: [MaDCoW: Marginal Distortion Correction for Wide-Angle Photography
  with Arbitrary Objects](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_MaDCoW_Marginal_Distortion_Correction_for_Wide-Angle_Photography_with_Arbitrary_Objects_CVPR_2025_paper.html)
- Venue: CVPR 2025

```bibtex
@InProceedings{Zhang_2025_CVPR,
    author = {Zhang, Kevin and Huang, Jia-Bin and Echevarria, Jose and DiVerdi, Stephen and Hertzmann, Aaron},
    title = {MaDCoW: Marginal Distortion Correction for Wide-Angle Photography with Arbitrary Objects},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2025},
    pages = {10923-10932}
}
```

## Current Pipeline

1. Load the input image, annotation JSON, and pipeline config.
2. Build the input camera from the annotation's `camera_model` and optional
   view metadata.
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
- `annotate.py` - Matplotlib GUI for manual line and ROI annotation.
- `config.json` - default mesh resolution, loss weights, and optimizer
  settings.
- `data/` - optional local workspace for input images, annotation JSON files,
  and ROI mask PNGs.
- `outputs/` - optional local workspace for generated correction outputs.
- `src/camera.py` - input camera models and pixel/ray conversion.
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

Use your own input image path. The command does not accept `--camera-model` or
`--fov`; camera setup happens inside the GUI.

```bash
./.venv/bin/python -m MaDCoW.annotate \
    --image <input-image> \
    --output-dir <annotation-output-dir>
```

The GUI first asks for the camera/view setup:

- `Pinhole`: use the original image as a pinhole image. The tool estimates
  horizontal FOV from EXIF when possible, falls back to 90 degrees, and lets the
  user edit the FOV text box before pressing `Done`.
- `Panorama`: treat the original image as a 360x180 equirectangular source,
  choose a centered/cropped annotation view, then press `Done`. The tool writes
  a derived view image into the output directory and saves
  `camera_model: "panorama_view"` with `source_image_path` and `view` metadata.

After camera setup, annotation is split into two phases:

- ROI phase: left-drag paints the current ROI mask.
- `Redraw`: clear the current ROI draft.
- `Next ROI` or `n` / `Tab`: accept the current non-empty ROI and start the
  next ROI draft.
- `Brush -` / `Brush +` or `-` / `+`: adjust brush radius.
- `Done ROI` or `d` / `Enter`: lock ROI masks and switch to line annotation.
- Line phase: left-drag along an input-image curve corresponding to a
  real-world straight structure. The stroke is resampled to 128 view-sphere
  points and saved as `points_dir`.
- `Redraw` or `r` / `Esc`: clear the pending ROI or line draft.
- `Next` or `n` / `Enter` / `Space`: accept the pending line.
- `Save` or `s` / `Ctrl+S` / `Cmd+S`: save JSON and masks.
- `Save+Close`: save the annotation JSON and close the GUI after a successful
  save.

The GUI writes `<image_stem>.json` and one `mask_<roi>.png` file per non-empty
ROI into the output directory.

## Run Correction

If the annotation JSON already contains the correct `image_path`, no image
argument is needed:

```bash
./.venv/bin/python -m MaDCoW.main \
    --annotations <annotation-json> \
    --config MaDCoW/config.json \
    --output <output-image> \
    --crop
```

For non-`panorama_view` annotations, `--image` can override the image path saved
inside the annotation JSON:

```bash
./.venv/bin/python -m MaDCoW.main \
    --image <input-image> \
    --annotations <annotation-json> \
    --config MaDCoW/config.json \
    --output <output-image>
```

`--crop` is optional. When enabled, the renderer returns a validity mask and
the output is cropped to the largest black-border-free rectangle with the
input aspect ratio. Because cropping uses the validity mask, legitimate black
pixels in the input are not treated as borders.

`main.py` does not provide a CLI flag for choosing the camera model. The camera
model must already be present in the annotation JSON.

## Annotation JSON

The annotation file contains the input image path, input camera model,
straight-line constraints, and ROI mask paths. Pinhole annotations also contain
the input horizontal FOV:

```json
{
    "image_path": "input.jpg",
    "camera_model": "pinhole",
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

A full equirectangular input can be provided by external tools or manually
authored annotations using `camera_model: "360"`:

```json
{
    "image_path": "panorama.jpg",
    "camera_model": "360",
    "lines": [],
    "regions": []
}
```

The current `annotate.py` panorama setup writes a derived annotation view with
v2 metadata:

```json
{
    "image_path": "panorama_view.png",
    "source_image_path": "panorama.jpg",
    "camera_model": "panorama_view",
    "schema_version": 2,
    "view": {
        "type": "panorama_view",
        "source_camera_model": "360",
        "projection": "equirectangular_crop",
        "source_size": [4000, 2000],
        "view_size": [2000, 1000],
        "preview_size": [1200, 600],
        "center_yaw_rad": 0.0,
        "center_pitch_rad": 0.0,
        "crop_original_px": [1000.0, 0.0, 3000.0, 2000.0],
        "crop_preview_px": [300.0, 0.0, 900.0, 600.0],
        "horizontal_fov_deg": 179.0,
        "vertical_fov_deg": 179.0
    },
    "lines": [],
    "regions": []
}
```

Paths inside the JSON are resolved relative to the JSON file. Line samples are
view-sphere directions in radians: `lambda` is yaw and `phi` is pitch. The
annotated curve may appear curved in the input image, but it should represent
a real-world straight structure. The output line loss forces the warped curve
samples to become collinear.

Annotations missing `camera_model` are rejected by `main.py`. Supported camera
models are `pinhole`, `360`, and `panorama_view`.

For `panorama_view` annotations, `main.py` must run on the saved
`image_path`. Passing `--image` with a different path is rejected to avoid
mixing the derived view geometry with another image.

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

The mesh extent is derived from the input camera rays; changing the config mesh
values changes sampling density, not a separate output camera model.

## Smoke Test With Your Own Files

Create or provide an annotation JSON for your own image, then run:

```bash
./.venv/bin/python -m MaDCoW.main \
    --image <input-image> \
    --annotations <annotation-json> \
    --config MaDCoW/config.json \
    --output <output-image> \
    --crop
```

For `panorama_view` annotations, omit `--image` unless it is exactly the same
path as the annotation's `image_path`.

## Development Checks

Several modules contain small executable self-checks:

```bash
./.venv/bin/python -m compileall MaDCoW
./.venv/bin/python -m MaDCoW.main --help
./.venv/bin/python -m MaDCoW.annotate --help
./.venv/bin/python -m MaDCoW.src.camera
./.venv/bin/python -m MaDCoW.src.render
./.venv/bin/python -m MaDCoW.src.losses
./.venv/bin/python -m MaDCoW.src.stage2_warp
```
