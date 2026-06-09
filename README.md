# Marginal Distortion Correction

This repository provides an annotation-guided workflow for marginal distortion
correction in wide-angle photographs. The top-level pipeline combines a shared
camera/view setup GUI, SAM2-assisted ROI masks, interactive snapping for
straight-structure line constraints, and the MaDCoW correction algorithm.

The integrated workflow is designed around one active annotation view. The user
selects the camera/view setup first, and both ROI masks and line constraints are
created on that same view. This avoids the wrong workflow where ROI annotation
and line annotation are launched as two independent GUIs with different camera
metadata and merged afterward.

## Project Components

- `annotation_gui/` stores shared image loading, preview/original coordinate
  mapping, pinhole FOV setup, and panorama center/crop setup.
- `sam2/` contains the SAM2-assisted ROI annotation GUI. It uses box and point
  prompts to predict masks and saves each accepted ROI as an 8-bit PNG.
- `interactive_snapping_2d/` contains the rough-stroke snapping GUI. It snaps
  user strokes to image evidence and exports MaDCoW `lines[].points_dir`
  samples.
- `MaDCoW/` contains the marginal distortion correction pipeline and the
  original `MaDCoW.main` CLI.
- `pipeline/` contains the top-level integration controller and the fallback
  annotation merge utility.
- `annotate.py` runs the integrated annotation workflow.
- `run_mad.py` runs MaDCoW on a complete annotation JSON.

## Pipeline

```text
input image
  -> annotate.py
     -> shared camera/view setup
        -> pinhole: edit horizontal FOV, then Done
        -> panorama: choose center and crop range, then Done
     -> SAM2 ROI annotation on the active annotation image
     -> interactive snapping line annotation on the same active annotation image
     -> complete MaDCoW-compatible annotation JSON
  -> run_mad.py
     -> MaDCoW correction
     -> corrected output image
```

For pinhole inputs, the active annotation image is the original image and the
JSON stores `camera_model: "pinhole"` plus `fov_deg`.

For panorama inputs, setup writes a derived cropped panorama view under the
workspace and the JSON stores `camera_model: "panorama_view"`,
`source_image_path`, `schema_version: 2`, and `view` metadata.

## Setup

Use Python 3. The setup script creates `.venv`, installs the Python
dependencies from `requirements.txt`, downloads the SAM2.1 checkpoints into
`sam2/checkpoints/`, and runs a lightweight import verification for the main
pipeline modules.

To install and keep the virtual environment active in the current shell:

```bash
source ./setup.sh
```

To install without activating the caller shell:

```bash
bash setup.sh
source .venv/bin/activate
```

The script requires either `curl` or `wget` to download SAM2 checkpoints. If a
checkpoint file already exists and is non-empty, the download is skipped.

The downloaded checkpoint files are:

```text
sam2/checkpoints/sam2.1_hiera_tiny.pt
sam2/checkpoints/sam2.1_hiera_small.pt
sam2/checkpoints/sam2.1_hiera_base_plus.pt
sam2/checkpoints/sam2.1_hiera_large.pt
```

The quick-start command below uses the large checkpoint.

## Quick Start

Create a workspace directory for generated masks, panorama views, intermediate
annotation files, logs, and the final annotation JSON.

```bash
mkdir -p data/example_workspace
```

Create one complete annotation JSON:

```bash
python annotate.py \
    --image data/test.png
    --workspace data/test \
    --output-annotation data/test/annotation.json \
    --sam2-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
    --sam2-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --device auto \
    --snap-config interactive_snapping_2d/config/snap_config.json
```

Run MaDCoW correction:

```bash
python run_mad.py \
    --annotations data/test/annotation.json \
    --config MaDCoW/config.json \
    --output data/test/test_corrected.png \
    --crop
```

For non-`panorama_view` annotations, `run_mad.py` also accepts `--image` as an
optional override if the annotation JSON does not contain the desired input
image path. For `panorama_view` annotations, omit `--image` unless it is exactly
the same path as the JSON `image_path`.

## Annotation GUI Workflow

The first GUI window is always the shared camera/view setup.

- `Pinhole`: preview the original image, edit horizontal FOV, then press
  `Done`.
- `Panorama`: drag the panorama preview to choose the view center, drag crop
  edges to choose the annotation range, then press `Done`.

After setup, SAM2 ROI annotation starts on the active annotation image.

- `Box`: draw a box prompt and predict the current ROI mask.
- `Point`: add positive point prompts and predict the current ROI mask.
- `Redraw`: clear the current ROI draft.
- `Next ROI`: accept the current non-empty ROI and start the next one.
- `Save+Close`: save the ROI phase output and close the ROI window.

After ROI annotation, interactive snapping starts on the same active annotation
image.

- Left-drag: draw a rough stroke near a real-world straight structure.
- `Type`: choose `line` or `curve` if the GUI exposes both modes.
- `Redraw`: discard the pending snapped result.
- `Next`: accept the pending snapped result.
- `Save+Close`: save the line phase output and close the line window.

The integrated controller then writes one final annotation JSON at
`--output-annotation` and writes `annotation_summary.json` in the workspace.

## Annotation JSON

The final JSON is compatible with `MaDCoW/main.py`.

Common fields:

- `image_path`: active annotation image path.
- `camera_model`: `pinhole` or `panorama_view`.
- `lines`: snapped line annotations.
- `regions`: ROI mask annotations.

Pinhole fields:

- `fov_deg`: horizontal field of view in degrees.

Panorama-view fields:

- `source_image_path`: original equirectangular source image.
- `schema_version`: `2`.
- `view`: center, crop, size, and FOV metadata for the derived annotation view.

Line entries:

- `lines[].points_dir`: exactly 128 `(lambda, phi)` view-sphere samples in
  radians. These samples represent image curves corresponding to real-world
  straight structures.

Region entries:

- `regions[].name`: ROI name.
- `regions[].mask_path`: path to an 8-bit PNG mask, relative to the annotation
  JSON when possible.

## Line Constraint Notes

MaDCoW line constraints represent real-world straight structures. The
integrated workflow uses interactive snapping rather than pure manual line
drawing or automatic line detection. If curve mode is available in the snapping
GUI, use it only when the underlying real-world structure is actually straight.

## Workspace And Generated Files

`annotate.py` creates workspace subdirectories as needed:

```text
<workspace>/
  annotation/
  masks/
  views/
  logs/
  annotation_summary.json
```

Generated checkpoints, masks, panorama views, annotation JSON files, and
corrected images are local assets. Do not commit large checkpoints or generated
workspace outputs.

## Fallback Merge Utility

The normal top-level annotation path shares one camera/view setup and writes one
final JSON. `pipeline.annotation_merge` exists only for fallback and testing
when separate ROI-only and line-only JSON files already exist.

```bash
python -m pipeline.annotation_merge \
    --roi-json <roi-json> \
    --line-json <line-json> \
    --output <full-json>
```

The merge utility validates image path, camera model, pinhole FOV,
`source_image_path`, and panorama `view` metadata before writing output.

## Development Checks

Run these checks after code changes:

```bash
python -m compileall MaDCoW annotation_gui interactive_snapping_2d sam2 pipeline annotate.py run_mad.py
python annotate.py --help
python run_mad.py --help
python -m MaDCoW.main --help
python -m sam2.annotate_ROI_auto --help
python -m interactive_snapping_2d.annotate_line_aid --help
python -m pipeline.annotation_merge --help
python -m pipeline.annotation_merge --self-check
```

The merge self-check creates temporary annotation JSON files and a temporary
mask file, verifies successful merge behavior, and verifies clear errors for
mismatched image path, camera model, and pinhole FOV.

