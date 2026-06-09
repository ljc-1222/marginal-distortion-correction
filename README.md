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
- `pipeline/` contains top-level GUI integration helpers and the fallback
  annotation merge utility.
- `annotate.py` runs the integrated annotation workflow.
- `run_mad.py` runs MaDCoW on a complete annotation JSON.

## Pipeline

```text
input image
  -> annotate.py
     -> integrated annotation GUI
        -> choose image from data folder with preview
        -> confirm output directory
        -> choose one SAM2 model
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

The integrated GUI exposes these SAM2.1 model choices:

| Model | Checkpoint | Notes |
| --- | --- | --- |
| Tiny | `sam2/checkpoints/sam2.1_hiera_tiny.pt` | Fastest, lowest memory, lowest mask quality. |
| Small | `sam2/checkpoints/sam2.1_hiera_small.pt` | Fast with moderate memory use. |
| Base Plus | `sam2/checkpoints/sam2.1_hiera_base_plus.pt` | Balanced quality and speed. |
| Large | `sam2/checkpoints/sam2.1_hiera_large.pt` | Best mask quality, highest memory use. |

If a checkpoint is missing, its model option is shown as unavailable.

## Quick Start

Create one complete annotation JSON:

```bash
python annotate.py
```

The integrated GUI starts in `data/`, shows supported image files, displays a
preview after selecting an image filename, proposes an output directory from the
input image's parent folder name, and writes the final JSON as
`<output_directory>/annotation.json`. For example, selecting
`data/test1/test.png` defaults to `annotation/test1/annotation.json` unless the
output directory is edited.

Run MaDCoW correction:

```bash
python run_mad.py
```

The runner GUI starts in `annotation/`, lets you choose a complete annotation
JSON, proposes an output directory from the annotation image's parent folder,
offers an editable output image name and crop toggle, shows Stage 1 and Stage 2
optimizer progress bars plus a final render/save progress bar, and previews the
corrected output after saving. For example, an annotation whose `image_path` is
`data/test1/test.png` defaults to `outputs/test1/` and writes
`test_corrected.png` or `test_corrected_crop.png`.

The CLI path is still available:

```bash
python run_mad.py \
    --annotations annotation/test1/annotation.json \
    --output data/test1/test_corrected_crop.png \
    --crop
```
Or if you want the uncrop one:

```bash
python run_mad.py \
    --annotations annotation/test1/annotation.json \
    --output data/test1/test_corrected.png \
```

For non-`panorama_view` annotations, `run_mad.py` also accepts `--image` as an
optional override if the annotation JSON does not contain the desired input
image path. For `panorama_view` annotations, omit `--image` unless it is exactly
the same path as the JSON `image_path`.

## MaDCoW Runner GUI

`run_mad.py` opens a single Matplotlib window when launched without CLI
arguments.

1. `Select Annotation`: browse from `annotation/`, click a full annotation JSON,
   and preview the image referenced by the JSON.
2. `Select Output`: confirm or edit the output directory, output image name,
   and crop setting. The default output filename is derived from the annotation
   `image_path`.
3. `Run MaDCoW`: Stage 1 and Stage 2 progress bars update from MaDCoW optimizer
   callbacks, then `Finalize output` tracks render, crop, and save.
4. `Complete`: the corrected image is loaded back into the preview canvas.

## Annotation GUI Workflow

`annotate.py` opens one Matplotlib window with a fixed left sidebar for the step
counter, setup controls, and run summary. The right side stays as the preview or
annotation canvas throughout the workflow.

1. `Select Image`: browse from `data/`, click an image filename, and inspect
   the preview in the same window.
2. `Select Output Directory`: confirm or edit the workspace. The default is
   `annotation/<input_parent_folder>`. For `data/test1/test.png`, this is
   `annotation/test1`, and the final JSON path is
   `annotation/test1/annotation.json`.
3. `Select SAM2 Model`: choose Tiny, Small, Base Plus, or Large. The GUI maps
   the model name to its fixed checkpoint and config path.
4. `Camera/View Setup`: choose the active annotation view. `Pinhole` previews
   the original image and exposes editable horizontal FOV. `Panorama` lets the
   user choose view center and crop range. Press `Done` to finalize setup.
5. `SAM2 ROI Annotation`: use box or point prompts, `Redraw`, `Next ROI`, and
   `Done ROI` to create ROI masks on the active annotation image.
6. `Snapping Line Annotation`: draw rough strokes near real-world straight
   structures, accept snapped results with `Next`, and press `Save Final`.
7. `Save Complete Annotation`: the integrated controller writes one final
   annotation JSON at `<output_directory>/annotation.json` and writes
   `annotation_summary.json` in the workspace.

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

`annotate.py` uses the fixed snapping config
`interactive_snapping_2d/config/snap_config.json`. `run_mad.py` uses the fixed
MaDCoW config `MaDCoW/config.json`.

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
