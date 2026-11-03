# Using the system

[Project overview](../README.md) · [Results and analysis](results.md) · [Development reference](development.md)

## Installation

Run the commands from the repository root in a Bash shell. The annotation and runner interfaces use Matplotlib and need an interactive desktop display. SAM2 can use CUDA or CPU through `--device auto`; the recorded dependency file includes an NVIDIA/CUDA environment and should be reviewed when installing on another platform.

```bash
bash setup.sh --sam2-model small
source .venv/bin/activate
```

The setup script creates `.venv`, installs `requirements.txt`, downloads the selected SAM2.1 checkpoint, and checks imports for the integrated workflow. `source ./setup.sh --sam2-model small` also keeps the environment active in the current shell.

Select a checkpoint with `--sam2-model tiny`, `small`, `base-plus`, or `large`. The GUI exposes whichever checkpoints are present under `sam2/checkpoints/`. Running `bash setup.sh` without options preserves the original behavior and downloads all four models; `--sam2-model all` does the same explicitly. Existing non-empty checkpoints are reused. Downloads require `curl` or `wget`.

The checkpoints use the filenames `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_small.pt`, `sam2.1_hiera_base_plus.pt`, and `sam2.1_hiera_large.pt`. Their model configurations are bundled in the repository.

`requirements.txt` records the existing project environment, including pinned Python packages and CUDA components. The import check confirms that modules load; it does not run segmentation or image correction.

## Create an annotation

Start with the bundled image or supply your own path:

```bash
python annotate.py --image data/test1/image.png --workspace annotation/test1
```

The CLI arguments prefill the integrated GUI. The workflow stays in one window, with a left sidebar for the current step and the image canvas on the right.

1. **Select the image and output directory.** Inspect the preview. The default workspace is `annotation/<input_parent_folder>`; the example above uses `annotation/test1`.
2. **Select a SAM2 model.** Choose a model with an available checkpoint. If installation used `--sam2-model small`, choose **Small**.
3. **Set the camera and annotation view.** For a pinhole image, confirm the horizontal FOV in degrees. For an equirectangular panorama, select a view center and crop range. Press **Done** to use that view for both annotation stages.
4. **Create ROI masks.** Draw a box or add positive point prompts around an object to preserve. Review the predicted mask; use **Redraw** to discard the draft, **Next ROI** to accept it, and **Done ROI** to proceed. Mask post-processing keeps sufficiently large connected components and fills holes.
5. **Annotate structural features.** Draw a rough stroke near a real-world straight structure, inspect the snapped result, and press **Next** to accept it. Use **Redraw** to retry and **Save Final** to finish.

The final annotation is written to `<workspace>/annotation.json`. A custom path can be supplied through `--output-annotation`. `--device auto` prefers CUDA for SAM2 when available; `--device cpu` explicitly selects CPU inference.

### Choose the right snapping mode

**Line** mode returns an exactly straight segment in the displayed image plane. Use it when the target structure already appears straight in that view.

**Curve** mode follows a curved image boundary. Use it when a real-world straight structure appears curved because of the input projection. The selected feature still becomes a straightness constraint in the correction stage. Decorative curves and genuinely curved objects are unsuitable targets for this constraint.

Snapping uses nearby image evidence, so inspect the result before accepting it. Strong texture, adjacent edges, or weak contrast can pull the stroke toward a different boundary. See the [snapping reference](../interactive_snapping_2d/README.md) for algorithm details and controls.

### Camera and view consistency

For `pinhole`, annotations refer to the original image and store its horizontal FOV. For `panorama_view`, the workflow saves a derived view and its source, center, crop, and FOV metadata. ROI masks and line annotations are created on that same active view.

Keep the annotation JSON, masks, and derived view together when moving a workspace. Their paths are saved relative to the annotation JSON where possible. The [development reference](development.md#annotation-contract) describes the saved fields.

## Run correction

To select an annotation and output settings interactively:

```bash
python run_mad.py
```

The runner previews the image referenced by the JSON, lets you choose the output name and crop setting, displays progress for the two optimization stages and final rendering, and previews the saved result.

To run the same operation from the command line:

```bash
python run_mad.py \
    --annotations annotation/test1/annotation.json \
    --output outputs/test1/image_corrected.png \
    --summary outputs/test1/summary.json \
    --crop
```

`--summary` writes the camera type, annotation counts, configuration path, output path, and crop setting. It is a run record rather than a quality metric. Cropping uses the renderer's validity mask and preserves the input aspect ratio; omit `--crop` to keep the full output.

The wrapper uses `MaDCoW/config.json`. To run a different correction configuration, use the module entry point:

```bash
python -m MaDCoW.main \
    --annotations annotation/test1/annotation.json \
    --config MaDCoW/config.json \
    --output outputs/test1/image_corrected.png \
    --crop
```

For panorama-view annotations, use the image already recorded in the JSON so that the view geometry remains consistent. A pinhole annotation can use `--image` to override its image path.

## Output organization

For the example commands, the completed annotation is `annotation/test1/annotation.json`, ROI masks are stored in `annotation/test1/masks/`, and derived panorama images are stored in `annotation/test1/views/`. The workspace also includes `annotation_summary.json` and working subdirectories used by the integrated controller.

Correction outputs are saved under `outputs/test1/`. Checkpoints, working annotations, masks, and rendered outputs are local assets; the curated comparison images in `docs/assets/` are the public project examples.
