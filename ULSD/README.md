# ULSD Line Annotation

`ULSD/` contains a ULSD-based line detection and review frontend for MaDCoW. It
runs the detector on a user-provided image, lets the user keep, drop, or add
line candidates, then exports a MaDCoW-compatible annotation JSON with
`lines[].points_dir`.

It does not run MaDCoW optimization, ROI segmentation, SAM2, or interactive 2D
stroke snapping.

## Paper and Citation

ULSD is a learning-based line segment detector designed to work across pinhole,
fisheye, and spherical images without requiring camera-specific undistortion.
It represents possibly distorted line segments with equipartition points on a
Bezier curve and regresses those points with a neural network. In this
repository, ULSD is used as an automatic line-candidate generator whose output
is reviewed before conversion to MaDCoW `lines[].points_dir`.

- Paper: [ULSD: Unified Line Segment Detection across Pinhole, Fisheye, and
  Spherical Cameras](https://www.sciencedirect.com/science/article/pii/S0924271621001623)
- Code/models: [Unified-Line-Segment-Detection](https://github.com/lh9171338/Unified-Line-Segment-Detection)
- Venue: ISPRS Journal of Photogrammetry and Remote Sensing, 2021

```bibtex
@article{LI2021187,
    title = {ULSD: Unified line segment detection across pinhole, fisheye, and spherical cameras},
    author = {Hao Li and Huai Yu and Jinwang Wang and Wen Yang and Lei Yu and Sebastian Scherer},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {178},
    pages = {187-202},
    year = {2021},
    doi = {10.1016/j.isprsjprs.2021.06.004},
    url = {https://www.sciencedirect.com/science/article/pii/S0924271621001623}
}
```

## Current Pipeline

1. Load the input image, ULSD config, line-selection config, and model weights.
2. Run ULSD on the image and rescale detected line candidates back to the
   original image resolution.
3. Filter candidates by score, `top_k`, and duplicate distance.
4. Open a Matplotlib review GUI where candidates can be selected, dropped, or
   supplemented with hand-drawn lines.
5. Save a MaDCoW annotation JSON and a marked preview image of the selected
   lines.

## Repository Layout

- `annotate_line_auto.py` - command-line entry point and review GUI.
- `config/default.yaml` - ULSD model and inference configuration.
- `config/line_selection.json` - score, count, and duplicate filtering
  defaults.
- `model/` - optional local directory for ULSD model weights.
- `network/` - ULSD network implementation.
- `util/bezier.py` - line interpolation helper.
- `data/` - optional local workspace for input images, JSON output, and marked
  line images.
- `LICENSE` - upstream ULSD license file for the bundled code.

## Run Line Review

Use your own input image path. The current GUI treats the input as a full
equirectangular panorama, uses `spherical.pkl` unless `--model-name` overrides
the model filename, and exports MaDCoW v2 `panorama_view` metadata.

```bash
./.venv/bin/python -m ULSD.annotate_line_auto \
    --image <input-image> \
    --output-dir <annotation-output-dir> \
    --selection-config ULSD/config/line_selection.json \
    --gpu -1
```

Common optional arguments:

- `--model-name <weights-file>`: override the default camera-specific model
  filename.
- `--marked-image <output-image>`: choose where the selected-line preview is
  saved.
- `--json <annotation-json>`: choose the annotation JSON output path.
- `--score-thresh <value>`: override `score_thresh` from the selection config.
- `--top-k <count>`: override `top_k` from the selection config.
- `--duplicate-dist-px <pixels>`: override duplicate suppression distance.
- `--junc-score-thresh <value>` and `--line-score-thresh <value>`: override
  ULSD proposal thresholds.
- `--gpu <id>`: use a CUDA device when available. Use `-1` for CPU.

## GUI Controls

- `Review` or `v`: click a candidate line to toggle selected/dropped state.
- `Draw` or `l`: left-drag to add a hand-drawn line.
- `Select All` or `a`: select every candidate.
- `Drop All` or `d`: drop every candidate.
- `Invert` or `i`: invert candidate selection.
- `Rerun` or `r`: rerun detection with the current thresholds.
- `Undo` or `u` / `Ctrl+Z` / `Cmd+Z`: remove the last hand-drawn line.
- `Save` or `s` / `Ctrl+S` / `Cmd+S`: save JSON and marked image.
- `Save+Close`: save JSON and marked image, then close the GUI.
- `Esc`: cancel the active hand-drawn line.

## JSON Output

Save writes `<stem>.json` and a marked selected-line image. The JSON contains
line annotations and no ROIs. A full equirectangular panorama is recorded as a
`panorama_view` whose crop covers the whole source image:

```json
{
  "image_path": "input.jpg",
  "source_image_path": "input.jpg",
  "camera_model": "panorama_view",
  "schema_version": 2,
  "view": {
    "type": "panorama_view",
    "source_camera_model": "panorama",
    "projection": "equirectangular_crop",
    "source_size": [4000, 2000],
    "view_size": [4000, 2000],
    "preview_size": [1200, 600],
    "center_yaw_rad": 0.0,
    "center_pitch_rad": 0.0,
    "crop_original_px": [0.0, 0.0, 4000.0, 2000.0],
    "crop_preview_px": [0.0, 0.0, 1200.0, 600.0],
    "horizontal_fov_deg": 360.0,
    "vertical_fov_deg": 180.0
  },
  "lines": [
    {
      "points_dir": [[-3.02, -1.45], "... 126 more samples ..."]
    }
  ],
  "regions": []
}
```

Each selected line is resampled to 128 points, converted from image pixels to
view-sphere `(lambda, phi)` directions through the MaDCoW camera model, and
stored as `points_dir`.

## Selection Config

`config/line_selection.json` currently has this structure:

```json
{
    "score_thresh": 0.5,
    "top_k": 30,
    "duplicate_dist_px": 6.0
}
```

- `score_thresh`: keep candidates whose ULSD score is greater than this value.
- `top_k`: keep at most this many candidates after thresholding. `0` disables
  the count cap.
- `duplicate_dist_px`: suppress near-duplicate candidate lines within this
  average pixel distance.

## Run MaDCoW With the Line JSON

```bash
./.venv/bin/python -m MaDCoW.main \
    --image <input-image> \
    --annotations <annotation-json> \
    --config MaDCoW/config.json \
    --output <output-image> \
    --crop
```

## Development Checks

```bash
./.venv/bin/python -m compileall ULSD
./.venv/bin/python -m ULSD.annotate_line_auto --help
```
