# SAM2 ROI Annotation

`sam2/` contains a SAM2-assisted ROI annotation frontend for MaDCoW. It uses
box or point prompts to create ROI masks, then exports a MaDCoW-compatible
annotation JSON with an empty `lines` list.

It does not run MaDCoW optimization, line annotation, ULSD line detection, or
2D stroke snapping.

## Paper and Citation

SAM2 is Meta's promptable segmentation foundation model for images and videos.
It extends the Segment Anything family with a streaming-memory transformer
architecture for video while preserving the point/box prompt workflow used for
single-image mask prediction. In this repository, SAM2 is used only as an ROI
mask proposal tool for MaDCoW annotations.

- Paper: [SAM 2: Segment Anything in Images and Videos](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
- Code: [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- Venue/source: ICLR 2025; official repository citation uses the arXiv entry.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

## Current Pipeline

1. Load the input image, SAM2 checkpoint, model config, and inference device.
2. Run the shared camera/view setup used by the MaDCoW annotator.
3. Build a SAM2 image predictor and compute the image embedding for the active
   annotation image.
4. Use box or point prompts to predict the current ROI mask.
5. Accept each non-empty ROI and repeat as needed.
6. Save `<image_stem>.json` and one `mask_<roi>.png` per saved ROI.

## Repository Layout

- `annotate_ROI_auto.py` - command-line entry point and Matplotlib ROI prompt
  GUI.
- `sam2/` - local SAM2 package code used by the predictor.
- `checkpoints/` - optional local checkpoint directory.
- `data/` - optional local workspace for input images and generated
  annotations.
- `LICENSE` - upstream SAM2 license file for the bundled code.

## Annotate ROIs

Use your own input image path. The checkpoint path must point to an existing
SAM2 checkpoint on your machine.

```bash
./.venv/bin/python -m sam2.annotate_ROI_auto \
    --image <input-image> \
    --output-dir <annotation-output-dir> \
    --checkpoint <sam2-checkpoint.pt> \
    --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --device auto
```

The default checkpoint path is `sam2/checkpoints/sam2.1_hiera_large.pt`; use
`--checkpoint` when that file is not present.

`--device auto` uses CUDA when available and otherwise falls back to CPU. A
specific device string such as `cpu` or `cuda:0` can also be passed.

## GUI Controls

The GUI first asks for the camera/view setup:

- `Pinhole`: use the original image with an editable horizontal FOV.
- `Panorama`: choose a centered/cropped equirectangular annotation view. The
  saved JSON uses `camera_model: "panorama_view"` with `source_image_path` and
  `view` metadata.

After setup, ROI controls are:

- `Box` or `b`: draw box prompts.
- `Point` or `p` / `f`: add positive point prompts.
- Left-drag in box mode: create a box prompt and run SAM2 prediction.
- Left-click in point mode: add a point prompt and run SAM2 prediction.
- `Redraw` or `r` / `c`: clear the current ROI draft.
- `Next ROI` or `n` / `Tab`: accept the current non-empty ROI and start the
  next ROI draft.
- `Save` or `s` / `Ctrl+S` / `Cmd+S`: save JSON and masks.
- `Save+Close`: save JSON and masks, then close the GUI.
- `Esc`: cancel the active box drag.

## JSON Output

Save writes `<stem>.json` in the output directory. It contains ROI masks and no
line annotations:

```json
{
  "image_path": "input.jpg",
  "camera_model": "pinhole",
  "fov_deg": 90.0,
  "lines": [],
  "regions": [
    {
      "name": "region_1",
      "mask_path": "mask_region_1.png"
    }
  ]
}
```

Paths inside the JSON are relative to the JSON file when possible. Each mask is
saved as an 8-bit PNG where nonzero pixels are treated as the ROI by
`MaDCoW/main.py`.

## Run MaDCoW With the ROI JSON

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

```bash
./.venv/bin/python -m compileall sam2
./.venv/bin/python -m sam2.annotate_ROI_auto --help
```
