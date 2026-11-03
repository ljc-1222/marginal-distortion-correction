# Development reference

[Project overview](../README.md) · [Using the system](usage.md) · [Results and analysis](results.md)

## Code layout

- `annotate.py` starts the integrated annotation workflow; `run_mad.py` starts the correction runner.
- `annotation_gui/` provides image loading, coordinate mapping, camera/view setup, and shared session state.
- `pipeline/` connects the annotation stages, correction runner GUI, and fallback merge utility.
- `sam2/` contains the ROI frontend and bundled SAM2 predictor code.
- `interactive_snapping_2d/` contains the line/curve snapping frontend and algorithms.
- `MaDCoW/` contains the correction implementation, camera models, losses, optimization, and rendering.
- `ULSD/` contains the explored automatic line-candidate frontend.

The integrated annotator uses `interactive_snapping_2d/config/snap_config.json`. The top-level correction wrapper uses `MaDCoW/config.json`; the `MaDCoW.main` entry point additionally accepts `--config`.

## Annotation contract

The complete JSON is the interface between annotation and correction. Paths are resolved relative to the JSON file.

Common fields are `image_path`, `camera_model`, `lines`, and `regions`. Supported camera models are `pinhole` and `panorama_view`.

- A **pinhole** annotation stores the horizontal FOV in `fov_deg`.
- A **panorama-view** annotation stores `schema_version: 2`, `source_image_path`, and `view` metadata describing the derived annotation image, source size, view center, crop, and FOV.
- Each `lines[].points_dir` contains exactly 128 pairs of angular coordinates `(λ, φ)` in radians. The samples trace an input-image feature corresponding to a real-world straight structure.
- Each ROI has a `name` and `mask_path`. The mask is an 8-bit PNG matching the active annotation image dimensions; the correction loader selects pixels with values at least 128.

For panorama-view annotations, correction uses the recorded `image_path`. An `--image` override pointing elsewhere is rejected because the derived view metadata would no longer describe the input image.

The [MaDCoW reference](../MaDCoW/README.md#annotation-json) includes the full JSON examples. The [ROI reference](../sam2/README.md) and [snapping reference](../interactive_snapping_2d/README.md) describe standalone frontend outputs.

## Merge existing standalone annotations

The integrated workflow writes a complete JSON directly. To combine previously saved ROI-only and line-only annotations:

```bash
python -m pipeline.annotation_merge \
    --roi-json path/to/roi.json \
    --line-json path/to/lines.json \
    --output path/to/annotation.json
```

The utility checks image path, camera model, pinhole FOV, source path, and panorama-view metadata before merging.

## Checks

From an installed environment, run the entry-point and merge checks after changing integration code:

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

The merge self-check exercises successful merges and mismatched metadata. Geometry, loss, and rendering modules also include executable self-checks, listed in the [MaDCoW reference](../MaDCoW/README.md#development-checks).

For setup-script changes, `bash -n setup.sh` checks shell syntax and `bash setup.sh --help` lists checkpoint-selection options. Review the [recorded dependencies](../requirements.txt) separately when changing Python, PyTorch, or CUDA versions.

## Sources and licenses

**MaDCoW.** The correction implementation follows Kevin Zhang, Jia-Bin Huang, Jose Echevarria, Stephen DiVerdi, and Aaron Hertzmann, *MaDCoW: Marginal Distortion Correction for Wide-Angle Photography with Arbitrary Objects*, CVPR 2025. The [implementation reference](../MaDCoW/README.md#paper-and-citation) links the paper and includes its citation.

**SAM 2.** The ROI frontend uses bundled code and checkpoints from [facebookresearch/sam2](https://github.com/facebookresearch/sam2). Its upstream license is retained in [sam2/LICENSE](../sam2/LICENSE). The [ROI reference](../sam2/README.md#paper-and-citation) includes the paper citation.

**ULSD.** The optional line-candidate frontend uses ULSD by Hao Li et al. The upstream license is retained in [ULSD/LICENSE](../ULSD/LICENSE); its [module reference](../ULSD/README.md#paper-and-citation) links the source and publication.

**Project figures.** The [asset source list](assets/SOURCES.md) identifies the inputs used for the comparison figures. The project was developed by 林進成 and 黃義凱 for Computational Photography at National Tsing Hua University.
