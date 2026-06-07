# Interactive Snapping 2D

`interactive_snapping_2d/` is a MaDCoW annotation frontend for snapping a rough
mouse stroke to nearby 2D image evidence and exporting `lines[].points_dir`.

It does not run MaDCoW optimization, ROI correction, image warping, ULSD, or
full-image line detection. It only writes the annotation JSON consumed by
`MaDCoW/main.py`.

## Algorithm

The snapper routes by annotation mode. `camera_type` only controls panorama
seam unwrap/wrap; it does not change the geometry contract of `line` or
`curve`.

1. Clean, smooth, and uniformly resample the rough stroke.
2. Compute local edge evidence with multi-scale gradients, Canny edges, edge
   tangent direction, and Lab color contrast across candidate normals.
3. For `line`, fit an initial 2D line from the rough stroke, search nearby
   angle/rho candidates, score each angle's rho candidates in a vectorized
   batch by edge, orientation-gated edge evidence, color contrast, continuity,
   and distance to the rough stroke, then robustly refit and output exact
   collinear line samples.
4. For `curve`, extract local normal-profile peaks, optionally score paired
   edges around a wide band center, run peak-level dynamic programming with
   offset jump, polarity, orientation, and curvature penalties, apply subpixel
   peak refinement, then smooth and resample the curve.

The public API is:

```python
from interactive_snapping_2d.snap2d import snap_annotation

result = snap_annotation(image, stroke, camera_type="pinhole", mode="curve")
points_xy = result.points
```

## Modes

- `line`: always outputs a geometrically exact 2D straight line segment in
  input-image pixel coordinates. This is true for both `pinhole` and
  `panorama`.
- `curve`: outputs 2D curve samples that snap to nearby image boundaries rather
  than only smoothing the user's stroke.

## Camera Types

- `pinhole`: snaps directly in input image coordinates.
- `panorama`: horizontally unwraps strokes that cross the equirectangular
  seam, snaps in the unwrapped 2D image, then wraps output `x` coordinates back
  to `[0, W)`.

## Line-Aid GUI

```bash
./.venv/bin/python -m interactive_snapping_2d.annotate_line_aid \
    --image interactive_snapping_2d/data/test_1.jpg \
    --config-json interactive_snapping_2d/config/snap_config.json
```

Mouse and keyboard controls:

- `Camera`: choose `pinhole` or `panorama`.
- `Type`: choose `line` or `curve`.
- Left drag: draw a rough stroke; mouse release snaps it with the selected
  camera and mode.
- `Redraw`: discard the pending snapped result and draw again.
- `Next`: accept the pending snapped result and start the next annotation.
- `Save`: accept any pending result, then save a MaDCoW-compatible annotation
  JSON.
- `Save + Close`: save the annotation JSON and close the GUI after a successful
  save.
- `R` or `ESC`: discard the pending result or current stroke.
- `N`, `Enter`, or `Space`: accept the pending snapped result.
- `S`, `Ctrl+S`, or `Cmd+S`: save.

The default MaDCoW annotation is compatible with `MaDCoW/main.py`: it contains
`image_path`, `camera_model`, pinhole `fov_deg`, `lines[].points_dir` with 128
view-sphere samples, and an empty `regions` list. Use one camera type per saved
MaDCoW file; the MaDCoW export uses the current global `Camera` setting.

## Snap Parameters

Default GUI parameters are stored in:

```text
interactive_snapping_2d/config/snap_config.json
```

The current curve defaults keep `profile_top_k=11` so real edges are not
dropped from the candidate set. They smooth the selected normal-offset sequence
(`offset_smooth_window=13`, `offset_smooth_passes=2`) before final curve
construction, then apply a lower cutoff output smoother
(`output_smooth_window=15`, `output_smooth_passes=3`) to suppress remaining
high-frequency jitter. The `normal_gradient_consistency_*` parameters can run a
second curve-DP pass that penalizes reliable normal-gradient sign changes,
which is useful for testing same-side snapping on thick objects. It defaults to
`0.0` because the current real-image ablation favored offset smoothing as the
default same-side stabilizer. The
`band_center_*` parameters are available for wide shadow or thick-line cases,
but `band_center_weight` also defaults to `0.0` because the current real-image
ablation did not justify the extra runtime as a default.

## JSON Output

Save writes `<stem>.json` in the output directory. It converts the 2D snapped
annotations to the current `main.py` schema:

```json
{
  "image_path": "../data/test_1.jpg",
  "camera_model": "360",
  "lines": [
    {
      "points_dir": [[-3.02, -1.45], "... 126 more samples ..."]
    }
  ],
  "regions": []
}
```

## Known Limitations

- Low-contrast boundaries may snap inaccurately.
- Strong texture near the stroke can attract the path to the wrong edge.
- Endpoints are not automatically extended far beyond the user stroke.
- Panorama handling only performs 2D horizontal seam unwrap; it does not do
  spherical geometry optimization.
