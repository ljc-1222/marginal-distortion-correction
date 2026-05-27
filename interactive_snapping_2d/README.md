# Interactive Snapping 2D

`interactive_snapping_2d/` is an annotation frontend for snapping a rough
mouse stroke to nearby 2D image evidence. It outputs image-space points:

```json
[[123.4, 512.8], [124.9, 511.6], [126.2, 510.1]]
```

It does not run MaDCoW optimization, ROI correction, image warping, ULSD, or
full-image line detection. Downstream code can convert the 2D points to
longitude/latitude or view-sphere coordinates before marginal distortion
correction.

## Algorithm

The snapper uses a local 2D ribbon around the user stroke:

1. Clean, smooth, and uniformly resample the rough stroke.
2. Compute local edge strength with OpenCV gradients and optional Canny edges.
3. Build candidate points along each stroke normal.
4. Use dynamic programming to choose a low-cost path with edge, distance,
   orientation, and smoothness terms.
5. Postprocess the selected path according to camera type and annotation mode.

The public API is:

```python
from interactive_snapping_2d.snap2d import snap_annotation

result = snap_annotation(image, stroke, camera_type="pinhole", mode="curve")
points_xy = result.points
```

## Modes

- `curve`: keeps the snapped 2D curve shape, then lightly smooths and resamples
  it.
- `line` with `pinhole`: fits a 2D image-space straight line to the snapped
  path and samples only within the user stroke extent.
- `line` with `panorama`: keeps a 2D polyline. It does not force an
  equirectangular panorama line annotation to become a straight image-space
  line.

## Camera Types

- `pinhole`: snaps directly in input image coordinates.
- `panorama`: horizontally unwraps strokes that cross the equirectangular
  seam, snaps in the unwrapped 2D image, then wraps output `x` coordinates back
  to `[0, W)`.

## Line-Aid GUI

```bash
./.venv/bin/python -m interactive_snapping_2d.annotate_line_aid \
    --image interactive_snapping_2d/data/test_1.jpg
```

Mouse and keyboard controls:

- `Draw` button: enter rough-stroke drawing mode.
- Left drag in `Draw` mode: draw a rough stroke; mouse release snaps it into a selected candidate line.
- `Review` button: enter candidate review mode.
- Left click a candidate in `Review` mode: toggle selected/dropped.
- `Select All`, `Drop All`, `Invert`: batch-edit candidate selection.
- `Undo`: remove the most recently created snapped candidate.
- `Reset`: reset only the current in-progress stroke.
- `S`: save a MaDCoW-compatible annotation JSON, a 2D debug JSON, and a preview PNG.
- `P`: use `pinhole`.
- `E`: use `panorama`.
- `L`: use `line`.
- `C`: use `curve`.
- `D`: toggle debug edge overlay.
- `ESC`: reset the current stroke.

The default saved annotation is compatible with `MaDCoW/main.py`: it contains
`image_path`, `camera_model`, optional `fov_deg`, `lines[].points_dir` with
128 view-sphere samples, and an empty `regions` list. The snapping debug JSON
keeps the 2D image-space points.

## CLI Demo

```bash
./.venv/bin/python -m interactive_snapping_2d.demo_cli \
    --image interactive_snapping_2d/data/test_1.jpg \
    --stroke-json interactive_snapping_2d/examples/sample_strokes.json \
    --camera-type panorama \
    --mode line \
    --output-json interactive_snapping_2d/outputs/demo_annotation.json \
    --output-madcow-json interactive_snapping_2d/outputs/demo_madcow_annotation.json \
    --output-preview interactive_snapping_2d/outputs/demo_preview.png
```

## JSON Output

The exported schema keeps the core output in 2D input-image pixels:

```json
{
  "version": "0.1.0",
  "tool": "interactive_snapping_2d",
  "image_path": "interactive_snapping_2d/data/test_1.jpg",
  "coordinate_space": "input_image_pixel",
  "camera_type": "panorama",
  "mode": "line",
  "points": [[123.4, 512.8], [124.9, 511.6]],
  "source_stroke": [[120.0, 515.0], [130.0, 508.0]],
  "closed": false,
  "confidence": 0.86,
  "debug_summary": {
    "mean_edge_score": 0.72,
    "mean_orientation_score": 0.81,
    "mean_abs_offset_px": 5.2
  }
}
```

For MaDCoW, use the optional MaDCoW export. It converts the 2D snapped line to
the current `main.py` schema:

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

## Tests

```bash
./.venv/bin/python -m pytest interactive_snapping_2d/tests
```

## Known Limitations

- Low-contrast boundaries may snap inaccurately.
- Strong texture near the stroke can attract the path to the wrong edge.
- Endpoints are not automatically extended far beyond the user stroke.
- Panorama handling only performs 2D horizontal seam unwrap; it does not do
  spherical geometry optimization.
