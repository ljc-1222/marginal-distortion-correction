# Results and design analysis

[Project overview](../README.md) · [Using the system](usage.md)

The qualitative project experiments use a numbered test pattern, a classroom panorama view, and a pinhole wide-angle portrait to examine how user constraints guide local object appearance and scene structures through a global warp.

## Constraint comparison

The [numbered-pattern examples](../README.md#correction-examples) compare three independent settings on the same input: no annotations, one line constraint, and one ROI constraint. The line follows the divider between the two right-hand green columns; the ROI covers the green block containing 1/5/9/13 and 2/6/10/14.

Without annotations, the warp mainly follows the stereographic initialization and regularization. A line constraint adds a straightness target for the sampled feature. An ROI constraint instead fits a local projection and guides the global warp toward it. The line-only and ROI-only outputs show the different roles of these two forms of user input.

## Global correction

![Original and corrected panorama-view and pinhole examples.](assets/correction-examples.jpg)

In the classroom example, correction redistributes the wide field of view across the output. In the portrait example, the local face projection changes together with the surrounding ceiling, desks, and floor. The black areas in the corrected panels are outside the rendered valid image region; the examples retain these borders to show the full output extent.

## ROI and structural constraints

![Local comparisons without annotations and with annotations: classroom structures in the first row and the portrait face in the second row.](assets/local-constraints.jpg)

**Classroom structures.** The annotated result changes the projection screen and whiteboard boundaries relative to the unconstrained output. The structural annotations give the optimizer explicit straightness targets instead of leaving those features to the projection and regularization terms alone.

**Face appearance.** The ROI constraint guides the face toward its fitted local projection. This changes the shape around the forehead, cheek, and nose while the rest of the image adapts through the global warp. The selected region and its surrounding background remain coupled, which also explains the boundary artifacts discussed below.

## Line annotation design

The project also explored ULSD as an automatic line-candidate frontend. In the tested building image, the detections include fragmented structures, responses on secondary edges, and uneven coverage of useful geometric constraints. Changing detection thresholds did not remove the need to select which structures should guide MaDCoW.

![The ULSD building input and its detected line candidates.](assets/ulsd-candidates.jpg)

[ULSD](https://www.sciencedirect.com/science/article/pii/S0924271621001623) supports distorted line segments through a Bézier representation. The integrated workflow uses interactive snapping to focus annotation on structures selected by the user: local edge, orientation, color, continuity, and distance evidence refine each rough stroke before review and export. `Line` and `Curve` modes describe the input image trace; both can represent a real-world straight feature. See the [snapping reference](../interactive_snapping_2d/README.md) for details and the [ULSD frontend](../ULSD/README.md) for the separate automatic-detection path.

## Observed limitations

![Failure examples: deformation near an ROI boundary and a collapsed warp for an overly wide panorama view.](assets/failure-cases.jpg)

**Distortion near ROI boundaries.** Preserving an ROI can shift deformation into neighboring background regions. This becomes visible when strong structures sit close to the selected object. Softer boundary weights or boundary-aware regularization are possible follow-up directions.

**Excessive panorama extent.** Very wide selected views can produce heavily stretched or collapsed output meshes. The example illustrates a practical limit of the current workflow. Restricting view extent or checking mesh validity before rendering could make this failure easier to handle.

**Phone ultra-wide example.** In a tested 0.6× phone image with an effective FOV of about 114°, linear-projection initialization produced a more satisfactory result than stereographic initialization. This case illustrates the influence of initialization on the correction result.

## Figure sources

The visual examples come from the project's final-report assets and are arranged for this repository. The [asset source list](assets/SOURCES.md) records the original filenames and presentation changes.
