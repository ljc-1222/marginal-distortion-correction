# Figure sources

The figures are composed from the image assets accompanying the Computational Photography final report, *An Interactive Annotation System for MaDCoW-Based Marginal Distortion Correction*, by [@ljc-1222](https://github.com/ljc-1222) and [@tuna00662244](https://github.com/tuna00662244).

Source panels are placed on a neutral canvas with descriptive labels, preserving their aspect ratios and full extents. The local comparisons use the crops already included in the report assets. The numbered comparison uses six independent SVG files, each embedding its original PNG. Red overlays appear only in the separate line-input and ROI-input files. The other comparisons use JPEG encoding.

- **`correction-examples.jpg`** uses `360_view.png`, `360_result.png`, `pinhole_view.jpg`, and `pinhole_result.png` (report Fig. 4).
- **`local-constraints.jpg`** uses `360_wo_annotate.png`, `360_annotate.png`, `pinhole_wo_annotate.png`, and `pinhole_annotate.png` (report Fig. 6).
- **`numbered-input.svg` / `numbered-unconstrained.svg`** embed `number_original_view.png` and `number_stero.png` (report Fig. 5).
- **`numbered-line-input.svg` / `numbered-line-result.svg`** embed `number_original_view.png` with a red vertical-divider overlay and the unchanged `number_line.png` result.
- **`numbered-roi-input.svg` / `numbered-roi-result.svg`** embed `number_original_view.png` with a red outline around the right-hand green block and the unchanged `number_ROI.png` result. All numbered source panels retain their native 500-pixel width. The source spelling `stero` identifies the original unannotated-result file.
- **`ulsd-candidates.jpg`** uses `ULSD.png` and `ULSD_result.png` (report Fig. 2). The panels illustrate the project's ULSD exploration; the detector is attributed in [results and analysis](../results.md#line-annotation-design).
- **`failure-cases.jpg`** uses `fail_case_2.png` and `fail_case_3.png` (report Fig. 7).

The report PDF, LaTeX source, model checkpoints, and working annotation outputs are not included in this documentation asset set.
