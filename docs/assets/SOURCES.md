# Figure sources

The figures are composed from the image assets accompanying the Computational Photography final report, *An Interactive Annotation System for MaDCoW-Based Marginal Distortion Correction*, by [@ljc-1222](https://github.com/ljc-1222) and [@tuna00662244](https://github.com/tuna00662244).

Source panels are placed on a neutral canvas with descriptive labels, preserving their aspect ratios and full extents. The local comparisons use the crops already included in the report assets. The numbered comparison embeds the original PNG panels in an SVG and adds explanatory red callouts to the input panel; the other comparisons use JPEG encoding.

- **`correction-examples.jpg`** uses `360_view.png`, `360_result.png`, `pinhole_view.jpg`, and `pinhole_result.png` (report Fig. 4).
- **`local-constraints.jpg`** uses `360_wo_annotate.png`, `360_annotate.png`, `pinhole_wo_annotate.png`, and `pinhole_annotate.png` (report Fig. 6).
- **`numbered-constraints.svg`** embeds `number_original_view.png`, `number_stero.png`, `number_ROI.png`, and `number_line.png` at their native 500-pixel panel width (report Fig. 5). Red vector overlays on the input identify the right-hand green block and its vertical divider. The source spelling `stero` is retained here to identify the original file.
- **`ulsd-candidates.jpg`** uses `ULSD.png` and `ULSD_result.png` (report Fig. 2). The panels illustrate the project's ULSD exploration; the detector is attributed in [results and analysis](../results.md#why-use-interactive-snapping).
- **`failure-cases.jpg`** uses `fail_case_2.png` and `fail_case_3.png` (report Fig. 7).

The report PDF, LaTeX source, model checkpoints, and working annotation outputs are not included in this documentation asset set.
