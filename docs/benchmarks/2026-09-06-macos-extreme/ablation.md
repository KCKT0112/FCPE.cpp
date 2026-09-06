# Three alternating rounds

Each cell is the median of all 60 samples, in ms. Round medians and every sample remain in ablation.json. All 132 runs passed the original numerical gates.

| Variant | 0.1s | 1s | 3s | 11s |
| --- | ---: | ---: | ---: | ---: |
| current | 1.105 | 2.169 | 3.962 | 12.442 |
| legacy_copy | 1.106 | 2.201 | 4.100 | 13.148 |
| legacy_graph | 1.279 | 2.457 | 5.285 | 18.746 |
| legacy_dw_layout | 1.375 | 2.165 | 4.018 | 13.085 |
| legacy_dw_kernel | 1.300 | 2.519 | 5.097 | 17.238 |
| legacy_gn | 1.148 | 2.199 | 4.220 | 14.952 |
| legacy_im2col | 1.231 | 2.352 | 4.359 | 14.213 |
| no_glu_project | 1.151 | 2.067 | 4.123 | 13.379 |
| no_fusion | 1.168 | 2.065 | 4.247 | 14.364 |
| single_submit | 1.156 | 2.205 | 4.424 | 12.567 |
| baseline | 1.475 | 3.713 | 9.620 | 32.344 |
