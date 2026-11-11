# O re-score grid (§9 I11, patch option 1: MapCapProxy map_new_tile_cap = C), dev only

10 batch-2 A0 runs, `eval-dev-primary` (own state, marks 0.25M–5M), 560 episodes per C; `rescore.py --own-state`. O = O:behind ∪ O:side windows per episode (pooled; a step counts once). Rule (I11): **C = the largest grid value with pooled O in-window drop ≥ 0.80**. Outside O = all P on steps outside every O window, all 560 episodes. Exactness = P0 re-score vs the saved per-step p.

| C | reward hash | O inside old → new | O drop | passes | P outside O old → new | total P new ÷ old | D drop | O:behind / O:side drop | exactness max \|diff\| |
|---|---|---|---|---|---|---|---|---|---|
| 160 | `a5dd204d81f0` | 8423.3 → 3767.1 | 0.553 | no | 38949.8 → 28658.0 | 0.684 | 0.495 | 0.631 / 0.139 | 0e+00 |
| 80 | `7bde65e5f426` | 8423.3 → 2637.4 | 0.687 | no | 38949.8 → 18805.8 | 0.453 | 0.614 | 0.760 / 0.296 | 0e+00 |
| 40 | `8545939a2407` | 8423.3 → 2115.5 | 0.749 | no | 38949.8 → 12221.3 | 0.303 | 0.755 | 0.805 / 0.452 | 0e+00 |
| 20 | `4aa81ba4ad1f` | 8423.3 → 1206.9 | 0.857 | yes | 38949.8 → 6851.7 | 0.170 | 0.872 | 0.885 / 0.707 | 0e+00 |
| 10 | `958c1be3a595` | 8423.3 → 637.4 | 0.924 | yes | 38949.8 → 3560.6 | 0.089 | 0.939 | 0.938 / 0.851 | 0e+00 |
| 5 | `99666472e53e` | 8423.3 → 326.3 | 0.961 | yes | 38949.8 → 1817.3 | 0.045 | 0.970 | 0.968 / 0.925 | 0e+00 |

**Chosen: C = 20** (largest passing value). Per-mark rows in `mapcap-<C>.md`.
O's pooled share of P0 here: 8,423.3 ÷ 47,373.0 = 0.178 (matches the catalogue).
