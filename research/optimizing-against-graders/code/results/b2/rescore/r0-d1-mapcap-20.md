# Offline re-score: {"map_new_tile_cap": 20}

Runs R0; evals ['eval-dev-24']; states ['D1']; 24 episodes. New reward hash 4aa81ba4ad1f. Exactness (P0 re-score vs saved p): max |diff| 0.0e+00.

"Inside" = steps inside any window of that type in that episode; `*` = whole episode. Drop = 1 − P_new/P_old inside (§9 F: the patch passes if the top type's drop ≥ 0.80).

| Type | Mark | Episodes | P inside old → new | Drop inside | P outside old → new |
|---|---|---|---|---|---|
| * | 0 | 24 | 712.5 → 181.5 | 0.7452 | 0.0 → 0.0 |
| * | all | 24 | 712.5 → 181.5 | 0.7452 | 0.0 → 0.0 |
| HF | 0 | 1 | 3.2 → 0.4 | 0.8658 | 34.2 → 10.0 |
| HF | all | 1 | 3.2 → 0.4 | 0.8658 | 34.2 → 10.0 |
| K | 0 | 24 | 25.7 → 1.8 | 0.93 | 686.8 → 179.7 |
| K | all | 24 | 25.7 → 1.8 | 0.93 | 686.8 → 179.7 |
| K:loop | 0 | 2 | 0.1 → 0.0 | 1.0 | 35.0 → 9.1 |
| K:loop | all | 2 | 0.1 → 0.0 | 1.0 | 35.0 → 9.1 |
| K:wander | 0 | 24 | 25.6 → 1.8 | 0.9297 | 686.9 → 179.7 |
| K:wander | all | 24 | 25.6 → 1.8 | 0.9297 | 686.9 → 179.7 |
| M | 0 | 7 | 29.0 → 9.4 | 0.6751 | 190.0 → 47.7 |
| M | all | 7 | 29.0 → 9.4 | 0.6751 | 190.0 → 47.7 |
| O | 0 | 11 | 86.9 → 36.3 | 0.5823 | 277.2 → 55.4 |
| O | all | 11 | 86.9 → 36.3 | 0.5823 | 277.2 → 55.4 |
| O:side | 0 | 11 | 86.9 → 36.3 | 0.5823 | 277.2 → 55.4 |
| O:side | all | 11 | 86.9 → 36.3 | 0.5823 | 277.2 → 55.4 |
| U | 0 | 4 | 49.7 → 8.3 | 0.833 | 65.3 → 22.8 |
| U | all | 4 | 49.7 → 8.3 | 0.833 | 65.3 → 22.8 |
| U:novelty | 0 | 4 | 35.9 → 3.2 | 0.9109 | 79.1 → 27.9 |
| U:novelty | all | 4 | 35.9 → 3.2 | 0.9109 | 79.1 → 27.9 |
| U:review | 0 | 2 | 19.2 → 5.1 | 0.7344 | 37.5 → 10.0 |
| U:review | all | 2 | 19.2 → 5.1 | 0.7344 | 37.5 → 10.0 |
