# G5 part 2: 0-step mark (EXPLORATORY)

Each A0 / A2″ dev run's rebuilt init policy, own training state × 8 episodes (seeds 1000–1007), next to the same runs' 0.25M dev cells. Dev states only. Exploratory: does not change H0 (i).

Init rebuilt on AMD EPYC 7B13 (torch 2.15.0+cpu), hash-checked against each run's recorded `init_sha256`: 20/20 match. Excluded (no match): none.

| arm | state | runs | eps | T @0 | T @0.25M | P @0 | P @0.25M | legP @0 | legP @0.25M | P @0 explore / level / heal |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | D1 | 5 | 40 | 0.0 | 0.625 | 32.562 | 47.371 | 0.39 | 1.188 | 31.707 / 0.075 / 0.779 |
| A0 | D2 | 5 | 40 | 0.0 | 0.175 | 30.321 | 25.991 | 0.363 | 0.567 | 29.982 / 0.15 / 0.189 |
| A2pp | D1 | 5 | 40 | 0.0 | 0.0 | 32.562 | 5.992 | 0.39 | 0.142 | 31.707 / 0.075 / 0.779 |
| A2pp | D2 | 5 | 40 | 0.0 | 0.0 | 30.321 | 7.296 | 0.363 | 0.148 | 29.982 / 0.15 / 0.189 |

Pooled means over the kept runs' episodes (8 per run); 0.25M = the same runs' `eval-dev/` rows at 250,880 on their own state. Per-episode rows: `zero-step-episodes.jsonl`; hash check: `zero-step.json`.
