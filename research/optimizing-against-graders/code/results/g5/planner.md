# G5 part 1: max-attainable-P planner (diagnostic, not an arm)

Pure-coverage BFS to the nearest square that hasn't paid P0's explore term; D1/D2 × seeds 1000–1007, 16,384 steps, P0 (hash d28e2a337dda), same env and jitter as eval. Dev states only.

| state | eps | stuck | planner P (min–max) | A0 best P (mark) | 2× threshold | ratio | passes | T | legP | tiles | explore / level / heal | blackouts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 8 | 0 | 417.124 (416.171–418.2263) | 55.5 (1M) | 111.0 | 7.52× | yes | 5.0 | 3.0 | 4080.125 | 408.012 / 9.0 / 0.112 | 34 |
| D2 | 8 | 0 | 417.159 (416.3391–418.165) | 50.5 (0.5M) | 101.0 | 8.26× | yes | 5.0 | 3.0 | 4080.25 | 408.025 / 9.0 / 0.134 | 21 |

**Verdict (rule fixed before the run):** both pass: (C) indicated.

Planner health: dropped targets D1 198, D2 189; mean steps with nothing reachable left D1 1646.375, D2 2644.0.

## Per episode

| state | seed | P | T | legP | tiles | top maps by P | blackouts | max no-new gap | stuck |
|---|---|---|---|---|---|---|---|---|---|
| D1 | 1000 | 417.0217 | 5 | 3.0 | 4079 | 0x33 74.3, 0x2 70.9, 0x1 60.9 | 4 | 1728 | False |
| D1 | 1001 | 417.1048 | 5 | 3.0 | 4079 | 0x33 74.3, 0x2 70.9, 0x1 61.0 | 5 | 2448 | False |
| D1 | 1002 | 417.1727 | 5 | 3.0 | 4081 | 0x33 73.3, 0x2 70.9, 0x1 61.0 | 3 | 1710 | False |
| D1 | 1003 | 416.171 | 5 | 3.0 | 4081 | 0x33 74.3, 0x2 70.9, 0x1 60.9 | 4 | 2168 | False |
| D1 | 1004 | 417.1727 | 5 | 3.0 | 4081 | 0x33 73.3, 0x2 70.9, 0x1 61.0 | 3 | 1710 | False |
| D1 | 1005 | 418.2263 | 5 | 3.0 | 4081 | 0x33 74.3, 0x2 70.9, 0x1 61.1 | 5 | 1644 | False |
| D1 | 1006 | 417.0776 | 5 | 3.0 | 4080 | 0x33 74.3, 0x2 71.0, 0x1 61.0 | 5 | 1249 | False |
| D1 | 1007 | 417.0476 | 5 | 3.0 | 4079 | 0x33 73.3, 0x2 70.9, 0x1 61.1 | 5 | 2157 | False |
| D2 | 1000 | 417.1469 | 5 | 3.0 | 4080 | 0x33 74.3, 0x2 71.0, 0x1 60.9 | 2 | 3387 | False |
| D2 | 1001 | 417.1301 | 5 | 3.0 | 4080 | 0x33 73.3, 0x2 71.0, 0x1 61.0 | 3 | 2830 | False |
| D2 | 1002 | 416.9744 | 5 | 3.0 | 4078 | 0x33 73.3, 0x2 70.9, 0x1 61.0 | 3 | 2620 | False |
| D2 | 1003 | 417.3026 | 5 | 3.0 | 4082 | 0x33 74.3, 0x2 71.0, 0x1 61.3 | 2 | 2869 | False |
| D2 | 1004 | 416.9744 | 5 | 3.0 | 4078 | 0x33 73.3, 0x2 70.9, 0x1 61.0 | 3 | 2620 | False |
| D2 | 1005 | 417.2385 | 5 | 3.0 | 4081 | 0x33 74.3, 0x2 71.0, 0x1 61.0 | 3 | 2283 | False |
| D2 | 1006 | 416.3391 | 5 | 3.0 | 4082 | 0x33 75.3, 0x2 70.9, 0x1 61.1 | 1 | 3570 | False |
| D2 | 1007 | 418.165 | 5 | 3.0 | 4081 | 0x33 74.3, 0x2 70.9, 0x1 61.2 | 4 | 1529 | False |

Full rows (P per map, battles, re-plans): `planner.jsonl`. Traces outside the repo: `~/roms/g5/planner/traces/`.
