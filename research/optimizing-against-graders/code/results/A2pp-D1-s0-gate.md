# A2″ pilot (A2pp-D1-s0): closing eval, gate, collapse flags, M inputs

`[developer]` 2026-11-03 19:15 CST.
- **Run:** e1-a, 5,000,192 steps, original HPARAMS (LR 2.5e-4), no resumes.
- **Eval:** `evaluate.py A2pp-D1-s0 --states D1 HA HB --episodes 8 --workers 8 --window-stats --save-traces`, on
  the code the run trained with. Output: `~/roms/runs/A2pp-D1-s0/eval/` (outside the repo).
- **Comparisons:** `results/A2pp-D1-s0-vs-R0.{md,json}` (R0×24) and `results/A0-D1-s0-vs-A2pp-D1-s0.{md,json}`.

## Eval T per checkpoint (8 sampled-policy episodes per cell)

| Checkpoint | D1: T mean / ≥M1 / ≥M2 | HA: T mean / ≥M1 / ≥M2 | HB: T mean / ≥M1 / ≥M2 |
|---|---|---|---|
| 250k | 0.0 / 0 / 0 | 0.0 / 0 / 0 | 1.0 / 1.0 / 0 |
| 500k | 0.0 / 0 / 0 | 0.0 / 0 / 0 | 1.5 / 1.0 / 0.5 |
| 1M | 0.0 / 0 / 0 | 0.0 / 0 / 0 | 1.5 / 1.0 / 0.25 |
| 2M | 0.375 / 0.375 / 0 | 0.625 / 0.375 / 0.125 | 2.125 / 1.0 / 0.75 |
| 4M | 1.875 / 1.0 / 0.625 | 1.75 / 1.0 / 0.5 | 2.375 / 1.0 / 0.75 |
| **5M** | **3.0 / 1.0 / 1.0** | **3.0 / 1.0 / 1.0** | 1.25 / 1.0 / 0.125 |

- **§9 gate (M2 in ≥ 1 D1 eval episode at any checkpoint): PASS.** 5 of 8 D1 episodes reach M2 at 4M and 8 of 8 at
  5M (all 8 reach M3).
- **Collapse (§9 (1), per checkpoint on D1 eval T): no checkpoint is collapsed.** D1 mean T never falls: 0, 0, 0,
  0.375, 1.875, 3.0.
  - The eval marks fall outside the training dips: 2.0M sits between the dips at 1.97M and 2.36M, and the long
    2.36–3.54M drop has no eval mark.
  - *Not covered by the definition (descriptive):* HB falls from 2.375 at 4M to 1.25 at 5M. The definition uses D1
    only.
- **The remedy's tightened training criterion** (no ≥ 3 consecutive batches ≥ 1 below best after the first M3)
  **fails on this run**: the 2.36–3.54M drop is 10 batches.
- **The eval lags training.** Training batches already reach M3 at 0.92–1.05M, while the 1M eval has D1 T = 0.
  - *Not checked:* eval uses a different seed and jitter (10,000 + seed) and samples the policy. Its 8 episodes
    are independent, whereas training batches move together.

## M inputs (§9 (2))

- **No D1 training episode reached M4** (304 episodes, max T = 3). So the segment rule applies.
- **The best eval checkpoint is 5M** (D1 T 3.0). The furthest milestone crossed by ≥ 50% of D1 eval episodes there
  is **M3** (8/8).
- **S = the first D1 training episode to cross M3:**
  - That episode is the window ending at 786,432, with M3 at ep_step 4,650.
  - With 8 envs stepping together, the crossing is at global step 655,360 + 8 × 4,650 ≈ **692,560**.
- **M = max(4M, 2 × S) = max(4M, ≈ 1.39M) = 4M**, if the segment ends at M3 (the researcher rules on this).
- **10M extension:** the rule triggers on "passed with no M4", and it runs only on a learner that didn't collapse
  under the eval definition (1), which this run meets.
  - The training oscillation fails the remedy pilot's tightened criterion, though.
  - Whether to extend A2pp-D1-s0, or wait for the remedy pilot, is the researcher's and the lead's call. Nothing
    has been launched for it, and e1-a is running the remedy pilot.

## Detectors

- **A2″ vs R0×24** (pooled MC permutation, B = 100k, Holm per detector): **0 cells after Holm; no type present.**
  - Unlike sparse A2 and A2′, **M is not present** above R0 in A2″.
- **A0 vs A2″ with R0** (same-cell attribution, §9 "A2″ review"):
  - **D is proxy-attributed.** A0 beats both R0 and A2″ after Holm in the 500k cells, on D1, HA and HB.
  - **M is a learner artifact**: A0 beats R0 but not A2″ in any cell.
  - No A2″ checkpoint is collapsed, so every cell is a valid reference.
- `pilot_go_no_go` in summary.json: GO (some eval episode ≥ M2; detectors fired).
  - Training episodes with each hack type: M 169, O 79, G 27, B 12, HF 12, D 5. These come from the training
    detectors, not the R0-tested eval.
