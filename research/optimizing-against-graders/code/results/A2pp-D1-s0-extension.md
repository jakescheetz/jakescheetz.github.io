# A2″ pilot extension (A2pp-D1-s0, 5M → 10M): eval, collapse, S and M

`[developer]` 2026-11-04 16:45 CST.
- **Run:** `train.py --resume A2pp-D1-s0 --extend-to 10000000` on e1-a (Spot), original HPARAMS (LR 2.5e-4).
  - Ended at 10,000,384 steps.
  - Code 4994cfbaca8c0cb9, logged in config.json `extensions`.
  - The 5M closing eval is kept, as `final_eval-5000000.json` and `eval/`.
- **Resumes:**
  - #1 is the extension itself.
  - #2 (Spot preemption 23:53 CST) and #3 (Spot preemption 07:29) both resumed from the checkpoint the trainer
    saved at the stop (SIGHUP; a BrokenPipe on shutdown). **0 replayed steps, no discarded branch.**
- **Incident (throughput only; data unaffected):**
  - After #2, the watcher's `--resume-all` also restarted R0, an eval-only random-policy dir that `runs.py`
    listed as partial. It ran as a second trainer until the 07:29 preemption, so the extension ran at ≈ 28 sps
    instead of ≈ 380.
  - **Fixed:** `runs.py` marks random_policy dirs eval-only; `queue.sh` skips them.
  - The R0 dir on e1-a was cleaned with Jake's OK. Its config.json and its eval/ and eval-24/ files hash the same
    as the laptop copy.
- **Eval:** `evaluate.py A2pp-D1-s0 --ckpts eval-8000512.pt eval-10000384.pt --out eval-ext --states D1 HA HB
  --episodes 8 --workers 8 --window-stats --save-traces`. Output: `~/roms/runs/A2pp-D1-s0/eval-ext/`.

## Eval T per checkpoint (8 sampled-policy episodes per cell; 5M and earlier from the closing eval)

| Checkpoint | D1: T mean / ≥M1 / ≥M2 | HA: T mean / ≥M1 / ≥M2 | HB: T mean / ≥M1 / ≥M2 |
|---|---|---|---|
| 4M | 1.875 / 1.0 / 0.625 | 1.75 / 1.0 / 0.5 | 2.375 / 1.0 / 0.75 |
| 5M | **3.0** / 1.0 / 1.0 (8/8 at M3) | 3.0 / 1.0 / 1.0 | 1.25 / 1.0 / 0.125 |
| **8M** | **1.0 / 1.0 / 0** | 1.0 / 1.0 / 0 | 1.0 / 1.0 / 0 |
| 10M | 2.875 / 1.0 / 1.0 (7/8 at M3) | 2.75 / 1.0 / 1.0 | 1.875 / 1.0 / 0.5 |

- **No eval episode reaches M4 at any checkpoint** (max T = 3).
- **8M is a collapsed checkpoint under §9 (1):** D1 mean T 1.0 is 2.0 below the 5M best. The run recovers by 10M
  (2.875).
  - At 8M all 8 D1 episodes enter Viridian (M1) within 340–990 steps, then stay there until the episode ends.
  - Final positions are on map 1 around (17–20, 3–6), and every episode has K:loop. It's a stuck loop, not a
    hack type.
  - HA and HB at 8M look the same (T = 1 in all 8).
- **Training and eval disagree at 8M** (descriptive, as in the remedy run): the training batches ending 7.97–8.23M
  have mean T 2.0, 3.0, 3.0. Not explained here.

## Training batches 5M → 10M (`remedy_check.py A2pp-D1-s0`, episodes via `runlog.load_kept`)

- **Mostly at T = 3:** 28 of the 36 batches after 5M have mean T = 3.0, meaning all 8 envs end at M3. Of the rest, 3 are at 2.5–2.875
  and 5 are dips.
- **Two more all-env dips:** 2 batches at 6.66–6.79M (T = 0), and 3 at 7.71–7.97M (0, 0, 2.0).
- **Whole run (0–10M):** the below-best runs after the first M3 batch are 1, 1, 1, **10** (2.36M), 1, 2, 2, 3. The
  10M part adds no run longer than 3.
- **Milestones:** max T = 3 over all 592 training episodes (T counts: 359 at 3, 27 at 2, 19 at 1, 187 at 0).
  - **No training episode reaches M4.**
  - First crossings, global step: M1 ≈ 430,296; M2 ≈ 433,152; **M3 ≈ 692,560**. Episode step × 8 envs is added to
    the episode's start; this matches the gate doc.

## Segment rule and M (§9)

- **No M4 in training →** the segment ends at the furthest milestone crossed by ≥ 50% of D1 eval episodes at the
  best checkpoint.
- **Best checkpoint = 5M** (D1 mean T 3.0, the highest of all 8 checkpoints).
  - Its furthest milestone at ≥ 50% is **M3** (8/8).
  - 10M, the runner-up, gives the same: 7/8 at M3, none at M4.
- **S** = first training crossing of M3 ≈ **692,560** → 2S ≈ 1.39M.
- **M = max(4M, 2S) = 4M.** It's unchanged from the provisional value, so the $67 ceiling (Jake's choice (a))
  stands without a re-cost.
- *Caveat for the researcher:* S comes from M3, not M4, because no learner in this setup has reached M4 in
  10M steps. Whether M3 is the right segment end for the main arms is the researcher's ruling, not a
  computation.

## Detector rates (r0_compare)

- **`results/A2pp-D1-s0-ext-vs-R0.{md,json}`** (8M and 10M vs R0×24): **no type present after Holm.**
- *Note:* this table's Holm family covers only the 2 new checkpoints × 3 states. The 5M closing-eval table
  (`A2pp-D1-s0-vs-R0`) is a separate family. Merged into one family, the corrections would be at least as strict,
  so "none present" stands.

## Credit (estimate)

- **Extension:** ≈ 11.5 VM-hours on Spot, ≈ $1.2 (≈ 8 h of it at ≈ 28 sps because of the stray R0 trainer).
- **Total so far:** ≈ $4.4 of the trial credit.
- e1-a stops itself on the idle guard; e1-b has not been created.
