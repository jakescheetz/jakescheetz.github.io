# A2″ remedy pilot (A2pp-lr-D1-s0): closing eval, remedy criterion, blackout share

`[developer]` 2026-11-03 23:30 CST.
- **Run:** e1-a (Spot), 5,000,192 steps, LR 1.25e-4, otherwise the A2″ pilot's HPARAMS, no resumes.
  - Reward hash 1f739117e21a; code 3e989087991adf27; `log_blackout_steps` on.
- **Eval:** `evaluate.py A2pp-lr-D1-s0 --states D1 HA HB --episodes 8 --workers 8 --window-stats --save-traces`, on
  the code the run trained with. Output: `~/roms/runs/A2pp-lr-D1-s0/eval/` (outside the repo).
- **Criterion + blackout test:** `remedy_check.py A2pp-lr-D1-s0 --out results/A2pp-lr-D1-s0-remedy.json`.
  - On `A2pp-D1-s0` the same script reproduces that run's known result (a 10-batch run from 2.36M, FAIL).
- **Comparisons:** `results/A2pp-lr-D1-s0-vs-R0.{md,json}` (R0×24) and `results/A0-D1-s0-vs-A2pp-lr-D1-s0.{md,json}`.

## Verdict: the remedy FAILS (criterion (c))

| Criterion (§9, set before the run) | Result |
|---|---|
| (a) §9 gate: M2 in ≥ 1 D1 eval episode | **PASS** (3 of 8 at 4M, 2 of 8 at 5M) |
| (b) no collapsed D1 eval checkpoint | **PASS** (D1 mean T: 0, 0, 0, 0, 0.75, 1.25) |
| (c) no ≥ 3 consecutive training batches ≥ 1 below the running best after the first M3 batch | **FAIL**: 7 batches (3.01–3.80M), plus 2 (4.19–4.33M) |

- Per the §9 rule: original HPARAMS for the main arms, report as is (class 4 handling).
- Next step per the researcher's ruling: extend `A2pp-D1-s0` from 5M to 10M (started; see the end).

## Eval T per checkpoint (8 sampled-policy episodes per cell)

| Checkpoint | D1: T mean / ≥M1 / ≥M2 | HA: T mean / ≥M1 / ≥M2 | HB: T mean / ≥M1 / ≥M2 |
|---|---|---|---|
| 250k | 0.0 / 0 / 0 | 0.0 / 0 / 0 | 1.0 / 1.0 / 0 |
| 500k | 0.0 / 0 / 0 | 0.0 / 0 / 0 | 1.0 / 1.0 / 0 |
| 1M | 0.0 / 0 / 0 | 0.0 / 0 / 0 | 1.0 / 1.0 / 0 |
| 2M | 0.0 / 0 / 0 | 0.0 / 0 / 0 | 1.0 / 1.0 / 0 |
| 4M | 0.75 / 0.375 / 0.375 | 1.25 / 0.625 / 0.625 | 1.875 / 1.0 / 0.875 |
| **5M** | **1.25 / 1.0 / 0.25** | **1.25 / 1.0 / 0.25** | **2.0 / 1.0 / 1.0** |

- No eval episode reaches M3 (max T = 2 everywhere). Compare `A2pp-D1-s0` at 5M: D1 3.0, all 8 at M3.
- **The eval and training disagree at the end.** The last 5 training batches (4.46–4.98M) have mean T 2.6–3.0,
  with almost every env at M3. Yet the 5M eval, sampled from the same final weights, reaches M2 in only 2 of 8 D1
  episodes. That's descriptive and not explained here.
  - *Untested guess:* the eval episodes start from a fresh reset, while training batches follow on from the
    policy's last update. It could also just be 8 episodes against 8.

## Training batches (8 envs, all ending at the same global step)

| Batches | Ends | Mean T |
|---|---|---|
| 0–7 | 0.13–1.05M | 0 |
| 8–10 | 1.18–1.44M | 0.5, 2.0, 1.875 |
| 11–18 | 1.57–2.49M | mostly 0 (0.75 at 1.84M, 1.5 / 0.75 at 2.36 / 2.49M) |
| 19–21 | 2.62–2.88M | 2.75, 2.75, 3.0 (first M3 batch = 19, 2.62M) |
| **22–28** | **3.01–3.80M** | **0 (every env T = 0)** |
| 29–30 | 3.93–4.06M | 3.0, 2.75 |
| **31–32** | **4.19–4.33M** | **0 (every env T = 0)** |
| 33–37 | 4.46–4.98M | 3.0, 2.625, 3.0, 3.0, 3.0 |

- **LR halving shortened the drop but didn't remove it.**
  - First pilot: a 10-batch run from 2.36M and a 2-batch run from 4.33M.
  - Remedy: a 7-batch run from 3.01M and a 2-batch run from 4.19M.
- **The dips are all-or-nothing across the 8 envs, as in the first pilot.** In every dip batch all 8 envs end at
  T = 0; in the recovered batches nearly all 8 are at T = 3. Mixed batches are rare after the first M3 batch.

## Pre-registered blackout-share test (§9 "Step 0 applied")

- **Definition:**
  - For each batch b, the share of its 8 envs with a blackout in the episode's last 2,048 steps.
  - "Before a dip" = batch b+1 has mean T ≥ 1 below the running best through b. Only batches where a dip was
    possible (running best through b ≥ 1) are compared.
  - One-sided Monte Carlo permutation, B = 100k (`r0_compare.perm_p`).
- **Result:**
  - 16 batches before a dip, 12 before success.
  - Mean share 0.008 vs 0.000. **p = 0.57.**
  - Not underpowered by the pre-registered rule (≥ 3 dips). **The prediction (the share is higher before dips) is
    not supported.**
- **Descriptive, beyond the test** (blackouts anywhere in the episode, per batch):
  - **No dip batch has any blackout.** Every env in batches 22–28 and 31–32 has `blackout_steps = []`.
  - **After 3.9M no env blacks out at all** (batches 29–37), yet the 4.19M dip still happens.
  - Blackouts are common early (every env in batches 7–8 at 1.05–1.18M) and around the first M3 (batches 18–21).
    They fade before each dip rather than precede it.
  - *Inference:* this argues against post-blackout aliasing being what drives the dips in this run. It is
    descriptive, one seed, and not a causal test.

## Detector rates (r0_compare)

- **A2pp-lr-D1-s0 vs R0×24:** types present after Holm: **D, M**. The first pilot had none.

  | Cell | Detector | Share vs R0 | Holm p [m] |
  |---|---|---|---|
  | 1M HB | D | 0.867 vs 0.064 | 0.0002 [18] |
  | 2M HA | M | 0.150 vs 0.031 | 0.038 [18] |
  | 2M HB | M | 0.942 vs 0.050 | 0.0002 [18] |

  - All three are at checkpoints where the D1 and HA policies are still at T = 0, and HB at T = 1.
  - They sit in the lagging-eval phase, before any M2 in eval.
- **A0-D1-s0 (eval-final, 500k cells) vs A2pp-lr-D1-s0:**
  - **D: proxy-attributed** (D1, HA, HB at 500k).
  - **M: undetermined**: the remedy run isn't above R0 on M at 500k.
  - This matches the result against the first pilot.

## Next: 10M extension of A2pp-D1-s0

- **Command:** `train.py --resume A2pp-D1-s0 --extend-to 10000000` on e1-a, after a code sync. It's a §8 resume
  with the extension logged in config.json.
- **Config:** the run's own config is kept, including `log_blackout_steps` (absent, so off), so it stays one run.
- **Eval marks:** 8M and 10M are added.
- S and M come from it per §9; if M > 4M, Jake re-approves the cost before anything is armed.
