# A2″ pilot (A2pp-D1-s0): step-0 data for the peak-then-drop rules

`[developer]` 2026-11-03 18:25 CST. Asked for in experiment-e1.md §9 "A2″ peak-then-drop", step 0. These are training
logs from e1-a (`~/roms/runs/A2pp-D1-s0/metrics.jsonl`, `episodes.jsonl`) at 4.59M of 5M steps, read before the
final eval. They are numbers only: no frames and no ROM data. The classification is the researcher's call; nothing
here applies it.

## 1. Infrastructure

- **No resume or preemption in the whole run:** 0 resume rows in `metrics.jsonl`, and the watcher saw the trainer
  ALIVE at every 10-min check from 15:01 on. Check 1 doesn't apply.

## 2. Return vs T, per training batch (8 envs; their 8 episodes end at the same global step)

| Batch end step | mean T | mean R | mean R_shaping |
|---|---|---|---|
| 0.13–0.39M | 0.0 | 0.18–0.21 | 0.18–0.21 |
| 0.52M | 1.5 | 3.70 | 2.20 |
| 0.92M, 1.05M | 3.0 | 6.09, 6.01 | 3.09, 3.01 |
| 1.18–1.97M | 2.38, 1.0, 2.75, **0.0**, 2.75, 2.62, **0.0** | 4.84, 2.09, 5.72, 0.55, 5.73, 5.48, 0.29 | |
| 2.10M, 2.23M | 2.75, 3.0 | 5.58, 6.03 | 2.83, 3.03 |
| **2.36–3.54M (10 batches)** | **0.0** | 0.18–0.34 | same as R |
| 3.67M, 3.80M | 2.75, 3.0 | 5.58, 6.07 | 2.83, 3.07 |
| 3.93M | 0.5 | 1.28 | 0.78 |
| 4.06M, 4.19M | 2.88, 3.0 | 5.98, 5.96 | |
| 4.33M, 4.46M | 0.0 | 0.25, 0.28 | |
| 4.59M | 2.75 | 5.49 | 2.74 |

- **Return falls with T:** T = 0 batches earn 0.18–0.34, under the ≈ 1 leg-1 bound, and M3 batches earn ≈ 6
  (3 × ΔT + ≈ 3 progress). So no parking spot pays, and check 2 is negative.
- **The shape is oscillation, not one collapse.** Batches are bimodal: all 8 episodes reach about M3, or all 8
  stay at T = 0. There are single-batch dips at 1.57M and 1.97M, a 10-batch drop at 2.36–3.54M, and recovery at
  3.67M. Other dips follow at 3.93M and 4.33–4.46M.
  - *Inference, not checked:* the 8 envs reset together, and the policy is updated about 128 times within one
    episode window. So the batch outcome is set by the policy during the early (Pallet) part of that window,
    which would explain why the episodes in a batch agree.
- **Transition (the §9 definition):** the first of two consecutive batches whose mean T is ≥ 1 below the running
  best batch mean (3.0) is the batch ending at **2,359,296**.

## 3. Optimizer statistics (§9 check 3)

The reference values are medians over 1–2M: `approx_kl` 0.00309, `v_loss` 3.90e-5, entropy 1.486. The entropy at
1M is 1.493.

| 200k before the transition (2.16–2.36M) | Value | Threshold | Met? |
|---|---|---|---|
| approx_kl max | 0.0072 | > 4× median = 0.0124 | no |
| v_loss max | 1.0e-3 (25.8× median; 2 of 195 updates over 10×) | > 10× median = 3.9e-4 | **yes, mechanically** |
| entropy min / max | 0.845 / 1.751 | < 50% (0.746) or > 150% (2.24) of the 1M value | no |
| entropy median before → after | 1.510 → 1.431 | | |

**The base rate of the v_loss signature** (updates over 10× the 1–2M median, per 200k window, whole run):
- 151, 60, 35, 20, 83, 26, 53, 1, 68, 0, 42, **2** (the transition window), 0, 1, 0, 1, 0, 13, 59, 21, 91, 1, 60.
- It fires in nearly every window that holds milestone payouts. It goes quiet only in the windows where T = 0
  throughout, where the return is tiny and `v_loss` falls to about 1e-6.
- The windows before the other dips have 50, 68 and 91 such updates, all more than the transition window's 2.
- KL over 4× its median occurs once in the whole run (4.0–4.2M). The entropy per 250k stays between 1.37 and 1.80
  and has no trend at the transition.
- *Reading (mine; for the researcher):* `v_loss` > 10× its median follows reward events, not instability: the
  median is small because most updates see no milestone. The signature is met in the letter but doesn't single out
  the drop. Whether that makes this class 3 (value) or class 4 is for the researcher to rule; the data above is
  everything I have on it.

## 4. Still to come

Per-checkpoint D1/HA/HB eval T, from the closing eval (eval checkpoints 250k, 500k, 1M, 2M, 4M and final).

## 5. Sync-reset log test (researcher's "free test", §9 "Step 0 applied")

Data: `results/A2pp-D1-s0-syncreset.json`, one row per training batch (the episode window of the 8 envs, 128
updates). Each update's `maps` field is the share of the 8 envs on each map at that update.
- **"Early"** = Pallet (0x00), the two houses and the rival's house (0x25–0x27), Oak's Lab (0x28).
- **"Late"** = Route 2 (0x0D), the Route 2 gate (0x32), Viridian Forest (0x33).
- **first16** = the window's first 16 updates, which is the first 2,048 steps per env.
- **last16** = its last 16 updates.

Counted after the first M3 batch (0.92M).

| Prediction | Result |
|---|---|
| T = 0 batches fail in Pallet/lab within the first few updates | **Partly.** 12 of 14 T = 0 batches are 100% early-map for the first 2,048 steps, and 14 of 14 never have a single env on a late map in any update. But 4 of 14 M3 batches also start 100% early-map and still reach M3, and 2 T = 0 batches (1.57M, 2.36M) leave the early maps in the first 16 updates and still end at T = 0. A slow start doesn't decide the batch; never leaving does. |
| The batch before each dip ends on many updates of late-phase data | **Not supported in this form.** Its last16 late share averages 0.48 (0.125, 0.25, 0.875, 0.683; n = 4) for pre-dip batches and 0.76 (0.50–1.0; n = 7) for batches followed by another M3 batch. Pre-dip batches end with *less* late data. |

- **One difference that does show:** pre-dip batches end with **0.31 of envs back on early maps** (last16), against 0.02
  for batches followed by success.
  - *Inference, not checked:* the blackout map is Pallet (§2), so this could be blackouts late in the window. The
    last updates before a dip would then train on Pallet data from a blacked-out party. Checking needs per-env
    blackout timestamps from the traces; it isn't in metrics.jsonl.
- Raw per-batch reward over first16 is also in the json: about 1e-4 per step in slow-start batches, 1–3e-3 when the
  envs leave Pallet early.

## 6. Check (ii): critic probe (`a2pp_critic_probe.py`, `results/A2pp-D1-s0-critic-probe.json`)

Three observations, taken from the greedy planner under the training obs_mode (tilegrid) and pushed through every
retained checkpoint:
- **D1 start:** Oak's Lab, T = 0.
- **Pallet start:** the first step in Pallet after the lab, (0,5,11), T = 0.
- **Post-blackout Pallet:** the respawn square (0,5,6), with the party healed.
  - Seed 0 wipes out on Route 1 at T = 0. **Seed 2 wipes out at T = 3** (the hypothesis case).

**What the observation can tell apart:**
- The RAM part (74 dims) has no T, no event flags and no step counter.
- Post-blackout (T = 3) vs Pallet start (T = 0) differ in RAM only in y, level sum/max (7 → 9) and Pokédex seen
  (4 → 7), an L2 of 0.085 on the RAM part.
- The tile and sprite grids differ because the squares differ: (5,6) vs (5,11), L2 ≈ 2,100 on raw tile ids.
- So a post-blackout T = 3 agent and a fresh T = 0 agent look nearly identical except for where they stand.

| Checkpoint | V(D1 start) | V(Pallet start, T=0) | V(post-blackout, T=3) | Training batch at that step |
|---|---|---|---|---|
| eval-250880 | 0.164 | 0.003 | −0.007 | T = 0 (pre-learning) |
| eval-500736 | 0.182 | 0.080 | 0.121 | first M1/M2 batch is 0.52M |
| eval-1000448 | 0.286 | 0.273 | 0.190 | M3 |
| eval-2000896 | 0.056 | 0.037 | 0.024 | M3 (between dips at 1.97M and 2.36M) |
| eval-4000768 | 0.206 | 0.238 | 0.190 | M3 (4.06M) |
| ckpt-4077568 | **0.632** | **0.839** | 0.288 | M3 (4.06–4.19M) |
| ckpt-4425728 | **0.020** | **−0.044** | 0.018 | **T = 0 dip** (4.33–4.46M) |
| ckpt-4776960 | 0.058 | 0.059 | 0.041 | M3 at 4.72M |

- V is small throughout. An M3 episode returns ≈ 6 over ≈ 16k steps, so with γ = 0.998 the discounted value from
  the start should be ≈ 0.1–1, depending on when the milestones land. So the scale isn't absurd.
- **The post-blackout value tracks the Pallet-start value**, within about 0.05–0.1 at most checkpoints. The
  exception is ckpt-4077568, where the start (0.84) is valued well above the post-blackout square (0.29).
- **V at the start states drops to about 0 at the checkpoint inside the dip** (4.43M) and is highest at the
  checkpoint just before it (4.08M). That's one dip with two checkpoints around it, so it's suggestive only.
- *Not tested here:* whether the critic's start-of-episode values are *pulled down* by post-blackout experience,
  which is what the hypothesis needs. That would need V over many post-blackout and start observations across
  updates, and the retained checkpoints are too sparse for it. With `blackout_steps` logged, the remedy pilot will
  give per-episode blackout timing.
- Side finding: the greedy planner's seed 0 blacks out **twice**, not once as validation (b) reported. The second
  wipe-out (ep_step 1532) is poison damage in the overworld (the forest's north gate, 0x2F, HP 1 → 0, no battle),
  followed by a respawn in Pallet at T = 3. The planner only counted wipe-outs inside battle.

## 7. `blackout_steps` field: identity smoke check (Jake's conditions)

- **Code:** `pokered/env.py` records the ep_step of each wipe-out, using the same rule as
  `detectors.blackout_steps`: the party HP sum goes from > 0 to 0, read from the RAM snapshot already taken.
  `train.py` writes `blackout_steps` into each episode row when `cfg["log_blackout_steps"]` is set (default on;
  `--no-blackout-log` turns it off). config.json now also records `code_sha256` (the VM's CODE_SHA256) and a
  `code_note`.
- **Reward hash unchanged:** 1f739117e21a, the same as A2pp-D1-s0. The hash covers ProgressOracle + Proxy source
  and params, which weren't touched.
- **Smoke:** D1, seed 0, A2pp, 2 envs, 1 thread, 512-step episodes, 12 updates (3,072 steps), laptop. Three runs:
  - (old) the VM's code (CODE_SHA256 7e2f08bab31e21c0);
  - (on) the new code;
  - (off) the new code with `--no-blackout-log`.
  - **metrics.jsonl:** every non-timing field is identical across all three (losses, KL, entropy, rewards, map
    mix, step, update). The fields that can't match between any two runs are wall-clock ones: sps, sps_update,
    t_collect, t_learn, load1, cpu_mhz, time.
  - **episodes.jsonl:** identical, old = off, and on = off once the field is removed.
  - **Final weights:** the same SHA-1 (fa50a9bfe99b) in all three.
  - **Pass**, so the remedy pilot launches with the field.
- **The field fires correctly:** on the greedy planner's seed 0 it records [444, 1532]. 444 is the Route 1 battle
  wipe-out; 1532 is the overworld poison wipe-out above.

## 8. Matched squares: T = 0 vs post-blackout T = 3 (`a2pp_critic_probe.py --matched --seeds 2`, `results/A2pp-D1-s0-critic-probe-matched.json`)

- **Squares:** greedy planner seed 2 visits 19 Pallet squares at T = 0 before any wipe-out and 12 after its
  T = 3 respawn. **7 squares are in both:** (10,0)–(10,6), the north path to Route 1.
- **Observation:** the tile and sprite grids are **identical** (L2 0.0). The RAM part differs only in level
  sum/max (0.07 → 0.09) and Pokédex seen (0.026 → 0.046).
- **V, mean over the 7 squares** (T = 0 / T = 3):

  | Checkpoint | V (T = 0 / T = 3) |
  |---|---|
  | eval-250880 | 0.147 / 0.147 |
  | eval-500736 | 0.169 / 0.169 |
  | eval-1000448 | 0.346 / 0.347 |
  | eval-2000896 | 0.020 / 0.020 |
  | eval-4000768 | 0.165 / 0.165 |
  | ckpt-4077568 | 0.579 / 0.579 |
  | ckpt-4425728 | 0.002 / 0.002 |
  | ckpt-4776960 | 0.064 / 0.064 |
  | 5000192 | 0.698 / 0.698 |

  The mean |ΔV| is ≤ 0.0001 at every checkpoint, and the per-square correlation is 1.000.
- **Reading:** on the same square, the critic gives a T = 3 post-blackout agent the same value as a fresh T = 0
  agent. It is effectively blind to T and to the blackout; the small level/seen difference moves V by less than
  1e-4. Under A2″ the two situations have different returns ahead: after the respawn, ΔT is spent and only the
  leg's remaining progress pays, while from a fresh start every milestone is still to come. So the critic must
  average them, which is what the hypothesis needs.
  - Whether that averaging *causes* the dips isn't shown here. The remedy pilot's `blackout_steps` and the
    pre-registered blackout-share test address that.
