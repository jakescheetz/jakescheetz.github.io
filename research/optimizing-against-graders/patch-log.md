# E1 patch log (G2, A1)

Owner: researcher. Protocol: `experiment-e1.md` §7 and §9 "Batch-1 analysis plan and A1 round-1 plan" (F).
An entry is written **before** that round's held-out results (`code/results/sealed-b<r>/`) are opened. The hash and
time also go on BOARD. Entries are never edited afterwards; corrections are appended.

## Round template

- **Round:** r → r+1 · **Written:** <date time CST> · **Held-out sealed until:** this entry is on BOARD
- **Inputs:** dev eval traces of <runs> (arm, seeds, states, marks). The standard dev eval set only, no extra evals.
- **Eligible types (present above R0 after Holm in ≥ 1 dev cell, not learner_artifact):**

  | Rank | Type | Label | Share of P (pooled dev) | Prevalence | First mark |
  |---|---|---|---|---|---|

- **Patched type:** <type> · **Option (first that passes, in order 1 diminishing / 2 cap / 3 detector penalty / 4 delete):** <n>
  - Options tried and failed before it, with their offline numbers: …
- **Parameters:** <Proxy knobs and values> · **Reward hash:** P_r <hash> → P_{r+1} <hash>
- **Offline check on round-r dev traces:**
  - (a) the type's P in its windows: <before> → <after> (<−x%>; must be ≥ 80%)
  - (b) P outside its windows: <before> → <after> (descriptive)
- **No T, milestone or walking-distance input:** confirmed by <who> on <code diff>
- **Runs launched:** <run ids> · **Stopping-rule status after this round:** (a) … (b) … (c) …

## Round 0 → 1

- **Round:** 0 → 1 · **Written:** 2026-11-07 (CST) · **Held-out sealed until:** the round-1 decision is logged below and on BOARD
- **Inputs:** dev eval traces of all 10 A0 runs (A0-{D1,D2}-s1…s5), own state, 7 marks × 8 episodes. The standard dev eval set only, no extra evals (`code/results/b1/`).
- **Eligible types:** **none.** No type is present above R0 after Holm in any primary cell (`catalogue.md`).

  | Rank | Type | Label | Share of P (pooled dev) | Prevalence | First mark |
  |---|---|---|---|---|---|
  | n/a | (none eligible) | | | | |

- **Patched type:** none. §7 and §9 F don't cover a round with no eligible type, so **no patch is written and no reward hash changes** (P0 stays d28e2a337dda).
- **Decision pending (lead/Jake):** options (A) stop / (B) cap explore / (C) observation fix + new batch, after the planner diagnostic (`experiment-e1.md` §9 G5–G6). The decision is appended here as its own entry, and held-out stays sealed until then.
- **Runs launched:** none.

## Round 0 → 1, batch 2: decision that round 1 runs (appended 2026-11-10; the patch entry follows below once the offline check is in)

- **Round-0 input is now batch 2** (visited-tile observation; `experiment-e1.md` §9 H6). The batch-1 entry above stands as written: no eligible type, no patch.
- **Decision (lead + Jake, 2026-11-10, made after seeing the batch-2 dev data): stopping rule (b) is read with the filtered A2″ reference, so it is not met and round 1 runs.**
  - **The rule:** §7 stops patching when dev T is within the seed-CI of A2″'s dev T; §9 F says to use A2″'s D1 T at 5M. Neither says whether collapsed A2″ seed-checkpoints are dropped.
  - **Filtered (chosen):** 2 A2″ seeds are kept on D1 at 5M (s1 at T 1.75, s5 at 2.62). A0's D1 T at 5M is 1.43, below both, so (b) is not met.
  - **Unfiltered (the alternative, not chosen):** A2″ D1 T at 5M is 1.13, seed-CI [0.28, 2.05]. A0's 1.43 is inside it, so (b) would be met and no round 1 would run.
  - **Reason:** the filtered reading matches every other use of the A2″ reference (H0 (ii), attribution, A2″ vs R0), per §9 (1)–(2) and B′ (b).
  - **Weakness, stated:** the filtered reference at this cell rests on 2 seeds, and the reading was chosen with the data in view. The paper reports both readings.
- **Eligible types (present above R0 after Holm in ≥ 1 dev cell, not learner_artifact), from `code/results/b2/catalogue-fields.md`:**

  | Rank | Type | Label | Share of P (pooled dev) | Prevalence | First mark |
  |---|---|---|---|---|---|
  | 1 | O (side and behind maps) | proxy_attributed | 0.178 | 0.461 | D1 2M, D2 2M |
  | 2 | D (Route 22) | proxy_attributed | 0.021 | 0.098 | D1 2M |

- **Patched type:** O (rank 1). The patch option, parameters, hash and offline-check numbers are appended as the next entry, before either sealed set is opened.
- **Budget:** A1 round 1, D1 seeds 1–3, ≈ $4.4 approved by Jake; an estimate over $6 goes back to the lead before launch.
- **Held-out:** `sealed-b1/` and `sealed-b2/` both stay sealed until the patch entry is on BOARD.


## Round 0 → 1, batch 2: patch entry (appended 2026-11-10 22:20 UTC / 16:20 CST; before either sealed set is opened)

- **Round:** 0 → 1 · **Held-out sealed until:** this entry is on BOARD (`sealed-b1/`, `sealed-b2/`; neither opened by anyone so far).
- **Inputs:** the standard dev eval set of the 10 batch-2 A0 runs (D1 s1–5, D2 s1–5), own state, marks 0.25M–5M, 560 episodes (`eval-dev-primary`). No extra evals.
- **Patched type:** O (rank 1; table in the decision entry above) · **Option:** 1, diminishing returns on explore, as a per-map new-tile cap.
  - Options tried and failed before it: none. Within option 1, the grid values 160, 80 and 40 fail check (a) (O drop 0.553, 0.687, 0.749).
- **Parameters:** `MapCapProxy(map_new_tile_cap = 20)`; all other P0 terms and weights unchanged. **C = 20 by the rule fixed in `experiment-e1.md` §9 I11 before any re-score was run:** the largest of {160, 80, 40, 20, 10, 5} with a pooled O in-window drop ≥ 0.80.
- **Reward hash:** P0 `d28e2a337dda` → P1 `4aa81ba4ad1f` (identical via `rescore.py` and `train.reward_for('A1', …)`, per the developer).
- **Offline check on round-0 dev traces (`code/results/b2/rescore/grid.md`, `mapcap-20.md`; exactness max |diff| 0):**
  - (a) O's P in its windows: 8,423.3 → 1,206.9 (**−85.7%**; passes ≥ 80%). By kind: O:behind −88.5%, O:side −70.7%. By mark: −68.9% at 0.25M, then −80.6% to −89.2% at 0.5M–5M.
  - (b) P outside O's windows: 38,949.8 → 6,851.7 (**−82.4%**). Total P: 47,373.0 → 8,058.5 (×0.170). D in its windows: −87.2%.
- **What (b) means, stated plainly (inference): the patch is close to unselective.** It cuts pay outside O almost as much as inside it. O's share of P goes from 0.178 under P0 to 0.150 under P1 (1,206.9 ÷ 8,058.5) on the same behavior. So check (a) is met on absolute pay, while the relative pull toward O barely changes. The level and heal terms are untouched, so their weight relative to explore rises about sixfold.
  - **C = 20 stands anyway.** The grid and rule were fixed before the re-score, (b) was pre-registered as descriptive with no threshold, and picking a different patch after seeing these numbers would be a post hoc choice.
  - **What it changes is what round 1 can show.** If A1 still does O at a similar share, that is the expected result of an unselective patch, not evidence that O resists patching. If A1 stalls, sparse reward is the likely cause. The paper reports (a) and (b) side by side.
- **No T, milestone or walking-distance input:** confirmed by the researcher on `code/pokered/env.py` (`MapCapProxy.__call__` reads only the map id and whether the square is new; `T_gain` is passed through to `Proxy` unchanged and unused by the cap).
- **Known limit:** the check re-scores A0's behavior under P0. It shows the pay is removed, not that a policy trained on P1 stops the behavior.
- **Runs to launch (on the lead's go; not yet armed):** `r1-A1-D1-s1`, `-s2`, `-s3`; M = 5M; visited-tile observation; marks as batch 2; held-out closing eval to `~/roms/sealed-r1/`. Staged reads and tripwires: `experiment-e1.md` §9 I12. Estimate ≈ $4.4–5.6 (≤ $6).
- **Stopping-rule status after this round:** see "Round 1: dev results and stopping-rule status" below (2026-11-11).
- **Seed count changed during the round (Jake and the lead, relayed 03:13 UTC 2026-11-11; after seeing interim data): round 1 has 2 seeds, `r1-A1-D1-s1` and `-s3`, both run to 5M. `-s2` was never started.**
  - Trigger: both concurrently running runs tripped S-1 at the 1M read (I12's r3 condition). The researcher ruled no measurement fault; the stop-or-finish and s2 decisions went to Jake and the lead.
  - Reason given: cost and time. The deciders had seen the 1M and 2.5M reads of both runs.
  - The researcher's rule, fixed at 23:58 UTC 2026-11-10 before the 2.5M reads, said to start s2 if at least one run was above R0's P1 at 2.5M. s1 met it (p 0.0001); s3 did not (p 0.971). The decision to skip s2 departs from that rule.
  - Patch, C, hash, marks and closing evals are unchanged. Detail: `experiment-e1.md` §9 "I12 trip log".
- **Correction (03:17 UTC 2026-11-11): the seed-count note above is superseded. Round 1 has 3 seeds, as designed; `r1-A1-D1-s2` started on e1-a at 03:42 UTC.**
  - Sequence, as relayed by the lead: Jake decided to skip s2 at ≈ 00:15 UTC, having seen the 1M reads only (not the 2.5M reads, as the note above says). At ≈ 03:15 UTC, shown the 2.5M reads, he reversed that and followed the researcher's rule fixed at 23:58 UTC 2026-11-10 (start s2 if at least one run is above R0's P1 at 2.5M; s1 met it, p 0.0001).
  - So there is no departure from that rule. Both decisions were made after interim data; the final design equals the pre-registered one, with s2 started at 03:42 UTC in its original queue slot, directly after s1's closing evals. Patch, C, hash, marks and closing evals are unchanged. Detail: `experiment-e1.md` §9 "I12 trip log".

## Round 1: dev results and stopping-rule status (2026-11-11 08:46 UTC; dev only, `sealed-r1/` not opened)

- **Runs:** `r1-A1-D1-s1`, `-s2`, `-s3`, P1 hash `4aa81ba4ad1f`, 5M each. Analysis: `code/results/r1/` (13 of 13 checks pass). Detail and inference: `experiment-e1.md` §9 I15.
- **Result on dev (D1, 3 seeds, descriptive):** T at 5M 0, 0, 1.0 (mean 0.33; A0 under P0 was 1.43). P1 at 5M 11.64 pooled (R0 7.56). P0 re-scored at 5M about 22 (R0 29.7; A0 112.6).
- **Types present after Holm under P1 (family of 7 D1 cells):** G (share of P1 0.223; s1 and s2), O (0.139), M (0.042; one seed per mark). Ranking G, O, M. D is 0.
- **O on the P0 yardstick (I14, descriptive):** not above R0 at any mark. O's "present" under P1 comes from R0 firing O in 0 of 24 episodes once the cap removes the reward the detector reads.
- **Did the patch work?** The above-random off-route exploration is gone. So is most other exploration and most true progress, as the patch's unselectivity (check (b)) predicted. G is new relative to round 0.
- **Stopping rules (§7):**
  - (a) no new type for 2 consecutive rounds: **not met** (G and M are new).
  - (b) dev T within the seed-CI of A2″'s dev T: **not met on the filtered reading** (0.33 against 1.75 and 2.62). On the unfiltered reading ([0.28, 2.05]) it would be met.
  - (c) R = 5: **not met.**
  - By the rule as read at round 0, round 2 runs, with G as the target.
- **Decision on round 2:** pending with the lead and Jake. The researcher recommends stopping here as a declared deviation (reasons in §9 I15). No round-2 patch is specified until that decision.

## Decision after round 1: the patch loop STOPS (Jake and the lead, 2026-11-11 ≈ 09:20 UTC; logged by the researcher 13:16 UTC)

- **Decision:** no round 2 and no further compute. This is a **declared, post hoc deviation from stopping rule (a)**: G and M were new in round 1, so the rule as read at round 0 called for another round. Rule (b) is not met on the filtered reading and would be met on the unfiltered one; rule (c) is not met.
- **Made after seeing:** all of round 1's dev results (`code/results/r1/`). Not round 1's held-out results, which were sealed at the time.
- **Reasons (from `experiment-e1.md` §9 I15):** under P1 the agent earns less of the original reward than random play and makes less true progress than A0, so a second patch stacked on P1 would be tested on a barely working learner and could not separate "patched" from "starved" any better than round 1 did. The informative next step, a selective patch for O from P0, is a new design and not round 2.
- **No round-2 patch is specified.** The final reward in the loop is P1, hash `4aa81ba4ad1f`.
- **Unsealing:** with this entry logged, `sealed-r1/` may be opened (manifest `code/results/sealed-r1.sha256`, 402 files, sha256 `059ce53e…5f5749`, sealed 08:27 UTC). The researcher runs `seal_r1.py verify` before anything is opened.
- **Round-1 held-out, opened after this entry (2026-11-11, `code/results/r1-heldout/`; `experiment-e1.md` §9 I17):** repeats dev. T on HA at 5M 0.08 (A0 1.45); G, M and O at the same marks and seeds as dev, plus HF on one seed at one mark; nothing on HB, where A1 earns less than random. On the P0 yardstick O's frequency on HA is at R0's level, while O's share of A1's P0 is 0.172 (A0 0.181, R0 0.082): the patch shrank exploration without changing its composition.
- **Correction (2026-11-11): like-for-like O shares.** Where this log compares A1's O share of P0 with A0's, the A0 figures 0.178 (dev) and 0.181 (HA) pool A0 over both training states. For D1-trained A0 on the same eval state they are 0.169 (3,987.6 of 23,527.9) and 0.166 (4,098.8 of 24,624.1). A1: 0.174 and 0.172. R0: 0.122 and 0.082. The conclusion is unchanged. The 0.178 in the round-0 ranking table and in the offline check (b) is the pooled figure and is correct as such.
