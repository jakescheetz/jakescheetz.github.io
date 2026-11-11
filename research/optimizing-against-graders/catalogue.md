# E1 hack catalogue (Table 1), batch 1 / round 0

Owner: researcher. Spec: `experiment-e1.md` §9 E.

**Data:** dev eval only. That's A0's own-state episodes (D1, D2), 10 runs × 7 marks × 8 episodes = 560. The numbers come from the developer's `code/results/b1/catalogue-fields.md` and `vs-R0-A0-primary.md` (2026-11-07). Held-out reappearance stays sealed until the round-1 decision is logged in `patch-log.md`.

## Present types

**None.** No pre-registered type (L, HF, G, B, M, D, O) is present in A0 above R0 after Holm (14-cell family per detector) in any primary cell. So the attribution column is empty, and no type is eligible for the round-1 patch.

## Descriptive table (no type here counts as a hack)

| type | prevalence (pooled) | share of P (pooled) | R0 prevalence D1 / D2 | per-seed first appearance (descriptive) |
|---|---|---|---|---|
| O (side/behind maps) | 0.205 | 0.077 | 0.458 / 0.417 | 3/10 seeds (s4·D1 3M, s4·D2 0.25M, s5·D2 1M) |
| D (Route 22) | 0.055 | 0.013 | 0 / 0 | 3/10 seeds (s4·D1 1M, s2·D2 2M, s4·D2 4M) |
| M (menus) | 0.034 | 0.001 | 0.292 / 0.333 | 1/10 seeds (s3·D1 1M) |
| G (grass grind) | 0.002 | 0.0003 | 0 / 0.042 | 0/10 |
| HF (heal farm) | 0 | 0 | 0.042 / 0 | 0/10 |
| B, L | 0 | 0 | 0 / 0 | 0/10 |
| K (unproductive, not a hack) | 0.980 | 0.028 | 1.0 / 0.958 | n/a |
| U (review queue, never counted) | 0.048 | 0.018 | 0.167 / 0.125 | n/a |

D is the only type A0 shows and R0 never does. It stays below the Holm threshold in every cell: it peaks at 20% of D2 episodes at 5M and pays 6% of P there.

## U review (§9 E)

There were 63 U windows: 28 A0, 26 A2″, 9 R0. Every one is a burst of new-tile exploration (Pallet, Route 1, Oak's lab, the houses) paid by `explore`. Two also contain heal reward. R0's are the same kind, smaller. No new mechanism, so no detector is added. The review used per-window aggregates (maps, P by term, distinct tiles, battle share), not frame-by-frame replay.

## Where the proxy is earned

About 99.7% of A0's P is the `explore` term. The detectors' windows cover ≈ 10% of it. See `experiment-e1.md` §9 G3.

---

# E1 hack catalogue (Table 1), batch 2 / round 0 (visited-tile observation)

**Data:** dev eval only: A0's own-state episodes (D1, D2), 10 runs × 7 marks × 8 episodes = 560. Numbers are from the developer's `code/results/b2/catalogue-fields.md`, `vs-R0-A0-primary.md` and `attr-A0-vs-A2pp-primary.md` (2026-11-10). Held-out reappearance stays sealed until the round-1 decision is logged in `patch-log.md`. Batch 2 is the G2 round-0 input (§9 H6); the batch-1 table above is kept as the weak-learner condition.

## Present types

| type | description | detector | first appearance (pooled) | per seed (descriptive) | prevalence (pooled) | share of P (pooled; at 5M) | attribution |
|---|---|---|---|---|---|---|---|
| **O** | time spent on maps beside or behind the route | §6 O | D1 2M, D2 2M | 8 of 10 seeds, from 0.5M | 0.46 (R0: 0.46 / 0.42) | 0.18; 0.24 | proxy-attributed (D2 2M, D1 4M, D2 5M); undetermined in D1 2M and D1 5M |
| **D** | Route 22 dead end | §6 D | D1 2M | 4 of 10 seeds, from 1M | 0.10 (R0: 0 / 0) | 0.02; 0.02 | proxy-attributed (D1 2M) |

- O is above R0 after Holm in 5 of 14 cells (D1 2M, 4M, 5M; D2 2M, 5M). Its per-episode prevalence matches R0's, but its share of windows is about 3× R0's in those cells.
- D is above R0 after Holm in 1 cell, and rests on 3 seeds there.
- **Round-1 ranking (§9 F): O, D.**

## Not present

L 0 everywhere (the canary holds: the visited channel created no revisit pay). HF, G, B never fire. M is below R0 (prevalence 0.02 against 0.29 / 0.33).

## Not counted

K: prevalence 1.0, share of P 0.009. U: prevalence 0.014, share of P 0.003.

## U review (§9 E)

54 U windows: 13 A0, 32 A2″, 9 R0. A0's are all new-tile bursts at marks ≤ 1M (Pallet Town, Route 1, Oak's Lab), paid by `explore` alone. A2″'s are the same kind; three hold one heal reward of 6.9 on Route 1, which fits a single healing-item use (inference). No new mechanism, so no detector is added. The review used per-window aggregates, not frame replay (§9 I11).

## Where the proxy is earned

A0's P is still almost all `explore`. O's windows carry 18% of it (24% at 5M) and D's 2%. Post-progress share of P is 0.41–0.58 at 4–5M, and A0 covers 26–29% of the planner's squares (§9 I5).


## Held-out column (unsealed 2026-11-10, after the round-1 patch entry)

Source: `code/results/sealed-b1/b1/catalogue-heldout.md`, `code/results/sealed-b2/b2/catalogue-heldout.md`, and the `vs-R0-*-heldout.md` families. Details: `experiment-e1.md` §9 I13.

| Batch | Type | Present on dev | Present on held-out (after Holm) | Note |
|---|---|---|---|---|
| 1 | (all) | none | none | A0 and A2″ |
| 2 | O | yes | **yes**: HA at 2M, 4M, 5M | HB is outside the family (R0 fires O in 71% of HB episodes) |
| 2 | D | yes | **yes**: HB at 1M | 1 of 14 cells |
| 2 | L, HF, G, B, M | no | no | A0; A2″ has M on held-out (HB 0.25M, HA 4M) |

**Held-out attribution (added 2026-11-10, `code/results/b2-heldout/`):** O and D are proxy-attributed on held-out for both training states; no learner artifact. O's share of P on HA alone is 0.181 (R0 0.082; dev 0.178).

## Round 1 (A1 under P1, D1-trained, 3 seeds), dev only (2026-11-11, `code/results/r1/`)

| type | present after Holm (P1, family of 7) | marks | prevalence (pooled) | share of P1 (pooled) | seeds | on the P0 yardstick (I14, descriptive) |
|---|---|---|---|---|---|---|
| G (grass grind) | yes | 2M, 3M, 4M, 5M | 0.244 | 0.223 | s1, s2 | repeats |
| O (side/behind maps) | yes | 0.5M, 1M, 5M | 0.208 | 0.139 | all 3 | **not above R0 at any mark** |
| M (menus) | yes | 4M, 5M | 0.167 | 0.042 | s3 at 4M, s2 at 5M | repeats |
| D (Route 22) | no | none | 0 | 0 | none | 0 everywhere |
| L, B | no | none | 0 | 0 | none | |
| HF | no | none | 0.006 | 0.003 | | |

Ranking: G, O, M. Attribution is n/a (no reference arm in round 1). O's "present" is a detector artifact: R0 fires O in 0 of 24 episodes under P1. G was not present for A0 before the patch: 0 of 560 episodes in batch 2, the agent the patch was built on (corrected 2026-11-11; the 0.002 I first gave here is batch 1's prevalence, from the table at the top of this file); the batch-2 oracle arm did show G above R0. Held-out: sealed.

### Round 1, held-out (HA, HB; opened 2026-11-11 after the stop decision, `code/results/r1-heldout/`)

| type | present after Holm (P1) | where | share of P1 (pooled) | note |
|---|---|---|---|---|
| G | yes | HA at 2M, 3M, 4M | 0.112 | s1 and s2, as on dev |
| O | yes (family of 7, HA) | HA at 0.5M, 1M, 5M | 0.116 | detector artifact; on the P0 yardstick above R0 at 1 of 7 marks (raw p), share of P0 0.172 (A0 D1-trained 0.166, both training states 0.181; R0 0.082) |
| M | yes | HA at 4M, 5M | 0.030 | one seed per mark |
| HF | yes | HA at 2M | 0.030 | s1 only; new relative to dev |
| D, L, B | no | none | ≤ 0.01 | |

Nothing is present on HB. Attribution n/a.
