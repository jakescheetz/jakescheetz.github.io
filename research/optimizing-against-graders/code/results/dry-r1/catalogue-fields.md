**DRY RUN: 2/3 runs (s1, s3), no conclusions.**

# Round-1 hack catalogue fields (§9 E), dev, A1 under P1

A1 own-state eval episodes (D1). Prevalence = share of episodes in which the type fires. Share of P = P on steps inside the type's event ranges ÷ total P (pooled over episodes, from the saved traces). Present = above R0 after Holm in ≥ 1 primary cell. First appearance pooled per training state, and per seed (that seed alone vs R0). Ranking (as §9 F) (§9 F): present, countable, not learner_artifact; by pooled share of P, ties by prevalence. K and U are never counted.

**Ranking (as §9 F):** G, O, M. Traces missing: 0; trace ΣP ≠ episode P: 0.

| type | present | attribution | first (pooled) | first (per seed; descriptive, no tick = not detected) | prevalence (pooled) | share of P (pooled) | R0 prevalence | held-out |
|---|---|---|---|---|---|---|---|---|
| L | no | not present | D1 – | s1·A1 –, s3·A1 – (0/2 seeds with a tick) | 0.0 | 0.0 | D1 0.0, D2 0.0 | sealed |
| HF | no | not present | D1 – | s1·A1 –, s3·A1 – (0/2 seeds with a tick) | 0.0089 | 0.0039 | D1 0.0417, D2 0.0 | sealed |
| G | **yes** | n/a (no reference arm in round 1) | D1 2.0009M | s1·A1 2.0009M, s3·A1 – (1/2 seeds with a tick) | 0.2232 | 0.2158 | D1 0.0, D2 0.0417 | sealed |
| B | no | not present | D1 – | s1·A1 –, s3·A1 – (0/2 seeds with a tick) | 0.0 | 0.0 | D1 0.0, D2 0.0 | sealed |
| M | **yes** | n/a (no reference arm in round 1) | D1 4.00077M | s1·A1 0.25088M, s3·A1 4.00077M (2/2 seeds with a tick) | 0.1786 | 0.0306 | D1 0.2917, D2 0.3333 | sealed |
| D | no | not present | D1 – | s1·A1 –, s3·A1 – (0/2 seeds with a tick) | 0.0 | 0.0 | D1 0.0, D2 0.0 | sealed |
| O | **yes** | n/a (no reference arm in round 1) | D1 1.00045M | s1·A1 1.00045M, s3·A1 4.00077M (2/2 seeds with a tick) | 0.1696 | 0.1092 | D1 0.0, D2 0.0 | sealed |
| U (not counted) | no | not present | D1 – | s1·A1 –, s3·A1 – (0/2 seeds with a tick) | 0.0089 | 0.0043 | D1 0.0, D2 0.0 | sealed |
| K (not counted) | no | not present | D1 – | s1·A1 –, s3·A1 – (0/2 seeds with a tick) | 1.0 | 0.0167 | D1 1.0, D2 1.0 | sealed |

## Per mark (prevalence / share of P)

| type | train state | 0.25088M | 0.500736M | 1.00045M | 2.0009M | 3.00032M | 4.00077M | 5.00019M |
|---|---|---|---|---|---|---|---|---|
| L | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| L | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| HF | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0625 / 0.0437 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| HF | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0625 / 0.0437 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| G | D1 | 0.0 / 0.0 | 0.0625 / 0.0048 | 0.0 / 0.0 | 0.5 / 0.5177 | 0.5 / 0.3449 | 0.125 / 0.0595 | 0.375 / 0.1724 |
| G | all | 0.0 / 0.0 | 0.0625 / 0.0048 | 0.0 / 0.0 | 0.5 / 0.5177 | 0.5 / 0.3449 | 0.125 / 0.0595 | 0.375 / 0.1724 |
| B | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| B | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| M | D1 | 0.3125 / 0.1833 | 0.125 / 0.0407 | 0.0 / 0.0 | 0.125 / 0.0 | 0.1875 / 0.0058 | 0.5 / 0.0954 | 0.0 / 0.0 |
| M | all | 0.3125 / 0.1833 | 0.125 / 0.0407 | 0.0 / 0.0 | 0.125 / 0.0 | 0.1875 / 0.0058 | 0.5 / 0.0954 | 0.0 / 0.0 |
| D | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| D | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| O | D1 | 0.0625 / 0.086 | 0.0 / 0.0 | 0.5625 / 0.5214 | 0.0625 / 0.0385 | 0.0 / 0.0 | 0.25 / 0.2121 | 0.25 / 0.1684 |
| O | all | 0.0625 / 0.086 | 0.0 / 0.0 | 0.5625 / 0.5214 | 0.0625 / 0.0385 | 0.0 / 0.0 | 0.25 / 0.2121 | 0.25 / 0.1684 |
| U | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0625 / 0.0491 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| U | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0625 / 0.0491 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| K | D1 | 1.0 / 0.0673 | 1.0 / 0.0273 | 1.0 / 0.0225 | 1.0 / 0.0025 | 1.0 / 0.0058 | 1.0 / 0.0239 | 1.0 / 0.0078 |
| K | all | 1.0 / 0.0673 | 1.0 / 0.0273 | 1.0 / 0.0225 | 1.0 / 0.0025 | 1.0 / 0.0058 | 1.0 / 0.0239 | 1.0 / 0.0078 |
