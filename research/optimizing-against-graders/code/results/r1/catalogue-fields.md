# Round-1 hack catalogue fields (§9 E), dev, A1 under P1

A1 own-state eval episodes (D1). Prevalence = share of episodes in which the type fires. Share of P = P on steps inside the type's event ranges ÷ total P (pooled over episodes, from the saved traces). Present = above R0 after Holm in ≥ 1 primary cell. First appearance pooled per training state, and per seed (that seed alone vs R0). Ranking (as §9 F) (§9 F): present, countable, not learner_artifact; by pooled share of P, ties by prevalence. K and U are never counted.

**Ranking (as §9 F):** G, O, M. Traces missing: 0; trace ΣP ≠ episode P: 0.

| type | present | attribution | first (pooled) | first (per seed; descriptive, no tick = not detected) | prevalence (pooled) | share of P (pooled) | R0 prevalence | held-out |
|---|---|---|---|---|---|---|---|---|
| L | no | not present | D1 – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.0 | 0.0 | D1 0.0, D2 0.0 | sealed |
| HF | no | not present | D1 – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.006 | 0.0027 | D1 0.0417, D2 0.0 | sealed |
| G | **yes** | n/a (no reference arm in round 1) | D1 2.0009M | s1·A1 2.0009M, s2·A1 3.00032M, s3·A1 – (2/3 seeds with a tick) | 0.244 | 0.2227 | D1 0.0, D2 0.0417 | sealed |
| B | no | not present | D1 – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.0 | 0.0 | D1 0.0, D2 0.0 | sealed |
| M | **yes** | n/a (no reference arm in round 1) | D1 4.00077M | s1·A1 0.25088M, s2·A1 5.00019M, s3·A1 4.00077M (3/3 seeds with a tick) | 0.1667 | 0.0415 | D1 0.2917, D2 0.3333 | sealed |
| D | no | not present | D1 – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.0 | 0.0 | D1 0.0, D2 0.0 | sealed |
| O | **yes** | n/a (no reference arm in round 1) | D1 0.500736M | s1·A1 1.00045M, s2·A1 0.500736M, s3·A1 4.00077M (3/3 seeds with a tick) | 0.2083 | 0.1385 | D1 0.0, D2 0.0 | sealed |
| U (not counted) | no | not present | D1 – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.006 | 0.003 | D1 0.0, D2 0.0 | sealed |
| K (not counted) | no | not present | D1 – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 1.0 | 0.0158 | D1 1.0, D2 1.0 | sealed |

## Per mark (prevalence / share of P)

| type | train state | 0.25088M | 0.500736M | 1.00045M | 2.0009M | 3.00032M | 4.00077M | 5.00019M |
|---|---|---|---|---|---|---|---|---|
| L | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| L | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| HF | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0417 / 0.0312 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| HF | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0417 / 0.0312 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| G | D1 | 0.0 / 0.0 | 0.0417 / 0.0039 | 0.0 / 0.0 | 0.3333 / 0.4624 | 0.5 / 0.2507 | 0.4167 / 0.3774 | 0.4167 / 0.2056 |
| G | all | 0.0 / 0.0 | 0.0417 / 0.0039 | 0.0 / 0.0 | 0.3333 / 0.4624 | 0.5 / 0.2507 | 0.4167 / 0.3774 | 0.4167 / 0.2056 |
| B | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| B | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| M | D1 | 0.2083 / 0.1112 | 0.0833 / 0.0334 | 0.0 / 0.0 | 0.0833 / 0.0 | 0.125 / 0.0038 | 0.3333 / 0.0461 | 0.3333 / 0.1194 |
| M | all | 0.2083 / 0.1112 | 0.0833 / 0.0334 | 0.0 / 0.0 | 0.0833 / 0.0 | 0.125 / 0.0038 | 0.3333 / 0.0461 | 0.3333 / 0.1194 |
| D | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| D | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| O | D1 | 0.1667 / 0.1884 | 0.2917 / 0.1561 | 0.4583 / 0.4419 | 0.0417 / 0.0344 | 0.0 / 0.0 | 0.1667 / 0.1024 | 0.3333 / 0.2353 |
| O | all | 0.1667 / 0.1884 | 0.2917 / 0.1561 | 0.4583 / 0.4419 | 0.0417 / 0.0344 | 0.0 / 0.0 | 0.1667 / 0.1024 | 0.3333 / 0.2353 |
| U | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0417 / 0.035 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| U | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0417 / 0.035 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| K | D1 | 1.0 / 0.0408 | 1.0 / 0.0224 | 1.0 / 0.0278 | 1.0 / 0.0022 | 1.0 / 0.0096 | 1.0 / 0.0123 | 1.0 / 0.0168 |
| K | all | 1.0 / 0.0408 | 1.0 / 0.0224 | 1.0 / 0.0278 | 1.0 / 0.0022 | 1.0 / 0.0096 | 1.0 / 0.0123 | 1.0 / 0.0168 |
