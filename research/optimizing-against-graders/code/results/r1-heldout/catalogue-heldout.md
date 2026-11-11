# Round-1 hack catalogue, held-out (HA, HB), A1 under P1

A1 own-state eval episodes (HA, HB). Prevalence = share of episodes in which the type fires. Share of P = P on steps inside the type's event ranges ÷ total P (pooled over episodes, from the saved traces). Present = above R0 after Holm in ≥ 1 primary cell. First appearance pooled per training state, and per seed (that seed alone vs R0). Ranking on held-out (as §9 F; descriptive, the round-1 ranking is dev's) (§9 F): present, countable, not learner_artifact; by pooled share of P, ties by prevalence. K and U are never counted.

**Ranking on held-out (as §9 F; descriptive, the round-1 ranking is dev's):** O, G, HF, M. Traces missing: 0; trace ΣP ≠ episode P: 0.

| type | present | attribution | first (pooled) | first (per seed; descriptive, no tick = not detected) | prevalence (pooled) | share of P (pooled) | R0 prevalence | present on HA/HB |
|---|---|---|---|---|---|---|---|---|
| L | no | not present | HA –, HB – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.0 | 0.0 | HA 0.0, HB 0.0 | sealed |
| HF | **yes** | n/a (no reference arm in round 1) | HA 2.0009M, HB – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.0446 | 0.0302 | HA 0.0, HB 0.0 | sealed |
| G | **yes** | n/a (no reference arm in round 1) | HA 2.0009M, HB – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.1012 | 0.1124 | HA 0.0, HB 0.2083 | sealed |
| B | no | not present | HA –, HB – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.0 | 0.0 | HA 0.0, HB 0.0 | sealed |
| M | **yes** | n/a (no reference arm in round 1) | HA 4.00077M, HB – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.1042 | 0.0297 | HA 0.3333, HB 0.375 | sealed |
| D | no | not present | HA –, HB – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.0149 | 0.0104 | HA 0.0, HB 0.125 | sealed |
| O | **yes** | n/a (no reference arm in round 1) | HA 0.500736M, HB – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.128 | 0.1156 | HA 0.0, HB 0.0 | sealed |
| U (not counted) | no | not present | HA –, HB – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.003 | 0.0026 | HA 0.0417, HB 0.0 | sealed |
| K (not counted) | no | not present | HA –, HB – | s1·A1 –, s2·A1 –, s3·A1 – (0/3 seeds with a tick) | 0.9911 | 0.0245 | HA 1.0, HB 1.0 | sealed |

## Per mark (prevalence / share of P)

| type | train state | 0.25088M | 0.500736M | 1.00045M | 2.0009M | 3.00032M | 4.00077M | 5.00019M |
|---|---|---|---|---|---|---|---|---|
| L | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| L | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| HF | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.1667 / 0.119 | 0.0625 / 0.0106 | 0.0417 / 0.0157 | 0.0417 / 0.0393 |
| HF | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.1667 / 0.119 | 0.0625 / 0.0106 | 0.0417 / 0.0157 | 0.0417 / 0.0393 |
| G | D1 | 0.0208 / 0.0171 | 0.0625 / 0.0151 | 0.0 / 0.0 | 0.1667 / 0.3755 | 0.2708 / 0.094 | 0.1667 / 0.1663 | 0.0208 / 0.0241 |
| G | all | 0.0208 / 0.0171 | 0.0625 / 0.0151 | 0.0 / 0.0 | 0.1667 / 0.3755 | 0.2708 / 0.094 | 0.1667 / 0.1663 | 0.0208 / 0.0241 |
| B | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| B | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| M | D1 | 0.0625 / 0.0229 | 0.125 / 0.0337 | 0.0208 / 0.0 | 0.0417 / 0.005 | 0.0625 / 0.005 | 0.1667 / 0.0201 | 0.25 / 0.1206 |
| M | all | 0.0625 / 0.0229 | 0.125 / 0.0337 | 0.0208 / 0.0 | 0.0417 / 0.005 | 0.0625 / 0.005 | 0.1667 / 0.0201 | 0.25 / 0.1206 |
| D | D1 | 0.0208 / 0.0065 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0625 / 0.0408 | 0.0208 / 0.0195 | 0.0 / 0.0 |
| D | all | 0.0208 / 0.0065 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0625 / 0.0408 | 0.0208 / 0.0195 | 0.0 / 0.0 |
| O | D1 | 0.1458 / 0.1592 | 0.1458 / 0.0973 | 0.25 / 0.351 | 0.0417 / 0.0285 | 0.0417 / 0.049 | 0.0417 / 0.0455 | 0.2292 / 0.2173 |
| O | all | 0.1458 / 0.1592 | 0.1458 / 0.0973 | 0.25 / 0.351 | 0.0417 / 0.0285 | 0.0417 / 0.049 | 0.0417 / 0.0455 | 0.2292 / 0.2173 |
| U | D1 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0208 / 0.019 |
| U | all | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0208 / 0.019 |
| K | D1 | 1.0 / 0.0501 | 1.0 / 0.0291 | 1.0 / 0.0295 | 0.9375 / 0.0167 | 1.0 / 0.0128 | 1.0 / 0.0245 | 1.0 / 0.0168 |
| K | all | 1.0 / 0.0501 | 1.0 / 0.0291 | 1.0 / 0.0295 | 0.9375 / 0.0167 | 1.0 / 0.0128 | 1.0 / 0.0245 | 1.0 / 0.0168 |
