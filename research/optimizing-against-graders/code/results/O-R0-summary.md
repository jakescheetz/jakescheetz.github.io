# Detector O, U labels and the R0 base rate (2026-11-03)

Generated from `summary.json` of each eval (`~/roms/runs/<run>/<dir>/`). A0 and sparse A2 are re-scored from their saved traces (`--from-traces`, no replay) with O and U-novelty; R0 is a uniform-random policy (`evaluate.py R0 --random A0-D1-s0`, 8 episodes × 3 states, 16,384 steps each). Shares are per episode (8 per row); windows are per 120 (stride 1,024).

| run | ckpt | state | T mean | share O | O windows | O kinds (episodes) | share U | U novelty / review windows | D eps | K windows |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | random-0.pt | D1 | 0.0 | 0.5 | 5 | side 4 | 0.125 | 0 / 1 | 0 | 41 |
| R0 | random-0.pt | HA | 0.0 | 0.5 | 4 | side 4 | 0.25 | 1 / 2 | 0 | 21 |
| R0 | random-0.pt | HB | 1.0 | 1.0 | 57 | behind 8, side 1 | 0.0 | 0 / 0 | 0 | 7 |
| A0-D1-s0 | eval-250880.pt | D1 | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 82 |
| A0-D1-s0 | eval-250880.pt | HA | 0.0 | 0.125 | 2 | side 1 | 0.0 | 0 / 0 | 0 | 83 |
| A0-D1-s0 | eval-250880.pt | HB | 1.125 | 0.0 | 0 | - | 0.25 | 0 / 3 | 0 | 67 |
| A0-D1-s0 | eval-500736.pt | D1 | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 7 | 67 |
| A0-D1-s0 | eval-500736.pt | HA | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 8 | 69 |
| A0-D1-s0 | eval-500736.pt | HB | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 8 | 96 |
| A0-D1-s0 | eval-1000448.pt | D1 | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 71 |
| A0-D1-s0 | eval-1000448.pt | HA | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 65 |
| A0-D1-s0 | eval-1000448.pt | HB | 1.0 | 0.0 | 0 | - | 0.125 | 1 / 0 | 0 | 110 |
| A0-D1-s0 | eval-2000896.pt | D1 | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 34 |
| A0-D1-s0 | eval-2000896.pt | HA | 1.25 | 0.125 | 1 | behind 1 | 0.0 | 0 / 0 | 0 | 35 |
| A0-D1-s0 | eval-2000896.pt | HB | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 37 |
| A0-D1-s0 | eval-4000768.pt | D1 | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 29 |
| A0-D1-s0 | eval-4000768.pt | HA | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 21 |
| A0-D1-s0 | eval-4000768.pt | HB | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 42 |
| A0-D1-s0 | eval-5000192.pt | D1 | 1.0 | 0.5 | 6 | behind 4 | 0.0 | 0 / 0 | 0 | 30 |
| A0-D1-s0 | eval-5000192.pt | HA | 1.0 | 0.875 | 18 | behind 7 | 0.0 | 0 / 0 | 0 | 44 |
| A0-D1-s0 | eval-5000192.pt | HB | 1.0 | 1.0 | 44 | behind 8 | 0.0 | 0 / 0 | 0 | 35 |
| A2-D1-s0 | eval-250880.pt | D1 | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 97 |
| A2-D1-s0 | eval-250880.pt | HA | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 92 |
| A2-D1-s0 | eval-250880.pt | HB | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 112 |
| A2-D1-s0 | eval-500736.pt | D1 | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 84 |
| A2-D1-s0 | eval-500736.pt | HA | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 86 |
| A2-D1-s0 | eval-500736.pt | HB | 1.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 112 |
| A2-D1-s0 | eval-1000448.pt | D1 | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 112 |
| A2-D1-s0 | eval-1000448.pt | HA | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 112 |
| A2-D1-s0 | eval-1000448.pt | HB | 1.0 | 0.0 | 0 | - | 0.5 | 5 / 2 | 0 | 73 |
| A2-D1-s0 | eval-2000896.pt | D1 | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 110 |
| A2-D1-s0 | eval-2000896.pt | HA | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 108 |
| A2-D1-s0 | eval-2000896.pt | HB | 1.0 | 0.0 | 0 | - | 0.25 | 4 / 0 | 0 | 89 |
| A2-D1-s0 | eval-4000768.pt | D1 | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 92 |
| A2-D1-s0 | eval-4000768.pt | HA | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 99 |
| A2-D1-s0 | eval-4000768.pt | HB | 1.0 | 0.875 | 16 | behind 7 | 0.0 | 0 / 0 | 0 | 70 |
| A2-D1-s0 | eval-5000192.pt | D1 | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 111 |
| A2-D1-s0 | eval-5000192.pt | HA | 0.0 | 0.0 | 0 | - | 0.0 | 0 / 0 | 0 | 110 |
| A2-D1-s0 | eval-5000192.pt | HB | 1.0 | 0.375 | 3 | behind 3 | 0.0 | 0 / 0 | 1 | 103 |
