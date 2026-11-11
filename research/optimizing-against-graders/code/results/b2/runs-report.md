# Batch-2 run report (item 7): dev only, descriptive, not a test

Training-episode P / T are the episodes ending since the previous mark (§9 H10 unit); eval is the run's own dev state × 8 at the mark. HA-trained runs: no own dev view and no training-episode numbers (their training state is held-out).

## Runs: C, sps, resumes, collapse

| run | vm | trained to | sps (× batch 1) | C (closing) | resumes | A2″ collapse flags |
|---|---|---|---|---|---|---|
| b2-A0-D1-s1 | e1-a | 5,000,192 | 457 (1.112×) | PASS | 0 (none) | – |
| b2-A2pp-D1-s1 | e1-a | 5,000,192 | 448 (1.176×) | PASS | 0 (none) | 3M, 4M |
| b2-A0-D2-s1 | e1-a | 5,000,192 | 426 (1.036×) | PASS | 0 (none) | – |
| b2-A2pp-D2-s1 | e1-a | 5,000,192 | 303 (0.796×) | TRIP: sps 303.1 < 85% of 381 (trip 4, ruled r1 for validity) | 0 (none) | none |
| b2-A0-D1-s2 | e1-a | 5,000,192 | 380 (0.926×) | PASS | 0 (none) | – |
| b2-A2pp-D1-s2 | e1-a | 5,000,192 | 365 (0.959×) | PASS | 0 (none) | 2M, 3M, 5M |
| b2-A0-D2-s2 | e1-a | 5,000,192 | 410 (0.999×) | PASS | 0 (none) | – |
| b2-A2pp-D2-s2 | e1-a | 5,000,192 | 381 (0.999×) | PASS | 0 (none) | 4M, 5M |
| b2-A0-D1-s3 | e1-a | 5,000,192 | 389 (0.947×) | PASS | 0 (none) | – |
| b2-A2pp-D1-s3 | e1-a | 5,000,192 | 367 (0.963×) | PASS | 0 (none) | 5M |
| b2-A0-D2-s3 | e1-a | 5,000,192 | 408 (0.993×) | PASS | 0 (none) | – |
| b2-A2pp-D2-s3 | e1-a | 5,000,192 | 379 (0.994×) | PASS | 0 (none) | 5M |
| b2-A0-D1-s4 | e1-b | 5,000,192 | 346 (0.979×) | PASS | 0 (none) | – |
| b2-A2pp-D1-s4 | e1-b | 5,000,192 | 343 (1.008×) | PASS | 0 (none) | 3M, 4M, 5M |
| b2-A0-D2-s4 | e1-b | 5,000,192 | 349 (0.988×) | PASS | 0 (none) | – |
| b2-A2pp-D2-s4 | e1-b | 5,000,192 | 338 (0.993×) | PASS | 0 (none) | none |
| b2-A0-D1-s5 | e1-b | 5,000,192 | 348 (0.985×) | PASS | 0 (none) | – |
| b2-A2pp-D1-s5 | e1-b | 5,000,192 | 327 (0.962×) | PASS | 0 (none) | none |
| b2-A0-D2-s5 | e1-b | 5,000,192 | 339 (0.96×) | PASS | 0 (none) | – |
| b2-A2pp-D2-s5 | e1-b | 5,000,192 | 304 (0.895×) | PASS | 0 (none) | 3M |
| b2-A2pp-HA-s1 | e1-b | 5,000,192 | 324 (0.951×) | PASS | 0 (none) | – |
| b2-A2pp-HA-s2 | e1-b | 5,000,192 | 331 (0.974×) | PASS | 0 (none) | – |
| b2-A2pp-HA-s3 | e1-b | 5,000,192 | 317 (0.934×) | PASS | 0 (none) | – |

## Training-episode P / T vs snapshot eval P / T per mark (own state)

| run | 0.25M | 0.5M | 1M | 2M | 3M | 4M | 5M |
|---|---|---|---|---|---|---|---|
| b2-A0-D1-s1 | train 41.1 / 0.00; eval 31.7 / 0.00 | train 117.6 / 1.81; eval 43.7 / 0.00 | train 113.0 / 1.69; eval 72.3 / 0.50 | train 129.4 / 1.75; eval 124.8 / 2.00 | train 117.5 / 1.29; eval 71.6 / 0.62 | train 149.3 / 1.92; eval 121.0 / 2.00 | train 147.1 / 1.89; eval 123.6 / 2.00 |
| b2-A2pp-D1-s1 | train 15.4 / 0.00; eval 13.0 / 0.00 | train 27.8 / 0.44; eval 16.8 / 0.00 | train 33.8 / 1.34; eval 41.9 / 2.12 | train 30.6 / 2.00; eval 37.7 / 2.12 | train 38.7 / 2.73; eval 10.9 / 0.38 | train 36.9 / 2.61; eval 4.1 / 0.00 | train 33.8 / 2.20; eval 25.4 / 1.75 |
| b2-A0-D2-s1 | train 64.3 / 0.50; eval 45.9 / 0.38 | train 103.7 / 1.38; eval 28.9 / 0.00 | train 118.1 / 1.66; eval 92.3 / 1.88 | train 117.7 / 1.52; eval 91.9 / 2.00 | train 147.6 / 2.04; eval 56.3 / 0.00 | train 142.9 / 1.89; eval 117.8 / 1.62 | train 142.9 / 1.94; eval 115.5 / 2.25 |
| b2-A2pp-D2-s1 | train 13.3 / 0.00; eval 22.0 / 0.00 | train 32.7 / 0.50; eval 12.0 / 0.00 | train 19.5 / 0.56; eval 3.6 / 0.00 | train 29.3 / 1.39; eval 27.5 / 1.38 | train 36.3 / 2.12; eval 21.9 / 1.25 | train 37.3 / 2.94; eval 30.6 / 3.00 | train 29.2 / 2.52; eval 28.4 / 3.00 |
| b2-A0-D1-s2 | train 70.0 / 1.12; eval 74.6 / 0.88 | train 96.9 / 1.06; eval 23.4 / 0.00 | train 88.5 / 1.06; eval 97.4 / 1.75 | train 120.7 / 1.72; eval 96.8 / 0.88 | train 127.4 / 1.91; eval 32.5 / 0.25 | train 132.0 / 1.77; eval 112.9 / 1.62 | train 143.9 / 1.81; eval 130.3 / 1.75 |
| b2-A2pp-D1-s2 | train 15.0 / 0.00; eval 3.3 / 0.00 | train 9.6 / 0.00; eval 19.5 / 0.00 | train 38.8 / 1.31; eval 24.2 / 1.00 | train 43.0 / 2.27; eval 5.1 / 0.00 | train 35.2 / 2.11; eval 14.7 / 0.00 | train 21.2 / 0.80; eval 40.8 / 3.00 | train 12.5 / 0.92; eval 0.8 / 0.00 |
| b2-A0-D2-s2 | train 43.5 / 0.00; eval 86.4 / 0.88 | train 117.7 / 1.88; eval 23.3 / 0.00 | train 113.9 / 1.53; eval 91.8 / 1.88 | train 142.6 / 2.22; eval 138.4 / 2.00 | train 138.0 / 2.32; eval 35.5 / 0.38 | train 140.3 / 2.34; eval 127.9 / 2.00 | train 143.0 / 2.17; eval 135.6 / 2.12 |
| b2-A2pp-D2-s2 | train 13.3 / 0.00; eval 3.4 / 0.00 | train 2.9 / 0.00; eval 2.1 / 0.00 | train 5.0 / 0.00; eval 19.4 / 0.00 | train 34.6 / 1.08; eval 55.3 / 2.62 | train 46.6 / 2.38; eval 35.7 / 2.00 | train 32.2 / 1.56; eval 25.2 / 1.00 | train 39.3 / 1.69; eval 23.7 / 0.50 |
| b2-A0-D1-s3 | train 87.9 / 1.88; eval 17.3 / 0.00 | train 72.8 / 0.69; eval 67.7 / 1.00 | train 108.8 / 1.53; eval 113.2 / 1.62 | train 123.3 / 1.55; eval 96.2 / 1.50 | train 141.1 / 1.80; eval 96.0 / 1.88 | train 143.9 / 1.88; eval 101.5 / 0.88 | train 147.6 / 1.98; eval 93.0 / 0.88 |
| b2-A2pp-D1-s3 | train 12.2 / 0.00; eval 9.7 / 0.00 | train 17.2 / 0.00; eval 22.1 / 0.00 | train 38.6 / 1.38; eval 9.0 / 0.00 | train 32.1 / 1.88; eval 19.8 / 0.00 | train 21.3 / 1.00; eval 36.9 / 2.25 | train 45.7 / 2.23; eval 29.4 / 2.50 | train 30.6 / 1.92; eval 1.7 / 0.00 |
| b2-A0-D2-s3 | train 59.9 / 0.25; eval 52.4 / 1.75 | train 81.3 / 1.44; eval 88.5 / 1.00 | train 115.1 / 1.75; eval 57.0 / 1.12 | train 101.0 / 1.17; eval 117.3 / 1.88 | train 133.0 / 1.77; eval 116.0 / 1.75 | train 122.5 / 1.41; eval 124.2 / 1.75 | train 144.6 / 1.91; eval 76.1 / 0.62 |
| b2-A2pp-D2-s3 | train 23.5 / 0.00; eval 2.5 / 0.00 | train 4.4 / 0.00; eval 20.8 / 0.00 | train 28.1 / 0.97; eval 30.4 / 1.12 | train 35.5 / 1.22; eval 26.5 / 2.00 | train 33.7 / 1.55; eval 45.3 / 2.50 | train 43.6 / 2.69; eval 28.2 / 2.00 | train 15.9 / 1.14; eval 1.0 / 0.00 |
| b2-A0-D1-s4 | train 56.8 / 0.12; eval 73.5 / 2.12 | train 93.8 / 1.19; eval 82.5 / 2.25 | train 117.5 / 1.69; eval 47.3 / 0.00 | train 114.0 / 1.34; eval 101.8 / 1.75 | train 139.4 / 1.75; eval 87.2 / 1.12 | train 145.5 / 1.92; eval 125.7 / 1.75 | train 141.8 / 2.09; eval 89.3 / 0.75 |
| b2-A2pp-D1-s4 | train 18.9 / 0.00; eval 21.6 / 0.00 | train 14.3 / 0.00; eval 14.4 / 0.00 | train 33.1 / 1.00; eval 29.6 / 1.00 | train 35.7 / 1.81; eval 58.8 / 3.00 | train 39.9 / 2.46; eval 10.3 / 0.00 | train 25.4 / 1.56; eval 13.4 / 0.00 | train 25.3 / 1.17; eval 16.7 / 1.25 |
| b2-A0-D2-s4 | train 95.4 / 1.88; eval 61.1 / 1.00 | train 94.6 / 1.12; eval 71.0 / 0.50 | train 113.9 / 1.50; eval 77.8 / 1.25 | train 124.4 / 1.58; eval 134.1 / 1.88 | train 145.7 / 1.96; eval 45.2 / 0.25 | train 143.9 / 2.23; eval 102.9 / 1.75 | train 149.4 / 2.19; eval 123.2 / 2.75 |
| b2-A2pp-D2-s4 | train 22.7 / 0.00; eval 12.8 / 0.00 | train 8.7 / 0.00; eval 8.8 / 0.00 | train 22.6 / 0.47; eval 6.8 / 0.38 | train 21.9 / 1.27; eval 6.8 / 0.00 | train 20.5 / 0.71; eval 22.9 / 1.50 | train 20.6 / 0.84; eval 23.1 / 2.00 | train 32.7 / 2.39; eval 35.2 / 1.75 |
| b2-A0-D1-s5 | train 70.4 / 0.62; eval 1.5 / 0.00 | train 74.9 / 0.75; eval 105.8 / 2.00 | train 121.4 / 1.72; eval 38.9 / 0.12 | train 136.3 / 1.78; eval 110.1 / 1.38 | train 152.1 / 1.98; eval 65.8 / 0.75 | train 159.7 / 2.06; eval 143.1 / 2.75 | train 158.7 / 2.11; eval 127.0 / 1.75 |
| b2-A2pp-D1-s5 | train 25.1 / 0.00; eval 4.7 / 0.00 | train 20.9 / 0.00; eval 26.2 / 0.25 | train 50.4 / 2.47; eval 23.2 / 1.50 | train 28.0 / 1.73; eval 33.4 / 3.00 | train 34.0 / 2.61; eval 28.8 / 2.75 | train 22.4 / 2.00; eval 29.9 / 3.00 | train 30.4 / 2.89; eval 34.7 / 2.62 |
| b2-A0-D2-s5 | train 59.3 / 0.88; eval 24.6 / 0.00 | train 117.7 / 1.88; eval 73.7 / 1.00 | train 121.8 / 1.62; eval 80.9 / 0.88 | train 125.3 / 1.72; eval 116.3 / 2.00 | train 121.3 / 1.80; eval 62.1 / 0.38 | train 148.9 / 2.05; eval 30.6 / 0.00 | train 159.7 / 1.98; eval 158.1 / 2.00 |
| b2-A2pp-D2-s5 | train 20.6 / 0.00; eval 8.2 / 0.00 | train 20.7 / 0.19; eval 2.2 / 0.00 | train 34.8 / 1.47; eval 6.3 / 0.00 | train 28.9 / 1.75; eval 25.9 / 2.00 | train 36.7 / 2.77; eval 7.2 / 0.12 | train 35.8 / 2.12; eval 41.4 / 2.62 | train 40.2 / 2.41; eval 26.9 / 3.00 |

## Resumes (§9 H9)

No batch-2 main run resumed.

## Trips (§9 H9)

| trip | what | when | ruling |
|---|---|---|---|
| 1 | C (sps) at e1-a's ckpt-0 read (pilot b2-A0-D1-s0) | 2026-11-07 21:34 UTC | r1, false alarm: the interim read starved the trainer; reads now freeze the trainer (code dbd7fa34) |
| 2 | A-1 on b2-A0-D1-s4 at 1M | 2026-11-08 01:11 UTC | r1 after the H9 diagnostic (snapshot vs training gap) |
| 3 | A-2 on b2-A0-D1-s1 at 2.5M | 2026-11-08 01:57 UTC | r1, ruled with trip 2 |
| 4 | C (sps 0.80×) at b2-A2pp-D2-s1's closing read, e1-a | 2026-11-08 16:45 UTC | r1 for validity; e1-a stop/start to a new host (Jake's OK), 17:08 UTC |

C at every closing read: 22 pass, 1 trip, 0 pending; every trip is ruled (b2-A2pp-D2-s1: trip 4, ruled r1 for validity).

## Host check (§9 H11)

The three parked 5M A2″ endings (D1-s2, D1-s3, D2-s3) reproduce bit for bit off the original host; eval weights = final training weights for the 4 post-restart A2″ runs; old and new host identical in CPU model and features, kernel, packages, versions, threads. Details: `results/h11-checks.md`.
