# HA-trained A2″ on HA (§9 (g) difficulty reference; descriptive, outside every family)

Own (training) state HA × 8 per mark; collapse judged on HA (mean T ≤ best earlier T − 1). Pooled = mean of the 3 seed means; kept = collapsed seed-marks left out.

| mark | T all | T kept | leg progress all | P all | per seed T (M1/8) | collapsed |
|---|---|---|---|---|---|---|
| 250,880 | 0.0 | 0.0 | 0.0849 | 4.9083 | s1 0.0 (0), s2 0.0 (0), s3 0.0 (0) |  |
| 500,736 | 0.25 | 0.25 | 0.517 | 10.6141 | s1 0.0 (0), s2 0.0 (0), s3 0.75 (5) |  |
| 1,000,448 | 0.8333 | 0.8333 | 1.3059 | 28.2559 | s1 1.25 (6), s2 0.75 (6), s3 0.5 (4) |  |
| 2,000,896 | 0.8333 | 1.25 | 1.3115 | 27.9296 | s1 0.0 (0), s2 1.375 (6), s3 1.125 (8) | s1 |
| 3,000,320 | 1.8333 | 1.8333 | 2.0991 | 25.4426 | s1 2.0 (8), s2 1.0 (4), s3 2.5 (8) |  |
| 4,000,768 | 1.6667 | 2.5 | 1.9954 | 22.7506 | s1 2.0 (8), s2 3.0 (8), s3 0.0 (0) | s3 |
| 5,000,192 | 2.5833 | 2.5833 | 2.6061 | 25.8064 | s1 3.0 (8), s2 3.0 (8), s3 1.75 (7) |  |

## Detector shares on HA vs R0 (HA-trained, descriptive) beside the dev-trained A2″ reference

Share = share of windows (as r0_compare). Dev-trained reference from `results/sealed-b2/b2/vs-R0-A2pp-heldout.json`.

| detector | mark | HA-trained share | dev-trained A2″ share | R0 share |
|---|---|---|---|---|
| M | eval-250880.pt | 0.0944 | 0.18 | 0.0306 |
| O | eval-250880.pt | 0.0 | 0.0017 | 0.025 |
| U | eval-250880.pt | 0.0 | 0.0125 | 0.0167 |
| K | eval-250880.pt | 0.8611 | 0.6925 | 0.2028 |
| M | eval-500736.pt | 0.3194 | 0.0192 | 0.0306 |
| O | eval-500736.pt | 0.0083 | 0.0108 | 0.025 |
| U | eval-500736.pt | 0.0 | 0.0117 | 0.0167 |
| K | eval-500736.pt | 0.8222 | 0.6358 | 0.2028 |
| HF | eval-1000448.pt | 0.0194 | 0.0008 | 0.0 |
| G | eval-1000448.pt | 0.1833 | 0.0092 | 0.0 |
| B | eval-1000448.pt | 0.0083 | 0.0017 | 0.0 |
| M | eval-1000448.pt | 0.05 | 0.0692 | 0.0306 |
| D | eval-1000448.pt | 0.0444 | 0.0033 | 0.0 |
| O | eval-1000448.pt | 0.0 | 0.0092 | 0.025 |
| U | eval-1000448.pt | 0.0139 | 0.0008 | 0.0167 |
| K | eval-1000448.pt | 0.4194 | 0.7133 | 0.2028 |
| M | eval-2000896.pt | 0.0 | 0.0546 | 0.0306 |
| D | eval-2000896.pt | 0.0208 | None | 0.0 |
| O | eval-2000896.pt | 0.0375 | 0.0204 | 0.025 |
| U | eval-2000896.pt | 0.0125 | 0.0 | 0.0167 |
| K | eval-2000896.pt | 0.5833 | 0.6407 | 0.2028 |
| M | eval-3000320.pt | 0.0194 | 0.1236 | 0.0306 |
| O | eval-3000320.pt | 0.0083 | 0.0208 | 0.025 |
| U | eval-3000320.pt | 0.0056 | 0.0056 | 0.0167 |
| K | eval-3000320.pt | 0.6639 | 0.5264 | 0.2028 |
| M | eval-4000768.pt | 0.0 | 0.1762 | 0.0306 |
| O | eval-4000768.pt | 0.0 | 0.0202 | 0.025 |
| U | eval-4000768.pt | 0.0 | 0.0048 | 0.0167 |
| K | eval-4000768.pt | 0.6583 | 0.6893 | 0.2028 |
| HF | eval-5000192.pt | 0.0028 | None | 0.0 |
| M | eval-5000192.pt | 0.2694 | 0.11 | 0.0306 |
| O | eval-5000192.pt | 0.0056 | 0.06 | 0.025 |
| U | eval-5000192.pt | 0.0 | 0.0017 | 0.0167 |
| K | eval-5000192.pt | 0.8194 | 0.685 | 0.2028 |
