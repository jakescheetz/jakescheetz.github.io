#!/usr/bin/env python3
"""Batch-2 re-cost (experiment-e1.md §9 "Batch 2" H7, H4′), from batch 1's MEASURED times, not plan rates.

  python3 gcp/b2_cost.py --sps-ratio R [--margin 0.30]

Inputs (measured; see code/README 'Batch 2'):
  * batch-1 training hours per VM, from every run's metrics.jsonl timestamps (gaps > 10 min, i.e. resumes,
    capped): e1-a 42.6 h (12 runs), e1-b 44.7 h (11 runs);
  * batch-1 closing eval hours (D1 HA HB × 7 marks = 21 state-marks per run): e1-a 5.8 h, e1-b 10.4 h;
    the later D2 + D1-KL pass: e1-a 3.4 h, e1-b 5.3 h (folded into batch 2's single eval pass);
  * R = batch-2 sps ÷ batch-1 sps, measured in the smoke (tilegrid_visited vs tilegrid, same machine).
Batch 2: training ÷ R; eval = 32 state-marks per run (D1 D2 HA HB × marks 0 + 7), all with KL (+6%, measured
in batch 1's D2 pass) at batch 1's per-state-mark rate; the 4 tripwired runs' interim reads (3 × 8 episodes,
2 workers, beside training); the stage-1 pilots (2.5M each, concurrent; D1 × 8 at 6 checkpoints); idle-guard
tails (20 min per queue start: pilot + main on each VM); disks and IPv4 for the whole span.
Rates: the ledger's (~/roms/runs/gcp-budget.json): e1-a $0.343/h, e1-b $0.174/h. Ceiling = estimate × (1 + margin).
"""
import argparse

TRAIN_H = {"e1-a": 42.6, "e1-b": 44.7}
EVAL_H = {"e1-a": 5.8, "e1-b": 10.4}
RUNS = {"e1-a": 12, "e1-b": 11}
SM_B1 = 21                      # state-marks per run in batch 1's closing eval
SM_B2 = 32                      # 4 states × (0 + 7 marks)
KL = 1.06
RATE = {"e1-a": 0.343, "e1-b": 0.174}
SPS_B1 = {"e1-a": 411, "e1-b": 340}      # pilot arms: A0 on e1-a, A2″ on e1-b (batch-1 medians)
PILOT_STEPS = 2_500_000
IDLE_H = 20 / 60
IPV4 = 0.005
DISK_H = 30 * 0.10 / 730
SPAN_H = 4 * 24                 # disks exist for the whole batch (≈ 4 days incl. the staged wait)


def estimate(r, margin):
    rows, total = {}, 0.0
    for vm in TRAIN_H:
        per_sm = EVAL_H[vm] / (RUNS[vm] * SM_B1)
        train = TRAIN_H[vm] / r
        ev = RUNS[vm] * SM_B2 * per_sm * KL
        interim = 2 * 3 * per_sm * KL                     # 2 tripwired runs × 3 reads × 1 state-mark
        pilot = PILOT_STEPS / (SPS_B1[vm] * r) / 3600 + 6 * per_sm * KL
        idle = 2 * IDLE_H
        h = train + ev + interim + pilot + idle
        usd = h * (RATE[vm] + IPV4) + SPAN_H * DISK_H
        rows[vm] = {"train_h": round(train, 1), "eval_h": round(ev, 1), "interim_h": round(interim, 2),
                    "pilot_h": round(pilot, 2), "idle_h": round(idle, 2), "hours": round(h, 1),
                    "usd": round(usd, 2), "ceiling_hours": round(h * (1 + margin))}
        total += usd
    return rows, round(total, 2), round(total * (1 + margin), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sps-ratio", type=float, required=True)
    ap.add_argument("--margin", type=float, default=0.30)
    a = ap.parse_args()
    rows, est, ceil = estimate(a.sps_ratio, a.margin)
    for vm, r in rows.items():
        print(vm, r)
    print(f"estimate ${est}  ceiling ${ceil} (+{a.margin:.0%})  sps ratio {a.sps_ratio}")


if __name__ == "__main__":
    main()
