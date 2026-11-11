#!/usr/bin/env python3
"""Credit estimate and guard sizing for the E1 main arms on on-demand GCP (stdlib only).

  gcp/plan_cost.py estimate --M 4000000 [--M 6000000 ...]     whole-plan hours + credit, per VM, per batch
  gcp/plan_cost.py hours PLAN --vcpus 8 [--M N]                hours one queue file needs on one VM
                                                                (used by gcp/arm.sh to size the guards)

The plan (experiment-e1.md §8 base table, A2 -> A2″ per §9; the pilot is done and not counted; M = 5M per §9 "A2″ 10M extension"):
  batch 1   A0 5 seeds x {D1, D2}  +  A2″ 5 D1 + 5 D2 + 3 HA           23 runs, tiles-CNN
            (A2″ 13 runs: experiment-e1.md §9 "A2″ main-arm states and seeds", lead-approved 2026-11-03)
  batch 2-5 A1 rounds 1-4, 3 seeds each (a patch round sits between them) 3 runs each, tiles-CNN
  batch 6   R1: A0 on D1, screen-CNN, 2 seeds                             2 runs, ~0.6x tiles speed
Each batch's runs are split across e1-a (8 vCPU) and e1-b (4 vCPU) to even out wall-clock; each VM runs
its share one at a time (gcp/remote/queue.sh), each run followed by its eval.

Rates and prices are ESTIMATES (see the constants). The ceiling arm.sh enforces is the whole-plan VM-hours
in this file x (1 + MARGIN), priced at the same rates, plus disks.
"""
import argparse
import math
import re
import sys

# measured on e1-a (t2d-standard-8, 8 envs): 391 median in the 1-h bench, 340-445 in the pilots; use 370.
# e1-b (t2d-standard-4, us-central1-c) runs the same 8 envs as every main run (batch size = 128 x envs must
# match the pilots): measured 351 sps over its first 112k steps (A0, D1, 2026-11-04 bench-e1b-8env);
# plan with 330 (e1-a's 1-h bench was 391 against 370 in full runs).
SPS = {8: 370, 4: 330}
SCREEN_SPEED = 0.6          # screen-CNN vs tiles-CNN steps/s (laptop 47 vs 80)
EVAL_SHARE = 0.25 * 7 / 6    # §8: eval adds ≈25% for 6 marks (A2′/A2″ pilot eval on e1-a: ≈14%; e1-b has half the
                             # workers); x 7/6 for the 3M mark (§9 "A2″ 10M extension"): 7 checkpoints per run
IDLE_TAIL_H = 20 / 60       # idle guard: the VM stops this long after its queue empties
# on-demand us-central1, third-party trackers seen 2026-11-02 (UNVERIFIED; README "GCP trial")
RATE = {8: 0.338, 4: 0.169}  # t2d-standard-{8,4} $/h
IPV4 = 0.005                 # in-use external IPv4, $/h per VM (GCP list price; unverified for the trial)
DISK = 30 * 0.10 / 730       # 30 GB pd-balanced at ≈$0.10/GB-month, $/h per VM, billed while the disk exists
MARGIN = 0.30                # guard margin over the estimate

VMS = {"e1-a": 8, "e1-b": 4}


def run_hours(steps, vcpus, screen=False):
    sps = SPS[vcpus] * (SCREEN_SPEED if screen else 1.0)
    return steps / sps / 3600 * (1 + EVAL_SHARE)


def split(n_runs, steps, screen=False):
    """Assign n equal runs to the two VMs (each runs its share sequentially) to minimise wall-clock."""
    best = None
    for na in range(n_runs + 1):
        ha = na * run_hours(steps, 8, screen)
        hb = (n_runs - na) * run_hours(steps, 4, screen)
        if best is None or max(ha, hb) < best[0]:
            best = (max(ha, hb), na, ha, hb)
    _, na, ha, hb = best
    return {"e1-a": (na, ha), "e1-b": (n_runs - na, hb)}


def batches(M):
    out = [("1: A0 x10 + A2″ x13", split(23, M))]
    out += [(f"{r + 2}: A1 round {r + 1} x3", split(3, M)) for r in range(4)]
    out.append(("6: R1 screen-CNN x2", split(2, M, screen=True)))
    return out


def vm_cost(vm, hours):
    return hours * (RATE[VMS[vm]] + IPV4)


def estimate(M, days_on_disk):
    rows, tot_h, tot_usd = [], {v: 0.0 for v in VMS}, 0.0
    wall = 0.0
    for name, sp in batches(M):
        b_usd = 0.0
        for vm, (n, h) in sp.items():
            if n:
                h += IDLE_TAIL_H
            tot_h[vm] += h
            b_usd += vm_cost(vm, h)
        wall += max(h for _, h in sp.values())
        tot_usd += b_usd
        rows.append((name, sp, b_usd))
    disk = 2 * DISK * 24 * days_on_disk
    est = tot_usd + disk
    ceil_h = {vm: h * (1 + MARGIN) for vm, h in tot_h.items()}
    ceiling = sum(vm_cost(vm, h) for vm, h in ceil_h.items()) + disk
    return {"rows": rows, "vm_hours": tot_h, "compute_usd": tot_usd, "disk_usd": disk, "estimate_usd": est,
            "wall_h": wall, "ceiling_vm_hours": ceil_h, "ceiling_usd": ceiling}


def plan_lines(path, M):
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        train = line.split(" ; eval:")[0]
        if "{M}" in train and not M:
            sys.exit(f"{path}: has {{M}}; give --M")
        train = train.replace("{M}", str(M or 0))
        m = re.search(r"--steps[ =](\d+)", train)
        yield int(m.group(1)) if m else 5_000_000, "--policy screen" in train or "--policy=screen" in train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--M", type=int, action="append", required=True)
    e.add_argument("--days", type=float, default=14, help="days both disks exist (default 14)")
    h = sub.add_parser("hours")
    h.add_argument("plan")
    h.add_argument("--vcpus", type=int, choices=sorted(SPS), required=True)
    h.add_argument("--M", type=int)
    a = ap.parse_args()
    if a.cmd == "hours":
        hrs = sum(run_hours(s, a.vcpus, sc) for s, sc in plan_lines(a.plan, a.M))
        # guard window: estimate x (1 + margin) + 1 h (boot, setup, idle tail), whole hours
        print(math.ceil(hrs * (1 + MARGIN) + 1), round(hrs, 2), RATE[a.vcpus] + IPV4)
        return
    print(f"rates: e1-a {SPS[8]} sps @ ${RATE[8]}/h, e1-b {SPS[4]} sps (unmeasured) @ ${RATE[4]}/h, "
          f"+${IPV4}/h IPv4 each; eval +{EVAL_SHARE:.0%}; screen-CNN x{SCREEN_SPEED}; idle tail "
          f"{IDLE_TAIL_H * 60:.0f} min; disks 2 x 30 GB for {a.days:g} days. Prices UNVERIFIED (third-party).")
    for M in a.M:
        r = estimate(M, a.days)
        print(f"\nM = {M / 1e6:g}M")
        for name, sp, usd in r["rows"]:
            parts = ", ".join(f"{vm} {n} runs {h:.1f} h" for vm, (n, h) in sp.items())
            print(f"  batch {name:<24} {parts:<44} ${usd:6.2f}")
        vh = r["vm_hours"]
        print(f"  VM-hours   e1-a {vh['e1-a']:.0f} h, e1-b {vh['e1-b']:.0f} h; compute wall-clock ≈ {r['wall_h']:.0f} h "
              f"(+ the patch rounds' human time between batches 1-5, VMs stopped)")
        print(f"  ESTIMATE   ${r['estimate_usd']:.0f}  (compute ${r['compute_usd']:.2f} + disks ${r['disk_usd']:.2f})")
        ch = r["ceiling_vm_hours"]
        print(f"  CEILING    ${r['ceiling_usd']:.0f}  = e1-a ≤ {ch['e1-a']:.0f} h, e1-b ≤ {ch['e1-b']:.0f} h "
              f"(estimate + {MARGIN:.0%}) + disks")


if __name__ == "__main__":
    main()
