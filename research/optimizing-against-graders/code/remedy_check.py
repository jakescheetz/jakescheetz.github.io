#!/usr/bin/env python3
"""A2″ remedy success criterion + pre-registered blackout-share test (experiment-e1.md §9 "Step 0 applied").
No training; reads a finished run's episodes.jsonl and eval/summary.json.

  .venv/bin/python remedy_check.py RUN [--out results/RUN-remedy.json]

1. Training batches: the 8 episodes that end at the same global_step (8 envs reset together). Batch mean T.
2. Success criterion (all three):
   (a) §9 gate: M2 in >= 1 D1 eval episode at any checkpoint;
   (b) no collapsed D1 eval checkpoint: mean D1 eval T >= 1 below its best at any EARLIER checkpoint;
   (c) training: no run of >= 3 consecutive batches with mean T >= 1 below the running best (best over all
       earlier batches), counted from the first M3 batch (first batch with any episode at T >= 3) on.
3. Blackout share (needs blackout_steps in the episode rows): per batch b, the share of its 8 envs with a
   blackout in the last 2,048 steps of the episode. Batch b is "before a dip" if batch b+1 has mean T >= 1
   below the running best through b; otherwise "before success". Only batches where a dip was possible
   (running best through b >= 1) are compared. One-sided Monte Carlo permutation p (B = 100k, r0_compare.perm_p)
   for share(before dip) > share(before success). Fewer than 3 dips -> reported as underpowered, not a null.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import rom, runlog  # noqa: E402
import r0_compare as RC  # noqa: E402

LAST = 2048


def batches(rd):
    by = defaultdict(list)
    for e in runlog.load_kept(rd, "episodes.jsonl"):   # drops resume markers and discarded branches
        by[e["global_step"]].append(e)
    return [(gs, by[gs]) for gs in sorted(by)]


def train_criterion(bs):
    means = [sum(e["T"] for e in eps) / len(eps) for _, eps in bs]
    first_m3 = next((i for i, (_, eps) in enumerate(bs) if any(e["T"] >= 3 for e in eps)), None)
    runs, cur, best = [], [], float("-inf")
    for i, m in enumerate(means):
        below = first_m3 is not None and i > first_m3 and m <= best - 1
        if below:
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
        best = max(best, m)
    if cur:
        runs.append(cur)
    longest = max((len(r) for r in runs), default=0)
    return means, first_m3, runs, longest


def eval_criterion(rd):
    s = json.load(open(rd / "eval" / "summary.json"))["summary"]
    d1 = sorted((v["step"], v["T_mean"], v["share_ge_M2"]) for k, v in s.items() if k.endswith("|D1"))
    gate = any(m2 > 0 for _, _, m2 in d1)
    collapsed, best = [], float("-inf")
    for st, t, _ in d1:
        if t <= best - 1:
            collapsed.append(st)
        best = max(best, t)
    return d1, gate, collapsed


def blackout_share(bs, means):
    if not all("blackout_steps" in e for _, eps in bs for e in eps):
        return None
    share = [sum(any(s >= e["steps"] - LAST for s in e["blackout_steps"]) for e in eps) / len(eps) for _, eps in bs]
    any_bo = [sum(bool(e["blackout_steps"]) for e in eps) / len(eps) for _, eps in bs]
    dip, ok, best = [], [], float("-inf")
    for b in range(len(bs) - 1):
        best = max(best, means[b])
        if best < 1:
            continue
        (dip if means[b + 1] <= best - 1 else ok).append(b)
    out = {"share_last2048": share, "share_any_blackout": any_bo, "before_dip": dip, "before_success": ok}
    if dip and ok:
        a, c = [share[b] for b in dip], [share[b] for b in ok]
        p, method, n = RC.perm_p(a, c)
        out.update(mean_dip=sum(a) / len(a), mean_success=sum(c) / len(c), p=p, method=method, B=n)
    out["underpowered"] = len(dip) < 3
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out")
    a = ap.parse_args()
    rd = rom.runs_dir() / a.run
    bs = batches(rd)
    means, first_m3, runs, longest = train_criterion(bs)
    print(f"{a.run}: {len(bs)} batches; first M3 batch {first_m3} "
          f"(ends {bs[first_m3][0] if first_m3 is not None else None})")
    for i, (gs, eps) in enumerate(bs):
        print(f"  {i:3d} {gs:>10,}  T {means[i]:.3f}  max {max(e['T'] for e in eps)}"
              + ("  BELOW" if any(i in r for r in runs) else ""))
    print(f"(c) below-best runs after first M3: {[[bs[r[0]][0], len(r)] for r in runs]}; longest {longest} "
          f"-> {'FAIL' if longest >= 3 else 'pass'}")
    res = {"run": a.run, "batches": [{"end": gs, "meanT": m} for (gs, _), m in zip(bs, means)],
           "first_m3_batch": first_m3, "below_runs": [[bs[r[0]][0], len(r)] for r in runs],
           "longest_below_run": longest, "train_ok": longest < 3}
    if (rd / "eval" / "summary.json").exists():
        d1, gate, collapsed = eval_criterion(rd)
        print("D1 eval (step, T mean, >=M2):", d1)
        print(f"(a) gate {'PASS' if gate else 'FAIL'}; (b) collapsed checkpoints {collapsed or 'none'}")
        res.update(d1_eval=d1, gate=gate, collapsed=collapsed)
        res["success"] = gate and not collapsed and longest < 3
        print("REMEDY SUCCESS" if res["success"] else "REMEDY FAILS")
    bo = blackout_share(bs, means)
    if bo is None:
        print("blackout share: no blackout_steps in the episode rows")
    else:
        print(f"blackout share: {len(bo['before_dip'])} batches before a dip, {len(bo['before_success'])} before "
              f"success; " + (f"mean share {bo['mean_dip']:.3f} vs {bo['mean_success']:.3f}, p = {bo['p']:.4f}"
                              if "p" in bo else "no comparison") + ("  (UNDERPOWERED: < 3 dips)" if bo["underpowered"] else ""))
    res["blackout_share"] = bo
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
