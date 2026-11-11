#!/usr/bin/env python3
"""On the VM: state of training runs in ~/roms/runs (stdlib only; any python3).

  runs.py state RUN_ID   -> new | partial | done | eval-only   (done = last metrics step >= total_steps;
                                                       eval-only = random_policy baseline, e.g. R0)
  runs.py partial        -> run ids that have a config but are not done (bench-* excluded)
  runs.py table          -> one line per run: step / total, last sps_update, ep_T_mean, files
"""
import json
import sys
from pathlib import Path

RUNS = Path.home() / "roms" / "runs"


def last_row(d):
    f = d / "metrics.jsonl"
    if not f.exists():
        return None
    last = None
    with open(f) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line cut by a hard kill
                continue
            if "resume" not in r:         # resume marker rows (pokered/runlog.py) are not progress
                last = r
    return last


def state(d):
    if not (d / "config.json").exists():
        return "new"
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("random_policy"):
        # an eval-only baseline (R0: evaluate.py --r0 writes a copy of a run's config); never trained, so never
        # resumable. Before this check, --resume-all "resumed" R0 as an A0 trainer after a preemption.
        return "eval-only"
    r = last_row(d)
    return "done" if r and r["step"] >= cfg["total_steps"] else "partial"


def main():
    cmd = sys.argv[1]
    if cmd == "state":
        print(state(RUNS / sys.argv[2]))
    elif cmd == "partial":
        for d in sorted(RUNS.glob("*/")):
            if not d.name.startswith("bench-") and state(d) == "partial":
                print(d.name)
    elif cmd == "table":
        for d in sorted(RUNS.glob("*/")):
            if not (d / "config.json").exists():
                continue
            cfg = json.loads((d / "config.json").read_text())
            r = last_row(d) or {}
            ck = len(list(d.glob("ckpt-*.pt"))) + len(list(d.glob("eval-*.pt")))
            print(f"{d.name:<28} {state(d):<8} step {r.get('step', 0):>10,} / {cfg['total_steps']:<11,} "
                  f"sps_update {r.get('sps_update', '-'):>6}  T {r.get('ep_T_mean')}  ckpts {ck}"
                  f"{'  final_eval' if (d / 'final_eval.json').exists() else ''}")


if __name__ == "__main__":
    main()
