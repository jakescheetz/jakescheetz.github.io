#!/usr/bin/env python3
"""Reconstruct config.json["resume_log"] for a run whose resumes predate pokered/runlog.py (A0-D1-s0,
the first A2 attempt). Read-only unless --write.

  .venv/bin/python backfill_resume_log.py RUN_ID [--vm e1-a] [--zone us-central1-a] [--write]

Resumes come from the step drops in metrics.jsonl (runlog.synth_markers); restarts from
metrics.jsonl.before-restart-* files. Each entry's stop is dated by the first GCP stop event (Spot
preemption, guest shutdown, manual stop) after the stopped segment's last logged row, read with
`gcloud compute operations list` (laptop only; the VM holds no credentials). Entries carry
"backfilled": true, and basis is "metrics+vm_event" or "metrics" (no exact stop.json existed then).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from pokered import rom, runlog

KINDS = {"compute.instances.preempted": "preempted", "compute.instances.guestTerminate": "guest_shutdown",
         "stop": "stop", "compute.instances.hostError": "host_error"}


def vm_events(vm, zone, project):
    out = subprocess.run(["gcloud", "--project", project, "compute", "operations", "list", "--zones", zone,
                          "--format=json"], capture_output=True, text=True, check=True).stdout
    return [{"id": o["id"], "type": KINDS[o["operationType"]], "time": o["insertTime"], "vm": vm}
            for o in sorted(json.loads(out), key=lambda o: o["insertTime"])
            if o.get("operationType") in KINDS and o.get("targetLink", "").rsplit("/", 1)[-1] == vm]


def first_event_after(events, t0):
    after = [e for e in events if runlog._parse(e["time"]) >= t0]
    return after[0] if after else None


def entry(n, kind, ckpt_step, last, first_after, events):
    ev = first_event_after(events, last["time"]) if last and last.get("time") else None
    step, basis = runlog.stopped_at(last, None, ev)
    return {"resume": n, "kind": kind, "backfilled": True,
            "resumed_at": runlog._iso(first_after["time"]) if first_after and first_after.get("time") else None,
            "resumed_from_ckpt": f"ckpt-{ckpt_step}.pt" if kind == "resume" else None,
            "resumed_from_step": ckpt_step if kind == "resume" else 0,
            "stopped_at_step": step, "replayed_steps": (step - (ckpt_step if kind == "resume" else 0)) if step is not None else None,
            "basis": basis, "last_logged_step": last["step"] if last else None,
            "last_logged_time": runlog._iso(last["time"]) if last and last.get("time") else None,
            "max_step_before": last["step"] if last else None, "stop": None, "vm_event": ev}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--vm", default=os.environ.get("E1_VM", "e1-a"))
    ap.add_argument("--zone", default=os.environ.get("E1_ZONE", "us-central1-a"))
    ap.add_argument("--project", default=os.environ.get("E1_PROJECT", "sylvan-storm-513522-q9"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    d = rom.runs_dir() / a.run_id
    cfg = json.loads((d / "config.json").read_text())
    events = vm_events(a.vm, a.zone, a.project)
    out = []
    # restarts first (they happened before the current metrics.jsonl began)
    for f in sorted(glob.glob(str(d / "metrics.jsonl.before-restart-*"))):
        rows = [r for r in runlog._read(f) if "resume" not in r]
        cur = [r for r in runlog._read(d / "metrics.jsonl") if "resume" not in r]
        out.append(entry(len(out) + 1, "restart", 0, rows[-1] if rows else None, cur[0] if cur else None, events))
    metrics = [r for r in runlog._read(d / "metrics.jsonl") if "resume" not in r]
    for mi, _, c in runlog.synth_markers(cfg, metrics):
        out.append(entry(len(out) + 1, "resume", c, metrics[mi - 1], metrics[mi], events))
    print(json.dumps(out, indent=1))
    if cfg.get("resume_log") and not all(e.get("backfilled") for e in cfg["resume_log"]):
        sys.exit("config.json already has a live resume_log; not overwriting")
    if a.write:
        cfg["resume_log"] = out
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"wrote {len(out)} entries to {d / 'config.json'}")


if __name__ == "__main__":
    main()
