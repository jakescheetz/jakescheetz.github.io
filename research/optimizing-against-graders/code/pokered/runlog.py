"""Resume records for the paper: which steps of a run were trained twice.

When a run stops early (Spot preemption, cost guard, crash) and is resumed, training restarts from the
last checkpoint. The steps between that checkpoint and the stop get trained again with new env seeds.
Each resume appends one entry to config.json["resume_log"]:

  resume              1, 2, ... (restarts with no checkpoint are counted too; kind tells them apart)
  kind                "resume" (from a checkpoint) or "restart" (no checkpoint: from step 0)
  resumed_at          wall time of the resume (UTC)
  resumed_from_ckpt   checkpoint file name, or null for a restart
  resumed_from_step   its global step (0 for a restart)
  stopped_at_step     where the previous segment stopped, as best known (see basis)
  replayed_steps      stopped_at_step - resumed_from_step
  basis               how stopped_at_step was found:
                        "stop.json"            exact: the trainer saved on the shutdown signal and wrote
                                               stop.json (an update boundary; a rollout in progress, at
                                               most rollout x envs steps, is dropped)
                        "metrics+vm_event"     estimate: last logged step + its steps/s x the time between
                                               that log line and the GCP stop event
                        "metrics"              lower bound: the last logged step only
  last_logged_step, last_logged_time   the last metrics.jsonl row of the stopped segment
  stop                the stop.json contents (signal, time), when present
  vm_event            the GCP event that stopped the VM (type, time), when gcp/run.sh recorded one

gcp/run.sh writes GCP stop events to ~/roms/runs/vm-events.jsonl on the VM before it resumes anything.

The trainer also writes a resume row into metrics.jsonl and episodes.jsonl at each resume (§8 "Resume
rule"): {"resume": n, "kind", "step"/"global_step": checkpoint step, "ckpt", "time", "max_step_before"}.
Rows after it continue from the checkpoint step, so rows logged before it with a larger step are on a
discarded branch. kept_rows() drops them (and the resume rows); every analysis and plot should read the
logs through it. Logs written before resume rows existed are handled from the step drops + resume_log.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

VM_EVENTS = "vm-events.jsonl"


def _iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def _parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def last_metrics_row(path):
    path = Path(path)
    if not path.exists():
        return None
    last = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:                 # a line cut by a hard kill
                continue
            if "resume" not in r:              # resume rows are markers, not progress
                last = r
    return last


def max_step(path, key="step"):
    path = Path(path)
    if not path.exists():
        return None
    m = None
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "resume" not in r and r.get(key) is not None:
            m = r[key] if m is None else max(m, r[key])
    return m


def resume_row(entry, key="step"):
    """The marker row written into metrics.jsonl (key "step") and episodes.jsonl (key "global_step")."""
    return {"resume": entry["resume"], "kind": entry["kind"], key: entry["resumed_from_step"],
            "ckpt": entry["resumed_from_ckpt"], "time": time.time(), "time_iso": entry["resumed_at"],
            "max_step_before": entry.get("max_step_before")}


def kept_rows(rows, key=None):
    """Rows (in file order) minus the discarded branches and the resume markers. key: "step" (metrics)
    or "global_step" (episodes), guessed if None. Each marker at checkpoint step c drops the rows kept so
    far with step > c: they were trained on a branch the resume threw away."""
    rows = list(rows)
    if key is None:
        key = "global_step" if any("global_step" in r for r in rows if "resume" not in r) else "step"
    kept = []
    for r in rows:
        if "resume" in r:
            c = r.get(key, r.get("step"))
            kept = [k for k in kept if k.get(key) is None or k[key] <= c]
        else:
            kept.append(r)
    return kept


def _read(path):
    out = []
    if Path(path).exists():
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def synth_markers(cfg, metrics):
    """For logs written before resume rows existed: [(metrics index, episodes index, checkpoint step)] per
    resume, found from the step drops in metrics.jsonl. The checkpoint step is the resume_log entry's when
    there is one, else the first step after the drop minus one update (rollout x envs). The episodes index
    comes from the per-update "episodes" counts logged in metrics up to the drop."""
    per_update = cfg["hp"]["rollout"] * cfg["envs"]
    logged = [e["resumed_from_step"] for e in cfg.get("resume_log", []) if e.get("kind", "resume") == "resume"]
    out, n_eps = [], 0
    for i, r in enumerate(metrics):
        if i and r["step"] < metrics[i - 1]["step"]:
            guess = r["step"] - per_update
            c = next((x for x in logged if abs(x - guess) <= per_update), guess)
            out.append((i, n_eps, c))
        n_eps += r.get("episodes", 0) or 0
    return out


def load_kept(run_dir, name):
    """kept_rows() of run_dir/name ("metrics.jsonl" or "episodes.jsonl"). Logs with resume rows are used
    as they are; older logs get markers synthesized from metrics.jsonl (synth_markers)."""
    run_dir = Path(run_dir)
    rows = _read(run_dir / name)
    if any("resume" in r for r in rows):
        return kept_rows(rows)
    cfg = json.loads((run_dir / "config.json").read_text())
    metrics = [r for r in _read(run_dir / "metrics.jsonl") if "resume" not in r]
    key = "step" if name == "metrics.jsonl" else "global_step"
    marks = synth_markers(cfg, metrics)
    out, prev = [], 0
    for mi, ei, c in marks:
        cut = mi if name == "metrics.jsonl" else ei
        out += rows[prev:cut] + [{"resume": True, key: c}]
        prev = cut
    return kept_rows(out + rows[prev:], key)


def vm_event_after(runs_dir, t0):
    """The first recorded GCP stop event after time t0 (epoch seconds), or None."""
    p = Path(runs_dir) / VM_EVENTS
    if not p.exists():
        return None
    evs = []
    with open(p) as f:
        lines = f.readlines()
    for line in lines:
        try:
            e = json.loads(line)
            if _parse(e["time"]) >= t0:
                evs.append(e)
        except (ValueError, KeyError):
            continue
    return min(evs, key=lambda e: _parse(e["time"])) if evs else None


def stopped_at(last, stop, event):
    """(stopped_at_step, basis) from the last metrics row, stop.json and the VM event."""
    if stop and (not last or stop.get("time", 0) >= last.get("time", 0)):
        return stop["step"], "stop.json"
    if not last:
        return None, None
    if event and last.get("time"):
        dt = max(0.0, _parse(event["time"]) - last["time"])
        sps = last.get("sps_update") or last.get("sps") or 0
        return int(last["step"] + sps * dt), "metrics+vm_event"
    return last["step"], "metrics"


def resume_entry(run_dir, runs_dir, ckpt_step, ckpt_name, n, metrics_path=None, kind="resume"):
    run_dir = Path(run_dir)
    mpath = metrics_path or run_dir / "metrics.jsonl"
    last = last_metrics_row(mpath)
    stop_p = run_dir / "stop.json"
    stop = json.loads(stop_p.read_text()) if stop_p.exists() else None
    event = vm_event_after(runs_dir, last["time"]) if last and last.get("time") else None
    step, basis = stopped_at(last, stop, event)
    if stop_p.exists():
        stop_p.rename(run_dir / f"stop-{n}.json")      # one stop.json per segment
    return {"resume": n, "kind": kind, "resumed_at": _iso(time.time()),
            "resumed_from_ckpt": ckpt_name, "resumed_from_step": ckpt_step,
            "stopped_at_step": step, "replayed_steps": (step - ckpt_step) if step is not None else None,
            "basis": basis,
            "last_logged_step": last["step"] if last else None,
            "last_logged_time": _iso(last["time"]) if last and last.get("time") else None,
            "max_step_before": max_step(mpath),
            "stop": stop, "vm_event": event}


def write_stop(run_dir, step, reason):
    """Called from the trainer's finally: block, after the checkpoint is saved."""
    (Path(run_dir) / "stop.json").write_text(json.dumps({"step": step, "reason": reason, "time": time.time(),
                                                         "time_iso": _iso(time.time())}))
