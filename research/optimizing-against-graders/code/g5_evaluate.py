#!/usr/bin/env python3
"""§9 G5 part 2, the 0-step mark (exploratory): evaluate each A0 / A2″ dev run's *initial* policy on its own
training state × 8 episodes (seeds 1000–1007, as evaluate.py), next to the 0.25M mark. Dev states only.

  .venv/bin/python g5_evaluate.py manifest              (laptop) recorded init hash + thread count per run
                                                        from the dev views -> ~/roms/g5/zero-manifest.json
  .venv/bin/python g5_evaluate.py zero [--workers 8]    (e1-a)   rebuild each init, hash-check, evaluate
                                                        -> ~/roms/g5/zero/{episodes.jsonl,excluded.json}
  .venv/bin/python g5_evaluate.py report                (laptop) -> results/g5/zero-step.{json,md}

Hash rule (§9 G5, lead-approved): the init is rebuilt with evaluate.init_policy at the run's recorded thread
count; its state_dict sha256 must equal the run's recorded init_sha256 (from the --kl-init evals), else
the run is left out and listed. The QR in the orthogonal init rounds differently by CPU (the laptop's
Intel/MKL never matched), so `zero` runs on e1-a (AMD EPYC, where the e1-a runs were built); e1-b's runs
were built on 4-thread EPYC and are rebuilt at 4 threads on e1-a, excluded if they don't match.
Each episode is evaluate.episode() unchanged (same env, P0, jitter, sampling). No U (its threshold is
per checkpoint over D1+D2; here each run has one state), so detector events are listed without U.
The file name contains "evaluate.py" so the VM idle guard (pgrep evaluate.py) sees the job as busy.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import rom  # noqa: E402

DEV_RUNS = [f"{a}-{s}-s{k}" for k in range(1, 6) for s in ("D1", "D2") for a in ("A0", "A2pp")]
HERE = Path(__file__).resolve().parent


def g5_dir():
    return rom.runs_dir().parent / "g5"


def manifest():
    out = []
    for run in DEV_RUNS:
        d = rom.runs_dir() / run
        cfg = json.loads((d / "config.json").read_text())
        rows = [json.loads(l) for l in open(d / "eval-dev" / "episodes.jsonl")]
        rec = {(r["init_sha256"], r["init_threads"]) for r in rows if r["state"] == cfg["state"] and "init_sha256" in r}
        if len(rec) != 1:
            sys.exit(f"{run}: {len(rec)} distinct recorded (init_sha256, threads) on its own state")
        (sha, th), = rec
        vm = {r.get("vm") for r in rows} - {None}
        out.append({"run": run, "state": cfg["state"], "init_sha256": sha.split()[0], "init_threads": th,
                    "vm": sorted(vm), "cfg": cfg})
    p = g5_dir() / "zero-manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}: {len(out)} runs")


def zero(workers, manifest_path=None, out=None, runs=None, episodes=8):
    import torch
    import evaluate as EV
    from pokered import detectors
    man = json.loads(Path(manifest_path or g5_dir() / "zero-manifest.json").read_text())
    man = [m for m in man if not runs or m["run"] in runs]
    out = Path(out) if out else g5_dir() / "zero"
    (out / "inits").mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)
    jobs, excluded, checked = [], [], []
    for m in man:
        cfg = dict(m["cfg"], train_threads=m["init_threads"])
        model, sha, n = EV.init_policy(cfg)
        sha = sha.split()[0]
        ok = sha == m["init_sha256"]
        checked.append({"run": m["run"], "threads": n, "rebuilt": sha, "recorded": m["init_sha256"], "match": ok})
        print(f"{m['run']}: rebuilt {sha[:8]} at {n} threads, recorded {m['init_sha256'][:8]}: "
              f"{'match' if ok else 'NO MATCH -> excluded'}", flush=True)
        if not ok:
            excluded.append(m["run"])
            continue
        ck = out / "inits" / m["run"] / "eval-0.pt"
        ck.parent.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict()}, ck)
        sd = out / "traces" / m["run"]
        sd.mkdir(exist_ok=True)
        for i in range(episodes):
            jobs.append((cfg, str(ck), m["state"], 1000 + i, None, str(sd), None, False))
    (out / "hash-check.json").write_text(json.dumps({"checked": checked, "excluded": excluded,
                                                       "host_cpu": _cpu(), "torch": torch.__version__}, indent=1))
    print(f"{len(jobs)} episodes, {workers} workers; excluded {excluded}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool, open(out / "episodes.jsonl", "w") as f:
        # imap keeps job order, so each row is paired with its run
        for job, (r, wr) in zip(jobs, pool.imap(EV.episode, jobs)):
            run = Path(job[1]).parent.name
            events = detectors.merge(detectors.firings_of(wr))       # no apply_u: U is not scored here
            r.update(run=run, arm=run.split("-")[0], step=0,
                     event_types=sorted({e["type"] + (f":{e['kind']}" if e["kind"] else "") for e in events}))
            f.write(json.dumps(r) + "\n")
            print(f"  {run} {r['state']} seed {r['seed']}: T {r['T']} P {r['P']} ({r['seconds']} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


def _cpu():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["manifest", "zero", "report"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--manifest", help="zero: manifest path (default ~/roms/g5/zero-manifest.json)")
    ap.add_argument("--out", help="zero: output dir (default ~/roms/g5/zero); for smoke tests")
    ap.add_argument("--runs", nargs="*", help="zero: only these runs (smoke tests)")
    ap.add_argument("--episodes", type=int, default=8, help="zero: episodes per run (default 8)")
    a = ap.parse_args()
    {"manifest": manifest, "zero": lambda: zero(a.workers, a.manifest, a.out, a.runs, a.episodes), "report": report}[a.cmd]()


def report():
    """Per arm × state: mean T, P (and by term, from the traces), leg progress at step 0, next to the same
    runs' 0.25M dev cells. Labelled exploratory; changes nothing pre-registered (H0 (i))."""
    import gzip
    z = g5_dir() / "zero"
    hc = json.loads((z / "hash-check.json").read_text())
    rows = [json.loads(l) for l in open(z / "episodes.jsonl")]
    for r in rows:
        tf = z / "traces" / r["run"] / f"eval-0-{r['state']}-{r['seed']}.jsonl.gz"
        terms = dict(explore=0.0, level=0.0, heal=0.0, seen=0.0)
        with gzip.open(tf, "rt") as f:
            f.readline()
            for l in f:
                t = json.loads(l)
                for k in terms:
                    terms[k] += t[f"r_{k}"]
        r["P_terms"] = {k: round(v, 4) for k, v in terms.items()}
    kept = sorted({r["run"] for r in rows})
    first = {}
    for run in kept:
        d = rom.runs_dir() / run
        st = json.loads((d / "config.json").read_text())["state"]
        first[run] = [json.loads(l) for l in open(d / "eval-dev" / "episodes.jsonl")]
        first[run] = [e for e in first[run] if e["state"] == st and e["step"] == 250880]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    cells = []
    for arm in ("A0", "A2pp"):
        for st in ("D1", "D2"):
            rs = [r for r in rows if r["arm"] == arm and r["state"] == st]
            q = [e for run in kept if run.startswith(f"{arm}-{st}-") for e in first[run]]
            cells.append({"arm": arm, "state": st, "runs": sorted({r["run"] for r in rs}), "episodes": len(rs),
                          "T_0": mean([r["T"] for r in rs]), "P_0": mean([r["P"] for r in rs]),
                          "legP_0": mean([r["leg_progress"] for r in rs]),
                          "P_0_terms": {k: mean([r["P_terms"][k] for r in rs]) for k in ("explore", "level", "heal", "seen")},
                          "T_025M": mean([e["T"] for e in q]), "P_025M": mean([e["P"] for e in q]),
                          "legP_025M": mean([e["leg_progress"] for e in q]), "episodes_025M": len(q)})
    res = {"label": "EXPLORATORY 0-step mark (§9 G5); does not change H0 (i)", "hash_check": hc,
           "cells": cells, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    od = HERE / "results" / "g5"
    od.mkdir(parents=True, exist_ok=True)
    (od / "zero-step.json").write_text(json.dumps(res, indent=1))
    (od / "zero-step-episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    L = ["# G5 part 2: 0-step mark (EXPLORATORY)", "",
         "Each A0 / A2″ dev run's rebuilt init policy, own training state × 8 episodes (seeds 1000–1007), "
         "next to the same runs' 0.25M dev cells. Dev states only. Exploratory: does not change H0 (i).", "",
         f"Init rebuilt on {hc['host_cpu']} (torch {hc['torch']}), hash-checked against each run's recorded "
         f"`init_sha256`: {sum(c['match'] for c in hc['checked'])}/{len(hc['checked'])} match. "
         f"Excluded (no match): {', '.join(hc['excluded']) or 'none'}.", "",
         "| arm | state | runs | eps | T @0 | T @0.25M | P @0 | P @0.25M | legP @0 | legP @0.25M | P @0 explore / level / heal |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        t = c["P_0_terms"]
        L.append(f"| {c['arm']} | {c['state']} | {len(c['runs'])} | {c['episodes']} | {c['T_0']} | {c['T_025M']} | "
                 f"{c['P_0']} | {c['P_025M']} | {c['legP_0']} | {c['legP_025M']} | {t['explore']} / {t['level']} / {t['heal']} |")
    L += ["", "Pooled means over the kept runs' episodes (8 per run); 0.25M = the same runs' `eval-dev/` rows "
          "at 250,880 on their own state. Per-episode rows: `zero-step-episodes.jsonl`; hash check: `zero-step.json`.",
          "", "A0 and A2″ runs with the same seed share one init (same seed, same ActorCritic build) and the eval seeds "
          "are the same, so their step-0 rows are identical by construction; the arms differ only from training on."]
    (od / "zero-step.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
