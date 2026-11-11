#!/usr/bin/env python3
"""Batch-2 run-level report (item 7; experiment-e1.md §9 H4′, H9, H10, H11): dev only, descriptive.

  .venv/bin/python b2_runs_report.py [--out results/b2]     -> OUT/runs-report.{md,json}

Per run (from b2_reads.summary, which reads only the run's own DEV state; HA-trained runs have no own dev view):
  * training-episode P / T per mark (episodes ending since the previous mark; HA-trained: not summarised),
    beside the snapshot own-state eval P / T at the same mark
  * the resumes table (count, kind, from step, replayed steps; §9 H9 resume rule status)
  * C at the closing read (H4′: L, obs_version, reward hash, init = ckpt-0, sps vs 85 % of batch 1) and sps ratio;
    for HA-trained runs C is read from their dev D1/D2 rows
  * A2″ collapse flags (own state; rule: mean T ≤ best earlier T − 1)
  * the trip list (§9 H9: trips 1–4 and their rulings), and the H11 host check result
Nothing here is held-out, tested or used by the patch decision.
"""
import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import b2_reads as BR  # noqa: E402
import seal_b2  # noqa: E402

TRIPS = [
    ("1", "C (sps) at e1-a's ckpt-0 read (pilot b2-A0-D1-s0)", "2026-11-07 21:34 UTC",
     "r1, false alarm: the interim read starved the trainer; reads now freeze the trainer (code dbd7fa34)"),
    ("2", "A-1 on b2-A0-D1-s4 at 1M", "2026-11-08 01:11 UTC", "r1 after the H9 diagnostic (snapshot vs training gap)"),
    ("3", "A-2 on b2-A0-D1-s1 at 2.5M", "2026-11-08 01:57 UTC", "r1, ruled with trip 2"),
    ("4", "C (sps 0.80×) at b2-A2pp-D2-s1's closing read, e1-a", "2026-11-08 16:45 UTC",
     "r1 for validity; e1-a stop/start to a new host (Jake's OK), 17:08 UTC"),
]
RULED = {"b2-A2pp-D2-s1": "trip 4, ruled r1 for validity"}   # closing-read C trips already ruled (§9 H9)
MARKS = [250_880, 500_736, 1_000_448, 2_000_896, 3_000_320, 4_000_768, 5_000_192]
LAB = dict(zip(MARKS, ["0.25M", "0.5M", "1M", "2M", "3M", "4M", "5M"]))


def dev_c(run, cfg):
    """C for an HA-trained run, from its closing dev rows (D1/D2): L, obs_version, reward hash, init = ckpt-0."""
    rows = list(BR.jl(BR.runs_dir() / run / "eval" / "episodes.jsonl"))
    c = []
    if any(any(x["type"] == "L" for x in r.get("events", [])) for r in rows):
        c.append("L > 0 in dev rows")
    if cfg.get("obs_version") != BR.OBS_VERSION or any(r.get("obs_version") != BR.OBS_VERSION for r in rows):
        c.append("obs_version")
    if cfg.get("reward_hash") != BR.REWARD_HASH.get(cfg["arm"]):
        c.append("reward hash")
    if any(not str(r.get("init_sha256", "")).endswith("=ckpt-0") for r in rows):
        c.append("init hash")
    return c, len(rows)


def run_row(run, vm):
    s = BR.summary(run, vm) if (BR.runs_dir() / run / "config.json").exists() else None
    if s is None:
        return {"run": run, "vm": vm, "missing": True}
    cfg = json.loads((BR.runs_dir() / run / "config.json").read_text())
    marks = {m["step"]: m for m in s["marks"]}
    per = []
    for m in MARKS:
        x = marks.get(m, {})
        per.append({"mark": m, "eval_P": x.get("eval", {}).get("P"), "eval_T": x.get("eval", {}).get("T"),
                    "train_n": x.get("train", {}).get("episodes"), "train_P": x.get("train", {}).get("P"),
                    "train_T": x.get("train", {}).get("T"), "collapsed": x.get("eval", {}).get("collapsed")})
    c = s["C"]
    if s["state"] not in BR.DEV:
        reasons, n = dev_c(run, cfg)
        if s["sps_ratio"] is not None and s["sps_ratio"] < BR.SPS_MIN_FRAC:
            reasons.append("sps")
        c = {"status": "trip" if reasons else "pass", "reasons": reasons,
             "detail": f"from {n} dev D1/D2 rows (own state HA is sealed)"}
    return {"run": run, "vm": vm, "arm": s["arm"], "state": s["state"], "trained_to": s["trained_to"],
            "sps": s["sps_recent"], "sps_ratio": s["sps_ratio"], "C": c, "resumes": s["resumes"],
            "resume_rule": s["resume_rule"], "marks": per,
            "collapsed": [p["mark"] for p in per if p["collapsed"]]}


def fmt(x, k=1):
    return "–" if x is None else f"{x:.{k}f}"


def md(rows):
    L = ["# Batch-2 run report (item 7): dev only, descriptive, not a test", "",
         "Training-episode P / T are the episodes ending since the previous mark (§9 H10 unit); eval is the "
         "run's own dev state × 8 at the mark. HA-trained runs: no own dev view and no training-episode numbers "
         "(their training state is held-out).", "", "## Runs: C, sps, resumes, collapse", "",
         "| run | vm | trained to | sps (× batch 1) | C (closing) | resumes | A2″ collapse flags |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            L.append(f"| {r['run']} | {r['vm']} | not fetched | | | | |")
            continue
        L.append(f"| {r['run']} | {r['vm']} | {r['trained_to']:,} | {fmt(r['sps'], 0)} ({r['sps_ratio']}×) | "
                 f"{r['C']['status'].upper()}{': ' + '; '.join(r['C']['reasons']) if r['C']['reasons'] else ''}"
                 f"{' (' + RULED[r['run']] + ')' if r['run'] in RULED and r['C']['status'] == 'trip' else ''} | "
                 f"{len(r['resumes'])} ({r['resume_rule']['status']}) | "
                 f"{', '.join(LAB[m] for m in r['collapsed']) or ('none' if r['arm'] == 'A2pp' and r['state'] in BR.DEV else '–')} |")
    L += ["", "## Training-episode P / T vs snapshot eval P / T per mark (own state)", "",
          "| run | " + " | ".join(LAB[m] for m in MARKS) + " |", "|---|" + "---|" * len(MARKS)]
    for r in rows:
        if r.get("missing") or r["state"] not in BR.DEV:
            continue
        L.append(f"| {r['run']} | " + " | ".join(
            f"train {fmt(p['train_P'])} / {fmt(p['train_T'], 2)}; eval {fmt(p['eval_P'])} / {fmt(p['eval_T'], 2)}"
            for p in r["marks"]) + " |")
    res = [(r["run"], e) for r in rows if not r.get("missing") for e in r["resumes"]]
    L += ["", "## Resumes (§9 H9)", "",
          ("No batch-2 main run resumed." if not res else
           "| run | kind | from step | replayed | restored |\n|---|---|---|---|---|\n" + "\n".join(
               f"| {r} | {e.get('kind')} | {e.get('resumed_from_step')} | {e.get('replayed_steps')} | {e.get('restored')} |"
               for r, e in res)),
          "", "## Trips (§9 H9)", "", "| trip | what | when | ruling |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in TRIPS]
    L += ["", "C at every closing read: " + ", ".join(
              f"{sum(r['C']['status'] == s for r in rows if not r.get('missing'))} {s}" for s in ("pass", "trip", "pending")) +
          "; every trip is ruled (" + "; ".join(f"{k}: {v}" for k, v in RULED.items()) + ").",
          "", "## Host check (§9 H11)", "",
          "The three parked 5M A2″ endings (D1-s2, D1-s3, D2-s3) reproduce bit for bit off the original host; "
          "eval weights = final training weights for the 4 post-restart A2″ runs; old and new host identical in CPU "
          "model and features, kernel, packages, versions, threads. Details: `results/h11-checks.md`.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CODE / "results" / "b2"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [run_row(r, vm) for r, vm in seal_b2.batch2_runs()]
    (out / "runs-report.json").write_text(json.dumps({"runs": rows, "trips": TRIPS}, indent=1))
    (out / "runs-report.md").write_text(md(rows))
    print(f"{sum(not r.get('missing') for r in rows)}/{len(rows)} runs; wrote {out / 'runs-report.md'}")


if __name__ == "__main__":
    main()
