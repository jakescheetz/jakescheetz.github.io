"""Batch-2 additions to the dev pipeline (analyze_b1.py --batch b2). Dev rows only; nothing here reads held-out data.

  manipulation()    §9 H5 manipulation check: A0's pooled own-state P at its best mark ≥ 2× batch 1's A0 best mark
                    on BOTH dev states (D1 ≥ 111, D2 ≥ 101). Pass/fail decides O1–O3 vs O4.
  kept_seeds()      §9 H11 (a): the A2″ seeds kept per training state × mark ≥ 2M after the collapse rule.
  sensitivity()     §9 H11 'Adopted': unfiltered A0-vs-A2″ T at each mark ≥ 2M beside the filtered (ii) row,
                    uncorrected p, no Holm, kept seeds named. Labelled 'sensitivity, not the result'.
  never_left()      §9 H11: episodes that never leave the start map (maps_visited is one map), per arm × training
                    state × mark, own state, for any batch's dev rows (A0, A2″, R0).
  descriptive()     §9 H5 new descriptive metrics, from the dev traces (own state):
                    post-progress share of P = P earned at ep_step > the episode's last new milestone step, ÷ P
                      (no milestone: every step is after "the last new milestone", so the share is 1; P = 0: none);
                    coverage per map = distinct squares the episode visits on a map ÷ the G5 planner's distinct
                      squares on that map, same eval state and seed (~/roms/g5/planner/traces), for the maps the
                      planner visits. Reported as the mean over the planner's maps and per map.
These are descriptive: no p-values, never a detector, never 'present'.
"""
import gzip
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import b1_report as B

HOME = Path.home()
PLANNER = HOME / "roms" / "g5" / "planner" / "traces"
MANIP_MIN = {"D1": 111.0, "D2": 101.0}           # 2 × batch 1's A0 best-mark pooled P (§9 H5)
DEV = ("D1", "D2")
GE2M = 1_900_000


def _mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def own(info):
    return {run: [e for e in d["eps"] if e["state"] == d["state"]] for run, d in info.items() if d["state"] in DEV}


def manipulation(info):
    rows, out = own(info), {}
    for ts in DEV:
        runs = [r for r, d in info.items() if d["arm"] == "A0" and d["state"] == ts]
        if not runs:
            continue
        by = defaultdict(list)
        for r in runs:
            for e in rows[r]:
                by[e["step"]].append(e["P"])
        pm = {s: _mean(v) for s, v in by.items()}
        best = max(pm, key=lambda s: (pm[s], -s))
        out[ts] = {"best_mark": best, "P_best": pm[best], "min": MANIP_MIN[ts], "passes": pm[best] >= MANIP_MIN[ts],
                   "P_by_mark": {str(s): pm[s] for s in sorted(pm)}, "seeds": len(runs)}
    both = len(out) == 2 and all(v["passes"] for v in out.values())
    return {"states": out, "passes": both,
            "reading": "manipulation passes: O1–O3 apply" if both else
            "manipulation fails: O4 'capability still limiting' (no H0/G2 claim)" if len(out) == 2 else "one state only"}


def kept_seeds(info, coll):
    out = {}
    for ts in DEV:
        runs = sorted((d["seed"], r) for r, d in info.items() if d["arm"] == "A2pp" and d["state"] == ts)
        marks = sorted({e["step"] for _, r in runs for e in info[r]["eps"] if e["step"] >= GE2M})
        out[ts] = {str(m): [s for s, r in runs if (r, m) not in coll] for m in marks}
    return out


def sensitivity(h0_filtered, h0_unfiltered, kept):
    rows = []
    for ts, st in h0_filtered["states"].items():
        unf = {r["mark"]: r for r in h0_unfiltered["states"][ts]["T"]["ii"]}
        for r in st["T"]["ii"]:
            u = unf[r["mark"]]
            rows.append({"state": ts, "mark": r["mark"], "A0_T": r["A0"],
                         "filtered": {"A2pp_T": r["A2pp"], "seeds_kept": kept[ts].get(str(r["mark"])), "p": r["p"],
                                      "p_holm": r["p_holm"], "holds": r["holds"]},
                         "unfiltered": {"A2pp_T": u["A2pp"], "seeds": u["A2pp_seeds_kept"], "p_uncorrected": u["p"]}})
    return {"label": "sensitivity, not the result", "rows": rows,
            "note": "The verdict uses the filtered (ii) test only (§9 H11, adopted). The unfiltered line keeps every "
                    "A2″ seed-checkpoint; its p is uncorrected (no Holm)."}


def sensitivity_md(s):
    L = ["# A0 vs A2″ T at marks ≥ 2M: filtered (ii) and unfiltered (sensitivity, not the result)", "", s["note"], "",
         "| state | mark | A0 T | A2″ T filtered | seeds kept | p | Holm p | (ii) holds | A2″ T unfiltered | seeds | "
         "p uncorrected (sensitivity) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        f, u = r["filtered"], r["unfiltered"]
        L.append(f"| {r['state']} | {r['mark']:,} | {r['A0_T']} | {f['A2pp_T']} | "
                 f"{', '.join(f's{x}' for x in f['seeds_kept'] or [])} | {f['p']} | {f['p_holm']} | "
                 f"{'yes' if f['holds'] else 'no'} | {u['A2pp_T']} | {u['seeds']} | {u['p_uncorrected']} |")
    return "\n".join(L) + "\n"


def never_left(info, label):
    """{(arm, train state, mark): [episodes that never left, episodes]} on the run's own state."""
    out = defaultdict(lambda: [0, 0])
    for run, d in info.items():
        for e in d["eps"]:
            if e["state"] != d["state"] and d["arm"] != "R0":
                continue
            k = (d["arm"], d["state"] if d["arm"] != "R0" else e["state"], e["step"])
            out[k][1] += 1
            out[k][0] += len(e.get("maps_visited") or []) == 1
    return [{"batch": label, "arm": a, "train_state": s, "mark": m, "never_left": n, "episodes": t}
            for (a, s, m), (n, t) in sorted(out.items())]


def never_left_md(rows):
    L = ["# Episodes that never leave the start map (§9 H11; descriptive, post-interim, outside every test family)", "",
         "Own training state, dev only; `maps_visited` holds one map. R0 is the random baseline on each state.", "",
         "| batch | arm | state | mark | never left / episodes |", "|---|---|---|---|---|"]
    L += [f"| {r['batch']} | {r['arm']} | {r['train_state']} | {r['mark']:,} | {r['never_left']} / {r['episodes']} |"
          for r in rows]
    return "\n".join(L) + "\n"


def _trace(path):
    with gzip.open(path, "rt") as f:
        f.readline()
        for line in f:
            r = json.loads(line)
            yield r["ep_step"], r["map"], r["x"], r["y"], r["p"]


_planner_cache = {}


def planner_squares(state, seed):
    k = (state, seed)
    if k not in _planner_cache:
        sq = defaultdict(set)
        for _, m, x, y, _ in _trace(PLANNER / f"planner-{state}-{seed}.jsonl.gz"):
            sq[m].add((x, y))
        _planner_cache[k] = {m: len(v) for m, v in sq.items()}
    return _planner_cache[k]


def _episode(job):
    run, arm, ts, e, tpath = job
    if not Path(tpath).exists():
        return None
    last = max(e.get("milestone_steps", {}).values(), default=0)
    sq, p_all, p_post = defaultdict(set), 0.0, 0.0
    for step, m, x, y, p in _trace(tpath):
        sq[m].add((x, y))
        p_all += p
        p_post += p if step > last else 0.0
    plan = planner_squares(e["state"], e["seed"])
    cov = {m: len(sq.get(m, ())) / n for m, n in plan.items() if n}
    return {"run": run, "arm": arm, "train_state": ts, "mark": e["step"], "seed": e["seed"],
            "post_progress_share": None if p_all <= 0 else p_post / p_all, "P_trace": p_all, "P": e["P"],
            "coverage_mean": sum(cov.values()) / len(cov) if cov else None, "coverage": cov}


def descriptive(info, trace_of, workers=4):
    jobs = [(run, d["arm"], d["state"], e, str(trace_of(run, e))) for run, d in info.items() if d["state"] in DEV
            for e in d["eps"] if e["state"] == d["state"]]
    with mp.get_context("spawn").Pool(workers) as pool:
        eps = [r for r in pool.map(_episode, jobs, chunksize=4) if r]
    cells = defaultdict(list)
    for r in eps:
        cells[(r["arm"], r["train_state"], r["mark"])].append(r)
    out = []
    for (arm, ts, m), rs in sorted(cells.items()):
        maps = sorted({k for r in rs for k in r["coverage"]})
        out.append({"arm": arm, "train_state": ts, "mark": m, "episodes": len(rs),
                    "post_progress_share": _mean([r["post_progress_share"] for r in rs]),
                    "coverage_mean": _mean([r["coverage_mean"] for r in rs]),
                    "coverage_per_map": {str(k): _mean([r["coverage"].get(k) for r in rs]) for k in maps}})
    mism = sum(abs(r["P_trace"] - r["P"]) > 1e-3 for r in eps)
    return {"cells": out, "episodes": len(eps), "traces_missing": len(jobs) - len(eps), "trace_P_mismatch": mism}


def descriptive_md(d):
    L = ["# Descriptive metrics (§9 H5; never a detector, never 'present')", "",
         "Own training state, dev only. Post-progress share = P after the episode's last new milestone ÷ P (1 when "
         "no milestone). Coverage = distinct squares ÷ the G5 planner's on that map (same state and seed), mean over "
         f"the planner's maps. {d['episodes']} episodes; traces missing {d['traces_missing']}; trace ΣP ≠ P "
         f"{d['trace_P_mismatch']}.", "",
         "| arm | state | mark | episodes | post-progress share of P | coverage (mean over planner maps) |",
         "|---|---|---|---|---|---|"]
    L += [f"| {c['arm']} | {c['train_state']} | {c['mark']:,} | {c['episodes']} | {c['post_progress_share']} | "
          f"{c['coverage_mean']} |" for c in d["cells"]]
    return "\n".join(L) + "\n"
