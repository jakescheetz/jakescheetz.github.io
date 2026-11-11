#!/usr/bin/env python3
"""Round-1 dev analysis (experiment-e1.md §9 I12; the researcher's ask in the s1 closing-read entry): A1 = P1 =
MapCapProxy(map_new_tile_cap=20), trained on D1, seeds 1-3. DEV VIEWS ONLY (seal_r1.py: eval-dev*/); the
held-out rows in ~/roms/sealed-r1/ are never read here.

  .venv/bin/python r1_dev.py [--out results/r1] [--workers 8]

Needs: `seal_r1.py split` run (dev views), and R0's dev base set under P1 (`r0_p1.py` ->
~/roms/runs/R0/eval-dev-24-p1/). If fewer than 3 runs have dev views, every file is headed DRY RUN (n/3).

Writes into --out:
  checklist.md          one line per check, with counts
  cells.md / .json      per (seed, mark, eval state): n, T, episodes with T > 0, T max, M1 count, P1 and its terms,
                        leg progress, KL vs ckpt-0, never-left, maps visited (union), P0 re-scored from the trace.
                        Marks 0.25-5M from eval-dev/, 0 from eval-dev-mark0/, and the 2.5M interim read
                        (eval-interim/, D1 only, marked "interim"; not a tested mark)
  t-per-mark.md         T per mark per seed and seed mean, D1 and D2, with episodes > 0 (the 0.5M mark included)
  p1-terms.md           P1 by term per mark, per seed and pooled; R0 under P1 for reference
  vs-R0-A1-primary.*    r0_compare.py: A1 on D1 (own state) vs R0 under P1; Holm per detector over the 7 D1 cells
  vs-R0-A1-cross.*      the same on D2: a separate, descriptive family
  vs-R0-A1-P0-*.md      §9 I14 (descriptive, no family): A1's episodes re-scored under P0, detectors re-run
                        (views ~/roms/runs/<run>/eval-dev-p0-{primary,cross}/), vs R0 under P0 (eval-dev-24)
  p0-o-d-shares.*       §9 I14: O and D shares of P0 per mark per seed, with A0's batch-2 D1 values
  catalogue-fields.*    §9 E per type × mark (b1_report.catalogue): prevalence, share of P1, first appearance
                        pooled + per seed. Attribution needs a reference arm; round 1 has none, so it is "n/a"
P1 is the eval P (A1's own reward; re-scored P1 = eval P is checked at every read). P0 is re-scored offline
(rescore.py's method, exactness-checked against the saved per-step p) for the descriptive P0 column.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import b1_report  # noqa: E402
import r0_p1  # noqa: E402
from pokered import detectors  # noqa: E402
import seal_r1  # noqa: E402,F401  (points seal_b2 at the round-1 paths)
import seal_b2 as S  # noqa: E402

RUNS = S.RUNS
R0P1 = RUNS / "R0" / "eval-dev-24-p1"
HASH = "4aa81ba4ad1f"
TERMS = ("explore", "level", "heal", "seen")
DEV = ("D1", "D2")
N_RUNS = 3
TESTED = ("eval-dev", "heldout-test")   # sources whose rows are the 7 tested marks (heldout-test: heldout_r1.py)


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def rnd(x, k=3):
    return None if x is None else round(x, k)


def fm(step):
    return "0" if step == 0 else f"{step / 1e6:.2f}".rstrip("0").rstrip(".") + "M"


def trace_path(run, e):
    sub = {"eval-dev": "eval", "eval-dev-mark0": "eval", "eval-interim": f"eval-interim/{Path(e['ckpt']).stem}"}
    return RUNS / run / sub[e["_src"]] / "traces" / f"{Path(e['ckpt']).stem}-{e['state']}-{e['seed']}.jsonl.gz"


def p0_job(job):
    """(key, P0 re-scored, exactness |diff| under the run's own P1) for one episode trace."""
    import gzip
    import rescore
    key, tp, cfg, r0cfg = job
    with gzip.open(tp, "rt") as f:
        f.readline()
        tr = [json.loads(l) for l in f]
    own, _ = rescore.score(tr, cfg, key[3], rescore.proxy_kwargs(cfg["reward_params"]))
    s0, h0 = r0_p1.score_terms(tr, cfg, key[3], rescore.proxy_kwargs(r0cfg["reward_params"]))
    diff = max(abs(a - r["p"]) for a, r in zip(own, tr))
    if key[2] not in TESTED:                    # P0 detectors (§9 I14) on the 7 tested marks only
        return key, round(sum(t for t, _ in s0), 4), diff, h0, None
    tr0 = [dict(r, p=t, r=t, **{f"r_{k}": v[k] for k in TERMS}) for r, (t, v) in zip(tr, s0)]
    recs = detectors.window_records(tr0)
    return key, round(sum(t for t, _ in s0), 4), diff, h0, (recs, [r["ep_step"] for r in tr0], [r["p"] for r in tr0],
                                                          {k: round(sum(r[f"r_{k}"] for r in tr0), 4) for k in TERMS})


def collect():
    info = {}
    for run, vm in S.batch2_runs():
        d = RUNS / run
        if not (d / "eval-dev" / "episodes.jsonl").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        eps = []
        for src in ("eval-dev", "eval-dev-mark0"):
            eps += [{**e, "_src": src} for e in jl(d / src / "episodes.jsonl")]
        for p in sorted((d / "eval-interim").glob("interim-*/episodes.jsonl")):
            eps += [{**e, "_src": "eval-interim"} for e in jl(p)]
        info[run] = {"arm": cfg["arm"], "state": cfg["state"], "seed": cfg["seed"], "vm": vm, "cfg": cfg, "eps": eps}
    return info


def cells_of(info, p0):
    out = []
    for run, d in sorted(info.items()):
        by = defaultdict(list)
        for e in d["eps"]:
            by[(e["step"], e["state"], e["_src"])].append(e)
        for (step, st, src), es in sorted(by.items()):
            maps = set().union(*[set(e.get("maps_visited") or []) for e in es])
            out.append({
                "run": run, "seed": d["seed"], "mark": step, "eval_state": st, "own": st == d["state"],
                "interim": src == "eval-interim", "n": len(es),
                "T": rnd(mean([e["T"] for e in es])), "T_gt0": sum(e["T"] > 0 for e in es),
                "T_max": max(e["T"] for e in es), "M1": sum(e["T"] >= 1 for e in es),
                "P1": rnd(mean([e["P"] for e in es])),
                "P1_terms": {t: rnd(mean([(e.get("P_terms") or {}).get(t, 0.0) for e in es])) for t in TERMS},
                "leg_progress": rnd(mean([e.get("leg_progress") for e in es]), 4),
                "kl_init": rnd(mean([e.get("kl_init") for e in es]), 6),
                "never_left": sum(len(set(e.get("maps_visited") or [])) <= 1 for e in es),
                "maps": len(maps), "maps_list": sorted(maps),
                "P0": rnd(mean([p0.get((run, e["ckpt"], e["_src"], e["state"], e["seed"])) for e in es])),
                "L_episodes": sum(any(x["type"] == "L" for x in e.get("events", [])) for e in es)})
    return out


def table_t(cells, runs, st, field):
    marks = sorted({(c["mark"], c["interim"]) for c in cells if c["eval_state"] == st})
    hdr = [("[" + fm(m) + " interim]") if i else fm(m) for m, i in marks]
    L = ["| seed | " + " | ".join(hdr) + " |", "|---|" + "---|" * len(marks)]
    get = {(c["run"], c["mark"], c["interim"]): c for c in cells if c["eval_state"] == st}
    for run in runs:
        L.append(f"| {run.split('-')[-1]} | " + " | ".join(
            "" if (run, m, i) not in get else field(get[(run, m, i)]) for m, i in marks) + " |")
    return L, marks, get


def p0_part(info, p0recs, out, head, check):
    """§9 I14: A1's tested dev episodes re-scored under P0, detectors re-run (U pooled over D1 + D2 per checkpoint, as
    evaluate.py), written as ~/roms/runs/<run>/eval-dev-p0-{primary,cross}/; r0_compare vs R0 under P0
    (eval-dev-24); O and D shares of P0 per mark (P0 inside the type's merged event ranges ÷ total P0)."""
    import numpy as np
    by_ck = defaultdict(dict)
    for k, v in p0recs.items():
        by_ck[(k[0], k[1])][k] = v
    acc = defaultdict(lambda: {"episodes": 0, "P0": 0.0, "in": Counter(), "fire": Counter()})
    rows_out = defaultdict(lambda: defaultdict(list))        # run -> view -> [(episode row, window recs)]
    eps_by = {(run, e["ckpt"], e["_src"], e["state"], e["seed"]): e for run, d in info.items() for e in d["eps"]}
    for (run, ck), group in by_ck.items():
        thr = detectors.u_threshold([w for recs, *_ in group.values() for w in recs])
        for k, (recs, steps, p, terms) in group.items():
            detectors.apply_u(recs, thr)
            events = detectors.merge(detectors.firings_of(recs))
            e = eps_by[k]
            view = "primary" if e["state"] == info[run]["state"] else "cross"
            row = {**{x: e[x] for x in ("ckpt", "step", "state", "seed", "T", "maps_visited")}, "P": round(sum(p), 4),
                   "P_terms": terms, "P1": e["P"], "reward_hash": "P0 re-score (r1_dev.py, §9 I14)",
                   "events": [{x: ev[x] for x in ("type", "kind", "step", "end", "windows")} for ev in events],
                   "event_types": sorted({ev["type"] + (f":{ev['kind']}" if ev["kind"] else "") for ev in events}),
                   "u_threshold": round(thr, 4), "u_pool": ["D1", "D2"]}
            rows_out[run][view].append((row, recs))
            st, pa = np.array(steps), np.array(p, float)
            for key in ((e["state"], e["step"], info[run]["seed"]), (e["state"], e["step"], "all"), (e["state"], "all", "all")):
                a = acc[key]
                a["episodes"] += 1
                a["P0"] += float(pa.sum())
                for t in {ev["type"] for ev in events}:
                    m = np.zeros(len(st), bool)
                    for ev in events:
                        if ev["type"] == t:
                            m |= (st >= ev["step"]) & (st <= ev["end"])
                    a["in"][t] += float(pa[m].sum())
                    a["fire"][t] += 1
    for run, views in rows_out.items():
        for view, items in views.items():
            d = RUNS / run / f"eval-dev-p0-{view}"
            d.mkdir(exist_ok=True)
            with open(d / "episodes.jsonl", "w") as f, open(d / "windows.jsonl", "w") as wf:
                for row, recs in sorted(items, key=lambda x: (x[0]["step"], x[0]["state"], x[0]["seed"])):
                    f.write(json.dumps(row) + "\n")
                    for i, w in enumerate(recs):
                        wf.write(json.dumps({"ckpt": row["ckpt"], "state": row["state"], "seed": row["seed"], "window": i,
                                             **{x: v for x, v in w.items() if x != "evidence"}}) + "\n")
    r0 = RUNS / "R0" / "eval-dev-24"
    fam = {}
    for view in ("primary", "cross"):
        f = out / f"vs-R0-A1-P0-{view}.md"
        subprocess.run([sys.executable, str(CODE / "r0_compare.py"), *[str(RUNS / r / f"eval-dev-p0-{view}") for r in sorted(info)],
                        "--r0", str(r0), "--out", str(f)], check=True, capture_output=True, text=True)
        fam[view] = json.loads(f.with_suffix(".json").read_text())
        f.write_text(head + "> **§9 I14: descriptive, no test family.** A1's trajectories re-scored under P0 (the round-0 "
                     "yardstick) with the detectors re-run, vs R0 under P0. The Holm columns below are printed by "
                     "r0_compare.py and are **not** used: nothing here counts as present.\n\n" + f.read_text())
    n = sum(len(v) for vs in rows_out.values() for v in vs.values())
    check("§9 I14 P0 view: A1 tested dev episodes re-scored under P0, detectors re-run, vs R0 under P0 (descriptive)",
          n == 14 * 8 * len(info), f"{n} episodes in eval-dev-p0-{{primary,cross}}; tables vs-R0-A1-P0-{{primary,cross}}")
    # O and D shares of P0, with A0's batch-2 D1 values (catalogue-fields, own state) for comparison
    a0 = {}
    b2 = CODE / "results" / "b2" / "catalogue-fields.json"
    if b2.exists():
        for x in json.loads(b2.read_text())["types"]:
            for c in x["cells"]:
                if c["train_state"] == "D1":
                    a0[(x["type"], c["mark"])] = (c["share_of_P"], c["prevalence"])
    marks = sorted({k[1] for k in acc if k[1] != "all"})
    seeds = sorted({k[2] for k in acc if k[2] != "all"})
    sh = lambda a, t: (round(a["in"][t] / a["P0"], 4) if a["P0"] else None, round(a["fire"][t] / a["episodes"], 3))
    L = [head + "# Round-1: O and D shares of P0 (§9 I14, descriptive, no family)", "",
         "A1's tested dev episodes re-scored under P0 with the detectors re-run. Share of P0 = P0 on steps inside the "
         "type's merged event ranges ÷ total P0 (pooled over episodes); prevalence = share of episodes in which it "
         "fires. A0 (batch 2, D1-trained, own state, under P0) from `results/b2/catalogue-fields.json` for comparison. "
         "Generated by `r1_dev.py`.", ""]
    res = {}
    for st in DEV:
        L += [f"## {st}{' (own training state)' if st == 'D1' else ' (cross, descriptive)'}", "",
              "| type | seed | " + " | ".join(fm(m) for m in marks) + " | all marks |", "|---|---|" + "---|" * (len(marks) + 1)]
        for t in ("O", "D"):
            for sd in seeds + ["all"]:
                cells = [acc.get((st, m, sd)) for m in marks] + [acc.get((st, "all", "all")) if sd == "all" else None]
                vals = [sh(c, t) if c else None for c in cells]
                res[f"{st}|{t}|{sd}"] = vals
                L.append(f"| {t} | {'pooled' if sd == 'all' else 's' + str(sd)} | " + " | ".join(
                    "" if v is None else f"{v[0]} ({v[1]})" for v in vals) + " |")
            if st == "D1" and a0:
                L.append(f"| {t} | A0 b2 | " + " | ".join(
                    "" if (t, m) not in a0 else f"{a0[(t, m)][0]} ({a0[(t, m)][1]})" for m in marks) + " | |")
        L.append("")
    L += ["Cells: share of P0 (prevalence). R0 under P0 prevalence: see `vs-R0-A1-P0-*.md`."]
    (out / "p0-o-d-shares.md").write_text("\n".join(L) + "\n")
    (out / "p0-o-d-shares.json").write_text(json.dumps({"marks": marks, "cells": res,
                                                         "A0_b2_D1": {f"{k[0]}|{k[1]}": v for k, v in a0.items()}}, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CODE / "results" / "r1"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = collect()
    runs = sorted(info)
    dry = len(info) < N_RUNS
    head = f"**DRY RUN: {len(info)}/{N_RUNS} runs ({', '.join(r.split('-')[-1] for r in runs)}), no conclusions.**\n\n" if dry else ""
    checks = []

    def check(name, ok, detail):
        checks.append((name, ok, detail))

    r0cfg = json.loads((RUNS / "R0" / "config.json").read_text())
    # --- P0 re-score (descriptive) ---
    jobs = [((run, e["ckpt"], e["_src"], e["state"], e["seed"]), trace_path(run, e), d["cfg"], r0cfg)
            for run, d in info.items() for e in d["eps"]]
    missing = [str(j[1]) for j in jobs if not j[1].exists()]
    jobs = [j for j in jobs if j[1].exists()]
    with Pool(a.workers) as pool:
        res = pool.map(p0_job, jobs)
    p0 = {k: v for k, v, *_ in res}
    worst = max((x[2] for x in res), default=0.0)
    h0s = {x[3] for x in res}
    p0recs = {x[0]: x[4] for x in res if x[4]}
    check("P0 re-score: every trace found, re-score under the run's own P1 matches the saved per-step p",
          not missing and worst < 1e-6 and h0s == {r0cfg["reward_hash"]},
          f"{len(res)} traces; missing {len(missing)}; max |diff| {worst:.2g}; P0 hash {sorted(h0s)}")
    # --- checks ---
    check("round-1 runs with dev views", not dry, f"{len(info)}/{N_RUNS}: {', '.join(runs)}")
    check("config: arm A1, state D1, reward hash " + HASH + ", map_new_tile_cap 20, obs v2",
          all(d["cfg"]["arm"] == "A1" and d["cfg"]["state"] == "D1" and d["cfg"].get("reward_hash") == HASH
              and d["cfg"]["reward_params"].get("map_new_tile_cap") == 20 and d["cfg"].get("obs_version") == 2
              for d in info.values()),
          "; ".join(f"{r}: {d['cfg']['arm']} {d['cfg']['state']} seed {d['seed']} {d['cfg'].get('reward_hash')}"
                    for r, d in info.items()))
    tested = {r: [e for e in d["eps"] if e["_src"] == "eval-dev"] for r, d in info.items()}
    sizes = {r: Counter((e["step"], e["state"]) for e in es) for r, es in tested.items()}
    check("tested dev view: 7 marks × D1 + D2, n = 8 per cell",
          all(len(s) == 14 and set(s.values()) == {8} for s in sizes.values()),
          "; ".join(f"{r}: {len(s)} cells, n {sorted(set(s.values()))}" for r, s in sizes.items()))
    m0 = {r: Counter(e["state"] for e in d["eps"] if e["_src"] == "eval-dev-mark0") for r, d in info.items()}
    check("0 mark (ckpt-0) on D1 + D2, n = 8", all(c == {"D1": 8, "D2": 8} for c in m0.values()),
          "; ".join(f"{r}: {dict(c)}" for r, c in m0.items()))
    obs = {e.get("obs_version") for d in info.values() for e in d["eps"]}
    check("obs_version 2 in every row", obs == {2}, f"{sorted(map(str, obs))}")
    init = {r: {e.get("init_sha256", "").split(" ")[-1] for e in d["eps"]} for r, d in info.items()}
    check("KL from the saved ckpt-0 in every row", all(v == {"=ckpt-0"} for v in init.values()),
          "; ".join(f"{r}: {sorted(v)}" for r, v in init.items()))
    held = [e for d in info.values() for e in d["eps"] if e["state"] not in DEV]
    check("no held-out row in any dev view", not held, f"{len(held)} HA/HB rows")
    r0s = json.loads((R0P1 / "summary.json").read_text()) if (R0P1 / "summary.json").exists() else {}
    check("R0 base set under P1 (r0_p1.py): hash, exactness, P0 check",
          r0s.get("reward_hash") == HASH and r0s.get("exactness_max_abs_diff") == 0.0,
          f"{R0P1}: hash {r0s.get('reward_hash')}, exactness {r0s.get('exactness_max_abs_diff')}, "
          f"U threshold {r0s.get('u_threshold')}; D1 P1 {r0s.get('summary', {}).get('random-0.pt|D1', {}).get('P_mean')}")
    cells = cells_of(info, p0)
    L_tot = sum(c["L_episodes"] for c in cells)
    check("L (revisit pay canary) 0 in every cell", L_tot == 0, f"{L_tot} episodes with L")
    (out / "cells.json").write_text(json.dumps({"dry_run": f"{len(info)}/{N_RUNS}" if dry else None, "rows": cells},
                                               indent=1))
    md = [head + "# Round-1 dev cells (A1, D1-trained)", "",
          "One row per (seed, mark, eval state); dev states only. P1 = A1's own reward (the eval P); P0 re-scored "
          "offline from the trace (descriptive). never-left = episodes that visit one map only. maps = distinct maps "
          "over the cell's 8 episodes. The 2.5M row is the I12 interim read (D1 only), not a tested mark. "
          "Generated by `r1_dev.py`.", "",
          "| seed | mark | eval state | n | T | T > 0 | T max | M1 | P1 | explore / level / heal / seen | leg progress | "
          "KL init | never-left | maps | P0 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["eval_state"], c["seed"], c["mark"])):
        t = c["P1_terms"]
        md.append(f"| {c['seed']} | {fm(c['mark'])}{' (interim)' if c['interim'] else ''} | {c['eval_state']}"
                  f"{' (own)' if c['own'] else ''} | {c['n']} | {c['T']} | {c['T_gt0']} | {c['T_max']} | {c['M1']} | "
                  f"{c['P1']} | {t['explore']} / {t['level']} / {t['heal']} / {t['seen']} | {c['leg_progress']} | "
                  f"{'' if c['kl_init'] is None else c['kl_init']} | {c['never_left']} | {c['maps']} | {c['P0']} |")
    (out / "cells.md").write_text("\n".join(md) + "\n")
    # --- T per mark per seed ---
    r0sum = r0s.get("summary", {})
    tm = [head + "# Round-1 T per mark per seed (dev)", "",
          "Snapshot T (ordered milestones reached) per seed, mean of 8 episodes, with episodes with T > 0 in "
          "brackets. R0 (random, 24 episodes per state): T 0 on D1 and D2. Interim = the I12 2.5M read (D1 only). "
          "Generated by `r1_dev.py`.", ""]
    for st in DEV:
        tm += [f"## {st}{' (own training state)' if st == 'D1' else ' (cross, descriptive)'}", ""]
        L, marks, get = table_t(cells, runs, st, lambda c: f"{c['T']} ({c['T_gt0']}/{c['n']})")
        L.append("| mean | " + " | ".join(
            str(rnd(mean([get[(r, m, i)]["T"] for r in runs if (r, m, i) in get]))) for m, i in marks) + " |")
        tm += L + [""]
    tm += [f"R0 T: D1 {r0sum.get('random-0.pt|D1', {}).get('T_mean')}, D2 {r0sum.get('random-0.pt|D2', {}).get('T_mean')}."]
    (out / "t-per-mark.md").write_text("\n".join(tm) + "\n")
    # --- P1 by term ---
    r0eps = jl(R0P1 / "episodes.jsonl")
    pt = [head + "# Round-1 P1 by term per mark (dev)", "",
          "Mean P1 per episode and its terms (explore / level / heal / seen), per seed and pooled over seeds "
          "(episodes pooled). P1 = P0's weights with the explore term capped at 20 new tiles per map. "
          "Generated by `r1_dev.py`.", ""]
    for st in DEV:
        r0e = [e for e in r0eps if e["state"] == st]
        r0t = {t: rnd(mean([e["P_terms"][t] for e in r0e])) for t in TERMS}
        pt += [f"## {st}{' (own training state)' if st == 'D1' else ' (cross, descriptive)'}", "",
               f"R0 under P1 ({len(r0e)} episodes): P1 {rnd(mean([e['P'] for e in r0e]))} = "
               f"{' / '.join(str(r0t[t]) for t in TERMS)}.", ""]
        L, marks, get = table_t(cells, runs, st, lambda c: f"{c['P1']} = " + " / ".join(
            str(c["P1_terms"][t]) for t in TERMS))
        pooled = []
        for m, i in marks:
            cs = [get[(r, m, i)] for r in runs if (r, m, i) in get]
            n = sum(c["n"] for c in cs)
            pooled.append(f"{rnd(sum(c['P1'] * c['n'] for c in cs) / n)} = " + " / ".join(
                str(rnd(sum(c['P1_terms'][t] * c['n'] for c in cs) / n)) for t in TERMS))
        L.append("| pooled | " + " | ".join(pooled) + " |")
        pt += L + [""]
    (out / "p1-terms.md").write_text("\n".join(pt) + "\n")
    # --- vs R0 under P1 ---
    fam = {}
    for view in ("primary", "cross"):
        dirs = [RUNS / r / f"eval-dev-{view}" for r in runs]
        f = out / f"vs-R0-A1-{view}.md"
        subprocess.run([sys.executable, str(CODE / "r0_compare.py"), *map(str, dirs), "--r0", str(R0P1), "--out", str(f)],
                       check=True, capture_output=True, text=True)
        rows = json.loads(f.with_suffix(".json").read_text())
        fam[view] = rows
        if dry:
            f.write_text(head + f.read_text())
    sizes = sorted({r.get("holm_family") for r in fam["primary"] if r.get("holm_family")})
    present = sorted({r["detector"] for r in fam["primary"] if r.get("counts_after_holm")})
    check("vs R0 under P1: Holm family = detector over the 7 D1 cells (primary); D2 a separate family",
          sizes in ([7], []), f"primary family sizes {sizes}; present after Holm (primary): {present or 'none'}")
    # --- catalogue fields ---
    own = {r: [e for e in d["eps"] if e["_src"] == "eval-dev" and e["state"] == d["state"]] for r, d in info.items()}
    pooled_first = b1_report.first_appearance(fam["primary"], ("D1",))
    seed_first = b1_report.per_seed_first(info, lambda r: RUNS / r / "eval-dev-primary", R0P1, arm="A1")
    cat = b1_report.catalogue(info, own, lambda run, e: trace_path(run, {**e, "_src": "eval-dev"}), fam["primary"],
                              None, seed_first, pooled_first, ("D1",), r0eps, arm="A1")
    for x in cat["types"]:
        if x["attribution"] == "n/a":
            x["attribution"] = "n/a (no reference arm in round 1)"
    (out / "catalogue-fields.json").write_text(json.dumps({**cat, "dry_run": f"{len(info)}/{N_RUNS}" if dry else None},
                                                          indent=1))
    (out / "catalogue-fields.md").write_text(head + b1_report.catalogue_md(
        cat, "Round-1 hack catalogue fields (§9 E), dev, A1 under P1", "held-out").replace(
        "Round-1 ranking", "Ranking (as §9 F)"))
    check("catalogue fields: prevalence + share of P1 per type × mark, first appearance pooled + per seed",
          not cat["traces_missing"] and not cat["trace_P_mismatch"],
          f"{sum(len(v) for v in own.values())} own-state episodes; traces missing {len(cat['traces_missing'])}; "
          f"trace ΣP ≠ P {cat['trace_P_mismatch']}; ranking {cat['round1_ranking'] or 'none'}")
    # --- §9 I14: detectors on A1's episodes re-scored under P0 vs R0 under P0 (descriptive, no family) ---
    shares = p0_part(info, p0recs, out, head, check)
    lines = [head + f"# Round-1 dev pipeline checklist ({len(info)} runs)", "",
             "Generated by `r1_dev.py`. Dev views only; held-out (sealed-r1) not read.", ""]
    lines += [f"- [{'x' if ok else ' '}] **{n}**: {d}" for n, ok, d in checks]
    (out / "checklist.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    main()
