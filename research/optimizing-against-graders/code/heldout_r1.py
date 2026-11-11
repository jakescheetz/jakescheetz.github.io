#!/usr/bin/env python3
"""Round-1 held-out analysis (experiment-e1.md §9 I16; run after the stop decision was logged and sealed-r1/ was
verified, 2026-11-11 13:16 UTC). A1 = P1 = MapCapProxy(map_new_tile_cap=20), D1-trained, seeds 1-3, on HA and HB.

  .venv/bin/python heldout_r1.py [--out results/r1-heldout] [--workers 8]

Needs: `seal_r1.py split` (tested views ~/roms/sealed-r1/<run>/eval-heldout-test/) and R0's held-out base set under
P1 (`r0_p1.py --src ~/roms/runs/R0/eval-24 --states HA HB --check-pool D1 HA HB
--windows ~/roms/sealed-b1/R0/eval-24/windows.jsonl --out-dir ~/roms/runs/R0/eval-heldout-24-p1`).

I16, fixed before sealed-r1/ was opened (same rules as batch 2's held-out, §9 (g), I13):
  vs-R0-A1-heldout.*      A1 on HA + HB vs R0 under P1; Holm per detector over the 14 cells (HA, HB × 7 marks),
                          except O: its family is the 7 HA cells, O on HB descriptive only (r0_compare.DESCRIPTIVE)
  vs-R0-A1-P0-heldout.md  §9 I14 view: A1's HA/HB episodes re-scored under P0, detectors re-run (U pooled over HA + HB
                          per checkpoint, as evaluate.py ran them; views in ~/roms/r1-heldout-p0/<run>/eval-heldout-p0/),
                          vs R0 under P0 (sealed-b1/R0/eval-24, the batch-2 held-out reference). Descriptive, no family
  p0-o-d-shares.*         O and D shares of P0 per mark per seed, HA (O's headline) and HB (descriptive), with A0
                          batch-2 D1-trained on HA (results/b2-heldout/o-share-ha.md) and R0 on HA from its traces
  t-per-mark.md           T per mark per seed, HA and HB apart (HB starts at M1: not comparable with D1 or HA)
  p1-terms.md             P1 by term per mark, per seed and pooled; R0 under P1 beside it
  cells.*, catalogue-heldout.*, checklist.md
Attribution n/a (no reference arm). Outputs hold counts and means only; ROM-derived rows stay in ~/roms.
"""
import argparse
import gzip
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import b1_report  # noqa: E402
import r1_dev as D  # noqa: E402  (imports seal_r1, which points seal_b2 at the round-1 paths)
import seal_b2 as S  # noqa: E402
from pokered import detectors  # noqa: E402

RUNS, SR1 = S.RUNS, S.SEALED_RAW
R0P1H = RUNS / "R0" / "eval-heldout-24-p1"
R0H = Path.home() / "roms" / "sealed-b1" / "R0" / "eval-24"          # R0 held-out under P0 (batch-2 reference)
R0TR = RUNS / "R0" / "eval-24" / "traces"
P0VIEW = Path.home() / "roms" / "r1-heldout-p0"
HELD = ("HA", "HB")
TERMS = D.TERMS
NOTE = {"HA": " (D1's start, other starter)", "HB": " (starts at M1, Viridian Center door: T not comparable)"}
fm, rnd, mean, jl = D.fm, D.rnd, D.mean, D.jl


def trace_path(run, e):
    return SR1 / run / "eval-heldout" / "traces" / f"{Path(e['ckpt']).stem}-{e['state']}-{e['seed']}.jsonl.gz"


def collect():
    info = {}
    for run, vm in S.batch2_runs():
        h = SR1 / run / "eval-heldout-test" / "episodes.jsonl"
        if not h.exists():
            continue
        cfg = json.loads((RUNS / run / "config.json").read_text())
        eps = [{**e, "_src": "heldout-test"} for e in jl(h)]
        eps += [{**e, "_src": "heldout-mark0"} for e in jl(SR1 / run / "eval-heldout" / "episodes.jsonl")
                if e["ckpt"] == "ckpt-0.pt"]
        info[run] = {"arm": cfg["arm"], "state": cfg["state"], "seed": cfg["seed"], "vm": vm, "cfg": cfg, "eps": eps}
    return info


def a0_ha():
    """A0 batch-2 D1-trained on HA, O share of P0 per mark, and R0 HA, from results/b2-heldout/o-share-ha.md."""
    out, f = {}, CODE / "results" / "b2-heldout" / "o-share-ha.md"
    for line in (f.read_text().splitlines() if f.exists() else []):
        m = re.match(r"\| A0 \('D1', (\d+)\) \| \d+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def r0_o_share(state):
    """R0 under P0 on `state`: (O share of P0, O prevalence, D share, D prevalence) from its traces + saved events."""
    tot, ins, fire, n = 0.0, Counter(), Counter(), 0
    for e in jl(R0H / "episodes.jsonl"):
        if e["state"] != state:
            continue
        t, i = b1_report.p_inside(R0TR / f"random-0-{state}-{e['seed']}.jsonl.gz", e["events"])
        tot, n = tot + t, n + 1
        for k in ("O", "D"):
            ins[k] += i.get(k, 0.0)
            fire[k] += any(x["type"] == k for x in e["events"])
    return {k: (round(ins[k] / tot, 4), round(fire[k] / n, 3)) for k in ("O", "D")}, n


def p0_part(info, p0recs, out, check):
    import numpy as np
    by_ck = defaultdict(dict)
    for k, v in p0recs.items():
        by_ck[(k[0], k[1])][k] = v
    acc = defaultdict(lambda: {"episodes": 0, "P0": 0.0, "in": Counter(), "fire": Counter()})
    rows_out = defaultdict(list)
    eps_by = {(run, e["ckpt"], e["_src"], e["state"], e["seed"]): e for run, d in info.items() for e in d["eps"]}
    for (run, ck), group in by_ck.items():
        thr = detectors.u_threshold([w for recs, *_ in group.values() for w in recs])
        for k, (recs, steps, p, terms) in group.items():
            detectors.apply_u(recs, thr)
            events = detectors.merge(detectors.firings_of(recs))
            e = eps_by[k]
            row = {**{x: e[x] for x in ("ckpt", "step", "state", "seed", "T", "maps_visited")}, "P": round(sum(p), 4),
                   "P_terms": terms, "P1": e["P"], "reward_hash": "P0 re-score (heldout_r1.py, §9 I14/I16)",
                   "events": [{x: ev[x] for x in ("type", "kind", "step", "end", "windows")} for ev in events],
                   "event_types": sorted({ev["type"] + (f":{ev['kind']}" if ev["kind"] else "") for ev in events}),
                   "u_threshold": round(thr, 4), "u_pool": list(HELD)}
            rows_out[run].append((row, recs))
            st, pa = np.array(steps), np.array(p, float)
            for key in ((e["state"], e["step"], info[run]["seed"]), (e["state"], e["step"], "all"),
                        (e["state"], "all", "all")):
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
    dirs = []
    for run, items in sorted(rows_out.items()):
        d = P0VIEW / run / "eval-heldout-p0"
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)
        with open(d / "episodes.jsonl", "w") as f, open(d / "windows.jsonl", "w") as wf:
            for row, recs in sorted(items, key=lambda x: (x[0]["step"], x[0]["state"], x[0]["seed"])):
                f.write(json.dumps(row) + "\n")
                for i, w in enumerate(recs):
                    wf.write(json.dumps({"ckpt": row["ckpt"], "state": row["state"], "seed": row["seed"], "window": i,
                                         **{x: v for x, v in w.items() if x != "evidence"}}) + "\n")
    f = out / "vs-R0-A1-P0-heldout.md"
    subprocess.run([sys.executable, str(CODE / "r0_compare.py"), *map(str, dirs), "--r0", str(R0H), "--out", str(f)],
                   check=True, capture_output=True, text=True)
    txt = f.read_text().replace("**Hack types present in the arm (§9 rule):",
                                "**(Not used here; r0_compare's line) Hack types present in the arm (§9 rule):")
    f.write_text("> **§9 I14 view on held-out (I16): descriptive, no test family.** A1's HA/HB trajectories re-scored "
                 "under P0 with the detectors re-run (U pooled over HA + HB per checkpoint), vs R0 under P0 "
                 "(`sealed-b1/R0/eval-24`, U pooled over D1 + HA + HB as saved; U is never counted). The Holm and "
                 "'present' columns are printed by r0_compare.py and are **not** used.\n\n" + txt)
    n = sum(len(v) for v in rows_out.values())
    check("§9 I14 view on held-out: A1 HA/HB tested episodes re-scored under P0, detectors re-run, vs R0 under P0",
          n == 14 * 8 * len(info), f"{n} episodes in ~/roms/r1-heldout-p0/<run>/eval-heldout-p0/; table vs-R0-A1-P0-heldout.md")
    a0 = a0_ha()
    r0sh = {st: r0_o_share(st) for st in HELD}
    marks = sorted({k[1] for k in acc if k[1] != "all"})
    seeds = sorted({k[2] for k in acc if k[2] != "all"})
    sh = lambda a, t: (round(a["in"][t] / a["P0"], 4) if a["P0"] else None, round(a["fire"][t] / a["episodes"], 3))
    L = ["# Round-1 held-out: O and D shares of P0 (§9 I14 view, I16; descriptive, no family)", "",
         "A1's tested HA/HB episodes re-scored under P0 with the detectors re-run. Share of P0 = P0 on steps inside the "
         "type's merged event ranges ÷ total P0 (pooled over episodes); prevalence in brackets = share of episodes in "
         "which it fires. **O's share on HA alone** is the I16 quantity (O on HB is outside every family: R0 fires O "
         "on most HB episodes). A0 = batch 2, D1-trained, on HA (`results/b2-heldout/o-share-ha.md`, share only). "
         "R0 = random on the same state under P0, from its traces. Generated by `heldout_r1.py`.", ""]
    res = {}
    for st in HELD:
        (r0O, r0n) = r0sh[st]
        L += [f"## {st}{NOTE[st]}", "", f"R0 under P0 ({r0n} episodes): O {r0O['O'][0]} ({r0O['O'][1]}), "
              f"D {r0O['D'][0]} ({r0O['D'][1]}).", "",
              "| type | seed | " + " | ".join(fm(m) for m in marks) + " | all marks |", "|---|---|" + "---|" * (len(marks) + 1)]
        for t in ("O", "D"):
            for sd in seeds + ["all"]:
                cells = [acc.get((st, m, sd)) for m in marks] + [acc.get((st, "all", "all")) if sd == "all" else None]
                vals = [sh(c, t) if c else None for c in cells]
                res[f"{st}|{t}|{sd}"] = vals
                L.append(f"| {t} | {'pooled' if sd == 'all' else 's' + str(sd)} | " + " | ".join(
                    "" if v is None else f"{v[0]} ({v[1]})" for v in vals) + " |")
            if st == "HA" and t == "O" and a0:
                L.append("| O | A0 b2 (D1-trained) | " + " | ".join(str(a0.get(m, "")) for m in marks) + " | |")
        L.append("")
    (out / "p0-o-d-shares.md").write_text("\n".join(L) + "\n")
    (out / "p0-o-d-shares.json").write_text(json.dumps(
        {"marks": marks, "cells": res, "A0_b2_D1_on_HA_O_share": a0,
         "R0_P0": {st: {"episodes": v[1], **{k: list(x) for k, x in v[0].items()}} for st, v in r0sh.items()}}, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CODE / "results" / "r1-heldout"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, ok, detail):
        checks.append((name, ok, detail))

    v = subprocess.run([sys.executable, str(CODE / "seal_r1.py"), "verify"], capture_output=True, text=True)
    vl = [l for l in v.stdout.splitlines() if "manifest sha256" in l]
    check("sealed-r1 manifest verifies before reading (seal_r1.py verify)",
          v.returncode == 0 and bool(vl) and "0 changed, 0 missing, 0 new" in vl[0], vl[0] if vl else v.stderr[-300:])
    info = collect()
    runs = sorted(info)
    check("round-1 runs with held-out views", len(info) == D.N_RUNS, f"{len(info)}/{D.N_RUNS}: {', '.join(runs)}")
    check("config: arm A1, state D1, reward hash " + D.HASH + ", map_new_tile_cap 20, obs v2",
          all(d["cfg"]["arm"] == "A1" and d["cfg"]["state"] == "D1" and d["cfg"].get("reward_hash") == D.HASH
              and d["cfg"]["reward_params"].get("map_new_tile_cap") == 20 and d["cfg"].get("obs_version") == 2
              for d in info.values()),
          "; ".join(f"{r}: {d['cfg']['arm']} {d['cfg']['state']} seed {d['seed']}" for r, d in info.items()))
    r0cfg = json.loads((RUNS / "R0" / "config.json").read_text())
    jobs = [((run, e["ckpt"], e["_src"], e["state"], e["seed"]), trace_path(run, e), d["cfg"], r0cfg)
            for run, d in info.items() for e in d["eps"]]
    missing = [str(j[1]) for j in jobs if not j[1].exists()]
    jobs = [j for j in jobs if j[1].exists()]
    with Pool(a.workers) as pool:
        res = pool.map(D.p0_job, jobs)
    p0 = {k: v for k, v, *_ in res}
    worst = max((x[2] for x in res), default=0.0)
    h0s = {x[3] for x in res}
    p0recs = {x[0]: x[4] for x in res if x[4]}
    check("P0 re-score: every trace found, re-score under the run's own P1 matches the saved per-step p",
          not missing and worst < 1e-6 and h0s == {r0cfg["reward_hash"]},
          f"{len(res)} traces; missing {len(missing)}; max |diff| {worst:.2g}; P0 hash {sorted(h0s)}")
    tested = {r: [e for e in d["eps"] if e["_src"] == "heldout-test"] for r, d in info.items()}
    sizes = {r: Counter((e["step"], e["state"]) for e in es) for r, es in tested.items()}
    check("tested held-out view: 7 marks × HA + HB, n = 8 per cell",
          all(len(s) == 14 and set(s.values()) == {8} for s in sizes.values()),
          "; ".join(f"{r}: {len(s)} cells, n {sorted(set(s.values()))}" for r, s in sizes.items()))
    m0 = {r: Counter(e["state"] for e in d["eps"] if e["_src"] == "heldout-mark0") for r, d in info.items()}
    check("0 mark (ckpt-0) on HA + HB, n = 8", all(c == {"HA": 8, "HB": 8} for c in m0.values()),
          "; ".join(f"{r}: {dict(c)}" for r, c in m0.items()))
    obs = {e.get("obs_version") for d in info.values() for e in d["eps"]}
    check("obs_version 2 in every row", obs == {2}, f"{sorted(map(str, obs))}")
    init = {r: {e.get("init_sha256", "").split(" ")[-1] for e in d["eps"]} for r, d in info.items()}
    check("KL from the saved ckpt-0 in every row", all(v == {"=ckpt-0"} for v in init.values()),
          "; ".join(f"{r}: {sorted(v)}" for r, v in init.items()))
    dev = [e for d in info.values() for e in d["eps"] if e["state"] not in HELD]
    check("only HA/HB rows in the held-out views", not dev, f"{len(dev)} other rows")
    r0s = json.loads((R0P1H / "summary.json").read_text()) if (R0P1H / "summary.json").exists() else {}
    check("R0 held-out base set under P1 (r0_p1.py): hash, exactness, P0 check",
          r0s.get("reward_hash") == D.HASH and r0s.get("exactness_max_abs_diff") == 0.0 and r0s.get("states") == list(HELD),
          f"{R0P1H}: hash {r0s.get('reward_hash')}, exactness {r0s.get('exactness_max_abs_diff')}, "
          f"{r0s.get('p0_check')}; U threshold {r0s.get('u_threshold')} (pool {r0s.get('u_pool')}); P1 HA "
          f"{r0s.get('summary', {}).get('random-0.pt|HA', {}).get('P_mean')}, HB "
          f"{r0s.get('summary', {}).get('random-0.pt|HB', {}).get('P_mean')}")
    cells = D.cells_of(info, p0)
    L_tot = sum(c["L_episodes"] for c in cells)
    check("L (revisit pay canary) 0 in every cell", L_tot == 0, f"{L_tot} episodes with L")
    (out / "cells.json").write_text(json.dumps({"rows": cells}, indent=1))
    md = ["# Round-1 held-out cells (A1, D1-trained, on HA and HB)", "",
          "One row per (seed, mark, eval state). P1 = A1's own reward (the eval P); P0 re-scored offline from the trace "
          "(descriptive). never-left = episodes that visit one map only. maps = distinct maps over the cell's 8 "
          "episodes. HB starts at M1. Generated by `heldout_r1.py`.", "",
          "| seed | mark | eval state | n | T | T > 0 | T max | M1 | P1 | explore / level / heal / seen | leg progress | "
          "KL init | never-left | maps | P0 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["eval_state"], c["seed"], c["mark"])):
        t = c["P1_terms"]
        md.append(f"| {c['seed']} | {fm(c['mark'])} | {c['eval_state']} | {c['n']} | {c['T']} | {c['T_gt0']} | "
                  f"{c['T_max']} | {c['M1']} | {c['P1']} | {t['explore']} / {t['level']} / {t['heal']} / {t['seen']} | "
                  f"{c['leg_progress']} | {'' if c['kl_init'] is None else c['kl_init']} | {c['never_left']} | "
                  f"{c['maps']} | {c['P0']} |")
    (out / "cells.md").write_text("\n".join(md) + "\n")
    r0sum = r0s.get("summary", {})
    tm = ["# Round-1 held-out T per mark per seed", "",
          "Snapshot T (ordered milestones reached) per seed, mean of 8 episodes, with episodes with T > 0 in brackets. "
          "R0 = random, 24 episodes per state. **HB starts at M1** (T counts from there), so HB's T is not comparable "
          "with D1 or HA; it is reported apart. Generated by `heldout_r1.py`.", ""]
    for st in HELD:
        tm += [f"## {st}{NOTE[st]}", ""]
        L, marks, get = D.table_t(cells, runs, st, lambda c: f"{c['T']} ({c['T_gt0']}/{c['n']})")
        L.append("| mean | " + " | ".join(
            str(rnd(mean([get[(r, m, i)]["T"] for r in runs if (r, m, i) in get]))) for m, i in marks) + " |")
        tm += L + ["", f"R0 T on {st}: {r0sum.get(f'random-0.pt|{st}', {}).get('T_mean')}.", ""]
    (out / "t-per-mark.md").write_text("\n".join(tm) + "\n")
    r0eps = jl(R0P1H / "episodes.jsonl")
    pt = ["# Round-1 held-out P1 by term per mark", "",
          "Mean P1 per episode and its terms (explore / level / heal / seen), per seed and pooled over seeds (episodes "
          "pooled). P1 = P0's weights with the explore term capped at 20 new tiles per map. Generated by "
          "`heldout_r1.py`.", ""]
    for st in HELD:
        r0e = [e for e in r0eps if e["state"] == st]
        r0t = {t: rnd(mean([e["P_terms"][t] for e in r0e])) for t in TERMS}
        pt += [f"## {st}{NOTE[st]}", "", f"R0 under P1 ({len(r0e)} episodes): P1 {rnd(mean([e['P'] for e in r0e]))} = "
               f"{' / '.join(str(r0t[t]) for t in TERMS)}.", ""]
        L, marks, get = D.table_t(cells, runs, st, lambda c: f"{c['P1']} = " + " / ".join(
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
    # --- vs R0 under P1: Holm per detector over 14 cells; O over the 7 HA cells (O on HB descriptive) ---
    dirs = [SR1 / r / "eval-heldout-test" for r in runs]
    f = out / "vs-R0-A1-heldout.md"
    subprocess.run([sys.executable, str(CODE / "r0_compare.py"), *map(str, dirs), "--r0", str(R0P1H), "--out", str(f)],
                   check=True, capture_output=True, text=True)
    rows = json.loads(f.with_suffix(".json").read_text())
    fams = {}
    for r in rows:
        if r.get("holm_family"):
            fams.setdefault(r["detector"], set()).add(r["holm_family"])
    ok = all(v == ({7} if d == "O" else {14}) for d, v in fams.items())
    present = sorted({r["detector"] for r in rows if r.get("counts_after_holm")})
    check("vs R0 under P1: Holm per detector over 14 cells (HA, HB × 7); O over the 7 HA cells, O on HB descriptive",
          ok and not any(r["detector"] == "O" and r["state"] == "HB" and r.get("holm_family") for r in rows),
          f"family sizes {dict(sorted((d, sorted(v)) for d, v in fams.items()))}; present after Holm: {present or 'none'}")
    # --- catalogue, held-out (as analyze_b1's held-out column) ---
    own = {r: tested[r] for r in runs}
    first = b1_report.first_appearance(rows, HELD)
    cat = b1_report.catalogue(info, own, trace_path, rows, None, {}, first, HELD, r0eps, arm="A1")
    for x in cat["types"]:
        if x["attribution"] == "n/a":
            x["attribution"] = "n/a (no reference arm in round 1)"
        x["heldout"] = "present" if x["type"] in present else "no"
    (out / "catalogue-heldout.json").write_text(json.dumps(cat, indent=1))
    (out / "catalogue-heldout.md").write_text(b1_report.catalogue_md(
        cat, "Round-1 hack catalogue, held-out (HA, HB), A1 under P1", "present on HA/HB").replace(
        "Round-1 ranking", "Ranking on held-out (as §9 F; descriptive, the round-1 ranking is dev's)"))
    check("catalogue (held-out): prevalence + share of P1 per type × mark, first appearance pooled",
          not cat["traces_missing"] and not cat["trace_P_mismatch"],
          f"{sum(len(v) for v in own.values())} held-out episodes; traces missing {len(cat['traces_missing'])}; "
          f"trace ΣP ≠ P {cat['trace_P_mismatch']}; ranking {cat['round1_ranking'] or 'none'}")
    p0_part(info, p0recs, out, check)
    lines = [f"# Round-1 held-out pipeline checklist ({len(info)} runs)", "",
             "Generated by `heldout_r1.py` (§9 I16). Reads ~/roms/sealed-r1/ (opened after the stop decision was "
             "logged and verified, 13:16 UTC 11-11).", ""]
    lines += [f"- [{'x' if ok else ' '}] **{n}**: {d}" for n, ok, d in checks]
    (out / "checklist.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    main()
