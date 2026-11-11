#!/usr/bin/env python3
"""§9 H10 secondary, DESCRIPTIVE series: the learning process (training episodes) next to the snapshot evals.
Own training state only (D1/D2); HA-trained runs are left out (their training episodes are held-out data).

  .venv/bin/python train_series.py --batch b1|b2 [--out DIR]   -> DIR/train-series.{json,md}
                                                                 (default results/<batch>/)

As ruled in §9 H10 (amendment): only what training logs. No training code was changed mid-batch, so there is no
leg progress and no per-step trace; detector information is the ONLINE flag prevalence (hacks_active, K in
stuck_active: share of episodes with the flag), never shares of P.

  unit      completed training episodes (episodes.jsonl via runlog.load_kept: discarded resume branches dropped)
  bins      episodes whose end step falls in (m − 250k, m] for each mark m in 0.25, 0.5, 1, 2, 3, 4, 5M
  per bin   n, P, P by term (batch 2 only), T, share reaching ≥ M1 / ≥ M2 (milestone steps), flag prevalence
  snapshot  the run's own-state eval at the mark (batch 1: eval-dev/; batch 2: eval/ + eval-interim/), P and T
  gap       training − snapshot (P, T) and the P ratio training / snapshot
Readouts (seed-bootstrap 95% CIs over seeds, pooled per arm × state; no p-values reported; nothing here enters a
test family, a tripwire or the patch decision):
  S1 criteria    §9 H10′: decided on BOTH series by seed-bootstrap 95% CIs of per-seed differences (no p-values):
                 (i) per seed P(5M) − P(0.5M) > 0 (CI excludes 0 upward); (ii) per seed, paired by seed number,
                 A0 − A2″ T < 0 at >= 1 mark >= 2M (CI excludes 0 downward). The b1_report.h0_tests permutation
                 output is still written, labelled 'not used (pseudo-replication)' for training
  S2 divergence  'learning-process divergence' if A0's training series meets (i) and (ii) and the snapshot doesn't;
                 'snapshot-only divergence' for the reverse; else 'agree'. Per state, never merged into H0
  S3 contrast    A0 − A2″ T at 5M, training and snapshot side by side (paired by seed)
  S4 gap size    training / snapshot P ratio per mark per arm
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = Path.home() / "roms" / "runs"
DEV = ("D1", "D2")
NOMINAL = [250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]
BIN = 250_000
PLANS = {"b1": ["main-b1-e1a.txt", "main-b1-e1b.txt"], "b2": ["main-b2-e1a.txt", "main-b2-e1b.txt"]}


def runs_of(batch):
    out = []
    for p in PLANS[batch]:
        for line in (HERE / "gcp" / "plans" / p).read_text().splitlines():
            line = line.split("#")[0]
            if "--run-id" in line:
                out.append(line.split("--run-id")[1].split()[0])
    return out


def jl(p):
    if not Path(p).exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f]


def snapshot_rows(d, batch, state):
    srcs = [d / "eval-dev" / "episodes.jsonl"] if batch == "b1" else \
        sorted((d / "eval-interim").glob("*/episodes.jsonl")) + [d / "eval" / "episodes.jsonl"]
    out = {}
    for p in srcs:
        for r in jl(p):
            if r["state"] == state:
                out[(r["step"], r["seed"])] = r
    return list(out.values())


def mark_steps(snap):
    """nominal mark -> the run's actual eval step for it (nearest saved eval-*.pt step)."""
    steps = sorted({r["step"] for r in snap if r["step"] > 0})
    return {m: min(steps, key=lambda s: abs(s - m)) for m in NOMINAL if steps and min(abs(s - m) for s in steps) < 60_000}


def load(batch):
    from pokered import runlog
    info, train, snap = {}, {}, {}
    for run in runs_of(batch):
        d = RUNS / run
        if not (d / "config.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if cfg["state"] not in DEV:
            continue                                     # HA-trained: held-out training data, never read
        ep = [r for r in runlog.load_kept(d, "episodes.jsonl") if "P" in r and "global_step" in r]
        if not ep:
            continue
        info[run] = {"arm": cfg["arm"], "state": cfg["state"], "seed": cfg["seed"]}
        train[run] = ep
        snap[run] = snapshot_rows(d, batch, cfg["state"])
    return info, train, snap


def bins_of(eps, ms):
    """{actual mark step: [episodes ending in (m − 250k, m]]} using the nominal bin edges."""
    out = {}
    for m in NOMINAL:
        step = ms.get(m, m)
        out[step] = [e for e in eps if m - BIN < e["global_step"] <= m]       # nominal edges
    return out


def mean(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def per_bin(eps):
    flags = sorted({f for e in eps for f in (e.get("hacks_active") or []) + (e.get("stuck_active") or [])})
    terms = all("P_terms" in e for e in eps) and eps
    return {"n": len(eps), "P": mean(e["P"] for e in eps), "T": mean(e["T"] for e in eps),
            "P_terms": {k: mean(e["P_terms"][k] for e in eps) for k in ("explore", "level", "heal")} if terms else None,
            "share_ge_M1": mean(e["T"] >= 1 for e in eps), "share_ge_M2": mean(e["T"] >= 2 for e in eps),
            "flag_prevalence": {f: mean(f in ((e.get("hacks_active") or []) + (e.get("stuck_active") or [])) for e in eps)
                                for f in flags}}


def build(batch):
    import b1_report as B
    info, train, snap = load(batch)
    runs = {}
    pseudo_train, pseudo_snap = {}, {}
    for run in sorted(info):
        st = info[run]["state"]
        ms = mark_steps(snap[run])
        bins = bins_of(train[run], ms)
        marks = []
        pseudo_train[run], pseudo_snap[run] = [], []
        for m in NOMINAL:
            step = ms.get(m)
            eps = bins.get(step if step else m, [])
            s = [r for r in snap[run] if step and r["step"] == step]
            b = per_bin(eps)
            rec = {"mark": m, "step": step, "train": b,
                   "snapshot": {"n": len(s), "P": mean(r["P"] for r in s), "T": mean(r["T"] for r in s)}}
            if b["P"] is not None and rec["snapshot"]["P"]:
                rec["gap"] = {"P": round(b["P"] - rec["snapshot"]["P"], 4), "T": round(b["T"] - rec["snapshot"]["T"], 4),
                              "P_ratio": round(b["P"] / rec["snapshot"]["P"], 3)}
            marks.append(rec)
            if step:
                pseudo_train[run] += [{"step": step, "state": st, "seed": i, "T": e["T"], "P": e["P"]} for i, e in enumerate(eps)]
                pseudo_snap[run] += [{"step": step, "state": st, "seed": r["seed"], "T": r["T"], "P": r["P"]} for r in s]
        runs[run] = {**info[run], "marks": marks}
    # pooled per arm × state, seed-bootstrap CIs
    pooled = {}
    for st in DEV:
        for arm in ("A0", "A2pp"):
            rs = sorted(r for r, d in info.items() if d["arm"] == arm and d["state"] == st)
            if not rs:
                continue
            rows = []
            for i, m in enumerate(NOMINAL):
                tp = [[e["P"] for e in pseudo_train[r] if e["step"] == runs[r]["marks"][i]["step"]] for r in rs]
                tt = [[e["T"] for e in pseudo_train[r] if e["step"] == runs[r]["marks"][i]["step"]] for r in rs]
                sp = [[e["P"] for e in pseudo_snap[r] if e["step"] == runs[r]["marks"][i]["step"]] for r in rs]
                st_ = [[e["T"] for e in pseudo_snap[r] if e["step"] == runs[r]["marks"][i]["step"]] for r in rs]
                tp, tt, sp, st_ = ([g for g in x if g] for x in (tp, tt, sp, st_))
                row = {"mark": m, "seeds": len(tp)}
                for name, g in (("train_P", tp), ("train_T", tt), ("snap_P", sp), ("snap_T", st_)):
                    if g:
                        row[name] = mean(x for v in g for x in v)
                        row[name + "_ci"] = [round(v, 3) for v in B.seed_boot(g)]
                if row.get("train_P") and row.get("snap_P"):
                    row["P_ratio"] = round(row["train_P"] / row["snap_P"], 3)          # S4
                rows.append(row)
            pooled[f"{arm}|{st}"] = {"runs": rs, "marks": rows}
    # S1 / S2: H0 criteria on both series (only when every mark is present)
    def complete(ps):
        return all(len({e["step"] for e in ps[r]}) == len(NOMINAL) for r in ps)
    s1 = {}
    for name, ps in (("training", pseudo_train), ("snapshot", pseudo_snap)):
        if ps and complete(ps):
            coll = B.collapsed(info, ps)
            s1[name] = B.h0_tests(info, ps, coll, states=DEV, metrics=(("T", "primary"),))
        else:
            s1[name] = None
    # §9 H10′: S1/S2 are decided by seed-bootstrap 95% CIs of PER-SEED differences, on both series alike.
    # (i) per seed P(5M) − P(0.5M) > 0; (ii) per seed (paired by seed number) A0 − A2″ T < 0 at >= 1 mark >= 2M.
    def seed_ci(vals):
        v = np.asarray(vals, float)
        if len(v) == 0:
            return None, None, None
        rng = np.random.default_rng(B.SEED)
        means = v[rng.integers(0, len(v), (B.B_CI, len(v)))].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        return round(float(v.mean()), 3), round(float(lo), 3), round(float(hi), 3)

    def run_mean(ps, run, step, f):
        v = [e[f] for e in ps[run] if e["step"] == step]
        return float(np.mean(v)) if v else None

    def ci_criteria(ps, st):
        a0 = {info[r]["seed"]: r for r in ps if info[r]["arm"] == "A0" and info[r]["state"] == st}
        a2 = {info[r]["seed"]: r for r in ps if info[r]["arm"] == "A2pp" and info[r]["state"] == st}
        if not a0:
            return None
        step_of = lambda r, m: runs[r]["marks"][NOMINAL.index(m)]["step"]
        d_i = {s: run_mean(ps, r, step_of(r, 5_000_000), "P") - run_mean(ps, r, step_of(r, 500_000), "P")
               for s, r in a0.items() if run_mean(ps, r, step_of(r, 5_000_000), "P") is not None
               and run_mean(ps, r, step_of(r, 500_000), "P") is not None}
        m, lo, hi = seed_ci(list(d_i.values()))
        crit_i = {"per_seed": {k: round(v, 2) for k, v in sorted(d_i.items())}, "mean": m, "ci": [lo, hi],
                  "met": lo is not None and lo > 0}
        ii = []
        for mk in (2_000_000, 3_000_000, 4_000_000, 5_000_000):
            seeds = sorted(set(a0) & set(a2))
            d = {s: run_mean(ps, a0[s], step_of(a0[s], mk), "T") - run_mean(ps, a2[s], step_of(a2[s], mk), "T")
                 for s in seeds if run_mean(ps, a0[s], step_of(a0[s], mk), "T") is not None
                 and run_mean(ps, a2[s], step_of(a2[s], mk), "T") is not None}
            m2, lo2, hi2 = seed_ci(list(d.values()))
            ii.append({"mark": mk, "per_seed": {k: round(v, 3) for k, v in sorted(d.items())}, "mean": m2,
                       "ci": [lo2, hi2], "met": hi2 is not None and hi2 < 0})
        return {"i": crit_i, "ii": ii, "ii_met": any(x["met"] for x in ii),
                "meets_i_and_ii": crit_i["met"] and any(x["met"] for x in ii)}

    s1_ci, s2 = {}, {}
    for st in DEV:
        t, s = ci_criteria(pseudo_train, st), ci_criteria(pseudo_snap, st)
        s1_ci[st] = {"training": t, "snapshot": s}
        if not t or not s or not complete(pseudo_train) or not complete(pseudo_snap):
            s2[st] = "not computable yet (series incomplete)"
            continue
        tm, sm = t["meets_i_and_ii"], s["meets_i_and_ii"]
        s2[st] = ("learning-process divergence" if tm and not sm else "snapshot-only divergence" if sm and not tm
                  else "agree")
    # the permutation output is kept but NOT used for S1/S2 (§9 H10′)
    if s1.get("training"):
        s1["training"]["_status"] = "not used (pseudo-replication: correlated training episodes treated as independent)"
    if s1.get("snapshot"):
        s1["snapshot"]["_status"] = "permutation valid for snapshots (H5 primary) but not used for S2 (§9 H10′)"
    # S3: A0 − A2″ T at 5M, paired by seed
    s3 = {}
    for st in DEV:
        out = {}
        for name, ps in (("training", pseudo_train), ("snapshot", pseudo_snap)):
            a0 = {info[r]["seed"]: [e["T"] for e in ps[r] if e["step"] == runs[r]["marks"][-1]["step"]]
                  for r in ps if info[r]["arm"] == "A0" and info[r]["state"] == st}
            a2 = {info[r]["seed"]: [e["T"] for e in ps[r] if e["step"] == runs[r]["marks"][-1]["step"]]
                  for r in ps if info[r]["arm"] == "A2pp" and info[r]["state"] == st}
            seeds = sorted(s for s in set(a0) & set(a2) if a0[s] and a2[s])
            if seeds:
                lo, hi = B.seed_boot_diff([a2[s] for s in seeds], [a0[s] for s in seeds])
                out[name] = {"A0_minus_A2pp_T": round(mean(x for s in seeds for x in a0[s]) - mean(x for s in seeds for x in a2[s]), 4),
                             "ci": [round(lo, 3), round(hi, 3)], "seeds": len(seeds)}
        s3[st] = out
    return {"batch": batch, "label": "DESCRIPTIVE secondary series (§9 H10); not a test, not a tripwire",
            "not_available": "leg progress and per-step traces (training logs none); shares of P by detector"
                             + ("; P by term (batch 1 doesn't log it)" if batch == "b1" else ""),
            "runs": runs, "pooled": pooled, "S1_seed_ci": s1_ci, "S1_H0_mirror_permutation": s1,
            "S2_divergence": s2, "S2_rule": "§9 H10′: (i) and (ii) by seed-bootstrap 95% CIs of per-seed differences",
            "S3_contrast_T5M": s3}


def md(res):
    L = [f"# {res['batch']}: training-episode series vs snapshot evals (DESCRIPTIVE, §9 H10)", "",
         "Own training state only; HA-trained runs excluded. Training bins: episodes ending in (m − 250k, m]. "
         f"Not available: {res['not_available']}. Seed-bootstrap 95% CIs; nothing here is a test.", "",
         "## Pooled per arm × state (S4 = train/snapshot P ratio)", "",
         "| arm·state | mark | seeds | train P [CI] | snapshot P [CI] | S4 ratio | train T [CI] | snapshot T [CI] |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in res["pooled"].items():
        for r in v["marks"]:
            f = lambda n: f"{r.get(n)} {r.get(n + '_ci', '')}" if r.get(n) is not None else "–"
            L.append(f"| {k} | {r['mark'] / 1e6:g}M | {r['seeds']} | {f('train_P')} | {f('snap_P')} | {r.get('P_ratio', '–')} | "
                     f"{f('train_T')} | {f('snap_T')} |")
    L += ["", "## S1 / S2 (§9 H10′: seed-bootstrap 95% CIs of per-seed differences, both series)", "",
          "| state | series | (i) ΔP 5M − 0.5M per seed → mean [CI] | (i) met | (ii) marks with A0 − A2″ T CI < 0 | S2 |",
          "|---|---|---|---|---|---|"]
    for st, v in res["S1_seed_ci"].items():
        for name in ("training", "snapshot"):
            x = v.get(name)
            if not x:
                continue
            per = ", ".join(f"{d:+.0f}" for d in x["i"]["per_seed"].values())
            marks = ", ".join(f"{r['mark'] / 1e6:g}M" for r in x["ii"] if r["met"]) or "none"
            L.append(f"| {st} | {name} | {per} → {x['i']['mean']:+.1f} [{x['i']['ci'][0]:.0f}, {x['i']['ci'][1]:.0f}] | "
                     f"{'met' if x['i']['met'] else 'not met'} | {marks} | {res['S2_divergence'][st] if name == 'training' else ''} |")
    L += ["", "The H0-mirror permutation output in the JSON is kept but not used: training episodes are correlated "
          "(pseudo-replication)."]
    L += ["", "## S3: A0 − A2″ T at 5M (paired by seed)", ""]
    for st, v in res["S3_contrast_T5M"].items():
        L.append(f"- {st}: " + "; ".join(f"{n} {x['A0_minus_A2pp_T']} [{x['ci'][0]}, {x['ci'][1]}] ({x['seeds']} seeds)"
                                          for n, x in v.items()) if v else f"- {st}: not computable yet")
    L += ["", "## Per run (descriptive line: training-episode P and T per mark)", "",
          "| run | " + " | ".join(f"{m / 1e6:g}M" for m in NOMINAL) + " |", "|---|" + "---|" * len(NOMINAL)]
    for run, v in res["runs"].items():
        cells = []
        for r in v["marks"]:
            t, s = r["train"], r["snapshot"]
            cells.append(f"train P {t['P']} T {t['T']} (n {t['n']}) / snap P {s['P']} T {s['T']}" if t["n"] else "–")
        L.append(f"| {run} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", choices=["b1", "b2"], required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = build(a.batch)
    od = Path(a.out) if a.out else HERE / "results" / a.batch
    od.mkdir(parents=True, exist_ok=True)
    (od / "train-series.json").write_text(json.dumps(res, indent=1, default=str))
    (od / "train-series.md").write_text(md(res))
    print(md(res)[:4000])


if __name__ == "__main__":
    main()
