#!/usr/bin/env python3
"""Batch-2 held-out extras after unsealing (experiment-e1.md §9 I13 asks; §9 B′ (f), (g)). Descriptive tables.

  .venv/bin/python heldout_b2.py [--out results/b2-heldout]

(1) Held-out attribution: A0 vs the dev-trained A2″, matched by training state × eval state × mark (§9 (g)):
    r0_compare.py --vs per training state (D1-trained A0 vs D1-trained A2″ on HA/HB; same for D2), R0 = the
    held-out R0 eval (24 episodes per state), the dev collapse list dropped from the A2″ reference (§9 B′ (b)).
    -> attr-heldout-<D1|D2>.{md,json}
(2) The HA-trained A2″ runs (b2-A2pp-HA-s1..s3) on HA, the §9 (g) difficulty reference: T, leg progress and P by
    mark (pooled, per seed), collapse judged on HA (mean T ≤ best earlier T − 1); their detector shares on HA vs R0
    (descriptive, never in a Holm family), beside the dev-trained A2″ reference's HA rows.
    -> ha-trained-a2pp.{md,json}, vs-R0-A2pp-HA-trained-heldout.{md,json}; and the held-out Fig. 1 with the
    HA-trained line drawn in the two HA panels (pooled over its 3 seeds, collapse on HA) -> fig1-heldout-ha-ref.{svg,md,json}
(3) O's share of P on HA alone (§9 B′ (f): Σ p inside merged O event ranges ÷ Σ p), A0 dev-trained runs on HA,
    pooled over marks and per mark, by training state, with R0's share on HA beside it. -> o-share-ha.{md,json}
Reads the unsealed ~/roms/sealed-b2/ rows (tested view eval-heldout-test/, traces in eval-heldout/traces/). The
sealed manifest is not touched: outputs go to a new folder.
"""
import argparse
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import seal_b1  # noqa: E402
import seal_b2  # noqa: E402

RUNS, SB2 = seal_b2.RUNS, seal_b2.SEALED_RAW
R0H = seal_b1.SEALED_RAW / "R0" / "eval-24"
DEV = ("D1", "D2")


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def runs():
    out = {}
    for run, vm in seal_b2.batch2_runs():
        cfg = json.loads((RUNS / run / "config.json").read_text())
        out[run] = {"arm": cfg["arm"], "state": cfg["state"], "seed": cfg["seed"], "vm": vm,
                    "dir": SB2 / run / "eval-heldout-test"}
    return out


def r0c(dirs, out, vs=None, drop=None):
    cmd = [sys.executable, str(CODE / "r0_compare.py"), *map(str, dirs), "--r0", str(R0H), "--out", str(out)]
    if vs:
        cmd[2 + len(dirs):2 + len(dirs)] = ["--vs", *map(str, vs)]
    if drop:
        cmd += ["--drop", str(drop)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(Path(out).with_suffix(".json").read_text())


def rows_of(j):
    return j["rows"] if isinstance(j, dict) and "rows" in j else j


def merged(ranges):
    out = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def o_share(eps, trace_of):
    """Σ p inside merged O ranges ÷ Σ p (positive part = net here: P0 has no negative terms)."""
    tot = defaultdict(lambda: [0.0, 0.0, 0])
    missing = 0
    for key, e in eps:
        tf = trace_of(e)
        if not tf.exists():
            missing += 1
            continue
        rs = merged([(x["step"], x["end"]) for x in e.get("events", []) if x["type"] == "O"])
        inside = allp = 0.0
        with gzip.open(tf, "rt") as f:
            f.readline()
            for line in f:
                r = json.loads(line)
                allp += r["p"]
                if any(a <= r["ep_step"] <= b for a, b in rs):
                    inside += r["p"]
        for k in (key, "all"):
            tot[k][0] += inside
            tot[k][1] += allp
            tot[k][2] += 1
    return {str(k): {"O_inside": round(i, 2), "P": round(p, 2), "share": round(i / p, 4) if p else None, "episodes": n}
            for k, (i, p, n) in tot.items()}, missing


def collapse_flags(by_mark):
    best, out = None, {}
    for m in sorted(by_mark):
        t = by_mark[m]
        out[m] = best is not None and m > 0 and t <= best - 1
        best = t if best is None else max(best, t)
    return out


def fig1_ha_ref(info, coll_dev, coll_ha, out):
    import b1_report as BR
    dev = {r: d for r, d in info.items() if d["state"] in DEV}
    rows = {r: jl(d["dir"] / "episodes.jsonl") for r, d in info.items()}
    cells = [(ts, es) for ts in DEV for es in ("HA", "HB")]
    data = BR.fig1_data(dev, {r: rows[r] for r in dev}, cells, [], coll_dev)
    ha = {r: {**d, "arm": "A2pp"} for r, d in info.items() if d["state"] == "HA"}
    ref = BR.fig1_data(ha, {r: rows[r] for r in ha}, [("HA", "HA")], [], coll_ha)["HA→HA"]["A2pp"]
    for ts in DEV:
        data[f"{ts}→HA"]["A2ppHA"] = ref
    title = "Fig. 1 (held-out): proxy P and true T, A0 and A2″ by training state, with the HA-trained A2″ reference"
    cap = [" ".join(["Means pooled over seeds (8 eval episodes per seed per mark); band = 95% bootstrap over seeds (B = 10,000).",
           "Thin lines = one seed; hollow = collapsed seed-ckpt (dev-trained A2″ judged on its training state; HA-trained",
           "A2″ judged on HA, §9 (g)). The HA-trained A2″ line (3 seeds, trained and evaluated on HA) is the same in both",
           "HA panels: it is the difficulty reference, outside every family. P and T are stacked panels, not two axes."])]
    Path(out / "fig1-heldout-ha-ref.svg").write_text(BR.fig1_svg(data, {}, title, cap, False,
                                                                 [("A2ppHA", "A2″ HA-trained (HA panels)")]))
    Path(out / "fig1-heldout-ha-ref.md").write_text(BR.fig1_md(data, title, cap))
    Path(out / "fig1-heldout-ha-ref.json").write_text(json.dumps(data, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CODE / "results" / "b2-heldout"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = runs()
    drop = CODE / "results" / "b2" / "collapsed_a2pp.json"          # dev collapse (judged on the training state)
    res = {}
    # (1) attribution per training state
    for ts in DEV:
        a0 = [d["dir"] for r, d in info.items() if d["arm"] == "A0" and d["state"] == ts]
        a2 = [d["dir"] for r, d in info.items() if d["arm"] == "A2pp" and d["state"] == ts]
        j = r0c(a0, out / f"attr-heldout-{ts}.md", vs=a2, drop=drop)
        res[f"attr_{ts}"] = j["attribution"]
    # (2) HA-trained A2″ on HA
    ha = {r: d for r, d in info.items() if d["state"] == "HA"}
    per = {}
    for r, d in ha.items():
        rows = [e for e in jl(d["dir"] / "episodes.jsonl") if e["state"] == "HA"]
        bym = defaultdict(list)
        for e in rows:
            bym[e["step"]].append(e)
        T = {m: mean([e["T"] for e in v]) for m, v in bym.items()}
        per[r] = {"seed": d["seed"], "marks": {m: {"n": len(v), "T": T[m], "M1": sum(e["T"] >= 1 for e in v),
                                                    "leg_progress": mean([e.get("leg_progress") for e in v]),
                                                    "P": mean([e["P"] for e in v])} for m, v in sorted(bym.items())},
                  "collapsed": [m for m, c in collapse_flags(T).items() if c]}
    marks = sorted({m for v in per.values() for m in v["marks"]})
    pooled = {m: {k: mean([v["marks"][m][k] for v in per.values() if m in v["marks"]]) for k in ("T", "leg_progress", "P")}
              for m in marks}
    kept = {m: {k: mean([v["marks"][m][k] for v in per.values() if m in v["marks"] and m not in v["collapsed"]])
                for k in ("T", "leg_progress", "P")} for m in marks}
    hdrop = out / "collapsed_ha_trained.json"
    hdrop.write_text(json.dumps(sorted([r, f"eval-{m}.pt"] for r, v in per.items() for m in v["collapsed"])))
    vs = r0c([d["dir"] for d in ha.values()], out / "vs-R0-A2pp-HA-trained-heldout.md", drop=hdrop)
    dev_ref = json.loads((CODE / "results" / "sealed-b2" / "b2" / "vs-R0-A2pp-heldout.json").read_text())
    coll_dev = {(r, int(c.split("-")[1].split(".")[0])) for r, c in json.loads(drop.read_text())}
    fig1_ha_ref(info, coll_dev, {(r, m) for r, v in per.items() for m in v["collapsed"]}, out)
    res["ha_trained"] = {"per_run": per, "pooled_all_seeds": pooled, "pooled_kept_seeds": kept,
                         "detector_rows_HA": [r for r in rows_of(vs) if r.get("state") == "HA"]}
    # (3) O's share of P on HA, A0 dev-trained, and R0 on HA
    eps = []
    for r, d in info.items():
        if d["arm"] == "A0" and d["state"] in DEV:
            for e in jl(d["dir"] / "episodes.jsonl"):
                if e["state"] == "HA":
                    eps.append(((d["state"], e["step"]), {**e, "_run": r}))
    tr = lambda e: SB2 / e["_run"] / "eval-heldout" / "traces" / f"{Path(e['ckpt']).stem}-HA-{e['seed']}.jsonl.gz"
    a0s, miss = o_share(eps, tr)
    r0e = [(("R0", 0), e) for e in jl(R0H / "episodes.jsonl") if e["state"] == "HA"]
    r0s, miss0 = o_share(r0e, lambda e: next((d / f"{Path(e['ckpt']).stem}-HA-{e['seed']}.jsonl.gz" for d in (R0H / "traces", RUNS / "R0" / "eval-24" / "traces")
                                  if (d / f"{Path(e['ckpt']).stem}-HA-{e['seed']}.jsonl.gz").exists()), R0H / "missing"))
    res["o_share_HA"] = {"A0": a0s, "R0": r0s, "traces_missing": miss + miss0}
    (out / "heldout-b2.json").write_text(json.dumps(res, indent=1, default=str))
    # markdown
    L = ["# HA-trained A2″ on HA (§9 (g) difficulty reference; descriptive, outside every family)", "",
         "Own (training) state HA × 8 per mark; collapse judged on HA (mean T ≤ best earlier T − 1). "
         "Pooled = mean of the 3 seed means; kept = collapsed seed-marks left out.", "",
         "| mark | T all | T kept | leg progress all | P all | per seed T (M1/8) | collapsed |", "|---|---|---|---|---|---|---|"]
    for m in marks:
        L.append(f"| {m:,} | {pooled[m]['T']} | {kept[m]['T']} | {pooled[m]['leg_progress']} | {pooled[m]['P']} | " +
                 ", ".join(f"s{v['seed']} {v['marks'][m]['T']} ({v['marks'][m]['M1']})" for v in per.values() if m in v["marks"]) +
                 " | " + ", ".join(f"s{v['seed']}" for v in per.values() if m in v["collapsed"]) + " |")
    L += ["", "## Detector shares on HA vs R0 (HA-trained, descriptive) beside the dev-trained A2″ reference", "",
          "Share = share of windows (as r0_compare). Dev-trained reference from `results/sealed-b2/b2/vs-R0-A2pp-heldout.json`.", "",
          "| detector | mark | HA-trained share | dev-trained A2″ share | R0 share |", "|---|---|---|---|---|"]
    ref = {(r["detector"], r.get("ckpt"), r.get("state")): r for r in rows_of(dev_ref)}
    for r in res["ha_trained"]["detector_rows_HA"]:
        d = ref.get((r["detector"], r.get("ckpt"), "HA"), {})
        L.append(f"| {r['detector']} | {r.get('ckpt')} | {r.get('share')} | {d.get('share')} | {r.get('r0_share')} |")
    (out / "ha-trained-a2pp.md").write_text("\n".join(L) + "\n")
    M = ["# O's share of P on HA alone (§9 B′ (f): Σ p in merged O ranges ÷ Σ p)", "",
         f"A0, dev-trained runs evaluated on HA; R0 on HA (24 episodes). Traces missing: {res['o_share_HA']['traces_missing']}.", "",
         "| group | episodes | O inside | P | share |", "|---|---|---|---|---|"]
    for k, v in sorted(a0s.items(), key=lambda kv: (kv[0] != "all", kv[0])):
        M.append(f"| A0 {k} | {v['episodes']} | {v['O_inside']} | {v['P']} | {v['share']} |")
    M.append(f"| R0 HA | {r0s['all']['episodes']} | {r0s['all']['O_inside']} | {r0s['all']['P']} | {r0s['all']['share']} |")
    (out / "o-share-ha.md").write_text("\n".join(M) + "\n")
    print(f"wrote {out}: attr-heldout-D1/D2, ha-trained-a2pp, vs-R0-A2pp-HA-trained-heldout, o-share-ha; "
          f"traces missing {res['o_share_HA']['traces_missing']}")


if __name__ == "__main__":
    main()
