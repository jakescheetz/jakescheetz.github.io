#!/usr/bin/env python3
"""R0's dev base set re-scored under a patched proxy (round 1: P1 = MapCapProxy(map_new_tile_cap=20)), with the
detectors re-run, so the round-1 vs-R0 tables compare A1 with random under the same reward.

  .venv/bin/python r0_p1.py [--like r1-A1-D1-s1] [--src ~/roms/runs/R0/eval-dev-24] [--out-name eval-dev-24-p1]
  held-out (round 1, §9 I16): --src ~/roms/runs/R0/eval-24 --states HA HB
      --windows ~/roms/sealed-b1/R0/eval-24/windows.jsonl --out-dir ~/roms/runs/R0/eval-heldout-24-p1

Why: six of the §6 detectors read the proxy's per-step reward or its terms (L, K, HF, O, B through r_explore,
and U through P in W). R0's saved windows were scored under P0. A1 is scored under P1, so its base rate must be
too. Nothing is replayed: each saved R0 trace row holds the RAM fields the proxy reads (rescore.py's method).

Per trace (R0 dev: D1 + D2 × 24 episodes, random-0):
  1. exactness: re-score under R0's own reward params; every step must match the saved `p` (|diff| <= 1e-9);
  2. P0 check: the detectors on the unchanged trace must reproduce the saved windows' `fired` (before U);
  3. re-score under LIKE's reward params (P1, hash checked against LIKE's config); rewrite each row's p, r and
     r_<term>; run detectors.window_records.
U: threshold pooled over all 48 episodes (R0's dev pool is D1 + D2, as A1's closing eval pools D1 + D2 per
checkpoint), then apply_u; the P0 run of the same pool must reproduce the saved U firings (check 2 includes U).
Writes ~/roms/runs/R0/<out-name>/{episodes,windows}.jsonl + summary.json (ROM-derived: stays in ~/roms), and
prints counts and the P1 means only.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from pokered import detectors, env as E  # noqa: E402
import rescore  # noqa: E402

RUNS = Path.home() / "roms" / "runs"
TERMS = ("explore", "level", "heal", "seen")


def load_trace(p):
    with gzip.open(p, "rt") as f:
        head = json.loads(f.readline())
        return head, [json.loads(l) for l in f]


def score_terms(trace, cfg, state, kw):
    """Per-step (total, {term: value}) under Proxy kwargs kw, and the proxy's reward hash."""
    p = E.make_proxy(**kw)
    p.reset(rescore.start_mem(cfg, state))
    mem, out = bytearray(0x10000), []
    for row in trace:
        out.append(p(rescore.row_mem(row, mem), 0))
    return out, E.reward_hash(p)


def u_events(recs_by_ep):
    thr = detectors.u_threshold([w for wr in recs_by_ep.values() for w in wr])
    for wr in recs_by_ep.values():
        detectors.apply_u(wr, thr)
    return thr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--like", default="r1-A1-D1-s1", help="run whose reward params (P1) are applied")
    ap.add_argument("--src", default=str(RUNS / "R0" / "eval-dev-24"))
    ap.add_argument("--out-name", default="eval-dev-24-p1")
    ap.add_argument("--out-dir", help="write here instead of ~/roms/runs/R0/<out-name> (e.g. the held-out set)")
    ap.add_argument("--states", nargs="+", default=["D1", "D2"], help="eval states to take from --src (U pool)")
    ap.add_argument("--check-pool", nargs="+", help="U pool the saved windows were computed over, for the P0 check "
                    "only (default: --states); extra states are re-run for the check and not written")
    ap.add_argument("--windows", help="saved windows.jsonl for the P0 check (default: --src/windows.jsonl)")
    a = ap.parse_args()
    src = Path(a.src)
    r0cfg = json.loads((RUNS / "R0" / "config.json").read_text())
    like = json.loads((RUNS / a.like / "config.json").read_text())
    kw0, kw1 = rescore.proxy_kwargs(r0cfg["reward_params"]), rescore.proxy_kwargs(like["reward_params"])
    states = tuple(a.states)
    cpool = tuple(a.check_pool or states)
    saved = [e for e in map(json.loads, open(src / "episodes.jsonl")) if e["state"] in states + cpool]
    saved_w = {}
    for w in map(json.loads, open(a.windows or src / "windows.jsonl")):
        if w["state"] not in states:
            continue
        saved_w.setdefault((w["state"], w["seed"]), []).append(w)
    rows0, recs0, recs1, traces = {}, {}, {}, {}
    worst = 0.0
    for ep in saved:
        k = (ep["state"], ep["seed"])
        head, tr = load_trace(src / "traces" / f"{Path(ep['ckpt']).stem}-{ep['state']}-{ep['seed']}.jsonl.gz")
        s0, h0 = score_terms(tr, r0cfg, ep["state"], kw0)
        assert h0 == r0cfg["reward_hash"], (h0, r0cfg["reward_hash"])
        worst = max(worst, max(abs(t - r["p"]) for (t, _), r in zip(s0, tr)))
        if ep["state"] in cpool:
            recs0[k] = detectors.window_records([dict(r) for r in tr])
        if ep["state"] not in states:
            continue
        s1, h1 = score_terms(tr, like, ep["state"], kw1)
        assert h1 == like["reward_hash"], (h1, like["reward_hash"])
        tr1 = []
        for r, (tot, parts) in zip(tr, s1):
            r = dict(r, p=tot, r=tot, **{f"r_{t}": parts[t] for t in TERMS})
            tr1.append(r)
        recs1[k] = detectors.window_records(tr1)
        traces[k] = tr1
        rows0[k] = ep
    if worst > 1e-9:
        sys.exit(f"exactness check failed: max |diff| {worst}")
    u_events(recs0)
    bad = [k for k in recs0 if k[0] in states and [w["fired"] for w in recs0[k]] != [w["fired"] for w in saved_w[k]]]
    if bad:
        sys.exit(f"P0 check failed: re-run detectors differ from the saved R0 windows on {len(bad)} episodes: {bad[:3]}")
    thr = u_events(recs1)
    out = Path(a.out_dir) if a.out_dir else RUNS / "R0" / a.out_name
    out.mkdir(exist_ok=True)
    eps = []
    with open(out / "episodes.jsonl", "w") as f, open(out / "windows.jsonl", "w") as wf:
        for k, ep in rows0.items():
            wr, tr = recs1[k], traces[k]
            events = detectors.merge(detectors.firings_of(wr))
            r = {**ep, "P": round(sum(x["p"] for x in tr), 4),
                 "P_terms": {t: round(sum(x[f"r_{t}"] for x in tr), 4) for t in TERMS},
                 "P0": ep["P"], "reward_hash": like["reward_hash"], "reward_params": like["reward_params"],
                 "events": [{x: e[x] for x in ("type", "kind", "step", "end", "windows")} for e in events],
                 "event_types": sorted({e["type"] + (f":{e['kind']}" if e["kind"] else "") for e in events}),
                 "u_threshold": round(thr, 4), "u_pool": list(states), "source": f"{src.name} re-scored ({a.like})"}
            eps.append(r)
            f.write(json.dumps(r) + "\n")
            for i, w in enumerate(wr):
                wf.write(json.dumps({"ckpt": ep["ckpt"], "state": ep["state"], "seed": ep["seed"], "window": i,
                                     **{x: v for x, v in w.items() if x != "evidence"}, "source": r["source"]}) + "\n")
    summ = {}
    for st in states:
        rs = [r for r in eps if r["state"] == st]
        types = {}
        for r in rs:
            for t in {e["type"] for e in r["events"]}:
                types[t] = types.get(t, 0) + 1
        summ[f"random-0.pt|{st}"] = {"episodes": len(rs), "P_mean": round(sum(r["P"] for r in rs) / len(rs), 3),
                                     "P0_mean": round(sum(r["P0"] for r in rs) / len(rs), 3),
                                     "T_mean": round(sum(r["T"] for r in rs) / len(rs), 3), "episodes_with_type": types}
    (out / "summary.json").write_text(json.dumps(
        {"run_id": "R0", "states": list(states), "reward_hash": like["reward_hash"], "like_run": a.like,
         "reward_params": like["reward_params"], "u_pool": list(states), "u_threshold": round(thr, 4),
         "exactness_max_abs_diff": worst, "p0_check": "re-run detectors reproduce the saved R0 windows incl. U (U pool " + "+".join(cpool) + ")",
         "summary": summ, "note": "R0 dev base set re-scored under the patched proxy by r0_p1.py; detectors re-run"},
        indent=1))
    print(f"exactness max |diff| {worst}; P0 check ok ({len(rows0)} episodes, U pool {"+".join(cpool)}, windows incl. U identical); "
          f"U threshold under {like['reward_hash']}: {thr:.4f}")
    for k, v in summ.items():
        print(k, json.dumps(v))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
