#!/usr/bin/env python3
"""Batch-2 U-review window aggregates (§9 E; as the batch-1 U review): dev views only, never held-out.

  .venv/bin/python b2_ureview.py [--out results/b2]      -> OUT/u-review.{md,json}

Every U window (`"U" in fired`) in each batch-2 run's `eval-dev/` (D1 + D2, marks 0.25-5M; built by seal_b2.py)
and in R0's dev windows (`~/roms/runs/R0/eval-dev-24/`, the false-alarm floor). Per window, aggregate fields only:
maps (rows per map, named from pret map constants), P in W by term (explore / level / heal / seen), distinct
tiles, off-path fraction, P on revisits, and battle share (fraction of the window's steps with in_battle, from
the episode trace). Grouped by arm × training state × view (own state / cross state) × mark, and pooled per
arm. U is a review queue, never counted: no test, no p-value here.
"""
import argparse
import gzip
import json
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

from u_report import MAPS
import seal_b2

RUNS = Path.home() / "roms" / "runs"
R0 = RUNS / "R0" / "eval-dev-24"
LAB = {250_880: "0.25M", 500_736: "0.5M", 1_000_448: "1M", 2_000_896: "2M", 3_000_320: "3M", 4_000_768: "4M",
       5_000_192: "5M", 0: "0"}


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def mark(ckpt):
    return int(ckpt.split("-")[1].split(".")[0])


def mapname(h):
    return MAPS.get(int(h, 16), h)


def battle_share(trace, start, end):
    """Fraction of trace steps start..end (1-based episode steps, inclusive) with in_battle set."""
    if not trace.exists():
        return None
    n = b = 0
    with gzip.open(trace, "rt") as f:
        f.readline()                                    # episode header
        for i, line in enumerate(f, 1):
            if i < start:
                continue
            if i > end:
                break
            n += 1
            b += bool(json.loads(line).get("in_battle"))
    return round(b / n, 3) if n else None


def windows(run, arm, train_state, d, trace_dir, trace_name):
    out = []
    for w in jl(d / "windows.jsonl"):
        if "U" not in w["fired"]:
            continue
        m = mark(w["ckpt"])
        out.append({
            "run": run, "arm": arm, "train_state": train_state, "eval_state": w["state"],
            "view": "R0" if arm == "R0" else ("own" if w["state"] == train_state else "cross"),
            "mark": m, "seed": w["seed"], "window": w["window"], "start": w["start"], "end": w["end"],
            "u_kind": w.get("u_kind"), "p_in_w": w["p_in_w"],
            "p_terms": {k: w[f"p_{k}"] for k in ("explore", "level", "heal", "seen")},
            "p_on_revisits": w["p_on_revisits"], "distinct_tiles": w["distinct_tiles"],
            "offpath_fraction": w["offpath_fraction"], "no_milestone": w["no_milestone"],
            "maps": {mapname(k): v for k, v in sorted(w["maps"].items(), key=lambda kv: -kv[1])},
            "battle_share": battle_share(trace_dir / trace_name(w), w["start"], w["end"]),
        })
    return out


def collect():
    rows = []
    for run, _vm in seal_b2.batch2_runs():
        cfg = json.loads((RUNS / run / "config.json").read_text())
        d = RUNS / run / "eval-dev"
        rows += windows(run, cfg["arm"], cfg["state"], d, d / "traces",
                        lambda w: f"{Path(w['ckpt']).stem}-{w['state']}-{w['seed']}.jsonl.gz")
    rows += windows("R0", "R0", None, R0, R0 / "traces",
                    lambda w: f"random-0-{w['state']}-{w['seed']}.jsonl.gz")
    return rows


def q(v):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return f"{min(v):.3g} / {stats.median(v):.3g} / {max(v):.3g}"


def summarise(rs):
    maps = Counter()
    for r in rs:
        maps.update(r["maps"])
    tot = sum(maps.values()) or 1
    p = sum(r["p_in_w"] for r in rs) or 1
    return {"windows": len(rs), "episodes": len({(r["run"], r["mark"], r["eval_state"], r["seed"]) for r in rs}),
            "u_kind": dict(Counter(r["u_kind"] for r in rs)),
            "p_in_w": q([r["p_in_w"] for r in rs]),
            "p_share_by_term": {k: round(sum(r["p_terms"][k] for r in rs) / p, 3) for k in ("explore", "level", "heal", "seen")},
            "with_heal": sum(r["p_terms"]["heal"] > 0 for r in rs), "with_level": sum(r["p_terms"]["level"] > 0 for r in rs),
            "distinct_tiles": q([r["distinct_tiles"] for r in rs]),
            "offpath_fraction": q([r["offpath_fraction"] for r in rs]),
            "p_on_revisits": q([r["p_on_revisits"] for r in rs]),
            "battle_share": q([r["battle_share"] for r in rs]),
            "in_battle_any": sum((r["battle_share"] or 0) > 0 for r in rs),
            "no_milestone": sum(r["no_milestone"] for r in rs),
            "top_maps": {k: round(v / tot, 3) for k, v in maps.most_common(6)}}


def md(res):
    L = ["# Batch-2 U-review window aggregates (§9 E; dev only, descriptive; U is never counted)", "",
         "Every U window in the batch-2 dev views (`eval-dev/`: D1 + D2, marks 0.25-5M) and R0's dev windows. "
         "Aggregate fields only, as in the batch-1 review (maps, P by term, distinct tiles, battle share). "
         "Ranges are min / median / max. Generated by `b2_ureview.py`; per-window rows in `u-review.json`.", "",
         f"Total: {res['total']} U windows.", "", "## Per arm (pooled)", "",
         "| arm | view | windows | episodes | u_kind | P in W | P share explore / level / heal / seen | windows with heal / level P | "
         "distinct tiles | off-path fraction | P on revisits | battle share | windows in battle | top maps (share of rows) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in res["by_arm"].items():
        arm, view = k.split("|")
        t = s["p_share_by_term"]
        L.append(f"| {arm} | {view} | {s['windows']} | {s['episodes']} | "
                 f"{', '.join(f'{a} {b}' for a, b in s['u_kind'].items())} | {s['p_in_w']} | "
                 f"{t['explore']} / {t['level']} / {t['heal']} / {t['seen']} | {s['with_heal']} / {s['with_level']} | "
                 f"{s['distinct_tiles']} | {s['offpath_fraction']} | {s['p_on_revisits']} | {s['battle_share']} | "
                 f"{s['in_battle_any']} | {', '.join(f'{m} {v}' for m, v in s['top_maps'].items())} |")
    L += ["", "## Counts per arm × training state × view × mark", "",
          "| arm | training state | view | " + " | ".join(v for k, v in LAB.items() if k) + " | total |",
          "|---|---|---|" + "---|" * 8]
    for k, c in res["counts"].items():
        arm, ts, view = k.split("|")
        L.append(f"| {arm} | {ts} | {view} | " + " | ".join("–" if arm == "R0" else str(c.get(str(m), 0)) for m in LAB if m) +
                 f" | {sum(c.values())} |")
    L += ["", "R0 is the random policy (no marks): its row is the 24-episode dev base set per state."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "b2"))
    a = ap.parse_args()
    rows = collect()
    by_arm, counts = defaultdict(list), defaultdict(Counter)
    for r in rows:
        arm = f"{r['arm']}-HA" if r["train_state"] == "HA" else r["arm"]
        by_arm[f"{arm}|{r['view']}"].append(r)
        counts[f"{arm}|{r['train_state'] or '–'}|{r['view'] if arm != 'R0' else r['eval_state']}"][str(r["mark"])] += 1
    order = ["A0|own", "A0|cross", "A2pp|own", "A2pp|cross", "A2pp-HA|cross", "R0|R0"]
    res = {"total": len(rows),
           "by_arm": {k: summarise(by_arm[k]) for k in order if k in by_arm},
           "counts": {k: dict(v) for k, v in sorted(counts.items())}, "windows": rows}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "u-review.json").write_text(json.dumps(res, indent=1))
    (out / "u-review.md").write_text(md(res))
    print(f"{len(rows)} U windows; wrote {out / 'u-review.md'}")


if __name__ == "__main__":
    main()
