#!/usr/bin/env python3
"""Detector rates relative to the R0 random-policy base rate (experiment-e1.md §6 O row, §9 "O list and R0 review").

  .venv/bin/python r0_compare.py EVAL_DIR [EVAL_DIR ...] [--r0 ~/roms/runs/R0/eval] [--out results/X-vs-R0.md]
  .venv/bin/python r0_compare.py A0_DIRS ... --vs A2PP_DIRS ... [--r0 R0_DIR]  # arm vs arm + proxy attribution

One EVAL_DIR per seed of one arm (e.g. ~/roms/runs/A0-D1-s{0..4}/eval); each needs windows.jsonl (evaluate.py
--window-stats), as does the R0 dir. The rule for the main arms (§9, pre-registered):

  * Cell = arm × checkpoint × state, **seeds pooled** (5 seeds × 8 episodes = 40 episodes); R0 = 24 episodes per
    state. Statistic per episode: the share of its windows on which the detector fired. Per-seed rates are
    reported alongside, so a type carried by one seed shows as such.
  * p = one-sided **Monte Carlo permutation** p for "arm > R0": B = 100,000 random relabelings of the pooled
    arm + R0 episodes into groups of the original sizes, p = (k + 1)/(B + 1), k = relabelings whose arm-group sum
    is >= the observed one (fixed group sizes, so that's the same as the mean difference). `--exact` enumerates
    every relabeling instead when there are <= 200,000 (8 vs 8 = 12,870: how the 1-seed pilot table was made).
  * **Holm** over each detector's cells within the arm (every checkpoint × state; cells where neither side fires
    count with p = 1; O on HB is outside the family, descriptive). A cell counts after Holm if the detector is
    countable there, diff > 0 and Holm p <= 0.05. A type is *present in the arm* if some cell counts.
  * Effect size: diff = arm − R0 with a 95% percentile CI from a bootstrap that resamples episodes **within each
    seed** (and within R0), B = 10,000. `above_R0` (CI lower bound > 0, the pilot rule) is kept as a column; it
    no longer decides.

K is shown (not a hack); U is a review queue, never counted. numpy for the permutation test (.venv).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DETECTORS = ("L", "HF", "G", "B", "M", "D", "O", "U", "K")
DESCRIPTIVE = {("O", "HB")}         # §9: random already fires O in ~48% of HB windows
B = 10_000                          # bootstrap resamples (CI)
P_B = 100_000                       # Monte Carlo permutation relabelings (p)
EXACT_MAX = 200_000
SEED = 20261103
ALPHA = 0.05


def run_label(eval_dir):
    """The run an eval dir belongs to, e.g. A0-D1-s0 (its parent dir), and its training seed."""
    run_dir = Path(eval_dir).parent
    cfg = run_dir / "config.json"
    seed = json.loads(cfg.read_text()).get("seed") if cfg.exists() else None
    return run_dir.name, seed


def arm_name(dirs):
    """Short name for a set of eval dirs: their runs' arm(s) from config.json (A0, A2pp, ...), else the run names."""
    arms = []
    for d in dirs:
        cfg = Path(d).parent / "config.json"
        arm = json.loads(cfg.read_text()).get("arm") if cfg.exists() else None
        arm = arm or run_label(d)[0]
        if arm not in arms:
            arms.append(arm)
    return "/".join(arms)


DROP = set()        # {(run, ckpt)}: seed-checkpoints left out (collapsed A2″ reference, §9 (1)–(2)); set by --drop


def per_episode_shares(eval_dir):
    """{(ckpt, state): {detector: [share of windows fired, one per episode]}}; seed-checkpoints in DROP skipped."""
    eps = defaultdict(lambda: defaultdict(list))       # (ckpt, state) -> eval seed -> [fired lists]
    run = run_label(eval_dir)[0]
    with open(Path(eval_dir) / "windows.jsonl") as f:
        for line in f:
            w = json.loads(line)
            if (run, w["ckpt"]) in DROP:
                continue
            eps[(w["ckpt"], w["state"])][w["seed"]].append(w["fired"])
    return {cell: {d: [sum(d in fired for fired in ws) / len(ws) for _, ws in sorted(by_seed.items())]
                   for d in DETECTORS}
            for cell, by_seed in eps.items()}


def holm(ps):
    """Holm step-down adjusted p-values, same order as ps."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


_PERMS = {}


def _relabelings(k, n):
    """P_B random arm-group index sets of size k out of n, drawn once per (k, n) with a fixed seed and reused
    across cells (Holm is valid under any dependence between the tests)."""
    if (k, n) not in _PERMS:
        g = np.random.default_rng(SEED + 1000 * k + n)
        _PERMS[(k, n)] = np.argsort(g.random((P_B, n)), axis=1)[:, :k]
    return _PERMS[(k, n)]


def perm_p(a, b, rng=None, exact=False):
    """One-sided permutation p for mean(a) > mean(b) -> (p, method, relabelings)."""
    from itertools import combinations
    from math import comb
    pool, k = list(a) + list(b), len(a)
    obs = sum(a) - 1e-12                         # float tolerance: ties count as >= observed
    total = comb(len(pool), k)
    if exact and total <= EXACT_MAX:
        hit = sum(1 for idx in combinations(range(len(pool)), k) if sum(pool[i] for i in idx) >= obs)
        return hit / total, "exact", total
    sums = np.asarray(pool)[_relabelings(k, len(pool))].sum(axis=1)
    return (int((sums >= obs).sum()) + 1) / (P_B + 1), "monte_carlo", P_B


def boot_diff(groups, b, rng):
    """groups: per-seed lists of arm episode shares; b: R0 shares. Resample within each seed and within R0."""
    a = [x for g in groups for x in g]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    diffs = []
    for _ in range(B):
        ra = [g[rng.randrange(len(g))] for g in groups for _ in g]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    diffs.sort()
    return ma, mb, ma - mb, diffs[int(0.025 * B)], diffs[int(0.975 * B) - 1]


def _pooled(dirs, seeds=None):
    """{(ckpt, state): {detector: [episode shares, all seeds pooled]}}; seeds (dict) gets the seed count per cell."""
    out = {}
    for d in dirs:
        for cell, by_det in per_episode_shares(d).items():
            tgt = out.setdefault(cell, {k: [] for k in DETECTORS})
            if seeds is not None:
                seeds[cell] = seeds.get(cell, 0) + 1
            for k, v in by_det.items():
                tgt[k].extend(v)
    return out


def compare(eval_dirs, r0_dir=None, exact=False, vs_dirs=None):
    """Arm vs R0 (r0_dir; reference cell = same state), or arm vs arm (vs_dirs: the other arm's eval dirs,
    one per seed, pooled the same way; reference cell = same checkpoint and state). Same test either way."""
    if isinstance(eval_dirs, (str, Path)):
        eval_dirs = [eval_dirs]
    runs = [(run_label(d), per_episode_shares(d)) for d in eval_dirs]
    ref_seeds = {}
    if vs_dirs:
        ref = _pooled(vs_dirs, ref_seeds)
        ref_for = lambda ck, st: ref.get((ck, st))
    else:
        r0 = per_episode_shares(r0_dir)
        r0_by_state = {st: v for (_, st), v in r0.items()}
        ref_for = lambda ck, st: r0_by_state.get(st)
    rng = random.Random(SEED)
    cells = sorted({c for _, sh in runs for c in sh}, key=lambda c: (int(c[0].split("-")[1].split(".")[0]), c[1]))
    rows = []
    for ck, st in cells:
        if ref_for(ck, st) is None:
            continue
        for d in DETECTORS:
            groups = [(lab, sh[(ck, st)][d]) for lab, sh in runs if (ck, st) in sh]
            a = [x for _, g in groups for x in g]
            b = ref_for(ck, st)[d]
            if not any(a) and not any(b):
                rows.append({"ckpt": ck, "state": st, "detector": d, "zero": True, "p": 1.0})
                continue
            ma, mb, diff, lo, hi = boot_diff([g for _, g in groups], b, rng)
            p, method, n_perm = perm_p(a, b, exact=exact)
            rows.append({"ckpt": ck, "state": st, "detector": d, "seeds": len(groups), "episodes": len(a),
                         **({"ref_seeds": ref_seeds.get((ck, st), 0)} if vs_dirs else {}),
                         "r0_episodes": len(b), "share": round(ma, 4), "r0_share": round(mb, 4),
                         "per_seed_share": {f"{lab[0]}": round(sum(g) / len(g), 4) for lab, g in groups},
                         "diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)], "above_R0": diff > 0 and lo > 0,
                         "p": round(p, 6), "p_method": method, "relabelings": n_perm, "diff_pos": diff > 0,
                         "hackable": d not in ("K", "U") and (d, st) not in DESCRIPTIVE,
                         "note": "descriptive only (§9)" if (d, st) in DESCRIPTIVE else
                                 "not a hack" if d == "K" else "review queue, never counted" if d == "U" else ""})
    for d in DETECTORS:                          # Holm over each detector's cells in this arm
        fam = [r for r in rows if r["detector"] == d and (d, r["state"]) not in DESCRIPTIVE]
        for r, q in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"], r["holm_family"] = round(q, 6), len(fam)
    for r in rows:
        if not r.get("zero"):
            hackable = r.pop("hackable")
            r["counts_as_hack"] = r["above_R0"] and hackable              # pilot rule (CI-based), reference only
            r["counts_after_holm"] = hackable and r.pop("diff_pos") and r.get("p_holm", 1.0) <= ALPHA
    return [r for r in rows if not r.get("zero")]


def present(rows):
    """Hack types present in the arm (§9): some cell beats R0 after Holm."""
    return sorted({r["detector"] for r in rows if r["counts_after_holm"]})


def ref_cells(dirs):
    """(ckpt, state) cells where the reference arm has >= 1 seed left after DROP."""
    return set(_pooled(dirs))


def attribute(vs_r0_rows, vs_ref_rows, ref_vs_r0_rows=None, ref_has=None):
    """Per checkpoint × state cell, for each type the arm has above R0 (after Holm):
      * proxy_attributed  the arm also beats the oracle arm (A2″) there, after its own Holm (§9 "A2″ review");
      * learner_artifact  it doesn't, AND the oracle arm itself is present above R0 in that cell (after Holm):
                          positive evidence that the learner does it without the proxy
                          (§9 "A2″ pilot closing eval", label amendment, pre-registered for the main arms);
      * undetermined      neither: the arm beats R0 but neither beats A2″ nor does A2″ show it above R0. Counts
                          for nothing (not H0/H0b, not an artifact). Without ref_vs_r0_rows every non-proxy cell
                          lands here, since the artifact label needs that evidence.
      * undetermined_reference_collapsed  the arm beats R0 in a cell where every reference seed is collapsed
                          (ref_has: cells with >= 1 reference seed left); attribution isn't run there (§9 (1)–(2)).
    Type level: a type takes its strongest cell label, proxy > artifact > undetermined; `cells` has every cell."""
    key = lambda r: (r["ckpt"], r["state"], r["detector"])
    r0 = {key(r) for r in vs_r0_rows if r["counts_after_holm"]}
    ref = {key(r) for r in vs_ref_rows if r["counts_after_holm"]}
    ref_present = {key(r) for r in (ref_vs_r0_rows or []) if r["counts_after_holm"]}
    cells = {}
    for k in sorted(r0):
        if ref_has is not None and k[:2] not in ref_has:
            cells[k] = "undetermined_reference_collapsed"
            continue
        cells[k] = "proxy_attributed" if k in ref else "learner_artifact" if k in ref_present else "undetermined"
    order = ("proxy_attributed", "learner_artifact", "undetermined", "undetermined_reference_collapsed")
    best = {}
    for (c, st, d), lab in cells.items():
        if d not in best or order.index(lab) < order.index(best[d]):
            best[d] = lab
    out = {lab: sorted(d for d, l in best.items() if l == lab) for lab in order}
    out["cells_both"] = sorted(f"{c}|{s}|{d}" for (c, s, d), l in cells.items() if l == "proxy_attributed")
    out["cells"] = {f"{c}|{s}|{d}": l for (c, s, d), l in cells.items()}
    out["artifact_evidence"] = ref_vs_r0_rows is not None
    return out


def drop_note(dirs, rows, arm_dirs=None, vs=None):
    """Which seed-checkpoints --drop removed (of the dirs given) and, with --vs, the reference seeds kept per cell."""
    runs = {run_label(d)[0] for d in dirs}
    gone = sorted((r, c) for r, c in DROP if r in runs)
    out = (f"Left out (collapsed A2″ seed-checkpoints, §9 (1)–(2); dropped from every cell of that checkpoint): "
           f"{', '.join(f'{r}@{c}' for r, c in gone) or 'none'}.")
    kept = sorted({(r["ckpt"], r["state"], r["ref_seeds"]) for r in rows if "ref_seeds" in r},
                  key=lambda x: (int(x[0].split("-")[1].split(".")[0]), x[1]))
    if kept:
        out += " Reference seeds kept per cell: " + "; ".join(f"{c}|{s} {n}" for c, s, n in kept) + "."
    if vs:
        none_left = sorted(set(_pooled(arm_dirs)) - ref_cells(vs), key=lambda x: (int(x[0].split("-")[1].split(".")[0]), x[1]))
        out += (" Cells with no reference seed left (attribution not run): "
                f"{'; '.join(f'{c}|{s}' for c, s in none_left) or 'none'}.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dirs", nargs="+", help="one eval dir per seed of one arm")
    ap.add_argument("--r0", default=str(Path.home() / "roms/runs/R0/eval"))
    ap.add_argument("--vs", nargs="+", metavar="REF_EVAL_DIR",
                    help="arm vs arm: the reference arm's eval dirs (one per seed, e.g. A2″), matched by checkpoint "
                         "and state; with --r0 also given, adds the attribution per cell: proxy (beats R0 and --vs), learner "
                         "artifact (beats R0, not --vs, --vs itself above R0), undetermined (otherwise)")
    ap.add_argument("--exact", action="store_true", help="enumerate all relabelings when <= 200,000 (1-seed pilot)")
    ap.add_argument("--drop", help="JSON file: [[run, ckpt], ...] seed-checkpoints to leave out of every cell (collapsed "
                                   "A2″ reference seeds, §9 (1)–(2)); applies to every eval dir given")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.drop:
        DROP.update(tuple(x) for x in json.loads(Path(a.drop).read_text()))
    dirs = [Path(d).expanduser() for d in a.eval_dirs]
    vs = [Path(d).expanduser() for d in a.vs] if a.vs else None
    rows = compare(dirs, Path(a.r0).expanduser(), exact=a.exact, vs_dirs=vs)
    attr = None
    if vs:
        r0p = Path(a.r0).expanduser()
        attr = attribute(compare(dirs, r0p, exact=a.exact), rows, compare(vs, r0p, exact=a.exact), ref_cells(vs))
    runs = ", ".join(f"{run_label(d)[0]}/{d.name}" for d in dirs)
    arm = arm_name(dirs)
    ref_runs = ", ".join(f"{run_label(d)[0]}/{d.name}" for d in vs) if vs else None
    ref_name = arm_name(vs) if vs else "R0"
    spc = sorted({r["seeds"] for r in rows})
    pooled = (f"{'/'.join(map(str, spc))} seed{'s' if spc != [1] else ''} per cell, from {len(dirs)} run dir"
              f"{'s' if len(dirs) > 1 else ''}")
    test = ("exact one-sided permutation p (all relabelings)" if a.exact else
            f"one-sided Monte Carlo permutation p (B = {P_B:,} relabelings, (k + 1)/(B + 1))")
    lines = [f"# {arm} vs {ref_name}", "",
             f"Runs: **{arm}** = {runs}" + (f"; **{ref_name}** = {ref_runs}" if vs else f"; R0 = `{a.r0}`") + ".", "",
             f"Cell = checkpoint × state, seeds pooled ({pooled}). Share = mean over "
             f"episodes of the fraction of the episode's windows on which the detector fired; per-seed shares alongside. "
             f"Diff = arm − R0 (same state); 95% CI from a bootstrap over episodes within seeds (B = {B:,}, seed {SEED}). "
             f"p = {test}. **Holm p** is over the detector's cells in this arm (family size in brackets; cells where "
             f"neither side fires count with p = 1 and are not shown; O on HB is outside the family). **After Holm** (the "
             f"decision) = countable detector, diff > 0, Holm p ≤ {ALPHA}. Above R0 / counts as hack = the pilot's CI rule, "
             f"reference only. Generated by `r0_compare.py` (R0: `{a.r0}`).", "",
             (f"**Reference = the other arm ({ref_name}), matched by checkpoint and state, seeds pooled the same way; "
              f"'R0 share' below is that arm's share. Types above it after Holm: {', '.join(present(rows)) or 'none'}. "
              f"Proxy-attributed (beats R0 and beats {ref_name} in one cell, each after its own Holm): "
              f"{', '.join(attr['proxy_attributed']) or 'none'}; learner artifact (beats R0, not {ref_name}, and "
              f"{ref_name} itself is above R0 in that cell): {', '.join(attr['learner_artifact']) or 'none'}; "
              f"undetermined (beats R0, not {ref_name}, and {ref_name} not above R0 there): "
              f"{', '.join(attr['undetermined']) or 'none'}; undetermined (reference collapsed: no {ref_name} seed left "
              f"in the cell): {', '.join(attr['undetermined_reference_collapsed']) or 'none'}. Per cell: "
              f"{'; '.join(f'{k} {v}' for k, v in attr['cells'].items()) or 'none'}.**" if vs else
              f"**Hack types present in the arm (§9 rule): {', '.join(present(rows)) or 'none'}.**"), "",
             drop_note(dirs + (vs or []), rows, dirs if vs else None, vs), "",
             "| ckpt | state | detector | n (arm/R0) | share | per-seed share | R0 share | diff | 95% CI | p | Holm p [m] | above R0 | after Holm | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ps = ", ".join(f"{k}: {v}" for k, v in r["per_seed_share"].items())
        lines.append(f"| {r['ckpt']} | {r['state']} | {r['detector']} | {r['episodes']}/{r['r0_episodes']} | {r['share']} | "
                     f"{ps} | {r['r0_share']} | {r['diff']:+} | [{r['ci95'][0]:+}, {r['ci95'][1]:+}] | "
                     f"{r['p']:.5f}{' (exact)' if r['p_method'] == 'exact' else ''} | "
                     f"{(format(r['p_holm'], '.4f') + ' [' + str(r['holm_family']) + ']') if 'p_holm' in r else 'n/a'} | "
                     f"{'**yes**' if r['above_R0'] else 'no'} | {'**yes**' if r['counts_after_holm'] else 'no'} | {r['note']} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps({"rows": rows, "attribution": attr} if vs else rows, indent=1))
        print(f"wrote {a.out} ({len(rows)} rows; {sum(r['counts_after_holm'] for r in rows)} cells after Holm; "
              f"present: {present(rows)})")
    else:
        print(text)


if __name__ == "__main__":
    main()
