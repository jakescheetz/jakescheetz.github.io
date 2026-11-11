#!/usr/bin/env python3
"""Batch-1 dev-cell pipeline (experiment-e1.md §9 "Batch-1 analysis plan" B): inventory and checks, then the
R0 and A0-vs-A2″ detector tables per training state, on the DEV VIEWS only (seal_b1.py: eval-dev/, R0
eval-dev-24/). Held-out cells are never read here.

  .venv/bin/python analyze_b1.py --dry --vm e1-a --out results/dry-b1      # dry run: e1-a's 12 runs
  .venv/bin/python analyze_b1.py --out results/b1                           # all 23 runs
  .venv/bin/python analyze_b1.py --batch b2 --out results/b2                # batch 2 (§9 H6 "Pipeline")

--batch b2 (§9 H6): the same pipeline on batch 2's dev views (seal_b2.py: eval-dev*/ hold the 7 tested marks;
ckpt-0 in eval-dev-mark0/), R0's batch-1 dev rows as the base rate (R0 is random, so obs-independent). The
checklist gains: obs_version 2 in every row, the 0 mark present, KL from the saved ckpt-0 (bit-exact), the
manipulation check, the descriptive metrics (b2_extras.py), and the §9 H11 lines (kept A2″ seeds, the
unfiltered sensitivity line, never-left-start-map counts for both batches). Held-out part: results/sealed-b2/b2/.

Writes into --out (every file headed "DRY RUN: n/23 runs, no conclusions" with --dry):
  checklist.md        the §9 B pass/fail list, one line per check, with counts
  cells.md / .json    every (arm, training state, seed, mark, eval state) cell: n, T mean, P mean, leg progress,
                      KL vs init (where logged), vm
  vs-R0-<arm>-primary.md/.json     r0_compare.py: the arm's D1- and D2-trained runs, each on its own state, vs R0;
                                   Holm per detector over these 14 cells (§9 A′; family size in brackets)
  vs-R0-<arm>-cross.md/.json       the same on the other dev state: a separate, descriptive family
  attr-A0-vs-A2pp-{primary,cross}  r0_compare.py --vs: A0 vs A2″ + attribution labels
  fig1.svg / .md / .json           Fig. 1 (§9 C): P over T + leg progress, vs step (log) and vs KL, per training
                                   state on its own eval state; seed-bootstrap CIs, per-seed lines, collapsed A2″
                                   seed-ckpts hollow, first-appearance ticks (pooled + per seed); .md = table view
  h0.md / .json                    H0 (i)–(iii) per training state + verdict (§9 D); leg progress secondary
  catalogue-fields.md / .json      §9 E per type × mark: prevalence, share of P (from traces), first appearance
                                   pooled + per seed, attribution, round-1 ranking (§9 F)
Every JSON gets "dry_run": "n/23" with --dry (list JSONs become {"dry_run", "rows"}).

Held-out part (never in --out): results/sealed-b1/{dry-b1,b1}/ gets vs-R0-<arm>-heldout, fig1-heldout and
catalogue-heldout (U recomputed over held-out windows only). Only counts are printed. Re-run
`seal_b1.py split` afterwards so the manifest covers the new sealed files. Choices the plan leaves open
are listed in b1_report.py's docstring.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from pokered import runlog  # noqa: E402
import seal_b1  # noqa: E402
import seal_b2  # noqa: E402
import b1_report  # noqa: E402
import b2_extras  # noqa: E402

RUNS = seal_b1.RUNS
BATCH = {"b1": {"name": "Batch-1", "runs": seal_b1.batch1_runs, "sealed_raw": seal_b1.SEALED_RAW, "held_src": "eval",
                "sealed_root": "sealed-b1", "seal_tool": "seal_b1.py"},
         "b2": {"name": "Batch-2", "runs": seal_b2.batch2_runs, "sealed_raw": seal_b2.SEALED_RAW, "held_src": "eval-heldout-test",
                "sealed_root": "sealed-b2", "seal_tool": "seal_b2.py"}}
CFG = BATCH["b1"]
MARKS = 7
DEV = ("D1", "D2")


def t_from_milestones(ms):
    """T implied by an episode's milestone_steps keys ('M1', 'M2_route2', ...): the highest milestone index."""
    return max((int(k[1]) for k in ms if k[:1] == "M" and k[1:2].isdigit()), default=0)


def load(p):
    j = json.loads(Path(p).read_text())
    return j["rows"] if isinstance(j, dict) and "rows" in j and "attribution" not in j else j


def trace_of(src):
    return lambda run, e: RUNS / run / src / "traces" / f"{Path(e['ckpt']).stem}-{e['state']}-{e['seed']}.jsonl.gz"


def reference_seeds(info, coll, view):
    """{(ckpt, eval state): A2″ seeds left after dropping collapsed seed-ckpts} for the primary or cross view."""
    out = Counter()
    for run, d in info.items():
        if d["arm"] != "A2pp" or d["state"] not in DEV:
            continue
        for ck, st in {(e["ckpt"], e["state"]) for e in d["eps"]}:
            if (st == d["state"]) == (view == "primary") and (run, e_step(ck)) not in coll:
                out[(ck, st)] += 1
    cells = {(e["ckpt"], e["state"]) for run, d in info.items() if d["arm"] == "A2pp" and d["state"] in DEV
             for e in d["eps"] if (e["state"] == d["state"]) == (view == "primary")}
    return {c: out.get(c, 0) for c in cells}


def empty_cells(info, coll, view):
    return sorted(c for c, n in reference_seeds(info, coll, view).items() if n == 0)


def e_step(ckpt):
    return int(ckpt.split("-")[1].split(".")[0])


def report_parts(info, out, r0, kl_bad, check):
    """§9 C (Fig. 1), D (H0), E (catalogue fields) on the dev views."""
    rows = {run: d["eps"] for run, d in info.items()}
    own = {run: [e for e in d["eps"] if e["state"] == d["state"]] for run, d in info.items()}
    coll = b1_report.collapsed(info, rows)
    r0p = load(out / "vs-R0-A0-primary.json")
    pooled_first = b1_report.first_appearance(r0p, DEV)
    seed_first = b1_report.per_seed_first(info, lambda r: RUNS / r / "eval-dev-primary", r0)
    ticks = {}
    for (ts, t), step in pooled_first.items():
        if t in b1_report.HACKABLE:
            ticks.setdefault(f"{ts}→{ts}", []).append((step, t, 1))
    for (run, t), step in seed_first.items():
        if t in b1_report.HACKABLE:
            ticks.setdefault(f"{info[run]['state']}→{info[run]['state']}", []).append(
                (step, f"{t}·s{info[run]['seed']}", 2))
    kl_cap = ["x = KL: init rebuilt from each run's seed with its training thread count and checked by hash (one hash",
              "per seed, shared by A0 and A2″); verified by reproducibility, not against a stored step-0 checkpoint.",
              f"Runs left out of the KL panel: {', '.join(kl_bad) or 'none'}."]
    data = b1_report.fig1(info, rows, out, "fig1", [(s, s) for s in DEV], kl_bad, coll, ticks, True, kl_cap)
    n_pts = sum(len(a["marks"]) for c in data.values() for a in c.values())
    check("Fig. 1 (§9 C): P, T, leg progress vs step and vs KL, A0 + A2″, per training state", n_pts > 0,
          f"panels {list(data)}; {n_pts} pooled points; collapsed A2″ seed-ckpts {len(coll)}"
          f"{' (' + ', '.join(f'{r}@{s:,}' for r, s in sorted(coll)) + ')' if coll else ''}; "
          f"first-appearance ticks pooled {sum(v[2] == 1 for t in ticks.values() for v in t)}, "
          f"per seed {sum(v[2] == 2 for t in ticks.values() for v in t)}")
    h0 = b1_report.h0_tests(info, rows, coll)
    (out / "h0.json").write_text(json.dumps(h0, indent=1))
    (out / "h0.md").write_text(b1_report.h0_md(h0))
    check("H0 (i)–(iii) per training state (§9 D), MC permutation B = 100k, Holm over 4 marks ≥ 2M",
          len(h0["states"]) == len({d["state"] for d in info.values()} & set(DEV)) and
          all(len(st["T"]["ii"]) == 4 for st in h0["states"].values()),
          f"states {list(h0['states'])}; marks ≥ 2M {h0['marks_ge_2M']}; verdicts computed (values in h0.md)")
    attr = load(out / "attr-A0-vs-A2pp-primary.json")["attribution"]
    r0_dev = RUNS / "R0" / "eval-dev-24" / "episodes.jsonl"
    r0_eps = [json.loads(l) for l in open(r0_dev)] if r0_dev.exists() else []
    cat = b1_report.catalogue(info, own, trace_of("eval-dev"), r0p, attr, seed_first, pooled_first, DEV, r0_eps)
    (out / "catalogue-fields.json").write_text(json.dumps(cat, indent=1))
    (out / "catalogue-fields.md").write_text(b1_report.catalogue_md(cat, "Hack catalogue fields (§9 E), dev",
                                                                     "on HA/HB"))
    n_ep = next(c["episodes"] for x in cat["types"] for c in x["cells"] if c["mark"] == "all")
    check("catalogue fields (§9 E): prevalence + share of P per type × mark, first appearance pooled + per seed",
          not cat["traces_missing"] and not cat["trace_P_mismatch"],
          f"{n_ep} A0 own-state episodes; traces missing {len(cat['traces_missing'])}; trace ΣP ≠ P "
          f"{cat['trace_P_mismatch']}; per-seed first appearance from {len(seed_first)} (run, type) hits")


def sealed_part(info, sout, check):
    """Held-out panels and columns (§9 A2): written only under results/sealed-b1/; only counts are printed."""
    # D1/D2-trained runs only: the A2″-HA runs train on a held-out state and are not part of these families
    held_runs = {run: d for run, d in info.items() if d["state"] in DEV and
                 (CFG["sealed_raw"] / run / CFG["held_src"] / "episodes.jsonl").exists()}
    r0h = seal_b1.SEALED_RAW / "R0" / "eval-24"
    if not held_runs or not (r0h / "episodes.jsonl").exists():
        check("sealed part (HA/HB) runs", False, "no sealed rows found")
        return
    sout.mkdir(parents=True, exist_ok=True)
    held = {}
    for run in held_runs:                  # U over held-out windows only (§9 A′ (2): computed inside the sealed set)
        src = CFG["sealed_raw"] / run / CFG["held_src"]
        eps, wins = seal_b1.rows(src / "episodes.jsonl"), seal_b1.rows(src / "windows.jsonl")
        seal_b1.recompute_u(eps, wins)
        held[run] = eps
    info_h = {run: {**d, "eps": held[run]} for run, d in held_runs.items()}
    coll = b1_report.collapsed(info, {r: d["eps"] for r, d in info.items()})        # collapse is a dev-state property
    drop = sout / "collapsed_a2pp.json"
    drop.write_text(json.dumps(sorted([r, f"eval-{s}.pt"] for r, s in coll)))
    files = []
    for arm in ("A0", "A2pp"):
        dirs = [CFG["sealed_raw"] / r / CFG["held_src"] for r, d in held_runs.items() if d["arm"] == arm]
        if dirs:
            f = sout / f"vs-R0-{arm}-heldout.md"
            subprocess.run([sys.executable, str(CODE / "r0_compare.py"), *map(str, dirs), "--r0", str(r0h),
                            "--drop", str(drop), "--out", str(f)], check=True, capture_output=True, text=True)
            files.append(f.name)
    r0a = load(sout / "vs-R0-A0-heldout.json")
    cells = [(ts, es) for ts in DEV for es in sorted(seal_b1.HELD) if any(d["state"] == ts for d in info_h.values())]
    data = b1_report.fig1(info_h, held, sout, "fig1-heldout", cells, [], coll, {}, False,
                          ["Held-out panels: no first-appearance ticks (held-out families are separate, see "
                           "vs-R0-*-heldout)."])
    first = b1_report.first_appearance(r0a, sorted(seal_b1.HELD))
    r0_eps = [e for e in seal_b1.rows(r0h / "episodes.jsonl")]
    own = {run: [e for e in eps if e["state"] in seal_b1.HELD] for run, eps in held.items()}
    held_trace = (trace_of("eval") if CFG is BATCH["b1"] else
                  lambda run, e: CFG["sealed_raw"] / run / "eval-heldout" / "traces" /
                  f"{Path(e['ckpt']).stem}-{e['state']}-{e['seed']}.jsonl.gz")
    cat = b1_report.catalogue(info_h, own, held_trace, r0a, None, {}, first, sorted(seal_b1.HELD), r0_eps)
    pres = {r["detector"] for r in r0a if r.get("counts_after_holm")}
    for x in cat["types"]:
        x["sealed"] = "present" if x["type"] in pres else "no"
    (sout / "catalogue-heldout.json").write_text(json.dumps(cat, indent=1))
    (sout / "catalogue-heldout.md").write_text(b1_report.catalogue_md(cat, "Hack catalogue, held-out (SEALED)",
                                                                       "present on HA/HB"))
    files += ["fig1-heldout.{svg,md,json}", "catalogue-heldout.{md,json}"]
    n = sum(len(v) for v in own.values())
    check(f"sealed part (HA/HB): held-out Fig. 1 panels, vs-R0 families, catalogue column, under results/{CFG['sealed_root']}/",
          not cat["traces_missing"] and not cat["trace_P_mismatch"] and len(data) == len(cells),
          f"{len(held_runs)} runs, {n} held-out rows, {len(cells)} panels, traces missing "
          f"{len(cat['traces_missing'])}; files {', '.join(files)} in {sout.relative_to(CODE)} (counts only)")


def b2_checks(info, check):
    """§9 H6 'Pipeline' checklist lines for batch 2 (dev rows; mark 0 from eval-dev-mark0/)."""
    m0 = {run: [json.loads(l) for l in open(RUNS / run / "eval-dev-mark0" / "episodes.jsonl")]
          if (RUNS / run / "eval-dev-mark0" / "episodes.jsonl").exists() else [] for run in info}
    allrows = [e for run, d in info.items() for e in d["eps"] + m0[run]]
    obs = Counter(e.get("obs_version") for e in allrows)
    check("obs_version 2 in every dev row (marks 0-5M)", set(obs) == {2}, f"obs_version counts {dict(obs)}")
    bad0 = {run: Counter(e["state"] for e in rs) for run, rs in m0.items()
            if Counter(e["state"] for e in rs) != Counter({"D1": 8, "D2": 8})}
    check("the 0 mark (ckpt-0) present: D1 and D2 × 8 per run", not bad0,
          f"{sum(len(v) for v in m0.values())} mark-0 rows over {len(m0)} runs; wrong: {sorted(bad0) or 'none'}")
    kl0 = sum(str(e.get("init_sha256", "")).endswith("=ckpt-0") for e in allrows)
    z0 = [e["kl_init"] for rs in m0.values() for e in rs if "kl_init" in e]
    check("KL from the saved ckpt-0 (bit-exact): init hash checked against ckpt-0 in every row, KL = 0 at mark 0",
          kl0 == len(allrows) and z0 and all(x == 0 for x in z0),
          f"{kl0}/{len(allrows)} rows '=ckpt-0'; mark-0 KL values {sorted(set(z0))}")
    for run, d in info.items():             # mark-0 rows: descriptive only (never-left counts), not in any test
        d["mark0"] = m0[run]


def b2_part(info, out, coll, check):
    """§9 H5 manipulation check + descriptive metrics, §9 H11 kept seeds / sensitivity / never-left counts."""
    man = b2_extras.manipulation(info)
    (out / "manipulation.json").write_text(json.dumps(man, indent=1))
    check("manipulation check (§9 H5): A0 pooled P at its best mark ≥ 2× batch 1's on both dev states (D1 ≥ 111, D2 ≥ 101)",
          len(man["states"]) == 2, "computed for " + ", ".join(man["states"]) + "; pass/fail and values in manipulation.json")
    kept = b2_extras.kept_seeds(info, coll)
    rows = {run: d["eps"] for run, d in info.items()}
    h0f = json.loads((out / "h0.json").read_text())
    h0u = b1_report.h0_tests(info, rows, set(), metrics=(("T", "primary"),))
    sens = b2_extras.sensitivity(h0f, h0u, kept)
    (out / "kept-seeds.json").write_text(json.dumps(kept, indent=1))
    (out / "sensitivity.json").write_text(json.dumps(sens, indent=1))
    (out / "sensitivity.md").write_text(b2_extras.sensitivity_md(sens))
    check("§9 H11: kept A2″ seeds per mark ≥ 2M and the unfiltered A0-vs-A2″ T line ('sensitivity, not the result', "
          "uncorrected p, no Holm)", all(len(v) == 4 for v in kept.values()) and len(sens["rows"]) == 8,
          "kept seeds: " + "; ".join(f"{ts} " + " ".join(f"{int(m) / 1e6:g}M:{len(s)}" for m, s in v.items())
                                     for ts, v in kept.items()) + "; table in sensitivity.md")
    b1_info = {}
    for run, vm in seal_b1.batch1_runs():
        f = RUNS / run / "eval-dev" / "episodes.jsonl"
        if f.exists():
            cfg = json.loads((RUNS / run / "config.json").read_text())
            b1_info[run] = {"arm": cfg["arm"], "state": cfg["state"], "eps": [json.loads(l) for l in open(f)]}
    r0f = RUNS / "R0" / "eval-dev-24" / "episodes.jsonl"
    r0 = {"R0": {"arm": "R0", "state": "R0", "eps": [json.loads(l) for l in open(r0f)]}} if r0f.exists() else {}
    b2_all = {run: {**d, "eps": d["eps"] + d.get("mark0", [])} for run, d in info.items()}
    nl = (b2_extras.never_left(b2_all, "b2") + b2_extras.never_left(b1_info, "b1") + b2_extras.never_left(r0, "R0"))
    (out / "never-left-start.json").write_text(json.dumps(nl, indent=1))
    (out / "never-left-start.md").write_text(b2_extras.never_left_md(nl))
    check("§9 H11: never-left-start-map count per arm × mark, both batches + R0 (descriptive)", bool(nl),
          f"{len(nl)} cells (b2 {sum(r['batch'] == 'b2' for r in nl)}, b1 {sum(r['batch'] == 'b1' for r in nl)}, "
          f"R0 {sum(r['batch'] == 'R0' for r in nl)}); table in never-left-start.md")
    desc = b2_extras.descriptive(info, trace_of("eval-dev"))
    (out / "descriptive.json").write_text(json.dumps(desc, indent=1))
    (out / "descriptive.md").write_text(b2_extras.descriptive_md(desc))
    check("descriptive metrics (§9 H5): post-progress share of P, coverage per map vs the G5 planner",
          not desc["traces_missing"] and not desc["trace_P_mismatch"],
          f"{desc['episodes']} own-state episodes in {len(desc['cells'])} cells; traces missing {desc['traces_missing']}; "
          f"trace ΣP ≠ P {desc['trace_P_mismatch']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vm", help="only this VM's runs (dry run: e1-a)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", choices=sorted(BATCH), default="b1")
    a = ap.parse_args()
    global CFG
    CFG = BATCH[a.batch]
    runs = [(r, vm) for r, vm in CFG["runs"]() if not a.vm or vm == a.vm]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    head = f"**DRY RUN: {len(runs)}/23 runs, no conclusions.**\n\n" if a.dry else ""
    checks, cells, info = [], [], {}
    for run, vm in runs:
        cfg = json.loads((RUNS / run / "config.json").read_text())
        dev = RUNS / run / "eval-dev"
        eps = [json.loads(l) for l in open(dev / "episodes.jsonl")] if (dev / "episodes.jsonl").exists() else []
        info[run] = {"arm": cfg["arm"], "state": cfg["state"], "seed": cfg["seed"], "vm": vm, "eps": eps,
                     "cfg": cfg, "dev": dev}
    # --- checks ---
    def check(name, ok, detail):
        checks.append((name, ok, detail))

    by_cell = defaultdict(list)
    for run, d in info.items():
        for e in d["eps"]:
            by_cell[(run, e["step"], e["state"])].append(e)
    missing_states = {run: sorted(set(DEV) - {e["state"] for e in d["eps"]}) for run, d in info.items()}
    check("dev view exists for every run", all(d["eps"] for d in info.values()),
          f"{sum(bool(d['eps']) for d in info.values())}/{len(info)} runs have eval-dev rows")
    check("both dev states (D1, D2) evaluated", not any(missing_states.values()),
          "; ".join(f"{r}: missing {m}" for r, m in missing_states.items() if m) or "all runs have D1 and D2")
    ns = Counter(len(v) for v in by_cell.values())
    check("every (run, mark, eval state) cell has n = 8", set(ns) == {8}, f"cell sizes {dict(ns)}")
    marks = {run: sorted({e["step"] for e in d["eps"]}) for run, d in info.items()}
    bad_marks = {r: m for r, m in marks.items() if len(m) != MARKS or not any(2_900_000 < s < 3_100_000 for s in m)}
    check("7 marks per run, including 3M", not bad_marks,
          "; ".join(f"{r}: {len(m)} marks" for r, m in bad_marks.items()) or
          f"all runs: {MARKS} marks, e.g. {marks[next(iter(marks))]}")
    pairs = defaultdict(set)
    for run, d in info.items():
        pairs[(d["seed"], d["state"])].add(d["arm"])
    unpaired = {k: v for k, v in pairs.items() if k[1] in DEV and v != {"A0", "A2pp"}}
    same_vm = all(len({info[r]["vm"] for r in info if (info[r]["seed"], info[r]["state"]) == k}) == 1 for k in pairs)
    check("A0/A2″ paired by seed × training state (same VM)", not unpaired and same_vm,
          f"{sum(v == {'A0', 'A2pp'} for v in pairs.values())} pairs; unpaired {sorted(unpaired) or 'none'}; "
          f"pairs on one VM: {same_vm}")
    lp = sum("leg_progress" in e and e["leg_progress"] is not None for d in info.values() for e in d["eps"])
    n_eps = sum(len(d["eps"]) for d in info.values())
    check("leg progress present in every dev row", lp == n_eps, f"{lp}/{n_eps} rows")
    tm = [(r, e["ckpt"], e["state"], e["seed"]) for r, d in info.items() for e in d["eps"]
          if e["T"] != t_from_milestones(e["milestone_steps"])]
    check("T in eval rows matches milestone_steps", not tm, f"{len(tm)} mismatches" + (f", e.g. {tm[:3]}" if tm else ""))
    vmr = sum(e.get("vm") == info[r]["vm"] for r, d in info.items() for e in d["eps"])
    check("VM name in every row", vmr == n_eps, f"{vmr}/{n_eps} rows")
    kept = {}
    for run, d in info.items():
        raw = [json.loads(l) for l in open(RUNS / run / "episodes.jsonl")]
        k = runlog.load_kept(RUNS / run, "episodes.jsonl")
        kept[run] = (len(raw), len(k), d["cfg"].get("resumes", 0))
    check("runlog.load_kept applied to training episodes (resume markers / discarded branches dropped)", True,
          "; ".join(f"{r}: {n} rows -> {k} kept, {res} resumes" for r, (n, k, res) in kept.items()))
    kl = sum("kl_init" in e for d in info.values() for e in d["eps"])
    shas = {e["init_sha256"] for d in info.values() for e in d["eps"] if "init_sha256" in e}
    check("KL vs init logged (D2-trained runs on D2; D1-trained need the D1 KL pass)", kl > 0,
          f"{kl} rows with kl_init; {len(shas)} distinct init hashes")
    # §9 A′ KL rule: the init must be rebuilt with the training thread count on the training VM type; runs where it
    # isn't (or whose rebuild differs between rows) are left out of the KL panel and listed in its caption
    vm_threads = {"e1-a": 8, "e1-b": 4}
    kl_bad = sorted({r for r, d in info.items() for e in d["eps"] if "kl_init" in e and
                     (e.get("init_threads") != (d["cfg"].get("train_threads") or vm_threads[d["vm"]]) or
                      len({x["init_sha256"] for x in d["eps"] if "init_sha256" in x}) != 1)})
    check("KL init rebuilt with each run's training thread count (e1-a 8, e1-b 4), one hash per run", not kl_bad,
          f"runs to leave out of the KL panel: {kl_bad or 'none'}")
    (out / "kl_excluded.json").write_text(json.dumps(kl_bad))
    if a.batch == "b2":
        b2_checks(info, check)
    # --- cells table ---
    agg = defaultdict(list)
    for run, d in info.items():
        for e in d["eps"]:
            agg[(d["arm"], d["state"], d["seed"], e["step"], e["state"], d["vm"])].append(e)
    mean = lambda v: round(sum(v) / len(v), 4) if v else None
    for (arm, st, seed, step, es, vm), es_ in sorted(agg.items()):
        cells.append({"arm": arm, "train_state": st, "seed": seed, "mark": step, "eval_state": es, "vm": vm,
                      "n": len(es_), "T_mean": mean([e["T"] for e in es_]), "P_mean": mean([e["P"] for e in es_]),
                      "leg_progress_mean": mean([e["leg_progress"] for e in es_ if e.get("leg_progress") is not None]),
                      "kl_init_mean": mean([e["kl_init"] for e in es_ if "kl_init" in e]),
                      "primary": es == st})
    (out / "cells.json").write_text(json.dumps(cells, indent=1))
    md = [f"# {CFG['name']} dev cells", "", "One row per (arm, training state, seed, mark, eval state); dev states only. "
          "Primary = eval state is the run's own training state.", "",
          "| arm | train | seed | mark | eval | primary | n | T mean | P mean | leg progress | KL init | vm |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        md.append(f"| {c['arm']} | {c['train_state']} | {c['seed']} | {c['mark']:,} | {c['eval_state']} | "
                  f"{'yes' if c['primary'] else ''} | {c['n']} | {c['T_mean']} | {c['P_mean']} | "
                  f"{c['leg_progress_mean']} | {c['kl_init_mean'] if c['kl_init_mean'] is not None else ''} | {c['vm']} |")
    (out / "cells.md").write_text("\n".join(md) + "\n")
    # --- detector tables (r0_compare on dev views) ---
    r0 = RUNS / "R0" / "eval-dev-24"
    r0_states = {json.loads(l)["state"] for l in open(r0 / "windows.jsonl")} if (r0 / "windows.jsonl").exists() else set()
    check("R0 dev base rates (24 episodes) for D1 and D2", set(DEV) <= r0_states, f"R0 dev states {sorted(r0_states)}")
    # collapsed A2″ seed-checkpoints (§9 (1)–(2), B′ (b)): judged on the run's own training state, dropped from
    # the A2″ reference in every cell of that checkpoint (primary and cross), in attribution and A2″-vs-R0 alike
    coll = b1_report.collapsed(info, {run: d["eps"] for run, d in info.items()})
    drop = out / "collapsed_a2pp.json"
    drop.write_text(json.dumps(sorted([r, f"eval-{s}.pt"] for r, s in coll)))
    droparg = ["--drop", str(drop)]
    fam, ref_seed_cells = [], {}
    for view in ("primary", "cross"):     # §9 A′: Holm per detector × arm over the 14 primary cells; cross apart
        groups = {arm: [RUNS / r / f"eval-dev-{view}" for r in info
                        if info[r]["arm"] == arm and info[r]["state"] in DEV] for arm in ("A0", "A2pp")}
        for arm, dirs in groups.items():
            if dirs and r0_states:
                f = out / f"vs-R0-{arm}-{view}.md"
                subprocess.run([sys.executable, str(CODE / "r0_compare.py"), *map(str, dirs), "--r0", str(r0),
                                *droparg, "--out", str(f)], check=True, capture_output=True, text=True)
                rows = json.loads(f.with_suffix(".json").read_text())
                fam += [(arm, view, r["detector"], r.get("holm_family")) for r in rows]
        if groups["A0"] and groups["A2pp"] and r0_states:
            f = out / f"attr-A0-vs-A2pp-{view}.md"
            subprocess.run([sys.executable, str(CODE / "r0_compare.py"), *map(str, groups["A0"]), "--vs",
                            *map(str, groups["A2pp"]), "--r0", str(r0), *droparg, "--out", str(f)],
                           check=True, capture_output=True, text=True)
            aj = json.loads(f.with_suffix(".json").read_text())
            labs = aj["attribution"]
            ref_seed_cells[view] = sorted({(r["ckpt"], r["state"], r["ref_seeds"]) for r in aj["rows"]})
            check(f"attribution labels come out ({view}{'' if view == 'primary' else ', descriptive'})", True,
                  "; ".join(f"{k}: {len(labs[k])} types" for k in
                            ("proxy_attributed", "learner_artifact", "undetermined",
                             "undetermined_reference_collapsed")))
    fams = sorted({(a_, s_, d_, m_) for a_, s_, d_, m_ in fam if m_})
    prim = sorted({m_ for a_, v_, d_, m_ in fams if v_ == "primary" and a_ == "A0"})
    n_a2 = 14 - sum(1 for c in empty_cells(info, coll, "primary"))
    prim2 = sorted({m_ for a_, v_, d_, m_ in fams if v_ == "primary" and a_ == "A2pp"})
    check("Holm family = detector × arm over the 14 primary dev cells (§9 A′); cross cells a separate family",
          prim == [14] and prim2 in ([n_a2], []), f"A0 primary family sizes {prim}; A2″ {prim2} (14 minus {14 - n_a2} cells "
          f"with every A2″ seed collapsed); cross family sizes {sorted({m_ for a_, v_, d_, m_ in fams if v_ == 'cross'})}")
    expect = {view: {c: n for c, n in reference_seeds(info, coll, view).items()} for view in ref_seed_cells}
    got = {view: {(c, s_): n for c, s_, n in v} for view, v in ref_seed_cells.items()}
    ok = all(got[v][k] == expect[v][k] for v in got for k in got[v])
    check("collapsed A2″ seed-ckpts dropped from the A2″ reference in attribution and A2″-vs-R0, every cell of that "
          "checkpoint (§9 (1)–(2), B′ (b))", ok and bool(ref_seed_cells),
          f"{len(coll)} seed-ckpts dropped ({', '.join(f'{r}@{s_:,}' for r, s_ in sorted(coll)) or 'none'}); reference seeds "
          f"kept per primary cell {sorted(Counter(n for (_, _), n in got.get('primary', {}).items()).items())} (seeds: cells); "
          f"cells with none left: primary {len(empty_cells(info, coll, 'primary'))}, cross {len(empty_cells(info, coll, 'cross'))} "
          f"(labelled 'undetermined (reference collapsed)' where A0 is present); kept counts match the dev rows: {ok}")
    pools = {tuple(e.get("u_pool") or ()) for d in info.values() for e in d["eps"]}
    check("U decile over dev windows only (D1 + D2 per checkpoint, §9 A′)", pools == {("D1", "D2")},
          f"U pools in dev rows: {sorted(pools)}")
    check("O on HB excluded from the family", True, "HB is held-out: not in these dev tables (r0_compare "
          "DESCRIPTIVE = {(O, HB)} applies in the sealed tables)")
    report_parts(info, out, r0, kl_bad, check)
    if a.batch == "b2":
        b2_part(info, out, coll, check)
    sealed_out = CODE / "results" / CFG["sealed_root"] / (f"dry-{a.batch}" if a.dry else a.batch)
    sealed_part(info, sealed_out, check)
    tag = f"{len(runs)}/23"
    for d in (out, sealed_out):
        for f in d.glob("*.md"):
            if a.dry and not f.read_text().startswith("**DRY RUN"):
                f.write_text(head + f.read_text())
        for f in d.glob("*.json"):
            if a.dry:
                j = json.loads(f.read_text())
                if not (isinstance(j, dict) and "dry_run" in j):
                    j = {"dry_run": tag, **j} if isinstance(j, dict) else {"dry_run": tag, "rows": j}
                    f.write_text(json.dumps(j, indent=1))
        for f in d.glob("*.svg"):
            t = f.read_text()
            if a.dry and "DRY RUN" not in t:
                t = t.replace('font-weight="bold">Fig. 1', f'font-weight="bold">DRY RUN {tag}, no conclusions. Fig. 1', 1)
                f.write_text(t)
    lines = [head + f"# {CFG['name']} pipeline checklist ({len(runs)} runs{', ' + a.vm if a.vm else ''})", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else '**FAIL**'} | {d} |" for n, ok, d in checks]
    (out / "checklist.md").write_text("\n".join(lines) + "\n")
    for n, ok, d in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {n}: {d[:160]}")


if __name__ == "__main__":
    main()
