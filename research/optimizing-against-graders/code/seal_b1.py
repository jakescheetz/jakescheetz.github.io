#!/usr/bin/env python3
"""Batch-1 blinding (experiment-e1.md §9 "Batch-1 analysis plan" A2, §7.3): split every batch-1 eval into a dev
view (D1, D2) and a sealed held-out part (HA, HB), and hash everything held-out so it can be shown untouched.

  .venv/bin/python seal_b1.py split        build the views and write the manifest (re-runnable; idempotent)
  .venv/bin/python seal_b1.py verify       re-hash every sealed file against the manifest

Inputs: the batch-1 runs in gcp/plans/main-b1-e1{a,b}.txt (+ R0), with their evaluate.py outputs in ~/roms/runs:
`eval/` (D1 HA HB), `eval-d2/` (D2), R0's `eval-24/` and `eval-24-d2/`, and the D1-trained runs' `eval-kl-d1/`
(a D1 replay with --kl-init: its KL fields are merged onto the dev D1 rows only if T/P/final_pos are identical).

- **Dev view** (what the researcher reads): `~/roms/runs/<run>/eval-dev/` (R0: `eval-dev-24/`) with
  episodes.jsonl and windows.jsonl holding only D1/D2 rows (each row also gets `vm` and `source`),
  summary.json with only the D1/D2 cells, and traces/ as symlinks to the D1/D2 trace files.
- **Sealed raw:** HA/HB rows and cells go to `~/roms/sealed-b1/<run>/<source>/` (outside the repo; raw rows are
  too big for it). Derived held-out outputs (r0_compare cells, Fig. 1 held-out panels, catalogue columns) go to
  `code/results/sealed-b1/`.
- **The mixed originals** (`eval/`, `eval-24/`: D1 + HA + HB) and the HA/HB trace files stay where evaluate.py
  wrote them and count as sealed: they are hashed in the manifest, and the researcher doesn't open them.
- **U on dev (§9 A′ ruling 2):** evaluate.py set the U decile per call, over D1 + HA + HB windows in `eval/`.
  The dev view recomputes U offline per checkpoint over the run's **dev windows only** (D1 + D2): U is stripped
  from every dev window and episode, the decile is taken over that pool, apply_u runs, and U events are re-merged.
  Rows get `u_threshold` (the dev one) and `u_pool` (the states pooled). Dev summary cells drop their old U fields.
- **Primary / cross views (§9 A′ ruling 1):** `eval-dev-primary/` (own training state only) and `eval-dev-cross/`
  (the other dev state) per run, for the Holm families: primary = the 14 own-state cells per detector × arm,
  cross = a separate descriptive family.
- **Manifest:** `code/results/sealed-b1.sha256` (sha256 + path from ~, one line per sealed file). Its own sha256
  is printed for BOARD. `verify` reports any changed, missing or new sealed file.
- This prints only counts (files, rows, cells, n per cell): never a held-out value.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from pokered import detectors  # noqa: E402
HOME = Path.home()
RUNS = HOME / "roms" / "runs"
SEALED_RAW = HOME / "roms" / "sealed-b1"
SEALED_OUT = CODE / "results" / "sealed-b1"
MANIFEST = CODE / "results" / "sealed-b1.sha256"
DEV, HELD = {"D1", "D2"}, {"HA", "HB"}
PLANS = {"e1-a": CODE / "gcp/plans/main-b1-e1a.txt", "e1-b": CODE / "gcp/plans/main-b1-e1b.txt"}


def batch1_runs():
    out = []
    for vm, plan in PLANS.items():
        for line in plan.read_text().splitlines():
            line = line.split("#")[0]
            if "--run-id" in line:
                out.append((line.split("--run-id")[1].split()[0], vm))
    return out


def sources(run):
    """(source eval dir, dev view name) for a run."""
    if run == "R0":
        return [RUNS / "R0" / "eval-24", RUNS / "R0" / "eval-24-d2"], "eval-dev-24"
    return [RUNS / run / "eval", RUNS / run / "eval-d2"], "eval-dev"


def rows(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def write_rows(path, rs):
    with open(path, "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")


def recompute_u(eps, wins):
    """U over dev windows only, per checkpoint (§9 A′): strip U, take the decile over the pool, re-apply, re-merge."""
    by_ck = {}
    for w in wins:
        w["fired"] = [t for t in w["fired"] if t != "U"]
        w.pop("u_kind", None)
        w["evidence"] = {}
        by_ck.setdefault(w["ckpt"], []).append(w)
    thr = {ck: detectors.u_threshold(ws) for ck, ws in by_ck.items()}
    for ck, ws in by_ck.items():
        detectors.apply_u(ws, thr[ck])
    u_by_ep = {}
    for w in wins:
        if "U" in w["fired"]:
            u_by_ep.setdefault((w["ckpt"], w["state"], w["seed"]), []).append(
                {"type": "U", "step": w["start"], "end": w["end"], "evidence": {"kind": w["u_kind"]}})
        w.pop("evidence")
    pool = sorted({w["state"] for w in wins})
    for e in eps:
        u = detectors.merge(u_by_ep.get((e["ckpt"], e["state"], e["seed"]), []))
        ev = [x for x in e["events"] if x["type"] != "U"] + \
             [{k: x[k] for k in ("type", "kind", "step", "end", "windows")} for x in u]
        e["events"] = sorted(ev, key=lambda x: x["step"])
        e["event_types"] = sorted({x["type"] + (f":{x['kind']}" if x["kind"] else "") for x in e["events"]})
        e["u_threshold"] = round(thr.get(e["ckpt"], 0.0), 4)
        e["u_pool"] = pool
    return pool


KL_FIELDS = ("kl_init", "init_sha256", "init_threads")


def merge_kl(run, dev_eps, dev_cells):
    """D1-trained runs' KL comes from a separate D1 replay (`eval-kl-d1/`, same checkpoints and seeds, --kl-init).
    Copy kl_init / init_sha256 / init_threads onto the dev D1 rows, but only if the replay is the same episode
    (T, P, final_pos identical); a mismatch stops the split. Cells get kl_init_mean and init_sha256."""
    src = RUNS / run / "eval-kl-d1"
    if not (src / "episodes.jsonl").exists():
        return []
    kl = {(r["ckpt"], r["state"], r["seed"]): r for r in rows(src / "episodes.jsonl")}
    n = 0
    for e in dev_eps:
        k = (e["ckpt"], e["state"], e["seed"])
        if k not in kl or "kl_init" in e:
            continue
        r = kl[k]
        if (r["T"], r["P"], r["final_pos"]) != (e["T"], e["P"], e["final_pos"]):
            sys.exit(f"{run} {k}: eval-kl-d1 replay differs from {e['source']} (T/P/final_pos); not merged")
        e.update({f: r[f] for f in KL_FIELDS}, kl_source=src.name)
        n += 1
    summ = json.loads((src / "summary.json").read_text())["summary"]
    for key, c in summ.items():
        if key in dev_cells and "kl_init_mean" in c:
            dev_cells[key].update(kl_init_mean=c["kl_init_mean"], init_sha256=c.get("init_sha256"))
    missing = len(kl) - n
    return [f"eval-kl-d1: KL merged into {n} dev rows (replay T/P/final_pos identical){'' if not missing else f'; {missing} replay rows unmatched'}"]


def split_run(run, vm):
    srcs, view = sources(run)
    dev_dir = RUNS / run / view
    (dev_dir / "traces").mkdir(parents=True, exist_ok=True)
    dev_eps, dev_win, dev_cells, report = [], [], {}, []
    for src in srcs:
        if not (src / "episodes.jsonl").exists():
            report.append(f"{src.name}: missing")
            continue
        eps, wins = rows(src / "episodes.jsonl"), rows(src / "windows.jsonl")
        summ = json.loads((src / "summary.json").read_text())["summary"]
        held_dir = SEALED_RAW / run / src.name
        held_eps = [r for r in eps if r["state"] in HELD]
        held_win = [w for w in wins if w["state"] in HELD]
        odd = {r["state"] for r in eps} - DEV - HELD
        if odd:
            sys.exit(f"{src}: unknown states {odd}")
        for r in eps:
            if r["state"] in DEV:
                dev_eps.append({**r, "vm": vm, "source": src.name})
        dev_win += [{**w, "vm": vm, "source": src.name} for w in wins if w["state"] in DEV]
        dev_cells.update({k: v for k, v in summ.items() if k.split("|")[1] in DEV})
        for r in eps:
            if r["state"] in DEV:
                tf = src / "traces" / f"{Path(r['ckpt']).stem}-{r['state']}-{r['seed']}.jsonl.gz"
                link = dev_dir / "traces" / tf.name
                if tf.exists() and not link.exists():
                    link.symlink_to(tf)
        if held_eps:
            held_dir.mkdir(parents=True, exist_ok=True)
            write_rows(held_dir / "episodes.jsonl", held_eps)
            write_rows(held_dir / "windows.jsonl", held_win)
            (held_dir / "summary.json").write_text(json.dumps(
                {"run_id": run, "vm": vm, "source": str(src),
                 "summary": {k: v for k, v in summ.items() if k.split("|")[1] in HELD}}, indent=1))
        n_held = Counter((r["ckpt"], r["state"]) for r in held_eps)
        report.append(f"{src.name}: dev {sum(r['state'] in DEV for r in eps)} rows, held-out {len(held_eps)} rows in "
                      f"{len(n_held)} cells (n per cell {sorted(set(n_held.values())) or '-'})")
    report += merge_kl(run, dev_eps, dev_cells)
    pool = recompute_u(dev_eps, dev_win)
    for c in dev_cells.values():
        for k in ("share_U", "U_windows", "episodes_with_event_type"):
            c.pop(k, None)
    note = ("dev view (D1, D2) built by seal_b1.py; HA/HB are sealed (§9 A2). U recomputed over dev windows "
            f"{pool} per checkpoint (§9 A′); the cells' old U fields are dropped: use episodes/windows.")
    write_rows(dev_dir / "episodes.jsonl", dev_eps)
    write_rows(dev_dir / "windows.jsonl", dev_win)
    (dev_dir / "summary.json").write_text(json.dumps(
        {"run_id": run, "vm": vm, "states": sorted({r["state"] for r in dev_eps}), "sources": [s.name for s in srcs],
         "u_pool": pool, "summary": dev_cells, "note": note}, indent=1))
    if run != "R0":                       # primary (own training state) and cross (other dev state) views
        own = json.loads((RUNS / run / "config.json").read_text())["state"]
        for name, keep in (("eval-dev-primary", lambda st: st == own), ("eval-dev-cross", lambda st: st != own)):
            d = RUNS / run / name
            d.mkdir(exist_ok=True)
            write_rows(d / "episodes.jsonl", [r for r in dev_eps if keep(r["state"])])
            write_rows(d / "windows.jsonl", [w for w in dev_win if keep(w["state"])])
            (d / "summary.json").write_text(json.dumps(
                {"run_id": run, "vm": vm, "view": name, "train_state": own,
                 "summary": {k: v for k, v in dev_cells.items() if keep(k.split("|")[1])}, "note": note}, indent=1))
    n_dev = Counter((r["ckpt"], r["state"]) for r in dev_eps)
    return report, n_dev


def sealed_files():
    out = []
    for run, _ in batch1_runs() + [("R0", "e1-a")]:
        srcs, _ = sources(run)
        for src in srcs:
            for name in ("episodes.jsonl", "windows.jsonl", "summary.json"):
                if (src / name).exists() and src.name in ("eval", "eval-24"):   # mixed originals (D1 + HA + HB)
                    out.append(src / name)
            if (src / "traces").exists():
                out += sorted(p for p in (src / "traces").iterdir() if p.name.split("-")[2] in HELD)
    for root in (SEALED_RAW, SEALED_OUT):
        if root.exists():
            out += sorted(p for p in root.rglob("*") if p.is_file())
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    p = Path(p).resolve()
    return "~/" + str(p.relative_to(HOME)) if p.is_relative_to(HOME) else str(p)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "split":
        SEALED_OUT.mkdir(parents=True, exist_ok=True)
        for run, vm in batch1_runs() + [("R0", "e1-a")]:
            if not (RUNS / run / "config.json").exists():
                print(f"{run} ({vm}): not fetched yet, skipped")
                continue
            report, n_dev = split_run(run, vm)
            bad = {k: n for k, n in n_dev.items() if n != (24 if run == "R0" else 8)}
            print(f"{run} ({vm}): " + "; ".join(report) + f"; dev cells {len(n_dev)}"
                  + (f", WRONG n in {len(bad)}" if bad else ", all n ok"))
        files = sealed_files()
        MANIFEST.write_text("".join(f"{sha(p)}  {rel(p)}\n" for p in files))
        print(f"manifest {rel(MANIFEST)}: {len(files)} sealed files; sha256 {sha(MANIFEST)}")
    elif cmd == "verify":
        want = dict(reversed(l.split("  ", 1)) for l in MANIFEST.read_text().splitlines())
        have = {rel(p): p for p in sealed_files()}
        changed = [k for k, h in ((k, want[k]) for k in want if k in have) if sha(have[k]) != h]
        missing = [k for k in want if k not in have]
        new = [k for k in have if k not in want]
        print(f"manifest sha256 {sha(MANIFEST)}; {len(want)} files: {len(changed)} changed, {len(missing)} missing, "
              f"{len(new)} new")
        for k in changed + missing + new:
            print("  ", k)
        sys.exit(1 if changed or missing else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
