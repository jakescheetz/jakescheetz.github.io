#!/usr/bin/env python3
"""Batch-2 blinding (experiment-e1.md §9 "Batch 2" H8, item 6): the sealed-b2 manifest and the dev views.

  .venv/bin/python seal_b2.py split        check the split, build the dev views, write the manifest (idempotent)
  .venv/bin/python seal_b2.py verify       re-hash every sealed file against the manifest

Batch 2 was split at the source, so unlike seal_b1.py there is nothing to move:
- **Dev:** each run's closing `eval/` holds D1 + D2 only, with KL from ckpt-0 and U pooled over D1 + D2 per
  checkpoint already, because evaluate.py ran on exactly those states (§9 A′). `split` checks that: any HA/HB row,
  window or trace under `~/roms/runs/<run>/` stops it with a leak error.
- **Sealed:** HA + HB closing evals went straight to `~/roms/sealed-b2/<run>/eval-heldout/` (outside the runs
  tree and the repo). Derived held-out outputs go to `code/results/sealed-b2/`. Both are hashed; neither is opened
  except to count rows per (checkpoint, state) cell.
- **Tested views hold marks 0.25-5M** (the 14-cell families); ckpt-0 goes to `eval-dev-mark0/` (dev) and stays only
  in `eval-heldout/` (sealed), whose copy without it is `eval-heldout-test/` (sealed tree, hashed).
- **Views (§9 A′ ruling 1):** `eval-dev/` (both dev states, with trace links), `eval-dev-primary/` (own training state) and `eval-dev-cross/` (the other dev state)
  per run, as in batch 1. HA-trained runs have no own dev state: primary is empty and cross holds D1 + D2.
- **Manifest:** `code/results/sealed-b2.sha256` (sha256 + path from ~, one line per sealed file, sorted by path
  in C order), after '#' note lines (e.g. the dry-run folder). Its own sha256 is printed for BOARD. `verify` reports changed, missing or new sealed files.
- This prints only counts (runs, files, rows, cells, n per cell): never a held-out value.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent
HOME = Path.home()
RUNS = HOME / "roms" / "runs"
SEALED_RAW = HOME / "roms" / "sealed-b2"
SEALED_OUT = CODE / "results" / "sealed-b2"
MANIFEST = CODE / "results" / "sealed-b2.sha256"
DEV, HELD = {"D1", "D2"}, {"HA", "HB"}
PLANS = {"e1-a": CODE / "gcp/plans/main-b2-e1a.txt", "e1-b": CODE / "gcp/plans/main-b2-e1b.txt"}
MARKS = 8                                   # ckpt-0 + 7 eval marks
N = 8                                       # episodes per cell
NOTES = {                                   # '#' lines at the top of the manifest; verify skips them
    "results/sealed-b2/dry-b2/": "dry-run output (analyze_b1.py --batch b2 --dry --vm e1-a, 12 runs, 11-10); left in "
                                 "place unopened per the lead; excluded from the analysis",
}


def batch2_runs():
    out = []
    for vm, plan in PLANS.items():
        for line in plan.read_text().splitlines():
            line = line.split("#")[0]
            if "--run-id" in line:
                out.append((line.split("--run-id")[1].split()[0], vm))
    return out


def rows(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def write_rows(path, rs):
    with open(path, "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")


def leak_check(run):
    """Any held-out row, window, cell or trace inside the runs tree is a leak."""
    d = RUNS / run
    bad = []
    for f in d.rglob("*.jsonl"):
        if f.name in ("episodes.jsonl", "windows.jsonl") and f.parent.name.startswith("eval"):
            if any(r.get("state") in HELD for r in rows(f)):
                bad.append(str(f))
    for f in d.rglob("summary.json"):
        if f.parent.name.startswith("eval"):
            cells = json.loads(f.read_text()).get("summary", {})
            if any(k.split("|")[1] in HELD for k in cells if "|" in k):
                bad.append(str(f))
    bad += [str(p) for p in d.rglob("*.jsonl.gz") if p.name.split("-")[-2] in HELD]
    return bad


def split_run(run, vm):
    d, src = RUNS / run, RUNS / run / "eval"
    if not (src / "episodes.jsonl").exists():
        return [f"{run}: dev eval/ missing"], None, None
    leaks = leak_check(run)
    if leaks:
        sys.exit(f"{run}: held-out data inside the runs tree (leak), stopping: {leaks[:3]}")
    eps, wins = rows(src / "episodes.jsonl"), rows(src / "windows.jsonl")
    cells = json.loads((src / "summary.json").read_text())["summary"]
    own = json.loads((d / "config.json").read_text())["state"]
    pool = {}                                        # U decile pool per checkpoint = the states evaluate.py ran on
    for w in wins:
        pool.setdefault(w["ckpt"], set()).add(w["state"])
    tag = lambda r: {**r, "vm": vm, "source": "eval", "u_pool": sorted(pool.get(r["ckpt"], ())),
                     "kl_source": "ckpt-0" if str(r.get("init_sha256", "")).endswith("=ckpt-0") else None}
    note = ("dev view (D1, D2) built by seal_b2.py from the closing eval/ (D1 + D2 only; KL from ckpt-0; U pooled "
            "over D1 + D2 per checkpoint by evaluate.py); HA/HB are in " + rel(SEALED_RAW) + "/ (§9 H8). The tested views "
            "hold the 7 marks 0.25-5M (the 14-cell families, §9 H5); ckpt-0 (mark 0) is in eval-dev-mark0/ only")
    test = lambda r: r["ckpt"] != "ckpt-0.pt"
    views = (("eval-dev", lambda r: test(r)), ("eval-dev-primary", lambda r: test(r) and r["state"] == own),
             ("eval-dev-cross", lambda r: test(r) and r["state"] != own), ("eval-dev-mark0", lambda r: not test(r)))
    for name, keep in views:
        v = d / name
        v.mkdir(exist_ok=True)
        write_rows(v / "episodes.jsonl", [tag(r) for r in eps if keep(r)])
        write_rows(v / "windows.jsonl", [{**w, "vm": vm, "source": "eval"} for w in wins if keep(w)])
        (v / "summary.json").write_text(json.dumps(
            {"run_id": run, "vm": vm, "view": name, "train_state": own, "states": sorted(DEV),
             "summary": {k: c for k, c in cells.items() if keep({"ckpt": k.split("|")[0], "state": k.split("|")[1]})},
             "note": note}, indent=1))
        if name == "eval-dev":                     # trace links, as seal_b1's eval-dev/traces
            (v / "traces").mkdir(exist_ok=True)
            for r in eps:
                if keep(r):
                    tf = src / "traces" / f"{Path(r['ckpt']).stem}-{r['state']}-{r['seed']}.jsonl.gz"
                    link = v / "traces" / tf.name
                    if tf.exists() and not link.exists():
                        link.symlink_to(tf)
    n_dev = Counter((r["ckpt"], r["state"]) for r in eps)
    hsrc = SEALED_RAW / run / "eval-heldout"
    held = rows(hsrc / "episodes.jsonl")
    n_held = Counter((r["ckpt"], r["state"]) for r in held)
    if held:      # the tested held-out view, inside the sealed tree: the 7 marks 0.25-5M, rows copied unread
        hv = SEALED_RAW / run / "eval-heldout-test"
        hv.mkdir(exist_ok=True)
        write_rows(hv / "episodes.jsonl", [{**r, "vm": vm, "source": "eval-heldout"} for r in held if test(r)])
        write_rows(hv / "windows.jsonl", [w for w in rows(hsrc / "windows.jsonl") if test(w)])
        hs = json.loads((hsrc / "summary.json").read_text())
        hs["summary"] = {k: c for k, c in hs["summary"].items() if not k.startswith("ckpt-0.pt|")}
        hs["note"] = "tested held-out view (seal_b2.py): marks 0.25-5M; ckpt-0 stays in eval-heldout/ only"
        (hv / "summary.json").write_text(json.dumps(hs, indent=1))
    report = [f"dev {len(eps)} rows in {len(n_dev)} cells (n {sorted(set(n_dev.values()))})",
              f"held-out {len(held)} rows in {len(n_held)} cells (n {sorted(set(n_held.values())) or '-'})"]
    return report, n_dev, n_held


def sealed_files():
    out = []
    for root in (SEALED_RAW, SEALED_OUT):
        if root.exists():
            out += [p for p in root.rglob("*") if p.is_file()]
    return sorted(out, key=lambda p: rel(p).encode())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    p = Path(p).absolute()
    return "~/" + str(p.relative_to(HOME)) if p.is_relative_to(HOME) else str(p)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "split":
        SEALED_OUT.mkdir(parents=True, exist_ok=True)
        ok = missing = 0
        for run, vm in batch2_runs():
            if not (RUNS / run / "config.json").exists():
                print(f"{run} ({vm}): not fetched yet, skipped")
                missing += 1
                continue
            report, n_dev, n_held = split_run(run, vm)
            if n_dev is None:
                print("; ".join(report))
                missing += 1
                continue
            good = (len(n_dev) == 2 * MARKS and set(n_dev.values()) == {N}
                    and len(n_held) == 2 * MARKS and set(n_held.values()) == {N})
            ok += good
            print(f"{run} ({vm}): " + "; ".join(report) + ("; all n ok" if good else "; WRONG cell count or n"))
        files = sealed_files()
        notes = [f"# {rel(CODE / k)}: {v}\n" for k, v in NOTES.items() if (CODE / k).exists()]
        MANIFEST.write_text("".join(notes) + "".join(f"{sha(p)}  {rel(p)}\n" for p in files))
        print(f"{ok} runs complete, {missing} missing; manifest {rel(MANIFEST)}: {len(files)} sealed files; "
              f"sha256 {sha(MANIFEST)}")
    elif cmd == "verify":
        want = dict(reversed(l.split("  ", 1)) for l in MANIFEST.read_text().splitlines()
                    if not l.startswith("#"))
        have = {rel(p): p for p in sealed_files()}
        changed = [k for k in want if k in have and sha(have[k]) != want[k]]
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
