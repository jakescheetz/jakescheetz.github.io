#!/usr/bin/env python3
"""Round-1 blinding (experiment-e1.md §9 I12, round 1 = r1-A1-D1-s1 + s3; s2 skipped by Jake + lead, r3).

  .venv/bin/python seal_r1.py split        check the split, build the dev views, write the manifest (idempotent)
  .venv/bin/python seal_r1.py verify       re-hash every sealed file against the manifest

Exactly seal_b2.py's procedure (see its docstring) on round-1 paths: the round-1 plans (`gcp/plans/r1-e1{a,b}.txt`;
commented-out lines such as the skipped s2 are ignored), held-out closing evals in `~/roms/sealed-r1/<run>/eval-heldout/`,
derived held-out outputs in `code/results/sealed-r1/`, manifest `code/results/sealed-r1.sha256`. Dev views:
`eval-dev/`, `eval-dev-primary/` (D1), `eval-dev-cross/` (D2), `eval-dev-mark0/` per run. Prints counts only.
"""
import seal_b2 as S

S.SEALED_RAW = S.HOME / "roms" / "sealed-r1"
S.SEALED_OUT = S.CODE / "results" / "sealed-r1"
S.MANIFEST = S.CODE / "results" / "sealed-r1.sha256"
S.PLANS = {"e1-a": S.CODE / "gcp/plans/r1-e1a.txt", "e1-b": S.CODE / "gcp/plans/r1-e1b.txt"}
S.NOTES = {}

if __name__ == "__main__":
    print("round-1 runs:", ", ".join(f"{r} ({v})" for r, v in S.batch2_runs()))
    S.main()
