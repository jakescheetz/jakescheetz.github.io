#!/usr/bin/env python3
"""§9 "A2 oracle pilot" ask (1): replay the scripted §9.1 check trace (script_states.py stage "trace": D1 ->
Route 1 battles -> Viridian (M1) -> three Center heals -> Route 2 (M2)) with the A2 Oracle as the env reward
and check it pays exactly +1 at M1, +1 at M2 and 0 on every other step. Then the same trace under A2′
(ShapedOracle): its reward must equal ΔT + γΦ(s') − Φ(s) on every step, and sum to T + the telescoped Φ.

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python oracle_check.py

No screenshots. Outputs (outside the repo, rom.states_dir()): script-trace-oracle.jsonl, script-trace-a2p.jsonl
(+ their -summary.json from the trace stage), oracle-check.json (the result below).
"""
import json
import sys

from pokered import env as E
from pokered import ram, rom, shaping
import script_states as SS


def run(reward, name):
    for seed in range(8):
        try:
            SS._stage_trace(None, seed, reward=reward, name=name)
            break
        except SS.Blackout as e:
            print(f"  {name} seed {seed}: blackout ({e}); next seed")
    else:
        sys.exit(f"{name}: trace blacked out for every seed")
    with open(rom.states_dir() / f"{name}.jsonl") as f:
        return seed, [json.loads(l) for l in f]


def main():
    sd = rom.states_dir()
    with open(sd / "script-trace.jsonl") as f:
        ref = [(r["map"], r["x"], r["y"], r["T"]) for r in (json.loads(l) for l in f)]
    out = {}
    seed, rows = run(E.Oracle(), "script-trace-oracle")
    paid = [(i, r["ep_step"], r["r"], r["T"]) for i, r in enumerate(rows) if r["r"] != 0]
    ms = {n: next(i for i, r in enumerate(rows) if r["T"] >= n) for n in (1, 2)}
    out["oracle"] = {
        "seed": seed, "steps": len(rows), "nonzero_reward_steps": paid,
        "milestone_rows": ms, "maps_at_milestones": {n: hex(rows[i]["map"]) for n, i in ms.items()},
        "same_path_as_script_trace": [(r["map"], r["x"], r["y"], r["T"]) for r in rows] == ref,
        "pass": [p[0] for p in paid] == [ms[1], ms[2]] and all(p[2] == 1.0 for p in paid) and len(rows) and rows[-1]["T"] == 2,
    }
    seed2, rows2 = run(E.ShapedOracle(), "script-trace-a2p")
    ph = [shaping.phi(r["map"], r["x"], r["y"], r["T"]) for r in rows2]
    # the reward's state before row 0 is the reset state: same tile and T as row 0 except for the first press
    expect = [(rows2[i]["T"] - rows2[i - 1]["T"]) + shaping.GAMMA * ph[i] - ph[i - 1] for i in range(1, len(rows2))]
    err = max(abs(rows2[i]["r"] - expect[i - 1]) for i in range(1, len(rows2)))
    disc_total = sum(r["r"] for r in rows2[1:])
    out["a2p"] = {
        "seed": seed2, "steps": len(rows2), "max_abs_error_vs_formula": err,
        "sum_reward_rows_1_on": round(disc_total, 6),
        "T_plus_phi_telescoped_gamma_1_part": round(rows2[-1]["T"] - rows2[0]["T"] + ph[-1] - ph[0], 6),
        "phi": {"start": round(ph[0], 4), "at_M1": round(ph[ms[1]], 4) if ms else None,
                "at_M2": round(ph[-1], 4), "min": round(min(ph), 4), "max": round(max(ph), 4)},
        "reward_at_milestones": {n: round(rows2[i]["r"], 4) for n, i in
                                 {n: next(i for i, r in enumerate(rows2) if r["T"] >= n) for n in (1, 2)}.items()},
        "largest_single_step": sorted((round(r["r"], 4), i, hex(r["map"])) for i, r in enumerate(rows2))[-3:],
        "smallest_single_step": sorted((round(r["r"], 4), i, hex(r["map"])) for i, r in enumerate(rows2))[:3],
        "pass": err < 1e-9,
    }
    (sd / "oracle-check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
