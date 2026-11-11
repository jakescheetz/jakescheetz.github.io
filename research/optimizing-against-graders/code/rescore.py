#!/usr/bin/env python3
"""Offline re-score of saved eval traces under a patched proxy (experiment-e1.md §9 "Batch-1 analysis plan"
F, the patch's offline check). No training and no replay: each saved trace row has the RAM fields the proxy
reads (position, party, levels, HP, Pokédex seen), so the proxy runs unchanged on a RAM image rebuilt from
the row. The episode's start state (needed for Proxy.reset) comes from one emulator reset per state, so this
needs $POKEMON_RED_ROM. The ROM and traces stay in ~/roms; only the summary tables are written out.

  .venv/bin/python rescore.py RUN [RUN ...] --evals eval eval-d2 --proxy-params '{"map_new_tile_cap": 200}' \
      [--states D1 D2] [--own-state] [--out results/rescore-x]

- **P_old** is the episode's P as saved (the run's own reward params). **P_new** is the trace re-scored with
  the run's proxy params updated by --proxy-params (a JSON dict of Proxy / MapCapProxy kwargs: w_explore,
  w_level, w_heal, w_seen, tile_visit_cap, heal_decay, level_cap, map_new_tile_cap).
- **Exactness check:** every trace is also re-scored with no change and compared step by step with the
  saved per-step `p`; max |diff| is printed and stored, and the run fails if it is above 1e-9.
- **Per hack type** (event label type[:kind] from episodes.jsonl; a step is "inside" if it lies in any
  window of that type in that episode): P inside, before and after, and the drop share 1 − new/old.
  Outside = P on the steps outside every window of that type. Pooled over all episodes and per mark.
- **Blinding (§9 A2):** only dev states (D1, D2) by default. HA/HB need --heldout, and then --out must be
  under results/sealed-b1/.
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import env as E, ram, rom  # noqa: E402

DEV = ("D1", "D2")
HELD = ("HA", "HB")
KNOBS = ("tile_visit_cap", "heal_decay", "level_cap", "map_new_tile_cap")   # Proxy / MapCapProxy kwargs


def row_mem(row, mem=None):
    """A RAM image holding just what Proxy reads, from a trace row."""
    m = mem if mem is not None else bytearray(0x10000)
    m[ram.W_CUR_MAP], m[ram.W_X_COORD], m[ram.W_Y_COORD] = row["map"], row["x"], row["y"]
    m[ram.W_PARTY_COUNT] = row["party_count"]
    for k in range(6):
        b = ram.PARTY_STRIDE * k
        lv = row["levels"][k] if k < len(row["levels"]) else 0
        hp = row["hp"][k] if k < len(row["hp"]) else 0
        mx = row["max_hp"][k] if k < len(row["max_hp"]) else 0
        m[ram.W_PARTY_MON1_LEVEL + b] = lv
        m[ram.W_PARTY_MON1_HP + b], m[ram.W_PARTY_MON1_HP + b + 1] = hp >> 8, hp & 0xFF
        m[ram.W_PARTY_MON1_MAX_HP + b], m[ram.W_PARTY_MON1_MAX_HP + b + 1] = mx >> 8, mx & 0xFF
    seen = row["seen"]
    for i in range(ram.POKEDEX_BYTES):
        bits = max(0, min(8, seen - 8 * i))
        m[ram.W_POKEDEX_SEEN + i] = (1 << bits) - 1
    return m


_START = {}


def start_mem(cfg, state):
    """The RAM image at episode start (Proxy.reset input), from one emulator reset of this state."""
    key = (state, cfg["fast_mode"], cfg["frames_per_action"])
    if key not in _START:
        env = E.PokemonRedEnv(state_name=state, reward=E.Proxy(), max_steps=1, seed=0, obs_mode=cfg["obs_mode"],
                              fast_mode=cfg["fast_mode"], frames_per_action=cfg["frames_per_action"],
                              keep_trace=False)
        env.reset()
        m = env.mem
        _START[key] = row_mem({"map": m[ram.W_CUR_MAP], "x": m[ram.W_X_COORD], "y": m[ram.W_Y_COORD],
                               "party_count": ram.party_count(m), "levels": ram.party_levels(m),
                               "hp": ram.party_hp(m), "max_hp": ram.party_max_hp(m),
                               "seen": ram.popcount_bytes(m, ram.W_POKEDEX_SEEN, ram.POKEDEX_BYTES)})
    return _START[key]


def proxy_kwargs(params):
    """Proxy kwargs from a config's reward_params (+ any extra knobs stored there)."""
    w = params.get("w", {})
    kw = {f"w_{k}": v for k, v in w.items()}
    kw.update({k: v for k, v in params.items() if k in KNOBS})
    return kw


def score(trace, cfg, state, kw):
    """Per-step proxy reward of a trace under Proxy kwargs kw."""
    p = E.make_proxy(**kw)
    p.reset(start_mem(cfg, state))
    mem, out = bytearray(0x10000), []
    for row in trace:
        r, _ = p(row_mem(row, mem), 0)
        out.append(r)
    return out, E.reward_hash(p)


def windows_by_type(events):
    """Windows per type:kind, plus per type pooled over its kinds (e.g. "O" = O:behind ∪ O:side; a step counts
    once however many windows hold it)."""
    by = defaultdict(list)
    for e in events:
        by[e["type"] + (f":{e['kind']}" if e.get("kind") else "")].append((e["step"], e["end"]))
        if e.get("kind"):
            by[e["type"]].append((e["step"], e["end"]))
    return by


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--evals", nargs="+", default=["eval"], help="eval dir names under each run (default eval)")
    ap.add_argument("--proxy-params", default="{}", help="JSON dict of Proxy kwargs that override the run's own")
    ap.add_argument("--states", nargs="+", default=list(DEV))
    ap.add_argument("--own-state", action="store_true",
                    help="keep only episodes on each run's own training state (pools D1- and D2-trained runs)")
    ap.add_argument("--heldout", action="store_true", help="allow HA/HB (output must be under results/sealed-b1/)")
    ap.add_argument("--out", help="write OUT.json and OUT.md")
    a = ap.parse_args()
    held = [s for s in a.states if s in HELD]
    if held and not (a.heldout and a.out and "sealed-b1" in Path(a.out).resolve().parts):
        sys.exit(f"held-out states {held}: need --heldout and --out under results/sealed-b1/ (§9 A2 blinding)")
    patch = json.loads(a.proxy_params)
    tot = defaultdict(lambda: defaultdict(float))       # (type, step) -> sums; type "*" = whole episode
    n_ep, max_diff, hashes, missing = 0, 0.0, set(), []
    for run in a.runs:
        rd = rom.runs_dir() / run
        cfg = json.loads((rd / "config.json").read_text())
        base = proxy_kwargs(cfg["reward_params"])
        new = {**base, **patch}
        for ev in a.evals:
            ed = rd / ev
            if not (ed / "episodes.jsonl").exists():
                missing.append(f"{run}/{ev}")
                continue
            for line in open(ed / "episodes.jsonl"):
                ep = json.loads(line)
                if ep["state"] not in a.states or (a.own_state and ep["state"] != cfg["state"]):
                    continue
                name = f"{Path(ep['ckpt']).stem}-{ep['state']}-{ep['seed']}.jsonl.gz"
                tf = ed / "traces" / name
                if not tf.exists() and (rd / "eval" / "traces" / name).exists():
                    tf = rd / "eval" / "traces" / name     # dev views (seal_b2.py) hold rows only; traces stay in eval/
                if not tf.exists():
                    missing.append(str(tf.relative_to(rom.runs_dir())))
                    continue
                with gzip.open(tf, "rt") as f:
                    f.readline()
                    trace = [json.loads(l) for l in f]
                old, _ = score(trace, cfg, ep["state"], base)
                saved = [r["p"] for r in trace]
                max_diff = max(max_diff, max(abs(x - y) for x, y in zip(old, saved)))
                nw, h = score(trace, cfg, ep["state"], new)
                hashes.add(h)
                n_ep += 1
                wins = windows_by_type(ep["events"])
                for typ, ws in [("*", [(1, len(trace))])] + list(wins.items()):
                    inside = [any(s <= row["ep_step"] <= e for s, e in ws) for row in trace]
                    for key in ((typ, ep["step"]), (typ, "all")):
                        t = tot[key]
                        t["episodes"] += 1
                        t["in_old"] += sum(o for o, i in zip(saved, inside) if i)
                        t["in_new"] += sum(x for x, i in zip(nw, inside) if i)
                        t["out_old"] += sum(o for o, i in zip(saved, inside) if not i)
                        t["out_new"] += sum(x for x, i in zip(nw, inside) if not i)
    if n_ep and max_diff > 1e-9:
        sys.exit(f"exactness check FAILED: re-scored P0 differs from the saved per-step p by {max_diff}")
    rows = []
    for (typ, step), t in sorted(tot.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        drop = 1 - t["in_new"] / t["in_old"] if t["in_old"] else None
        rows.append({"type": typ, "step": step, **{k: round(v, 4) for k, v in t.items()},
                     "drop_inside": None if drop is None else round(drop, 4)})
    res = {"runs": a.runs, "evals": a.evals, "states": a.states, "own_state": a.own_state, "proxy_params": patch,
           "reward_hash_new": sorted(hashes), "episodes": n_ep, "exactness_max_abs_diff": max_diff,
           "missing": missing, "rows": rows}
    print(f"{n_ep} episodes; exactness max |diff| {max_diff:.2e}; new reward hash {sorted(hashes)}; "
          f"{len(missing)} missing")
    for r in rows:
        if r["step"] == "all":
            print(f"  {r['type']:>14}  P inside {r['in_old']:.1f} -> {r['in_new']:.1f}"
                  f"  (drop {r['drop_inside']})  outside {r['out_old']:.1f} -> {r['out_new']:.1f}")
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(res, indent=1))
        md = [f"# Offline re-score: {json.dumps(patch)}", "",
              f"Runs {', '.join(a.runs)}; evals {a.evals}; states {a.states}"
              f"{' (each run on its own training state)' if a.own_state else ''}; {n_ep} episodes. "
              f"New reward hash {', '.join(sorted(hashes))}. Exactness (P0 re-score vs saved p): max |diff| "
              f"{max_diff:.1e}.", "",
              "\"Inside\" = steps inside any window of that type in that episode; `*` = whole episode. "
              "Drop = 1 − P_new/P_old inside (§9 F: the patch passes if the top type's drop ≥ 0.80).", "",
              "| Type | Mark | Episodes | P inside old → new | Drop inside | P outside old → new |",
              "|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['type']} | {r['step']} | {int(r['episodes'])} | {r['in_old']:.1f} → {r['in_new']:.1f} "
                      f"| {r['drop_inside']} | {r['out_old']:.1f} → {r['out_new']:.1f} |")
        if missing:
            md += ["", f"Missing ({len(missing)}): " + ", ".join(missing[:20])]
        out.with_suffix(".md").write_text("\n".join(md) + "\n")
        print("wrote", out.with_suffix(".json"), out.with_suffix(".md"))


if __name__ == "__main__":
    main()
