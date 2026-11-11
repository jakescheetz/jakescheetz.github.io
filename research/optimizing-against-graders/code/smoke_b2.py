#!/usr/bin/env python3
"""Batch-2 smoke tests (experiment-e1.md §9 "Batch 2" H4 a–c; d is the train smoke, run separately). Needs the ROM.

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python smoke_b2.py [--steps 6000]
      -> prints PASS/FAIL per check and writes results/batch2-smoke.json

The driver is a replay: the action column of the G5 planner's D1 seed-1000 trace (~/roms/g5/planner/traces/),
which walks through many doors, map edges and battles. The same actions are replayed with the same seed in a
`tilegrid_visited` env and a batch-1 `tilegrid` env, step by step:

  (a) alignment, every settled overworld step (no battle, no text, ≥ 2 steps since the last map change, not
      mid ledge hop: a hop moves the coordinates a step before the screen finishes scrolling, so the tiles
      lag; the channel follows the coordinates, as the reward does; and not while wJoyIgnore is set, or a script presses
      the buttons (wSimulatedJoypadStatesIndex, this step or the last), i.e. a script moves the player, as in
      the Pewter escort and the museum push-back, with the same one-step lag); squares whose cell shows a font,
      text-box or menu tile (id >= 0x60; overworld map tiles are < 0x60) are overlays and aren't compared:
      - geometry: for every square in view, the bottom-left 8×8 tile of its cell block in wTileMap (rows
        VIS_ROW0 + 2dy + 1, col VIS_COL0 + 2dx) equals that square's bottom-left tile from the map data
        (wOverworldMap blocks × the tileset's blockset, as driver.walk_grid reads it). This pins VIS_ROW0/COL0.
      - content, every step: each cell is 1 exactly when its square (inverse map: cell (r, c) -> (map,
        x + (c - VIS_COL0)//2, y + (r - VIS_ROW0)//2)) is in the set of squares stood on so far this episode,
        rebuilt independently from the trace; the cell block under the player is 1; per map (only the
        current map's squares can light up).
      - reset: after env.reset() the grid holds only the start square.
  (b) identity: the replayed positions equal the trace's; tiles, sprites and the RAM vector equal the batch-1
      env's at every step; the flattened obs with the visited block removed equals batch 1's flat obs.
  (c) reward hashes: P0 = d28e2a337dda, A2″ = 1f739117e21a, built from the batch-1 run configs as train.py does.
Plus: ActorCritic("tilegrid_visited") runs on the flat obs, its first conv has 10 inputs, and with the
visited input zeroed its first conv equals a 9-input conv on the batch-1 input (same weights).
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from pokered import driver as D, ram, rom  # noqa: E402

HERE = Path(__file__).resolve().parent
W_SIM_JOYPAD = 0xCD38          # wSimulatedJoypadStatesIndex (pret wram.asm): a script is pressing the buttons


def square_tile(m, sx, sy):
    """Bottom-left 8×8 tile id of square (sx, sy) on the current map, from wOverworldMap + the blockset
    (driver.walk_grid's lookup, outside its loop)."""
    w = m[D.W_CUR_MAP_WIDTH]
    stride = w + 2 * D.MAP_BORDER
    bank = m[D.W_TILESET_BANK]
    blocks = m[D.W_TILESET_BLOCKS_PTR] | m[D.W_TILESET_BLOCKS_PTR + 1] << 8
    b = m[D.W_OVERWORLD_MAP + (sy // 2 + D.MAP_BORDER) * stride + sx // 2 + D.MAP_BORDER]
    addr = blocks + b * 16 + ((sy % 2) * 2 + 1) * 4 + (sx % 2) * 2
    return m[bank, addr] if addr >= 0x4000 else m[addr]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--trace", default=str(rom.runs_dir().parent / "g5" / "planner" / "traces" / "planner-D1-1000.jsonl.gz"))
    a = ap.parse_args()
    import torch
    import script_states as SS
    import train as TR
    from pokered import env as E
    from pokered.policy import ActorCritic, flatten_obs, TILE_CELLS
    res = {}

    # (c) reward hashes
    hashes = {}
    for run, want in (("A0-D1-s1", "d28e2a337dda"), ("A2pp-D1-s1", "1f739117e21a")):
        cfg = json.loads((rom.runs_dir() / run / "config.json").read_text())
        got = E.reward_hash(TR.make_reward(cfg))
        hashes[cfg["arm"]] = {"got": got, "want": want, "recorded": cfg["reward_hash"]}
    res["c_reward_hashes"] = {"pass": all(v["got"] == v["want"] == v["recorded"] for v in hashes.values()), **hashes}

    # replay
    with gzip.open(a.trace, "rt") as f:
        meta = json.loads(f.readline())
        trace = [json.loads(l) for l in f][:a.steps]
    cfg = json.loads((rom.runs_dir() / "A0-D1-s1" / "config.json").read_text())
    mk = lambda mode: E.PokemonRedEnv(state_name=meta["state"], reward=TR.make_reward(cfg), max_steps=10 ** 9,
                                      seed=meta["seed"], jitter=True, obs_mode=mode, fast_mode=cfg["fast_mode"],
                                      frames_per_action=cfg["frames_per_action"], keep_trace=False)
    ev, e1 = mk("tilegrid_visited"), mk("tilegrid")
    ov, o1 = ev.reset(), e1.reset()
    m = ev.pb.memory
    start = ram.position(m)
    reset_ok = int(ov["visited"].sum()) == 4 and ov["visited"][E.VIS_ROW0:E.VIS_ROW0 + 2, E.VIS_COL0:E.VIS_COL0 + 2].all()
    stood = {start}
    n = dict(steps=0, pos_mismatch=0, obs_mismatch=0, flat_mismatch=0, content_bad=0, under_player_bad=0,
             geom_checked=0, geom_bad=0, map_changes=0, maps=set())
    geom_examples, last_map, since_map = [], start[0], 99
    sd = D.Driver(ev.pb)
    DIR = {E.ACTIONS.index(d) for d in ("up", "down", "left", "right")}
    mid_hop = {i for i in range(len(trace) - 1)                 # the position moves on without a direction press:
               if (trace[i + 1]["map"], trace[i + 1]["x"], trace[i + 1]["y"]) != (trace[i]["map"], trace[i]["x"], trace[i]["y"])
               and trace[i + 1]["action"] not in DIR and trace[i + 1]["map"] == trace[i]["map"]}
    sim_prev = 0
    n["mid_hop_skipped"] = n["scripted_skipped"] = n["overlay_squares"] = 0
    for i, row in enumerate(trace):
        ov, _, _, _ = ev.step(row["action"])
        o1, _, _, _ = e1.step(row["action"])
        n["steps"] += 1
        mp, x, y = ram.position(m)
        n["maps"].add(mp)
        if mp != last_map:
            n["map_changes"] += 1
            last_map, since_map = mp, 0
        else:
            since_map += 1
        if (mp, x, y) != (row["map"], row["x"], row["y"]):
            n["pos_mismatch"] += 1
        stood.add((mp, x, y))
        # (b) identity with batch 1
        if not (np.array_equal(ov["tiles"], o1["tiles"]) and np.array_equal(ov["sprites"], o1["sprites"])
                and np.array_equal(ov["ram"], o1["ram"])):
            n["obs_mismatch"] += 1
        fv, f1 = flatten_obs("tilegrid_visited", ov), flatten_obs("tilegrid", o1)
        if not np.array_equal(np.concatenate([fv[:2 * TILE_CELLS], fv[3 * TILE_CELLS:]]), f1):
            n["flat_mismatch"] += 1
        # (a) content, from the independent set
        g = ov["visited"]
        want = np.zeros_like(g)
        for r in range(ram.TILE_ROWS):
            for c in range(ram.TILE_COLS):
                if (mp, x + (c - E.VIS_COL0) // 2, y + (r - E.VIS_ROW0) // 2) in stood:
                    want[r, c] = 1
        if not np.array_equal(g, want):
            n["content_bad"] += 1
        if not g[E.VIS_ROW0:E.VIS_ROW0 + 2, E.VIS_COL0:E.VIS_COL0 + 2].all():
            n["under_player_bad"] += 1
        # (a) geometry, settled overworld steps only
        if i in mid_hop:
            n["mid_hop_skipped"] += 1
        sim_now = m[W_SIM_JOYPAD]
        if i in mid_hop:
            pass
        elif m[D.W_JOY_IGNORE] or m[W_SIM_JOYPAD] or sim_prev:  # scripted movement (Pewter escort, museum push-back)
            n["scripted_skipped"] += 1
        elif not m[ram.W_IS_IN_BATTLE] and not sd.screen_text() and since_map >= 2:
            tm = np.frombuffer(bytes(m[ram.W_TILE_MAP:ram.W_TILE_MAP + 360]), dtype=np.uint8).reshape(18, 20)
            bad = []
            for dy in range(-4, 5):
                for dx in range(-4, 6):
                    t = tm[E.VIS_ROW0 + 2 * dy + 1, E.VIS_COL0 + 2 * dx]
                    if t >= 0x60:            # font / text-box / menu tile: an overlay, not map (overworld tiles < 0x60)
                        n["overlay_squares"] += 1
                        continue
                    if t != square_tile(m, x + dx, y + dy):
                        bad.append(dy)
            n["geom_checked"] += 1
            if bad:
                n["geom_bad"] += 1
                n.setdefault("bad_by_map", {}).setdefault(hex(mp), []).append(n["steps"])
                if len(geom_examples) < 5:
                    geom_examples.append({"step": n["steps"], "pos": [mp, x, y], "bad_squares": len(bad)})
        sim_prev = sim_now
    # reset again: only the new start square
    ov = ev.reset()
    reset2_ok = int(ov["visited"].sum()) == 4
    # P_terms logging sums to P
    pt_ok = abs(sum(ev.P_terms.values()) - ev.P) < 1e-9
    ev.close(), e1.close()
    n["maps"] = sorted(hex(v) for v in n["maps"])
    res["a_alignment"] = {"pass": n["content_bad"] == 0 and n["under_player_bad"] == 0 and n["geom_bad"] == 0
                          and n["geom_checked"] > 0 and reset_ok and reset2_ok and n["map_changes"] > 0,
                          "reset_ok": bool(reset_ok and reset2_ok), "geom_examples": geom_examples,
                          "bad_by_map": {k: v[:5] for k, v in n.get("bad_by_map", {}).items()},
                          **{k: n[k] for k in ("steps", "content_bad", "under_player_bad", "geom_checked", "geom_bad",
                                               "mid_hop_skipped", "scripted_skipped", "overlay_squares", "map_changes", "maps")}}
    res["b_identity"] = {"pass": n["pos_mismatch"] == 0 and n["obs_mismatch"] == 0 and n["flat_mismatch"] == 0,
                         **{k: n[k] for k in ("pos_mismatch", "obs_mismatch", "flat_mismatch")}}
    # policy
    torch.manual_seed(0)
    net = ActorCritic("tilegrid_visited")
    c0 = net.enc.conv[0]
    x = torch.from_numpy(fv)[None]
    logits, v = net(x)
    xz = x.clone()
    xz[:, 2 * TILE_CELLS:3 * TILE_CELLS] = 0
    t = xz[:, :TILE_CELLS].long().clamp(0, 255).view(-1, 18, 20)
    inp9 = torch.cat([net.enc.emb(t).permute(0, 3, 1, 2), xz[:, TILE_CELLS:2 * TILE_CELLS].view(-1, 1, 18, 20)], 1)
    ref = torch.nn.functional.conv2d(inp9, c0.weight[:, :9], c0.bias, padding=1)
    inp10 = torch.cat([inp9, xz[:, 2 * TILE_CELLS:3 * TILE_CELLS].view(-1, 1, 18, 20)], 1)
    res["policy"] = {"pass": c0.in_channels == 10 and logits.shape == (1, 7) and torch.allclose(c0(inp10), ref),
                     "first_conv_in": c0.in_channels}
    res["P_terms_sum_to_P"] = {"pass": pt_ok}
    res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    out = HERE / "results" / "batch2-smoke.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v if not isinstance(v, dict) else ("PASS" if v["pass"] else "FAIL"),
              "" if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "pass"})
    print("wrote", out)


if __name__ == "__main__":
    main()
