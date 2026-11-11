#!/usr/bin/env python3
"""A2″ critic probe (experiment-e1.md §9 "Step 0 applied", researcher check (ii) + the blackout hypothesis).
No training.

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python a2pp_critic_probe.py [--run A2pp-D1-s0] [--seeds 0 2]

Hypothesis (researcher, inference): after a blackout the agent stands in Pallet with T >= 1, and that looks to the
critic like Pallet at the start of an episode (T = 0). The observation has no T, no event flags and no step
counter (pokered/env.py ram_features), so V can't tell the two apart.

1. Observations, from the BFS-greedy planner (greedy_planner.py) under the training obs_mode, per seed:
   * d1_start         the D1 reset observation (Oak's Lab, T = 0)
   * pallet_start     the first step in Pallet after leaving the lab (T = 0)
   * post_blackout    the first step with control after a wipe-out (game respawns the party; T >= 1)
     (seeds 0 and 2 black out in the A2″ validation run; seed 1 doesn't)
2. For every retained checkpoint of the run (eval-*.pt, ckpt-*.pt), the critic's V at each observation, and the
   policy's action distribution there.
3. How far apart the observations are: the L2 distance between the observation vectors, and which RAM features
   differ.

Output: results/<run>-critic-probe.json (numbers only; no frames, no tiles).

--matched (researcher's follow-up): instead, every Pallet square the planner stands on at T = 0 before any wipe-out
and at T >= 1 after one (seed 2: respawn at T = 3, then it walks north again). On each square visited in both
phases, V(T = 0) vs V(T >= 1) per checkpoint, and the RAM features that differ. Output:
results/<run>-critic-probe-matched.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import greedy_planner as G  # noqa: E402
from pokered import env as E, ram, rom  # noqa: E402

RAM_NAMES = ["hp_frac", "lvl_sum", "lvl_max", "wild", "trainer", "party", "x", "y", "menu", "seen"]


def collect(seed, obs_mode, budget=6000):
    """Run the greedy planner once, recording the three observations."""
    got = {}
    base = E.PokemonRedEnv

    class Rec(base):
        def __init__(self, *a, **k):
            k["obs_mode"] = obs_mode
            super().__init__(*a, **k)
            self._lost = False

        def reset(self):
            o = super().reset()
            got.setdefault("d1_start", (o, self._where()))
            return o

        def _where(self):
            m = self.pb.memory
            return {"map": m[ram.W_CUR_MAP], "x": m[0xD362], "y": m[0xD361], "T": self.T,
                    "hp": ram.party_hp(m), "levels": ram.party_levels(m)}

        def step(self, a):
            o, r, d, info = super().step(a)
            m = self.pb.memory
            if m[ram.W_IS_IN_BATTLE] == 0xFF:
                self._lost = True
            w = self._where()
            if w["map"] == 0x00 and not self._lost and self.T == 0:
                got.setdefault("pallet_start", (o, w))
            if self._lost and not m[ram.W_IS_IN_BATTLE] and w["map"] in (0x00, 0x25, 0x26) \
                    and "post_blackout" not in got:
                got["post_blackout"] = (o, w)
            return o, r, d, info

    G.PokemonRedEnv = Rec
    try:
        G.run(seed, target=4, budget=budget, log=lambda *a: None)
    finally:
        G.PokemonRedEnv = base
    return got


def collect_matched(seed, obs_mode, budget=6000):
    """{(x, y): obs} in Pallet at T = 0 before any wipe-out, and at T >= 1 after one (first visit each)."""
    pre, post = {}, {}
    base = E.PokemonRedEnv

    class Rec(base):
        def __init__(self, *a, **k):
            k["obs_mode"] = obs_mode
            super().__init__(*a, **k)
            self._lost = False

        def step(self, a):
            o, r, d, info = super().step(a)
            m = self.pb.memory
            if m[ram.W_IS_IN_BATTLE] == 0xFF or (self.blackout_steps and self.blackout_steps[-1] == self.steps):
                self._lost = True
            if m[ram.W_CUR_MAP] == 0x00 and not m[ram.W_IS_IN_BATTLE]:
                xy = (m[0xD362], m[0xD361])
                if not self._lost and self.T == 0:
                    pre.setdefault(xy, (o, self.T))
                elif self._lost and self.T >= 1:
                    post.setdefault(xy, (o, self.T))
            return o, r, d, info

    G.PokemonRedEnv = Rec
    try:
        G.run(seed, target=4, budget=budget, log=lambda *a: None)
    finally:
        G.PokemonRedEnv = base
    return pre, post


def matched(a, cfg, rd):
    import torch
    from pokered.policy import ActorCritic, flatten_obs
    pre, post = collect_matched(a.seeds[0], cfg["obs_mode"])
    common = sorted(set(pre) & set(post))
    print(f"seed {a.seeds[0]}: {len(pre)} T=0 squares, {len(post)} post-blackout squares, {len(common)} in common")
    X0 = {xy: flatten_obs(cfg["policy"], pre[xy][0]) for xy in common}
    X1 = {xy: flatten_obs(cfg["policy"], post[xy][0]) for xy in common}
    diff = {f"{x},{y}": {n: [round(float(X0[(x, y)][-74 + j]), 3), round(float(X1[(x, y)][-74 + j]), 3)]
                         for j, n in enumerate(RAM_NAMES) if abs(X0[(x, y)][-74 + j] - X1[(x, y)][-74 + j]) > 1e-6}
            | {"tiles_sprites_l2": round(float(np.linalg.norm(X0[(x, y)][:-74] - X1[(x, y)][:-74])), 2)}
            for x, y in common}
    cks = sorted(list(rd.glob("eval-*.pt")) + list(rd.glob("ckpt-*.pt")), key=lambda p: int(p.stem.split("-")[1]))
    per_ck = {}
    for ck in cks:
        model = ActorCritic(cfg["policy"])
        model.load_state_dict(torch.load(ck, weights_only=False)["model"])
        model.eval()
        with torch.no_grad():
            v0 = model(torch.from_numpy(np.stack([X0[xy] for xy in common])))[1].reshape(-1).numpy()
            v1 = model(torch.from_numpy(np.stack([X1[xy] for xy in common])))[1].reshape(-1).numpy()
        d = v1 - v0
        per_ck[ck.name] = {"mean_V_T0": round(float(v0.mean()), 4), "mean_V_post": round(float(v1.mean()), 4),
                           "mean_diff": round(float(d.mean()), 4), "mean_abs_diff": round(float(np.abs(d).mean()), 4),
                           "corr": round(float(np.corrcoef(v0, v1)[0, 1]), 3) if len(common) > 2 else None,
                           "per_square": {f"{x},{y}": [round(float(p), 4), round(float(q), 4)]
                                          for (x, y), p, q in zip(common, v0, v1)}}
        print(ck.name, {k: v for k, v in per_ck[ck.name].items() if k != "per_square"}, flush=True)
    out = Path(__file__).resolve().parent / "results" / f"{a.run}-critic-probe-matched.json"
    out.write_text(json.dumps({"run": a.run, "seed": a.seeds[0], "post_T": sorted({post[xy][1] for xy in common}),
                               "squares": [f"{x},{y}" for x, y in common], "obs_diff": diff, "per_ckpt": per_ck},
                              indent=1))
    print("wrote", out)


def main():
    import torch
    from pokered.policy import ActorCritic, flatten_obs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="A2pp-D1-s0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--matched", action="store_true", help="matched Pallet squares, T = 0 vs post-blackout (seed = first of --seeds)")
    a = ap.parse_args()
    rd = rom.runs_dir() / a.run
    cfg = json.loads((rd / "config.json").read_text())
    if a.matched:
        return matched(a, cfg, rd)
    obs = {}
    for s in a.seeds:
        for k, (o, w) in collect(s, cfg["obs_mode"]).items():
            obs[f"s{s}:{k}"] = (flatten_obs(cfg["policy"], o), w)
    keys = sorted(obs)
    ram_off = -74
    dist = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            v1, v2 = obs[k1][0], obs[k2][0]
            diff = {n: [round(float(v1[ram_off + j]), 3), round(float(v2[ram_off + j]), 3)]
                    for j, n in enumerate(RAM_NAMES) if abs(v1[ram_off + j] - v2[ram_off + j]) > 1e-6}
            dist[f"{k1} | {k2}"] = {"l2_all": round(float(np.linalg.norm(v1 - v2)), 3),
                                    "l2_ram": round(float(np.linalg.norm(v1[ram_off:] - v2[ram_off:])), 4),
                                    "ram_diff": diff}
    cks = sorted(list(rd.glob("eval-*.pt")) + list(rd.glob("ckpt-*.pt")), key=lambda p: int(p.stem.split("-")[1]))
    per_ck = {}
    for ck in cks:
        model = ActorCritic(cfg["policy"])
        model.load_state_dict(torch.load(ck, weights_only=False)["model"])
        model.eval()
        row = {}
        for k in keys:
            with torch.no_grad():
                logits, v = model(torch.from_numpy(obs[k][0])[None])
            p = torch.softmax(logits, -1)[0].numpy()
            row[k] = {"V": round(float(v.reshape(-1)[0]), 4),
                      "pi": {n: round(float(x), 3) for n, x in zip(("down", "left", "right", "up", "a", "b", "start"), p)}}
        per_ck[ck.name] = row
        print(ck.name, {k: row[k]["V"] for k in keys}, flush=True)
    out = Path(__file__).resolve().parent / "results" / f"{a.run}-critic-probe.json"
    out.write_text(json.dumps({"run": a.run, "where": {k: obs[k][1] for k in keys}, "obs_distance": dist,
                               "per_ckpt": per_ck}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
