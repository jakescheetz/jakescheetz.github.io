#!/usr/bin/env python3
"""A2′ pilot diagnostics, no training (experiment-e1.md §9 "A2′ pilot", researcher ask (1)).

  .venv/bin/python a2p_diagnostics.py traces   [--run A2p-D1-s0]            # from the saved eval traces
  .venv/bin/python a2p_diagnostics.py critic   [--run A2p-D1-s0] [--steps 4096] [--episodes 2]

traces: per-step reward split (ΔT, shaping = r − ΔT) at the two crossings (lab door → Pallet, Pallet →
        Route 1) for the D1 eval episodes that make them; where the policy parks (x, y histogram at Pallet
        y <= 2 and the action mix there), per checkpoint.
critic: for each checkpoint, D1 episodes under the checkpoint's own sampled policy, logging the critic's
        V′(s) and Φ(s) every step. Absorption (shaped PPO = unshaped PPO with the critic offset by Φ)
        predicts V′ ≈ c − Φ: corr(V′, −Φ) → 1 and Ṽ = V′ + Φ flat across positions. Reported per
        checkpoint: corr, slope of V′ on −Φ, std(V′), std(Ṽ), and std(Ṽ) / std(V′).
Outputs: results/A2p-D1-s0-diagnostics.{md,json} (numbers only: positions, rewards, values; no frames).
"""
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import rom, shaping  # noqa: E402

ACTION_NAMES = ("down", "left", "right", "up", "a", "b", "start")
LAB, PALLET, ROUTE1 = shaping.OAKS_LAB, shaping.PALLET, shaping.ROUTE1


def rows_of(path):
    with gzip.open(path, "rt") as f:
        return [r for r in map(json.loads, f) if "map" in r]


def ckpt_step(name):
    return int(name.split("-")[1].split(".")[0])


def crossing_window(rows, i, before=4, after=4):
    out = []
    for r, prev in zip(rows[max(1, i - before):i + after + 1], rows[max(0, i - before - 1):i + after]):
        dT = r["T"] - prev["T"]
        out.append({"ep_step": r["ep_step"], "map": r["map"], "x": r["x"], "y": r["y"],
                    "action": ACTION_NAMES[r["action"]], "r": round(r["r"], 5), "dT": dT,
                    "shaping": round(r["r"] - dT, 5),
                    "phi": round(shaping.phi(r["map"], r["x"], r["y"], r["T"]), 4)})
    return out


def traces(run):
    tdir = rom.runs_dir() / run / "eval" / "traces"
    by_ck = defaultdict(list)
    for p in sorted(tdir.glob("eval-*-D1-*.jsonl.gz")):
        by_ck[p.name.split("-D1-")[0]].append(p)
    res = {"crossings": {}, "parking": {}, "shaping_stats": {}}
    for ck in sorted(by_ck, key=lambda c: ckpt_step(c)):
        lab_exit, to_r1 = [], []
        park, acts, n_rows, sh = Counter(), Counter(), 0, defaultdict(list)
        for p in by_ck[ck]:
            rows = rows_of(p)
            for i in range(1, len(rows)):
                a, b = rows[i - 1], rows[i]
                if a["map"] == LAB and b["map"] == PALLET and len(lab_exit) < 2:
                    lab_exit.append({"episode": p.name, "rows": crossing_window(rows, i)})
                if a["map"] == PALLET and b["map"] == ROUTE1 and len(to_r1) < 2:
                    to_r1.append({"episode": p.name, "rows": crossing_window(rows, i)})
            for r, prev in zip(rows[1:], rows):
                n_rows += 1
                if r["map"] == PALLET and r["y"] <= 2:
                    park[(r["x"], r["y"])] += 1
                    acts[ACTION_NAMES[r["action"]]] += 1
                # shaping by move kind: toward / away / still (Φ change)
                d = shaping.phi(r["map"], r["x"], r["y"], r["T"]) - shaping.phi(prev["map"], prev["x"], prev["y"], prev["T"])
                kind = "toward" if d > 1e-9 else "away" if d < -1e-9 else "level"
                sh[kind].append(r["r"] - (r["T"] - prev["T"]))
        res["crossings"][ck] = {"lab_to_pallet": lab_exit, "pallet_to_route1": to_r1}
        tot = sum(park.values())
        res["parking"][ck] = {"rows": n_rows, "pallet_y_le_2_rows": tot, "share": round(tot / max(n_rows, 1), 4),
                              "top_squares": [[f"{x},{y}", c] for (x, y), c in park.most_common(8)],
                              "actions": dict(acts.most_common())}
        res["shaping_stats"][ck] = {k: {"n": len(v), "mean": round(sum(v) / len(v), 5)} for k, v in sh.items() if v}
    return res


def critic(run, steps, episodes):
    import numpy as np
    import torch
    import train as TR
    from pokered import env as E
    from pokered.policy import ActorCritic, flatten_obs
    rd = rom.runs_dir() / run
    cfg = json.loads((rd / "config.json").read_text())
    out = {}
    for ck in sorted(rd.glob("eval-*.pt"), key=lambda p: ckpt_step(p.name)):
        model = ActorCritic(cfg["policy"])
        model.load_state_dict(torch.load(ck, weights_only=False)["model"])
        model.eval()
        V, PHI, POS = [], [], []
        for ep in range(episodes):
            seed = 1000 + ep
            env = E.PokemonRedEnv(state_name="D1", reward=TR.make_reward(cfg), max_steps=steps, seed=seed,
                                  obs_mode=cfg["obs_mode"], fast_mode=cfg["fast_mode"],
                                  frames_per_action=cfg["frames_per_action"], keep_trace=False)
            g = torch.Generator().manual_seed(seed)
            o, done = env.reset(), False
            while not done:
                with torch.no_grad():
                    logits, v = model(torch.from_numpy(flatten_obs(cfg["policy"], o))[None])
                    a = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).item()
                mp, x, y = env.pb.memory[0xD35E], env.pb.memory[0xD362], env.pb.memory[0xD361]
                V.append(float(v.reshape(-1)[0]))
                PHI.append(shaping.phi(mp, x, y, env.T))
                POS.append((mp, x, y))
                o, _, done, _ = env.step(a)
            env.close()
        V, PHI = np.array(V), np.array(PHI)
        Vt = V + PHI
        sd_phi = PHI.std()
        corr = float(np.corrcoef(V, -PHI)[0, 1]) if sd_phi > 0 and V.std() > 0 else None
        slope = float(np.polyfit(-PHI, V, 1)[0]) if sd_phi > 0 else None
        out[ck.name] = {"steps": len(V), "distinct_squares": len(set(POS)), "phi_range": [round(PHI.min(), 4), round(PHI.max(), 4)],
                        "std_phi": round(float(sd_phi), 5), "corr_V_negphi": None if corr is None else round(corr, 4),
                        "slope_V_on_negphi": None if slope is None else round(slope, 4),
                        "std_V": round(float(V.std()), 5), "std_Vtilde": round(float(Vt.std()), 5),
                        "mean_V": round(float(V.mean()), 4), "ratio_std_Vtilde_V": round(float(Vt.std() / V.std()), 4) if V.std() else None}
        print(ck.name, out[ck.name], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["traces", "critic"])
    ap.add_argument("--run", default="A2p-D1-s0")
    ap.add_argument("--steps", type=int, default=4096)
    ap.add_argument("--episodes", type=int, default=2)
    a = ap.parse_args()
    out = Path(__file__).resolve().parent / "results" / f"{a.run}-diagnostics.json"
    blob = json.loads(out.read_text()) if out.exists() else {}
    if a.what == "traces":
        blob["traces"] = traces(a.run)
    else:
        blob["critic"] = {"steps_per_episode": a.steps, "episodes": a.episodes, "per_ckpt": critic(a.run, a.steps, a.episodes)}
    out.write_text(json.dumps(blob, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
