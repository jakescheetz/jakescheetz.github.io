#!/usr/bin/env python3
"""§9 H9 trip-2 diagnostic D-1..D-5: does eval measure the trained policy? Dev state (D1) only. Needs the ROM.

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python gap_diag.py d1 RUN CKPT [--episodes 2]
        training rollout path, frozen: train.VecEnv (the trainer's env factory and worker loop, 8 envs, seeds
        cfg seed·1000 + idx), batched forward on torch.from_numpy(venv.obs), Categorical(logits).sample(),
        no updates; each env plays --episodes full episodes; rows = the worker's own episode dicts (env.P etc.)
  ... gap_diag.py d2 RUN CKPT [--steps 256]
        bit check: one env per factory (train.worker's arguments vs evaluate.episode's) with the same seed;
        the same action each step (sampled once from the training-path logits); compare obs and logits
        (training path: batched [1, D] from the stacked obs; eval path: flatten_obs(...)[None]); diff the env
        kwargs; load the checkpoint with strict=False and list missing / unexpected keys
  ... gap_diag.py d34 RUN                  D-3 step-0 anchor + D-4 counting, from the training rows
Outputs: results/gap-diag-<RUN>-<part>.json (+ printed). D-5 uses gap_check.py --mode train (training jitter
seeds, the 5th-episode draw) vs --mode eval; the evaluate.py leg of D-1 is evaluate.py itself.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import rom  # noqa: E402

HERE = Path(__file__).resolve().parent


def load(run, ck):
    import torch
    from pokered.policy import ActorCritic
    cfg = json.loads((rom.runs_dir() / run / "config.json").read_text())
    m = ActorCritic(cfg["policy"])
    sd = torch.load(rom.runs_dir() / run / ck, weights_only=False)["model"]
    res = m.load_state_dict(sd, strict=False)
    m.eval()
    return cfg, m, {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys),
                    "n_tensors": len(sd)}


def d1(run, ck, episodes):
    import numpy as np
    import torch
    import train as TR
    cfg, model, keys = load(run, ck)
    torch.set_num_threads(cfg.get("train_threads") or 4)
    torch.manual_seed(cfg["seed"])
    n = cfg["envs"]
    venv = TR.VecEnv(cfg, n)
    obs = torch.from_numpy(venv.obs)
    eps, per_env = [], [0] * n
    while min(per_env) < episodes:
        with torch.no_grad():
            logits, _ = model(obs)
            a = torch.distributions.Categorical(logits=logits).sample()
        o, r, d, infos = venv.step(a.numpy())
        obs = torch.from_numpy(o)
        for i, ep in enumerate(infos):
            if ep and per_env[i] < episodes:
                per_env[i] += 1
                eps.append({"env": i, "episode": per_env[i], **{k: ep[k] for k in ("P", "T", "steps", "P_terms")}})
                print(json.dumps(eps[-1]), flush=True)
    venv.close()
    mean = lambda k: round(sum(e[k] for e in eps) / len(eps), 3)
    out = {"part": "D-1 training rollout path (frozen)", "run": run, "ckpt": ck, "n": len(eps), "P_mean": mean("P"),
           "T_mean": mean("T"), "steps": sorted({e["steps"] for e in eps}),
           "P_terms_mean": {k: round(sum(e["P_terms"][k] for e in eps) / len(eps), 3) for k in ("explore", "level", "heal")},
           "load": keys, "episodes": eps}
    (HERE / "results" / f"gap-diag-{run}-d1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}))


def d2(run, ck, steps):
    import inspect
    import numpy as np
    import torch
    import evaluate as EV
    import train as TR
    from pokered import env as E
    from pokered.policy import flatten_obs
    torch.set_num_threads(1)
    cfg, model, keys = load(run, ck)
    seed = cfg["seed"] * 1000                       # env idx 0 of the trainer
    kw_train = dict(state_name=cfg["state"], max_steps=cfg["hp"]["episode_steps"], seed=seed, jitter=True,
                    run_id=cfg["run_id"], meta={"arm": cfg["arm"], "seed": cfg["seed"], "round": cfg["round"]},
                    obs_mode=cfg["obs_mode"], fast_mode=cfg["fast_mode"], frames_per_action=cfg["frames_per_action"],
                    keep_trace=False, sanity=not cfg["smoke"], expect_levels=cfg.get("expect_levels"))
    kw_eval = dict(state_name=cfg["state"], max_steps=cfg["hp"]["episode_steps"], seed=seed,
                   obs_mode=cfg["obs_mode"], fast_mode=cfg["fast_mode"], frames_per_action=cfg["frames_per_action"],
                   keep_trace=True, expect_levels=None)
    # the evaluate.py call, read from its source so this diff can't drift from the real code
    src = inspect.getsource(EV.episode)
    defaults = {k: v.default for k, v in inspect.signature(E.PokemonRedEnv.__init__).parameters.items()
                if v.default is not inspect.Parameter.empty}
    full = lambda kw: {k: kw.get(k, defaults.get(k)) for k in sorted(set(defaults) | set(kw))}
    ft, fe = full(kw_train), full(kw_eval)
    cfg_diff = {k: [ft[k], fe[k]] for k in ft if ft[k] != fe[k]}
    et = E.PokemonRedEnv(reward=TR.make_reward(cfg), **kw_train)
    ee = E.PokemonRedEnv(reward=TR.make_reward(cfg), **kw_eval)
    ot, oe = et.reset(), ee.reset()
    g = torch.Generator().manual_seed(0)
    n_obs_diff = n_logit_diff = 0
    max_logit = 0.0
    first_obs_diff = None
    for t in range(steps):
        ft_, fe_ = flatten_obs(cfg["policy"], ot), flatten_obs(cfg["policy"], oe)
        batch = torch.from_numpy(np.stack([ft_]))                      # the trainer's path (VecEnv stacks)
        single = torch.from_numpy(fe_)[None]                            # evaluate.py's path
        if not np.array_equal(ft_, fe_):
            n_obs_diff += 1
            first_obs_diff = first_obs_diff if first_obs_diff is not None else t
        with torch.no_grad():
            lt, _ = model(batch)
            le, _ = model(single)
        dl = float((lt - le).abs().max())
        max_logit = max(max_logit, dl)
        n_logit_diff += dl != 0.0
        a = torch.multinomial(torch.softmax(lt, -1), 1, generator=g).item()
        ot, _, _, _ = et.step(a)
        oe, _, _, _ = ee.step(a)
    out = {"part": "D-2 bit check", "run": run, "ckpt": ck, "seed": seed, "steps": steps,
           "obs_identical_steps": steps - n_obs_diff, "first_obs_diff_step": first_obs_diff,
           "logits_identical_steps": steps - n_logit_diff, "max_abs_logit_diff": max_logit,
           "positions_equal_at_end": [list(et.reward.coords) == list(ee.reward.coords), et.P == ee.P],
           "env_kwargs_diff_train_vs_eval": cfg_diff, "load": keys,
           "evaluate_episode_env_call": [l.strip() for l in src.splitlines() if "PokemonRedEnv(" in l or "frames_per_action=cfg" in l]}
    et.close(), ee.close()
    (HERE / "results" / f"gap-diag-{run}-d2.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


def d34(run):
    d = rom.runs_dir() / run
    cfg = json.loads((d / "config.json").read_text())
    rows = [json.loads(l) for l in open(d / "episodes.jsonl")]
    rows = [r for r in rows if "P" in r and "global_step" in r]
    span = cfg["hp"]["episode_steps"] * cfg["envs"]                      # global steps one env-episode spans
    first = sorted(rows, key=lambda r: r["global_step"])[:cfg["envs"]]
    ev0 = [json.loads(l) for l in open(d / "eval-interim" / "ckpt-0" / "episodes.jsonl")]
    w = [r for r in rows if 500_736 < r["global_step"] <= 1_000_448]
    chk = [abs(sum(r["P_terms"].values()) - r["P"]) < 1e-3 for r in rows if "P_terms" in r]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    out = {"part": "D-3 step-0 anchor + D-4 counting", "run": run,
           "D3": {"first_training_episodes": len(first), "their_global_steps": [r["global_step"] for r in first],
                  "policy_span": f"each was played from global step 0 to ≈{span:,} (≈{span // (128 * cfg['envs'])} PPO updates)",
                  "train_P_first": mean([r["P"] for r in first]), "eval_P_ckpt0": mean([r["P"] for r in ev0]),
                  "train_T_first": mean([r["T"] for r in first]), "eval_T_ckpt0": mean([r["T"] for r in ev0])},
           "D4": {"definition": ("one row per finished env-episode, written when it ends: P = env.P, the proxy summed over "
                                 "all steps of that episode (reset to done; done at episode_steps or the last milestone); "
                                 "global_step = the trainer's step when it ended. 'Since 0.5M' = rows with global_step in "
                                 "(500,736, 1,000,448]; each was played over the preceding ≈ episode_steps × envs global steps"),
                  "episode_steps": cfg["hp"]["episode_steps"], "envs": cfg["envs"], "global_span_per_episode": span,
                  "window_rows": len(w), "window_steps_values": sorted({r["steps"] for r in w}),
                  "window_P_mean_recomputed": mean([r["P"] for r in w]),
                  "window_episodes_started_between": [min(r["global_step"] for r in w) - span, max(r["global_step"] for r in w) - span],
                  "P_equals_sum_of_terms": f"{sum(chk)}/{len(chk)} rows"}}
    (HERE / "results" / f"gap-diag-{run}-d34.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["d1", "d2", "d34"])
    ap.add_argument("run")
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=256)
    a = ap.parse_args()
    {"d1": lambda: d1(a.run, a.ckpt, a.episodes), "d2": lambda: d2(a.run, a.ckpt, a.steps), "d34": lambda: d34(a.run)}[a.part]()


if __name__ == "__main__":
    main()
