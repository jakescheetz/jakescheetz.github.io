#!/usr/bin/env python3
"""Train-vs-eval P gap diagnostic (lead's ask, 2026-11-08; §9 H4′ trip 2 context). Dev state only. Needs the ROM.

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python gap_check.py RUN CKPT --mode eval|train [--n 8] [--skip 4]
      -> one JSON line per episode + a mean, to stdout and results/gap-check-<RUN>-<mode>.jsonl

A frozen checkpoint, scored exactly as evaluate.py scores (env.P, T, leg progress), in one of two env configs:
  eval   evaluate.episode's env: seeds 1000.., a fresh env per episode (first jitter draw of each seed)
  train  train.worker's env, argument for argument: seed = cfg seed·1000 + idx + 7919·resumes for env idx 0..n-1,
         keep_trace False, the same run_id/meta; --skip K episode resets first, so the jitter draw is the one the
         worker used for its (K+1)-th episode (the training window (0.5M, 1M] = episodes 4..7 of each worker)
Actions: torch.distributions.Categorical(logits).sample() as in the trainer (eval's multinomial(softmax) is the same
distribution), torch seeded per episode. Only the env config differs between the modes; the policy is the same frozen
checkpoint, so a gap between them is an env-config effect and no gap means the cause is elsewhere (the policy).
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import rom  # noqa: E402

HERE = Path(__file__).resolve().parent


def episode(job):
    import torch
    torch.set_num_threads(1)
    import train as TR
    from pokered import env as E, walkdist
    from pokered.policy import ActorCritic, flatten_obs
    run, ck, mode, idx, skip = job
    cfg = json.loads((rom.runs_dir() / run / "config.json").read_text())
    model = ActorCritic(cfg["policy"])
    model.load_state_dict(torch.load(rom.runs_dir() / run / ck, weights_only=False)["model"])
    model.eval()
    if mode == "train":
        env = E.PokemonRedEnv(
            state_name=cfg["state"], reward=TR.make_reward(cfg), max_steps=cfg["hp"]["episode_steps"],
            seed=cfg["seed"] * 1000 + idx + 7919 * cfg.get("resumes", 0), jitter=True,
            run_id=cfg["run_id"], meta={"arm": cfg["arm"], "seed": cfg["seed"], "round": cfg["round"]},
            obs_mode=cfg["obs_mode"], fast_mode=cfg["fast_mode"], frames_per_action=cfg["frames_per_action"],
            keep_trace=True, sanity=not cfg["smoke"], expect_levels=cfg.get("expect_levels"))
        for _ in range(skip):
            env.reset()
        seed = cfg["seed"] * 1000 + idx
    else:
        seed = 1000 + idx
        env = E.PokemonRedEnv(state_name=cfg["state"], reward=TR.make_reward(cfg), max_steps=cfg["hp"]["episode_steps"],
                              seed=seed, obs_mode=cfg["obs_mode"], fast_mode=cfg["fast_mode"],
                              frames_per_action=cfg["frames_per_action"], keep_trace=True,
                              expect_levels=cfg.get("expect_levels"))
    torch.manual_seed(seed)
    o, done = env.reset(), False
    while not done:
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(flatten_obs(cfg["policy"], o))[None])
            a = torch.distributions.Categorical(logits=logits).sample().item()
        o, _, done, _ = env.step(a)
    out = {"mode": mode, "idx": idx, "seed": seed, "skip": skip, "ckpt": ck, "P": round(env.P, 4), "T": env.T,
           "steps": env.steps, "P_terms": {k: round(v, 3) for k, v in env.P_terms.items()},
           "leg_progress": walkdist.leg_progress(env.trace, E.walk_dist()),
           "distinct_tiles": len({(r["map"], r["x"], r["y"]) for r in env.trace})}
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("ckpt")
    ap.add_argument("--mode", choices=["eval", "train"], required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(a.run, a.ckpt, a.mode, i, a.skip if a.mode == "train" else 0) for i in range(a.n)]
    out = HERE / "results" / f"gap-check-{a.run}-{a.mode}.jsonl"
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool, open(out, "w") as f:
        for r in pool.imap(episode, jobs):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
    m = lambda k: round(sum(r[k] for r in rows) / len(rows), 3)
    print(json.dumps({"mode": a.mode, "ckpt": a.ckpt, "n": len(rows), "P_mean": m("P"), "T_mean": m("T"),
                      "legP_mean": m("leg_progress"), "tiles_mean": m("distinct_tiles")}))


if __name__ == "__main__":
    main()
