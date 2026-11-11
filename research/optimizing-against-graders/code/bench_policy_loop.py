#!/usr/bin/env python3
"""Env steps/s with a CNN policy in the loop: PyBoy vs Crafter, on this machine.

Setup (typical of PPO on pixels): N env worker processes, one learner process
that batches observations, runs a small CNN policy (torch, CPU), and sends
actions back. We time (1) acting only, and (2) the learner's PPO-style
update cost (forward + backward per sample), then combine them into training
steps/s for 4 PPO epochs per batch.

PyBoy runs a synthetic ROM built in memory by bench_pyboy.build_rom (no game
ROM is downloaded or needed). Each action advances FRAMES_PER_ACTION frames.
Observation: 72x80 grayscale screen (a downsample of 144x160).
Crafter: 64x64 RGB, its standard observation.
pokered: the real E1 env (pokered/env.py) on Jake's ROM: 24 frames/action,
RAM readers, proxy reward and trace rows all included in the step cost.

Usage:  python bench_policy_loop.py [--envs 1 2 3 4] [--steps 2000]
Output: table on stdout.
"""
import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import time

import numpy as np

FRAMES_PER_ACTION = 24


def worker(kind, rom, conn):
    if kind == "pokered":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pokered.env import PokemonRedEnv
        env = PokemonRedEnv(state_name=os.environ.get("E1_STATE") or None, max_steps=10 ** 9)
        conn.send(env.reset()["screen"][None])

        def step(a):
            return env.step(int(a) % 7)[0]["screen"][None]
    elif kind == "pyboy":
        from pyboy import PyBoy
        pb = PyBoy(rom, window="null", sound_emulated=False)
        pb.set_emulation_speed(0)

        def obs():
            s = pb.screen.ndarray[::2, ::2, :3].mean(axis=2).astype(np.uint8)
            return s[None]                                  # 1x72x80

        def step(a):
            pb.tick(FRAMES_PER_ACTION, True)
            _ = pb.memory[0xD356]                           # reward reads RAM
            return obs()
        conn.send(obs())
    else:
        import crafter
        env = crafter.Env(size=(64, 64), seed=0)
        o = env.reset()

        def step(a):
            o, r, done, info = env.step(int(a))
            if done:
                o = env.reset()
            return o.transpose(2, 0, 1)                     # 3x64x64
        conn.send(o.transpose(2, 0, 1))
    while True:
        a = conn.recv()
        if a is None:
            break
        conn.send(step(a))


def make_policy(in_ch, n_act):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten(), nn.LazyLinear(512), nn.ReLU(), nn.Linear(512, n_act + 1))


def bench(kind, rom, n_envs, steps):
    import torch
    torch.set_num_threads(1)
    pipes, procs = [], []
    for _ in range(n_envs):
        a, b = mp.Pipe()
        p = mp.Process(target=worker, args=(kind, rom, b), daemon=True)
        p.start()
        pipes.append(a)
        procs.append(p)
    obs = np.stack([c.recv() for c in pipes])
    n_act = {"pyboy": 8, "pokered": 7}.get(kind, 17)
    pol = make_policy(obs.shape[1], n_act)
    with torch.no_grad():
        pol(torch.zeros((1,) + obs.shape[1:]))
    t0 = time.perf_counter()
    done = 0
    while done < steps:
        with torch.no_grad():
            logits = pol(torch.from_numpy(obs).float() / 255)[:, :n_act]
            acts = torch.distributions.Categorical(logits=logits).sample().numpy()
        for c, a in zip(pipes, acts):
            c.send(a)
        obs = np.stack([c.recv() for c in pipes])
        done += n_envs
    act_sps = done / (time.perf_counter() - t0)
    for c in pipes:
        c.send(None)
    for p in procs:
        p.join()

    # learner cost: fwd+bwd on minibatches of 256 (threads = all cores, envs idle)
    torch.set_num_threads(os.cpu_count())
    opt = torch.optim.Adam(pol.parameters(), 2.5e-4)
    x = torch.rand((256,) + obs.shape[1:])
    t0 = time.perf_counter()
    for _ in range(8):
        loss = pol(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    learn_sps = 8 * 256 / (time.perf_counter() - t0)
    train_sps = 1 / (1 / act_sps + 4 / learn_sps)              # 4 PPO epochs
    return act_sps, learn_sps, train_sps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--kinds", nargs="+", default=["pyboy", "crafter"],
                    help="pyboy (synthetic ROM), pokered (real ROM via $POKEMON_RED_ROM, "
                         "optional start state $E1_STATE), crafter")
    a = ap.parse_args()
    import torch
    from bench_pyboy import build_rom
    print(f"torch {torch.__version__}, cpus {os.cpu_count()}, "
          f"pyboy frames/action {FRAMES_PER_ACTION}")
    print(f"{'env':>7} {'n_envs':>6} {'act sps':>8} {'learn sps':>9} {'train sps':>9} {'train/h':>9}")
    with tempfile.TemporaryDirectory() as d:
        rom = os.path.join(d, "bench.gb")
        build_rom(rom)
        for kind in a.kinds:
            for n in a.envs:
                act, learn, train = bench(kind, rom, n, a.steps)
                print(f"{kind:>7} {n:>6} {act:>8.0f} {learn:>9.0f} {train:>9.0f} {train*3600:>9.0f}")


if __name__ == "__main__":
    main()
