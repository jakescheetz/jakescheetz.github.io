#!/usr/bin/env python3
"""Lean-compute benchmark for E1 on the real ROM (code/README.md, "Lean compute").

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python bench_lean.py

Parts:
  1. env-only steps/s for each config (1 process, random actions, seed 0):
     obs mode (screen / tiles / ram), PyBoy fast_mode, frames per action, trace kept or not;
  2. fast_mode fidelity: same actions with and without fast_mode, compare RAM per step;
  3. policy cost: CNN on the screen vs small CNN on tiles vs MLP on RAM features;
  4. end-to-end training steps/s for the lean config with n parallel envs.
Bench start state: until Jake's D1 exists, a deterministic overworld state made by
3,000 seeded random steps from power-on, cached as <states dir>/bench-boot3000.state
(outside the repo). Pass --state D1 to use a real start state instead.
"""
import argparse
import hashlib
import io
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import env as E  # noqa: E402
from pokered import rom  # noqa: E402

N_ACT = len(E.ACTIONS)


def bench_state(name):
    if name:
        return (rom.states_dir() / f"{name}.state").read_bytes()
    path = rom.states_dir() / "bench-boot3000.state"
    if not path.exists():
        print("making bench state (3,000 seeded random steps from power-on)...", flush=True)
        env = E.PokemonRedEnv(obs_mode="ram", jitter=False, keep_trace=False, max_steps=10 ** 9)
        env.reset()
        rng = random.Random(0)
        for _ in range(3000):
            env.step(rng.randrange(N_ACT))
        buf = io.BytesIO()
        env.pb.save_state(buf)
        path.write_bytes(buf.getvalue())
        env.close()
    return path.read_bytes()


def make_env(st, cfg):
    return E.PokemonRedEnv(state_bytes=st, state_name="bench", sanity=False, jitter=False,
                           max_steps=10 ** 9, **cfg)


def env_speed(st, cfg, steps):
    env = make_env(st, cfg)
    env.reset()
    rng = random.Random(0)
    t = time.perf_counter()
    for _ in range(steps):
        env.step(rng.randrange(N_ACT))
    sps = steps / (time.perf_counter() - t)
    env.close()
    return sps


SOUND_SERIAL = set(range(0xFF01, 0xFF03)) | set(range(0xFF10, 0xFF40))   # §5b: allowed to differ


def fidelity(st, steps, jitter_seed=0):
    """§5b check: same actions and jitter, fast_mode off vs on; compare WRAM
    0xC000-0xDFFF and HRAM 0xFF80-0xFFFE at every step (HRAM holds the RNG,
    hRandomAdd/Sub at 0xFFD3-4, and hFrameCounter)."""
    runs = {}
    for fast in (False, True):
        env = make_env(st, {"obs_mode": "ram", "fast_mode": fast, "keep_trace": True})
        env.jitter, env.rng = True, random.Random(jitter_seed)
        env.reset()
        rng = random.Random(0)
        snaps = []
        for _ in range(steps):
            env.step(rng.randrange(N_ACT))
            snaps.append(bytes(env.pb.memory[0xC000:0xE000]) + bytes(env.pb.memory[0xFF80:0xFFFF]))
        runs[fast] = (snaps, env.trace)
        env.close()
    a, b = runs[False][0], runs[True][0]
    first, addrs = None, []
    for i in range(steps):
        if a[i] != b[i]:
            first = i
            addrs = [(0xC000 + j if j < 0x2000 else 0xFF80 + j - 0x2000)
                     for j in range(len(a[i])) if a[i][j] != b[i][j]]
            break
    tr = runs[False][1]
    coverage = {"wild_battle_steps": sum(r["in_battle"] == 1 for r in tr),
                "trainer_battle_steps": sum(r["in_battle"] == 2 for r in tr),
                "maps": sorted({r["map"] for r in tr}),
                "rng_changes": sum(1 for i in range(1, steps)
                                   if a[i][0x2000 + 0x53:0x2000 + 0x55] != a[i - 1][0x2000 + 0x53:0x2000 + 0x55])}
    return first, addrs, coverage


# --- policies ---------------------------------------------------------------
def policies():
    import torch.nn as nn

    class TileNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(512, 8)
            self.conv = nn.Sequential(nn.Conv2d(8, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 2), nn.ReLU(),
                                      nn.Flatten(), nn.LazyLinear(256), nn.ReLU(), nn.Linear(256, N_ACT + 1))

        def forward(self, x):
            return self.conv(self.emb(x.long().clamp(0, 511)).permute(0, 3, 1, 2))

    cnn = nn.Sequential(nn.Conv2d(1, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                        nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.LazyLinear(512), nn.ReLU(),
                        nn.Linear(512, N_ACT + 1))
    class TileGridNet(nn.Module):
        """§5a tiles-CNN: wTileMap ids (embedded) + sprite grid -> CNN, concat RAM MLP.
        Input is one flat vector: 360 tile ids, 360 sprite counts, 74 RAM features."""
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(256, 8)
            # slim encoder: 16/32/32 channels, 128 hidden (2.4x faster to train than 32/64/64 + 256)
            self.conv = nn.Sequential(nn.Conv2d(9, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2), nn.ReLU(),
                                      nn.Conv2d(32, 32, 3, 1), nn.ReLU(), nn.Flatten())
            self.ram = nn.Sequential(nn.Linear(74, 64), nn.ReLU())
            self.head = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, N_ACT + 1))

        def forward(self, x):
            t = x[:, :360].long().clamp(0, 255).view(-1, 18, 20)
            sp = x[:, 360:720].view(-1, 1, 18, 20)
            g = torch.cat([self.emb(t).permute(0, 3, 1, 2), sp], 1)
            return self.head(torch.cat([self.conv(g), self.ram(x[:, 720:])], 1))

    import torch
    mlp = nn.Sequential(nn.Linear(74, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, N_ACT + 1))
    return {"cnn-screen": (cnn, (1, 72, 80)), "cnn-tiles": (TileNet(), (18, 20)), "mlp-ram": (mlp, (74,)),
            "cnn-tilegrid": (TileGridNet(), (794,))}


def policy_cost(n_envs=2):
    import torch
    out = {}
    for name, (net, shape) in policies().items():
        x1 = torch.rand((n_envs,) + shape) * (255 if "tile" in name else 1)
        torch.set_num_threads(1)
        with torch.no_grad():
            net(x1)
            t = time.perf_counter()
            for _ in range(200):
                net(x1)
            infer_ms = (time.perf_counter() - t) / 200 * 1000
        torch.set_num_threads(os.cpu_count())
        opt = torch.optim.Adam(net.parameters(), 2.5e-4)
        xb = torch.rand((256,) + shape) * (255 if "tile" in name else 1)
        t = time.perf_counter()
        for _ in range(8):
            loss = net(xb).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        learn_sps = 8 * 256 / (time.perf_counter() - t)
        out[name] = (infer_ms, learn_sps)
    return out


# --- end-to-end -------------------------------------------------------------
OBS_KEY = {"mlp-ram": "ram", "cnn-screen": "screen", "cnn-tiles": "tiles", "cnn-tilegrid": "tilegrid"}


def worker(st, cfg, key, conn):
    env = make_env(st, cfg)

    def pick(o):
        if key == "tilegrid":
            return np.concatenate([o["tiles"].ravel(), o["sprites"].ravel(), o["ram"]]).astype(np.float32)
        return o[key][None] if key == "screen" else o[key]
    conn.send(pick(env.reset()))
    while True:
        a = conn.recv()
        if a is None:
            break
        conn.send(pick(env.step(int(a))[0]))
    env.close()


def end_to_end(st, cfg, n_envs, steps, learn_sps, pol="mlp-ram"):
    import torch
    torch.set_num_threads(1)
    net = policies()[pol][0]
    key = OBS_KEY[pol]
    pipes = []
    for _ in range(n_envs):
        a, b = mp.Pipe()
        mp.Process(target=worker, args=(st, cfg, key, b), daemon=True).start()
        pipes.append(a)
    obs = np.stack([c.recv() for c in pipes])
    with torch.no_grad():
        net(torch.from_numpy(obs).float())
    t = time.perf_counter()
    done = 0
    while done < steps:
        with torch.no_grad():
            x = torch.from_numpy(obs).float() / (255 if key == "screen" else 1)
            acts = torch.distributions.Categorical(logits=net(x)[:, :N_ACT]).sample().numpy()
        for c, a in zip(pipes, acts):
            c.send(a)
        obs = np.stack([c.recv() for c in pipes])
        done += n_envs
    act = done / (time.perf_counter() - t)
    for c in pipes:
        c.send(None)
    return act, 1 / (1 / act + 4 / learn_sps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--skip", nargs="*", default=[], help="parts to skip: env fidelity policy e2e")
    ap.add_argument("--fidelity-steps", type=int, default=10000)
    a = ap.parse_args()
    st = bench_state(a.state)
    import pyboy  # noqa: F401
    import torch
    print(f"torch {torch.__version__}, cpus {os.cpu_count()}, state {a.state or 'bench-boot3000'}")

    configs = [
        ("baseline: screen obs, 24 f/a", dict(obs_mode="screen")),
        ("tiles obs (no render)", dict(obs_mode="tiles")),
        ("ram obs (no render)", dict(obs_mode="ram")),
        ("ram + no trace kept", dict(obs_mode="ram", keep_trace=False)),
        ("ram + fast_mode", dict(obs_mode="ram", fast_mode=True)),
        ("ram + fast_mode + 16 f/a*", dict(obs_mode="ram", fast_mode=True, frames_per_action=16)),
        ("screen + fast_mode", dict(obs_mode="screen", fast_mode=True)),
        ("tiles + fast_mode", dict(obs_mode="tiles", fast_mode=True)),
        ("tilegrid (§5a) + fast_mode", dict(obs_mode="tilegrid", fast_mode=True)),
    ]
    if "env" not in a.skip:
        print("\n1. env-only, 1 process (* = 16 f/a validity pending --check-frames)")
        for name, cfg in configs:
            print(f"  {name:<34} {env_speed(st, cfg, a.steps):>6.0f} steps/s", flush=True)
    if "fidelity" not in a.skip:
        first, addrs, cov = fidelity(st, a.fidelity_steps)
        bad = [x for x in addrs if x not in SOUND_SERIAL]
        verdict = ("PASS (byte-identical)" if first is None else
                   "PASS (sound/serial only)" if not bad else "FAIL")
        print(f"\n2. fast_mode fidelity, WRAM+HRAM, {a.fidelity_steps} steps: {verdict}; "
              f"first divergence step {first}, addresses {[hex(x) for x in addrs[:12]]}")
        print(f"   coverage: {cov}")
    costs = policy_cost()
    if "policy" not in a.skip:
        print("\n3. policy cost (inference: 2 envs, 1 thread; learn: minibatch 256, all threads)")
        for name, (ms, ls) in costs.items():
            print(f"  {name:<12} infer {ms:>6.2f} ms/batch   learn {ls:>7.0f} samples/s")
    if "e2e" not in a.skip:
        print("\n4. end-to-end, 4 PPO epochs per batch")
        for cfg_name, cfg, pol in [("ram, MLP", dict(obs_mode="ram"), "mlp-ram"),
                                   ("ram + fast_mode, MLP", dict(obs_mode="ram", fast_mode=True), "mlp-ram"),
                                   ("screen + fast_mode, CNN", dict(obs_mode="screen", fast_mode=True), "cnn-screen"),
                                   ("tilegrid + fast_mode, tiles-CNN", dict(obs_mode="tilegrid", fast_mode=True),
                                    "cnn-tilegrid")]:
            for n in (1, 2, 3, 4):
                act, train = end_to_end(st, cfg, n, a.steps, costs[pol][1], pol)
                print(f"  {cfg_name:<26} envs {n}: act {act:>5.0f}  train {train:>5.0f} steps/s "
                      f"({train * 3600 / 1e3:.0f}k/h)", flush=True)


if __name__ == "__main__":
    main()
