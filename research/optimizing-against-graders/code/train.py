#!/usr/bin/env python3
"""PPO trainer for E1, with checkpoint/resume (experiment-e1.md §4, §5, §8).

  # main arms (needs Jake's start states)
  .venv/bin/python train.py --arm A0 --state D1 --seed 0 --steps 5000000 --envs 3
  .venv/bin/python train.py --arm A2 --state HA --seed 1 ...
  .venv/bin/python train.py --arm A2p --state D1 --seed 0 ...                # A2′ shaped oracle (§4)
  .venv/bin/python train.py --arm A1 --round 2 --proxy '{"tile_visit_cap": 600}' --state D1 --seed 0 ...
  # resume after a crash, reboot or spot preemption
  .venv/bin/python train.py --resume A0-D1-s0
  # smoke run without start states (power-on, sanity check off, RAM-MLP)
  .venv/bin/python train.py --arm A0 --smoke --steps 20000 --envs 2

Everything goes to $POKEMON_RUNS_DIR/<run_id>/ (default ~/roms/runs/, outside the repo):
  config.json      full config, incl. reward params/hash, fast_mode, frames/action, versions
  metrics.jsonl    one row per PPO update: cumulative sps, sps_update (this update only),
                   collect/learn seconds, load average, CPU MHz, map mix, losses, episode P/T means
  episodes.jsonl   one row per finished training episode (P, T, milestone steps, hacks)
  ckpt-<step>.pt   rolling checkpoints (last 3); written to a temp file, then renamed
  ckpt-0.pt        the step-0 (initial) policy, saved once at the start of a new run and never pruned
                   (§9 "Batch-1 analysis plan" A′: KL from the init policy; evaluate.py --kl-init checks it)
  eval-<step>.pt   log-spaced checkpoints kept for §5 evaluation (0.25M, 0.5M, 1M, ...)
  <run_id>.png/.json  final frame + sidecar of a closing D1 eval episode (§10.6)

Resume is exact for the learner (weights, optimizer, step counters, RNG states).
In-progress episodes restart from the start state, so a resumed run is a valid run
of the same config, not a bit-identical replay. Hyperparameters are pinned at the
pilot and then frozen for all arms (§5); the defaults below are the pre-pilot choice.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import platform
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import runlog
from pokered import env as E  # noqa: E402
from pokered import ram, rom  # noqa: E402

HPARAMS = dict(rollout=128, epochs=4, minibatch=256, lr=2.5e-4, gamma=0.998, gae_lambda=0.95,
               clip=0.1, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, episode_steps=2048 * 8)
KEEP_CKPTS = 3
SLOW_SPS = 60      # print a line when one update runs below this (the 21-31 sps dip seen in a smoke run)


def perf_row(steps, t_total, t_collect, buf_o):
    """Per-update speed and diagnostics, to tell a game-phase slowdown (maps change)
    from CPU contention (load up) or throttling (MHz down)."""
    from pokered.policy import RAM_DIM
    try:
        with open("/proc/cpuinfo") as f:
            mhz = [float(l.split(":")[1]) for l in f if l.startswith("cpu MHz")]
        mhz = round(sum(mhz) / len(mhz)) if mhz else None
    except OSError:
        mhz = None
    onehot = buf_o[:, :, -RAM_DIM + 10:].reshape(-1, 64)     # map id % 64 one-hot in the RAM vector
    counts = onehot.sum(0)
    maps = {int(k): round(float(counts[k]) / len(onehot), 3) for k in counts.nonzero().flatten().tolist()}
    return {"sps_update": round(steps / t_total, 1), "t_collect": round(t_collect, 2),
            "t_learn": round(t_total - t_collect, 2), "load1": round(os.getloadavg()[0], 2),
            "cpu_mhz": mhz, "maps": maps}


# --- env workers ------------------------------------------------------------
def reward_for(arm, proxy, gamma):
    """A0/A1: the proxy; A2: sparse oracle; A2p (A2′, §4): oracle + potential shaping at the PPO γ;
    A2pp (A2″, §9): oracle + progress bonus on walking distance."""
    if arm == "A2":
        return E.Oracle(**proxy)
    if arm == "A2p":
        return E.ShapedOracle(gamma=gamma, **proxy)
    if arm == "A2pp":                                   # A2″ (§9): ΔT + progress on BFS walking distance
        return E.ProgressOracle(**proxy)
    return E.make_proxy(**proxy)                       # P0 knobs; MapCapProxy if map_new_tile_cap is set


def make_reward(cfg):
    return reward_for(cfg["arm"], cfg["proxy"], cfg["hp"]["gamma"])


def worker(cfg, idx, conn):
    signal.signal(signal.SIGINT, signal.SIG_IGN)      # the trainer owns shutdown (checkpoint first)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    from pokered.policy import flatten_obs
    env = E.PokemonRedEnv(
        state_name=cfg["state"], reward=make_reward(cfg), max_steps=cfg["hp"]["episode_steps"],
        seed=cfg["seed"] * 1000 + idx + 7919 * cfg.get("resumes", 0), jitter=True,
        run_id=cfg["run_id"], meta={"arm": cfg["arm"], "seed": cfg["seed"], "round": cfg["round"]},
        obs_mode=cfg["obs_mode"], fast_mode=cfg["fast_mode"], frames_per_action=cfg["frames_per_action"],
        keep_trace=False, sanity=not cfg["smoke"], expect_levels=cfg.get("expect_levels"))
    kind = cfg["policy"]
    conn.send(flatten_obs(kind, env.reset()))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        o, r, done, info = env.step(int(msg))
        ep = None
        if done:
            ep = {"P": round(env.P, 4), "T": env.T, "steps": env.steps,
                  "R": round(env.R, 4), "R_dT": env.R_dT, "R_shaping": round(env.R - env.R_dT, 4),
                  "milestone_steps": env.milestone_steps, "hacks_active": sorted(env.episode_hacks),
                  "stuck_active": sorted(env.episode_stuck),
                  "hacks_first_step": env.hacks_first,
                  "P_terms": {k: round(v, 4) for k, v in env.P_terms.items()},     # logging only
                  "obs_version": cfg.get("obs_version", 1)}
            if cfg.get("log_blackout_steps", False):
                ep["blackout_steps"] = list(env.blackout_steps)
            o = env.reset()
        conn.send((flatten_obs(kind, o), r, done, ep, info["dT"]))
    env.close()


class VecEnv:
    def __init__(self, cfg, n):
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(cfg, i, b), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)
        self.obs = np.stack([c.recv() for c in self.pipes])

    def step(self, actions):
        for c, a in zip(self.pipes, actions):
            c.send(int(a))
        out = [c.recv() for c in self.pipes]
        self.obs = np.stack([o[0] for o in out])
        self.last_dT = np.array([o[4] for o in out], np.float32)     # ΔT part of each env's reward
        return self.obs, np.array([o[1] for o in out], np.float32), np.array([o[2] for o in out]), [o[3] for o in out]

    def close(self):
        for c in self.pipes:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)


# --- checkpoints ------------------------------------------------------------
EXTRA_EVAL_MARKS = [3_000_000]   # §9 "A2″ 10M extension": a 3M mark for every main arm (new runs only)


def eval_marks(total, extra=()):
    """Log-spaced eval checkpoints: 0.25M, 0.5M, 1M, 2M, ... up to total (§5), plus the run's extra marks
    (cfg["extra_eval_marks"], recorded at creation, so a resumed older run keeps the marks it started with)."""
    marks, m = [], 250_000
    while m < total:
        marks.append(m)
        m *= 2
    marks += [x for x in extra if x < total]
    return sorted(set(marks)) + [total]


def save_ckpt(path, model, opt, state):
    import torch
    blob = {"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}}
    tmp = str(path) + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def latest_ckpt(run_dir):
    cands = glob.glob(str(run_dir / "ckpt-*.pt")) + glob.glob(str(run_dir / "eval-*.pt"))
    return max(cands, key=lambda p: int(Path(p).stem.split("-")[1])) if cands else None


def prune(run_dir):
    ck = sorted(glob.glob(str(run_dir / "ckpt-*.pt")), key=lambda p: int(Path(p).stem.split("-")[1]))
    ck = [p for p in ck if int(Path(p).stem.split("-")[1]) > 0]      # ckpt-0.pt (the init policy) is kept
    for p in ck[:-KEEP_CKPTS]:
        os.remove(p)


# --- eval -------------------------------------------------------------------
def eval_episode(cfg, model, state, seed, save_frame=False, training_step=None):
    """One sampled-policy episode from `state`; returns its summary, optionally
    saving the final frame + sidecar (§10.6)."""
    import torch
    from pokered.policy import flatten_obs
    env = E.PokemonRedEnv(
        state_name=state, reward=make_reward(cfg), max_steps=cfg["hp"]["episode_steps"], seed=seed,
        run_id=cfg["run_id"], meta={"arm": cfg["arm"], "seed": cfg["seed"], "round": cfg["round"],
                                    "eval_seed": seed},
        obs_mode=cfg["obs_mode"], fast_mode=cfg["fast_mode"], frames_per_action=cfg["frames_per_action"],
        keep_trace=False, sanity=not cfg["smoke"])
    g = torch.Generator().manual_seed(seed)
    o, done = env.reset(), False
    while not done:
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(flatten_obs(cfg["policy"], o))[None])
            a = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).item()
        o, _, done, _ = env.step(a)
    out = {"state": state, "seed": seed, "P": round(env.P, 4), "T": env.T,
           "milestone_steps": env.milestone_steps, "hacks_active": sorted(env.episode_hacks),
           "stuck_active": sorted(env.episode_stuck)}
    if save_frame:
        out["frame"] = str(env.save_run_frame(training_step))
    env.close()
    return out


# --- main -------------------------------------------------------------------
def build_cfg(a):
    import pyboy
    import torch
    proxy = json.loads(a.proxy) if a.proxy else {}
    state = a.state or (None if a.smoke else "D1")
    policy = "ram" if (a.smoke and a.policy is None) else (a.policy or "tilegrid")
    run_id = a.run_id or f"{a.arm}{'-r' + str(a.round) if a.arm == 'A1' else ''}-{state or 'boot'}-s{a.seed}"
    reward = reward_for(a.arm, proxy, a.gamma if getattr(a, "gamma", None) is not None else HPARAMS["gamma"])
    hp = dict(HPARAMS)
    for k in HPARAMS:
        v = getattr(a, k, None)
        if v is not None:
            hp[k] = v
    return {
        "run_id": run_id, "arm": a.arm, "round": a.round, "seed": a.seed, "state": state, "smoke": a.smoke,
        "policy": policy, "obs_mode": policy, "fast_mode": a.fast_mode and not a.normal_core,
        "frames_per_action": a.fpa, "envs": a.envs, "total_steps": a.steps, "threads": a.threads,
        "proxy": proxy, "reward": reward.name, "reward_params": reward.params(),
        "reward_hash": E.reward_hash(reward), "hp": hp, "expect_levels": json.loads(a.expect_levels)
        if a.expect_levels else None, "final_eval_state": a.final_eval_state or state,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "pyboy": getattr(pyboy, "__version__", "2.9.0"), "machine": platform.machine()},
        "rom_sha1": rom.RED_SHA1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "resumes": 0,
    }


def train(cfg, run_dir, resume_from=None):
    import torch
    from pokered.policy import OBS_DIM, ActorCritic
    hp = cfg["hp"]
    torch.set_num_threads(cfg["threads"] or os.cpu_count())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = ActorCritic(cfg["policy"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], eps=1e-5)
    st = {"global_step": 0, "updates": 0, "marks_done": []}
    if resume_from:
        blob = torch.load(resume_from, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        st = blob["state"]
        torch.set_rng_state(blob["rng"]["torch"])
        np.random.set_state(blob["rng"]["numpy"])
        random.setstate(blob["rng"]["python"])
        print(f"resumed {cfg['run_id']} at step {st['global_step']} from {Path(resume_from).name}", flush=True)
    elif not (run_dir / "ckpt-0.pt").exists():
        save_ckpt(run_dir / "ckpt-0.pt", model, opt, st)            # the init policy (§9 A′)

    stop = {"flag": False}

    def on_signal(sig, _):
        print(f"signal {sig}: checkpointing and exiting", flush=True)
        stop["flag"] = True
        stop["reason"] = f"signal {sig}"
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    # tmux sends SIGHUP to the pane when its server dies (VM shutdown, spot preemption). Its default
    # action kills the process outright, skipping the finally: checkpoint, so treat it like SIGTERM.
    signal.signal(signal.SIGHUP, on_signal)

    n, T = cfg["envs"], hp["rollout"]
    D = OBS_DIM[cfg["policy"]]
    venv = VecEnv(cfg, n)
    obs = torch.from_numpy(venv.obs)
    buf_o = torch.zeros((T, n, D))
    buf_a = torch.zeros((T, n), dtype=torch.long)
    buf_lp, buf_r, buf_d, buf_v = (torch.zeros((T, n)) for _ in range(4))
    metrics = open(run_dir / "metrics.jsonl", "a")
    episodes = open(run_dir / "episodes.jsonl", "a")
    marks = [m for m in eval_marks(cfg["total_steps"], cfg.get("extra_eval_marks", ())) if m not in st["marks_done"]]
    st.setdefault("interim_done", [])
    interim = [m for m in sorted(cfg.get("interim_marks", ())) if m not in st["interim_done"] and m < cfg["total_steps"]]
    last_ckpt = time.time()
    done_t = torch.zeros(n)
    t_start, s_start = time.time(), st["global_step"]
    try:
        while st["global_step"] < cfg["total_steps"] and not stop["flag"]:
            ep_P, ep_T = [], []
            sum_dT = 0.0
            t_upd = time.time()
            for t in range(T):
                with torch.no_grad():
                    logits, v = model(obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                buf_o[t], buf_a[t], buf_lp[t], buf_v[t], buf_d[t] = obs, a, dist.log_prob(a), v, done_t
                o, r, d, eps = venv.step(a.numpy())
                buf_r[t] = torch.from_numpy(r)
                sum_dT += float(venv.last_dT.sum())
                obs, done_t = torch.from_numpy(o), torch.from_numpy(d.astype(np.float32))
                for ep in eps:
                    if ep:
                        ep.update(global_step=st["global_step"] + (t + 1) * n)
                        episodes.write(json.dumps(ep) + "\n")
                        ep_P.append(ep["P"])
                        ep_T.append(ep["T"])
            st["global_step"] += T * n
            t_collect = time.time() - t_upd
            # GAE
            with torch.no_grad():
                _, next_v = model(obs)
            adv = torch.zeros((T, n))
            last = torch.zeros(n)
            for t in reversed(range(T)):
                nonterm = 1.0 - (done_t if t == T - 1 else buf_d[t + 1])
                nv = next_v if t == T - 1 else buf_v[t + 1]
                delta = buf_r[t] + hp["gamma"] * nv * nonterm - buf_v[t]
                last = delta + hp["gamma"] * hp["gae_lambda"] * nonterm * last
                adv[t] = last
            ret = adv + buf_v
            # PPO update
            bo, ba, blp = buf_o.reshape(T * n, D), buf_a.reshape(-1), buf_lp.reshape(-1)
            badv, bret = adv.reshape(-1), ret.reshape(-1)
            idx = np.arange(T * n)
            stats = []
            for _ in range(hp["epochs"]):
                np.random.shuffle(idx)
                for s in range(0, T * n, hp["minibatch"]):
                    mb = torch.from_numpy(idx[s:s + hp["minibatch"]])
                    logits, v = model(bo[mb])
                    dist = torch.distributions.Categorical(logits=logits)
                    lp = dist.log_prob(ba[mb])
                    ratio = (lp - blp[mb]).exp()
                    a_mb = badv[mb]
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                    pg = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - hp["clip"], 1 + hp["clip"])).mean()
                    vl = 0.5 * (v - bret[mb]).pow(2).mean()
                    ent = dist.entropy().mean()
                    loss = pg + hp["vf_coef"] * vl - hp["ent_coef"] * ent
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hp["max_grad_norm"])
                    opt.step()
                    stats.append((pg.item(), vl.item(), ent.item(), ((ratio - 1) - ratio.log()).mean().item()))
            st["updates"] += 1
            t_total = time.time() - t_upd
            sps = (st["global_step"] - s_start) / (time.time() - t_start)
            row = {"step": st["global_step"], "update": st["updates"], "sps": round(sps, 1),
                   **perf_row(T * n, t_total, t_collect, buf_o),
                   "pg_loss": float(np.mean([x[0] for x in stats])), "v_loss": float(np.mean([x[1] for x in stats])),
                   "entropy": float(np.mean([x[2] for x in stats])), "approx_kl": float(np.mean([x[3] for x in stats])),
                   "mean_reward_per_step": float(buf_r.mean()), "episodes": len(ep_P),
                   # paid reward split (§4): ΔT (milestones) vs shaping (= reward − ΔT; A2p only), sums over the update
                   "reward_sum": float(buf_r.sum()), "reward_dT_sum": sum_dT,
                   "reward_shaping_sum": float(buf_r.sum()) - sum_dT,
                   "ep_P_mean": float(np.mean(ep_P)) if ep_P else None,
                   "ep_T_mean": float(np.mean(ep_T)) if ep_T else None, "time": time.time()}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            episodes.flush()
            if row["sps_update"] < SLOW_SPS:
                print(f"slow update {st['updates']}: {row['sps_update']} sps (collect {row['t_collect']} s, "
                      f"learn {row['t_learn']} s, load {row['load1']}, {row['cpu_mhz']} MHz, maps {row['maps']})",
                      flush=True)
            if st["updates"] % 10 == 0:
                print(f"step {st['global_step']:>9}  sps {sps:6.1f}  ent {row['entropy']:.3f}  "
                      f"P {row['ep_P_mean']}  T {row['ep_T_mean']}", flush=True)
            while marks and st["global_step"] >= marks[0]:
                st["marks_done"].append(marks.pop(0))
                save_ckpt(run_dir / f"eval-{st['global_step']}.pt", model, opt, st)
            while interim and st["global_step"] >= interim[0]:     # §9 H4′: descriptive interim-only checkpoints,
                st["interim_done"].append(interim.pop(0))           # named interim-*, so the closing eval skips them
                save_ckpt(run_dir / f"interim-{st['global_step']}.pt", model, opt, st)
            if time.time() - last_ckpt > cfg.get("ckpt_minutes", 15) * 60:
                save_ckpt(run_dir / f"ckpt-{st['global_step']}.pt", model, opt, st)
                prune(run_dir)
                last_ckpt = time.time()
    finally:
        save_ckpt(run_dir / f"ckpt-{st['global_step']}.pt", model, opt, st)
        if st["global_step"] < cfg["total_steps"]:     # stopped early: record exactly where (pokered.runlog)
            exc = sys.exc_info()[1]
            runlog.write_stop(run_dir, st["global_step"],
                              stop.get("reason") or (f"exception {type(exc).__name__}: {exc}" if exc else "stopped"))
        prune(run_dir)
        venv.close()
        metrics.close()
        episodes.close()
    finished = st["global_step"] >= cfg["total_steps"]
    if finished and cfg.get("final_eval", True):
        res = eval_episode(cfg, model, cfg["final_eval_state"], seed=10_000 + cfg["seed"], save_frame=True,
                           training_step=st["global_step"])
        (run_dir / "final_eval.json").write_text(json.dumps(res, indent=2))
        print(f"final eval on {cfg['final_eval_state'] or 'boot'}: T={res['T']} P={res['P']} -> {res['frame']}")
    return finished


def check_state_core(cfg):
    """§2 condition 3 / §5b Final: a start state must have been made on the core the run uses. The arms
    use the normal-core set in the states dir; the retired fast_mode set is archived elsewhere."""
    for name in {cfg["state"], cfg.get("final_eval_state")} - {None}:
        meta = rom.states_dir() / f"{name}.json"
        if not meta.exists():
            continue                                  # hand-made state without metadata: nothing to check
        made_fast = json.loads(meta.read_text()).get("fast_mode")
        if made_fast is not None and bool(made_fast) != bool(cfg["fast_mode"]):
            sys.exit(f"start state {name} was made with fast_mode={made_fast}, but this run uses "
                     f"fast_mode={cfg['fast_mode']}; mixing cores is not allowed (experiment-e1.md §5b)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", metavar="RUN_ID")
    ap.add_argument("--extend-to", type=int, metavar="STEPS",
                    help="with --resume: raise the run's total_steps to STEPS (the §9 10M extension); logged in "
                         "config.json 'extensions'; adds the eval marks past the old total")
    ap.add_argument("--arm", choices=["A0", "A1", "A2", "A2p", "A2pp"], default="A0")
    ap.add_argument("--round", type=int, default=0, help="patch round (A1)")
    ap.add_argument("--proxy", help="JSON kwargs for Proxy (patches), e.g. '{\"tile_visit_cap\": 600}'")
    ap.add_argument("--state", help="start state name in the states dir (default D1; none with --smoke)")
    ap.add_argument("--final-eval-state", help="state for the closing eval + screenshot (default: --state)")
    ap.add_argument("--expect-levels", help="JSON list, checked at reset (held-out B)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id")
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--envs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = all cores)")
    ap.add_argument("--policy", choices=["tilegrid", "tilegrid_visited", "screen", "ram"])
    ap.add_argument("--normal-core", action="store_true", help="(default since the §5b check failed; kept for old commands)")
    ap.add_argument("--fast-mode", action="store_true",
                    help="PyBoy fast_mode: benches/smoke only. §5b FAILED on D1 (Cable Club, RNG diverges): not for arms")
    ap.add_argument("--fpa", type=int, default=E.FRAMES_PER_ACTION, help="frames per action")
    ap.add_argument("--smoke", action="store_true",
                    help="no start-state sanity check, RAM-MLP by default; power-on start unless --state is given")
    ap.add_argument("--ckpt-minutes", type=float, default=15)
    ap.add_argument("--no-final-eval", action="store_true")
    ap.add_argument("--interim-marks", type=int, nargs="*", default=[],
                    help="extra checkpoints saved as interim-<step>.pt for interim reads only (§9 H4′; not eval marks)")
    ap.add_argument("--no-extra-marks", action="store_true", help="no extra eval marks (old 0.25/0.5/1/2/4M.. set)")
    ap.add_argument("--no-blackout-log", action="store_true",
                    help="omit blackout_steps from episodes.jsonl (logging only; for the identity smoke check)")
    for k in HPARAMS:
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(HPARAMS[k]), default=None)
    a = ap.parse_args()

    try:
        rom.rom_path(check=True)
        runs = rom.runs_dir()
    except rom.RomError as e:
        sys.exit(f"train: {e}")

    if a.resume:
        run_dir = runs / a.resume
        cfg = json.loads((run_dir / "config.json").read_text())
        ck = latest_ckpt(run_dir)
        n = len(cfg.get("resume_log", [])) + 1
        if not ck:
            # stopped before its first checkpoint: nothing to resume, so start the same run over
            print(f"no checkpoint in {run_dir}: restarting {a.resume} from step 0 (same config)", flush=True)
            entry = runlog.resume_entry(run_dir, runs, 0, None, n, kind="restart")
            for f in ("metrics.jsonl", "episodes.jsonl"):
                if (run_dir / f).exists():
                    (run_dir / f).rename(run_dir / f"{f}.before-restart-{n}")
            cfg["restarts"] = cfg.get("restarts", 0) + 1
            cfg.setdefault("resume_log", []).append(entry)
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
            print(f"resume_log: {json.dumps(entry)}", flush=True)
            train(cfg, run_dir)
            return
        if a.extend_to:
            old = cfg["total_steps"]
            if a.extend_to <= old:
                sys.exit(f"--extend-to {a.extend_to} is not above the run's total_steps {old}")
            sha = Path(__file__).resolve().parent / "CODE_SHA256"
            ext = {"from_steps": old, "to_steps": a.extend_to, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "code_sha256": sha.read_text().strip() if sha.exists() else None}
            if (run_dir / "final_eval.json").exists():   # the old total's closing eval, kept under its step
                (run_dir / "final_eval.json").rename(run_dir / f"final_eval-{old}.json")
                ext["kept_final_eval"] = f"final_eval-{old}.json"
            for suf in (".png", ".json"):                 # the §10.6 screenshot + sidecar of the old total
                f = run_dir / f"{a.resume}{suf}"
                if f.exists():
                    f.rename(run_dir / f"{a.resume}-{old}{suf}")
            cfg.setdefault("extensions", []).append(ext)
            cfg["total_steps"] = a.extend_to
            print(f"extension: total_steps {old:,} -> {a.extend_to:,}", flush=True)
        entry = runlog.resume_entry(run_dir, runs, int(Path(ck).stem.split("-")[1]), Path(ck).name, n)
        # §9 H9 standing ask: what this resume restores. The learner is exact; the emulator/env is not: every
        # in-progress episode restarts from its start state, and env workers are reseeded (+7919 per resume).
        entry["restored"] = {"weights": True, "optimizer": True, "step_counters": True,
                             "rng": ["torch", "numpy", "python"], "emulator_state": False,
                             "env_episodes": "restarted from the start state", "env_seed_offset": 7919 * (cfg.get("resumes", 0) + 1),
                             "full_restore": False, "ckpt_kind": "recovery (wall-clock, 15 min)" if Path(ck).name.startswith("ckpt-")
                             else "step-indexed eval/interim mark"}
        cfg.setdefault("resume_log", []).append(entry)
        # marker rows: analysis drops rows logged before this with a larger step (runlog.kept_rows)
        for f, key in (("metrics.jsonl", "step"), ("episodes.jsonl", "global_step")):
            with open(run_dir / f, "a") as fh:
                fh.write(json.dumps(runlog.resume_row(entry, key)) + "\n")
        print(f"resume_log: {json.dumps(entry)}", flush=True)
        cfg["resumes"] = cfg.get("resumes", 0) + 1
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        train(cfg, run_dir, resume_from=ck)
        return
    cfg = build_cfg(a)
    check_state_core(cfg)
    cfg["ckpt_minutes"] = a.ckpt_minutes
    cfg["final_eval"] = not a.no_final_eval
    cfg["log_blackout_steps"] = not a.no_blackout_log          # logging only; learner and RNG unchanged
    cfg["extra_eval_marks"] = [] if a.no_extra_marks else EXTRA_EVAL_MARKS   # checkpoint saves only; RNG unchanged
    cfg["interim_marks"] = list(a.interim_marks)               # checkpoint saves only; RNG unchanged (§9 H4′)
    from pokered.policy import OBS_VERSION
    cfg["obs_version"] = OBS_VERSION.get(cfg["policy"], 1)    # batch 2 = 2 (visited channel, §9 "Batch 2" H2)
    cfg["train_threads"] = a.threads or os.cpu_count()   # = train()'s torch.set_num_threads (init rebuild, §9 A′)
    # which code trained this run (gcp/config.sh sync_code writes CODE_SHA256 on the VM; absent on the laptop)
    sha = Path(__file__).resolve().parent / "CODE_SHA256"
    cfg["code_sha256"] = sha.read_text().strip() if sha.exists() else None
    cfg["code_note"] = ("2026-11-03: + episodes.jsonl blackout_steps (logging only; §9 Step 0 check (i)); "
                        "2026-11-04: + 3M eval mark (extra_eval_marks; §9 'A2″ 10M extension'); "
                        "2026-11-06: + ckpt-0.pt init policy saved at start, train_threads recorded (RNG unchanged; §9 A′); "
                        "2026-11-07: + obs_version, --policy tilegrid_visited (§9 'Batch 2'), episodes.jsonl P_terms "
                        "(logging only), --interim-marks (interim-*.pt; §9 H4′)")
    run_dir = runs / cfg["run_id"]
    if (run_dir / "config.json").exists():
        sys.exit(f"{run_dir} exists; use --resume {cfg['run_id']} or a new --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"run {cfg['run_id']} -> {run_dir}  policy {cfg['policy']}  fast_mode {cfg['fast_mode']}  "
          f"reward {cfg['reward']} {cfg['reward_hash']}", flush=True)
    train(cfg, run_dir)


if __name__ == "__main__":
    main()
