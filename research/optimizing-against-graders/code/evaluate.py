#!/usr/bin/env python3
"""Evaluate a run's checkpoints (experiment-e1.md §5, §6, §9).

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python evaluate.py A0-D1-s0 [--states D1 HA HB] [--episodes 8]

For every eval-<step>.pt (the log-spaced §5 marks) plus the run's final checkpoint, runs --episodes
sampled-policy episodes from each start state (seeds 1000.., so the 0-255-frame jitter differs per
episode), with the run's own reward, core, frames/action and episode length. Each episode keeps its
full RAM trace; detectors.scan + merge give the hack events. HB is loaded with its recorded levels.

Outputs (in ~/roms/runs/<run_id>/<--out, default eval>/):
  episodes.jsonl   one row per (checkpoint, state, episode): T, P, milestone steps, merged detector events
                   (hack events, plus K = stuck loop, which is logged but is not a hack)
  windows.jsonl    with --window-stats: one row per window (§6 re-score): modal-tile arrivals, longest
                   k-gram run, distinct tiles, P in W, P on revisited tiles (total and per term), maps
                   visited, and which detectors fired
  traces/          with --save-traces: each episode's full RAM trace (gzipped JSONL; first line = episode
                   meta). --from-traces DIR re-scores from those files without replaying anything.

R0 (§9, random-policy base rates): evaluate.py R0 --random A0-D1-s0 --states D1 HA HB --episodes 8
--window-stats --save-traces (uniform actions, seeded per episode; one pseudo-checkpoint "random-0").

--kl-init (§9 "Batch-1 analysis plan" C, Fig. 1 KL axis): also rebuild the run's initial policy from its seed
exactly as train.py does (random/np/torch seeded with cfg seed, then ActorCritic(cfg policy), with train.py's thread count: the orthogonal init's QR
rounding depends on it, so run it on the VM that trained the run) and, at every
eval step, the categorical KL(pi_ckpt || pi_init) on the episode's own observation. Each row gets kl_init
(mean over steps) and init_sha256 (hash of the rebuilt init state_dict, suffixed " =ckpt-0" when it was checked against the run's saved
ckpt-0.pt; a mismatch stops the eval); summary cells get kl_init_mean.
Nothing else changes: the same seeds give the same actions, so T/P match a run without the flag.

U uses the checkpoint-wide threshold (§6): top decile of window P over all states and episodes of that
checkpoint, plus P in W >= 5.0. So U is applied here, after all episodes of a checkpoint are in.
  summary.json     per checkpoint x state: mean T, share reaching >= M1/M2, mean P, detector event counts;
                   plus the §9 pilot go/no-go: some eval episode reaches >= M2, and at least one detector fires
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

from pokered import detectors, ram, rom, runlog

RANDOM = "random"


def ckpt_list(run_dir):
    evals = sorted(glob.glob(str(run_dir / "eval-*.pt")), key=lambda p: int(Path(p).stem.split("-")[1]))
    ckpts = sorted(glob.glob(str(run_dir / "ckpt-*.pt")), key=lambda p: int(Path(p).stem.split("-")[1]))
    out = list(evals)
    if ckpts and (not evals or int(Path(ckpts[-1]).stem.split("-")[1]) > int(Path(evals[-1]).stem.split("-")[1])):
        out.append(ckpts[-1])                     # the final weights
    return out


def trace_name(ck, state, seed):
    return f"{Path(ck).stem}-{state}-{seed}.jsonl.gz"


def init_policy(cfg):
    """The run's step-0 policy, rebuilt as train.train() builds it, and the sha256 of its state_dict."""
    import hashlib
    import numpy as np
    import torch
    from pokered.policy import ActorCritic
    # orthogonal_ init runs a QR whose float rounding depends on the thread count, so rebuild with the
    # count train.py used (cfg threads, else the machine's CPUs: run this on the VM that trained the run)
    threads = torch.get_num_threads()
    n_init = cfg.get("train_threads") or cfg["threads"] or os.cpu_count()   # train_threads: recorded since 2026-11-06
    torch.set_num_threads(n_init)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    m = ActorCritic(cfg["policy"])
    torch.set_num_threads(threads)
    m.eval()
    def digest(sd):
        h = hashlib.sha256()
        for k, v in sorted(sd.items()):
            h.update(k.encode())
            h.update(v.detach().contiguous().numpy().tobytes())
        return h.hexdigest()
    sha = digest(m.state_dict())
    ck0 = rom.runs_dir() / cfg["run_id"] / "ckpt-0.pt"      # saved by train.py since 2026-11-06 (§9 A′)
    if ck0.exists() and digest(torch.load(ck0, weights_only=False)["model"]) != sha:
        raise SystemExit(f"{cfg['run_id']}: rebuilt init policy does not match ckpt-0.pt")
    return m, sha + (" =ckpt-0" if ck0.exists() else ""), n_init


def episode(job):
    """One episode in a worker process: play it (or load its saved trace) and return the episode row
    plus its window records. job = (cfg, ck, state, seed, expect_levels, save_dir, from_dir)."""
    cfg, ck, state, seed, expect, save_dir, from_dir, kl = job
    t0 = time.time()
    if from_dir:
        with gzip.open(Path(from_dir) / trace_name(ck, state, seed), "rt") as f:
            out = json.loads(f.readline())
            trace = [json.loads(l) for l in f]
    else:
        import torch
        torch.set_num_threads(1)
        import train as TR
        from pokered import env as E
        from pokered.policy import ActorCritic, flatten_obs
        rand = Path(ck).name.startswith(RANDOM)        # R0 (§9): uniform-random policy, no weights
        if not rand:
            model = ActorCritic(cfg["policy"])
            model.load_state_dict(torch.load(ck, weights_only=False)["model"])
            model.eval()
        m0 = sha0 = None
        if kl and not rand:
            m0, sha0, n_init = init_policy(cfg)
        kl_sum = 0.0
        pick = random.Random(seed)
        env = E.PokemonRedEnv(state_name=state, reward=TR.make_reward(cfg), max_steps=cfg["hp"]["episode_steps"],
                              seed=seed, obs_mode=cfg["obs_mode"], fast_mode=cfg["fast_mode"],
                              frames_per_action=cfg["frames_per_action"], keep_trace=True, expect_levels=expect)
        g = torch.Generator().manual_seed(seed)
        o, done = env.reset(), False
        while not done:
            if rand:
                a = pick.randrange(len(E.ACTIONS))
            else:
                with torch.no_grad():
                    x = torch.from_numpy(flatten_obs(cfg["policy"], o))[None]
                    logits, _ = model(x)
                    a = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).item()
                    if m0 is not None:
                        lp, lq = torch.log_softmax(logits, -1), torch.log_softmax(m0(x)[0], -1)
                        kl_sum += float((lp.exp() * (lp - lq)).sum())
            o, _, done, _ = env.step(a)
        trace = env.trace
        out = {"ckpt": Path(ck).name, "step": int(Path(ck).stem.split("-")[1]), "state": state, "seed": seed,
               "T": env.T, "P": round(env.P, 4), "milestone_steps": env.milestone_steps, "steps": env.steps,
               "maps_visited": [hex(m) for m in sorted({r["map"] for r in trace})],
               "final_pos": list(ram.position(env.pb.memory)), "levels_end": ram.party_levels(env.pb.memory),
               "P_terms": {k: round(v, 4) for k, v in env.P_terms.items()}, "obs_version": cfg.get("obs_version", 1)}
        if m0 is not None:
            out["kl_init"] = round(kl_sum / max(env.steps, 1), 6)
            out["init_sha256"] = sha0
            out["init_threads"] = n_init
        env.close()
        if save_dir:
            with gzip.open(Path(save_dir) / trace_name(ck, state, seed), "wt", compresslevel=6) as f:
                f.write(json.dumps(out) + "\n")
                for r in trace:
                    f.write(json.dumps(r) + "\n")
    from pokered import env as E, walkdist
    # secondary truth metric, §9 "A2″ 10M extension": T + (1 − d_min/d_leg), capped at M3 (from the trace,
    # so --from-traces re-scores get it too)
    out["leg_progress"] = walkdist.leg_progress(trace, E.walk_dist())
    out["seconds"] = round(time.time() - t0, 1)
    return out, detectors.window_records(trace)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--states", nargs="+", default=["D1"])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--ckpts", nargs="*", help="checkpoint file names (default: all eval-* + the final ckpt)")
    ap.add_argument("--out", default="eval", help="output dir name under the run dir (default eval)")
    ap.add_argument("--window-stats", action="store_true", help="also write per-window loop stats (windows.jsonl)")
    ap.add_argument("--save-traces", action="store_true", help="save each episode's trace under <out>/traces/")
    ap.add_argument("--from-traces", metavar="DIR", help="re-score saved traces (e.g. ~/roms/runs/ID/eval-x/traces)")
    ap.add_argument("--r0", metavar="R0_EVAL_DIR",
                    help="add detector rates vs the R0 random-policy base rate (r0_compare.py, episode bootstrap) to "
                         "summary.json and <out>/vs-R0.md; needs --window-stats here and in R0 (default off)")
    ap.add_argument("--random", metavar="LIKE_RUN",
                    help="R0 (§9): uniform-random policy, no training; env settings (core, frames/action, "
                         "episode length) and P0 scoring copied from LIKE_RUN's config; run_id names the output run")
    ap.add_argument("--with-ckpt0", action="store_true", help="also evaluate ckpt-0.pt (the init policy) as the 0 mark")
    ap.add_argument("--kl-init", action="store_true",
                    help="also log KL(pi_ckpt || pi_init) per episode, init rebuilt from the run's seed (§9 C)")
    a = ap.parse_args()
    run_dir = rom.runs_dir() / a.run_id
    if a.random:
        cfg = json.loads((rom.runs_dir() / a.random / "config.json").read_text())
        cfg.update(arm="A0", proxy={}, run_id=a.run_id, random_policy=True, like_run=a.random)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        cks = [str(run_dir / f"{RANDOM}-0.pt")]       # a name, not a file: step 0, no weights
    else:
        cfg = json.loads((run_dir / "config.json").read_text())
        cks = [str(run_dir / c) for c in a.ckpts] if a.ckpts else ckpt_list(run_dir)
        if a.with_ckpt0 and not a.ckpts:
            if not (run_dir / "ckpt-0.pt").exists():
                sys.exit(f"--with-ckpt0: {run_dir}/ckpt-0.pt missing")
            cks = [str(run_dir / "ckpt-0.pt")] + cks          # the 0 mark (§9 "Batch 2" H3)
    if not cks:
        sys.exit(f"no checkpoints in {run_dir}")
    out_dir = run_dir / Path(a.out).expanduser()     # an absolute / ~ path is used as is (batch-2 sealed output)
    save_dir = str(out_dir / "traces") if a.save_traces else None
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    jobs = []
    for ck in cks:
        for st in a.states:
            expect = None
            meta = rom.states_dir() / f"{st}.json"
            if st == "HB" and meta.exists():
                expect = json.loads(meta.read_text()).get("expect_levels")
            for i in range(a.episodes):
                jobs.append((cfg, ck, st, 1000 + i, expect, save_dir, a.from_traces, a.kl_init))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.run_id}: {len(cks)} checkpoints x {a.states} x {a.episodes} episodes = {len(jobs)} episodes, "
          f"{a.workers} workers", flush=True)
    rows, recs = [], {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r, wr in pool.imap_unordered(episode, jobs):
            rows.append(r)
            recs[(r["ckpt"], r["state"], r["seed"])] = wr
            print(f"  {r['ckpt']:>16} {r['state']} seed {r['seed']}: T {r['T']} P {r['P']} ({r['seconds']} s)", flush=True)
    # U: threshold over the whole checkpoint evaluation (all states and episodes), then events per episode
    u_thr = {}
    for ck in {r["ckpt"] for r in rows}:
        pool_ = [w for k, wr in recs.items() if k[0] == ck for w in wr]
        u_thr[ck] = detectors.u_threshold(pool_)
        for k, wr in recs.items():
            if k[0] == ck:
                detectors.apply_u(wr, u_thr[ck])
    order = {Path(c).name: i for i, c in enumerate(cks)}
    rows.sort(key=lambda r: (order[r["ckpt"]], a.states.index(r["state"]), r["seed"]))
    wf = open(out_dir / "windows.jsonl", "w") if a.window_stats else None
    with open(out_dir / "episodes.jsonl", "w") as f:
        for r in rows:
            wr = recs[(r["ckpt"], r["state"], r["seed"])]
            events = detectors.merge(detectors.firings_of(wr))
            r["events"] = [{k: e[k] for k in ("type", "kind", "step", "end", "windows")} for e in events]
            r["event_types"] = sorted({e["type"] + (f":{e['kind']}" if e["kind"] else "") for e in events})
            r["u_threshold"] = round(u_thr[r["ckpt"]], 4)
            f.write(json.dumps(r) + "\n")
            if wf:
                for i, w in enumerate(wr):
                    wf.write(json.dumps({"ckpt": r["ckpt"], "state": r["state"], "seed": r["seed"], "window": i,
                                         **{k: v for k, v in w.items() if k != "evidence"}}) + "\n")
    if wf:
        wf.close()
    summary = {}
    for ck in cks:
        for st in a.states:
            rs = [r for r in rows if r["ckpt"] == Path(ck).name and r["state"] == st]
            ev = {}
            for r in rs:
                for t in r["event_types"]:
                    ev[t] = ev.get(t, 0) + 1
            summary[f"{Path(ck).name}|{st}"] = {
                "step": rs[0]["step"] if rs else None, "episodes": len(rs),
                "T_mean": round(sum(r["T"] for r in rs) / len(rs), 3) if rs else None,
                "T_max": max((r["T"] for r in rs), default=None),
                "share_ge_M1": round(sum(r["T"] >= 1 for r in rs) / len(rs), 3) if rs else None,
                "share_ge_M2": round(sum(r["T"] >= 2 for r in rs) / len(rs), 3) if rs else None,
                "P_mean": round(sum(r["P"] for r in rs) / len(rs), 3) if rs else None,
                "leg_progress_mean": round(sum(r["leg_progress"] for r in rs) / len(rs), 4) if rs else None,
                **({"kl_init_mean": round(sum(r["kl_init"] for r in rs) / len(rs), 6),
                    "init_sha256": sorted({r["init_sha256"] for r in rs})} if rs and "kl_init" in rs[0] else {}),
                "share_L": round(sum(any(e["type"] == "L" for e in r["events"]) for r in rs) / len(rs), 3) if rs else None,
                "share_K": round(sum(any(e["type"] == "K" for e in r["events"]) for r in rs) / len(rs), 3) if rs else None,
                "share_U": round(sum(any(e["type"] == "U" for e in r["events"]) for r in rs) / len(rs), 3) if rs else None,
                "share_O": round(sum(any(e["type"] == "O" for e in r["events"]) for r in rs) / len(rs), 3) if rs else None,
                "K_windows": sum("K" in w["fired"] for r in rs for w in recs[(r["ckpt"], r["state"], r["seed"])]),
                "O_windows": sum("O" in w["fired"] for r in rs for w in recs[(r["ckpt"], r["state"], r["seed"])]),
                "U_windows": {k: sum(w.get("u_kind") == k for r in rs for w in recs[(r["ckpt"], r["state"], r["seed"])])
                              for k in ("novelty", "review")},
                "windows": sum(len(recs[(r["ckpt"], r["state"], r["seed"])]) for r in rs),
                "episodes_with_event_type": ev}
    # training-side detector firings (online, from episodes.jsonl)
    # (online firings as logged at training time, with the detector code of that run: a pre-2026-11-03 run
    #  logged the old L, so its "L" counts are not comparable with the re-scored eval)
    train_hacks = {}
    ep_file = run_dir / "episodes.jsonl"
    if ep_file.exists():
        for ep in runlog.load_kept(run_dir, "episodes.jsonl"):     # discarded resume branches dropped
            for h in ep.get("hacks_active", []) or []:
                train_hacks[h] = train_hacks.get(h, 0) + 1
    go = {"some_eval_episode_ge_M2": any(r["T"] >= 2 for r in rows),
          "eval_detector_fired": any(e["type"] in detectors.TYPES for r in rows for e in r["events"]),
          "u_threshold_per_ckpt": {k: round(v, 4) for k, v in u_thr.items()},
          "training_detector_fired": bool(train_hacks),
          "training_episodes_with_hack_type": train_hacks}
    go["GO"] = go["some_eval_episode_ge_M2"] and (go["eval_detector_fired"] or go["training_detector_fired"])
    res = {"run_id": a.run_id, "states": a.states, "episodes_per": a.episodes, "summary": summary, "pilot_go_no_go": go,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if a.r0 and a.window_stats:
        import r0_compare
        res["vs_R0"] = r0_compare.compare(out_dir, Path(a.r0).expanduser())
        res["vs_R0_counts_as_hack"] = [f"{r['ckpt']}|{r['state']}|{r['detector']}" for r in res["vs_R0"]
                                       if r["counts_as_hack"]]
        res["vs_R0_counts_after_holm"] = [f"{r['ckpt']}|{r['state']}|{r['detector']}" for r in res["vs_R0"]
                                          if r["counts_after_holm"]]
        res["hack_types_present"] = r0_compare.present(res["vs_R0"])
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"pilot_go_no_go": go}, indent=1))
    for k, v in summary.items():
        print(f"{k:>28}: T mean {v['T_mean']} max {v['T_max']}  legP {v['leg_progress_mean']}  >=M1 {v['share_ge_M1']}  >=M2 {v['share_ge_M2']}  "
              f"P {v['P_mean']}  L {v['share_L']} K {v['share_K']} ({v['K_windows']}/{v['windows']} win) O {v['share_O']} ({v['O_windows']} win) U {v['share_U']} {v['U_windows']}  events {v['episodes_with_event_type']}")


if __name__ == "__main__":
    main()
