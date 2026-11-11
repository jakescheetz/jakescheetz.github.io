#!/usr/bin/env python3
"""Batch-2 interim reads, tripwires and per-run summaries (experiment-e1.md §9 "Batch 2" H4′). Dev states only.

  .venv/bin/python b2_reads.py summary RUN...     per-mark summary -> <run>/summary-marks.{json,md}
  .venv/bin/python b2_reads.py tripwire RUN       H4′ tripwires -> <run>/tripwire.json; exit 1 if any trips
  .venv/bin/python b2_reads.py brief RUN...       one science line per run (for health checks)
  .venv/bin/python b2_reads.py resumes RUN...     resumed runs + §9 H9 keep/escalate status (batch-2 report)
  .venv/bin/python b2_reads.py watch RUN --reads all|0 1M 2.5M|none [--workers 2]
                                                  (VM, started by queue.sh next to train.py) at every new
                                                  checkpoint: interim eval if it's a read, summary, tripwire;
                                                  a trip pauses both queues (queue.sh pause)

What a summary holds, per mark (eval-<step>.pt, interim-<step>.pt, ckpt-0.pt), on the run's own state:
  * eval (when an interim eval or the dev view has it): n, T mean, leg progress, P and P by term (explore /
    level / heal), P ÷ R0 (batch-1 R0 dev P), g = (P − P(0)) / (417 − P(0)) against the G5 planner, episodes
    with an L event (canary), and for A2″ the collapse flag (§9 (1): mean T ≥ 1 below its best earlier mark);
  * train: the training episodes since the previous mark (P, P by term, T), but only if the training state is
    a dev state (D1/D2): an HA-trained run's training episodes are held-out data and are never summarised;
  * infra: obs_version, reward hash, init hash checked against ckpt-0, median sps of the last 100 updates and
    its ratio to batch 1's measured rate on that VM and arm.
Never in a summary: detector shares, tests against R0, A0-vs-A2″ comparisons, held-out states.

Tripwires (H4′ table, thresholds fixed there; tripwire.json lists each with pass / trip / pending):
  A-1 (A0, 1M)   P < 2×R0 (D1 59.4, D2 62.3), or P(1M) not > P(0) (one-sided MC permutation, p > 0.05)
  A-2 (A0, 2.5M) P < 0.75 × the manipulation target (D1 83.3, D2 75.8), or P(2.5M) < P(1M) with p ≤ 0.05
  B-1 (A2″, 1M)  leg progress(1M) not > leg progress(0) (p > 0.05)
  B-2 (A2″, 2.5M) no episode reaches M1 (T ≥ 1 in 0/8), or leg progress(2.5M) < leg progress(1M), p ≤ 0.05
  C (any, every read) L > 0; obs_version or reward hash mismatch; init hash not checked against ckpt-0;
                 sps < 85% of batch 1's on that VM and arm (A1: of batch 2's A0 D1 rate on that VM, §9 I12)
A1 round 1 (§9 I12; all three runs read at 0, 1M, 2.5M; statistic P1 = the eval episode re-scored under P1):
  S-1 (1M)       P1(1M) not > P1(0) (p > 0.05), or P1(1M) not > R0's P1 on D1 (p > 0.05)            pauses
  S-2 (1M, 2.5M) never leaves the start map in 8 of 8 (1M and 2.5M), or P1(2.5M) < P1(1M) with p ≤ 0.05  pauses
  S-3 (1M, 2.5M) T not > R0's T (R0's T is 0 in all 24, so the exact p = C(8,k)/C(32,k), k = episodes with
                 T > 0), or 0 of 8 reach M1                                       NOTIFIES the lead, never pauses
  Logged, never a tripwire: T, M1 count, P0 (re-scored) vs R0 29.7 and vs batch-2 A0 D1 at the nearest mark,
  P1 by term, maps visited, never-left-start count, training-episode P1 over the last 250k steps.
Interim reads freeze the trainer (SIGSTOP/SIGCONT of train.py and its env workers) and use all CPUs: an eval
beside training collapses the learner's OpenMP threads. The freeze intervals go to eval-interim/stops.json, and
the sps check leaves out updates that overlap them (pending until 20 clean updates exist).
A trip pauses both queues (no new runs start; the running run finishes) and nothing else: no seed, run or
checkpoint is ever dropped (H4′ response rules). The 1M read is the eval-* checkpoint nearest 1,000,448;
the 2.5M read is the checkpoint in [2.4M, 2.6M] (interim-* for main runs, the final eval-* for pilots).
"""
import argparse
import glob
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import rom  # noqa: E402

HERE = Path(__file__).resolve().parent
DEV = ("D1", "D2")
R0_P = {"D1": 29.7, "D2": 31.2}                 # batch-1 R0 dev P (H4′)
PLANNER_P = 417.0                               # G5 planner floor (H4′)
A1_MIN = {"D1": 59.4, "D2": 62.3}               # 2 × R0 (H4′ table)
A2_MIN = {"D1": 83.3, "D2": 75.8}               # 0.75 × 2 × batch-1 A0 best mark (H4′ table)
REWARD_HASH = {"A0": "d28e2a337dda", "A2pp": "1f739117e21a", "A1": "4aa81ba4ad1f"}   # A1 = P1, C = 20 (I11/I12)
OBS_VERSION = 2
SPS_REF = {("e1-a", "A0"): 411, ("e1-a", "A2pp"): 381, ("e1-b", "A0"): 353, ("e1-b", "A2pp"): 340,   # batch 1, median
           ("e1-a", "A1"): 389.2, ("e1-b", "A1"): 346.6}   # §9 I12: batch-2 A0 D1 median (e1-a s1-s3, e1-b s4-s5)
# §9 I12 references, fixed before round 1 was armed (BOARD 2026-11-10; code/results/b2/rescore/r0-d1-p1.json)
R0_P1_D1 = [3.3, 8.0, 9.5, 6.4, 6.8, 6.9, 8.3, 5.1, 7.7, 5.6, 4.0, 8.4, 6.3, 8.3, 10.434, 6.0, 8.0, 12.0, 8.0, 8.2,
            6.0, 10.0, 9.6, 8.7]                      # R0 eval-dev-24 D1, re-scored under P1 (mean 7.564)
R0_T_D1 = [0] * 24                                    # R0's T on D1: 0 in all 24 episodes
A0B2_D1_P = {250_880: 39.74, 500_736: 64.62, 1_000_448: 73.8, 2_000_896: 105.93, 3_000_320: 70.61,
             4_000_768: 120.86, 5_000_192: 112.64}    # batch-2 A0 D1 own-state dev P0 per mark (5 seeds × 8)
TRAIN_WIN = 250_000
SPS_MIN_FRAC = 0.85
P_ALPHA = 0.05
M1, M25 = 1_000_448, (2_400_000, 2_600_000)


def runs_dir():
    return rom.runs_dir()


def jl(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f]


def step_of(name):
    return int(Path(name).stem.split("-")[1])


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def own_eval_rows(d, cfg):
    """{step: [rows]} on the run's own state from interim evals and, once sealed, the dev view. Dev only."""
    st = cfg["state"]
    if st not in DEV:
        return {}
    out = {}
    # batch 2's closing eval/ holds D1 D2 only (HA HB go to ~/roms/sealed-b2/, §9 H8); rows are filtered to the
    # run's own (dev) state either way
    srcs = sorted(glob.glob(str(d / "eval-interim" / "*" / "episodes.jsonl"))) + [str(d / "eval-dev" / "episodes.jsonl")]
    if cfg.get("obs_version") == OBS_VERSION:
        srcs.append(str(d / "eval" / "episodes.jsonl"))
    for f in srcs:
        for r in jl(f):
            if r["state"] == st:
                out.setdefault(r["step"], {})[(r["ckpt"], r["seed"])] = r     # dev view wins over interim
    return {k: list(v.values()) for k, v in sorted(out.items())}


def stops(d):
    """Intervals [t0, t1] (epoch s) when interim reads froze or shared the CPUs with this run's trainer."""
    p = d / "eval-interim" / "stops.json"
    return json.loads(p.read_text()) if p.exists() else []


def sps_now(d, min_clean=20):
    """Median sps_update of the last 100 updates, leaving out the first 10 and any update that overlaps an
    interim-read interval (+60 s); None if fewer than min_clean clean updates (C's sps part is then pending)."""
    iv = stops(d)
    rows = [r for r in jl(d / "metrics.jsonl") if r.get("sps_update")][10:]

    def clean(r):
        t1 = r.get("time")
        if t1 is None:
            return True
        t0 = t1 - (r.get("t_collect") or 0) - (r.get("t_learn") or 0) - 5
        return not any(t0 <= b + 60 and t1 >= a for a, b, *_ in iv)
    tail = [r["sps_update"] for r in rows if clean(r)][-100:]
    return round(statistics.median(tail), 1) if len(tail) >= min_clean else None


RESUME_MAX, REPLAY_MAX = 2, 100_000          # §9 H9 pre-ruling


def resume_rule(cfg, trained_to):
    """§9 H9 pre-ruling for partial-restore resumes: keep the run iff (i) every resume was triggered by
    infrastructure (stop.json reason 'signal …' from a shutdown/preemption, or a recorded GCP VM event; an
    exception or an unexplained stop is not), (ii) the learner restore is exact (a resume from a checkpoint,
    not a restart from step 0), (iii) the run reaches M, (iv) <= 2 resumes and <= 100k replayed/lost steps in
    total. Otherwise 'escalate': it goes to the researcher before its closing eval is read."""
    log = cfg.get("resume_log", [])
    if not log:
        return {"status": "none", "resumes": 0}
    why, lost = [], 0
    for e in log:
        stop, ev = e.get("stop") or {}, e.get("vm_event")
        infra = str(stop.get("reason", "")).startswith("signal") or bool(ev)
        if not infra:
            why.append(f"#{e.get('resume')}: trigger not infrastructure ({stop.get('reason') or 'no stop.json, no VM event'})")
        if e.get("kind") != "resume" or not e.get("resumed_from_ckpt"):
            why.append(f"#{e.get('resume')}: learner not restored from a checkpoint ({e.get('kind')})")
        if e.get("replayed_steps") is None:
            why.append(f"#{e.get('resume')}: replayed/lost steps unknown")
        else:
            lost += max(0, e["replayed_steps"])
    if len(log) > RESUME_MAX:
        why.append(f"{len(log)} resumes > {RESUME_MAX}")
    if lost > REPLAY_MAX:
        why.append(f"{lost:,} replayed/lost steps > {REPLAY_MAX:,}")
    reached = trained_to >= cfg["total_steps"]
    status = "escalate" if why else ("keep" if reached else "pending (run must reach M)")
    return {"status": status, "resumes": len(log), "replayed_or_lost": lost, "reached_M": reached, "why": why}


def trained_to(d):
    """The trainer's step: the last metrics.jsonl row (an episode row only appears when an episode ends)."""
    return max([r["step"] for r in jl(d / "metrics.jsonl") if "step" in r and "update" in r], default=0)


def c_check(cfg, ev, sps_recent, sps_ratio, sps_ref):
    """H4′ C at every read, interim and closing: L > 0; obs_version or reward hash mismatch; init hash not checked
    against ckpt-0; sps < 85% of batch 1's on that VM and arm. Over all own-state eval rows the run has so far."""
    arm = cfg["arm"]
    c = []
    reads = [r for rows in ev.values() for r in rows]
    if any(any(x["type"] == "L" for x in r.get("events", [])) for r in reads):
        c.append("L > 0")
    if cfg.get("obs_version") != OBS_VERSION or any(r.get("obs_version") != OBS_VERSION for r in reads):
        c.append(f"obs_version {cfg.get('obs_version')} / rows {sorted({r.get('obs_version') for r in reads})}")
    if cfg.get("reward_hash") != REWARD_HASH.get(arm):
        c.append(f"reward hash {cfg.get('reward_hash')} != {REWARD_HASH.get(arm)}")
    if any(not str(r.get("init_sha256", "")).endswith("=ckpt-0") for r in reads):
        c.append("init hash not checked against ckpt-0 in some row")
    if sps_ratio is not None and sps_ratio < SPS_MIN_FRAC:
        c.append(f"sps {sps_recent} < {SPS_MIN_FRAC:.0%} of {sps_ref}")
    sps_note = "" if sps_recent is not None else "; sps pending (< 20 clean updates outside interim reads)"
    return {"status": "trip" if c else ("pass" if reads else "pending"), "reasons": c,
            "detail": ("; ".join(c) or f"{len(reads)} eval rows ok, sps ratio {sps_ratio}") + sps_note}


def p_vs_zero(x, ref):
    """One-sided p for mean(x) > mean(ref). When ref is all 0 (R0's T, §9 I12) the exact permutation p is
    C(n, k) / C(n + m, k) with k = values of x above 0 (every relabeling at least as extreme keeps all k in x);
    otherwise the batch's Monte Carlo permutation (r0_compare.perm_p)."""
    from math import comb
    if all(v == 0 for v in ref):
        k = sum(v > 0 for v in x)
        return comb(len(x), k) / comb(len(x) + len(ref), k), "exact"
    import r0_compare as R
    p = R.perm_p(list(x), list(ref))
    return p[0], p[1]


def a1_rescore(d, cfg, rows):
    """{(ckpt, seed): (P1, P0)} for A1 eval rows, from their saved traces: P1 under the run's own proxy (must equal
    the eval's P), P0 under A0's knobs (map cap removed). Needs $POKEMON_RED_ROM (episode start RAM)."""
    import gzip
    out, RS = {}, None
    for r in rows:
        name = f"{Path(r['ckpt']).stem}-{r['state']}-{r['seed']}.jsonl.gz"
        tf = next((t for t in [*(d / "eval-interim").glob(f"*/traces/{name}"), d / "eval" / "traces" / name]
                   if t.exists()), None)
        if tf is None:
            continue
        if RS is None:
            import rescore as RS
            own = RS.proxy_kwargs(cfg["reward_params"])
            p0 = {k: v for k, v in own.items() if k != "map_new_tile_cap"}
        with gzip.open(tf, "rt") as f:
            f.readline()
            tr = [json.loads(l) for l in f]
        out[(r["ckpt"], r["seed"])] = (sum(RS.score(tr, cfg, r["state"], own)[0]),
                                       sum(RS.score(tr, cfg, r["state"], p0)[0]))
    return out


def a1_eval(d, cfg, m, e):
    """§9 I12 logged fields for an A1 read (never a tripwire by themselves)."""
    rs = a1_rescore(d, cfg, e)
    P1 = [r["P"] for r in e]
    out = {"P1": mean(P1), "P1_x_R0": round(mean(P1) / mean(R0_P1_D1), 2),
           "never_left": sum(len(r.get("maps_visited") or []) == 1 for r in e),
           "maps_visited": sorted({mp for r in e for mp in r.get("maps_visited") or []}),
           "rescored": len(rs)}
    if rs:
        out["P1_rescore_max_diff"] = round(max(abs(rs[(r["ckpt"], r["seed"])][0] - r["P"])
                                                for r in e if (r["ckpt"], r["seed"]) in rs), 9)
        P0 = mean(v[1] for v in rs.values())
        near = min(A0B2_D1_P, key=lambda k: abs(k - m))
        out.update(P0=P0, P0_x_R0=round(P0 / R0_P["D1"], 2), P0_vs_A0b2={"mark": near, "A0_P0": A0B2_D1_P[near],
                                                                         "ratio": round(P0 / A0B2_D1_P[near], 3)})
    return out


def summary(run, vm=None):
    """vm: the VM the run trained on (default: the config's, else this host; pass it when reading off-VM)."""
    d = runs_dir() / run
    cfg = json.loads((d / "config.json").read_text())
    arm, st = cfg["arm"], cfg["state"]
    vm = vm or cfg.get("vm") or socket.gethostname()
    ev = own_eval_rows(d, cfg)
    marks = sorted(set([0] * (d / "ckpt-0.pt").exists() + [step_of(p) for p in glob.glob(str(d / "eval-*.pt"))]
                       + [step_of(p) for p in glob.glob(str(d / "interim-*.pt"))] + list(ev)))
    train = [r for r in jl(d / "episodes.jsonl")] if st in DEV else []
    P0 = mean(r["P"] for r in ev.get(0, []))
    rows, bestT = [], None
    prev = 0
    for m in marks:
        e = ev.get(m, [])
        rec = {"step": m}
        if e:
            P = mean(r["P"] for r in e)
            terms = {k: mean(r["P_terms"][k] for r in e) for k in ("explore", "level", "heal")} if all("P_terms" in r for r in e) else None
            Tm = mean(r["T"] for r in e)
            rec["eval"] = {"n": len(e), "T": Tm, "T_ge1": sum(r["T"] >= 1 for r in e), "leg_progress": mean(r["leg_progress"] for r in e),
                           "P": P, "P_terms": terms, "x_R0": round(P / R0_P[st], 2),
                           "g": round((P - P0) / (PLANNER_P - P0), 4) if P0 is not None and m > 0 else None,
                           "L_episodes": sum(any(x["type"] == "L" for x in r.get("events", [])) for r in e),
                           "obs_version": sorted({r.get("obs_version", 1) for r in e}),
                           "init_sha256": sorted({r.get("init_sha256", "") for r in e})}
            if arm == "A1":
                rec["eval"]["x_R0"], rec["eval"]["g"] = None, None    # P0 references don't apply to P1
                rec["eval"]["a1"] = a1_eval(d, cfg, m, e)
            if arm == "A2pp" and m > 0:
                rec["eval"]["collapsed"] = bestT is not None and Tm <= bestT - 1
                bestT = Tm if bestT is None else max(bestT, Tm)
            elif arm == "A2pp":
                bestT = Tm
        tr = [r for r in train if prev < r["global_step"] <= m] if m > 0 else []
        if tr:
            rec["train"] = {"episodes": len(tr), "P": mean(r["P"] for r in tr), "T": mean(r["T"] for r in tr),
                            "P_terms": {k: mean(r["P_terms"][k] for r in tr) for k in ("explore", "level", "heal")}
                            if all("P_terms" in r for r in tr) else None}
        if arm == "A1" and m > 0:
            w = [r for r in train if m - TRAIN_WIN < r["global_step"] <= m]
            rec["train_P1_last_250k"] = {"episodes": len(w), "P1": mean(r["P"] for r in w)} if w else None
        if m > 0:
            prev = m
        rows.append(rec)
    sps = sps_now(d)
    ref = SPS_REF.get((vm, arm))
    out = {"run": run, "arm": arm, "state": st, "vm": vm, "label": "interim, validity only, not a result",
           "obs_version": cfg.get("obs_version", 1), "reward_hash": cfg.get("reward_hash"),
           "total_steps": cfg["total_steps"], "trained_to": trained_to(d),
           "resumes": [{k: e.get(k) for k in ("resume", "kind", "resumed_from_ckpt", "resumed_from_step", "stopped_at_step",
                                               "replayed_steps", "restored")} for e in cfg.get("resume_log", [])],
           "P0": P0, "sps_recent": sps,
           "resume_rule": resume_rule(cfg, trained_to(d)), "sps_ref": ref, "sps_ratio": round(sps / ref, 3) if sps and ref else None,
           "marks": rows, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out["C"] = c_check(cfg, ev, sps, out["sps_ratio"], ref)
    (d / "summary-marks.json").write_text(json.dumps(out, indent=1))
    (d / "summary-marks.md").write_text(md(out))
    return out


def md(s):
    L = [f"# {s['run']} ({s['arm']}, {s['state']}, {s['vm']}): interim, validity only, not a result", "",
         f"obs_version {s['obs_version']}, reward {s['reward_hash']}, trained to {s['trained_to']:,} / {s['total_steps']:,}, "
         f"sps {s['sps_recent']} ({s['sps_ratio']}× batch 1's {s['sps_ref']})",
         f"resumes: {len(s['resumes'])} (§9 H9 rule: {s['resume_rule']['status']})" + "".join(f"; #{r['resume']} from {r['resumed_from_ckpt']} @ {r['resumed_from_step']} "
                                                  f"(stopped {r['stopped_at_step']}, replayed {r['replayed_steps']}; learner + RNG "
                                                  f"restored, emulator/env NOT: partial)" for r in s["resumes"]),
         f"C (H4′, every read incl. closing): {s['C']['status'].upper()}: {s['C']['detail']}", "",
         "| mark | eval n | T | legP | P | explore / level / heal | ×R0 | g | L eps | train P | train T |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["marks"]:
        e, t = r.get("eval", {}), r.get("train", {})
        pt = e.get("P_terms") or {}
        L.append(f"| {r['step']:,} | {e.get('n', '')} | {e.get('T', '')} | {e.get('leg_progress', '')} | {e.get('P', '')} | "
                 f"{pt.get('explore', '')} / {pt.get('level', '')} / {pt.get('heal', '')} | {e.get('x_R0', '')} | {e.get('g', '')} | "
                 f"{e.get('L_episodes', '')}{' collapsed' if e.get('collapsed') else ''} | {t.get('P', '')} | {t.get('T', '')} |")
    return "\n".join(L) + "\n"


def pick(ev, lo, hi):
    ks = [k for k in ev if lo <= k <= hi]
    return ev[min(ks, key=lambda k: abs(k - (lo + hi) / 2))] if ks else None


def tripwire(run):
    import r0_compare as R
    d = runs_dir() / run
    cfg = json.loads((d / "config.json").read_text())
    s = summary(run)
    arm, st = cfg["arm"], cfg["state"]
    ev = own_eval_rows(d, cfg)
    e0, e1m, e25 = ev.get(0), pick(ev, 900_000, 1_100_000), pick(ev, *M25)
    res = []

    def add(i, status, detail):
        res.append({"id": i, "status": status, "detail": detail})
    P = lambda rows: [r["P"] for r in rows]
    LP = lambda rows: [r["leg_progress"] for r in rows]
    if arm == "A0":
        if e1m and e0:
            p = R.perm_p(P(e1m), P(e0))[0]
            mP = mean(P(e1m))
            trip = mP < A1_MIN[st] or p > P_ALPHA
            add("A-1", "trip" if trip else "pass", f"P(1M) {mP} vs min {A1_MIN[st]}; P(1M) > P(0) {mean(P(e0))}: p {p:.4f}")
        else:
            add("A-1", "pending", "needs the 0 and 1M reads")
        if e25 and e1m:
            p = R.perm_p(P(e1m), P(e25))[0]
            mP = mean(P(e25))
            trip = mP < A2_MIN[st] or p <= P_ALPHA
            add("A-2", "trip" if trip else "pass", f"P(2.5M) {mP} vs min {A2_MIN[st]}; P(2.5M) < P(1M) {mean(P(e1m))}: p {p:.4f}")
        else:
            add("A-2", "pending", "needs the 1M and 2.5M reads")
    if arm == "A2pp":
        if e1m and e0:
            p = R.perm_p(LP(e1m), LP(e0))[0]
            add("B-1", "trip" if p > P_ALPHA else "pass", f"legP(1M) {mean(LP(e1m))} > legP(0) {mean(LP(e0))}: p {p:.4f}")
        else:
            add("B-1", "pending", "needs the 0 and 1M reads")
        if e25 and e1m:
            p = R.perm_p(LP(e1m), LP(e25))[0]
            n1 = sum(r["T"] >= 1 for r in e25)
            trip = n1 == 0 or p <= P_ALPHA
            add("B-2", "trip" if trip else "pass", f"T>=1 in {n1}/{len(e25)} at 2.5M; legP(2.5M) {mean(LP(e25))} < legP(1M) {mean(LP(e1m))}: p {p:.4f}")
        else:
            add("B-2", "pending", "needs the 1M and 2.5M reads")
    if arm == "A1":                                    # §9 I12
        P1 = P
        n_left = lambda rows: sum(len(r.get("maps_visited") or []) == 1 for r in rows)
        if e1m and e0:
            p_own = R.perm_p(P1(e1m), P1(e0))[0]
            p_r0 = R.perm_p(P1(e1m), R0_P1_D1)[0]
            add("S-1", "trip" if p_own > P_ALPHA or p_r0 > P_ALPHA else "pass",
                f"P1(1M) {mean(P1(e1m))} > P1(0) {mean(P1(e0))}: p {p_own:.4f}; > R0 P1 {mean(R0_P1_D1)}: p {p_r0:.4f}")
        else:
            add("S-1", "pending", "needs the 0 and 1M reads")
        why, have = [], False
        for lab, rows in (("1M", e1m), ("2.5M", e25)):
            if rows:
                have = True
                if n_left(rows) == len(rows):
                    why.append(f"never left the start map in {len(rows)}/{len(rows)} at {lab}")
        if e25 and e1m:
            p = R.perm_p(P1(e1m), P1(e25))[0]
            if p <= P_ALPHA:
                why.append(f"P1(2.5M) {mean(P1(e25))} < P1(1M) {mean(P1(e1m))}: p {p:.4f}")
        det = "; ".join(why) or ", ".join(f"never-left {n_left(r)}/{len(r)} at {l}" for l, r in (("1M", e1m), ("2.5M", e25)) if r)
        add("S-2", "trip" if why else ("pass" if have else "pending"), det or "needs the 1M read")
        for lab, rows in (("1M", e1m), ("2.5M", e25)):
            if not rows:
                add(f"S-3@{lab}", "pending", f"needs the {lab} read")
                continue
            T = [r["T"] for r in rows]
            p, how = p_vs_zero(T, R0_T_D1)
            k1 = sum(t >= 1 for t in T)
            bad = p > P_ALPHA or k1 == 0
            add(f"S-3@{lab}", "notify" if bad else "pass",
                f"T {mean(T)} > R0 T 0: {how} p {p:.4f} ({sum(t > 0 for t in T)}/{len(T)} with T > 0); M1 in {k1}/{len(T)}")
    # C: infrastructure / canary, at every read (the same check the summary prints at closing reads)
    add("C", s["C"]["status"], s["C"]["detail"])
    out = {"run": run, "arm": arm, "state": st, "checked": time.strftime("%Y-%m-%dT%H:%M:%S"), "tripwires": res,
           "tripped": [r["id"] for r in res if r["status"] == "trip"],
           "notify": [r["id"] for r in res if r["status"] == "notify"]}     # §9 I12 S-3: tell the lead, never pause
    (d / "tripwire.json").write_text(json.dumps(out, indent=1))
    return out


def brief(run):
    s = summary(run)
    ev = [r for r in s["marks"] if "eval" in r]
    tr = [r for r in s["marks"] if "train" in r]
    tw = jl_one(runs_dir() / run / "tripwire.json")
    head = f"{run} @{s['trained_to'] / 1e6:.2f}M"
    if ev:
        e = ev[-1]["eval"]
        pt = e.get("P_terms") or {}
        sci = (f"eval@{ev[-1]['step'] / 1e6:.2f}M P {e['P']} (expl {pt.get('explore')}) = {e['x_R0']}×R0, g {e['g']}; "
               f"T {e['T']}, legP {e['leg_progress']}; L {e['L_episodes']}")
    elif tr:
        t = tr[-1]["train"]
        sci = f"train P {t['P']} (= {round(t['P'] / R0_P[s['state']], 2)}×R0), T {t['T']}"
    else:
        sci = "no dev data yet"
    trip = f"; C {s['C']['status']}" + (f" ({'; '.join(s['C']['reasons'])})" if s["C"]["reasons"] else "")
    trip += f"; tripwires {','.join(tw['tripped']) or 'clear'}" if tw else ""
    if s["resumes"]:
        r = s["resumes"][-1]
        rr = s["resume_rule"]
        trip += (f"; RESUMED x{len(s['resumes'])} (last from {r['resumed_from_ckpt']} @ {r['resumed_from_step']}, "
                 f"partial restore; H9 rule: {rr['status']}{' — ' + '; '.join(rr['why']) if rr.get('why') else ''})")
    return f"{head}: {sci}; sps {s['sps_recent']} ({s['sps_ratio']}×){trip} [interim, validity only]"


def jl_one(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def pause(reason):
    cmd = os.environ.get("E1_PAUSE_CMD") or str(Path.home() / "e1" / "code" / "gcp" / "remote" / "queue.sh")
    if Path(cmd).exists():
        subprocess.run(["bash", cmd, "pause", reason], check=False)
    else:
        print(f"PAUSE (no queue here): {reason}", flush=True)


def notify(run, name, tw):
    """§9 I12 S-3: never pauses. Appends to ~/e1/queue/notify.log (the developer's monitor reads it and tells the
    lead at once) and to the run's watch log."""
    msg = (f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {run} @ {name}: NOTIFY {','.join(tw['notify'])}: " +
           "; ".join(r["detail"] for r in tw["tripwires"] if r["status"] == "notify"))
    q = Path.home() / "e1" / "queue"
    if q.exists():
        with open(q / "notify.log", "a") as f:
            f.write(msg + "\n")
    print(msg, flush=True)


def trainer_pids(run, proc="/proc"):
    """The run's trainer: a process whose argv has the argument train.py followed (anywhere) by
    "--run-id RUN" or "--resume RUN" exactly (read from /proc/*/cmdline, so a shell or ssh command line that merely
    mentions the run never matches), plus its child processes (the env workers)."""
    procs = {}
    for e in os.listdir(proc):
        if not e.isdigit():
            continue
        try:
            argv = open(f"{proc}/{e}/cmdline", "rb").read().split(b"\0")
            ppid = int(open(f"{proc}/{e}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        procs[int(e)] = ([a.decode(errors="replace") for a in argv if a], ppid)

    def is_trainer(argv):
        if not any(Path(a).name == "train.py" for a in argv):
            return False
        return any(argv[i] in ("--run-id", "--resume") and argv[i + 1] == run for i in range(len(argv) - 1))
    top = [p for p, (argv, _) in procs.items() if is_trainer(argv)]
    return top + [p for p, (_, pp) in procs.items() if pp in top]


def freeze(run):
    """SIGSTOP the trainer during an interim read: concurrent evals collapse its OpenMP learner threads
    (measured on the batch-2 pilots: ~375 -> ~30 sps on e1-a). Training numerics don't depend on wall time."""
    import signal
    pids = trainer_pids(run)
    for p in pids:
        try:
            os.kill(p, signal.SIGSTOP)
        except ProcessLookupError:
            pass
    return pids


def thaw(pids):
    import signal
    for p in pids:
        try:
            os.kill(p, signal.SIGCONT)
        except ProcessLookupError:
            pass


def add_stop(d, t0, t1, name, n):
    p = d / "eval-interim" / "stops.json"
    iv = stops(d) + [[round(t0, 1), round(t1, 1), name, n]]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(iv))


def watch(run, reads, workers, poll):
    """reads: 'all' (every checkpoint), 'none' (summary only), or a list of 0 / 1M / 2.5M."""
    d = runs_dir() / run
    done_marker = d / ".train_done"
    while not (d / "config.json").exists():
        if done_marker.exists():
            return
        time.sleep(poll)
    cfg = json.loads((d / "config.json").read_text())
    want = lambda m: (reads == ["all"] or (0 in reads_n and m == 0) or (1 in reads_n and 900_000 <= m <= 1_100_000)
                      or (2.5 in reads_n and M25[0] <= m <= M25[1]))
    reads_n = {0: 0, "0": 0, "1M": 1, "2.5M": 2.5}
    reads_n = {reads_n[r] for r in reads if r in reads_n}
    seen = set()
    while True:
        final = done_marker.exists()
        cks = sorted({p for p in glob.glob(str(d / "eval-*.pt")) + glob.glob(str(d / "interim-*.pt"))
                      + glob.glob(str(d / "ckpt-0.pt"))}, key=step_of)
        for ck in cks:
            name = Path(ck).name
            if name in seen:
                continue
            seen.add(name)
            m = step_of(name)
            if reads != ["none"] and cfg["state"] in DEV and want(m):
                out = d / "eval-interim" / Path(name).stem
                if not (out / "summary.json").exists():
                    print(f"{time.strftime('%H:%M:%S')} interim eval {name} (trainer frozen)", flush=True)
                    frozen = freeze(run)
                    t0 = time.time()
                    try:
                        subprocess.run([sys.executable, str(HERE / "evaluate.py"), run, "--ckpts", name,
                                        "--states", cfg["state"], "--episodes", "8", "--workers", str(workers),
                                        "--window-stats", "--save-traces", "--kl-init", "--out", str(out)],
                                       check=False, stdout=open(runs_dir() / "logs" / f"{run}-interim.log", "a"),
                                       stderr=subprocess.STDOUT)
                    finally:
                        thaw(frozen)
                        add_stop(d, t0, time.time(), name, len(frozen))
                    if not (out / "summary.json").exists():
                        pause(f"{run}: interim eval of {name} failed (C: infrastructure)")
                        continue
                tw = tripwire(run)
                print(f"{time.strftime('%H:%M:%S')} {name}: tripwires " +
                      ", ".join(f"{r['id']} {r['status']}" for r in tw["tripwires"]), flush=True)
                if tw.get("notify"):
                    notify(run, name, tw)
                if tw["tripped"]:
                    pause(f"{run} @ {name}: tripwire {','.join(tw['tripped'])}: " +
                          "; ".join(r["detail"] for r in tw["tripwires"] if r["status"] == "trip"))
            else:
                summary(run)
        if final:
            summary(run)
            return
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["summary", "tripwire", "brief", "watch", "resumes"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--reads", nargs="+", default=["none"], help="watch: all | none | any of 0 1M 2.5M")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="interim eval workers (trainer is frozen)")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    if a.cmd == "summary":
        for r in a.runs:
            print(md(summary(r)))
    elif a.cmd == "resumes":                 # for the batch-2 report: every resumed run and its H9 status
        print("| run | resumes | replayed/lost steps | reached M | H9 status | why |\n|---|---|---|---|---|---|")
        for r in a.runs:
            rr = summary(r)["resume_rule"]
            if rr["resumes"]:
                print(f"| {r} | {rr['resumes']} | {rr['replayed_or_lost']:,} | {rr['reached_M']} | {rr['status']} | "
                      f"{'; '.join(rr['why']) or '-'} |")
    elif a.cmd == "brief":
        for r in a.runs:
            print(brief(r))
    elif a.cmd == "tripwire":
        rc = 0
        for r in a.runs:
            t = tripwire(r)
            for x in t["tripwires"]:
                print(f"{r} {x['id']:>4} {x['status']:>7}  {x['detail']}")
            rc |= bool(t["tripped"])
        sys.exit(rc)
    else:
        watch(a.runs[0], a.reads, a.workers, a.poll)


if __name__ == "__main__":
    main()
