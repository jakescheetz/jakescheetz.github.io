#!/usr/bin/env python3
"""§9 G5 part 1: max-attainable-P planner (diagnostic, not an arm). No training, dev states only.

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python g5_planner.py [--states D1 D2] [--seeds 1000 ... 1007]
      [--workers 4]  -> ~/roms/g5/planner/ (traces) + results/g5/planner.{jsonl,json,md}

Pure coverage, no route knowledge: at every overworld step it walks to the *nearest walkable square that
hasn't paid P0's explore term yet* (not in the proxy's own `coords` set), by BFS over walkgrid.json's move
graph (steps, ledge hops, warps/doors, map connections; walkdist.WalkDist.fwd). The plan is kept until its
target is paid or the player leaves the planned path, then re-planned. Visible NPC squares on the current
map are blocked in the BFS; a press that doesn't move us twice re-plans, and a target that fails 6 presses
in a row is dropped for the episode (counted as `dropped_targets`). A target not reached within
2 × its path length + 64 steps, summed over every time it was the target (battle steps don't count; text
does, since push-back scripts are dialogue) is dropped; a square that is
the failure point of two such timeouts (a script pushing the player back, e.g. the museum's ticket desk) becomes a *barrier*: impassable for the rest of the episode (logged in `barriers`). So does a square
where two battles that wiped out the party began (a fight the planner can't win, e.g. the Route 22 rival).
Battles, text and off-graph positions are handled as in greedy_planner.py (validation (b)): Fight / move 1
(once move 1's PP is gone: the next damaging move with PP, else RUN from a wild battle; see make_script),
B through text, B while coordinates are stale; a wipe-out is logged and the planner goes on.
The episode ends where the env ends it in eval: 16,384 steps or reaching the last milestone (M5).
Same env as evaluate.py: P0 (reward hash d28e2a337dda, from an A0 run's config), core, 24 frames/action,
16,384 steps, D1/D2, seeds 1000–1007 with the 0–255-frame jitter. Every input is an env.step.

Per episode: P (and by term), T, leg progress, distinct tiles, P per map, blackouts, battles, plus planner
health: dropped targets, steps with nothing reachable left, the longest stretch with no new tile.
Stuck rule (§9 G5): >= 2,048 steps with no new tile while an unvisited reachable square remains = planner
bug (`stuck: true`), to be fixed, not counted as a result.
2× rule (fixed before the planner ran, §9 G5 'Approved by the lead'): A0's pooled P at its best mark,
D1 55.5 (1M), D2 50.5 (0.5M) -> pass if planner mean P >= 111 (D1) / >= 101 (D2); (C) only if both pass.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import re
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import ram, rom  # noqa: E402

HERE = Path(__file__).resolve().parent
LIKE_RUN = "A0-D1-s1"                      # P0 + env settings come from an A0 run's config
P0_HASH = "d28e2a337dda"
STEPS = 16384
STUCK = 2048
A0_BEST = {"D1": (55.5, "1M"), "D2": (50.5, "0.5M")}   # §9 G5 'Approved by the lead'


class Budget(Exception):
    pass


NON_DAMAGING = ("TAIL WHIP", "GROWL", "LEER", "WITHDRAW")   # the starters' status moves in this segment


def cursor_move(t):
    """Move menu text -> (name under the '>' cursor, its current PP), read off the screen, as the game shows it:
    'RATTATA 3  SQUIRTLE 7 TYPE/ NORMAL 24/ 24 35/35  >TACKLE TAIL WHIP' (HP pair, then the cursor move's PP)."""
    pairs = re.findall(r"(\d+)/\s*(\d+)", t.split(">")[0])
    name = t.split(">", 1)[1] if ">" in t else ""
    return name, int(pairs[-1][0]) if pairs else None


def make_script(SS, pb, step):
    """script_states.Script with one change to fight(). Validation (b)'s rule is "Fight, move 1" every turn;
    its runs never emptied move 1, but a 16,384-step coverage episode does, and then the game answers
    "No PP left!" forever. Here, in the move menu: A on the cursor move if it is damaging and has PP, else
    `down` to the next move. If a full cycle finds none: wild battle -> RUN; trainer battle -> any move with
    PP (and with no PP at all the game uses Struggle without the menu)."""
    class PlannerScript(SS.Script):
        pp_switches = runs = 0

        def fight(self, rival=False):
            mem = self.mem
            st = {"cycle": 0, "run": False, "any": False}
            for i in range(1200):
                flag = mem[ram.W_IS_IN_BATTLE]
                if flag == 0xFF or (flag and ram.party_hp(mem) and sum(ram.party_hp(mem)) == 0):
                    raise SS.Blackout(f"party wiped out on map {mem[ram.W_CUR_MAP]:#04x}")
                if not flag and not self.d.screen_text():
                    return i
                t = self.d.screen_text()
                if "FIGHT" in t and "RUN" in t:
                    self.mem_cursor_top()
                    if st["run"]:
                        self.runs += 1
                        for b in ("down", "right", "a"):
                            self.d.act(b)
                    else:
                        st["cycle"] = 0
                        self.d.act("a")
                    continue
                if "TYPE/" in t and ">" in t:
                    name, pp = cursor_move(t)
                    ok = pp is not None and pp > 0 and (st["any"] or not name.startswith(NON_DAMAGING))
                    if ok:
                        self.d.act("a")
                        continue
                    st["cycle"] += 1
                    if st["cycle"] > 4:
                        if mem[ram.W_IS_IN_BATTLE] == 1:
                            st["run"] = True
                            self.d.act("b")                 # back to the main menu, then RUN
                        else:
                            st["any"] = True
                        st["cycle"] = 0
                        continue
                    self.pp_switches += 1
                    self.d.act("down")
                    continue
                self.d.act("b")
            raise SS.ScriptError(f"battle did not end; screen {self.d.screen_text()!r}")
    return PlannerScript(pb, None, step_fn=step, log=lambda *a: None)


def bfs_target(fwd, start, visited, blocked, dropped):
    """Nearest square (by moves) not yet paid: returns [(how, node), ...] from start, or None."""
    prev, q = {start: None}, deque([start])
    while q:
        u = q.popleft()
        if u != start and u not in visited and u not in dropped:
            path = []
            while prev[u] is not None:
                p, how = prev[u]
                path.append((how, u))
                u = p
            return path[::-1]
        for v, how in fwd.get(u, ()):
            if v in prev or v in blocked:
                continue
            prev[v] = (u, how)
            q.append(v)
    return None


def episode(job):
    state, seed = job
    import greedy_planner as G
    import script_states as SS
    import train as TR
    from pokered import env as E, walkdist
    cfg = json.loads((rom.runs_dir() / LIKE_RUN / "config.json").read_text())
    reward = TR.make_reward(cfg)
    assert E.reward_hash(reward) == P0_HASH, E.reward_hash(reward)
    wd = E.walk_dist()
    env = E.PokemonRedEnv(state_name=state, reward=reward, max_steps=STEPS, seed=seed, jitter=True,
                          obs_mode="ram", fast_mode=cfg["fast_mode"], frames_per_action=cfg["frames_per_action"],
                          keep_trace=True)
    env.reset()

    def step(b):
        if env.steps >= STEPS or env.T == len(ram.MILESTONES):     # the env's own episode end (as in eval)
            raise Budget
        env.step(E.ACTIONS.index(b))

    s = make_script(SS, env.pb, step)
    h = {"battles": 0, "blackouts": 0, "text_boxes": 0, "waits_off_graph": 0, "replans": 0, "npc_replans": 0,
         "dropped_targets": 0, "exhausted_steps": 0, "script_errors": 0, "max_no_new_gap": 0, "stuck": False,
         "stuck_at": None, "script_error_msgs": [], "barriers": []}
    plan, fails, tfails, dropped = [], {}, 0, set()
    barriers = set()          # squares a script keeps pushing us off (e.g. a ticket desk): impassable from then on
    spent, last_steps = {}, 0  # target -> non-battle steps spent on it, summed over every time it was the target
    fail_sq = {}              # square -> timeouts it was the failure point of
    tgt = None                # current target: {"node", "t0", "path", "far"} (far = furthest path index reached)
    last_new, n_paid = 0, len(reward.coords)
    t0 = time.time()
    try:
        while env.steps < STEPS and env.T < len(ram.MILESTONES):
            if len(reward.coords) > n_paid:
                n_paid, last_new = len(reward.coords), env.steps
            gap = env.steps - last_new
            h["max_no_new_gap"] = max(h["max_no_new_gap"], gap)
            if gap >= STUCK and not h["stuck"]:
                here = s.d.pos()
                if here in wd.fwd and bfs_target(wd.fwd, here, reward.coords, barriers, dropped):
                    h["stuck"], h["stuck_at"] = True, {"step": env.steps, "pos": list(here)}
            if s.mem[ram.W_IS_IN_BATTLE]:
                h["battles"] += 1
                bpos = s.d.pos()
                try:
                    s.fight()
                except SS.Blackout:
                    h["blackouts"] += 1
                    # a fight the planner can't win (e.g. the Route 22 rival): two wipe-outs that began on the
                    # same square make it a barrier, so the planner stops walking back into it
                    fail_sq[("wipe",) + bpos] = fail_sq.get(("wipe",) + bpos, 0) + 1
                    if fail_sq[("wipe",) + bpos] >= 2 and bpos in wd.fwd:
                        barriers.add(bpos)
                        h["barriers"].append({"step": env.steps, "square": list(bpos), "why": "2 wipe-outs"})
                    plan, tgt = [], None
                    s.talk_until(lambda: not s.mem[ram.W_IS_IN_BATTLE] and s.control(4), limit=2000)
                except SS.ScriptError as e:
                    h["script_errors"] += 1
                    h["script_error_msgs"] = (h["script_error_msgs"] + [f"{env.steps}: {str(e)[:80]}"])[:5]
                try:
                    s.talk_until(lambda: s.control(4) or s.mem[ram.W_IS_IN_BATTLE] not in (0, 0xFF), limit=400)
                except SS.ScriptError as e:
                    h["script_errors"] += 1
                    h["script_error_msgs"] = (h["script_error_msgs"] + [f"{env.steps}: {str(e)[:80]}"])[:5]
                plan = []
                last_steps = env.steps            # battle steps are not charged to the target
                continue
            if s.d.screen_text():
                h["text_boxes"] += 1
                s.d.act("b")
                continue
            here = s.d.pos()
            if here not in wd.fwd:
                h["waits_off_graph"] += 1
                s.d.act("b")
                continue
            if tgt and tgt["node"] not in reward.coords:
                if here in tgt["path"]:
                    tgt["far"] = max(tgt["far"], tgt["path"].index(here))
                spent[tgt["node"]] = spent.get(tgt["node"], 0) + env.steps - last_steps
                last_steps = env.steps
                if spent[tgt["node"]] > 2 * len(tgt["path"]) + 64:
                    # timed out (every step since the target was set except battle steps: dialogue loops count): drop the target; the
                    # first path square never stood on becomes a barrier once it has failed twice
                    b = tgt["path"][min(tgt["far"] + 1, len(tgt["path"]) - 1)]
                    dropped.add(tgt["node"])
                    h["dropped_targets"] += 1
                    fail_sq[b] = fail_sq.get(b, 0) + 1
                    if fail_sq[b] >= 2:
                        barriers.add(b)
                        h["barriers"].append({"step": env.steps, "square": list(b), "target": list(tgt["node"])})
                    plan, tgt = [], None
            if not plan or plan[-1][1] in reward.coords or plan[0][2] != here:
                npcs = {(here[0], x, y) for x, y in G.visible_npcs(s.mem)}
                path = bfs_target(wd.fwd, here, reward.coords, npcs | (barriers - {here}), dropped)
                h["replans"] += 1
                if not path:
                    h["exhausted_steps"] += 1
                    s.d.act("b")
                    plan = []
                    continue
                plan, cur = [], here
                for how, node in path:
                    plan.append((how, node, cur))
                    cur = node
                tfails = 0
                if not tgt or tgt["node"] != path[-1][1]:      # a new target (re-plans to the same one keep its clock)
                    last_steps = env.steps
                    tgt = {"node": path[-1][1], "t0": env.steps, "path": [here] + [n for _, n in path], "far": 0}
            how, node, src = plan[0]
            s.d.act(G.button_for(how, *src, wd))
            now = s.d.pos()
            if now == node:
                plan.pop(0)
                fails.pop(src, None)
            elif now == src and not s.mem[ram.W_IS_IN_BATTLE] and not s.d.screen_text():
                fails[src] = fails.get(src, 0) + 1
                tfails += 1
                if fails[src] >= 2:                      # an NPC (or a turn) in the way: re-plan around it
                    h["npc_replans"] += 1
                    plan = []
                if tfails >= 6:
                    dropped.add(plan[-1][1] if plan else node)
                    h["dropped_targets"] += 1
                    plan, tfails = [], 0
            # else: mid-hop / warp with stale coordinates / pushed elsewhere: plan[0][2] != here re-plans
    except Budget:
        pass
    trace = env.trace
    terms = {k: round(sum(r[f"r_{k}"] for r in trace), 4) for k in ("explore", "level", "heal", "seen")}
    per_map = {}
    for r in trace:
        per_map[hex(r["map"])] = per_map.get(hex(r["map"]), 0.0) + r["p"]
    out = {"state": state, "seed": seed, "steps": env.steps,
           "ended": "M5 (episode end, as in eval)" if env.T == len(ram.MILESTONES) else "16,384 steps", "T": env.T, "P": round(env.P, 4), "P_terms": terms,
           "leg_progress": walkdist.leg_progress(trace, wd), "milestone_steps": env.milestone_steps,
           "distinct_tiles": len({(r["map"], r["x"], r["y"]) for r in trace}), "paid_tiles": len(reward.coords),
           "P_per_map": {k: round(v, 4) for k, v in sorted(per_map.items(), key=lambda kv: -kv[1])},
           "blackout_steps": env.blackout_steps, **h, "final_pos": list(ram.position(env.pb.memory)),
           "pp_switch_presses": s.pp_switches, "run_attempts": s.runs, "reward_hash": E.reward_hash(reward), "walkgrid": wd.hash, "seconds": round(time.time() - t0, 1)}
    env.close()
    td = rom.runs_dir().parent / "g5" / "planner" / "traces"
    td.mkdir(parents=True, exist_ok=True)
    with gzip.open(td / f"planner-{state}-{seed}.jsonl.gz", "wt", compresslevel=6) as f:
        f.write(json.dumps(out) + "\n")
        for r in trace:
            f.write(json.dumps(r) + "\n")
    return out


def report(rows, od):
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    res = {"cells": {}, "rule": "pass if planner mean P >= 2x A0's pooled P at its best mark; (C) only if both pass"}
    for st in sorted({r["state"] for r in rows}):
        rs = [r for r in rows if r["state"] == st]
        ok = [r for r in rs if not r["stuck"]]
        ref, mark = A0_BEST[st]
        P = mean([r["P"] for r in ok])
        res["cells"][st] = {"episodes": len(rs), "stuck": len(rs) - len(ok), "P_mean": P,
                            "P_min": min((r["P"] for r in ok), default=None), "P_max": max((r["P"] for r in ok), default=None),
                            "P_terms_mean": {k: mean([r["P_terms"][k] for r in ok]) for k in ("explore", "level", "heal", "seen")},
                            "T_mean": mean([r["T"] for r in ok]), "legP_mean": mean([r["leg_progress"] for r in ok]),
                            "distinct_tiles_mean": mean([r["distinct_tiles"] for r in ok]),
                            "blackouts": sum(r["blackouts"] for r in ok), "dropped_targets": sum(r["dropped_targets"] for r in ok),
                            "exhausted_steps_mean": mean([r["exhausted_steps"] for r in ok]),
                            "A0_best": ref, "A0_best_mark": mark, "threshold": round(2 * ref, 1),
                            "ratio": round(P / ref, 2) if P is not None else None,
                            "passes_2x": P is not None and P >= 2 * ref and len(ok) == len(rs)}
    c = res["cells"]
    res["verdict"] = ("both pass: (C) indicated" if all(v["passes_2x"] for v in c.values()) else
                      "both fail: null real at this horizon (explanation 4)" if not any(v["passes_2x"] for v in c.values())
                      else "mixed: lead/Jake decide, no default")
    if any(v["stuck"] for v in c.values()):
        res["verdict"] = "INVALID: planner stuck in some episode (bug, fix and re-run)"
    res["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (od / "planner.json").write_text(json.dumps(res, indent=1))
    L = ["# G5 part 1: max-attainable-P planner (diagnostic, not an arm)", "",
         "Pure-coverage BFS to the nearest square that hasn't paid P0's explore term; D1/D2 × seeds 1000–1007, "
         "16,384 steps, P0 (hash d28e2a337dda), same env and jitter as eval. Dev states only.", "",
         "| state | eps | stuck | planner P (min–max) | A0 best P (mark) | 2× threshold | ratio | passes | T | legP | tiles | explore / level / heal | blackouts |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st, v in c.items():
        t = v["P_terms_mean"]
        L.append(f"| {st} | {v['episodes']} | {v['stuck']} | {v['P_mean']} ({v['P_min']}–{v['P_max']}) | {v['A0_best']} ({v['A0_best_mark']}) | "
                 f"{v['threshold']} | {v['ratio']}× | {'yes' if v['passes_2x'] else 'no'} | {v['T_mean']} | {v['legP_mean']} | "
                 f"{v['distinct_tiles_mean']} | {t['explore']} / {t['level']} / {t['heal']} | {v['blackouts']} |")
    L += ["", f"**Verdict (rule fixed before the run):** {res['verdict']}.", "",
          "Planner health: dropped targets " + ", ".join(f"{st} {v['dropped_targets']}" for st, v in c.items()) +
          "; mean steps with nothing reachable left " + ", ".join(f"{st} {v['exhausted_steps_mean']}" for st, v in c.items()) + ".",
          "", "## Per episode", "",
          "| state | seed | P | T | legP | tiles | top maps by P | blackouts | max no-new gap | stuck |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["state"], r["seed"])):
        top = ", ".join(f"{k} {v:.1f}" for k, v in list(r["P_per_map"].items())[:3])
        L.append(f"| {r['state']} | {r['seed']} | {r['P']} | {r['T']} | {r['leg_progress']} | {r['distinct_tiles']} | {top} | "
                 f"{r['blackouts']} | {r['max_no_new_gap']} | {r['stuck']} |")
    L += ["", "Full rows (P per map, battles, re-plans): `planner.jsonl`. Traces outside the repo: `~/roms/g5/planner/traces/`."]
    (od / "planner.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", nargs="+", default=["D1", "D2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1008)))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results" / "g5"))
    a = ap.parse_args()
    if set(a.states) - {"D1", "D2"}:
        sys.exit("dev states only (D1, D2)")
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    jobs = [(st, sd) for st in a.states for sd in a.seeds]
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool, open(od / "planner.jsonl", "w") as f:
        for r in pool.imap_unordered(episode, jobs):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  {r['state']} seed {r['seed']}: P {r['P']} T {r['T']} tiles {r['distinct_tiles']} "
                  f"blackouts {r['blackouts']} gap {r['max_no_new_gap']} stuck {r['stuck']} ({r['seconds']} s)", flush=True)
    report(rows, od)


if __name__ == "__main__":
    main()
