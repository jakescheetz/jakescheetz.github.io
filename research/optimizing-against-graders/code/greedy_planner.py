#!/usr/bin/env python3
"""A2″ validation (b): a BFS-greedy planner must reach M2 from D1 in the emulator, ideally M4
(experiment-e1.md §9 "A2′ pilot"). No training.

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python greedy_planner.py [--seeds 0 1 2] [--target 4]

Policy, per env step (every input is an env.step under the A2″ reward, so the rows are trainer rows):
  * overworld with control: press the move whose destination has the smallest walking distance d to
    the next milestone (pokered/walkdist.WalkDist.best_moves: a step, a ledge hop, an edge exit or a
    warp). That is strict greedy descent on the BFS field, which is a shortest path. If a *visible*
    NPC sprite holds that square, or 2 presses there didn't move us (an NPC stepped in), it takes the
    shortest same-map detour around the NPC squares to the nearest square with smaller d (counted as
    npc_detours); with no detour, or 6 failed presses, it waits a step (B, counted as stuck_waits).
  * on a warp square: press toward the map edge it sits on (indoor mats: down), as the game needs.
  * wild/trainer battle (wIsInBattle != 0): "Fight, move 1" every turn, text advanced with B, i.e.
    script_states.Script.fight, the rule the start states were made with (§2). A wipe-out is logged
    as a blackout and the planner just carries on from where the game puts it.
  * any other text box (signs, NPC lines, trainer "!" approach, item pickups): B.
  * position not in the graph (mid ledge hop, the 1-2 stale-coordinate rows after a warp): B (wait).
No healing, no menus, no items: the planner gets only what walking and move-1 battles give.

Outputs (outside the repo, rom.states_dir()): greedy-<seed>.jsonl (env trace rows + d, progress),
greedy-summary.json (per seed: milestone steps, battles, text boxes, blackouts, stuck events, A2″
progress paid per leg).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import ram, rom, walkdist  # noqa: E402
from pokered.env import ACTIONS, PokemonRedEnv, ProgressOracle  # noqa: E402
import script_states as SS  # noqa: E402


def visible_npcs(mem):
    """Squares held by *visible* NPC sprites. driver.sprites() also lists hidden ones (a toggled-off
    object keeps its slot; pret: SPRITESTATEDATA1_IMAGEINDEX = $ff when hidden), e.g. Viridian's lying
    old man after the Pokédex. driver.py is left as is so the approved start states replay unchanged."""
    from pokered.driver import W_SPRITE_STATE_1, W_SPRITE_STATE_2
    out = set()
    for i in range(1, 16):
        if mem[W_SPRITE_STATE_1 + 16 * i] == 0 or mem[W_SPRITE_STATE_1 + 16 * i + 2] == 0xFF:
            continue
        out.add((mem[W_SPRITE_STATE_2 + 16 * i + 5] - 4, mem[W_SPRITE_STATE_2 + 16 * i + 4] - 4))
    return out


def detour(wd, leg, start, here_d, npcs, limit=400):
    """Shortest path on the same map around NPC squares to the nearest square with d < here_d:
    a moving NPC on the greedy line costs a short detour instead of a wait."""
    from collections import deque
    prev, q = {start: None}, deque([start])
    while q:
        u = q.popleft()
        du = wd.d(leg, *u)
        if u != start and du is not None and du < here_d:
            while prev[u][0] != start:
                u = prev[u][0]
            return prev[u][1]
        if len(prev) > limit:
            break
        for v, how in wd.fwd.get(u, ()):
            if v in prev or v[0] != start[0] or v[1:] in npcs:
                continue
            prev[v] = (u, how)
            q.append(v)
    return None


def button_for(how, mp, x, y, wd):
    if how == "warp":
        g = wd.maps[mp]
        return "down" if y >= g["h"] - 1 else "up" if y == 0 else "left" if x == 0 else \
            "right" if x >= g["w"] - 1 else "down"
    return how.split("-")[-1]


def run(seed, target, budget, log):
    sd = rom.states_dir()
    path = sd / f"greedy-{seed}.jsonl"
    path.unlink(missing_ok=True)
    reward = ProgressOracle()
    wd = reward.wd
    env = PokemonRedEnv(state_name="D1", obs_mode="ram", seed=seed, jitter=True, max_steps=10 ** 9,
                        trace_path=str(path), run_id=f"greedy-{seed}", reward=reward)
    env.reset()
    s = SS.Script(env.pb, None, step_fn=lambda b: env.step(ACTIONS.index(b)), log=lambda *a: None)
    ev = {"battles": [], "text_boxes": 0, "text_samples": {}, "blackouts": 0, "stuck_waits": 0,
          "npc_detours": 0, "waits_off_graph": 0, "milestone_steps": {}, "progress_by_leg": {}}
    fails = {}
    t0 = time.time()

    def battle():
        mp = s.mem[ram.W_CUR_MAP]
        kind = {1: "wild", 2: "trainer"}.get(s.mem[ram.W_IS_IN_BATTLE], s.mem[ram.W_IS_IN_BATTLE])
        st, hp0 = env.steps, sum(ram.party_hp(s.mem))
        try:
            s.fight()
            lost = False
        except SS.Blackout:
            # fight() raises as soon as the party is wiped out (wIsInBattle = $ff), before the blackout
            # text and the warp: press B through them until the game hands back control
            lost = True
            ev["blackouts"] += 1
            s.talk_until(lambda: not s.mem[ram.W_IS_IN_BATTLE] and s.control(4), limit=2000)
        s.talk_until(lambda: s.control(4) or s.mem[ram.W_IS_IN_BATTLE] not in (0, 0xFF), limit=400)
        ev["battles"].append({"map": mp, "kind": kind, "steps": env.steps - st, "hp_before": hp0,
                              "hp_after": sum(ram.party_hp(s.mem)), "lost": lost, "at": st})

    while env.T < target and env.steps < budget:
        leg_T = env.T
        if s.mem[ram.W_IS_IN_BATTLE]:
            battle()
        elif s.d.screen_text():
            t = s.d.screen_text()
            ev["text_boxes"] += 1
            key = f"{s.mem[ram.W_CUR_MAP]:#04x}"
            ev["text_samples"].setdefault(key, [])
            if len(ev["text_samples"][key]) < 3 and t[:40] not in ev["text_samples"][key]:
                ev["text_samples"][key].append(t[:40])
            s.d.act("b")
        else:
            mp, x, y = s.d.pos()
            here = wd.d(env.T + 1, mp, x, y)
            moves = wd.best_moves(env.T + 1, mp, x, y)
            if here is None or not moves:
                ev["waits_off_graph"] += 1
                s.d.act("b")
            else:
                npcs = visible_npcs(s.mem)
                best = moves[0]
                k = fails.get((mp, x, y), 0)
                how = None
                if best[0] < here and not (best[2][0] == mp and best[2][1:] in npcs) and k < 2:
                    how = best[1]                                   # plain greedy step
                else:                                               # an NPC (or a failed press): detour
                    how = detour(wd, env.T + 1, (mp, x, y), here, npcs)
                    if how:
                        ev["npc_detours"] += 1
                if how is None or k >= 6:
                    ev["stuck_waits"] += 1
                    fails[(mp, x, y)] = 0
                    s.d.act("b")
                else:
                    s.d.act(button_for(how, mp, x, y, wd))
                    if s.d.pos() == (mp, x, y) and not s.mem[ram.W_IS_IN_BATTLE] and not s.d.screen_text():
                        fails[(mp, x, y)] = k + 1
                    else:
                        fails.pop((mp, x, y), None)
        if env.T > leg_T:
            for k in range(leg_T, env.T):
                ev["milestone_steps"][ram.MILESTONES[k]] = env.steps
            log(f"  seed {seed}: {ram.MILESTONES[env.T - 1]} at step {env.steps} "
                f"(battles {len(ev['battles'])}, blackouts {ev['blackouts']}, {time.time() - t0:.0f} s)")
    # A2″ progress paid per leg, from the trace rows (r − ΔT); flush first so the last rows are in the file
    env.trace_file.flush()
    ev["progress_total"] = round(reward.progress.paid, 6)
    rows = [json.loads(l) for l in open(path)]
    T_prev = rows[0]["T"] if rows else 0
    for i, r in enumerate(rows):
        leg = (T_prev if r["T"] == T_prev else r["T"] - 1) + 1
        prog = r["r"] - (r["T"] - T_prev)
        ev["progress_by_leg"][leg] = round(ev["progress_by_leg"].get(leg, 0.0) + prog, 6)
        T_prev = r["T"]
    env.close()
    wild = [b for b in ev["battles"] if b["kind"] == "wild"]
    return {"seed": seed, "T": env.T, "steps": env.steps, "target": target, "budget": budget,
            "reached_M2": env.T >= 2, "reached_M4": env.T >= 4, "wall_s": round(time.time() - t0, 1),
            "battles_wild": len(wild), "battles_trainer": len(ev["battles"]) - len(wild),
            "battle_steps": sum(b["steps"] for b in ev["battles"]),
            "d_leg": wd.d_leg, "walkgrid": wd.hash, "trace": str(path), **ev}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--target", type=int, default=4, help="stop at this many milestones (4 = Pewter)")
    ap.add_argument("--budget", type=int, default=16384, help="env steps per seed (default: one episode)")
    a = ap.parse_args()
    out = []
    for sd in a.seeds:
        r = run(sd, a.target, a.budget, print)
        print(f"seed {sd}: T={r['T']} in {r['steps']} steps; wild battles {r['battles_wild']}, trainer "
              f"{r['battles_trainer']}, blackouts {r['blackouts']}, text boxes {r['text_boxes']}, "
              f"stuck waits {r['stuck_waits']}, progress by leg {r['progress_by_leg']}")
        out.append(r)
    p = rom.states_dir() / "greedy-summary.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
