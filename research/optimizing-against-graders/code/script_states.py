#!/usr/bin/env python3
"""Make the E1 start states by script, no hand play (experiment-e1.md §2 "Scripted states", §9.1).

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python script_states.py            # everything
  .venv/bin/python script_states.py --only starters HB ...                           # some stages
  .venv/bin/python script_states.py --only trace jitter                              # checks only

Stages (each waypoint ends in a RAM check that fails loudly, and a screenshot):
  PRE_STARTER  power-on -> intro -> default names (RED / BLUE) -> bedroom -> Pallet -> Oak stops
               you at Route 1 -> Oak's Lab, control returned just before the starter choice
  starters     D1 / D2 / HA from PRE_STARTER with ONE shared jitter: starter (Squirtle / Charmander /
               Bulbasaur), rival battle (move 1; losing it is allowed by the game), parcel from Viridian
               Mart, full heal at Mom (not a Center: SetLastBlackoutMap), back to Oak, Pokédex.
               §2 start-condition rules: no blackouts (a wipe-out aborts that jitter), full HP, levels
               equal if any jitter in SEARCH_JITTERS gives that, else a 1-level gap (>= 2 is an error).
  HB           from D1: straight to the Viridian Pokémon Center door, no grass detours, only
               unavoidable battles, "Fight, move 1", no blackouts; then healed by RAM write (HP := max,
               status := 0) with a WRAM diff check and wLastBlackoutMap == D1's (§2)
  trace        §9.1 check trace through PokemonRedEnv: HP falls/heals, battle flag, M1, M2; one heal
               alone in a window (HF must not fire) and two heals in one window (HF must fire)
  jitter       each state x 8 jitter values x 200 scripted steps: same jitter -> identical RAM,
               different jitter -> different RNG bytes (hRandomAdd/Sub)

Outputs, all outside the repo (rom.states_dir(), default ~/roms/states/):
  <NAME>.state + <NAME>.json      (snapshot, party species, sanity check, ROM SHA-1, PyBoy version, core)
  script-frames/<NN>-<waypoint>.png + script-frames/waypoints.jsonl (RAM values per frame, for the review)
  script-trace.jsonl, script-trace-summary.json, jitter-check.json, start-states-log.json
No state files or TAS inputs are downloaded: all inputs are generated here.
"""
import argparse
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

from pokered import detectors, ram, rom
from pokered.driver import (DIRS, SPECIES, W_JOY_IGNORE, W_PARTY_MON1_EXP, W_PARTY_SPECIES,
                            Driver, ScriptError)

import os
# Same core as the trainer (§2 condition 3). §5b failed on D1 (2026-11-02), so the normal core.
# E1_STATES_FAST_MODE=1 only to reproduce the earlier fast_mode states. Recorded in every .json.
FAST_MODE = os.environ.get("E1_STATES_FAST_MODE") == "1"
STARTER_OF = {"D1": "SQUIRTLE", "D2": "CHARMANDER", "HA": "BULBASAUR"}


# --- shared helpers -----------------------------------------------------------------------------
class Blackout(ScriptError):
    """The party was wiped out in a battle that counts (§2 rules: no blackouts on the scripted path)."""


class Script:
    warp_fallback = True        # fastmode_check.py --repro-first turns this off to replay its first run
    def __init__(self, pb, frames_dir, log=print, step_fn=None):
        """frames_dir None: RAM checks only, no screenshots (used while searching for a jitter)."""
        self.pb = pb
        self.d = Driver(pb, step_fn=step_fn, log=log)
        self.frames_dir = frames_dir
        self.log = log
        self.n_frame = len(list(frames_dir.glob("*.png"))) if frames_dir else 0
        self.battles = []          # one entry per battle: map, kind, won/lost

    @property
    def mem(self):
        return self.pb.memory

    # waypoint = RAM check + screenshot + a row in waypoints.jsonl
    def waypoint(self, name, expect=None):
        m = self.mem
        snap = ram.snapshot(m)
        snap["species"] = party_species(m)
        snap["exp"] = party_exp(m)
        snap["T_if_from_here"] = ram.update_T(0, m)
        bad = []
        for k, v in (expect or {}).items():
            got = snap[k] if k in snap else None
            if callable(v):
                if not v(got):
                    bad.append(f"{k}={got!r} fails check")
            elif got != v:
                bad.append(f"{k}={got!r}, expected {v!r}")
        png = Path("(no frame)")
        # Always advance exactly one frame here, rendered only when saving a screenshot: the timing (and
        # so the RNG) must be the same with and without frames, or the jitter search would not replay.
        self.pb.tick(1, bool(self.frames_dir), False)
        if self.frames_dir:
            self.n_frame += 1
            png = self.frames_dir / f"{self.n_frame:02d}-{name}.png"
            self.pb.screen.image.convert("RGB").save(png)
            row = {"n": self.n_frame, "waypoint": name, "frame": png.name, "ok": not bad, "problems": bad,
                   "screen_text": self.d.screen_text(), **snap}
            with open(self.frames_dir / "waypoints.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        self.log(f"  [{'ok' if not bad else 'FAIL'}] {name}: map {snap['map']:#04x} ({snap['x']},{snap['y']}) "
                 f"levels {snap['levels']} hp {snap['hp']} species {snap['species']} -> {png.name}")
        if bad:
            raise ScriptError(f"waypoint {name}: " + "; ".join(bad) + f" (see {png})")
        return snap

    def control(self, n=6):
        """True when the overworld is idle: no text on screen and input not ignored, n checks in a row."""
        for _ in range(n):
            if self.d.screen_text() or self.mem[W_JOY_IGNORE]:
                return False
            self.d.wait(1)
        return True

    def talk_until(self, done, limit=400, button="b", on_text=None):
        """Advance dialogue with `button` until done() is true. on_text(text) may return a button to
        press instead (menus, YES/NO); it sees the decoded screen each step."""
        for i in range(limit):
            if done():
                return i
            t = self.d.screen_text()
            b = on_text(t) if on_text else None
            self.d.act(b or button)
        raise ScriptError(f"talk_until: no progress after {limit} actions; screen: {self.d.screen_text()!r}")

    def walk_to(self, goal, limit=400, battle=None, grass_cost=8):
        """Walk to (x, y) on the current map, replanning every step (NPCs move). Handles battles and
        dialogue that interrupt the walk. Returns when on the goal or the map changes."""
        m0 = self.mem[ram.W_CUR_MAP]
        for _ in range(limit):
            mp, x, y = self.d.pos()
            if mp != m0 or (x, y) == goal:
                return
            if self.mem[ram.W_IS_IN_BATTLE]:
                (battle or self.fight)()
                continue
            if self.d.screen_text():
                self.d.act("b")
                continue
            path = self.d.plan(goal, grass_cost=grass_cost)
            if not path:
                self.d.act("b")       # blocked (an NPC on the only path): wait a step
                continue
            self.d.act(path[0])
        raise ScriptError(f"walk_to {goal} on map {m0:#04x}: stuck at {self.d.pos()}")

    def step_off(self, direction, limit=40, battle=None):
        """Walk `direction` until the map changes (edge connections, door mats, stairs).
        Battles and dialogue that interrupt (an NPC, a wild encounter) are handled first."""
        m0 = self.mem[ram.W_CUR_MAP]
        for _ in range(limit):
            if self.mem[ram.W_CUR_MAP] != m0:
                break
            if self.mem[ram.W_IS_IN_BATTLE]:
                (battle or self.fight)()
            elif self.d.screen_text():
                self.d.act("b")
            else:
                self.d.act(direction)
        else:
            raise ScriptError(f"step_off {direction}: still on map {m0:#04x} at {self.d.pos()}")
        self.talk_until(self.control, limit=60)

    def warp_to(self, dest_map, direction="down", battle=None):
        """Walk onto the warp square that leads to dest_map, then into it."""
        ws = [w for w in self.d.warps() if w["map"] == dest_map or (dest_map == "out" and w["map"] == 0xFF)]
        if not ws:
            raise ScriptError(f"no warp to {dest_map} from map {self.mem[ram.W_CUR_MAP]:#04x}: {self.d.warps()}")
        _, x, y = self.d.pos()
        w = min(ws, key=lambda w: abs(w["x"] - x) + abs(w["y"] - y))
        # some warp pairs have one square the player cannot stand on (e.g. the Viridian Forest south
        # gate's (4,0)); only then switch to a reachable one. The nearest-square rule is kept otherwise
        # so the approved start states replay identically.
        if self.warp_fallback and self.d.plan((w["x"], w["y"]), allow_goal_blocked=False) is None:
            reach = [(len(pth), v) for v in ws
                     if (pth := self.d.plan((v["x"], v["y"]), allow_goal_blocked=False)) is not None]
            if reach:
                w = min(reach, key=lambda t: t[0])[1]
        m0 = self.mem[ram.W_CUR_MAP]
        self.walk_to((w["x"], w["y"]), battle=battle)
        if self.mem[ram.W_CUR_MAP] == m0 and self.d.pos()[1:] == (w["x"], w["y"]):
            self.step_off(direction, battle=battle)
        else:
            self.talk_until(self.control, limit=60)

    # battles: "Fight, move 1" every turn (§2 condition 2); dialogue advanced with B
    def fight(self, rival=False):
        """rival=True: the first rival battle, which the game lets you lose without a blackout."""
        entry = {"map": self.mem[ram.W_CUR_MAP], "kind": self.mem[ram.W_IS_IN_BATTLE], "rival": rival,
                 "hp_before": ram.party_hp(self.mem), "lost": False}
        self.battles.append(entry)

        def on_text(t):
            if "FIGHT" in t and "RUN" in t:
                return "a"                     # cursor rests on FIGHT
            if t.startswith(">") or "TYPE/" in t or "PP" in t:
                return "a"                     # move menu: first move
            return None
        for i in range(600):
            flag = self.mem[ram.W_IS_IN_BATTLE]
            if flag == 0xFF or (flag and ram.party_hp(self.mem) and sum(ram.party_hp(self.mem)) == 0):
                entry["lost"] = True                 # wIsInBattle -1 = lost (pret wram.asm)
                if not rival:
                    raise Blackout(f"party wiped out on map {entry['map']:#04x} (battle {len(self.battles)})")
            if not flag and not self.d.screen_text():
                entry["hp_after"] = ram.party_hp(self.mem)
                return i
            t = self.d.screen_text()
            b = on_text(t)
            if "FIGHT" in t and "RUN" in t:
                self.mem_cursor_top()
            self.d.act(b or "b")
        raise ScriptError(f"battle did not end; screen {self.d.screen_text()!r}")

    def mem_cursor_top(self):
        """Battle main menu: make sure the cursor is on FIGHT (top-left) before pressing A."""
        for b in ("up", "left"):
            self.d.act(b)


def party_species(m):
    return [SPECIES.get(m[W_PARTY_SPECIES + i], hex(m[W_PARTY_SPECIES + i])) for i in range(ram.party_count(m))]


def party_exp(m):
    return [m[W_PARTY_MON1_EXP + ram.PARTY_STRIDE * k] << 16 | m[W_PARTY_MON1_EXP + ram.PARTY_STRIDE * k + 1] << 8
            | m[W_PARTY_MON1_EXP + ram.PARTY_STRIDE * k + 2] for k in range(ram.party_count(m))]


# --- maps (pret constants/map_constants.asm) ----------------------------------------------------
PALLET_TOWN, VIRIDIAN_CITY, ROUTE_1, ROUTE_2 = 0x00, 0x01, 0x0C, 0x0D
REDS_HOUSE_1F, REDS_HOUSE_2F, OAKS_LAB = 0x25, 0x26, 0x28
VIRIDIAN_POKECENTER, VIRIDIAN_MART = 0x29, 0x2A


# --- stage 1: power-on -> PRE_STARTER -----------------------------------------------------------
def stage_pre_starter(s):
    d = s.d
    s.talk_until(lambda: "NEW GAME" in d.screen_text(), limit=200,
                 on_text=lambda t: None, button="start")
    s.waypoint("main-menu", {})
    d.act("a")
    names = []

    def on_text(t):
        if "NEW NAME" in t:
            names.append(t)
            d.act("down")            # first preset under NEW NAME: RED, then BLUE
            return "a"
        return "a"
    s.talk_until(lambda: len(names) == 2 and s.control(), limit=400, on_text=on_text)
    player = decode_name(s.mem, 0xD158)
    rival = decode_name(s.mem, 0xD34A)
    s.waypoint("bedroom", {"map": REDS_HOUSE_2F, "party_count": 0})
    if (player, rival) != ("RED", "BLUE"):
        raise ScriptError(f"names {player}/{rival}, expected the default RED/BLUE")
    s.warp_to(REDS_HOUSE_1F)
    s.waypoint("house-1f", {"map": REDS_HOUSE_1F})
    s.warp_to("out")
    s.waypoint("pallet-town", {"map": PALLET_TOWN})
    # north exit: Oak's script stops the player at y == 1 and leads him to the lab
    s.walk_to((10, 1))
    s.talk_until(lambda: s.mem[ram.W_CUR_MAP] == OAKS_LAB and s.control(10), limit=600)
    s.waypoint("PRE_STARTER", {"map": OAKS_LAB, "party_count": 0, "ev_pokedex": False})
    return {"player": player, "rival": rival}


def decode_name(m, addr):
    from pokered.driver import CHARS
    out = ""
    for i in range(11):
        c = m[addr + i]
        if c == 0x50:
            break
        out += CHARS.get(c, "?")
    return out


# --- stage 2: PRE_STARTER -> starter -> rival -> parcel -> Pokédex -------------------------------
BALLS = [(6, 3), (7, 3), (8, 3)]    # the three Poké Balls on Oak's table; which is which is read from the dialogue


def go_edge(s, direction, battle=None):
    """Leave the current map through its edge in `direction` (map connections)."""
    passable, _ = s.d.walk_grid()
    H, Wd = len(passable), len(passable[0])
    if direction in ("up", "down"):
        yy = 0 if direction == "up" else H - 1
        cands = [(x, yy) for x in range(Wd) if passable[yy][x]]
    else:
        xx = 0 if direction == "left" else Wd - 1
        cands = [(xx, y) for y in range(H) if passable[y][xx]]
    best = None
    for c in cands:
        p = s.d.plan(c)
        if p is not None and (best is None or len(p) < best[0]):
            best = (len(p), c)
    if not best:
        raise ScriptError(f"no reachable {direction} edge on map {s.mem[ram.W_CUR_MAP]:#04x}")
    s.walk_to(best[1], battle=battle)
    if s.d.pos()[1:] == best[1]:
        s.step_off(direction, battle=battle)


# map graph for the segment start: (from, to) -> how. Recomputed every hop, so a blackout (which
# drops the player back in Pallet) just reroutes instead of breaking the script.
EDGES = {
    PALLET_TOWN: {ROUTE_1: ("edge", "up"), OAKS_LAB: ("warp", "up"), REDS_HOUSE_1F: ("warp", "up")},
    ROUTE_1: {PALLET_TOWN: ("edge", "down"), VIRIDIAN_CITY: ("edge", "up")},
    VIRIDIAN_CITY: {ROUTE_1: ("edge", "down"), ROUTE_2: ("edge", "up"),
                    VIRIDIAN_MART: ("warp", "up"), VIRIDIAN_POKECENTER: ("warp", "up")},
    ROUTE_2: {VIRIDIAN_CITY: ("edge", "down")},
    OAKS_LAB: {PALLET_TOWN: ("out", "down")},
    REDS_HOUSE_1F: {PALLET_TOWN: ("out", "down")},
    VIRIDIAN_MART: {VIRIDIAN_CITY: ("out", "down")},
    VIRIDIAN_POKECENTER: {VIRIDIAN_CITY: ("out", "down")},
}


def next_hop(cur, target):
    from collections import deque
    prev, q = {cur: None}, deque([cur])
    while q:
        u = q.popleft()
        if u == target:
            break
        for v in EDGES.get(u, {}):
            if v not in prev:
                prev[v] = u
                q.append(v)
    if target not in prev:
        raise ScriptError(f"no route from map {cur:#04x} to {target:#04x}")
    v = target
    while prev[v] != cur:
        v = prev[v]
    return v


def travel(s, target, battle=None, log=None, limit=40):
    for _ in range(limit):
        # wait out fades, warps and blackout text: the map header is only valid once control returns
        s.talk_until(lambda: s.control(6), limit=300)
        cur = s.mem[ram.W_CUR_MAP]
        if cur == target:
            return
        hop = next_hop(cur, target)
        how, direction = EDGES[cur][hop]
        if log is not None:
            log.append({"from": cur, "to": hop})
        if how == "edge":
            go_edge(s, direction, battle=battle)
        elif how == "warp":
            s.warp_to(hop, direction=direction, battle=battle)
        else:
            s.warp_to("out", direction=direction, battle=battle)
    raise ScriptError(f"travel to {target:#04x}: gave up at {s.d.pos()}")


def stage_starter(s, want, name):
    d = s.d
    s.waypoint(f"{name}-from-PRE_STARTER", {"map": OAKS_LAB, "party_count": 0})
    chosen, seen = [], []

    def on_text(t):
        for sp in SPECIES.values():          # the ball first shows that species' Pokédex page
            if t.startswith(sp):
                seen.append(sp)
        if "YES" in t and "NO" in t and seen:
            if seen[-1] == want:
                chosen.append(want)
                return "a"                   # YES
            return "b"                       # NO: wrong ball
        return None
    for bx, by in BALLS:
        s.walk_to((bx, by + 1))
        d.act("up")                          # face the ball
        d.act("a")
        s.talk_until(lambda: chosen or (seen and s.control()), limit=80, on_text=on_text)
        if chosen:
            break
    if not chosen:
        raise ScriptError(f"{want} not found on the table")
    # nickname prompt (B = NO), rival takes his ball
    s.talk_until(lambda: ram.party_count(s.mem) == 1 and s.control(10), limit=200)
    s.waypoint(f"{name}-starter", {"species": [want], "map": OAKS_LAB})
    # leaving the lab triggers the rival battle
    s.walk_to((5, 6))
    s.talk_until(lambda: s.mem[ram.W_IS_IN_BATTLE] != 0 and "FIGHT" in d.screen_text(), limit=300)
    s.waypoint(f"{name}-rival-battle", {"in_battle": 2})
    s.fight(rival=True)
    s.talk_until(lambda: s.control(10), limit=300)
    s.waypoint(f"{name}-after-rival", {"map": OAKS_LAB, "in_battle": 0})
    # parcel errand. Mom heals first (in-game, Pallet): the errand's Route 1 battles at L5-7 with no heal
    # wiped the party in 14 of 24 jitters, and blackouts are not allowed on the scripted path (§2)
    route = []
    travel(s, REDS_HOUSE_1F, log=route)
    mom_heal(s)
    s.waypoint(f"{name}-mom-healed-before-errand", {"map": REDS_HOUSE_1F, "hp": lambda v: v == ram.party_max_hp(s.mem)})
    travel(s, ROUTE_1, log=route)
    s.waypoint(f"{name}-route1", {"map": ROUTE_1})
    travel(s, VIRIDIAN_CITY, log=route)
    s.waypoint(f"{name}-viridian", {"map": VIRIDIAN_CITY})
    travel(s, VIRIDIAN_MART, log=route)
    s.talk_until(lambda: s.control(10), limit=300)
    s.waypoint(f"{name}-mart-parcel", {"map": VIRIDIAN_MART})
    travel(s, ROUTE_1, log=route)
    s.waypoint(f"{name}-route1-back", {"map": ROUTE_1})
    # full heal in-game before the Pokédex (§2 start-condition rules): Mom, not a Center, because a
    # Center heal calls SetLastBlackoutMap and would move the blackout point to Viridian
    travel(s, REDS_HOUSE_1F, log=route)
    blackout_map = s.mem[ram.W_LAST_BLACKOUT_MAP]
    mom_heal(s)
    s.waypoint(f"{name}-mom-healed", {"map": REDS_HOUSE_1F, "hp": lambda v: v == ram.party_max_hp(s.mem),
                                      "blackout_map": blackout_map})
    travel(s, OAKS_LAB, log=route)
    s.waypoint(f"{name}-lab-return", {"map": OAKS_LAB})
    # hand Oak the parcel: stand below him and talk
    oak = oak_square(s)
    s.walk_to((oak[0], oak[1] + 1))
    d.act("up")
    d.act("a")
    s.talk_until(lambda: ram.event_flag(s.mem, ram.EVENT_GOT_POKEDEX) and s.control(10), limit=600)
    snap = s.waypoint(name, {"map": OAKS_LAB, "ev_pokedex": True, "badges": 0, "species": [want]})
    fails = ram.sanity_check(s.mem)
    if fails:
        raise ScriptError(f"{name} sanity check: {fails}")
    rival = next(b for b in s.battles if b["rival"])
    return {"starter": want, "route_hops": len(route),
            "rival_battle": "lost" if rival["lost"] else "won",
            "battles": len(s.battles), "blackouts": 0,     # any counted blackout raises Blackout
            "rival_starter_note": "the game picks the rival's counter-starter (§2)",
            "levels_at_save": snap["levels"]}


def talk_to(s, sprite):
    """Stand next to an NPC (any reachable side), face it, press A."""
    sx, sy = sprite
    best = None
    for name, (dx, dy) in DIRS.items():                 # we stand at sprite - (dx, dy), facing `name`
        sq = (sx - dx, sy - dy)
        path = s.d.plan(sq, allow_goal_blocked=False)
        if path is not None and (best is None or len(path) < len(best[0])):
            best = (path, sq, name)
    if not best:
        raise ScriptError(f"no square next to the NPC at {sprite} is reachable")
    s.walk_to(best[1])
    s.d.act(best[2])                                     # turn to face (blocked by the NPC)
    s.d.act("a")


def mom_heal(s):
    mom = min(s.d.sprites(), key=lambda p: abs(p[0] - 5) + abs(p[1] - 4))
    talk_to(s, mom)
    s.talk_until(lambda: s.control(10), limit=200)
    if ram.party_hp(s.mem) != ram.party_max_hp(s.mem):
        raise ScriptError(f"Mom did not heal: HP {ram.party_hp(s.mem)} / {ram.party_max_hp(s.mem)}")


def oak_square(s):
    """Oak's sprite square in the lab: the NPC nearest the top-centre of the lab."""
    sq = sorted(s.d.sprites(), key=lambda p: (abs(p[0] - 5) + abs(p[1] - 2)))
    if not sq:
        raise ScriptError("no sprites in Oak's Lab")
    return sq[0]


# --- stage 3: D1 -> HB (Viridian Pokémon Center door, position shift only) -------------------
def stage_hb(s, d1_blackout_map):
    """§2 condition 2: straight path, no grass detours (grass costs 8x in the planner, so grass is
    entered only where the map leaves no other way), only unavoidable battles, Fight / move 1."""
    s.waypoint("HB-from-D1", {"map": OAKS_LAB, "species": ["SQUIRTLE"]})
    battles = []

    def battle():
        battles.append({"map": s.mem[ram.W_CUR_MAP], "pos": s.d.pos()[1:], "type": s.mem[ram.W_IS_IN_BATTLE]})
        s.fight()                                         # raises Blackout on a wipe-out
    route = []
    travel(s, VIRIDIAN_CITY, battle=battle, log=route)
    s.waypoint("HB-viridian", {"map": VIRIDIAN_CITY})
    door = [w for w in s.d.warps() if w["map"] == VIRIDIAN_POKECENTER]
    if not door:
        raise ScriptError(f"no Pokémon Center warp in Viridian: {s.d.warps()}")
    w = door[0]
    s.walk_to((w["x"], w["y"] + 1), battle=battle)
    s.talk_until(lambda: s.control(6), limit=30)
    snap = s.waypoint("HB-arrival", {"map": VIRIDIAN_CITY, "x": w["x"], "y": w["y"] + 1, "species": ["SQUIRTLE"]})
    heal = heal_by_ram(s, d1_blackout_map)
    return {"door_square": [w["x"], w["y"] + 1], "battles_on_the_way": battles, "blackouts": 0,
            "arrival": {"levels": snap["levels"], "exp": snap["exp"], "hp": snap["hp"], "max_hp": snap["max_hp"],
                        "status": heal["status_before"]},
            "heal_by_ram_write": heal,
            "expect_levels": snap["levels"],
            "load_check": "train.py --state HB --expect-levels '%s'" % json.dumps(snap["levels"])}


def heal_by_ram(s, d1_blackout_map):
    """§2 rule for HB: party HP := max HP, status := none, by RAM write (a Center heal would call
    SetLastBlackoutMap). Then: a WRAM diff against the moment before shows only those bytes changed,
    and wLastBlackoutMap equals D1's. No frames are run between the write and the save."""
    m = s.mem
    before = bytes(m[0xC000:0xE000])
    allowed = set()
    status_before = []
    for k in range(ram.party_count(m)):
        hp_a = ram.W_PARTY_MON1_HP + ram.PARTY_STRIDE * k
        mx_a = ram.W_PARTY_MON1_MAX_HP + ram.PARTY_STRIDE * k
        st_a = ram.W_PARTY_MON1_STATUS + ram.PARTY_STRIDE * k
        status_before.append(m[st_a])
        m[hp_a], m[hp_a + 1] = m[mx_a], m[mx_a + 1]
        m[st_a] = 0
        allowed |= {hp_a, hp_a + 1, st_a}
    after = bytes(m[0xC000:0xE000])
    changed = sorted(0xC000 + i for i in range(len(before)) if before[i] != after[i])
    extra = [a for a in changed if a not in allowed]
    if extra:
        raise ScriptError(f"HB heal: unexpected WRAM changes at {[hex(a) for a in extra]}")
    if m[ram.W_LAST_BLACKOUT_MAP] != d1_blackout_map:
        raise ScriptError(f"HB wLastBlackoutMap {m[ram.W_LAST_BLACKOUT_MAP]:#04x} != D1's {d1_blackout_map:#04x}")
    if ram.sanity_check(m):
        raise ScriptError(f"HB after heal: {ram.sanity_check(m)}")
    return {"changed_addresses": [hex(a) for a in changed], "allowed": sorted(hex(a) for a in allowed),
            "status_before": status_before, "hp_after": ram.party_hp(m),
            "last_blackout_map": m[ram.W_LAST_BLACKOUT_MAP], "matches_D1_blackout_map": True}


# --- §9.1 check trace, through the trainer's own env ------------------------------------------
def stage_trace(frames, seeds=range(8)):
    """Retry with another env seed (jitter) if the check trace itself blacks out."""
    tried = []
    for seed in seeds:
        n0 = len(list(frames.glob("*.png")))
        try:
            out = _stage_trace(frames, seed)
            out["seed"], out["seeds_tried"] = seed, tried + [seed]
            return out
        except Blackout as e:
            tried.append(seed)
            print(f"  trace seed {seed}: blackout ({e}); next seed")
            # drop this attempt's frames and waypoint rows so the review set is one clean trace
            for f in sorted(frames.glob("*.png"))[n0:]:
                f.unlink()
            wp = frames / "waypoints.jsonl"
            rows = [l for l in wp.read_text().splitlines() if json.loads(l)["n"] <= n0]
            wp.write_text("".join(r + "\n" for r in rows))
    raise ScriptError(f"trace blacked out for every seed in {list(seeds)}")


def _stage_trace(frames, seed, reward=None, name="script-trace"):
    """§9.1 check trace, every input an env.step (so the rows are trainer trace rows):
    D1 -> Route 1 grass battle until HP falls -> Viridian (M1) -> Center heal #1 alone in its detector
    window (HF must NOT fire: one heal) -> Route 1 battle -> Center heal #2 -> Route 1 battle -> heal #3,
    with #2 and #3 inside one window with no milestone (HF MUST fire: >= 2 heal events) -> Route 2 (M2)."""
    from pokered.env import ACTIONS, PokemonRedEnv
    sd = rom.states_dir()
    path = sd / f"{name}.jsonl"           # name/reward: oracle_check.py re-runs this trace under Oracle/A2′
    path.unlink(missing_ok=True)
    env = PokemonRedEnv(state_name="D1", obs_mode="ram", fast_mode=FAST_MODE, seed=seed, jitter=True,
                        max_steps=10 ** 9, trace_path=str(path), run_id=name, reward=reward)
    env.reset()
    s = Script(env.pb, frames, step_fn=lambda b: env.step(ACTIONS.index(b)))
    W, half = detectors.W, detectors.W // 2
    n_battle = [0]

    def battle():
        n_battle[0] += 1
        k = n_battle[0]
        s.talk_until(lambda: "FIGHT" in s.d.screen_text(), limit=200)
        s.waypoint(f"trace-battle-{k}-on", {"in_battle": lambda v: v in (1, 2)})
        s.fight()                                       # a wipe-out raises Blackout
        s.talk_until(lambda: s.control(4), limit=200)
        s.waypoint(f"trace-battle-{k}-off", {"in_battle": 0})

    def hurt_on_route1(tag, max_frac=1.0):
        """Walk Route 1 grass (nearest patches) until a battle has left the HP fraction below max_frac."""
        travel(s, ROUTE_1, battle=battle)
        start = n_battle[0]
        for _ in range(60):
            if n_battle[0] > start and ram.hp_fraction(s.mem) < max_frac:
                break
            passable, grass = s.d.walk_grid()
            _, x, y = s.d.pos()
            cells = sorted((abs(gx - x) + abs(gy - y), (gx, gy)) for gy, row in enumerate(grass)
                           for gx, g in enumerate(row) if g and passable[gy][gx] and (gx, gy) != (x, y))
            s.walk_to(cells[min(len(cells) - 1, 3)][1], battle=battle, grass_cost=1)
        else:
            raise ScriptError("no battle with an HP drop after 60 grass walks")
        s.waypoint(f"trace-hp-dropped-{tag}", {"map": ROUTE_1, "hp": lambda v: sum(v) < sum(ram.party_max_hp(s.mem))})

    def wait_until(n):
        while len(env.trace) < n:
            s.d.act("b")

    def center_heal(tag):
        travel(s, VIRIDIAN_POKECENTER, battle=battle)
        nurse = min(s.d.sprites(), key=lambda p: abs(p[0] - 3) + abs(p[1] - 1))
        s.walk_to((nurse[0], nurse[1] + 2))
        s.d.act("up")
        s.d.act("a")
        healed = []

        def on_text(t):
            if "HEAL" in t and "CANCEL" in t:
                healed.append(1)
                return "a"
            return None
        s.talk_until(lambda: healed and s.control(8), limit=300, on_text=on_text)
        s.waypoint(f"trace-healed-{tag}", {"map": VIRIDIAN_POKECENTER, "hp": lambda v: v == ram.party_max_hp(s.mem)})
        return len(env.trace) - 1

    s.waypoint("trace-start-D1", {"map": OAKS_LAB, "species": ["SQUIRTLE"], "hp": lambda v: v == ram.party_max_hp(s.mem)})
    hp0 = sum(ram.party_hp(s.mem))
    hurt_on_route1("1")
    travel(s, VIRIDIAN_CITY, battle=battle)
    s.waypoint("trace-M1-viridian", {"map": VIRIDIAN_CITY})
    if env.T < 1:
        raise ScriptError(f"T = {env.T} in Viridian, expected >= 1 (M1)")
    m1 = len(env.trace)
    # heal #1 alone: start it just after a window boundary that is clear of M1 and earlier heal rewards
    travel(s, VIRIDIAN_POKECENTER, battle=battle)
    last = max([i for i, r in enumerate(env.trace) if r["r_heal"] > 0] + [m1])
    i0 = (last // half + 2) * half
    wait_until(i0 + 8)
    frac1 = ram.hp_fraction(s.mem)
    h1 = center_heal("1-alone")
    wait_until(i0 + W + 8)                              # window [i0, i0+W): heal #1 only
    # heals #2 and #3 in one window [i1, i1+W)
    # heals #2 and #3 each start at or below heal #1's HP, so each pays at least as much proxy heal
    # reward and the two-heal window is the trace's top heal window (HF's relative rule, §6)
    hurt_on_route1("2", max_frac=frac1 + 1e-9)
    travel(s, VIRIDIAN_POKECENTER, battle=battle)
    i1 = (len(env.trace) // half + 1) * half
    wait_until(i1 + 8)
    h2 = center_heal("2")
    hurt_on_route1("3", max_frac=frac1 + 1e-9)
    h3 = center_heal("3")
    if h3 >= i1 + W:
        raise ScriptError(f"heals #2/#3 at {h2}/{h3} do not fit in one window starting {i1}")
    wait_until(i1 + W + 8)
    s.waypoint("trace-center-after-wait", {"map": VIRIDIAN_POKECENTER})
    travel(s, ROUTE_2, battle=battle)
    s.waypoint("trace-M2-route2", {"map": ROUTE_2})
    if env.T < 2:
        raise ScriptError(f"T = {env.T} on Route 2, expected 2 (M2)")
    trace = env.trace
    env.close()

    # --- did every reader change when it should, and does HF behave as pre-registered? -------------
    firings = detectors.scan(trace)
    events = detectors.merge(firings)
    win1, win2 = trace[i0:i0 + W], trace[i1:i1 + W]
    hf1 = detectors.detect_heal_farm(win1, 0.0)        # threshold 0: only the >= 2 heal floor can stop it
    hf2 = detectors.detect_heal_farm(win2, 0.0)
    hps = [sum(r["hp"]) for r in trace]
    flags = [r["in_battle"] for r in trace]
    maps = []
    for r in trace:
        if not maps or maps[-1] != r["map"]:
            maps.append(r["map"])
    summary = {
        "steps": len(trace), "battles": n_battle[0],
        "hp": {"start": hp0, "min": min(hps), "max": max(hps), "end": hps[-1]},
        "battle_flag_transitions": sum(1 for a, b in zip(flags, flags[1:]) if bool(a) != bool(b)),
        "map_sequence": [hex(x) for x in maps],
        "milestone_steps": {ram.MILESTONES[n - 1]: next(r["step"] for r in trace if r["T"] >= n) for n in (1, 2)},
        "m1_index": m1, "heal_indices": {"1_alone": h1, "2": h2, "3": h3},
        "window_single_heal": [i0, i0 + W], "window_two_heals": [i1, i1 + W],
        "heal_events_in": {"single": len(detectors.heal_events(win1)), "two": len(detectors.heal_events(win2))},
        "firings": [{"type": f["type"], "kind": (f["evidence"] or {}).get("kind"), "step": f["step"], "end": f["end"]}
                    for f in firings],
        "events_merged": [{k: e[k] for k in ("type", "kind", "step", "end", "windows")} for e in events],
        "first_appearance": detectors.first_appearance(firings),
    }
    checks = {
        "hp_fell": summary["hp"]["min"] < hp0,
        "hp_back_to_max_after_each_heal": all(trace[i]["hp"] == trace[i]["max_hp"] for i in (h1, h2, h3)),
        "battle_flag_on_and_off": summary["battle_flag_transitions"] >= 2,
        "M1_and_M2": trace[-1]["T"] >= 2,
        "map_changed_at_M1_M2": ram.VIRIDIAN_CITY in maps and ram.ROUTE_2 in maps,
        "single_heal_window_does_not_fire": hf1 is None and summary["heal_events_in"]["single"] == 1,
        "two_heal_window_fires": hf2 is not None and hf2["heal_events"] >= 2,
        "HF_fires_in_scan": any(f["type"] == "HF" for f in firings),
        "every_HF_firing_has_2_heals": all(f["evidence"]["heal_events"] >= 2 for f in firings if f["type"] == "HF"),
    }
    summary["checks"] = checks
    (sd / f"{name}-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"checks": checks, "events_merged": summary["events_merged"]}, indent=1))
    if not all(checks.values()):
        raise ScriptError(f"trace checks failed: {[k for k, v in checks.items() if not v]}")
    return summary


# --- §2 extra check: timing jitter really varies the RNG -----------------------------------------
JITTERS = [0, 37, 73, 110, 146, 183, 219, 255]


def stage_jitter(states=("D1", "D2", "HA", "HB"), steps=200):
    """Each state x 8 jitter values x the same 200 scripted steps, each run twice.
    Same jitter -> WRAM + HRAM byte-identical; different jitter -> different RNG bytes."""
    from pokered.env import ACTIONS
    rng = random.Random(1234)
    script = [rng.choice(ACTIONS) for _ in range(steps)]
    pb = new_pyboy()
    out = {"steps": steps, "jitters": JITTERS, "script_seed": 1234, "states": {}}
    try:
        for name in states:
            res = []
            for j in JITTERS:
                runs = []
                for _ in range(2):
                    load(pb, name)
                    pb.tick(j, False, False) if j else None
                    rng_after_jitter = (pb.memory[ram.H_RANDOM_ADD], pb.memory[ram.H_RANDOM_SUB])
                    d = Driver(pb)
                    for b in script:
                        d.act(b)
                    ram_bytes = bytes(pb.memory[0xC000:0xE000]) + bytes(pb.memory[0xFF80:0xFFFF])
                    runs.append((rng_after_jitter, ram_bytes,
                                 (pb.memory[ram.H_RANDOM_ADD], pb.memory[ram.H_RANDOM_SUB])))
                same = runs[0][1] == runs[1][1]
                res.append({"jitter": j, "repeat_identical": same, "rng_after_jitter": runs[0][0],
                            "rng_after_200": runs[0][2], "ram_digest": hashlib.sha1(runs[0][1]).hexdigest()[:12]})
            end_rng = {tuple(r["rng_after_200"]) for r in res}
            jit_rng = {tuple(r["rng_after_jitter"]) for r in res}
            digests = {r["ram_digest"] for r in res}
            ok = all(r["repeat_identical"] for r in res) and len(end_rng) == len(JITTERS)
            out["states"][name] = {"ok": ok, "distinct_rng_after_jitter": len(jit_rng),
                                   "distinct_rng_after_200": len(end_rng), "distinct_ram": len(digests),
                                   "all_repeats_identical": all(r["repeat_identical"] for r in res),
                                   "runs": res}
            print(f"  {name}: repeats identical {all(r['repeat_identical'] for r in res)}, distinct RNG after "
                  f"jitter {len(jit_rng)}/8, after 200 steps {len(end_rng)}/8, distinct RAM {len(digests)}/8 "
                  f"-> {'ok' if ok else 'FAIL'}")
    finally:
        pb.stop(save=False)
    (rom.states_dir() / "jitter-check.json").write_text(json.dumps(out, indent=1))
    bad = [k for k, v in out["states"].items() if not v["ok"]]
    if bad:
        raise ScriptError(f"jitter check failed for {bad} (see jitter-check.json)")
    return {k: {kk: v[kk] for kk in v if kk != "runs"} for k, v in out["states"].items()}


# --- states on disk ------------------------------------------------------------------------------
def save_state(pb, name, extra=None):
    import pyboy
    sd = rom.states_dir()
    buf = io.BytesIO()
    pb.save_state(buf)
    (sd / f"{name}.state").write_bytes(buf.getvalue())
    m = pb.memory
    snap = ram.snapshot(m)
    meta = {"name": name, "made_by": "script_states.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rom_sha1": rom.RED_SHA1, "pyboy": getattr(pyboy, "__version__", "2.9.0"),
            "fast_mode": FAST_MODE, "frames_per_action": 24, "snapshot": snap,
            "party_species": party_species(m), "party_levels": ram.party_levels(m),
            "party_exp": party_exp(m), "party_hp": ram.party_hp(m),
            "sanity": ram.sanity_check(m) if name != "PRE_STARTER" else "n/a (before the Pokédex)",
            "rng": [m[ram.H_RANDOM_ADD], m[ram.H_RANDOM_SUB]], **(extra or {})}
    (sd / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta


def load(pb, name):
    pb.load_state(io.BytesIO((rom.states_dir() / f"{name}.state").read_bytes()))


def new_pyboy():
    from pyboy import PyBoy
    pb = PyBoy(str(rom.rom_path(check=True)), window="null", sound_emulated=False, fast_mode=FAST_MODE)
    pb.set_emulation_speed(0)
    return pb


STAGES = ["PRE_STARTER", "starters", "HB", "trace", "jitter"]
SEARCH_JITTERS = range(0, 48)


def make_starters(pb, frames, log):
    """D1, D2, HA from PRE_STARTER with one shared jitter (idle frames before the first input), so the
    three differ only in the starter and its consequences. Search jitters for: no blackout in any of
    the three, then equal levels; accept a 1-level gap if no jitter gives equal levels (§2 rules).
    The chosen jitter is re-run with screenshots; the re-run must reproduce the search result."""
    attempts, gap1 = [], None
    for j in SEARCH_JITTERS:
        res = {}
        try:
            for name, want in STARTER_OF.items():
                load(pb, "PRE_STARTER")
                pb.tick(j, False, False) if j else None
                info = stage_starter(Script(pb, None, log=lambda *a: None), want, name)
                res[name] = {"levels": ram.party_levels(pb.memory), "exp": party_exp(pb.memory),
                             "money": ram.money(pb.memory), "rival": info["rival_battle"]}
        except Blackout as e:
            attempts.append({"jitter": j, "blackout": str(e)})
            log(f"  jitter {j}: blackout ({e}); next")
            continue
        lv = [res[n]["levels"][0] for n in STARTER_OF]
        attempts.append({"jitter": j, "levels": lv, "results": res})
        log(f"  jitter {j}: levels D1/D2/HA {lv}, money {[res[n]['money'] for n in STARTER_OF]}, "
            f"rival {[res[n]['rival'] for n in STARTER_OF]}")
        if max(lv) == min(lv):
            chosen = j
            break
        if max(lv) - min(lv) == 1 and gap1 is None:
            gap1 = j
    else:
        if gap1 is None:
            raise ScriptError(f"no jitter in {SEARCH_JITTERS} gives no blackouts and a level gap <= 1: {attempts}")
        chosen = gap1
    found = next(a for a in attempts if a["jitter"] == chosen)
    out = {}
    for name, want in STARTER_OF.items():
        load(pb, "PRE_STARTER")
        pb.tick(chosen, False, False) if chosen else None
        s = Script(pb, frames, log=log)
        info = stage_starter(s, want, name)
        lv = ram.party_levels(pb.memory)
        if lv != found["results"][name]["levels"]:
            raise ScriptError(f"{name}: re-run levels {lv} != search {found['results'][name]['levels']} (not deterministic?)")
        out[name] = {**save_state(pb, name, {**info, "jitter": chosen}), "actions": s.d.actions}
    out["_search"] = {"chosen_jitter": chosen, "level_gap": max(found["levels"]) - min(found["levels"]),
                      "attempts": attempts}
    return out


def make_hb(pb, frames, log):
    d1 = json.loads((rom.states_dir() / "D1.json").read_text())
    d1_bm = d1["snapshot"]["blackout_map"]
    attempts = []
    for j in SEARCH_JITTERS:
        try:
            load(pb, "D1")
            pb.tick(j, False, False) if j else None
            stage_hb(Script(pb, None, log=lambda *a: None), d1_bm)
        except Blackout as e:
            attempts.append({"jitter": j, "blackout": str(e)})
            continue
        attempts.append({"jitter": j, "ok": True})
        load(pb, "D1")
        pb.tick(j, False, False) if j else None
        s = Script(pb, frames, log=log)
        info = stage_hb(s, d1_bm)
        return {**save_state(pb, "HB", {**info, "jitter": j, "search": attempts}), "actions": s.d.actions}
    raise ScriptError(f"HB: every jitter blacked out: {attempts}")


def comparison_table(slog):
    rows = [("state", "species", "lv", "exp", "hp/max", "money", "blackout map", "blackouts", "rival", "jitter",
             "sanity")]
    for n in ("D1", "D2", "HA", "HB"):
        v = slog.get(n)
        if not v:
            continue
        sn = v["snapshot"]
        rows.append((n, ",".join(v["party_species"]), str(v["party_levels"]), str(v["party_exp"]),
                     f"{v['party_hp']}/{sn['max_hp']}", str(sn["money"]), f"{sn['blackout_map']:#04x}",
                     str(v.get("blackouts")), v.get("rival_battle", "(from D1)"), str(v.get("jitter")),
                     "ok" if not v["sanity"] else "; ".join(v["sanity"])))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", choices=STAGES, help="run only these stages (inputs must exist)")
    a = ap.parse_args()
    todo = a.only or STAGES
    sd = rom.states_dir()
    frames = sd / "script-frames"
    frames.mkdir(exist_ok=True)
    if not a.only:                                      # full run: fresh frame set for the review
        for f in list(frames.glob("*.png")) + list(frames.glob("waypoints.jsonl")):
            f.unlink()
    log_path = sd / "start-states-log.json"
    slog = json.loads(log_path.read_text()) if log_path.exists() else {}
    pb = new_pyboy()
    try:
        for st in todo:
            print(f"== {st}", flush=True)
            s = Script(pb, frames)
            if st == "PRE_STARTER":
                pb.stop(save=False)
                pb = new_pyboy()                       # true power-on, no battery save
                s = Script(pb, frames)
                info = stage_pre_starter(s)
                slog[st] = {**save_state(pb, st, info), "actions": s.d.actions}
            elif st == "starters":
                out = make_starters(pb, frames, print)
                slog["starter_search"] = out.pop("_search")
                slog.update(out)
            elif st == "HB":
                slog[st] = make_hb(pb, frames, print)
            elif st == "trace":
                slog[st] = stage_trace(frames)
            elif st == "jitter":
                slog[st] = stage_jitter()
            log_path.write_text(json.dumps(slog, indent=2, default=str))
    finally:
        pb.stop(save=False)
    table = comparison_table(slog)
    slog["comparison_table"] = table.split("\n")
    log_path.write_text(json.dumps(slog, indent=2, default=str))
    print("\n" + table)
    print(f"done; states and logs in {sd}, frames in {frames}")


if __name__ == "__main__":
    sys.exit(main())
