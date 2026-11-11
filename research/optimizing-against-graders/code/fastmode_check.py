#!/usr/bin/env python3
"""§5b fast_mode acceptance on D1 (experiment-e1.md §5b).

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python fastmode_check.py [--steps 10000] [--jitter 0]

Two PyBoy instances in lockstep, normal core (A) and fast_mode (B), both loaded from D1 with the
same jitter. The script steers A (script_states driver); every input goes to both. After every
step (24 frames, as in the env) WRAM 0xC000-0xDFFF and HRAM 0xFF80-0xFFFE are compared.

Route (criterion 1): D1 -> Route 1 grass battles -> Viridian -> Pokémon Center heal -> Route 2 ->
Viridian Forest -> talk to forest NPCs until a trainer battle starts -> fight on in forest grass
until the party blacks out (if it does within the budget) -> then seeded random inputs up to --steps.
Verdict per §5b: PASS if byte-identical throughout; any WRAM/HRAM divergence is a FAIL (sound and
serial registers are outside the compared ranges). Writes ~/roms/states/fastmode-check.json.
"""
import argparse
import json
import random
import sys
import time

import script_states as S
from pokered import ram, rom
from pokered.driver import FRAMES_PER_ACTION, PRESS_FRAMES

ROUTE_2_GATE, VIRIDIAN_FOREST_SOUTH_GATE, VIRIDIAN_FOREST = 0x31, 0x32, 0x33   # pret map_constants.asm
WRAM, HRAM = (0xC000, 0xE000), (0xFF80, 0xFFFF)
AUDIO_WRAM = (0xC000, 0xC0F3)   # wUnusedMusicByte .. wTempoModifier (pret pokered.sym): the audio engine's state
REGIONS = {
    "audio_wram": lambda x: AUDIO_WRAM[0] <= x < AUDIO_WRAM[1],
    "other_wram": lambda x: AUDIO_WRAM[1] <= x < WRAM[1],
    "rng_bytes": lambda x: x in (0xFFD3, 0xFFD4),
    "other_hram": lambda x: HRAM[0] <= x < HRAM[1] and x not in (0xFFD3, 0xFFD4),
}


def new(fast):
    from pyboy import PyBoy
    pb = PyBoy(str(rom.rom_path(check=True)), window="null", sound_emulated=False, fast_mode=fast)
    pb.set_emulation_speed(0)
    return pb


class Lockstep:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.steps = 0
        self.first = None
        self.diff_steps = 0
        self.addrs = set()
        self.first_by_region = {}   # region -> (step, addresses at that step)
        self.trace = []             # per step: map, battle, phase (for locating divergences)
        self.events = {}            # what the trajectory covered, first step of each
        self.phase = "start"
        self.prev = {}

    @staticmethod
    def text_of(pb):
        from pokered.driver import CHARS
        m = pb.memory
        rows = ("".join(CHARS.get(m[ram.W_TILE_MAP + r * 20 + c], " ") for c in range(20)) for r in range(ram.TILE_ROWS))
        return " ".join(" ".join(r.split()) for r in rows).strip()[:160]

    def mem(self, pb):
        m = pb.memory
        return bytes(m[WRAM[0]:WRAM[1]]) + bytes(m[HRAM[0]:HRAM[1]])

    def act(self, button):
        for pb in (self.a, self.b):
            pb.button(button, PRESS_FRAMES)
            pb.tick(FRAMES_PER_ACTION, False, False)
        self.steps += 1
        ma, mb = self.mem(self.a), self.mem(self.b)
        m = self.a.memory
        self.trace.append({"step": self.steps, "button": button, "pos": list(ram.position(m)),
                           "battle": m[ram.W_IS_IN_BATTLE], "phase": self.phase,
                           "text": self.text_of(self.a)})
        if ma != mb:
            self.diff_steps += 1
            n = WRAM[1] - WRAM[0]
            now = [WRAM[0] + i if i < n else HRAM[0] + i - n for i in range(len(ma)) if ma[i] != mb[i]]
            self.addrs.update(now)
            if self.first is None:
                self.first = self.steps
            for reg, ok in REGIONS.items():
                hit = [x for x in now if ok(x)]
                if hit and reg not in self.first_by_region:
                    self.first_by_region[reg] = {"step": self.steps, "addresses": [hex(x) for x in hit[:24]],
                                                 "map": hex(m[ram.W_CUR_MAP]), "battle": m[ram.W_IS_IN_BATTLE],
                                                 "phase": self.phase}
        self.note()

    def note(self):
        m = self.a.memory
        cur = {"map": m[ram.W_CUR_MAP], "battle": m[ram.W_IS_IN_BATTLE], "hp": sum(ram.party_hp(m)),
               "rng": (m[ram.H_RANDOM_ADD], m[ram.H_RANDOM_SUB])}
        p = self.prev
        if p:
            if cur["battle"] == 1 and p["battle"] != 1:
                self.events.setdefault("wild_battle", self.steps)
                self.events["wild_battles"] = self.events.get("wild_battles", 0) + 1
            if cur["battle"] == 2 and p["battle"] != 2:
                self.events.setdefault("trainer_battle", self.steps)
                self.events["trainer_battle_map"] = hex(cur["map"])
            if cur["battle"] == 0xFF and p["battle"] != 0xFF:
                self.events.setdefault("blackout", self.steps)
            if cur["map"] != p["map"]:
                self.events.setdefault(f"map_{cur['map']:#04x}", self.steps)
            if cur["rng"] != p["rng"]:
                self.events["rng_changes"] = self.events.get("rng_changes", 0) + 1
        self.prev = cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--repro-first", action="store_true",
                    help="old warp choice (stuck at the forest gate, then random inputs): replays the first run")
    ap.add_argument("--out", default="fastmode-check.json")
    a = ap.parse_args()
    if a.repro_first:
        S.Script.warp_fallback = False
    A, B = new(False), new(True)
    for pb in (A, B):
        S.load(pb, "D1")
        if a.jitter:
            pb.tick(a.jitter, False, False)
    ls = Lockstep(A, B)
    s = S.Script(A, None, log=lambda *x: None, step_fn=ls.act)
    t0 = time.time()
    phase = "start"

    def set_phase(p):
        nonlocal phase
        phase = p
        ls.phase = p
    try:
        def fight_allow_loss():
            s.fight(rival=True)                 # rival=True: a loss is recorded, not raised
        set_phase("viridian")
        S.travel(s, S.VIRIDIAN_CITY, battle=fight_allow_loss)
        set_phase("center-heal")
        S.travel(s, S.VIRIDIAN_POKECENTER, battle=fight_allow_loss)
        nurse = min(s.d.sprites(), key=lambda p: abs(p[0] - 3) + abs(p[1] - 1))
        s.walk_to((nurse[0], nurse[1] + 2))
        s.d.act("up")
        s.d.act("a")
        healed = []
        s.talk_until(lambda: healed and s.control(8), limit=300,
                     on_text=lambda t: (healed.append(1) or "a") if "HEAL" in t and "CANCEL" in t else None)
        ls.events["center_heal"] = ls.steps
        set_phase("forest")
        S.EDGES[S.ROUTE_2] = {S.VIRIDIAN_CITY: ("edge", "down"), ROUTE_2_GATE: ("warp", "up"),
                              VIRIDIAN_FOREST_SOUTH_GATE: ("warp", "up")}
        S.EDGES[VIRIDIAN_FOREST_SOUTH_GATE] = {VIRIDIAN_FOREST: ("warp", "up"), S.ROUTE_2: ("out", "down")}
        S.EDGES[VIRIDIAN_FOREST] = {VIRIDIAN_FOREST_SOUTH_GATE: ("warp", "down")}
        S.travel(s, VIRIDIAN_FOREST, battle=fight_allow_loss)
        set_phase("trainer")
        tried = set()
        while "trainer_battle" not in ls.events and len(tried) < 12 and ls.steps < a.steps:
            if A.memory[ram.W_CUR_MAP] != VIRIDIAN_FOREST:
                S.travel(s, VIRIDIAN_FOREST, battle=fight_allow_loss)
            _, x, y = s.d.pos()
            cands = [p for p in s.d.sprites() if p not in tried]
            if not cands:
                break
            npc = min(cands, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
            tried.add(npc)
            try:
                S.talk_to(s, npc)
                s.talk_until(lambda: A.memory[ram.W_IS_IN_BATTLE] or s.control(8), limit=200)
                if A.memory[ram.W_IS_IN_BATTLE]:
                    fight_allow_loss()
                    s.talk_until(lambda: s.control(8), limit=300)
            except S.ScriptError:
                continue                         # unreachable NPC: try the next one
        set_phase("grass-until-blackout")
        while "blackout" not in ls.events and ls.steps < a.steps - 500:
            if A.memory[ram.W_CUR_MAP] != VIRIDIAN_FOREST:
                break                            # blacked out (warped to the Center) or wandered off
            passable, grass = s.d.walk_grid()
            _, x, y = s.d.pos()
            cells = sorted((abs(gx - x) + abs(gy - y), (gx, gy)) for gy, row in enumerate(grass)
                           for gx, g in enumerate(row) if g and passable[gy][gx] and (gx, gy) != (x, y))
            if not cells:
                break
            s.walk_to(cells[min(len(cells) - 1, 5)][1], battle=fight_allow_loss, grass_cost=1, limit=200)
    except S.ScriptError as e:
        ls.events["script_stopped"] = f"{phase}: {e}"
    set_phase("random")
    rng = random.Random(5)
    from pokered.env import ACTIONS
    while ls.steps < a.steps:
        ls.act(rng.choice(ACTIONS))
    dt = time.time() - t0
    sound_serial = [x for x in ls.addrs if 0xFF01 <= x <= 0xFF02 or 0xFF10 <= x <= 0xFF3F]
    other = sorted(x for x in ls.addrs if x not in sound_serial)
    verdict = ("PASS (byte-identical)" if ls.first is None else
               "PASS (sound/serial only)" if not other else "FAIL")
    by_region = {r: [x for x in ls.addrs if ok(x)] for r, ok in REGIONS.items()}
    out = {"verdict": verdict, "steps": ls.steps,
           "first_divergence_by_region": ls.first_by_region,
           "divergent_address_count_by_region": {r: len(v) for r, v in by_region.items()}, "jitter": a.jitter, "first_divergence_step": ls.first,
           "divergent_steps": ls.diff_steps, "divergent_addresses_non_audio": [hex(x) for x in sorted(other) if not REGIONS["audio_wram"](x)][:64],
           "coverage": ls.events,
           "steps_before_first_divergence": [r for r in ls.trace if ls.first and ls.first - 40 <= r["step"] <= ls.first],
           "screen_B_at_first_divergence": None, "seconds": round(dt, 1), "compared": "WRAM 0xC000-0xDFFF + HRAM 0xFF80-0xFFFE, every step",
           "frames_per_action": FRAMES_PER_ACTION, "rom_sha1": rom.RED_SHA1,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (rom.states_dir() / a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    for pb in (A, B):
        pb.stop(save=False)
    return 0 if verdict.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
