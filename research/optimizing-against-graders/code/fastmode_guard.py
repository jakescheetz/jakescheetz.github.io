#!/usr/bin/env python3
"""§5b 'Ruling on the exemption': the four acceptance tests for fast_mode with a guard.

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python fastmode_guard.py --state HB --seed 1 [--steps 12000]

Normal core (A) and fast_mode (B) in lockstep from D1 or HB, same jitter. Inputs are biased toward the
Viridian Pokémon Center: the run repeatedly walks there and picks an activity at random:
  attendant  talk to the Cable Club receptionist (11,2), then mash A (accept), mash B (decline) or random
  pc         use the Center PC (hidden event 13,3, facing up)
  trade      Route 2 trade house: talk to the in-game trade NPC (4,1), random answers
  nurse      talk to the nurse, random answers
  wander     random env inputs in the Center
each followed by 10-60 random env inputs. WRAM + HRAM are compared after every step (24 frames).

Serial-path signal (verified against pret pokered.sym, symbols branch): PyBoy execution hooks on
CableClubNPC (01:71C5) and its labels, on the Serial interrupt handler (00:2125) and every Serial_*
routine, plus PC/trade routines for test 4. Hooks are registered identically on both cores.

On each divergence (tests 1-3): record the lag from the last CableClubNPC entry and whether any serial-path
hook fired since the last sync; then drive EACH core on its own with the same dialogue policy until it has
control again (no text, input not ignored, 6 checks in a row; 3,000-step cap = soft-lock), recording the
steps it took and the code labels it hit (= the text path); then compare game state (map, x, y, facing,
party, bag, money, badges, event flags, wLastBlackoutMap; plus a full WRAM diff with RNG, frame/timer
counters, audio-engine WRAM and sprite state excluded); then copy A's state into B and continue.
Output: ~/roms/states/fastmode-guard/<state>-s<seed>.json, screenshots in ~/roms/states/fastmode-frames/.
"""
import argparse
import faulthandler
import io
import signal
import json
import random
import sys
import time

import script_states as S
from pokered import ram, rom
from pokered.driver import CHARS, FRAMES_PER_ACTION, PRESS_FRAMES, W_JOY_IGNORE
from pokered.env import ACTIONS

WRAM, HRAM = (0xC000, 0xE000), (0xFF80, 0xFFFF)
ROUTE_2_TRADE_HOUSE = 0x30
HOOKS = {   # label: (bank, address) from pret pokered.sym
    "CableClubNPC": (0x01, 0x71C5),
    "CableClubNPC.establishConnectionLoop": (0x01, 0x71EB),
    "CableClubNPC.establishedConnection": (0x01, 0x721A),
    "CableClubNPC.failedToEstablishConnection": (0x01, 0x7287),
    "CableClubNPC.choseNo": (0x01, 0x728F),
    "CableClubNPC.didNotConnect": (0x01, 0x7298),
    "CableClubNPC.connected": (0x01, 0x72A8),
    "LinkMenu": (0x01, 0x5C0A),
    "Serial": (0x00, 0x2125),
    "Serial_ExchangeBytes": (0x00, 0x216F),
    "Serial_ExchangeByte": (0x00, 0x219A),
    "SerialDisconnected": (0x00, 0x223F),
    "Serial_ExchangeLinkMenuSelection": (0x00, 0x2247),
    "Serial_PrintWaitingTextAndSyncAndExchangeNybble": (0x00, 0x226E),
    "Serial_SyncAndExchangeNybble": (0x00, 0x227F),
    "Serial_ExchangeNybble": (0x00, 0x22C3),
    "Serial_SendZeroByte": (0x00, 0x22ED),
    "Serial_TryEstablishingExternallyClockedConnection": (0x00, 0x22FA),
    "OpenPokemonCenterPC": (0x18, 0x6516),
    "ActivatePC": (0x05, 0x7E2C),
    "DoInGameTradeDialogue": (0x1C, 0x5AD9),
    "InGameTrade_DoTrade": (0x1C, 0x5C07),
}
# Every Pokémon Center's map script calls Serial_TryEstablishingExternallyClockedConnection once per frame
# (pret scripts/*Pokecenter.asm): an external-clock listen, no transfer without a cable, identical on both
# cores. It is counted separately, not as a serial-path event.
CENTER_LISTEN = "Serial_TryEstablishingExternallyClockedConnection"
SERIAL_PATH = {k for k in HOOKS if (k.startswith("Serial") or k.startswith("CableClubNPC.") or k == "LinkMenu")
               and k != CENTER_LISTEN}
SERIAL_HRAM = (0xFFA9, 0xFFAE)   # hSerialReceivedNewData .. hSerialReceiveData (pret pokered.sym): serial state
# Excluded from the test-2 full diff: RNG, frame/timer counters, audio engine, sprite/OAM state.
EXCLUDE = [(0xC000, 0xC0F3), (0xC100, 0xC300), (0xC300, 0xC3A0), (0xFFD3, 0xFFD5), (0xFFD5, 0xFFD6),
           (0xDEE3, 0xE000),      # the stack (wStack grows down from 0xDFFF): residue of the serial interrupt
           (0xD5CD, 0xD5CE),      # wSavedSpriteImageIndex: sprite state
           (0xFF92, 0xFF93)]      # hSpriteScreenY: sprite state
EXCLUDE_NAMES = {"audio_wram": (0xC000, 0xC0F3), "stack": (0xDEE3, 0xE000), "sprite_state": None,
                 "rng": (0xFFD3, 0xFFD5), "frame_counter": (0xFFD5, 0xFFD6)}
FIELDS = {  # test 2 game state
    "map": (0xD35E, 0xD35F), "y_x": (0xD361, 0xD363), "facing": (0xC109, 0xC10A),
    "party": (0xD163, 0xD2F7), "bag": (0xD31D, 0xD347), "money": (0xD347, 0xD34A), "badges": (0xD356, 0xD357),
    "event_flags": (0xD747, 0xD887), "last_blackout_map": (0xD719, 0xD71A),
}


def new(fast):
    from pyboy import PyBoy
    pb = PyBoy(str(rom.rom_path(check=True)), window="null", sound_emulated=False, fast_mode=fast)
    pb.set_emulation_speed(0)
    return pb


def mem(pb):
    m = pb.memory
    return bytes(m[WRAM[0]:WRAM[1]]) + bytes(m[HRAM[0]:HRAM[1]])


def addr_of(i):
    n = WRAM[1] - WRAM[0]
    return WRAM[0] + i if i < n else HRAM[0] + i - n


def text(pb):
    m = pb.memory
    rows = ("".join(CHARS.get(m[ram.W_TILE_MAP + r * 20 + c], " ") for c in range(20)) for r in range(ram.TILE_ROWS))
    return " ".join(" ".join(r.split()) for r in rows).strip()[:160]


class Core:
    def __init__(self, name, fast, ls):
        self.name, self.pb, self.ls = name, new(fast), ls
        self.events = []                                  # (step, label)
        for label, (bank, addr) in HOOKS.items():
            self.pb.hook_register(bank, addr, self.hit, label)

    def hit(self, label):
        self.events.append((self.ls.steps + 1, label))

    def press(self, button):
        self.pb.button(button, PRESS_FRAMES)
        self.pb.tick(FRAMES_PER_ACTION, False, False)

    def idle(self):
        self.pb.tick(FRAMES_PER_ACTION, False, False)

    def has_control(self):
        return not text(self.pb) and not self.pb.memory[W_JOY_IGNORE]


class Lockstep:
    def __init__(self, args, rng):
        self.steps = 0
        self.rng = rng
        self.A = Core("A-normal", False, self)
        self.B = Core("B-fast", True, self)
        self.policy = "b"
        self.last_sync = 0
        self.resolve_steps = 0
        self.next_report = 500
        self.limit = args.steps
        self.divergences = []
        self.frames = rom.states_dir() / "fastmode-frames"
        self.frames.mkdir(exist_ok=True)
        self.tag = f"{args.state}-s{args.seed}"

    def act(self, button):
        self.A.press(button)
        self.B.press(button)
        self.steps += 1
        if mem(self.A.pb) != mem(self.B.pb):
            self.resolve()
        if self.steps >= self.next_report:
            self.next_report += 500
            print(f"[{self.tag}] step {self.steps} pos {ram.position(self.A.pb.memory)} divergences "
                  f"{len(self.divergences)} resolve-steps {self.resolve_steps} policy {self.policy}", flush=True)
        if self.steps > self.limit + 2000:
            raise S.ScriptError("step budget exceeded inside an activity")

    def policy_button(self, i):
        if self.policy == "random":
            return ACTIONS[(i * 3 + 1) % len(ACTIONS)]       # cycles all 7 buttons; same sequence on both cores
        return self.policy

    def resolve(self):
        n = self.steps
        ma, mb = mem(self.A.pb), mem(self.B.pb)
        diff = [addr_of(i) for i in range(len(ma)) if ma[i] != mb[i]]
        since = lambda c: [(s, l) for s, l in c.events if s > self.last_sync]
        entries = [s for s, l in self.A.events if l == "CableClubNPC"]
        serial_since = sorted({l for s, l in since(self.A) + since(self.B) if l in SERIAL_PATH})
        # the current visit: first connection-loop pass after the latest CableClubNPC entry
        loop_steps = [s for s, l in self.A.events
                      if l == "CableClubNPC.establishConnectionLoop" and entries and s >= entries[-1]]
        ev = {"step": n, "pos": list(ram.position(self.A.pb.memory)), "policy": self.policy,
              "first_addresses": [hex(x) for x in diff[:24]], "n_addresses": len(diff),
              "rng_bytes_differ": any(x in (0xFFD3, 0xFFD4) for x in diff),
              "lag_from_CableClubNPC_entry": (n - entries[-1]) if entries else None,
              "lag_from_serial_loop_start": (n - min(loop_steps)) if loop_steps else None,
              # the test-1 signal: the first Serial interrupt (00:2125) of this visit on the normal core,
              # i.e. the event fast_mode never delivers
              "lag_from_first_serial_interrupt": (n - min(irq)) if (irq := [s for s, l in self.A.events
                  if l == "Serial" and entries and s >= entries[-1]]) else None,
              "serial_path_hooks_since_sync": serial_since,
              "text_A": text(self.A.pb), "text_B": text(self.B.pb)}
        # drive each core on its own until it has control again
        out = {}
        for c in (self.A, self.B):
            k0 = len(c.events)
            texts, stable, k = [], 0, 0
            while k < 3000 and stable < 6:
                t = text(c.pb)
                if t and (not texts or texts[-1] != t):
                    texts.append(t)
                if c.has_control():
                    stable += 1
                    c.idle()
                else:
                    stable = 0
                    if t:
                        c.press(self.policy_button(k))
                    else:
                        c.idle()
                k += 1
            self.resolve_steps += k
            out[c.name] = {"steps_to_control": k, "softlock": stable < 6,
                           "labels": [l for _, l in c.events[k0:] if l.startswith("CableClubNPC") or l == "LinkMenu"],
                           "texts": [t for t in texts if len(t) > 12][-12:]}
            c.pb.tick(1, True, False)
            p = self.frames / f"guard-{self.tag}-div{len(self.divergences)}-{c.name}.png"
            c.pb.screen.image.convert("RGB").save(p)
            out[c.name]["frame"] = p.name
        ev["after"] = out
        # test 2: game state once both have control
        fa, fb = self.A.pb.memory, self.B.pb.memory
        ev["field_diffs"] = {k: [hex(x) for x in range(a, b) if fa[x] != fb[x]][:8]
                             for k, (a, b) in FIELDS.items() if any(fa[x] != fb[x] for x in range(a, b))}
        excl = lambda x: any(a <= x < b for a, b in EXCLUDE)
        ma, mb = mem(self.A.pb), mem(self.B.pb)
        alld = [addr_of(i) for i in range(len(ma)) if ma[i] != mb[i]]
        ev["excluded_diffs_by_category"] = {
            "audio_wram": sum(0xC000 <= x < 0xC0F3 for x in alld), "stack": sum(0xDEE3 <= x < 0xE000 for x in alld),
            "sprite_state": sum(0xC100 <= x < 0xC3A0 or x in (0xD5CD, 0xFF92) for x in alld),
            "rng_and_frame_counter": sum(0xFFD3 <= x < 0xFFD6 for x in alld)}
        rest = [x for x in alld if not excl(x)]
        ev["serial_hram_diffs"] = [hex(x) for x in rest if SERIAL_HRAM[0] <= x < SERIAL_HRAM[1]]
        ev["other_wram_hram_diffs"] = [hex(x) for x in rest if not SERIAL_HRAM[0] <= x < SERIAL_HRAM[1]][:32]
        ev["same_game_state"] = not ev["field_diffs"] and not ev["other_wram_hram_diffs"]
        ev["same_text_path"] = out["A-normal"]["labels"] == out["B-fast"]["labels"]
        ev["B_not_sooner"] = out["B-fast"]["steps_to_control"] >= out["A-normal"]["steps_to_control"]
        self.divergences.append(ev)
        # resync B to A
        buf = io.BytesIO()
        self.A.pb.save_state(buf)
        buf.seek(0)
        self.B.pb.load_state(buf)
        self.last_sync = self.steps


def orphan_serial(ls, window=400):
    """Serial-path hook hits (either core) with no CableClubNPC entry in the preceding `window` steps."""
    out = []
    for c in (ls.A, ls.B):
        entries = [st for st, l in c.events if l == "CableClubNPC"]
        for st, l in c.events:
            if l in SERIAL_PATH and not any(0 <= st - e <= window for e in entries):
                out.append({"core": c.name, "step": st, "label": l})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=12000)
    a = ap.parse_args()
    faulthandler.register(signal.SIGUSR1)                 # kill -USR1 <pid> prints the stack
    rng = random.Random(a.seed)
    ls = Lockstep(a, rng)
    jit = rng.randrange(256)
    for c in (ls.A, ls.B):
        S.load(c.pb, a.state)
        c.pb.tick(jit, False, False) if jit else None
    s = S.Script(ls.A.pb, None, log=lambda *x: None, step_fn=ls.act)
    S.EDGES[S.ROUTE_2] = {S.VIRIDIAN_CITY: ("edge", "down"), ROUTE_2_TRADE_HOUSE: ("warp", "up")}
    S.EDGES[ROUTE_2_TRADE_HOUSE] = {S.ROUTE_2: ("out", "down")}
    fight = lambda: s.fight(rival=True)                  # losses allowed (blackouts happen in random play)
    counts, errors = {}, []
    t0 = time.time()
    while ls.steps < a.steps:
        # the Route 2 trade NPC is behind Cut trees (unreachable in E1); test 4 covers it statically
        act = rng.choices(["attendant", "pc", "nurse", "wander"], [40, 20, 15, 25])[0]
        ls.policy = rng.choice(["a", "a", "b", "random"])
        counts[act] = counts.get(act, 0) + 1
        try:
            if act == "trade":
                S.travel(s, ROUTE_2_TRADE_HOUSE, battle=fight)
                S.talk_to(s, (4, 1))
            else:
                S.travel(s, S.VIRIDIAN_POKECENTER, battle=fight)
                if act == "attendant":
                    S.talk_to(s, (11, 2))
                elif act == "nurse":
                    s.walk_to((3, 3), battle=fight)          # across the counter, as in the §9.1 trace
                    s.d.act("up")
                    s.d.act("a")
                elif act == "pc":
                    s.walk_to((13, 4), battle=fight)
                    s.d.act("up")
                    s.d.act("a")
            if act != "wander":
                for i in range(400):                     # answer the dialogue with this activity's policy
                    if s.control(4):
                        break
                    s.d.act(ls.policy_button(i) if s.d.screen_text() else "b")
            for _ in range(rng.randrange(10, 61) if act != "wander" else rng.randrange(30, 121)):
                s.d.act(rng.choice(ACTIONS))
        except S.ScriptError as e:
            errors.append({"step": ls.steps, "activity": act, "error": str(e)[:200]})
            if ls.steps >= a.steps:
                break
            for _ in range(20):
                s.d.act(rng.choice(ACTIONS))
    hooks = {}
    for c in (ls.A, ls.B):
        hc = {}
        for _, l in c.events:
            hc[l] = hc.get(l, 0) + 1
        hooks[c.name] = hc
    divs = ls.divergences
    res = {"state": a.state, "seed": a.seed, "jitter": jit, "steps": ls.steps, "seconds": round(time.time() - t0, 1),
           "activities": counts, "script_errors": errors[:20], "hook_counts": hooks,
           "tests": {
               "1_single_trigger": all(d["lag_from_first_serial_interrupt"] is not None
                                       and 0 <= d["lag_from_first_serial_interrupt"] <= 1 for d in divs),
               "1_attendant_visits_vs_divergences": [hooks["A-normal"].get("CableClubNPC", 0), len(divs)],
               "1_every_divergence_after_serial_event": all(d["serial_path_hooks_since_sync"] for d in divs),
               "2_same_game_state": all(d["same_game_state"] for d in divs),
               "3_same_text_path": all(d["same_text_path"] for d in divs),
               "3_no_softlock": not any(v["softlock"] for d in divs for v in d["after"].values()),
               "3_B_not_sooner": all(d["B_not_sooner"] for d in divs),
               "4_serial_only_after_attendant": orphan_serial(ls) == [],
               "4_pc_used": hooks["A-normal"].get("OpenPokemonCenterPC", 0),
               "4_center_listen_identical": hooks["A-normal"].get(CENTER_LISTEN, 0) == hooks["B-fast"].get(CENTER_LISTEN, 0),
               "serial_interrupts": {c: hooks[c].get("Serial", 0) for c in hooks}},
           "serial_events_without_attendant": orphan_serial(ls)[:20],
           "divergences": divs}
    (rom.states_dir() / "fastmode-guard").mkdir(exist_ok=True)
    (rom.states_dir() / "fastmode-guard" / f"{a.state}-s{a.seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("state", "seed", "steps", "seconds", "activities", "tests")}
                     | {"n_divergences": len(divs), "n_script_errors": len(errors)}))
    for c in (ls.A, ls.B):
        c.pb.stop(save=False)


if __name__ == "__main__":
    sys.exit(main())
