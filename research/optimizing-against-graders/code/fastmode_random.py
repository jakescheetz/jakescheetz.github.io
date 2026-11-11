#!/usr/bin/env python3
"""Characterize fast_mode divergence under random inputs (experiment-e1.md §5b, researcher's follow-up).

  POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python fastmode_random.py --state HB --seed 1 [--steps 10000]

Normal core (A) and fast_mode (B) in lockstep from a start state, same jitter, seeded uniform-random
env actions (the 7 env buttons, 24 frames each). WRAM 0xC000-0xDFFF + HRAM 0xFF80-0xFFFE compared
after every step. On each divergence:
  - log the last 60 steps (button, position, battle flag, decoded screen text of A)
  - classify it: "link" if the Cable Club / link text appeared on screen in those 60 steps, else "other"
  - follow both cores for --follow steps on the same inputs (text of A and B, whether B's screen and RAM
    still change, i.e. not hung) and save screenshots of both at +0/+10/+50/+150 to
    ~/roms/states/fastmode-frames/ (outside the repo)
  - then resync: copy A's full state into B and continue, so later, independent divergences are caught too
Also logs every attendant interaction (link text on screen) with or without a divergence, so a run that
never talks to the attendant can be checked for byte-identity.
Output: ~/roms/states/fastmode-random/<state>-s<seed>.json
"""
import argparse
import io
import json
import random
import sys
import time

import script_states as S
from pokered import ram, rom
from pokered.driver import CHARS, FRAMES_PER_ACTION, PRESS_FRAMES
from pokered.env import ACTIONS

WRAM, HRAM = (0xC000, 0xE000), (0xFF80, 0xFFFF)
LINK_WORDS = ("cable club", "link", "colosseum", "trade center")


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


def is_link(t):
    t = t.lower()
    return any(w in t for w in LINK_WORDS)


def step(pbs, button):
    for pb in pbs:
        pb.button(button, PRESS_FRAMES)
        pb.tick(FRAMES_PER_ACTION, False, False)


def shot(pb, path):
    pb.screen.image.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--follow", type=int, default=150)
    a = ap.parse_args()
    sd = rom.states_dir()
    out_dir = sd / "fastmode-random"
    frames = sd / "fastmode-frames"
    out_dir.mkdir(exist_ok=True)
    frames.mkdir(exist_ok=True)
    A, B = new(False), new(True)
    rng = random.Random(a.seed)
    jit = rng.randrange(256)
    for pb in (A, B):
        S.load(pb, a.state)
        pb.tick(jit, False, False) if jit else None
    hist = []                       # recent steps of A
    divergences, link_events = [], []
    maps_visited = set()
    n = 0
    t0 = time.time()
    while n < a.steps:
        b = rng.choice(ACTIONS)
        step((A, B), b)
        n += 1
        tA = text(A)
        mp, x, y = ram.position(A.memory)
        maps_visited.add(mp)
        hist.append({"step": n, "button": b, "pos": [mp, x, y], "battle": A.memory[ram.W_IS_IN_BATTLE], "text": tA})
        hist = hist[-60:]
        if is_link(tA) and (not link_events or n - link_events[-1]["step"] > 30):
            link_events.append({"step": n, "pos": [mp, x, y], "text": tA})
        ma, mb = mem(A), mem(B)
        if ma == mb:
            continue
        diff = [addr_of(i) for i in range(len(ma)) if ma[i] != mb[i]]
        k = len(divergences)
        ev = {"step": n, "pos": [mp, x, y], "first_addresses": [hex(x) for x in diff[:24]], "n_addresses": len(diff),
              "rng_bytes_differ": any(x in (0xFFD3, 0xFFD4) for x in diff),
              "kind": "link" if any(is_link(h["text"]) for h in hist) else "other",
              "last_60_steps": list(hist)}
        # follow both cores on the same inputs: what does B do?
        follow, snaps = [], {}
        prevB = mem(B)
        for j in range(a.follow + 1):
            if j in (0, 10, 50, a.follow):
                for name, pb in (("A-normal", A), ("B-fast", B)):
                    pb.tick(1, True, False)                     # one rendered frame on both (keeps them paired)
                    p = frames / f"{a.state}-s{a.seed}-div{k}-{name}-plus{j:03d}.png"
                    shot(pb, p)
                    snaps.setdefault(j, {})[name] = p.name
            if j == a.follow:
                break
            bb = rng.choice(ACTIONS)
            step((A, B), bb)
            n += 1
            nowB = mem(B)
            follow.append({"plus": j + 1, "button": bb, "A_text": text(A), "B_text": text(B),
                           "A_pos": list(ram.position(A.memory)), "B_pos": list(ram.position(B.memory)),
                           "B_ram_changed": nowB != prevB, "same": mem(A) == nowB})
            prevB = nowB
        ev["follow"] = follow
        ev["frames"] = snaps
        ev["B_ram_static_steps_in_follow"] = sum(1 for f in follow if not f["B_ram_changed"])
        ev["rejoined_without_resync"] = any(f["same"] for f in follow)
        divergences.append(ev)
        # resync B to A and continue
        buf = io.BytesIO()
        A.save_state(buf)
        buf.seek(0)
        B.load_state(buf)
        hist = []
    res = {"state": a.state, "seed": a.seed, "jitter": jit, "steps": n, "seconds": round(time.time() - t0, 1),
           "maps_visited": sorted(hex(m) for m in maps_visited),
           "center_visited": 0x29 in maps_visited,
           "link_interactions": link_events, "divergences": divergences,
           "summary": {"divergences": len(divergences), "link": sum(d["kind"] == "link" for d in divergences),
                       "other": sum(d["kind"] == "other" for d in divergences),
                       "link_interactions": len(link_events)}}
    (out_dir / f"{a.state}-s{a.seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("state", "seed", "steps", "seconds", "center_visited", "summary")}))
    for pb in (A, B):
        pb.stop(save=False)


if __name__ == "__main__":
    sys.exit(main())
