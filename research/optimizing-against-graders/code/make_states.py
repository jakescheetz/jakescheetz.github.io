#!/usr/bin/env python3
"""Play Pokémon Red in a window and save the E1 start states with one key each.

Jake's whole to-do (experiment-e1.md §2, §9.1):

    POKEMON_RED_ROM=~/roms/pokemon_red.gb .venv/bin/python make_states.py

then play, and press a number key at the right moment:

    1  save D1   Oak's Lab right after the Pokédex, Squirtle
    2  save D2   same point, Charmander
    3  save HA   same point, Bulbasaur              (held-out A)
    4  save HB   Viridian Pokémon Center door, Squirtle, no extra grinding (held-out B)
    m  drop a marker in the check trace (e.g. "entered battle", "healed")
    Esc  quit

Game controls are PyBoy's: arrows, A = a, B = s, Start = Enter, Select = Backspace,
Space = fast-forward. (PyBoy's own z/x save/load and o/i/u dump keys still work;
their files land in the states dir, outside the repo.)

What it does:
  - refuses to start unless the ROM's SHA-1 is English Red (§3);
  - writes <name>.state plus <name>.json (RAM snapshot, party levels, sanity-check
    result) to the states dir, default <rom dir>/states/;
  - logs every RAM reader once per 24-frame step to check-trace-<time>.jsonl in the
    states dir (the §9.1 harness-check trace), with your markers in it;
  - if you start from a saved state: --load D1 (useful for making HB from D1).
Nothing is written into the repo.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import ram, rom  # noqa: E402

KEYS = {"1": "D1", "2": "D2", "3": "HA", "4": "HB"}
TRACE_EVERY = 24          # frames per trace row, same as the env's action length


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", help="start from a saved state name (e.g. D1)")
    ap.add_argument("--scale", type=int, default=4, help="window scale")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (testing)")
    a = ap.parse_args()

    try:
        rom_file = rom.rom_path(check=True)
        sdir = rom.states_dir()
    except rom.RomError as e:
        sys.exit(f"make_states: {e}")
    print(f"ROM OK ({rom.RED_SHA1[:8]}…)  states -> {sdir}")
    os.chdir(sdir)        # PyBoy's own screenshot/recording/state keys write to cwd

    import sdl2
    from pyboy import PyBoy
    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    pb = PyBoy(str(rom_file), window="null" if headless else "SDL2", scale=a.scale,
               sound_emulated=not headless)
    pb.set_emulation_speed(1)
    if a.load:
        with open(sdir / f"{a.load}.state", "rb") as f:
            pb.load_state(f)
        print(f"loaded {a.load}")

    trace_path = sdir / f"check-trace-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    trace = open(trace_path, "w")
    scancodes = {sdl2.SDL_GetScancodeFromKey(getattr(sdl2, f"SDLK_{k}")): k for k in list(KEYS) + ["m"]}
    held = set()
    frame = 0
    T = 0
    print("Keys: 1=D1 2=D2 3=HA 4=HB  m=marker  Esc=quit")
    while pb.tick(1, True):
        frame += 1
        mem = pb.memory
        T = ram.update_T(T, mem)
        pressed = set()
        if not headless:
            ks = sdl2.SDL_GetKeyboardState(None)
            pressed = {k for sc, k in scancodes.items() if ks[sc]}
        for k in pressed - held:              # rising edges only
            if k == "m":
                trace.write(json.dumps({"frame": frame, "marker": True, **ram.snapshot(mem)}) + "\n")
                print(f"marker at frame {frame}: map {mem[ram.W_CUR_MAP]:#04x}")
            else:
                save(pb, sdir, KEYS[k], frame)
        held = pressed
        if frame % TRACE_EVERY == 0:
            trace.write(json.dumps({"frame": frame, "T": T, **ram.snapshot(mem)}) + "\n")
        if a.max_frames and frame >= a.max_frames:
            break
    trace.close()
    pb.stop(save=False)
    print(f"check trace: {trace_path}")


def save(pb, sdir, name, frame):
    mem = pb.memory
    path = sdir / f"{name}.state"
    with open(path, "wb") as f:
        pb.save_state(f)
    fails = ram.sanity_check(mem)
    snap = ram.snapshot(mem)
    meta = {"name": name, "frame": frame, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "rom_sha1": rom.RED_SHA1, "sanity_failures": fails,
            "party_levels": snap["levels"], "snapshot": snap}
    (sdir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    status = "OK" if not fails else "CHECK FAILED: " + "; ".join(fails)
    print(f"saved {name} -> {path}  map {snap['map']:#04x} ({snap['x']},{snap['y']}) "
          f"levels {snap['levels']}  [{status}]")


if __name__ == "__main__":
    main()
