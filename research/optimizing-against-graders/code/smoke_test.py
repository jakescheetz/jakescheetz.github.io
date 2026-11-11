#!/usr/bin/env python3
"""Harness smoke test on the real ROM: env speed, readers, run outputs, frame timing.

  .venv/bin/python smoke_test.py                    # from power-on (no start state yet)
  .venv/bin/python smoke_test.py --state D1         # from a start state (runs the sanity check)
  .venv/bin/python smoke_test.py --state D1 --check-frames

--check-frames: for several frames-per-action values, press one direction and
count how often the player moves exactly one tile. Needs an overworld state.
Screenshots and sidecars go to $POKEMON_RUNS_DIR (default <rom dir>/runs/).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import detectors, env as E, ram  # noqa: E402


def speed(state, steps):
    env = E.PokemonRedEnv(state_name=state, max_steps=steps, run_id=f"smoke-{time.strftime('%H%M%S')}",
                          meta={"arm": "smoke", "seed": 0, "round": 0}, capture_hack_frames=True)
    env.reset()
    rng = random.Random(0)
    t = time.perf_counter()
    done = False
    while not done:
        _, _, done, info = env.step(rng.randrange(len(E.ACTIONS)))
    dt = time.perf_counter() - t
    png = env.save_run_frame()
    print(f"env steps/s (1 env, random actions, no policy): {steps / dt:.0f}")
    print(f"last row: {json.dumps({k: env.trace[-1][k] for k in ('map', 'x', 'y', 'levels', 'hp', 'T')})}")
    print(f"detector firings over the trace: {detectors.first_appearance(detectors.scan(env.trace)) or 'none'}")
    print(f"screenshot + sidecar: {png} (+ .json)")
    env.close()


def check_frames(state, trials):
    """Shortest reliable frames/action (experiment-e1.md §8). From `state`, the driver walks to an open
    square in Pallet Town (no wild encounters), saved in memory. Then, per frames/action value, on the
    trainer's core (fast_mode): `trials` sequences of 6 presses (straight runs and direction changes,
    each with a random 0-63 frame jitter), each kept inside the open area. Per action we check:
      ok    the player moved exactly one tile in the pressed direction (or only turned, when 24 f/a
            also only turns), and RAM shows it at the end of the action
      extra the position still changed during a 64-frame settle after the sequence (queued input)
    A value is reliable if its outcomes match 24 f/a's on every action and extra == 0."""
    import io
    import script_states as S
    from pokered.driver import Driver, PRESS_FRAMES
    pb = S.new_pyboy()                           # trainer core (normal since §5b failed)
    S.load(pb, state)
    sc = S.Script(pb, None, log=lambda *a: None)
    S.travel(sc, S.PALLET_TOWN)
    passable, _ = sc.d.walk_grid()
    H, W = len(passable), len(passable[0])

    def open_run(x, y, dx, dy, n):
        return all(0 <= x + dx * k < W and 0 <= y + dy * k < H and passable[y + dy * k][x + dx * k]
                   for k in range(1, n + 1))
    dirs = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    best = max(((sum(open_run(x, y, dx, dy, 3) for dx, dy in dirs.values()), (x, y))
                for y in range(H) for x in range(W) if passable[y][x]), key=lambda t: t[0])
    if best[0] < 4:
        sys.exit("no open square with 3 free tiles in every direction in Pallet Town")
    sc.walk_to(best[1])
    sc.talk_until(lambda: sc.control(6), limit=30)
    home = io.BytesIO()
    pb.save_state(home)
    home = home.getvalue()

    rng = random.Random(7)
    seqs = []
    for t in range(trials):
        seq, x, y = [], *best[1]
        for _ in range(6):                       # stay inside the 3-tile box around the start square
            opts = [d for d, (dx, dy) in dirs.items() if abs(x + dx - best[1][0]) <= 2 and abs(y + dy - best[1][1]) <= 2
                    and passable[y + dy][x + dx]]
            if t % 2 == 0 and seq and seq[-1] in opts:
                d = seq[-1]                      # even trials: straight runs where possible
            else:
                d = rng.choice(opts)
            seq.append(d)
            x, y = x + dirs[d][0], y + dirs[d][1]
        seqs.append((rng.randrange(64), seq))

    def run(fpa):
        outcomes, extra = [], 0
        for jit, seq in seqs:
            pb.load_state(io.BytesIO(home))
            pb.tick(jit, False, False) if jit else None
            for d in seq:
                before = ram.position(pb.memory)
                pb.button(d, min(PRESS_FRAMES, fpa))
                pb.tick(fpa, False, False)
                after = ram.position(pb.memory)
                dx, dy = after[1] - before[1], after[2] - before[2]
                outcomes.append("move" if (dx, dy) == dirs[d] else "none" if (dx, dy) == (0, 0) else "wrong")
            p0 = ram.position(pb.memory)
            pb.tick(64, False, False)
            extra += ram.position(pb.memory) != p0
        return outcomes, extra
    base, base_extra = run(24)
    print(f"open square {best[1]} in Pallet; {trials} sequences x 6 presses; trainer core (normal; §5b)")
    results = {}
    for fpa in (8, 12, 16, 20, 24, 32):
        out, extra = run(fpa) if fpa != 24 else (base, base_extra)
        same = sum(a == b for a, b in zip(out, base))
        mv = out.count("move")
        results[fpa] = {"moved_one_tile": mv, "turn_only": out.count("none"), "wrong": out.count("wrong"),
                        "same_as_24": same, "actions": len(out), "settle_moves": extra,
                        "reliable": same == len(out) and extra == 0 and not out.count("wrong")}
        r = results[fpa]
        print(f"frames/action {fpa:>2}: one tile {mv / len(out):6.1%}  turn-only {r['turn_only']:>4}  wrong {r['wrong']:>3}  "
              f"same as 24 {same / len(out):6.1%}  moves after settle {extra:>3}  -> {'RELIABLE' if r['reliable'] else 'no'}")
    shortest = min((f for f, r in results.items() if r["reliable"]), default=None)
    print(f"shortest reliable frames/action: {shortest}")
    (E.rom.states_dir() / "check-frames.json").write_text(json.dumps(
        {"state": state, "square": best[1], "trials": trials, "results": results, "shortest_reliable": shortest,
         "core": "normal" if not S.FAST_MODE else "fast_mode", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    pb.stop(save=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--check-frames", action="store_true")
    ap.add_argument("--trials", type=int, default=100)
    a = ap.parse_args()
    speed(a.state, a.steps)
    if a.check_frames:
        if not a.state:
            sys.exit("--check-frames needs --state (an overworld start state)")
        check_frames(a.state, a.trials)


if __name__ == "__main__":
    main()
