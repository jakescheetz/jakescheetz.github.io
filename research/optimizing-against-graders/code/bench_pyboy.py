#!/usr/bin/env python3
"""Measure PyBoy emulation speed on this machine without any game ROM.

Builds a tiny synthetic Game Boy ROM in memory (a busy loop that keeps the CPU
and LCD running and writes to WRAM), so no copyrighted or downloaded ROM is
involved. A real game does more work per frame, so treat these numbers as an
upper bound; the README applies a discount.

Usage:  python bench_pyboy.py [--frames 20000] [--procs 1 2 4]
Output: a table on stdout (frames/s per process, aggregate, RSS per instance).
"""
import argparse
import multiprocessing as mp
import os
import resource
import tempfile
import time


def build_rom(path):
    rom = bytearray(0x8000)
    rom[0x100:0x104] = bytes([0x00, 0xC3, 0x50, 0x01])        # NOP; JP 0x0150
    rom[0x134:0x13F] = b"BENCH".ljust(11, b"\0")               # title
    rom[0x147] = 0x00                                          # ROM only
    # 0x150: LD A,0x91; LDH (0x40),A  (LCD on)
    #        loop: LD HL,0xC000; INC (HL); INC HL; LD A,(HL); ADD A,B; LD (HL),A; JR loop
    code = [0x3E, 0x91, 0xE0, 0x40,
            0x21, 0x00, 0xC0, 0x34, 0x23, 0x7E, 0x80, 0x77, 0x18, 0xF6]
    rom[0x150:0x150 + len(code)] = bytes(code)
    chk = 0
    for b in rom[0x134:0x14D]:
        chk = (chk - b - 1) & 0xFF
    rom[0x14D] = chk
    with open(path, "wb") as f:
        f.write(rom)


def run(args):
    rom, frames, render = args
    from pyboy import PyBoy
    pb = PyBoy(rom, window="null", sound_emulated=False)
    pb.set_emulation_speed(0)
    pb.tick(60, False)
    t = time.perf_counter()
    if render:
        for _ in range(frames):
            pb.tick(1, True)
            _ = pb.screen.ndarray  # pixel observation, as an RL env would read
    else:
        pb.tick(frames, False)
    dt = time.perf_counter() - t
    _ = pb.memory[0xC000]  # RAM read, as a reward function would do
    pb.stop(save=False)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return frames / dt, rss_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20000)
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4])
    a = ap.parse_args()
    import pyboy
    print(f"pyboy {pyboy.__version__ if hasattr(pyboy, '__version__') else '?'}, "
          f"cpus {os.cpu_count()}, frames/proc {a.frames}")
    with tempfile.TemporaryDirectory() as d:
        rom = os.path.join(d, "bench.gb")
        build_rom(rom)
        print(f"{'render':>6} {'procs':>5} {'fps/proc':>10} {'fps total':>10} {'RSS MB':>7}")
        for render in (False, True):
            for n in a.procs:
                with mp.Pool(n) as pool:
                    res = pool.map(run, [(rom, a.frames, render)] * n)
                fps = [r[0] for r in res]
                print(f"{str(render):>6} {n:>5} {sum(fps)/n:>10.0f} {sum(fps):>10.0f} "
                      f"{max(r[1] for r in res):>7.0f}")


if __name__ == "__main__":
    main()
