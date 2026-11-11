#!/usr/bin/env python3
"""Build a contact sheet from per-run screenshots, outside the repo.

Reads <runs dir>/<run_id>/<run_id>.png + .json (and hack-*.png with --hacks),
tiles them with a caption (run id, arm, seed, round, T, P, hacks), and writes
contact-<time>.png into the runs dir. The writer picks frames from it later,
with Jake's approval (experiment-e1.md §10.6).

Usage:
  .venv/bin/python contact_sheet.py                 # all runs
  .venv/bin/python contact_sheet.py RUN_ID ...      # chosen runs
  .venv/bin/python contact_sheet.py --arm A0 --hacks --cols 6
  .venv/bin/python contact_sheet.py --waypoints     # script_states.py frames + RAM read at each (§9.1 review)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import rom  # noqa: E402

TILE_W, TILE_H, CAP_H, SCALE = 160, 144, 44, 2


def tiles_for(run_dir, hacks):
    meta_path = run_dir / f"{run_dir.name}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out = []
    main = run_dir / f"{run_dir.name}.png"
    if main.exists():
        cap = (f"{run_dir.name}\n{meta.get('arm')} s{meta.get('seed')} r{meta.get('round')} "
               f"T={meta.get('T')} P={meta.get('P')}\nhacks: {','.join(meta.get('hacks_active', [])) or '-'}")
        out.append((main, cap, meta))
    if hacks:
        for p in sorted(run_dir.glob("hack-*.png")):
            out.append((p, f"{run_dir.name}\n{p.stem}", meta))
    return out


def waypoint_tiles():
    """script_states.py waypoints: each frame captioned with the RAM values read at that moment, so a
    reviewer can compare the screen against the readers (experiment-e1.md §9.1)."""
    d = rom.states_dir() / "script-frames"
    out = []
    for line in open(d / "waypoints.jsonl"):
        r = json.loads(line)
        cap = (f"{r['n']:02d} {r['waypoint']}\nmap {r['map']:#04x} ({r['x']},{r['y']}) battle {r['in_battle']}\n"
               f"lv {r['levels']} hp {r['hp']}/{r['max_hp']}")
        out.append((d / r["frame"], cap, r))
    return out, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--arm")
    ap.add_argument("--hacks", action="store_true", help="include first-detection hack frames")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--waypoints", action="store_true", help="script_states.py frames with their RAM values")
    a = ap.parse_args()
    if a.waypoints:
        tiles, root = waypoint_tiles()
    else:
        root = rom.runs_dir()
        dirs = [root / r for r in a.runs] if a.runs else sorted(p for p in root.iterdir() if p.is_dir())
        tiles = [t for d in dirs for t in tiles_for(d, a.hacks)]
    if a.arm:
        tiles = [t for t in tiles if t[2].get("arm") == a.arm]
    if not tiles:
        sys.exit("no screenshots found")
    cols = min(a.cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    tw, th = TILE_W * SCALE, TILE_H * SCALE + CAP_H
    sheet = Image.new("RGB", (cols * tw, rows * th), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (path, cap, _) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * th
        img = Image.open(path).convert("RGB").resize((TILE_W * SCALE, TILE_H * SCALE), Image.NEAREST)
        sheet.paste(img, (x, y))
        draw.multiline_text((x + 4, y + TILE_H * SCALE + 2), cap, fill="black", font=font, spacing=1)
    out = root / (f"contact-waypoints.png" if a.waypoints else f"contact-{time.strftime('%Y%m%d-%H%M%S')}.png")
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
