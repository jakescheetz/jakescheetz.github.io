#!/usr/bin/env python3
"""A2″ walking-distance grid (experiment-e1.md §9 "A2′ pilot"; pokered/walkdist.py).

  .venv/bin/python walkgrid.py build PRET_DIR      # writes ~/roms/walkgrid.json (outside the repo)
  .venv/bin/python walkgrid.py show [MAP ...]      # ASCII map: # blocked, . passable, digits = d mod 10
  .venv/bin/python walkgrid.py check TRACE ...     # validation (a): every square a trace stood on is walkable

The grid is derived from the game's map data (pret), so it lives next to the ROM, never in the repo.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import walkdist as WD  # noqa: E402


def cmd_build(a):
    out, g = WD.write_grid(a.pret, a.out)
    wd = WD.WalkDist(g)
    print(f"wrote {out}  grid_hash {WD.grid_hash(g)}  maps {len(g['maps'])}  nodes {len(wd.fwd)}  "
          f"ledges {sum(len(m['ledges']) for m in g['maps'].values())}")
    print("leg starts:", wd.leg_start)
    print("d_leg:", wd.d_leg)


def cmd_show(a):
    wd = WD.WalkDist(WD.load_grid(a.grid))
    for mp in (a.maps or [str(m) for m in wd.maps]):
        mp = int(mp, 0)
        g = wd.maps[mp]
        print(f"map {mp:#04x} {g['name']} ({g['tileset']}) {g['w']}x{g['h']}  leg {a.leg} d:")
        for y in range(g["h"]):
            row = ""
            for x in range(g["w"]):
                d = wd.d(a.leg, mp, x, y)
                row += ("#" if g["passable"][y][x] == "0" else "?") if d is None else str(d % 10)
            print(f"  {y:2d} {row}")


def read_rows(path):
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [r for r in map(json.loads, f) if "map" in r]     # traces start with a header row


def trace_squares(path, settle=WD.SETTLE_STEPS):
    """(map, x, y) a trace stood on. Rows within `settle` steps after a map change are left out: a warp
    updates wCurMap 1-2 steps before wXCoord/wYCoord, so those rows pair the new map with old coordinates
    (the same rule A2″ uses for d_min, §9 (c))."""
    out, since = set(), None
    for i, r in enumerate(read_rows(path)):
        if i == 0 or r.get("ep_step") == 1:
            since = 0                               # episode start: step 1 may already be a warp (HB)
        elif r["map"] != prev["map"]:
            since = 0
        else:
            since += 1
        if since > settle:
            out.add((r["map"], r["x"], r["y"]))
        prev = r
    return out


def check(wd, paths):
    res = {"files": len(paths), "squares": 0, "outside_region": 0, "unwalkable": [], "no_distance": []}
    seen = set()
    for p in paths:
        seen |= trace_squares(p)
    for mp, x, y in sorted(seen):
        if mp not in wd.maps:
            res["outside_region"] += 1
            continue
        res["squares"] += 1
        if (mp, x, y) not in wd.fwd:
            res["unwalkable"].append((mp, x, y))
        elif all(wd.d(k, mp, x, y) is None for k in wd.dist):
            res["no_distance"].append((mp, x, y))
    return res


def cmd_check(a):
    wd = WD.WalkDist(WD.load_grid(a.grid))
    paths = [Path(p).expanduser() for p in a.traces]
    res = check(wd, paths)
    print(json.dumps({**res, "unwalkable": [f"{m:#04x}:{x},{y}" for m, x, y in res["unwalkable"]],
                      "no_distance": [f"{m:#04x}:{x},{y}" for m, x, y in res["no_distance"]]}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("pret")
    b.add_argument("--out")
    s = sub.add_parser("show")
    s.add_argument("maps", nargs="*")
    s.add_argument("--leg", type=int, default=1)
    s.add_argument("--grid")
    c = sub.add_parser("check")
    c.add_argument("traces", nargs="+")
    c.add_argument("--grid")
    c.add_argument("--out")
    a = ap.parse_args()
    {"build": cmd_build, "show": cmd_show, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
