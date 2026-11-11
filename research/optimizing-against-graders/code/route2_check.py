#!/usr/bin/env python3
"""Check shaping.py's Route 2 split (y >= ROUTE2_SPLIT_Y = south part, no passage north except through the
Forest) against pret's block data (§9 "O list and R0 review", ask b). Reads a local pret/pokered checkout;
no ROM.

  python3 route2_check.py PRET_DIR

Walkability per step square (2×2 tiles): the tile at the square's bottom-left, looked up in the Overworld
tileset's passable list (data/tilesets/collision_tile_ids.asm, Overworld_Coll), from maps/Route2.blk (10×36
blocks) and gfx/blocksets/overworld.bst (16 tile ids per block). Cut trees and ledges are not passable tiles,
so a flood fill from the Viridian edge going north is what an agent without Cut can reach (ledges only drop
south, so they're barriers going north). NPCs are ignored (they only block more). The Route 2 Gate is crossed
as a building: its south door (15,39) leads to its north door (16,35).
"""
import re
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pokered import shaping as S  # noqa: E402


def main():
    pret = Path(sys.argv[1]).expanduser()
    coll = (pret / "data/tilesets/collision_tile_ids.asm").read_text()
    body = coll.split("Overworld_Coll::", 1)[1].split("::", 1)[0]
    passable = {int(h, 16) for h in re.findall(r"\$([0-9a-fA-F]{2})", body.split("\n\n")[0])}
    blk = (pret / "maps/Route2.blk").read_bytes()
    bst = (pret / "gfx/blocksets/overworld.bst").read_bytes()
    bw, bh = S.SIZE_BLOCKS[S.ROUTE2]
    assert len(blk) == bw * bh, len(blk)
    W, H = 2 * bw, 2 * bh

    def tile(sx, sy):
        b = blk[(sy // 2) * bw + sx // 2]
        return bst[b * 16 + (2 * (sy % 2) + 1) * 4 + 2 * (sx % 2)]

    walk = [[tile(x, y) in passable for x in range(W)] for y in range(H)]
    gate = {(15, 39): (16, 35), (16, 35): (15, 39)}

    def flood(starts):
        seen, q = set(starts), deque(starts)
        while q:
            x, y = q.popleft()
            nxt = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] + ([gate[(x, y)]] if (x, y) in gate else [])
            for nx, ny in nxt:
                if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen and (walk[ny][nx] or (nx, ny) in gate
                                                                            or (nx, ny) in S_WARPS):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return seen

    global S_WARPS
    S_WARPS = {(x, y) for x, y, *_ in S.WARPS[S.ROUTE2]}
    south = flood([(x, H - 1) for x in range(W) if walk[H - 1][x]])
    north = flood([(x, 0) for x in range(W) if walk[0][x]])
    print("map (from the south edge: S = reachable, from the north edge: N, both: B; # = blocked):")
    for y in range(H):
        row = ""
        for x in range(W):
            s, n = (x, y) in south, (x, y) in north
            row += "B" if s and n else "S" if s else "N" if n else ("." if walk[y][x] else "#")
        print(f"  {y:2d} {row}")
    meet = south & north
    res = {"south_reaches_north_edge": any(y == 0 for _, y in south),
           "south_reaches_forest_north_gate_door": (3, 11) in south,
           "south_reaches_gate_south_door_15_39": (15, 39) in south,
           "min_y_reached_from_south": min(y for _, y in south),
           "max_y_reached_from_north": max(y for _, y in north),
           "squares_reachable_from_both": len(meet),
           "split_row_consistent": all(y >= S.ROUTE2_SPLIT_Y for _, y in south - north)
                                   and all(y < S.ROUTE2_SPLIT_Y for _, y in north - south)}
    for k, v in res.items():
        print(f"{k}: {v}")
    # validate the walkability rule on the scripted trace: every square it stood on must be passable
    import json
    tr = Path.home() / "roms/states/script-trace.jsonl"
    if tr.exists():
        with open(tr) as f:
            rows = [json.loads(l) for l in f]
        for name, mid in (("Route1", 0x0C), ("ViridianCity", 0x01), ("PalletTown", 0x00), ("Route2", 0x0D)):
            b = (pret / f"maps/{name}.blk").read_bytes()
            w = S.SIZE_BLOCKS[mid][0]
            t = lambda sx, sy: bst[b[(sy // 2) * w + sx // 2] * 16 + (2 * (sy % 2) + 1) * 4 + 2 * (sx % 2)]
            sq = {(r["x"], r["y"]) for i, r in enumerate(rows) if i and r["map"] == mid and rows[i - 1]["map"] == mid}
            bad = [q for q in sq if t(*q) not in passable]
            print(f"rule check on script-trace.jsonl, {name}: {len(sq)} squares stood on, {len(bad)} not passable {bad[:5]}")


if __name__ == "__main__":
    main()
