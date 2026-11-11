"""A2″ walking distance: BFS over walkable step squares, joined by warps and map connections
(experiment-e1.md §9 "A2′ pilot" → A2″).

Two parts:

  build_grid(pret_dir)  reads a local pret/pokered checkout and writes the move grid for the segment's
                        maps (shaping.SIZE) to ~/roms/walkgrid.json, i.e. *outside the repo*: it is
                        derived from the game's map data, so it stays with the ROM (§12). Run it through
                        `walkgrid.py build PRET_DIR`; the VM gets a copy over scp, like the states.
  WalkDist(grid)        per leg k (T = k-1 → milestone k), d_k(map, x, y) = fewest env moves from that
                        square to the leg's target, by BFS run backwards over the directed move graph.

The game's movement rules, from pret (paths below are in the pret tree):
  * A step square (2×2 tiles) is passable when its bottom-left 8×8 tile is in the map tileset's
    collision list (data/tilesets/collision_tile_ids.asm; the tileset's blockset is in gfx/tilesets.asm,
    the map's blocks in maps/<Name>.blk via maps.asm, the map's tileset in data/maps/headers/<Name>.asm).
    That is home/overworld.asm's rule, and driver.walk_grid's, which reads the same thing from WRAM.
  * Tile-pair collisions (data/tilesets/pair_collision_tile_ids.asm, TilePairCollisionsLand): moving
    between a square whose tile is t1 and one whose tile is t2 is blocked, in both directions. In this
    segment only the FOREST tileset (Viridian Forest) has them.
  * Ledges (engine/overworld/ledges.asm, data/tilesets/ledge_tiles.asm; OVERWORLD tileset only): facing
    direction D, standing on tile S, with the ledge tile L in front, pressing D hops two squares. The
    table has down, left and right ledges; none goes up, so ledges are one-way.
  * Warps (data/maps/objects/*.asm, resolved in shaping.WARPS / shaping.exits): standing on a warp
    square, one more move lands on the destination warp square. A warp square is used only if its tile
    is passable, like any square: some warp pairs have a half the player can't stand on (Viridian Forest
    south gate (4, 0)), and no trace in validation (a) ever stood on a non-passable warp square.
  * Map connections (data/maps/headers/*.asm, shaping.CONNECTIONS): from an edge square, a move off
    the edge lands on the matching square of the next map, if that square is passable.
NPCs, signs, trainers' sight lines and Cut trees are not modeled beyond the tiles: NPCs only block,
and the cuttable trees are impassable tiles anyway (no Cut in this segment).

Each move costs 1 (one env action), so d is in env moves; a ledge hop counts 1 (one press).
"""
import hashlib
import json
import re
from collections import deque
from pathlib import Path

from . import ram, shaping

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
DIR_BIT = {"up": 1, "down": 2, "left": 4, "right": 8}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
FACING = {"down": 0x00, "up": 0x04, "left": 0x08, "right": 0x0C}   # SPRITE_FACING_* (constants)
GRID_VERSION = 2          # 2: STAY NPCs/items visible in the start state are blocked squares
SETTLE_STEPS = 2          # §9 (c): no d_min update for 2 steps after a map change


def default_path():
    from . import rom
    return rom.states_dir().parent / "walkgrid.json"


# --- build from pret -------------------------------------------------------------------------------
def _labels_to_value(text, pattern):
    """asm with chained labels ("A::\\nB:: INCBIN x"): {label: value of the next INCBIN/coll line}."""
    out, pending = {}, []
    for line in text.splitlines():
        m = re.match(r"\s*(\w+)::?\s*(.*)", line)
        if m and not line.startswith((" ", "\t")):
            pending.append(m.group(1))
            rest = m.group(2)
        else:
            rest = line
        v = re.search(pattern, rest)
        if v and pending:
            for p in pending:
                out[p] = v.group(1)
            pending = []
    return out


W_TOGGLEABLE_OBJECT_FLAGS = 0xD5A6     # pret pokered.sym: wToggleableObjectFlags (bit set = hidden), 32 bytes


def build_grid(pret_dir, toggle_flags=None):
    """toggle_flags: wToggleableObjectFlags bytes from the start state (D1), so toggleable NPCs and items
    are blocked exactly when that state shows them; None = pret's initial ON/OFF states."""
    pret = Path(pret_dir).expanduser()
    used = []

    def read(rel, binary=False):
        p = pret / rel
        used.append(rel)
        return p.read_bytes() if binary else p.read_text()

    # map ids, in constants order
    ids = {}
    for line in read("constants/map_constants.asm").splitlines():
        m = re.match(r"\s*map_const\s+(\w+),", line)
        if m:
            ids[m.group(1)] = len(ids)
    tileset_ids = [m.group(1) for m in re.finditer(r"^\s*const (\w+)", read("constants/tileset_constants.asm"), re.M)]
    ts_names = [m.group(1) for m in re.finditer(r"^\s*tileset (\w+),", read("data/tilesets/tileset_headers.asm"), re.M)]
    ts_label = dict(zip(tileset_ids, ts_names))                       # FOREST -> Forest
    block_file = _labels_to_value(read("gfx/tilesets.asm"), r'INCBIN "(gfx/blocksets/\w+\.bst)"')
    coll_raw = read("data/tilesets/collision_tile_ids.asm")
    coll = {}
    pending = []
    for line in coll_raw.splitlines():
        m = re.match(r"(\w+)_Coll::", line)
        if m:
            pending.append(m.group(1))
        elif "coll_tiles" in line and pending and "MACRO" not in line:
            vals = [int(h, 16) for h in re.findall(r"\$([0-9a-fA-F]{2})", line)]
            for p in pending:
                coll[p] = vals
            pending = []
    blk_file = _labels_to_value(read("maps.asm"), r'INCBIN "(maps/\w+\.blk)"')
    pairs = [(m.group(1), int(m.group(2), 16), int(m.group(3), 16)) for m in re.finditer(
        r"db (\w+), \$([0-9A-Fa-f]{2}), \$([0-9A-Fa-f]{2})",
        read("data/tilesets/pair_collision_tile_ids.asm").split("TilePairCollisionsWater")[0])]
    facing_const = {"SPRITE_FACING_DOWN": "down", "SPRITE_FACING_UP": "up", "SPRITE_FACING_LEFT": "left",
                    "SPRITE_FACING_RIGHT": "right"}
    ledges = [(facing_const[m.group(1)], int(m.group(2), 16), int(m.group(3), 16)) for m in re.finditer(
        r"db (SPRITE_FACING_\w+),\s*\$([0-9A-Fa-f]{2}), \$([0-9A-Fa-f]{2})", read("data/tilesets/ledge_tiles.asm"))]

    # toggleable objects: global index = order in toggleable_objects.asm (pret: TOGGLE_* constants)
    toggles, cur = {}, None
    for line in read("data/maps/toggleable_objects.asm").split("ToggleableObjectStates:")[1].splitlines():
        m = re.match(r"\s*toggleable_objects_for (\w+)", line)
        if m:
            cur = m.group(1)
        m = re.match(r"\s*toggle_object_state (\w+),\s*(ON|OFF)", line)
        if m:
            toggles[m.group(1)] = (len(toggles), m.group(2) == "ON")

    def visible(objname):
        if objname not in toggles:
            return True
        idx, on = toggles[objname]
        if toggle_flags is None:
            return on
        return not (toggle_flags[idx // 8] >> (idx % 8)) & 1

    headers = {}
    for p in sorted((pret / "data/maps/headers").glob("*.asm")):
        m = re.search(r"map_header (\w+), (\w+), (\w+)", p.read_text())
        if m:
            headers[ids[m.group(2)]] = (m.group(1), m.group(3), f"data/maps/headers/{p.name}")

    maps = {}
    for mp in sorted(shaping.SIZE):
        name, ts, hdr = headers[mp]
        used.append(hdr)
        lab = ts_label[ts]
        bst = read(block_file[f"{lab}_Block"], binary=True)
        blk = read(blk_file[f"{name}_Blocks"], binary=True)
        bw, bh = shaping.SIZE_BLOCKS[mp]
        assert len(blk) == bw * bh, (name, len(blk), bw, bh)
        W, H = 2 * bw, 2 * bh

        def tile(sx, sy):
            return bst[blk[(sy // 2) * bw + sx // 2] * 16 + (2 * (sy % 2) + 1) * 4 + 2 * (sx % 2)]

        obj_txt = read(f"data/maps/objects/{name}.asm")
        names = re.findall(r"const_export (\w+)", obj_txt)
        objs = re.findall(r"object_event\s+(\d+),\s*(\d+),\s*(\w+),\s*(\w+)", obj_txt)
        npc_block = {(int(x), int(y)) for (x, y, spr, mv), nm in zip(objs, names + [None] * len(objs))
                     if mv == "STAY" and visible(nm)}
        ok = set(coll[lab])
        bad_pairs = {(a, b) for t, a, b in pairs if t == ts} | {(b, a) for t, a, b in pairs if t == ts}
        rows_pass, rows_moves, hops = [], [], []
        for y in range(H):
            pr, mr = "", ""
            for x in range(W):
                pr += "1" if tile(x, y) in ok and (x, y) not in npc_block else "0"
                bits = 0
                for dname, (dx, dy) in DIRS.items():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and tile(nx, ny) in ok and (nx, ny) not in npc_block \
                            and (tile(x, y), tile(nx, ny)) not in bad_pairs:
                        bits |= DIR_BIT[dname]
                    if ts == "OVERWORLD" and 0 <= nx < W and 0 <= ny < H:
                        lx, ly = x + 2 * dx, y + 2 * dy
                        for fd, s, l in ledges:
                            if fd == dname and tile(x, y) == s and tile(nx, ny) == l and 0 <= lx < W and 0 <= ly < H \
                                    and (lx, ly) not in npc_block:
                                hops.append([x, y, dname])
                mr += f"{bits:x}"
            rows_pass.append(pr)
            rows_moves.append(mr)
        maps[str(mp)] = {"name": name, "tileset": ts, "w": W, "h": H, "passable": rows_pass,
                         "moves": rows_moves, "ledges": hops,
                         "npc_blocked": sorted(npc_block)}
    src = {rel: hashlib.sha1((pret / rel).read_bytes()).hexdigest()[:12] for rel in sorted(set(used))}
    return {"version": GRID_VERSION, "maps": maps, "pret_files": src,
            "npc_visibility": "start state" if toggle_flags is not None else "pret initial"}


def start_state_flags(state="D1"):
    """wToggleableObjectFlags from a start state (needs the ROM; the bytes never leave ~/roms)."""
    import io
    from pyboy import PyBoy
    from . import rom
    pb = PyBoy(str(rom.rom_path(check=True)), window="null", sound_emulated=False)
    pb.load_state(io.BytesIO((rom.states_dir() / f"{state}.state").read_bytes()))
    flags = bytes(pb.memory[W_TOGGLEABLE_OBJECT_FLAGS:W_TOGGLEABLE_OBJECT_FLAGS + 32])
    pb.stop(save=False)
    return flags


def write_grid(pret_dir, out=None, state="D1"):
    g = build_grid(pret_dir, start_state_flags(state) if state else None)
    g["npc_state"] = state
    out = Path(out) if out else default_path()
    from . import rom
    rom._outside_repo(out, "walk grid")
    out.write_text(json.dumps(g))
    return out, g


def load_grid(path=None):
    p = Path(path) if path else default_path()
    if not p.exists():
        raise FileNotFoundError(f"{p}: build it with `walkgrid.py build PRET_DIR` (stays outside the repo)")
    return json.loads(p.read_text())


def grid_hash(grid):
    return hashlib.sha1(json.dumps({k: grid[k] for k in ("version", "maps")}, sort_keys=True).encode()).hexdigest()[:12]


# --- distances -------------------------------------------------------------------------------------
class WalkDist:
    """Forward move graph over (map, x, y) and per-leg BFS distances to each leg's target."""

    def __init__(self, grid):
        self.grid = grid
        self.hash = grid_hash(grid)
        self.maps = {int(k): v for k, v in grid["maps"].items()}
        self.warp_sq = {mp: {(x, y) for x, y, *_ in shaping.WARPS.get(mp, [])} for mp in self.maps}
        self.fwd = self._graph()
        rev = {}
        for u, vs in self.fwd.items():
            for v, _ in vs:
                rev.setdefault(v, []).append(u)
        self.rev = rev
        self.dist = {k: self._bfs(k) for k in range(1, len(ram.MILESTONES) + 1)}
        self.leg_start, self.d_leg = self._reference()

    def passable(self, mp, x, y):
        g = self.maps.get(mp)
        return bool(g) and 0 <= x < g["w"] and 0 <= y < g["h"] and g["passable"][y][x] == "1"

    def _graph(self):
        """(map, x, y) -> [((map, x, y), how)]; how = direction name, 'ledge-<dir>', or 'exit-<dir>'."""
        fwd = {}
        for mp, g in self.maps.items():
            W, H = g["w"], g["h"]
            for y in range(H):
                for x in range(W):
                    if g["passable"][y][x] != "1":
                        continue
                    out = fwd.setdefault((mp, x, y), [])
                    bits = int(g["moves"][y][x], 16)
                    for dname, (dx, dy) in DIRS.items():
                        nx, ny = x + dx, y + dy
                        if bits & DIR_BIT[dname]:
                            out.append(((mp, nx, ny), dname))
            for x, y, dname in g["ledges"]:
                dx, dy = DIRS[dname]
                fwd.setdefault((mp, x, y), []).append(((mp, x + 2 * dx, y + 2 * dy), f"ledge-{dname}"))
            for x, y, dest, wid in shaping.WARPS.get(mp, []):
                if dest is None:
                    dm, (lx, ly) = shaping._parent_warp(mp, wid)
                else:
                    dm, (lx, ly) = dest, shaping.WARPS[dest][wid - 1][:2]
                if dm in self.maps and self.passable(mp, x, y) and self.passable(dm, lx, ly):
                    fwd.setdefault((mp, x, y), []).append(((dm, lx, ly), "warp"))
            for (src, direction), (dm, off) in shaping.CONNECTIONS.items():
                if src != mp or dm not in self.maps:
                    continue
                dw, dh = self.maps[dm]["w"], self.maps[dm]["h"]
                if direction in ("north", "south"):
                    edge = [(x, 0 if direction == "north" else H - 1, x - 2 * off,
                             dh - 1 if direction == "north" else 0) for x in range(W)]
                else:
                    edge = [(0 if direction == "west" else W - 1, y, dw - 1 if direction == "west" else 0,
                             y - 2 * off) for y in range(H)]
                how = "exit-" + {"north": "up", "south": "down", "west": "left", "east": "right"}[direction]
                for ex, ey, lx, ly in edge:
                    if self.passable(mp, ex, ey) and self.passable(dm, lx, ly):
                        fwd.setdefault((mp, ex, ey), []).append(((dm, lx, ly), how))
        return fwd

    def _targets(self, leg):
        tgt = shaping.TARGETS[leg]
        if isinstance(tgt, tuple):
            return [tgt]
        g = self.maps[tgt]
        return [(tgt, x, y) for y in range(g["h"]) for x in range(g["w"]) if (tgt, x, y) in self.fwd]

    def _bfs(self, leg):
        dist = {}
        q = deque()
        for t in self._targets(leg):
            dist[t] = 0
            q.append(t)
        while q:
            v = q.popleft()
            for u in self.rev.get(v, ()):
                if u not in dist:
                    dist[u] = dist[v] + 1
                    q.append(u)
        return dist

    def d(self, leg, mp, x, y):
        """Walking distance in moves from (mp, x, y) to leg's target, or None (unwalkable, stale
        coordinates, outside the region, or no path)."""
        if leg not in self.dist:
            return None
        return self.dist[leg].get((mp, x, y))

    def best_moves(self, leg, mp, x, y):
        """Moves from (mp, x, y) sorted by the distance they lead to (greedy planner)."""
        out = []
        for v, how in self.fwd.get((mp, x, y), ()):
            dv = self.dist[leg].get(v)
            if dv is not None:
                out.append((dv, how, v))
        return sorted(out, key=lambda t: (t[0], t[1]))

    def path(self, leg, start):
        """Shortest path (list of nodes) from start to the leg's target, following d downhill."""
        cur, out = start, [start]
        while self.dist[leg].get(cur, None):
            nxt = min((self.dist[leg][v], v) for v, _ in self.fwd[cur] if v in self.dist[leg])
            cur = nxt[1]
            out.append(cur)
        return out

    def _reference(self):
        """Leg k's start on the reference path: D1's tile for leg 1, then where leg k-1's shortest path
        enters milestone k-1's map. d_leg(k) = d_k there."""
        starts = {1: shaping.D1_START}
        for k in range(1, len(ram.MILESTONES)):
            p = self.path(k, starts[k])
            starts[k + 1] = p[-1]
        d_leg = {k: self.d(k, *starts[k]) for k in starts}
        return starts, d_leg


# --- A2″ progress term -----------------------------------------------------------------------------
class Progress:
    """r_progress = max(0, d_min − d_t) / d_leg for the current leg (§9 "A2′ pilot" → A2″).

    d_t: walking distance to the next milestone; d_min: the best d reached so far in this leg of this
    episode (starts unset at reset and at each milestone, then takes the first usable d without pay).
    A d is usable when it's defined (walkable square in the region) and the map hasn't changed in the last
    SETTLE_STEPS steps (§9 (c): a warp updates wCurMap before wXCoord/wYCoord). On a milestone step the
    leg's remaining d_min is paid, as if d had reached 0, so a leg pays d_first / d_leg in total (= 1 from
    the leg's reference start) and the milestone itself is still ΔT. Loops, pacing and backtracking pay 0.
    """

    def __init__(self, wd, settle=SETTLE_STEPS):
        self.wd, self.settle = wd, settle

    def reset(self, mp, x, y, T):
        self.T, self.map, self.since = T, mp, self.settle + 1
        self.d_min = None
        self.paid = 0.0
        self._update(mp, x, y)

    def _update(self, mp, x, y):
        if self.T >= len(ram.MILESTONES) or self.since <= self.settle:
            return 0.0
        d = self.wd.d(self.T + 1, mp, x, y)
        if d is None:
            return 0.0
        if self.d_min is None:
            self.d_min = d
            return 0.0
        if d < self.d_min:
            gain = (self.d_min - d) / self.wd.d_leg[self.T + 1]
            self.d_min = d
            return gain
        return 0.0

    def step(self, mp, x, y, T):
        """Progress paid for arriving at (mp, x, y) with milestone count T (after this step)."""
        self.since = 0 if mp != self.map else self.since + 1
        self.map = mp
        r = 0.0
        if T > self.T:
            if self.d_min is not None and self.T < len(ram.MILESTONES):
                r += self.d_min / self.wd.d_leg[self.T + 1]       # the rest of the completed leg
            self.T, self.d_min = T, None
        r += self._update(mp, x, y)
        self.paid += r
        return r


def leg_progress(trace, wd, cap=3, settle=SETTLE_STEPS):
    """Secondary truth metric (experiment-e1.md §9 "A2″ 10M extension", pre-registered 2026-11-04):
    T + (1 − d_min / d_leg) at episode end, capped at `cap` (the segment end, M3).

    trace: the episode's per-step RAM rows (map, x, y, T), as env.trace / the saved eval traces. d_min is the
    best walking distance to the next milestone in the episode's last (unfinished) leg, tracked exactly as the
    A2″ Progress term does (first usable d, SETTLE_STEPS after a map change); the fraction is clamped to
    [0, 1] (a start further out than the leg's reference start counts as 0). Truth side only: computed from
    RAM position, never part of any proxy or of what the patcher sees (§3a).
    """
    if not trace:
        return 0.0
    p = Progress(wd, settle)
    r0 = trace[0]
    p.reset(r0["map"], r0["x"], r0["y"], r0["T"])
    for r in trace[1:]:
        p.step(r["map"], r["x"], r["y"], r["T"])
    if p.T >= cap:
        return float(cap)
    frac = 0.0 if p.d_min is None else min(1.0, max(0.0, 1.0 - p.d_min / wd.d_leg[p.T + 1]))
    return round(min(float(cap), p.T + frac), 4)
