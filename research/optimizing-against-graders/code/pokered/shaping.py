"""A2′ shaped oracle: potential Φ(map, x, y, T) from the segment's map geometry (experiment-e1.md §4).

    r_t = ΔT_t + γ Φ(s_{t+1}) − Φ(s_t),   γ = 0.998 (the PPO γ)

Φ uses only the segment's truth (milestone order + map geometry from the pret disassembly), no proxy
term. Per leg k (T = k-1 → milestone k):

    d_k(map, x, y)   route distance in tiles to the milestone-k map (M6: to the tile below Brock):
                     on the current map, Manhattan to an exit (warp tile or connected edge tile) + 1
                     step through it, then the precomputed distance from where that exit lands.
                     Off-route maps route back through their exits the same way (Dijkstra over every
                     exit of every map in the region, run backwards from the target at import).
    d_leg(k)         d_k at the point where leg k starts on the reference path (leg 1: D1's start
                     tile in Oak's Lab; leg k > 1: where leg k-1's shortest path enters milestone
                     k-1's map), so each leg is worth about 1, like a milestone.

    Φ(s) = −[(5 − T) + min(d_{T+1}(s) / d_leg(T+1), D_CAP)]   for T < 6,    Φ = 0 at T = 6.

The (5 − T) term is the number of legs left after the current one. It is a function of T, which is
part of the state, so Φ is still a state potential and the Ng et al. guarantee holds; it makes Φ
continuous across a milestone (Φ ≈ −(5 − T) just before, −(5 − (T+1)) − ≈1 just after), so the
milestone step still pays ≈ +1 net instead of ΔT (+1) being cancelled by Φ jumping down by 1.
[Deviation from the §4 text Φ = −d/d_leg, flagged for the researcher; both give the same optimal
policies, they differ only in where the reward lands.]

Bounded: Φ ∈ [−(5 + D_CAP), 0]. Maps outside the region (none are reachable in the segment without
HMs or badges, by pret) get d/d_leg = D_CAP.

Map data: pret/pokered master (local copy, 2026-11-02): data/maps/headers/*.asm (connections and
their offsets), data/maps/objects/*.asm (warp_event x, y, dest map, dest warp id; LAST_MAP = the
outside map whose warp leads in), constants/map_constants.asm (map sizes in blocks; 1 block = 2×2
tiles). Coordinates are wXCoord/wYCoord tile coordinates, as in ram.position().

Known simplifications (all inference, for review; none of them can make a loop pay, see the tests):
  * Manhattan ignores walls, ledges, trees and NPCs (the §4 "known trap": it can cost learning speed,
    not validity).
  * An edge exit is every tile of the edge row/column that overlaps the connected map; whether each
    one is walkable is not checked.
  * Route 2 is two pieces without Cut: the south part (y >= ROUTE2_SPLIT_Y, entered from Viridian,
    holds the Viridian Forest south gate) and the north part (Pewter side, forest north gate, Diglett's
    Cave, trade house, Route 2 gate). They are treated as separate maps with no passage between them
    except through the Forest. The split row is my reading of the warp layout (south gate door at
    y=43, Route 2 gate's south door at y=39), not checked against the block data.
"""
import heapq

from . import ram

GAMMA = 0.998
D_CAP = 2.0
N_LEGS = len(ram.MILESTONES)           # 6

# --- pret map data (sizes in tiles = 2 × blocks) ----------------------------------------------------
PALLET, VIRIDIAN, PEWTER, ROUTE1, ROUTE2, ROUTE3 = 0x00, 0x01, 0x02, 0x0C, 0x0D, 0x0E
ROUTE21, ROUTE22 = 0x20, 0x21
REDS1F, REDS2F, BLUES, OAKS_LAB = 0x25, 0x26, 0x27, 0x28
V_CENTER, V_MART, V_SCHOOL, V_NICKNAME, V_GYM = 0x29, 0x2A, 0x2B, 0x2C, 0x2D
DIGLETT_R2, NORTH_GATE, R2_TRADE, R2_GATE, SOUTH_GATE, FOREST = 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33
MUSEUM1F, MUSEUM2F, GYM, P_NIDORAN, P_MART, P_SPEECH, P_CENTER = 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A
ROUTE22_GATE = 0xC1

SIZE_BLOCKS = {  # map_constants.asm: width, height in blocks
    PALLET: (10, 9), VIRIDIAN: (20, 18), PEWTER: (20, 18), ROUTE1: (10, 18), ROUTE2: (10, 36),
    ROUTE3: (35, 9), ROUTE21: (10, 45), ROUTE22: (20, 9), REDS1F: (4, 4), REDS2F: (4, 4), BLUES: (4, 4),
    OAKS_LAB: (5, 6), V_CENTER: (7, 4), V_MART: (4, 4), V_SCHOOL: (4, 4), V_NICKNAME: (4, 4),
    V_GYM: (10, 9), DIGLETT_R2: (4, 4), NORTH_GATE: (5, 4), R2_TRADE: (4, 4), R2_GATE: (5, 4),
    SOUTH_GATE: (5, 4), FOREST: (17, 24), MUSEUM1F: (10, 4), MUSEUM2F: (7, 4), GYM: (5, 7),
    P_NIDORAN: (4, 4), P_MART: (4, 4), P_SPEECH: (4, 4), P_CENTER: (7, 4), ROUTE22_GATE: (5, 4),
}
SIZE = {m: (2 * w, 2 * h) for m, (w, h) in SIZE_BLOCKS.items()}

# headers/*.asm: (map, direction) -> (dest map, offset in blocks). Only connections inside the region;
# Route 21 -> Cinnabar (Surf), Route 3 -> Route 4 and Route 22 -> Route 23 are left out (unreachable).
CONNECTIONS = {
    (PALLET, "north"): (ROUTE1, 0), (PALLET, "south"): (ROUTE21, 0),
    (ROUTE1, "north"): (VIRIDIAN, -5), (ROUTE1, "south"): (PALLET, 0),
    (VIRIDIAN, "north"): (ROUTE2, 5), (VIRIDIAN, "south"): (ROUTE1, 5), (VIRIDIAN, "west"): (ROUTE22, 4),
    (ROUTE2, "north"): (PEWTER, -5), (ROUTE2, "south"): (VIRIDIAN, -5),
    (PEWTER, "south"): (ROUTE2, 5), (PEWTER, "east"): (ROUTE3, 4),
    (ROUTE3, "west"): (PEWTER, -4),
    (ROUTE21, "north"): (PALLET, 0),
    (ROUTE22, "east"): (VIRIDIAN, -4),
}

# objects/*.asm: map -> [(x, y, dest map or None for LAST_MAP, dest warp id, 1-based)]
WARPS = {
    PALLET: [(5, 5, REDS1F, 1), (13, 5, BLUES, 1), (12, 11, OAKS_LAB, 2)],
    OAKS_LAB: [(4, 11, None, 3), (5, 11, None, 3)],
    REDS1F: [(2, 7, None, 1), (3, 7, None, 1), (7, 1, REDS2F, 1)],
    REDS2F: [(7, 1, REDS1F, 3)],
    BLUES: [(2, 7, None, 2), (3, 7, None, 2)],
    VIRIDIAN: [(23, 25, V_CENTER, 1), (29, 19, V_MART, 1), (21, 15, V_SCHOOL, 1), (21, 9, V_NICKNAME, 1),
               (32, 7, V_GYM, 1)],
    V_CENTER: [(3, 7, None, 1), (4, 7, None, 1)],
    V_MART: [(3, 7, None, 2), (4, 7, None, 2)],
    V_SCHOOL: [(2, 7, None, 3), (3, 7, None, 3)],
    V_NICKNAME: [(2, 7, None, 4), (3, 7, None, 4)],
    V_GYM: [(16, 17, None, 5), (17, 17, None, 5)],
    ROUTE22: [(8, 5, ROUTE22_GATE, 1)],
    ROUTE22_GATE: [(4, 7, None, 1), (5, 7, None, 1)],      # its north door (Route 23) needs badges
    ROUTE2: [(12, 9, DIGLETT_R2, 1), (3, 11, NORTH_GATE, 2), (15, 19, R2_TRADE, 1), (16, 35, R2_GATE, 2),
             (15, 39, R2_GATE, 3), (3, 43, SOUTH_GATE, 3)],
    DIGLETT_R2: [(2, 7, None, 1), (3, 7, None, 1)],        # (4, 4) -> Diglett's Cave: out of the region
    R2_TRADE: [(2, 7, None, 3), (3, 7, None, 3)],
    R2_GATE: [(4, 0, None, 4), (5, 0, None, 4), (4, 7, None, 5), (5, 7, None, 5)],
    SOUTH_GATE: [(4, 0, FOREST, 4), (5, 0, FOREST, 5), (4, 7, None, 6), (5, 7, None, 6)],
    FOREST: [(1, 0, NORTH_GATE, 3), (2, 0, NORTH_GATE, 4), (15, 47, SOUTH_GATE, 2), (16, 47, SOUTH_GATE, 2),
             (17, 47, SOUTH_GATE, 2), (18, 47, SOUTH_GATE, 2)],
    NORTH_GATE: [(4, 0, None, 2), (5, 0, None, 2), (4, 7, FOREST, 1), (5, 7, FOREST, 1)],
    PEWTER: [(14, 7, MUSEUM1F, 1), (19, 5, MUSEUM1F, 3), (16, 17, GYM, 1), (29, 13, P_NIDORAN, 1),
             (23, 17, P_MART, 1), (7, 29, P_SPEECH, 1), (13, 25, P_CENTER, 1)],
    MUSEUM1F: [(10, 7, None, 1), (11, 7, None, 1), (16, 7, None, 2), (17, 7, None, 2), (7, 7, MUSEUM2F, 1)],
    MUSEUM2F: [(7, 7, MUSEUM1F, 5)],
    GYM: [(4, 13, None, 3), (5, 13, None, 3)],
    P_NIDORAN: [(2, 7, None, 4), (3, 7, None, 4)],
    P_MART: [(3, 7, None, 5), (4, 7, None, 5)],
    P_SPEECH: [(2, 7, None, 6), (3, 7, None, 6)],
    P_CENTER: [(3, 7, None, 7), (4, 7, None, 7)],
}
BROCK_FRONT = (GYM, 4, 2)               # Brock stands at (4, 1) facing down (objects/PewterGym.asm)

# Route 2 is two maps without Cut (see the module docstring).
ROUTE2_SPLIT_Y = 40
R2N, R2S = ("R2", "north"), ("R2", "south")

# leg k targets: entering this map completes milestone k (ram.milestone_hits); leg 6 is Brock's tile
LEG_TARGET = {1: VIRIDIAN, 2: ROUTE2, 3: FOREST, 4: PEWTER, 5: GYM, 6: None}
D1_START = (OAKS_LAB, 5, 3)             # D1's tile (start-states log; first row of script-trace.jsonl)


def node(mp, y=None):
    """The graph map for a (map, y): Route 2 splits into its two pieces."""
    if mp == ROUTE2 and y is not None:
        return R2S if y >= ROUTE2_SPLIT_Y else R2N
    return mp


def raw(gm):
    return ROUTE2 if gm in (R2N, R2S) else gm


def _parent_warp(mp, warp_id):
    """LAST_MAP: the outside map whose warp leads into mp; dest warp id counts on that map."""
    for outer, ws in WARPS.items():
        if any(d == mp for _, _, d, _ in ws):
            return outer, ws[warp_id - 1][:2]
    raise KeyError(f"no outside map leads into {mp:#x}")


def exits(mp):
    """[(x, y, dest map, dest x, dest y)] for every way off map mp (raw map id)."""
    out = []
    for x, y, dest, wid in WARPS.get(mp, []):
        if dest is None:
            dmap, (dx, dy) = _parent_warp(mp, wid)
        else:
            dmap, (dx, dy) = dest, WARPS[dest][wid - 1][:2]
        out.append((x, y, dmap, dx, dy))
    w, h = SIZE[mp]
    for (src, direction), (dmap, off) in CONNECTIONS.items():
        if src != mp:
            continue
        dw, dh = SIZE[dmap]
        if direction in ("north", "south"):
            for x in range(w):
                dx = x - 2 * off            # pret offset in blocks; Route 1 -> Viridian (-5): x + 10
                if 0 <= dx < dw:
                    out.append((x, 0, dmap, dx, dh - 1) if direction == "north" else (x, h - 1, dmap, dx, 0))
        else:
            for y in range(h):
                dy = y - 2 * off
                if 0 <= dy < dh:
                    out.append((0, y, dmap, dw - 1, dy) if direction == "west" else (w - 1, y, dmap, 0, dy))
    return out


def _all_exits():
    """graph map -> [(x, y, dest graph map, dest x, dest y)], with Route 2 split by y."""
    g = {}
    for mp in SIZE:
        for x, y, dmap, dx, dy in exits(mp):
            g.setdefault(node(mp, y), []).append((x, y, node(dmap, dy), dx, dy))
    return g


EXITS = _all_exits()


def _leg_tables(target):
    """Dijkstra backwards from the target: cost[(graph map, x, y) arrival point] = tiles to the target
    from there; then per graph map, [(exit x, exit y, 1 + cost of where it lands)]."""
    arrivals = {(dm, dx, dy) for es in EXITS.values() for _, _, dm, dx, dy in es}

    def here(gm, x, y):          # distance to the target if standing on (gm, x, y), not leaving gm
        if target is None:
            return None
        if isinstance(target, tuple):
            tm, tx, ty = target
            return abs(x - tx) + abs(y - ty) if gm == tm else None
        return 0 if raw(gm) == target else None

    cost = {}
    pq = []
    tie = iter(range(10 ** 9))     # heap tie-break: graph maps mix ints and Route 2's tuples
    for a in arrivals:
        h = here(*a)
        if h is not None:
            cost[a] = h
            heapq.heappush(pq, (h, next(tie), a))
    # reverse edges: an arrival point p on map gm reaches exit e of gm at Manhattan cost, then e's landing
    by_map = {}
    for a in arrivals:
        by_map.setdefault(a[0], []).append(a)
    into = {}                      # landing point -> [(map, exit x, exit y)] that land there
    for gm, es in EXITS.items():
        for x, y, dm, dx, dy in es:
            into.setdefault((dm, dx, dy), []).append((gm, x, y))
    while pq:
        c, _, land = heapq.heappop(pq)
        if c > cost.get(land, 1e18):
            continue
        for gm, ex, ey in into.get(land, []):
            for a in by_map.get(gm, []):
                nc = c + 1 + abs(a[1] - ex) + abs(a[2] - ey)
                if nc < cost.get(a, 1e18):
                    cost[a] = nc
                    heapq.heappush(pq, (nc, next(tie), a))
    table = {}
    for gm, es in EXITS.items():
        table[gm] = [(x, y, 1 + cost[(dm, dx, dy)]) for x, y, dm, dx, dy in es if (dm, dx, dy) in cost]
    return table


TARGETS = {k: (BROCK_FRONT if k == 6 else LEG_TARGET[k]) for k in range(1, N_LEGS + 1)}
TABLES = {k: _leg_tables(TARGETS[k]) for k in TARGETS}


def distance(leg, mp, x, y):
    """d_leg(map, x, y) in tiles, or None if the map is outside the region."""
    gm = node(mp, y)
    tgt = TARGETS[leg]
    if isinstance(tgt, tuple):
        if gm == tgt[0]:
            return abs(x - tgt[1]) + abs(y - tgt[2])
    elif raw(gm) == tgt:
        return 0
    es = TABLES[leg].get(gm)
    if not es:
        return None
    return min(abs(x - ex) + abs(y - ey) + c for ex, ey, c in es)


def _reference_starts():
    """Leg k's start on the reference path: D1_START for leg 1, then where leg k-1's shortest path
    lands on milestone k-1's map."""
    starts = {1: D1_START}
    for k in range(1, N_LEGS):
        mp, x, y = starts[k]
        while True:
            gm = node(mp, y)
            best = min(((abs(x - ex) + abs(y - ey) + 1 + _land_cost(k, dm, dx, dy), i, (dm, dx, dy))
                        for i, (ex, ey, dm, dx, dy) in enumerate(EXITS[gm]) if _land_cost(k, dm, dx, dy) is not None))
            dm, dx, dy = best[2]
            mp = raw(dm)
            x, y = dx, dy
            if distance(k, mp, x, y) == 0:
                starts[k + 1] = (mp, x, y)
                break
    return starts


def _land_cost(leg, gm, x, y):
    return distance(leg, raw(gm), x, y)


LEG_START = _reference_starts()
D_LEG = {k: distance(k, *LEG_START[k]) for k in LEG_START}


def phi(mp, x, y, T):
    if T >= N_LEGS:
        return 0.0
    leg = T + 1
    if mp in SIZE:
        # a warp updates wCurMap 1-2 env steps before wXCoord/wYCoord (seen in script-trace.jsonl), so
        # for those rows the coordinates belong to the old map; clamp them into the new one. Still a
        # function of the state, so it can't make a loop pay; it only shrinks the transient.
        w, h = SIZE[mp]
        x, y = min(x, w - 1), min(y, h - 1)
    d = distance(leg, mp, x, y)
    frac = D_CAP if d is None else min(d / D_LEG[leg], D_CAP)
    return -((N_LEGS - 1 - T) + frac)


def phi_mem(mem, T):
    return phi(*ram.position(mem), T)


def table_hash():
    """Hash of the map data, leg lengths and constants, so reward_hash changes if Φ does."""
    import hashlib
    import json
    blob = json.dumps({"size": sorted(SIZE.items()), "conn": sorted((f"{k}", v) for k, v in CONNECTIONS.items()),
                       "warps": sorted(WARPS.items()), "brock": BROCK_FRONT, "split": ROUTE2_SPLIT_Y,
                       "d_leg": sorted(D_LEG.items()), "cap": D_CAP, "start": D1_START}, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]
