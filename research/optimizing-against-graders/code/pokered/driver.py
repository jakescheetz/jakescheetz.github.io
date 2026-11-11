"""Scripted play for Pokémon Red: button presses, on-screen text, map-wide pathing.

Used by script_states.py to make the start states without hand play
(experiment-e1.md §2 "Scripted states", §9.1). Nothing here is used by the trainer.

Inputs go through one primitive, `act(button)`: press for PRESS_FRAMES, then run
to FRAMES_PER_ACTION frames. That is exactly an env step (env.py), so the same
driver can steer PyBoy directly (state making) or a PokemonRedEnv (the §9.1 trace,
whose rows are then the trainer's own trace rows).

Addresses are pret pokered.sym (symbols branch bd77855a9f64), checked 2026-11-02:
  wOverworldMap c6e8, wCurMapHeight d368, wCurMapWidth d369, wTilesetBank d52b,
  wTilesetBlocksPtr d52c, wTilesetCollisionPtr d530 (lists live in bank 0, e.g.
  Overworld_Coll 00:1735), wGrassTile d535, wNumberOfWarps d3ae, wWarpEntries d3af,
  wSpriteStateData1 c100, wSpriteStateData2 c200, wJoyIgnore cd6b, wTileMap c3a0.
Map walkability follows the game's own rule: a 16x16 step square is passable when
its lower-left 8x8 tile is in the tileset's collision list (pret home/overworld.asm).
"""
import heapq

from . import ram

W_OVERWORLD_MAP = 0xC6E8
W_CUR_MAP_HEIGHT = 0xD368
W_CUR_MAP_WIDTH = 0xD369
W_TILESET_BANK = 0xD52B
W_TILESET_BLOCKS_PTR = 0xD52C
W_TILESET_COLLISION_PTR = 0xD530
W_GRASS_TILE = 0xD535
W_NUMBER_OF_WARPS = 0xD3AE
W_WARP_ENTRIES = 0xD3AF
W_SPRITE_STATE_1 = 0xC100
W_SPRITE_STATE_2 = 0xC200
W_JOY_IGNORE = 0xCD6B
W_PARTY_SPECIES = 0xD164
W_PARTY_MON1_EXP = 0xD179     # 3 bytes, big-endian
MAP_BORDER = 3                # wOverworldMap has a 3-block border on every side

PRESS_FRAMES = 8
FRAMES_PER_ACTION = 24
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

# pret charmap.asm, enough to read dialogue and menus
CHARS = {0x7F: " ", 0xE8: ".", 0xE7: "!", 0xE6: "?", 0xF4: ",", 0xE3: "-", 0xBA: "é",
         0xED: ">", 0xEE: "v", 0xE0: "'", 0x9C: ":", 0xF3: "/", 0xEF: "M", 0xF5: "F"}
CHARS.update({0x80 + i: chr(65 + i) for i in range(26)})
CHARS.update({0xA0 + i: chr(97 + i) for i in range(26)})
CHARS.update({0xF6 + i: str(i) for i in range(10)})

SPECIES = {0xB1: "SQUIRTLE", 0xB0: "CHARMANDER", 0x99: "BULBASAUR"}   # pret constants/pokemon_constants.asm


class ScriptError(RuntimeError):
    pass


class Driver:
    def __init__(self, pb, step_fn=None, log=print):
        """pb: a PyBoy. step_fn(button) overrides `act` (e.g. env.step for the trace)."""
        self.pb = pb
        self.step_fn = step_fn
        self.log = log
        self.actions = 0

    # --- primitives ---------------------------------------------------------
    @property
    def mem(self):
        return self.pb.memory

    def act(self, button):
        self.actions += 1
        if self.step_fn:
            return self.step_fn(button)
        self.pb.button(button, PRESS_FRAMES)
        self.pb.tick(FRAMES_PER_ACTION, False, False)

    def wait(self, n=1):
        """Idle n actions. The env has no no-op action, so in env mode this presses B,
        which does nothing in the overworld when no text is open."""
        if self.step_fn:
            for _ in range(n):
                self.act("b")
            return
        self.pb.tick(FRAMES_PER_ACTION * n, False, False)

    def text(self):
        m = self.mem
        rows = []
        for r in range(ram.TILE_ROWS):
            rows.append("".join(CHARS.get(m[ram.W_TILE_MAP + r * 20 + c], "#") for c in range(20)))
        return rows

    def screen_text(self):
        """Readable text on screen (dialogue box and menus), squashed to one line."""
        return " ".join(" ".join(r.replace("#", " ").split()) for r in self.text()).strip()

    def pos(self):
        return ram.position(self.mem)

    # --- map-wide walkability ----------------------------------------------
    def walk_grid(self):
        """passable[y][x] and grass[y][x] for the whole current map, in step squares."""
        m = self.mem
        h, w = m[W_CUR_MAP_HEIGHT], m[W_CUR_MAP_WIDTH]
        bank = m[W_TILESET_BANK]
        blocks = m[W_TILESET_BLOCKS_PTR] | m[W_TILESET_BLOCKS_PTR + 1] << 8
        coll = m[W_TILESET_COLLISION_PTR] | m[W_TILESET_COLLISION_PTR + 1] << 8
        ok = set()
        while m[coll] != 0xFF:
            ok.add(m[coll])
            coll += 1
        grass_tile = m[W_GRASS_TILE]
        stride = w + 2 * MAP_BORDER

        def tile(sx, sy):
            b = m[W_OVERWORLD_MAP + (sy // 2 + MAP_BORDER) * stride + sx // 2 + MAP_BORDER]
            r, c = (sy % 2) * 2 + 1, (sx % 2) * 2
            addr = blocks + b * 16 + r * 4 + c
            return m[bank, addr] if addr >= 0x4000 else m[addr]

        H, Wd = 2 * h, 2 * w
        passable = [[False] * Wd for _ in range(H)]
        grass = [[False] * Wd for _ in range(H)]
        for y in range(H):
            for x in range(Wd):
                t = tile(x, y)
                passable[y][x] = t in ok
                grass[y][x] = t == grass_tile
        return passable, grass

    def sprites(self):
        """Map squares occupied by NPC sprites (slot 0 is the player)."""
        m = self.mem
        out = set()
        for i in range(1, 16):
            if m[W_SPRITE_STATE_1 + 16 * i] == 0:          # picture id 0: empty slot
                continue
            y, x = m[W_SPRITE_STATE_2 + 16 * i + 4] - 4, m[W_SPRITE_STATE_2 + 16 * i + 5] - 4
            out.add((x, y))
        return out

    def warps(self):
        m = self.mem
        return [dict(y=m[W_WARP_ENTRIES + 4 * k], x=m[W_WARP_ENTRIES + 4 * k + 1],
                     warp=m[W_WARP_ENTRIES + 4 * k + 2], map=m[W_WARP_ENTRIES + 4 * k + 3])
                for k in range(m[W_NUMBER_OF_WARPS])]

    def plan(self, goal, grass_cost=8, allow_goal_blocked=True):
        """Dijkstra from the player to goal (x, y); grass squares cost extra so paths avoid
        wild encounters when there is another way. Returns a list of directions."""
        passable, grass = self.walk_grid()
        H, Wd = len(passable), len(passable[0])
        _, sx, sy = self.pos()
        block = self.sprites()
        start = (sx, sy)
        dist = {start: 0}
        prev = {}
        pq = [(0, start)]
        while pq:
            d, (x, y) = heapq.heappop(pq)
            if (x, y) == goal:
                break
            if d > dist[(x, y)]:
                continue
            for name, (dx, dy) in DIRS.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < Wd and 0 <= ny < H):
                    continue
                if (nx, ny) != goal or not allow_goal_blocked:
                    if not passable[ny][nx] or (nx, ny) in block:
                        continue
                nd = d + (grass_cost if grass[ny][nx] else 1)
                if nd < dist.get((nx, ny), 1 << 30):
                    dist[(nx, ny)] = nd
                    prev[(nx, ny)] = ((x, y), name)
                    heapq.heappush(pq, (nd, (nx, ny)))
        if goal not in dist:
            return None
        path, cur = [], goal
        while cur != start:
            cur, name = prev[cur]
            path.append(name)
        return path[::-1]
