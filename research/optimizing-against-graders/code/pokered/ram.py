"""Pokémon Red RAM map and pure readers (experiment-e1.md §3).

Every address and constant below was checked against pret/pokered:
  - symbols branch bd77855a9f64, pokered.sym (absolute WRAM addresses)
  - master 0dbf95cd12f5, constants/map_constants.asm and constants/event_constants.asm
    (event indices recomputed by expanding const / const_skip / const_next)
  https://github.com/pret/pokered

Readers take `mem`, anything indexable by address that returns a byte (a PyBoy
`pyboy.memory`, a bytes/bytearray of the full 64 KiB address space, or a dict).
They have no side effects, so the same code runs on the live emulator and on
logged traces.

Pitfalls handled here (experiment-e1.md §3):
  1. Event flag n is at wEventFlags + n // 8, bit n % 8, LSB-first (pret CheckEvent).
  2. HP is 2-byte big-endian; party struct stride is 0x2C.
"""

# --- WRAM addresses (pokered.sym) -----------------------------------------
W_CURRENT_MENU_ITEM = 0xCC26
W_BATTLE_MON_HP = 0xD015
W_IS_IN_BATTLE = 0xD057      # 0 none, 1 wild, 2 trainer, 0xFF lost
W_PARTY_COUNT = 0xD163
W_PARTY_MON1 = 0xD16B
W_PARTY_MON1_HP = 0xD16C     # 2 bytes, big-endian
W_PARTY_MON1_STATUS = 0xD16F # wPartyMon1Status (party_struct: species, hp(2), box level, status)
W_PARTY_MON1_LEVEL = 0xD18C
W_PARTY_MON1_MAX_HP = 0xD18D  # 2 bytes, big-endian
PARTY_STRIDE = 0x2C          # wPartyMon2 (0xD197) - wPartyMon1 (0xD16B)
W_POKEDEX_OWNED = 0xD2F7     # 19 bytes bitfield
W_POKEDEX_SEEN = 0xD30A      # 19 bytes bitfield
POKEDEX_BYTES = 19
W_PLAYER_MONEY = 0xD347      # 3 bytes BCD
W_OBTAINED_BADGES = 0xD356
W_CUR_MAP = 0xD35E
W_Y_COORD = 0xD361
W_X_COORD = 0xD362
W_LAST_BLACKOUT_MAP = 0xD719
W_EVENT_FLAGS = 0xD747       # 320 bytes, 2560 flags, ends before wGrassRate 0xD887
W_SHADOW_OAM = 0xC300        # 40 sprites x 4 bytes (y, x, tile, attr), copied to OAM by DMA
W_TILE_MAP = 0xC3A0          # 20 x 18 tiles visible on screen (text and menus are drawn here too)
TILE_COLS, TILE_ROWS = 20, 18
H_RANDOM_ADD = 0xFFD3        # RNG state (HRAM), stirred from rDIV in Random_
H_RANDOM_SUB = 0xFFD4

# --- Map IDs (map_constants.asm) -------------------------------------------
PALLET_TOWN = 0x00
VIRIDIAN_CITY = 0x01
PEWTER_CITY = 0x02
ROUTE_1 = 0x0C
ROUTE_2 = 0x0D
ROUTE_22 = 0x21
REDS_HOUSE_1F = 0x25
OAKS_LAB = 0x28
VIRIDIAN_POKECENTER = 0x29
VIRIDIAN_FOREST = 0x33
PEWTER_GYM = 0x36
PEWTER_POKECENTER = 0x3A
POKECENTERS = frozenset({VIRIDIAN_POKECENTER, PEWTER_POKECENTER})
GRASS_MAPS = frozenset({ROUTE_1, ROUTE_2, VIRIDIAN_FOREST})

# --- Event flag indices (event_constants.asm) ------------------------------
EVENT_GOT_POKEDEX = 37
EVENT_BEAT_PEWTER_GYM_TRAINER_0 = 114
EVENT_GOT_TM34 = 118
EVENT_BEAT_BROCK = 119
EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_0 = 1378
EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_1 = 1379
EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_2 = 1380
BIT_BOULDERBADGE = 0


# --- Primitive readers ------------------------------------------------------
def u16be(mem, addr):
    return 256 * mem[addr] + mem[addr + 1]


def event_flag(mem, n):
    """Event flag n, LSB-first: byte wEventFlags + n//8, bit n%8."""
    return bool((mem[W_EVENT_FLAGS + n // 8] >> (n % 8)) & 1)


def popcount_bytes(mem, addr, n):
    return sum(bin(mem[addr + i]).count("1") for i in range(n))


def bcd(mem, addr, n):
    v = 0
    for i in range(n):
        b = mem[addr + i]
        v = v * 100 + (b >> 4) * 10 + (b & 0xF)
    return v


# --- Game-state readers -----------------------------------------------------
def position(mem):
    return mem[W_CUR_MAP], mem[W_X_COORD], mem[W_Y_COORD]


def party_count(mem):
    return min(mem[W_PARTY_COUNT], 6)


def party_levels(mem):
    return [mem[W_PARTY_MON1_LEVEL + PARTY_STRIDE * k] for k in range(party_count(mem))]


def party_hp(mem):
    return [u16be(mem, W_PARTY_MON1_HP + PARTY_STRIDE * k) for k in range(party_count(mem))]


def party_max_hp(mem):
    return [u16be(mem, W_PARTY_MON1_MAX_HP + PARTY_STRIDE * k) for k in range(party_count(mem))]


def hp_fraction(mem):
    hp, mx = sum(party_hp(mem)), sum(party_max_hp(mem))
    return hp / mx if mx else 0.0


def badges(mem):
    return mem[W_OBTAINED_BADGES]


def in_battle(mem):
    return mem[W_IS_IN_BATTLE]


# --- Tile-grid observation (experiment-e1.md §5a) ---------------------------
def tile_grid(mem):
    """wTileMap as 18 rows x 20 columns of tile ids."""
    return [[mem[W_TILE_MAP + r * TILE_COLS + c] for c in range(TILE_COLS)] for r in range(TILE_ROWS)]


def sprite_grid(mem):
    """wShadowOAM rasterised onto the 20x18 tile grid: cell = number of 8x8 sprite
    tiles covering it. OAM y is screen y + 16 and x is screen x + 8; y = 0 or >= 160
    (and x = 0 or >= 168) means hidden."""
    g = [[0] * TILE_COLS for _ in range(TILE_ROWS)]
    for k in range(40):
        y, x = mem[W_SHADOW_OAM + 4 * k], mem[W_SHADOW_OAM + 4 * k + 1]
        if not (0 < y < 160 and 0 < x < 168):
            continue
        r, c = (y - 16) // 8, (x - 8) // 8
        if 0 <= r < TILE_ROWS and 0 <= c < TILE_COLS:
            g[r][c] += 1
    return g


# --- True metric T (§3a). Never used by any proxy or the patcher. ------------
MILESTONES = ("M1_viridian", "M2_route2", "M3_forest", "M4_pewter", "M5_gym", "M6_brock")


def milestone_hits(mem):
    """Which milestone conditions hold *right now* (unordered)."""
    m = mem[W_CUR_MAP]
    return {
        "M1_viridian": m == VIRIDIAN_CITY,
        "M2_route2": m == ROUTE_2,
        "M3_forest": m == VIRIDIAN_FOREST,
        "M4_pewter": m == PEWTER_CITY,
        "M5_gym": m == PEWTER_GYM,
        "M6_brock": event_flag(mem, EVENT_BEAT_BROCK)
                    and bool((badges(mem) >> BIT_BOULDERBADGE) & 1),
    }


def update_T(reached, mem):
    """Ordered milestone count. `reached` is the count so far this episode;
    a milestone counts only once all earlier ones are reached."""
    hits = milestone_hits(mem)
    while reached < len(MILESTONES) and hits[MILESTONES[reached]]:
        reached += 1
    return reached


# --- Snapshot: one row of a RAM trace ---------------------------------------
def snapshot(mem):
    """Everything §3 says to log, as a flat JSON-friendly dict."""
    mp, x, y = position(mem)
    return {
        "map": mp, "x": x, "y": y,
        "party_count": party_count(mem),
        "levels": party_levels(mem),
        "hp": party_hp(mem),
        "max_hp": party_max_hp(mem),
        "in_battle": in_battle(mem),
        "blackout_map": mem[W_LAST_BLACKOUT_MAP],
        "menu_item": mem[W_CURRENT_MENU_ITEM],
        "badges": badges(mem),
        "seen": popcount_bytes(mem, W_POKEDEX_SEEN, POKEDEX_BYTES),
        "owned": popcount_bytes(mem, W_POKEDEX_OWNED, POKEDEX_BYTES),
        "money": bcd(mem, W_PLAYER_MONEY, 3),
        "ev_pokedex": event_flag(mem, EVENT_GOT_POKEDEX),
        "ev_brock": event_flag(mem, EVENT_BEAT_BROCK),
        "ev_tm34": event_flag(mem, EVENT_GOT_TM34),
        "ev_gym_trainer0": event_flag(mem, EVENT_BEAT_PEWTER_GYM_TRAINER_0),
        "ev_forest_trainers": [event_flag(mem, n) for n in (
            EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_0,
            EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_1,
            EVENT_BEAT_VIRIDIAN_FOREST_TRAINER_2)],
    }


# --- Start-state sanity (§2) ------------------------------------------------
def money(mem):
    return bcd(mem, W_PLAYER_MONEY, 3)


def sanity_check(mem, expect_levels=None, expect_money=None, full_hp=True):
    """Return a list of failure strings; empty means the state is usable.
    full_hp (§2 start-condition rules, 2026-11-02): every party mon at max HP, no status."""
    fails = []
    if full_hp:
        hp, mx = party_hp(mem), party_max_hp(mem)
        if hp != mx:
            fails.append(f"party HP {hp} != max {mx}")
        st = [mem[W_PARTY_MON1_STATUS + PARTY_STRIDE * k] for k in range(party_count(mem))]
        if any(st):
            fails.append(f"party status {st}, expected none")
    if expect_money is not None and money(mem) != expect_money:
        fails.append(f"money {money(mem)} != recorded {expect_money}")
    if not event_flag(mem, EVENT_GOT_POKEDEX):
        fails.append("EVENT_GOT_POKEDEX not set")
    if badges(mem) != 0:
        fails.append(f"wObtainedBadges = {badges(mem):#04x}, expected 0")
    if party_count(mem) < 1:
        fails.append("party is empty")
    if expect_levels is not None and party_levels(mem) != list(expect_levels):
        fails.append(f"party levels {party_levels(mem)} != recorded {list(expect_levels)}")
    return fails
