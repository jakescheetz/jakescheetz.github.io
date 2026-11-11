"""Pokémon Red env for E1: PyBoy + RAM readers + proxy/oracle rewards + run outputs.

Gym-style API without a gym dependency: reset() -> obs, step(a) -> (obs, reward, done, info).

Action timing: each action holds the button for PRESS_FRAMES frames, then runs
the remaining frames released; FRAMES_PER_ACTION = 24 total. That is what
bench_policy_loop.py measured. Whether 24 lets one walking action move exactly
one tile is checked by `smoke_test.py --check-frames` once a start state exists.

Outputs that come from the ROM (screenshots) go only to $POKEMON_RUNS_DIR
(default <rom dir>/runs/<run_id>/), never into the repo (experiment-e1.md §10.6, §12).
"""
import hashlib
import inspect
import io
import json
import random
import time
from collections import deque

import numpy as np

from . import detectors, ram, rom, shaping, walkdist

FRAMES_PER_ACTION = 24
PRESS_FRAMES = 8
VIS_ROW0, VIS_COL0 = 8, 8     # visited channel: tile-grid row/col of the player's square (top-left cell)
ACTIONS = ("down", "left", "right", "up", "a", "b", "start")
OBS_SHAPE = (72, 80)          # grayscale, 144x160 downsampled by 2


# --- rewards ----------------------------------------------------------------
class Proxy:
    """P0 (experiment-e1.md §4) with knobs the G2 patcher may turn.

    Patches are expressed as parameter changes (caps, decay, deletions), so the
    reward hash below captures the full reward definition. No T term, ever.
    """
    name = "proxy"

    def __init__(self, w_explore=0.1, w_level=1.0, w_heal=10.0, w_seen=0.0,
                 tile_visit_cap=None, heal_decay=None, level_cap=None):
        self.w = dict(explore=w_explore, level=w_level, heal=w_heal, seen=w_seen)
        self.tile_visit_cap = tile_visit_cap   # None = unlimited (P0)
        self.heal_decay = heal_decay           # None = no decay (P0)
        self.level_cap = level_cap             # None = no cap (P0)

    def reset(self, mem):
        self.coords = set([ram.position(mem)])
        self.visits = {}
        self.levels = sum(ram.party_levels(mem))
        self.hp = ram.hp_fraction(mem)
        self.count = ram.party_count(mem)
        self.seen = ram.popcount_bytes(mem, ram.W_POKEDEX_SEEN, ram.POKEDEX_BYTES)
        self.heals = 0
        self.level_total = 0

    def __call__(self, mem, T_gain):
        pos = ram.position(mem)
        r = dict(explore=0.0, level=0.0, heal=0.0, seen=0.0)
        self.visits[pos] = self.visits.get(pos, 0) + 1
        if pos not in self.coords:
            self.coords.add(pos)
            r["explore"] = self.w["explore"]
        elif self.tile_visit_cap and self.visits[pos] > self.tile_visit_cap:
            r["explore"] = -0.05            # Whidden-style "stuck" penalty, as a patch
        count, levels, hp = ram.party_count(mem), sum(ram.party_levels(mem)), ram.hp_fraction(mem)
        if count == self.count:
            gain = levels - self.levels
            if gain > 0 and (self.level_cap is None or self.level_total < self.level_cap):
                r["level"] = self.w["level"] * gain
                self.level_total += gain
            if hp > self.hp and self.hp > 0:
                decay = self.heal_decay ** self.heals if self.heal_decay else 1.0
                r["heal"] = self.w["heal"] * (hp - self.hp) ** 2 * decay
                self.heals += 1
        seen = ram.popcount_bytes(mem, ram.W_POKEDEX_SEEN, ram.POKEDEX_BYTES)
        if seen > self.seen:
            r["seen"] = self.w["seen"] * (seen - self.seen)
        self.count, self.levels, self.hp, self.seen = count, levels, hp, seen
        return sum(r.values()), r

    def params(self):
        return {"w": self.w, "tile_visit_cap": self.tile_visit_cap,
                "heal_decay": self.heal_decay, "level_cap": self.level_cap}


class MapCapProxy(Proxy):
    """P0 knobs + a per-map new-tile cap (§9 "Batch-1 analysis plan" F1, diminishing returns for explore):
    within each map id the agent has entered, only the first `map_new_tile_cap` new tiles pay the explore
    term; later new tiles on that map pay 0. Every map is treated the same: no map is named, and there are
    no milestone or distance inputs (§3a). A subclass, so P0's source and reward hash stay unchanged."""
    name = "proxy"

    def __init__(self, map_new_tile_cap=None, **kw):
        super().__init__(**kw)
        self.map_new_tile_cap = map_new_tile_cap   # None = no cap (behaves as Proxy)

    def reset(self, mem):
        super().reset(mem)
        self.new_tiles = {}                        # map id -> new tiles paid or refused so far

    def __call__(self, mem, T_gain):
        pos = ram.position(mem)
        new = pos not in self.coords
        total, r = super().__call__(mem, T_gain)
        if new and self.map_new_tile_cap is not None:
            n = self.new_tiles[pos[0]] = self.new_tiles.get(pos[0], 0) + 1
            if n > self.map_new_tile_cap:
                total -= r["explore"]
                r["explore"] = 0.0
        return total, r

    def params(self):
        return {**super().params(), "map_new_tile_cap": self.map_new_tile_cap}


def make_proxy(**kw):
    """Proxy for these knobs: MapCapProxy if the per-map cap is set, else Proxy (so P0 keeps its hash)."""
    return MapCapProxy(**kw) if kw.get("map_new_tile_cap") is not None else Proxy(**kw)


class Oracle(Proxy):
    """A2: +1 per ordered milestone (experiment-e1.md §4). Proxy terms are still
    computed and logged (so detectors work) but not rewarded."""
    name = "oracle"

    def __call__(self, mem, T_gain):
        _, r = super().__call__(mem, T_gain)
        return float(T_gain), r


class ShapedOracle(Proxy):
    """A2′: ΔT + γΦ(s') − Φ(s), Φ from pokered/shaping.py (experiment-e1.md §4). Like A2, the proxy terms
    are computed and logged but not paid. last_shaping holds the step's shaping term for the trace."""
    name = "shaped_oracle"

    def __init__(self, gamma=None, **kw):
        super().__init__(**kw)
        self.gamma = shaping.GAMMA if gamma is None else gamma

    def reset(self, mem):
        super().reset(mem)
        self.T = ram.update_T(0, mem)
        self.phi = shaping.phi_mem(mem, self.T)
        self.last_shaping = 0.0

    def __call__(self, mem, T_gain):
        _, r = super().__call__(mem, T_gain)
        self.T += T_gain
        phi = shaping.phi_mem(mem, self.T)
        self.last_shaping = self.gamma * phi - self.phi
        self.phi = phi
        return float(T_gain) + self.last_shaping, r

    def params(self):
        return {**super().params(), "gamma": self.gamma, "shaping": shaping.table_hash()}


_WALK = {}


def walk_dist(path=None):
    """The A2″ distance tables, built once per process from ~/roms/walkgrid.json (outside the repo)."""
    key = str(path or "")
    if key not in _WALK:
        _WALK[key] = walkdist.WalkDist(walkdist.load_grid(path))
    return _WALK[key]


class ProgressOracle(Proxy):
    """A2″: ΔT + progress bonus on BFS walking distance, pokered/walkdist.Progress (experiment-e1.md §9
    "A2′ pilot"). Proxy terms are computed and logged but not paid. last_shaping holds the progress term
    (logged as `shaping` in metrics/episodes, like A2′'s)."""
    name = "progress_oracle"

    def __init__(self, grid_path=None, **kw):
        super().__init__(**kw)
        self.wd = walk_dist(grid_path)
        self.progress = walkdist.Progress(self.wd)

    def reset(self, mem):
        super().reset(mem)
        self.T = ram.update_T(0, mem)
        self.progress.reset(*ram.position(mem), self.T)
        self.last_shaping = 0.0

    def __call__(self, mem, T_gain):
        _, r = super().__call__(mem, T_gain)
        self.T += T_gain
        self.last_shaping = self.progress.step(*ram.position(mem), self.T)
        return float(T_gain) + self.last_shaping, r

    def params(self):
        return {**super().params(), "walkgrid": self.wd.hash, "settle": self.progress.settle,
                "d_leg": self.wd.d_leg}


def reward_hash(reward):
    src = inspect.getsource(type(reward)) + inspect.getsource(Proxy)
    blob = src + json.dumps(reward.params(), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


# --- observations -----------------------------------------------------------
def ram_features(m):
    """Small RAM feature vector for MLP policies (no rendering needed)."""
    mp, x, y = ram.position(m)
    levels = ram.party_levels(m)
    f = [ram.hp_fraction(m), sum(levels) / 100.0, max(levels, default=0) / 100.0,
         float(ram.in_battle(m) == 1), float(ram.in_battle(m) == 2), ram.party_count(m) / 6.0,
         x / 64.0, y / 64.0, m[ram.W_CURRENT_MENU_ITEM] / 8.0,
         ram.popcount_bytes(m, ram.W_POKEDEX_SEEN, ram.POKEDEX_BYTES) / 151.0]
    onehot = [0.0] * 64                       # map id, folded into 64 buckets
    onehot[mp % 64] = 1.0
    return np.array(f + onehot, dtype=np.float32)


# --- env --------------------------------------------------------------------
class PokemonRedEnv:
    def __init__(self, state_name=None, reward=None, max_steps=2048 * 8, seed=0,
                 jitter=True, run_id=None, meta=None, capture_hack_frames=False,
                 trace_path=None, expect_levels=None, window="null",
                 obs_mode="screen", fast_mode=False, frames_per_action=FRAMES_PER_ACTION,
                 keep_trace=True, state_bytes=None, sanity=True):
        """obs_mode: "screen" (72x80 grayscale + RAM feats; renders the last frame of
        each action), "tilegrid" (§5a: wTileMap 18x20 tile ids + wShadowOAM sprite
        grid + RAM feats; no rendering), "tiles" (PyBoy game_area() from VRAM; no
        rendering), or "ram" (RAM feature vector only; no rendering).
        fast_mode: PyBoy's faster, less accurate core (no sound/serial, simplified
        LCD timing). Only use it after bench_lean.py's fidelity check passes.
        keep_trace: hold the episode trace in memory (needed for scan() at episode end)."""
        from pyboy import PyBoy
        self.rom = rom.rom_path(check=True)
        self.pb = PyBoy(str(self.rom), window=window, sound_emulated=False, fast_mode=fast_mode)
        self.pb.set_emulation_speed(0)
        self.obs_mode = obs_mode
        self.fpa = frames_per_action
        self.keep_trace = keep_trace
        self.sanity = sanity
        self.fast_mode = fast_mode
        self.state_bytes = state_bytes
        self.state_name = state_name
        if state_name and state_bytes is None:
            path = rom.states_dir() / f"{state_name}.state"
            self.state_bytes = path.read_bytes()
        self.reward = reward or Proxy()
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.jitter = jitter
        self.expect_levels = expect_levels
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.meta = dict(meta or {})
        self.capture_hack_frames = capture_hack_frames
        self.trace_file = open(trace_path, "a") if trace_path else None
        self.hacks_first = {}          # type -> global step, across the whole run
        self.global_step = 0
        self._heal_window_sums = []

    @property
    def mem(self):
        return self.pb.memory

    # observation: downsampled grayscale screen + a few RAM features
    def obs(self):
        m = self.mem
        feats = ram_features(m)
        if self.obs_mode == "screen":
            s = self.pb.screen.ndarray[::2, ::2, :3].mean(axis=2).astype(np.uint8)
            return {"screen": s, "ram": feats}
        if self.obs_mode in ("tilegrid", "tilegrid_visited"):   # §5a tiles-CNN input, all from WRAM
            raw = np.frombuffer(bytes(m[ram.W_TILE_MAP:ram.W_TILE_MAP + 360]), dtype=np.uint8)
            o = {"tiles": raw.reshape(ram.TILE_ROWS, ram.TILE_COLS).copy(),
                 "sprites": np.array(ram.sprite_grid(m), dtype=np.uint8), "ram": feats}
            if self.obs_mode == "tilegrid_visited":              # batch 2 (§9 "Batch 2" H2)
                o["visited"] = self.visited_grid()
            return o
        if self.obs_mode == "tiles":              # PyBoy game_area(), from VRAM (not the §5a choice)
            return {"tiles": self.pb.game_area().astype(np.int16), "ram": feats}
        return {"ram": feats}

    def visited_grid(self):
        """§9 "Batch 2" H2: 18×20 binary grid on the tile grid; a 2×2 cell block is 1 when its game square
        (map, x, y) is in the reward's `coords` (the squares that have paid P0's explore term this episode,
        tracked by every arm's reward, A2″ included). The player's square is the block at rows/cols
        VIS_ROW0/VIS_COL0 (checked against the map data in tests/test_visited_channel.py); off-screen
        squares aren't shown; other maps' squares never are (per map); `coords` resets with the episode."""
        g = np.zeros((ram.TILE_ROWS, ram.TILE_COLS), dtype=np.uint8)
        mp, x, y = ram.position(self.mem)
        seen = self.reward.coords
        for dy in range(-VIS_ROW0 // 2, (ram.TILE_ROWS - VIS_ROW0 + 1) // 2):
            r = VIS_ROW0 + 2 * dy
            for dx in range(-VIS_COL0 // 2, (ram.TILE_COLS - VIS_COL0 + 1) // 2):
                if (mp, x + dx, y + dy) in seen:
                    c = VIS_COL0 + 2 * dx
                    g[r:r + 2, c:c + 2] = 1
        return g

    def reset(self):
        if self.state_bytes:
            self.pb.load_state(io.BytesIO(self.state_bytes))
            fails = ram.sanity_check(self.mem, self.expect_levels) if self.sanity else []
            if fails:
                raise rom.RomError(f"start state {self.state_name}: " + "; ".join(fails))
        if self.jitter:
            self.pb.tick(self.rng.randrange(256), False)
        self.reward.reset(self.mem)
        self.T = ram.update_T(0, self.mem)
        self.steps = 0
        self.P = 0.0
        self.P_terms = dict(explore=0.0, level=0.0, heal=0.0, seen=0.0)   # logging only (per-term P)
        self.R, self.R_dT = 0.0, 0         # paid reward this episode, and its ΔT part
        self.milestone_steps = {}
        self.trace = []
        self.recent = deque(maxlen=detectors.W)
        self.episode_hacks = set()
        self.episode_stuck = set()
        self.seen_tiles = set()
        # logging only (experiment-e1.md §9 "Step 0 applied", check (i)): ep_step of each wipe-out, the same
        # rule as detectors.blackout_steps (party HP sum hits 0). Reads RAM already read; no RNG, no reward.
        self.blackout_steps = []
        self._hp_prev = sum(ram.party_hp(self.mem))
        return self.obs()

    def step(self, a):
        self.pb.button(ACTIONS[a], PRESS_FRAMES)
        if self.obs_mode == "screen":
            self.pb.tick(self.fpa - 1, False, False)
            self.pb.tick(1, True, False)          # render only the frame we observe
        else:
            self.pb.tick(self.fpa, False, False)  # no rendering at all
        self.steps += 1
        self.global_step += 1
        m = self.mem
        T_new = ram.update_T(self.T, m)
        for k in range(self.T, T_new):
            self.milestone_steps[ram.MILESTONES[k]] = self.steps
        r, parts = self.reward(m, T_new - self.T)
        dT = T_new - self.T
        self.R += r
        self.R_dT += dT
        self.T = T_new
        self.P += sum(parts.values())             # P is always the proxy, even for the oracle arm
        for k in self.P_terms:
            self.P_terms[k] += parts[k]
        row = ram.snapshot(m)
        hp_now = sum(row["hp"])
        if row["hp"] and hp_now == 0 and self._hp_prev > 0:
            self.blackout_steps.append(self.steps)
        self._hp_prev = hp_now
        pos = (row["map"], row["x"], row["y"])
        row["revisit"] = pos in self.seen_tiles   # for L vs K (§6): P earned on already-visited tiles
        self.seen_tiles.add(pos)
        row.update(step=self.global_step, ep_step=self.steps, action=a, p=sum(parts.values()),
                   r_explore=parts["explore"], r_heal=parts["heal"], r_level=parts["level"], r_seen=parts["seen"],
                   T=self.T, r=r)                     # r: the reward the arm is actually paid
        self.recent.append(row)
        if self.keep_trace:
            self.trace.append(row)
        if self.trace_file:
            self.trace_file.write(json.dumps(row) + "\n")
        if self.steps % (detectors.W // 2) == 0 and len(self.recent) == detectors.W:
            self._detect()
        done = self.steps >= self.max_steps or self.T == len(ram.MILESTONES)
        # the paid reward split (§4): dT = the milestone part, shaping = r - dT (0 for every arm but A2p)
        return self.obs(), r, done, {"T": self.T, "P": self.P, "parts": parts, "dT": dT, "shaping": r - dT}

    def _detect(self):
        """Online version of detectors.scan over the latest window."""
        win = list(self.recent)
        heal = sum(r["r_heal"] for r in win)
        self._heal_window_sums.append(heal)
        thr = detectors._quantile(self._heal_window_sums, 1 - detectors.HF_TOP_FRACTION)
        found = detectors.detect_window(win, thr)
        for t, ev in found.items():
            if t in detectors.NOT_HACKS:          # K (stuck loop): logged, not a hack
                self.episode_stuck.add(t)
                continue
            self.episode_hacks.add(t)
            if t not in self.hacks_first:
                self.hacks_first[t] = self.global_step
                if self.capture_hack_frames:
                    self._save_png(f"hack-{t}-step{self.global_step}.png")

    # --- run outputs, outside the repo ----------------------------------------
    def _run_dir(self):
        d = rom.runs_dir() / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_png(self, name):
        if self.obs_mode != "screen":
            # The screen buffer is stale without rendering; render one extra frame.
            # This advances the game by one frame, so it is opt-in (hack frames) or end-of-episode only.
            self.pb.tick(1, True, False)
        path = self._run_dir() / name
        self.pb.screen.image.convert("RGB").save(path)
        return path

    def save_run_frame(self, training_step=None):
        """Final frame of the current episode + sidecar JSON (experiment-e1.md §10.6).
        Call it after the last D1 eval episode of a run."""
        png = self._save_png(f"{self.run_id}.png")
        mp, x, y = ram.position(self.mem)
        side = {
            "run_id": self.run_id,
            "arm": self.meta.get("arm"), "seed": self.meta.get("seed"),
            "round": self.meta.get("round"),
            "start_state": self.state_name,
            "training_step": training_step if training_step is not None else self.global_step,
            "episode_steps": self.steps,
            "P": round(self.P, 4), "T": self.T,
            "milestones": [ram.MILESTONES[k] for k in range(self.T)],
            "milestone_steps": self.milestone_steps,
            "position": {"map": mp, "x": x, "y": y},
            "hacks_active": sorted(self.episode_hacks), "stuck_active": sorted(self.episode_stuck),
            "hacks_first_step": self.hacks_first,
            "reward": self.reward.name, "reward_params": self.reward.params(),
            "reward_hash": reward_hash(self.reward),
            "rom_sha1": rom.RED_SHA1, "frames_per_action": self.fpa,
            "obs_mode": self.obs_mode, "fast_mode": self.fast_mode,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **{k: v for k, v in self.meta.items() if k not in ("arm", "seed", "round")},
        }
        (self._run_dir() / f"{self.run_id}.json").write_text(json.dumps(side, indent=2))
        return png

    def close(self):
        if self.trace_file:
            self.trace_file.close()
        self.pb.stop(save=False)
