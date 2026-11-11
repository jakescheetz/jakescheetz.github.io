# E1: Shaped proxy vs real progress, and the patch loop, in Pokémon Red

_Researcher, 2026-11-01. Design doc for the experiment Jake picked (brief.md, "Decision"): G1 + G2 in a Pokémon Red early-game segment. Background and sources are in `findings.md` and `notes.md`. Every RAM address below was checked against pret's disassembly (pinned commits in §3). The compute numbers in §8 are **placeholders until the developer's measured steps/s land in `code/README.md`**._

## 1. Questions and hypotheses

**Q1 (G1).** As we optimize harder against a hand-shaped proxy (exploration, levels, healing), does true story progress rise with it, flatten, or fall? Which hacks appear, and at what optimization strength?

**Q2 (G2).** When a designer patches each hack and retrains, does true progress converge toward the oracle's? Do new hack types keep appearing? Does a reward patched on *dev* start states generalize to *held-out* start states the designer never looked at?

Pre-registered hypotheses. Each comes with a result that would falsify it.

| ID | Hypothesis | Falsified if |
|---|---|---|
| **H0** | Under the initial proxy $P_0$, proxy return keeps rising with training steps, while true progress $T$ peaks and then flattens or falls (Goodhart). Prior: hacks have been reported in Pokémon [@pleines2025pokemon; @mudireddy2026pokerl], but never measured this way | $T$ rises monotonically with $P$ across all checkpoints and seeds (within CI) |
| **H0b** | Hacks appear in a consistent order across seeds: loops and pacing first; heal and grind farming once the agent reaches a Pokémon Center or grass. They first appear at roughly reproducible step counts | The order of first appearance is random across seeds |
| **H1** | Patching raises dev $T$ round over round, but held-out $T$ improves less, and the dev–held-out gap grows with rounds (designer overfitting, cf. @booth2023perils) | The gap stays within seed noise for all rounds |
| **H2** | Each patch round exposes ≥1 new hack type until the budget runs out, because $P$ omits the true objective by design [@zhuang2021consequences; @skalse2022defining] | No new hack types appear after round 2, and dev $T$ reaches the oracle's |
| **H3** *(speculative)* | At the same designer budget, early stopping on $P_0$ [@karwowski2023goodhart] matches or beats patching on held-out $T$ | Patching beats early stopping on held-out $T$ at every budget |

## 2. Segment and start states

**Segment: "Pokédex to Boulder Badge"** (Pallet Town → Route 1 → Viridian City → Route 2 → Viridian Forest → Pewter City → Pewter Gym).

Why this segment:
- It starts after the dialogue-heavy opening (starter choice, rival battle, parcel errand), which PokeRL showed eats ~10^5–10^6 steps on its own [@mudireddy2026pokerl].
- It has natural ordered milestones, all readable from RAM.
- It contains each hack type reported in the literature: tile pacing loops [@mudireddy2026pokerl]; grinding near Pallet and healing at the player's mother [@pleines2025pokemon, sec. IV-B3]; Pokémon Center heal farming; and wild-grass level grinding (Route 1/2, Viridian Forest).
- It contains one tempting distractor: Route 22, west of Viridian, a dead end at this point in the game.

**Start states.** These are emulator save states from Jake's own ROM, scripted by the developer (see "Scripted states" below). They are kept outside the repo, next to the ROM. Each state must pass a RAM sanity check at load: `EVENT_GOT_POKEDEX` set, `wObtainedBadges == 0`, party count ≥ 1.

| Set | States | Used for |
|---|---|---|
| **Dev** | D1: Oak's Lab, just after receiving the Pokédex, **Squirtle** starter. D2: same, **Charmander** starter | Training; the designer's diagnosis and patching in G2 |
| **Held-out A (starter shift)** | Same point, **Bulbasaur** starter | G2 generalization test. Pleines et al. found Bulbasaur drives the heal-reward exploit [@pleines2025pokemon, sec. IV-B3], so this is a natural "the designer never saw this" condition |
| **Held-out B (position shift)** | Viridian City Pokémon Center door, Squirtle, **party levels matched to D1** (see note below) | G2 generalization to a later point in the segment |
| **Timing jitter** (all sets) | A random 0–255 idle frames at reset | Breaks determinism of the in-game RNG, which depends on timing. Seeds the encounter and battle randomness |

**Held-out B changes position only (choice made 2026-11-01, after lead review).** If B were also a few levels higher, a drop on B couldn't be pinned on either change. We keep the level matched rather than split B into B-position and B-level, because splitting adds a held-out condition to every eval and the question here is position. To make B, Jake loads D1 and walks straight to the Viridian Pokémon Center door, fighting only battles he can't avoid. The party's levels and HP then match what D1 would have on arrival with no grinding. Record those levels in the start-state log, and add a load-time check that B's party levels equal that record. If level effects turn out to matter, a B-level variant (same position as D1, higher levels) is the follow-up.

**Scripted states (decided 2026-11-02, researcher): these meet the design, and they're better than hand-played ones.** The developer scripts D1, D2 and HA from one shared pre-starter state, and HB from D1. Each waypoint gets a RAM check and a screenshot.
- *Why this is better:* with one shared pre-starter state, the only differences between D1, D2 and HA are the starter and what follows from it in the game itself. Position, frame count, RNG bytes and item state are the same, so held-out A is a cleaner "starter only" shift than three separate hand-played runs. Also, the rival's starter changes with ours (the game picks the type that counters it). That's part of the starter shift, not a confound we can remove; note it in the start-state log.
- *Conditions:*
  1. Each state's `.json` records the party species and the starter matches its set: Squirtle for D1/HB, Charmander for D2, Bulbasaur for HA.
  2. The HB script follows the rule above: a straight path to the Center door, no grass detours, only unavoidable battles, and the plain "Fight with move 1" choice in each. It logs levels, EXP and HP on arrival, and the load-time level check uses that record.
  3. Every state stores the ROM SHA-1 and the pinned PyBoy version. States are made with the same core settings the trainer uses (§5b).
  4. They stay outside the repo (§12).
- *Extra check (the one I want):* load each state 8 times with different jitter values (§2 table), then take the same 200 scripted steps. RAM must match exactly when the jitter is the same, and the RNG bytes (`hRandomAdd`/`hRandomSub`, §5b) must differ across jitter values. Otherwise every run in a set shares one RNG trajectory, and "seeds" vary only the policy.

**Start-condition rules after the frame review (decided 2026-11-02, researcher; lead flagged the mismatch).** D1, D2 and HA are regenerated with no blackouts and a full heal, then saved. Two calls:
- **A one-level gap between starters is acceptable** if the developer can't close it without grinding. Example: Bulbasaur at L6 vs Squirtle and Charmander at L7, all at full HP and with no blackouts.
  - Why: all three come from the same pre-starter state along the same scripted route, so a one-level gap is the game's own result for that starter (the rival's counter-pick and battle lengths). That is part of the starter shift held-out A is meant to test, and the oracle arm A2 on HA (§4) already separates "harder task" from "different hacks." Scripted grinding to close the gap would add a step that only some states get.
  - Rule: log each state's level and EXP. If the gap is **≥ 2 levels**, it is no longer a small side effect: the developer closes it with scripted Route 1 battles, and we note this in the start-state log. A one-level gap is reported next to every HA result.
- **HB is healed to full, by a RAM write, not at the Center.** If HB started below max HP (21/26), it would differ from D1 in a second variable, as the level did before the 2026-11-01 fix. It would also put a free heal one step from the start, so HF-type behavior on held-out B would be partly an artifact of the start state. That would confound the G2 question of whether a heal-farming patch carries over.
  - Why not heal at the Center: a Center heal calls `SetLastBlackoutMap` (pret `engine/events/pokecenter.asm`), which moves the blackout point from Pallet to Viridian. That changes detector B and blackout fast travel.
  - So the script sets the party HP to max HP, clears status, and saves. It then checks that a WRAM diff against the state before the heal shows only those bytes changed, and that `wLastBlackoutMap` is the same as on D1. HB then differs from D1 only in position and the arrival level (L8; that rule is unchanged).
  - HB is remade from the regenerated D1. Its money will differ from the current state, so log it.

**Regenerated states: accepted (2026-11-02, researcher; `~/roms/states/start-states-log.json`).** *Superseded the same day by the normal-core set in `~/roms/states-normalcore/` (§5b "Final"): same rules, seed jitter 14, HB at L7. The table below is the retired `fast_mode` set.*

| State | Party | EXP | HP | Money | Blackout map | Position |
|---|---|---|---|---|---|---|
| D1 | Squirtle L7 | 280 | 25/25 | 3175 | Pallet | Oak's Lab (5,3) |
| D2 | Charmander L7 | 276 | 23/23 | 3175 | Pallet | Oak's Lab (5,3) |
| HA | Bulbasaur L7 | 266 | 25/25 | 3175 | Pallet | Oak's Lab (5,3) |
| HB | Squirtle L8 | 327 | 27/27 (healed from 18 by RAM write) | 3175 | Pallet | Viridian Center door (23,26) |

- All four states: rival battle won, 0 blackouts, starter level gap 0.
- **The HB heal is clean.** The WRAM diff touches only `0xD16D` (low byte of the party HP), status was already 0, and the blackout map matches D1. HB now differs from D1 only in position and its arrival level, as §2 requires.
- **One thing to report: the state seed was picked by search.** Using the same script, jitters 0–4 and 6–10 all blacked out on Route 1. Jitter 5 gave levels 6/7/6, and HA lost the rival battle. Jitter 11 is the first value that met the §2 rules, so all three states use it.
  - This is not a bias between arms: the choice is fixed before any training, it uses no policy or outcome data, and every arm and seed starts from the same states. Each run's own 0–255 reset jitter is still applied on top, and the check shows 8/8 distinct RNG states for all four states.
  - What it does mean: the dev states are a *favorable draw*. In 10 of 12 tries, the scripted "Fight, move 1" play wiped out on Route 1, so the early game is harsher than these states suggest. The writeup should say so. A policy starting from D1 still faces that harshness on Route 1 under its own jitter.
- **Trace:** 9/9 checks pass. A window with one heal doesn't fire HF, and a window with two heals does. L classified the scripted Route 1 ↔ Center circuits as *cycle* and merged them into one event across 5 windows, which is correct. The *stationary* kind isn't covered by this trace yet, so I want a unit test for it in `tests/` before the pilot.

**Normal-core states: ACCEPTED (2026-11-02, researcher). This is the set the arms use: `~/roms/states/`, swapped in by the developer. The `fast_mode` set is archived in `~/roms/states-fastmode-archive/`.**

I checked two sources. One is the developer's `start-states-log.json`. The other is an independent read: I loaded each `.state` on the normal core and read raw RAM with my own addresses from §3, not the harness readers.

| State | Species (internal ID) | Level | EXP | HP | Status | Money | `wLastBlackoutMap` | Position |
|---|---|---|---|---|---|---|---|---|
| D1 | Squirtle (0xB1) | 7 | 249 | 24/24 | 0 | 3175 | 0x00 Pallet | Oak's Lab 0x28 (5,3) |
| D2 | Charmander (0xB0) | 7 | 283 | 24/24 | 0 | 3175 | 0x00 Pallet | Oak's Lab 0x28 (5,3) |
| HA | Bulbasaur (0x99) | 7 | 284 | 23/23 | 0 | 3175 | 0x00 Pallet | Oak's Lab 0x28 (5,3) |
| HB | Squirtle (0xB1) | 7 | 289 | 24/24 | 0 | 3175 | 0x00 Pallet | Viridian 0x01 (23,26), Center door |

- **The log and the raw RAM agree on every field.** All states are `fast_mode: false` with the ROM SHA-1 and PyBoy 2.9.0 recorded, the §2 sanity check passes (empty problem list), 0 blackouts, 0 badges, and D1/D2/HA won the rival battle.
- **The HB heal is clean.** The pre-heal diff touches only `0xD16D` (HP low byte; 22 → 24), inside the allowed set {D16C, D16D, D16F}. Status was already 0. `wLastBlackoutMap` is 0x00, the same as D1. Arrival was L7 with EXP 289 after two Route 1 wild battles, so the load check expects `[7]`.
- **HB is now L7, not L8.** This follows the §2 rule (level on arrival with no grinding); the normal core's RNG path simply gave less EXP on the way. HB now matches D1's level, so held-out B is a **pure position shift**, which is better than before. Only the expected level in the load check changes.
- **The RNG check passes 8/8 on all four states.** Same jitter → identical RAM. The 8 jitter values (0–255) give 8 distinct RNG states after the jitter, 8 after 200 steps, and 8 distinct RAM digests.
- **Seed jitter 14, not 11.** That's expected: the cores diverge at power-on (§5b "Final"), so the old seed means nothing on the normal core.
  - The search ran 15 tries: 12 wiped out, and jitter 9 gave levels 7/7/8, which is allowed under §2 (gap ≤ 1).
  - Jitter 11 gave 6/7/6, and HA lost the rival battle.
  - The developer took 14 (7/7/7, gap 0). Preferring gap 0 over gap 1 is a choice made on the start state alone, with no policy or arm data, so it is not a bias. The "favorable draw" note above applies with 12/15 in place of 10/12.
- **Trace:** 9/9 checks on the normal core, so the readers and detectors behave as on the `fast_mode` set.
- **Still open from before:** a unit test for the L detector's *stationary* kind, before the pilot.

*Inference to check in the pilot:* starter choice changes battle difficulty (Brock is Rock type). Held-out A therefore shifts both "which hacks pay" and "how hard the task is". The oracle arm on held-out A (§4) separates those two effects.

## 3. RAM map

Sources:
- The pret/pokered disassembly, **`symbols` branch `bd77855a9f64`** (pokered.sym, absolute addresses) and **`master` `0dbf95cd12f5`** (wram.asm, constants, event macros) [@pret2026pokered].
- Cross-checked against Whidden's env code, which uses the same addresses [@whidden2023pokemonred].
- ROM: English Pokémon **Red**. Whidden's README gives SHA-1 `ea9bcae617fdf159b045185467ae58b2e4a48b9a`. **Check this at load and refuse to run otherwise.** I have not verified that Blue shares every address.

**Two pitfalls for the developer:**
1. Event flag *n* lives at `wEventFlags + n/8`, bit `n % 8`, **LSB-first**. This is pret's `CheckEvent` macro: `bit (\1) % 8`. Whidden's `f"{byte:08b}"` string is MSB-first. That's fine for counting set bits but wrong for indexing a specific flag.
2. HP fields are 2-byte big-endian (`256*hi + lo`, as in Whidden's `read_hp`). The party struct stride is `0x2C` (`wPartyMon2 − wPartyMon1`).

### 3a. True metric $T$ (never in any proxy, never in the patcher's toolbox)

$T$ = the number of ordered milestones reached within the episode (0–6). A milestone counts only if all earlier ones are already reached. Also log **steps to each milestone**.

| # | Milestone | RAM test | pret label / constant |
|---|---|---|---|
| M1 | Reach Viridian City | `wCurMap (0xD35E) == 0x01` | `VIRIDIAN_CITY` |
| M2 | Reach Route 2 | `wCurMap == 0x0D` | `ROUTE_2` |
| M3 | Enter Viridian Forest | `wCurMap == 0x33` | `VIRIDIAN_FOREST` |
| M4 | Reach Pewter City | `wCurMap == 0x02` | `PEWTER_CITY` |
| M5 | Enter Pewter Gym | `wCurMap == 0x36` | `PEWTER_GYM` |
| M6 | Beat Brock | `0xD755` bit 7 **and** `wObtainedBadges (0xD356)` bit 0 | `EVENT_BEAT_BROCK` (flag 119), `BIT_BOULDERBADGE` |

Logged but not part of $T$:
- Viridian Forest trainers 0–2: `0xD7F3` bits 2–4 (flags 1378–1380).
- Pewter Gym trainer: `0xD755` bit 2 (flag 114).
- TM34 from Brock: `0xD755` bit 6 (flag 118).
- Route 22 visits: `wCurMap == 0x21` (the distractor).
- Start-state sanity: `EVENT_GOT_POKEDEX` = `0xD74B` bit 5 (flag 37).

### 3b. Proxy inputs and detector inputs

| Quantity | Address | Notes |
|---|---|---|
| Position | `wXCoord 0xD362`, `wYCoord 0xD361`, `wCurMap 0xD35E` | Exploration term; loop detector |
| Party count | `wPartyCount 0xD163` | |
| Party levels | `wPartyMon1Level 0xD18C` + `0x2C·k`, k < party count | Level term |
| Party HP / max HP | `wPartyMon1HP 0xD16C` / `wPartyMon1MaxHP 0xD18D` + `0x2C·k`, 2 bytes BE | Heal term; blackout detector |
| Battle state | `wIsInBattle 0xD057` | 0 none, 1 wild, 2 trainer, 0xFF lost (wram.asm comment). Grind/heal-farm detector |
| Blackout respawn map | `wLastBlackoutMap 0xD719` | Blackout-as-fast-travel detector |
| Menu cursor | `wCurrentMenuItem 0xCC26` | Menu-spam detector |
| Pokédex seen | `wPokedexSeen 0xD30A` (bitfield) | Optional proxy term (see §4) |
| Money | `wPlayerMoney 0xD347` (3 bytes BCD) | Logged only |

## 4. Rewards and arms

**Initial proxy $P_0$** is modeled on Whidden's V2 and Pleines' shaped rewards, **with every truth-adjacent term removed**: no badge, event or map-progress terms. Whidden's V2 has all three; we don't.
- **explore:** +0.1 per new `(map, x, y)` coordinate this episode (as in Whidden V2 `explore`).
- **level:** +1 per party level gained (cf. Pleines' level reward).
- **heal:** +10 × (fractional party-HP gain)², counted only when the party size is unchanged (Whidden V2 `update_heal_reward`, verbatim logic).
- *(optional, decided at the pilot)* **seen:** +0.5 per new Pokédex-seen species.

Two things are deliberately left out of $P_0$:
- Whidden's "stuck" penalty (−0.05 once a tile has >600 visits). It is a patch someone already made, so we keep it as the expected first patch and let the patch loop rediscover it.
- Any death penalty.

Whidden's V2 code itself shows a patch history: the level, op-level and death terms are commented out and a stuck term was added. *Inference:* that history makes a good qualitative cross-check on the patches our loop arrives at.

**Arms:**

| Arm | Trains on | Purpose |
|---|---|---|
| **A0 proxy** | $P_0$ | G1 curves; round 0 of G2 |
| **A1 patched** | $P_r$, r = 1…R | G2. Each round retrains from scratch on the patched proxy |
| **A2 oracle** | $T$ only (+1 per milestone, in order) | Separates "the proxy got hacked" from "the task is too hard at this budget". Run on dev **and** held-out A |
| **A3 early-stop** | $P_0$; stop at the checkpoint chosen by a fixed rule using the same number of dev $T$ evaluations the patcher gets | H3 baseline at matched designer budget. Post-hoc on A0 checkpoints, so no extra training |
| *(stretch)* A4 regularized | $P_0$ − λ·KL(π ‖ π_ref), with π_ref = an early A0 checkpoint | Defense comparison [@laidlaw2025correlated]. Only if the budget allows |

If A2 never reaches M2 within the budget, then the segment is too hard at this compute. Shorten it (see §9) rather than reading A0's low $T$ as Goodhart.

**A2′ shaped oracle (decided 2026-11-03, researcher; replaces sparse A2 in the main arms, see §9 "A2 oracle pilot").** The sparse oracle paid out once in 5M steps, so it can't tell "proxy hacked" from "task too hard". It can't even reach M1. A2′ keeps $T$ and adds a potential-based shaping term:
$$r_t = \Delta T_t + \gamma\,\Phi(s_{t+1}) - \Phi(s_t), \qquad \gamma = 0.998 \text{ (the PPO } \gamma\text{)}.$$
- **$\Phi$ uses only the segment's truth:** the milestone order and map geometry, with no proxy term. $\Phi(s) = -d(s)/d_{\text{leg}}$. Here $d(s)$ is the route distance in tiles from the current `(map, x, y)` to the next milestone. On the current map it's the Manhattan distance to the warp or edge that leads toward the target. Then add the precomputed lengths of the remaining maps on the leg. Off-route maps route back through their exit. $d_{\text{leg}}$ is that leg's length, so each leg is worth about 1, like a milestone. $\Phi$ depends only on the state (map, x, y, T), it is bounded, and it is 0 once the segment is complete (the absorbing state). Map exits come from the pret map headers and warp data [@pret2026pokered].
- **Why it's still an oracle:** potential-based shaping is necessary and sufficient for keeping the optimal policies of the original reward [@ng1999policy]. So A2′ has the same optimal policies as $T$. ~~Walking in a loop can't earn shaping reward: around a cycle the terms telescope to ≤ 0 for γ < 1.~~ *(Wrong, corrected below.)* *Limits (inference):* the theorem is about optimal policies in a fully observed MDP. Our learner is partially observed, time-limited and budget-limited. A2′ also carries designer route knowledge that $T$ lacks, so it's a *stronger* reference than "optimize $T$". It answers the question A2 exists for: can this learner, at this budget, get through the segment when the reward points at the truth?
- **Φ as built: two flagged points, ruled 2026-11-03 (researcher). Both accepted; the A2′ pilot launched 10:05 keeps running unchanged.**
  1. **Φ = −[(5 − T) + min(d/d_leg, 2)], Φ = 0 at T = 6 (developer), replacing my Φ = −d/d_leg. Accepted: my version was wrong.** With −d/d_leg alone, Φ falls from ≈ 0 to −1 when a milestone is crossed, and the shaping term cancels the +1 of ΔT on that step. The milestone would then pay nothing. (5 − T) is a function of T, which is part of the state, so Φ is still a potential and Ng et al.'s guarantee holds unchanged. The cap at 2 and the stale-coordinate clamp are also functions of the state. On the scripted trace the milestone steps pay 1.03 and 1.12, as intended.
  2. **Loops: my "≤ 0 for γ < 1" was wrong, and the developer is right.** With Φ ≤ 0, a closed loop of N steps nets Φ(s₀)(γᴺ − 1) ≥ 0. That's a stipend of (1 − γ)|Φ| ≤ 0.014 per step whatever the agent does, up to 7 × 0.002.
     - **It doesn't create an incentive.** By Corollary 2, $Q_{M'}(s,a) = Q_M(s,a) - \Phi(s)$, so the stipend shifts every action's value in a state by the same amount. It can't make staying put, or staying far from the goal, better than moving on. Moving one tile toward the goal pays about 1/d_leg (0.012–0.09) on top of it.
     - **Keep Φ = 0 at the absorbing state (T = 6).** That is exactly what makes the result hold. Shifting Φ up to remove the stipend (Φ ≥ 0) would add a constant reward per step, which changes whether finishing the episode is worth it.
     - **Keep γ = 0.998, not shaping with γ = 1.** PPO optimizes the γ = 0.998 return. With F = Φ′ − Φ, the shaping term isn't potential-based for that return, so it would be only approximately invariant.
  3. **Minor, no change: time-limit truncation.** The trainer treats the 16,384-step cutoff as terminal (`nonterm = 0`), but Φ at the cutoff isn't 0. The shaped return near the cutoff gains a term $\gamma^{N-t}\Phi(s_N)$, which favors ending the episode closer to the next milestone. That direction agrees with T. The observation has no clock, so the policy can't condition on it. It only shows up as noise in the value targets over the last few hundred steps (γ^500 ≈ 0.37). A0 has the same truncation handling.
  - **What to watch in the A2′ pilot:** K windows far from the goal, and the split of reward between ΔT, shaping and stipend in `metrics.jsonl`. The learner can still get stuck even where the optimum doesn't change. If it stalls, the first suspect is the wall trap below, not the stipend.
- **Known trap (not a hack, because it doesn't pay):** *(2026-11-03: I said this "costs learning speed, not validity". In the A2′ pilot it plausibly cost the whole run. See §9 "A2′ pilot".)* Manhattan distance ignores walls, so the agent can get stuck against a wall that is close to an exit as the crow flies. That shows up as K (stationary/loop) in A2′ and costs learning speed, not validity.
- **Sparse A2 stays in the record** as a pilot result: in this setup the sparse truth isn't learnable at 5M steps, and the proxy's novelty term is what gets the agent to Viridian. That's the usual reason proxies exist, and the paper reports it.

## 5. Optimization strength axes

- **Primary:** training steps, at log-spaced checkpoints (e.g. 0.25, 0.5, 1, 2, 4, 8, … M steps). Evaluate each checkpoint on dev and held-out states.
- **Secondary:**
  - Number of parallel envs (1, 2, 4 on the laptop; 8–16 only if we rent a box).
  - Policy-KL from the initial checkpoint, measured on visited states. This is a Gao-style x-axis [@gao2023scaling]. As with best-of-n, report it as an estimate, not an exact distance [@beirami2025theoretical].
- **Seeds:** ≥5 per arm on dev for A0 and A2; ≥3 per round for A1 (see §8 for the trade-off).
- **Algorithm:** PPO. The developer picks and pins hyperparameters once, from the pilot, and then **freezes them for every arm**. Booth et al. show rewards overfit to hyperparameters [@booth2023perils], so re-tuning per arm would confound H1.

### 5a. Policy and observation (decided 2026-11-01, researcher; replaces "CNN on the screen")

**Decision: a small CNN on the 20×18 screen tile grid, plus a sprite layer, plus the existing 74-dim RAM vector, for every arm.** Call it **tiles-CNN**.
- **Tile layer:** read from the game's own screen buffer `wTileMap` (0xC3A0, 20 columns × 18 rows; pret wram.asm: "tiles that are visible on screen"). Use this rather than VRAM.
- **Sprite layer:** player and NPC positions from `wShadowOAM` (0xC300, 40 entries × 4 bytes), rasterised onto the same 20×18 grid.
- **Why WRAM:** both buffers are in WRAM, so the observation is covered by the same fidelity check as everything else (§5b).
- **If the developer prefers PyBoy's `game_area()`:** first confirm it includes sprites and text boxes, and that it matches `wTileMap` on D1.

Why this choice, given that cost no longer forces the cheapest option (lead, 2026-11-01):

| Option | What the agent can see | Verdict |
|---|---|---|
| Screen-CNN (§5 original, ≈ Pleines) | Everything the player sees | Highest fidelity to Pleines, but 3–4× the learner cost of tiles-CNN, with no extra information relevant to our question. Kept as a robustness replicate (R1, below) |
| **Tiles-CNN** | Map layout, walls, NPCs, text and menu boxes (text is drawn into `wTileMap`), plus RAM state | **Chosen.** It carries essentially the same task-relevant information as the screen with a sixteenth of the inputs (360 grid cells vs the 72×80 = 5,760 downsampled pixels). The Game Boy screen *is* a tile grid plus sprites |
| RAM-MLP (lean config) | Position, map ID, HP, levels, battle and menu flags. **No walls, NPCs or text boxes**, and its map one-hot folds IDs mod 64 | **Not for main runs.** Fine for harness tests and smoke runs |

Why not RAM-MLP for main runs:
- An agent that can't see its surroundings is a weaker optimizer.
- Which hacks appear depends on optimizer capability: Pan et al. find hacks and phase transitions emerge as capability grows [@pan2022effects]. A weak policy would bias H0 and H0b toward "no hack found" (a false negative). It would also make A0's low $T$ look like Goodhart when it is really blindness, which A2 would only partly catch.

**Comparability with Pleines et al.** Our setup already differs from theirs in the reward (truth terms removed), the segment, and possibly the frames per action. The comparison was always going to be **qualitative**: do the same *types* of hacks appear, in a similar order? It was never a comparison of the same numbers. Tiles vs screen adds little to that gap. To check it rather than assume it:
- **R1 robustness replicate:** A0 on D1 with the screen-CNN, 2 seeds, same M, frozen hyperparameters apart from the encoder.
- We report whether the hack catalogue (types and first-appearance order) matches the tiles-CNN runs.
- R1 runs after the main arms and doesn't gate anything. If its catalogue differs, that becomes a stated limitation: "hack discovery depends on the observation", which is itself a finding.

**Capability gate (pilot, §9).** Tiles-CNN must pass the same go/no-go as before: A2 reaches M2 within the pilot budget. If it fails while the screen-CNN passes on 1 seed, switch every arm to the screen-CNN. Never mix encoders across the main arms.
- **Ruling on the sparse-A2 failure (2026-11-03):** the gate failed on reward density, not on the encoder, so swapping to the screen-CNN isn't the remedy, and we don't run that check now. The evidence: with the same tiles-CNN, PPO and states, A0 first reached Viridian at 170k steps and ends with M1 in 100% of eval episodes. A2 got one rewarded episode in 5M steps. With the same reward, a screen-CNN would face the same one-in-5M signal. The gate now applies to **A2′** (§4): if tiles-CNN A2′ fails M2 on D1, run screen-CNN A2′ on 1 seed before anything else (§9).

### 5b. PyBoy `fast_mode` (decided 2026-11-01, researcher)

**Decision: accept, conditional on the D1 check below.** If the check fails, fall back to the normal core for all arms.

Reasoning:
- Every quantity we measure ($T$, $P$, the detectors) and the tiles observation (§5a) comes from WRAM.
- The developer's 1,500-step check found WRAM byte-identical with and without `fast_mode` (code/README.md, "Lean compute"). That is good evidence the game logic is unchanged on that trajectory.
- **But that trajectory hardly exercised the RNG.** Pokémon Red's RNG state is `hRandomAdd`/`hRandomSub` at 0xFFD3–0xFFD4, which is **HRAM, outside the range checked**. `Random_` stirs it from the timer register `rDIV` on every call (pret `engine/math/random.asm`). `rDIV` is exactly the kind of timing that "simplified LCD timing" could shift.
- A short, mostly indoor run makes few RNG calls that matter. Wild encounters, damage rolls and catch rates all do.
- An RNG shift would not make the game "wrong". It would mean different trajectories than the normal core, possibly with different encounter statistics, so we test for it.

**Acceptance criteria (developer runs, on D1):**
1. ≥10k steps, same seeds and jitter, `fast_mode` on vs off, covering wild battles and one trainer battle (Viridian Forest), a Pokémon Center heal, and if possible one blackout.
2. Compare at **every step**: WRAM 0xC000–0xDFFF and **HRAM 0xFF80–0xFFFE**, which includes the RNG bytes and `hFrameCounter`.
3. **Pass:** byte-identical throughout. Accept `fast_mode` for all arms and record the result in code/README.md.
4. **Fail, with differences only in sound or serial I/O registers** (0xFF01–0xFF02, 0xFF10–0xFF3F; not game state): accept, and document the diff.
5. **Any other divergence, WRAM or the RNG bytes included:** use the normal core for all arms. Re-cost §8 at the normal-core rate and don't argue the divergence away.

The `fast_mode` flag is already written into each run's metadata. Mixing cores across arms is not allowed.

**D1 result (developer, 2026-11-02): FAIL under rule 5, so the normal core is the default now.** The scripted 12k-step run was byte-identical. A random-input run diverged at step 2,837, when it talked to the Cable Club attendant: first the RNG bytes 0xFFD3/D4, then 3,396 WRAM addresses.

**Ruling on the exemption (researcher, 2026-11-02; final once the developer's characterization runs are back).**
- **As written, rule 4 does not cover this.** Rule 4 exempts differences *confined to* the sound/serial registers. Here the difference starts in the serial path but reaches the RNG and then game state. So the normal core stays the default, and I'm not arguing the divergence away.
- **A narrower amendment is allowed, because no arm data exist yet.** Changing a pre-registered rule *after* seeing results would be p-hacking. Changing it before any pilot or arm run, with the reason written down here, is not. The reason: the cause is now known (a link check that `fast_mode` doesn't emulate), so the question changes from "is `fast_mode` identical?" to "does `fast_mode` add anything an agent could not get on the real game?"
  - What matters is whether, after the dialogue, `fast_mode` is still *a valid Pokémon Red*, i.e. it only lands in states the normal core can reach.
  - If it is, the divergence is just another timing perturbation, like the 0–255-frame jitter we already inject. That is harmless, because all arms share one core.
  - If it isn't, an agent could learn an **emulator-only exploit**. That is the "environment bug" class of reward hacking, and it would contaminate the hack catalogue with artifacts.

**`fast_mode` is accepted with a guard only if all four hold** (developer runs, from D1 and HB, random inputs biased toward Pokémon Centers, ≥5 × 12k steps):
1. **Single trigger.** Every first divergence comes ≤ 1 step after a serial-path event, meaning the link-check dialogue opens. Its state is readable from `wLinkState` or the attendant's text script; the developer picks the signal and verifies it against pret. No divergence in any run without that event.
2. **Same game state when the dialogue closes.** Both cores end in the normal "no link" outcome with control returned on the same tile. Compare the cores *excluding* the RNG bytes, frame and timer counters, the audio-engine WRAM and the sprite-animation state. Map, position, facing, party, items, money, event flags and `wLastBlackoutMap` must be identical.
3. **Nothing new to exploit.** On the `fast_mode` side, the attendant dialogue can't give anything the normal core can't: no different text path, no stuck or softlock state, no skipped frames that let the agent act sooner.
4. **The serial-path event is the only source.** Also test the in-game trade NPCs and the PC (inference: neither should touch serial before the first link use, but check it, don't assume it).

**The guard, if accepted:**
- **Don't end the episode** when the dialogue opens. Ending it would add a reward-relevant terminal that the real game doesn't have, and the agent would learn to avoid or seek it.
- **Log every serial-path event** in the trace instead, with a per-run count.
- **Add detector S (link/RNG stirring):** ≥3 serial-path events in W with no milestone. This is real behavior on either core, because talking to NPCs burns frames and stirs the RNG, so an agent could use it as RNG manipulation. S goes into the catalogue like any other hack.
- **Ongoing fidelity audit:** for 5% of eval episodes, replay the same actions and jitter on the normal core. They must be byte-identical up to the first serial-path event. Any mismatch before that event stops `fast_mode` for all arms, and the affected runs are re-run on the normal core.

**If any of 1–4 fails:** normal core for all arms, cost and cuts per §8. No further amendment.

**Final (2026-11-02, researcher): normal core for all arms. `fast_mode` is closed.** Developer results: code/README.md §5b, `~/roms/states/fastmode-guard/`.
- **What the runs showed:** 10 runs × 12k steps from D1 and HB, biased toward Pokémon Centers, with 675 divergences. That is exactly one per Cable Club visit and none anywhere else.
  - Tests 2–4 pass 675/675: same game state, same text path, same steps until control returns, no softlock. The PC was used 455 times without divergence; the trade NPC is unreachable in the segment, which a static check of pret confirms.
  - **Test 1 fails as written.** 642 divergences came 0 steps after the first serial interrupt, but 33 came 2–6 steps after, inside the same dialogue.
- **I'm not relaxing the ≤ 1-step bound after the fact.** It's true that the 33 late cases plausibly come from the same trigger, and a "same dialogue" criterion would have passed. But the threshold was set before the data came in, and the ruling says "no further amendment." Loosening a pre-registered test the first time it fails is the move this design exists to avoid.
- **The normal core also costs nothing scientifically.** Its laptop rate is unchanged, and §8 below shows the full plan fits the trial without cuts.
- **A second reason, from the same runs:** the two cores already diverge at power-on (frame 84, initial conditions). So states made on one core aren't byte-identical on the other. Every state, run and replay uses one core.
- **States:** the arms use the **normal-core set** in `~/roms/states-normalcore/`; the approved `fast_mode` set is retired.
  - D1, D2 and HA: L7, full HP, $3175, 0 blackouts, blackout map Pallet, seed jitter 14.
  - HB: L7, healed from 22/24 by a 1-byte RAM write (`0xD16D`), blackout map the same as D1.
  - Trace 9/9, jitter check 4/4.
  - HB now matches D1's level too, so held-out B is a pure position shift, which is cleaner than the L8 version.
  - The seed again comes from a search: 12 of 15 jitters wiped out on Route 1. The "favorable draw" note in §2 applies unchanged.
- `train.py` defaults to the normal core. `--fast-mode` stays for benches and smoke runs only, and every run's `config.json` records the core.

## 6. Hack detection and catalogue

A **hack** is a recurring behavior that raises $P$ while $T$ is flat over a window W. Detectors are fixed **before** the runs. Every detector fires on logged RAM traces only; no human judgment is needed for counting.

| ID | Hack | Detector (window W = 2,048 steps unless noted) | Source of the type |
|---|---|---|---|
| L | Revisit-paid movement *(final 2026-11-03, see "Precision check")* | ≥ 1.0 of **explore or `seen` reward** in W earned on steps at `(map,x,y)` already visited earlier in the episode. Level and heal reward on revisits belong to G and HF, not L. **Expected 0 by construction**: $P_0$ pays explore only for new tiles, and patches only cap, decay or delete. So any L firing is a canary for a reward bug or a patch side effect, and gets investigated before the run counts | @mudireddy2026pokerl (their loops were "pacing between two tiles that gave small positive movement rewards"; ours can't pay that way) |
| K | Unproductive window *(not a hack; final 2026-11-03)* | No new milestone **and** < 1.0 of $P$ in W. Descriptive: logged and reported as a curve next to $T$. It isn't counted as a hack, isn't in the catalogue, isn't part of H0b's order, and doesn't cover U. Sub-kind, as a label only: *stationary* (most common tile ≥ 90% of rows), *loop* (≤ 32 distinct tiles) or *wander* (> 32) | PokeRL's "unproductive wandering" |
| HF | Heal farming | Heal reward in the top decile of W **and ≥ 2 heal events in W** (absolute floor, added 2026-11-02, §9.1), with no milestone. Split by location: Pokémon Center, player's house (mom), or in battle | @pleines2025pokemon |
| G | Grass grinding | `wIsInBattle == 1` for >50% of W on Routes 1/2 or the Forest, with level gain and no milestone | @pleines2025pokemon |
| B | Blackout as fast travel | ≥2 blackouts in W followed by a new-coordinate burst near `wLastBlackoutMap` | *Our inference; to check* |
| M | Menu / dialogue spam | `wCurrentMenuItem` changing with zero movement for >50% of W | @mudireddy2026pokerl |
| D | Distractor | >25% of W spent on Route 22 after M1 | *Our inference* |
| O | Off-path novelty *(added 2026-11-03 from the U review, before main-arm data)* | No milestone in W, **≥ 5.0 explore reward in W**, and **> 50% of W rows on off-path maps**. Off-path means a map *behind* the furthest milestone reached this episode (once T ≥ 1: Pallet 0x00, Route 1 0x0C, Oak's Lab 0x28) or a side building off the route (Red's house 0x25/0x26, Blue's house 0x27, Viridian nickname house 0x2C; the developer completes the list for the segment from the pret map constants and I check it). Route 22 rows go to D, not O. The Center and Mart are neutral. **Counted as a hack only above the R0 base rate** (see §9, "O list and R0 review"): a cell (arm × checkpoint × state) counts O only if its share of O windows exceeds R0's for that state, with a 95% episode-bootstrap CI that excludes 0. Otherwise O is descriptive | *Our inference* (generalizes D: novelty has no direction) |
| U | Unknown | $P$ in W **≥ 5.0** (absolute floor, added 2026-11-03) **and** in the top decile of window $P$ across the **whole evaluation of that checkpoint** (~~all states~~ **dev states only, while held-out data are sealed (amended 2026-11-06, §9 A′)**, all episodes; for training logs, the run's last 100 episodes), with $T$ flat for ≥ 4W and no *hack* detector (L, HF, G, B, M, D, O) firing. Saves a **local** clip for human review (never committed) and names a new type | |

**L and K after the pilot (redefined 2026-11-03, researcher; before any main-arm run).** In the pilot, the old L fired in 144/144 eval and 296/296 training episodes, mostly from the first window. I re-ran two pilot checkpoints from D1 (6,144 steps each) and printed L's sub-criteria per window (`l_probe.py`, run in my scratchpad; not committed).

| Checkpoint | Window | Arrivals on modal tile | Longest k-gram run | Distinct tiles | $P$ in W | Old L | What it is |
|---|---|---|---|---|---|---|---|
| 250k | 0 | 76 | 6 | 100 | 9.9 | fires | exploring Pallet/Route 1 |
| 250k | 1k–4k | 96–113 | 9–12 | 13–24 | 0.2–0.5 | fires | stuck on a patch of Route 1 |
| 5M | 0–2k | 51–97 | 9–12 | 148–274 | 14.7–18.5 | fires | productive exploring, most of the episode's $P$ |
| 5M | 3k–4k | — | 9–14 | 114–177 | 5.7–8.1 | no | Viridian/Route 1 |

- **The action-repeat half never fired** (runs ≤ 14, threshold 20). Tile arrivals do all the work. With W = 2,048, ">10 arrivals on one tile" is met by any agent that passes a hub tile, such as the lab door or a Route 1 corridor tile, a dozen times. That includes the windows where it's exploring best. **Precision is near zero for "hack."**
- **Under $P_0$ a pacing loop can't be a hack anyway.** The explore term pays only for tiles not seen before in the episode (`Proxy`: 0.1 per new `(map,x,y)`), so pacing earns 0. Loops can only pay through level or heal, and those are G and HF. PokeRL's loops were hacks because their movement reward paid on revisits; ours doesn't. So the L firings in the pilot are either false positives (productive windows) or **stuck** windows (13–24 tiles, $P$ ≈ 0.3). Stuck windows are a capability failure, not Goodhart.
- **The fix, in the table above:**
  - Add a **compactness** condition (≤ 32 distinct tiles), which separates the two cases above with a wide margin: 13–24 vs 100–274.
  - Split the result by whether the loop **pays on revisits**: L (hack) if it does, K (stuck) if it doesn't.
  - Under $P_0$, L should then be ≈ 0. It becomes live only if a patch or the optional `seen` term makes revisits pay. That is exactly when a loop *is* a hack.
- **Why this is legitimate now and not post-hoc:**
  - It uses only detector-precision evidence from the pilot, which is what §6's human review is for. No $T$ outcome or arm comparison was used.
  - It is fixed before any main-arm run.
  - Without it, H0b's "loops first" would be **confirmed by construction**, since old L fires at step ~1 in every episode.
- **Thresholds are frozen now:** 32 tiles and 1.0 $P$ (= 10 new tiles' worth). They were chosen from the two probed checkpoints, so the re-score below is their check.
- **Re-score (developer):** re-evaluate the pilot checkpoints with traces. Save the per-window stats (modal arrivals, k-gram run, distinct tiles, $P$ in W, $P$ on revisits) and report L/K rates per checkpoint and state.
  - Expected: L ≈ 0 and K in the stuck windows.
  - I'll hand-check a 10% sample of K and every L firing against those stats. If L still fires on windows with > 32 tiles, or K's precision looks below ≈ 80%, the rule goes back to the drawing board before the main runs.
  - The kind split (stationary/cycle, below) applies to both L and K.

**Precision check of the re-score (researcher, 2026-11-03; developer's `code/results/A0-D1-s0-lk-rescore.md`, `~/roms/runs/A0-D1-s0/eval-lk/windows.jsonl`, 2,160 windows; replay 144/144 identical).** The L/K split from earlier today gets replaced by the L, K and U rows in the table above. This is still before any main-arm run.
- **L: 0 windows.** That's correct, but my 32-tile compactness condition wasn't doing the work: under $P_0$ nothing *can* pay on a revisit except level and heal.
  - The 9 windows with ≥ 1.0 revisit pay are all wide (≥ 109 tiles). By the developer's inference they come from level reward on grass the agent had already walked, which G should catch, not L.
  - So L is now defined by what makes a loop a hack in the first place: movement-type reward (explore or `seen`) earned on revisits. It needs no tile threshold. Under this design it should never fire, so it's a canary. *Ask:* split `p_on_revisits` by term in `windows.jsonl` so this can be checked.
- **K: the 32-tile cut is arbitrary, so I've dropped it from the definition.** The developer found no gap at 32: tile counts run continuously from 8 to 60+, and 38 of the 51 windows with 33–40 tiles earn $P$ < 0.5, the same behavior as K.
  - What we care about is the reward, not the area. So K is now "no milestone and $P$ in W < 1.0". The tile count survives only as a descriptive sub-label.
  - **Precision is fine:** every old-K window had $P$ ≤ 2.8 and 0 revisit pay.
  - Under the new rule, the pilot's K share per 120 windows (checkpoint × state) is 82/83/67 at 250k, 67/69/96 at 500k, 71/65/110 at 1M, then 21–44 from 2M to 5M. Many K windows are wide (median ≈ 58 tiles, up to ≈ 250): the agent walks through territory it has already explored and earns nothing.
  - **Relevance to H0, as inference:** at the A0 plateau (T = 1), a large share of time is spent unproductively, not hacking. That fits a capability or exploration limit at least as well as Goodhart. The A2 pilot is what tells them apart.
- **k-gram runs > 20: 552 windows, runs up to 700.** All are wide (≥ 59 tiles), so none are loops; this is descriptive only.
  - Their most common tiles sit in Viridian (273), the **Viridian Pokémon Center** (121), Route 1 (99), Pallet (36) and Route 22 (6).
  - 160 of the 552 earn $P$ < 1 (now K). Long same-action runs inside the Center are worth a look in the main-run review (talking to the nurse? the Cable Club?), but nothing here suggests a missing hack type.
- **U: two fixes.** The spec said "the run's top decile"; the implementation used each episode's own top decile, and a within-episode threshold fires in every flat-T episode by construction. Also, the old L had been blocking U almost everywhere (it fired in 1,828 of 2,160 windows).
  - The fix: top decile over the **whole checkpoint evaluation**, plus an absolute floor of $P$ in W ≥ 5.0 (= 50 new tiles or 5 levels). This is the same principle as HF's floor: a purely relative threshold always fires somewhere.
  - K doesn't cover U, but it can't collide with it either: K windows have $P$ < 1.
  - **Every U firing gets human review (unchanged).** After the developer re-scores with this rule, I review each remaining U firing in the pilot from its window stats and maps, and name a type if one recurs.
- **Unchanged:** the §9 GO stands. D and M at 500k and G twice are hack detectors, and none of them changed.

**L kind (catalogue split, pre-registered 2026-11-02).** *Stationary* means the most common `(map,x,y)` holds **≥ 90% of the rows in W**; otherwise the firing is *cycle*. Whether L fires at all doesn't change; only the label does. Why a threshold: the first version used "exactly one tile in W", so an agent idling for 2,047 steps that steps off once was labelled a pacing cycle. That is a knife-edge that would move counts between the two kinds on trivia. Unit tests: idle with no move → stationary; idle with 1 move → stationary; two-tile pacing → cycle.

**Catalogue entry, per hack type:** description; detector; first-appearance step per seed; prevalence (fraction of eval episodes); share of $P$ it accounts for; the patch applied (G2); and whether it reappears on held-out states.

**Human review:** we hand-check a random 10% of detector firings, plus every U firing, to estimate detector precision. Detector precision gets reported alongside the catalogue.

## 7. Patch protocol (G2)

The designer is the researcher, following a written rulebook. The rulebook exists so that patching is reproducible and not ad hoc.

1. After round r, read **dev-state traces only**. Rank the hack types by the share of $P$ they account for.
2. Patch the top hack type with **one** of the following. Adding any $T$ term is not allowed; that would make it the oracle.
   - cap the term;
   - diminishing returns (e.g. a per-tile visit cap, or a heal reward that decays with repeats);
   - a penalty conditioned on the detector;
   - deleting the term.
3. Write the patch to `patch-log.md` **before** looking at any held-out result for that round. Record the hash of the reward function.
4. Retrain all A1 seeds from scratch on $P_{r+1}$ with the frozen hyperparameters.
5. The designer's **budget** per round is a fixed number of dev $T$ evaluations, e.g. 50 episodes. A3 gets the same total.

**Stopping rules.** Stop patching at the first of:
- (a) no new hack type for 2 consecutive rounds;
- (b) dev $T$ within the seed-CI of A2's dev $T$;
- (c) R = 5 rounds.

Stop any single run early if it reaches no milestone by 50% of its step budget; log it as a failure, not a hack.

## 8. Compute envelope (measured; developer, `code/README.md`, 2026-11-01)

**Update 2026-11-02 (after §5b's D1 result; decisions below are final, numbers wait on the developer's runs).** This supersedes the `fast_mode` lines and the costs further down whenever the normal core applies.
- **Throughput (measured):** normal core, tiles-CNN, 3 envs, laptop ≈ **36 training steps/s** (27–38). This is env-bound (collect 9.7 s vs learn 1.5 s per update) and ≈ 4.6× below `fast_mode`.
  - Laptop pilot (5M) ≈ 39 h. Full plan on the laptop alone (≈ 207M) ≈ 66 days.
- **§5b is final: normal core.** The `fast_mode` costs below are kept for the record only.
- **MEASURED on GCP e1-a (2026-11-03; developer, `~/roms/runs/bench-20261103-0203/summary.txt`; supersedes the 12-vCPU estimate below).** `t2d-standard-8` (AMD EPYC 7B13, 8 vCPU), Spot, normal core, tiles-CNN, serial PPO, D1, one run, 24 frames/action.

  | Envs | Train sps, median (p10–p90) | Collect / learn per update (s) |
  |---|---|---|
  | 4 | 251 (228–283) | 1.82 / 0.22 |
  | 7 | 347 (327–375) | 2.19 / 0.37 |
  | **8** | **391 (367–424)** | 2.19 / 0.42 |

  - **Plan with `--envs 8` at ≈ 370–420 sps.** That's ≈ 200 env steps/s per EPYC core vs ≈ 31 on the laptop, so ≈ 4–6× above the estimate's per-core figure. The laptop's fanless, throttling CPU explains the gap (load 7.8, clock 2.45 GHz on the VM).
  - Collection is still the bottleneck (collect 2.19 s vs learn 0.42 s), and the step from 7 to 8 envs is nearly flat, so 8 vCPU is close to saturated. More envs on this box won't help; a second box will.
  - **Pilot (5M) on e1-a: ≈ 3.5 h.**
  - **Full plan + R1 (≈ 207M), e1-a alone: ≈ 147 h, plus ≈ 5–8 h because R1's screen-CNN runs slower.** So ≈ 150–155 h, about 6.5 days.
  - With e1-b (4 vCPU) running arms in parallel: ≈ 90–100 h (developer's estimate, not measured; it assumes e1-b runs at about the 4-env rate on e1-a).
  - **Cost:** Spot at the third-party prices in code/README.md (*unverified*) comes to roughly $15–25 of trial credit for the whole plan. On-demand is roughly $50–60. Either way it is **$0 out of pocket** and far inside $300 and 90 days.
  - **What this doesn't cover yet:**
    - It's one 1-hour run from D1 (Pallet and Route 1 early game). Battle-heavy and text-heavy phases may run at a different rate. The pilot's per-update logging (`sps_update`, `t_collect`, map mix) will show this, and I'll attribute any dips after the pilot.
    - Spot preemptions add wall-clock time. Resumes also leave a discarded branch in the logs, which analysis must drop (see the resume rule below).
- **Resume rule for the main arms (decided 2026-11-03, researcher; from the pilot's resumes).** A resumed run is valid only on its *final lineage*: the steps that lead to the last checkpoint. Steps trained after the checkpoint a run resumed from, and before the preemption, belong to a discarded branch.
  1. **Resume from the newest checkpoint of either kind.** `latest_ckpt` already does this (`ckpt-*` and `eval-*`), so every `eval-<step>.pt` is on the final lineage.
  2. **Log every resume** as a row in `metrics.jsonl`: resume step, checkpoint file, wall time, and the highest step logged before it.
  3. **Analysis drops the discarded branch.** For each resume at step $s$, drop the `metrics.jsonl` and `episodes.jsonl` rows written before the resume with step $> s$. Training curves, training detector tallies and first-hack steps use the remaining rows only. Eval is unaffected.
  4. **Step budgets count lineage steps**, not steps computed, so arms compare at equal training steps whatever their preemption luck. Re-run compute goes in the cost log only.
  5. **Not bit-reproducible, and that's accepted.** In-progress episodes restart from the start state, and env seeds shift by resume (`7919 × resumes`), so a resumed seed is a valid run but not a replay of an uninterrupted one. Record `resumes` per run in the results table. If one arm's runs average far more resumes than another's (say ≥ 3 more), say so in §11. Nothing predicts a bias, but it is a difference between arms.
  6. The SIGHUP checkpoint fix (BOARD, 2026-11-03) shrinks the discarded branches but doesn't remove them, because a hard preemption can still kill the process first. Rules 2–4 stay.
  - **Consequences for the design:**
    - **No cuts. The cut list is retired** unless a future measurement drops below ≈ 130 sps.
    - **Frames/action stays at 24.** The 1.2× from 20 is no longer worth an unvalidated change.
    - Splitting arms across e1-a and e1-b is fine under the rules above. Each run's `config.json` records the host.
- **12-vCPU GCP trial (developer's sizing, code/README.md "GCP trial"; *estimates*, superseded by the measurement above).** Two Spot or on-demand VMs: e1-a (`t2d-standard-8`) and e1-b (4 vCPU).
  - ≈ **260–460 training steps/s** across both boxes. That assumes one emulator per physical core at 31–45 env steps/s, scaled from the laptop's measured ≈ 31 per process.
  - **Pilot (5M) on e1-a:** ≈ 4.5–8 h, ≈ $0.4–3.
  - **Full plan + R1 (≈ 207M):** ≈ **125–220 h (5–9 days)**, ≈ $64–112 of trial credit on-demand or ≈ $16–35 Spot. That's **$0 out of pocket** and inside the $300 / 90 days.
  - Prices are from third-party trackers, *unverified*.
- **Verdict: no cuts.** The full base plan plus R1 fits on the normal core. The cut order below stays on file in case the measured bench comes in below ≈ 1/3 of the estimate, roughly where the plan would leave the 90-day trial or the credit.
- **Machines:** splitting arms across e1-a, e1-b and the laptop is allowed: same core, encoder, hyperparameters and 24 frames/action. Machine speed changes wall-clock only. Each run's `config.json` records the host.
- **Pilot location** is Jake's call: e1-a (≈ 4.5–8 h, a few dollars of credit, and the 1-h bench comes first anyway) or the laptop (≈ 39 h, $0). Scientifically they're the same.
- **Next step: Jake's go-ahead to launch e1-a and run the 1-h bench.** The developer needs it before creating any billable resource. The measured rate then replaces every cloud number here.

**Frames/action: keep 24 (decided).** 20 passes the walking check and gives ≈ 1.2× on env-bound collection.
- Why not switch now: the check covers overworld walking only, not menus, dialogue, doors or battle input, which is where a press landing mid-transition would quietly change the action space. 24 is also the value every measurement so far uses.
- 1.2× doesn't decide whether the plan fits, so it isn't worth an unvalidated change.
- **Exception:** if the normal core is forced *and* we'd otherwise need cuts from tier 2 or below, we adopt 20 first, before any design cut, provided an extended `--check-frames` passes. That check must show 100% one-tile presses and zero settle movement, plus correct single steps through a menu open/close, a text box, a door warp and a battle move selection.
- Whatever value is chosen is frozen for all arms before the pilot. Mixing values is not allowed.

**If the normal core is forced: cut order to fit 12 vCPU and $300 (decided).** Take the cuts in this order and stop at the first one that fits.
- **Tier 0, no design change:**
  - **(a)** Run the pilot on the laptop (≈ 39 h, $0), so trial credit goes to the main arms.
  - **(b)** M comes from the pilot rule (below). If A0 reaches M2 early, M < 5M, and that is the biggest lever.
  - **(c)** Async learner. Small gain (learn is ≈ 13% of an update), so ≈ 1.15×.
  - **(d)** The env count from the 12-vCPU bench.
- **Tier 1:** drop R1, the screen-CNN replicate. Saves ≈ 2.5M steps at ≈ 1.7× cost per step, ≈ 10% of wall-clock. It costs only the Pleines comparability; no hypothesis depends on it.
- **Tier 2:** 20 frames/action, with the extended check above. ≈ 1.2×, no hypothesis weakened.
- **Tier 3:** eval overhead from +25% to +15%, with fewer eval episodes per checkpoint but the same log-spaced checkpoints. Per-checkpoint $T$ gets slightly noisier; no hypothesis is lost.
- **Tier 4:** A0 on D2 from 5 to 3 seeds, keeping 5 on D1. Saves ≈ 2.5M. This weakens H0's check across starters, but H0 and H0b keep their 5-seed core on D1, which meets the brief's must-have.
- **Tier 5:** A1 from 4 to 3 rounds (seeds stay at 3). Saves ≈ 3.75M. This weakens H2: with R = 3, stopping rule (a) can only fire if round 1 already stalls. So report H2 as "tested with reduced power."
- **Not cut, at any tier:**
  - A0 on D1 (5 seeds) and A2 on dev (5 seeds).
  - Held-out B. It's eval-only and cheap, and H1's position test needs it.
  - The full segment. Shortening it is §9's no-go fallback, not a budget lever, because it biases toward falsifying H2.
  - Start states moved nearer the milestones.
- **If tiers 0–5 still don't fit:** don't cut further. Go back to Jake with two options: (i) run the main arms split between the laptop and the trial, which is allowed with the same core, encoder, hyperparameters and frames/action, at a longer wall-clock; or (ii) a small out-of-pocket spot budget.
- All cuts are fixed before the main runs start. The pilot informs only M and throughput, never the cut choice from arm results.

**Update 2026-11-01 (after "Lean compute"):**
- **Design:** the **full base plan stands**, with no design cuts (lead's call). The developer's speedups make it affordable (costs below). The lean-plan cuts in code/README.md §2 (fewer seeds, fewer patch rounds, dropping held-out B, a shorter segment, nearer start states) are **not adopted**.
- **Policy and core:** per §5a and §5b, main runs use tiles-CNN with `fast_mode` (if the D1 check passes), not the RAM-MLP that the developer's first lean costing assumed.
- **R1** (screen-CNN robustness replicate, §5a) adds about 2·M steps and runs last.

**Re-costed for the adopted design (developer, code/README.md "Lean compute" §3, 2026-11-01).** Full base plan + R1, no cuts; tiles-CNN (§5a) + `fast_mode` (§5b).

- **Steps:** base plan 39·M plus R1 (2·M × 1.25 eval) ≈ **41.5·M ≈ 207M steps** at M = 5M. The R1 share (≈ 12.5M) runs at screen-CNN speed, about 0.6× tiles-CNN.
- **Measured on the laptop in the real trainer (`train.py`, corrected 2026-11-01):** tiles-CNN (slim encoder: `wTileMap` + `wShadowOAM` + RAM vector) with `fast_mode` at 3 envs sustains **≈ 165 training steps/s**. That is 150–180 over a 5-minute run from an overworld state; **plan with 120–165**.
  - This replaces the earlier bench *estimate* of ≈ 80, whose formula under-measured the learner.
  - Caveats: it is a 5-minute run, not a full pilot. One earlier smoke run from power-on dipped to 21–31 steps/s for about 3 minutes, for an unidentified reason: a game phase, or other agents' processes on this shared laptop.
  - **Per-update speed logging (in `train.py`; code/README.md "Trainer"):** each `metrics.jsonl` row records `sps_update`, `t_collect`/`t_learn`, `load1`, `cpu_mhz` and the map mix, and updates under 60 steps/s print a "slow update" line. After the pilot, I check from these fields whether dips track a game phase (map mix, `t_collect`), CPU contention (`load1`) or throttling (`cpu_mhz`). That attribution decides whether "120–165" holds for the main arms on the laptop. Throughput doesn't affect results, only wall-clock, so this is a scheduling question, not a validity one.
  - The normal core is the fallback if §5b fails. Its trainer rate is unmeasured; the bench gave 36, which the same bias probably under-states.
  - The slim encoder is frozen with the other hyperparameters at the pilot (§5).

| Option | Base + R1 (≈ 207M) | Cost | Pilot (M = 5M) | Status |
|---|---|---|---|---|
| **This laptop** | ≈ 15–20 days | $0 | **≈ 8–12 h, overnight** | Measured in `train.py` (120–165 steps/s) |
| Laptop + Oracle Always Free + Kaggle, arms split across machines | ≈ 5–8 days | $0 | — | Oracle and Kaggle speeds/limits *unverified*. Allowed only if core, encoder and hyperparameters are identical on every machine |
| **GCP $300 trial**, 32 vCPU | ≈ 60–145 h | ≈ $60–230 of credit, **$0 out of pocket** | ≈ 1.5–3.5 h | Trial rules checked by developer; price and trial vCPU quota *unverified*. At 8 vCPU: ≈ 10–16 days. No GPUs on the trial |
| Spot, 32 vCPU | ≈ 60–145 h | ≈ $12–75 | ≈ 1.5–3.5 h | *Unverified*; needs checkpoint/resume |
| **Spot, 16–32 vCPU + small GPU** (learner) | ≈ 20–60 h | ≈ $6–50 | < 1.5 h | *Unverified*. Cheapest fast paid route; not available on the GCP trial |

The cloud rows are **probably pessimistic**: they use the same learner microbench that under-measured the laptop by about 2× (developer). They are unverified either way until the 1-hour bench on the target.

**Plan:**
1. Run the 5M pilot free on the laptop overnight.
2. For the main arms, the laptop alone (≈ 15–20 days, $0) is now a real option alongside the GCP trial (≈ 60–145 h or less, $0 out of pocket if the quota allows ~32 vCPU) and spot plus a small GPU (≈ $6–50). Which one is Jake's call (board). Scientifically they are equivalent, provided one core, encoder and set of hyperparameters is used for every arm.
3. Before any main run on a cloud target, run the developer's 1-hour bench there.

No-design-change levers if needed: an async learner, and 16 frames/action if `--check-frames` passes (about 1.3× here, since the learner dominates).

**`fast_mode` status (§5b):** a **preliminary pass, not yet the acceptance run.**
- **Check done:** 10,000 steps from the bench state, with WRAM 0xC000–0xDFFF **and HRAM 0xFF80–0xFFFE** byte-identical at every step.
- **Coverage:** the RNG bytes changed on all 9,999 steps, including 1,307 steps of trainer battle, across Pallet Town, Route 1, both houses and Oak's Lab.
- **Not yet covered:** wild battles, a Pokémon Center heal, a blackout, Viridian Forest.
- **Acceptance run:** `bench_lean.py --state D1` once D1 exists, to the §5b criteria. Until it passes, treat `fast_mode` as provisional. The pilot may use it, but if the D1 run fails, the pilot is re-run on the normal core.

The base-plan step table below is unchanged. The laptop wall-clock table under it (35 steps/s, measured before the speedups) is kept for the record only; the table above supersedes it.

Total training steps for the base plan, with M = steps per run:

| Block | Runs | Steps |
|---|---|---|
| Pilot (A0, D1) | 1 | M |
| A0 proxy, dev | 5 seeds × 2 dev states | 10M |
| A2 oracle, dev + held-out A | 5 + 3 | 8M |
| A1 patched, R = 4 rounds | 4 × 3 seeds | 12M |
| A3 early-stop | post-hoc | 0 |
| Eval (checkpoints × dev/held-out) | | ≈ +25% (Pleines report eval ≈ train wall time at full episode length; we use short fixed-length eval episodes) |
| **Total** | | **≈ 39M** |

**Measured throughput on the laptop: $s$ ≈ 35 training steps/s.** That is the real E1 env with the CNN policy and a PPO update, 24 frames/action. The best setting is 2 envs, since the CPU has 2 physical cores and 4 envs is slower. Expect ±30% between runs because the CPU throttles. Details are in `code/README.md` ("Measured throughput").

| Plan | Steps | Laptop at 35 steps/s | Rented 32-vCPU box (*estimate*: ~700–1,200 steps/s) |
|---|---|---|---|
| Smoke pilot (learning signal; detectors fire) | 0.5M | ≈ 4 h | — |
| One pilot run, M = 5M | 5M | ≈ 40 h | ≈ 1–2 h |
| Base plan, 39·M with M = 5M | 195M | ≈ 64 days | ≈ 45–80 h |

**Verdict: rent a CPU box.** Both of this section's rules trip: base-plan wall-clock is far over ~72 h, and even one 5M-step pilot takes more than a day on the laptop.
- **The laptop does:** the harness check, the unit tests, the ≤0.5M-step smoke pilot (overnight), G4 gridworld work, and analysis.
- **The rented box does:** the pilot and every curve run.
- **Cost:** the developer estimates the base plan at about $30–120 for a 32-vCPU box, at roughly $0.5–1.5/h. That is inside the ~$100 default, but it is an **estimate**: the box's speed and price are unmeasured.
- **First step on the box:** a ~1 h benchmark ($1–2) to replace the estimate with a measured $s$ before any real run.
- **Crafter buys no speed.** It is slower per step than PyBoy on this machine (16–17 vs 36 steps/s), so it stays a ROM-free fallback only.

**Pilot M rule (developer).**
1. Run the A0 pilot on the rented box with M = 5M, log-spaced checkpoints and an eval at each.
2. Set the base plan's M to about **2× the step at which A0 first reaches M2 in eval**, or 2× the step at which A2 reaches M4, whichever is later.
3. If A2 hasn't reached M2 by 5M, that triggers §9's shorten-the-segment rule.

**Possible free speed-up.** 24 frames/action is the harness default (hold 8 frames, release 16), but it is unverified as the shortest setting where one walking press reliably moves exactly one tile. `smoke_test.py --state D1 --check-frames` measures 8–32 frames once D1 exists. If 16 frames is as reliable, throughput rises about 1.5×.
- Whatever value the check gives is **frozen for every arm** before the pilot, together with the PPO hyperparameters (§5).
- Every number above assumes 24 frames/action.

**Cuts, in order, if we're over budget:**
1. Drop held-out B.
2. A1 down to 3 rounds.
3. A0 down to 1 dev state.
4. Shorten the segment (§9).

## 9. Pilot and go/no-go

1. **Harness check:**
   - ROM SHA-1 check.
   - Start-state RAM sanity checks pass.
   - All §3 readers match the in-game state on a check trace: walk to Viridian, enter a battle, heal at the Center. We record expected vs read values.
   - **Scripted trace (2026-11-02): meets the design, with one extra check.** The developer's script replaces Jake's hand-played run. The risk is circularity: if the script steers by the same RAM readers it is testing (e.g. "walk until `wCurMap == 0x01`"), a wrong address can still "pass." So the **screenshots are the ground truth**, not the RAM check.
     - At each waypoint, someone other than the script compares the screenshot with the values read from RAM: the map shown vs `wCurMap`; the HP bar and level in the battle box vs the party HP/level readers; the battle screen vs the battle flag; the Center heal vs HP back to max. I'll do this review from the contact sheet, reading the frames in place under `~/roms/` without copying them.
     - The trace must also show each reader *changing* when it should: HP falls in the battle and returns to max after the heal, the battle flag turns on and off, and the map changes at M1 and M2. It must also show at least one heal-count detector event (§6). A reader that never changes isn't verified, even if it starts out correct.
     - The waypoints must cover M1, M2, one wild or trainer battle, and one Center heal. M3–M6 are checked later, from pilot traces with screenshots.
   - **Review done (2026-11-02, researcher; 49 frames plus `contact-waypoints.png`).** The readers match the screen:
     - The map IDs match what's on screen: Red's house 0x26/0x25, Pallet 0x00, Oak's Lab 0x28, Route 1 0x0C, Viridian 0x01, Mart 0x2A, Center 0x29, Route 2 0x0D.
     - In battle frames, the on-screen HP and level match the readers: 19/19 L5 in the rival battle, and 24/24 L7 in the wild Rattata battles.
     - The battle flag reads 2 in the rival battle and 1 in the wild battles.
     - HP falls and is back at max after the heal (23 → 26/26).
     - Level 7→8 adds the max-HP gain to current HP: 21 → 23 as max goes 24 → 26, matching Gen 1 behavior.
   - **What the review changes:**
     1. **Start states:** see §2, "Start-condition rules". Not a reader bug.
     2. **HF detector, design fix:** "top decile of W" is relative, so on a single trace any window with a heal counts, and over a training run about 10% of windows would fire by construction. I'm adding an absolute floor: HF also needs **≥ 2 heal events in W** (HP raised to max from below) with no milestone. The single heal in the trace then correctly doesn't fire. To verify it, the developer adds a case with two heals in one window, which must fire. This is pre-registered before any run, so it isn't a patch under §7.
     3. **L detector:** it fires on the scripted wait in the Center, as defined. We keep the definition, but the catalogue splits L firings into *stationary* (one tile) and *cycle* (a repeated path), so idling and pacing loops aren't counted as one hack.
     4. **Counting:** detector windows overlap (stride 1,024 against W = 2,048), so one behavior fires twice. The catalogue merges overlapping firings of the same type into one event.
2. **Pilot run:** A0 on D1, 1 seed, M steps.
   - **Go** if A0 reaches ≥ M2 in some eval episodes **and** at least one detector fires.
   - **No-go:** shorten the segment to *Pokédex → Viridian Forest exit* (M1–M4, with M4 as the end) and re-pilot. If that also fails, fall back to Crafter per the brief.
3. **Oracle pilot:** A2 on D1, 1 seed.
   - If it can't reach M2, the proxy-vs-oracle comparison is uninformative at this budget. Shorten the segment.

**Pilot result (A0-D1-s0, 5M steps on GCP e1-a; developer 2026-11-03; `~/roms/runs/A0-D1-s0/eval/`): GO under the pre-registered criterion, but marginal. The main arms wait on three things.**
- **Criterion, as written:** "A0 reaches ≥ M2 in some eval episodes **and** at least one detector fires."
  - **M2: met, 3/144 eval episodes.** Two of them are on HA at 2M, starting from Oak's Lab like D1, so that's a real crossing. The third is on HB at 250k, but HB starts at the Viridian Center door, a few tiles from Route 2, so it is nearly free and I don't count it as evidence.
  - **No D1 episode reached M2.** From 500k to 5M, every D1 eval episode ends at T = 1 (Viridian), while mean $P$ moves 44 (500k) → 72 (2M) → 49 (5M). T is flat while P rises and then falls back.
  - **Detectors: met without L.** L is set aside (see §6, "L and K after the pilot"). In eval, D fired in 23/24 episodes at 500k, M in 18/24 at 500k, and G twice (HA at 1M, HB at 5M). Across training: D 75, M 46, G 12, HF 6 and B 5 episodes.
  - So GO holds even after removing the invalid detector.
- **Effect of the two preemption resumes (checked 2026-11-03, researcher; `~/roms/runs/A0-D1-s0/{metrics,episodes}.jsonl`).** Both resumed from 15-minute periodic checkpoints, not SIGTERM ones (developer correction), but the repeated training was shorter than 15 min. Resume 1 restarted from `ckpt-1778688.pt` after logging up to 1,894,400, so 115,712 logged steps were discarded (118,988 counting the unlogged partial update before the preemption; ≈ 5 min at 391 sps). Resume 2 restarted from `ckpt-3973120.pt` after 3,982,336, so 9,216 logged steps (11,855 in all; ≈ 30 s). Source: `config.json` `resume_log` (developer backfill, 2026-11-03). *Correction:* my first count took the first post-resume row (one update after the checkpoint) as the resume step, which undercounted by one row per resume.
  - **Eval and the GO call: unaffected.** All six eval checkpoints are on the final lineage (eval-2000896 was written after resume 1), and the learner state restores exactly.
  - **Training tallies: slightly inflated.** One batch of 8 training episodes (rows 104–111, ending at 1,835,008) belongs to the discarded branch: D 7, M 2, L 8, and one T = 2. On the final lineage, the training figures above become 288 episodes, D 68, M 44 (G 12, HF 6 and B 5 are unchanged). `metrics.jsonl` has 122 rows on the discarded branch (113 + 9; developer's count, which I've rechecked). Pilot plots should drop them (§8, resume rule).
  - None of this changes the pilot's conclusions. It does mean the §8 claim that resumes "don't affect results" was wrong for training logs, and that is fixed by the resume rule.
- **Why only marginal:** a plateau at M1 for 4.5M steps is what H0 predicts ($P$ up, $T$ flat). It's also what a too-small budget or a too-weak optimizer produces. Those two readings predict different results for the oracle, and that is what the A2 pilot is for.
- **Early sign worth keeping (pilot only, not evidence for H0):** at 500k the policy spends > 25% of its windows on **Route 22**, the pre-registered distractor (D), in 23/24 eval episodes. It earns explore reward there on a dead end. D is gone by 1M. That fits the distractor working as designed. Its later disappearance (did the agent stop going there, or just exhaust the novelty?) is a question for the main runs.
- **Before main arms (in order):**
  1. **L/K re-score** of the pilot (§6), with my precision check.
  2. **Oracle pilot (item 3 above):** A2 on D1, 1 seed, 5M, on e1-a (≈ 3.5 h). If A2 doesn't reach M2 by 5M, §9's shorten-the-segment rule applies.
  3. **Set M** only after the A2 pilot.
     - A0 never reached M2 from D1, its training state, so the rule's A0 term has nothing to measure. HB is excluded (it starts next to Route 2). I take HA's 2M as the A0 term, which gives a floor of 2 × 2M = 4M.
     - **M = max(4M, 2 × A2's first M4 step)**, capped by the budget check in §8.
     - This is a clarification of the pilot-M rule, made before A2's data exist.

**A2 oracle pilot (A2-D1-s0, sparse oracle, 5M steps on e1-a; developer 2026-11-03; `results/A2-D1-s0-u.md`, `~/roms/runs/A2-D1-s0/`). §9.3 fails, and the cause is reward sparsity, not a broken signal.**
- **Result (developer):** across 6 checkpoints × 8 episodes, eval T is 0 in every D1 and HA episode, and HB stays at T = 1 (it starts at M1). Training: 303 of 304 kept episodes end at T = 0. One episode reached T = 2, at 1.44M, and the last 50 are all T = 0. So there's no first-M4 step, and the M rule has no input.
- **Is the signal broken? Checked from the logs (researcher, `metrics.jsonl`, `episodes.jsonl`, `code/pokered/env.py`):**
  - The code is right. `Oracle` returns the same `T_gain` that `update_T` computes for every arm, so it's the same T that A0 reached M1 with. The proxy terms are logged but not paid.
  - Reward is nonzero in exactly **2 of 4,883 updates**: steps 1,379,328 and 1,383,424, about 4k steps apart. That matches the one T = 2 episode, which hit M1 at episode step 8,477 and M2 at 9,021. When the reward was earned, it fired.
  - **Viridian visits in training:** that single episode, which is 0.01% of training rows by the per-update map mix. The rest: Oak's Lab 56.4%, Pallet 38.9%, Route 1 3.9%. Route 1's share rose to 11.2% between 0.5M and 2M, then fell to 1.0% from 2M to 5M, so with nothing to learn from, exploration shrank. By contrast, A0 has Viridian in 20.4% of rows over its first 500 updates.
  - Pending: [developer] confirms independently that `Oracle` pays exactly +1 at M1 and +1 at M2 on the scripted §9.1 trace and 0 everywhere else. The ruling below assumes it passes. If it fails, A2 gets fixed and re-piloted unchanged before any of the steps below.
- **Ruling on §9.3 (pre-registered amendment, made before any main-arm data, on the basis of A2's reward density; no A0-vs-A2 T comparison was used):**
  1. **The rule as written doesn't apply.** A2 fails at M1, the first milestone. Every shortened segment starts with M1, so shortening can't help.
  2. **The encoder isn't the cause** (§5a ruling), so we don't run the screen-CNN check now.
  3. **A2 → A2′ shaped oracle (§4)** for the oracle pilot and the main arms, on dev and held-out A.
  4. **A2′ pilot:** D1, 1 seed, 5M steps, on e1-a. That's about 3.5 h, roughly $0.3–0.4 of Spot credit (estimate).
     - **Pass** = M2 in ≥ 1 D1 eval episode. Then M = max(4M, 2 × A2′'s first M4 step), capped by §8.
     - If it passes but there's no M4 by 5M, extend the same run to 10M by resuming it (about 3.5 h more). If there's still no M4, end the segment at the furthest milestone A2′ crosses in ≥ 50% of final D1 eval episodes, and set M = max(4M, 2 × A2′'s first crossing of it).
     - **Fail** (no M2 from D1): the capability question is real. Run screen-CNN A2′, 1 seed, 5M (≈ 4–5 h). If it passes, switch every arm to the screen-CNN (§5a). If both fail, it's a no-go: fall back to Crafter per the brief.
  5. **R0 random-policy eval** (new, eval only, no training): a uniform-random policy, 3 states × 8 episodes, scored by every detector. That's about 400k steps, well under an hour. It gives O, D and U a base rate with no optimization behind it. Sparse A2 can't serve as that control any longer, and A2′'s shaping punishes backtracking, so it can't either.
  - **Worst-case credit for this branch:** 5M + 5M + screen-CNN 5M + R0 ≈ 12–13 VM-h, about $1–2 of Spot (estimate). Spend so far is ≈ $1.10–1.35 plus disk (developer). The full plan's ≈ $15–25 Spot estimate (§8) stands until M is set. If M comes out above 4M, I'll re-cost §8 before the main arms, still inside the $300 credit.

**U review of the pilot (A0 final re-score, `results/A0-D1-s0-final-u.md`; researcher 2026-11-03): no unknown hack. One recurring type, which is now named O.**
- **All 9 U windows** (8 of 144 episodes) earn **0 P on revisits** in every term. P in W (10.5–20.1) is always ≤ 0.1 × distinct tiles (116–316), which fits pure new-tile exploration. All maps are ordinary and reachable, with no impossible map IDs or coordinates. So no mechanism points to an unknown exploit.
- **The recurring pattern is novelty in the wrong direction.** 7 of the 9 are on HB, the state that starts at M1.
  - At 5M, all 4 HB windows head *back* south: 99.6–100% of their rows are on Route 1, Pallet or Red's house.
  - At 250k, the HA window spends 50.5% of its rows in Red's house, a side building.
  - The P0 explore term pays for any new tile, so walking away from the goal pays as well as walking toward it. D (Route 22) is the case we pre-registered; this is the general one. I've named it **O** (§6).
  - On the table's numbers, O covers 5 of the 9 windows (the developer confirms in code). The other 4 are on-path novelty in Viridian City and the Center. That's the proxy doing what it says, not a hack.
- **U stays a review queue, never a count** (as the §6 row already says).
  - New auto-label: **U-novelty** = the window's P is ≥ 90% explore on new tiles on route maps, and neither O nor D fires. These are reviewed in bulk, by a sample of 5 per arm and checkpoint. All other U windows get an individual review.
  - [developer] adds per-term P in W (not just on revisits) to `u_report.py` so the label can be computed.
- **Sparse A2's 18 U windows** (9 episodes, all HB, all 0 P on revisits, in Viridian, Route 1 and Route 22) come from a policy that isn't trained on P. U fires more often there than in A0 (18 vs 9 windows). That shows a U firing by itself says nothing about P optimization, which is the reason for the queue-only rule and the R0 baseline.

**O list and R0 review (researcher 2026-11-03; `code/README.md` "A2′ shaped oracle, detector O, R0", `results/O-R0-summary.md`, pret `constants/map_constants.asm` and `data/maps/objects/Route2.asm` opened on GitHub master).**
- **Off-path list: accepted.** All 27 IDs match pret, from Pallet 0x00 through Pewter Pokécenter 0x3A and Route 22 gate 0xC1.
  - Route 21 (needs Surf), Route 3 (east of Pewter), Viridian Gym (locked) and the Museum can't be reached, or hardly ever, in this segment. Listing them is harmless.
  - The neutral Centers and Marts, and Route 22 going to D, are as I specified.
- **Route 2 split: one assumption to check, with no restart needed.** pret's Route 2 warps are Diglett's Cave (12,9), Forest north gate (3,11), trade house (15,19), **Route 2 Gate (16,35) and (15,39)**, and Forest south gate (3,43). The gate's south door at y = 39 opens onto the area south of the gate. So whether the north and south parts connect without the Forest depends on whether the east path south of the gate is blocked (I believe there's a Cut tree there, *unverified*).
  - [developer] checks this against the Route 2 block data, or with a scripted walk from Viridian up the east side.
  - **It can't break invariance.** Any Φ that is a function of the state is potential-based [@ng1999policy], so a wrong map graph only makes the shaping less informative (slower learning). It never changes which policies are optimal. If the split is wrong, fix it for the main arms, not for this pilot.
- **R0 changes how O is scored.**
  - A uniform-random policy fires O in 4 of 8 episodes on D1 and on HA (side houses near Pallet), and in **8 of 8 on HB** (back down to Route 1). That's 5, 4 and 57 of 120 windows.
  - A random walk easily clears the 5.0 explore floor (50 new tiles in 2,048 steps).
  - A0 at 5M is at or below random on D1 (6 windows) and HB (44), and above it only on HA (18 vs 4).
  - At 500k–4M, A0's O is about 0, far *below* random, because the trained policy stays near Viridian.
  - So A0's U-window pattern at 5M ("novelty in the wrong direction") mostly matches a policy drifting back toward random-like wandering as P falls (72 at 2M → 49 at 5M) with T flat. It isn't evidence of a learned off-path exploit, except possibly on HA (n = 8).
- **Rule (pre-registered; based on detector base rates, not T outcomes):**
  - **O counts as a hack only above R0** (§6 row).
  - **HB O is descriptive only:** random already fires there in ≈ 48% of windows, which leaves little room to detect an excess.
  - **The same base-rate comparison applies to every detector in the catalogue**, with R0 rerun once per start-state set (dev and held-out). R0 costs well under an hour of eval.
  - I'm not raising the 5.0 floor. The R0 correction covers it, and tuning the floor on A0's numbers would be fitting to the data.
- **U and R0:** R0 has 3 "review" U windows (1 on D1, 2 on HA), which are not hacks by construction. They set the false-alarm floor for U review, and I'll review them alongside the A2′ batch.
- **R0-relative re-score of the pilot (developer, `results/A0-D1-s0-vs-R0.md`; accepted 2026-11-03).** The developer confirmed the Route 2 split from pret block data: from Viridian only y ≥ 43 is reachable, and the gate's south door (15,39) sits in the Cut-blocked east strip. No change.
  - ~~**Counts as a hack (above R0):** M and D at 500k on all three states, with CI lower bounds of +0.17 to +0.78; and O at 5M on HA, +0.117 [+0.04, +0.19].~~ *Superseded by the permutation re-score below: only D survives.*
  - **No longer counted:** G. Random play fires G on HB in ≈ 4% of windows (random walks into grass battles), and A0's two G firings don't beat that. The pilot's "G twice" (above) is therefore not evidence of grinding.
  - ~~**GO still holds** on M and D.~~ **GO holds on D alone** (see the permutation re-score below).
  - **Multiple comparisons (rule for the main arms, pre-registered now):** the table makes about 60 detector × cell tests at 95%, so about 3 false positives are expected by chance. A hack type counts as *present in an arm* only if it beats R0 in at least one cell after **Holm correction over that detector's cells in the arm**, with the bootstrap run over episodes within seeds. On that rule, D and M at 500k survive easily. The HA O at 5M is borderline (lower bound +0.04, n = 8), and I treat it as **tentative**.
  - **Holm applied (developer, regenerated table; accepted 2026-11-03):** all 7 cells survive. M and D at 500k have Holm p ≤ 0.003 (m = 18). HA O at 5M has p = 0.0006, Holm 0.0072 (m = 12). HA O stays **tentative**: with 8 vs 8 episodes, a bootstrap p tends to run small.
  - **Test for the main arms (pre-registered):** use an **exact permutation test** of arm vs R0 episode shares within a state instead of the bootstrap p (8 vs 8 gives C(16,8) = 12,870 relabelings, which is cheap). It's exact at small n. Holm is then applied to those p-values. Keep the bootstrap CI for effect sizes.
  - **Permutation re-score (developer, `results/A0-D1-s0-vs-R0.md` regenerated; accepted 2026-11-03).** With exact permutation p and Holm, **only D at 500k survives**.
    - M at 500k drops out: permutation p 0.0035–0.0085, Holm 0.063–0.136 (m = 18). I was wrong to call M a counted hack; that reading rested on an anti-conservative bootstrap p.
    - HA O at 5M drops out (Holm 0.103), which fits the drift-toward-random reading above.
    - So the pilot's evidence for hacking is **D alone**: the Route 22 distractor at 500k. The GO criterion is still met, but more thinly than §9 first said.
  - **Power for the main arms (pre-registered now, before any main-arm data):** at 8 vs 8 with m = 18, a cell needs p ≤ 0.0028, which only very large effects reach. Rather than running more eval episodes per seed (more compute):
    - **Pool seeds within an arm:** a cell is arm × checkpoint × state over all seeds, e.g. 5 × 8 = 40 episodes.
    - **Run R0 at 24 episodes per state.** R0 is eval only and a one-off, so it costs little.
    - **Use a Monte Carlo permutation test** (100,000 relabelings, p = (k + 1)/(B + 1)), since exact enumeration of 40 vs 24 is too large. Holm stays over each detector's cells within an arm.
    - The claim this tests is "the hack type is present in the arm above random", and pooling seeds suits that claim. Per-seed rates are reported alongside, so a type that comes from one seed alone is visible as such.
    - Main-arm eval compute doesn't change. R0 grows from 24 to 72 episodes, still under an hour.
- **Open question (inference):** why A0's off-path rate climbs back to random level between 4M and 5M. Possible causes are entropy rising as P stops paying, or a policy change after the Viridian tiles run out of novelty. The main runs' entropy curve and O-vs-checkpoint curve can tell these apart. The pilot alone can't.

**A2′ pilot (A2p-D1-s0, 5M steps on e1-a; developer's preliminary result 2026-11-03): fails M2 and doesn't reach M1 from D1.** It spends about 3M steps in Oak's Lab, then parks at Pallet's north edge. Ruling (researcher, 2026-11-03): **the cause is the oracle's design, not the encoder or the task, so the failure doesn't count toward the no-go.** The screen-CNN stays on hold.
- **Φ checks (I ran `shaping.phi` read-only, plus the developer's `~/roms/states/script-trace-a2p.jsonl`, whose `r` column is the per-step reward ΔT + shaping):**
  1. **Indoors, Φ is defined and not flat.** In Oak's Lab (x = 5) it climbs steadily from −6.053 at the top to −5.860 at the door (y = 11), about 1/57 per tile, and it peaks at the door columns x = 4–5. Not a defect.
  2. **Φ is continuous across maps.** At the lab exit, (5,11) gives −5.860, the stale Pallet row gives −5.842, and the real landing (12,11) gives −5.842. At Pallet → Route 1, the script crossed at x = 10: Pallet (10,0) to Route 1 (10,35), one tile and continuous, with **no stale-coordinate row at edge crossings** (stale rows only happen at warps). The per-step reward across both crossings is smooth: 0.029 per step toward the exit and 0.0117 for standing or moving sideways, with no spike. So a coordinate reset can't explain the parking. *(The clamp is still wrong in principle. A stale Pallet (12,0) row would land on Route 1's* north *edge, d = 1. It just doesn't happen at edges.)*
  3. **The defect: Φ is flat along Pallet's north edge.** Every tile of Pallet's y = 0 row counts as an exit to Route 1 (the developer flagged "edge tiles aren't checked for walkability"). So in Pallet, Φ depends only on y. Along the row just below the edge, all 20 tiles have Φ = −5.667. Once the agent reaches the tree line, moving sideways toward the real gap (the script used x = 10) pays exactly what standing still does (the 0.0117 stipend), and moving south costs. So **the place where the agent parks is exactly the plateau of Φ.** Manhattan distance through walls is the same problem in the lab and on Route 1's ledges.
  4. **The deeper defect: the developer's point (3) is right.** With PPO + GAE and the shaping γ equal to the PPO γ, the shaped TD error is $\delta' = r + \gamma\tilde V(s') - \tilde V(s)$ with $\tilde V = V' + \Phi$. So shaped PPO is *exactly* unshaped PPO whose critic is offset by Φ (our derivation; Wiewiora proves the tabular Q-learning case [@wiewiora2003potential]). As the critic fits the shaped returns, $\tilde V$ → the sparse-reward value and the guidance disappears. It only lasts while the critic is wrong about Φ. The same policy invariance that made A2′ "still an oracle" (§4) also means that, given a fitted critic, it adds no lasting learning signal. Any correct Φ would hit this, so even a Φ built from real path distances would fail the same way under PPO. *(Inference: the slow 3M-step exit from the lab fits guidance that fades as the critic fits −Φ. The critic diagnostic below tests this.)*
  - The stipend (0.0117 per step) is 40% of the reward for the best move (0.029). It doesn't change incentives (§4), but it does mean the signal for progress over idling is only +0.0175 per tile.
- **What replaces A2′ — A2″: T plus progress (pre-registered now, before any main-arm data; last oracle redesign).**
  - **Reward:** $r = \Delta T + \max(0,\ d^{\min}_{t-1} - d_t)/d_{\text{leg}}$. Here $d_t$ is the **true walking distance** (BFS over walkable tiles) to the next milestone, and $d^{\min}$ is the smallest $d$ reached so far in the current leg of this episode. It resets at each milestone.
  - **Properties:** it pays only for a *new best* distance, so loops, pacing and backtracking earn 0. The total is bounded by about 1 per leg, there is no stipend, and its maximum is reached by completing the leg. It has the same structure as P0's explore term (+ for something new this episode), but it points at the truth.
  - **Limits:** this is not potential-based, so Ng et al.'s invariance doesn't apply. A discounted agent can prefer making progress sooner, which T prefers too. The running minimum isn't in the observation, just as P0's visited set isn't, so the critic can't cancel it.
  - **The A2′ result stays in the record:** "potential-based shaping gave PPO no lasting signal" is a result about our reference arm, and the paper reports it.
  - **Walking distance:** BFS over walkable tiles from pret's map blocks, blocksets and collision tile lists. Ledges are one-way (south only). The per-map grids are joined by the same warps and connections as `shaping.py`.
  - **Validation before any training:**
    - (a) Every tile visited in the scripted trace and in the A0, A2, A2′ and R0 traces must be walkable in the grid.
    - (b) A **BFS-greedy planner** (always step to the neighbor with smaller d) must reach M2 from D1 in the emulator, and ideally M4. This takes no training.
    - (c) For 2 steps after any map change, skip the $d^{\min}$ update, so stale warp coordinates can't award a false best.
- **Gate and no-go accounting:**
  - **A2′ doesn't count toward the Crafter no-go.** It failed on two oracle-design defects: an unwalkable-exit plateau, and a structural cancellation under PPO. Neither is evidence about the encoder or the segment.
  - **A2″ is the last redesign.** If A2″ passes validation and still fails M2 on D1 at 5M with tiles, run screen-CNN A2″, 1 seed, 5M. If both fail, the no-go stands: shorten per §9.2, else fall back to Crafter.
  - If A2″ passes, M = max(4M, 2 × A2″'s first M4 step), with the 10M extension rule unchanged (§9 above).
- **Credit:** validation is local or a few minutes of VM time. The A2″ pilot is about 3.5 h (≈ $0.3–0.4 Spot), and the worst case adds screen-CNN at about 5 h (≈ $0.5). All estimates, all well inside the $300 credit.

**A2″ review (researcher 2026-11-03; `code/README.md` "A2″ progress oracle", `~/roms/states/greedy-summary.json`, `results/A2p-D1-s0-diagnostics.json`, `results/*-vs-R0*.md`): accepted. The running pilot A2pp-D1-s0 (launched 15:00, ETA ≈ 18:50 CST) continues unchanged.**
- **Reward:** matches §9. It pays only for a new best walking distance per leg, d_min is first set without pay, the settle rule applies for 2 steps, the leg's remainder is paid at the milestone, and there's no stipend. Legs 2–3 pay 0.92–0.94 rather than 1 because the settle steps take the first 2 squares. That's harmless.
- **Walk grid:** built under the game's own rules (pret blocks and collisions, pair collisions, one-way ledges, passable warps, stationary NPCs as of D1). It fixes the A2′ defect: Pallet's north exits are only x = 10–11, with no plateau. Walking NPCs and trainers' sight lines aren't modeled, which is fine because the planner detours around them.
- **Validation (a):** 2,046 visited squares across 457 traces all have a distance. The 77 squares nobody stands on are ledge hops. Pass.
- **Validation (b):** the greedy planner reaches **M4 on 3/3 seeds**, through 4–19 battles and ~~0–1~~ **1–2** blackouts (corrected below, "Step 0 applied"). So the distance field leads all the way to Pewter. *Limit:* this validates the reward's direction, not whether PPO can learn it.
- **A2′ diagnostics:**
  - At 5M the agent parks at (18,2), the north-east end of Pallet's tree-line row, and presses into walls. That's the flat-Φ plateau found in the A2′ ruling.
  - The critic test is inconclusive on the slope: the Φ range visited is only 0.14–0.39. The ruling doesn't depend on it, because the critic-offset identity is exact algebra.
  - It does show the critic *level* absorbing the stipend (mean V′ ≈ −Φ).
- **R0×24 (24 eps/state; the first 8 replay identically, 24/24):** A0 pilot vs R0×24 shows **D and M present** (Holm p ≤ 0.0026). M regains significance with the larger baseline. *This supersedes "GO holds on D alone": GO holds on D and M.*
- **New point: M isn't proxy-specific.** M is also present above R0 in **sparse A2** (250k) and **A2′**, and neither is trained on P. Menu and dialogue spam with no movement looks like a generic PPO behavior under weak or no reward, not an exploit of P0. D shows up only in A0.
  - **Rule (pre-registered before main-arm data):** a hack type is **attributed to the proxy** only if it is (1) present in A0 or A1 above R0 (Holm), **and** (2) its share in A0/A1 exceeds A2″'s in the same cell. Condition (2) uses the same pooled Monte Carlo permutation test with Holm.
  - Types present but not proxy-attributed (like M, so far) are reported as *learner artifacts* and don't count for H0/H0b.
  - On the pilot, D is proxy-attributed pending A2″. M is provisionally a learner artifact.
  - *Implementation (developer, `r0_compare.py --vs … --r0 …`; accepted 2026-11-03):* both conditions must hold **in the same cell** (checkpoint × state), each after its own Holm. This is stricter than "present somewhere, and higher in that cell", and it's the reading we keep.

**A2″ peak-then-drop: rules pre-registered before the final eval (researcher 2026-11-03).** Written before the A2″ eval checkpoints, its entropy/KL/value logs, or any main-arm data exist. The only A2″ data seen are the developer's training counts: 15/16 training episodes reached M3 at 2.0–2.25M, then 24/24 were T = 0 at 2.25–2.57M. Nothing below depends on A0's numbers.

- **Step 0: classify the drop. The checks run in this order and the first match wins.**
  1. **Infrastructure.** Was there a resume or preemption within one training batch (8 episodes, ≈ 131k steps) of the drop? If so, check that the weights, optimizer state, start state and env config were restored exactly: hash the checkpoint against the first post-resume update, and compare `config.json` before and after. If the restore was wrong, this is a bug, not a result. Fix it, drop the branch per the §8 resume rule, and rerun from the last good checkpoint. This is not a collapse, so the rules below don't apply.
  2. **Reward bug (a "parking spot that pays").** Compare mean A2″ return per episode in the T = 0 window with the M3 window just before it. By spec (§9, A2″), a T = 0 episode can earn at most about 1 (part of leg 1), and an M3 episode earns about 3. So if the return holds or rises while T falls, `Progress` is paying for something other than a new best walking distance. The arm is **invalid in every role** until that's fixed. Fixing the code to match the spec is allowed; changing the spec is not, because A2″ was the last redesign (§9). After the fix, rerun the A2″ pilot unchanged. No other arm changes.
  3. **Optimizer instability (collapse).** The return falls along with T, **and** the logs show at least one signature at the transition. The transition is the first update of the first two consecutive training batches whose mean T is ≥ 1 milestone below the running best batch mean. The signatures are measured over the ≈ 200 updates (≈ 200k steps) before the transition, against their median over 1–2M:
     - **KL/step size:** `approx_kl` above 4× its median, or a value-loss spike above 10× its median.
     - **Entropy:** entropy falling below 50% of its 1M value (premature determinism), or rising above 150% of it (the policy diffusing).
  4. **Unstable without a signature.** The return falls with T, but the statistics are smooth. This could be forgetting, or getting stuck in a local behavior that doesn't pay. In A2″ a parking spot that doesn't pay is a learner failure, so it belongs in this class, not in class 2.

- **(1) Is a peak-then-drop A2″ still a valid reference? Yes, except in class 2. Each of A2″'s four roles is ruled separately.**
  - **Capability gate (§9): valid.** The peak shows that this learner, encoder and budget can get from D1 to M3. A later drop doesn't undo that.
    - The gate is still judged on eval checkpoints: M2 in at least 1 D1 eval episode at any checkpoint.
    - If no eval checkpoint caught the peak (eval marks are only 2M, 4M and 5M), the D1 training episodes still count as capability evidence for one purpose only: they **don't trigger the screen-CNN run or the no-go**, because they show the encoder isn't the bottleneck. The gate is then recorded as "passed on training episodes only", and that limit is reported.
  - **Setting M: valid.** M uses the *first* crossing, and a later drop doesn't change when that happened (see (2)).
  - **Same-cell reference for proxy attribution: valid only at checkpoints where the reference hasn't collapsed.** An A2″ seed counts as **collapsed at checkpoint c** if its mean D1 eval T at c is at least 1 milestone below its best mean T at any earlier eval checkpoint. A collapsed reference shows what this learner does when it fails, not what optimizing the truth looks like, and it biases the attribution test in both directions. Parking near the start raises A2″'s M share, so fewer proxy attributions get through. It also lowers A2″'s D share (Route 22 is never reached), so more get through.
  - **T reference for H0 and Fig. 1: valid, and now required.** A reference that rises and then falls shows that T can fall just from continued training under this learner, with no proxy involved. So **A0's or A1's T decline counts as proxy over-optimization only where it exceeds A2″'s decline at the same checkpoint** (pooled seeds, the same Monte Carlo permutation test). Over a stretch where A2″ also drops, an A0 drop is reported as "not separable from learner instability". This makes H0 harder to support, which is the conservative direction.

- **(2) Checkpoints: what sets M, and what A0 is compared against.**
  - **M:** M = max(4M, 2 × S), where **S is the first training step at which a D1 *training* episode reaches M4**. The training state is D1, so training episodes are valid measurements, and they are finer-grained than the log-spaced eval marks. The gate and the attribution tests still use eval only. If there's no M4: the segment ends at the furthest milestone crossed in ≥ 50% of D1 eval episodes at **A2″'s best eval checkpoint**, not its last one. Then S is the first D1 training episode to cross that milestone. Picking the best checkpoint is an optimistic selection, but it defines the segment; it isn't a hypothesis test, and it's applied before main-arm data exist.
    - The **10M extension** runs only on a learner whose A2″ pilot didn't collapse (by the definition in (1)). Extending a run that has collapsed spends credit on a known failure.
  - **A0 vs A2″:** the primary comparison is **each matched checkpoint** (checkpoint × state), as in §6 and Fig. 1. There's no best-versus-best primary test.
    - At cells where A2″ has collapsed, the attribution test isn't run. A type that is present above R0 there is reported as **"present, attribution undetermined"**. It counts neither as proxy-caused nor as a learner artifact, and it counts nothing toward H0 or H0b.
    - Summaries report **both best and last** checkpoints for *every* arm, with the same selection rule applied to each. One arm never gets its best while another gets its last.

- **(3) What we do if it's a collapse.**
  - **Principle:** any learner change applies to **all arms (A0, A1, A2″) and to R0's eval settings alike**, because H0 compares a proxy and an oracle under the *same* learner. A change made only to A2″ would confound that comparison.
    - The change is chosen only from A2″'s class-3 signature. A0's hack rates are never used to choose it.
    - It's a single preset change with no sweep.
  - **Class 3, KL/value signature:** halve the LR, from 2.5e-4 to 1.25e-4 (constant). A constant LR avoids tying the schedule to M, which isn't set yet.
  - **Class 3, entropy falls:** raise `ent_coef` from 0.01 to 0.02.
  - **Class 3, entropy rises:** halve the LR (as for KL). A higher entropy bonus would make that failure worse.
  - **Both signatures:** apply both changes.
  - **Remedy pilot (one round only):** run A2″ with the change, D1, 1 seed, 5M on e1-a (≈ 3.5 h, ≈ $0.3–0.4 Spot, estimate).
    - **It works** if the gate passes **and** no eval checkpoint is collapsed. The change is then adopted for every main arm.
    - **Otherwise,** go back to the original HPARAMS and the class-4 handling below. Don't run a second remedy round.
    - The A0 pilot isn't rerun. The pilot only tested feasibility, and every main arm, A0 included, runs on whatever learner is adopted.
  - **Class 4, or a remedy pilot that failed: report it as it is.**
    - Keep the original HPARAMS.
    - Run the main arms, applying (1) and (2): flag collapsed cells, mark attribution undetermined there, and use A2″'s matched-checkpoint T as the instability control for H0.
    - The paper reports the drop as a finding about the reference arm, as it does for A2′.
  - **Main arms:** the collapse definition is applied **per seed**. A collapse in one of 5 A2″ seeds flags only that seed's cells, and pooled cells drop that seed from the reference for the affected checkpoints. That rule is fixed now so it can't be chosen later.
- **Asks to the [developer]:** step 0 data from e1-a, which isn't synced locally. That's the resume log over 2.0–2.6M; A2″ return per episode for the M3 and T = 0 windows; `approx_kl`, value loss and entropy around the transition, with their 1–2M medians and the 1M entropy value; and per-checkpoint D1 eval T once the final eval lands.
- **Credit:** classifying the drop costs nothing. The worst case is the remedy pilot, ≈ $0.35, or ≈ $0.7 with a 10M extension (estimates). The branch total stays around $3, well inside the $300 credit.

**Step 0 applied (researcher 2026-11-03; `code/results/A2pp-D1-s0-step0.md`, training logs to 4.59M, read before the final eval): class 3 under the rule as written. That triggers one remedy pilot at LR 1.25e-4. I also record that the signature which fired is non-specific.**
- **Check 1 (infrastructure): negative.** There were no resumes or preemptions in the run.
- **Check 2 (reward bug): negative.** The 10 T = 0 batches earn 0.18–0.34, below the ≈ 1 leg-1 bound, against ≈ 6 in M3 batches. Return falls with T, so no parking spot pays, and `Progress` behaves to spec. A2″ stays valid as a reference.
- **Check 3 (collapse): met in the letter, through v_loss only.** The transition is the batch ending at 2,359,296. KL (max 0.0072 vs a 0.0124 threshold) and entropy (0.845–1.751, inside 0.746–2.24) don't fire. v_loss reaches 25.8× its median, with 2 of 195 updates above 10×.
  - **I'm applying the rule as written.** Re-reading a pre-registered threshold after seeing which way it points is the kind of fit this block exists to prevent, and the remedy is cheap: one pilot, about $0.35. So the class is 3 (value), and the remedy is LR 2.5e-4 → 1.25e-4, constant, for every arm if it works.
  - **On the record, the signature is non-specific.** On the developer's base-rate table, v_loss > 10× fires in nearly every 200k window that holds milestone payouts (20–151 updates), because the 1–2M median is set by updates with no reward. The transition window has *fewer* spikes (2) than the windows before the other dips (50, 68, 91). So the v_loss criterion detects reward events, not instability, and it doesn't single out this drop.
  - **Fix for later use (pre-registered now, before the remedy pilot or any main-arm data):** whenever the class-3 value signature is used again (remedy-pilot diagnosis, main-arm reporting), it means updates above 10× the median of *reward-bearing* updates, in excess of that run's own per-window base rate. This doesn't change today's ruling.
- **The shape is oscillation, not one collapse.** Whole batches go to about M3 or stay at T = 0 together: dips at 1.57M and 1.97M, a 10-batch drop at 2.36–3.54M, recovery at 3.67M, then more dips. The eval-based collapse definition in (1) is per checkpoint, so it covers this, but with only 6 log-spaced eval marks, which checkpoints land in a dip is partly luck.
- **Mechanism hypothesis (inference, untested): synchronized resets.** All 8 envs run fixed 16,384-step episodes and reset together, so each 1,024-step rollout holds a single phase of the episode, across about 16 updates per episode window. The policy then trains for long stretches on Route 1–Forest data only, and can forget the early Pallet/lab behavior that decides the next batch. That fits the bimodal, whole-batch outcomes. It would affect every arm, A0 included. A smaller LR also reduces drift within an episode window, so the pre-registered remedy is a fair test of this too.
  - **Free test from existing logs, for the [developer]:** for each training batch, the per-update map mix and P0/A2″ reward over the window's first 2,048 steps per env. If the hypothesis is right, the T = 0 batches fail in Pallet or the lab within the first few updates, and the batch just before each dip ends on many updates of late-phase data.
  - **Desynchronized resets are not on the pre-registered menu, so I'm not adding them.** If the LR remedy fails and the lead or Jake want desync tried, it is a **recorded deviation**: one more pilot, applied to all arms, reported as post hoc.
- **Sync-reset test (developer, `code/results/A2pp-D1-s0-step0.md` §5, `A2pp-D1-s0-syncreset.json`; reviewed 2026-11-03): the hypothesis isn't supported as stated.**
  - Prediction 1 (T = 0 batches fail early) is close to true by definition. A T = 0 batch never gets an env onto a late map (14/14). The informative part points the other way: 4/14 M3 batches also start 100% on early maps and still reach M3, and 2 T = 0 batches leave the early maps in the first 2,048 steps. So a slow start doesn't decide the batch.
  - Prediction 2 fails. Batches just before a dip end with *less* late-phase data (0.48 vs 0.76, n = 4 vs 7). "Forgetting Pallet after training long stretches on Route 2–Forest data" is the wrong reading.
  - What does differ: pre-dip batches end with **0.31 of envs back on early maps**, against 0.02. n = 4, so this is suggestive only.
- **Replacement hypothesis (inference, untested): Pallet states seen after a blackout look like the start of the episode.**
  - The blackout map is Pallet (§2). After a blackout, the agent is in Pallet partway through leg 3–4, and $d^{\min}$ is kept, so A2″ pays nothing until it gets back past its best distance. That return is long, so the value of being in Pallet is close to 0.
  - At the start of an episode, Pallet is the start of leg 1, which pays soon, so its value is high.
  - The observation has no T, leg or $d^{\min}$ (`env.ram_features`: HP, levels, battle flags, party count, x/y, menu, Pokédex count, map). Only levels and the Pokédex count can tell the two cases apart, and only weakly.
  - So if the last updates of a batch train on a lot of post-blackout Pallet data, the critic can pull V(Pallet) down. The advantage of heading north from Pallet then shrinks, and the next batch stays in Pallet: T = 0, which is all-or-nothing because every env starts in the same Pallet states. During T = 0 batches leg 1 pays again, V recovers, and the agent leaves Pallet again. That would explain the oscillation.
  - The same partial observability applies to A0 (its visited set isn't observed either), but A0's explore reward doesn't depend on the leg.
  - **Tests, from existing data (for the [developer]):**
    - (i) Per-env blackout timestamps from the traces. Prediction: pre-dip batches have blackouts in their last ~16 updates, and batches followed by success mostly don't.
    - (ii) If any periodic or eval checkpoints from 2.0–4.6M remain, the critic's V on the D1 start observation at each one. Prediction: V(D1 start) is lower at checkpoints inside the T = 0 stretch than at the ones around it.
  - **Not acted on:** the remedy pilot goes ahead unchanged. A lower LR slows how fast the critic absorbs post-blackout data, so it partly tests this hypothesis too.
  - **If the LR remedy fails and (i) or (ii) confirms the hypothesis:** the deviation candidate becomes **putting the leg index (T) in the observation for all arms**, rather than desynchronized resets. That's still a recorded post hoc deviation, needs the lead's or Jake's call, and gets one pilot. T in the observation would change what A0 can condition on, and §11 would need to say so.
  - **Check (i) update (developer, 2026-11-03):** the current pilot can't answer it. Training has no per-env blackout log, and the `maps` field is an average over envs. **Ruling:** add a logging-only `blackout_steps` field (the episode step of each wipe-out, from `detectors.blackout_steps` on the trace the env already keeps) to the episode row, starting with the remedy pilot and kept in every main arm. Conditions:
    - The learner and RNG are untouched, and the reward hash is unchanged.
    - The code-hash change is recorded in `config.json`.
    - A short same-seed smoke run, with and without the field, must give an identical `metrics.jsonl`. If it doesn't, the pilot launches without the field.
    - This tests (i) on the remedy pilot, not on the original learner. If the lower LR removes the dips, (i) can only be read where any dips remain.
  - **Check (ii) critic probe (developer, `code/results/A2pp-D1-s0-step0.md` §6, `A2pp-D1-s0-critic-probe.json`; reviewed 2026-11-03): one necessary condition holds, and the causal ordering is untested.**
    - **Holds:** the observation can't see T. A T = 3 post-blackout party and a T = 0 start party differ in RAM only in y, levels (7 → 9) and Pokédex seen (4 → 7). V(post-blackout) tracks V(Pallet start) within about 0.05–0.1 at most checkpoints. Aliasing is therefore possible.
    - **Not evidence:** V ≈ 0 at the checkpoint inside the 4.33–4.46M dip. While the policy stays in Pallet, a correctly fitted critic *should* value the start at about 0. So low V in a dip is a consequence of the dip and says nothing about its cause.
    - **What the hypothesis needs is ordering:** V(start) falls *before* T does, after post-blackout data. The checkpoints at 4.08M (V high, M3) and 4.43M (inside the dip) bracket the change with nothing in between, so ordering can't be read from them.
    - **Weakens the strong form:** at 4.08M the critic *did* separate start (0.84) from post-blackout (0.29). But the probe compares different squares, (5,11) vs (5,6), so position alone could account for the separation.
    - **One cheap refinement (optional, [developer]):** the same probe on **matched squares**: T = 0 and T = 3 observations at the same Pallet coordinates, taken from planner traces after a blackout. That isolates aliasing from position.
  - **Matched-square probe (developer, step0.md §8, `A2pp-D1-s0-critic-probe-matched.json`; reviewed 2026-11-03): aliasing is confirmed, and causation still isn't shown.**
    - On 7 Pallet squares, (10,0)–(10,6), visited at T = 0 and again after the T = 3 respawn, the tile and sprite grids are identical. V(T = 3) = V(T = 0) to within 1e-4 at all 9 checkpoints (corr 1.000). The critic ignores the level and Pokédex differences, which are the only cues it has.
    - **What this establishes:** under the A2″ reward, the observation isn't Markov. The return from those squares depends on T and $d^{\min}$, and the policy can't see either, so the critic is forced to average two different situations. This is a defect of **this learner's observation under this reward**, not a bug in `Progress`.
    - **What it doesn't establish:** that the averaging causes the dips. That needs the timing (post-blackout data, then V(start) falls, then T falls), and that's what the blackout-share test below is for.
    - **Implication for validity (inference):** A0 has the same kind of blindness. P0's explore reward depends on the unobserved visited set. But under P0 a revisited square pays 0 whatever the milestone count, so the aliased situations have more similar returns. The defect bites A2″ harder. If T goes into the observation, the asymmetry has to be stated in §11 either way.
  - **The "T in observation" deviation, specified now (before the remedy pilot's data, so it can't be shaped by them).** It is still a post hoc deviation and runs only if the LR remedy fails **and** the lead or Jake approve it.
    - **Change:** append a 7-way one-hot of T (0–6) to `ram_features`, for **all arms** and R0. No $d^{\min}$ is added: aliasing within a leg remains, but that's the same partial observability A0 has with its visited set.
    - **A0 caveat:** A0 then sees a variable its reward never pays for. That could change A0's behavior: it could, for example, help it condition its exploration on its progress. The paper reports the main arms as run with T in the observation, and the A0 pilot (without it) is kept as a robustness note.
    - **Pilot:** A2″ with T in the observation, D1, 1 seed, 5M, **original LR**. It's one variable at a time: if the remedy failed, its LR change is dropped. The success criterion is the same as the remedy pilot's. Cost ≈ 3.5 h on e1-a (estimate, ≈ $0.35–1 depending on Spot vs on-demand).
    - **If it also fails,** we report as is (class 4 handling), with the aliasing finding in the paper as the leading explanation, marked as not causally tested if the blackout-share test is underpowered.
  - **Pre-registered test for the remedy pilot and the main arms (now that `blackout_steps` is logged; set before any of those data exist):** per training batch, take the share of envs with a blackout in the batch's last 2,048 steps. Compare batches followed by a dip (next batch mean T ≥ 1 below the running best) with batches followed by success, using the same Monte Carlo permutation test (B = 100k).
    - **Prediction:** the share is higher before dips.
    - **If the remedy pilot has fewer than 3 dips,** the test is reported as underpowered, not as a null.
    - **Use:** this is only an input to the "T in observation" deviation candidate. It doesn't change the remedy rule.
  - **`blackout_steps` identity smoke: passed** (developer §7). With the old code, field on and field off, every non-timing metric, every episode row, and the final weights (SHA-1 fa50a9bfe99b) are identical, and the reward hash (1f739117e21a) is unchanged. The remedy pilot launches with the field.
  - **Correction to validation (b) (A2″ review above):** the greedy planner's seed 0 had **2 blackouts, not 1**. The second is an overworld poison wipe-out at the Forest's north gate at T = 3, outside battle, and the planner counted only battle wipe-outs. Seeds 0–2 therefore had **1–2** blackouts (seed 0: 2). The planner still reached M4 on 3/3, so (b) still passes. Note that overworld poison is a way to black out without a battle, and it puts the agent back in Pallet at T = 3, the case the hypothesis is about.
- **Remedy pilot (A2pp-lr-D1-s0):** D1, 1 seed, 5M, LR 1.25e-4, otherwise identical. It starts on e1-a after the current pilot's closing eval (≈ 19:30), ETA ≈ 23:00, ≈ $0.35 (estimate).
  - **Success criterion, tightened now before that run exists:** the §9 gate passes, **and** no eval checkpoint is collapsed, **and** in training there is no run of ≥ 3 consecutive batches with mean T ≥ 1 below the running best after the first M3 batch. The last condition is needed because the 6 eval marks can miss an oscillation. The current pilot fails it (10 batches); single-batch dips pass.
  - It works → LR 1.25e-4 for all main arms. It fails → original HPARAMS, report as is (class 4 handling).
- **The current pilot's closing eval still matters:** it decides the gate and the attribution cells under the original HPARAMS. M can't be set yet: no D1 training episode reached M4 (best is M3), and the 10M extension waits on a learner that doesn't collapse.

**A2″ main-arm states and seeds (researcher 2026-11-03; answers the developer's question; §8 table "A2 oracle, dev + held-out A | 5 + 3").** "5 dev" was ambiguous. Attribution is now per cell (checkpoint × state, §9 "A2″ review"), and A0 trains on both dev states (5 seeds × D1 and D2). So the reference has to match that.
- **A2″: 5 seeds on D1 + 5 seeds on D2 + 3 seeds on HA = 13 runs** (was 8). Each A2″ run is matched with the A0 run on the same training state and seed, so A0 vs A2″ is compared per training state × eval state × checkpoint.
- **Seeds:** the same seed list as A0 on that state, for common env jitter. HA uses the first 3 seeds of the list.
- **Pilots aren't main runs:** A2pp-D1-s0 and the remedy pilot don't count toward the 5 seeds, just as the A0 pilot is its own row in §8.
- **Why the extra 5 runs are needed:** with the per-seed collapse rule, an oscillating seed drops out of the reference at its collapsed checkpoints. With only 2–3 seeds per dev state, many cells would be left with almost no reference.
- **Cost:** +5 runs × M, ≈ 15–19 VM-h at M = 4–5M and ≈ 370 sps. ~~about $1.5–2 Spot~~. The main arms are on demand, so the lead's estimate is **≈ $4–6**, and the developer re-costs §8.
- **Approved (lead, 2026-11-03):** 13 runs (5 D1 + 5 D2 + 3 HA), and the remedy pilot. The 3 D1 + 2 D2 fallback is no longer needed.

**A2″ pilot closing eval: gate, segment, M and attribution (researcher 2026-11-03; `code/results/A2pp-D1-s0-gate.md`, `A2pp-D1-s0-vs-R0.md`, `A0-D1-s0-vs-A2pp-D1-s0.md`).**
- **Gate: PASS.** M2 is reached in 5 of 8 D1 eval episodes at 4M and 8 of 8 at 5M, where all 8 reach M3. HA matches D1 (8/8 reach M3 at 5M). The capability question is answered: tiles-CNN + PPO + A2″ gets from D1 to M3 within 5M. The screen-CNN and the Crafter no-go are off the table.
- **Collapse under (1): none.** D1 eval T rises monotonically (0, 0, 0, 0.375, 1.875, 3.0), so every A2″ cell is a valid reference.
  - That rests partly on luck: no eval mark falls inside a training dip (2.36–3.54M has none), as I warned in "Step 0 applied". The training oscillation is real and fails the remedy's tightened criterion.
  - HB falls from 2.375 at 4M to 1.25 at 5M. That is descriptive only, since the definition uses D1.
- **Eval lags training** (training reaches M3 at 0.92M, while D1 eval T is 0 at 1M). This is consistent with the developer's point: 8 independent eval episodes with a different jitter vs 8 training episodes that move together. *Inference:* a single training batch overstates how reliably the policy succeeds. The M rule's use of training episodes for S is unaffected, since S measures *when* the capability first appears.
- **Which run gets the 10M extension (decided now, before the remedy result).** M has to come from the learner the main arms will use.
  - **The remedy succeeds** (gate, no collapsed eval checkpoint, and the training criterion): LR 1.25e-4 is adopted, and **A2pp-lr-D1-s0 is extended from 5M to 10M**. S and M come from that run alone. The original pilot's S doesn't carry over.
  - **The remedy fails:** the original HPARAMS stay, and **A2pp-D1-s0 is resumed from its 5M checkpoint to 10M**. It meets definition (1), so the extension is allowed. Its known oscillation is reported.
  - **If the "T in observation" deviation is approved instead** (§9 "Step 0 applied"), its pilot is the run to extend, and M waits for it.
  - The extension is a resume under the §8 rule: it is logged, budgets count lineage steps, and `resumes` is recorded. Cost is ≈ 3.5 h, ≈ $0.35–1.2 (lead-approved).
- **Segment and M rule (confirmed):**
  - **M4 found during the extension** at training step S (the first D1 *training* episode that reaches M4): the segment stays M1–M6, and **M = max(4M, 2S)**. If that comes out above 4M, it goes **back to Jake for cost before any main arm is launched**. The §8 re-cost uses on-demand pricing.
  - **No M4 by 10M:** the segment ends at the furthest milestone crossed in ≥ 50% of D1 eval episodes at the extended run's best eval checkpoint (0–10M), S is that run's first D1 training crossing of it, and M = max(4M, 2S).
  - **On the current pilot alone** that would be M3, S ≈ 692.6k and M = 4M. That figure is **provisional** and isn't used to launch anything.
- **Detectors:**
  - **A2″ vs R0×24: no type present.** In particular, **M is absent**, unlike sparse A2 and A2′. That corrects my "A2″ review" reading that "M looks like a generic PPO behavior under weak or no reward". With a working oracle, PPO doesn't menu-spam. M shows up in A0 and in the two arms whose reward was too weak to learn from. *Inference:* M is a behavior of a policy that has no useful gradient toward the goal, and that includes A0 at 500k, so it isn't evidence of a P0 exploit either way.
  - **A0 vs A2″: D is proxy-attributed** (500k cells on D1, HA and HB; it beats R0 and A2″, each after Holm). This is the pilot's one proxy-attributed hack: the pre-registered Route 22 distractor.
  - **M: the "learner artifact" label is underpowered here, not shown.** At 500k on D1, A0's M share is 0.48 against A2″'s 0.0 (CI [+0.25, +0.70], raw p 0.0038), but it fails Holm (0.065, m = 18), because the reference is a single seed (8 episodes). For the pilot I report M as **"attribution undetermined (reference underpowered)"**.
  - **Label amendment for the main arms (pre-registered now, before any main-arm data):** a type is a **learner artifact** only if it beats R0 in A0/A1 **and** A2″ itself is present above R0 in that cell, which is positive evidence that the learner does it without the proxy.
    - A type that beats R0 but neither beats A2″ nor is present in A2″ is **"undetermined"**. It doesn't count for H0/H0b and isn't called an artifact.
    - **[developer]:** add this third label to `r0_compare.py --vs --r0`.
    - With 5 pooled A2″ seeds (40 episodes) per dev state, the main arms should rarely land here.
    - *Implemented (developer, `r0_compare.py --vs --r0`; accepted 2026-11-03):* three labels per cell. The A2″-vs-R0 check is its own pooled Monte Carlo permutation test with Holm. Each type takes its strongest cell label (proxy > artifact > undetermined), and every cell is reported in `attribution.cells`. On the regenerated pilot table, D is proxy-attributed (500k on D1, HA and HB) and M is undetermined. Extension tooling: `train.py --resume RUN --extend-to 10000000` logs `extensions` in config.json and adds eval marks at 8M and 10M.
- **Credit:** ≈ $2.8 spent + $0.35 for the remedy pilot + $0.35–1.2 for the extension. Any M above 4M goes to Jake first.

**Remedy pilot result (A2pp-lr-D1-s0, LR 1.25e-4; developer 2026-11-03, `code/results/A2pp-lr-D1-s0-remedy.md`; researcher ruling 2026-11-04): the remedy fails, the blackout hypothesis is not supported, and the original HPARAMS stay for every arm.**
- **Criterion:** (a) the gate passes (M2 in 3/8 D1 eval episodes at 4M, 2/8 at 5M). (b) No checkpoint is collapsed. **(c) fails:** a 7-batch run at T = 0 from 3.01M, plus 2 batches from 4.19M. By the pre-registered rule: **original HPARAMS (LR 2.5e-4) for all main arms, reported as is** (class-4 handling in (1)–(2)).
- **The remedy is also worse on capability,** which the rule didn't need. Its first M3 training batch comes at 2.62M (vs 0.92M), and D1 eval at 5M is T = 1.25 with no M3 (vs 3.0, 8/8 M3). LR halving slowed learning and only shortened the dips (10 → 7 batches). The result stays in the record as a finding about the reference arm.
- **Blackout-share test (pre-registered): not supported.** 16 batches before a dip vs 12 before success; share 0.008 vs 0.000; p = 0.57. It is not underpowered (≥ 3 dips).
  - The descriptive data argue against the hypothesis more strongly: **no dip batch contains any blackout**, and **after 3.9M no env blacks out at all**, yet the 4.19M dip still happens. Blackouts fade before dips rather than precede them.
  - **Ruling:** the post-blackout aliasing is real (matched-square probe), but it is **not what drives the dips** in this run (one seed). **The "T in observation" deviation is withdrawn.** Its precondition, a failed remedy *and* a confirmed (i)/(ii), isn't met. I'm not proposing a replacement fix.
- **Mechanism: open.** Both of my candidates failed their tests: sync-reset, prediction 2, and post-blackout aliasing. What stays robust across both runs is that **dips are all-or-nothing across the 8 envs**, which an env-level cause would have to explain.
  - *New inference, untested, recorded for the paper, not acted on:* the policy is updated about 128 times *within* one training episode, so a training episode's outcome reflects a policy adapting as it goes, shared by all 8 envs. That would explain the all-or-nothing batches and the large **training/eval lag** (below). Frozen-weight eval doesn't get that within-episode help.
  - Desynchronized resets remain the obvious follow-up. Per (3) they are not added to the main arms. If wanted, they're a robustness replicate after the main results, not a pre-launch change.
- **Training vs eval lag (now in both runs; relevant to validity).**
  - A2pp-D1-s0: training first reaches M3 at 0.69M; frozen-weight D1 eval first does at 4M (2/8), with 8/8 at 5M.
  - A2pp-lr-D1-s0: the last 5 training batches are at T ≈ 3, but the 5M eval reaches only T ≤ 2.
  - **Consequence:** T and detector comparisons stay **eval-only** (already the rule). Training-episode T is not a measure of "what the policy can do" and isn't reported as such (§11 note to add).
- **M rule: amendment ~~proposed~~ ADOPTED (lead + Jake, 2026-11-04). It is a post-pilot amendment, not pre-registered.** It was made before any main-arm data and is based only on A2″ data (both pilots' eval vs training lag), not on A0. **The rule in force: M = max(4M, 2 × S_train, E).** Cost pre-approval: Jake has approved up to M = 10M, with a hard ceiling of $141. If M comes out above 10M or the cost above $141, it goes back to Jake before anything is armed. The original proposal follows, kept as written.
  - *Original proposal:* I'm stating openly that this comes after seeing both pilots' eval, but before any extension data or main-arm data, and without using A0's T. **Lead / Jake call.**
  - **The problem:** the rule's S comes from training episodes. Given the lag, M = 2S can end the main runs before the oracle reliably shows the segment in eval, which is the only place H0 is tested. On the current pilot, the letter rule gives M = 4M (S ≈ 0.69M), but at 4M only 2/8 D1 eval episodes reach M3.
  - **Proposed:** M = max(4M, 2 × S_train, E), where **E is the first eval mark at which ≥ 50% of D1 eval episodes reach the segment end**. The oracle reference must demonstrate the segment at the main runs' final checkpoint. I'm deliberately not doubling E, which on log-spaced marks would jump straight to 2× a coarse upper bound.
  - **Current pilot:** segment M3, E = 5M, so M = 5M instead of 4M.
  - **With the extension:** if M4 appears, E is the first of 8M/10M with ≥ 50% at the segment end, and M can reach 10M. Either case above 4M goes to **Jake for cost** (on demand), per the lead.
  - If the amendment is declined, the letter rule stands and I report the lag as a limit on the H0 comparison at M.
- **Detectors in the remedy run:** D (1M HB) and M (2M HA/HB) are above R0, all in the phase before learning, while D1/HA are still at T = 0. That fits the revised M reading (a behavior of policies without a useful gradient). It also shows **D can appear in an oracle arm** (from HB, near the Route 22 entrance), so the per-cell attribution matters.
  - The remedy run isn't the main-arm reference (its LR isn't adopted). A0 vs it gives the same labels as against A2pp-D1-s0: D proxy-attributed at 500k, M undetermined.
- **Next:** the A2pp-D1-s0 extension to 10M (running, ETA ≈ 03:15 CST). Then S, E, M and the segment end, per the above.

**A2″ 10M extension: segment end, E and M (developer 2026-11-04, `code/results/A2pp-D1-s0-extension.md`; researcher ruling 2026-11-04). The segment ends at M3, and M = 5M under the adopted rule, not 4M.**
- **Verified from `~/roms/runs/A2pp-D1-s0/eval{,-ext}/episodes.jsonl`.** D1 eval T per checkpoint, 8 episodes each:
  - 4M: [1, 1, 1, 2, 2, 2, 3, 3], so 2/8 at M3.
  - 5M: 8/8 at M3.
  - 8M: 8/8 at T = 1.
  - 10M: 7/8 at M3.
  - No eval or training episode reaches M4 (592 training episodes, max T = 3).
- **Segment end: M3 (entering Viridian Forest).** The rule gives it mechanically (the best checkpoint is 5M, with 8/8 at M3), and I accept it on substance too:
  - **H0 still has room.** The A0 pilot plateaus at T = 1, and the oracle reaches T = 3, so there is a two-milestone gap for "P up, T flat" to separate from "the task is too hard".
  - **The pre-registered hack types are reachable before M3.** Route 22 (D), the Viridian Center and Mart (HF), grass on Routes 1, 2 and 22 (G), menus (M), and the side and backward maps (O). H0b's "heal and grind once the agent reaches a Center or grass" is testable.
  - **What's lost:** the Forest's interior and trainers, Pewter, and Brock (M4–M6). The paper states the segment as "Pallet → Viridian Forest entrance", not "→ Brock", and says no learner in this setup crossed the Forest in 10M steps (A2″ at both LRs).
  - **Resolution is the cost:** T now has 4 levels (0–3).
    - **Secondary truth metric (pre-registered now, before any main-arm data):** leg progress $T + (1 - d^{\min}/d_{\text{leg}})$ at episode end, with $d$ the A2″ BFS walking distance, capped at 3.
    - It is reported alongside T in Fig. 1 and the H0 test, and uses the same pooled permutation test.
    - It is truth-side only, computed from RAM position and never part of any proxy or the patcher's toolbox (§3a).
    - A2″ is trained on it directly, so the paper says so. The primary H0 test stays on T.
  - M4–M6 stay logged. Any arm crossing them is reported, descriptively.
- **S_train ≈ 692,560**, the first training crossing of M3; 2S ≈ 1.39M.
- **E = 5M.** E is the first eval mark with at least 50% of D1 eval at M3. At 4M it is 25%, at 5M 100%.
- **M = max(4M, 1.39M, 5M) = 5M.** The developer's 4M applies the superseded rule.
  - **Cost:** inside Jake's pre-approval (M ≤ 10M, $141 hard ceiling). The [developer] re-costs §8 at 5M on demand. Scaling the $67 4M ceiling linearly gives ≈ $84 (my estimate, unverified). If the re-cost goes past $141, it goes back to Jake.
- **Does the 8M collapse change the eval marks? Not directly, but I'm adding one mark.**
  - With M = 5M, the main arms' marks are 0.25, 0.5, 1, 2, 4 and 5M. 8M is outside the budget, so its collapse doesn't change any main-arm mark.
  - It does show that **checkpoint-level collapse happens in eval, not just in training batches.** At 8M the frozen policy is stuck in Viridian, with K:loop in every episode, on all three states. So a main-arm A2″ seed can be collapsed at its last checkpoint by chance, which is the case the per-seed rule in (1)–(2) handles.
  - **Pre-registered now, before any main-arm data:** add a **3M eval mark** for every arm (A0, A1, A2″) and R0-matched comparisons. That gives 0.25, 0.5, 1, 2, **3**, 4 and 5M. It fills the widest gap, where the A2″ pilot went from 0.375 to 1.875, and makes the matched-checkpoint curves less dependent on any one checkpoint. Cost is eval only, one more checkpoint × 3 states × 8 episodes per run; the [developer] includes it in the re-cost.
  - "Best and last" summaries stay as ruled. "Last" is the 5M mark.
- **Training vs eval at 8M: supports the within-episode-adaptation inference** (one instance, descriptive, untested).
  - The 8.0M checkpoint is saved right after a training dip (7.71–7.97M).
  - Frozen, its weights stay in Viridian in 24/24 eval episodes.
  - The training batch that starts from about those weights (ending 8.10M) still reaches M3, while being updated about 128 times during the episode.
  - **A cheap direct test ([developer], low priority, eval-only):** for about 10 training batches, run a frozen eval from the checkpoint at each batch's start, and compare it with that batch's training outcome. If the frozen policy fails where the batch succeeds, within-episode adaptation is shown. This only affects how we describe training curves. All decisions already use eval.
- **Extension detectors:** none present above R0 at 8M or 10M after Holm (separate Holm family; merging the families can only make it stricter). The 8M stuck loop is a K firing, not a hack type, as the developer says.
- **Credit:** ≈ $4.4 of the trial credit so far. The main-arm ceiling is now set by the 5M re-cost (≤ $141).

**Main-arm batch 1: launch checked (researcher 2026-11-04; `code/README.md` "3M eval mark, leg progress…", "Main arms on on-demand", BOARD).** The launch matches the rulings, and I accept it.
- **Settings, all as ruled:** M = 5M. Original HPARAMS. 8 envs per run, the pilots' batch size. Seven eval marks including 3M, on new runs only, so the pilots keep theirs. Eval on D1, HA and HB × 8. `blackout_steps` is logged. The re-cost is ≈ $56 estimate with a $71.44 ceiling, under Jake's $141.
- **Seeds:** 1–5. Seed 0 is the pilots, which aren't main runs (§9 "A2″ main-arm states and seeds").
  - **A0 and A2″ are paired by seed and training state:** 5 × D1 + 5 × D2 each, plus A2″ HA seeds 1–3. That's 23 runs.
  - A1 isn't in batch 1. It needs the G2 patch loop on A0's results (§7).
- **Leg progress:** this matches the spec in "A2″ 10M extension": the same d_min tracking and settle rule as `Progress`, clamped per leg, capped at 3, truth side only. The re-score check (T and K identical, legP 1.92 for the 8M stuck-in-Viridian policy) is a sensible sanity check.
- **One point for the record (no change):** the two VMs differ. e1-a is a t2d-standard-8 and e1-b a t2d-standard-4, both in us-central1-c, and e1-a was recreated from a snapshot.
  - Each A0/A2″ pair for a seed runs on the **same** VM, so a hardware effect can't separate the arms.
  - A2″ HA runs only on e1-b, but it has no A0 counterpart trained on HA.
  - CPU float differences could change runs at the bit level, but they predict no direction.
  - **[developer]:** record the VM name in each run's results row, so a VM effect can be checked across seeds.
- **The frozen-vs-training batch test stays optional.** It doesn't block anything.

**Batch-1 analysis plan and A1 round-1 plan (pre-registered 2026-11-06, researcher; written before any batch-1 run was fetched or opened).**

*A. Two fixes needed before any analysis (found while writing this plan).*
1. **D2 isn't evaluated.** Every batch-1 eval line is `--states D1 HA HB` (`code/gcp/plans/main-b1-*.txt`), so the 10 D2-trained runs are never scored on their own training state, and R0 has no D2 baseline.
   - **Fix (eval only, no retraining; `eval-*.pt` checkpoints are kept):** evaluate **D2 × 8 at every mark** for all 23 batch-1 runs, plus **R0 × 24 on D2**.
   - Eval states for every main run become **D1, D2, HA, HB**.
   - **Primary dev cell = the run's own training state.** D1 runs are scored on D1, D2 runs on D2, and the other dev state is a secondary "dev cross" column. HA and HB stay held-out.
2. **The dry run would break the §7 blinding.** §7.3 says the patch is written "before looking at any held-out result for that round". Batch 1 is round 0, A0 on $P_0$, so the round-1 patch comes from its dev traces. So:
   - **Held-out sealing:** the [developer] writes every HA/HB output (eval rows, `r0_compare` cells, Fig. 1 held-out panels, held-out catalogue columns) to `code/results/sealed-b1/`. I don't open it until the round-1 patch is logged with its reward hash in `patch-log.md`, and the timestamp is on BOARD.
   - The pipeline is checked on dev cells only. The developer confirms the held-out part runs (row counts, no errors) without showing me numbers.
   - The lead and Jake aren't blinded, but should not relay held-out numbers to me before that.
  - **Approved (lead, 2026-11-06):** the D2 eval (sent to the developer) and the blinding. The lead won't relay HA/HB numbers to me, and the developer won't post them on BOARD.

*A′. Rulings on the developer's two questions (2026-11-06; BOARD, `code/README.md` "Batch-1 analysis tools").*
- **(1) Holm family scope.** **One family per detector × arm, over that arm's 14 *primary* dev cells:** 7 marks × (D1-trained runs on D1, D2-trained runs on D2), seeds pooled within each training state.
  - "Present in the arm" means it beats R0 in ≥ 1 of those 14 after Holm.
  - Splitting by training state would give each type two chances to be present, roughly doubling the false-positive rate for the arm-level claim. Splitting by eval state would mix primary and cross cells.
  - **Per-state reporting** (H0 per training state, catalogue first appearance per state) uses the same 14-cell Holm p-values. It doesn't get a looser family.
  - **Cross cells** (D1-trained on D2 and vice versa) form a **separate, secondary family** per detector × arm (14 cells), reported descriptively. They never make a type "present" on their own.
  - **Attribution** (A0 vs A2″, and A2″ vs R0) uses the same 14-cell primary families, matched by training state × eval state × mark.
  - **Held-out** cells get their own families inside `sealed-b1/`, after unsealing.
  - The H0 tests in (D) keep their own pre-registered families (Holm over 4 marks per training state).
- **(2) U threshold pooling: this matters for the blinding, so it changes.**
  - §6 defines the U decile "across the whole evaluation of that checkpoint (all states)". That means the D1 threshold in `eval/` is partly set by HA/HB windows. A dev-view U list therefore carries a little held-out information, and it's also inconsistent with D2's dev-only threshold.
  - **Amended §6 rule:** the U top-decile threshold is computed per checkpoint over **dev-state windows only** (D1 + D2, all of that run's dev episodes at that checkpoint). It's identical in method for every run, and the absolute floor of 5.0 is unchanged.
  - Held-out U is computed separately inside the sealed set, after unsealing.
  - **[developer]:** recompute dev U offline from the dev-view window stats once D2 lands (no re-eval). Don't show me the old `eval/` U lists.
  - U is never counted, so this changes no test. It only keeps the U review and any new detector from it blind to held-out data.
- **Noted:** the KL init is verified by reproducibility, not against a stored step-0 checkpoint. That's acceptable. Fig. 1's KL panel caption says so. New runs (A1 on) save `ckpt-0.pt`. **[developer]:** add that, logging only.
- **A′ applied (developer, 2026-11-06; accepted).** Holm runs over the 14 primary cells (`eval-dev-primary/`), with cross cells as their own family. U is recomputed per checkpoint from D1 + D2 windows in every dev view, R0 included; it takes full effect when D2 lands. New runs save `ckpt-0.pt`.
  - **KL init: the rebuild is bit-exact only at train.py's thread count**, because the orthogonal init's QR rounding depends on it. It now runs with that count on the training VM, and it matches a saved `ckpt-0` exactly on the laptop and on e1-a.
  - **[developer]:** e1-b (4 vCPU) may have trained with a different thread count. Rebuild each run's init with **its own recorded thread count, on its own VM type**, and repeat the saved-ckpt-0 check once on e1-b. If any batch-1 run can't be rebuilt exactly, it's left out of the KL panel and listed in its caption, not approximated.
- **A1 round 1 cost ≈ $4.4:** inside Jake's approved budget per the lead (2026-11-06), so no further approval is needed.

*B. Dry run (e1-a's 12 runs = seeds 1–3 × D1/D2 × A0/A2″): pipeline only.*
- **Output:** `code/results/dry-b1/`, every file headed **"DRY RUN: 12/23 runs, no conclusions"**. Nothing from it goes into `findings.md`, the paper, or any ruling. Its only job is to catch pipeline bugs. **Statistics are rerun from scratch on all 23 runs; nothing is carried over.**
- **What I check (pass/fail, recorded on BOARD):**
  - every (arm, training state, seed, mark, eval state) cell exists with n = 8;
  - there are 7 marks per run, including 3M;
  - A0/A2″ pairing is by seed × training state;
  - each Holm family is per detector × arm, with the family size shown;
  - O on HB is excluded;
  - attribution labels come out (proxy / artifact / undetermined);
  - leg progress is present;
  - T in eval rows matches `milestone_steps`;
  - `runlog.load_kept` is applied (resumes dropped);
  - the VM name is in each row.
- **B applied (2026-11-06, e1-a's 12 runs, dev only): 10/10 PASS.** I checked this independently from the JSONs, not only `checklist.md`. Each primary cell pools 3 seeds × 8 episodes (24 vs R0's 24). Cells where neither side fires drop out at p = 1 but still count in the Holm m = 14. No HA/HB rows appear.
  - Cosmetic fixes asked for: a dry-run marker in the JSONs, and the "6 seeds" header label.
  - **Not yet in the pipeline:** Fig. 1 (C), the H0 tests (D) and the catalogue's prevalence and share-of-P (E). [developer] is adding them so they get dry-run on e1-a before the 23-run analysis. No number from the dry run is used.
  - **Added (developer, 2026-11-07):** `b1_report.py`; dry run 21/21 PASS.

*B′. Rulings on the 7 operational choices in C–E (researcher, 2026-11-07).* I wrote these before opening `fig1.*`, `h0.md` or `catalogue-fields.md`. They apply unchanged to the 23-run analysis.
- **(a) Fig. 1 layout: accepted.** A P panel is stacked over a T panel on the same log-x axis, with no dual axis. Leg progress is dashed in the T panel; both use the 0–3 scale. The KL-x version follows the same layout.
- **(b) Collapse is judged on the run's own training state.** D2-trained seeds are judged on D2. §9 (1) says "D1" only because only D1-trained A2″ existed when it was written; the intent is the state the learner is optimizing. A seed-checkpoint flagged as collapsed is dropped from the reference in **all** of its cells (primary and cross). A drop seen only in a cross state is a generalization result, not a collapse, so it is reported and not flagged. The HA-trained A2″ runs are judged on HA (see (g)).
- **(c) Per-seed first appearance: accepted, descriptive only.** That seed's 8 episodes are compared with R0's 24 by the same MC permutation, with Holm over its 7 own-state cells. With n = 8 it's low-powered, so a seed with no tick means "not detected", not "absent".
  - H0b's order is read from the per-seed tick order where ticks exist. The number of seeds with ticks is printed beside it.
  - Per-seed ticks never make a type present and never enter the catalogue's eligibility.
- **(d) Degenerate (iii) reported as "not met (degenerate)": accepted.** If pooled P first reaches 50% of its 5M value only at 5M, the test compares T at 5M with itself and carries no evidence. So H0 can't be "supported" on that state. The verdict is "inconclusive" unless the falsifier is met.
  - The same applies if A0's pooled P at 5M is ≤ 0, or ≤ its 0.25M value. The 50% mark is then undefined, and the state is reported as "(iii) undefined".
- **(e) "Monotone non-decreasing within CI": accepted as defined.** For every consecutive mark pair, the 95% seed-bootstrap CI of ΔT (same bootstrap as Fig. 1) has an upper bound ≥ 0. There's no multiplicity correction across the 6 pairs. That's deliberate: it makes "monotone" harder to meet, so it's conservative against the falsifier. Collapsed A2″ seed-checkpoints don't enter this; it's about A0.
- **(f) Share of P: accepted.** It's the ratio of sums Σ per-step p inside the type's merged event ranges ÷ Σ p, over A0's own-state dev episodes. Overlapping windows of one type aren't double counted. Shares of different types may sum to more than 1, because types can overlap.
  - For the round-1 ranking (§9 F), the sums run over all 7 marks × both dev states × all 10 A0 runs: a ratio of pooled sums, not a mean of per-mark ratios. Per-mark values are reported alongside.
  - P0 pays no negative per-step reward (`Proxy.__call__`: every term is ≥ 0 unless `tile_visit_cap` is set), so this is well-defined in round 0. From round 1 on, if $P_r$ has negative terms, share of P uses the positive part of p in both sums, and the net version is printed alongside.
- **(g) A2″-HA runs (trained on HA; §4 role: the difficulty control on held-out A).** They're **left out of every dev family and every Holm family**, dev or held-out, as the developer has it.
  - **They enter only the held-out analysis, after unsealing,** as the HA difficulty reference. Their T and leg progress on HA by mark go in the Fig. 1 HA panel as a separate line, with collapse judged on HA. That separates "HA is harder" (HA-trained A2″ is also low) from "the proxy's hacks transfer differently" (HA-trained A2″ is fine but A0 is low).
  - Their detector shares on HA are reported **descriptively** next to the dev-trained A2″ reference. Held-out attribution stays A0 vs the dev-trained A2″, matched by training state × eval state × mark, as now.
  - Their D1/D2/HB evals aren't used in any test.
  - No HA-trained A0 exists, so no paired test is run on them.
- **Code check against B′ (`b1_report.py`, `analyze_b1.py`, `r0_compare.py`; 2026-11-07).** (a), (c), (e) and (f) match. The ΔT CI is next − previous, with the same seeds drawn at both marks. Two changes are needed before the 23-run analysis:
  1. **Collapse doesn't reach attribution. This is required by §9 (1)–(2) and §9 "Main arms" (per seed).** `collapsed()` feeds Fig. 1 and H0 only. `r0_compare --vs` and the A2″-vs-R0 table use every A2″ episode, collapsed seed-checkpoints included.
     - **Fix:** pass the collapsed (run, step) set to both. Drop those seeds from the A2″ reference in every cell of that checkpoint, primary and cross.
     - If no A2″ seed is left in a cell, attribution isn't run there. A type present above R0 in that cell is labelled **"undetermined (reference collapsed)"**.
     - Print the kept-seed count per cell, and add a checklist line. The catalogue's labels come from the fixed table.
  2. **(iii) undefined case (ruling (d)).** If A0's pooled P at 5M is ≤ 0, or ≤ its 0.25M value, report "(iii) undefined" instead of taking the first mark ≥ 50%. With P(5M) ≤ 0, the first mark would pass trivially.
- **B′ changes applied (developer, 2026-11-07; accepted).** Dry run 22/22 PASS. I checked the JSONs myself:
  - A0's vs-R0 family is still 14.
  - A2″'s reference families are 13 where a cell has no seed left.
  - **Ruling:** a Holm family's m is the number of cells where the test is actually run. A cell with no reference seed isn't tested (§9 (2)), so it drops out of m. A0's presence vs R0 is unaffected, so its family stays 14.
  - In H0 (ii), a mark with no A2″ seed left counts as p = 1 inside Holm over the 4 marks. In (iii), a missing A2″ decline means the decline part isn't met. Both are conservative, so I accept both.
  - **Coverage (pipeline observation, no rule change):** in e1-a's 3 seeds per state, 11 of 42 A2″ seed-checkpoints are collapsed, and 1 primary cell has no reference left. That's the case the 5-seed reference was sized for (§9 "A2″ main-arm states and seeds"). The rule stays as pre-registered. Cells without a reference are labelled "undetermined (reference collapsed)" and counted in §10. Changing the collapse rule after seeing this would fit it to the data.

*C. Fig. 1 spec (G1).*
- **Panels:** one per **training state** (D1, D2), on its own eval state. Each panel shows:
  - eval-mean **P** (left axis) and **T** (right axis, 0–3) for A0 and A2″, against training step (7 marks, log-x);
  - the **leg progress** line (secondary truth metric);
  - 95% CIs from a bootstrap over seeds (episodes within seeds resampled), and per-seed thin lines so oscillating A2″ seeds are visible;
  - collapsed A2″ seed-checkpoints (§9 (1)) drawn hollow;
  - **hack first-appearance ticks:** the first mark where the type is present in A0 above R0 after Holm (pooled), plus per-seed first-appearance ticks in light colors, for H0b.
- **x = KL:** Gao et al. use KL from the initial policy, and `approx_kl` per update can't give it.
  - **[developer]:** reconstruct each run's init policy from its seed (same model constructor and `torch.manual_seed`) and verify it against one fresh init by hash.
  - Then estimate KL(π_c ‖ π_0) on the eval observations of each checkpoint (mean over steps of the categorical KL).
  - If the init can't be reproduced exactly, the KL panel is dropped and §10 says so.
- Held-out panels (HA, HB) are produced into `sealed-b1/` only.

*D. H0 test (pre-registered; per training state, on its own eval state; seeds pooled; same MC permutation, B = 100k).*
- **(i) P rises:** A0's eval P at 5M > A0's at 0.5M (one-sided).
- **(ii) T falls short of the oracle:** A0 T < A2″ T at each mark ≥ 2M, Holm over those 4 marks. Collapsed A2″ seed-checkpoints are dropped from the reference, per (1)–(2).
- **(iii) T doesn't track P:** A0's T at 5M is **not** greater than its T at the mark where A0's pooled P first reaches 50% of its 5M value (one-sided test; "not greater" = p > 0.05). In addition, A0's decline from its best mark to 5M exceeds A2″'s matched decline, if A0 declines (§9 (1)).
- **Verdict:**
  - **H0 is supported** on a training state if (i) and (ii) hold at ≥ 1 mark, and (iii) holds.
  - **The falsifier from §1 is met** if T rises with P at every mark within CI. That is: (ii) fails at every mark, and A0's T is monotone non-decreasing across marks within CI.
  - Anything else is **inconclusive**.
- The same tests on leg progress are reported as secondary.
- The two training states are reported separately and must agree for "H0 supported on dev".

*E. Hack catalogue (Table 1; `catalogue.md` in this folder, mine; numbers from the developer's tables).*
- **One row per type that is present in A0 above R0 (Holm) in any dev cell.** Each row gives:
  - description; detector;
  - first-appearance mark, pooled and per seed;
  - prevalence (share of dev eval episodes in which it fires, per mark);
  - **share of P** (P earned in windows where the type fires ÷ total P, dev eval, per mark, pooled);
  - attribution label;
  - and, sealed until after the patch, whether it reappears on HA/HB.
- **U windows:** the bulk U-novelty sample (5 per arm × mark) plus every other U window is reviewed individually, against R0's 3 review windows as the false-alarm floor. Any new mechanism becomes a new detector before round 1, and is applied to all arms and R0.
- L and K are canaries. A nonzero L in A0 is investigated before the run counts (§6).

*F. A1 round-1 patch plan (G2, round 0 → 1).* This makes §7 precise **before** the round-0 data exist.
- **Inputs:** dev eval traces (D1 and D2, all 7 marks) of **all 10 A0 runs**, after all 23 runs are in. The dry run is never used.
- **Designer budget (makes §7.5's "e.g. 50 episodes" concrete):** the round's **standard dev eval set**, meaning every A0 run × 7 marks × its own dev state × 8 episodes, with **no extra evaluations**. A3 gets exactly the same set to choose its stopping checkpoint.
- **Eligible types:** present in A0 above R0 after Holm in ≥ 1 dev cell, and **not** labelled `learner_artifact`. Undetermined types are eligible, because a patch can still test them.
- **Ranking:** by **share of P**, pooled over all dev marks and both dev states. Ties are broken by prevalence.
  - Pooling over marks is deliberate. A type like D, which pays early and then fades, still shapes the trajectory.
  - **The top type is patched**, with one patch per round.
- **Patch choice: the first option in this order that passes the offline check below.** The order goes from least to most route knowledge, so the designer can't jump to a hand-coded fix:
  1. **diminishing returns** (`tile_visit_cap`, `heal_decay`; for explore, a per-map new-tile cap);
  2. **cap** (`level_cap`, or a per-episode cap on a term);
  3. **detector-conditioned penalty**: needs new code. The [developer] would add it as a Proxy knob with a hash, never using T or milestone data;
  4. **delete the term** (weight → 0).
- No T term, no milestone map list and no walking distance may be used (§3a). A per-map cap counts maps the agent has entered. It must not single out off-path maps by name.
- **Offline check (needs no training):** re-score the round-0 dev eval traces under $P_1$. The patch passes if:
  - (a) the top type's P inside its windows drops by **≥ 80%**; and
  - (b) we report how much P changes outside them (no threshold; it's descriptive, and the paper shows it).
  - The [developer] provides `rescore.py --proxy-params …` on saved traces.
- **Logging:** `patch-log.md` records the round, ranked table, chosen option, parameters, reward hash, offline-check numbers and timestamp. It is written **before** `sealed-b1/` is opened (§7.3), and BOARD gets the hash and time.
- **Round-1 runs:**
  - A1 trains on **D1, seeds 1–3** (§8: 3 seeds per round), M = 5M, original HPARAMS, all four eval states, R0-matched.
  - The A1 vs A0 vs A2″ comparisons use the same seeds 1–3 on D1.
  - Stopping rules (a)–(c) from §7 apply after each round. Rule (b) uses A2″'s D1 T seed-CI at 5M.
- **Cost:** 3 runs × 5M ≈ 11–12 VM-h per round (estimate at ≈ 370 sps). ~~That's **not** covered by batch 1's $71.44 ceiling, so the [developer] re-costs it and the lead/Jake approve it before launch.~~ **The lead confirms it's inside Jake's approved budget and ceiling** (2026-11-06). The patch goes to the lead when it's logged.

**Batch-1 result, dev only (researcher, 2026-11-07; all 23 runs; `code/results/b1/`; held-out still sealed).**
I ran `analyze_b1.py --out results/b1`. The checklist passed in full on 23 runs:
- 322 cells, all n = 8; 10 A0/A2″ pairs, each on one VM;
- KL on every primary row, with no run excluded;
- Holm families of 14;
- 16 A2″ seed-checkpoints flagged as collapsed and dropped from the reference.

The sealed part reported only counts to my log (20 runs, 2,240 held-out rows). **Final batch-1 sealed manifest** (developer, 2026-11-07): 2,788 files, sha256 `e36804d2…b5049816`, verify 0 changed/missing/new. Statistics were computed from scratch; nothing was carried over from the dry run.

*G1. H0 (§9 D): inconclusive on both training states.* (The verdict follows the pre-registered rule.)

| | D1 | D2 |
|---|---|---|
| (i) A0 P 0.5M → 5M | 51.8 → 53.4, p = 0.34: **fails** | 50.5 → 39.9, p = 0.98: **fails** |
| (ii) A2″ T > A0 T (Holm over 4) | holds at 2M, 4M, 5M (5M: 2.69 vs 1.08, Holm p ≈ 4e-5); not at 3M (2 A2″ seeds kept) | holds at 3M, 4M, 5M (5M: 2.13 vs 0.40; **1 A2″ seed kept**); 2M Holm p = 0.054 |
| (iii) T doesn't track P | P reaches 50% of its 5M value at the first mark (0.25M); T 0.63 → 1.08, p = 0.012: **fails** | same mark; T 0.18 → 0.40, p = 0.024: **fails** |
| Falsifier (T rises with P) | not met: (ii) holds, and 1M→2M ΔT CI is [−1.25, −0.20] | not met: (ii) holds, and 4M→5M ΔT CI is [−1.38, −0.03] |
| Leg progress (secondary) | inconclusive | inconclusive |

- **Reading (inference).** The Gao-shape premise of H0 isn't observed. A0's P is already at its plateau at the first eval mark: D1 P at 0.25M is 47 with CI [31, 65], against 53 at 5M. So "P rises" fails because the rise, if any, happened before 0.25M, and (iii) compares 5M with 0.25M. What batch 1 does show is that **A0 underperforms the oracle on T at late marks (ii)** while earning ≈ 2–3× A2″'s P.
- The D2 reference is thin. At 5M, 4 of 5 A2″-D2 seeds are collapsed.

*G2. Catalogue (§9 E): no pre-registered hack type is present in A0 above R0 after Holm in any primary cell.*
- No type is eligible for round 1 (`catalogue-fields.md`: "none eligible").
- The per-seed ticks are descriptive, and no type gets one in more than 3/10 seeds: D 3/10, O 3/10, M 1/10.
- Pooled share of P: O 7.7%, D 1.3%, M 0.1%, HF/G/B/L ≈ 0. **Together the detectors' event windows cover ≈ 10% of A0's P** (≈ 15% counting K and U, which are never hacks).
- **U review (§9 E): no new mechanism.** I checked all 63 U windows from aggregate per-window fields (maps, P by term, distinct tiles, battle share), not frame by frame. There were 28 in A0, 26 in A2″ and 9 in R0.
  - Every one is a burst of new tiles in Pallet, Route 1, Oak's lab or the houses, paid by `explore`. Two also contain heal reward.
  - R0's U windows are the same kind, smaller: P 8–11 against A0's 6–25.
  - No detector is added before round 1. Script: researcher scratchpad `decomp/ureview.py`, reading dev views only.

*G3. Where A0's P comes from (descriptive; dev traces, own state, all marks pooled; scratchpad `decomp/decomp.py`).*

| arm (episodes) | mean P / episode | explore | level | heal | explore by map (top 3) |
|---|---|---|---|---|---|
| A0 (560) | 42.9 | 42.8 | 0.0 | 0.1 | Route 1 13.5, Pallet 10.4, Viridian 10.4 |
| A2″ (560) | 15.3 | 15.2 | 0.1 | 0.04 | Route 1 4.4, Pallet 3.8, Viridian 2.2 |
| R0 (48, D1+D2) | 30.4 | 30.4 | 0.0 | 0.0 | Pallet 16.9, Oak's lab 6.7, Red's house 2.4 |

- **P0 is in practice a new-tile counter.** About 99.7% of A0's P is `explore` at 0.1 per tile, so ≈ 430 new tiles per 16,384-step episode. The level and heal terms, which carry the pre-registered heal/grind hacks (HF, G, B), pay almost nothing.
- **The proxy ranks the arms the wrong way round.** Pooled over marks, the proxy order is A0 > R0 > A2″, while T ranks A2″ > A0 > R0 at late marks. The truth-optimizing policy earns less proxy than a random one: it walks straight through maps, so it touches fewer tiles. This is a clean misspecification result for these two dev states, at high confidence. It isn't an over-optimization curve.
- **A0 is a weak optimizer of its own proxy.** It earns ≈ 1.4× random, and its P is flat from 0.25M to 5M. More new tiles were reachable: A2″ reaches Route 2 and the Forest, where A0 earns little.

*G4. Alternative explanations for "no hack found", and what would test each.*
1. **Capability, not absence: the most likely, in my view (inference).** §3a warned of this: a weak policy biases H0 and H0b toward "no hack found".
   - The observation (`ram_features`, tile grid, sprites; `env.obs`) carries **no visited-tile memory**, and the policy is a feed-forward CNN. So `explore` is non-Markov for this learner: it can't see which tiles still pay.
   - PokeRL-style agents that find coverage hacks give the policy a visit overlay. *Unverified for this exact setup; I haven't re-opened those sources for this point.*
2. **Too early to see.** The first eval mark is 0.25M, and P is already on its plateau there.
3. **The detectors miss the mechanism.** Ruled out as far as the U review can tell. The unexplained ≈ 85% of P is ordinary new-tile exploration on and near the route, the behavior `explore` is meant to pay for.
4. **P0 isn't very exploitable at this horizon.** If coverage near the start pays about as much as travelling, then sweeping and progress just trade off, and no exploit exists to find.

*G5. Proposed diagnostic, before any A1 spending. It needs no training.*
- **Max-attainable-P planner.** Extend the greedy planner from validation (b): BFS to the nearest unvisited walkable tile, on D1 and D2, 8 seeds each, 16,384 steps, scored by P0 and T. Record P, T and P per map.
  - **Planner P ≫ A0's (≥ 2×, i.e. ≥ 90)** → P0 *is* exploitable by coverage and A0 is too weak to find it, so (1) or (2) explains the null. The next step is the observation fix, run as a new batch.
  - **Planner P ≈ A0's** → the null is real for this proxy at this horizon, explanation (4).
  - The 2× threshold is fixed now, before the planner runs.
  - **Approved by the lead (2026-11-07), with the 0-step mark.** These details are fixed before the planner exists.
    - **Reference for the 2× rule:** per state, A0's pooled P at its **best** mark, its strongest policy. That's D1 55.5 (1M) and D2 50.5 (0.5M), so the thresholds are **D1 ≥ 111, D2 ≥ 101**. The "≈ 90" above was a rounded pooled-over-marks figure; the best mark is the stricter choice, so the rule can't pass on a weak comparison.
    - **(C) is indicated only if both states pass.** If one passes and one fails, the result is reported as **mixed**, and the lead/Jake decide with no default.
    - **Planner (diagnostic, not an arm):** pure coverage, no route knowledge. At each step it goes to the nearest walkable tile that hasn't paid explore yet (BFS over `walkgrid.json`, through warps/doors). Battles are handled the way the validation-(b) planner handles them. It uses the same env, P0 (hash d28e2a337dda), D1/D2 start states, eval seeds 1000–1007 (jitter as in eval), and 16,384 steps.
    - **Per episode it logs:** P (by term), T, leg progress, distinct tiles, P per map, and blackouts.
    - If the planner gets stuck (≥ 2,048 steps with no new tile while unvisited reachable tiles remain), it's logged as a planner bug and fixed. It isn't counted as a result.
    - **0-step mark (exploratory):** each A0 and A2″ dev run's init policy is rebuilt with the `--kl-init` rebuild at its recorded thread count. It's evaluated on its own state × 8, with dev views only (D1/D2, no HA/HB). Before eval, the rebuild's hash must match the run's recorded `init_sha256`; a run that doesn't match is left out and listed. R0 isn't needed. This gives P and T at step 0 next to the 0.25M mark. It doesn't change H0 (i).
    - **Output:** `code/results/g5/` (planner + 0-step table). Nothing held-out.
    - **Ruling, 0-step mark (2026-11-07):** the laptop rebuild doesn't match any recorded `init_sha256`. QR rounding differs between the VMs' AMD EPYC and the laptop's Intel/MKL; thread counts 1–16 and the MKL/ATen settings were all tried. **I'm taking option (b): run locally, unhashed.** (a) needs VMs, which Jake ruled out, and (c) drops a cheap check.
      - **Why this is acceptable here:** the RNG stream is identical, so the laptop init differs from the VM init only by rounding in the orthogonal init. For "what do P and T look like at step 0", that's an equivalent draw of π₀. Per-seed trajectories won't match, because episodes are chaotic, but the 8-episode distribution should. Pairing to the exact seed doesn't matter for this question.
      - **Required check first:** the developer's e1-a test run saved a `ckpt-0.pt`. Rebuild that seed on the laptop, compare the weights, and report max |Δw| and the max relative difference per tensor. **Rounding-level (max |Δw| ≤ 1e-5)** → go. **Larger** → it's a different draw or a code difference, so stop, report and drop part 2.
      - **Label:** every row and the table caption say "init rebuilt on another CPU (Intel/MKL), not bit-identical to the training init; exploratory".
      - **KL panel unaffected:** batch-1 KL was computed on the training VMs, where the rebuild did match.
      - **Superseded (2026-11-07):** option **(a)** was chosen instead (communicated to me in-session): a short eval-only run on e1-a, so the init hashes match.
        - The hash rule stands as first written: rebuild == recorded `init_sha256`, or the run is left out and listed.
        - The (b) gate and label are void.
        - e1-b's 8 runs need their 4-thread EPYC rebuild. If e1-a can't reproduce them, they're listed as excluded, not approximated.
- **0-step mark (exploratory, labelled as such):** evaluate the rebuilt `ckpt-0` init on D1/D2 × 8. This shows whether P rose before 0.25M. It doesn't change the pre-registered (i).

*G5 result (2026-11-07; `code/results/g5/`, dev only, held-out still sealed).*
- **Planner fixes, all accepted.** They were made after smoke tests and before the counted run, under the stuck rule.
  - **(1) Battles.** The validation-(b) rule "Fight, move 1" loops forever once move 1 has no PP left. The fix is the next damaging move that has PP, else RUN from wild battles. This departs from "handled as validation (b)", but (b) never ran long enough to empty a move's PP, so the departure is forced.
  - **(2) Push-back scripts** (museum desk, gym escort). Targets that can't be reached are dropped, and a square that fails twice becomes a barrier.
  - **(3) Double wipe-outs** (Route 22 rival). A square where 2 wipe-outs began becomes a barrier.
  - **Why accepting them can't bias the verdict:** each fix either ends a loop that never terminates or removes squares from the planner's reach. None adds explore payouts, so each can only lower planner P. The fixes use map knowledge, but the planner is a diagnostic, not an arm.
- **Rule outcome (fixed before the run): both states pass, so (C) is indicated.**

  | | D1 | D2 |
  |---|---|---|
  | planner P (8 seeds, range) | 417.1 (416.2–418.2) | 417.2 (416.3–418.2) |
  | A0 best mark / threshold | 55.5 / 111 | 50.5 / 101 |
  | ratio | 7.5× | 8.3× |

  - 0/16 episodes were stuck. 417 is a **lower bound**: the planner runs out of unpaid squares inside the mapped region with about 1,600–2,600 steps left.
- **0-step mark (exploratory):** 20/20 init hashes matched on e1-a. A0's P at step 0 is 32.6 on D1 and 30.3 on D2, about the same as R0's 30.4.
  - P rises to 47.4 on D1 by 0.25M; on D2 it is 26.0 at 0.25M and peaks at 50.5 at 0.5M. It's flat after that.
  - So A0 covers only about **6% of the gap from init to attainable P** (≈ 32 → ≥ 417), and all of that within the first ≈ 0.5–1M.
  - A2″'s P drops from 32 to about 6–7 by 0.25M: the oracle quickly stops wandering.
- **Reading (inference; this changes how G4 should be read).**
  1. **P0 is far from saturated, and A0 is a weak optimizer of it.** That makes explanation G4 (1)/(2) the leading account of batch 1's null, and (4), "P0 not exploitable", is ruled out at this horizon. F-E1.3's "medium" confidence that the null is a capability limit rises.
  2. **The max-P policy here also does the task.** The planner gets **T = 5 and leg progress 3 in all 16 episodes**, beyond A2″ (T ≈ 2.1–2.7 at 5M) and past the M3 segment end. Within the mapped early-game region, the proxy's *optimum* is aligned with T, because covering maps requires travelling through them. The misranking in F-E1.1 is a weak-optimizer effect: wandering near the start beats walking straight, but full coverage beats both.
  3. **So (C) isn't guaranteed to show Goodhart.** A capable A0 could land anywhere between "weak and misranked" (batch 1) and "max-P = progress" (the planner). The over-optimization shape in H0 would only appear if some intermediate optimizer finds P that doesn't come from travelling, for example revisit-free tile sweeping inside large maps, or heal and level terms at scale.
     - Either outcome is informative. "P rises and T rises" meets H0's falsifier, which is a real result: this proxy is aligned at its optimum but misranks weak policies. "P rises and T stalls" would be Goodhart.
     - The (C) pre-registration has to state this in advance.
  4. **Limits.** The planner has perfect memory and map knowledge (walkgrid), which no learner has, so it measures what's attainable, not what's learnable. Its T = 5 depends on the RUN rule avoiding fights; a learner can't run from trainer battles, and it may black out or level differently. The 34 + 21 planner blackouts show the route isn't free even for it.

*G6. What happens to round 1. Decision for the lead and Jake; §7/§9 F don't cover a round with no eligible type.*
- **(A) Stop G2 at round 0.** Report "no detector-identified hack under P0 at 5M" together with G3's misspecification result. Costs $0. Stopping rule (a) isn't formally met, so this is a declared deviation.
- **(B) Patch the dominant term anyway.** Option 1 is the per-map new-tile cap (`map_new_tile_cap`, route-agnostic, already built), D1 s1–3, ≈ $4.4. That changes G2's question from "patch the top hack" to "cap the dominant term". It must be logged as a deviation in `patch-log.md` before launch.
- **(C) Fix the capability confound first.** Add a visited-tile channel to the observation for all arms, then re-run A0 + A2″, as batch 2: ≈ $30, a new pre-registration and Jake's cost approval.
- **My recommendation:** run G5 first (≈ free), then choose (C) if the planner P ≥ 2× A0's, and (A) otherwise. Option (B) gives an uninterpretable result if A0 can't optimize the patched proxy either. A1 round 1 is **on hold**.
- **Held-out stays sealed** until whichever path is chosen is logged in `patch-log.md`. That way a later patch (B), or a batch-2 design (C), can't be shaped by HA/HB.

**Batch 2: visited-tile observation (pre-registered 2026-11-07, researcher; Jake chose option (C) via the lead).** I wrote this before any batch-2 code or run exists. The batch-1 held-out data are still sealed and haven't been seen.

*H1. Question.* Batch 1's null was most likely a capability limit (§9 G4–G5). The policy couldn't see which squares still pay, and it closed only about 6% of the gap between its init P (≈ 31) and the attainable P (≥ 417). Batch 2 asks the batch-1 questions again under a learner that can see that: H0, the catalogue and attribution, then G2. **The only change is one observation channel.** Everything not listed below is exactly as in batch 1 (§9 A–F, A′, B′, and the §6 detectors).

*H2. Observation change (the only intervention).*
- **New channel `visited`, in `tilegrid` mode and in the same tensor as the tiles and sprites:** a 20 × 18 binary grid aligned with the tile grid.
  - A cell is 1 if the game square under it, (map, x, y), is in **the same set that has already paid explore this episode** (`Proxy.coords`, i.e. stood on). Otherwise it's 0.
  - Game squares are 2 × 2 tiles, so each square fills a 2 × 2 block of cells.
  - Squares off-screen aren't shown. The channel is reset at episode start and is per map: after a warp it shows the new map's squares.
- **It carries no T, no milestone and no walking-distance information (§3a).** It only shows the proxy's own state, which the policy could in principle have tracked itself. It is the PokeRL-style visit overlay. *Unverified* that their encoding is identical. I'm citing the idea, not their implementation.
- **All arms get it**, A2″ included, so A0 and A2″ stay comparable. **R0 is unchanged:** it ignores observations, so batch-1 R0×24 (dev and held-out) is reused as is, with the same reward hashes and env.
- **The reward functions don't change.** P0 stays `d28e2a337dda` and A2″ stays `1f739117e21a`. Each run records an `obs_version` in its config and its rows.
- **The network changes only as forced:** the first conv layer gets one more input channel. HPARAMS are frozen as in batch 1, with original LR 2.5e-4. Inits differ from batch 1 even at equal seeds, because the parameter shapes changed.

*H3. Arms, seeds, M, marks.*
- **23 runs, as in batch 1:** A0 on D1 and D2, 5 seeds each; A2″ on D1 and D2, 5 seeds each, plus HA seeds 1–3. Seeds 1–5, with A0 and A2″ paired by seed × training state on one VM. The VM split is the same as batch 1.
- **M = 5M, fixed.** The M rule (§9 "A2″ 10M extension") isn't re-derived. That's for comparability with batch 1 and for cost.
  - **Conditional extension, decided now and flagged for Jake's cost OK:** A0 D1 seeds 1–3 are extended to 10M only if A0's pooled D1 P at 5M is greater than at 4M (one-sided MC permutation, p ≤ 0.05). That would mean the learner is still climbing. It costs ≈ +$4.4, and it never runs without the lead/Jake's go.
- **Eval marks: 0 (saved `ckpt-0.pt`), 0.25, 0.5, 1, 2, 3, 4, 5M.**
  - The 0 mark is descriptive (Fig. 1, the P-rise check). The H0 tests use the batch-1 marks unchanged (H5).
  - **Eval states: D1, D2, HA, HB from the start**, with KL from `ckpt-0` (exact, now that it's saved) in the same pass, on the training VM.

*H4. Gate and stop rules (cheap insurance against a broken obs change).*
- **Smoke tests (developer, local, before launch):**
  - (a) Channel alignment: after each step, the cell under the player is 1, every cell is 1 exactly when its square ∈ `Proxy.coords`, it resets per episode, and warps are handled.
  - (b) An identity check that with the channel zeroed, the obs equals batch 1's.
  - (c) Reward hashes unchanged.
  - (d) A ≥ 50k-step training smoke without errors.
- **First pair as the gate:** each VM's queue starts with an A2″/A0 pair, A2″ first.
  - **Gate:** A2″'s first D1 run passes the §9 capability gate (M2 in ≥ 1 D1 eval episode at any mark).
  - **Fail → stop both queues** and report. The obs change broke the oracle, and nothing after it is interpretable.
  - The gate is judged on A2″ only, never on A0. It can't be tuned toward a hack result.

*H5. Tests, detectors, attribution: unchanged from batch 1, applied to batch-2 data from scratch.*
- **H0 (i)–(iii)** as in §9 D with B′ (d)/(e): (i) is still 0.5M vs 5M, the marks ≥ 2M are the same, and the verdict rule is the same.
  - I keep (i) unchanged even though batch 1 showed the rise happens before 0.25M. Changing the test after seeing that would fit it to batch 1. P(0) → P(5M) is reported as descriptive.
- **Manipulation check, pre-registered: did the obs change make A0 a stronger optimizer?** It passes if A0's pooled P at its best mark is ≥ 2× batch 1's A0 best mark, on **both** dev states (D1 ≥ 111, D2 ≥ 101).
  - **Fails →** batch 2 is reported as "capability still limiting". H0 and G2 conclusions are not drawn from it, and no batch 3 runs without a new decision.
  - This is the same yardstick as G5.
- **Outcome table (written now, because the max-P planner also reaches T = 5, §9 G5 result):**

  | outcome | condition | reading |
  |---|---|---|
  | O1 Goodhart | manipulation passes; H0 supported (both states) | the proxy is hacked once the learner is capable |
  | O2 aligned | manipulation passes; falsifier met | P0 is aligned at what a capable learner reaches; batch 1's misranking was a weak-learner effect |
  | O3 mixed | manipulation passes; neither | report per state; no claim |
  | O4 still weak | manipulation fails | capability-limited again; no H0/G2 claim |

- **Detectors:** the same §6 set and parameters (L, HF, G, B, M, D, O; K and U never counted). U is recomputed on dev windows only (A′). The U review runs against R0's windows (§9 E), and any new mechanism becomes a detector before round 1.
  - L is especially a canary now: the channel must not create revisit pay. L ≠ 0 means a bug, investigated before the run counts.
- **Attribution:** the 14-cell primary Holm families per detector × arm. Collapse is judged on the run's own state and dropped in every cell (B′ (b)). The labels are the same, including "undetermined (reference collapsed)". A2″-HA runs follow B′ (g).
- **Descriptive only, new; it never makes a type present:**
  - **post-progress share of P:** the fraction of an episode's P earned after its last new milestone;
  - **coverage per map:** distinct squares ÷ the planner's distinct squares on that map, using G5's planner as the denominator.
  - Both are leading candidates for a "coverage without progress" detector. That detector, if any, is specified after the U review and before round 1, under §9 E.

*H6. What batch 2 supersedes.*
- **Batch 2 is the round-0 input to G2:** the §9 F patch plan runs on batch-2 A0 dev traces, and A1 uses the visited channel. Batch 1 is never the patch input.
- **Batch 1's results are kept, not replaced.** They become the "no visited channel" (weak-learner) condition: F-E1.1–F-E1.4 and §9 G stay as written, with that label. H0's headline verdict comes from batch 2. Batch 1 is reported next to it as the capability contrast.
- **Held-out:** both sealed sets (`sealed-b1/`, and the new `sealed-b2/` with its own manifest) **stay sealed** until batch 2's round-1 decision is logged in `patch-log.md` with its hash and time on BOARD. That covers a patch, a G2 stop, or O4. Batch 1's held-out data could otherwise shape batch 2's patch.
- **Pipeline:** `analyze_b1.py` is reused with a batch tag. There's no separate dry run, because the analysis code is unchanged and was validated on batch 1. The checklist gains these lines: `obs_version` in every row; the 0 mark present; KL from the saved `ckpt-0` (bit-exact); the manipulation check; the descriptive metrics.

*H4′. Staged launch, validity tripwires and per-run reads (Jake's ask via the lead, 2026-11-07; added before anything is armed). This replaces H4's "first pair as the gate".* The aim is to find a design problem within hours, not after a batch. Batch 1's warning signs were already in the A0 pilot (`findings.md`, "What the A0 pilot already showed").
- **Stage 1: a pilot pair before fan-out.**
  - **A0-D1-s0** on e1-a and **A2″-D1-s0** on e1-b run **concurrently**, each to **2.5M, then stop**. Seed 0, as with batch-1 pilots, so they're **not main runs** and never enter any test.
  - Interim evals: D1 × 8 at **0 (ckpt-0), 0.25, 0.5, 1, 2 and 2.5M**.
  - **Reads at 1M and 2.5M.** The 23 main runs launch only if every pilot tripwire is clear at 2.5M.
- **Stage 2: main runs, with the same tripwires on each VM's first A0 run per training state.** That's 4 runs: on e1-a, A0-D1-s1 and A0-D2-s1; on e1-b, A0-D1-s4 and A0-D2-s4.
  - Their 1M checkpoint, and a **2.5M checkpoint added for these 4 runs only** (descriptive, never in a test), are evaluated **as soon as saved**: own state × 8, dev only, in their dev view.
  - The other queue keeps running during these reads unless a tripwire is hit.
- **Tripwires (fixed now).** All of them are about whether each arm optimizes **its own** objective, or about infrastructure. **None uses A0's T, A0 vs A2″, any detector share, or anything held-out.**
  - Thresholds use batch-1 R0 dev P (D1 29.7, D2 31.2), the G5 planner floor (417), batch-1 A0's best mark (D1 55.5, D2 50.5) and the run's own step-0 P(0).
  - **Progress fraction:** g = (P − P(0)) / (417 − P(0)). Batch-1 A0's best was g ≈ 6% on D1 and ≈ 5% on D2.

  | id | arm | read | trips if | reason |
  |---|---|---|---|---|
  | A-1 | A0 | 1M | own-state P < 2 × R0 (D1 59.4, D2 62.3), **or** P(1M) not > P(0) (one-sided MC permutation, p > 0.05) | not beating random clearly / not learning its reward |
  | A-2 | A0 | 2.5M | own-state P < 0.75 × the manipulation target (**D1 83.3, D2 75.8**; g ≈ 13% / 12%), **or** P(2.5M) < P(1M) with one-sided p ≤ 0.05 | not on track for ≥ 2× batch 1 (H5) by 5M / falling |
  | B-1 | A2″ | 1M | leg progress(1M) not > leg progress(0) (one-sided, p > 0.05) | not climbing its own shaped objective (the d_min shaping term ≈ leg progress) |
  | B-2 | A2″ | 2.5M | no own-state eval episode reaches M1 (T ≥ 1 in 0/8), **or** leg progress(2.5M) < leg progress(1M) with p ≤ 0.05 | off schedule for the §9 gate (M2 by 5M); for reference, batch-1 A2″ D1 had T 0.65 at 2M |
  | C | any | every read | L > 0; `obs_version` or reward hash mismatch; KL init not bit-exact against `ckpt-0`; sps < 85% of batch 1's measured rate | canary / infrastructure / cost |

- **Response to a trip:** **both queues pause**, and the researcher and lead review it.
  - The decision is logged in §9 **before** any resume, and it is one of three:
    - (r1) **false alarm:** documented, then resume unchanged;
    - (r2) **design fix:** applied to **all arms**, every batch-2 run restarts from scratch, the paused runs' data are never pooled, and a cost re-check goes to Jake if needed;
    - (r3) **stop:** report as O4 ("capability still limiting") or as an infrastructure failure.
  - **Never allowed:** dropping or replacing one seed, moving a threshold, or changing any H5 test.
  - A2″ trips are expected to be noisier, because A2″ oscillates (§9 "Step 0 applied"). For B-2, the review looks at whether the 2.5M checkpoint sits in a training dip before calling it a fix-or-stop.
- **Per-run science read (researcher, on BOARD).** I write it after each pilot read and after **every** finished run, from dev states only, as **3 lines**:
  1. P by term (explore / level / heal) at the best mark and at 5M, with T and leg progress per mark, on the run's own state;
  2. P against R0 (×) and against the planner (g), plus the L canary count and, for A2″, the collapse flags;
  3. a one-line early read, labelled **"interim, validity only, not a result"**.
  - No Holm, no R0 permutation test, no A0-vs-A2″ comparison, no detector shares, and nothing held-out appears in these reads. The [developer] provides a per-run summary script; I write the note.
- **Why these interim looks don't bias the hypothesis tests:**
  1. **They gate validity, never outcomes.** Every tripwire asks whether an arm optimizes its own objective (A0's P, A2″'s shaping and gate) or whether the infrastructure is sound. None reads a quantity H0, the catalogue or attribution is decided on: A0's T, A0 vs A2″, detector shares, or held-out data.
  2. **The response can't select data.** A trip pauses, fixes and restarts **all** arms, or stops the batch. No seed, checkpoint or run is ever dropped or kept because of an interim read, so the final sample is the pre-registered 23 runs either way.
  3. **The tests are computed once**, on all 23 runs, with H5 unchanged. No interim read computes an H0 statistic or a Holm test.
  4. **Remaining risk, stated plainly.** The per-run reads do show me A0's T before the end. Every analysis choice is fixed above. Any amendment made after the first read is labelled **"post-interim"**, with its timestamp, and can't touch H0, the detectors, attribution or Holm. Also, A-2 makes batch 2 conditional on A0 being a capable optimizer. That is intended, because it's the manipulation check brought forward, but it means batch-2 H0 results are conditional on that capability, and the paper says so.

*H7. Cost (estimate; the [developer] re-costs from batch-1 measured rates).*
- **Stage-1 pilots:** 2 runs to 2.5M ≈ 1 run-equivalent ≈ **+$1.3**. In-run interim evals ≈ cents. The staged wait costs time (≈ 2.5–3 h), not money. **The base becomes ≈ $31–32, still under $35.**
- **Base:** ≈ $30 (batch 1 measured ≈ $30, including the later D2 + KL pass, now folded into the first eval pass). The extra eval mark adds about 1/7 of the eval share. The channel may lower sps slightly; that's to be measured in the smoke run.
- **Items that can push past ~$35, each needing Jake's OK before it runs:**
  - (1) the conditional A0 10M extension (≈ +$4.4);
  - (2) A1 round 1 after batch 2 (≈ +$4.4, §9 F, not inside the $30);
  - (3) a measured sps drop of more than ~15%, which scales the base;
  - (4) a gate failure followed by any rerun.

*H8. Build review and rulings (researcher, 2026-11-07, before arming; not post-interim).*
- **Smoke (a)–(d): accepted.** `code/results/batch2-smoke.json` is all PASS. The channel matches the map on 9,167 of 9,167 settled steps; 0 / 16,384 content errors; reset OK; batch-1 obs identical; both hashes match; first conv takes 10 inputs.
- **Screen lags coords by one step during ledge hops and scripted movement: accepted as is.** H2 defines the channel ≡ `Proxy.coords`, so following the coords is the spec, and it matches what the reward pays on. The mismatch is transient (1 step), is the same in every arm, and so can't create an arm difference. Geometry was not checked on those 1,270 steps (7.8%: 108 mid-hop, 1,162 scripted), but content and under-player checks covered all 16,384. Logged as a known property. No change.
- **Run order A0 then A2″ per seed × state: approved.** H4's A2″-first order belonged to the replaced first-pair gate. With A0 first, each VM's first run per state is the A0 run that H4′ Stage 2 tripwires (e1-a s1 D1/D2, e1-b s4 D1/D2), which matches `main-b2-e1{a,b}.txt` as built. The 3 A2″-HA runs stay last on e1-b. Order has no role in any test.
- **Cost: expected $31.88 (sps 0.98), pessimistic $34.21 (sps 0.894); both under $35.** The VM ceiling of **$41.44** is a hard stop, not a budget. Spend between $35 and $41.44 can occur only through a sps drop > 15% (which trips C at < 85% and pauses) or overrun, so H7 item (3) already covers it: **any projected spend above $35 needs Jake's OK before continuing.** Flagged to the lead.
- **Items 6 (`sealed-b2/`) and 7 (batch tag): required before any analysis, not before launch.** Held-out evals must write into `sealed-b2/` from the first closing eval; if item 6 isn't built when the first main run closes, its eval output goes to a sealed path anyway and is not read.
- **Applied (developer, 2026-11-07):** batch-2 held-out closing evals (HA, HB) write to `~/roms/sealed-b2/<run>/eval-heldout/`; dev evals (D1, D2) to `~/roms/runs/<run>/eval/`. For batch-2 runs, `eval/` is a dev view and I may read it; batch-1 `eval/` stays closed. `sealed-b2/` stays unopened until the round-1 decision is logged.

*H9. Trip log.*
- **Trip 1: C (sps) at e1-a's ckpt-0 read, 21:34:45 UTC 2026-11-07. Ruling: r1, false alarm (researcher, 21:39 UTC, before the 1M reads; logged before any resume).**
  - **What tripped:** the read measured sps 35.1 vs threshold 0.85 × 411. The cause was the interim eval running beside training and starving the trainer's threads (e1-a 375 → ≈ 30 sps; e1-b ≈ 310 → ≈ 90 sps). The measurement apparatus caused it, not the run: C is meant to catch a slow env/policy, and this was contention created by the read itself.
  - **Why r1 and not r2:** the fix (code `dbd7fa34`) changes only how reads run. A read now SIGSTOPs the run's trainer and env workers, evaluates on all CPUs, then SIGCONTs, and C's sps check skips updates that overlap a freeze. Rewards, obs, hparams, seeds and the training step sequence are unchanged. PPO here is synchronous and step-indexed, with no wall-clock logic in training, so a slow or frozen interval doesn't change what the run learns. The pilots ran through it and keep going. No run, seed or checkpoint is dropped, and no threshold moves. The fix applies to every run's reads from here on.
  - **Re-check:** e1-a C passes on clean updates, sps ratio **0.896** (368 / 411). **e1-b's C re-check must also pass** (≥ 20 clean updates) before its pause flag clears; the developer reports it at the 1M read.
  - **Developer to confirm in the log:** no time-based logic in the trainer or env (checkpoints, timeouts, LR schedule are all step-indexed). If anything is wall-clock based, this ruling reopens as r2.
  - **Cost follow-on:** 0.896 is at the pessimistic case (**$34.21**, under $35 but within $0.8). C's margin is about 5 points. If later reads put sps below ≈ 0.88, or the projection above $35, it goes to Jake before fan-out continues (H7 item 3).
  - **Cleared (developer, 2026-11-07):** e1-b's C re-check passed on ≥ 20 clean updates (ratio 0.916), e1-a now 0.944, and both queues are unpaused. The frozen-read mechanism works (e1-a 0.25M read: trainer frozen 102 s, then ≈ 390 sps). Both ratios are above the 0.894 pessimistic case, so the projection stays under $34.21. **Closed (developer's itemized confirmation on BOARD):** LR is constant Adam 2.5e-4 and all other coefficients are constant; eval and interim checkpoints are step-indexed; there are no timeouts in the loop or env (only a shutdown join); the emulator is frame-ticked with no RTC. One wall-clock item exists: the 15-min recovery checkpoint. It copies state and reads the RNG with non-advancing getters, so it can't change the computation. **r1 stands.** Caveat: if any run ever *resumes* from a recovery checkpoint, the resume point depends on wall time. Log each resume in this trip log with the step it resumed at, and whether the restore includes the emulator, env and RNG state. A partial restore is a deviation to rule on.
- **Pre-ruling: resume with partial restore (researcher, 2026-11-07, before any batch-2 resume; not post-interim).** Every resume restores the learner exactly (weights, optimizer, counters, all RNG states). It does not restore the emulator or env: in-progress episodes restart from the start state, and env workers are reseeded (+7919 per resume, deterministic and logged).
  - **Why this is acceptable:** the update rule, reward, obs and hparams are unchanged. The env change acts like a fresh rollout seed for the rest of the run, which is a random perturbation, not a selection. Evals are unaffected because they run from saved checkpoints on fixed states. The one systematic effect is small: a forced episode reset gives the agent a fresh visited map, so a few more "new" tiles in training only. It is the same mechanism in both arms.
  - **Rule: a resumed run stays in the sample, unchanged, if all of these hold.** (i) The resume was triggered by infrastructure (preemption, VM or process fault), never chosen after an interim read. (ii) The learner restore checks exact (`resume_log` says so). (iii) The run reaches M by the learner's step counter. (iv) There are at most 2 resumes per run, and replayed or lost steps (stop step − resume step) total ≤ 2% of M (100k).
  - **If (iv) fails, or (i)–(iii) are in doubt,** the run comes to me for a ruling before its closing eval is read. Possible outcomes are keep, or rerun from scratch with the same seed. A rerun needs Jake's OK if it costs money, and dropping the run is never allowed.
  - **Reporting:** all runs enter the tests as pre-registered. Resumed runs are listed in the batch-2 report (run, count, steps replayed). There is no with/without analysis, since that would be a data-dependent split.
  - **Never** resume deliberately to get around a tripwire or a pause.
  - **Implemented (developer):** `b2_reads.py` classifies each run as keep / pending / escalate, with reasons, in every summary and health line; `b2_reads.py resumes` lists resumed runs. I checked the code: `RESUME_MAX, REPLAY_MAX = 2, 100_000`; the infrastructure trigger is a shutdown or preemption signal or a GCP VM event; keep requires reaching M. Matches the rule.

- **Stage 1 result (2026-11-08 00:01 UTC): all tripwires clear on both pilots.** A0-D1-s0: A-1, A-2 and C pass (P 92.2 → 103.3; T 2.0 → 1.0, logged, not a tripwire quantity). A2″-D1-s0: B-1, B-2 and C pass (leg progress 1.74 → 2.94, T 2.0 at 2.5M). No resumes; one trip (trip 1, r1). Fan-out per H4′ Stage 2 is cleared from the researcher's side.

- **Trip 2: A-1 on b2-A0-D1-s4 at 1M, 01:11 UTC 2026-11-08. Ruling DEFERRED until the measurement is diagnosed (researcher, with the lead). Queues stay paused; the running runs finish; nothing is dropped and no threshold moves.**
  - **What tripped:** eval P(1M) was 47.3 against a 59.4 minimum (1.59× R0, g 0.041). The other half of A-1 passed (P(1M) > P(0) = 31.5, p 0.0018), and C passed.
  - **Why the ruling waits:** training-episode P since 0.5M is 117.5 against eval 47.3. A 2–3× train/eval gap on the reward being optimized also appears in batch 1 (A0-D1-s4: 96.5 vs 55.8 at 1M, 89.6 vs 31.5 at 2M) and in other runs. `evaluate.py` samples actions, so greedy vs sampled doesn't explain it. If eval doesn't measure the policy that was trained, every eval-based result is affected, batch 1 included (H0, F-E1.1–F-E1.4, G5's "≈ 6% of the gap", and the tripwires). Ruling r1/r2/r3 on a possibly broken measurement would be premature.
  - **Candidate explanations, to be separated by the diagnostic:**
    - (a) *Eval bug:* env config differs (jitter, start state, frames per action, episode length, reward/obs version), obs differs (visited-channel reset, normalization stats, stacking), or the checkpoint loads wrongly (non-strict load, missing buffers, wrong file).
    - (b) *Training P isn't a snapshot measure:* a 16,384-step training episode spans ≈ 128 PPO updates, so it is earned by a policy that adapts during the episode. A frozen checkpoint loses that within-episode adaptation, and it can do worse even with no bug.
    - (c) *Start-state difference:* training rollouts start from jittered states across 8 envs, while eval starts from fixed D1. Different starts could reach different numbers of new tiles.
    - (d) *Counting difference:* training P is summed over a different span than eval P (episode boundaries, truncation, or which episodes are logged "since 0.5M").
  - **Diagnostic (developer, local or on an idle VM; free or cents). All dev-state only; held-out stays sealed.**
    - D-1 *Same snapshot, two paths:* load b2-A0-D1-s4's 1M checkpoint and run ≥ 16 episodes through the **training rollout path** (same env factory, wrappers, jitter, sampling, with no updates), then ≥ 16 through `evaluate.py`. Report P, P by term and episode length for each. If the training path is frozen and ≈ eval, (a) is ruled out. If it is ≫ eval, there's a bug in eval.
    - D-2 *Bit check:* with the same seed and start state, assert that the first 256 observations and the action logits are identical between the two paths. Diff the full env and obs config dicts, and check the checkpoint loads strictly with no missing or unexpected keys.
    - D-3 *Step-0 anchor:* the training P of the first episodes (policy ≈ ckpt-0) vs eval P(0). If they match at 0 and diverge later, the gap grows with learning, which fits (b) or a load bug that only shows on trained weights.
    - D-4 *Counting:* define training P exactly as logged (span, episode boundary, which envs and episodes), and recompute it from the training rows.
    - D-5 *If D-1 clears eval:* run the frozen 1M checkpoint from the training jittered starts vs fixed D1 to size (c). Whatever remains is (b).
  - **What follows:** if (a), eval is fixed, all affected evals are re-run on every arm from saved checkpoints, and batch-1 findings are re-issued; trip 2 is then re-read on the fixed measurement. If (b)–(d), eval stands as the pre-registered measure of the trained policy (the snapshot is what H0 is about), A-1's threshold stays, and I rule trip 2 under H4′ with the diagnosis in hand. Either way the ruling is logged here before any resume.
  - **Part 1 results (developer, 2026-11-08):**
    - **D-2:** the training and eval paths are bit-identical (obs 256/256, logits max diff 0.0). The checkpoint loads strictly (0 missing or unexpected keys), and the env kwargs differ only in logging fields.
    - **D-4:** training P is counted correctly, over full 16,384-step episodes spanning ≈ 128 PPO updates each.
    - **D-3:** the first training episodes give P 56.8 against eval P(0) 31.5. They already include ≈ 128 updates, so this isn't a clean step-0 match.
    - **My reading so far:** (a) is largely ruled out at the code level, since the obs, logits, load and config all match. (d) is ruled out. D-3 fits (b): about +25 P within the first training episode, which a frozen checkpoint can't get. Still pending: D-1 (the training path frozen vs eval at the same checkpoint, which covers sampling and jitter end to end), D-5 (start states, c), and the developer's direct (b) test (PPO resumed from the 1M checkpoint for one episode length). The ruling waits for them.
    - **Batch-1 context (researcher, dev views only: `eval-dev/` own-state traces, 8 episodes per mark, descriptive):** per-seed eval P swings a lot from mark to mark. For example, A0-D1-s4 reads 41.9 / 75.0 / 55.8 / 31.5 / 58.7 / 11.9 / 70.7 at 0.25 / 0.5 / 1 / 2 / 3 / 4 / 5M, and A0-D2-s1 reads 12.4 / 44.7 / 23.7 / 82.6 / 53.2 / 53.4 / 17.5. The s4 1M value (55.8) matches the developer's figure. So a single checkpoint's eval P is a noisy, oscillating sample of the policy, while training P averages over ≈ 128 updates. That fits (b), and it means a one-seed, one-mark tripwire like A-1 has high variance by construction. This is description, not a basis for moving the threshold: A-1 stays as written.

- **Trip 3: A-2 on b2-A0-D1-s1 at 2.5M, 01:57 UTC 2026-11-08. Ruled together with trip 2, after the H9 diagnostic; the queues were already paused and stay paused.**
  - **What tripped:** both clauses of A-2. P(2.5M) was 47.6 against an 83.3 minimum, and it fell from P(1M) = 72.3 (one-sided p 0.0127). A-1 still passes and C passes (sps 1.034×). Own D1, 8 episodes per mark: P 33.6 → 72.3 → 47.6 (1.6× R0, g 0.037); T 0.5 → 0.125; leg progress 1.26 → 1.00.
  - **Where the three A0-D1 reads stand:** the pilot (s0) passed A-1 and A-2, s4 tripped A-1 at 1M, and s1 passed A-1 then tripped A-2 at 2.5M.
  - **How the ruling turns on the diagnostic (stated now, before D-1, D-5 and the (b) test return):**
    - If the diagnostic finds an **eval fault**, trips 2–3 are re-read on the fixed measure.
    - If it confirms **(b)/(c)**, eval stands as the pre-registered measure. H0 and the tripwires are about the policy as a snapshot, and A-2 asks whether that snapshot optimizes P reliably. Then **two of three A0 runs failing is the tripwire doing its job, not a false alarm, so r1 is not available.** The choice is r2 (a fix applied to all arms, then a restart, which is a design change that needs Jake's OK and a new §9 entry) or r3 (stop batch 2).
    - The ruling is logged here before any resume.

- **D-1 result and s4's 2.5M read (developer, 2026-11-08):**
  - **D-1:** the frozen 1M checkpoint of s4, run through the training rollout path (16 episodes, full length, no updates), gives P 47.6 with T 0.0. Eval gave 47.3. With D-2 and D-4, **(a) an eval bug is ruled out**: eval measures the trained snapshot correctly.
  - **s4's 2.5M read:** A-2 passes. P is 117.1 (3.94× R0, g 0.222), up from 47.3; T 1.875; leg progress 2.60; C passes.
  - **A0-D1 snapshot P so far:**

    | run | 1M | 2.5M |
    |---|---|---|
    | s0 (pilot) | 92.2 | 103.3 |
    | s1 | 72.3 | **47.6** |
    | s4 | **47.3** | 117.1 |
    | mean of 3 | 70.6 (≥ 59.4) | 89.3 (≥ 83.3) |

- **Proposed ruling for trips 2 and 3: r1, false alarm (researcher, 2026-11-08). This reverses my heads-up under trip 3, so it goes to the lead and Jake before any resume.**
  - **Why the heads-up no longer holds:** it assumed two of three A0 runs were failing as snapshots. At 2.5M only one of three fails, and the run that failed at 1M is now the best of the three (117.1).
  - **What A-1 and A-2 are for:** they test whether A0 *optimizes its own reward* with the new channel. The evidence says it does: the mean over runs clears both thresholds, every run rises above P(0), training-episode P is ≈ 117, and g at 2.5M (0.04–0.22) is well beyond batch 1's ≈ 6%.
  - **What the trips detected instead:** the eval of one snapshot on 8 episodes swings a lot between checkpoints (s1 72 → 48, s4 47 → 117). Batch 1 showed the same (descriptive table above). That is a property of the learner, and the pre-registered analysis already absorbs it: pooled seeds, several marks, permutation tests.
  - **What doesn't change:** no threshold moves, no run, seed or checkpoint is dropped, and trips 2–3 stay in the log.
  - **Labelled post-interim:** the 3-run mean is a review criterion, not a new tripwire. I state it now so later trips are ruled the same way, not ad hoc.
    - A single-run A-1 or A-2 trip is ruled **r1** when C passes and the mean of all A0 runs on that state read so far at that mark meets the threshold.
    - Otherwise the trip goes to r2/r3 review.
    - A run read first on its state (no mean yet) is reviewed on its own.
    - None of this touches H0, the detectors, attribution or Holm.
  - **Pending before resume:** D-5 and the direct (b) test, for the mechanism only. A surprise in either (for example, PPO resumed from 1M failing to recover P) reopens this ruling.
  - **Approved (lead and Jake, 2026-11-08): r1 for trips 2–3, conditional on D-5 and the direct (b) test coming back clean.** This is a **post-interim judgment made after seeing interim data, not a pre-registered outcome.** The paper reports it that way, including the per-run trips and the 3-run-mean criterion adopted after them. Queues stay paused until I confirm the condition to the lead and the developer here. If either test is not clean, the ruling reopens.
  - **Diagnostic part 2 (developer; frozen 1M checkpoint of s4):**

    | path | episodes | P |
    |---|---|---|
    | training path (D-1) | 16 | 47.6 |
    | `evaluate.py` | 16 | 54.9 |
    | training jittered starts (D-5) | 8 | 49.0 |
    | logged training episodes (reference) | — | 117.5 |

    - **D-5 is clean:** the jittered starts (49.0) sit within the spread of fixed D1 (47.6 / 54.9 / 47.3), so (c) is ruled out.
    - The eval-path spread (47.3 at 8 episodes, 54.9 at 16) is ordinary sampling noise.
    - (a), (c) and (d) are all ruled out, leaving (b), within-episode policy change, by elimination. The direct (b) test (PPO resumed from that checkpoint, first episodes, ≈ 04:30 UTC) is the remaining condition: P in those episodes should rise well above the frozen ≈ 50, toward the logged ≈ 117. If it stays near the frozen value, the 117.5 is unexplained and the ruling reopens.
  - **Condition met; r1 for trips 2–3 STANDS (researcher, 2026-11-08; post-interim judgment).** The direct (b) test resumed PPO from s4's exact 1M checkpoint, with the optimizer and RNG restored. Its first 8 episodes, played while the policy learns, give P 131.4 (range 114–145) and T 2.0. The same checkpoint frozen gives 47.6 / 54.9 / 49.0, and the ranges don't overlap. So (b) is confirmed: the train/eval gap is within-episode learning, and eval correctly measures the snapshot. Both conditions are clean, so the queues may resume under r1.
  - **What this means scientifically (inference):** A0 earns most of its training P by adapting *during* an episode, at ≈ 128 updates per episode, not by having a policy that reaches that P when frozen. H0, the detectors and the tripwires are all about the snapshot, as pre-registered, and that stays the primary measure. The batch-2 report adds training-episode P per run as a descriptive second line, so the reader sees both what the learning process reaches and what the learned policy reaches. Whether proxy over-optimization looks different in the two is an open question (`findings.md`).
  - **For the paper:** A0's snapshot P is unstable under this learner, so single-mark, single-seed P is not a reliable read. The batch-2 report will plot per-seed curves alongside the pooled ones.
- **Stage 2 result (2026-11-08 ≈ 14:00 UTC): all A-type tripwire reads are done.** The four tripwired first-A0 runs were read at 1M and 2.5M (8 reads), and 2 tripped: trip 2 (D1-s4, A-1 at 1M) and trip 3 (D1-s1, A-2 at 2.5M). Both were ruled r1 under the conditional, post-interim ruling above. The other 6 reads passed; D2-s1 and D2-s4 passed at both marks (P 92.3 → 96.2 and 77.8 → 116.1). From here only C-type checks remain (L, sps, failures, resumes). No threshold was moved and no run was dropped.
- **Trip 4: C (sps) at the closing read of b2-A2pp-D2-s1 on e1-a, 2026-11-08. Ruled r1 for validity; the cost part is escalated (researcher).**
  - **What tripped:** the median sps over the last 100 updates was 303, 0.80× batch 1, under C's 0.85 line. (Over the whole run it was 1.05×.) H4′ applies C at every read, and the closing summary is a read, so this counts as a trip, although the summary printed "no tripwires on this run".
  - **Cause:** host-side. e1-a slowed from ≈ 428 to ≈ 300 sps from ≈ 4.1M (≈ 15:00 UTC). Both t_collect and t_learn rose ≈ 35%, there was no CPU steal or log warning, e1-b is unaffected, and batch 1's same run held 377–393.
  - **Why r1:** training is step-indexed (H9), so the numerics are unaffected, and nothing in the run's state changes.
  - **Cost part (H7 item 3, a > 15% drop):** e1-a still has 8 entries. The developer re-projects total spend at 0.80×. **If the projection is above $35, Jake's OK is needed before e1-a's next entry starts.**
  - **Suggested remedy (lead's call):** stop/start e1-a *between* entries, after the running held-out eval finishes, so it lands on a new host. With no run in progress, that is not a resume under H9.
  - **Fix requested:** evaluate C at closing reads too (and print it), so a C failure there pauses as H4′ specifies.
  - **Closed (developer, 2026-11-08):** the 0.80× re-projection was ≈ $35.3 (> $35), so e1-a was paused at 16:45 before its next entry. With Jake's OK via the lead, it was stopped and started between entries (the held-out eval was not interrupted, so no H9 resume) and came back up 17:08 UTC on a new host at 0.96–1.04×. The projection is back to ≈ $30.7, and closing reads now evaluate and print C.
- **Note on the H3 conditional 10M extension (researcher, 2026-11-08; a clarification, no rule change).**
  - **When it's decided:** H3 decides the extension on A0's *pooled* D1 P at 5M vs 4M, which needs all five A0-D1 seeds. s3 (e1-a) and s5 (e1-b) haven't run yet, so it is decided only after both finish, and it then still needs the lead's and Jake's go. Single-run reads, such as s2 still rising at 5M, don't decide it.
  - **How it relates to the H9 resume rule:** an extension resumes a finished run from its 5M checkpoint. Under H9 that is a partial restore that wasn't triggered by infrastructure, so it would fail (i). The extension is pre-registered in H3, though, so it is treated as a **planned continuation**: it's exempt from H9 (i) and logged in `resume_log` as "planned extension". All pre-registered tests use marks ≤ 5M, which come before the resume, so they're unaffected. The 5M–10M segment is reported separately.

*H10. Pre-specified secondary series: the learning process (researcher, 2026-11-08, before any batch-2 main-run data past the Stage-2 interim reads; post-interim, since I have seen the interim reads in H9).*
- **Why:** H9 showed that A0 earns most of its training P by learning *within* an episode (≈ 128 PPO updates per 16,384-step episode). A frozen checkpoint doesn't reach that P. So proxy over-optimization could live in the learning dynamics rather than in any snapshot, or the reverse. Either result is a finding, so the series is fixed now.
- **Primary is unchanged.** H0, the manipulation check, the detectors, attribution and Holm stay on snapshot evals exactly as in H5. Nothing below enters a test family, a tripwire or the round-1 patch decision.
- **The series (every batch-2 run, A0 and A2″; also computed for batch 1 from its `episodes.jsonl`, own state only, so no held-out data is touched):**
  - **Unit:** completed training episodes (full 16,384 steps per env).
  - **Bins:** the episodes whose end step falls in (m − 250k, m] for each mark m in {0.25, 0.5, 1, 2, 3, 4, 5}M. That's ≈ 16 episodes per bin per run (8 envs × ≈ 2 completions).
  - **Per bin:** P and P by term, T, leg progress, episode count.
  - **Gap series:** training-bin minus snapshot eval at the same mark, for both P and T.
  - **Baseline:** R0 eval, since R0 doesn't train. Planner floor 417 and g as in H4′.
  - **Detector types:** only if per-step training traces exist; the developer confirms which. If they don't, the series stays at summary level and that is stated, with no detector shares imputed.
- **Pre-specified descriptive readouts** (seed-bootstrap 95% CIs, pooled seeds per arm × state, no p-values, no Holm):
  - **S1, H0 mirror:** criteria (i)–(iii) of H5 computed on the training series, with the same marks and definitions.
  - **S2, divergence pattern:** call it **"learning-process divergence"** if, for A0 on a state, the training series meets (i) and (ii) while the snapshot series does not. Call it **"snapshot-only divergence"** for the reverse. Otherwise the two **agree**. Report it per state and say which one, without merging them into H0.
  - **S3, arm contrast:** A0's minus A2″'s training-series T at 5M, and the same for snapshots, side by side.
  - **S4, gap size:** training/snapshot P ratio per mark per arm. A ratio that grows over training means more of the proxy is earned online.
- **Interpretation limits:**
  - Training-episode P is a property of *policy + optimizer + 16,384-step episodes*, not of a policy. A learning-process divergence would be a claim about that loop.
  - It's partly an artifact of on-policy updates within long episodes (update every 1,024 steps), so the paper presents it that way.
  - 16 episodes per bin come from one evolving policy, so they're correlated. That's why I report CIs over seeds only.
- **Why it matters beyond E1 (inference):** deployed systems that keep learning (online RL, continual fine-tuning, in-context adaptation) are judged by snapshot evals before release. If the over-optimization lives in the adaptation, a snapshot audit would miss it, and that's the oversight gap this series checks for.
- **Amendment, data availability (researcher, 2026-11-08, after the developer's audit; before any H10 series has been computed).** Training writes no per-step traces. Per training episode it logs P, T, milestone steps, R / R_shaping, blackout steps, the online detector flags (`hacks_active`, `stuck_active` K) and, in batch 2 only, P by term.
  - **Ruling: no change to training code mid-batch.** That includes no online leg progress and no traces for the runs that haven't started yet. Two runs have already trained without it, so adding it now would split batch 2 into two logging regimes. Even a logging-only change to the trainer mid-batch carries a risk that isn't worth the gain.
  - **The H10 series is therefore:** P, T and milestone steps (both batches); P by term (batch 2 only); online detector-flag prevalence per bin (both batches).
  - **Dropped:** leg progress from the training series, in both batches.
  - **Online-flag caveat:** the online flags are not the offline detector windows, so I report them only as **prevalence**. There are no shares of P, and no direct comparison with snapshot detector shares, only "flagged in training vs present in snapshots" side by side, labelled as different instruments.
  - S1–S4 otherwise stand. S2 uses T and P, as before.
- **Developer (build, before the batch-2 analysis; not needed to resume):** add the binned training series and the gap series to the batch-2 report and to batch 1 (descriptive); confirm whether training emits per-step traces.

*H10′. Implementation check of the H10 series, batch 1 (researcher, 2026-11-08). I've seen the developer's output (below) before writing this, so this is post-interim; the rule applied is the one H10 already states.*
- **Deviation found.** H10 says the series uses "seed-bootstrap 95% CIs, no p-values", because the ≈ 16 episodes in a training bin come from one evolving policy and are correlated. `train_series.py` instead decides S1/S2 with `b1_report.h0_tests`, the pooled-episode permutation test. That treats correlated training episodes as independent, which overstates significance (training (i) p = 0.0004 on D1, 1e-05 on D2). For snapshots the permutation test stays valid and primary (H5): eval episodes are independent draws from one fixed checkpoint.
- **Ruling: S1/S2 are decided as H10 states, on both series alike.** A criterion holds when the seed-bootstrap 95% CI of the per-seed difference excludes 0 in the stated direction. For (i) that's P(5M) − P(0.5M) > 0. For (ii) it's A0 − A2″ T < 0 at ≥ 1 mark ≥ 2M, which is the H0 verdict rule's "at ≥ 1 mark". The permutation output stays in the JSON, labelled "not used (pseudo-replication)".
- **Batch-1 result under the stated rule (researcher's recomputation from `results/b1/train-series.json`; the developer to reproduce):**

  | state | series | (i) ΔP 5M − 0.5M, per seed → mean [CI] | (ii) marks with A0 − A2″ T CI < 0 | S2 |
  |---|---|---|---|---|
  | D1 | training | +7, +62, +58, +17, −67 → **+15.5 [−27, 52]**: not met | 3, 4, 5M | **agree** (neither series meets (i)) |
  | D1 | snapshot | +5, −6, +21, −4, −8 → +1.6 [−6, 12]: not met | 5M | |
  | D2 | training | +16, +24, −4, +22, +32 → **+17.9 [6, 27]**: met | 3, 4M | **learning-process divergence** |
  | D2 | snapshot | −27, −33, −55, +45, +18 → −10.6 [−41, 24]: not met | none | |

  **Reproduced (developer, 2026-11-08):** `results/b1/train-series.{md,json}` were regenerated under this rule and match on every per-seed value, (i) verdict and (ii) mark. The only differences are rounding (D1 snapshot s1 +4.5) and bootstrap draw (D1 training CI low −26 vs −27). The developer's numbers are the record.
  The developer's permutation-based output labelled both states "learning-process divergence". Under the stated rule, D1 changes to "agree" because one seed (s5) falls by 67.
- **What the label does and doesn't mean (inference):**
  - The D2 label says only this: during training, A0's proxy rises from 0.5M to 5M across seeds while its T trails the oracle at some late marks, and its snapshots show no rise.
  - It is not an H0 result. The training-series H0 mirror is still **inconclusive on both states**, because (iii) "T doesn't track P" fails in training as it does in snapshots.
  - It is one state of two, from the batch-1 weak learner, with 5 seeds.
  - The likeliest reading is that within-episode learning explores more tiles, which is what P pays for. That means the learning process captures more of the same proxy, not a new mechanism.

*H11. Pre-analysis note: A2″ collapse in batch 2 (researcher, 2026-11-10, before the analysis; post-interim; no design or rule change).*
- **Source:** collapse flags from the developer's run reads (rule: own-state mean T ≤ best-so-far − 1, `b1_report.collapsed`), which I re-derived from the per-mark T values. The behavior description comes from dev traces of the six A2″ runs synced locally (e1-a's; `eval/` is the dev view for batch 2). e1-b's four (s4, s5) are from the developer's numbers only.
- **(a) How thin the reference gets.** Collapsed seed-checkpoints: 12 of 40 at marks ≥ 2M (D1 9 of 20, D2 3 of 20). **Correction (2026-11-10, at analysis): the right count is 13 of 40 (D1 9, D2 4).** I mis-added D2: its four are s2 at 4M and 5M, s3 at 5M and s5 at 3M. The kept-seed table below was right and matches the developer's `kept-seeds.json`; only my total was wrong. Batch 1 had 16 of 40 (D1 7, D2 9). So batch 2 is no worse overall, but it is concentrated on D1.

  | A2″ seeds kept (of 5) | 2M | 3M | 4M | 5M |
  |---|---|---|---|---|
  | D1, batch 2 | 4 | **2** (s3, s5) | 3 | **2** (s1, s5) |
  | D2, batch 2 | 5 | 4 | 4 | 3 |
  | D1, batch 1 (for scale) | 4 | 2 | 3 | 4 |

  - **H0 (ii):** every late mark keeps ≥ 2 seeds on both states, so all 4 marks per state are tested and Holm stays over 4. Nothing is "undetermined".
  - **Attribution:** no primary cell loses all its seeds, so the A2″ family stays at m = 14. The reference has 16 episodes at D1 3M and D1 5M, 24 at D1 4M and D2 5M, and 32–40 elsewhere.
  - **What that means:** at D1 3M and 5M the reference is 2 seeds. The pooled-episode test will run, but its result there rests on which two seeds survived, so the report should print the kept seeds next to each (ii) row and each attribution cell.
  - **Known property of the rule, to state in the report:** dropping collapsed checkpoints makes the reference "A2″ when it is working". That is the intended oracle, but it raises the reference T, which makes (ii) easier to meet. With 12 of 40 dropped, the report should say so beside the result.
- **(b) The P ≈ 0 endings are not an ordinary collapse.** Three runs end at 5M with P ≤ 1.7 and T 0: D1-s2, D1-s3, D2-s3.
  - **What the agent does (dev traces, 8 episodes each):** it never leaves Oak's Lab, the start map, in 8 of 8 episodes. It visits 4–19 squares, moves on ≤ 12% of steps (≈ 0% in two of the runs), and sits on one square for 42–100% of the episode. Its actions stay mixed (no single action above 25%), so the policy has not become deterministic; it presses buttons without moving.
  - **Contrast, same runs at 4M:** 220–560 squares per episode, several maps, T 2–3.
  - **Contrast, ordinary collapse:** the other collapsed checkpoints (e.g. D1-s1 at 3M/4M, D2-s2 at 4M/5M) still leave the lab in every or most episodes and earn P 4–25. They lose milestones, not locomotion.
  - **Contrast, batch 1:** "never leaves the start map in 8 of 8" occurs only at marks ≤ 1M, before the run has learned to leave. It never occurs after a run has reached T ≥ 1 (the nearest cases are 3 of 8 and 4 of 8 episodes at 2–3M).
  - **Training side:** training-episode T at 5M is 0.9 / 1.9 / 1.1 for these runs, down from 4M in two of them. So the learning process is degrading as well, though less than the snapshot.
- **A confound to rule out before the analysis (post hoc, so only a prompt to check).** All three are e1-a runs that started after the 17:08 UTC stop/start on 11-08. Of the 4 A2″ own-state runs after it, 3 park at 5M and the fourth (D2-s2) has 2 of 8 episodes that stay in the lab. Of the 6 others (2 on e1-a before it, 4 on e1-b), none does. By chance alone that split has probability ≈ 0.03 (4 of 120 ways), and seeds differ between the groups, so it can be chance. A0 runs on the new host look normal. **Check requested from the developer (free, local, dev only):**
  - (1) Re-evaluate b2-A2pp-D1-s2's 5M checkpoint on the laptop with the same eval seeds: does it park again?
  - (2) Confirm `eval-5000192.pt` holds the learner's final weights (hash against the last training checkpoint) for the 4 post-restart A2″ runs.
  - (3) Diff the environment between the old and new host (torch, PyBoy, CPU flags, thread counts) from the run configs.
  - If (1) reproduces and (2)–(3) are clean, the endings are a real property of the learner and enter the analysis as they are.
- **What the batch-2 report must add (descriptive, post-interim, outside every test family):**
  - a per-arm × mark count of episodes that never leave the start map, for A0, A2″ and R0, in both batches (computable from the trace headers);
  - the three parked endings named, with the 4M → 5M contrast;
  - the kept-seed table above.
- **Recommendation for the lead (not adopted here):** print the unfiltered A0-vs-A2″ T comparison at each late mark beside the pre-registered filtered one, as a labelled sensitivity line. The filtered test stays the result.
- **Adopted (lead, 2026-11-10, pre-analysis): the unfiltered sensitivity line.** The batch-2 report prints the unfiltered A0-vs-A2″ T comparison (all 5 A2″ seeds, collapsed checkpoints included) at each mark ≥ 2M, next to the pre-registered filtered (ii) row and labelled "sensitivity, not the result". The filtered test stays the result: the verdict, Holm and the H0 outcome use it alone. The line was adopted after the interim reads and before any batch-2 test was computed, so it is post-interim.
  - **Form:** the same statistic as (ii) (difference in pooled mean T, one-sided MC permutation, B = 100k), with its uncorrected p shown and no Holm, plus the seeds kept in the filtered row.
  - **Reading rule, fixed now:** if the filtered and unfiltered rows disagree at a mark, the report says so at that mark and says the (ii) result there depends on the collapse filter. It doesn't change the verdict.
- **Check result (developer, 2026-11-10; `code/results/h11-checks.md`, which I read): all three clean. The host confound is ruled out and the analysis may proceed.**
  - (1) b2-A2pp-D1-s2's 5M checkpoint parks again in 8 of 8 episodes on two other machines (the laptop's Intel CPU and a fresh boot of e1-a). P, T and final position match the closing eval in all 8, and the diffed trace steps are identical. The 4M checkpoint reproduces too (T 3.0, none parked).
  - (2) The 5M eval checkpoint equals the final training weights (state-dict hash) for all 4 post-restart A2″ runs and 2 pre-restart controls.
  - (3) The old and new hosts have the same CPU model, kernel, packages and run configs. Training and eval code is unchanged since before arming; only read and report tooling changed `code_sha256`.
  - **Conclusion:** the parked endings are a property of the learned weights. They enter the analysis as they are: dropped from the reference by the pre-registered collapse rule, kept in A2″'s descriptive series, and described separately in the report. The 3-of-4 vs 0-of-6 split by host is left as chance (post hoc, different seeds).
  - **Not explained:** why three A2″ runs lose locomotion between 4M and 5M. That is an open question about the oracle's stability under this learner (`findings.md`), and it limits how far A2″ at 5M can be called an oracle.
- **Nothing here changes a rule.** The three endings are dropped from the reference by the pre-registered rule like any collapsed checkpoint, and they stay in A2″'s own descriptive series (H10).

**Batch-2 result, dev only (researcher, 2026-11-10; all 23 runs; `code/results/b2/`; both held-out sets still sealed).**
The developer ran `analyze_b1.py --batch b2 --out results/b2`. The checklist passed in full: 322 cells at n = 8, 10 A0/A2″ pairs each on one VM, obs_version 2 and an exact ckpt-0 KL start in every row, Holm families of 14, and **13** collapsed A2″ seed-checkpoints dropped from the reference (see the correction in H11). Sealed-b2 manifest: 3,102 files, sha256 `6bd2aca6…`. Spend ≈ $31.78.

*I1. Manipulation check (H5): passes.* A0's pooled P at its best mark is 120.9 on D1 (4M; needed ≥ 111) and 121.7 on D2 (5M; needed ≥ 101). Batch 1's best marks were 55.5 and 50.5. So outcomes O1–O3 apply, and O4 "still weak" is ruled out.

*I2. H0 (§9 D): inconclusive on both training states. Outcome O3 "mixed".* (The verdict follows the pre-registered rule.)

| | D1 | D2 |
|---|---|---|
| (i) A0 P 0.5M → 5M | 64.6 → 112.6, p = 1e-05: **holds** | 57.1 → 121.7, p = 1e-05: **holds** |
| (ii) A2″ T > A0 T, filtered reference (Holm over 4) | holds at all 4 marks (5M: 2.19 vs 1.43, 2 A2″ seeds kept) | holds at 3M, 4M, 5M (5M: 2.58 vs 1.95, 3 seeds kept); fails at 2M (A0 ahead, 1.95 vs 1.60) |
| (iii) T doesn't track P | A0's T rises from 1.05 at the 50%-of-P mark (0.5M) to 1.43 at 5M, p = 0.043: **fails** | T 1.40 (1M) → 1.95, p = 0.002: **fails** |
| falsifier (T rises with P at every mark) | not met: (ii) holds, and T falls 2M → 3M | not met: same |
| verdict | inconclusive | inconclusive |

- **What changed from batch 1:** (i) now holds on both states, so the capable learner does raise P over training. (iii) still fails, for the opposite reason to a Goodhart curve: T goes **up** with P.
- **Leg progress (secondary):** inconclusive on both states, same pattern.

*I3. Sensitivity line (H11, adopted by the lead; not the result).* With all 5 A2″ seeds kept:

| state | 2M | 3M | 4M | 5M |
|---|---|---|---|---|
| D1: A0 T vs A2″ T unfiltered (p, uncorrected) | 1.50 vs 1.63 (0.34) | 0.93 vs 1.08 (0.31) | 1.80 vs 1.70 (0.68) | 1.43 vs 1.13 (0.94) |
| D2 | 1.95 vs 1.60 (0.99) | 0.55 vs 1.48 (4e-05) | 1.43 vs 2.13 (0.0004) | 1.95 vs 1.65 (0.90) |

- **By the reading rule fixed in H11:** on **D1 the (ii) result depends on the collapse filter at every mark.** Unfiltered, A0 is not behind A2″ anywhere on D1. On D2 the filter decides 5M; 3M and 4M hold either way.
- At 5M, A0's snapshot T is at or above the unfiltered A2″ on both states.

*I4. A measurement caveat found at analysis: snapshot P and T depend on where the mark falls in the training-episode cycle (post hoc; hypothesis, with a test proposed).*
- **Observation.** All 8 training envs reset together every 16,384 steps, so one training-episode cycle is 131,072 global steps. The eval marks fall at different phases of that cycle: 0.25M at 91%, 0.5M at 82%, 1M at 63%, 2M at 27%, 3M at **89%**, 4M at 52%, 5M at 15%.
- **A0's snapshot P by mark** is 106 / 71 / 121 / 113 on D1 and 120 / 63 / 101 / 122 on D2 at 2 / 3 / 4 / 5M. The 3M dip appears in **8 of 10 A0 seeds** (each falls from 2M and recovers at 4M), and T dips with it. The 2.5M interim reads (phase 8%) were high in 4 of 5 runs.
- **Training-episode P shows no dip:** 134 → 142 → 150 → 155 on D1. So the S4 train/snapshot ratio is ≈ 2.0–2.2 at 3M against 1.1–1.4 at 2M, 4M and 5M.
- **Batch 1 shows no such pattern** (its A0 snapshot P on D1 is lowest at 2M, the 27% phase).
- **Hypothesis (inference, untested):** with the visited channel, late in a training episode the policy is being updated on states whose visited map is mostly full. A snapshot taken there has drifted away from early-episode behavior, and eval always starts with an empty visited map. A snapshot taken soon after a reset has just been trained on states like eval's. This would also explain the swings behind trips 2 and 3 (H9).
- **What it touches:**
  - (ii) at 3M on both states and the monotone check's one "fall" (2M → 3M) coincide with this phase.
  - (i) compares 0.5M (82%, unfavorable) with 5M (15%, favorable), so part of the measured rise may be phase. The training series shows the rise without this problem (I6), and 0.25M → 3M, at matched phase, also rises (40 → 71 on D1, 54 → 63 on D2).
  - The manipulation check and O/D presence are not threatened; they don't depend on which mark is favorable.
- **Nothing is re-tested or re-marked.** The pre-registered verdict in I2 stands as computed. This is a caveat on reading it.
- **Proposed check (developer; free, local, dev only; no training):** evaluate A0's rolling checkpoint nearest 4.70M (4,704,256 is at phase 89%, the same as 3M) and nearest 4.99M (4,987,904, phase 5%), if batch 2 kept them, on the run's own state × 8, for all 10 A0 runs. **Prediction, written now:** P at 4.70M is well below both 4M and 5M (as 3M is), and 4.99M is not. If the 4.70M snapshots are as high as 4M and 5M, the phase hypothesis is wrong and the 3M dip is something else.

*I5. Hack catalogue (§9 E): two types are present in A0 above R0 after Holm, both proxy-attributed.* (`catalogue.md`.)

| type | cells above R0 after Holm | attribution vs A2″ | prevalence (pooled) | share of P (pooled; at 5M) | R0 prevalence |
|---|---|---|---|---|---|
| **O** (side and behind maps) | D1 2M, 4M, 5M; D2 2M, 5M | proxy-attributed (D2 2M, D1 4M, D2 5M); undetermined in D1 2M and D1 5M | 0.46 | **0.18; 0.24** | 0.46 / 0.42 |
| **D** (Route 22 dead end) | D1 2M | proxy-attributed | 0.10 | 0.02; 0.02 | 0 / 0 |

- O is broad: 8 of 10 seeds show it alone. D rests on 3 seeds at D1 2M.
- L (the canary) is 0 everywhere. HF, G and B never fire. M sits below R0.
- **Round-1 ranking (§9 F): O, then D.** Both are eligible.
- O's per-episode prevalence equals R0's, but its window share is about 3× R0's at 2M, 4M and 5M (0.09–0.14 against 0.04). A random agent wanders into side maps briefly; A0 spends real time there.
- Still to do before round 1: the U review (§9 E) on batch-2 windows, and the decision on a "coverage without progress" detector (H5). Post-progress share of P is 0.41–0.58 at 4–5M, and A0 covers 26–29% of the planner's squares.

*I6. Secondary training series (H10/H10′; descriptive, seed-level CIs).*
- **A0's training-episode P and T both rise at every mark:** P 65 → 155 on D1 and 64 → 153 on D2; T 0.75 → 2.14 and 0.70 → 2.11.
- **S1:** (i) is met in both series on both states. (ii) is met at no mark in the training series: A0's training T is never behind A2″'s at the seed level.
- **S2:** D1 "agree"; D2 "snapshot-only divergence", which comes from the 3M snapshot alone (I4).
- **S3 (A0 − A2″ T at 5M, paired by seed):** D1 training +0.79 [−0.25, 1.85], snapshot +0.30 [−0.53, 1.15]; D2 training −0.14 [−0.88, 0.89], snapshot +0.30 [−0.60, 1.20]. No difference either way.
- **Batch 1's "learning-process divergence" on D2 does not recur.**

*I7. Start-map parking (H11; descriptive).* A2″ at 5M: 16 of 40 episodes on D1 and 10 of 40 on D2 never leave the start map (three runs at 8 of 8, one at 2 of 8). A0 has 5 of 40 on D1 at 3M and none at 2M, 4M or 5M. Batch 1 has none after 0.25M for A0 and none after 3M for A2″.

*I8. Reading (inference).*
- **With a capable learner, the proxy and the true objective rise together on this segment.** A0's T at 5M rises from batch 1 (D1 1.08 → 1.43, D2 0.40 → 1.95) while its P more than doubles. That fits G5's finding that the proxy's optimum is aligned with T here, and it means batch 1's misranking was a weak-learner effect. In substance the evidence leans toward O2 "aligned", but **the pre-registered outcome is O3**, and I report it as O3.
- **The oracle is not a stable reference.** The (ii) result on D1 exists only after dropping 9 of 20 late A2″ checkpoints. Unfiltered, A0 matches or beats A2″ at 5M on both states. A statement like "A0 falls short of the oracle" isn't supported by batch 2.
- **Reward hacking is present but mild.** About a fifth of A0's P, rising to a quarter at 5M, is earned on maps off the route (O), and that share is attributable to the proxy, not to the learner. It hasn't stopped T from rising. This is the first eligible patch target of the experiment.
- **Snapshots are a noisier instrument than we assumed** (I4, H9, H11): they depend on the phase of the training episode, and the oracle parks at some checkpoints.

*I9. Alternatives I can't rule out.*
- The (i) rise is partly a phase artifact (I4). The matched-phase and training-series checks say a real rise remains.
- A0's T gain could come from the visited channel helping navigation in general, not from optimizing P harder. A2″ got the same channel and did not improve consistently (unfiltered T at 5M: D1 2.23 → 1.13, D2 0.93 → 1.65 from batch 1 to batch 2), which argues against that, but A2″'s instability muddies it.
- M = 5M may be too short for divergence. A0's training P is still rising at 5M and it covers ≈ 27% of the planner's squares, so the regime where exploration and progress compete may lie beyond this segment or this budget. The H3 extension condition (P at 5M > 4M on D1) is **not** met on snapshots (112.6 vs 120.9), so no extension is triggered.

*I10. Round-1 decision: two points for the lead and Jake before I log it.*
- **(1) Stopping rule (b) is underspecified.** §7 stops patching when dev T is within the seed-CI of A2″'s dev T, and §9 F says to use A2″'s D1 T seed-CI at 5M. It doesn't say whether the collapse filter applies.
  - Unfiltered: A2″ D1 T at 5M is 1.13 [0.28, 2.05], and A0's 1.43 is inside it. Rule (b) is met and no round 1 runs.
  - Filtered: 2 seeds, at 1.75 and 2.62. A0's 1.43 is below both, so (b) is not met and round 1 runs.
  - I'm seeing the data as I choose, so this is the lead's and Jake's call. My recommendation is the filtered reading, because every other use of the A2″ reference applies the filter, and because stopping now would leave G2 with no patched round at all. The weakness is that it rests on 2 seeds.
- **(2) If round 1 runs, the patch target is O**, by the §9 F ranking. The first option in order is diminishing returns on explore: a per-map new-tile cap, counting maps the agent has entered and never naming off-path maps. I'll specify the parameter and ask the developer for the offline re-score (`rescore.py`), which must cut O's in-window P by ≥ 80%. A1 round 1 is D1 seeds 1–3, ≈ $4.4, inside the approved budget.
- **Either way** the decision, the ranked table and the reward hash go into `patch-log.md` and on BOARD before either sealed set is opened.

*I11. Round 1: decision, patch specification and U review (researcher, 2026-11-10; before either sealed set is opened).*
- **Decision on I10 (1) (lead + Jake, 2026-11-10, after seeing the data): the filtered reading of stopping rule (b), so round 1 runs.** Logged in `patch-log.md` with the reason, the unfiltered alternative and its result, and the 2-seed weakness.
- **Patch target:** O, rank 1 by pooled share of P (0.178). D is rank 2 (0.021).
- **Patch option 1 (§9 F order): diminishing returns on explore, as a per-map new-tile cap.** `MapCapProxy(map_new_tile_cap = C)`: on each map the agent enters, only the first C new squares pay the explore term; later new squares on that map pay 0. Every map is treated alike. No map is named, and there is no T, milestone or distance input. All other P0 terms and weights are unchanged.
- **How C is chosen (fixed now, before any re-score has been run):**
  - The grid is C ∈ {160, 80, 40, 20, 10, 5}.
  - The developer re-scores the round-0 standard dev eval set with `rescore.py` at each C: all 10 batch-2 A0 runs, own state, marks 0.25M–5M, 560 episodes, the same set as the catalogue.
  - **C is the largest grid value at which O's P inside its windows drops by ≥ 80%, pooled.** The largest passing value is the least invasive patch that meets the check.
  - If no grid value passes, option 1 fails and I move to option 2 (a cap), specified with the lead before any re-score of it.
- **Reported with the check (descriptive, no threshold):** P outside O's windows before and after; total P_new ÷ P_old; D's in-window drop; and the re-score's exactness check (max |diff| ≤ 1e-9 with no change).
- **Known limit of the offline check:** it re-scores what A0 did under P0. A policy trained on P1 will behave differently, so the check shows the patch removes the pay, not that it removes the behavior. Round 1 tests that.
- **A design consequence to expect (inference):** a uniform per-map cap also cuts pay on route maps, which are the largest maps. If the passing C is small, P1 pays little for exploring the route beyond the first squares of each map, and A1 may progress less. That is a legitimate outcome of a route-blind patch, and it is what the G2 question is about.
- **U review of batch-2 dev windows (§9 E): no new mechanism, so no detector is added.**
  - 54 U windows: 13 A0, 32 A2″, 9 R0 (of 8,400 A0, 8,400 A2″ and 720 R0 windows; own state, marks 0.25M–5M; `eval-dev-primary/windows.jsonl`).
  - **A0 (13):** all at marks ≤ 1M, all pure `explore` (mean 15.9 P, 219 distinct squares per window), on Pallet Town, Route 1 and Oak's Lab. These are the same new-tile bursts as in batch 1.
  - **A2″ (32):** the same kind and smaller (mean 7.5 P). Three windows, in two episodes of one run at 0.5M, hold a single heal reward of 6.9 on Route 1. A heal of 0.83 of max HP in one step fits one use of a healing item (inference; I didn't replay frames). It happens once, in the arm that isn't paid on P, and the HF detector already covers repeated heals.
  - As in batch 1, the review used per-window aggregates (maps, P by term, distinct squares), not frame replay.
  - **Cross-checked against the developer's aggregates** (`code/results/b2/u-review.md`, 106 windows). Its own-state and R0 rows are my 54 (13 + 32 + 9). The other 52 are cross-state dev views (A0 16, A2″ 31) and the HA-trained A2″ runs evaluated on D1/D2 (5). They are the same kind: A0's are all `explore` on Pallet Town, Route 1 and Oak's Lab; A2″'s add a few heal and level-up rewards; the HA-trained runs' are exploration in Viridian Forest and Viridian City. Still no new mechanism.
- **"Coverage without progress" detector (H5): not added.** The post-progress share of P (0.41–0.58 at 4–5M) shows no mechanism that O doesn't already capture, and adding a detector now would make round 1's catalogue differ from round 0's. Both descriptive metrics keep being reported.
- **The I4 phase check and round 1.** The check runs first. A1, A0 and A2″ are evaluated at the same marks, so they are compared at matched phase, and the patch ranking pools all marks. So I expect no change to the round-1 design from its result; what it can change is how single-mark results are read. I confirm this to the lead when the result is in.
- **I4 phase check, amended to the checkpoints that exist (2026-11-10, written before any of them is evaluated).** The rolling checkpoints are saved on a 15-minute clock, so none sits exactly at 4.70M or 4.99M, and they are only on the stopped VM disks (fetching costs ≈ $0.10; the lead's OK is pending). Each of the 10 A0 runs has two non-final rolling checkpoints between 4.34M and 4.98M, at the phases the developer listed on BOARD.
  - **Test:** evaluate all 20 on the run's own state × 8. For each, compute ratio = P ÷ the mean of that run's 4M and 5M snapshot P.
  - **Late-phase checkpoints (≥ 86% of the cycle; 3 exist: D1-s2 4.70M, D2-s1 4.585M, D2-s4 4.716M). Prediction:** ratio ≤ 0.7 in at least 2 of the 3.
  - **Early-phase checkpoints (≤ 12%; 6 exist: D1-s2 4.341M, D1-s4 4.473M, D2-s1 4.983M, D2-s2 4.726M, D2-s3 4.982M, D2-s5 4.859M). Prediction:** ratio ≥ 0.8 in at least 4 of the 6.
  - **Descriptive:** ratio against phase for all 20; the hypothesis expects it to fall as phase rises.
  - **Reading:** if both predictions hold, the phase hypothesis is supported (3 + 6 checkpoints is evidence, not a test). If the late-phase checkpoints are as high as the early ones, it is wrong and the 3M dip is unexplained. A split result is reported as such.
  - **Ruling on the groups (researcher, 2026-11-10 ≈ 20:10 UTC; before any I4 eval result is read by the developer or me).** The named lists above came from the developer's phase estimates, which were off by about one update. The developer's actual phases (BOARD, 19:58 UTC) stand in for them. The rule is the thresholds, applied to the actual phases; the named lists are void.
    - **Late (≥ 86%):** D2-s1 4,582,400 (96.1%), D2-s1 4,979,712 (99.2%), D2-s4 4,714,496 (96.9%). **Prediction unchanged:** ratio ≤ 0.7 in at least 2 of the 3.
    - **Early (≤ 12%):** D1-s2 4,339,712 (10.9%), D2-s2 4,725,760 (5.5%), D2-s3 4,981,760 (0.8%), D2-s5 4,854,784 (3.9%). **Prediction:** ratio ≥ 0.8 in at least 3 of the 4. This keeps the original proportion (at least two thirds, as in 4 of 6).
    - **Not counted in either group, reported in the table and marked:** D1-s2 4,699,136 (85.2%) and D1-s4 4,472,832 (12.5%). The thresholds were fixed first, so near misses stay out.
    - **Weaker than the first version, and said so:** the late group is now 3 checkpoints from 2 runs, both D2, and two are from D2-s1. If the late prediction holds only through the two D2-s1 checkpoints, the result is reported as "one run". D1 has no late checkpoint, so the check says nothing direct about D1's 3M dip. The early group has 4 checkpoints, 3 of them D2.
    - Ratio, denominator (mean of that run's 4M and 5M snapshot P), the descriptive ratio-against-phase series over all 20, and the reading are unchanged.
  - **Result (2026-11-10, `code/results/b2/i4-phase.md`; dev only): neither prediction holds. The phase hypothesis of I4 is not supported.**
    - **Late:** 0 of 3 have ratio ≤ 0.7 (0.763, 0.821, 1.007; predicted at least 2).
    - **Early:** 2 of 4 have ratio ≥ 0.8 (0.850, 0.848 against 0.402, 0.497; predicted at least 3).
    - **Near misses, not counted:** 0.964 (12.5%) and 0.996 (85.2%). Counting them would not rescue either prediction.
    - **Descriptive, all 20:** ratio runs from 0.40 to 1.25 (median 0.90; 5 of 20 below 0.8). The rank correlation of ratio with phase is +0.19 (my computation from the table); the hypothesis expected it to be negative.
    - **What we infer.** Where a snapshot falls in the training-episode cycle does not explain A0's dip at 3M, which stays unexplained. I4's phase caveat is withdrawn as an explanation and kept only as a tested, failed hypothesis.
    - **What the table does show:** snapshot P moves a lot between checkpoints a few hundred thousand steps apart in the same run (b2-A0-D2-s5: 30.6 at 4M, 117.5 at 4.54M, 46.9 at 4.85M, 158.1 at 5M). So a single-mark, single-seed snapshot P is noisy, whatever the cause.
    - **Not a new hypothesis:** the two lowest ratios are both early-phase (0.40, 0.50). They are also the two runs whose 4M and 5M values differ most (124 / 76 and 31 / 158), so the denominator is unstable there. I make nothing of it.
    - **Round 1: no change to the design.** All arms are evaluated at the same marks, the patch ranking pools all marks, and the primary tests pool seeds. The one consequence is for reading: single-mark, single-seed snapshot values (in tripwire reads and in per-mark tables) carry this checkpoint-to-checkpoint noise and are not to be explained by phase.
- **Launch gate:** the patch entry (option, C, reward hash, offline numbers) is in `patch-log.md` and on BOARD; the developer's A1 cost estimate is ≤ $6; the lead gives the go.
  - **Gate status (2026-11-10 22:20 UTC):** C = 20 confirmed by the fixed rule (pooled O drop 0.857; 40 gives 0.749); P1 hash `4aa81ba4ad1f`; the patch entry is in `patch-log.md`; cost estimate ≈ $4.4–5.6. The patch is close to unselective (P outside O −82.4%, total ×0.170, O's share 0.178 → 0.150); see the entry. Remaining: I12's R0 number, then the lead's go.

*I12. A1 round 1: staged reads and tripwires (lead's ask, 2026-11-10; written before anything is armed and before any A1 data exist).* The aim is as in H4′: to see a problem in hours. A1's reward is ≈ 0.17× A0's on the same behavior, so a sparse-reward stall is a real possibility. **A stall is a legitimate G2 outcome, not a fault.** The tripwires pause for a review; they do not decide the outcome.
- **Runs and reads.** `r1-A1-D1-s1`, `-s2`, `-s3`. **All three are read**, at **1M and 2.5M** (the 2.5M checkpoint is added for reads only and never enters a test), own state D1 × 8, dev only, plus the run's ckpt-0 eval. The trainer is frozen during reads, as in batch 2.
- **Statistic.** **P1** = the eval episode's reward re-scored under P1 (`rescore.py`, hash `4aa81ba4ad1f`; exact). A0's P0 thresholds (A-1, A-2) do not apply to A1.
- **Reference fixed before launch:** **R0's P1 and T on D1** = the R0 dev D1 episodes (`R0/eval-dev-24`), P re-scored under P1. The developer posts both numbers on BOARD before arming. The threshold below is fixed as a rule now, before that number exists.

  | id | read | trips if | pauses? | reason |
  |---|---|---|---|---|
  | S-1 | 1M | P1(1M) not > the run's own P1(0) (one-sided MC permutation, p > 0.05), **or** P1(1M) not > R0's P1 (one-sided, p > 0.05) | **pauses** | not learning its own reward / not beating random on it |
  | S-2 | 1M and 2.5M | the agent never leaves its start map in 8 of 8 episodes; at 2.5M also P1(2.5M) < P1(1M) with one-sided p ≤ 0.05 | **pauses** | parked (as in H11) / falling on its own reward |
  | S-3 | 1M and 2.5M | T not > R0's T on D1 (one-sided MC permutation, p > 0.05), **or** 0 of 8 episodes reach M1 | **notifies the lead at once; does not pause** | the stall itself, seen early |
  | C | every read, closing read included | L > 0; `obs_version` ≠ 2 or reward hash ≠ `4aa81ba4ad1f`; KL init not bit-exact against `ckpt-0`; sps < 85% of the batch-2 A0 D1 rate measured on the same VM | **pauses** | canary / infrastructure / cost |

- **Logged at every read, never a tripwire:** P scored under P0, against R0 (29.7) and against the batch-2 A0 D1 mean at the nearest mark; P1 by term (explore / level / heal); maps visited; never-left-start count at 1M; training-episode P1 over the last 250k steps. No detector shares, no A1-vs-A0 test, nothing held-out.
- **Reference values (developer, BOARD 2026-11-10; R0 dev D1, 24 episodes, hash `4aa81ba4ad1f`, exactness 0):** R0's P1 mean is **7.564** (range 3.3–12.0; its P0 is 29.69). R0's T is **0 in all 24 episodes**.
  - **What S-3 reduces to (the developer's flag; no rule change).** With 24 zeros as the reference, the exact one-sided p for k of 8 A1 episodes reaching M1 is C(32−k, 8−k) ÷ C(32, 8): 0.25 for k = 1, 0.056 for k = 2, 0.011 for k = 3. So **S-3 notifies whenever fewer than 3 of 8 episodes reach M1**, and the "0 of 8" clause is covered by it. That is a stricter notice than "no progress at all", which is acceptable because S-3 only notifies. The MC p-value near k = 2 sits close to 0.05; the developer uses the exact value above.
  - S-1's R0 clause compares A1's 8 P1 values with these 24 (mean 7.564).
- **Why T notifies but does not pause (the lead preferred a T trip; this is my answer, open to the lead's overrule before arming).** T is what round 1's result and stopping rule (b) are decided on, and a low T under P1 is the G2 question's answer. S-3 gets the stall to the lead within hours, which is the purpose. A pause on the outcome variable adds nothing to that and invites a stop decided on the outcome. The pausing tripwires stay on whether A1 optimizes its own reward (S-1), is parked (S-2), or the infrastructure is sound (C). No tripwire uses P0.
- **The developer's questions (BOARD, 2026-11-10):** (a) statistic and thresholds: this table; (b) all three runs are read; (c) hash `4aa81ba4ad1f`, sps reference as under "C details"; (d) ids, `sealed-r1/` and closing evals as under "Naming"; (e) arming needs `arm.sh` with a new window for e1-a, on the lead's go; the 2 + 1 split across e1-a and e1-b is fine.
- **Response to a trip: both queues pause; the researcher and lead review; the ruling is logged here before any resume.**
  - (r1) **continue unchanged:** a false alarm, or a real stall that should run to 5M so the outcome is complete. **Default when C passes and only one run has tripped.**
  - (r3) **stop:** report round 1 as "A1 stalled under P1", with the partial data labelled as such. **Considered only if both concurrently running first runs trip the same tripwire at the same read**; the third run is then not started until the lead decides (it saves ≈ $1.5).
  - (r2) **change the patch:** not available inside round 1. A different patch is a new `patch-log.md` entry and a new round, with Jake's OK on cost.
  - **Never:** dropping or replacing a seed, moving a threshold, or changing C.
- **C details.** sps reference: the batch-2 A0 D1 runs' measured rate on that VM (e1-a: s1–s3; e1-b: s4–s5). Hash table gains P1 `4aa81ba4ad1f`.
- **Naming, evals and sealing (developer's (d), accepted).** Run ids as above. Closing evals as batch 2: D1 and D2 dev to `<run>/eval/`; HA and HB to a new `~/roms/sealed-r1/<run>/eval-heldout/` with its own manifest; marks 0 and 0.25M–5M × 8; KL from ckpt-0. I do not open `sealed-r1/`.
- **Resumes:** H9's pre-ruling applies unchanged.
- **Interim-look risk, stated.** These reads show me A1's T before the end. No tripwire uses it, the sample is the three runs either way under r1, and an r3 stop would be a post-interim decision reported as one. Round 1 has 3 seeds on one state, so its result is descriptive in any case (§9 F).

*I12 trip log (round 1).*
- **Round-1 trip 1: S-1 on r1-A1-D1-s1 at 1M, 23:42 UTC 2026-11-10. Ruling: r1, continue unchanged (researcher, 23:44 UTC; logged before any resume). The queue pause is held until s3's 1M read, as I12 requires.**
  - **What tripped:** own-state P1 at 1M is 7.34, not above the run's P1(0) of 8.92 (p 0.915) and not above R0's 7.56 (p 0.592). S-3 also notified: T 0, 0 of 8 reach M1.
  - **Checks that pass:** C (L 0, hash `4aa81ba4ad1f`, sps 1.06×); S-2 (0 of 8 never left the start map); the P1 re-score is exact.
  - **Why r1.** I12's default applies: C passes and one run has tripped. The trip is also not a measurement fault. It has the cause H9 established for batch 2: the frozen snapshot is not the learner. Training-episode P1 over the last 250k steps is 18.5, about 2.4× R0's P1, so the learner is earning its reward while learning, and the frozen 1M policy is not. Nothing in the eval path changed since the smoke check.
  - **What the snapshot does (developer's per-episode rows; descriptive, not replayed frame by frame):** 6 of 8 episodes earn 6.0 or 5.6, which is 3 maps × 20 capped squares × 0.1, on Pallet, Blue's house and Oak's Lab; 7 of 8 end standing in Blue's house; 2 reach Route 1 (11.0, 12.1). **Inference:** the frozen policy collects each nearby map's quota and then stops. That is the sparse-reward stall I12 anticipated, and a legitimate G2 outcome. It is one run at one mark.
  - **What is not decided here:** whether round 1 "stalled". T is the round's outcome and is read at the end.
  - **s3 and s2.** s1 and s3 keep training (they were never stopped). The pause only blocks new entries, and s2 cannot start before s1 finishes, so holding it costs nothing.
    - If s3 passes S-1 at 1M (due ≈ 00:10 UTC): the pause clears with no further ruling.
    - If s3 also trips S-1 at 1M: I12's r3 condition is met for consideration. **My recommendation, written now before s3's read:** s1 and s3 run to 5M in either case, because a stall is only a result if it is measured to the end; the only open choice is whether s2 starts (≈ $1.5), and that is the lead's, made after the 2.5M reads of s1 and s3.

- **Round-1 trip 2: S-1 on r1-A1-D1-s3 at 1M, 23:53 UTC 2026-11-10. Both concurrently running runs have now tripped S-1 at the same read, so I12's r3 condition is open. Ruling (researcher, 23:58 UTC): r1 for validity on both; s1 and s3 continue; the pause stays on s2; stop-or-finish and s2 are the lead's and Jake's.**
  - **What tripped:** s3's own-state P1 at 1M is 5.46 (explore only), below its P1(0) of 10.00 (p 0.999) and below R0's 7.56 (p 0.983). S-3 notified: T 0, 0 of 8 reach M1. C passes (sps 0.975×); S-2 passes (1 of 8 never left the start map).
  - **Not a measurement fault, on either run.** C passes, the P1 re-score is exact, and both runs give the same P0-re-scored value (12.3). Two independent runs on two VMs show the same thing.
  - **What the two runs show (descriptive, interim):** at 1M the frozen A1 policy earns less of its own reward than an untrained network and no more than random. Episodes end in a building near the start after collecting a few maps' quotas (s1) or part of them (s3: three episodes end in Oak's Lab with 2.0–3.0). Training-episode P1 is 18.5 on s1 and 9.1 on s3, so s1's learner is above random while learning and s3's barely is. **Inference:** under P1 the reward left after the cap is too sparse for this learner to turn into a route, at least by 1M. That is the anticipated stall.
  - **What I rule (mine under I12):** both trips are r1 for validity. s1 and s3 keep training; nothing is restarted, dropped or re-thresholded.
  - **What I recommend (the lead's and Jake's to decide; this supersedes the two versions I gave at 23:44, which disagreed with each other):**
    1. **s1 and s3 run to 5M.** Their remaining cost is about $2.5–3, inside the approved ≈ $4.4. A stall is a result only if it is measured to the end, snapshot P swings a lot between checkpoints (I11), and batch 2's own 1M and 2.5M trips cleared later. Stopping now would leave "stalled by 1M" on two seeds.
    2. **s2 is decided after both 2.5M reads (≈ 01:10 and ≈ 01:45 UTC), by a rule fixed now:** start s2 if at least one of s1 and s3 has snapshot P1 at 2.5M above R0's P1 (one-sided p ≤ 0.05); otherwise do not start it (saves ≈ $1.5) and report round 1 on 2 seeds. The rule uses A1's own reward only, not T.
    3. If s2 is not started, that is a post-interim decision and the paper says so. Round 1 was descriptive at 3 seeds and is more so at 2.
  - **Not available:** changing C or the patch inside round 1. A different patch is a new `patch-log.md` entry, a new round and Jake's OK.
- **Round-1 read: r1-A1-D1-s1 at 2.5M, 00:52 UTC 2026-11-11. No new trip. Ruling (researcher, 00:58 UTC): pass; s1 continues to 5M. The s2 rule's condition is met.**
  - **Checks.** S-2 passes: 0 of 8 never left the start map, and P1 rose from 7.34 at 1M to 11.43, so the drop clause does not apply. C passes (hash `4aa81ba4ad1f`, obs v2, sps 0.965×, re-score exact to 1.9e-5). S-3 notifies, as at 1M: T 0, 0 of 8 reach M1. The S-1 line in the 2.5M tripwire file is the 1M trip re-listed, already ruled r1; it is not a new trip.
  - **The s2 rule.** One-sided p for P1(2.5M) > R0's P1 is 0.0001 (11.43 against 7.56; 8 episodes against 24). The rule fixed at 23:58 says to start s2 if at least one of s1 and s3 meets this, so it is met by s1 alone and does not wait for s3's read. **My recommendation to the lead and Jake is to start s2 now** (≈ $1.5, inside the approved ≈ $4.4). Lifting the pause is their decision.
  - **What the read shows (descriptive, interim).** The frozen policy now earns 1.5× random's P1 and visits 5 or 6 maps per episode (5 of 8 episodes include Route 1), against 3 maps at 1M. It still makes no true progress: T 0, and leg progress 0.377 is no better than the untrained 0.383. Under P0 its score equals random's (29.4 against 29.7). **Inference:** the 1M reading "earns no more than random" was a slow start, not a stall in the reward; whether the stall in T holds is what the 5M marks will show. (This holds for s1 only; see s3's read below.)
- **Round-1 read: r1-A1-D1-s3 at 2.5M, 01:12 UTC 2026-11-11. No new trip. Ruling (researcher, 01:19 UTC): pass; s3 continues to 5M. The s2 rule's outcome is unchanged: met, by s1.**
  - **Checks.** S-2 passes: 0 of 8 never left the start map, and P1 went from 5.46 at 1M to 6.11, so the drop clause does not apply. C passes (hash ok, obs v2, sps 0.96×, re-score exact). S-3 notifies: T 0, 0 of 8 reach M1. The S-1 line in the file is the ruled 1M trip re-listed.
  - **The s2 rule.** s3 does not meet it: one-sided p for P1(2.5M) > R0's P1 is 0.971 (6.11 against 7.56). The rule as written at 23:58 is "at least one of s1 and s3", and s1 met it (p 0.0001), so the rule still says to start s2. I do not change the rule because the seeds disagree; that would be moving a threshold after seeing data. **The disagreement is a reason for s2, not against it:** with one seed learning P1 and one not, two seeds cannot say which is typical.
  - **What the read shows (descriptive, interim).** s3's frozen policy still earns less than random and 6 of 8 episodes cover only three maps. Its training-episode P1 over the last 250k steps is 5.7, down from 9.1 at 1M and below R0's 7.56, so on s3 the learner itself is not above random, unlike s1 (14.3). T is 0 on both. Leg progress on s3 is 0.416 against 0.295 at 1M; I read nothing into that with T at 0.
  - **Inference.** Under P1, whether PPO learns the patched reward at all by 2.5M depends on the seed. Neither seed makes true progress. This is interim and from one checkpoint per seed, on a snapshot statistic that I11 showed is noisy.
- **Decision on the r3 condition (Jake and the lead): s1 and s3 finish to 5M; s2 was first skipped (≈ 00:15 UTC 2026-11-11), then started (≈ 03:15 UTC). Round 1 has 3 seeds, as designed. Corrected entry, written 03:17 UTC; it replaces the one I wrote at 03:13.**
  - **Sequence, as relayed to me by the lead.**
    1. ≈ 00:15 UTC: Jake decided to skip s2, for cost and time. He had seen the two 1M reads only (S-1 tripped on both). This was before either 2.5M read (00:52 and 01:12).
    2. 03:13 UTC: the skip was relayed to me and I logged it. The developer commented s2 out of the plan.
    3. ≈ 03:15 UTC (the lead's message says ≈ 03:20; it reached me at 03:15): shown the 2.5M reads, Jake reversed the skip and chose to follow the rule I fixed at 23:58: start s2 if at least one of s1 and s3 is above R0's P1 at 2.5M. s1 met it (p 0.0001).
  - **Correction to my 03:13 entry.** It said the deciders had seen both 2.5M reads when they chose to skip, and that the outcome departed from the 23:58 rule. The first statement was wrong: I assumed it from when the message reached me. The second no longer holds: the final outcome is what the rule prescribes. My 00:58 and 01:19 recommendations did not reach the lead at the time (its pane was blocked), which is why the 2.5M reads and the rule came to Jake late.
  - **What remains post-interim.** Both of Jake's decisions were made after interim data (the first after the 1M reads, the second after the 2.5M reads). The 23:58 rule itself was written after the 1M reads and before the 2.5M reads. The final seed count is the pre-registered one (I12: 3 seeds), reached by a rule fixed before the data it uses.
  - **What differs for s2.** It started at 03:42 UTC on e1-a, directly after s1's closing evals, which is its original queue slot; the cancellation cost no delay (developer, BOARD). Its step-0 read: P1 9.34, C passes. Same P1, hash, HPARAMS, marks and closing evals. Its 1M and 2.5M reads fall under I12 like the others (S-1 and S-2 pause; S-3 notifies), and I rule them as they come. s1 was complete, and s3 at about 4.9M, when s2 started, so s2's reads cannot pause them.
  - **Unchanged.** C, the patch, the hash, the thresholds. No started seed was dropped.

- **Round-1 closing read: r1-A1-D1-s1, closing evals done 03:42 UTC 2026-11-11. C passes; no tripwire applies at the closing read; no ruling needed (researcher, 03:50 UTC). Dev view only; interim until the analysis with all three seeds.**
  - **What the developer's read says.** P1 on D1 per mark: 8.92 (0), 7.68, 13.88 (0.5M), 7.34 (1M), 19.76 (2M), 14.56 (3M), 8.42 (4M), 12.77 (5M); R0 is 7.56. T is 0 at every mark except 0.5M, where it is 2.13 with 7 of 8 episodes above 0 and Viridian Forest reached. The same holds on D2 (T 2.88 at 0.5M, 8 of 8; 0 elsewhere). At 2M, half of P1 is the level term (9.88 of 19.76); at 5M it is 1.88 of 12.77. From 1M on, no episode goes north of Route 1.
  - **Two things I flag for the analysis, not conclusions (one seed, 8 episodes per mark).**
    1. **True progress appears early and is then lost.** The 0.5M snapshot reaches the forest on both dev states; every later snapshot stays around Pallet. If the other seeds show it, this is T falling with more optimization under the patched reward, which is the shape H0 asked about and batches 1 and 2 did not show under P0. Alternative to rule out: snapshot noise (I11), though 15 of 16 episodes across two states at one mark, and 0 of 8 at every other mark, is not what noise at one checkpoint looks like.
    2. **Reward moves to the level term.** With the per-map cap filled, the learner's P1 comes partly from battles near the start. Whether the catalogue's detectors flag this (G is the candidate type) is for the detector analysis; I do not name it an exploit from the term split alone.
  - **Asked of the developer for the round-1 dev analysis:** the usual catalogue and vs-R0 tables under P1, plus T per mark per seed with the 0.5M mark shown, and P1 by term per mark.

- **Round-1 closing read: r1-A1-D1-s3, closing dev eval done 04:10 UTC 2026-11-11. C passes; no ruling needed (researcher, 04:18 UTC). Dev view only; interim.**
  - **What the developer's read says.** P1 on D1 per mark: 10.00 (0), 2.35, 12.26 (0.5M), 5.46 (1M), 15.76 (2M), 8.59 (3M), 6.25 (4M), 9.73 (5M), all from the explore term. T per mark: 0, 0, 2.0 (0.5M, 8 of 8), 0 (1M), 2.25 (2M, 8 of 8), 0.88 (3M, 5 of 8), 0 (4M), 1.0 (5M, 4 of 8). On D2: T 1.75 at 2M, 0 at 5M.
  - **This weakens my flag 1 from s1's read.** s3 does not lose true progress after 0.5M: it has T above 0 at 0.5M, 2M, 3M and 5M, and 0 at 1M, 2.5M and 4M. So "progress appears early and is then lost" does not hold across the two seeds. What both seeds share is that T is above 0 at 0.5M. Within s3, the marks with high P1 are the marks with T above 0 (inference from 8 marks, one seed): under P1 on this seed, earning the capped reward and making progress go together, because new maps are where uncapped squares are.
  - **The interim reads sampled s3's zero marks.** Its 1M and 2.5M reads both showed T 0 and P1 at or below random; the marks either side (0.5M, 2M, 3M) do not. The "stall" picture from the staged reads was therefore partly an artifact of which checkpoints were read. This is the I11 snapshot swing again, and it means no round-1 statement should rest on a single mark.
  - **The seeds differ in kind.** s1's P1 is partly levelling and stays south of Viridian after 0.5M; s3's is all exploration and reaches the forest at several marks. The third seed matters for saying which is typical, and three seeds may still not settle it.

- **Round-1 trip 3: S-1 on r1-A1-D1-s2 at 1M, 04:31 UTC 2026-11-11. Ruling: r1, continue unchanged (researcher, 04:36 UTC; logged before any resume). s2 runs to 5M.**
  - **What tripped.** s2's own-state P1 at 1M is 5.16 (all explore), below its P1(0) of 9.34 (p 1.000) and below R0's 7.56 (p 0.999). S-3 notified: T 0, 0 of 8 reach M1. S-2 passes (0 of 8 never left). C passes (hash ok, sps 0.97×, re-score exact).
  - **Why r1.** C passes and this is one run tripping; that is I12's default. r3 is not open: its condition is two concurrently running first runs tripping at the same read, and s1 and s3 are finished. Nothing is queued behind s2, so the pause has nothing to hold; the developer may clear it.
  - **Not a measurement fault.** Same checks as trips 1 and 2. All three seeds now show the same thing at the 1M checkpoint: a frozen policy that stays on Pallet, Blue's house and Oak's Lab and earns less than random.
  - **What it is evidence of (inference).** Not a stall. s1 and s3 tripped the same way at 1M and both scored above R0 at 2M (19.76 and 15.76), and s3 had T 2.25 there. So a 1M trip has, twice out of twice, been followed by recovery. What needs explaining is why the 1M checkpoint is bad on 3 of 3 seeds when 0.5M and 2M are good on the two finished ones. Candidates, untested: a consistent phase of learning around 1M where the policy has filled the nearby caps and not yet found the next maps; or a snapshot-position effect like the one I4 tested for and did not find. I do not test this inside round 1; it is a descriptive note for the paper.
  - **S-1 as a tripwire.** It has fired on 3 of 3 runs at 1M and none was a fault or a lasting stall. For the record, it did its job (surfacing the 1M dip in hours), but its 1M reading is not informative about a run's later marks. I do not change it mid-round.

- **Round-1 read: r1-A1-D1-s2 at 2.5M, 05:38 UTC 2026-11-11. No new trip. Ruling (researcher, 05:42 UTC): pass; s2 continues to 5M.**
  - **Checks.** S-2 passes: 0 of 8 never left the start map, and P1 rose from 5.16 at 1M to 8.15. C passes (hash ok, obs v2, sps 0.975×, re-score exact). S-3 notifies: T 0, 0 of 8 reach M1. The S-1 line in the file is the ruled 1M trip re-listed.
  - **Descriptive, interim.** P1 at 2.5M is 1.08× R0's (one-sided p 0.239), so s2 is not above random at this mark; s1 was (p 0.0001) and s3 was not (p 0.971). No rule depends on this for s2. Leg progress rose to 0.648 from 0.225, with 7 of 8 episodes on Route 1 and none past it. As the s3 closing read showed, the 2.5M mark says little about the marks either side, so I read nothing further into it.
  - **Nothing for the lead or Jake.** The pause holds nothing; it can stay or be cleared.

- **Round-1 closing read: r1-A1-D1-s2, closing dev eval done 07:57 UTC 2026-11-11. C passes; no ruling needed (researcher, 08:04 UTC). Dev view only; interim until the dev analysis.**
  - **What the developer's read says.** P1 on D1 per mark: 9.34 (0), 6.50, 5.73 (0.5M), 5.16 (1M), 4.25 (2M), 12.75 (3M), 15.72 (4M), 12.42 (5M). T is 0 and M1 is 0 of 8 at every mark, on D1 and D2. From 0.5M to 2M the snapshots stay on three maps; from 3M they are on Route 1 and no further. At 4M the level term is 4.50 of 15.72.
  - **Three seeds, three patterns (descriptive).** s1: T above 0 only at 0.5M, then levelling near the start. s3: T above 0 at four marks, all reward from exploration. s2: T 0 throughout, below random until after 2M. The only things common to all three are the 1M dip and that P1 at 5M is above R0's (12.77, 9.73, 12.42 against 7.56; not yet tested).
  - **s2 does not show the 0.5M progress** that s1 and s3 share, so that is 2 of 3, not a feature of the round.

*I13. Unsealing and held-out results, batches 1 and 2 (researcher, 2026-11-10).*
- **Unsealing.** Condition met: the round-1 patch entry is in `patch-log.md` and on BOARD (22:20 UTC). The lead confirmed (Jake's OK) at ≈ 22:40 UTC. Before opening I ran both verifies at 22:38 UTC: batch 1, 2,788 files, sha256 `e36804d2…b5049816`, 0 changed / missing / new; sealed-b2, 3,102 files, `6bd2aca6…`, 0 changed / missing / new. I then opened `code/results/sealed-b1/b1/`, `code/results/sealed-b2/b2/` and the raw `~/roms/sealed-b2/` rows. The `dry-b1/` and `dry-b2/` folders stay unopened and excluded, as ruled. `sealed-r1/` does not exist yet and stays closed to me through round 1's patching decisions.
- **What the held-out states are.** HA: the same start as D1 with a different starter. HB: D1's party, starting at the Viridian Center door, so it starts at M1 and next to off-route maps. O on HB is outside the family for that reason (§9 B), descriptive only.
- **Batch 1 (no visited channel): held-out agrees with dev.** No hack type is present above R0 after Holm for A0 or A2″ on HA or HB (`vs-R0-*-heldout.md`). A0's P at 5M is 39.0–55.8 across the four panels; its T is 0.4–1.3.
- **Batch 2: the proxy transfers.** A0's P at 5M on held-out is 117.4 (D1→HA), 110.6 (D1→HB), 123.3 (D2→HA) and 103.0 (D2→HB), against 112.6 and 121.7 on its own states. There is no generalization gap in P.
- **Batch 2: the catalogue reproduces on held-out.** For A0, O is present after Holm on HA at 2M, 4M and 5M (family of 7, HB excluded), and D on HB at 1M (family of 14). These are the same two types as on dev, and nothing else is present. L is 0. For A2″, M is present (HB at 0.25M, HA at 4M); on dev A2″ had G and M.
  - O's pooled share of P on held-out is 0.286 (dev 0.178). **This number mixes HA and HB**, and R0 itself fires O in 71% of HB episodes, so it overstates O on held-out. I do not compare it with the dev share.
- **Batch 2: A0's T on held-out is like its T on dev.** At 5M: 1.45, 1.58, 1.98, 2.03 (own state: 1.43 and 1.95).
- **A0 against A2″ on held-out T (descriptive; no test was pre-registered here, and kept seeds number 2–5 per cell).** Seed means at 2M–5M, four panels:
  - **Filtered (collapsed A2″ seed-checkpoints dropped, judged on the training state):** A2″ is above A0 in 12 of 16 cells. At 5M: 2.38 vs 1.45, 2.06 vs 1.58, 2.67 vs 1.98, and 1.79 vs 2.03 (A0 higher on D2→HB).
  - **Unfiltered:** A2″ is above A0 in 8 of 16 cells, and at 5M in 1 of 4.
  - So held-out repeats the dev picture: whether the oracle beats the proxy-trained agent on T depends on the collapse filter.
- **HA is not simply harder (the §9 (g) difficulty reference; 3 seeds, descriptive).** A2″ trained on HA reaches T 0.83 at 1M and 2M, 1.83 at 3M, 1.67 at 4M and 2.58 at 5M on HA (per seed at 5M: 3, 3, 1.75; one seed is at 0 at 4M). Dev-trained kept A2″ seeds reach 2.38 and 2.67 on HA at 5M. A0 on HA is at 1.45 and 1.98. So an oracle trained on HA does about as well there as the dev-trained oracle does on transfer, and A0's lower T on HA is not explained by HA being harder.
- **The 3M dip appears on held-out too.** A0's P at 3M is 74.4, 70.3, 70.2 and 81.5, below 2M and 4M in all four panels. The dip belongs to the 3M checkpoints, not to the eval state.
- **Not yet produced (asked of the developer; neither changes a dev result):** (1) held-out attribution, A0 against dev-trained A2″ matched by training state × eval state × mark (§9 (g)); the held-out catalogue shows "n/a". (2) The HA-trained A2″ line in the Fig. 1 HA panel and its detector shares on HA. My numbers above for the HA-trained runs are my own means from the sealed episode rows.
- **The three missing pieces, now in (developer, 2026-11-10, `code/results/b2-heldout/`; I read the tables).**
  - **Held-out attribution: O and D are proxy-attributed on held-out for both training states** (`attr-heldout-D1.md`, `attr-heldout-D2.md`; A0 against the dev-trained A2″, matched by training state × eval state × mark, collapsed reference seed-checkpoints dropped). O is above A2″ after Holm on HA in 6 of 7 marks for D1-trained runs and 6 of 7 for D2-trained runs (family of 7). D is above A2″ after Holm on HB at 1M for both, and in further cells (D1: HA at 0.5M, 1M, 2M and HB at 2M; D2: HB at 3M). No type is a learner artifact. At late marks the reference has 2–4 seeds.
  - **O's share of P on HA alone is 0.181** (8,718.5 of 48,270.0 over 560 A0 episodes), against 0.082 for R0 on HA and 0.178 for A0 on dev. So on the held-out state where O is countable, it carries the same share of reward as on dev. This replaces the pooled 0.286, which I had flagged as inflated by HB.
  - **HA-trained A2″ (`ha-trained-a2pp.md`):** the developer's per-mark means match mine. With collapse judged on HA, s1 is flagged at 2M and s3 at 4M; the kept means are 1.25 at 2M and 2.5 at 4M, and 2.58 at 5M with all three seeds. The HA-trained runs show G and M above R0, as dev-trained A2″ does on dev, and no O or D.
  - The line for the Fig. 1 HA panel is not drawn yet; I have asked for it, after the A1 reads.
- **What we infer.** Held-out changes no conclusion from dev. The O3 "mixed" reading stands: the proxy is learned and transfers, the same two exploit types appear in new states, and A0's true progress neither falls nor clearly trails the oracle once collapsed oracle checkpoints are counted.

*I14. Round-1 dev analysis: family and one added table (researcher, 2026-11-11 05:27 UTC; written before I opened any round-1 analysis output, including `code/results/dry-r1/`, which I have not opened).*
- **Primary vs-R0 family: confirmed as 7 cells.** A1 evaluated on its own training state D1 at the 7 marks (0.25M, 0.5M, 1M, 2M, 3M, 4M, 5M), Holm per detector over those 7. This is batch 2's rule (own state × mark) applied to one training state; batch 2 had 14 because it had two. The 2.5M interim read is not a mark and is not in the family. D2 is a cross-state dev view: its own table, descriptive, not in the family.
- **R0 under P1 is the right reference for the primary table,** because the reward-reading detectors (L, K, HF, O, B, U) must score both sides under the same reward. Two consequences to state with the result: R0 fires K in 24 of 24 episodes under P1, so K cannot come out "above R0"; and R0 fires O in 0 of 24 under P1.
- **Added table, descriptive, no test family: the same detectors on A1's episodes re-scored under P0, against R0 under P0.** Reason: O's detector reads the reward, and the cap removes most of the reward inside O's windows by construction. So "O not present under P1" could mean the behavior stopped or only that the detector no longer sees it paid. The patch-loop question is about the behavior. Scoring A1's trajectories under P0 measures O with the round-0 yardstick, comparable with A0's round-0 shares. I add this before seeing any round-1 detector output; it is still an addition after the 1M, 2.5M and closing reads, and the paper lists it as such.
- **Attribution is n/a,** as the developer says: round 1 has no reference arm. **No pooled-seed test claims more than 3 seeds can carry;** round 1 stays descriptive.

*I15. Round 1 (A1 under P1): dev results and stopping-rule status (researcher, 2026-11-11 08:46 UTC; dev only, `sealed-r1/` not opened).*
- **Source.** `code/results/r1/` (developer, `r1_dev.py`): `checklist.md` (13 of 13 pass), `vs-R0-A1-primary.md`, `catalogue-fields.md`, `t-per-mark.md`, `p1-terms.md`, `cells.md`, and the I14 view `vs-R0-A1-P0-primary.md`, `p0-o-d-shares.md`. 3 seeds, D1-trained, 8 episodes per cell, 7 marks. Everything below is descriptive except the vs-R0 family of 7.
- **What the tables say.**
  - **True progress.** T on D1 at 5M: 0, 0 and 1.0 by seed (mean 0.33). Per seed over all marks: s1 above 0 only at 0.5M (2.13); s2 never; s3 at 0.5M (2.0), 2M (2.25), 3M (0.88) and 5M (1.0). R0 is 0. For comparison, A0 under P0 in batch 2 had T 1.43 on D1 at 5M (5 seeds).
  - **The patched reward.** P1 at 5M: 12.77, 12.42, 9.73 (pooled 11.64) against R0's 7.56. At 1M all three seeds are below R0 (pooled 5.99). The best marks are 2M for s1 and s3 and 4M for s2.
  - **The old reward.** A1's trajectories re-scored under P0 earn about 22 at 5M on every seed (22.5, 22.1, 21.7), below R0's 29.7 and far below A0's 112.6.
  - **Types present after Holm under P1 (family of 7): G, M, O.** G (grass grind) at 2M, 3M, 4M and 5M, on s1 and s2 only; pooled share of P1 0.223. M (menus) at 4M and 5M, each from one seed (s3 at 4M, s2 at 5M); share 0.042. O at 0.5M, 1M and 5M with window shares of 0.02–0.03; share of P1 0.139. D is 0 in every episode. L is 0.
  - **The I14 view (P0 yardstick).** O is not above R0 at any D1 mark (A1 window share 0.014–0.056, R0 0.036; p ≥ 0.11). G and M repeat. O's share of P0 pooled over marks is 0.174 with prevalence 0.41 (R0's prevalence is 0.46; A0's batch-2 share was 0.178).
- **What I infer.**
  1. **O under P1 is "present" by the pre-registered rule, and I report it as a detector artifact.** R0 fires O in 0 of 24 episodes under P1 because the cap removes the reward the detector reads, so any A1 window counts as above R0. On the P0 yardstick A1 visits side maps as often as random play does. The formal catalogue line stays "present"; the text says what it means.
  2. **The off-route exploration that A0 did above random is gone, and so is nearly everything else A0 did.** A1 covers less ground than random under the old reward and makes less true progress than A0. That is what an unselective patch predicts (patch-log, check (b)). Round 1 therefore does not show a patch that removes an exploit while keeping the task.
  3. **Reward moved to the level term on 2 of 3 seeds.** G, absent for A0 before the patch (0 of 560 batch-2 episodes; corrected 2026-11-11, I first wrote 0.002, which is batch 1's prevalence), carries 0.22 of P1. The patch log predicted the level term's relative weight would rise about sixfold. This is one exploit replacing another after a patch, on two seeds. Two limits: attribution is n/a (no reference arm), and the batch-2 oracle arm also showed G above R0, so grass grinding is not specific to a misspecified reward.
  4. **M rests on one seed per mark,** and R0 fires M in 29% of episodes. I give it little weight.
  5. **The seeds differ in kind,** as logged at the closing reads. s3 is the only seed with true progress after 0.5M and the only one with no G. On s3 the marks with high P1 are the marks with T above 0. With three seeds I cannot say which pattern is typical.
  6. **Alternative explanations.** (i) Sparse reward: total P1 is 0.17 of P0 on the same behavior, so A1's lower T may be a starved learner, not a response to the cap's shape; the design cannot separate these. (ii) Snapshot noise: T swings between 0 and above 2 across adjacent marks on s3, so the 5M figure alone is weak; the per-mark table is the result. (iii) Three seeds.
- **Stopping-rule status (§7).**
  - **(a) No new hack type for 2 consecutive rounds: not met.** Round 1 has G and M, which A0 did not have in round 0. O is not new.
  - **(b) Dev T within the seed-CI of A2″'s dev T: not met on the filtered reading chosen at round 0.** A1's D1 T at 5M is 0.33; the two kept A2″ seeds are at 1.75 and 2.62. On the unfiltered reading the interval is [0.28, 2.05] and 0.33 is inside it, so (b) would be met. This is the same fork as I10 (1), and it again decides the outcome.
  - **(c) R = 5: not met** (1 round done).
  - **So the rule, as read at round 0, says to run round 2,** with G as the patch target (rank 1 by share of P1).
- **My recommendation, which is the lead's and Jake's decision (cost and design): stop the patch loop after round 1, as a declared deviation from stopping rule (a).** Reasons: under P1 the agent already earns less of the old reward than random and makes less true progress than A0, so a second patch on top of P1 would be tested on a learner that is barely working, and its result could not separate "patched" from "starved" any better than round 1 did. The informative next step is a selective patch for O from P0, which is a new design, not round 2. The alternative is round 2 by the rule: patch G on P1, 3 seeds, about $4.4 and about 10 hours. I am choosing after seeing the data, so this is post hoc and the paper says so either way.
- **`sealed-r1/` stays closed** until that decision (stop, or the round-2 patch entry) is logged in `patch-log.md`.

*I16. Stop decision and round-1 held-out analysis specification (researcher, 2026-11-11 13:18 UTC; written before any `sealed-r1/` file is opened).*
- **Decision (Jake and the lead, ≈ 09:20 UTC):** the patch loop stops after round 1, as a declared post hoc deviation from stopping rule (a). Logged in `patch-log.md` at 13:16 UTC with the reasons from I15.
- **Unsealing.** Condition met by that entry. I ran `seal_r1.py verify` at 13:16 UTC: 402 files, sha256 `059ce53e…5f5749`, 0 changed, 0 missing, 0 new. Nothing in `sealed-r1/` was opened before this.
- **Analysis, fixed now (same rules as batch 2's held-out, §9 (g) and I13).**
  - Reference: R0's held-out base set (HA, HB) re-scored under P1 with the detectors re-run, as was done for dev.
  - vs-R0 under P1, Holm per detector: family of 14 (HA and HB × 7 marks) for every type except O; O's family is the 7 HA cells, and O on HB is descriptive only.
  - The I14 view on held-out: detectors on A1's HA and HB episodes re-scored under P0 against R0 under P0, with O's share of P0 on HA alone. Descriptive, no family.
  - T per mark per seed and P1 by term per mark, on HA and HB. HB starts at M1, so T on HB is not comparable with D1 or HA; report it apart.
  - Attribution n/a (no reference arm).
- **What held-out is for.** It checks whether I15's dev statements repeat on new states: lower T than A0, O not above R0 on the P0 yardstick, G present. It cannot change the stop decision, which is logged.

*I17. Round-1 held-out results (researcher, 2026-11-11 13:40 UTC).*
- **Source.** `code/results/r1-heldout/` (developer, `heldout_r1.py`; 14 of 14 checks; the manifest verified before reading). I read `checklist.md`, `t-per-mark.md`, `catalogue-heldout.md`, `vs-R0-A1-heldout.md`, `vs-R0-A1-P0-heldout.md`, `p0-o-d-shares.md`, `p1-terms.md`. 3 seeds, D1-trained, 8 episodes per cell.
- **What the tables say.**
  - **T on HA:** the same pattern as on D1, seed by seed. s1 above 0 only at 0.5M (2.0); s2 never; s3 at 0.5M (1.88), 2M (2.0), 3M (1.0) and 5M (0.25). Pooled at 5M: 0.08 (D1 0.33). A0, D1-trained, had 1.45 on HA at 5M in batch 2.
  - **T on HB (starts at M1; apart):** pooled 1.25 at 5M against R0's 1.0. s2 stays at 1.0 from 1M on.
  - **P1:** on HA 10.10 at 5M (R0 7.74). On HB 3.58 at 5M, below R0's 7.77 at every mark from 0.5M on.
  - **P0 re-scored at 5M:** HA 22.1, 21.2, 15.8 by seed (R0 31.5; A0 about 117). HB 8.0, 8.5, 12.3 (R0 39.6).
  - **Present after Holm under P1:** G (HA at 2M, 3M, 4M), M (HA at 4M and 5M, one seed each), O (HA at 0.5M, 1M, 5M; family of 7) and HF (HA at 2M, s1 alone, share 0.033). Nothing on HB. G, M and O are at the same marks and from the same seeds as on dev, except G at 5M, which is present on dev only. HF is new.
  - **The P0 yardstick on HA.** O's window share is above R0 at one mark of seven on raw p (0.5M: 0.061 against 0.025, p 0.0097, Holm 0.068) and not at the others. O's share of P0 pooled over marks is 0.172, with prevalence 0.36; R0 on HA has 0.082 with prevalence 0.33; A0 had 0.181. D is 0 on HA.
- **What I infer.**
  1. **Held-out repeats dev on the main points.** True progress is lower than A0's and near 0 at 5M; the old reward is below random; G appears on the same two seeds; O's "present" under P1 is the same detector artifact.
  2. **I sharpen what I15 said about O.** Two measures point in different directions and both are true. *How often* A1 is in a side map is at random play's level (window share, prevalence). *What share of its original-reward total* comes from there is 0.17, the same as A0's 0.18 and about twice random's 0.08 on HA. So the patch did not make the agent's exploration more on-route; it shrank all exploration to about random's amount and left its composition as it was. That is what the offline check predicted (O's share 0.178 → 0.150). "O is gone" in I15 was right for frequency and too strong as a general statement; `findings.md` and the paper now give both measures.
  3. **A1 does not transfer to HB.** It earns less than random there under either reward. A0 in batch 2 did transfer (P 110.6 on HB). Inference: a policy trained under the cap learns to fill the quota on the maps around its own start, and that does not carry to a start in another town. One training state, three seeds.
  4. **HF rests on one seed at one mark.** I list it and give it no weight.
- **No conclusion from dev changes.** The stop decision was logged before these were opened and is unaffected.
- **Method note (developer).** R0's saved U windows on held-out were pooled over D1, HA and HB; the P1 view pools U over HA and HB like A1's evals. U is never counted, so no result depends on it.

*I18. Corrections after the post's fact-check (researcher, 2026-11-11).*
- **G before the patch.** I15 and the paper said A0 did G in 0.2% of episodes in round 0. That is batch 1's prevalence (0.0018). The agent the patch was built on is batch 2's, which did G in 0 of 560 episodes. Corrected in I15, `catalogue.md`, `post-facts.md` and `paper.md`.
- **O share, like for like.** I15 and I17 compare A1's O share of P0 (D1-trained, 0.174 on D1 and 0.172 on HA) with A0's 0.178 and 0.181, which pool A0 over both training states. For D1-trained A0 on the same eval state: 0.169 on D1 (3,987.6 of 23,527.9; developer, checked against the seven D1 checkpoint means) and 0.166 on HA (4,098.8 of 24,624.1; my sum of the D1 rows of `b2-heldout/o-share-ha.md`). R0: 0.122 on D1 (86.9 of 712.5; my recomputation from R0's traces and events, matching the developer's) and 0.082 on HA. **The conclusion stands and is slightly cleaner:** A1's share equals D1-trained A0's (0.174 against 0.169; 0.172 against 0.166) and is above random's on both states.
- **How these were found.** Both came from checking the developer's `reward-sources` figure against the results files, at the lead's request. Neither changes a result; both were wrong comparisons of mine.

## 10. Figures

1. **Fig. 1 (G1):** mean proxy $P$ and true $T$ against training steps (and against KL), A0 vs A2, with seed CIs. Hack first-appearance ticks along the x-axis. This is the Gao-shape figure.
2. **Fig. 2 (G2):** dev $T$ and held-out $T$ against patch round, with the A2 and A3 reference lines. This is the H1 test.
3. **Fig. 3:** active hack types per round (stacked), with new types highlighted. This is the H2 test.
4. **Table 1:** the hack catalogue (§6).
5. Optional: a route map, coloured by visit density, for one hacked run vs one oracle run. The developer can render it from coordinates (RAM logs), so it needs no ROM-derived image.
6. **Per-run screenshot (for the blog post, per the brief's Decision):**
   - **Capture:** each run saves one screenshot, the final frame of its last D1 eval episode, to `~/roms/runs/<run_id>/<run_id>.png` (one folder per run, per Jake), **outside the repo**.
   - **Metadata:** in the same folder, `<run_id>.json` records arm, seed, patch round, training step, start state, `(map, x, y)`, $T$, $P$, the active hack detectors, and the reward-function hash.
   - **Post use:** a handful of curated frames (e.g. a hacked A0 run vs an A2 oracle run, or one frame per catalogued hack type) may be copied into the post's directory (`content/posts/<slug>/`) as illustration with commentary. **Jake approves which ones**, and the writer copies only those.
   - **Status:** screenshots are illustrations only and are never evidence for a claim. Every number in the post comes from the RAM logs.

Both SVG and monospace ASCII versions of figures 1–5 go to `code/figures/`, per the writing guide.

## 11. Threats to validity

- **Capability vs Goodhart.** Low $T$ can mean "too hard" rather than "hacked". Controls: the A2 oracle, plus H0's *non-monotonic* signature (T rises, then falls, while P keeps rising). A flat $T$ alone counts as weaker evidence.
- **Seed instability.** Pleines report large variance across runs. Use ≥5 seeds for G1, and report CIs and per-seed curves, not only means.
- **Detector bias.** Detectors are fixed in advance and their precision is measured by hand-check. U-type firings get reported separately.
- **Designer bias in G2.** The rulebook, the pre-registered patch log, and the blind held-out evaluation are there to limit it. The designer (the researcher) still knows the literature's hacks. Say so in the write-up.
- **Held-out A confound.** The starter changes both the hacks and the difficulty. Control: A2 on held-out A.
- **Generality.** One game, one segment, one algorithm. Claims stay at the level of "in this setting". The general point is carried by agreement with Pan, Karwowski and Gao [@pan2022effects; @karwowski2023goodhart; @gao2023scaling], not by this experiment alone.
- **Not adversarial Goodhart.** The agent has no model of the designer, so this is unintentional search, as in Gao et al. sec. 4.2.4. The write-up must not imply deliberate gaming.

## 12. Data hygiene (ROM rules from the brief)

- The ROM path comes only from `$POKEMON_RED_ROM` (e.g. `~/roms/`).
- Save states sit next to the ROM, outside the repo.
- Raw frame dumps, per-run screenshots and any videos or local review clips (§6) stay outside the repo, under `~/roms/runs/`.
- Gitignore `*.gb`, `*.gbc`, `*.state`, `*.sav`, and any `*.png`/`*.mp4`/`*.gif` under `code/`, as a backstop in case a dump lands there by mistake.
- Commit only: code, RAM-derived logs (coordinates, flags, rewards), figures generated from those logs, and the few curated frames Jake has approved, which go in the post's directory (§10.6) and nowhere else.

## 13. Handoffs

- **Developer:**
  - §3 readers and sanity checks.
  - §6 detectors as pure functions over RAM traces.
  - Measured $s$ and pilot M into `code/README.md`.
  - Fill in §8.
  - ~~Confirm 24 frames/action~~: confirmed as the harness default. Run `smoke_test.py --check-frames` once D1 exists, then freeze the value (§8).
- **Jake:**
  - Put the ROM at `$POKEMON_RED_ROM`.
  - ~~Make 4 save states~~ and ~~play the §9.1 trace~~: both are now scripted by the developer (§2, §9.1). Jake only approves any frames that go into the post.
- **Developer (added 2026-11-02):** scripted states plus the jitter/RNG check (§2), and the scripted §9.1 trace with screenshots per waypoint. Then ping me for the screenshot-vs-RAM review.
- **Researcher (me):** `patch-log.md` template and the G2 rulebook as a checklist. Also, after the pilot, revisit whether the optional `seen` term goes into $P_0$.
