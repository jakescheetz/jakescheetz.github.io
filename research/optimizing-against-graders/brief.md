# Optimizing against graders

Working title. The thread may narrow once the field map is done.

**Question:** When a model is optimized against an automated grader (a reward model, an LLM judge, a test suite, a monitor), how and when does it find the grader's blind spots? And which defenses actually close them, measured per unit of human labeling or compute?

**Why it matters:** Every current alignment pipeline puts optimization pressure on some learned or automated checker. If that checker has exploitable gaps, training amplifies them, whether or not anyone intends it. That covers reward hacking, overoptimization, and scalable oversight, and it is the natural follow-up to our reward-hacking post.

**Scope:**
- In: small, reproducible experiments we can run ourselves (open-weight models up to roughly 1–8B, best-of-n or light RL/DPO, small reward models and judges); measuring how proxy and true quality come apart; comparing defenses (bigger RMs, ensembles, iterated/adversarial RM retraining, KL budgets, monitors).
- Out: frontier-scale training; attacks on real deployed systems.

**Security angle (only where it's real):** The optimizer is an unintentional adversary searching the grader's input space, so "find inputs that break the checker, then patch and add regression tests" maps onto RM retraining. Grader poisoning (corrupted preference data) and judge manipulation are real threat models. We stay analytical: failure classes and mitigations, no working exploits against real systems.

**Candidate experiments (researcher to verify novelty and feasibility, then rank):**
1. *Patch-and-regress.* Does iteratively retraining a small RM on the current policy's top-scoring outputs delay overoptimization more than spending the same label budget on a bigger or ensembled RM? (Extends Gao et al. 2023 to a defense comparison.)
2. *Judge exploitability and transfer.* Use best-of-n or light optimization against one LLM judge, then measure how much of the gain transfers to other judges and to a held-out "gold" judge. Which surface features carry the exploit (length, confidence, formatting, flattery)?
3. *Test-gaming in code RL.* With reward = unit tests passing, how often does a small model learn to special-case the tests, and does a cheap monitor (a second model reading the diff) catch it?
4. *Preference-data poisoning.* How many corrupted comparisons does it take to plant a behavior that survives RL, and do the defenses from (1) remove it?

## Direction from Jake (2026-11-01): go small and fundamental

Jake prefers experiments that are **simple and fundamental**: Jupyter notebooks, or a local video game, used as an abstraction that proves a general point about optimizing against a proxy. He suggested **a Pokémon ROM** as a fun testbed. This is now the **primary track**. The four LLM candidates above become secondary or later.

Constraints: this laptop has 4 CPU cores, 8 GB RAM, and no GPU. Prefer things that run here (CPU RL, small envs, notebooks), or that need only a cheap rented box.

Game-track ideas (researcher to verify prior work and sharpen; developer to check feasibility):
- **G1. Shaped reward vs real progress in Pokémon Red/Blue** (e.g. via the PyBoy emulator). Train an RL agent on a hand-designed proxy (exploration, levels, HP, items), then measure *true* progress (badges, story flags read from RAM) as optimization strength grows. Catalogue the hacks the agent finds. Known prior work to check: Peter Whidden's PokemonRedExperiments, and LLM agents playing Pokémon. What would be new: treating it as a Goodhart/overoptimization measurement (proxy vs true curves, à la Gao et al.), not just "get further in the game".
- **G2. Patch-and-retest in the game.** Every time the agent finds a hack, patch the reward and retrain. Do new hacks keep appearing? How fast, and how far does true progress get per patch? This is the fuzzing lens, applied loosely.
- **G3. Learned reward model in the game.** Replace the hand-written proxy with a small reward model trained on "human" (scripted) preferences over short clips. That reproduces RLHF's structure cheaply, and the reward model can be overoptimized.
- **G4. Notebook-scale gridworld versions** of the same points (CoastRunners-style loops, a tampering cell, a proxy that diverges off-distribution). These run in seconds, make good post figures, and serve as a fallback if the ROM is too slow.

ROM note: Pokémon ROMs are copyrighted. Jake would supply his own legally dumped ROM. We never download one, commit one, or redistribute one; `.gitignore` it. Prefer a design where an open-source homebrew Game Boy game could stand in.

## Decision (Jake, 2026-11-01): Pokémon first

E1 = **G1 (+ G2 patch loop) in Pokémon Red**, scoped to an early-game segment from a save state. Notebook gridworld (G4) is optional support for post figures, not a gate. Crafter is the fallback only if the ROM path fails.

- ROM: Jake's own dump, kept **outside the repo** (e.g. `~/roms/`), passed via env var `POKEMON_RED_ROM`. No ROM, save state derived from it, or screenshot dump gets committed. **Screenshots (Jake, 2026-11-01):** capture one screenshot per run, saved outside the repo in `~/roms/runs/`, with run metadata. For the post, a handful of curated frames may be copied into the post dir as illustration and commentary, and Jake approves which ones.
- Design must-haves (from findings.md): true metric (badges/story flags from RAM) excluded from the proxy; oracle arm trained on true progress; optimization strength axis (training steps, n envs) with ≥5 seeds; hack catalogue timestamped by first appearance; held-out start states for the patch loop.

**Done when:** one experiment picked with Jake, run, written up as `paper.md` with figures and verified citations, plus a blog post and a build-log post.

**Outputs:** findings.md (field map) · code/README.md (feasibility) · paper.md (white paper) · blog post(s) · code/figures

**Open for Jake:** compute budget (default assumption: one consumer GPU, or about $100 of cloud).

## Decision (Jake, 2026-11-11): two versions of the E1 write-up, website first

- **Website post (now).** Conversational, storytelling, in Jake's voice (tone reference: his published post `content/posts/your-reward-model-is-an-attack-surface/`). Tell it as a story: the question, the setup, what the agent did, the twists (batch 1 learned nothing, the visited-squares fix, the patch that backfired, grass grinding). **Shortcomings go in one honest section near the end**, not as caveats on every paragraph; keep only the hedges a claim needs to stay true. Numbers come from `paper.md` / `findings.md` and stay exact; no new claims.
- **Website graphics.** Easy to read, game-styled (Game Boy-ish palette, pixel feel), drawn from RAM logs and our own numbers, not from ROM art or sprites. Candidates: a schematic of the route with milestones; "score vs real progress" over training; where agents went (visit heatmaps A0 / patched A1 / random); where the reward came from before and after the patch; round-1 seeds as small multiples. Curated game screenshots only if Jake approves specific frames (rule above).
- **Paper (after).** `paper.md` keeps the academic tone for an arXiv-style version. Its charts get redone later: easy to understand but academically conventional (axes, CIs, captions). Not started yet.
- Post stays `draft: true`; only Jake publishes.
