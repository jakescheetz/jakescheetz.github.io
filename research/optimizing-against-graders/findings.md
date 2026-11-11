# Findings: field map for "optimizing against graders"

_Researcher, 2026-11-01. Centered on the game/notebook track (G1–G4) per Jake's "go small and fundamental" direction; the LLM candidates (1–4) are ranked at the end as secondary. Sources in `sources.bib` (all opened; full-text vs abstract-only is marked there), details in `notes.md`. Confidence: **high** = directly shown in multiple primary sources; **medium** = shown once or in a narrow setting; **low** = our inference._

## Bottom line

| Rank | Idea | Novelty | Runs on the laptop? | Why this rank |
|---|---|---|---|---|
| **1** | **G2 patch-and-retest, built first on a G4 notebook world, then ported to the game** | Medium–high | Notebook: yes (minutes). Game: only an early-game segment | The one question the literature hasn't answered: when you patch each hack the optimizer finds, does true performance converge, or do you overfit the patches to what you looked at? |
| 2 | G1 shaped proxy vs real progress in Pokémon (or Crafter) | Medium | Early-game segment: probably; full game: no | The method is known (Pan 2022, Gao 2023). What's new is the setting: Gao-style curves plus a hack catalogue indexed by optimization strength. Best showcase/post value |
| 3 | G4 standalone gridworld demos (loop, tampering, off-distribution proxy) | Low | Yes (seconds) | Already exists (AI Safety Gridworlds, Karwowski 2023). Do it anyway as the substrate and figures for #1, not as a result |
| 4 | G3 learned reward model in the game | Low–medium | No (labels + RM on frames + RL) | Christiano 2017 and Ibarz 2018 already showed frozen learned RMs being hacked in Atari. Only worth it later, as a scaling study |

**Recommendation:** run **E1** (below) as the first experiment: G2 inside a G4-style "mini-Pokémon" gridworld, with a held-out true metric. If it shows something, port the same protocol to a Pokémon early-game segment (G1 + G2) as the showcase. Crafter is a ROM-free fallback for the game step. The developer's feasibility numbers for the game track are still pending in `code/README.md`, so the game-step ranking may move once they land.

---

## State of knowledge

**F1. Optimizing a misspecified proxy harder produces higher proxy and lower true reward, often non-monotonically, across many RL settings.** *High.*
Pan et al. vary optimization power (model size, training time, action resolution) across four environments and nine proxies and find "more capable agents often exploit reward misspecifications", including sudden "phase transitions" [@pan2022effects]. Karwowski et al. find a Goodhart drop in 19.3% of 30,400 sampled tabular settings, so it is "common (albeit not universal)" [@karwowski2023goodhart]. Gao et al. find the same rise-then-fall for learned RMs in RLHF [@gao2023scaling]. *Implication:* "proxy diverges with optimization" is not a finding on its own. A new experiment has to say something about **when**, **how much**, or **what fixes it**.

**F2. Pokémon Red reward shaping produces concrete, nameable hacks, but no one has measured them as overoptimization.** *High (that the hacks exist); medium (that no one has measured them, based on what I found).*
Pleines et al. (with Whidden) document a heal-reward loop (endless Zubat battles sustained by Leech Seed/Leech Life), a starter chosen "simply because its reward is triggered sooner", a navigation weight that blocks beating Brock, routine skipping of a mandatory badge, and a level reward whose removal "yields the best ablation performance" while it "introduces side effects as well" [@pleines2025pokemon, sec. IV-B4] *(wording tightened 2026-11-10 on re-reading)*. PokeRL reports loop episodes in 41.2% of a 1,000-episode baseline sample, cut to 4.7% by a three-layer anti-loop system whose penalty values were settled "after several iterations" [@mudireddy2026pokerl, sec. V-A, VIII-C]. *(Corrected 2026-11-10 on re-reading: the paper does not call the sample "early-game".)* Both are qualitative. Neither plots proxy against true progress as optimization grows, and neither holds out a true metric. I found no paper doing Gao-style measurement in Pokémon (searched 2026-11-01).

**F3. Learned reward models get hacked in games when frozen, and online relabeling fixes the specific loopholes.** *High.*
Christiano et al.'s offline-RM ablation gives a Pong agent that avoids losing but never scores [@christiano2017deep, sec. 3.3]. Ibarz et al. freeze a trained RM in Atari: "perceived performance … increases, while the actual performance … plummets" in three games. Online labeling fixes loopholes "similar to adversarial training" [@ibarz2018reward, sec. 3.2, fig. 4]. This is G3's result, already in hand.

**F4. Patching rewards by trial and error overfits them.** *Medium (one paper, small domains).*
Booth et al. show dense reward functions get "overfit to learning algorithms and their hyperparameters", both computationally and in a controlled study of expert designers, and that myopic per-state reward design produces "invalid task specifications" [@booth2023perils]. Theory points the same way: unhackable proxies are essentially trivial over all stochastic policies [@skalse2022defining], and a proxy that omits attributes can lose unbounded utility [@zhuang2021consequences]. *Inference:* patching may never converge, and the patcher's own evaluation becomes a second proxy.

**F5. Iterating the grader helps but plateaus; this is measured for LLM RMs, not for hand-designed game rewards.** *Medium.*
In AlpacaFarm, overoptimisation falls over RLHF iterations with diminishing gains [@wolf2025reward]. Gao et al.'s model-based analysis says iteration helps extremal Goodhart but not regressional Goodhart [@gao2023scaling, sec. 4.3]. Bai et al. iterate because PMs are miscalibrated at high scores [@bai2022training, sec. 4.5]. Nothing comparable exists for *patching a hand-designed reward*, which is G2.

**F6. There are principled alternatives to patching that could serve as baselines.** *Medium.*
- Early stopping with a provable rule [@karwowski2023goodhart].
- Occupancy-measure (χ²) regularization to a reference policy, argued to beat action-KL [@laidlaw2025correlated].
- Myopic optimization with non-myopic approval for multi-step hacks, including gridworld sensor tampering [@farquhar2025mona].
- Design principles that remove tampering incentives [@everitt2021reward].

Patching is what practitioners actually do (PokeRL, Whidden's V2 exploration reward), yet nobody has compared it with these at equal designer effort.

**F7. The standard toy environments already exist.** *High.*
AI Safety Gridworlds has a CoastRunners-style boat race and a tomato-watering env where the agent can blind its own sensor with a bucket, each with a hidden performance function [@leike2017ai]. The anecdote catalogues are large [@clark2016faulty; @krakovna2020specification; @lehman2018surprising]. G4 as "demonstrate the known failure modes" is reproduction, which is fine for posts but not a contribution.

**F8. Off-distribution failure has two different explanations, and they must be kept apart.** *High.*
Inverse reward design frames proxies as valid only in the scenarios they were designed for [@hadfieldmenell2017inverse]. Goal misgeneralization is a capable agent pursuing the wrong goal out of distribution even with a correct training reward [@langosco2022goal]. A G1/G4 "proxy diverges off-distribution" result has to say which one it is.

## Gaps (where a small experiment can say something new)

1. **Patch dynamics** (G2). When each discovered hack is patched and the agent retrained, how does true performance evolve per patch? Do new hack types keep appearing? How big is the gap between the true metric the patcher watched and one it didn't (Booth-style overfitting)? *Not measured anywhere I found.*
2. **Patching vs principled defenses at equal designer budget.** Patching vs early stopping vs occupancy regularization vs MONA, with budget = number of true-reward evaluations or inspections the designer gets. *Not found.*
3. **Quantitative overoptimization curves in a long-horizon game with natural progress flags** (G1). Pokémon (or Crafter) gives a "true" metric (badges, story flags, achievements) that is cleanly separable from the shaped proxy. The hacks are already known qualitatively (F2).
4. *(Lower priority)* Gao-style RM-size/data scaling for a learned RM in a game (G3). Atari shows the effect exists (F3); a scaling law there would be new, but it is expensive and overlaps the LLM track.

## The game-track ideas in detail

### G2: Patch-and-retest. **Rank 1** (run on a G4 substrate first)
- **Prior work:** Booth 2023 (designer overfitting), Ibarz 2018 (online labels as patches), PokeRL and Whidden V2 (informal patching), Wolf 2025 (the LLM analogue), Skalse 2022 and Zhuang 2021 (theory predicting non-convergence).
- **Novel:** measuring patch dynamics with a held-out true metric, and comparing patching to principled defenses per unit of designer budget.
- **Why first in a notebook:** each patch needs a retrain. In Pokémon a single full-scale run took about 36 h on a cluster [@pleines2025pokemon]; in a gridworld a retrain takes seconds, so we can afford 10+ seeds × 10+ patch rounds × several arms.
- **Security lens (where it helps):** this is the regression-test loop from security work. The policy is an unintentional fuzzer, each patch is a fix, and Booth's result is the familiar failure of fixing to the test suite. The held-out metric plays the independent red team.

### G1: Shaped proxy vs real progress. **Rank 2**
- **Prior work:** Pleines 2025, PokeRL 2026, and Whidden's repo (Pokémon); Pan 2022 (the method); Gao 2023 (the curve shape).
- **Novel:** proxy-vs-true curves against optimization strength (training steps, policy size, and number of envs; ≥5 seeds), with each hack timestamped when it first appears. Pan-style "phase transitions" would be the interesting result.
- **Design must-haves:**
  - The true metric (badges and story flags from RAM) must be **removed from the proxy**. Pleines' event reward overlaps with truth (F2).
  - Include an **oracle arm** trained on true progress at the same budget, to separate "proxy hacked" from "task too hard" (low true reward can just be a capability failure).
  - Scope to an early-game segment from a save state; PokeRL saw loops within ~10^5–10^6 steps.
- **Risks:** seed instability (Pleines report large variance); CPU throughput on 4 cores (developer to measure); ROM handling (Jake's own dump, gitignored).
- **Crafter alternative:** open source, achievements built in as the true metric [@hafner2022crafter]. It costs some "fun" but gives better science hygiene.

### G4: Notebook gridworlds. **Rank 3** as a standalone; essential as the substrate for #1
- **Prior work:** AI Safety Gridworlds (boat-race loop, tomato tampering), Karwowski 2023 (Goodhart curves in tabular MDPs), IRD (off-distribution proxy).
- **Novel:** nothing on its own. Value: post figures in seconds, and the testbed for E1. Reuse or reimplement the Leike environments and cite them.

### G3: Learned RM in the game. **Rank 4**
- **Prior work:** Christiano 2017 and Ibarz 2018 did this in Atari and showed the frozen-vs-online contrast.
- **Novel only if** it adds RM-size/label-budget scaling, which turns it into LLM candidate 1 in game form.
- **Cheap variant worth keeping:** "G3-lite" in the E1 gridworld, with a tiny RM trained on scripted trajectory preferences. That gives a Gao-shaped figure for the post in minutes. It is not a contribution.

---

## Proposed experiment E1 (for the developer): patch-and-retest in a mini-Pokémon gridworld

**Question.** When a designer repeatedly patches the hacks an RL agent finds in a shaped reward, does true performance converge? Does the patched reward generalize to conditions the designer never inspected? How does patching compare with principled defenses at equal designer budget?

**Environment.** A procedurally generated gridworld that mirrors Pokémon's reward structure from Pleines et al.:
- rooms and corridors;
- "grass" tiles with respawning encounters that give XP (the level proxy);
- a heal tile (the heal proxy);
- tiles that count as exploration (the navigation proxy);
- ordered "badge" flags (the true reward);
- a few planted loophole types: a respawn loop (CoastRunners), a heal–damage cycle (the Zubat loop), a reward that fires sooner on a worse branch (the starter bias), and an optional sensor-tampering tile (tomato watering).

Maps come from a seed; there are separate **dev** seeds and **held-out** seeds. The proxy is a weighted sum of the shaped terms, with no badge term. The true reward is badges collected (plus optionally steps-to-badge).

**Agent and optimization strength.**
- (a) Tabular or small-MLP Q-learning/PPO, with optimization strength = training steps.
- (b) For noise-free curves, soft-optimal policies $\pi_\beta \propto \exp(\beta Q_{\text{proxy}})$ computed exactly by value iteration, with optimization strength = $\beta$. Report occupancy-KL from a reference (random or "novice") policy as the x-axis, à la Gao and Karwowski.

**Patch loop (automated designer).**
1. Train on the current proxy.
2. On **dev** seeds only, find the proxy term or state–action features with the largest (proxy − true) contribution on the policy's occupancy. This is an operational hack detector in the spirit of Laidlaw et al.'s correlation-breakdown definition.
3. Patch: down-weight that term, cap it, or add a penalty, as a human designer would.
4. Retrain and repeat for R rounds.

The designer's **budget** is the number of true-reward evaluations or inspections used.

**Arms, at matched designer budget:**
1. Patching.
2. Early stopping with Karwowski's rule.
3. Occupancy / χ² or KL regularization to the reference policy (Laidlaw).
4. MONA, for the tampering loophole only.
5. Oracle: train on the true reward.
6. No defense.

**Measurements, per round and arm, ≥10 seeds:** true reward on dev seeds; true reward on held-out seeds; held-out under a *different learning algorithm* (Booth's overfitting axis); number of distinct hack types active (from loop and occupancy signatures); proxy–true correlation on visited states.

**Hypotheses (falsifiable):**
- **H1:** patching raises dev true reward round over round, but held-out true reward plateaus lower. The dev–held-out gap grows with rounds. *Falsified if* the gap stays within seed noise.
- **H2:** each patch tends to expose a new hack type while the proxy omits the badge term. The number of hack types stays above zero until the omitted feature is effectively added back. *Falsified if* hacks die out after the planted loopholes are patched, i.e. patching converges.
- **H3** (*speculative*): at small designer budgets, the regularization and early-stopping arms beat patching on held-out true reward; patching catches up only with large budgets.

**Alternative explanations to control:**
- Capability vs Goodhart: the oracle arm.
- Seed variance: ≥10 seeds, with confidence intervals.
- Detector subjectivity: fixed operational definitions, plus a hand-checked sample of rollouts.
- Planted-loophole artifact: also run maps whose loopholes are not hand-planted but arise from random proxy weights (Karwowski's random-proxy protocol).

**Compute (my estimate; developer to confirm):** tabular and value-iteration arms take seconds to minutes per run on CPU, so the full grid fits on the laptop in hours.

**Outputs:**
- Fig. 1: true vs proxy against optimization strength. This is the Gao-shape figure for the post.
- Fig. 2: dev vs held-out true reward against patch round, per arm.
- Fig. 3: hack types per round.
- A hack catalogue table.

**Phase 2 (if E1 is interesting):** port the same patch loop to a Pokémon early-game segment (G1 + G2) or to Crafter. Use 3–5 patch rounds, with badges and story flags as true reward and a held-out start state as the held-out condition.

---

## Secondary: the LLM candidates, re-ranked

All need a GPU box or cloud (per the developer's table), which is now a mark against them under the new direction.

| Rank | Candidate | Novelty after this search | Feasibility | Note |
|---|---|---|---|---|
| 1 | (1) Patch-and-regress for RMs | Medium. Wolf 2025 studies iterated-RLHF overoptimisation; Coste and Eisenstein cover ensembles. Still open: a **label-budget-matched** comparison of iterated vs bigger vs ensembled RMs | BoN on one GPU, 20–40 GPU-h | The LLM version of E1. Do it after E1 if E1 is interesting |
| 2 | (2) Judge exploitability and transfer | Low. Cross-family transfer of judge hacks is reported [@zhou2026convincing]; gold-judge setups show hacking "regardless of judge size" [@liu2026examining]; null-model and master-key exploits [@zheng2025cheating; @zhao2025onetoken] | Easiest: BoN only, 10–25 GPU-h | Fine as a small replication post; weak as a paper |
| 3 | (4) Preference-data poisoning | Low–medium. Attacks are well covered [@rando2024universal; @baumgartner2024bestofvenom; @wu2024preference; @souly2025poisoning]. "Do iterated-RM defenses remove a planted preference?" may be open (not verified) | 30–60 GPU-h; RL part on cloud | Security-flavored. Keep the marker behavior benign |
| 4 | (3) Test-gaming in code RL | Low. Crowded: [@zhong2025impossiblebench; @gabor2025evilgenie; @wang2026catch; @baker2025monitoring] | Hardest: 50–150 GPU-h, discovery risk | Drop unless a specific gap appears |

## E1 batch 1 (Pokémon Red, round 0): dev results, 2026-11-07

All 23 runs; dev states D1/D2 only; held-out still sealed. Protocol and numbers are in `experiment-e1.md` §9 G, Table 1 in `catalogue.md`, and tables in `code/results/b1/`. These are our own results, not a source claim.

- **F-E1.1. The proxy misranks policies. Confidence: high, on these two dev states.**
  - Pooled over marks, P0 ranks A0 (42.9 per episode) > random R0 (30.4) > the truth-optimizing A2″ (15.3).
  - T ranks A2″ > A0 > R0 at late marks. At 5M, A2″ beats A0 on D1 (2.69 vs 1.08) and on D2 (2.13 vs 0.40), Holm p ≈ 4e-5 on each.
  - About 99.7% of P is the new-tile term, so "doing the task" (walking straight) earns less proxy than wandering.
- **F-E1.2. H0, the Gao-shape over-optimization curve, is inconclusive on both training states. Confidence: high that it's inconclusive; it says little about H0 itself.**
  - A0's P is already flat by the first eval mark (0.25M), so neither "P rises" nor "T falls as P rises" can be observed at these marks.
- **F-E1.3. No pre-registered hack type is present above random. Confidence: medium, because the null is likely a capability limit.**
  - None is present in any primary cell after Holm, and the U review found no new mechanism. So the G2 patch loop has nothing eligible to patch.
  - *Most likely alternative (inference):* the policy has no visited-tile memory in its observation, so it can't optimize a new-tile reward well. A0 earns only ≈ 1.4× random, which fits this. A planner diagnostic that tells "A0 is too weak to find the hack" from "the proxy isn't exploitable" is proposed in §9 G5.
- **F-E1.4. P0 is far from saturated, and its optimum is aligned with T on this segment. Confidence: high for "attainable"; whether it's learnable is open.**
  - A pure-coverage planner (perfect memory and map knowledge) earns P ≥ 417 on both dev states, 7.5–8.3× A0's best. It also reaches **T = 5 in 16/16 episodes**, beyond the oracle A2″.
  - A0 climbs about 6% of the way from its init P (≈ 31, about random) to that level, all within ≈ 0.5–1M steps (exploratory 0-step mark).
  - *Inference:* F-E1.1's misranking is a weak-optimizer effect. Batch 1's null is most likely a capability limit, so F-E1.3's confidence that it is one rises to medium-high.
  - It's open whether a stronger learner shows Goodhart or simply makes progress (`experiment-e1.md` §9 G5 result).
- **What the A0 pilot already showed, and we missed (A0-D1-s0, 2026-11-03).** I re-checked this on 2026-11-07 from its D1 eval traces (`~/roms/runs/A0-D1-s0/eval-final/`, 8 episodes per mark):
  - **P was explore-only at every mark.** Level 0.0 and heal ≤ 0.1 throughout.
  - **P was flat from 500k.** Eval P went 16.9 (0.25M) → 44.4 (0.5M) → 54.1 → 72.0 → 67.3 → 49.1 (5M), with no upward trend after 0.5M.
  - **T was stuck at 1.0 (Viridian) in every D1 episode from 0.5M to 5M.** Training-episode P was also flat from 0.5M.
  - **These are batch 1's signatures:** a proxy that counts tiles, an early plateau, and a learner too weak to exploit its own reward.
  - **Why we missed it:** the pilot gate checked only the oracle's capability (the A2 gate), plus "A0 reaches M2 and some detector fires". We compared detector shares with R0, but **never P itself** against R0 (≈ 30) or an attainable ceiling, and **never P by term**.
  - In the §9 pilot write-up I read "T flat while P rises and then falls back" as a possible over-optimization sign. In fact it fit a weak optimizer better.
  - **Lesson, now in batch 2 (§9 H4′):** check that each arm optimizes its own reward early, using P by term, P against R0 and against the planner, at 1M and 2.5M.
- **Security lens (brief):** a grader that pays coverage rewards wandering over task completion. A capable optimizer would be expected to exploit that gap. Batch 1 shows the gap exists (F-E1.1), but not that this learner exploits it.

## Open questions

- **E1 round 1 (for the lead/Jake):** with no eligible type, A1 round 1 is on hold. Options (A) stop / (B) cap the explore term / (C) fix the observation and re-run, after the free planner diagnostic, are in `experiment-e1.md` §9 G6.

- **For Jake:** E1 first (notebook, laptop, most novel), or straight to Pokémon G1 (most fun, slower, less novel)? And is Crafter acceptable as the game if the ROM path is too slow?
- **For the developer:** PyBoy and Crafter steps/s with a policy in the loop on 4 cores. Can a Pokémon early-game segment (save state → Viridian/Pewter) reach the PokeRL-style loop hacks within ~1–5M steps?
- **Unverified, would change the ranking:** whether anyone has already published patch-round dynamics with a held-out metric outside Booth 2023. I ran targeted searches and found nothing; a citation search on Booth 2023 would settle it.
- **Scientific:** in E1, is "the patcher's evaluation becomes a second proxy" (H1) distinguishable from ordinary overfitting to the dev maps? The different-algorithm held-out condition is meant to separate them. Check whether it does.

### Open question (2026-11-08, from E1 batch-2 trip diagnostics)

- **Snapshot vs learning process.** In E1, A0's training-episode P (≈ 117 at 1M) is 2–3× the P of the same checkpoint frozen (≈ 50). A direct test confirms the cause is learning within the episode: PPO resumed from that checkpoint gives P 131 in its first episodes (§9 H9). Goodhart-style results are usually stated about a fixed policy. Here, though, much of the proxy is earned by the optimizer acting online. Does over-optimization of P against T show up in the learning process (training episodes) when it doesn't in snapshots, or the reverse? Batch 2's primary tests stay on snapshots, as pre-registered. Training-episode P and T are reported descriptively. Confidence: the mechanism is high (direct test); the implication for H0 is untested.
- **Batch-1 learning-process series (§9 H10′, descriptive, seed-level CIs).** Training-episode P exceeds snapshot P at every mark (ratio 1.2–2.6). On D2 the learning process shows a proxy rise from 0.5M to 5M (+17.9, CI [6, 27]) that snapshots don't (−10.6, CI [−41, 24]), with T trailing A2″ at 3–4M, so it's labelled "learning-process divergence". On D1 the two series agree, because neither shows a robust rise. The H0 mirror on the training series is still inconclusive on both states. Confidence: low. It's one state of two, 5 seeds of a weak learner, and P is explore-only, so this most likely means more of the same proxy, not a new mechanism. Batch 2 tests whether it recurs.
- **Oracle instability: parked endings (2026-11-10, §9 H11).** In batch 2, three of ten A2″ own-state runs end at 5M with the agent never leaving the start map (8 of 8 episodes, P ≤ 1.7, T 0), after reaching T 2–3 at 3–4M. The developer reproduced one bit for bit on two other machines and ruled out an eval or host cause, so it is a property of the learned weights. Batch 1 never shows this after a run has reached T ≥ 1. What we don't know is the cause (candidates, untested: late-training policy drift under the progress-bonus shaping (A2″ is not potential-based; wording corrected 2026-11-11), or an interaction with the visited channel) or whether training past 5M would recover. Consequence: the "oracle" reference is a policy that works at some checkpoints, and results that lean on A2″ at 5M carry that caveat. Confidence in the observation: high. In any cause: none yet.

## E1 batch 2 (visited-tile observation), dev only, 2026-11-10

Source for all of these: our own runs, `experiment-e1.md` §9 I1–I10 and `code/results/b2/`. Held-out results are still sealed.

- **F-E2.1. The observation fix made A0 a capable optimizer of its proxy.** Pooled best-mark P is 120.9 (D1) and 121.7 (D2), against 55.5 and 50.5 in batch 1; the pre-registered manipulation check passes. Confidence: high.
- **F-E2.2. H0 is inconclusive on both states (outcome O3), and the reason is that T rises with P.** P rises over training (i), but A0's T at 5M is higher than at the mark where P reached half its final value, so (iii) fails. Training-episode P and T both rise at every mark. Confidence in the verdict: high (pre-registered rule). In the reading "P0 is aligned on this segment once the learner is capable": medium, since the formal falsifier is not met.
- **F-E2.3. "A0 trails the oracle" depends on the collapse filter.** The pre-registered (ii) holds at 7 of 8 late cells, but on D1 it holds at no mark once collapsed A2″ checkpoints are kept, and at 5M A0 is at or above the unfiltered A2″ on both states. Confidence: high (both lines computed).
- **F-E2.4. Mild, proxy-attributed reward hacking appears: off-route exploration (O) carries about 18% of P, 24% at 5M; Route 22 (D) about 2%.** It coexists with rising T. Confidence: medium-high for O (8 of 10 seeds, 5 cells after Holm); low for D (one cell, 3 seeds).
- **F-E2.5. Snapshot evals depend on the phase of the training-episode cycle: tested, not supported.** A0's snapshot P dips at 3M in 8 of 10 seeds while training-episode P does not; 3M sits at 89% of the 131,072-step cycle. The check on 20 rolling checkpoints, with predictions written first (§9 I11), failed: late-phase checkpoints were not low (0 of 3 at or under 0.7 of the run's 4M/5M mean), early-phase ones were not reliably high (2 of 4 at or over 0.8), and the rank correlation of ratio with phase is +0.19. The 3M dip is unexplained. The table does show that snapshot P swings widely between nearby checkpoints of one run (0.40 to 1.25 of the 4M/5M mean). Confidence that phase is not the explanation: medium (3 late checkpoints, from 2 D2 runs; none on D1).
- **What I infer, beyond the data:** on a task where the proxy's optimum is aligned (G5), a stronger optimizer closes the proxy-true gap. Batch 1's misranking of A0 and A2″ by P was a property of a weak learner, and the Goodhart shape needs a regime where exploration and progress compete, which this segment and budget may not reach.
- **Security lens:** the detectable "hack" here is reward earned off the intended path, and it is small and benign. The larger oversight risk this batch shows is in measurement: a checkpoint-based audit gives different answers depending on when the checkpoint was taken, and the reference policy used to judge the agent was itself unstable.

### Open questions (batch 2)

- What causes A0's snapshot dip at 3M, now that training-cycle phase is ruled out as far as the rolling checkpoints can say (§9 I11)? And how large is checkpoint-to-checkpoint noise in snapshot P relative to the between-arm effects?
- Which reading of stopping rule (b) applies, and does round 1 (patching O) run (§9 I10)?
- Would a longer horizon or a later segment produce divergence, given A0 covers only ≈ 27% of the planner's squares at 5M?


## E1 held-out results, batches 1 and 2 (unsealed 2026-11-10, after the round-1 patch entry)

Source: `experiment-e1.md` §9 I13; `code/results/sealed-b1/b1/`, `code/results/sealed-b2/b2/`. Held-out states: HA (different starter), HB (later start, at M1).

- **F-H.1. The proxy transfers to held-out states.** Batch-2 A0's P at 5M is 103.0–123.3 on HA/HB, against 112.6 and 121.7 on its own states. Confidence: high (10 seeds, 80 episodes per cell).
- **F-H.2. The same exploit types reappear on held-out, and no others.** Batch 2: O above R0 after Holm on HA at 2M, 4M and 5M; D on HB at 1M. Batch 1: none on dev, none on held-out. Confidence: medium-high for O (3 of 7 cells; HB excluded by design), low-medium for D (1 of 14 cells).
- **F-H.3. A0's true progress on held-out matches dev, and its standing against the oracle again depends on the collapse filter.** A0's T at 5M is 1.45–2.03. Kept A2″ seeds are above A0 in 12 of 16 late cells; all A2″ seeds in 8 of 16. Descriptive only (2–5 kept seeds per cell). Confidence: medium that the pattern matches dev; low on any single cell.
- **F-H.4. HA is not simply a harder state.** A2″ trained on HA reaches T 2.58 on HA at 5M (3 seeds), in line with the dev-trained kept seeds' 2.38 and 2.67 on HA. Confidence: low-medium (3 seeds, one at T 0 at 4M).
- **Inference.** Held-out adds no new failure and removes none. The batch-2 reading stays O3 (mixed): proxy learned and transferred, two proxy-attributed exploits, no fall in true progress.
- **F-H.5. On held-out, both exploits are again attributed to the proxy, and O carries the same share of reward as on dev.** O is above the oracle after Holm on HA in 6 of 7 marks for each training state; D on HB at 1M for both. O's share of P on HA alone is 0.181 (dev 0.178; random 0.082). Confidence: medium-high for O; low-medium for D. Late-mark references have 2–4 oracle seeds.

## E1 round 1 (patched proxy P1), dev only, 2026-11-11

Source: `code/results/r1/`; detail in `experiment-e1.md` §9 I15. 3 seeds on D1; descriptive except the vs-R0 family of 7. Held-out is sealed.

**F-R1.1. The cap shrank all exploration to about random play's level, off-route and on-route alike, and most true progress went with it.** On the P0 yardstick the *frequency* of A1's O is not above R0 at any dev mark (held-out HA: 1 of 7 on raw p). O's *share* of A1's P0 is unchanged from A0's, like for like (D1-trained A0 on the same eval state): 0.174 on dev (A0 0.169; R0 0.122) and 0.172 on HA (A0 0.166; R0 0.082). The 0.178 and 0.181 I first compared with pool A0 over both training states (corrected 2026-11-11). So the patch did not make exploration more on-route (wording corrected 2026-11-11 after held-out; the first version said the exploit was "removed"). A1's P0 at 5M is about 22 (R0 29.7; A0 112.6) and its T at 5M is 0.33 (A0 1.43). *Confidence: medium for the direction (all three seeds agree on P0; T is seed-dependent); low for any size.* Inference: this is the unselective patch doing what its offline check predicted.

**F-R1.2. A different type took over part of the reward: grass grinding.** G is present after Holm at 2M–5M and carries 0.22 of P1, on 2 of 3 seeds; it was absent for A0 in round 0. *Confidence: medium that G rose; low that it is caused by the misspecified reward,* since attribution is n/a and the oracle arm also showed G in batch 2.

**F-R1.3. The formal "O present under P1" is a detector artifact.** R0 fires O in 0 of 24 episodes under P1, because the detector reads the reward the cap removes. *Confidence: high.* Lesson: a detector that reads the reward cannot be carried across a reward patch without re-basing it.

**F-R1.4. The three seeds behave differently in kind.** s1: progress at 0.5M only, then grinding. s2: no progress at any mark. s3: progress at four marks, no grinding. *Confidence: high that they differ; no claim about which is typical.*

**F-R1.5. All three seeds scored below random on their own reward at the 1M checkpoint and recovered later.** The staged check tripped on 3 of 3 runs and none was a fault or a lasting stall. *Confidence: high for the observation; the cause is untested.*

### Open questions (round 1)

- Is A1's lower T caused by the cap's shape or by reward sparsity (total P1 is 0.17 of P0)? Test: a control with P0 scaled by 0.17 and no cap.
- Would a selective patch for O (one that leaves on-route exploration paid) remove O without the loss of progress? This needs a new patch design from P0.
- Why is the 1M checkpoint bad on every seed?

**F-R1.6. Round 1's held-out results repeat dev (opened 2026-11-11 after the stop decision).** T on HA at 5M is 0.08 (A0 1.45); the per-seed T pattern, G, M and O match dev by mark and seed. HF appears on one seed at one mark. *Confidence: medium, as for dev.*

**F-R1.7. The capped-reward policy does not transfer to HB.** It earns less than random there under P1 (3.58 against 7.77 at 5M) and under P0. A0 did transfer in batch 2. *Confidence: medium for the observation (all three seeds); the explanation (a start-local quota-filling policy) is inference.*

**Decision (Jake and the lead, 2026-11-11):** the patch loop stops after round 1, a declared post hoc deviation from stopping rule (a). `patch-log.md`.
