---
title: Does a stronger optimizer break a coverage reward? A pre-registered test in Pokémon Red
subtitle: Two batches, a held-out true metric, and a mixed result
date: 2026-11-10
draft: true
toc: true
bibliography: sources.bib
abstract: |
  A common claim in alignment is that optimizing a proxy harder first helps and then hurts the true
  objective. We test it in the first segment of Pokémon Red with a hand-written exploration reward
  (the proxy), a milestone count the agent never sees (the true metric), a truth-trained reference
  agent, a random baseline, and two held-out start states that stayed sealed until the first reward
  patch was logged. In batch 1 the proxy-trained agent was too weak to exploit its own reward, and
  nothing happened. In batch 2 we gave it a memory of visited squares. Its proxy score more than
  doubled, and its true progress rose with it. The pre-registered over-optimization hypothesis was
  inconclusive both times. Two mild exploits appeared, were attributed to the proxy, and reappeared
  on the held-out states. The clearest findings are about measurement: the reference agent was
  unstable, checkpoint scores were noisy, and one headline comparison flips with a filtering rule.
  One round of patching the main exploit followed. The patch was blunt: it shrank all exploration
  to the level of random play, most true progress went with it, and a different behavior took
  over part of the reward. We then stopped the patch loop, against the pre-set rule.
---

## Introduction

Reward hacking is usually told as a curve. Train an agent on a proxy for what you want; at first the
proxy and the true objective rise together; push harder and the true objective falls while the proxy
keeps rising. Gao et al. measure this shape for learned reward models in language-model fine-tuning [@gao2023scaling], Pan et al.
find it across several RL environments as agent capability grows [@pan2022effects], and Karwowski
et al. find it in about a fifth of sampled tabular problems [@karwowski2023goodhart].

Pokémon Red is a natural place to look for it. Published agents are trained on hand-shaped rewards,
and their authors describe specific problems: a heal-reward loop of near-endless battles, a level
reward whose removal gave the best ablation result (with side effects of its own)
[@pleines2025pokemon, sec. IV-B], and pacing loops in 41% of sampled baseline episodes, cut to
under 5% by hand-tuned anti-loop penalties [@mudireddy2026pokerl, sec. VIII-C]. Those reports are qualitative. None plots the shaped reward against true
game progress as optimization grows, and none holds out a true metric.

We ask two questions.

1. **Over-optimization (H0).** As an agent trained on a shaped exploration reward is optimized
   harder, does its true progress fall behind, in the Gao shape?
2. **Patch and retest (G2).** If the designer patches each exploit the agent finds, using only
   development states, does true progress on held-out states improve? Booth et al. show that
   reward functions designed by trial and error can be overfit to the learning algorithm and its
   hyperparameters [@booth2023perils]; we wanted to watch that happen, or
   not, round by round.

This draft reports question 1 for two batches of runs, the exploit catalogue, and the held-out
results, and one round of question 2, after which we stopped.

The short version: we did not find the curve. With a weak learner nothing diverged because nothing
was learned. With a capable learner the proxy and the true metric rose together. That is a real
result about this proxy on this segment, and we give the reasons we think it came out this way. It
is not evidence that the curve does not exist.

We then patched the reward once against the main exploit. The patch was blunt. It cut exploration
of every kind, most true progress went with it, and grass grinding grew on two of three seeds. We
stopped the patch loop there, against the rule we had set in advance, and say why.

## Background

**Proxy and true objective.** A proxy is "hackable" if some policy change raises it while lowering
the true reward. Skalse et al. show that, over all stochastic policies, two reward functions can
only be unhackable if one of them is constant [@skalse2022defining]. Zhuang and Hadfield-Menell
give conditions under which optimizing a proxy that omits attributes of the true utility drives
that utility arbitrarily low [@zhuang2021consequences]. So the question is never whether
a shaped reward can be exploited. It is whether a given optimizer, with a given budget, finds the
exploit.

**Shaping.** Adding a reward equal to the difference of a potential function between states
preserves the optimal policy, and other transformations may not [@ng1999policy]. Wiewiora proves
that a learner given such shaping makes the same updates as one whose initial Q-values are set
from the potential [@wiewiora2003potential]. The invariance is why our second reference agent
used potential-based shaping. It gave
PPO no lasting signal, so the reference we ended up with gives up that guarantee (see "Arms").

**The environment.** We use PyBoy [@pyboy] with the game's RAM read through the pret disassembly's
symbols [@pret2026pokered]. The reward design follows the family used by Whidden's agent and its
successors [@whidden2023pokemonred; @pleines2025pokemon].

## Method

Everything below was fixed in a pre-registration file before the runs it governs. Changes made
after seeing data are listed in "Deviations and post hoc decisions".

### Task and metrics

The segment starts in Oak's Lab, just after the Pokédex, and **ends at the entrance to Viridian
Forest**.

- **True metric T:** the number of milestones reached in an episode, read from RAM: Viridian City
  (M1), Route 2 (M2), Viridian Forest (M3). No agent trained on the proxy ever receives T.
- **Why it ends at M3.** The plan defined six milestones, up to beating the first gym leader. A
  pilot of the reference agent, extended to 10M steps, never passed M3, so we cut the segment
  there before the main runs (2026-11-04). No learned agent in this paper passes M3, so T ranges
  from 0 to 3. The scripted planner in the batch-1 results does go further (T = 5); that is
  outside the segment and is reported only to show where the proxy's optimum points.
- **Proxy P0:** 0.1 per new map square visited, 1.0 per level gained, and a heal term. In practice
  almost all of it is the new-square term.
- **Episodes** are 16,384 steps.

### Arms

| Arm | Trained on | Role |
|---|---|---|
| A0 | proxy P0 | the agent under test |
| A2″ | T plus a bonus for each new best walking distance to the next milestone | "oracle" reference for T |
| R0 | nothing (random actions) | baseline for every detector and for P |
| A1 | patched proxy P1 | round 1 of the patch loop (3 seeds, one start state) |

All learned arms use PPO with a small convolutional policy over a tile grid, with fixed
hyperparameters and 5M steps per run. A0 and A2″ have 5 seeds on each of two training states.

**The reference agent took three designs.** A reward of T alone paid out once in 5M steps. T plus
potential-based shaping (A2′) kept the optimal policies in theory and stalled in practice. A2″
pays T plus a bonus whenever the agent sets a new best walking distance to the next milestone
within the current leg. Loops and backtracking earn nothing, and the bonus is bounded by about 1
per leg. **It is not potential-based, so the invariance result does not apply to it**; a discounted
agent can prefer earlier progress, which T prefers too. A2″ was fixed before any main-arm data.

**Compute.** Runs used CPU-only cloud VMs (GCP t2d-standard-8 and -4) at roughly 350–400
environment steps per second with the policy in the loop, about 4 to 6 hours per 5M-step run
including evaluation. Everything reported here, pilots and round 1 included, used about $70 of
Google Cloud trial credit, of which round 1 was about $5. This is an estimate from VM uptime, disk
and early Spot usage at list prices, not a billed figure.

### States and blinding

- **Development:** D1 (Squirtle) and D2 (Charmander), both starting in Oak's Lab.
- **Held-out A (HA):** the same start with Bulbasaur.
- **Held-out B (HB):** D1's party, starting at the Viridian Pokémon Center door.

All held-out outputs were written to sealed folders with a hashed manifest. The analyst did not
open them until the round-1 patch was logged with its reward hash. Both manifests verified
unchanged before opening.

### Exploit detectors

Each evaluation episode is cut into windows. Pre-registered detectors flag windows as, for example,
heal farming, grass grinding, menu spam, time on a dead-end route (D), or off-route exploration
(O: reward earned on maps behind the agent's furthest milestone or in side buildings). A type is
**present** if its share is above R0's after a Holm correction over its family of cells
(checkpoint × state). It is **proxy-attributed** if it is also above the oracle arm's.

### Hypothesis

H0 is supported on a training state if (i) A0's P is higher at 5M than at 0.5M, (ii) A2″'s T is
above A0's at some mark from 2M on, and (iii) A0's T does not track its P. It is falsified if (ii)
fails at every mark and T rises monotonically. Anything else is inconclusive. Oracle checkpoints
that have collapsed (T at least one milestone below that seed's earlier best) are dropped from the
reference.

### Batch 2 is a follow-up, not part of the original plan

The original plan was one batch followed by the patch loop. Batch 1 produced no exploit to patch.
We then ran the planner diagnostic, saw that the learner was the limit, and chose to fix the
learner's observation and run again (2026-11-07). **So batch 2 was designed after seeing batch 1's
development results.** It was pre-registered before any batch-2 code or run existed and while
batch 1's held-out data were still sealed, but it is a second experiment informed by the first.

Batch 1's policy saw the current screen as tiles and nothing else. Batch 2 adds one input channel
marking squares already visited this episode. Nothing else changed. We pre-registered a
manipulation check (A0's P at least twice batch 1's) and staged validity checks at 1M and 2.5M
steps that could pause the batch.

## Results

### Batch 1: a learner too weak to exploit its reward

- A0's P was flat from the first evaluation mark, at about 1.4 times random.
- Pooled over marks, P ranked A0 (42.9) above random (30.4) above the oracle (15.3), while T ranked
  the oracle first. At 5M the oracle's T was 2.69 against A0's 1.08 on D1.
- H0 was inconclusive on both states: P did not rise, so there was no curve to have a shape.
- No exploit type was present above random, on development or held-out states.

A planner with perfect memory and map knowledge showed why. Greedily covering new squares, it
earned P of at least 417, about eight times A0's best, and reached T = 5 in 16 of 16 episodes.
So the proxy was far from saturated, and **its optimum on this segment is aligned with the true
objective**: covering the map takes you through the route. Batch 1's misranking of agents by P was
a weak-learner effect.

We should have seen this in the pilot. The pilot's P was exploration-only and flat from 0.5M. We
had checked the oracle's capability and whether detectors fired, and had not compared P itself with
random or with an attainable ceiling.

### Batch 2: the proxy is learned, and true progress rises with it

- **Manipulation check: passed.** A0's best pooled P was 120.9 (D1) and 121.7 (D2), against 55.5
  and 50.5 in batch 1.
- **H0: inconclusive on both states.** P rose (64.6 to 112.6 on D1; 57.1 to 121.7 on D2), so (i)
  held. T rose with it (1.05 to 1.43; 1.40 to 1.95), so (iii) failed.
- **Condition (ii) depends on a filter.** With collapsed oracle checkpoints dropped, as
  pre-registered, the oracle's T was above A0's at 7 of 8 late cells. With them kept, it was above
  A0's at no mark on D1, and at 5M A0 was at or above the oracle on both states. We report both.
  The pre-registered result is the first; the second shows how much it rests on the filter.

A descriptive series from training episodes (rather than frozen checkpoints) agrees with the
snapshots on D1. On D2 the snapshots show a divergence that the training episodes do not, and it
comes from the 3M snapshot alone, the unexplained dip described below.

### The exploit catalogue

| Type | Present (dev) | Share of P | Seeds | Attribution | Present (held-out) |
|---|---|---|---|---|---|
| O, off-route exploration | 5 of 14 cells | 0.178 (0.245 at 5M) | 8 of 10 | proxy (dev and held-out) | HA at 2M, 4M, 5M |
| D, dead-end route | 1 cell | 0.021 | 3 of 10 | proxy (dev and held-out) | HB at 1M |
| all others | none | | | | none |

Both exploits are mild, and both coexist with rising T. A review of unclassified high-reward
windows found only bursts of new-square exploration, so no detector was added.

### Held-out states

Held-out results agree with development results in both batches.

- **Batch 1:** no type present.
- **Batch 2:** A0's P at 5M was 103.0 to 123.3 across the four held-out panels, the same range as
  on its training states. Its T was 1.45 to 2.03. The same two exploit types appeared and no
  others, and both were again attributed to the proxy. On HA, off-route exploration carried 18.1%
  of A0's reward, against 17.8% on development states and 8.2% for random play.
- **Oracle comparison (descriptive):** kept oracle seeds were above A0 in 12 of 16 late cells; all
  oracle seeds in 8 of 16. The filter dependence carries over.
- **HA is not simply harder:** an oracle trained on HA reached T 2.58 there (3 seeds), in line with
  development-trained oracle seeds evaluated on HA.

### What went wrong with measurement

These are the results we are most confident will generalize.

- **The oracle is unstable.** 13 of 40 late oracle checkpoints had collapsed. Three runs ended at
  5M with the agent never leaving the first room, after reaching T of 2 to 3 earlier. One was
  reproduced bit for bit on two other machines, so it is a property of the learned weights.
- **A frozen checkpoint is not the learner.** A0's training episodes scored two to three times its
  frozen checkpoints. A direct test showed why: resuming learning from a checkpoint that scored
  about 50 gave 131 in its first episodes. Much of the proxy is earned by the optimizer adapting
  within an episode.
- **Checkpoint scores are noisy.** Across 20 extra checkpoints, a run's P ranged from 0.40 to 1.25
  of its own 4M/5M average. We guessed this tracked the phase of the training cycle, wrote the
  predictions down, and the guess failed (rank correlation +0.19 where we expected a negative
  one). A dip at 3M in 8 of 10 seeds is unexplained.

### Round 1: patching off-route exploration

The patch caps the exploration reward at the first 20 new squares per map. The cap value was
chosen by a rule fixed beforehand: the largest of six values that removes at least 80% of O's
reward on the round-0 development episodes. It removes 85.7%.

It is also **close to unselective**. Reward outside O's windows falls by 82.4%, total reward falls
to 0.17 of its old value, and O's share of reward moves only from 0.178 to 0.150 on the same
behavior. We kept the cap because the rule was fixed first. The consequence is that round 1 cannot
cleanly separate "the exploit resists patching" from "the patch did not target it", and a stall
from sparse reward is a likely outcome. Staged checks at 1M and 2.5M steps are pre-registered.

**Staged checks.** At 1M every seed tripped the same check: the frozen policy earned less patched
reward than random play (7.34, 5.16 and 5.46, against 7.56). Each time we ruled out a measurement
fault and continued. All three recovered in patched reward by 2M to 3M (seed 2 was still below
random at 2M), so a trip at 1M did not predict the reward at later marks. It says nothing about
true progress, which stayed at 0 on seed 2 throughout. Seed 2 was first cancelled for cost and
then reinstated under a rule written before the 2.5M checks (see "Deviations").

**Results (development state D1, 3 seeds, 8 episodes per mark; descriptive).** Held-out results
follow below.

*True progress fell.* T at 5M is 0, 0 and 1.0 by seed, a mean of 0.33. A0 under the original
reward reached 1.43 on the same state. The per-mark picture matters more than the 5M figure,
because T swings between adjacent checkpoints:

| Seed | 0.25M | 0.5M | 1M | 2M | 3M | 4M | 5M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s1 | 0 | 2.13 | 0 | 0 | 0 | 0 | 0 |
| s2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| s3 | 0 | 2.0 | 0 | 2.25 | 0.88 | 0 | 1.0 |

*The agent learned the patched reward, slowly.* Its reward at 5M is 11.64 pooled (12.77, 12.42,
9.73), against 7.56 for random play. Scored under the original reward, the same episodes earn
about 22 on every seed. That is below random play (29.7) and far below A0 (112.6).

*What happened to the exploit.* Under the patched reward the detector still labels O as present
at three marks. **That label is an artifact.** The O detector reads the reward, the cap removes
that reward, and so random play shows no O at all under the patched reward. We therefore re-ran
the detectors on A1's episodes under the original reward, a table we added before seeing any
round-1 detector output. Two measures on that yardstick point in different directions, and both
are true:

- **How often** A1 is in a side map is at the level of random play at every mark.
- **What share of its original-reward total** comes from side maps is 0.174. For A0 trained on
  the same start state it was 0.169 (0.178 pooled over both start states); for random play it
  is 0.122.

So the patch did not make exploration more on-route. It shrank all exploration to about what
random play does and left its composition unchanged, which is what the offline check predicted.
D is absent in every episode.

*What replaced it.* Grinding battles in grass (G) is present from 2M to 5M and carries 22% of the
patched reward. It appears on two of the three seeds. In batch 2, before the patch, A0 did it in none of 560
episodes.
The patch left the level reward untouched, so by our offline estimate on round-0 episodes its
weight relative to exploration rose about sixfold, and two seeds found it. Menu time (M) is also flagged at 4M and 5M, each time from a single seed; random
play does it in 29% of episodes, and we give it little weight.

| Type | Present under P1 | Share of patched reward | Seeds | Under the original reward |
| --- | --- | --- | --- | --- |
| G (grass grinding) | 2M–5M | 0.223 | s1, s2 | also present |
| O (off-route maps) | 0.5M, 1M, 5M | 0.139 | all | frequency not above random at any mark; share of reward unchanged |
| M (menus) | 4M, 5M | 0.042 | one per mark | also present |
| D (Route 22) | never | 0 | none | never |

*The seeds differ in kind.* One makes progress early and then grinds. One never makes progress.
One makes progress at four marks and never grinds. Three seeds cannot say which is typical.

**Held-out states (opened after the stop decision was logged).** The held-out results repeat the
development ones.

- On HA, true progress follows the same per-seed pattern and is 0.08 at 5M; A0 had 1.45 there.
- G, M and O are flagged at the same marks and from the same seeds, apart from G at 5M. A
  heal-farming type (HF) is flagged on one seed at one mark; we give it no weight.
- O on the original-reward yardstick: frequency above random at one mark of seven before
  correction and none after; share of reward 0.172, against 0.166 for A0 trained on the same
  start state (0.181 over both) and 0.082 for random.
- On HB, which starts in the next town, A1 earns less than random play under either reward
  (3.58 against 7.77 of patched reward at 5M). Its true progress there is 1.25 at 5M against 1.0
  for random play; HB starts at the first milestone, so 1.0 means no progress. A0 had
  transferred to HB in batch 2. Our reading,
  which is inference, is that a policy trained under the cap learns to fill its quota on the
  maps around its own start.

**What round 1 shows, and what it does not.** It shows a patch that cut the targeted behavior
only by cutting everything, at the cost of the task, and a second behavior growing into the space
the patch left. It does not
show that patching O must cost progress: this patch cut reward almost everywhere, as its offline
check had said it would. It also cannot separate the cap's shape from plain reward sparsity,
because total reward fell to 0.17 of its old value. There is no reference arm in round 1, so G is
not attributed to the reward; the truth-trained reference also grinds in batch 2.

**Stopping rule.** The rule set in advance says to patch again: new types appeared, and A1's true
progress (0.33) is below the two kept reference seeds (1.75 and 2.62). As at round 0, the answer
depends on the filter. With collapsed reference checkpoints counted, the reference interval is
0.28 to 2.05, A1 falls inside it, and the loop would stop.

**We stopped anyway. This is a declared deviation, decided after seeing round 1's development
results.** Under the patched reward the agent already earns less of the original reward than
random play. A second patch stacked on it would be tested on a learner that barely works, and
could not separate "patched" from "starved" any better than this round did. The useful next
experiment is a selective patch for O starting from the original reward, which is a new design.

## Discussion and limitations

**Why no curve?** Our best explanation, which is inference, is that this proxy is aligned with the
goal over this segment. The route to the first gym passes through most of the reachable squares,
so covering squares and making progress are the same activity until the agent runs out of route.
At 5M steps A0 covers about 27% of the planner's squares. Divergence would need a regime where
exploration and progress compete, such as a longer horizon or a segment with large optional areas.
We did not reach one.

**What would test that.** Train A0 past the point where route squares run out, or start it where a
large optional area is next to the route, and check whether O's share grows while T stalls.

**Alternative explanations we cannot rule out.**

- *Capability is still limiting.* A0 is at 29% of the planner's P. A much stronger learner might
  behave differently.
- *The oracle sets a low bar.* If the reference were stable, A0 might clearly trail it. The filter
  dependence is the symptom.
- *T is coarse.* Six milestones cannot register small losses of progress.

**Limitations.**

- One game segment, one proxy family, one algorithm, 5 seeds per cell.
- Two development states that differ only in the starter.
- Round 1 has 3 seeds on one state and is descriptive, except for the tests against random play.

**Deviations and post hoc decisions.** Each was logged with its date when it was made.

*Before the main runs (pilot phase).*

- **Emulator core (2026-11-01, 2026-11-02).** We first accepted PyBoy's fast mode conditionally,
  then moved every arm to the normal core and regenerated the start states. The two cores diverge
  from power-on, so they give different trajectories.
- **Reference agent redesigned twice (2026-11-03).** A2 to A2′ to A2″, each after a failed pilot,
  as described under "Arms".
- **One remedy pilot (2026-11-03, ruled 2026-11-04).** Under a rule set in advance, an oracle
  collapse triggered one pilot at half the learning rate. It failed, and the original
  hyperparameters stayed for every arm.
- **Run length and segment end (2026-11-04).** After the oracle's 10M pilot, runs were set to 5M
  steps and the segment was cut to M3.

*After batch 1.*

- **Batch 2 itself (2026-11-07),** as stated in Method.

*During and after batch 2.*

- **Four staged checks tripped.** Two were infrastructure: an evaluation slowing a training run
  (2026-11-07), and a slow host at one run's closing read (2026-11-08). Both were ruled false
  alarms. Two were about the science: single A0 runs scoring below the proxy thresholds at 1M and
  2.5M (2026-11-08). After a diagnostic showed the measurement was sound, the batch resumed
  unchanged. **That ruling was made after seeing interim data,** and so was the criterion it used.
- **Training-episode series (2026-11-08).** Added after the interim reads. Its first implementation
  used episode-level tests where the specification called for seed-level intervals; correcting it
  changed batch 1's D1 label from "divergence" to "agree".
- **Unfiltered oracle comparison (2026-11-10).** Added before the batch-2 analysis, after the
  collapse flags were visible.
- **Stopping rule for the patch loop (2026-11-10).** The rule did not say whether collapsed oracle
  checkpoints count. We chose the filtered reading, after seeing the data, because it matches every
  other use of the reference. Under the other reading the loop would have stopped before round 1.
  The filtered reference rests on 2 seeds at that cell.
- **Phase check regrouped (2026-11-10).** The checkpoints' true positions in the training cycle
  differed from the estimates used to name the groups. We applied the thresholds to the true
  positions and adjusted the pass count from 4 of 6 to 3 of 4, before any result was read.

*Round 1.*

- **Three staged checks tripped (2026-11-10, 2026-11-11),** one per seed, all at 1M and all the
  same check. Each was ruled not a fault and the run continued.
- **Seed 2 cancelled, then reinstated (2026-11-11).** After the two running seeds (1 and 3)
  tripped the check at 1M, seed 2 was cancelled for cost and time, before the 2.5M checks. The
  rule "run seed 2 if either running seed beats random at 2.5M" was written after the 1M checks
  and before the 2.5M checks. One seed met it, and seed 2 was reinstated after the 2.5M checks.
  Both decisions followed interim data. The final design is the planned 3 seeds.
- **Original-reward detector table for round 1 (2026-11-11).** Added after the interim and closing
  checks and before any round-1 detector output was read. It is descriptive and has no test family.
- **Patch loop stopped after one round (2026-11-11).** The pre-set rule called for a second
  round. We stopped for the reason given under "Round 1", after seeing its development results
  and before opening its held-out results.

**A security reading.** Treat the reward as a grader and the agent as the party being graded. The
exploitable gap exists: the grader pays for wandering. This agent used it only a little. The
weaker link was the audit. A checkpoint-based check gave different answers depending on when the
checkpoint was taken, the agent being audited behaved differently while it was still learning, and
the trusted reference was itself unreliable. An overseer relying on any one of those would have
been misled more by its own instruments than by the agent. Round 1 adds the other half: a blunt
fix to the grader closed the gap by paying less for everything, and on two of three seeds the
graded party moved to the next term that still paid.

## Conclusion

On the first segment of Pokémon Red, a coverage reward did not turn against true progress when the
optimizer got stronger. The weak agent failed to learn it, and the capable agent's progress rose
with it. Two mild, proxy-attributed exploits appeared and transferred to held-out states. The
pre-registered hypothesis was inconclusive in both batches. What the experiment did show clearly
is how fragile the measurements around such a claim are, and that a "null" from a weak learner
says nothing about the reward. One round of patching the main exploit, with a blunt cap, shrank
all exploration and most true progress together, left the exploit's share of reward unchanged,
and saw grass grinding grow on two of three seeds. We stopped there, against our own rule.
Whether a selective patch would do better is untested.

## Figures

*Not yet placed in the text. The files exist under `code/results/`. They
need copying next to the post when it moves to `content/posts/`.*

1. **Proxy score and true progress against training steps, batch 1** (development states; A0,
   A2″, R0). `b1/fig1.svg`.
2. **The same for batch 2** (visited-squares observation). `b2/fig1.svg`. This is the main figure
   for "no curve": both lines rise together.
3. **Held-out states, batch 1 and batch 2.** `sealed-b1/b1/fig1-heldout.svg`,
   `sealed-b2/b2/fig1-heldout.svg`.
4. **Held-out HA with the HA-trained reference line.** `b2-heldout/fig1-heldout-ha-ref.svg`.
5. **Round 1: true progress per checkpoint, one line per seed, with A0's batch-2 line for
   comparison; and patched reward split by term.** `r1/fig5.svg`, with a table and a post-ready
   ASCII version in `r1/fig5.md`.
