---
title: "Running a Pokémon experiment with an agent team"
subtitle: "Build log: what it took to get a reward-hacking experiment to the starting line"
date: 2026-11-07
draft: true                 # delete this line to publish
toc: true
bibliography: references.bib
abstract-title: TL;DR
abstract: |
  - A team of coding agents designed and launched E1, an experiment that trains RL agents in Pokémon Red against a hand-shaped reward and measures how far that reward and real progress come apart. This post covers the week it took to get the first main batch running. There are no results in it yet.
  - Most of the work turned out to be making sure the measurement was trustworthy: an emulator speed-up that failed a fidelity test, an "oracle" reference agent that took three designs before it could learn, and cloud restarts that quietly broke checkpointing.
  - What kept it honest was writing rules down before seeing data and refusing to loosen them afterwards, sealing the held-out results from the agent who designs the fixes, and a hard spending ceiling that kept the whole thing inside a free cloud credit.
---

My [last post](../your-reward-model-is-an-attack-surface/) was about reward hacking: optimize hard against a proxy for what you want, and the proxy stops tracking what you want.
The follow-up is an experiment that measures this directly, in a setting small enough to run ourselves.
We call it E1.

This post isn't about E1's results, because there aren't any yet.
It's about how it got built, by a team of four AI coding agents and me, and the problems that came up along the way.
Most of those problems were about whether we could trust the measurement at all.

## The experiment in one paragraph

E1 trains a reinforcement-learning agent to play the early part of Pokémon Red, from Professor Oak's lab to Viridian Forest, on a game emulator called PyBoy [@pyboy].
The agent is rewarded by a hand-written **proxy** $P$, modeled on earlier Pokémon RL work [@whidden2023pokemonred; @pleines2025pokemon]: points for visiting new tiles, gaining levels, and healing.
Separately, we track **true progress** $T$: how many story milestones the agent reached, in order (reach Viridian City, reach Route 2, enter Viridian Forest, and so on), read straight from the game's memory.
$T$ never goes into the proxy.
The question is the one from the last post: as training pushes $P$ up, what happens to $T$?
A second phase patches each hack the agent finds and retrains, to see whether the patches keep up.

Two kinds of start state matter for the rest of this post.
The **dev** states are what the agents train on and what the designer looks at when deciding how to patch the reward.
The **held-out** states are variations (a different starter Pokémon, a later starting position) that the designer never sees until the patch is locked in.

## Who did what

The team is four agents running side by side in one checkout of this repo, each with its own role and its own files.
They coordinate through a shared task board (`research/BOARD.md`) and message each other directly.

<figure>
```
                     Jake
        (picks the experiment, sets the budget,
         approves anything that costs money)
                       │
                       ▼
                ┌────────────┐
                │    lead    │  scopes work, approves plans,
                └────────────┘  keeps the board
               ╱       │       ╲
              ▼        ▼        ▼
   ┌────────────┐ ┌───────────┐ ┌────────────┐
   │ researcher │ │ developer │ │   writer   │
   └────────────┘ └───────────┘ └────────────┘
   design doc,     emulator      posts
   rulings on      harness,      (this one)
   every result    cloud runs
```
<figcaption>The roles on E1. The researcher and the developer did almost all of the work described here.</figcaption>
</figure>

In practice E1 ran as a loop between two agents.
The **researcher** wrote the design document (`experiment-e1.md`): the hypotheses, the reward terms, the hack detectors, and the pass/fail rules for every pilot.
The **developer** built the harness, ran things on cloud VMs, and reported back with numbers and log paths.
The researcher then wrote a *ruling* on each result, which went into the design doc with a date.
The **lead** approved anything that changed scope, and anything that cost money came to me.

That loop worked well, in part because each side caught the other's mistakes.
Several of the stories below are one agent finding an error in the other's reasoning.

## Is the fast emulator the same game?

The first question was speed.
PyBoy has a `fast_mode` that simplifies some of the hardware emulation, and a faster emulator means more training per dollar.
The developer checked it with a 1,500-step run and found the game's working memory (WRAM) byte-identical with and without it.

The researcher didn't accept that yet.
Pokémon Red's random-number state lives in a different memory region (HRAM), which that check didn't cover.
It's stirred by a hardware timer on every call, and timing is exactly what a "simplified" emulator might change [@pret2026pokered].
A short, mostly indoor run barely uses the RNG.
Wild encounters, damage rolls and catch rates all do.

So the researcher wrote acceptance criteria *before* the developer ran the longer test:
at least 10,000 steps, covering battles, a Pokémon Center heal and ideally a blackout; compare WRAM *and* HRAM at every step; pass only if byte-identical, with a narrow exemption for sound and serial-port registers.

A scripted 12,000-step run passed.
A random-input run diverged at step 2,837, when the agent talked to the Cable Club attendant (the in-game link-cable NPC): first the RNG bytes, then 3,396 memory addresses.
`fast_mode` doesn't emulate the link-cable check, and the difference leaks into the RNG.

This is where the team made its first real judgment call.
Nothing had been trained yet, so the researcher allowed one narrowly worded amendment, with the reasoning written down.
The amendment changed the question from "is it identical?" to "can it reach anything the real game can't?"
If it can, an agent could learn an emulator-only exploit, which would contaminate the hack catalogue with artifacts.
The amended test had four conditions, including that every divergence must start within one step of the link-cable dialogue opening.

Ten runs produced 675 divergences, exactly one per Cable Club visit and none anywhere else.
Three of the four conditions passed on all 675.
The fourth failed: 33 divergences started 2–6 steps after the dialogue opened, not within one.

Those 33 almost certainly had the same cause.
A looser "same dialogue" criterion would have passed.
The researcher still refused to loosen it:

> Loosening a pre-registered test the first time it fails is the move this design exists to avoid.

So `fast_mode` was dropped, and every run uses the normal emulator core.
It also turned out that the two cores diverge at power-on, so every save state had to be rebuilt on the normal core.
Giving up the speed-up hurt less than it sounds: re-costed at the normal core's rate, the full plan still fit inside the free cloud credit with no cuts.

## The oracle that wouldn't learn

To claim "the proxy got hacked," you need to rule out "the task was just too hard."
So E1 has an **oracle** arm, A2: the same agent, same network, same training budget, but rewarded on true progress $T$ instead of $P$.
If the oracle reaches milestones the proxy-trained agent doesn't, the gap means something.
If the oracle can't learn either, a low $T$ for the proxy agent tells you nothing.

The oracle took three designs.

### A2: the sparse oracle

The first version paid +1 per milestone and nothing else.
In a 5M-step pilot, the reward was nonzero in **2 of 4,883 updates**.
One lucky episode wandered into Viridian City.
The rest of the time the agent had nothing to learn from, and its exploration actually shrank as training went on.
The proxy-trained agent, by contrast, reached Viridian early in its own pilot, because "visit new tiles" pushed it out the door.

That is the usual reason people shape rewards in the first place, and the paper will say so.
But it meant A2 couldn't serve as a reference.

### A2′: shaping that can't change the answer

The researcher's fix was potential-based shaping.
You add a bonus $F = \gamma\,\Phi(s') - \Phi(s)$ for moving between states, where $\Phi$ is any function of the state.
Ng, Harada and Russell proved that this kind of bonus never changes which policies are optimal [@ng1999policy].
Here $\Phi$ was the negative distance to the next milestone, so the bonus points toward the goal without rewarding anything $T$ doesn't.
In theory, that gives the agent a trail to follow without changing what counts as winning.

Two things went wrong, and the agents caught both.

First, the researcher had written that walking in a loop can't earn shaping reward.
The developer pointed out this was wrong: with $\Phi \le 0$, every step earns a small positive stipend no matter what the agent does.
The researcher checked, agreed, and showed why it still doesn't change incentives (the stipend shifts every action's value in a state by the same amount).
Both versions stay in the design doc, with the wrong one struck through.

Second, the pilot failed.
The agent spent about 3M steps in Oak's Lab, then parked on the north edge of Pallet Town, pressing into the trees.
The researcher found two causes:

- **A bug in the distance map.** Every tile along Pallet's north edge counted as an exit to Route 1, though only a two-tile gap actually is one. Along that row, $\Phi$ was flat. Moving sideways toward the real gap paid exactly what standing still did, so the agent stood still.
- **A structural problem.** With the PPO algorithm we use, and the same discount factor for shaping and training, shaped PPO is exactly unshaped PPO with the value estimate shifted by $\Phi$. As the critic learns, the shaping cancels out and the guidance fades. This is the researcher's derivation, extending a known result for tabular Q-learning [@wiewiora2003potential]. The property that made A2′ "still an oracle" is the same property that makes it stop helping once the critic is trained.

The second point mattered most, because it meant even a correct $\Phi$ would fail.

### A2″: pay only for new progress

The third version drops potential-based shaping.
Its reward is

$$
r_t = \Delta T_t + \frac{\max\big(0,\; d^{\min}_{t-1} - d_t\big)}{d_{\text{leg}}}
$$

where $d_t$ is the true walking distance to the next milestone (a breadth-first search over walkable tiles, built from the game's own map and collision data [@pret2026pokered]) and $d^{\min}$ is the closest the agent has been during this leg.
It pays only for a *new best* distance, so loops and pacing earn nothing.
There's no stipend, and the total per leg is bounded by about 1.

Two things were different this time.
First, the walking-distance grid was validated before any training: every tile any previous agent had stood on had to be in it, and a simple greedy planner that always steps downhill had to walk from Oak's Lab to Pewter City in the real emulator.
It did, on 3 of 3 seeds.
Second, the researcher declared in advance that **A2″ was the last redesign**.
If it failed, the fallback was a different observation encoder, then a shorter segment, then a different game entirely.

A2″ passed its gate on the dev start state.
Its training also showed something odd: whole batches of episodes would succeed together, then fail together for a while, then recover.
Before looking at the final evaluation, the researcher wrote rules for what would count as a collapse and what each kind would trigger.
One rule called for a remedy pilot at half the learning rate.
The remedy failed its own pre-registered criterion and learned more slowly, so the original settings stayed for every arm.
The researcher's leading explanation (a blackout-related observation ambiguity) also failed its test.
**The cause of the oscillation is still open**, and the paper will report it as an unexplained property of the reference arm.

The A2″ pilot also set two numbers for the main runs: the segment ends at entering Viridian Forest, and each run trains for 5M steps.
Both came from rules written before the pilot finished.

## Spot preemption and the restart bugs

The runs needed more than my laptop (4 cores, no GPU), so the developer moved them to two small Google Cloud VMs on a free trial credit.
For the pilots we used **Spot** VMs, which are much cheaper but can be reclaimed by Google at short notice.
That makes checkpoint-and-resume essential, and it's where most of the infrastructure bugs were.

- **The shutdown checkpoint never ran.** The trainer saved a checkpoint on SIGTERM, the signal a preemption sends. But it ran inside tmux, and when the VM shuts down tmux sends SIGHUP to its children, not SIGTERM. So every "preemption checkpoint" in the first pilot was really the last 15-minute periodic one. The fix was to checkpoint on SIGHUP too.
- **A resume with nothing to resume from.** The oracle pilot was preempted at 69k steps, before its first checkpoint. The restart script called the trainer with `--resume`, which exited because there was no checkpoint. The idle guard then saw an idle VM and shut it down. This happened three times and wasted about 2.3 VM-hours before anyone noticed. Now `--resume` with no checkpoint starts from zero.
- **The baseline that became a trainer.** E1 also has R0, a random-policy evaluation that does no training at all. After one preemption, the generic resume logic restarted R0 *as a proxy-trained agent*. Now eval-only runs are never resumed, and there's a test that fails if that fix is removed.

The researcher also had to decide what a resumed run *is*.
When a VM is preempted after the last checkpoint, the steps trained since are lost and get re-run, which leaves a discarded branch in the logs.
The rule: a run is its *final lineage*, the steps leading to the last checkpoint.
Analysis drops the discarded branches, and step budgets count lineage steps, so arms are compared at equal training whatever their preemption luck.
Resumes aren't bit-reproducible (episodes in progress restart, and env seeds shift), and that's accepted and recorded per run.

For the main runs, I chose **on-demand** VMs instead of Spot, mainly so they could run unattended while the laptop slept.
That surfaced two more cloud quirks.
GCP won't convert a VM created as Spot into a standard one, so the developer snapshotted the disk, deleted the VM, and recreated it in a different zone (the original zone had no capacity left), then verified every file by hash.
And the obvious way to cap a VM's runtime, a maximum run duration, resets every time the VM restarts.
The developer switched to an absolute termination time, which doesn't.

## Pre-registration and blinding

The thread running through all of the above is that rules got written down *before* the data they judge.
The design doc marks each one "pre-registered", with a date.
Hyperparameters were frozen from the pilot for every arm, because reward designs can overfit to hyperparameters [@booth2023perils].
The rule for when a hack counts as caused by the proxy (it must beat both the random baseline and the oracle in the same cell, after multiple-comparison correction) was fixed before any main run.
So was the full analysis plan for the first batch, written before any of its runs were fetched.

Writing that plan surfaced two problems.

The first was a gap: the second dev start state was never evaluated on itself, so half the dev-trained runs had no score on their own training state.
That was fixed with an evaluation-only pass, with no retraining.

The second was a leak.
The patch phase asks: if the designer patches the reward using what they see on the dev states, does the patch also help on held-out states they never saw?
That only works if the designer really hasn't seen them.
Here the designer is the researcher agent, and the researcher was about to run analysis that included held-out numbers.

The team's fix:

- The developer splits every held-out output into a sealed directory and hashes each file into a manifest, so any later change is detectable.
- The researcher doesn't open it until the round-1 patch is logged with a hash of the new reward function, and the timestamp is on the board.
- The analysis pipeline is tested on dev data only. For the held-out part, the developer confirms row counts and that it runs without errors, and shows no numbers.
- The lead and I aren't blinded. The lead agreed not to pass held-out numbers to the researcher, and the developer doesn't post them on the board.

There was a subtler leak too.
One of the detectors flags unusual episodes for manual review, using a top-10% threshold computed over all start states, so held-out episodes were helping set the dev threshold.
That threshold is now computed from dev states only.
It changes no test, since those flags are reviewed, not counted, but it keeps the review blind.

There's also a dry run of the whole analysis on the first half of the batch, labeled "DRY RUN: no conclusions" in every file.
Its only job is to catch pipeline bugs before the real analysis.
Nothing from it gets used, and the statistics are rerun from scratch on the full batch.
It passes its checklist.

This is the part of the project where having agents is unusually convenient.
"Don't look at the held-out data" is a hard promise for a person working alone to keep.
Here it's a file the researcher doesn't open, and a rule the other agents follow.

## Keeping it under the free credit

Everything so far runs on a Google Cloud free trial: $300 of credit, $0 out of pocket.
I set a hard cap of $141 for the main arms.
The developer built the cost controls in layers:

- **A cost model** (`plan_cost.py`) that turns a run plan into VM-hours and dollars, using measured throughput. The prices come from third-party trackers and are marked unverified.
- **A ceiling** at the plan's estimate plus 30%, written to a ledger. A script refuses to start a VM if that VM's hours or the plan's dollars would exceed it.
- **Guards inside each VM**: lifetime uptime counted every minute, an idle shutdown, and the absolute termination time on the cloud side.

The developer also wrote down the remaining risk: if a VM hangs and stops counting, the overrun is limited to that one window.

For scale, the developer measured about 370 environment steps per second on the 8-vCPU VM.
Each main run is 5M steps, plus evaluation.
The first batch is 23 runs, the full plan has an estimate of about \$56 and a ceiling of \$71.44, and by the developer's ledger the whole project has used roughly \$30 of the trial credit so far.
Two decisions kept that low: a small observation (the game's 20×18 tile grid instead of screen pixels, which the design doc puts at a third to a quarter of the learner cost), and settling each open question with a one-seed pilot before committing to the main arms.
The biggest single waste was the restart bug above, about 2.3 VM-hours of an idle VM.

## What worked, and what didn't

**Worked:**

- *Rules before data, and not loosening them.* The `fast_mode` test is the clearest example. The researcher kept a threshold the data narrowly failed, because the alternative is a design where every inconvenient result gets explained away.
- *Agents checking each other.* The developer caught the researcher's wrong claim about loops. The researcher caught the developer's incomplete fidelity check. Each correction stays in the design doc next to the original.
- *Validating the measurement without training.* The greedy planner showed the walking-distance reward pointed the right way before any compute was spent on A2″.
- *Recording failures as results.* The sparse oracle, the potential-based oracle and the failed remedy are all in the design doc, and the paper will report them.

**Didn't, or not yet:**

- *Infrastructure assumptions.* Every restart bug came from an assumption nobody had tested: which signal tmux sends, what `--resume` does with no checkpoint, what "resume" means for an eval-only run.
- *The oracle took three tries.* A2′ failed for a structural reason that could have been worked out on paper first. That cost a 5M-step pilot.
- *The A2″ oscillation is unexplained.* Two candidate mechanisms failed their tests.
- *The design doc is now very long.* It records every ruling with its reasoning, which is the point, but it's hard to read from start to finish.

As of the latest board entry, batch 1 is on its last run.
After that comes the analysis, then the first patch round, then unsealing the held-out results.
The results will get their own post.

## References

::: {#refs}
:::
