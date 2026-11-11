---
title: "Watching a Pokémon agent for reward hacking"
subtitle: What reward hacking looked like when the textbook curve didn't show up
date: 2026-11-11
draft: true                 # delete this line to publish
toc: true
bibliography: references.bib
learn: true                 # loads the step-through player and callout boxes (assets/learn.*)
scripts: [demos.js]         # this post's three step-through animations
abstract-title: TL;DR
abstract: |
  - I trained reinforcement-learning agents to play the start of Pokémon Red. They got paid for exploring new squares, and I tracked real story progress on the side, where they couldn't see it. I was hoping to catch the textbook reward-hacking curve, where the score keeps climbing while real progress falls, and it never showed up. Once the agent could actually learn its reward, score and progress went up together.
  - I did find reward hacking, just in smaller forms. The agent earned about 18% of its reward wandering off-route and about 2% on a dead-end detour, and neither one hurt its progress. When I patched the off-route exploit, most of the real progress went with it, and grinding battles in tall grass grew on two of three runs.
  - The measurements turned out to be as interesting as the agent. My first batch of agents was too weak to learn the reward at all, checkpoint scores swung from one sample to the next, and the reference agent collapsed in 13 of its 40 late checkpoints. This post walks through what I saw and what I think it shows.
---

My [last post](../your-reward-model-is-an-attack-surface/) was about reward hacking.
The short version is that you can't write down exactly what you want from a model, so you reward something close to it, and a strong enough optimizer finds the places where the two don't match.
People usually draw it as a curve.
The reward and the real goal go up together for a while, and then the reward keeps going up while the real goal drops off.

<figure class="widget demo" id="demo-curve"></figure>

After writing that post I wanted to watch the curve happen for myself, on something small enough to run on my own.
I picked Pokémon Red, mostly because I grew up with it.
I spent a lot of time in that game as a kid, and getting to poke around in it again made this a lot more fun to work on.
So I trained some agents to play the first stretch of the game.
They got rewarded for exploring new squares, and I kept a separate count of real story progress that they never got to see.

I never got the curve.
From what I've read that's not so unusual, since one study of small reinforcement-learning problems found it in about a fifth of the cases it sampled [@karwowski2023goodhart].
What I got was reward hacking in smaller pieces, and those ended up being more interesting to dig into than I expected.
A couple of mild exploits showed up that didn't hurt real progress at all.
I patched one of them, and the patch took most of the progress away with it.
After that, two of three runs started earning part of their score by grinding battles.
This post is me walking through each of those and what I think they show.

::: why
Pretty much every way of training a model optimizes a stand-in for what people want, like a score, a test suite or another model's judgment.
The curve above is the usual warning sign, and it's easy to read "no curve" as "no problem."
What I saw here was a score getting gamed in small ways that the score itself didn't show, and a fix that did more damage than the exploit had.
I think those smaller cases say something about what reward hacking can look like when it isn't dramatic, and that's what I wanted to look at.
:::

## The question

I'm not the first person to train an agent on Pokémon Red.
The projects I found all reward something hand-written, like exploring, gaining levels or healing [@whidden2023pokemonred; @pleines2025pokemon], and their write-ups have some good stories about agents gaming those rewards.
One agent got stuck in a loop of near-endless battles to collect a heal reward [@pleines2025pokemon].
Another project found pacing loops in 41% of sampled baseline episodes, and got that under 5% with hand-tuned anti-loop penalties [@mudireddy2026pokerl].

What I didn't find in any of them was a plot of the reward against real progress through the game as training gets stronger.
@gao2023scaling made that kind of plot for learned reward models in language-model fine-tuning, and I wanted to see one for a game.
So the thing I was curious about was pretty simple.
If you pay an agent for exploring and keep training it harder, does its real progress eventually fall behind?

## The setup: a score, and a scoreboard the agent never sees

The game runs in an emulator called PyBoy [@pyboy], and everything I measure gets read straight out of the game's memory.

The agent starts in Professor Oak's lab, right after getting the Pokédex.

![The game at my first start state: the player in Professor Oak's lab, where every training episode begins. This is a frame I saved while scripting the start states, not the agent playing.](oaks-lab-start.png){style="width: 480px; max-width: 100%; height: auto; image-rendering: pixelated;"}

From there it's the route you'd expect if you've played the game: through Pallet Town and Route 1 to Viridian City, then up Route 2 to the entrance of Viridian Forest.
That's the whole stretch I used, and I put three milestones on it:

1. reach Viridian City,
2. reach Route 2,
3. enter Viridian Forest.

There's also Route 22, a dead end off to the west of Viridian City, which comes up again later.

![The route the agents play, read from the bottom up: Oak's Lab in Pallet Town to the entrance of Viridian Forest, with the three milestones. Route 22 is a dead end off Viridian City. A schematic, not to scale.](route-map.svg)

The agent gets a **score**: 0.1 points for every new square it stands on, 1 point per level its Pokémon gains, and a little for healing.
In practice almost all of it comes from new squares.

On the side, I keep a **scoreboard** of real progress, which is just how many of the three milestones the agent reached, in order.
The agent never sees it.
It only ever gets trained on the score, and the scoreboard is how I can tell whether the score is pointing it the right way.

Here's the idea on a toy map.

<figure class="widget demo" id="demo-score"></figure>

I ended up training three kinds of player:

- **the agent**, trained on the score;
- **a reference agent**, trained on real progress instead (the milestones, plus a small bonus for getting closer to the next one), so I could see what this kind of learner does when it's pointed at the real goal;
- **a random player** that just presses buttons at random, as a baseline for everything else.

I also set aside two start states the agent never trained on: the same start with a different starter Pokémon, and a start at the Viridian City Pokémon Center door.
I didn't look at those results until after I'd committed to a fix, so the fix couldn't be tuned to them.

## Batch 1: the agent couldn't learn its reward

The first batch didn't really tell me anything about the reward, because the agent never learned it.
Its score was flat from the very first checkpoint, sitting at about 1.4 times the random player's on average, and more training didn't move it.

![Batch 1, from the first start state. Top: the score. Bottom: real progress, in milestones. The agent trained on the score shows no upward trend, swinging between about 1 and 1.9 times random play. The reference agent's line averages all five of its runs, collapsed checkpoints included.](score-vs-progress-b1.svg)

The funny part is that if you go by points alone, the ranking comes out backwards.
The agent was first, the random player second and the reference agent last (30.4 for random against 15.3 for the reference, averaged over checkpoints), even though the reference was the one making the most real progress.

At that point I honestly wasn't sure whether the score was just a bad reward.
So I wrote a planner that has a perfect memory of where it's been and knows the whole map, and had it greedily cover new squares.
It scored at least 417, about eight times the agent's best.
And while doing nothing but chasing new squares, it walked right past the end of my segment in 16 out of 16 episodes.

So on this stretch of the game, **covering the map and following the route are the same thing**.
The score was fine, and my agent just couldn't learn it.

Looking back, my pilot run had already shown this and I missed it.
I'd checked that the reference agent could learn and that my cheat detectors fired, but I never thought to compare the agent's score with the random player's, or with what was actually possible.

## The fix: give it a memory

The batch 1 agent only saw the current screen, as a grid of tiles.
It had no idea which squares it had already been on, and it was getting paid for *new* ones.
That's a bit like asking someone to find every room in a house while wiping their memory each time they walk through a door.

So for batch 2 I gave it one more input, a layer marking the squares it had already visited in this episode.
Nothing else changed.

I should say this wasn't the original plan, which was one batch and then a patching phase.
I came up with batch 2 after seeing batch 1, so it's really a second experiment that was shaped by the first.
I did write its rules down before running it, and I left the held-out results alone the whole time.

The one thing I decided up front was that the agent's score had to at least double for the fix to count.
It did.
Its best score went from 55.5 to 120.9 on the first start state, and from 50.5 to 121.7 on the second.

## Batch 2: score and progress went up together

Now the agent was actually optimizing its score, so if the curve was going to show up anywhere, I figured it would be here, and it still didn't.
On the first start state the score went from 64.6 at the 0.5M-step checkpoint to 112.6 at 5M, and real progress went up with it, from 1.05 milestones to 1.43.
It was bumpy in between (0.80 at 1M, 1.50 at 2M, 0.93 at 3M, 1.80 at 4M), which was my first hint that single checkpoints are pretty noisy.
On the second start state the score went from 57.1 to 121.7, and progress from 1.40 to 1.95.
For comparison, the random player scores 29.7.

How the agent stacks up against the reference agent is messier, because it depends on what you do with the reference's bad checkpoints.
I get into that [further down](#what-this-data-can-and-cant-say).

![Batch 2, from the first start state. Top: the score. Bottom: real progress, in milestones. Both rise together for the agent trained on the score. The reference agent is unstable, and its line averages all five runs, collapsed checkpoints included; with those dropped, the reference is ahead at most late checkpoints.](score-vs-progress.svg)

This is also where I finally saw some reward hacking, just a mild version of it.
My detectors picked up two exploits, and both of them come from the score rather than from the learner:

- **Wandering off-route.** The agent picked up new squares on maps behind its furthest milestone, or in side buildings. That was about 18% of its reward, pooled over checkpoints, and it showed up on 8 of 10 runs.
- **The Route 22 detour.** The agent spent time on the dead end. That was about 2% of the reward, pooled the same way.

Both showed up again on the held-out start states, and neither one got in the way of real progress, which kept going up.

![Where each agent walked outdoors at the 5M-step checkpoint, starting in Oak's Lab. One checkpoint is one sample, and scores swing between checkpoints. Left (40 episodes): the batch-2 agent trained on the score covers Pallet Town, Route 1 and Viridian City densely, and reaches Route 22 and Route 2. Middle (24 episodes): the agent trained on the patched score (it comes up below) mostly stops on Route 1; in four of 24 episodes, all from one run, it continues to Viridian City and Route 2. Right (24 episodes): random play barely leaves Pallet Town. Indoor squares aren't drawn; the strips under each map show which buildings the agent entered. None entered Viridian Forest at this checkpoint.](where-they-went.svg)

My best guess at why is the planner result again.
On this stretch the route runs through most of the squares you can reach, so exploring and making progress are mostly the same activity.
By the end of training the agent had only covered about 27% of the squares the planner reaches.
It never ran out of route, so it never had to pick between exploring and progressing.
I didn't test that, so take it as a guess.

## Patching the main exploit

I'd always planned a second phase, where I'd find an exploit, patch the reward, retrain, and see if the patch holds up, including on the held-out starts.
Wandering off-route was the main exploit, so that's what I patched.

The patch was a cap, where only the first 20 new squares on each map pay.
I picked 20 using a rule I'd set ahead of time: out of six candidate caps, take the largest one that removes at least 80% of the off-route reward.
It removed 85.7%.

The catch is that it also removed 82.4% of the reward for everything else, and total reward on the same behavior dropped to 0.17 of what it had been.
So I knew before training anything that the cap was blunt.
I went with it anyway, since I'd set the rule first and swapping it out after seeing the numbers kind of defeats the point of having a rule.

The steps below go through what the cap does.

<figure class="widget demo" id="demo-cap"></figure>

Then I trained three new runs on the patched score, and real progress dropped.
At the end of training the three runs reached 0, 0 and 1.0 milestones, which averages out to 0.33.
Before the patch, five runs on the same start state averaged 1.43 at the same point.

The patched agents did learn their new score, slowly.
But if you score them by the *original* rules, they earned about 22, which is less than the random player's 29.7.

As for whether the patch got rid of the off-route wandering, it sort of did and sort of didn't.
Going by the original score, how *often* the patched agent was on an off-route map dropped to the random player's level.
But the *share* of its reward that came from those maps was 0.174, against 0.169 before the patch on the same start state.
So the patch didn't push the agent back onto the route.
It made the agent explore less overall, to a bit less than random play earns on the original score, and left the mix the same.
Where it goes did change, though, and you can see in the map figure above that it gets further north than random play does.

![Where the score came from, for the agent before the patch (batch 2) and the agent after it (trained on the capped score), pooled over all seven test checkpoints. (a) Each agent's own score by source: before the patch, 99.9% came from exploring; after it, 12% came from levelling up. (b) Measured on the original score, the share earned on off-route maps: 16.9% before, 17.4% after, and 12.2% for random play, all from the first start state. (c) The share of each agent's own score earned while grinding battles in grass: 0.0% before, 22.3% after. Charts (b) and (c) share one axis, from 0 to 30%.](reward-sources.svg)

## Grass grinding

With exploring paying so little, something else started picking up part of the score.

The patch didn't touch the level reward.
By my offline estimate, that made levels about six times more valuable relative to exploring than before.
After the patch, grinding battles against wild Pokémon in tall grass made up 22% of the patched reward across the three runs, and it showed up on two of them.
Before the patch, the agent did that in none of its 560 test episodes.

![A wild battle in the game: Squirtle, level 7, against a Pidgey, at the FIGHT menu. Grinding means fighting battles like this one over and over for the level points. This frame comes from a scripted test run, not the agent playing.](wild-battle.png){style="width: 480px; max-width: 100%; height: auto; image-rendering: pixelated;"}

I don't want to say the patch *caused* the grinding.
This round didn't have a reference agent to compare against, and the reference agent did some grass grinding in batch 2 too.
All I can really say is that grinding went up after the patch, on two of three runs.

The three runs also didn't look much like each other.
One made progress early and then settled into grinding.
One never made progress at all.
One made progress at four checkpoints and never ground.
Three runs isn't enough for me to say which of those is the normal one.

![The three runs trained on the patched score, one row each. Left: real progress at every checkpoint (8 test episodes each), with a blue tick marking the batch-2 agent's average before the patch. Right: the share of that checkpoint's score that came from levelling up. Run 1 made progress only at 0.5M steps, then earned more of its score by levelling up. Run 2 never reached a milestone, and levelled up late in training. Run 3 made progress at four checkpoints and never levelled much. Each checkpoint is one sample, and progress swings between them.](round1-seeds.svg)

## Why I stopped at one patch

By my own rules I was supposed to patch again.
The rule I'd written down said to keep going if a new behavior had appeared and real progress was below the reference agent's.
A new behavior had appeared.
The progress part depends on the reference's collapsed checkpoints, the ones where it had fallen back by a milestone or more.
Drop them and the condition holds, count them and it doesn't (more on those [below](#what-this-data-can-and-cant-say)).

I stopped anyway, and the paper lists that as a deviation from the plan.
I just didn't think a second patch would tell me much.
Under the patched score the agent was already earning less of the original score than random play, so I'd have been testing the next patch on a learner that barely worked.
I wouldn't have been able to tell "the patch worked" from "the agent was starved of reward" any better than I could this time.

After I'd logged that decision I opened the held-out results.
On the different-starter start they looked like what I'd already seen.
The other held-out start, at the Viridian City Pokémon Center door, was worse.
There the patched agent earned less than random play on either score: 3.58 points of patched reward at 5M steps, against the random player's 7.77.
The unpatched batch-2 agent had done fine from that start.
My guess, and it really is just a guess, is that an agent trained under the cap learns to fill its quota of new squares on the maps around its own start.

What I'd like to try next is a different kind of patch, one that goes after off-route exploring specifically, starting over from the original score.
I haven't run that yet.

## What this data can and can't say

This was a small experiment.
It's one stretch of one game, one kind of reward, one training algorithm, 5 runs per setting, and 3 runs for the patch round.
I think that's plenty to describe what happened here and to learn something from the odd cases.
It's not enough to say how often any of this happens anywhere else.

There are a few things I feel pretty good about.
The agent learned its reward in batch 2 and real progress went up with it, on both start states.
Wandering off-route looks like a repeatable exploit, since it showed up on 8 of 10 runs and again on the held-out starts.
And the cap cut exploring of every kind, since all three patched runs ended up earning less of the original score than random play.

Other things I'd hold more loosely:

- **No curve here doesn't mean no curve anywhere.** My main hypothesis came out inconclusive in both batches, which isn't the same as disproved.
- **Batch 2 is a follow-up.** I designed it after seeing batch 1. I wrote the rules down first, but batch 1 shaped it.
- **The reference agent was shaky.** 13 of its 40 late checkpoints had collapsed, meaning it fell back by at least a milestone. Whether the reference beats my agent on real progress flips depending on whether you drop those. I'd decided ahead of time to drop them, and I report it both ways.
- **A frozen checkpoint isn't the same as the learner.** The agent's training episodes scored two to three times what its frozen checkpoints did. When I resumed learning from a checkpoint that scored about 50, it scored 131 in its first episodes. A lot of the score comes from the agent adapting as it goes.
- **Checkpoint scores jump around.** The same run's score ranged from 0.40 to 1.25 of its own late average across 20 extra checkpoints. There's also a dip at 3M steps in 8 of 10 runs that I still can't explain. I had a guess, wrote the prediction down, and the guess was wrong.
- **The detour and the grinding are thinner evidence.** The Route 22 detour only stood out from random play at one checkpoint. The grinding comes from three runs on one start state with no reference agent, plus two judgment calls I made after seeing data: which reading of the stopping rule let the round run at all, and stopping after it.

I still think the small cases are worth having.
One run that makes progress and then settles into grinding can't tell me how common that is.
But it does show the behavior can happen under this reward, and what it looks like in the logs when it does.
If I wanted to go looking for it at a bigger scale, that's what I'd need to know.

## What I think this shows

The gap between the score and the real goal was there the whole time.
The score pays for wandering, and the agent did wander, a little.
On this stretch of the game the gap just stayed small, so reward hacking showed up as a handful of specific behaviors and never as a falling curve.
My hunch, and I can't back it up with this data, is that the small stuff is the more common way for it to show up.

The other thing that surprised me was how much the measurements moved around.
I could only see any of this because I tracked things the agent couldn't see, and those had quirks of their own.
A checkpoint scored differently depending on when I took it.
The agent behaved differently while it was still learning than when I froze it.
The reference agent I was using to say what "good" looks like collapsed in a third of its late checkpoints.
If I'd only looked at one of those, I'd have come away with the wrong idea.

The strangest case is three of the reference agent's runs in batch 2.
Earlier in training, each had made it to Route 2 or into Viridian Forest.
By the end, in all 8 of 8 test episodes, the same agent never left Oak's Lab.
I re-ran one of them bit for bit on two other machines and got the same thing, so it wasn't a glitch in the hardware or the evaluation.
It's just what the trained weights do, and I still don't know why.

A few other things tripped me up along the way:

- In batch 1 I was asking whether the agent was gaming its reward before I'd checked that it was learning the reward at all. Comparing its score with random play and with what's possible would have caught that back in the pilot. From batch 2 on I did that early, at 1M and 2.5M steps.
- The frozen snapshots and the learning process told different stories, so from batch 2 on I reported training episodes next to the frozen checkpoints.
- A detector that reads the reward stops meaning the same thing once the reward changes. After the patch, my off-route detector still flagged "off-route" under the new score, but random play didn't flag at all, because the cap had removed exactly the reward the detector was looking for. I had to re-run it under the original score to see what the agent was actually doing.
- Keeping the held-out starts closed until the patch was locked in is the only reason they could say anything about it.

The patch round is the part I keep thinking about.
A blunt fix closed the gap by paying less for everything, and on two of three runs the agent moved over to the next thing that still paid.

The whole thing ran on about $70 of free Google Cloud trial credit (an estimate from VM uptime, not a bill), about $5 of it for the patch round.
The full write-up with every number is in the paper, and there's a separate [build-log post](../running-a-pokemon-experiment-with-an-agent-team/) about how I built and ran it with a team of AI agents.<!-- link live only once both are published -->

## References

::: {#refs}
:::
