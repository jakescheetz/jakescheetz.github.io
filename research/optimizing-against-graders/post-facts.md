# E1 post: fact sheet for the writer (researcher, 2026-11-11)

Source of truth: `paper.md` (draft 4). Where this sheet and the paper differ, the paper wins; tell me.
I fact-check the outline and the draft against the paper when they are on the board.

## What the experiment found (safe to say)

1. **We did not find the "optimize harder, then it turns bad" curve.** The pre-registered hypothesis was
   inconclusive in both batches. It was not refuted.
2. **Batch 1:** the agent was too weak to learn its own reward, so nothing happened. A null from a weak
   learner says nothing about the reward.
3. **Batch 2:** with a memory of visited squares, the agent's best proxy score was more than double
   batch 1's (about 121 against 55.5), and its true progress rose with its score during training. On its own start states at 5M steps: score 112.6 and 121.7 (random play 29.7);
   true progress 1.43 and 1.95 on a 0–3 scale.
4. **Two mild exploits appeared:** wandering into off-route maps (about 18% of the reward) and a detour to
   Route 22 (about 2%). Both were attributed to the reward and both reappeared on the held-out start
   states.
5. **The measurements were the weak part.** The truth-trained reference agent was unstable (13 of 40 late
   checkpoints collapsed), checkpoint scores swung a lot, and whether the reference beats the
   proxy-trained agent on true progress depends on whether collapsed checkpoints are dropped.
6. **One patch round.** We capped the exploration reward at 20 new squares per map. The cap was blunt: it
   cut reward almost everywhere (total reward fell to 0.17 of its old value on the same behavior).
   - True progress fell: 0.33 at 5M against 1.43 before the patch (3 seeds: 0, 0, 1.0).
   - Off-route wandering dropped to the level of random play in frequency, but its share of the agent's
     reward did not change (0.174 against 0.169 before the patch, same start state; random play
     0.122; corrected 2026-11-11, the 0.178 I first gave pools both start states). The patch shrank all exploration; it did not make it
     more on-route.
   - Grass grinding grew: 22% of the patched reward, on 2 of 3 seeds. Before the patch (batch 2) the agent did
     it in none of 560 test episodes. (Corrected 2026-11-11: an earlier version said 0.2%, which is batch 1's
     agent.)
7. **We stopped after one round, against our own pre-set rule,** and said so. The reason: a second patch
   stacked on a barely working learner could not tell "patched" from "starved".
8. **Cost:** about $70 of Google Cloud trial credit (an estimate, not a billed figure), about $5 of it for
   the patch round.

## Wording to avoid (each would be wrong)

- **"We tried to make the agent cheat."** We did not. We trained it on an ordinary exploration reward and
  watched whether it would exploit it. The placeholder slug says "tried to make a Pokémon agent cheat";
  please change it. Something like "we watched for cheating and mostly did not find it" is accurate.
- **"The agent cheated" / "broke the reward" / "reward hacking was found"** without the size. The exploits
  were mild and true progress went up, not down.
- **"The proxy is safe" / "the curve does not exist" / "disproved".** The result is inconclusive on one
  game segment, one reward family, one algorithm.
- **"The patch removed the exploit."** Frequency fell to random's level; share of reward did not change.
- **"The patch caused grinding."** Round 1 has no reference arm, and the truth-trained agent also grinds
  in batch 2. Say grinding grew after the patch on two of three seeds.
- **"The oracle" as if it were ground truth.** It is a truth-trained reference agent, and it was unstable.
- **"Beat the first gym" / "reached Brock".** The segment ends at Viridian Forest. No run got further.
- **Any claim that rests on one checkpoint.** Scores swing between checkpoints; use the per-checkpoint
  picture or say "at 5M".
- **"All three patched seeds stalled."** All three scored below random on their own reward at the 1M
  check and recovered in reward later. One seed never made true progress; one made it early only; one
  made it at four checkpoints.

## Things that were decided after seeing data (the post should not hide them)

- Batch 2 itself was a follow-up, not part of the original plan.
- The reading of the stopping rule that let round 1 run.
- Stopping after round 1.
- The paper's "Deviations" list has the rest.

## Not for the post

- No ROM-derived pixels, save states or screenshots in the repo (team rule). Screenshots need Jake's OK
  and live outside the repo.
- No citations I have not opened. The verified ones are in `sources.bib`; the paper says what each is
  cited for.
