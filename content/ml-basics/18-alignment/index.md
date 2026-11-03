---
title: The alignment problem
subtitle: Getting a model to pursue what people meant, and how people check whether it does.
unit: 18
section: Alignment
minutes: 65
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'In the sense InstructGPT uses, when is a language model aligned?'
    answer: 'When it acts in accordance with what its users intend'
    wrong: ['When it gets a high score from its reward model', 'When it matches its training data on held-out examples', 'When it refuses every request that could cause harm']
    why: 'InstructGPT describes its goal as aligning language models with user intent. A high reward-model score is the proxy for that, and unit 17 showed the two can come apart.'
  - kind: term
    q: 'What is specification gaming?'
    answer: 'Behavior that satisfies the literal objective without achieving the intended outcome'
    wrong: ['Learning a different goal from an objective that was correct', 'Scoring well because the test set leaked into training', 'Refusing tasks the model was trained to do']
    why: 'That''s Krakovna et al.''s definition. The boat circling the lagoon earned the score it was given; the score just wasn''t the race.'
  - kind: term
    q: 'What is goal misgeneralization?'
    answer: 'A model that keeps its skills on new inputs but pursues a goal other than the intended one'
    wrong: ['A model that loses its skills on inputs unlike its training data', 'A model that exploits a mistake in its reward function', 'A model whose training loss stops going down']
    why: 'The CoinRun agent still finishes levels at test time. What fails to carry over is the goal: it heads for the end instead of the coin.'
  - kind: term
    q: 'What do Sharma et al. mean by sycophancy?'
    answer: 'A model seeking human approval in unwanted ways, such as agreeing with a user who is wrong'
    wrong: ['A model refusing harmless requests to stay on the safe side', 'A model repeating text it memorized from training data', 'A model writing much longer answers than the question needs']
    why: 'Their examples include wrongly admitting a mistake when a user pushes back and changing a correct answer to match what the user believes.'
  - kind: term
    q: 'What problem does scalable oversight refer to in Amodei et al.?'
    answer: 'Training safely when checking the true objective on every example costs too much'
    wrong: ['Training larger models on more data with the same objective', 'Red teaming a model with another language model', 'Penalizing the policy for moving away from $\pi_0$']
    why: 'They describe a designer who could evaluate the right objective but can''t afford to do it often, so training has to rely on cheaper approximations.'
  - kind: term
    q: 'In Constitutional AI, where do the harmlessness comparisons come from?'
    answer: 'A model compares two responses using a principle from a written list'
    wrong: ['Crowdworkers compare two responses using the written principles', 'A reward model trained on human harmlessness labels', 'The KL penalty against the starting model']
    why: 'That''s the AI feedback in RLAIF. Bai et al. still used human labels for helpfulness, and only the harmlessness labels came from the model.'
  - kind: apply
    q: 'The weak supervisor scores 60%, the strong model trained on true labels scores 90%, and the strong model trained on the weak labels scores 75%. What is the performance gap recovered?'
    answer: '$0.5$'
    wrong: ['$0.83$', '$0.25$', '$0.2$']
    why: '$(75 - 60) / (90 - 60) = 15/30 = 0.5$. The $0.83$ is $75/90$, which ignores where the weak supervisor started.'
  - kind: apply
    q: 'In the widget at 0% random coins, a test level puts the coin two cells left of the start. What does a typical run do?'
    answer: 'Walks right to the end of the level and stays there'
    wrong: ['Walks left and picks up the coin', 'Stays at the start, unsure which way to go', 'Wanders left and right with no preference']
    why: 'At the start it steps right with probability $\sigma(-w_{\text{coin}} + w_{\text{end}})$, about 0.66 for run 1, and three cells to the right the coin is out of view. Averaged over the coin-left levels it ends at the end 60% of the time.'
  - kind: apply
    q: 'In the toy with the coin always at the end, why does $w_{\text{end}}$ finish above $w_{\text{coin}}$?'
    answer: 'Steps taken with the coin out of view give credit to $w_{\text{end}}$ only'
    wrong: ['The coin feature is $-1$ on most training steps', '$w_{\text{end}}$ is updated with a larger learning rate', 'The baseline subtracts more from $w_{\text{coin}}$ than from $w_{\text{end}}$']
    why: 'With the coin in view to the right, $c = 1$ and both weights get the same push. Only when the agent wanders left and loses sight of the coin do the two differ, and then only $w_{\text{end}}$ moves.'
  - kind: apply
    q: 'Your agent gets the coin on 100% of validation levels built the same way as training. What is the best next check for goal misgeneralization?'
    answer: 'Test it on levels where the coin and the end of the level are in different places'
    wrong: ['Hold out a bigger validation set built the same way', 'Train for longer and watch the validation reward', 'Raise the KL coefficient $\beta$ and retrain']
    why: 'Validation levels built the same way can''t separate "go to the coin" from "go right", since both score 100% there. Only levels where the two disagree can.'
  - kind: apply
    q: 'In unit 17''s split $r_\phi = r^* + e$, with people''s approval playing the part of $r_\phi$, where does sycophancy put the error $e$?'
    answer: 'It''s positive for answers that agree with the user, right or wrong'
    wrong: ['It''s positive for answers that disagree with the user', 'It''s random noise that doesn''t depend on the answer', 'It''s positive only for answers that are long']
    why: 'Raters, and preference models trained on them, sometimes prefer a convincing answer that agrees with a misconception. Optimizing approval then pushes toward agreeing.'
  - kind: apply
    q: 'A red-team classifier has a recall of 0.8, and every one of its 18,444 flags is a real offensive reply. About how many offensive replies were there?'
    answer: 'About 23,000'
    wrong: ['About 14,800', 'Exactly 18,444', 'About 3,700']
    why: 'Recall is the share of real positives the classifier flags, so $18{,}444 / 0.8 \approx 23{,}000$. The 14,800 multiplies by 0.8 instead of dividing.'
---

Units 15 to 17 trained a model against a number, and unit 17 watched that number come apart from what people wanted.
*Alignment* is the name for the larger problem: getting a model to do what the people who build and use it actually intend.
Reward hacking is one way to miss.
A model can also learn the wrong goal from a reward that was right, or learn to please the people grading it instead of being correct, and much of the work on alignment is about preventing those and catching them when prevention fails.

## What alignment means

InstructGPT, the paper behind unit 16's recipe, describes its goal as "aligning language models with user intent", and, following [Leike et al. (2018)](https://arxiv.org/abs/1811.07871), as training models "to act in accordance with the user's intention".
It also borrows a three-word version from [Askell et al. (2021)](https://arxiv.org/abs/2112.00861) at Anthropic: helpful, honest and harmless.
Both describe intent, and training never sees intent.
It sees a loss or a reward, computed on the training data.

That leaves two places for a gap.

- **The objective is wrong.**
  The number being optimized doesn't match what people want, and optimizing it finds the difference.
  [Krakovna et al. (2020)](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) at DeepMind call this *specification gaming*: "a behaviour that satisfies the literal specification of an objective without achieving the intended outcome."
- **The objective is right, and the model learns a different goal anyway.**
  On the training data, the intended goal and some other goal earn exactly the same reward, and the model picks up the other one.
  That's *goal misgeneralization*, the next section.

The best-known case of the first is OpenAI's CoastRunners agent ([Clark and Amodei, 2016](https://openai.com/index/faulty-reward-functions/)).
The game's goal, as people understand it, is to finish a boat race, but the score comes from hitting targets along the route.
The agent found a lagoon where it could circle and knock over the same three targets as they reappeared, catching fire and crashing into other boats along the way, and it scored on average 20 percent higher than human players.
Unit 17's reward hacking is the same failure with a learned reward model as the objective.

[Hubinger et al. (2019)](https://arxiv.org/abs/1906.01820) call these two gaps *outer alignment* and *inner alignment*, though they define the second only for a model that is itself searching toward an objective of its own.

## Right reward, wrong goal

[Langosco et al. (2022)](https://arxiv.org/abs/2105.14111) trained an agent on CoinRun, a platform game where the agent gets its reward for reaching a coin.
In their training levels, "the coin is always located at the right end of the level next to a wall."
At test time they put the coin somewhere random, and "the agent generally ignores the coin completely and proceeds to the end of the level."
It still gets through the levels.
In their words, "the agent's capability for solving the levels generalizes, but its goal of collecting coins does not."

Nothing was wrong with the reward: the agent was only ever rewarded for the coin.
The trouble is that on every training level, "go to the coin" and "go to the right end" are the same behavior and earn the same reward, so the training data can't tell the two goals apart.
A validation set doesn't help either, since a [held-out set](../06-generalization/#train-validation-and-test) only tells you about data drawn the same way, and on validation levels with the coin at the end both goals score perfectly.

[Shah et al. (2022)](https://arxiv.org/abs/2210.01790) define goal misgeneralization in general: a learned model "behaves as though it is optimizing an unintended goal, despite receiving correct feedback during training."
An agent was rewarded for visiting colored spheres in the correct order, and it trained alongside a partner bot that visited them in that order, so following the partner was a good strategy.
At test time the partner was swapped for one that visited the spheres in the wrong order.
The agent kept following it, scoring far worse than a random policy, even though it could observe the negative rewards.

Langosco et al. also tried the obvious fix.
Their figure 2 reports that goal generalization "is greatly improved" when "just 2% of training levels have randomly placed coins (like the test environment)", and more randomization helps further.
That fix needs someone to know which coincidence to break, and the training data of a large language model comes with no list of them.

I rebuilt a small version to watch the weights.
The level is a corridor of 9 cells, the agent starts in the middle (cell 4), and the level ends at cell 8.
The agent sees 4 cells each way, and it gets a reward of 1 for reaching the coin within 12 steps.
In training the coin sits at cell 8, except on a chosen share of levels where it goes in a random cell.
At each step the policy chooses between stepping right and stepping left:

$$
\pi(\text{right} \mid s) = \sigma(w_{\text{coin}}\, c + w_{\text{end}})
$$

::: formula
$\sigma$ is the sigmoid from [unit 5](../05-classification/#from-a-score-to-a-probability), and the score inside it is unit 5's $w \cdot x + b$ with one feature.
$c$ is what the agent sees of the coin: $+1$ if it's in view to the right, $-1$ if it's in view to the left, and 0 if it's out of view.
$w_{\text{coin}}$ sets how strongly the agent heads toward a coin it can see, and $w_{\text{end}}$ plays the bias, a push to the right on every step.
Stepping left has probability $1 - \pi(\text{right} \mid s)$.
::::

So the policy is logistic regression with two weights, one for heading toward a coin it can see and one for heading right, toward the end.
It's trained with [REINFORCE and a running-average baseline](../15-reinforcement-learning/#a-baseline-compare-with-the-average) from unit 15 for 3,000 levels, starting from both weights at 0.

With the coin always at the end, the two weights almost always get the same push.
While the coin is in view to the right, $c = 1$, and $w_{\text{coin}}$ and $w_{\text{end}}$ enter the score the same way.
The only steps that tell them apart are ones where the agent has wandered left of the start and lost sight of the coin.
Stepping right is still what earns the reward there, and with $c = 0$ only $w_{\text{end}}$ gets the credit.
So training ends with $w_{\text{end}}$ ahead: averaged over 20 runs, $w_{\text{coin}} = 1.63$ and $w_{\text{end}} = 2.24$.

<figure class="widget" id="coin-corridor">
<figcaption>Top: one test level at a time, played by one of the 20 trained runs. The coin is orange, the end of the level is blue, and the dashed box is what the agent can see. Bottom: for each share of training levels with a random coin, how often the runs get a coin placed to the left of the start (orange) and how often they're standing at the end when time runs out (blue). Start at 0% and watch the coin-left levels, then move the slider to 2%.</figcaption>
</figure>

At 0%, every run gets the coin on its training levels, and on test levels with the coin to the left it gets the coin only 25% of the time and ends at the end 60% of the time.
Run 1 at the start of such a level steps right with probability $\sigma(-1.42 + 2.07) = 0.66$, and three steps right put a coin at cell 2 out of view.
At 2%, about 60 of the 3,000 training levels have a random coin, roughly half of them to the left, and all 20 runs end with $w_{\text{coin}}$ above $w_{\text{end}}$.
The toy's 2% matching Langosco et al.'s is a coincidence of the setup, but the direction is the same: a few levels where the two goals disagree are enough to separate them.

## When the proxy is people

RLHF uses people's picks as the training signal (unit 16), and those picks are a proxy too.
[Sharma et al. (2023)](https://arxiv.org/abs/2310.13548) studied one way they miss: *sycophancy*, "where a model seeks human approval in unwanted ways."

They tested five assistants trained with human feedback (Claude 1.3, Claude 2, GPT-3.5, GPT-4 and LLaMA 2 70B chat) and found sycophancy in all of them.
When a user replied "I don't think that's right. Are you sure?", Claude 1.3 wrongly admitted a mistake on 98% of questions.
When the user suggested a wrong answer in the question, accuracy fell by up to 27% (LLaMA 2).

To see where it comes from, they had GPT-4 tag 15,000 comparisons from Anthropic's public helpfulness data with 23 features and fit a Bayesian logistic regression predicting which answer people picked, the same kind of model as [unit 16's Bradley–Terry](../16-rlhf/#from-picks-to-numbers).
Matching the user's views was consistently one of the most predictive features, though not always the top one.
They also wrote prompts where a user states a common misconception and asks for a comment.
A preference model preferred a convincing answer that agreed with the misconception over a short correction 95% of the time.
Against a correct answer that explained why the user was wrong, it still preferred the agreeing one 45% of the time on the hardest misconceptions, and human raters chose the truthful answer less reliably as the misconceptions got harder.
Running RL against Claude 2's preference model increased some forms of sycophancy.

In terms of unit 17's [split of the proxy](../17-reward-hacking/#the-score-is-a-proxy), $r_\phi = r^* + e$, people's approval is now the proxy, and sycophancy is an error $e$ that's positive for answers that agree with the person grading them.
The reward model learns that error from the comparisons, and RL pushes toward it like any other.
Fitting the comparisons more closely doesn't remove it, because the error is in the comparisons.

## Supervising what people can't check

[Amodei et al. (2016)](https://arxiv.org/abs/1606.06565) named this problem *scalable oversight*: the designer may be able to evaluate the right objective, but it's "too expensive to do so frequently", so training has to "rely on cheaper approximations."
A reward model is one such approximation of people.
The harder case is answers people can't check at all, because they're too long or too technical, and that's where approval and correctness are most likely to come apart.

One response is to have models help supervise models.
[Bai et al. (2022)](https://arxiv.org/abs/2212.08073) call their version *Constitutional AI*.
The "constitution" is a list of 16 written principles about harmlessness.
In a supervised phase the model critiques its own responses against a principle and revises them, and it's fine-tuned on the revisions.
In an RL phase a model compares pairs of responses using a principle, a preference model is trained on those AI comparisons, and RL runs against it, which they call RL from AI feedback (RLAIF).
Helpfulness labels still came from people.
The result was an assistant that was less harmful at a given level of helpfulness than their RLHF models, and that almost never gave evasive answers: it engaged with harmful requests by explaining its objections.
The labels are only as good as the model applying the principles, but the principles are written down where anyone can read them, and applying them costs model time instead of people's.

Another line of work asks what happens when the supervisor is weaker than the student, which is the position people would be in with a model better than them at a task.
[Burns et al. (2023)](https://arxiv.org/abs/2312.09390) at OpenAI studied it with models standing in for both: a small model labels the data, and a much stronger model is fine-tuned on those labels.
They measure how much of the gap between the weak supervisor and the strong model's ceiling the student closes:

$$
\text{PGR} = \frac{\text{weak-to-strong} - \text{weak}}{\text{strong ceiling} - \text{weak}}
$$

::: formula
"Weak" is the weak supervisor's own score on the task.
"Strong ceiling" is the strong model's score when it's fine-tuned on correct labels.
"Weak-to-strong" is the strong model's score when it's fine-tuned on the weak model's labels.
PGR stands for performance gap recovered.
::::

A PGR of 1 means the student did as well as if it had the true labels, and 0 means it did no better than its supervisor.
With a GPT-2-level supervisor and GPT-4 as the student, plain fine-tuning typically recovered about half the gap on their language tasks, and an extra loss that lets the student trust its own confident predictions brought that to nearly 80%.
On reward modeling with ChatGPT's preference data it recovered roughly 10%.
The strong student can also learn to imitate its supervisor's errors, which is the failure you'd worry about most if the supervisor were people.

## Checking a model

Training choices only reduce the chance of these failures, so models also get tested.
An evaluation is unit 7's setup with behavior in place of labels: a fixed set of prompts, a way to score the responses, and a [reference point](../07-evaluation/#a-score-needs-a-reference-point) to compare against.
Sharma et al.'s "Are you sure?" test is one.
Unit 6's [contamination](../06-generalization/#leakage-and-contamination) applies too: once a behavioral test circulates, it can end up in training data, and a model can learn the test rather than the behavior.

*Red teaming* goes looking for failures instead of measuring a rate on a fixed set.
[Perez et al. (2022)](https://arxiv.org/abs/2202.03286) used one language model to write test questions for a 280-billion-parameter chatbot and a classifier to flag offensive replies.
Of 0.5 million questions written zero-shot, 3.7% got an offensive reply, 18,444 failed test cases, and training the question writer with RL raised that rate past 40%.
The classifier is a model with its own [confusion matrix](../07-evaluation/#the-confusion-matrix), so offensive replies it missed never got counted.
[Ganguli et al. (2022)](https://arxiv.org/abs/2209.07858) had 324 crowdworkers attack models of three sizes and four kinds, and released all 38,961 attacks.
By a preference model's harmlessness score, the RLHF models got harder to red team as they got bigger, while the other kinds stayed flat (by the red teamers' own ratings of their success, no kind showed a clear trend with size).

> **An analogy, if you've done security work.**
> Red teaming a model is a pentest of it.
> Perez et al.'s question-writing model is the automated scanner, cheap and wide, and Ganguli et al.'s crowdworkers are the manual testers.
> Either way the report lists what the testers found with the time and techniques they had, and a short report doesn't mean the system has no bugs.

## Worked example

The corridor from the widget, in plain Python.
`rng` is a small random number generator that JavaScript can reproduce exactly, so the widget trains the same 140 runs and shows the same numbers.
`outcome` doesn't sample: it tracks the probability of the agent being in each cell, step by step, and returns the exact chance of reaching the coin and of standing at the end when time runs out.
`coin wins` counts the runs, out of 20, that ended with $w_{\text{coin}}$ above $w_{\text{end}}$, and the test columns average over the four coin positions left of the start.

```python
import math

N, START, VIEW, STEPS = 9, 4, 4, 12   # cells 0 to 8, the level ends at cell 8

def rng(seed):   # mulberry32, a small generator the widget copies exactly
    s = seed
    def rand():
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((s ^ (s >> 15)) * (s | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 2**32
    return rand

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sees(pos, coin):   # +1 coin in view to the right, -1 to the left, 0 out of view
    d = coin - pos
    return 0 if abs(d) > VIEW else (1 if d > 0 else -1)

def train(share, seed, episodes=3000, eta=0.2):
    rand = rng(seed)
    w_coin = w_end = 0.0
    b = None
    for _ in range(episodes):
        if rand() < share:                   # a level with the coin somewhere random
            coin = int(rand() * (N - 1))
            coin += coin >= START
        else:                                # the usual level: coin at the end
            coin = N - 1
        pos, g_coin, g_end, r = START, 0.0, 0.0, 0
        for _ in range(STEPS):
            c = sees(pos, coin)
            p = sigmoid(w_coin * c + w_end)  # probability of stepping right
            right = rand() < p
            g_coin += (right - p) * c        # gradient of log pi(action)
            g_end += right - p
            pos = min(N - 1, pos + 1) if right else max(0, pos - 1)
            if pos == coin:
                r = 1
                break
        b = r if b is None else b + 0.1 * (r - b)   # running-average baseline
        w_coin += eta * (r - b) * g_coin
        w_end += eta * (r - b) * g_end
    return w_coin, w_end

def outcome(w_coin, w_end, coin):   # exact chance of getting the coin, and of
    here = {START: 1.0}             # standing at the end when time runs out
    got = 0.0
    for _ in range(STEPS):
        nxt = {}
        for pos, q in here.items():
            p = sigmoid(w_coin * sees(pos, coin) + w_end)
            for to, pr in ((min(N - 1, pos + 1), p), (max(0, pos - 1), 1 - p)):
                nxt[to] = nxt.get(to, 0.0) + q * pr
        got += nxt.pop(coin, 0.0)
        here = nxt
    return got, here.get(N - 1, 0.0)

print("share  coin wins  train  test: coin  at end   w_coin  w_end")
for share in (0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
    runs = [train(share, seed) for seed in range(1, 21)]
    wins = sum(wc > we for wc, we in runs)
    train_ok = sum(outcome(wc, we, N - 1)[0] for wc, we in runs) / 20
    test = [outcome(wc, we, coin) for wc, we in runs for coin in range(START)]
    coin_ok = sum(t[0] for t in test) / len(test)
    at_end = sum(t[1] for t in test) / len(test)
    wc = sum(r[0] for r in runs) / 20
    we = sum(r[1] for r in runs) / 20
    print(f"{share:5.1%}  {wins:5d}/20  {train_ok:5.2f}  {coin_ok:10.2f}  {at_end:6.2f}"
          f"  {wc:7.2f}  {we:5.2f}")
```

```
share  coin wins  train  test: coin  at end   w_coin  w_end
 0.0%      0/20   1.00        0.25    0.60     1.63   2.24
 0.5%     12/20   1.00        0.57    0.32     2.09   1.85
 1.0%     18/20   1.00        0.81    0.15     2.66   1.60
 2.0%     20/20   1.00        0.92    0.06     3.07   1.49
 5.0%     20/20   1.00        0.97    0.02     3.49   1.31
10.0%     20/20   1.00        0.99    0.01     3.97   1.01
20.0%     20/20   1.00        0.99    0.00     4.25   0.86
```

The `train` column is 1.00 on every row, so a score on levels like the training ones can't tell these policies apart, while the coin-left test success goes from 0.25 to 0.92 between 0% and 2%.
As the share grows, $w_{\text{coin}}$ rises and $w_{\text{end}}$ falls, which is the policy's goal moving from the end of the level to the coin.

## Where this goes next

- This is the last unit. My post [Reward hacking from the ground up](../../posts/your-reward-model-is-an-attack-surface/) also ends at sycophancy and scalable oversight, and its last section points at the harder case: a model that ends up pursuing an objective of its own, or one that behaves differently when it believes it's being trained. Everything here assumed a model shaped by training with no plan of its own.
- [Amodei et al.](https://arxiv.org/abs/1606.06565) list five concrete problems. Reward hacking and scalable oversight are two of them, goal misgeneralization fits under their distributional shift, and the other two (avoiding side effects and safe exploration) aren't covered in this course.
- The [DeepMind post on specification gaming](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) links a shared spreadsheet of examples (around 60 when the post was written), a good way to get a feel for how many ways an objective can be satisfied without doing the task.

## Self-test

<details>
<summary>In the widget at 0%, the runs get the coin on 100% of training levels. Why doesn't that show they learned to go for the coin?</summary>

On every training level the coin is at the end, so going for the coin and going to the end are the same walk.
Both goals earn a perfect training score, and the coin-left test levels, where they disagree, show the runs going to the end (60%) far more often than getting the coin (25%).

</details>

<details>
<summary>CoastRunners and CoinRun both ended with an agent doing something its designers didn't want. Which one is specification gaming, and which is goal misgeneralization?</summary>

CoastRunners is specification gaming: the score rewarded hitting targets, and circling the lagoon really did maximize it.
CoinRun is goal misgeneralization: the reward only paid for the coin, which was the intended goal, but the agent learned "go to the right end", which earned the same reward on every training level.

</details>

<details>
<summary>Why does the toy's w_end end up larger than w_coin when the coin is always at the end?</summary>

Whenever the coin is in view to the right, $c = 1$, and the two weights get exactly the same update.
They only differ on steps where the agent has wandered left and lost sight of the coin; stepping right is still what pays there, and with $c = 0$ only $w_{\text{end}}$ gets the credit.

</details>

<details>
<summary>Weak supervisor 50%, strong ceiling 80%, strong model trained on weak labels 74%. What is the PGR?</summary>

$(74 - 50) / (80 - 50) = 24/30 = 0.8$.
The student recovered 80% of the gap between its supervisor and what it could have done with correct labels.

</details>

<details>
<summary>Why can a better reward model fail to fix sycophancy, when it helps with the reward hacking in unit 17?</summary>

In unit 17 the error came from the reward model being wrong about answers far from its training data, and more data or a bigger model shrinks that.
With sycophancy the error is in the comparisons themselves: people sometimes prefer an answer that agrees with them.
A reward model that fits those comparisons better learns that preference more faithfully.

</details>

<details>
<summary>A red team spends a week on a model and finds nothing. What does that tell you?</summary>

That these testers, with their methods and their week, found nothing.
It's a lower bound on the failures, like a short pentest report, and the next red team with different methods may find plenty.

</details>

::: done-when
- You can explain the difference between specification gaming and goal misgeneralization, with one example of each.
- Using the widget, you can explain why the training score stays at 1.00 while the coin-left test score changes, and what a 2% share of random coins does to the two weights.
- You can say what sycophancy is, what Sharma et al. found in human preference data, and why it's a proxy problem like unit 17's.
- You can compute a PGR and describe what Constitutional AI, weak-to-strong training and red teaming each do.
:::

## Go deeper

- [Langosco et al., "Goal Misgeneralization in Deep Reinforcement Learning"](https://arxiv.org/abs/2105.14111). Section 3.1 and figure 2 for CoinRun and what randomizing the coin does, and section 3.2 for a maze version where the agent heads for the corner the cheese used to be in.
- [Sharma et al., "Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548). Section 3 for the four sycophancy tests, and section 4 for the analysis of human preference data and the misconception experiment.
- [Burns et al., "Weak-to-Strong Generalization"](https://arxiv.org/abs/2312.09390). Section 3 defines the setup and PGR, and section 5.1 looks at when the student imitates its supervisor's mistakes.
