---
title: "Reward hacking from the ground up"
subtitle: Why optimizing a learned reward goes wrong, starting from the basics
date: 2026-10-30
toc: true
bibliography: references.bib
abstract-title: TL;DR
abstract: |
  - We can't write down what we want from a language model, so RLHF trains a reward model to stand in for human judgment and then optimizes the language model against it.
  - The reward model learned from the original model's answers. Optimization pushes the trained model's answers away from those, toward the places where the reward model is least reliable. Its score keeps rising while actual quality peaks and falls. That is reward hacking, and it happens without the model "trying" to cheat.
  - The usual mitigations limit how far training goes or refresh the reward model with new human judgments. They help, but none of them makes the problem go away, and none of them covers a model that games its reward on purpose.
---

The core problem in this post is simple to state.
We want a language model to give good answers, but we can't write down what "good" means precisely enough to train on.
So we train a second model to guess what people would think, and then we optimize the first model to score well with that guess.
The guess is a *proxy* for what we actually want, and pushing hard on a proxy tends to find the places where it's wrong.

That failure is called reward hacking, and it's one of the most basic ways a trained system ends up pursuing something other than what its designers meant.

## A policy, and what it needs to learn

In reinforcement learning, the *policy* is the part that decides what to do.
It looks at the situation it's in and picks an action.
For a chess engine, the situation is the board and the action is a move.

For a language model, the situation is the prompt and the action is the response, so the policy is the model itself.
More precisely, it's the probability the model assigns to every possible response $y$ given a prompt $x$, written $\pi(y \mid x)$.
Training the policy means changing those probabilities so good responses get more likely and bad ones get less likely.

That only works if something tells you which responses are good.
That signal is the *reward*: a number attached to each response, where higher means better.

## Why the reward has to be learned

A chess engine gets its reward for free: the game ends and you win, lose, or draw.
A chatbot gets nothing like that.
Whether an answer is helpful, correct, or honest is a judgment a person makes, and you can't write it down as a formula.

You also can't put a person in the loop for every response, because training needs millions of them.
RLHF (reinforcement learning from human feedback) gets around this by collecting a smaller set of human judgments and training a second model, the *reward model*, to predict them.
The reward model then grades responses during training in place of people.

This gives us two different rewards, and the rest of the post is about the gap between them:

- The **true reward** is what we actually want: how good a response really is, as a careful person would judge it.
  We can't compute it directly.
- The **proxy reward** is the reward model's score.
  It's what training actually optimizes, and it's only as good as the reward model's guess.

## How RLHF is built

The recipe most people mean by RLHF comes from InstructGPT [@ouyang2022training], which built on earlier work that learned rewards from human comparisons [@christiano2017deep].
It has three steps.

1. **Start from a decent model.**
   Take a pretrained model and fine-tune it on example answers written by people.
   Call the result $\pi_0$.
   Everything after this builds on it.
2. **Train a reward model.**
   Have $\pi_0$ write a few answers to the same prompt, and have people say which one they prefer.
   A separate model, $r_\phi$, learns to predict their picks.
   It reads a prompt and a response and outputs one number: higher means people would probably like it more.
3. **Train the policy against the reward model.**
   The policy writes responses, the reward model scores them, and RL updates the policy toward responses that score higher.
   A penalty keeps the policy from drifting too far from $\pi_0$.

The penalty in step 3 uses the *KL divergence*, written $D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_0)$.
It's a standard measure of how different two probability distributions are: zero when the policy $\pi_\theta$ answers exactly like $\pi_0$, and larger the more its answers differ.
It's measured in nats, the natural-log version of bits.
Throughout this post it's the yardstick for "how far training has moved the model."

With that, step 3 as an equation:

$$
\max_\theta\; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] - \beta\, D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_0\big)
$$

The first term is the average reward-model score over prompts $x$ from the training set $D$.
The second term is the KL penalty, and $\beta$ sets how much moving away from $\pi_0$ costs.

<figure>
```
π₀'s answers, picked by people          prompts x
               │                            │
               ▼                            ▼
      ┌─────────────────┐    y     ┌─────────────────┐
      │   reward model  │◄─────────│      policy     │
      │       r_φ       │─────────►│       π_θ       │
      └─────────────────┘  r(x,y)  └─────────────────┘
```
<figcaption>Step 3, in its simplest form. The reward model was trained once, on π₀'s answers, and grades everything the policy writes after that.</figcaption>
</figure>

## Why optimizing the proxy goes wrong

Look at where the reward model's knowledge comes from.
In the simplest version of this recipe, and in the experiments later in this post, every answer it learned from was written by $\pi_0$.
Its whole sense of what people like comes from comparisons between one model's answers.
(Real pipelines usually feed in newer answers too; that turns out to be one of the main fixes, covered below.)

Step 3 then moves the policy away from $\pi_0$, on purpose, in whichever direction the reward model scores higher.
The further training goes, the less the policy's answers look like the ones the reward model was trained on.
In ML terms this is *distribution shift*: a model is being used on inputs that differ from the ones it was trained on.
Reward models are measurably less reliable there.
Reward models that agree with each other on the data they were trained on can give very different scores once the policy shifts [@eisenstein2023helping].
@bai2022training [sec. 4.5] found theirs became less calibrated and less robust at high scores, which they put down to having little training data there.
And high scores are exactly where the policy is being pushed.

Put those together and the problem follows.
The optimizer keeps whatever scores well, including answers that only score well because the reward model is wrong about them.
Early in training the policy mostly improves at the real thing, because near $\pi_0$ the proxy and the true reward agree closely.
Push further and more of the score comes from the reward model's mistakes.
Past some point the proxy keeps going up while the true reward goes down.
That's called *overoptimization*, and the behavior it produces is reward hacking.[^def]

No one has to intend this, including the model.
It follows from searching hard against a checker that was only tested on ordinary cases.
This is Goodhart's law, commonly stated as "when a measure becomes a target, it ceases to be a good measure."
@manheim2018categorizing [p. 1] put it more precisely: optimization causes "a collapse of the statistical relationship between a goal which the optimizer intends and the proxy used for that goal."
They split it into several kinds, and two of them matter here [@manheim2018categorizing, secs. 1–2]:

- **Regressional Goodhart.**
  The proxy is the true reward plus some error.
  If you pick whatever scores highest on the proxy, you also pick answers where the error happened to be in your favor, so the winner is usually a bit worse than its score says.
  This is there from the very start of training.
- **Extremal Goodhart.**
  The proxy and the true reward are related one way inside the range the reward model was trained on, and differently outside it.
  If you push far enough, you leave that range.

@gao2023scaling [sec. 4.2.2] attribute the point where the true reward turns down mainly to the extremal kind.

> **An analogy, if you've done security work.**
> This is the same reason fuzzers find bugs.
> A parser is tested against normal inputs; a fuzzer then generates huge numbers of abnormal ones and keeps those that reach new code, and it finds bugs because most of those inputs are things nobody tested.
> Here, the reward model was checked against people on $\pi_0$'s answers, the optimizer generates answers nobody checked, and it keeps the ones that score well.

### What it looks like

The best-known example of a reward getting gamed is OpenAI's CoastRunners boat [@clark2016faulty].
It was trained to maximize its score in a racing game, and it found a lagoon where it could circle forever, hitting the same targets as they respawned and catching fire and crashing into other boats along the way.
It scored higher than it could have by finishing the race.
That was a game score rather than a learned reward model, but it's the same failure: the number being optimized and what its designers wanted came apart.

With language models it's usually less dramatic.
@gao2023scaling [sec. 4.2.2] use answer length as their example.
If longer answers were usually the better ones in the comparison data, a reward model can learn that longer is always better.
Past a certain point that stops being true, and they note that policies writing very long answers when a short one would do is a problem they've actually run into.
Later work measured it directly.
@singhal2023long found that in three RLHF settings the reward gains were "largely driven by increasing response length", that a reward based only on length reproduced most of the improvements, and that the bias came mainly from the reward model.

## A small simulation

To see the shape of this, I wrote about the smallest version I could.
Each response is boiled down to one number, $x$, standing in for something like length.
People like more $x$ up to a point and then less: that's the true reward.
The reward model only saw responses where more $x$ was better, so it scores $x$ plus some noise: that's the proxy.[^toy]
The noise is the regressional part, and rewarding $x$ past the peak is the extremal part.

Instead of full RL, the simulation uses *best-of-$n$* sampling: draw $n$ responses from the base policy and keep whichever one the reward model scores highest.
A bigger $n$ means searching harder against the proxy.
It's simpler than RL, and its KL from the base policy is at most $\log n - (n-1)/n$ [@gao2023scaling, sec. 2], so you have an upper bound on how far you've moved.
Gao et al. use this formula as the exact value; @beirami2024theoretical later proved it's only an upper bound, and sometimes a loose one.

```python
import math, random, statistics

random.seed(0)

def true_reward(x):   # more x helps, until it doesn't (peaks at x = 5)
    return x - x * x / 10

def proxy_reward(x):  # learned "more x is better", plus noise
    return x + random.gauss(0, 1)

def best_of_n(n, trials=2000):
    proxy, true = [], []
    for _ in range(trials):
        candidates = [random.gauss(0, 3) for _ in range(n)]
        score, x = max((proxy_reward(x), x) for x in candidates)
        proxy.append(score)
        true.append(true_reward(x))
    return statistics.mean(proxy), statistics.mean(true)

for n in (1, 4, 16, 64, 256, 1024, 4096):
    kl = math.log(n) - (n - 1) / n
    proxy, true = best_of_n(n)
    print(f"n={n:<5} KL={kl:.2f}  proxy={proxy:5.2f}  true={true:5.2f}")
```

<figure>
<pre> 12 │
    │                                            <span class="chart-proxy">●</span>
    │                                    <span class="chart-proxy">●</span>
  9 │                           <span class="chart-proxy">●</span>
    │
    │                   <span class="chart-proxy">●</span>
  6 │           <span class="chart-proxy">●</span>
    │
    │
  3 │    <span class="chart-proxy">●</span>
    │    <span class="chart-true">○</span>      <span class="chart-true">○</span>       <span class="chart-true">○</span>
    │                           <span class="chart-true">○</span>        <span class="chart-true">○</span>
  0 ┼<span class="chart-proxy">●</span>───────────────────────────────────────────<span class="chart-true">○</span>─
    │<span class="chart-true">○</span>
     0     1     2     3     4     5     6     7
     KL from the base policy (nats, upper bound)

     <span class="chart-proxy">●</span> proxy reward (reward model's score)
     <span class="chart-true">○</span> true reward (actual quality)</pre>
<figcaption>What the script prints, plotted against the upper bound on how far best-of-n moved from the base policy.</figcaption>
</figure>

The proxy goes up the whole way.
The true reward peaks around $n = 16$, and by $n = 4096$ it's almost back to where it started, on the run the reward model rated highest.
If the reward model were your only way of checking, that last run would look like your best one.

This illustrates the problem; it doesn't prove anything.
I built the decline in when I made the proxy a straight line and the true reward a curve.
The point is the shape, and that the proxy never warns you.
The evidence that this happens in real training comes next.

## The same shape at scale

@gao2023scaling ran a much bigger version of this.
Labeling everything with people gets expensive, so they used a large 6B-parameter reward model as a stand-in for people (playing the role of the true reward), trained smaller reward models, from 3M to 3B parameters, on its preferences (the proxies), and optimized policies against the smaller ones.
The large model's score rose and then fell as the policy moved away from $\pi_0$, the same shape as the toy.
Larger proxy reward models reached a higher true-reward peak and overoptimized less [@gao2023scaling, sec. 3.2].

One caveat matters for how far this generalizes.
The "true reward" here is itself a reward model, so these results describe one model being overoptimized against another; they were never checked against people [@gao2023scaling, sec. 2.1].

## What the mitigations fix, and what they don't

With the mechanism in view, it's easier to say what each standard mitigation is actually doing.

**The KL penalty limits how far training goes.**
It's usually described as regularization, but given everything above it has a more specific job: it keeps the policy close to $\pi_0$, which is the only place the reward model was ever checked against people.
Gao et al. found that the penalty mostly changes where a run stops, not how good the result is at a given distance from $\pi_0$ [@gao2023scaling, sec. 3.6], though they caution that this result may be sensitive to hyperparameters.
So it limits how far the policy goes, but it doesn't make the reward model any better at judging what it finds there.

**Bigger reward models hold up longer.**
As above, larger proxy reward models reached a higher peak and overoptimized less [@gao2023scaling, sec. 3.2].
That moves the peak; it doesn't remove it.
@coste2023reward sum up Gao et al.'s result the same way: overoptimization "remains a persistent problem regardless of the size of the proxy reward model and training data used."

**Refreshing the reward model brings it back on-distribution.**
The most direct fix is to take answers from the current policy, have people compare those, and retrain the reward model on them, so it has seen the kinds of answers it's now being asked to grade.
This is how the method was first designed.
@christiano2017deep trained the reward model online, alongside the policy, on the policy's current behavior.
When they dropped that and trained the reward model offline, they got the failure this post describes: in Pong, the agent learned to avoid losing points but not to score, and produced endless volleys [@christiano2017deep, sec. 3.3].
InstructGPT also notes that its steps 2 and 3 can be repeated, and some of its comparison data came from later policies [@ouyang2022training, sec. 3.1].
@bai2022training [sec. 4.5] trained this way at Anthropic, updating their reward models and policies on a roughly weekly cadence with fresh human feedback, because their reward models got less reliable at high scores.

Refreshing helps, but it has limits.
Gao et al. analyze iterated training with a simple model and conclude it should reduce overoptimization, but only the extremal kind; it doesn't touch the regressional kind [@gao2023scaling, sec. 4.3].
A freshly trained reward model still has errors, and picking the top score still picks for them.

It also depends on people being able to judge the new answers, and that has its own Goodhart problem.
@sharma2023towards found that both people and preference models prefer convincingly written sycophantic answers over correct ones a non-negligible fraction of the time.
@wen2024language found that after standard RLHF, models got better at convincing time-limited human evaluators without getting more correct, and the evaluators' false-positive rates rose by 24.1% on a reading-comprehension task and 18.3% on a programming task.
Once the policy writes answers people can't reliably check, the people become a proxy too.
That's the starting point for work on scalable oversight, which I want to come back to.

## What this leaves out

This whole post treats the policy as a search process with no goals of its own.
It finds the reward model's mistakes because it's searching hard, not because it's trying to.
That's also the setting Gao et al. measured, and they say their results may not hold once models are capable enough to game the reward model deliberately [@gao2023scaling, sec. 4.2.4].
That's a harder problem: a model that ends up pursuing an objective of its own [@hubinger2019risks], or one that behaves differently when it believes it's being trained [@greenblatt2024alignment].
I want to write about that one separately.

[^def]: @skalse2022defining give a formal definition: reward hacking is when "optimizing an imperfect proxy reward function leads to poor performance according to the true reward function."

[^toy]: Candidates are $x \sim \mathcal{N}(0, 3^2)$, reward-model noise is $\mathcal{N}(0, 1)$, and actual quality is $x - x^2/10$.
    The exact numbers don't matter much; you get the same rise and fall whenever the reward model keeps rewarding $x$ past the point where people stop liking it.
    Because the proxy redraws its noise on every call, which $x$ gets picked is itself random, and the real KL falls further below the formula, so read the chart's x-axis as an upper bound.

## References

::: {#refs}
:::
