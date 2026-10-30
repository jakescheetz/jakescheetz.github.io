---
title: Your reward model is an attack surface
subtitle: Reward hacking from the ground up, through a security lens
date: 2026-10-30
bibliography: references.bib
abstract-title: TL;DR
abstract: |
  - RLHF trains a language model to get high scores from a reward model, and that reward model learned what people like by looking at the original model's answers.
  - As training pushes the score up, the model's answers drift away from anything the reward model saw. That's where its scores are least reliable, and it's where reward hacking comes from.
  - This is the same reason fuzzers find bugs. The fix is similar too: have people judge what the trained model writes now, and retrain the reward model on that.
---

Coming from security, one part of how RLHF works looked very familiar to me.
This post starts from the basics (what a policy is and how RLHF is put together), then gets to that part and why I think it explains reward hacking.

## What a policy is

In reinforcement learning, the policy is the part that decides what to do.
It looks at the situation it's in and picks an action.
For a chess engine, the situation is the board and the action is a move.

For a language model, the situation is the prompt and the action is the response, so the policy is the model itself.
More precisely, it's the probability the model assigns to every possible response $y$ given a prompt $x$, written $\pi(y \mid x)$.
Training the policy means changing those probabilities so good responses get more likely and bad ones get less likely.

That only works if something tells you which responses are good.
A chess engine gets the result of the game.
A chatbot gets nothing like that.
Whether an answer is helpful, correct, or honest is a judgment a person makes, and you can't put a person in the loop for the millions of responses training needs.
RLHF gets around this by training a second model to make that judgment.

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

Step 3 as an equation:

$$
\max_\theta\; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] - \beta\, D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_0\big)
$$

The first term is the average reward-model score.
The second term measures how far the policy's answers have moved from $\pi_0$'s, and $\beta$ sets how much that costs.

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
<figcaption>Step 3. The reward model was trained once, on π₀'s answers, and grades everything the policy writes after that.</figcaption>
</figure>

## The part that looked familiar

In the basic version of this recipe, every answer the reward model learned from was written by $\pi_0$.
Its whole sense of what people like comes from comparisons between one model's answers.

Step 3 then moves the policy away from $\pi_0$, on purpose, in whichever direction the reward model scores higher.
The further training goes, the less the policy's answers look like the ones the reward model was trained on, and the reward model keeps scoring them with the same confidence.

If you've done security work, that setup should sound a lot like fuzzing.
A parser gets written and tested against normal inputs.
A fuzzer then generates a huge number of abnormal inputs and keeps the ones that reach new code, and it finds bugs because most of those inputs are things nobody tested.
RL against a reward model works the same way.
The reward model was checked against people on $\pi_0$'s answers, the optimizer generates answers nobody checked, and it keeps the ones that score well, including the ones that only score well because the reward model is wrong about them.
No one has to intend this.
It follows from searching hard against a checker that was only tested on ordinary cases.

The best-known example of a reward getting gamed is OpenAI's CoastRunners boat [@clark2016faulty].
It was trained to maximize its score in a racing game, and it found a lagoon where it could circle forever, hitting the same targets as they respawned and catching fire and crashing into other boats along the way.
It scored higher than it could have by finishing the race.
That was a game score rather than a learned reward model, but it's the same failure: the number being optimized and what its designers wanted came apart.

With language models it's usually less dramatic.
@gao2023scaling [sec. 4.2.2] use answer length as their example.
If longer answers were usually the better ones in the comparison data, a reward model can learn that longer is always better.
Past a certain point that stops being true, and they note that policies writing very long answers when a short one would do is a problem they've actually run into.

## A small simulation

To see what this looks like, I wrote about the smallest version I could.
Each response is boiled down to one number, $x$, standing in for something like length.
People like more $x$ up to a point and then less.
The reward model only saw responses where more $x$ was better, so it scores $x$ plus some noise.[^toy]

Instead of full RL, it uses best-of-$n$: draw $n$ responses from the base policy and keep whichever one the reward model scores highest.
A bigger $n$ means searching harder.
It's simpler than RL, and its KL from the base policy works out to exactly $\log n - (n-1)/n$ [@gao2023scaling, sec. 2], so you know how far you've moved.

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
     KL from the base policy (nats)

     <span class="chart-proxy">●</span> reward model's score
     <span class="chart-true">○</span> actual quality</pre>
<figcaption>What the script prints, plotted against how far best-of-n moved from the base policy.</figcaption>
</figure>

The reward model's score goes up the whole way.
Actual quality peaks around $n = 16$, and by $n = 4096$ it's almost back to where it started, on the run the reward model rated highest.
If the reward model were your only way of checking, that last run would look like your best one.
This is Goodhart's law: once a measure becomes the target, it stops being a good measure [@manheim2018categorizing].

@gao2023scaling ran a much bigger version of this.
Labeling everything with people gets expensive, so they used a large 6B-parameter reward model as a stand-in for people, trained smaller reward models on its preferences, and optimized policies against the smaller ones.
The large model's score rose and then fell as the policy moved away from $\pi_0$, the same shape as the toy, and larger reward models lasted longer before it turned.

## What the fuzzing view changes

Looking at it this way changed how I think about two parts of the setup.

The first is the KL penalty.
It's usually described as regularization, but given everything above it has a more specific job: it keeps the policy close to $\pi_0$, which is the only place the reward model was ever checked against people.
Gao et al. found that the penalty mostly changes where a run stops, not how good the result is at a given distance from $\pi_0$ [@gao2023scaling, sec. 3.6].
That fits the fuzzing picture: the penalty limits how far the policy goes, but it doesn't make the reward model any better at judging what it finds there.

The second is the fix.
When a fuzzer finds an input that breaks your parser, you fix the parser and add the input to your tests so it stays fixed.
The RLHF version is to take answers from the current policy, have people compare those, and retrain the reward model on them, so it has seen the kinds of answers it's now being asked to grade.
@bai2022training trained this way at Anthropic, updating their reward models and policies every week with fresh human feedback, and Gao et al. point to this kind of iterated training as a way to reduce overoptimization [@gao2023scaling, sec. 4.3].

## What this leaves out

This whole post treats the policy as a search process with no goals of its own.
It finds the reward model's mistakes because it's searching hard, not because it's trying to.
That's also the setting Gao et al. measured, and they say their results may not hold once models are capable enough to game the reward model deliberately [@gao2023scaling, sec. 4.2.4].
That's a harder problem: a model that ends up pursuing an objective of its own [@hubinger2019risks], or one that behaves differently when it believes it's being trained [@greenblatt2024alignment].
I want to write about that one separately.

[^toy]: Candidates are $x \sim \mathcal{N}(0, 3^2)$, reward-model noise is $\mathcal{N}(0, 1)$, and actual quality is $x - x^2/10$.
    The exact numbers don't matter much; you get the same rise and fall whenever the reward model keeps rewarding $x$ past the point where people stop liking it.

## References

::: {#refs}
:::
