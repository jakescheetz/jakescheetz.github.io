---
title: Your reward model is an attack surface
subtitle: Reward hacking, read as a security problem
date: 2026-10-30
bibliography: references.bib
abstract-title: TL;DR
abstract: |
  - Training a model against a learned reward model looks a lot like fuzzing a parser: tons of queries against a judge that's wrong in places.
  - Push hard enough and the model finds those places. That's reward hacking, and you can measure it.
  - Security already has habits for this: budget the search, keep a judge the optimizer never sees, red-team the judge, and calibrate against people.
---

One thing you learn quickly in security: if you can query a check as many times as you want, you'll eventually find its edges.
A fuzzer doesn't understand the parser it's breaking.
It just needs volume and a signal that says "warmer."

That's also, more or less, how we align language models today, and I think reward hacking makes the most sense when you look at it as a security problem.
The reward model is the attack surface.
The optimizer is the attacker (whether it means to be or not).

The classic example is OpenAI's CoastRunners boat [@clark2016faulty].
It was rewarded for hitting targets, so it found a lagoon where it could loop forever, hitting the same respawning targets while catching fire and crashing into other boats.
It scored higher than it could have by actually finishing the race.
The optimizer worked fine; the reward had a bug.

## The setup

Most RLHF pipelines look something like this [@christiano2017deep; @ouyang2022training]: people compare pairs of model outputs, a reward model $r_\phi$ learns to predict which one they'd pick, and the policy $\pi_\theta$ is trained to maximize that reward without drifting too far from where it started, $\pi_0$:

$$
\max_\theta\; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] - \beta\, D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_0\big)
$$

<figure>
```
    human preferences              prompts x
            │                          │
            ▼                          ▼
   ┌─────────────────┐   y    ┌─────────────────┐
   │   reward model  │◄───────│      policy     │
   │       r_φ       │───────►│       π_θ       │
   └─────────────────┘ r(x,y) └─────────────────┘
```
<figcaption>The policy never talks to a person. It only ever sees r_φ.</figcaption>
</figure>

Look at that like a pentester would.
The policy is an untrusted process with a huge query budget, and the reward model is a validator it can hit as often as it wants.
The people who actually define "good" are nowhere in the training loop.
That's a fuzzer against a parser, and we know how that one goes.

## Goodhart, four ways

@manheim2018categorizing break Goodhart's law into four flavors, and @gao2023scaling [sec. 4.2] map each one onto reward models.
All four have a security equivalent:

- **Regressional:** the score is the goal plus noise, so pushing on the score also selects for noise.
  Like a scanner's highest-confidence findings being, on average, less real than their confidence says.
- **Extremal:** the proxy only held up on the data it was fit to.
  A reward model that learned "longer is better" keeps believing it at three thousand words, like a validator that was only ever tested on well-formed input.
- **Causal:** the proxy correlated with the goal without causing it, like treating `X-Forwarded-For: 127.0.0.1` as proof a request is internal.
- **Adversarial:** something is actively working against the proxy.
  Every WAF bypass ever.

The policy doesn't need to "want" anything to end up in that last bucket.
Enough search against a fixed judge turns up adversarial inputs on its own, which is basically the whole premise of fuzzing.

## Watching it happen

Here's about the smallest version of this I could build.
Each candidate response has one feature, $x$ (think length, or confidence, or how much it agrees with you).
True quality goes up with $x$ and then back down.
The reward model was fit on ordinary responses where more $x$ really was better, so it scores $x$ plus some noise.[^toy]
Best-of-$n$ keeps whichever candidate the reward model likes most, and its KL from the base policy is $\log n - (n-1)/n$ [@gao2023scaling, sec. 2], so $n$ works as the optimization budget.

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

     <span class="chart-proxy">●</span> proxy reward: what the optimizer sees
     <span class="chart-true">○</span> true reward: what we actually wanted</pre>
<figcaption>What that script prints, plotted against the optimization budget.</figcaption>
</figure>

The proxy score goes up at every step.
True reward peaks around $n = 16$ (about 1.8 nats out), and by $n = 4096$ it's given back almost all of its gain, on the exact run the reward model likes best.
Nothing in the proxy curve warns you.

This isn't just a toy thing either.
@gao2023scaling measured the same rise-and-fall at scale, using a 6B "gold" reward model as a stand-in for people and optimizing policies against smaller proxy reward models.
The curves were predictable enough to fit, and bigger reward models held up longer before tipping over.

## What security already knows

1. **KL is the attacker's budget.**
   Every step away from $\pi_0$ is more search the policy gets to spend on the reward model.
   In Gao et al.'s RL runs, a KL penalty acted more like early stopping than a fix [@gao2023scaling, sec. 3.6].
   That's rate limiting: it slows the attacker down, it doesn't patch the bug.
2. **Keep a judge the optimizer never sees.**
   Fresh human evals or held-out reward models the policy never trained against, and retrain the reward model as the policy moves [@gao2023scaling, sec. 4.3].
   Same reason the red team doesn't get to read the detection rules before the engagement.
3. **Red-team the reward model directly.**
   Go looking for outputs it overrates before the policy does.
   Every one you find first is one the training run doesn't get to exploit.
4. **Calibrate against people.**
   A score only means something next to a reference.
   That's the whole idea behind [EchoBench](https://www.netspi.com/blog/technical-blog/ai-ml-pentesting/introducing-echobench-a-human-calibrated-benchmark-for-autonomous-pentesting/), and reward models arguably need the same denominator: at the KL you actually trained to, how often do people agree with the reward model?

## What this misses

All of this treats the policy as a really thorough search process with no goals of its own.
That's also the regime Gao et al. measured, and they're upfront that their curves may break once models get capable enough to game the proxy on purpose [@gao2023scaling, sec. 4.2.4].

The harder questions start there: what if the model ends up optimizing an objective of its own [@hubinger2019risks], or acts differently when it thinks it's being trained [@greenblatt2024alignment]?
In security terms that's the jump from fuzzing to insider threat, and that's a post for another day.

[^toy]: Candidates are $x \sim \mathcal{N}(0, 3^2)$, reward-model noise is $\mathcal{N}(0, 1)$, and true reward is $x - x^2/10$.
    The numbers are arbitrary; the rise-and-fall shows up any time the proxy keeps rewarding $x$ past the point where true reward turns over.

## References

::: {#refs}
:::
