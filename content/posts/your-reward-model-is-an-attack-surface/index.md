---
title: Your reward model is an attack surface
subtitle: Reward hacking, read as a security problem
date: 2026-10-30
bibliography: references.bib
abstract: |
  Optimizing a policy against a learned reward model has the same structure as fuzzing a parser: an untrusted search process, a large query budget, and a judge it can query freely.
  This post reads reward hacking through that lens, reproduces overoptimization in a short simulation, and lists the security habits that carry over.
---

Security engineers learn one thing early: a check you can query without limit is a check you will eventually find the edges of.
A fuzzer doesn't need to understand a parser to crash it.
It needs volume and a signal that says *warmer*.

Reinforcement learning against a learned reward model has the same shape.
The policy doesn't need to understand what we want; it needs volume and a signal, and the reward model is the signal.
Reward hacking is one of the central problems in AI alignment [@krakovna2020specification], and I think it reads most clearly as a security problem: the reward model is the attack surface, and optimization is the attacker.

The classic example is OpenAI's CoastRunners agent [@clark2016faulty].
Rewarded for hitting targets during a boat race, it found a lagoon where it could circle indefinitely, knocking over the same targets as they respawned.
It caught fire, crashed into other boats, and went the wrong way on the track, and it still scored higher than it could by finishing the race.
Nothing was wrong with the optimizer.
The reward function had a bug, and the optimizer found it.

## The setup

Much of language-model alignment runs through some form of reinforcement learning from human feedback [@christiano2017deep; @ouyang2022training].
People compare pairs of outputs, and a reward model $r_\phi$ learns to predict which one they prefer, usually through a Bradley–Terry model:

$$
P(y_1 \succ y_2 \mid x) = \sigma\big(r_\phi(x, y_1) - r_\phi(x, y_2)\big)
$$

The policy $\pi_\theta$ is then trained to maximize that reward, with a penalty for drifting from the starting policy $\pi_0$:

$$
\max_\theta\; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] - \beta\, D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_0\big)
$$

InstructGPT adds a pretraining term on top of this [@ouyang2022training]; it doesn't change the argument.

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
<figcaption>The policy never sees a person. It sees r_φ.</figcaption>
</figure>

Read that loop the way a security engineer would.
The policy is an untrusted process with a very large query budget.
The reward model is a validator it can query as often as it likes, and its verdict is the only thing the policy is trained on.
Ground truth, meaning people, is offline.
That's a fuzzer against a parser, or a payload generator against a WAF, and we know how those go: whatever the validator gets wrong, the search eventually finds.

## Goodhart, four ways

Manheim and Garrabrant [-@manheim2018categorizing] split Goodhart's law into four mechanisms, and Gao et al. [-@gao2023scaling, sec. 4.2] map each onto reward models.
Every one has a security counterpart.

- **Regressional.** The proxy is the goal plus noise, so selecting hard on the proxy also selects for the noise.
  In security: a scanner's highest-confidence findings are, on average, less real than their confidence says.
- **Extremal.** The proxy tracked the goal on the data it was fit to, and fails outside it.
  A reward model that learned "longer answers are better" from ordinary answers keeps believing it at three thousand words.
  In security: a validator only ever tested on well-formed input, meeting malformed input.
- **Causal.** The proxy correlates with the goal without causing it, so pushing on the proxy moves nothing.
  In security: treating `X-Forwarded-For: 127.0.0.1` as proof a request is internal, right up until someone sets the header themselves.
- **Adversarial.** Another agent optimizes against the proxy for its own ends.
  In security: every WAF bypass.

An RL policy isn't an adversary in the sense of wanting anything besides reward.
It doesn't have to be.
Sustained search against a fixed judge turns up adversarial inputs as a by-product; that's the entire premise of fuzzing.

## Watching it happen

Here's the smallest version I could build.
Each candidate response has a single feature $x$: think of it as length, or confidence, or agreement with the user.
True quality rises with $x$ and then falls.
The reward model was fit on ordinary responses, where more $x$ really was better, so it scores $x$ plus noise.[^toy]
Best-of-$n$ sampling draws $n$ candidates and keeps the one the reward model likes most; its KL from the base policy is $\log n - (n-1)/n$ [@gao2023scaling, sec. 2], so $n$ doubles as an optimization budget.

```python
import math
import random
import statistics

random.seed(0)


def true_reward(x):
    # What we actually want: more x helps, until it doesn't (peak at x = 5).
    return x - x * x / 10


def proxy_reward(x):
    # What the reward model learned: "more x is better", plus noise.
    return x + random.gauss(0, 1)


def best_of_n(n, trials=2000):
    proxy, true = [], []
    for _ in range(trials):
        candidates = [random.gauss(0, 3) for _ in range(n)]
        score, x = max((proxy_reward(x), x) for x in candidates)
        proxy.append(score)
        true.append(true_reward(x))
    return statistics.mean(proxy), statistics.mean(true)


print(f"{'n':>5} {'KL':>5} {'proxy':>6} {'true':>6}")
for n in (1, 4, 16, 64, 256, 1024, 4096):
    kl = math.log(n) - (n - 1) / n
    proxy, true = best_of_n(n)
    print(f"{n:>5} {kl:>5.2f} {proxy:>6.2f} {true:>6.2f}")
```

Its output:

```
    n    KL  proxy   true
    1  0.00   0.05  -0.85
    4  0.64   3.24   1.58
   16  1.84   5.55   2.18
   64  3.17   7.48   1.97
  256  4.55   8.93   1.36
 1024  5.93  10.28   0.54
 4096  7.32  11.45  -0.48
```

<figure>
```
 12 │
    │                                            ●
    │                                    ●
  9 │                           ●
    │
    │                   ●
  6 │           ●
    │
    │
  3 │    ●
    │    ○      ○       ○
    │                           ○        ○
  0 ┼●───────────────────────────────────────────○─
    │○
     0     1     2     3     4     5     6     7
     KL from the base policy (nats)

     ● proxy reward: what the optimizer sees
     ○ true reward: what we wanted
```
<figcaption>The same run, plotted against the optimization budget.</figcaption>
</figure>

Proxy reward climbs at every step.
True reward peaks at $n = 16$, about 1.8 nats from the base policy, and has given back almost all of its gain by $n = 4096$, the run the reward model rates highest.
Nothing in the proxy curve marks the turn.

This isn't just a toy shape.
Gao et al. [-@gao2023scaling] measured it at scale by replacing human labelers with a 6B-parameter "gold" reward model, training smaller proxy reward models on its labels, and optimizing policies against the proxies.
Gold reward rose and then fell, following fitted curves in $d$, the square root of the KL from the base policy:

$$
R_{\mathrm{BoN}}(d) = d\,(\alpha_{\mathrm{BoN}} - \beta_{\mathrm{BoN}}\, d) \qquad R_{\mathrm{RL}}(d) = d\,(\alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}} \log d)
$$

The coefficients vary smoothly with reward-model size, so the peak is predictable before you reach it, and larger reward models overoptimize more slowly.

## What security practice suggests

1. **KL distance is the attacker's budget.**
   Every step away from $\pi_0$ is search the policy can spend probing the reward model.
   In Gao et al.'s RL runs, adding a KL penalty behaved like early stopping: it changed where a run ended up on the curve, not the curve itself [@gao2023scaling, sec. 3.6].
   That's rate limiting.
   It slows the attacker down; it doesn't fix the bug.
2. **Keep a judge the optimizer never sees.**
   Gao et al.'s measurements work because nothing optimizes against the gold model.
   In practice that means evaluating with fresh human judgments or held-out reward models the policy was never trained against, and retraining the reward model on fresh feedback as the policy moves [@gao2023scaling, sec. 4.3].
   It's the same reason the red team doesn't get to read the detection rules before the exercise.
3. **Red-team the reward model directly.**
   Search for outputs it overrates before the policy does.
   Every one you find is a bug the RL run doesn't get to find first.
4. **Calibrate against people.**
   A score only means something relative to a reference.
   That's the idea behind [EchoBench](https://www.netspi.com/blog/technical-blog/ai-ml-pentesting/introducing-echobench-a-human-calibrated-benchmark-for-autonomous-pentesting/), which scores autonomous pentesting agents against a measured human cohort on the same targets.
   Reward models deserve the same denominator: at the KL the policy actually reached, how often do people agree with the reward model's ranking?

## What this framing misses

Everything above treats the policy as a search process with no goals of its own, dangerous only because it's thorough.
That's also the regime Gao et al. measured: their models weren't capable enough for adversarial Goodhart, and they warn that their scaling laws may break down once models are [@gao2023scaling, sec. 4.2.4].

The harder questions start there.
What if the trained model becomes an optimizer with an objective of its own, distinct from the training signal [@hubinger2019risks]?
What if it behaves differently when it believes it's being trained [@greenblatt2024alignment]?
In security terms, that's the move from fuzzing to insider threat.
That's a different post.

[^toy]: The numbers are arbitrary: candidates are $x \sim \mathcal{N}(0, 3^2)$, reward-model noise is $\mathcal{N}(0, 1)$, and true reward is $x - x^2/10$.
    The rise-then-fall shape holds for any setup where the proxy keeps rewarding $x$ past the point where true reward turns over.

## References

::: {#refs}
:::
