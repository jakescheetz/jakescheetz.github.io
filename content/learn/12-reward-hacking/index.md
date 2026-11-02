---
title: Reward hacking
subtitle: Why pushing a reward model's score up eventually makes the answers worse.
unit: 12
minutes: 50
scripts: [widgets.js]
---

Unit 11 ended with a policy trained to get high scores from a reward model.
Push that training too far and the reward model's score keeps climbing while the answers people would actually prefer get worse.

## The score is a proxy

Goodhart's law, in the wording [Gao et al.](https://arxiv.org/abs/2210.10760) quote, says "When a measure becomes a target, it ceases to be a good measure."
RLHF's step 3 is that situation: the reward model was built to measure what people like, and then it becomes the number the policy is trained to push up.

What you want is answers people prefer.
Call the score a person would give a response $y$ its true reward, $r^*(y)$.
(I leave the prompt out of the notation in this unit, since everything here happens for one prompt at a time.)
Nobody can ask a person about every response during training, so step 3 of RLHF optimizes the reward model's score $r_\phi(y)$ instead.
That score is the proxy.

Split the proxy into the part that's right and the part that's wrong:

$$
r_\phi(y) = r^*(y) + e(y)
$$

where $e(y)$ is the reward model's error on $y$.
The optimizer only ever sees the sum.
When it searches for responses with a high $r_\phi$, it finds responses where $r^*$ is high and also responses where $e$ is high, and it can't tell those apart.

Gao et al. sort these errors using a taxonomy from Manheim and Garrabrant, and two of its kinds matter here.

- **Regressional:** $e$ is random noise. Part of the search goes into picking responses that got lucky noise, so the true reward rises less than the score, but it still rises.
- **Extremal:** $e$ is small where the reward model was trained and large somewhere else. Once the search reaches somewhere else, the true reward can fall while the score keeps going up.

Gao et al. show that, when the noise is independent of the true reward and well behaved (normal, for example), noise alone can't make the true reward go down as the score goes up.
So when the true reward does go down, the second kind is responsible.

## Why the errors grow as training goes on

The reward model learned from people's picks between $\pi_0$'s answers (unit 11).
It has only ever been checked on answers that look like $\pi_0$'s.
Step 3 then moves the policy $\pi_\theta$ away from $\pi_0$, deliberately, toward whatever $r_\phi$ scores highly.
The further it moves, the less its answers look like anything in the training data, and the more room $e$ has to be large.
The optimizer also prefers the places where $e$ is large and positive, because those score well.

Gao et al.'s example of the second kind is answer length (sec. 4.2.2).
If longer answers were usually the better ones in the comparison data, a reward model can learn that longer is always better.
Past some length that stops being true, and they report policies writing very long answers when a short one would do.

To see both kinds of error with real numbers, I use the same toy as my blog post.
Each response is boiled down to one number, $x$, standing in for something like length.
(In this unit $x$ is that number, not the prompt.)
$\pi_0$ writes responses with $x \sim \mathcal{N}(0, 3^2)$, a normal distribution with mean 0 and standard deviation 3.
People like more $x$ up to $x = 5$ and less after that:

$$
r^*(x) = x - \frac{x^2}{10}
$$

About 95% of $\pi_0$'s responses have $x$ below 5, so in the comparisons people made, the answer with more $x$ was usually the better one.
The reward model learned a straight line plus noise:

$$
r_\phi(x) = x + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)
$$

That makes its error $e(x) = \varepsilon + x^2/10$.
The noise $\varepsilon$ is the regressional part.
The $x^2/10$ is the extremal part: it's small near 0, where $\pi_0$ spends its time, and large past 5, where $\pi_0$ rarely goes and the reward model never got a correction.

## Best-of-n, and measuring how far it moves

Best-of-$n$ does the same kind of search as RL in a much simpler way: draw $n$ responses from $\pi_0$ and keep the one the reward model scores highest.
At $n = 1$ you get $\pi_0$ itself, and a larger $n$ searches harder.
Gao et al. study it alongside RL.

To compare amounts of search you need a measure of how far a policy has moved from $\pi_0$.
The standard one is the KL divergence, the same quantity unit 11's penalty uses:

$$
D_{\mathrm{KL}}(\pi \,\|\, \pi_0) = \mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y)}{\pi_0(y)}\right]
$$

Draw responses from the new policy $\pi$, take the log of how many times more likely $\pi$ makes each one than $\pi_0$ did, and average.
It's 0 when the two policies are the same and grows as $\pi$ moves its probability onto responses $\pi_0$ rarely wrote.
With the natural log the unit is nats.

For best-of-$n$ it has a closed form:

$$
D_{\mathrm{KL}}(\pi_{\mathrm{bon}} \,\|\, \pi_0) = \log n - \frac{n-1}{n}
$$

For a response $y$, let $u$ be the fraction of $\pi_0$'s responses that the reward model scores below $y$, so $u$ is between 0 and 1.
Best-of-$n$ returns $y$ when one of the $n$ draws is $y$ and the other $n - 1$ all score below it.
There are $n$ draws that could be $y$, so

$$
\pi_{\mathrm{bon}}(y) = n\, u^{n-1}\, \pi_0(y)
$$

and the log ratio inside the KL is $\log n + (n-1)\log u$.
The winner's $u$ is the largest of $n$ uniform numbers between 0 and 1, which has density $n u^{n-1}$, and the average of $\log u$ under that density is $\int_0^1 n u^{n-1} \log u \, du = -1/n$.
Plug that in and you get the formula.

The derivation assumes two different responses never get exactly the same score, which holds in the toy.
For real text, where the same response can be drawn twice, [Beirami et al.](https://arxiv.org/abs/2401.01879) show the formula is an upper bound on the true KL rather than exact.

The formula grows like $\log n$.
Best-of-16 is 1.84 nats from $\pi_0$, and best-of-4096, with 256 times the search, is 7.32.

<figure class="widget" id="best-of-n">
<figcaption>Best-of-n on the toy, running live. Left: π₀'s spread of x (grey), the reward model's straight line (blue, dashed) and what people actually think (orange), with the last few picks scored both ways. Right: average scores at every n, placed by KL. Slide n and watch the picks move past x = 5, where the line and the curve split.</figcaption>
</figure>

At small $n$ both colors rise together, because near $\pi_0$ the line and the curve agree.
Around $n = 16$ the average pick reaches $x = 5$, the top of the orange curve.
After that the picks keep moving right, the blue score keeps climbing, and the orange score falls.
The readout shows how unusual those picks are for $\pi_0$, which is the reason nobody corrected the reward model there.

## What Gao et al. measured

[Gao, Schulman and Hilton (2023)](https://arxiv.org/abs/2210.10760) ran this on real language models.
Labeling everything with people was too expensive, so a 6-billion-parameter reward model from InstructGPT stood in for people (the "gold" reward model).
It labeled 100,000 synthetic comparisons, which trained proxy reward models from 3 million to 3 billion parameters.
They optimized policies against the proxies with best-of-$n$ and with PPO and tracked the gold score along the way.

The gold score rose and then fell, the same shape as the toy.
They plotted it against $d = \sqrt{D_{\mathrm{KL}}}$ and found these fits:

$$
R_{\mathrm{bon}}(d) = d\,(a - b\,d), \qquad R_{\mathrm{RL}}(d) = d\,(a - b \log d)
$$

The paper calls the coefficients $\alpha$ and $\beta$; I renamed them because $\beta$ is already the KL coefficient.
Each method gets its own $a$ and $b$.
The $a d$ part is the early rise, and the $b$ term is what eventually pulls the score back down.

Other results from the paper:

- Larger proxy reward models overoptimize less, and the coefficients change smoothly enough with size to predict the peak gold score.
- More comparison data helps too. Below about 2,000 comparisons the proxies were barely better than chance.
- RL uses up far more KL than best-of-$n$ for the same change in score, so KL is a fair ruler within one method but not between methods.

## Two things people do about it

The first is the KL penalty from unit 11:

$$
\max_\theta\; \mathbb{E}_{y \sim \pi_\theta}\big[r_\phi(y)\big] - \beta\, D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_0)
$$

Every nat of distance from $\pi_0$ costs $\beta$ points of reward.
The policy keeps moving outward while extra distance still buys more score than it costs, so a larger $\beta$ settles closer to $\pi_0$.

Gao et al. ran RL with several values of $\beta$ (sec. 3.6).
The gold score depended only on how far the policy had moved.
A bigger penalty made a run level off at a smaller KL, but on the same gold-versus-KL curve, so its effect was like stopping training early.
They add that this result might be sensitive to hyperparameters.
In the widget's terms, choosing $\beta$ is like choosing where to leave the slider: it changes where you stop, and the orange curve stays the same.

The second fix goes after the errors themselves.
If the reward model is wrong in places it never saw, collect comparisons from those places.
[Bai et al. (2022)](https://arxiv.org/abs/2204.05862) at Anthropic found their preference models got less reliable at high scores and attributed it to having little data there.
Their "iterated online RLHF" trains the best policy it can, has crowdworkers compare that policy's answers, mixes those comparisons into the existing data, and retrains the preference models and then the policies, on a weekly cadence.
Crowdworkers clearly preferred the resulting models over ones trained only on the original data.

In the toy, comparisons between answers around $x = 5$ to $8$ would show the reward model that people stop wanting more $x$, and the line would bend.
Gao et al. (sec. 4.3) extended their RL fit to this kind of iteration and, under simplifying assumptions, predicted that it raises the final gold score through the $b$ term, which they attribute mostly to distribution shift, and leaves the part they tie to noise unchanged.
That matches the toy: fresh comparisons can teach the reward model the missing $x^2/10$, but they can't remove $\varepsilon$.

My post [Reward hacking from the ground up](../../posts/your-reward-model-is-an-attack-surface/) tells this as one story, including why it reminded me of fuzzing.

## Worked example

This runs best-of-$n$ on the toy with NumPy and prints the average over 2,000 trials at each $n$.
The `check` column estimates the KL directly from the derivation, averaging $\log n + (n-1)\log u$ over the winners (the reward model's scores on $\pi_0$'s responses are $\mathcal{N}(0, 10)$, so $u$ comes from the normal CDF).
The last column swaps in a reward model with only the regressional kind of error, $r^*(x) + \varepsilon$.

```python
import math
import numpy as np

rng = np.random.default_rng(0)
erf = np.vectorize(math.erf)

def true_reward(x):   # people like more x up to x = 5, then less
    return x - x**2 / 10

def best_of_n(n, proxy, trials=2000):
    x = rng.normal(0, 3, (trials, n))       # n candidates from pi_0 per trial
    noise = rng.normal(0, 1, (trials, n))   # the reward model's random error
    scores = proxy(x) + noise
    best = scores.argmax(axis=1)
    rows = np.arange(trials)
    return x[rows, best], scores[rows, best]

print("    n    KL  check  pick x  proxy   true | true (noise only)")
for n in (1, 4, 16, 64, 256, 1024, 4096):
    kl = math.log(n) - (n - 1) / n
    x, s = best_of_n(n, lambda x: x)        # the reward model from the text
    u = 0.5 * (1 + erf(s / math.sqrt(20)))  # where s ranks among pi_0's scores
    check = np.mean(math.log(n) + (n - 1) * np.log(u))
    x2, _ = best_of_n(n, true_reward)       # a reward model that only adds noise
    print(f"{n:5d} {kl:5.2f} {check:6.2f} {x.mean():7.2f} {s.mean():6.2f} "
          f"{true_reward(x).mean():6.2f} | {true_reward(x2).mean():6.2f}")
```

```
    n    KL  check  pick x  proxy   true | true (noise only)
    1  0.00   0.00   -0.08  -0.09  -0.99 |  -0.80
    4  0.64   0.64    2.90   3.22   1.60 |   1.52
   16  1.84   1.84    5.01   5.59   2.17 |   2.03
   64  3.17   3.17    6.65   7.42   1.97 |   2.20
  256  4.55   4.56    8.06   8.97   1.34 |   2.27
 1024  5.93   5.97    9.27  10.29   0.50 |   2.30
 4096  7.32   7.32   10.31  11.47  -0.50 |   2.33
```

The `check` column matches the formula to within 0.04, and the proxy score rises on every row while the true reward peaks at $n = 16$, where the average pick is $x = 5.01$.
With noise as the only error, the true reward rises on every row toward its maximum of 2.5, so the fall in the `true` column comes from the straight line.

## Where this goes next

- Unit 11's DPO has no separate reward model, but [Rafailov et al. (2024)](https://arxiv.org/abs/2406.02900) found DPO-style methods degrade in a similar way at larger KL budgets, often before a single pass over the data.
- Best-of-$n$ also works as a decoding method next to the ones in unit 09: at generation time, sample $n$ responses and return the one a scorer likes best, with $n$ trading compute for distance from $\pi_0$.
- Everything here treats the policy as a search with no goals of its own. Gao et al. (sec. 4.2.4) note that a model able to game the reward model on purpose could break their trends, and the last section of [my blog post](../../posts/your-reward-model-is-an-attack-surface/) points at that problem.

## Self-test

<details>
<summary>In the widget, at roughly which n does true quality peak, and where is the average pick there?</summary>

Around $n = 16$, with the average pick near $x = 5$.
That's the top of $r^*(x) = x - x^2/10$, so any further search moves the picks down the far side of the curve.

</details>

<details>
<summary>What is the KL from π₀ for best-of-1, and for best-of-2?</summary>

Best-of-1 is $\pi_0$, so $\log 1 - 0 = 0$.
Best-of-2 is $\log 2 - 1/2 \approx 0.19$ nats.

</details>

<details>
<summary>Why does noise alone (the last column of the worked example) never make the true reward go down?</summary>

With $r_\phi = r^* + \varepsilon$ and independent noise, a higher score is still evidence of a higher $r^*$, just diluted by the noise.
Picking higher scores wastes some search on lucky noise, but on average it still picks better responses.
A fall needs an error that grows in the direction the search is going, like the missing $x^2/10$.

</details>

<details>
<summary>Why did Gao et al. find the KL penalty acts like early stopping rather than a fix?</summary>

The penalty limits how far the policy moves from $\pi_0$, but it doesn't change what the reward model gets wrong at any given distance.
So runs with different $\beta$ ended at different points on the same gold-versus-KL curve.

</details>

<details>
<summary>Why does collecting new comparisons on the current policy's answers help, when collecting more on π₀'s answers helps less?</summary>

The extremal errors live where the policy is now and $\pi_0$ rarely went.
Comparisons from $\pi_0$ mostly add data where the reward model is already accurate, while comparisons from the current policy put data exactly where it's wrong.

</details>

::: done-when
- You can write $r_\phi = r^* + e$ and explain which kind of error makes the true reward fall while the score rises.
- You can compute the best-of-$n$ KL for a given $n$ and derive $\log n - (n-1)/n$ from the probability that a response wins.
- Using the widget, you can explain why the blue score climbs at every $n$ while the orange one peaks near $n = 16$.
- You can say what the KL penalty and iterated RLHF each change about overoptimization, and what each leaves alone.
:::

## Go deeper

- [Gao, Schulman and Hilton, "Scaling Laws for Reward Model Overoptimization"](https://arxiv.org/abs/2210.10760). Read section 3.6 for the KL penalty result and section 4.2 for how each kind of Goodhart shows up in their curves.
- [Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"](https://arxiv.org/abs/2204.05862). Section 4.2 splits the comparison data to measure overfitting to the preference model, and section 4.5 describes iterated online RLHF.
- [Beirami et al., "Theoretical guarantees on the best-of-n alignment policy"](https://arxiv.org/abs/2401.01879), for when the best-of-n KL formula is exact and when it's only an upper bound.
