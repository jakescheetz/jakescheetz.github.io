---
title: Reward hacking
subtitle: Why pushing a reward model's score up eventually makes the answers worse.
unit: 17
section: Alignment
minutes: 50
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'What does Goodhart''s law say?'
    answer: 'When a measure becomes a target, it ceases to be a good measure'
    wrong: ['A model''s error shrinks as you train it on more data', 'Optimizing a proxy always lowers the true reward', 'Distance from the starting policy grows like the log of the search']
    why: 'The reward model was built to measure what people like, and step 3 turns it into the number the policy pushes up.'
  - kind: term
    q: 'What is $r^*(y)$ in this unit?'
    answer: 'The score a person would give the response'
    wrong: ['The score the reward model gives the response', 'The highest reward any response can get', 'The reward model''s error on the response']
    why: '$r^*$ is the true reward. The reward model''s score is $r_\phi = r^* + e$, where $e$ is its error.'
  - kind: term
    q: 'What is an extremal error in a reward model?'
    answer: 'Error that is small where it was trained and large elsewhere'
    wrong: ['Random noise added the same way to every score', 'Error from labelers disagreeing on the same pair', 'Error that shrinks as the policy moves from $\pi_0$']
    why: 'Extremal error grows away from the training data. Once the search gets there, the true reward can fall while the score keeps rising.'
  - kind: term
    q: 'What does best-of-$n$ do?'
    answer: 'Draws $n$ responses from $\pi_0$ and keeps the one the reward model scores highest'
    wrong: ['Trains the policy for $n$ steps against the reward model', 'Draws $n$ responses from $\pi_0$ and keeps the one people prefer', 'Draws $n$ responses from $\pi_0$ and averages their reward-model scores']
    why: 'Nothing gets trained. The reward model picks the winner, so it can be fooled the same way it can be during RL.'
  - kind: term
    q: 'In Gao et al., what was the "gold" reward model?'
    answer: 'A large reward model that stood in for people'
    wrong: ['The proxy reward model the policies were trained against', 'The best policy found at the peak of the curve', 'A reward model trained only on $\pi_0$''s answers']
    why: 'Labeling everything with people was too expensive, so a 6-billion-parameter reward model from InstructGPT played the people. It labeled the comparisons the proxies trained on.'
  - kind: apply
    q: 'What is the KL divergence from $\pi_0$ of best-of-8 (natural log)?'
    answer: '$1.20$ nats'
    wrong: ['$2.08$ nats', '$0.88$ nats', '$2.95$ nats']
    why: '$\log 8 - 7/8 = 2.08 - 0.875 = 1.20$. The $2.08$ forgets the second term.'
  - kind: apply
    q: 'In the toy, $r^*(x) = x - x^2/10$. What is the true reward of a pick at $x = 8$?'
    answer: '$1.6$'
    wrong: ['$8$', '$2.5$', '$7.2$']
    why: '$8 - 64/10 = 1.6$. The reward model''s line says 8 (plus noise), and $7.2$ comes from forgetting to square $x$.'
  - kind: apply
    q: 'In the toy, which $x$ do people like best?'
    answer: '$x = 5$'
    wrong: ['$x = 10$', '$x = 3$', '$x = 0$']
    why: 'The slope of $x - x^2/10$ is $1 - x/5$, which is 0 at $x = 5$, where $r^* = 2.5$. At $x = 10$ the true reward is back down to 0.'
  - kind: apply
    q: 'In the toy, how far off is the reward model at $x = 8$, ignoring the noise $\varepsilon$?'
    answer: 'It scores $6.4$ too high'
    wrong: ['It scores $1.6$ too high', 'It scores $0.8$ too high', 'It scores $6.4$ too low']
    why: 'The error is $e(x) = \varepsilon + x^2/10$, and $64/10 = 6.4$. The line says 8 while people give $1.6$.'
  - kind: apply
    q: 'Swap the toy''s reward model for one with only noise, $r^*(x) + \varepsilon$. What happens to the true reward of best-of-$n$ as $n$ grows?'
    answer: 'It keeps rising toward 2.5'
    wrong: ['It rises and then falls past $n = 16$', 'It stays where $\pi_0$ started', 'It falls once the picks pass $x = 5$']
    why: 'With independent noise a higher score is still evidence of a higher $r^*$, so the search keeps finding better responses. The worked example''s last column climbs on every row.'
  - kind: apply
    q: 'Gao et al. ran RL with a bigger KL coefficient $\beta$. What changed about the gold score?'
    answer: 'The run leveled off at a smaller KL, on the same gold-versus-KL curve'
    wrong: ['The gold-versus-KL curve moved up, so the peak got higher', 'The run leveled off at a larger KL, on the same curve', 'The gold score kept rising instead of falling']
    why: 'The gold score depended only on how far the policy had moved, so a bigger penalty acted like stopping training earlier.'
  - kind: apply
    q: 'Your policy has learned that the reward model loves very long answers. Which move goes after the reward model''s error itself?'
    answer: 'Have people compare the current policy''s answers, then retrain the reward model'
    wrong: ['Raise $\beta$ so the policy stops closer to $\pi_0$', 'Have people compare more of $\pi_0$''s answers, then retrain', 'Switch from RL to best-of-$n$ with a large $n$']
    why: 'The error lives where the policy is now and $\pi_0$ rarely went, so new comparisons there correct it. Raising $\beta$ only changes where you stop.'
---

::: why
A teacher who is graded only on test scores can raise them by drilling the test.
The scores go up, and the students don't learn more.
RLHF has the same weak spot.
Unit 16 ended with a policy trained to get high scores from a reward model.
Push that training too far and the policy finds answers that score high without being better.
That's *reward hacking*: the score keeps climbing while the answers people would actually prefer get worse.
:::

## The score is a proxy

*Goodhart's law*, in the wording [Gao et al.](https://arxiv.org/abs/2210.10760) quote, says "When a measure becomes a target, it ceases to be a good measure."
RLHF's step 3 is that situation.
The reward model was built to measure what people like.
Then it becomes the number the policy is trained to push up.

Two earlier units showed smaller versions of this.
In unit 6, training loss kept falling while [validation loss climbed back up](../06-generalization/#weight-decay-early-stopping-and-more-data), because the training loss was only a stand-in for error on new data.
In unit 7, a [benchmark](../07-evaluation/#a-score-needs-a-reference-point) only measured something while its answers stayed out of training.
Here the stand-in is a learned model, and the optimizer pushes on it directly.

What you want is answers people prefer.
Call the score a person would give a response $y$ its *true reward*, $r^*(y)$.
(I leave the prompt out of the notation in this unit, since everything here happens for one prompt at a time.)
Nobody can ask a person about every response during training.
So step 3 of RLHF optimizes the reward model's score $r_\phi(y)$ instead.
That score is the *proxy*, a stand-in you can measure for the thing you actually want.

Split the proxy into the part that's right and the part that's wrong:

$$
r_\phi(y) = r^*(y) + e(y)
$$

The reward model's score on $y$ is the true reward plus $e(y)$.
That $e(y)$ is the reward model's *error* on $y$, and it can be positive or negative.
Say people would give an answer a 2 and the reward model gives it a 3.
Then the error is 1.

The optimizer only ever sees the sum.
When it searches for responses with a high $r_\phi$, it finds responses where $r^*$ is high.
It also finds responses where $e$ is high, and it can't tell the two apart.

Gao et al. sort these errors using a taxonomy from Manheim and Garrabrant.
Two of its kinds matter here.

- **Regressional:** $e$ is random noise. Part of the search goes into picking responses that got lucky noise. The true reward rises less than the score, but it still rises.
- **Extremal:** $e$ is small where the reward model was trained and large somewhere else. Once the search reaches somewhere else, the true reward can fall while the score keeps going up.

Gao et al. show that noise alone can't make the true reward go down as the score goes up.
(This holds when the noise is independent of the true reward and well behaved, normal for example.)
So when the true reward does go down, the second kind is responsible.

## Why the errors grow as training goes on

The reward model learned from people's picks between $\pi_0$'s answers (unit 16).
It has only ever been checked on answers that look like $\pi_0$'s.
The usual check is agreement with held-out human picks (unit 7).
That check holds out comparisons between $\pi_0$'s answers too, and a [held-out set](../06-generalization/#train-validation-and-test) only tells you about data drawn the same way.

Step 3 then moves the policy $\pi_\theta$ away from $\pi_0$, on purpose, toward whatever $r_\phi$ scores highly.
The further it moves, the less its answers look like anything in the training data.
That gives $e$ more room to be large.
The optimizer also prefers the places where $e$ is large and positive, because those score well.

Gao et al.'s example of the second kind is answer length (sec. 4.2.2).
If longer answers were usually the better ones in the comparison data, a reward model can learn that longer is always better.
Past some length that stops being true.
They report policies writing very long answers when a short one would do.

To see both kinds of error with real numbers, I use the same toy as my blog post.
Each response is boiled down to one number, $x$, standing in for something like length.
(In this unit $x$ is that number, not the prompt.)
$\pi_0$ writes responses with $x \sim \mathcal{N}(0, 3^2)$, a [normal distribution](../03-probability/#random-variables-and-distributions) with mean 0 and standard deviation 3.
People like more $x$ up to $x = 5$ and less after that:

$$
r^*(x) = x - \frac{x^2}{10}
$$

::: formula
The $x$ term says more is better, and the $-x^2/10$ term is a penalty that grows faster.
The two balance at $x = 5$, where the true reward is $5 - 2.5 = 2.5$.
:::

So the true reward peaks at $x = 5$, at 2.5, and falls after that.

About 95% of $\pi_0$'s responses have $x$ below 5.
So in the comparisons people made, the answer with more $x$ was usually the better one.
The reward model learned a straight line plus noise:

$$
r_\phi(x) = x + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, 1)
$$

::: formula
$\varepsilon$ (epsilon) is a fresh draw of noise from a normal distribution with mean 0 and standard deviation 1.
:::

So the score is $x$ plus random noise $\varepsilon$.
Subtract $r^*(x)$ and the error is $e(x) = \varepsilon + x^2/10$.
The noise $\varepsilon$ is the regressional part.
The $x^2/10$ is the extremal part.
It's small near 0, where $\pi_0$ spends its time.
It's large past 5, where $\pi_0$ rarely goes and the reward model never got a correction.

The demo draws the toy and measures the error at three places.

<figure class="widget demo" id="demo-proxy"></figure>

## Best-of-n, and measuring how far it moves

*Best-of-$n$* does the same kind of search as RL in a much simpler way.
Draw $n$ responses from $\pi_0$ and keep the one the reward model scores highest.
At $n = 1$ you get $\pi_0$ itself, and a larger $n$ searches harder.
Gao et al. study it alongside RL.

Here it is on the toy, starting with 4 draws.

<figure class="widget demo" id="demo-pick"></figure>

To compare amounts of search you need a measure of how far a policy has moved from $\pi_0$.
The standard one is the KL divergence, the same quantity [unit 16's penalty](../16-rlhf/#the-three-steps) uses:

$$
D_{\mathrm{KL}}(\pi \,\|\, \pi_0) = \mathbb{E}_{y \sim \pi}\left[\log \frac{\pi(y)}{\pi_0(y)}\right]
$$

::: formula
$\pi(y)$ and $\pi_0(y)$ are the probabilities the new policy and the starting policy give response $y$, and $\mathbb{E}_{y \sim \pi}$ is the [expectation](../03-probability/#expectation-spread-and-sampling) over responses drawn from $\pi$.
:::

In words: draw responses from the new policy $\pi$.
For each one, take the log of how many times more likely $\pi$ makes it than $\pi_0$ did.
Then average.
The result is 0 when the two policies are the same.
It grows as $\pi$ moves its probability onto responses $\pi_0$ rarely wrote.
With the natural log the unit is the *nat*, the natural-log version of a bit.

For best-of-$n$ it has a closed form:

$$
D_{\mathrm{KL}}(\pi_{\mathrm{bon}} \,\|\, \pi_0) = \log n - \frac{n-1}{n}
$$

::: formula
$\pi_{\mathrm{bon}}$ is the policy "draw $n$ from $\pi_0$ and keep the top-scoring one", and $n$ is the number of draws.
:::

Here's where the two terms come from.

For a response $y$, let $u$ be the fraction of $\pi_0$'s responses that the reward model scores below $y$.
So $u$ is between 0 and 1.
Best-of-$n$ returns $y$ when one of the $n$ draws is $y$ and the other $n - 1$ all score below it.
There are $n$ draws that could be $y$, so

$$
\pi_{\mathrm{bon}}(y) = n\, u^{n-1}\, \pi_0(y)
$$

::: formula
Read right to left: $\pi_0(y)$ is the chance one draw is $y$, $u^{n-1}$ is the chance the other $n - 1$ draws all score below it, and the $n$ counts which draw it was.
Dividing by $\pi_0(y)$ and taking the log gives the log ratio below.
The winner's $u$ is the largest of $n$ uniform numbers between 0 and 1, which has density $n u^{n-1}$, and the average of $\log u$ under that density is $\int_0^1 n u^{n-1} \log u \, du = -1/n$.
So the $\log n$ in the formula comes from the factor $n$, and the $-(n-1)/n$ is $(n-1)$ times that average $-1/n$.
:::

So the log ratio inside the KL is $\log n + (n-1)\log u$.
Averaging it over the winners gives the formula above.

The derivation assumes two different responses never get exactly the same score, which holds in the toy.
For real text the same response can be drawn twice.
There, [Beirami et al.](https://arxiv.org/abs/2401.01879) show the formula is an upper bound on the true KL rather than exact.

The formula grows like $\log n$, which is slow.
Best-of-16 is 1.84 nats from $\pi_0$.
Best-of-4096 does 256 times the search and is 7.32 nats away.

The demo plots the worked example's results at each $n$, placed by KL.

<figure class="widget demo" id="demo-curve"></figure>

The widget runs the same search live.
Set $n$ yourself and watch where the picks land.

<figure class="widget" id="best-of-n">
<figcaption>Best-of-n on the toy, running live. Left: π₀'s spread of x (grey), the reward model's straight line (blue, dashed) and what people actually think (orange), with the last few picks scored both ways. Right: average scores at every n, placed by KL. Slide n and watch the picks move past x = 5, where the line and the curve split.</figcaption>
</figure>

At small $n$ both colors rise together, because near $\pi_0$ the line and the curve agree.
Around $n = 16$ the average pick reaches $x = 5$, the top of the orange curve.
After that the picks keep moving right.
The blue score keeps climbing, and the orange score falls.
The readout shows how unusual those picks are for $\pi_0$, which is the reason nobody corrected the reward model there.

## What Gao et al. measured

[Gao, Schulman and Hilton (2023)](https://arxiv.org/abs/2210.10760) ran this on real language models.
Labeling everything with people was too expensive.
So a 6-billion-parameter reward model from InstructGPT stood in for people.
They call it the *gold reward model*.
It labeled 100,000 synthetic comparisons, which trained proxy reward models from 3 million to 3 billion parameters.
They optimized policies against the proxies with best-of-$n$ and with PPO, and tracked the gold score along the way.
The gold model plays the part of unit 6's test set.
The policy is never trained against it, so its score isn't inflated by the search.

The gold score rose and then fell, the same shape as the toy.
They plotted it against $d = \sqrt{D_{\mathrm{KL}}}$, the square root of the KL from $\pi_0$, and found these fits:

$$
R_{\mathrm{bon}}(d) = d\,(a - b\,d), \qquad R_{\mathrm{RL}}(d) = d\,(a - b \log d)
$$

::: formula
$R_{\mathrm{bon}}(d)$ and $R_{\mathrm{RL}}(d)$ are the gold score at distance $d$, counted from where $\pi_0$ started, for best-of-$n$ and for RL.
Each method gets its own $a$ and $b$.
The paper calls them $\alpha$ and $\beta$; I renamed them because $\beta$ is already the KL coefficient.
Multiplied out, best-of-$n$ is $a d - b d^2$ and RL is $a d - b\, d \log d$.
:::

In both, $a$ and $b$ are numbers fitted to the data.
The $a d$ part is the early rise.
The $b$ term grows faster than $a d$ for large $d$, so it eventually pulls the score back down.

Other results from the paper:

- Larger proxy reward models overoptimize less, and the coefficients change smoothly enough with size to predict the peak gold score.
- More comparison data helps too. Below about 2,000 comparisons the proxies were barely better than chance.
- RL uses up far more KL than best-of-$n$ for the same change in score, so KL is a fair ruler within one method but not between methods.

## Two things people do about it

The first is the KL penalty from [unit 16](../16-rlhf/#the-three-steps):

$$
\max_\theta\; \mathbb{E}_{y \sim \pi_\theta}\big[r_\phi(y)\big] - \beta\, D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_0)
$$

It's unit 16's step 3 objective with the prompt left out.
Take the average reward-model score of the policy's responses, and subtract $\beta$ times the KL from $\pi_0$.
Every nat of distance from $\pi_0$ costs $\beta$ points of reward.
The policy keeps moving outward while extra distance still buys more score than it costs.
So a larger $\beta$ settles closer to $\pi_0$.

Gao et al. ran RL with several values of $\beta$ (sec. 3.6).
The gold score depended only on how far the policy had moved.
A bigger penalty made a run level off at a smaller KL, but on the same gold-versus-KL curve.
So its effect was like [stopping training early](../06-generalization/#weight-decay-early-stopping-and-more-data).
They add that this result might be sensitive to hyperparameters.
In the widget's terms, choosing $\beta$ is like choosing where to leave the slider.
It changes where you stop, and the orange curve stays the same.

The second fix goes after the errors themselves.
If the reward model is wrong in places it never saw, collect comparisons from those places.
It's the same answer unit 6 gives for a test set that's been used up: fresh data, here drawn from where the policy now writes.

[Bai et al. (2022)](https://arxiv.org/abs/2204.05862) at Anthropic found their preference models got less reliable at high scores.
They attributed it to having little data there.
Their fix is called *iterated online RLHF*, and it runs on a weekly cadence.
Train the best policy you can, and have crowdworkers compare that policy's answers.
Mix those comparisons into the existing data.
Then retrain the preference models, and after them the policies.
Crowdworkers clearly preferred the resulting models over ones trained only on the original data.

In the toy, comparisons between answers around $x = 5$ to $8$ would show the reward model that people stop wanting more $x$, and the line would bend.
Gao et al. (sec. 4.3) extended their RL fit to this kind of iteration.
Under simplifying assumptions, they predicted that it raises the final gold score through the $b$ term, which they attribute mostly to distribution shift.
It leaves the part they tie to noise unchanged.
That matches the toy: fresh comparisons can teach the reward model the missing $x^2/10$, but they can't remove $\varepsilon$.

My post [Reward hacking from the ground up](../../posts/your-reward-model-is-an-attack-surface/) tells this as one story, including why it reminded me of fuzzing.

## Worked example

This runs best-of-$n$ on the toy with NumPy and prints the average over 2,000 trials at each $n$.
The `check` column estimates the KL directly from the derivation, by averaging $\log n + (n-1)\log u$ over the winners.
(The reward model's scores on $\pi_0$'s responses are $\mathcal{N}(0, 10)$, so $u$ comes from the normal CDF.)
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

The `check` column matches the formula to within 0.04.
The proxy score rises on every row.
The true reward peaks at $n = 16$, where the average pick is $x = 5.01$.
With noise as the only error, the true reward rises on every row toward its maximum of 2.5.
So the fall in the `true` column comes from the straight line.

::: real-world
- **Feeds tuned for clicks.** A feed that ranks posts by predicted clicks is optimizing a proxy for what people want to read. Clickbait is what a high score on that proxy looks like when the proxy is wrong.
- **Game agents.** OpenAI trained an agent on a boat racing game with points as the reward. It learned to circle one spot and hit the same bonus targets over and over instead of finishing the race.
- **Fuzzing.** A fuzzer throws huge numbers of unusual inputs at a program to find the ones its developers never tested. A policy searching against a reward model does the same thing to the reward model, which is why I think of the reward model as an attack surface.
:::

## Where this goes next

- Unit 16's DPO has no separate reward model, but [Rafailov et al. (2024)](https://arxiv.org/abs/2406.02900) found DPO-style methods degrade in a similar way at larger KL budgets, often before a single pass over the data.
- Best-of-$n$ also works as a decoding method next to the ones in unit 14: at generation time, sample $n$ responses and return the one a scorer likes best, with $n$ trading compute for distance from $\pi_0$.
- Everything here treats the policy as a search with no goals of its own. Gao et al. (sec. 4.2.4) note that a model able to game the reward model on purpose could break their trends, and the last section of [my blog post](../../posts/your-reward-model-is-an-attack-surface/) points at that problem.
- Unit 18 puts reward hacking next to the other ways a model ends up doing something people didn't mean, learning the wrong goal from a correct reward and learning to agree with its raters, and looks at how people supervise and test models for them.

::: review
Goodhart's law
:   "When a measure becomes a target, it ceases to be a good measure."

True reward, $r^*(y)$
:   The score a person would give a response. It's what you actually want to raise.

Proxy
:   A stand-in you can measure for the thing you want. Here it's the reward model's score $r_\phi(y)$.

Error, $e(y)$
:   The gap between the proxy and the true reward: $r_\phi(y) = r^*(y) + e(y)$.

Regressional error
:   Error that is random noise. It slows the true reward's rise but can't reverse it.

Extremal error
:   Error that is small where the reward model was trained and large elsewhere. It can make the true reward fall while the score rises.

Reward hacking
:   The policy finds responses that score high because the reward model is wrong about them. Gao et al. call it overoptimization.

Best-of-$n$
:   Draw $n$ responses from $\pi_0$ and keep the one the reward model scores highest.

KL divergence, in nats
:   How far a policy has moved from $\pi_0$. For best-of-$n$ it's $\log n - (n-1)/n$.

Gold reward model
:   A large reward model that stands in for people when measuring true reward. The policy is never trained against it.

Iterated online RLHF
:   Collect new comparisons on the current policy's answers, then retrain the reward model and the policy.
:::

::: say
The reward model's score is a proxy for what people want, and it's only accurate near the answers it was trained on.
Search hard for high scores and you move away from those answers, into places where the score is wrong.
There the score keeps rising while the true reward falls.
A KL penalty limits how far the policy moves, which works like stopping early.
Collecting fresh comparisons where the policy now writes fixes the errors themselves.
:::

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
