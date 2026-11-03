---
title: Probability basics
subtitle: Distributions, averages, conditional probability, and why models work in logs.
unit: 3
section: Math
minutes: 55
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'What does $\mathbb{E}[X]$ mean for a random variable $X$?'
    answer: 'The average of its values, each weighted by its probability'
    wrong: ['Its single most likely value', 'The middle value when its values are sorted', 'The largest value it can take']
    why: '$\mathbb{E}[X] = \sum_x x\,P(X = x)$. For a fair die it''s 3.5, which isn''t a face you can roll.'
  - kind: term
    q: 'What does $P(A \mid B)$ mean?'
    answer: 'The probability of $A$, counting only the cases where $B$ happened'
    wrong: ['The probability that $A$ and $B$ both happen', 'The probability of $B$, counting only the cases where $A$ happened', 'The probability that $A$ happens and $B$ doesn''t']
    why: 'It''s $P(A, B) / P(B)$: throw away every case where $B$ didn''t happen, then ask what share of the rest also has $A$.'
  - kind: term
    q: 'What does it mean for two events to be independent?'
    answer: 'Knowing one happened doesn''t change the probability of the other'
    wrong: ['They can never both happen', 'Their probabilities add up to 1', 'One of them always happens first']
    why: 'Independence means $P(A \mid B) = P(A)$, which is the same as $P(A, B) = P(A)\,P(B)$. Events that can''t both happen are the opposite case: knowing one happened tells you the other didn''t.'
  - kind: term
    q: 'What does the standard deviation $\sigma$ of a random variable measure?'
    answer: 'How far its values typically land from the mean'
    wrong: ['How far its mean is from zero', 'The gap between its largest and smallest values', 'How likely its most likely value is']
    why: 'It''s the square root of the variance, the average squared distance from $\mathbb{E}[X]$. The square root puts it back in the same units as $X$.'
  - kind: term
    q: 'In $\mathcal{N}(\mu, \sigma^2)$, what are the two numbers?'
    answer: 'The mean and the variance'
    wrong: ['The mean and the standard deviation', 'The peak height and the width', 'The smallest and largest values']
    why: 'The second number is the variance, $\sigma$ squared. So $\mathcal{N}(0, 3^2)$ has mean 0 and standard deviation 3.'
  - kind: term
    q: 'What is a Monte Carlo estimate of an expectation?'
    answer: 'The average of many random samples, used in place of the exact sum'
    wrong: ['The exact expectation, summed over every possible value', 'The value that came up most often in a batch of samples', 'The largest value seen in a batch of samples']
    why: 'When there are too many outcomes to sum over (every possible response of a language model, say), you draw samples and average them. The average closes in on the expectation as you draw more.'
  - kind: apply
    q: 'A random variable is 2 with probability 0.75 and 10 with probability 0.25. What is its expectation?'
    answer: '$4$'
    wrong: ['$6$', '$2$', '$12$']
    why: '$2 \times 0.75 + 10 \times 0.25 = 1.5 + 2.5 = 4$. Averaging 2 and 10 without the weights gives 6, which treats them as equally likely.'
  - kind: apply
    q: 'An average of 100 samples typically misses the true expectation by 0.2. About how many samples does it take to typically miss by 0.02?'
    answer: '10,000'
    wrong: ['1,000', '200', '1,000,000']
    why: 'The typical miss is $\sigma / \sqrt{N}$. Making it 10 times smaller takes $10^2 = 100$ times as many samples.'
  - kind: apply
    q: 'A filter flags 90% of phishing emails and 5% of legit ones, and 10% of all email is phishing. What share of flagged emails is phishing?'
    answer: 'About 67%'
    wrong: ['About 90%', 'About 10%', 'About 95%']
    why: 'Out of 1,000 emails, 100 are phishing and 90 of those get flagged, while 900 are legit and 45 of those get flagged. That''s $90 / 135 \approx 0.67$.'
  - kind: apply
    q: 'You roll two fair dice. What is the probability that both show 6?'
    answer: '$1/36$'
    wrong: ['$1/6$', '$1/3$', '$1/12$']
    why: 'The two rolls are independent, so the probabilities multiply: $1/6 \times 1/6 = 1/36$.'
  - kind: apply
    q: 'A model gives three tokens in a row probabilities 0.5, 0.5 and 0.1. What is the log-probability of the three-token text?'
    answer: '$-3.69$'
    wrong: ['$0.025$', '$0.095$', '$-1.11$']
    why: 'The log of a product is the sum of the logs: $2\log 0.5 + \log 0.1 = -1.39 - 2.30 = -3.69$, which is $\log 0.025$.'
  - kind: apply
    q: 'In the Bayes widget, you lower the false alarms from 5% to 1% and leave the other two sliders at 1% and 90%. What happens to P(phish | flag)?'
    answer: 'It rises from about 0.15 to about 0.48'
    wrong: ['It stays at about 0.15, since the catch rate didn''t change', 'It rises from about 0.15 to about 0.90', 'It falls, because fewer emails get flagged']
    why: 'The flagged legit emails drop from 0.0495 to 0.0099 of all email, so the 0.009 that''s phishing becomes $0.009 / 0.0189 \approx 0.48$ of the flagged pile.'
---

Every prediction a language model makes is a list of probabilities, one for each token in its vocabulary.
The loss in unit 11, the sampling settings in unit 14 and the reward averages in unit 15 are all arithmetic on lists like that.
The arithmetic is small, and two examples carry it here: a six-sided die and a phishing filter.

## Random variables and distributions

Roll a fair die and call the result $X$.
$X$ is a *random variable*: a number whose value comes out of a random process.
Its *distribution* lists every value it can take and the probability of each.
For the die that's $P(X = k) = 1/6$ for each face $k$ from 1 to 6, where $P(X = k)$ reads "the probability that $X$ comes out as $k$."

Every distribution follows two rules.
Each probability is between 0 and 1, and all of them add up to exactly 1, because some outcome always happens.

The outcomes don't have to be numbers.
After "I'd like a cup of", a model might give tea 0.5, coffee 0.4 and water 0.1.
That's a categorical distribution: one probability for each label in a fixed set.
A language model outputs one of these at every position, with one label per token in its vocabulary (unit 11 builds it).

Some quantities can land anywhere on a continuous scale, like a person's height or a model's raw score.
For those, the distribution is a curve called a density, written $p(x)$.
The probability of landing in a range is the area under the curve over that range, and the total area is 1, which is the "adds up to 1" rule again.
A single exact value has probability 0, since a single point has no width.

The density you'll see most is the normal distribution, the bell curve:

$$
p(x) = \frac{1}{\sigma\sqrt{2\pi}}\; e^{-(x - \mu)^2 / (2\sigma^2)}
$$

::: formula
$e \approx 2.718$ is the base of the natural log, and $\pi \approx 3.14$ is the usual pi.
Read the exponent from the inside out.
$(x - \mu)/\sigma$ is how far $x$ is from the center, counted in widths.
Squaring it, halving it and putting a minus sign in front means the curve is highest at $x = \mu$ and falls off fast as you move away in either direction.
The factor in front, $1/(\sigma\sqrt{2\pi})$, is only there to make the total area come out to 1.
A wider bell is lower, so the area stays the same.
:::

It has two settings: $\mu$ (mu) is the center of the bell, and $\sigma$ (sigma) is its width.
The next section names them properly as the mean and the standard deviation.

The usual shorthand is $\mathcal{N}(\mu, \sigma^2)$, and "$X \sim \mathcal{N}(0, 3^2)$" reads "$X$ is drawn from a normal distribution centered at 0 with width 3."
The second number is $\sigma^2$, not $\sigma$, which is easy to trip on.
About 68% of the area lies within one $\sigma$ of $\mu$, and about 95% within two.

## Expectation, spread and sampling

The *expectation* of a random variable, also called its *mean*, is the average of its values with each one weighted by its probability:

$$
\mathbb{E}[X] = \sum_x x \, P(X = x)
$$

::: formula
The $\sum_x$ means "add this up over every value $x$ that $X$ can take."
Each term is a value times how likely it is, so likely values pull the answer toward themselves and unlikely ones barely count.
:::

For the die every face has weight $1/6$, so $\mathbb{E}[X] = (1 + 2 + 3 + 4 + 5 + 6)/6 = 3.5$.
No roll ever comes up 3.5.
It's the number the average of many rolls settles toward, which the widget below shows.

The same sum works for anything computed from $X$.
Swap the value $x$ for $f(x)$ and you get $\mathbb{E}[f(X)] = \sum_x f(x)\,P(X = x)$, the average of $f$ over the distribution.
In unit 15 the outcome is an action and $f$ is the reward it earns, and training pushes the expected reward up.

The expectation says nothing about spread.
A variable that's always exactly 3.5 has the same mean as the die.
The *variance* measures spread as the average squared distance from the mean:

$$
\mathrm{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big]
$$

::: formula
Inside the brackets, $X - \mathbb{E}[X]$ is how far one outcome lands from the mean.
Squaring makes misses on both sides count as positive, and the outer $\mathbb{E}$ averages those squares with the same probability weights as before.
:::

For the die the squared distances from 3.5 are $6.25, 2.25, 0.25, 0.25, 2.25, 6.25$, and their average is $17.5 / 6 \approx 2.92$.
Squaring also squares the units, so the usual measure of spread is the square root of the variance, the standard deviation $\sigma = \sqrt{2.92} \approx 1.71$.
That's the same $\sigma$ as the [normal distribution's width](#random-variables-and-distributions).

Drawing a value at random from a distribution is called sampling.
Roll the die $N$ times and average the results, and you have an estimate of $\mathbb{E}[X]$.
As $N$ grows, the average closes in on 3.5.
Estimating an expectation by averaging samples like this is called a Monte Carlo estimate.

How close it gets has a formula.
If the samples don't influence each other (independent, defined in the next section), the average of $N$ of them misses the true mean by a typical amount of

$$
\frac{\sigma}{\sqrt{N}}
$$

where $\sigma$ is the standard deviation of a single sample.
More spread-out samples give a noisier average, and more samples give a steadier one, but only by the square root.
With 100 rolls the average typically misses 3.5 by about $1.71 / 10 = 0.17$, and with 10,000 rolls by about 0.017.
Each extra digit of accuracy costs 100 times as many samples.

For a die you'd skip all this and do the sum.
For a language model you can't: the possible responses to one prompt are every sequence of tokens, far too many to list, so expected values over responses are always estimated from samples.

<figure class="widget" id="dice">
<figcaption>Rolls of a die. Left: the share of rolls that landed on each face (blue) against the true probabilities (orange ticks). Right: the running mean after N rolls, on a log scale, against E[X] (orange). The dashed curves sit at E[X] ± σ/√N, the typical miss after N rolls. Press play, roll by hand, or switch to a die that lands on 6 half the time.</figcaption>
</figure>

The loaded die has $\mathbb{E}[X] = 4.5$ and $\sigma \approx 1.80$, so its funnel is a little wider and centered higher.
On the log axis, every tenfold step in rolls narrows the funnel by the same factor, $\sqrt{10} \approx 3.2$.
After 10,000 rolls the typical miss is 0.017 for the fair die and 0.018 for the loaded one.

## Conditional probability and Bayes' rule

The phishing filter from the intro: 1% of the email it sees is phishing, it flags 90% of the phishing, and it also flags 5% of the legit email.
Each email has two random facts attached, phishing or legit, and flagged or not.
The joint probability $P(A, B)$ is the probability that both $A$ and $B$ happen.
The four combinations cover every email, so their joint probabilities add up to 1:

|          | flagged | not flagged | total |
|----------|--------:|------------:|------:|
| phishing |  0.009  |    0.001    |  0.01 |
| legit    |  0.0495 |    0.9405   |  0.99 |
| total    |  0.0585 |    0.9415   |  1    |

A conditional probability $P(A \mid B)$, read "$A$ given $B$," is the probability of $A$ counting only the cases where $B$ happened:

$$
P(A \mid B) = \frac{P(A, B)}{P(B)}
$$

::: formula
The numerator is the share of all cases where both happen, and dividing by $P(B)$ rescales it so that "every case where $B$ happened" counts as the whole.
:::

In the table, that means staying inside one row or column.
$P(\text{flag} \mid \text{phish}) = 0.009 / 0.01 = 0.9$, the catch rate.
Rearranged, $P(A, B) = P(A \mid B)\,P(B)$, which is how I filled in the table: 90% of the 1% is 0.009.

Two events are independent when knowing one happened doesn't change the probability of the other, $P(A \mid B) = P(A)$.
Put that into the rearranged formula and you get $P(A, B) = P(A)\,P(B)$: for independent events, the joint probability is the product.
Two dice are independent, so the chance both show 6 is $1/6 \times 1/6 = 1/36$.
Flags and phishing are not independent, and shouldn't be: 5.85% of all email gets flagged, but 90% of phishing does.

When the filter flags an email, the number you want is the reverse of the catch rate, $P(\text{phish} \mid \text{flag})$.
The joint probability can be split either way, $P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$, and dividing both sides by $P(B)$ gives Bayes' rule:

$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
$$

::: formula
$P(A)$ is what you'd believe before seeing $B$, and $P(A \mid B)$ is what you believe after.
$P(B \mid A)$ says how likely the evidence is when $A$ is true, and $P(B)$ is how likely the evidence is overall.
:::

It turns a probability you know in one direction into the other direction, from the *prior* $P(A)$ to the *posterior* $P(A \mid B)$.
For the filter, $P(\text{flag})$ is the "flagged" column total, the flagged phishing plus the flagged legit:

$$
\begin{aligned}
P(\text{phish} \mid \text{flag}) &= \frac{0.9 \times 0.01}{0.9 \times 0.01 + 0.05 \times 0.99} \\
&= \frac{0.009}{0.0585} \approx 0.154
\end{aligned}
$$

So a flagged email is phishing only about 15% of the time, even though the filter catches 90% of phishing.
Out of 10,000 emails, 100 are phishing and 90 of those get flagged, while 9,900 are legit and 495 of those get flagged.
Only 90 of the 585 flagged emails are phishing, because the 5% false-alarm rate applies to 99 times as many emails as the 90% catch rate does.

<figure class="widget" id="bayes">
<figcaption>1,000 emails, one square each (counts rounded to whole emails). Orange squares are phishing, solid if the filter flagged them and faint if it missed them. Solid blue squares are legit emails it flagged, and faint grey ones are legit and left alone. The flagged pile is all the solid squares, and P(phish | flag) is the orange share of it. Base rate is P(phish), catch rate is P(flag | phish), and false alarms is P(flag | legit).</figcaption>
</figure>

## Why models work in log-probabilities

The rearranged conditional formula, $P(A, B) = P(A \mid B)\,P(B)$ from [the previous section](#conditional-probability-and-bayes-rule), chains.
The probability of a whole sequence is the probability of the first item, times the probability of the second given the first, and so on.
For a text of tokens $x_1, \dots, x_n$:

$$
P(x_1, \dots, x_n) = P(x_1)\; P(x_2 \mid x_1) \cdots P(x_n \mid x_1, \dots, x_{n-1})
$$

::: formula
Each factor is the probability of token $x_t$ given every token before it.
:::

Each factor is one next-token prediction, exactly what a language model outputs, so multiplying its predictions along a text gives the probability of the whole text.
If the model gives "of" probability 0.6 after "I'd like a cup" and "tea" probability 0.5 after "I'd like a cup of", the two tokens together get $0.6 \times 0.5 = 0.3$.

A real response is hundreds or thousands of tokens, and every factor is below 1.
A 1,000-token text where every token got probability 0.5 has probability about $9.3 \times 10^{-302}$, and at 2,000 tokens the product drops below the smallest number a 64-bit float can hold (about $5 \times 10^{-324}$) and comes out as exactly 0.
The text didn't become impossible; the computer just can't represent its probability.

Logs fix this, because the log of a product is the sum of the logs:

$$
\log P(x_1, \dots, x_n) = \sum_{t=1}^{n} \log P(x_t \mid x_1, \dots, x_{t-1})
$$

::: formula
Each term is the log of one next-token probability, and the sum adds them up over the $n$ tokens.
Logs here are natural logs (base $e$).
:::

A probability between 0 and 1 has a log between $-\infty$ and 0, with $\log 1 = 0$ and $\log 0.5 \approx -0.69$.
The 2,000-token text's log-probability is $2000 \times \log 0.5 \approx -1386.3$, an ordinary number.
And since the log only goes up as its input goes up, whatever has the highest probability also has the highest log-probability, so making one bigger makes the other bigger.

So ML code rarely handles the raw probability of a whole text: libraries return log-probabilities, and unit 11's training loss is the average of $-\log p$ over the tokens.

## Worked example

```python
import numpy as np

faces = np.arange(1, 7)
p = np.full(6, 1 / 6)                      # a fair die
mean = np.sum(faces * p)                   # E[X]
var = np.sum((faces - mean) ** 2 * p)      # Var(X)
print(p.sum(), mean, var, np.sqrt(var))

# Average N rolls, 1,000 separate times, and measure how spread out the averages are.
rng = np.random.default_rng(0)
for n in [10, 100, 1_000, 10_000]:
    averages = rng.integers(1, 7, size=(1_000, n)).mean(axis=1)
    print(n, averages[0], averages.std(), np.sqrt(var / n))

# Share of N(0, 3^2) samples within one and two standard deviations of the mean.
x = rng.normal(0, 3, size=100_000)
print(np.mean(np.abs(x) < 3), np.mean(np.abs(x) < 6))

# Bayes: 1% of emails are phishing; the filter flags 90% of those and 5% of the rest.
p_phish, p_flag_if_phish, p_flag_if_legit = 0.01, 0.90, 0.05
p_flag = p_flag_if_phish * p_phish + p_flag_if_legit * (1 - p_phish)
print(p_flag, p_flag_if_phish * p_phish / p_flag)

# A 2,000-token text where the model gave every token probability 0.5.
probs = np.full(2_000, 0.5)
print(np.prod(probs[:1_000]), np.prod(probs), np.sum(np.log(probs)))
```

```
0.9999999999999999 3.5 2.9166666666666665 1.707825127659933
10 2.8 0.5328130535187741 0.5400617248673216
100 3.66 0.17209417305649835 0.1707825127659933
1000 3.553 0.0541584607979215 0.05400617248673217
10000 3.4994 0.016719667909381453 0.01707825127659933
0.68498 0.95433
0.0585 0.15384615384615385
9.332636185032189e-302 0.0 -1386.2943611198907
```

In each sample-size row, the second number is one run's average, the third is how spread out the 1,000 averages were, and the fourth is $\sigma / \sqrt{N}$; the last two agree to within a few percent at every size, and each tenfold increase in rolls shrinks them by about $\sqrt{10} \approx 3.2$.
The first line's `0.9999999999999999` is six copies of $1/6$ added up in floating point, and the last line shows the product of 2,000 halves underflowing to 0 while the sum of their logs stays at $-1386.29$.

## Where this shows up later

- Unit 5: logistic regression outputs the probability of a yes/no class, and its loss is a negative log-probability.
- Unit 7: a test score is an average over test examples, so a small test set gives a noisy score, with the same $\sigma / \sqrt{N}$ behavior.
- Unit 11: softmax turns a model's scores into a categorical distribution over tokens, and the loss is the average $-\log p$ of the correct token.
- Unit 14: generating text means sampling from that distribution, one token at a time.
- Unit 15: reinforcement learning maximizes an expected reward, estimated by averaging over sampled actions.
- Unit 16: the Bradley–Terry model gives the probability that people prefer one answer over another.

## Self-test

<details>
<summary>A random variable is 0 with probability 0.5, 1 with probability 0.3 and 2 with probability 0.2. What are its mean, variance and standard deviation?</summary>

Mean: $0 \times 0.5 + 1 \times 0.3 + 2 \times 0.2 = 0.7$.
Variance, each value's squared distance from 0.7 weighted by its probability:

$$
\begin{aligned}
&(0 - 0.7)^2 \times 0.5 + (1 - 0.7)^2 \times 0.3 \\
&\quad + (2 - 0.7)^2 \times 0.2 \\
&= 0.245 + 0.027 + 0.338 = 0.61
\end{aligned}
$$

Standard deviation: $\sqrt{0.61} \approx 0.78$.

</details>

<details>
<summary>In the dice widget, switch to the loaded die. Where does the running mean settle, and why is the dashed funnel a little wider than for the fair die?</summary>

It settles near 4.5: $0.1 \times (1 + 2 + 3 + 4 + 5) + 0.5 \times 6 = 1.5 + 3 = 4.5$.
The funnel is $\sigma / \sqrt{N}$, and the loaded die's $\sigma$ is about 1.80 against the fair die's 1.71, so at every $N$ it's about 5% wider.

</details>

<details>
<summary>Why does 100 times as many samples only make a Monte Carlo estimate about 10 times more accurate?</summary>

The typical miss is $\sigma / \sqrt{N}$, so it shrinks with the square root of the number of samples.
Multiplying $N$ by 100 divides the miss by $\sqrt{100} = 10$.

</details>

<details>
<summary>In the Bayes widget, keep the catch rate at 90% and false alarms at 5%, and raise the base rate to 50%. What is P(phish | flag) now, and why is it so different from 15%?</summary>

About 0.95, from Bayes' rule with the new base rate:

$$
\frac{0.9 \times 0.5}{0.9 \times 0.5 + 0.05 \times 0.5} = \frac{0.45}{0.475} \approx 0.947
$$

With half the email phishing, the 5% false alarms only apply to as many emails as the catch rate does, so they no longer swamp the real catches.

</details>

<details>
<summary>Why do models and libraries report log-probabilities for whole texts instead of probabilities?</summary>

A text's probability is a product of one probability per token, all below 1, so for long texts it gets too small for a float and comes out as 0.
The log turns the product into a sum of ordinary-sized negative numbers, and since log only increases, comparing log-probabilities ranks texts the same way as comparing probabilities.

</details>

::: done-when
- You can compute the expectation, variance and standard deviation of a small discrete distribution by hand.
- You can estimate an expectation by averaging samples in NumPy, and say how many more samples it takes to cut the typical miss by 10.
- You can fill in a joint probability table from a base rate, a catch rate and a false-alarm rate, and read $P(\text{phish} \mid \text{flag})$ off it with Bayes' rule.
- You can explain why the probability of a long text is computed as a sum of log-probabilities.
:::

## Go deeper

- [Seeing Theory](https://seeing-theory.brown.edu/), chapters 1 and 2 (Basic Probability, Compound Probability): interactive versions of expectation, variance and conditional probability, including a die whose running mean converges like the one above.
- 3Blue1Brown's [Bayes' theorem](https://www.3blue1brown.com/lessons/bayes-theorem): Bayes' rule as areas of a square, and why thinking in counts ("4 out of 24") makes it click.
- [Mathematics for Machine Learning](https://mml-book.github.io/), chapter 6, sections 6.2 to 6.5: distributions, the sum and product rules, Bayes' theorem, expectations and variances, and the Gaussian, done more formally.
