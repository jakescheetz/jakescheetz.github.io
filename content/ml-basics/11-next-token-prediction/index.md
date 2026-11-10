---
title: Predicting the next token
subtitle: How a model turns scores into probabilities, and how training grades them.
unit: 11
section: Language models
minutes: 50
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'What is a logit?'
    answer: 'A score for one vocabulary token that can be any real number'
    wrong: ['The probability the model gives one token', 'The loss for one predicted token', 'A score between 0 and 1 for one token']
    why: 'Logits come straight out of $z = W_U h$. They can be negative and don''t add up to 1 until softmax turns them into probabilities.'
  - kind: term
    q: 'What does softmax do to a list of logits?'
    answer: 'Makes every entry positive and scales them to add up to 1, keeping their order'
    wrong: ['Sets the largest entry to 1 and every other entry to 0', 'Divides each entry by the sum of all the entries', 'Squashes each entry into the range 0 to 1 on its own']
    why: 'It raises $e$ to each logit, which makes everything positive, then divides by the total. Dividing raw logits by their sum fails as soon as one is negative.'
  - kind: term
    q: 'With a one-hot target, what is the cross-entropy loss for one prediction?'
    answer: '$-\log p_c$, where $c$ is the correct token'
    wrong: ['$1 - p_c$, where $c$ is the correct token', '$-\log$ of the largest probability in $p$', '$-\sum_i \log p_i$ over every token']
    why: 'The general form is $-\sum_i y_i \log p_i$, and with $y$ one-hot every term but the correct token''s drops out.'
  - kind: term
    q: 'What is perplexity?'
    answer: '$e^{L}$, where $L$ is the average loss'
    wrong: ['$\log L$, where $L$ is the average loss', '$e^{-L}$, where $L$ is the average loss', '$1 / L$, where $L$ is the average loss']
    why: 'It turns the average loss back into a count: a perplexity of 4 means as unsure as choosing evenly between 4 tokens.'
  - kind: term
    q: 'What does the unembedding matrix $W_U$ do?'
    answer: 'Scores $h$ against one row per vocabulary token, giving the logits'
    wrong: ['Turns each token ID into a vector of width $d$', 'Turns the logits into probabilities that add to 1', 'Mixes the vectors of earlier positions into $h$']
    why: '$W_U$ has $V$ rows and $d$ columns, so $z = W_U h$ is one dot product per token. GPT-2 reuses its embedding table $E$ here.'
  - kind: apply
    q: 'Two tokens have logits 3 and 1. How many times more likely is the first?'
    answer: '$e^{2} \approx 7.39$ times'
    wrong: ['$3$ times', '$2$ times', '$e^{4} \approx 54.6$ times']
    why: 'Only the difference matters, and a difference of 2 gives a probability ratio of $e^{3} / e^{1} = e^{2}$.'
  - kind: apply
    q: 'The model gives the correct token probability 0.2. What is the loss?'
    answer: '$1.61$'
    wrong: ['$0.80$', '$0.70$', '$-1.61$']
    why: '$-\ln 0.2 = 1.61$. The loss uses the natural log, and the minus sign makes it positive.'
  - kind: apply
    q: 'Softmax gives $p = (0.5, 0.3, 0.2)$ and the second token is correct. What is $\nabla_z L$?'
    answer: '$(0.5, -0.7, 0.2)$'
    wrong: ['$(-0.5, 0.7, -0.2)$', '$(0.5, -0.3, 0.2)$', '$(0.5, 0.3, 0.2)$']
    why: 'The gradient is $p - y$ with $y = (0, 1, 0)$, so only the correct entry changes: $0.3 - 1 = -0.7$. The entries add up to 0.'
  - kind: apply
    q: 'After one gradient step on the logits, which wrong token''s logit goes down the most?'
    answer: 'The wrong token with the highest probability'
    wrong: ['The wrong token with the lowest probability', 'Every wrong token, all by the same amount', 'The wrong token with the lowest logit']
    why: 'Each wrong logit moves down by $\eta$ times its own probability, so the token that grabbed the most probability gets pushed hardest.'
  - kind: apply
    q: 'A training sequence has 513 tokens. How many graded predictions does it give?'
    answer: '$512$'
    wrong: ['$513$', '$511$', '$1$']
    why: 'Every token except the last has a next token to predict, and all of them come out of one forward pass.'
  - kind: apply
    q: 'The losses at three positions are 0.5, 1.0 and 1.5. What is the perplexity?'
    answer: '$e^{1} \approx 2.72$'
    wrong: ['$1.0$', '$e^{3} \approx 20.1$', '$3.0$']
    why: 'The average loss is $(0.5 + 1.0 + 1.5) / 3 = 1$, and perplexity is $e$ to the average, not to the sum.'
  - kind: apply
    q: 'A fresh model with a 1,000-token vocabulary has all its logits near 0. Roughly what loss does it start at?'
    answer: '$\log 1000 \approx 6.91$'
    wrong: ['$1{,}000$', '$0$', '$1/1000 = 0.001$']
    why: 'Equal logits give every token $p = 1/1000$, so the loss is $-\log(1/1000) = \log 1000$. Its perplexity is 1,000, the vocabulary size.'
---

::: why
Your phone keyboard guesses your next word as you type.
A language model does the same job, and it is the only job it is trained on.
Given the tokens so far, it puts a probability on every token that could come next.
Training then grades those probabilities against the token that really came next.
Sampling in unit 14 and the RL units after it all work with these same probabilities.
:::

## One score per token

At the end of the network, each position in the sequence has a vector $h$ with $d$ entries (unit 13 covers how it gets built).
To make a prediction, the model multiplies $h$ by an *unembedding matrix* $W_U$.
This matrix has one row per vocabulary token, so $V$ rows and $d$ columns:

$$
z = W_U\, h
$$

::: formula
From [unit 1](../01-vectors/#a-matrix-is-a-stack-of-dot-products), $W_U h$ is "score $h$ against every row of $W_U$", so entry $i$ of $z$ is the dot product of $h$ with token $i$'s row.
:::

The result $z$ has $V$ entries, one score per token.
These scores are called *logits*.
GPT-2 reuses its embedding table $E$ from unit 10 as $W_U$, so a token's logit there is its own embedding dotted with $h$.
With GPT-2's vocabulary that's 50,257 logits at every position.

A logit can be any real number.
Higher means "more likely next".
But logits aren't probabilities yet: they can be negative, and they don't add up to 1.

Here is the running example for this unit.
The text so far is "the cat sat on the".
The vocabulary is shrunk to five tokens: mat, floor, sofa, dog and ran.
The model outputs

$$
z = (2.0,\ 1.0,\ 0.5,\ -1.0,\ -2.0)
$$

in that order, so it already leans toward "mat".

## Softmax turns scores into probabilities

*Softmax* is the function that turns the $V$ logits into probabilities.
The result is a categorical distribution ([unit 3](../03-probability/#random-variables-and-distributions)): one probability per token, each between 0 and 1, adding up to 1.
The recipe has two moves.
Raise $e$ to each logit, then divide by the total:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

::: formula
The top is $e \approx 2.718$ raised to token $i$'s logit $z_i$.
The bottom adds up the same exponential for every token $j$ in the vocabulary, so it's one number shared by all $V$ probabilities.
We write the whole vector as $p = \mathrm{softmax}(z)$.
:::

The exponential makes every entry positive.
It also keeps the order, so a bigger logit always gets a bigger probability.
Dividing by the sum makes the entries add up to 1.

For the example, $e^{2.0} = 7.39$ and the five exponentials add up to 12.26, so $p_{\text{mat}} = 7.39 / 12.26 = 0.60$.
All five come out to $p = (0.60,\ 0.22,\ 0.13,\ 0.03,\ 0.01)$.

<figure class="widget demo" id="demo-softmax"></figure>

Only the differences between logits matter.
Adding the same number $c$ to every logit multiplies every $e^{z_i}$ by $e^c$.
That factor sits on the top and the bottom of the fraction, so it cancels.
Code uses this to subtract the largest logit before exponentiating, which keeps $e^{z}$ from overflowing and leaves $p$ unchanged.

It also gives logits a concrete meaning.
Two tokens whose logits differ by 1 have probabilities in the ratio $e \approx 2.72$.
Mat and floor differ by 1 in the example, and $0.603 / 0.222 \approx 2.72$.

Softmax is the many-class version of the sigmoid from [unit 5](../05-classification/#from-a-score-to-a-probability).
With two tokens, dividing top and bottom by $e^{z_1}$ gives $p_1 = 1 / (1 + e^{-(z_1 - z_2)}) = \sigma(z_1 - z_2)$, the sigmoid of the gap between the two logits.

## The loss: −log of the right answer's probability

During training we know which token actually came next.
Call its index $c$ (here, mat).
The loss for this one prediction is

$$
L = -\log p_c
$$

the negative log of the probability the model gave the right answer, with $\log$ the natural log.
For the example that's $-\log 0.60 = 0.51$.
It's the same rule as log loss in [unit 5](../05-classification/#log-loss), with $V$ possible answers instead of two.

This loss is called *cross-entropy*.
The general formula compares the model's distribution $p$ with a target distribution $y$ as $-\sum_i y_i \log p_i$.
In words: for every token $i$, multiply the target's probability $y_i$ by the log of the model's probability $p_i$, add them all up and flip the sign.
Here $y$ is *one-hot*: 1 for the correct token and 0 for the rest.
So every term but one drops out, and you're left with $-\log p_c$.
Unit 5's binary cross-entropy is the same sum with two classes, where $y$ and $1 - y$ play the part of the one-hot target.

The log is what makes confident mistakes expensive.
Here is the loss for a few values of $p_c$:

| $p_c$ | 0.9 | 0.5 | 0.1 | 0.01 | 0.001 |
|---|---|---|---|---|---|
| $-\log p_c$ | 0.11 | 0.69 | 2.30 | 4.61 | 6.91 |

Being sure and right costs almost nothing.
Every time the probability on the right answer drops by a factor of 10, the loss goes up by another 2.30, with no ceiling.
Think of a weather forecaster who says "0.1% chance of rain" on a day it pours.
A model that puts 0.001 on the token that really came next pays far more than one that spread its bets.

There's a second reason for the log, covered in [unit 3](../03-probability/#why-models-work-in-log-probabilities).
The probability of a whole text is the product of its next-token probabilities, and the log turns that product into a sum.
So adding up $-\log p_c$ over every token is the same as maximizing the probability the model assigns to the training text.
It also avoids a product that underflows to zero.

<figure class="widget demo" id="demo-loss"></figure>

Now move the logits yourself and watch the probabilities and the loss change together.

<figure class="widget" id="softmax-loss">
<figcaption>Drag the logits and watch the probabilities (mat in blue) and the loss. The curve on the right is −log p, with the dot at p(mat). Step does one gradient step on the logits, z ← z − (p − y); Train keeps stepping. Try pushing floor up and mat down to see how fast the loss climbs.</figcaption>
</figure>

## The gradient is p − y

To train, we need the gradient of $L$ with respect to the logits ([unit 2](../02-gradients/#the-gradient-points-uphill)).
It tells us how the loss changes when each logit goes up a little.
Gradient descent ([unit 4](../04-gradient-descent/#walk-downhill)) then steps the other way.
Writing out the softmax inside the log splits the loss into two terms:

$$
L = -\log \frac{e^{z_c}}{\sum_j e^{z_j}} = -z_c + \log \sum_j e^{z_j}
$$

::: formula
The split uses $\log(a/b) = \log a - \log b$ and $\log e^{z_c} = z_c$.
The first term, $-z_c$, is minus the correct token's logit; its derivative with respect to $z_i$ is $-1$ if $i = c$ and 0 otherwise.
The second, $\log \sum_j e^{z_j}$, is the log of the softmax's denominator, and it depends on every logit.
Its derivative is $e^{z_i} / \sum_j e^{z_j}$, which is $p_i$: the log differentiates to one over the sum (the $1/p$ rule from [unit 5](../05-classification/#training-with-gradient-descent)), and by the chain rule ([unit 2](../02-gradients/#the-rules-you-need)) that gets multiplied by the sum's own derivative, $e^{z_i}$.
:::

Put together, with $y$ the one-hot vector for the correct token:

$$
\nabla_z L = p - y
$$

Entry $i$, $\partial L / \partial z_i$, is $p_i - 1$ for the correct token and $p_i$ for every other one.
It's the same "predicted minus actual" that log loss gives for the sigmoid in [unit 5](../05-classification/#training-with-gradient-descent), one entry per token.
For the example that's $(-0.40,\ 0.22,\ 0.13,\ 0.03,\ 0.01)$.

A gradient step with learning rate $\eta$ subtracts $\eta$ times the gradient from the logits.
The correct token's logit goes up by $\eta\,(1 - p_c)$.
Every other logit goes down by $\eta$ times its own probability.
So the wrong token that grabbed the most probability gets pushed down the hardest, and a token the model already ignores barely moves.

<figure class="widget demo" id="demo-gradient"></figure>

The entries of the gradient always add up to zero, because $p$ and $y$ both add up to 1.
The gradient is zero only at $p = y$, which softmax can approach but never reach with finite logits.

The −log also keeps the gradient large when the model is wrong, for the same reason squared error stalls in [unit 5](../05-classification/#log-loss).
The pull on the correct logit is $1 - p_c$, which approaches 1 when the model is badly wrong.
Suppose the loss were $1 - p_c$ instead.
Then the pull would be $p_c(1 - p_c)$, which shrinks toward zero exactly when $p_c$ is tiny.
The model would learn slowest from its worst mistakes.

In a real model the logits aren't free parameters.
They come from the parameters $\theta$ through $z = W_U h$.
Backpropagation ([unit 8](../08-neural-networks/#backpropagation)) starts from $p - y$ and carries it back through every weight.
The Step button in the widget above skips that and moves the logits directly, which is enough to see which way they want to go.

## Every position is a training example

Training text gives you a prediction at every position, and the labels are free.
The target at each position is just the next token of the text.

<figure>
<pre>position   1     2     3     4     5
input      the   cat   sat   on    the
           │     │     │     │     │
target     <span class="blue">cat</span>   <span class="blue">sat</span>   <span class="blue">on</span>    <span class="blue">the</span>   <span class="blue">mat</span>
p(target)  0.02  0.30  0.45  0.85  0.60
loss       <span class="orange">3.91</span>  <span class="orange">1.20</span>  <span class="orange">0.80</span>  <span class="orange">0.16</span>  <span class="orange">0.51</span></pre>
<figcaption>Position t sees the input tokens 1 to t and is graded on token t + 1. The probabilities are made up; the losses are −log of them.</figcaption>
</figure>

A sequence of tokens $x_1, \dots, x_n$ gives $n - 1$ predictions.
The training loss is the average of their $-\log p$:

$$
L = -\frac{1}{n-1} \sum_{t=1}^{n-1} \log p_\theta(x_{t+1} \mid x_1, \dots, x_t)
$$

::: formula
The sum runs over positions $t$ from 1 to $n - 1$, and the $\frac{1}{n-1}$ in front turns it into an average.
Each term is the log of $p_\theta(x_{t+1} \mid x_1, \dots, x_t)$, the probability the model with parameters $\theta$ gives the actual next token $x_{t+1}$ after seeing $x_1$ to $x_t$.
The bar reads "given", as in [conditional probability](../03-probability/#conditional-probability-and-bayes-rule).
The sum is the log-probability of the text from [unit 3](../03-probability/#why-models-work-in-log-probabilities), leaving out the first token, so $L$ is minus that log-probability divided by $n - 1$.
:::

The figure's text, "the cat sat on the mat", is six tokens.
So it gives five predictions, and $L = (3.91 + 1.20 + 0.80 + 0.16 + 0.51)/5 = 1.32$.

All $n - 1$ predictions come out of one forward pass.
The model computes a vector $h$ at every position at once.
Each position is only allowed to look at tokens up to its own (unit 12 shows how attention enforces that).
So a single training sequence of 1,024 tokens, GPT-2's context length, gives 1,023 graded predictions for the price of one pass.

You'll often see the loss reported as *perplexity*, which is $e^{L}$: $e$ raised to the average loss.
For the figure, $e^{1.3176} \approx 3.73$.
On average the model was as unsure as if it were choosing evenly between 3.73 tokens.

A model whose logits are all equal gives every token probability $1/V$.
Its loss is $\log V$ and its perplexity is $V$.
For GPT-2's vocabulary that's $\log 50257 = 10.82$.
A freshly initialized model starts roughly there, since its logits begin close to zero.

## Worked example

```python
import numpy as np

vocab = ["mat", "floor", "sofa", "dog", "ran"]
# Logits after "the cat sat on the"; the answer is "mat".
z = np.array([2.0, 1.0, 0.5, -1.0, -2.0])
correct = 0

def softmax(z):
    # Subtracting the max avoids overflow, same p.
    e = np.exp(z - z.max())
    return e / e.sum()

p = softmax(z)
print(p.round(3))
loss = -np.log(p[correct])
print(loss)

y = np.zeros(len(vocab))
y[correct] = 1.0
grad = p - y
print(grad.round(3), grad.sum())

# Finite-difference check of the "floor" entry.
eps = 1e-6
z2 = z.copy()
z2[1] += eps
print((-np.log(softmax(z2)[correct]) - loss) / eps)

# Three gradient steps on the logits, learning rate 1.
for step in range(3):
    z = z - 1.0 * (softmax(z) - y)
    step_loss = -np.log(softmax(z)[correct])
    print(step + 1, z.round(3), step_loss.round(4))

# Probability given to each correct next token in
# "the cat sat on the mat", one per position.
p_correct = np.array([0.02, 0.30, 0.45, 0.85, 0.60])
losses = -np.log(p_correct)
print(losses.round(2))
L = losses.mean()
print(L, np.exp(L))
print(np.log(50257))
```

```
[0.603 0.222 0.134 0.03  0.011]
0.5062827054670117
[-0.397  0.222  0.134  0.03   0.011] -6.245004513516506e-17
0.2217327798037516
1 [ 2.397  0.778  0.366 -1.03  -2.011] 0.3176
2 [ 2.669  0.634  0.27  -1.054 -2.02 ] 0.227
3 [ 2.872  0.53   0.198 -1.073 -2.027] 0.1755
[3.91 1.2  0.8  0.16 0.51]
1.3175696118471238 3.7343344571387243
10.82490511970208
```

The finite difference for floor, 0.22173, matches $p_{\text{floor}} = 0.222$ from $p - y$.
The gradient's sum of $-6 \times 10^{-17}$ is zero up to floating-point rounding.
Each step raises mat and lowers the others, floor most of all, and the loss falls from 0.506 to 0.1755 in three steps.

::: real-world
- **Chatbots and autocomplete.** A chat model writes its reply one token at a time. Each token comes from the probabilities in this unit, and then the model runs again with that token added to the text.
- **Training logs.** The "loss" printed while a language model trains is this average cross-entropy. Papers often report the same number as perplexity.
- **Spotting odd text.** Some defenses score an incoming prompt's perplexity under a language model. Machine-generated attack strings often look like gibberish, so they score far higher than normal prose.
:::

## Where this shows up later

- Unit 12: the rule that each position only looks at tokens up to its own is what lets all $n - 1$ predictions train in one pass.
- Unit 13: the transformer block builds $h$, and the final layer norm and $W_U$ turn it into these logits.
- Unit 14: sampling picks the next token from $p$, and temperature divides $z$ before the softmax.
- Unit 15: a language model is a policy over tokens, $\pi_\theta$, and the policy gradient needs $\nabla \log \pi_\theta$, which for the logits is $y - p$ (this unit's gradient with the sign flipped) with $y$ the one-hot vector for the sampled token.
- Unit 16: the reward model's loss, $-\log \sigma(r_w - r_l)$, is cross-entropy over a choice between two answers.

::: review
Logit, $z_i$
:   The model's raw score for token $i$. Any real number, higher means more likely.

Unembedding matrix, $W_U$
:   A $V \times d$ matrix. Multiplying it by $h$ gives one logit per vocabulary token.

Softmax
:   Raise $e$ to each logit and divide by the total. The results are positive and add up to 1.

Cross-entropy loss
:   $-\log p_c$, the negative log of the probability the model gave the correct token $c$.

One-hot target, $y$
:   A vector with a 1 for the correct token and 0 for every other token.

Gradient of the loss, $p - y$
:   How the loss changes with each logit. A step raises the correct logit and lowers each wrong one by its own probability.

Perplexity
:   $e$ raised to the average loss. It reads as "choosing evenly between this many tokens."
:::

::: say
A language model ends with one score per token, called a logit.
Softmax turns the logits into probabilities that add up to 1.
The loss is the negative log of the probability the model gave the token that really came next, so a confident mistake costs a lot.
The gradient of that loss with respect to the logits is p minus y, and every training step starts from it.
One pass over a text grades a prediction at every position.
:::

## Self-test

<details>
<summary>Four tokens all have logit 1. What is p, and what is the loss if the second token was correct?</summary>

Every $e^{z_i}$ is the same, so $p = (0.25, 0.25, 0.25, 0.25)$.
The loss is $-\log 0.25 = \log 4 \approx 1.39$, whichever token was correct.

</details>

<details>
<summary>You add 10 to every logit. What happens to the probabilities?</summary>

Nothing.
Every $e^{z_i}$ gets multiplied by $e^{10}$, which cancels when you divide by the sum.

</details>

<details>
<summary>Why use −log p of the correct token as the loss instead of 1 − p?</summary>

Two reasons from this unit.
With −log, a confident wrong answer costs a lot (there's no ceiling), while $1 - p$ never costs more than 1.
And with −log the pull on the correct logit is $1 - p_c$, which stays near 1 when the model is badly wrong, while with $1 - p$ it's $p_c(1 - p_c)$, which vanishes right when the model most needs to change.

</details>

<details>
<summary>In the widget, press Reset, then push z(floor) to 4 and z(mat) to −4. What happens to the loss, and which gradient bars get longest?</summary>

Floor takes 0.96 of the probability and $p_{\text{mat}}$ drops to 0.0003, so the loss is about 8.04.
The curve's axis stops at 5, so the dot parks at its top while the readout shows the real number.
The orange bar for mat sits at −1.00 and the one for floor at +0.96, so a step raises mat and lowers floor by nearly a full unit each, while the other three barely move.

</details>

<details>
<summary>A training sequence has 2,049 tokens. How many predictions does it give, and why don't they need 2,048 separate forward passes?</summary>

2,048, one for each token that has a next token.
They come out of one pass because the model computes every position at once, and each position can only see tokens up to its own, so no prediction gets to peek at the answer.

</details>

<details>
<summary>A model's average loss is 2.0. What is its perplexity?</summary>

$e^{2.0} \approx 7.39$.
On average it's as unsure as choosing evenly between about seven tokens.

</details>

::: done-when
- You can compute softmax and the cross-entropy loss by hand for three or four logits.
- You can explain with −log why a confident wrong prediction costs far more than an unsure one.
- You can write down the gradient of the loss with respect to the logits and say which way each logit moves on a step.
- You can turn an average loss into a perplexity and say what the number means.
:::

## Go deeper

- 3Blue1Brown's [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt): the last part, on the unembedding matrix and softmax, is this unit drawn out visually.
- Andrej Karpathy's [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo): builds a character-level next-token model from counts, then trains it with this exact loss.
- Dive into Deep Learning, [Softmax Regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html): the same softmax, cross-entropy and $p - y$ derivation, written for classification.
