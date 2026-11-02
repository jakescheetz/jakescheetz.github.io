---
title: Predicting the next token
subtitle: How a model turns scores into probabilities, and how training grades them.
unit: 6
minutes: 50
scripts: [widgets.js]
---

A language model is trained on a single task: given the tokens so far, put a probability on every token that could come next.
The probabilities come from a softmax over scores called logits, training grades them with cross-entropy loss, and the gradient of that loss is where every training step starts.
Sampling in unit 9 and the RL units after it all work with these same probabilities.

## One score per token

At the end of the network, each position in the sequence has a vector $h$ of width $d$ (unit 8 covers how it gets built).
To make a prediction, the model multiplies $h$ by an unembedding matrix $W_U$ that has one row per vocabulary token, so $V$ rows and $d$ columns:

$$
z = W_U\, h
$$

The result $z$ has $V$ entries, one per token, and they're called logits.
From unit 1, $W_U h$ is "score $h$ against every row of $W_U$", so each logit is the dot product of $h$ with that token's row.
GPT-2 reuses its embedding table $E$ from unit 5 as $W_U$, so a token's logit there is its own embedding dotted with $h$.
With GPT-2's vocabulary that's 50,257 logits at every position.

A logit can be any real number.
Higher means "more likely next", but logits aren't probabilities yet: they can be negative and they don't add up to 1.

For a running example, the text so far is "the cat sat on the", the vocabulary is shrunk to five tokens (mat, floor, sofa, dog and ran), and the model outputs

$$
z = (2.0,\ 1.0,\ 0.5,\ -1.0,\ -2.0)
$$

in that order, so it already leans toward "mat".

## Softmax turns scores into probabilities

Softmax raises $e$ to each logit, then divides by the total:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

We write the whole vector as $p = \mathrm{softmax}(z)$.
The exponential makes every entry positive and keeps the order (a bigger logit always gets a bigger probability), and dividing by the sum makes the entries add up to 1.

For the example, $e^{2.0} = 7.39$ and the five exponentials add up to 12.26, so $p_{\text{mat}} = 7.39 / 12.26 = 0.60$.
All five come out to $p = (0.60,\ 0.22,\ 0.13,\ 0.03,\ 0.01)$.

Only the differences between logits matter.
Adding the same number $c$ to every logit multiplies every $e^{z_i}$ by $e^c$, and that factor cancels between the top and the bottom.
Code uses this to subtract the largest logit before exponentiating, which keeps $e^{z}$ from overflowing and leaves $p$ unchanged.
It also gives logits a concrete meaning: two tokens whose logits differ by 1 have probabilities in the ratio $e \approx 2.72$.
Mat and floor differ by 1 in the example, and $0.603 / 0.222 \approx 2.72$.

## The loss: −log of the right answer's probability

During training we know which token actually came next.
Call its index $c$ (here, mat).
The loss for this one prediction is

$$
L = -\log p_c
$$

the negative log of the probability the model gave the right answer.
For the example that's $-\log 0.60 = 0.51$.

This is called cross-entropy.
The general formula compares the model's distribution $p$ with a target distribution $y$ as $-\sum_i y_i \log p_i$.
Here $y$ is one-hot (1 for the correct token, 0 for the rest), so every term but one drops out and you're left with $-\log p_c$.

The log is what makes confident mistakes expensive.
The loss for a few values of $p_c$:

| $p_c$ | 0.9 | 0.5 | 0.1 | 0.01 | 0.001 |
|---|---|---|---|---|---|
| $-\log p_c$ | 0.11 | 0.69 | 2.30 | 4.61 | 6.91 |

Being sure and right costs almost nothing.
Every time the probability on the right answer drops by a factor of 10, the loss goes up by another 2.30, with no ceiling.
A model that puts 0.001 on the token that really came next, because it was confident about something else, pays far more than one that spread its bets.

There's a second reason for the log.
The probability of a whole piece of text is the product of the probabilities of each of its tokens, and the log turns that product into a sum.
So adding up $-\log p_c$ over every token is the same as maximizing the probability the model assigns to the training text.
Working in logs also avoids multiplying thousands of numbers below 1, which would underflow to zero in floating point.

<figure class="widget" id="softmax-loss">
<figcaption>Drag the logits and watch the probabilities (mat in blue) and the loss. The curve on the right is −log p, with the dot at p(mat). Step does one gradient step on the logits, z ← z − (p − y); Train keeps stepping. Try pushing floor up and mat down to see how fast the loss climbs.</figcaption>
</figure>

## The gradient is p − y

To train, we need the gradient of $L$ with respect to the logits (units 2 and 3).
Writing out the softmax inside the log gives

$$
L = -\log \frac{e^{z_c}}{\sum_j e^{z_j}} = -z_c + \log \sum_j e^{z_j}
$$

The derivative of the first term with respect to $z_i$ is $-1$ if $i = c$ and 0 otherwise.
The derivative of the second term is $e^{z_i} / \sum_j e^{z_j}$, which is $p_i$.
Put together, with $y$ the one-hot vector for the correct token:

$$
\nabla_z L = p - y
$$

For the example that's $(-0.40,\ 0.22,\ 0.13,\ 0.03,\ 0.01)$.
A gradient step with learning rate $\eta$ moves each logit against its gradient.
The correct token's logit goes up by $\eta\,(1 - p_c)$, and every other logit goes down by $\eta$ times its own probability.
So the wrong token that grabbed the most probability gets pushed down the hardest, and a token the model already ignores barely moves.
The entries always add up to zero, because $p$ and $y$ both add up to 1.
The gradient is zero only at $p = y$, which softmax can approach but never reach with finite logits.

The −log also keeps the gradient large when the model is wrong.
The pull on the correct logit is $1 - p_c$, which approaches 1 when the model is badly wrong.
If the loss were $1 - p_c$ instead, the pull would be $p_c(1 - p_c)$, which shrinks toward zero exactly when $p_c$ is tiny, so the model would learn slowest from its worst mistakes.

In a real model the logits aren't free parameters.
They come from $\theta$ through $z = W_U h$, and backpropagation (unit 4) starts from $p - y$ and carries it back through every weight.
The Step button in the widget above skips that and moves the logits directly, which is enough to see which way they want to go.

## Every position is a training example

Training text gives you a prediction at every position, and the labels are free: the target at each position is just the next token of the text.

<figure>
<pre>position   1     2     3     4     5
input      the   cat   sat   on    the
           │     │     │     │     │
target     <span class="blue">cat</span>   <span class="blue">sat</span>   <span class="blue">on</span>    <span class="blue">the</span>   <span class="blue">mat</span>
p(target)  0.02  0.30  0.45  0.85  0.60
loss       <span class="orange">3.91</span>  <span class="orange">1.20</span>  <span class="orange">0.80</span>  <span class="orange">0.16</span>  <span class="orange">0.51</span></pre>
<figcaption>Position t sees the input tokens 1 to t and is graded on token t + 1. The probabilities are made up; the losses are −log of them.</figcaption>
</figure>

A sequence of tokens $x_1, \dots, x_n$ gives $n - 1$ predictions, and the training loss is their average:

$$
L = -\frac{1}{n-1} \sum_{t=1}^{n-1} \log p_\theta(x_{t+1} \mid x_1, \dots, x_t)
$$

where $p_\theta(x_{t+1} \mid x_1, \dots, x_t)$ is the probability the model with parameters $\theta$ gives the actual next token.
The figure's text, "the cat sat on the mat", is six tokens, so it gives five predictions and $L = (3.91 + 1.20 + 0.80 + 0.16 + 0.51)/5 = 1.32$.

All $n - 1$ predictions come out of one forward pass.
The model computes a vector $h$ at every position at once, and each position is only allowed to look at tokens up to its own (unit 7 shows how attention enforces that).
So a single training sequence of 1,024 tokens, GPT-2's context length, gives 1,023 graded predictions for the price of one pass.

You'll often see the loss reported as perplexity, which is $e^{L}$.
For the figure, $e^{1.3176} \approx 3.73$: on average the model was as unsure as if it were choosing evenly between 3.73 tokens.
A model whose logits are all equal gives every token probability $1/V$, so its loss is $\log V$ and its perplexity is $V$.
For GPT-2's vocabulary that's $\log 50257 = 10.82$, roughly where a freshly initialized model starts, since its logits begin close to zero.

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

The finite difference for floor, 0.22173, matches $p_{\text{floor}} = 0.222$ from $p - y$, and the gradient's sum of $-6 \times 10^{-17}$ is zero up to floating-point rounding.
Each step raises mat and lowers the others, floor most of all, and the loss falls from 0.506 to 0.1755 in three steps.

## Where this shows up later

- Unit 7: the rule that each position only looks at tokens up to its own is what lets all $n - 1$ predictions train in one pass.
- Unit 8: the transformer block builds $h$, and the final layer norm and $W_U$ turn it into these logits.
- Unit 9: sampling picks the next token from $p$, and temperature divides $z$ before the softmax.
- Unit 10: a language model is a policy over tokens, $\pi_\theta$, and the policy gradient needs $\nabla \log \pi_\theta$, which for the logits is $y - p$ (this unit's gradient with the sign flipped) with $y$ the one-hot vector for the sampled token.
- Unit 11: the reward model's loss, $-\log \sigma(r_w - r_l)$, is cross-entropy over a choice between two answers.

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
