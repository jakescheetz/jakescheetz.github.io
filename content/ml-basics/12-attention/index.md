---
title: Attention
subtitle: How each token pulls in information from the tokens before it.
unit: 12
section: Language models
minutes: 75
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'In attention, what does the query $q_i$ describe?'
    answer: 'What position $i$ is looking for in earlier tokens'
    wrong: ['What position $i$ offers to the tokens after it', 'What position $i$ hands over when it gets picked', 'Where position $i$ sits in the sequence']
    why: 'The query is compared with every visible key by a dot product. The key is what a position offers, and the value is what it hands over.'
  - kind: term
    q: 'What is the value vector $v_j$ for?'
    answer: 'It''s what position $j$ hands over, mixed in by its weight'
    wrong: ['It''s what position $j$ is looking for in earlier tokens', 'It''s what position $j$ offers to be compared against', 'It''s the softmax weight that position $j$ receives']
    why: 'The output is $o_i = a_{i1} v_1 + \cdots + a_{ii} v_i$, a weighted sum of values. Queries and keys only decide the weights.'
  - kind: term
    q: 'What does the causal mask do?'
    answer: 'Sets the scores for later positions to $-\infty$, so they get zero weight'
    wrong: ['Sets the scores for earlier positions to $-\infty$, so they get zero weight', 'Removes the diagonal, so no token can attend to itself', 'Hides a random subset of tokens during training']
    why: 'Position $i$ may only see positions 1 to $i$, itself included. Since $e^{-\infty} = 0$, the masked scores get no weight after the softmax.'
  - kind: term
    q: 'What is $d_k$?'
    answer: 'The number of entries in each query and key'
    wrong: ['The width of each token vector $h_i$', 'The number of heads in the layer', 'The number of tokens a position can see']
    why: '$W_Q$ and $W_K$ are $d_k \times d$, so they turn a width-$d$ vector into $d_k$ numbers. In GPT-2 small $d = 768$ and $d_k = 64$.'
  - kind: term
    q: 'What does an inference server keep in a KV cache?'
    answer: 'The keys and values of earlier positions, reused for each new token'
    wrong: ['The attention weights of earlier positions, reused for each new token', 'The queries of earlier positions, reused for each new token', 'The token IDs of earlier positions, re-embedded for each new token']
    why: 'Thanks to the mask, earlier keys and values never depend on later tokens, so they can be computed once. Each new token only needs its own query, key and value.'
  - kind: apply
    q: 'A query is $(2, 1, 0, 1)$ and a key is $(1, 3, 0, -1)$, with $d_k = 4$. What is the score?'
    answer: '$2$'
    wrong: ['$4$', '$1$', '$8$']
    why: 'The dot product is $2 + 3 + 0 - 1 = 4$, and dividing by $\sqrt{4} = 2$ gives 2.'
  - kind: apply
    q: 'The query at position 3 has scores $(2, 0, 0)$ for positions 1 to 3. What weight goes to position 1?'
    answer: '$0.79$'
    wrong: ['$1.00$', '$0.33$', '$0.88$']
    why: '$e^2 / (e^2 + e^0 + e^0) = 7.39 / 9.39 = 0.79$. Positions with score 0 still get $e^0 = 1$ on the bottom.'
  - kind: apply
    q: 'A sequence has $n = 5$ tokens. How many of the 25 entries in the score grid does the mask set to $-\infty$?'
    answer: '$10$'
    wrong: ['$15$', '$5$', '$20$']
    why: 'Everything above the diagonal gets masked: $4 + 3 + 2 + 1 = 10$. The 15 on and below the diagonal stay.'
  - kind: apply
    q: 'One head runs on $n = 6$ tokens with $d_k = 16$. What shape is the weight matrix $A$?'
    answer: '$6 \times 6$'
    wrong: ['$6 \times 16$', '$16 \times 16$', '$16 \times 6$']
    why: '$A$ holds one weight per (query, key) pair, so it''s $n \times n$. The output $AV$ is the one that''s $6 \times 16$.'
  - kind: apply
    q: 'Query and key entries are random with mean 0 and variance 1, and $d_k = 256$. Before scaling, how big is $q \cdot k$ typically?'
    answer: 'About 16'
    wrong: ['About 256', 'About 1', 'About 128']
    why: 'The variance of the dot product is $d_k = 256$, so its standard deviation is $\sqrt{256} = 16$. Dividing by $\sqrt{d_k}$ brings it back to about 1.'
  - kind: apply
    q: 'A model with $d = 512$ splits it evenly across 8 heads, the way GPT-2 does. What is $d_k$ for each head?'
    answer: '$64$'
    wrong: ['$512$', '$8$', '$4{,}096$']
    why: 'Each head gets $512 / 8 = 64$ entries, and joining the 8 outputs end to end gives back a vector of 512.'
  - kind: apply
    q: 'In the attention widget with "it" as the query, you drag the score scale from 1 to 2. What happens to the weight on "cat"?'
    answer: 'It goes up, from 0.76 to 0.99'
    wrong: ['It goes down, as weight spreads to the other tokens', 'It stays at 0.76, since only score differences matter', 'It drops to an equal share with every other token']
    why: 'Scaling multiplies every score, so the gaps between them grow too. Bigger gaps push the softmax to put more weight on the top score.'
---

After unit 10 every token has a vector, but each vector only describes its own token.
The vector for "it" is the same row of the embedding table whether "it" refers to a cat or a mat.
Attention is the step where each position looks back at the earlier positions and mixes some of their vectors into its own.
It's the only place in a transformer (unit 13) where information moves from one token to another.

## Mixing in earlier tokens

Take a sequence of $n$ tokens $x_1, \dots, x_n$, where token $x_i$ has a vector $h_i$ with $d$ entries.
In the first layer these are the embedding rows from unit 10.
In later layers they're whatever the layers below produced.

For each position $i$ I want a new vector that carries information from positions 1 to $i$.
The simplest way to get one is a weighted average, with a weight $a_{ij}$ for how much position $i$ takes from position $j$:

$$
o_i = a_{i1} h_1 + a_{i2} h_2 + \cdots + a_{ii} h_i
$$

::: formula
Each term is the weight $a_{ij}$ times $h_j$, a vector scaled by a number, and the terms run over the positions 1 to $i$ that position $i$ can see.
:::

The weights are at least 0 and add up to 1, so $o_i$ is a blend of the vectors it can see.

Equal weights would give every earlier word the same say.
In "the cat sat on the mat because it was tired", the new vector for "it" should take most of its weight from "cat", and "mat" should get almost none.
So the weights have to depend on the tokens themselves.

Softmax ([unit 11](../11-next-token-prediction/#softmax-turns-scores-into-probabilities)) turns any list of scores into weights that are positive and add up to 1, so what's left is where the scores come from.

## Queries, keys and values

Attention makes three new vectors from every $h_i$, each with its own learned matrix:

$$
q_i = W_Q h_i \qquad k_i = W_K h_i \qquad v_i = W_V h_i
$$

::: formula
Each is a matrix times $h_i$, as in [unit 1](../01-vectors/#a-matrix-is-a-stack-of-dot-products).
$W_Q$ and $W_K$ are $d_k \times d$, so every query $q_i$ and key $k_i$ has $d_k$ entries.
$W_V$ makes the value vectors $v_i$, and in this unit they have $d_k$ entries too, as in the original transformer paper.
:::

$d_k$ is the *key size*, and it's usually much smaller than $d$.

- **Query** $q_i$: what position $i$ is looking for.
- **Key** $k_j$: what position $j$ offers, written in the same $d_k$ numbers so the two can be compared with a [dot product](../01-vectors/#the-dot-product).
- **Value** $v_j$: what position $j$ hands over if it's picked.

> **An analogy.** It's a soft version of a search. The query is what you type, each key is what a document is filed under, and the value is the document's contents. A normal search hands back the best match; attention hands back a blend of every value, weighted by how well its key matched the query.

The score of position $j$ for the query at position $i$ is a scaled dot product, large when $i$'s query and $j$'s key point the same way:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

::: formula
The top is the dot product of $i$'s query with $j$'s key.
The bottom divides by the square root of the key size, for a reason the next section explains.
:::

The weights are the softmax of those scores over the positions $i$ is allowed to see, $j = 1, \dots, i$, and the output is the weighted sum of the values:

$$
\begin{aligned}
a_{ij} &= \frac{e^{s_{ij}}}{e^{s_{i1}} + \cdots + e^{s_{ii}}} \\
o_i &= a_{i1} v_1 + a_{i2} v_2 + \cdots + a_{ii} v_i
\end{aligned}
$$

::: formula
The first formula is softmax from [unit 11](../11-next-token-prediction/#softmax-turns-scores-into-probabilities), run over the $i$ scores in row $i$: $e$ to one score on top, $e$ to every visible score added up on the bottom.
The second is the weighted average from the first section, with the values $v_j$ in place of the raw vectors $h_j$.
:::

Scoring with $h_i \cdot h_j$ directly would make one vector play all three roles.
What "it" is looking for (a noun that could be the subject) is different from what "it" offers to the words after it, and both are different from the information it should pass along.
With three matrices, training can tune each role on its own.

<figure class="widget" id="attention-picker">
<figcaption>Orange bars are the weights from the blue query token to every token up to and including it. Tokens after the query are masked. Click a word or use the slider to change the query, and drag the score scale to see the softmax sharpen or flatten. The query and key vectors are four numbers I picked by hand, not taken from a trained model.</figcaption>
</figure>

In the widget, $d_k$ is 4 and I gave the four entries rough meanings: animal noun, thing noun, verb, pronoun.
"it" has the query $q = (3, -1, 0, 0)$, which reads as "looking for an animal, not a thing."
"cat" has the key $k = (2, 0, 0, 0)$, so its score is $6 / \sqrt{4} = 3$.
"mat" has $k = (0, 2, 0, 0)$ and scores $-2 / 2 = -1$.
Most other keys are zero vectors and score 0.
After the softmax, "it" puts 0.76 of its weight on "cat" and 0.01 on "mat".

In a trained model nobody picks these numbers.
$W_Q$ and $W_K$ are learned by gradient descent ([unit 4](../04-gradient-descent/#walk-downhill)), and their entries don't come with names.

The score scale slider multiplies every score by the same number.
At 0 every score is 0, so the weights are all equal.
At 2, "it" puts 0.99 on "cat".
Bigger scores push more of the weight onto the top score, and the $\sqrt{d_k}$ in the score formula is there to keep scores from getting that big.

## Why divide by the square root of the key size

Suppose the entries of $q$ and $k$ are independent ([unit 3](../03-probability/#conditional-probability-and-bayes-rule)) random numbers with mean 0 and variance 1 ([unit 3](../03-probability/#expectation-spread-and-sampling)).
Each product $q_m k_m$ of matching entries then has variance 1, and $q \cdot k$ adds up $d_k$ of them.
Variances of independent terms add, so the variance of $q \cdot k$ is $d_k$ and its typical size (the standard deviation, the square root of the variance) is $\sqrt{d_k}$.
At $d_k = 64$, the head size in GPT-2 and in the original paper, raw dot products are typically around 8 in size.

Softmax reacts to differences between scores, and differences of 8 are huge for it.
Scores $(1, 0)$ give weights $(0.73, 0.27)$.
Scores $(8, 0)$ give $(0.9997, 0.0003)$.
When one weight is almost 1, nudging the scores barely changes the weights, so the gradient that reaches $W_Q$ and $W_K$ is tiny and they learn slowly.
That's the explanation Vaswani et al. offer in the paper, and dividing by $\sqrt{d_k}$ brings the typical score back to about 1 for any key size.

<figure class="widget" id="scaling">
<figcaption>One random query against eight random keys, all entries drawn with mean 0 and variance 1. Blue bars are the scores (clipped at ±20, a notch marks a clipped bar), orange bars their softmax weights. "Scaling" divides every score by √d_k. Raise d_k with scaling off and the largest weight heads to 1. Turn scaling on and the scores stay within a few units of 0 at every d_k.</figcaption>
</figure>

## The causal mask

During training, every position predicts its next token in parallel (unit 11).
If position $i$ could attend to position $i + 1$, it could read the answer it's supposed to predict.
So position $i$ only gets to see positions $1$ to $i$.

In practice you compute all $n \times n$ scores at once and overwrite the ones where $j > i$ with $-\infty$.
Since $e^{-\infty} = 0$, those positions get zero weight and the softmax spreads all the weight over the rest.
For four tokens the score grid looks like this, with one row per query and one column per key:

<figure>
<pre>         k1     k2     k3     k4
  q1    <span class="blue">s11</span>     −∞     −∞     −∞
  q2    <span class="blue">s21</span>    <span class="blue">s22</span>     −∞     −∞
  q3    <span class="blue">s31</span>    <span class="blue">s32</span>    <span class="blue">s33</span>     −∞
  q4    <span class="blue">s41</span>    <span class="blue">s42</span>    <span class="blue">s43</span>    <span class="blue">s44</span></pre>
<figcaption>Scores on and below the diagonal are kept. The ones above it are set to −∞, so their weight is 0.</figcaption>
</figure>

Stack the $q_i$ as the rows of a matrix $Q$ (shape $n \times d_k$), and the same for $K$ and $V$.
Then all the scores are one matrix product, $Q K^\top$, and the whole layer is

$$
A = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) \qquad \text{output} = A V
$$

::: formula
$K^\top$ is $K$ with rows and columns swapped, so entry $(i, j)$ of $Q K^\top$ is row $i$ of $Q$ dotted with row $j$ of $K$, which is $q_i \cdot k_j$.
Inside the softmax, $Q K^\top / \sqrt{d_k}$ is the grid of every score $s_{ij}$ at once, and $M$ is the mask: 0 on and below the diagonal and $-\infty$ above it, so adding it leaves the visible scores alone and wipes out the rest.
The softmax runs along each row, and row $i$ of $AV$ is $a_{i1} v_1 + \cdots + a_{ii} v_i$, so multiplying the weights by $V$ forms every $o_i$ at once.
:::

$A$ is the $n \times n$ weight matrix.
The mask leaves nothing above the diagonal, so it's lower-triangular, and every row adds up to 1.

The mask also helps when generating text.
The keys and values of earlier positions never depend on later tokens, so they can be computed once and stored.
Each new token only needs its own query, key and value, and inference servers keep the stored ones in what's called a KV cache.

## Several heads

One set of $W_Q, W_K, W_V$ gives each position one set of weights.
A sentence usually needs more than one way of looking back, for example one that finds the noun "it" refers to and another that follows the previous word.

Multi-head attention runs several of these in parallel, called heads, each with its own three matrices and a smaller key size.
In GPT-2 small, $d = 768$ and there are 12 heads with $d_k = 64$ each.
Each head produces its own output vector with 64 entries, the 12 outputs are joined end to end into one vector of 768, and a final learned $d \times d$ matrix $W_O$ mixes them.
Because each head is smaller, 12 heads cost about the same as one head with $d_k = 768$.

## Worked example

```python
import numpy as np

rng = np.random.default_rng(0)
n, d, d_k = 4, 8, 4                     # 4 tokens, width 8, key size 4

H = rng.normal(size=(n, d))             # row i is the token vector h_i
W_Q = rng.normal(size=(d_k, d)) / np.sqrt(d)
W_K = rng.normal(size=(d_k, d)) / np.sqrt(d)
W_V = rng.normal(size=(d_k, d)) / np.sqrt(d)

Q = H @ W_Q.T                           # row i is q_i = W_Q h_i
K = H @ W_K.T                           # row j is k_j
V = H @ W_V.T                           # row j is v_j

S = Q @ K.T / np.sqrt(d_k)              # S[i, j] = q_i · k_j / sqrt(d_k)
future = np.triu(np.ones((n, n), dtype=bool), k=1)   # True where j > i
S[future] = -np.inf

A = np.exp(S - S.max(axis=1, keepdims=True))         # softmax of each row
A = A / A.sum(axis=1, keepdims=True)
out = A @ V                             # row i is the weighted sum of values

np.set_printoptions(precision=3, suppress=True)
print(A)
print(A.sum(axis=1))
print(out.shape)
print(out[0])
print(V[0])
```

```
[[1.    0.    0.    0.   ]
 [0.092 0.908 0.    0.   ]
 [0.296 0.406 0.298 0.   ]
 [0.1   0.519 0.137 0.244]]
[1. 1. 1. 1.]
(4, 4)
[-0.725  0.204  0.719 -0.682]
[-0.725  0.204  0.719 -0.682]
```

The weight matrix is lower-triangular and each row sums to 1.
The first token can only see itself, so its output is exactly its own value vector, and the last token spreads its weight over all four, with the most (0.519) on token 2.
Dividing the random matrices by $\sqrt{d}$ keeps the entries of $q$ and $k$ near variance 1.
If you delete those three divisions, rows 2 and 4 put 1.0 and 0.998 of their weight on a single token.

## Where this shows up later

- Unit 13: an attention layer and a small neural network (unit 8) together make a transformer block. Nothing in a score $q_i \cdot k_j$ says how far apart positions $i$ and $j$ are, so unit 13 also adds position information.
- Unit 13 again: the mask is what lets a transformer train on every position of a sequence at once, with the cross-entropy loss from unit 11.
- Unit 14: generating text one token at a time, where each new token attends over the cached keys and values of everything before it.

## Self-test

<details>
<summary>In the widget with "it" as the query, why does "mat" get less weight than "the"?</summary>

"the" has a zero key, so its score is 0.
"mat" has the key (0, 2, 0, 0), and the query for "it" is (3, −1, 0, 0), so its score is −2 / 2 = −1.
A lower score means a smaller weight after the softmax: 0.01 for "mat" against 0.04 for each "the".

</details>

<details>
<summary>In the widget, set the score scale to 0. What happens to the weights, and why?</summary>

Every score becomes 0, and $e^0 = 1$ for every visible position, so all weights are equal.
A query at position $i$ puts $1/i$ on each of the $i$ tokens it can see.

</details>

<details>
<summary>Why does a language model need the causal mask during training?</summary>

Every position is trained to predict the token after it, all at the same time.
Without the mask, position $i$ could attend to position $i + 1$ and copy the answer, so the model would learn to copy instead of predict, and copying fails at generation time when the next token doesn't exist yet.

</details>

<details>
<summary>Why are the scores divided by the square root of d_k?</summary>

With random entries of variance 1, a dot product of two vectors with $d_k$ entries has standard deviation $\sqrt{d_k}$.
Big scores push the softmax to put almost all weight on one position, where its gradients are tiny.
Dividing by $\sqrt{d_k}$ keeps the scores around size 1 whatever the key size is.

</details>

<details>
<summary>A sequence has 10 tokens and one head has d_k = 64. What are the shapes of Q, Q times K-transpose, and the head's output?</summary>

$Q$ is $10 \times 64$.
$Q K^\top$ is $10 \times 10$, one score per (query, key) pair.
The output $A V$ is $10 \times 64$, one new vector per token.

</details>

::: done-when
- Given small $q$ and $k$ vectors for three tokens, you can compute the causal attention weights for the third token by hand.
- You can explain what the mask does and why training on every position at once needs it.
- You can explain the $\sqrt{d_k}$ with the variance argument and say what goes wrong without it.
- Given $n$ and $d_k$, you can write down the shapes of $Q$, $K$, $V$, $Q K^\top$ and the output.
:::

## Go deeper

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), section 3.2: scaled dot-product attention and multi-head attention in two pages, with the variance argument for $\sqrt{d_k}$ in footnote 4.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar: step-by-step pictures of queries, keys, values and multiple heads. Its self-attention pictures come from the encoder, which has no mask, so every token sees every other token there.
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy: builds one causal self-attention head in PyTorch, including the lower-triangular mask, then multiple heads.
