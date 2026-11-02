---
title: The transformer block
subtitle: Attention and a small network, added onto every token's vector, stacked many times.
unit: 8
minutes: 70
scripts: [widgets.js]
---

Unit 7 built attention, the step that lets one token's vector pull in information from earlier tokens.
A language model is attention plus a few more parts, wrapped into a block that gets stacked a dozen or a hundred times.
The parts go from token IDs at the bottom to the logits of unit 6 at the top.
Every model in the rest of the route uses this architecture.

## The residual stream

Start with the input.
Each token $x_i$ becomes a vector $h$ of width $d$: its row of the embedding table $E$ from unit 5, plus a vector that encodes its position (more on that below).
For GPT-2 small, $d = 768$.

Now picture each token's vector as a lane running straight up through the model.
Each block reads the vector in a lane, computes something, and adds the result back into the same lane.
It never replaces the vector, only adds to it.
This lane is called the residual stream, and the addition is called a residual connection.

Written out, with $f$ standing for whatever a sublayer computes:

$$
h \leftarrow h + f(h)
$$

After $N$ blocks, each with an attention part that adds $a$ and an MLP part that adds $m$, the vector at the top is a plain sum:

$$
h_{\text{top}} = h_0 + a_1 + m_1 + a_2 + m_2 + \cdots + a_N + m_N
$$

where $h_0$ is the embedding the lane started with.
I think of it as a shared notepad.
Every sublayer reads the whole notepad and writes a note at the bottom, and nobody erases anything.

<figure class="widget" id="residual-stream">
<figcaption>Four tokens flow up through two blocks of a tiny transformer with random weights, computed live in this page. Each row is every token's vector (8 numbers as bars) after one more sublayer. The faint bar is what came in, the colored piece is what that sublayer added (orange for attention, green for the MLP), and the tick is the new value. Orange lines show the selected token's attention weights to earlier tokens. Click a column or use the sliders.</figcaption>
</figure>

Two things to look for in the widget.
Only the attention rows draw lines between columns, because attention is the only part that moves information from one position to another.
The MLP rows work on each column alone.
And every row is the row below plus a colored piece, so the bottom row's numbers are still in there at the top.

There's a training reason for the additions too.
In unit 4, backprop multiplied gradients through every layer on the way down.
With $h + f(h)$, the derivative of the output with respect to $h$ is the identity plus the derivative of $f$, so the gradient always has a direct path back to the embeddings, however many blocks sit on top.
It's a big part of why stacks as deep as GPT-3's 96 blocks still train.

## Inside one block

A block has two sublayers, one after the other.
First attention, which mixes information across positions.
Then an MLP (the small network from unit 4), which processes each position on its own.
Each sublayer reads a normalized copy of the stream and adds its output back:

$$
h \leftarrow h + \mathrm{Attn}(\mathrm{LN}(h))
$$

$$
h \leftarrow h + \mathrm{MLP}(\mathrm{LN}(h))
$$

<figure>
<pre>h ──┬──────────────────► (+) ──┬──────────────────► (+) ──► h
    │                     ▲    │                     ▲
    └──► <span class="cyan">LN</span> ──► <span class="orange">attn</span> ─────┘    └──► <span class="cyan">LN</span> ──► <span class="green">MLP</span> ──────┘</pre>
<figcaption>One pre-norm block. The straight line along the top is the residual stream. Each sublayer branches off it, and its result is added back in.</figcaption>
</figure>

$\mathrm{LN}$ is layer norm.
It works on one token's vector at a time: subtract the mean of its $d$ entries, divide by their standard deviation, then multiply entry by entry by a learned vector $\gamma$ and add a learned vector $\beta$.
The stream tends to get longer as blocks keep adding to it, and layer norm hands each sublayer an input at the same scale no matter how long the stream has gotten.

Placing layer norm at the input of each sublayer is called pre-norm.
The original 2017 transformer normalized after the addition instead, $\mathrm{LN}(h + f(h))$, which puts a normalization on the main path itself.
GPT-2 moved it to the sublayer inputs, and most models since then do the same.
With pre-norm, the stream itself never passes through anything except additions.

$\mathrm{Attn}$ is unit 7's attention, with one more matrix at the end.
Each head produces a vector of size $d_k$, the heads' outputs are placed side by side, and an output matrix $W_O$ ($d \times d$) maps that back to width $d$ so it can be added to the stream.

$\mathrm{MLP}$ is two layers: a matrix that widens the vector from $d$ to $4d$, a nonlinearity, and a matrix that brings it back to $d$.
GPT-2 uses GELU, a smoothed version of ReLU, as the nonlinearity.
Per block, attention has about $4d^2$ weights and the MLP about $8d^2$, so two thirds of a block's weights sit in its MLP.

## Telling positions apart

Attention on its own has no idea what order the tokens are in.
The scores are dot products between vectors, and the output is a weighted sum, and neither one looks at where a vector sits in the sequence.

Take "dog bites man ." and "man bites dog .".
At the final ".", the first attention layer sees the same four vectors in both sentences, just in a different order.
Without position information its output there is identical, even though the two sentences mean different things.
(The causal mask does let later layers recover some order, because each position sees a different number of tokens, but models don't leave it to that.)

GPT-2's fix is a second learned table $P$, one row per position, 1024 rows of width $d$.
The input to the first block is $h_0 = E_{x_i} + P_i$: row $x_i$ of $E$ (the token's embedding, as in unit 5) plus row $i$ of $P$ (its position's embedding).
Training shapes $P$ the same way it shapes $E$.

Many newer models, Llama among them, skip the position table and rotate each query and key vector by an angle that depends on its position, inside every attention layer.
The rotation is set up so that $q \cdot k$ depends on how far apart the two tokens are, not on their absolute positions.
Either way, position has to be put in on purpose.

## From the top of the stream to a prediction

After the last block comes one more layer norm, then the unembedding matrix $W_U$ ($V \times d$) from unit 6 turns each position's vector into $V$ logits:

$$
z = W_U \, \mathrm{LN}(h_{\text{top}})
$$

GPT-2 reuses the embedding table here, $W_U = E$, so it adds no new parameters.
A softmax gives the probabilities for the next token, as in unit 6.

Every position gets its own logits.
Position $i$ predicts token $x_{i+1}$, and because of the causal mask in every attention layer, nothing at position $i$ has seen $x_{i+1}$ or anything after it.
So one forward pass over $n$ tokens gives $n$ predictions, none of which saw its answer, and the training loss averages the cross-entropy over all of them.
When generating text (unit 9), only the last position's logits get used.

## Worked example

One pre-norm block on 4 tokens of width 8, with random weights.
It uses a single attention head with $d_k = d = 8$ and ReLU in the MLP to stay close to units 4 and 7.
It prints the shape at each step and the length $\|h\|$ of each token's vector, then counts parameters.

```python
import numpy as np

rng = np.random.default_rng(0)
n, d, d_ff = 4, 8, 32                 # 4 tokens, width 8, MLP hidden size 4d

def layer_norm(h, gamma, beta, eps=1e-5):
    mu = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    return gamma * (h - mu) / np.sqrt(var + eps) + beta

def softmax(s):
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def param(*shape, scale=0.3):
    return rng.normal(0, scale, shape)

p = {
    "ln1_g": np.ones(d), "ln1_b": np.zeros(d),
    "W_Q": param(d, d), "W_K": param(d, d), "W_V": param(d, d), "W_O": param(d, d),
    "b_Q": np.zeros(d), "b_K": np.zeros(d), "b_V": np.zeros(d), "b_O": np.zeros(d),
    "ln2_g": np.ones(d), "ln2_b": np.zeros(d),
    "W_1": param(d, d_ff), "b_1": np.zeros(d_ff),
    "W_2": param(d_ff, d), "b_2": np.zeros(d),
}

def attention(x):
    q, k, v = x @ p["W_Q"] + p["b_Q"], x @ p["W_K"] + p["b_K"], x @ p["W_V"] + p["b_V"]
    scores = q @ k.T / np.sqrt(d)
    scores[np.triu_indices(n, 1)] = -np.inf  # causal mask
    return softmax(scores) @ v @ p["W_O"] + p["b_O"]

def mlp(x):
    return np.maximum(0, x @ p["W_1"] + p["b_1"]) @ p["W_2"] + p["b_2"]

h = rng.normal(0, 1, (n, d))           # token + position embeddings, one row per token
print("h in:       ", h.shape, np.linalg.norm(h, axis=1).round(2))

a = attention(layer_norm(h, p["ln1_g"], p["ln1_b"]))
h = h + a
print("attn adds:  ", a.shape, np.linalg.norm(a, axis=1).round(2))

m = mlp(layer_norm(h, p["ln2_g"], p["ln2_b"]))
h = h + m
print("mlp adds:   ", m.shape, np.linalg.norm(m, axis=1).round(2))
print("h out:      ", h.shape, np.linalg.norm(h, axis=1).round(2))

count = sum(v.size for v in p.values())
print("parameters in this block:", count, "=", 12 * d * d + 13 * d)

D, V, ctx, blocks = 768, 50257, 1024, 12   # GPT-2 small
per_block = 12 * D * D + 13 * D
print("GPT-2 small, one block:", per_block)
print("GPT-2 small, total:", blocks * per_block + V * D + ctx * D + 2 * D)
```

```
h in:        (4, 8) [3.12 3.96 1.72 2.44]
attn adds:   (4, 8) [1.24 0.53 1.46 0.83]
mlp adds:    (4, 8) [3.23 2.47 2.23 3.01]
h out:       (4, 8) [3.75 5.01 3.35 3.08]
parameters in this block: 872 = 872
GPT-2 small, one block: 7087872
GPT-2 small, total: 124439808
```

Every step keeps the shape at (4, 8), one row per token, which is what lets blocks stack.
The count $12d^2 + 13d$ is four $d \times d$ attention matrices, the two MLP matrices ($8d^2$ together), and the biases and layer norm vectors.
At GPT-2 small's sizes it gives 124,439,808 parameters, the commonly quoted "124M", and only about 39 million of those are the embedding table.

## Where this shows up later

- Unit 9: generation runs this stack, takes the last position's logits, picks a token, appends it, and runs again.
- Unit 10: the policy $\pi_\theta(y \mid x)$ is this whole stack, and $\theta$ is every weight in it.
- Unit 11: a reward model is usually the same kind of transformer with $W_U$ replaced by a layer that outputs one number.

## Self-test

<details>
<summary>In the widget, why do only the attention rows draw lines between columns?</summary>

Attention is the only part of a block that reads other positions.
Layer norm and the MLP each work on one token's vector by itself, so in an MLP row each column changes based only on its own lane.

</details>

<details>
<summary>Why does each sublayer add its output to h instead of replacing h?</summary>

Adding keeps everything earlier layers wrote available to later ones.
It also gives the gradient a direct path back through every block, since the derivative of h + f(h) with respect to h is the identity plus the derivative of f.

</details>

<details>
<summary>A model has no position embeddings. Why does its first attention layer give the same output at the "." in "dog bites man ." and "man bites dog ."?</summary>

At that position it attends over the same four token vectors in both sentences.
The scores and the weighted sum only depend on the vectors, not on their order, so the output comes out the same.

</details>

<details>
<summary>In a pre-norm block, where does layer norm go, and what does it never touch?</summary>

At the input of each sublayer, and once more before the unembedding.
It never sits on the residual stream itself: the stream only ever gets things added to it.

</details>

<details>
<summary>Why can training compute a loss at all n positions in one forward pass without the model cheating?</summary>

The causal mask stops position i from attending to anything after it, in every layer.
So the vector at position i depends only on x1 to xi, and predicting x(i+1) from it is a fair test.

</details>

::: done-when
- You can write the two lines of a pre-norm block and say what each part does.
- You can explain why the top vector of the stream is the embedding plus a sum of every sublayer's contribution.
- You can explain why attention needs position information and name the two ways it gets added.
- You can compute a block's parameter count from $d$ using $12d^2 + 13d$.
:::

## Go deeper

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar: diagrams of the whole stack, from token embeddings to the output logits.
- [nanoGPT's model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) by Andrej Karpathy: a complete GPT-2 in about 330 lines of PyTorch. Its `Block` class is the two lines from this unit.
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Anthropic, 2021): the section on the residual stream treats it as a shared channel that layers read from and write to, which is the view this unit uses.
