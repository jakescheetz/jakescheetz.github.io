---
title: Neural networks and backpropagation
subtitle: Weighted sums with a bend between them, and the chain rule run backward to train them.
unit: 8
section: Deep learning
minutes: 75
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'What does ReLU do to a number?'
    answer: 'Keeps it if it''s positive and turns it into 0 if it''s negative'
    wrong: ['Squashes it into the range from 0 to 1', 'Takes its absolute value', 'Turns it into 1 if it''s positive and 0 if it''s negative']
    why: '$\mathrm{ReLU}(z) = \max(0, z)$, so $0.5$ stays $0.5$ and $-0.5$ becomes 0.'
  - kind: term
    q: 'In a layer $h = \mathrm{ReLU}(Wx + b)$, what is one row of $W$?'
    answer: 'One neuron''s weight vector'
    wrong: ['One input''s weights into every neuron', 'One training example', 'The layer''s biases']
    why: 'Each neuron''s weights are stacked as a row, so each output entry is one row dotted with $x$, plus that neuron''s bias.'
  - kind: term
    q: 'What is a hidden unit?'
    answer: 'An entry of a middle layer''s output'
    wrong: ['A weight that training never updates', 'An input feature the model ignores', 'A neuron whose ReLU is always off']
    why: 'The entries of $h$ never appear in the input or the output, which is where the name comes from.'
  - kind: term
    q: 'What is a dead unit?'
    answer: 'A hidden unit whose $z$ is negative for every training input'
    wrong: ['A hidden unit whose weights all start at exactly zero', 'A hidden unit that gets deleted after training', 'A hidden unit that copies another unit''s output']
    why: 'ReLU''s local derivative is 0 wherever $z$ is negative, so no gradient ever reaches its weights and it can''t recover.'
  - kind: term
    q: 'What is backpropagation?'
    answer: 'The chain rule applied from the loss backward, reusing forward-pass values'
    wrong: ['Running the network in reverse to recover the input from the output', 'Estimating each gradient by nudging every weight both ways', 'Feeding the loss back in as the next input']
    why: 'Each node multiplies the gradient arriving from its right by its own local derivative, and those local derivatives read values the forward pass already stored.'
  - kind: apply
    q: 'A neuron has $w = (1, 2)$ and $b = -3$. What does it output for $x = (1, 0.5)$?'
    answer: '$0$'
    wrong: ['$-1$', '$2$', '$5$']
    why: '$z = 1 + 1 - 3 = -1$, and ReLU turns that into 0. Forgetting the ReLU gives $-1$, and forgetting the bias gives 2.'
  - kind: apply
    q: 'A layer has $W$ of shape $8 \times 2$. How many numbers does it read, and how many does it output?'
    answer: 'Reads 2, outputs 8'
    wrong: ['Reads 8, outputs 2', 'Reads 2, outputs 2', 'Reads 16, outputs 1']
    why: 'An $m \times n$ weight matrix takes $n$ inputs and has one neuron per row, so $m$ outputs.'
  - kind: apply
    q: 'You delete every ReLU from a three-layer network. What can it compute now?'
    answer: 'Only what a single linear layer can'
    wrong: ['The same functions as before', 'Anything, if the layers are wide enough', 'Only functions that pass through the origin']
    why: 'Multiplying the layers out gives one matrix times $x$ plus one bias. It can''t fit XOR anymore.'
  - kind: apply
    q: 'The output is $\hat y = 0.5$, the target is $y = 1$ and $L = (\hat y - y)^2$. What is $\partial L / \partial \hat y$?'
    answer: '$-1$'
    wrong: ['$0.25$', '$-0.5$', '$1$']
    why: 'The loss''s local derivative is $2(\hat y - y) = 2(-0.5) = -1$. The loss itself is 0.25.'
  - kind: apply
    q: 'In the backward pass, a ReLU node had $z = 0.4$ going forward and receives $\partial L / \partial h = -0.6$. What is $\partial L / \partial z$?'
    answer: '$-0.6$'
    wrong: ['$0$', '$-0.24$', '$0.6$']
    why: '$z$ was positive, so ReLU''s local derivative is 1 and the gradient passes through unchanged.'
  - kind: apply
    q: 'An output is $\hat y = w \cdot h + b$ with $w = (1, 0.5)$ and $h = (2, 0.5)$, and $\partial L / \partial \hat y = -3$. What is $\partial L / \partial w$?'
    answer: '$(-6, -1.5)$'
    wrong: ['$(-3, -1.5)$', '$(6, 1.5)$', '$(-3, -3)$']
    why: 'The derivative of a weighted sum with respect to a weight is the input it multiplies, so it''s $-3 \cdot h$. Using $w$ instead gives $(-3, -1.5)$, which is $\partial L / \partial h$.'
  - kind: apply
    q: 'Your new backprop code makes the training loss go down. Is the gradient correct?'
    answer: 'Not necessarily, so compare a few entries with a central difference'
    wrong: ['Yes, a wrong gradient would make the loss go up', 'Yes, as long as the loss reaches zero', 'No, a correct gradient would make the loss drop every step']
    why: 'A slightly wrong gradient (a transposed matrix, a missing factor of 2) often still lowers the loss. The finite-difference check catches it.'
---

A neural network is a stack of the matrix-vector products from [unit 1](../01-vectors/#a-matrix-is-a-stack-of-dot-products), with a simple function applied between them.
Training one with [gradient descent](../04-gradient-descent/#walk-downhill) needs the gradient of the loss with respect to every weight, and the procedure that computes it is called *backpropagation*.
Every model in the rest of this route is trained with it.

## One neuron

A neuron takes an input vector $x$, computes a weighted sum of its entries using a weight vector $w$, adds a number $b$ called the bias, and passes the result through a nonlinearity:

$$
z = w \cdot x + b, \qquad h = \mathrm{ReLU}(z) = \max(0, z)
$$

::: formula
$w$ and $b$ are the neuron's parameters, and training is what sets them.
The first part, $z$, is the same score $w \cdot x + b$ that logistic regression computes in [unit 5](../05-classification/#from-a-score-to-a-probability), where it went through the sigmoid to become a probability.
:::

The nonlinearity here is *ReLU* (short for rectified linear unit), which keeps positive numbers as they are and turns negative ones into zero.

Take $w = (0.5, -0.5)$ and $b = 0$.
For $x = (1, 0)$ the weighted sum is $z = 0.5$, so $h = 0.5$.
For $x = (0, 1)$ it's $z = -0.5$, so $h = 0$.

The weighted sum is a dot product, so the reading from [unit 1](../01-vectors/#the-dot-product) applies: $z$ is large when $x$ points along $w$.
So a neuron works like a detector: it stays at zero until $w \cdot x$ gets past a threshold, which the bias sets, and then grows in a straight line.

With two inputs, the points where $z = 0$ form a line across the plane.
On one side the neuron outputs zero, and on the other its output rises with distance from the line, so I picture ReLU as folding the plane along that line.

## Layers, and why the bend matters

A layer is many neurons reading the same input.
Stack their weight vectors as the rows of a matrix $W$ and their biases into a vector $b$, and the whole layer is one line:

$$
h = \mathrm{ReLU}(W x + b)
$$

::: formula
ReLU is applied to each entry separately.
If $W$ is $m \times n$, the input has $n$ entries and the layer outputs $m$ of them, one per neuron.
:::

A multi-layer perceptron (MLP) is a few layers in a row, each one reading the previous one's output.
The network in this unit has two layers:

$$
h = \mathrm{ReLU}(W_1 x + b_1), \qquad \hat y = W_2 h + b_2
$$

The entries of $h$ are called hidden units, since you never see them in the input or the output.
The last layer has no ReLU, so the output $\hat y$ can be any number, including a negative one.

To see why the ReLU has to be there, take it out and expand:

$$
W_2 (W_1 x + b_1) + b_2 = (W_2 W_1)\, x + (W_2 b_1 + b_2)
$$

That's one matrix times $x$ plus one bias, which is a single linear layer.
The same expansion works for any number of layers, so a stack of linear layers can never compute anything that one linear layer can't.

XOR is the standard example of something a linear function can't do.
It takes two inputs that are each 0 or 1 and outputs 1 when exactly one of them is 1.
Suppose a linear function $f(x) = a \cdot x + c$ fit it, with outputs above 0.5 meaning 1 and below meaning 0.
Adding up its values at the two corners where XOR is 1 gives $f(0, 1) + f(1, 0) = a_1 + a_2 + 2c$, which must be above 1.
Adding up the other two corners gives $f(0, 0) + f(1, 1) = a_1 + a_2 + 2c$, the same number, which must be below 1.
Both can't be true, so no linear function fits XOR.

Two ReLU units are enough.
Set $h_1 = \mathrm{ReLU}(x_1 + x_2)$, $h_2 = \mathrm{ReLU}(x_1 + x_2 - 1)$ and $\hat y = h_1 - 2 h_2$:

<figure>
<pre>x1  x2   h1  <span class="orange">h2</span>   <span class="blue">ŷ</span> = h1 − 2·h2
 0   0    0   <span class="orange">0</span>   <span class="blue">0</span>
 0   1    1   <span class="orange">0</span>   <span class="blue">1</span>
 1   0    1   <span class="orange">0</span>   <span class="blue">1</span>
 1   1    2   <span class="orange">1</span>   <span class="blue">0</span></pre>
<figcaption>A hand-built XOR network. h2 stays at zero until both inputs are on, then cancels h1.</figcaption>
</figure>

In practice you start from random weights and let gradient descent find good ones, which needs the gradient.

## Backpropagation

Gradient descent needs $\partial L / \partial \theta$ for every parameter $\theta$ (every entry of $W_1$, $b_1$, $W_2$ and $b_2$).
The loss $L$ here is the squared error $(\hat y - y)^2$ between the output $\hat y$ and the target $y$ for a single input, the squared miss that the [mean squared error](../04-gradient-descent/#a-loss-is-one-number-for-how-wrong) in unit 4 averages over points.
Unit 5 trained its classifier with [log loss](../05-classification/#log-loss) because its output was a probability; this network's output can be any number, so squared error is the simpler fit.
The [chain rule](../02-gradients/#the-rules-you-need) says how to get the gradient: a nudge to a weight propagates forward through the network, and the rates of change along the way multiply.

To use the chain rule mechanically, break the network into a computational graph.
Each node computes one simple thing from the nodes feeding into it: a weighted sum, a ReLU, the loss.
Take the two-layer network from above with two hidden units.
$W_1$ has rows $(0.5, -0.5)$ and $(-1, 1)$, $b_1 = (0, 0.5)$, $W_2$ is the single row $(1, 0.5)$, and $b_2 = 0$.
The input is $x = (1, 0)$ and the target is $y = 1$.

The forward pass computes every node from left to right and keeps each value.
The backward pass then starts at the loss with $\partial L / \partial L = 1$ and moves right to left.
Each node takes the gradient arriving from the node after it and multiplies by its own local derivative:

$$
\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial \hat y}\,\frac{\partial \hat y}{\partial h_1}
$$

::: formula
Read it right to left through the network: $\partial L / \partial \hat y$ is the gradient that has already arrived at the output, $\partial \hat y / \partial h_1$ is the output node's local derivative with respect to the hidden unit $h_1$, and their product is how much the loss changes per unit change in $h_1$.
:::

The local derivative of the loss is $2(\hat y - y)$.
For a weighted sum, the derivative with respect to an input is that input's weight, and with respect to a weight it's the input that weight multiplies.
For ReLU it's 1 if $z > 0$ and 0 otherwise.

<figure class="widget" id="backprop-steps">
<figcaption>Step through the forward pass (blue numbers above each node) and then the backward pass (orange numbers below, each one ∂L/∂node). Underlined values are forward results that a backward step reads again.</figcaption>
</figure>

Two things make this cheap.
First, every local derivative uses a value the forward pass already computed: the ReLU step reads $z$, the gradient for $W_2$ reads $h$, the gradient for $W_1$ reads $x$.
Second, each node's gradient is computed once and then shared by everything to its left, so one backward pass gets every weight's gradient for a small multiple of the cost of one forward pass.

At step 13 in the widget, the gradient into $z_2$ is 0 because $z_2$ was negative and ReLU passed nothing on.
So neither weight into $z_2$ learns anything from this example.
A hidden unit whose $z$ is negative for every training input gets no gradient at all and can never recover, which is called a *dead unit*.

Written for a whole layer at once, the backward pass is four lines:

$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat y}\, h^\top, \qquad
\frac{\partial L}{\partial h} = W_2^\top \frac{\partial L}{\partial \hat y}
$$

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \odot [z > 0], \qquad
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z}\, x^\top
$$

::: formula
$W^\top$ is $W$ with rows and columns swapped, $h^\top$ and $x^\top$ turn a column vector into a row, and $\odot$ multiplies two vectors entry by entry.
The first line says each weight in $W_2$ gets the output's gradient times the hidden value it multiplied.
The second sends the output's gradient back to the hidden units, each scaled by its weight in $W_2$.
The third is the ReLU step: $[z > 0]$ is the vector with a 1 where $z$ is positive and a 0 elsewhere, so the gradient passes only through units that were on.
The fourth gives each weight in $W_1$ the gradient at its $z$ times the input it multiplied.
:::

They're the local-derivative rules above, applied to every unit of a layer at once.
The bias gradients are $\partial L / \partial \hat y$ and $\partial L / \partial z$ themselves, and the worked example below is these four lines in NumPy.

## Checking the gradient, then training

Backprop code is easy to get slightly wrong (a transposed matrix, a missing factor of 2), and a slightly wrong gradient often still lowers the loss, so watching the loss won't catch the bug.
To catch it, compare against the [central difference](../02-gradients/#checking-a-gradient-with-nudges) on a few weights:

$$
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + \varepsilon) - L(\theta_i - \varepsilon)}{2\varepsilon}
$$

It nudges one parameter $\theta_i$ up and down by a tiny $\varepsilon$, leaving the rest alone, and divides the change in the loss by the $2\varepsilon$ distance between the two nudges.
That costs two forward passes per parameter, so it's fine for spot-checking a few weights and far too slow for training.

Training then repeats three steps: forward pass on all four XOR inputs, backward pass, and the [update rule](../04-gradient-descent/#walk-downhill) $\theta \leftarrow \theta - \eta \nabla L$, which moves every parameter a step of size $\eta$ (the learning rate) against its gradient.

<figure class="widget" id="xor-training">
<figcaption>The worked example's network and η = 0.1, training live. Blue shading is where the output is above 0.5, orange below, and dashed lines are the hidden units' folds (where z = 0). Try 1 and 2 hidden units, pressing New start a few times at each.</figcaption>
</figure>

The output is a sum of folded planes, so the blue region always has straight edges that only bend where a dashed line crosses.
With one hidden unit it never fits.
The output only changes as you move along one direction in the plane, so the 0.5 boundary is a single straight line, and the XOR argument above showed no line works.
Two units are enough in principle (the hand-built network uses two), but from a random start gradient descent often gets stuck with a dead unit or a bad fold.
When I pressed New start 20 times at each size and ran 1,000 steps, the network solved it 4 times with two hidden units, 12 times with three, and all 20 times with eight.

## Worked example

```python
import numpy as np

# XOR: the target is 1 when exactly one input is 1.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)

rng = np.random.default_rng(0)
H = 8                                       # hidden units
W1 = rng.normal(0, 1, (H, 2)); b1 = np.zeros(H)
W2 = rng.normal(0, 0.5, (1, H)); b2 = np.zeros(1)

def forward(W1, b1, W2, b2):
    z = X @ W1.T + b1                       # (4, H) weighted sums
    h = np.maximum(0, z)                    # ReLU
    y_hat = (h @ W2.T + b2)[:, 0]           # (4,) outputs
    L = np.mean((y_hat - y) ** 2)
    return z, h, y_hat, L

def backward(z, h, y_hat):
    d_yhat = 2 * (y_hat - y) / len(y)       # dL/dy_hat
    dW2 = d_yhat[None, :] @ h               # reuses h from the forward pass
    db2 = d_yhat.sum(keepdims=True)
    dh = d_yhat[:, None] * W2               # dL/dh
    dz = dh * (z > 0)                       # ReLU passes gradient only where z > 0
    dW1 = dz.T @ X                          # reuses the inputs
    db1 = dz.sum(axis=0)
    return dW1, db1, dW2, db2

# Gradient check on one weight before training.
z, h, y_hat, L = forward(W1, b1, W2, b2)
dW1, db1, dW2, db2 = backward(z, h, y_hat)
eps = 1e-6
Wp = W1.copy(); Wp[0, 0] += eps
Wm = W1.copy(); Wm[0, 0] -= eps
numeric = (forward(Wp, b1, W2, b2)[3] - forward(Wm, b1, W2, b2)[3]) / (2 * eps)
print("backprop:", dW1[0, 0], " finite difference:", numeric)

eta = 0.1
for step in range(501):
    z, h, y_hat, L = forward(W1, b1, W2, b2)
    if step % 100 == 0:
        print(step, f"{L:.6f}")
    dW1, db1, dW2, db2 = backward(z, h, y_hat)
    W1 -= eta * dW1; b1 -= eta * db1
    W2 -= eta * dW2; b2 -= eta * db2

print(np.round(y_hat, 3))
```

```
backprop: 0.062015460943072497  finite difference: 0.06201546093320687
0 0.374618
100 0.109105
200 0.005582
300 0.000098
400 0.000002
500 0.000000
[0. 1. 1. 0.]
```

The two gradients agree to ten decimal places, so the backward pass is right before any training happens.
The loss falls from 0.374618 to below 0.0001 by step 300, and the four outputs round to 0, 1, 1, 0, which is XOR.

## Where this shows up later

- Unit 10: the embedding table is a layer's weights, trained by this same backward pass.
- Unit 11: the last layer outputs one number per vocabulary token, and the backward pass starts from the cross-entropy loss instead of $(\hat y - y)^2$.
- Unit 13: each transformer block contains a two-layer MLP like this one, applied to every token's vector (GPT-2 uses GELU, a smooth version of ReLU).
- Unit 15: a reward from the environment isn't a node you can backprop through, so reinforcement learning needs another way to get a gradient.

## Self-test

<details>
<summary>A neuron has w = (2, −1) and b = −1. What does it output for x = (2, 1), and for x = (0, 1)?</summary>

For $(2, 1)$: $z = 4 - 1 - 1 = 2$, so $h = 2$.
For $(0, 1)$: $z = -1 - 1 = -2$, so ReLU outputs 0.

</details>

<details>
<summary>Why doesn't stacking ten linear layers, with no ReLU between them, give you a more capable model than one layer?</summary>

Multiplying out the layers gives one matrix times $x$ plus one bias, which is a single linear layer.
The nonlinearity is what stops the layers collapsing into one.

</details>

<details>
<summary>In the step-through widget, the gradient into z2 is 0. Why, and what does that mean for the weights into z2?</summary>

$z_2 = -0.5$ was negative, so ReLU's local derivative there is 0 and nothing passes back through it.
Those weights get zero gradient from this example, so gradient descent leaves them alone.

</details>

<details>
<summary>In the training widget, one hidden unit never solves XOR, whatever the start. Why?</summary>

The output is $W_2$ times one ReLU plus a bias, so it only changes as you move along one direction in the plane, and the boundary where it crosses 0.5 is a single straight line.
XOR can't be split by a straight line.

</details>

<details>
<summary>Your backprop gradient and the finite-difference check disagree in the third decimal place. What do you suspect?</summary>

A bug in the backward pass, like a missing factor or a transposed matrix; the central difference should agree to many more digits.
The one innocent cause is a ReLU switching on or off between $\theta - \varepsilon$ and $\theta + \varepsilon$, which a smaller $\varepsilon$ usually rules out.

</details>

::: done-when
- You can run a small network's forward pass by hand and say which ReLUs are off.
- You can show with the algebra why two linear layers with nothing between them collapse into one.
- You can do the backward pass on the step-through widget's graph without looking, getting each gradient from the one to its right.
- You can write a finite-difference check for one weight and say how closely it should agree.
:::

## Go deeper

- 3Blue1Brown, [What is backpropagation really doing?](https://www.3blue1brown.com/lessons/backpropagation) and [Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus): the intuition first, then the chain rule on a chain of single neurons.
- Andrej Karpathy, [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0): two and a half hours turning the step-through widget's graph into a tiny autograd engine.
- Michael Nielsen, [Neural Networks and Deep Learning, chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html): the backward pass in matrix form, derived carefully (with a sigmoid where this unit uses ReLU).
