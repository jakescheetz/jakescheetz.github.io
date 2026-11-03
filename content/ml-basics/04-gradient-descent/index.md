---
title: Loss functions and gradient descent
subtitle: Score how wrong the model is with one number, then walk downhill.
unit: 4
section: Classical ML
minutes: 60
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'What does the loss $L$ measure?'
    answer: 'How wrong the model''s outputs are, as one number'
    wrong: ['How many parameters the model has', 'How far each update step moves the parameters', 'How fast the model is learning per step']
    why: 'It turns a whole set of predictions into one number where lower is better, like the mean squared error of a line on four points.'
  - kind: term
    q: 'What does the learning rate $\eta$ control in $\theta \leftarrow \theta - \eta \nabla L(\theta)$?'
    answer: 'How far each step goes'
    wrong: ['Which direction each step goes', 'How many points go into the loss', 'Where the bottom of the loss is']
    why: 'The direction comes from $-\nabla L$. $\eta$ is a small positive number that scales the step.'
  - kind: term
    q: 'In the bowl $L(\theta) = \frac{1}{2}\lambda\theta^2$, what does $\lambda$ describe?'
    answer: 'The curvature, how quickly the walls get steep'
    wrong: ['The learning rate', 'Where the bottom of the bowl is', 'The loss at the starting point']
    why: 'The gradient is $\lambda\theta$, so a big $\lambda$ means the slope grows fast as you move away from the bottom.'
  - kind: term
    q: 'What is a mini-batch gradient?'
    answer: 'The gradient of the loss on a random sample of $B$ examples'
    wrong: ['The gradient on the $B$ examples with the largest misses', 'The full gradient divided by $B$', 'The full gradient taken with a small learning rate']
    why: 'It''s noisy, but averaged over all the batches you could draw it equals the full gradient, and it costs $B$ examples instead of $N$.'
  - kind: term
    q: 'What is an epoch?'
    answer: 'One pass through the whole dataset in mini-batches'
    wrong: ['One gradient descent step', 'One mini-batch of $B$ examples', 'All the steps until the loss stops falling']
    why: 'With $N$ examples and batches of $B$, an epoch is about $N / B$ steps.'
  - kind: apply
    q: 'What is the mean squared error of the flat line $w = 0$, $b = 3$ on the points $(0, 1)$, $(1, 3)$, $(2, 4)$, $(3, 7)$?'
    answer: '$5.25$'
    wrong: ['$-0.75$', '$21$', '$1.75$']
    why: 'The misses are $2, 0, -1, -4$, their squares add to 21, and $21 / 4 = 5.25$. Averaging the misses without squaring gives $-0.75$.'
  - kind: apply
    q: 'A parameter is at $\theta = 2$, its gradient is 5, and $\eta = 0.1$. Where does one step of gradient descent put it?'
    answer: '$1.5$'
    wrong: ['$2.5$', '$-3$', '$1.9$']
    why: '$2 - 0.1 \times 5 = 1.5$. Adding the gradient instead gives 2.5, which climbs the loss.'
  - kind: apply
    q: 'For the line $w = 2$, $b = 0$ on the four points, the misses are $-1, -1, 0, -1$. What is $\partial L / \partial b$?'
    answer: '$-1.5$'
    wrong: ['$-0.75$', '$1.5$', '$-3$']
    why: '$\frac{2}{N}\sum e_i = \frac{2}{4}(-3) = -1.5$. It''s negative, so raising $b$ lowers the loss.'
  - kind: apply
    q: 'On the bowl $\frac{1}{2}\lambda\theta^2$ with $\lambda = 10$ and $\eta = 0.15$, what does gradient descent do?'
    answer: 'Overshoots back and forth, but closes in on the bottom'
    wrong: ['Creeps toward the bottom without overshooting', 'Lands exactly on the bottom in one step', 'Overshoots by more each step and diverges']
    why: '$\eta\lambda = 1.5$, so each step multiplies $\theta$ by $1 - 1.5 = -0.5$: the sign flips and the distance halves.'
  - kind: apply
    q: 'On the bowl $\frac{1}{2}\lambda\theta^2$ with $\lambda = 5$, above which learning rate does gradient descent diverge?'
    answer: '$\eta = 0.4$'
    wrong: ['$\eta = 0.2$', '$\eta = 2$', '$\eta = 0.1$']
    why: 'It diverges once $\eta\lambda > 2$, so $\eta > 2 / 5 = 0.4$. At $\eta = 0.2$ it lands on the bottom in one step.'
  - kind: apply
    q: 'A stretched bowl has curvature 20 in its steep direction and 1 in its flat direction. Above which learning rate does gradient descent diverge?'
    answer: '$0.1$'
    wrong: ['$2$', '$0.05$', '$0.095$']
    why: 'The steep direction sets the speed limit: $2 / 20 = 0.1$. The flat direction alone would allow up to 2.'
  - kind: apply
    q: 'Your loss gets bigger every step and heads off to infinity. What do you change first?'
    answer: 'Lower the learning rate'
    wrong: ['Raise the learning rate', 'Add the gradient instead of subtracting it', 'Run more steps']
    why: 'Growing overshoots mean $\eta$ times the steepest curvature is above 2. A smaller $\eta$ brings it back under.'
---

Unit 2 ended with the gradient, the direction that makes a function go up fastest.
Training uses it the other way around: score how wrong the model's outputs are with a single number, the loss, and lower that number a little at a time by stepping against the gradient.
Every model in the rest of the route, from the network in unit 8 to the reward model in unit 16, is trained by some version of this loop.

## A loss is one number for how wrong

Take four points, $(0, 1)$, $(1, 3)$, $(2, 4)$ and $(3, 7)$, and try to draw a straight line through them.
A line is $\hat y = wx + b$, where $w$ is the slope, $b$ is where it crosses the y-axis, and $\hat y$ ("y hat") is the line's prediction at $x$.
The two numbers $w$ and $b$ are the model's parameters, and I'll write them together as $\theta = (w, b)$.

No line goes through all four points, so the question is which line misses by the least.
For each point $(x_i, y_i)$ the miss is $\hat y_i - y_i$, the prediction minus the target.
Square each miss and average over the $N$ points:

$$
L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (\hat y_i - y_i)^2 = \frac{1}{N} \sum_{i=1}^{N} (w x_i + b - y_i)^2
$$

::: formula
$\hat y_i$ is the line's prediction at $x_i$, and the sum runs over the points $i = 1$ to $N$.
The second form just writes the prediction out as $w x_i + b$.
:::

This is the *mean squared error*, and $L$ is the *loss*: one number for the whole line, lower is better.

Try $w = 2$, $b = 0$.
The predictions are $0, 2, 4, 6$, the misses are $-1, -1, 0, -1$, and the loss is $(1 + 1 + 0 + 1) / 4 = 0.75$.
The best possible line for these points is $w = 1.9$, $b = 0.9$, with loss $0.175$.
A bad guess like $w = -1$, $b = 3$ (sloping the wrong way) misses by $2, -1, -3, -7$ and scores $63 / 4 = 15.75$.

Squaring does two jobs.
It makes every miss count as positive, so a miss above the line can't cancel one below it.
It also makes big misses cost much more than small ones: missing by 7 costs 49, while missing by 1 costs 1.

## Walk downhill

The loss is a function of $w$ and $b$, so you can picture it as a surface over the $(w, b)$ plane, with height $L$.
Every point on that surface is one line through the data.
Fitting the line means finding the lowest point.

For two parameters you could solve for the bottom with algebra.
A language model has billions of parameters and a loss with no closed-form answer, so instead you start somewhere and repeatedly take a small step downhill.
[Unit 2](../02-gradients/#the-gradient-points-uphill) showed that the gradient $\nabla L$ points in the steepest uphill direction, so the step goes the opposite way:

$$
\theta \leftarrow \theta - \eta \nabla L(\theta)
$$

::: formula
$\eta$ (eta) is a small positive number.
In words: from the current parameters $\theta$, compute the gradient of the loss there, scale it by $\eta$, and subtract it.
:::

The *learning rate* $\eta$ sets how far each step goes, and repeating the update until the loss stops falling is *gradient descent*.

For the line fit, write $e_i = w x_i + b - y_i$ for the miss at point $i$; the [chain rule](../02-gradients/#the-rules-you-need) gives the two partial derivatives:

$$
\frac{\partial L}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} e_i x_i, \qquad \frac{\partial L}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} e_i
$$

::: formula
The chain rule step: $e_i$ changes by $x_i$ when $w$ moves by 1 and by $1$ when $b$ moves by 1.
Each partial averages the misses over the $N$ points and doubles them, the 2 coming from the square.
:::

For $w$ every miss is weighted by its $x_i$, so points far to the right count for more; for $b$ every miss counts the same.

Start from the bad guess $w = -1$, $b = 3$, where the misses are $2, -1, -3, -7$.
Then $\partial L / \partial w = \frac{2}{4}(0 - 1 - 6 - 21) = -14$ and $\partial L / \partial b = \frac{2}{4}(2 - 1 - 3 - 7) = -4.5$.
Both are negative, so raising either parameter lowers the loss.
With $\eta = 0.1$ the step gives $w = -1 + 1.4 = 0.4$ and $b = 3 + 0.45 = 3.45$, and the loss drops from 15.75 to about 3.08.

## The learning rate

$\eta$ is a choice you make, and a bowl with one parameter shows what happens as you change it.
Take $L(\theta) = \frac{1}{2}\lambda\theta^2$, a parabola with its bottom at $\theta = 0$.
The number $\lambda$ is the curvature: how quickly the slope grows as you move away from the bottom, so a big $\lambda$ means steep walls.
The gradient is $\lambda\theta$, and one step of gradient descent gives

$$
\theta \leftarrow \theta - \eta\lambda\theta = (1 - \eta\lambda)\,\theta
$$

Every step multiplies the distance from the bottom by the same factor, $1 - \eta\lambda$.
If $\eta\lambda$ is small, the factor is just under 1 and progress is slow.
At $\eta\lambda = 1$ the factor is 0 and the first step lands exactly on the bottom.
Between 1 and 2 the factor is negative, so each step overshoots to the other side, but by less each time.
Above 2 the factor is below −1, every step overshoots by more than the last, and $\theta$ runs off to infinity.

<figure>
<pre>  ηλ   step 0       1       2       3       4       5
  0.1     1.00    0.90    0.81    0.73    0.66    0.59   slow
<span class="green">  0.9     1.00    0.10    0.01    0.00    0.00    0.00   fast</span>
  1.5     1.00   -0.50    0.25   -0.12    0.06   -0.03   zig-zag
<span class="orange">  2.5     1.00   -1.50    2.25   -3.38    5.06   -7.59   diverges</span></pre>
<figcaption>θ after each step on the bowl ½λθ², starting from θ = 1. Each row multiplies by 1 − ηλ per step.</figcaption>
</figure>

The line-fit loss is a bowl too, stretched so its contour lines are long, thin ellipses.
Along the short axis of the ellipses the walls are steep, with curvature 8.41.
Along the long axis they're nearly flat, with curvature 0.59.
(These are the eigenvalues of the matrix of second derivatives of $L$.)
The steep direction is mostly $w$, because changing the slope moves the prediction at $x = 3$ three times as far as changing $b$ does.

One $\eta$ has to serve both directions at once.
The steep direction sets the speed limit: above $\eta = 2 / 8.41 \approx 0.238$ it diverges, and above $1 / 8.41 \approx 0.119$ it zig-zags.
The flat direction sets how long the trip takes: at $\eta = 0.1$ each step only multiplies its distance by $1 - 0.1 \cdot 0.59 = 0.94$.
So the first step or two remove most of the error along the steep direction.
The rest of the run crawls along the valley floor, with $w$ rising and $b$ falling together.

<figure class="widget" id="descent">
<figcaption>Right: contours of the loss over w (across) and b (up), with the bottom marked by the green cross. The blue path is gradient descent from w = −1, b = 3; the orange dot is the current (w, b), and the orange line on the left is the same (w, b) drawn through the data, with dashed misses. Try the four presets, drag η, or click in the bowl to start somewhere else.</figcaption>
</figure>

The zig-zag preset ($\eta = 0.2$) overshoots on its first step, and the loss goes up, to 8.31 against 3.08 at $\eta = 0.1$.
But its flat-direction factor is $0.88$ instead of $0.94$, and after 10 steps its loss is 0.41 against 1.03.
It gets within 0.0001 of the lowest loss in 41 steps, where $\eta = 0.1$ needs 84.
For a bowl like this the fastest fixed rate is $2 / (8.41 + 0.59) \approx 0.22$, which zig-zags.
The ratio between the two curvatures, here about 14, decides how slow this gets.
The steep direction caps $\eta$ below 0.238, and even right at that cap the flat direction only shrinks by a factor of about 0.86 per step.

## Mini-batches

The gradient above is an average over all $N$ points.
With four points that's free.
A language model trains on billions of tokens, and computing the exact gradient over all of them for every single step would be far too slow.

*Stochastic gradient descent* uses a random sample instead.
Each step picks a *mini-batch* of $B$ examples, computes the loss and its gradient on just those, and takes the step.
The batch gradient is noisy, but averaged over all the batches you could have drawn it equals the full gradient, so the steps still point downhill on average.
The path wobbles where full gradient descent would be smooth, and in exchange each step costs $B$ examples instead of $N$.
One pass through the whole dataset in batches is called an *epoch*.

Large models are trained this way, usually with an optimizer called Adam (or its variant AdamW) instead of the plain update.
Adam steps along a running average of recent gradients.
It also divides each parameter's step by a running estimate of that parameter's typical gradient size, which helps with the steep-versus-flat problem from the last section.
Unit 9 works through both, with the formulas.

## Worked example

```python
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([1.0, 3.0, 4.0, 7.0])

w, b = -1.0, 3.0                     # a bad first guess: the line slopes the wrong way
eta = 0.1
for step in range(201):
    err = w * x + b - y              # prediction minus target, one per point
    loss = np.mean(err ** 2)
    if step in (0, 1, 2, 5, 10, 20, 50, 100, 200):
        print(f"step {step:3d}  w = {w:7.4f}  b = {b:7.4f}  loss = {loss:.4f}")
    grad_w = 2 * np.mean(err * x)
    grad_b = 2 * np.mean(err)
    w -= eta * grad_w
    b -= eta * grad_b

print(np.polyfit(x, y, 1))           # exact least-squares answer: slope, intercept
```

```
step   0  w = -1.0000  b =  3.0000  loss = 15.7500
step   1  w =  0.4000  b =  3.4500  loss = 3.0775
step   2  w =  0.6850  b =  3.3900  loss = 2.4658
step   5  w =  0.9223  b =  2.9870  loss = 1.7549
step  10  w =  1.1806  b =  2.4359  loss = 1.0306
step  20  w =  1.5104  b =  1.7318  loss = 0.4259
step  50  w =  1.8381  b =  1.0321  loss = 0.1813
step 100  w =  1.8971  b =  0.9062  loss = 0.1750
step 200  w =  1.9000  b =  0.9000  loss = 0.1750
[1.9 0.9]
```

The first step takes the loss from 15.75 to 3.08, and the next 49 steps only get it down to 0.1813.
The first step fixes most of the steep direction, and the rest is the walk along the flat one, with $b$ sliding from 3.45 down to 0.9.
`np.polyfit` solves the same problem exactly and agrees: slope 1.9, intercept 0.9.

## Where this shows up later

- Unit 8: backpropagation is how you get $\nabla L$ when the model is a network with many layers. The update step stays the same.
- Unit 9: how noisy mini-batch gradients are, and how momentum, Adam and AdamW change the update step.
- Unit 11: language models use a different loss, cross-entropy, and train it with mini-batches of text.
- Unit 15: reinforcement learning climbs a reward instead of descending a loss, so the step flips sign to $\theta \leftarrow \theta + \eta \nabla J$.
- Unit 16: the reward model is trained by gradient descent on a loss built from people's comparisons.

## Self-test

<details>
<summary>What is the mean squared error of the line w = 1, b = 1 on the four points?</summary>

The predictions are $1, 2, 3, 4$ and the misses are $0, -1, -1, -3$.
The loss is $(0 + 1 + 1 + 9) / 4 = 2.75$.

</details>

<details>
<summary>Why does the update subtract the gradient instead of adding it?</summary>

The gradient points in the direction where the loss rises fastest.
You want the loss to fall, so you step the opposite way.

</details>

<details>
<summary>On the bowl ½λθ² with λ = 4, which learning rates diverge, and which one reaches the bottom in a single step?</summary>

The factor per step is $1 - 4\eta$.
It's below −1 when $\eta > 0.5$, so those diverge.
At $\eta = 0.25$ the factor is 0 and the first step lands on the bottom.

</details>

<details>
<summary>In the widget, the zig-zag preset's loss goes up on the first step, yet it reaches the bottom sooner than the steady preset. Why?</summary>

The first step overshoots in the steep direction, which costs loss right away, but that direction still shrinks by a factor of 0.68 each step.
The slow part of the trip is the flat direction, and there the larger $\eta$ shrinks the distance by 0.88 per step instead of 0.94.

</details>

<details>
<summary>Why do squared misses beat plain misses as a loss?</summary>

Plain misses can be positive or negative, so a line that misses by +3 on one point and −3 on another would score 0.
Squaring makes every miss count and makes big misses cost the most.

</details>

<details>
<summary>Why train on mini-batches instead of computing the exact gradient every step?</summary>

The exact gradient needs a pass over the whole dataset for every step.
A mini-batch gradient is a noisy but unbiased estimate that costs only $B$ examples, so you get many more steps for the same compute.

</details>

::: done-when
- You can compute the mean squared error of a given line on a few points by hand.
- You can write the update θ ← θ − η∇L and compute one step of it for the line fit.
- You can say what happens on the bowl ½λθ² for ηλ below 1, between 1 and 2, and above 2.
- You can explain from the widget why a stretched bowl makes gradient descent slow.
:::

## Go deeper

- [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Gabriel Goh, Distill). Interactive, and the first half is this unit's bowl analysis done properly, eigenvalues and all.
- 3Blue1Brown's [gradient descent lesson](https://www.3blue1brown.com/lessons/gradient-descent): the same idea applied to a neural network's loss, a good bridge to unit 8.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma and Ba). Read algorithm 1 on page 2 to see how little it adds to the plain update.
