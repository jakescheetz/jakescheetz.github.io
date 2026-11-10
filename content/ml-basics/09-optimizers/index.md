---
title: Optimizers in practice
subtitle: Momentum, Adam, weight decay and learning-rate schedules, each a small change to the gradient descent step.
unit: 9
section: Deep learning
minutes: 50
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'In momentum, what does $\beta$ control?'
    answer: 'How much of the previous velocity carries over into each step'
    wrong: ['How far each step moves for a given gradient', 'How many examples go into each mini-batch', 'How quickly the learning rate falls over training']
    why: 'Each step keeps $\beta v$ and adds the new gradient, so $\beta$ sets how long old gradients keep counting. With $\beta = 0$ it''s plain gradient descent.'
  - kind: term
    q: 'In Adam, what does $s$ keep track of?'
    answer: 'A running average of each parameter''s squared gradient'
    wrong: ['A running average of each parameter''s gradient', 'The sum of every gradient seen so far', 'The squared length of the whole parameter vector']
    why: '$s \leftarrow \beta_2 s + (1 - \beta_2) g^2$, entry by entry, so $\sqrt{\hat s}$ is the typical size of that parameter''s gradient. The running average of $g$ itself is $m$.'
  - kind: term
    q: 'What problem does Adam''s bias correction fix?'
    answer: '$m$ and $s$ start at zero, so their early values are too small'
    wrong: ['Mini-batch gradients point the wrong way on average', 'Parameters with large gradients take steps that are too big', 'The learning rate is too high in the first steps']
    why: 'After one step $m = 0.1g$ and $s = 0.001g^2$. Dividing by $1 - \beta_1^t$ and $1 - \beta_2^t$ scales them back up to $g$ and $g^2$.'
  - kind: term
    q: 'What is learning-rate warmup?'
    answer: 'Raising $\eta$ from near zero to its peak over the first steps'
    wrong: ['Lowering $\eta$ smoothly to its minimum over the last steps', 'Running a few steps with $\eta = 0$ to fill Adam''s averages', 'Training on small batches first and large ones later']
    why: 'Warmup is the rising part at the start (LLaMA used 2,000 steps). The smooth fall at the end is the decay, often a cosine.'
  - kind: term
    q: 'What does AdamW change compared with Adam plus an L2 penalty in the loss?'
    answer: 'The weight decay is applied outside Adam''s division by $\sqrt{\hat s}$'
    wrong: ['The weight decay is divided by $\sqrt{\hat s}$ along with the gradient', 'It drops the running average of squared gradients', 'It applies weight decay only to the biases']
    why: 'With the penalty in the loss, $\lambda\theta$ gets divided by $\sqrt{\hat s}$ and weights with big gradients barely decay. AdamW shrinks every weight by the same factor $1 - \eta\lambda$.'
  - kind: term
    q: 'What does a cosine decay schedule do to $\eta$ after warmup?'
    answer: 'Lowers it from its peak to a minimum, slowly at first and at the end and fastest in the middle'
    wrong: ['Raises it from zero to its peak in a straight line', 'Halves it every fixed number of steps', 'Lowers it from its peak to a minimum at a constant rate']
    why: 'It follows half a cosine wave from $\eta_{\max}$ down to $\eta_{\min}$, which is flat at both ends and steepest halfway.'
  - kind: apply
    q: 'With momentum $\beta = 0.5$ and a gradient that stays at $g$ for many steps, what does the velocity $v$ approach?'
    answer: '$2g$'
    wrong: ['$0.5g$', '$g$', '$10g$']
    why: '$v = g + 0.5g + 0.25g + \cdots = g / (1 - 0.5) = 2g$. The $10g$ is what $\beta = 0.9$ gives.'
  - kind: apply
    q: 'Adam''s first step sees the gradient $(-140, -4.5)$, with $\eta = 0.01$. How far does each parameter move?'
    answer: 'Both move by 0.01'
    wrong: ['$w$ by 1.4 and $b$ by 0.045', 'Both move by about 0.0316', '$w$ by 0.01 and $b$ by about 0.0003']
    why: 'After bias correction $\hat m = g$ and $\hat s = g^2$, so the step is $\eta$ times the sign of each entry. 0.0316 is what you''d get without the correction.'
  - kind: apply
    q: 'Along one direction the gradient flips sign every step. With momentum $\beta = 0.5$, how big does the step along that direction get, compared with plain gradient descent?'
    answer: 'About two-thirds as big'
    wrong: ['About twice as big', 'About half as big', 'The same size']
    why: 'The terms alternate, $g - 0.5g + 0.25g - \cdots = g / 1.5$, so the zig-zag is damped to two-thirds.'
  - kind: apply
    q: 'You train with batch size 64 and want half the gradient noise. What batch size do you need?'
    answer: '256'
    wrong: ['128', '32', '4,096']
    why: 'The noise of an average shrinks like $1/\sqrt{B}$, so halving it takes 4 times the batch.'
  - kind: apply
    q: 'AdamW with $\eta = 0.001$ and $\lambda = 0.1$: by what factor does each step shrink every weight, apart from the Adam step?'
    answer: '0.9999'
    wrong: ['0.9', '0.999', '0.99']
    why: 'The factor is $1 - \eta\lambda = 1 - 0.0001$.'
  - kind: apply
    q: 'Cosine decay from $\eta_{\max} = 0.1$ to $\eta_{\min} = 0$ over 200 steps, no warmup. What is $\eta$ at step 100?'
    answer: '0.05'
    wrong: ['0.1', '0', '0.025']
    why: 'Halfway, $\cos(\pi/2) = 0$, so $\eta = \frac{1}{2} \cdot 0.1 \cdot (1 + 0) = 0.05$.'
---

::: why
[Unit 4](../04-gradient-descent/#walk-downhill) fit a line to four points with plain gradient descent.
It took 84 steps, and most of them were spent crawling along the flat floor of a stretched bowl.
Large networks are rarely trained with that plain update, $\theta \leftarrow \theta - \eta \nabla L(\theta)$.
They use Adam or AdamW, on mini-batches, with a learning rate that rises and then falls over the run.
Each of those is a few extra lines around the plain update.
I'll use unit 4's four points and bowl to show what each one does.
:::

## Mini-batch gradients are noisy

The gradient in [unit 4's update](../04-gradient-descent/#walk-downhill) was an average over all four points.
A [mini-batch](../04-gradient-descent/#mini-batches) averages over a random few instead.
A batch of one uses a single point's gradient.

For one point $(x_i, y_i)$ the miss is $e_i = w x_i + b - y_i$.
The gradient of that point's squared miss $e_i^2$ is

$$
2 e_i\,(x_i,\ 1)
$$

::: formula
The first entry is $\partial / \partial w$ and the second is $\partial / \partial b$.
The miss $e_i$ says how wrong the line is at that point, $x_i$ and $1$ are how much a nudge to $w$ or $b$ moves the prediction there, and the 2 comes from squaring the miss.
:::

Take the starting line from unit 4, $w = -1$ and $b = 3$.
The four single-point gradients are $(0, 4)$, $(-2, -2)$, $(-12, -6)$ and $(-42, -14)$.
Their average is the full gradient, $(-14, -4.5)$.

The four don't agree with each other.
The first point says lower $b$, while the full gradient says raise it.
The last one points roughly the right way but is three times as long.
So a single step can go the wrong way.
Averaged over which point you drew, though, it's the full-gradient step.
Early in training the full gradient is big, and that's enough to make progress.

Near the bottom the noise causes trouble.
At the best line, $w = 1.9$ and $b = 0.9$, the full gradient is zero.
But the single-point gradients are $(0, -0.2)$, $(-0.4, -0.4)$, $(2.8, 1.4)$ and $(-2.4, -0.8)$.
Each step still gets a kick of up to about 3, times $\eta$.

<figure class="widget demo" id="demo-noise"></figure>

With a fixed $\eta$ the parameters never settle at the bottom.
They keep bouncing around it, and a bigger $\eta$ means bigger bounces.

A bigger batch averages more of those kicks together.
[Unit 3](../03-probability/#expectation-spread-and-sampling) showed that an average of $N$ samples misses its expected value by about $\sigma / \sqrt{N}$, where $\sigma$ is the standard deviation of one sample.
A batch gradient is an average of $B$ single-example gradients, so its noise shrinks like $1 / \sqrt{B}$.
Four times the batch buys half the noise, at four times the compute per step.

A small batch gives many cheap, noisy steps.
A large batch gives fewer, cleaner ones.
Past some size the cleaner gradient stops helping much.
The direction is already accurate, and the step is limited by $\eta$ and the shape of the bowl.
[McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) measured where that happens: the largest useful batch differs a lot between tasks, and a statistic of the gradient's noise predicts it.

Below that size, a GPU works through a batch's examples in parallel, so a bigger batch often costs little extra time per step.
That's why large models use big batches: [LLaMA](https://arxiv.org/abs/2302.13971)'s were 4 million tokens per step.

## Momentum

[Unit 4's stretched bowl](../04-gradient-descent/#the-learning-rate) had two problems at once.
The steep direction zig-zags, and the flat direction crawls.

Think of a heavy ball rolling down the bowl.
It doesn't turn the instant the slope changes, because it carries speed from the slopes it already rolled over.
*Momentum* gives gradient descent that memory.
It keeps a running total of past gradients, called the *velocity* $v$, and steps along that instead of the latest gradient alone:

$$
v \leftarrow \beta v + \nabla L(\theta), \qquad \theta \leftarrow \theta - \eta v
$$

::: formula
$v$ has one entry per parameter and starts at zero.
Each step multiplies the old velocity by $\beta$, a number between 0 and 1, and adds the new gradient $\nabla L(\theta)$; then $\theta$ moves against $v$, scaled by the learning rate $\eta$ as before.
:::

$\beta$ is the fraction of the old velocity each step keeps.
It's usually 0.9.
With $\beta = 0$ the velocity is just the gradient, and this is plain gradient descent.

Unroll the first update, writing $g_t$ for the gradient at step $t$.
This is what the velocity holds after step $t$:

$$
v_t = g_t + \beta\, g_{t-1} + \beta^2 g_{t-2} + \cdots
$$

It's every gradient so far, with one from $k$ steps ago weighted by $\beta^k$.
With $\beta = 0.9$, a gradient from 10 steps ago still counts about a third ($0.9^{10} \approx 0.35$).
One from 50 steps ago counts for almost nothing ($0.005$).

Two cases cover the bowl.

- **The gradient keeps pointing the same way.** The sum is a geometric series, and $v$ grows to $g / (1 - \beta)$. The step is $1 / (1 - \beta)$ times bigger: 2 times for $\beta = 0.5$, 10 times for $\beta = 0.9$.
- **The gradient flips sign every step.** The terms alternate and partly cancel, and $v$ settles at $g / (1 + \beta)$, smaller than a single gradient.

<figure class="widget demo" id="demo-momentum"></figure>

On the stretched bowl the flat direction is the steady case, so the crawl along the floor gets the bigger step.
In the steep direction a step that big overshoots and flips the gradient's sign.
That's the cancelling case, so the zig-zag across the valley stays under control.
With $\eta = 0.1$ and $\beta = 0.5$ momentum gets within 0.0001 of the lowest loss in 36 steps, where plain gradient descent needs 84.

More momentum isn't automatically better.
At the same $\eta$, $\beta = 0.9$ takes 101 steps.
Ten times the step along the floor carries it past the bottom, and it swings back and forth across it several times before settling.
With $\beta = 0.9$ you'd normally lower $\eta$.
At $\eta = 0.05$ it takes 67 steps, against 171 for plain gradient descent at that rate.

## Adam

Momentum still uses one $\eta$ for every parameter, and the steep direction decides how big it can be.
On unit 4's line the starting gradient is $(-14, -4.5)$, three times bigger for $w$ than for $b$.
An $\eta$ that's safe for the parameters with big gradients is slow for the rest.

*Adam* gives each parameter its own scale.
It keeps two running averages per parameter: $m$ of its gradient $g$, and $s$ of the squared gradient:

$$
m \leftarrow \beta_1 m + (1 - \beta_1)\, g, \qquad s \leftarrow \beta_2 s + (1 - \beta_2)\, g^2
$$

::: formula
Everything here works on each parameter separately, and $g^2$ squares each entry of the gradient.
$m$ is like momentum's velocity, except the weights add up to 1, so it's an average of recent gradients instead of a sum.
With $\beta_1 = 0.9$ it mostly reflects the last 10 or so steps.
$s$ averages the squared gradients, so $\sqrt{s}$ is the typical size of that parameter's gradient regardless of sign.
With $\beta_2 = 0.999$ it reflects roughly the last thousand steps.
Both start at zero.
:::

The step divides one by the other:

$$
\hat m = \frac{m}{1 - \beta_1^t}, \qquad \hat s = \frac{s}{1 - \beta_2^t}, \qquad \theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat s} + \varepsilon}
$$

::: formula
$t$ is the step number, starting at 1, and $\beta_1^t$ is $\beta_1$ to the power $t$.
$\hat m$ and $\hat s$ are the averages after bias correction, explained below.
The Adam paper writes $\alpha$ for $\eta$ and $v$ for $s$.
:::

The fraction is the average gradient divided by the typical gradient size, entry by entry.
The tiny $\varepsilon$ underneath only keeps it from dividing by zero.

When a parameter's gradient keeps pointing the same way, the fraction is close to $\pm 1$.
When the gradient flips or is noisy, the fraction is closer to 0.
So each parameter moves about $\eta$ per step at most, however large or small its gradients are.
The Adam paper calls this being "invariant to diagonal rescaling of the gradients".
Multiply one parameter's gradients by any positive constant and its steps don't change (ignoring $\varepsilon$).
The paper's defaults are $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ and $\varepsilon = 10^{-8}$.

The hats on $\hat m$ and $\hat s$ are *bias correction*.
It's there because $m$ and $s$ start at zero.
After the first step $m = 0.1g$ and $s = 0.001g^2$, both far too small.
Uncorrected, the step would be $\eta \cdot 0.1g / \sqrt{0.001g^2} \approx 3.16\eta$.
Dividing by $1 - 0.9 = 0.1$ and $1 - 0.999 = 0.001$ gives back $\hat m = g$ and $\hat s = g^2$.
So the first step is exactly $\eta$ in each entry, in the direction against the gradient.

The correction fades as $\beta^t$ shrinks with the step number $t$, but slowly for $s$.
After 100 steps, $1 - 0.999^{100} \approx 0.095$, so $s$ is still being multiplied by about 10.5.

Back on unit 4's line, the starting gradient is $(-14, -4.5)$.
Plain gradient descent's first step at $\eta = 0.1$ moves $w$ by 1.4 and $b$ by 0.45.
Adam's moves both by exactly 0.1.

<figure class="widget demo" id="demo-adam"></figure>

That evenness pays off when the scales are far apart.
Measure $x$ in different units, so the points sit at $x = 0, 10, 20, 30$.
The best line is the same line in the new units (slope 0.19, intercept 0.9, lowest loss 0.175).
But the steep curvature jumps to about 701 while the flat one is 0.71, a ratio of 984 instead of 14.

Gradient descent has to keep $\eta$ below $2 / 701 \approx 0.00285$.
At $\eta = 0.0025$ it needs 2,722 steps, and momentum needs 1,356.
Adam, with $\eta$ still 0.1, needs 358, and its first step is still $(0.1, 0.1)$.

On the original points, though, Adam takes 117 steps, more than plain gradient descent's 84.
Its per-parameter scaling can only undo stretch that lines up with the parameter axes.
Multiplying every $x$ by 10 stretched the bowl almost purely along $w$ (the steep direction is within 2.5° of the $w$ axis).
If the bowl is stretched along a diagonal, both parameters see the same mix of big and small gradients.
No per-parameter division can separate them.

The widget races all three on a bowl you can reshape.

<figure class="widget" id="optimizer-race">
<figcaption>Gradient descent (GD, blue), momentum (orange) and Adam (purple) start from w = −1, b = 3 on a bowl with its bottom at the green cross. One η drives all three, and β is momentum's. Stretch and tilt reshape the bowl (unit 4's bowl is 14× and 25°). Try the three bowl buttons, or click anywhere to start from there.</figcaption>
</figure>

At the starting settings the widget reproduces the 36, 84 and 117 steps above.
Press "1000× along w" for a bowl shaped like the $x \times 10$ case.
After 600 steps gradient descent and momentum are still 0.0068 and 0.0025 above the lowest loss, and Adam got there in 77 steps.
"1000× tilted 45°" keeps the same stretch and only turns it.
Now Adam doesn't get there in 600 steps either.

## Weight decay and AdamW

[Unit 6](../06-generalization/#weight-decay-early-stopping-and-more-data) kept weights small by adding a penalty $\frac{\lambda}{2}\|\theta\|^2$ to the loss.
$\|\theta\|^2$ is the sum of the squared parameters, and $\lambda$ sets how much they cost.
The penalty's gradient is $\lambda\theta$.
In plain gradient descent the update becomes

$$
\theta \leftarrow \theta - \eta\,(\nabla L + \lambda\theta) = (1 - \eta\lambda)\,\theta - \eta \nabla L
$$

The left form is the usual step on the loss plus penalty.
The right form regroups it: shrink every weight by the factor $1 - \eta\lambda$, then take the normal step on the loss alone.
That steady shrinking is why it's called *weight decay*.

In Adam the two forms stop being the same.
If you add $\lambda\theta$ to the gradient, it flows into $m$ and $s$ and gets divided by $\sqrt{\hat s}$ along with everything else.
A weight whose gradients are usually large has a large $\hat s$, so its decay gets divided down a lot.
A weight with small gradients gets decayed hard.

Loshchilov and Hutter pointed this out and proposed *AdamW*, which takes the decay back out of the gradient:

$$
\theta \leftarrow \theta - \eta \left( \frac{\hat m}{\sqrt{\hat s} + \varepsilon} + \lambda\theta \right)
$$

The fraction is Adam's step from above, and here $m$ and $s$ only see the loss's gradient.
The $\lambda\theta$ sits outside the division.
So every weight shrinks by the same factor $1 - \eta\lambda$ per step, as it does in plain gradient descent.
This is the form PyTorch's `AdamW` uses.

Take $\eta = 3 \times 10^{-4}$ and $\lambda = 0.1$ (LLaMA's peak learning rate for its smaller models, and its weight decay).
The factor is $1 - 0.00003$.
10,000 steps of that alone would shrink a weight to 0.74 of its size.

## Learning-rate schedules

A *schedule* lets $\eta$ change with the step number, written $\eta_t$.
Most language model runs use two pieces.

The first is *warmup*.
$\eta$ climbs in a straight line from near zero to its peak $\eta_{\max}$ over the first $W$ steps:

$$
\eta_t = \eta_{\max}\, \frac{t}{W} \qquad (t \le W)
$$

At step $t$ you're the fraction $t / W$ of the way through warmup, and you get that fraction of the peak.

Warmup is there because the start of training is rough.
The weights are random, the gradients are large and change quickly, and Adam's $\hat s$ is an average over only a few steps.
[Liu et al. (2019)](https://arxiv.org/abs/1908.03265) argue that this last part is what warmup fixes.
With so few gradients behind it, $\sqrt{\hat s}$ is a shaky estimate, so the step size $\eta / \sqrt{\hat s}$ swings a lot from step to step.
Small steps until the averages settle keep that from pushing training somewhere bad early on.

After warmup, $\eta$ comes down again.
A common shape is *cosine decay* over the remaining steps, down to a floor $\eta_{\min}$ at the last step $T$:

$$
\eta_t = \eta_{\min} + \frac{1}{2}\,(\eta_{\max} - \eta_{\min}) \left( 1 + \cos\!\left( \pi\, \frac{t - W}{T - W} \right) \right)
$$

::: formula
The fraction $(t - W) / (T - W)$ runs from 0 right after warmup to 1 at the end.
Over that range the cosine goes from 1 to −1, so the bracket goes from 2 to 0 and $\eta_t$ goes from $\eta_{\max}$ to $\eta_{\min}$.
:::

It falls slowly at first, fastest in the middle, and levels off at the end.
LLaMA used AdamW with $\beta_1 = 0.9$ and $\beta_2 = 0.95$, 2,000 warmup steps, and cosine decay to 10% of the peak.

<figure class="widget demo" id="demo-schedule"></figure>

The decay is the fix for the bouncing from the first section.
The bounce grows with $\eta$, so lowering $\eta$ toward the end lets the parameters settle into the bottom.
The widget shows that on unit 4's four points.

<figure class="widget" id="sgd-schedule">
<figcaption>Left: 200 steps of mini-batch gradient descent on unit 4's four points, with each batch drawn at random. Right: the learning rate over the run (purple), and where the run is now (orange). The readout averages the last ten runs' final loss, so you can compare settings over several runs.</figcaption>
</figure>

I ran the widget's setup 1,000 times with a batch of one point and a peak $\eta$ of 0.05.
With a constant rate the runs ended on average 0.085 above the lowest loss, and with cosine decay 0.012.

With all four points in the batch there's no noise, and the comparison flips.
The constant rate reaches the bottom in 171 steps.
Cosine decay ends 0.007 above it, because it slowed down before arriving.

Warmup barely changes this bowl's numbers (0.012 either way).
A two-parameter bowl has none of the random weights or short $\hat s$ history that warmup is meant for.

## Worked example

The three update rules from this unit, raced on unit 4's points and on the same points with every $x$ multiplied by 10:

```python
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([1.0, 3.0, 4.0, 7.0])
lowest = 0.175                     # the best line's loss, from unit 4

def loss_and_grad(theta, xs):
    err = theta[0] * xs + theta[1] - y
    return np.mean(err ** 2), np.array([2 * np.mean(err * xs), 2 * np.mean(err)])

def gd(g, state, t, eta):
    return eta * g

def momentum(g, state, t, eta, beta=0.5):
    state["v"] = beta * state.get("v", 0) + g             # decaying sum of past gradients
    return eta * state["v"]

def adam(g, state, t, eta, b1=0.9, b2=0.999, eps=1e-8):
    state["m"] = b1 * state.get("m", 0) + (1 - b1) * g        # running average of g
    state["s"] = b2 * state.get("s", 0) + (1 - b2) * g ** 2   # running average of g squared
    m_hat = state["m"] / (1 - b1 ** t)                        # bias correction
    s_hat = state["s"] / (1 - b2 ** t)
    return eta * m_hat / (np.sqrt(s_hat) + eps)

def steps_to_bottom(update, xs, eta):
    theta, state = np.array([-1.0, 3.0]), {}
    for t in range(1, 10_001):
        loss, g = loss_and_grad(theta, xs)
        if loss - lowest < 1e-4:
            return t - 1
        theta = theta - update(g, state, t, eta)

for scale, eta in [(1, 0.1), (10, 0.0025)]:
    xs = scale * x
    g = loss_and_grad(np.array([-1.0, 3.0]), xs)[1]
    print(f"x times {scale}: gradient at the start {g}")
    print(f"  first step:  gradient descent {-gd(g, {}, 1, eta)}   Adam {-adam(g, {}, 1, 0.1)}")
    print(f"  steps:       gradient descent {steps_to_bottom(gd, xs, eta)}"
          f"   momentum {steps_to_bottom(momentum, xs, eta)}"
          f"   Adam {steps_to_bottom(adam, xs, 0.1)}")
```

```
x times 1: gradient at the start [-14.   -4.5]
  first step:  gradient descent [1.4  0.45]   Adam [0.1 0.1]
  steps:       gradient descent 84   momentum 36   Adam 117
x times 10: gradient at the start [-770.   -31.5]
  first step:  gradient descent [1.925   0.07875]   Adam [0.1 0.1]
  steps:       gradient descent 2722   momentum 1356   Adam 358
```

Adam's first step is $(0.1, 0.1)$ both times, even though the gradient on the second problem is 55 times bigger in $w$.
Gradient descent's step follows the gradient's size.
On the second problem it had to drop $\eta$ by a factor of 40, and it slowed down by a factor of 32.
Adam used the same $\eta$ for both and slowed down by a factor of 3.

::: real-world
- **The default optimizer.** `torch.optim.AdamW` with warmup and a decaying learning rate is the usual starting point for training or fine-tuning a transformer.
- **Training recipes in papers.** Model papers list the batch size, peak learning rate, warmup steps, $\beta$ values and weight decay. LLaMA's line reads AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1 and 2,000 warmup steps.
- **Adversarial attacks.** An attack that searches for an input to fool a model runs an optimizer too. It takes repeated gradient steps on the input instead of the weights, and some attacks add momentum.
:::

## Where this shows up later

- Unit 11: the cross-entropy loss there is what real language model runs minimize with AdamW, warmup and cosine decay, on batches of millions of tokens.
- Unit 13: a transformer's embedding table, attention matrices and layer-norm parameters are all updated by one AdamW optimizer, each parameter with its own $\hat s$.
- Unit 15: reinforcement learning estimates its gradient from sampled episodes, and those estimates are much noisier than a mini-batch gradient on labeled data.
- Unit 16: the reward model and the RLHF policy are trained with these same optimizers; only the loss changes.

::: review
Mini-batch gradient
:   The gradient averaged over a random few examples. It's a noisy estimate of the full gradient, and the noise shrinks like $1 / \sqrt{B}$ for batch size $B$.

Momentum
:   Stepping along a decaying sum of past gradients. Steady gradients add up and flipping ones cancel.

Velocity, $v$
:   Momentum's running total of gradients.

$\beta$
:   The fraction of the old velocity (or old average) each step keeps. Usually 0.9 for momentum.

Adam
:   An optimizer that divides each parameter's average gradient $m$ by its typical gradient size $\sqrt{s}$, so every parameter moves about $\eta$ per step at most.

Bias correction
:   Dividing $m$ and $s$ by $1 - \beta^t$ to make up for starting them at zero.

Weight decay
:   Shrinking every weight by the factor $1 - \eta\lambda$ each step.

AdamW
:   Adam with weight decay applied outside the division by $\sqrt{\hat s}$, so every weight shrinks by the same factor.

Warmup
:   Raising $\eta$ in a straight line from near zero to its peak over the first $W$ steps.

Cosine decay
:   Lowering $\eta$ from its peak to a floor along half a cosine wave.
:::

::: say
A mini-batch gradient is a noisy copy of the full gradient, and that noise keeps the parameters bouncing near the bottom.
Momentum remembers past gradients, so it speeds up where they agree and slows down where they flip.
Adam divides each parameter's gradient by its usual size, so every parameter takes steps of about the same length.
AdamW adds weight decay that shrinks every weight by the same factor.
A schedule warms the learning rate up at the start and decays it at the end so the run can settle.
:::

## Self-test

<details>
<summary>Momentum has β = 0.9 and the gradient stays at 2 for a long time. What does the velocity approach, and how far does each step go with η = 0.01?</summary>

The velocity approaches $2 / (1 - 0.9) = 20$.
Each step moves $\eta v = 0.01 \cdot 20 = 0.2$, ten times what plain gradient descent would.

</details>

<details>
<summary>Why does Adam need bias correction, and why does it matter more for s than for m?</summary>

Both averages start at zero, so for the first steps they're mostly zero and far too small.
The correction divides by $1 - \beta^t$, the share of the average that real gradients have filled so far.
With $\beta_2 = 0.999$, $s$ fills up slowly: after 100 steps only about 9.5% of it is real gradients, while $m$ with $\beta_1 = 0.9$ is nearly all real gradients by then.

</details>

<details>
<summary>In the race widget, "1000× along w" lets Adam reach the bottom in 77 steps, but "1000× tilted 45°" doesn't. Why does turning the bowl matter to Adam and not to gradient descent?</summary>

Adam divides each parameter's step by that parameter's own typical gradient size.
When the steep direction is the $w$ axis, $w$ has big gradients and $b$ small ones, and the division evens them out.
Tilted 45°, both parameters get a mix of the steep and flat directions, so their gradient sizes look alike and dividing them separately can't undo the stretch.
Gradient descent and momentum only see the two curvatures, not which way they point.
(Their numbers in the widget still change a little when you turn the bowl, because the start point doesn't turn with it.)

</details>

<details>
<summary>Why does mini-batch gradient descent with a constant learning rate keep bouncing near the bottom, and what are two ways to calm it?</summary>

At the bottom the full gradient is zero, but each batch's gradient isn't, so every step still gets a random kick of size about $\eta$ times the batch gradient.
A bigger batch makes the kicks smaller (the noise shrinks like $1/\sqrt{B}$), and a decaying learning rate makes each kick count for less near the end.

</details>

<details>
<summary>With Adam plus an L2 penalty in the loss, which weights get decayed the least, and how does AdamW change that?</summary>

The weights whose gradients are usually large: their $\hat s$ is large, and the penalty's gradient gets divided by $\sqrt{\hat s}$ along with the rest.
AdamW applies $\lambda\theta$ outside that division, so every weight shrinks by the same $1 - \eta\lambda$ each step.

</details>

::: done-when
- You can write the momentum update and say what $v$ grows to when the gradient is steady and when it flips sign.
- You can compute Adam's first step for a given gradient, with and without bias correction.
- You can explain with the race widget why Adam handles a stretch along the parameter axes but not a diagonal one.
- You can sketch warmup plus cosine decay and say which problem each part addresses.
:::

## Go deeper

- [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Gabriel Goh, Distill). Its section "The Dynamics of Momentum" does this unit's momentum analysis exactly, and the "Optimal parameters" part gives the best $\beta$ for a bowl with a given stretch.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma and Ba). Section 2.1 explains why the step is about $\eta$ in size, and section 3 derives the bias correction.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov and Hutter). Section 2 shows why L2 and weight decay are the same for gradient descent and different for Adam.
