---
title: Derivatives and gradients
subtitle: How much the output moves when you nudge an input, and which way is uphill.
unit: 2
minutes: 50
scripts: [widgets.js]
---

Training a model means changing its parameters so that one number, the loss, goes down.
To do that you need to know, for each parameter, which way to nudge it and how much the loss will respond.
A derivative answers that for one input, and a gradient answers it for all the inputs at once.
Unit 3 uses the gradient to walk downhill, and unit 4 computes it for a whole network.

## The slope at a point

Take $f(x) = x^2$ and stand at $x = 3$, where $f(3) = 9$.
Step a little to the right, to $3 + h$, and measure how much $f$ changed per unit of step:

$$
\frac{f(3 + h) - f(3)}{h}
$$

With $h = 1$ that's $(16 - 9)/1 = 7$.
With $h = 0.1$ it's $(9.61 - 9)/0.1 = 6.1$, and with $h = 0.01$ it's $6.01$.
As the step shrinks, the answer settles on 6.

Each of those numbers is the slope of a line through two points on the curve, called a secant line.
As $h$ shrinks, the second point slides onto the first, and the secant turns into the tangent: the line that just touches the curve at $x = 3$.
The tangent's slope is the derivative, written $f'(x)$ or $\frac{df}{dx}$:

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

For $x^2$ you can see where the 6 comes from by expanding.
$(x + h)^2 - x^2 = 2xh + h^2$, so the ratio is exactly $2x + h$, and letting $h$ go to zero leaves $f'(x) = 2x$.

In practice I use a derivative as an exchange rate.
Nudge the input by a small $h$, and the output moves by about $f'(x)$ times the nudge.

$$
f(x + h) \approx f(x) + f'(x)\,h
$$

At $x = 3$, a nudge of 0.01 should move $x^2$ by about $6 \times 0.01 = 0.06$, and the exact change is 0.0601.
The sign tells you which way $f$ moves when $x$ increases.
At $x = -2$ the slope is $-4$, so increasing $x$ makes $f$ go down.
At $x = 0$ the slope is 0, which is what the bottom of a valley looks like.

<figure class="widget" id="tangent">
<figcaption>The curve is f(x) = x². The blue line is the tangent at x, with slope 2x. The orange line goes through the points at x and x + h, and its slope is the dashed rise divided by the dashed run h. Shrink h and the orange slope closes in on the blue one. Press play to slide x along the curve and watch the blue slope change sign at the bottom.</figcaption>
</figure>

## The rules you need

You rarely need the limit itself, because a few rules cover nearly everything in this route.

The power rule: the derivative of $x^n$ is $n x^{n-1}$.
So $x^2$ gives $2x$, $x^3$ gives $3x^2$, and $x$ gives 1.
A constant on its own has derivative 0, since it doesn't move when $x$ does.
Constant factors stay in front and sums split up, so $x^3 + 5x$ has derivative $3x^2 + 5$.

The chain rule handles one function fed into another.
Take a model with a single weight $w$ that predicts $2w$ for an input of 2.
The right answer is 3, and the loss is the squared error

$$
L(w) = (2w - 3)^2
$$

Split it into two steps: first $u = 2w - 3$, then $L = u^2$.
Each step has its own exchange rate.
$u$ moves 2 times as much as $w$, so $\frac{du}{dw} = 2$.
$L$ moves $2u$ times as much as $u$, so $\frac{dL}{du} = 2u$.
A nudge to $w$ passes through both steps, so the two rates multiply:

$$
\frac{dL}{dw} = \frac{dL}{du} \cdot \frac{du}{dw} = 2u \cdot 2
$$

At $w = 1$ we get $u = -1$ and $\frac{dL}{dw} = -4$.
To check it, move $w$ from 1 to 1.01.
Then $u$ goes from $-1$ to $-0.98$, a change of 0.02.
$L$ goes from 1 to 0.9604, a change of $-0.0396$, which is close to $-4 \times 0.01$.
The sign is negative because $w = 1$ is below the best weight, 1.5, so increasing $w$ lowers the loss.

A neural network is the same thing with many more steps.
Each layer's output feeds the next, and the loss sits at the end.
The effect of one weight on the loss is the local rates along the way, multiplied together (and when a value feeds into several later steps, the contributions from each route add up).
Unit 4 computes these products efficiently by starting at the loss and working backward, which is what backpropagation means.

## More than one input

A loss depends on many parameters at once, so we need derivatives for functions of several inputs.
The example for the rest of this unit is

$$
f(x, y) = x^2 + 3y^2
$$

Its graph is a bowl with its lowest point at $(0, 0)$, stretched so the sides are steeper in the $y$ direction than in $x$.

A partial derivative changes one input and holds the others fixed.
For $\frac{\partial f}{\partial x}$, treat $y$ as a constant, so $3y^2$ contributes nothing and $\frac{\partial f}{\partial x} = 2x$.
In the same way, $\frac{\partial f}{\partial y} = 6y$.
The curly $\partial$ is a reminder that there are other inputs and they're being held still.

The gradient collects all the partial derivatives into one vector:

$$
\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right) = (2x, 6y)
$$

At the point $(1, 1)$ that's $(2, 6)$.
For a model with parameters $\theta$ and loss $L$, the gradient $\nabla L$ has one entry per parameter, so a model with a billion parameters has a gradient with a billion entries.

## The gradient points uphill

The exchange-rate picture works with several inputs too.
Start at a point $v$ and take a small step $s = (s_1, s_2)$.
The $x$ part of the step changes $f$ by about $\frac{\partial f}{\partial x} s_1$, the $y$ part changes it by about $\frac{\partial f}{\partial y} s_2$, and for small steps the two changes add:

$$
f(v + s) \approx f(v) + \nabla f \cdot s
$$

The $\nabla f \cdot s$ term is the dot product from unit 1.

Fix the step's length at 1 and only choose its direction.
Unit 1 says $\nabla f \cdot s = \|\nabla f\|\,\|s\|\cos\varphi$, where $\varphi$ is the angle between the step and the gradient.
This is largest when $\varphi = 0$.
So stepping in the gradient's direction raises $f$ faster than any other direction, and the rate you get is $\|\nabla f\|$, the gradient's length.
Stepping the opposite way, along $-\nabla f$, lowers $f$ the fastest, and that's the direction unit 3 steps in.

At $\varphi = 90°$ the change is zero, so $f$ stays the same to first order.
The directions where $f$ stays the same run along the contour line through the point, the curve where $f$ has the same value everywhere (like the lines on a hiking map).
So the gradient is always perpendicular to the contour line it starts from.
At $(1, 1)$ the gradient $(2, 6)$ has length $\sqrt{40} \approx 6.32$, so that's the steepest rate available there.

<figure class="widget" id="gradient">
<figcaption>Contour lines of f(x, y) = x² + 3y². The orange ellipse is the contour through the point, and the blue arrow is the gradient there, drawn at a fifth of its length. The green arrow is a step s of length 1 in the direction you choose, and the readout shows how fast f changes along it. Drag the point, click anywhere on the map, or press play.</figcaption>
</figure>

In the widget, the gradient at $(1, 1)$ doesn't point straight away from the bottom of the bowl (that would be the direction $(1, 1)$).
It leans toward $y$, because $f$ climbs three times faster in $y$ there.
Unit 3 shows what this does to gradient descent in a stretched bowl.

## Checking a gradient with nudges

Gradients worked out by hand are easy to get wrong, and so is code that computes them.
The definition gives you a cheap test: nudge one input by a small $h$, evaluate $f$ again, and divide.
It's more accurate to nudge both ways and take the slope across the middle:

$$
\frac{\partial f}{\partial x} \approx \frac{f(x + h, y) - f(x - h, y)}{2h}
$$

This is a central difference.
The one-sided version is off by an amount proportional to $h$ (for $x^2$ it's off by exactly $h$, which is the 6.1 from earlier).
The central version is off by an amount proportional to $h^2$, which is much smaller.
In the worked example below, with $h = 10^{-5}$, the estimate for $L$ lands within about $4 \times 10^{-12}$ of the exact $-4$, and that leftover is floating-point rounding.

It's too expensive to compute gradients this way for training.
A central difference takes two evaluations of $f$ for every input, so for a model with a million parameters, one gradient costs two million passes through the model.
So it's only used to spot-check a few entries, and the real gradient comes from the chain rule.

## Worked example

```python
import numpy as np

def f(x):
    return x ** 2

# Secant slopes at x = 3 for smaller and smaller h.
for h in [1.0, 0.1, 0.01, 0.001]:
    print(h, (f(3 + h) - f(3)) / h)

def g(v):                          # g(x, y) = x^2 + 3y^2
    return v[0] ** 2 + 3 * v[1] ** 2

def grad_g(v):                     # by hand: (2x, 6y)
    return np.array([2 * v[0], 6 * v[1]])

def numerical_grad(fn, v, h=1e-5):
    grad = np.zeros_like(v)
    for i in range(len(v)):
        step = np.zeros_like(v)
        step[i] = h
        grad[i] = (fn(v + step) - fn(v - step)) / (2 * h)
    return grad

v = np.array([1.0, 1.0])
print(grad_g(v), np.linalg.norm(grad_g(v)))
print(numerical_grad(g, v))

def L(w):                          # L(w) = (2w - 3)^2
    return (2 * w - 3) ** 2

print(2 * (2 * 1.0 - 3) * 2, (L(1.0 + 1e-5) - L(1.0 - 1e-5)) / 2e-5)
```

```
1.0 7.0
0.1 6.100000000000012
0.01 6.009999999999849
0.001 6.000999999999479
[2. 6.] 6.324555320336759
[2. 6.]
-4.0 -4.000000000004
```

The secant slopes come out as $6 + h$, apart from floating-point noise in the last digits.
The hand-derived gradient $(2, 6)$ and the central-difference estimate agree to every digit NumPy prints, and the chain-rule answer of $-4$ for $L$ matches its nudge estimate to about twelve digits.

## Where this shows up later

- Unit 3: gradient descent repeats the update $\theta \leftarrow \theta - \eta \nabla L$, a small step along $-\nabla L$.
- Unit 4: backpropagation is the chain rule run backward through a network, and a central-difference check like the one above is how you test it.
- Unit 6: the gradient of the next-token loss with respect to the model's outputs works out, by the chain rule, to the predicted probabilities minus a vector with a 1 at the correct token.
- Unit 10: the policy gradient is the gradient of expected reward with respect to the policy's parameters, estimated from samples.

## Self-test

<details>
<summary>What is the derivative of 4x³ − x, and what is it at x = 1?</summary>

$12x^2 - 1$, by the power rule on each term.
At $x = 1$ it's 11.

</details>

<details>
<summary>In the tangent widget, set x = 1 and try a few values of h. What does the orange slope read, and why is it always above the blue one?</summary>

It reads $2 + h$.
For $x^2$ the secant slope is exactly $2x + h$, so with $h > 0$ it sits $h$ above the tangent slope $2x$ and only meets it as $h$ goes to zero.

</details>

<details>
<summary>For L(w) = (2w − 3)², what is dL/dw at w = 2, and which way should w move to lower the loss?</summary>

$u = 2 \cdot 2 - 3 = 1$, so $\frac{dL}{dw} = 2u \cdot 2 = 4$.
It's positive, so increasing $w$ raises the loss, and $w$ should move down (toward 1.5).

</details>

<details>
<summary>What is the gradient of x² + 3y² at (2, −1), and how fast does f rise if you step along it?</summary>

$\nabla f = (2 \cdot 2, 6 \cdot (-1)) = (4, -6)$.
f rises at $\|\nabla f\| = \sqrt{52} \approx 7.21$ per unit of step.

</details>

<details>
<summary>In the contour widget, the gradient at (1, 1) doesn't point straight away from the center of the bowl. Why not?</summary>

The gradient points where $f$ climbs fastest from that spot, and $f$ climbs faster in $y$ than in $x$ there ($6y = 6$ against $2x = 2$).
So the arrow leans toward $y$.
It's always perpendicular to the contour through the point, and the contours are stretched ellipses, not circles, so perpendicular to the contour isn't the same as straight away from the center.

</details>

<details>
<summary>Why check gradients with finite differences instead of computing them that way?</summary>

Each finite difference costs two evaluations of the function per input.
For a model with millions of parameters that's millions of forward passes for a single gradient, while backpropagation gets all of them for a small multiple of one pass.

</details>

::: done-when
- You can differentiate a polynomial with the power rule and a two-step function like $(2w - 3)^2$ with the chain rule.
- You can compute the gradient of a two-input function and its length at a given point.
- You can explain with the dot product why the gradient points in the steepest uphill direction and is perpendicular to contour lines.
- You can write a central-difference gradient check in NumPy and compare it to a hand-derived gradient.
:::

## Go deeper

- 3Blue1Brown's Essence of Calculus: [the paradox of the derivative](https://www.3blue1brown.com/lessons/derivatives) and [visualizing the chain rule](https://www.3blue1brown.com/lessons/chain-rule-and-product-rule). The nudge picture from this unit, animated.
- [CS231n notes on optimization](https://cs231n.github.io/optimization-1/), the section "Computing the gradient": numerical and analytic gradients side by side, and why you use one to check the other.
- [Mathematics for Machine Learning](https://mml-book.github.io/), chapter 5, sections 5.1 and 5.2: derivatives, partial derivatives and gradients done more formally.
