---
title: Classification and logistic regression
subtitle: Turn a score into a probability, then train it with log loss.
unit: 5
section: Classical ML
minutes: 60
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'What does the sigmoid function do?'
    answer: 'Squashes any score into a number between 0 and 1'
    wrong: ['Rounds the score to exactly 0 or 1', 'Picks the class with the highest score', 'Scales a list of scores so they add up to 1']
    why: '$\sigma(z) = 1/(1 + e^{-z})$ is close to 0 for very negative $z$, close to 1 for very positive $z$, and exactly 0.5 at $z = 0$. It never outputs exactly 0 or 1.'
  - kind: term
    q: 'In logistic regression, what is the decision boundary?'
    answer: 'The points where $w \cdot x + b = 0$, so $p = 0.5$'
    wrong: ['The points where $p = 1$', 'The points where the loss is exactly 0', 'The line through the average of each class']
    why: 'The sigmoid crosses 0.5 exactly at a score of 0, so the boundary is the set of inputs with $w \cdot x + b = 0$, a straight line for two features.'
  - kind: term
    q: 'Solving $p = \sigma(z)$ for $z$ gives what?'
    answer: '$z = \log \frac{p}{1 - p}$, the log-odds'
    wrong: ['$z = \log p$, the log-probability', '$z = \frac{p}{1 - p}$, the odds', '$z = -\log(1 - p)$, the loss']
    why: 'The score is the log of the odds of class 1. That''s why a weight of 1 multiplies the odds by $e \approx 2.72$ per unit of its feature.'
  - kind: term
    q: 'What does log loss (binary cross-entropy) charge for one example?'
    answer: 'Minus the log of the probability the model gave the true label'
    wrong: ['The squared gap between $p$ and $y$', 'Minus the log of $p$, whatever the label is', '1 if the prediction is wrong and 0 if right']
    why: 'For a spam email it''s $-\log p$ and for a normal one $-\log(1 - p)$. Either way it''s the probability of what actually happened.'
  - kind: term
    q: 'Which way does $w$ point relative to the decision boundary?'
    answer: 'Perpendicular to it, toward the class-1 side'
    wrong: ['Along the boundary', 'Perpendicular to it, toward the class-0 side', 'From the origin to the nearest training point']
    why: 'Any two points on the boundary have the same $w \cdot x$, so the step between them has a zero dot product with $w$. Moving along $w$ raises the score, so it points toward spam.'
  - kind: apply
    q: 'With $w = (0.5, 1)$ and $b = -5$, what probability of spam does an email with 2 links and 4 exclamation marks get?'
    answer: '$0.5$'
    wrong: ['$0$', '$0.73$', '$0.27$']
    why: '$z = 0.5 \cdot 2 + 4 - 5 = 0$ and $\sigma(0) = 0.5$. The email sits exactly on the boundary.'
  - kind: apply
    q: 'A spam email gets $p = 0.1$. What is its log loss?'
    answer: '$2.30$'
    wrong: ['$0.11$', '$0.81$', '$0.90$']
    why: 'The label is spam, so the loss is $-\log 0.1 = 2.30$. $0.11$ is $-\log 0.9$, the loss it would get if it weren''t spam.'
  - kind: apply
    q: 'A normal email ($y = 0$) with $x = (2, 1)$ gets $p = 0.6$. What is $\partial L / \partial w$ for it?'
    answer: '$(1.2, 0.6)$'
    wrong: ['$(-0.8, -0.4)$', '$(-1.2, -0.6)$', '$(0.6, 0.6)$']
    why: 'The gradient is $(p - y)\,x = 0.6 \cdot (2, 1)$. A step subtracts it, which lowers the score of emails like this one.'
  - kind: apply
    q: 'You multiply both $w$ and $b$ by 3. What happens?'
    answer: 'The boundary stays put and the probabilities move toward 0 and 1'
    wrong: ['The boundary moves three times farther from the origin', 'The boundary rotates and the probabilities stay the same', 'Nothing changes, because the sigmoid undoes the scaling']
    why: 'Every score triples, so the points with $z = 0$ are the same, but every other score is farther from 0 and its probability is more extreme.'
  - kind: apply
    q: 'With $w_2 = 1$, what does one more exclamation mark do to the email''s odds of spam?'
    answer: 'Multiplies them by about 2.72'
    wrong: ['Adds 1 to the probability of spam', 'Multiplies the probability of spam by about 2.72', 'Adds about 2.72 to the odds']
    why: 'It adds 1 to the score, which is the log-odds, so the odds get multiplied by $e^1 \approx 2.72$. The probability changes by less, and by different amounts depending on where it started.'
  - kind: apply
    q: 'A spam email scores $z = -6$, so $p \approx 0.0025$. Which loss pushes harder on $z$?'
    answer: 'Log loss, by a factor of about 200'
    wrong: ['Squared error, by a factor of about 200', 'They push equally hard', 'Neither, because the gradient is 0 at $z = -6$']
    why: 'Log loss gives $p - y \approx -0.998$. Squared error gives $2(p - y)\,p(1 - p) \approx -0.005$, because the sigmoid is nearly flat at $z = -6$.'
  - kind: apply
    q: 'You train on data that a line separates perfectly, with plain gradient descent and no penalty. What happens to $\|w\|$?'
    answer: 'It keeps growing, slowly, for as long as you train'
    wrong: ['It shrinks toward 0', 'It grows until the loss hits exactly 0, then stops', 'It stops changing once every point is classified right']
    why: 'Once every point is on its correct side, a longer $w$ pushes every $p$ closer to its label and lowers the loss a little more. The loss never reaches 0, so there is no point where it stops.'
---

Unit 4's line predicts a number, but a lot of questions have a yes-or-no answer.
A spam filter has to decide whether an email is spam, and a language model has to pick one token out of 50,257.
*Logistic regression* is the simplest model that answers a yes-or-no question with a probability.
Its loss and its gradient come back almost unchanged when unit 11 trains a language model and unit 16 trains a reward model.

## From a score to a probability

The example for this unit is a toy spam filter.
Each email is described by two numbers, $x = (x_1, x_2)$: $x_1$ is how many links it contains and $x_2$ is how many exclamation marks.
Each email also has a label $y$, which is 1 if it's spam and 0 if it isn't.
The training set is 22 emails, 11 of each, plotted in the widget further down.

The model starts by giving each email a score:

$$
z = w \cdot x + b = w_1 x_1 + w_2 x_2 + b
$$

$w = (w_1, w_2)$ holds one weight per feature, and $b$ is a single number called the *bias*.
The first part is the dot product from [unit 1](../01-vectors/#the-dot-product): multiply each feature by its weight and add the results up.
A positive weight means that feature pushes the score up, toward spam.
The bias then shifts every score by the same amount, which sets how much evidence an email needs before it counts as spammy.

Pick some weights by hand for now: $w = (0.5, 1)$ and $b = -5$, so $z = 0.5 x_1 + x_2 - 5$.
Email A has 4 links and 2 exclamation marks, so $z = 2 + 2 - 5 = -1$.
Email B has 6 links and 5 exclamation marks, so $z = 3 + 5 - 5 = 3$.
B scores higher, so it looks more like spam.

A score can be any number, though, and $-1$ doesn't say how likely spam is.
The *sigmoid* function $\sigma$ (here a function, not the standard deviation from [unit 3](../03-probability/#expectation-spread-and-sampling)) squashes any score into the range between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Here $e \approx 2.718$ is the base of the natural log, and $e^{-z}$ is $e$ raised to minus the score.
When $z$ is large and positive, $e^{-z}$ is close to 0, the bottom of the fraction is close to 1, and $\sigma(z)$ is close to 1.
When $z$ is large and negative, $e^{-z}$ is huge and $\sigma(z)$ is close to 0.
At $z = 0$, $e^0 = 1$, so $\sigma(0) = 1/2$.
Some values in between:

| $z$ | −3 | −1 | 0 | 1 | 3 |
|---|---|---|---|---|---|
| $\sigma(z)$ | 0.05 | 0.27 | 0.50 | 0.73 | 0.95 |

The model's output is $p = \sigma(z)$, read as the probability that the email is spam.
Email A gets $p = \sigma(-1) = 0.27$ and email B gets $p = \sigma(3) = 0.95$.
The curve is symmetric around zero, $\sigma(-z) = 1 - \sigma(z)$, so the probability of "not spam", $1 - p$, is the sigmoid of the negated score.

You can also run the sigmoid backward.
Solving $p = \sigma(z)$ for $z$ gives

$$
z = \log \frac{p}{1 - p}
$$

The fraction $p / (1 - p)$ is the *odds* of spam: the probability of spam divided by the probability of not spam, so $p = 0.75$ gives odds of 3 to 1.
The log is the natural log, so the score is the log of the odds.
That gives each weight a concrete meaning.
One more exclamation mark adds $w_2 = 1$ to the log-odds, which multiplies the odds of spam by $e^1 \approx 2.72$.
One more link adds $0.5$, which multiplies them by $e^{0.5} \approx 1.65$.
Scores that get fed into a sigmoid are often called *logits*, after this log-odds function, and unit 11 uses the same name.

## The decision boundary is a line

To turn the probability into a yes or no, call an email spam when $p \ge 0.5$.
The sigmoid crosses 0.5 exactly at $z = 0$ and only ever goes up, so $p \ge 0.5$ is the same as $z \ge 0$.
The emails the model can't decide about are the ones with

$$
w \cdot x + b = 0
$$

that is, every point whose score is exactly zero.
With two features that set is a straight line, the *decision boundary*.
For the hand-picked weights it's $0.5 x_1 + x_2 = 5$, which passes through $(0, 5)$ and $(10, 0)$.
Every email on one side has $z > 0$ and gets called spam, and every email on the other side has $z < 0$.
With three features the boundary is a flat plane, and with more features it's the same kind of flat cut in more dimensions.

The dot product also tells you how $w$ sits relative to that line.
Take two points $u$ and $v$ on the boundary.
Both have $w \cdot x = -b$, so $w \cdot (u - v) = 0$, and from [unit 1](../01-vectors/#the-dot-product) a zero dot product means a right angle.
So $w$ points straight across the boundary, toward the side where scores are positive (spam).
Dividing the score by the length of $w$, $\|w\| = \sqrt{w_1^2 + w_2^2}$, gives each point's signed distance from the line, $z / \|w\|$.
That's the shadow onto $w$'s direction from [unit 1](../01-vectors/#the-dot-product), measured from the boundary instead of from the origin.
For $w = (0.5, 1)$ the length is $\|w\| = 1.12$, so email A, with $z = -1$, sits 0.89 units over on the not-spam side.

The length of $w$ also sets how sure the model is.
Multiply $w$ and $b$ by 2 and every score doubles, but the points with $z = 0$ stay the same, so the line doesn't move.
What changes is how quickly $p$ goes from near 0 to near 1 as you cross it.
In the widget the dashed lines mark $p = 0.1$ and $p = 0.9$, which are scores of $-2.2$ and $2.2$, so they sit $2.2 / \|w\|$ either side of the boundary.
A longer $w$ squeezes them in.

The 0.5 cutoff is a choice.
A filter that should almost never bin real mail could require $p \ge 0.9$, which moves the cut over to the $z = 2.2$ line.
Unit 7 is about choosing it.

The straight line is also the model's limit.
If spam sat in two opposite corners of the plot, no single line could fence it off.
Unit 8's XOR example is the standard case, and a neural network gets around it by learning new features first.

## Log loss

Training needs a loss, [one number for how wrong the model is](../04-gradient-descent/#a-loss-is-one-number-for-how-wrong).
For a single email, look at the probability the model gave to what actually happened: $p$ if the email is spam, and $1 - p$ if it isn't.
A good model puts that number close to 1.
The loss is minus its log:

$$
L = -\big[\, y \log p + (1 - y) \log(1 - p) \,\big]
$$

The $y$ and $1 - y$ in front of the two terms work as a switch.
For a spam email $y = 1$, the second term gets multiplied by 0 and drops out, and what's left is $L = -\log p$.
For a normal email $y = 0$, the first term drops out instead, leaving $L = -\log(1 - p)$.
Either way the loss is minus the log of the probability given to the true label.
The log of a number below 1 is negative, so the minus sign makes the loss positive, and the loss is 0 only when that probability is exactly 1.
The loss for the whole training set is the average of $L$ over the 22 emails.
This is called *log loss*, or *binary cross-entropy*.

With the hand-picked weights, email A is a normal email with $p = 0.27$, so its loss is $-\log 0.73 = 0.31$.
Email B is spam with $p = 0.95$, and its loss is $-\log 0.95 = 0.05$.
The spam email at $(3, 2)$ scores $z = -1.5$, so $p = \sigma(-1.5) = 0.182$ and the loss is $-\log 0.182 = 1.70$.
The model is wrong about that one, and fairly sure of itself, and it costs more than A and B put together.

| probability on the true label | 0.9 | 0.5 | 0.1 | 0.01 |
|---|---|---|---|---|
| loss | 0.11 | 0.69 | 2.30 | 4.61 |

Each time the probability on the true label drops by a factor of 10, the loss goes up by another 2.30, with no ceiling.
There's also a probability reason for the log ([unit 3](../03-probability/#why-models-work-in-log-probabilities)).
Adding up $-\log$ of each email's probability is the same as taking $-\log$ of their product.
If the emails are treated as independent ([unit 3](../03-probability/#conditional-probability-and-bayes-rule)), that product is the probability the model gives to the whole set of labels.
So minimizing log loss makes the labels you actually saw as likely as possible under the model.

Squared error from [unit 4](../04-gradient-descent/#a-loss-is-one-number-for-how-wrong), $(p - y)^2$, looks like it should work too, but it gets stuck on confident mistakes.
Its gradient with respect to the score goes through the sigmoid, so by the [chain rule](../02-gradients/#the-rules-you-need) it picks up the sigmoid's slope:

$$
\frac{\partial (p - y)^2}{\partial z} = 2(p - y) \cdot p(1 - p)
$$

The first factor, $2(p - y)$, is the derivative of the square with respect to $p$.
The second, $p(1 - p)$, is how fast $p$ changes as $z$ changes, which is the slope of the sigmoid.
That slope is 0.25 at $z = 0$ and close to 0 far out on either side, where the curve is flat.
Take a spam email the model scores at $z = -6$, so $p = 0.0025$.
Squared error pushes on its score with a gradient of $-0.0049$.
Log loss, as the next section shows, pushes with $p - y = -0.9975$, about 200 times harder.
With squared error the model learns slowest from exactly the emails it's most wrong about.

## Training with gradient descent

The gradient of log loss for one email comes from the chain rule in two steps, from $z$ to $p$ and from $p$ to $L$.
The derivative of $\log p$ is $1/p$ (a rule unit 2 didn't need), so

$$
\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1 - y}{1 - p} = \frac{p - y}{p(1 - p)}
$$

The middle expression differentiates each of the two terms of $L$, and the right side is the same thing put over a common denominator.
The other step, $\partial p / \partial z$, is the sigmoid's slope $p(1 - p)$ from above.
Multiplying the two, the $p(1 - p)$ cancels:

$$
\frac{\partial L}{\partial z} = p - y
$$

The push on the score is the miss.
For a spam email it's $p - 1$, which is negative, so a step raises the score.
For a normal email it's $p$, which is positive, so a step lowers it.
The cancellation is why log loss doesn't stall where the sigmoid is flat.

Next, the score is $z = w \cdot x + b$.
Nudging $w_1$ changes $z$ by $x_1$ per unit, nudging $w_2$ changes it by $x_2$, and nudging $b$ changes it by 1, so

$$
\frac{\partial L}{\partial w} = (p - y)\,x, \qquad \frac{\partial L}{\partial b} = p - y
$$

Each email's gradient for $w$ is its own feature vector times its miss $p - y$.
For the full training set, average these over the $N$ emails and take the step from [unit 4](../04-gradient-descent/#walk-downhill):

$$
w \leftarrow w - \eta\,\frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)\,x_i, \qquad b \leftarrow b - \eta\,\frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)
$$

$\eta$ is the learning rate, $x_i$ and $y_i$ are email $i$'s features and label, and $p_i$ is the model's current probability for it.
In words: every email pulls $w$ along its own feature vector, toward spam-like directions if it's spam and away if it isn't, in proportion to how wrong the model is about it.
Email A (not spam, $p = 0.27$, $x = (4, 2)$) has gradient $0.27 \cdot (4, 2) = (1.08, 0.54)$.
The step subtracts $\eta$ times that, which lowers the score of every email that looks like A.
An email the model already gets right with $p = 0.999$ barely pulls at all.

<figure class="widget" id="logistic">
<figcaption>Gradient descent from w = 0, b = 0 on the 22 emails (orange dots spam, blue rings not spam). Left: the boundary (solid), the p = 0.1 and p = 0.9 lines (dashed), w as an arrow, and shading for p. Right: each email at its score on the sigmoid, with its miss p − y dashed. Steps speed up as training goes on. Try the separable data and watch |w|.</figcaption>
</figure>

Training from $w = 0$, $b = 0$ is slow: the loss falls from 0.69 to 0.43 in the first 100 steps, then needs another 900 to reach 0.26.
This is the stretched bowl from [unit 4](../04-gradient-descent/#the-learning-rate) again.
Every feature is a positive count between 0 and 9, so changing $b$ and changing $w$ move the scores in similar ways.
At the start, the steepest direction of the loss curves about 170 times faster than the flattest one, and the flattest one is almost entirely $b$.
The usual fix is to subtract each feature's average before training (4.2 links and 3.4 exclamation marks here).
On centered features the same $\eta = 0.1$ gets the loss to 0.2443 in 100 steps, against 0.4266 without centering.

Now switch the widget to the separable data, which drops the two emails no line can get right.
Once every email is on its correct side, making $w$ longer pushes every $p$ closer to its label and lowers the loss a bit more.
The loss never reaches 0, so $w$ never stops growing: after 10,000 steps $\|w\|$ is 2.44, the loss is 0.0175, and both are still moving.
Unit 6 adds a penalty on the size of $w$ that stops this.

## Worked example

```python
import numpy as np

# One row per email: (links, exclamation marks). y = 1 for spam, 0 for not spam.
X = np.array([[0, 0], [1, 0], [0, 2], [2, 1], [1, 3], [3, 0], [2, 4], [4, 2],
              [3, 3], [5, 1], [6, 3],
              [5, 6], [7, 3], [6, 5], [8, 7], [4, 7], [9, 4], [6, 8], [7, 6],
              [3, 6], [8, 2], [3, 2]], dtype=float)
y = np.array([0] * 11 + [1] * 11, dtype=float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.zeros(2)
b = 0.0
eta = 0.1
for step in range(5001):
    p = sigmoid(X @ w + b)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    if step in (0, 1, 10, 100, 1000, 5000):
        right = np.sum((p >= 0.5) == y)
        print(f"step {step:4d}  loss {loss:.4f}  right {right}/22  w = {np.round(w, 3)}  b = {b:.3f}")
    w -= eta * (p - y) @ X / len(y)   # average of (p - y) x over the emails
    b -= eta * np.mean(p - y)

p = sigmoid(X @ w + b)
print("misclassified:", X[(p >= 0.5) != y].tolist())
print("new emails:", np.round(sigmoid(np.array([[4, 2], [6, 5]]) @ w + b), 3))
```

```
step    0  loss 0.6931  right 11/22  w = [0. 0.]  b = 0.000
step    1  loss 0.6131  right 11/22  w = [0.089 0.084]  b = 0.000
step   10  loss 0.5830  right 13/22  w = [0.077 0.155]  b = -0.147
step  100  loss 0.4266  right 17/22  w = [0.164 0.333]  b = -1.303
step 1000  loss 0.2570  right 20/22  w = [0.534 0.749]  b = -4.662
step 5000  loss 0.2436  right 20/22  w = [0.718 0.995]  b = -6.415
misclassified: [[6.0, 3.0], [3.0, 2.0]]
new emails: [0.175 0.946]
```

The loss starts at 0.6931, which is $\log 2$: with $w = 0$ and $b = 0$ every email gets $p = 0.5$.
That also explains step 0's 11 right (0.5 counts as spam, so exactly the spam emails are right) and why $b$ is still 0 after step 1 (the average of $0.5 - y_i$ is 0 when half the labels are 1).
The last 4,000 steps only take another 0.0134 off the loss, which is the slow $b$ direction from the previous section.
The two misses are the normal email at $(6, 3)$ and the spam email at $(3, 2)$, which sits inside the cluster of normal emails, so no straight line can put it on the spam side without taking normal emails with it.
The trained model gives emails A and B probabilities 0.175 and 0.946, close to the 0.27 and 0.95 from my hand-picked weights.

## Where this shows up later

- Unit 6: the separable case, and the penalty on $\|w\|$ that keeps it finite.
- Unit 7: choosing the threshold instead of 0.5, and checking whether a $p$ of 0.8 really means right 80% of the time.
- Unit 8: a neural network for a yes-or-no question usually ends with exactly this model on top of features it learned, which lets the boundary bend.
- Unit 11: softmax turns one score per class into probabilities for many classes. With two classes it gives $\sigma(z_1 - z_2)$, and the gradient is again $p - y$.
- Unit 16: the Bradley–Terry reward model says a person picks answer A over answer B with probability $\sigma(r_A - r_B)$, where $r_A$ and $r_B$ are the two answers' rewards. That's logistic regression on the difference of two rewards, and its loss is this unit's log loss with $y = 1$.

## Self-test

<details>
<summary>With w = (0.5, 1) and b = −5, what probability of spam does an email with 2 links and 6 exclamation marks get?</summary>

$z = 0.5 \cdot 2 + 6 - 5 = 2$, and $\sigma(2) = 0.88$.

</details>

<details>
<summary>Why does squared error learn so slowly from a spam email the model scores at z = −6?</summary>

Its gradient with respect to $z$ includes the sigmoid's slope $p(1 - p)$, which is nearly 0 that far out.
So the push is about $-0.005$, while log loss's push is $p - y \approx -1$ because that factor cancels.

</details>

<details>
<summary>A spam email gets p = 0.2. Which way does one gradient step move w and b?</summary>

Its gradient is $(p - y)\,x = -0.8\,x$ for $w$ and $-0.8$ for $b$.
The step subtracts $\eta$ times that, so $w$ moves $0.8\eta\,x$ toward the email's own feature vector and $b$ goes up by $0.8\eta$.
Both raise this email's score.

</details>

<details>
<summary>What happens to the boundary if you double b but leave w alone?</summary>

It moves without turning.
With the hand-picked weights, $0.5 x_1 + x_2 = 5$ becomes $0.5 x_1 + x_2 = 10$, a parallel line twice as far from the origin.
$w$ still sets the direction, and $b$ sets where along $w$ the line sits.

</details>

<details>
<summary>In the widget's separable data, why do the dashed lines keep closing in on the boundary?</summary>

Every email is already on its correct side, so a longer $w$ only pushes each $p$ closer to its label and lowers the loss.
The dashed lines sit $2.2 / \|w\|$ from the boundary, so as $\|w\|$ keeps growing they keep moving in.

</details>

::: done-when
- You can compute $z$, $\sigma(z)$ and the log loss for one email by hand.
- Given $w$ and $b$, you can sketch the decision boundary and say which side is class 1.
- You can derive $\partial L / \partial z = p - y$ and explain why squared error trains slower through a sigmoid.
- You can explain from the widget why $\|w\|$ keeps growing on separable data.
:::

## Go deeper

- Jurafsky and Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), the chapter "Logistic Regression and Text Classification": the same model on sentiment analysis, the full gradient derivation, and the many-class version that leads into unit 11.
- Google's Machine Learning Crash Course, [Logistic regression](https://developers.google.com/machine-learning/crash-course/logistic-regression): a short module on the sigmoid, why log loss instead of squared loss, and regularization.
