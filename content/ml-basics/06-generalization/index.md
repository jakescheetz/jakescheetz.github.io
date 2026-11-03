---
title: Overfitting and generalization
subtitle: Why a low training loss can mislead you, and how held-out data catches it.
unit: 6
section: Classical ML
minutes: 55
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'What is the validation set for?'
    answer: 'Choosing settings like the degree or $\lambda$, using data the weights weren''t fit to'
    wrong: ['Fitting the model''s weights, alongside the training set', 'Giving one final score at the very end, after every choice is made', 'Checking that the training loss has stopped going down']
    why: 'The weights are fit on the training set. Choices the fit doesn''t make itself (hyperparameters) are made by comparing validation losses, and the final score comes from the test set.'
  - kind: term
    q: 'Which of these describes overfitting?'
    answer: 'Training loss keeps falling while the loss on held-out data goes up'
    wrong: ['Training and held-out loss both stay high because the model is too simple', 'Training loss stops falling because the learning rate is too small', 'Held-out loss comes out lower than training loss']
    why: 'The model is fitting the noise in its own training points. Both losses staying high is underfitting.'
  - kind: term
    q: 'The L2-penalized loss is the training MSE plus $\lambda/2$ times the sum of the squared weights. What does $\lambda$ control?'
    answer: 'How much a large weight costs compared with a missed point'
    wrong: ['How far each gradient descent step goes', 'The highest power of $x$ in the polynomial', 'How many training points the fit uses']
    why: 'At $\lambda = 0$ the penalty is off. The bigger $\lambda$ gets, the more the fit trades accuracy on the training points for smaller weights.'
  - kind: term
    q: 'What is benchmark contamination?'
    answer: 'Benchmark questions and answers ending up in a model''s training data'
    wrong: ['A benchmark whose answer labels contain mistakes', 'A model scoring differently each time the benchmark is rerun', 'A benchmark too easy to tell current models apart']
    why: 'A model that saw the answers during training can score well without being able to do the task, so the benchmark stops measuring what it was built for.'
  - kind: term
    q: 'What does a model''s capacity mean?'
    answer: 'How wide a range of functions it can fit'
    wrong: ['How many training examples it can hold in memory', 'The largest value any one weight is allowed to take', 'The lowest validation loss it can reach on this data']
    why: 'A degree-9 polynomial can fit anything a degree-3 one can, and more. More capacity can only lower the training loss.'
  - kind: apply
    q: 'You fit a degree-9 polynomial with no penalty to 10 training points with different $x$ values. What is the training MSE?'
    answer: '$0$'
    wrong: ['$0.0625$, the noise level', 'About $1.47$', '$0.0979$']
    why: 'Ten weights for ten points is enough to pass through every point exactly. The 1.4732 is its validation loss, and 0.0979 is degree 3''s training loss.'
  - kind: apply
    q: 'With learning rate $\eta = 0.2$ and penalty $\lambda = 0.5$, weight decay multiplies each weight by what factor per step?'
    answer: '$0.9$'
    wrong: ['$0.8$', '$0.1$', '$0.5$']
    why: 'The factor is $1 - \eta\lambda = 1 - 0.1 = 0.9$. Getting $0.8$ means doubling, which happens if you forget the $\frac{1}{2}$ in the penalty.'
  - kind: apply
    q: 'The noise on every point has standard deviation 0.25. About what validation MSE does even the true curve get?'
    answer: '$0.0625$'
    wrong: ['$0.25$', '$0.5$', '$0$']
    why: 'The true curve misses each point by its noise, and the average squared noise is the variance, $0.25^2 = 0.0625$.'
  - kind: apply
    q: 'You try 20 versions of a model and keep the one with the best test score. What went wrong?'
    answer: 'The test set was used to choose, so its score is now optimistic'
    wrong: ['Nothing, as long as no test example was trained on', 'The training loss of the chosen model will be too high', 'The validation set is now contaminated with test data']
    why: 'Picking the best of 20 on one set favors the version whose errors happened to suit that set. The test set has turned into a validation set.'
  - kind: apply
    q: 'In the early-stopping widget, validation loss is lowest at step 305 and training loss is lowest at step 1,000,000. Which weights do you keep?'
    answer: 'The ones from step 305'
    wrong: ['The ones from step 1,000,000', 'The average of the weights over all steps', 'The ones from the step where training loss levels off']
    why: 'Validation loss is the estimate of how the model does on new data. After step 305 the extra training only fits noise in the ten points.'
  - kind: apply
    q: 'Which of these leaks information from the validation set into training?'
    answer: 'Normalizing inputs with statistics computed before splitting'
    wrong: ['Shuffling the examples randomly before splitting them', 'Making the validation set larger than the test set', 'Fitting the weights on the training set only']
    why: 'If the mean and standard deviation used for normalizing come from all the data, they include the validation points, so the training inputs carry a little information about them.'
  - kind: apply
    q: 'In the capacity widget at degree 9 with no penalty, you click "Add 30 training points". What happens to the validation MSE?'
    answer: 'It drops from about 1.47 to about 0.06'
    wrong: ['It rises, because there is more noise to fit', 'It stays the same, because the degree didn''t change', 'It drops to 0, because the curve now hits every point']
    why: 'Ten weights can''t pass through 40 points, so the fit has to follow the trend they share. It goes from 1.4732 to 0.0574.'
---

Units 4 and 5 trained models by pushing a loss down, and both measured that loss on the same examples the model was trained on.
That number tells you how well the model fits data it has already seen.
What you actually want is a model that does well on new inputs, and the loss on those can go up while the training loss keeps going down.
Every model later in the route, from the network in unit 8 to the language models in units 11 to 14, gets checked on data it never trained on, and the fixes here (held-out splits, weight decay, early stopping) show up in all of them.

## Fitting ten noisy points

I'll start with a small problem where I know the right answer.
The hidden rule is $y = \sin(\pi x)$ for $x$ between −1 and 1, a curve with one dip and one hump.
Every measurement comes with noise, so what I actually observe is

$$
y = \sin(\pi x) + \varepsilon
$$

where $\varepsilon$ is fresh random noise for each point, with standard deviation 0.25.

::: formula
$\varepsilon$ (epsilon) is drawn from a normal distribution with mean 0 and standard deviation 0.25 ([unit 3](../03-probability/#random-variables-and-distributions)).
:::

So a typical point sits about 0.25 above or below the curve, and nothing about $x$ tells you which way.

I take ten training points at evenly spaced $x$ and fit a polynomial of degree $k$ to them:

$$
\hat y = w_0 + w_1 x + w_2 x^2 + \cdots + w_k x^k
$$

::: formula
$\hat y$ is the prediction at $x$, and $w_0, \dots, w_k$ are the $k + 1$ weights the fit gets to choose.
Degree 0 is a flat line at height $w_0$, degree 1 is unit 4's straight line, and each extra degree adds one more power of $x$ with its own weight.
:::

The prediction is still a weighted sum, with the powers $1, x, \dots, x^k$ playing the role of inputs.
So the best weights come from the same mean squared error as in [unit 4](../04-gradient-descent/#a-loss-is-one-number-for-how-wrong), and for a model this small NumPy can solve for them directly instead of walking downhill.

The training loss is that mean squared error on the ten training points.
It drops as the degree goes up: 0.6244 at degree 0, 0.0979 at degree 3, and 0.0000 at degree 9.
Degree 9 has ten weights for ten points, which is enough to pass through every point exactly.

To see how each fit does on new data, I drew 20 more points from the same rule and never let the fit see them.
This is a *validation set*, and the mean squared error on it is the *validation loss*.
At degree 0 it's 0.4196, at degree 3 it's 0.0699, and at degree 9 it's 1.4732, worse than the flat line.
How well a model does on data it hasn't seen is called *generalization*, and the difference between its validation loss and its training loss is the *generalization gap*.

There's a floor under the validation loss.
Even the true curve misses each point by that point's noise $\varepsilon$, and the average of $\varepsilon^2$ is the noise's variance, $0.25^2 = 0.0625$ ([unit 3](../03-probability/#expectation-spread-and-sampling)).
So 0.0625 is about the best any model can expect, and degree 3's 0.0699 is close to it.
A training loss far below 0.0625 means the fit is following the noise in those ten particular points, and the next ten points will have different noise.

Degrees 0 to 2 *underfit*.
They can't bend enough to follow a dip and a hump, so both losses stay high (degree 1 scores 0.4364 on training and 0.1946 on validation).
Degree 3 is the first that can turn twice.
Past degree 4 the fit *overfits*: training loss keeps falling while validation loss climbs, from 0.0986 at degree 5 to 0.2061 at degree 8 and 1.4732 at degree 9.
The degree-9 curve swings far above and below the data between training points.
Its largest weight is 270.4, with big positive and negative terms cancelling each other to land on every point.

The range of functions a model can represent is its *capacity*.
A degree-$k$ polynomial can do anything a lower degree can (set the extra weights to 0), so adding degrees can only lower the training loss.
On new data, extra capacity helps until the model can fit the noise, and then it hurts.

<figure class="widget" id="capacity">
<figcaption>Left: the ten training points (filled blue), the 20 validation points (hollow orange), the true curve sin(πx) (dashed) and the current fit. Right: training (blue) and validation (orange) error for every degree on a log scale, with the noise level 0.0625 dashed. Drag the degree first. Then, at degree 9, turn up the penalty λ, or add 30 training points.</figcaption>
</figure>

Counting weights works as a measure of capacity for polynomials, but it's a poor guide for large neural networks.
[Zhang et al. (2016)](https://arxiv.org/abs/1611.03530) showed that standard image-classification networks can fit their training images even when every label is replaced by a random one, so they have room to memorize anything.
The same networks still generalize well when the labels are real.
What a big network ends up learning depends on the data and on how it's trained, as well as on its size, so in practice you measure generalization on held-out data instead of predicting it from the parameter count.

## Train, validation and test

Before training anything, split the data three ways.

- The **training set** is what the weights are fit to.
- The **validation set** is for choices the fit doesn't make itself: the degree, the penalty strength $\lambda$ below, when to stop training. These settings are called *hyperparameters*.
- The **test set** gives one final score, at the end, for the model you picked.

The test set exists because choosing on the validation set makes the validation score optimistic.
With ten degrees to choose from, the winner is partly the one whose errors happened to suit those 20 particular points.
The test set played no part in any choice, so its score doesn't have that bias.
In the worked example below, validation picks degree 3 and its test loss is 0.0490.
That's lower than its validation loss of 0.0699, the opposite direction from the bias I just described, because a score from 20 points is noisy either way (unit 7 measures how noisy).

Once you change the model because of its test score, the test set has become another validation set, and an unbiased number needs fresh data.
When data is scarce, *cross-validation* stretches it: split the training data into five parts, train five times with a different part held out each time, and average the five validation losses.

## Weight decay, early stopping and more data

Picking a lower degree is one fix for overfitting.
Three others keep the big model and stop it from fitting the noise.

The first is an *L2 penalty*, an extra term in the loss that charges for large weights:

$$
L(w) = \frac{1}{N} \sum_{i=1}^{N} (\hat y_i - y_i)^2 + \frac{\lambda}{2} \sum_{j=1}^{k} w_j^2
$$

::: formula
The first term is the training MSE over the $N$ training points, as before.
The second adds up the squared weights and multiplies by $\lambda/2$.
$\lambda$ (lambda) is a different number from the curvature that [unit 4](../04-gradient-descent/#the-learning-rate) also called $\lambda$.
The sum starts at $j = 1$ and leaves out $w_0$, because shifting the whole curve up or down doesn't make it wiggle.
The $\frac{1}{2}$ is there so the gradient comes out without a stray 2.
:::

The penalty strength $\lambda$ is a number you choose, 0 or more: at 0 the penalty is off, and the larger it is, the more a big weight costs compared with a missed point.

The fit now has to balance two things: weights big enough to follow the data, small enough to keep the penalty low.
At degree 9, $\lambda = 0.0001$ already cuts the validation loss from 1.4732 to 0.1325, and the largest weight from 270.4 to 5.0.
At $\lambda = 0.01$ the validation loss is 0.0663, slightly better than degree 3's 0.0699.
At $\lambda = 1$ the penalty wins and flattens the curve, with training loss 0.4667 and validation loss 0.2629.
So $\lambda$ is a hyperparameter, picked on the validation set like the degree.

For a general model with parameters $\theta$, the penalty is $\frac{\lambda}{2}\|\theta\|^2$, half of $\lambda$ times the squared length of the parameter vector ([unit 1](../01-vectors/#a-vector-is-a-list-of-numbers)).
Its gradient is $\lambda\theta$, so add that to the gradient in the [update rule](../04-gradient-descent/#walk-downhill):

$$
\theta \leftarrow \theta - \eta\,(\nabla L_{\text{data}} + \lambda\theta) = (1 - \eta\lambda)\,\theta - \eta \nabla L_{\text{data}}
$$

::: formula
By the [power rule](../02-gradients/#the-rules-you-need), each penalty term $\frac{\lambda}{2}\theta_j^2$ has derivative $\lambda\theta_j$, which is where $\lambda\theta$ comes from.
$\nabla L_{\text{data}}$ is the gradient of the loss without the penalty, and $\eta$ is the learning rate.
:::

The right-hand side says what a step does: multiply every parameter by $1 - \eta\lambda$, a number just under 1, and then take the usual gradient step.
With $\eta = 0.1$ and $\lambda = 0.001$ the factor is $0.9999$.
A weight the data stops pushing on shrinks toward zero a little every step, which is why this is called *weight decay*.
Unit 9 covers AdamW, which applies the decay separately from the optimizer's own step.

The second fix is *early stopping*.
Train the degree-9 polynomial with gradient descent instead of solving for it, starting from all weights at zero with $\eta = 0.3$, and record both losses as it goes.

<figure class="widget" id="early-stopping">
<figcaption>Gradient descent on the degree-9 fit. Left: the curve as training goes. Right: training (blue) and validation (orange) loss against the step number, both on log scales, up to a million steps. The green line marks the step with the lowest validation loss so far.</figcaption>
</figure>

Validation loss bottoms out at 0.0679 at step 305.
By step 1,000 it's back up to 0.1020, while training loss has kept falling, from 0.0836 at step 305 to 0.0373.
After a million steps training loss is 0.0114 and validation loss is 0.1542.
Early stopping means keeping the weights from the step with the lowest validation loss, or stopping once validation loss hasn't improved for a while.

Unit 4's [steep and flat directions](../04-gradient-descent/#the-learning-rate) explain why this works.
Here the curvature of the loss ranges from 3.12 in its steepest direction to 0.00000014 in its flattest.
The broad shape of the data lies along the steep directions, which gradient descent fixes in the first few hundred steps.
Bending the curve through each noisy point needs the flat directions, which take millions of steps.
Stopping early leaves the weights along those directions close to their starting value of zero, much like an L2 penalty would.

The third fix is more data.
With 40 training points (the original ten plus 30 more from the same rule), degree 9 with no penalty scores 0.0432 on training and 0.0574 on validation.
Ten weights can't pass through 40 points, so the fit follows the trend the points share, and the noise, which pushes different points in different directions, mostly cancels out.

## Leakage and contamination

Everything above assumes the validation and test points really are unseen.
*Leakage* is when information from them reaches training anyway, and it makes held-out scores look better than the model is.
It's rarely deliberate:

- The same example, or a near copy, lands on both sides of the split. This is common in data scraped from the web.
- Inputs are normalized with a mean and standard deviation computed over the whole dataset before splitting, so the training inputs carry a little information about the held-out points.
- Data with a time order is split at random, so the model trains on examples from after the ones it's tested on.

Language models have a large-scale version of this, called *benchmark contamination*.
They're pretrained on huge crawls of the public web, and benchmark questions get posted online along with their answers.
If those pages end up in the training data, a high score can mean the model has seen the answers.
[Zhang et al. (2024)](https://arxiv.org/abs/2405.00332) wrote a fresh set of grade-school math problems matched in style and difficulty to GSM8k, a widely used benchmark, and found accuracy drops of up to 8% on the new set for some model families, though many models, especially the strongest ones, showed little sign of it.
Benchmark authors try to make filtering possible: BIG-bench puts a fixed canary string in every task file so that people building training sets can find and drop documents that contain it.

From the security side this is a familiar problem.
A test only measures something while the answers stay secret, and a public benchmark's answers are on the internet from the day it's released.
Some newer evaluations keep their test sets private, or keep writing new questions, for that reason.

## Worked example

```python
import numpy as np

rng = np.random.default_rng(96)
def noisy(x):                    # the truth, sin(pi x), plus noise with std 0.25
    return np.round(np.sin(np.pi * x) + rng.normal(0, 0.25, len(x)), 2)

x_train = np.round(np.linspace(-1, 1, 10), 2); y_train = noisy(x_train)
x_val = np.round(rng.uniform(-1, 1, 20), 2);   y_val = noisy(x_val)
x_test = np.round(rng.uniform(-1, 1, 20), 2);  y_test = noisy(x_test)
x_more = np.round(rng.uniform(-1, 1, 30), 2);  y_more = noisy(x_more)

def fit(x, y, k, lam=0.0):
    """Degree-k polynomial minimizing MSE + (lam/2) * (w_1^2 + ... + w_k^2)."""
    X = np.vander(x, k + 1, increasing=True)     # columns 1, x, x^2, ..., x^k
    R = np.sqrt(lam * len(x) / 2) * np.eye(k + 1)[1:]  # the penalty, as k extra rows
    A, b = np.vstack([X, R]), np.concatenate([y, np.zeros(k)])
    return np.linalg.lstsq(A, b, rcond=None)[0]

def mse(w, x, y):
    return np.mean((np.vander(x, len(w), increasing=True) @ w - y) ** 2)

val = []
for k in range(10):
    w = fit(x_train, y_train, k)
    val.append(mse(w, x_val, y_val))
    print(f"degree {k}  train {mse(w, x_train, y_train):.4f}  val {val[-1]:.4f}")

k = int(np.argmin(val))
print(f"picked degree {k}  test {mse(fit(x_train, y_train, k), x_test, y_test):.4f}")

for lam in [0, 1e-4, 1e-2, 1]:
    w = fit(x_train, y_train, 9, lam)
    print(f"degree 9, lam {lam:<6g}  train {mse(w, x_train, y_train):.4f}"
          f"  val {mse(w, x_val, y_val):.4f}  largest |w| {np.abs(w).max():.1f}")

x_big, y_big = np.concatenate([x_train, x_more]), np.concatenate([y_train, y_more])
w = fit(x_big, y_big, 9)
print(f"degree 9, 40 points  train {mse(w, x_big, y_big):.4f}  val {mse(w, x_val, y_val):.4f}")
```

```
degree 0  train 0.6244  val 0.4196
degree 1  train 0.4364  val 0.1946
degree 2  train 0.4349  val 0.1882
degree 3  train 0.0979  val 0.0699
degree 4  train 0.0940  val 0.0702
degree 5  train 0.0428  val 0.0986
degree 6  train 0.0281  val 0.1140
degree 7  train 0.0262  val 0.1273
degree 8  train 0.0092  val 0.2061
degree 9  train 0.0000  val 1.4732
picked degree 3  test 0.0490
degree 9, lam 0       train 0.0000  val 1.4732  largest |w| 270.4
degree 9, lam 0.0001  train 0.0275  val 0.1325  largest |w| 5.0
degree 9, lam 0.01    train 0.0954  val 0.0663  largest |w| 2.4
degree 9, lam 1       train 0.4667  val 0.2629  largest |w| 0.3
degree 9, 40 points  train 0.0432  val 0.0574
```

The training column only ever goes down, while the validation column bottoms out at degree 3 and blows up at degree 9.
The extra rows `R` are the penalty written as least squares: each one asks a single weight $w_j$ to be 0, and its squared miss, $\frac{\lambda N}{2} w_j^2$, gets added to the summed squared error, which makes the total $N$ times the penalized loss $L(w)$ above, so it has the same minimum.

## Where this shows up later

- Unit 7: what to measure on the held-out set, and how much a score from a few dozen examples can move.
- Unit 8: neural networks usually have far more parameters than the polynomials here, and you watch their training and validation curves the same way.
- Unit 9: AdamW, the optimizer most large models train with, and its decoupled weight decay.
- Unit 17: in reward hacking, the reward model's score keeps rising while the true quality falls, the same shape as the training and validation curves in this unit.

## Self-test

<details>
<summary>Degree 9 gets a training MSE of 0.0000 on the ten points. Why is that bad news here rather than good news?</summary>

The noise alone gives even the true curve an average squared miss of 0.0625.
A training loss far below that means the fit is matching the noise in these ten points, and its validation loss of 1.4732 confirms it.

</details>

<details>
<summary>Why keep a test set separate from the validation set?</summary>

The validation set was used to pick the model, so the winner's validation score includes some luck from fitting those particular points.
The test set played no part in the choice, so it gives an unbiased score, as long as you don't go back and change the model because of it.

</details>

<details>
<summary>In the capacity widget, set degree 9 and click "Add 30 training points". What happens to the curve and the validation error, and why?</summary>

The big swings between points mostly go away (a small hook is left at the far left, where there are few points), and the validation error drops from 1.4732 to 0.0574.
Ten weights can't pass through 40 points, so the fit has to follow what the points have in common instead of each point's noise.

</details>

<details>
<summary>With η = 0.01 and λ = 0.1, by what factor does weight decay shrink each weight per step?</summary>

$1 - \eta\lambda = 1 - 0.001 = 0.999$.
After 1,000 steps with no push from the data, a weight would be down to about $0.999^{1000} \approx 0.37$ of its starting value.

</details>

<details>
<summary>In the early-stopping widget, training loss keeps falling for a million steps. Why does validation loss turn back up after step 305?</summary>

Gradient descent fixes the steep directions of the loss first, and those carry the broad shape of the data.
After that it slowly moves along the flat directions, which is where the curve bends to hit each noisy training point, and that makes predictions between the points worse.

</details>

<details>
<summary>Give two ways held-out data can influence training without anyone copying it in on purpose.</summary>

Duplicates or near copies that end up on both sides of a split, and normalizing with statistics computed before splitting.
For language models, benchmark questions posted on the web that end up in the pretraining crawl.

</details>

::: done-when
- You can explain why training loss falls with every extra degree while validation loss turns back up, and what the noise level has to do with it.
- You can say what the training, validation and test sets are each for, and when a test set stops being one.
- You can write the L2-penalized loss and derive the weight-decay update from it.
- You can name two ways evaluation data leaks into training, including benchmark contamination.
:::

## Go deeper

- [An Introduction to Statistical Learning](https://www.statlearning.com/) (free PDF), sections 2.2 (training versus test error), 5.1 (cross-validation) and 6.2 (ridge regression, which is this unit's L2 penalty). The same ideas with more care and more examples.
- [Deep Learning, chapter 7](https://www.deeplearningbook.org/contents/regularization.html) (Goodfellow, Bengio and Courville). Section 7.1.1 works through L2 penalties and weight decay, and section 7.8 shows when early stopping is equivalent to an L2 penalty.
- [Zhang et al., "A Careful Examination of Large Language Model Performance on Grade School Arithmetic"](https://arxiv.org/abs/2405.00332). How the GSM1k comparison was built, and how they linked the score drops to models having memorized GSM8k.
