---
title: Reward models and RLHF
subtitle: How people's picks between two answers become a reward a policy can train against.
unit: 11
minutes: 60
scripts: [widgets.js]
---

Unit 10 trained a policy against a reward that came from the environment, but a chatbot has no environment that hands out a score for an answer.
RLHF (reinforcement learning from human feedback) builds one: it trains a reward model on people's choices between answers, then runs RL against that model.
Most of the page is about that reward model: what its training data looks like and how "A beat B" becomes a number.
DPO, at the end, trains on the same choices without a separate reward model.

## The three steps

The recipe most people mean by RLHF comes from InstructGPT, OpenAI's 2022 paper on training GPT-3 to follow instructions.
It has three steps.

1. **Start from a decent model.**
   Take a pretrained model and fine-tune it on example answers written by people.
   Call the result $\pi_0$.
2. **Train a reward model.**
   Have $\pi_0$ write a few answers to the same prompt, and have people say which one they prefer.
   A separate model, $r_\phi$, learns to predict their picks.
   It reads a prompt and a response and outputs one number: higher means people would probably like it more.
3. **Train the policy against the reward model.**
   The policy writes responses, the reward model scores them, and RL updates the policy toward responses that score higher.
   A penalty keeps the policy from drifting too far from $\pi_0$.

Step 3 as an equation, where $D$ is the set of training prompts:

$$
\max_\theta\; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] - \beta\, D_{\mathrm{KL}}\big(\pi_\theta \,\|\, \pi_0\big)
$$

The first term is the average reward-model score of the policy's answers.
The second term measures how far the policy's answers have moved from $\pi_0$'s, and $\beta$ sets how much that costs.

That second term is the KL divergence.
For one answer $y$ it's estimated as

$$
\log \pi_\theta(y \mid x) - \log \pi_0(y \mid x)
$$

which is positive when the policy now makes $y$ more likely than $\pi_0$ did.
A single answer can come out negative, but the average over the policy's own answers can't, and it's zero only when the two policies give every answer the same probability.
InstructGPT subtracts $\beta$ times this from each answer's reward.
They apply it token by token, which adds up to the same thing, because an answer's log probability is the sum of its tokens' log probabilities (unit 06).

From there, step 3 is unit 10's setting.
InstructGPT describes it as a bandit: a prompt comes in, the policy writes one response, the reward model scores it, and the episode ends.
The RL algorithm is PPO, and the rest of this page is about step 2.

## What a reward model is

A reward model reads a prompt $x$ and a response $y$ and returns one number, $r_\phi(x, y)$, where $\phi$ stands for its parameters.
InstructGPT built theirs from the fine-tuned model: they took off the final unembedding layer (unit 08), the one that turns the last vector into $V$ logits, and trained the network to output a single number instead.
They used a 6-billion-parameter reward model throughout.

Take the prompt "How do I reverse a list in Python?" and say $\pi_0$ wrote four answers:

- **A**: correct, with a one-line code example.
- **B**: correct, no example.
- **C**: long, hedges a lot, half right.
- **D**: wrong.

A reward model doing its job might score them 1.5, 0.6, −0.4 and −1.7.
Nobody gave it those numbers, though.
In step 2 people see two or more answers and say which is better, and that's all the reward model learns from.

A score out of 10 means something different to each rater, and a choice between two answers doesn't need a shared scale.
People still disagree on the choices: InstructGPT's labelers agreed with each other 72.6% of the time.
InstructGPT had them rank 4 to 9 answers at a time and counted every pair inside a ranking as one comparison, so one ranking of 9 answers gives 36.

## From picks to numbers

So the training data is a pile of results like "A beat B", and the reward model has to output numbers.
The link between the two is the Bradley–Terry model.
Write $y_w$ for the answer people picked (the winner), $y_l$ for the other one (the loser), and $y_w \succ y_l$ for "$y_w$ is preferred to $y_l$".
Bradley–Terry says

$$
P(y_w \succ y_l) = \sigma\big(r(y_w) - r(y_l)\big), \qquad \sigma(t) = \frac{1}{1 + e^{-t}}
$$

where $r(y)$ is short for $r_\phi(x, y)$, since both answers share the prompt.
The sigmoid $\sigma$ squashes any number into a probability between 0 and 1, with $\sigma(0) = 0.5$.

For A and B the gap is $1.5 - 0.6 = 0.9$, and $\sigma(0.9) = 0.71$.
So the model says people pick A over B 71% of the time, and B over A the other 29%.
A against D is a gap of 3.2, which gives 0.96.

Only the gap enters the formula.
Add 10 to every answer's reward and every probability stays the same, so a reward of 0 means nothing on its own.
InstructGPT shifted their trained reward model so that answers written by people averaged 0 before running RL.

Training picks the rewards that make the observed picks most likely.
For one comparison that means making $\sigma(r_w - r_l)$ big, where $r_w = r(y_w)$ and $r_l = r(y_l)$, and as in unit 06 you minimize the negative log instead:

$$
L = -\log \sigma(r_w - r_l)
$$

averaged over all comparisons.
This is unit 06's cross-entropy with two classes, where the correct class is the answer people picked.
If people picked A over B, the loss is $-\log 0.71 = 0.34$.
If they picked B, the gap flips to $-0.9$ and the loss is $-\log 0.29 = 1.24$.

The gradient is what made this loss make sense to me.
The derivative of $-\log \sigma(t)$ is $-(1 - \sigma(t))$, so a gradient step (unit 03) raises $r_w$ and lowers $r_l$ by the same amount, proportional to

$$
1 - \sigma(r_w - r_l)
$$

That's the probability the model gave to the pick people didn't make.
When people pick A, the model had already given that 0.71, so the push is 0.29.
When they pick B, the model had given that only 0.29, so the push is 0.71.
A comparison the model called at 0.95 moves the rewards by 0.05, and one it called at 0.05 moves them by 0.95.

<figure class="widget" id="bt-curve">
<figcaption>The curve is σ of the reward gap. The blue part is the probability the model gives to the pick people made, and the orange part is what's left, which sets how hard that comparison pushes the two rewards apart. Drag the gap below 0 to see a pick the model didn't expect.</figcaption>
</figure>

One comparison moves two rewards a little.
Over thousands of comparisons the rewards settle where the gaps match how often each answer won.
At the exact best fit, each answer's predicted wins, added up over all its comparisons, equal its actual wins.

The widget below runs that with the four answers from the example.
Simulated people pick between random pairs using the hidden rewards 1.5, 0.6, −0.4 and −1.7 through Bradley–Terry.
The model starts every answer at 0 and only ever sees who won.

<figure class="widget" id="bt-fit">
<figcaption>Rings are the hidden rewards, dots are the model's estimates. The last comparison's winner is green and its loser orange. Each update adds to one reward and subtracts the same amount from the other, so the estimates always average 0, like the hidden ones.</figcaption>
</figure>

Even the hidden rewards don't call every pick, because the simulated people pick at random with the Bradley–Terry probabilities.
Averaged over the six pairs, the hidden rewards should call about 83% of picks right, and over many comparisons no set of estimates calls more.
A bigger step size gets close faster and then jitters more, the same tradeoff as the learning rate in unit 03.

In the widget each answer has its own number.
A real reward model has one set of weights $\phi$ shared by every answer, and each comparison's push goes through backpropagation (unit 04) into those weights.
That's how it can score answers nobody compared, which step 3 depends on, since almost everything the policy writes there is new.

## Skipping the reward model: DPO

Rafailov et al. (2023) found a way to train the policy on the comparisons directly, with no reward model and no RL loop, and called it Direct Preference Optimization (DPO).
It starts from the step 3 objective, which has an exact best policy if you could optimize it perfectly:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_0(y \mid x)\, \exp\big(r(x, y)/\beta\big)
$$

That's $\pi_0$'s probabilities, with each answer scaled up by $e$ to the power of its reward over $\beta$.
$Z(x)$ is the sum over every possible answer that makes the probabilities add up to 1, and for a language model nobody can compute it.
Solve for the reward instead:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_0(y \mid x)} + \beta \log Z(x)
$$

Now plug that into Bradley–Terry.
Both answers in a comparison share the prompt, so the $\beta \log Z(x)$ terms cancel in $r_w - r_l$.
Put the policy being trained, $\pi_\theta$, in place of $\pi^*$, and what's left is a loss on the policy itself:

$$
L_{\mathrm{DPO}} = -\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_0(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_0(y_l \mid x)}\Big)
$$

It's the reward model's loss with $\beta \log(\pi_\theta / \pi_0)$ standing in for the reward.
Each training step computes the log probability of both answers under $\pi_\theta$ and under a frozen copy of $\pi_0$, then takes a gradient step.
The policy never writes anything during training, and it learns from the same comparison data as step 2.
In the paper, DPO matched or beat PPO-based RLHF on summarization and single-turn dialogue.

## Worked example

A reward model that's linear in four hand-made features, $r(y) = w \cdot f(y)$, fitted to 2,000 simulated comparisons.
It has no bias term, since a constant added to every reward would cancel in $r_w - r_l$.
The simulated people pick using hidden weights, and the question is whether the fit gets them back from nothing but who won.

```python
import numpy as np

rng = np.random.default_rng(0)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def reward(f, w):                    # r(y) = w . f(y), for every row of f at once
    return (f * w).sum(axis=-1)

# 300 responses, each boiled down to 4 hand-made features:
# answers the question (0/1), has a code example (0/1),
# length in hundreds of words (0 to 4), apologizes (0/1)
n = 300
F = np.column_stack([
    rng.integers(0, 2, n),
    rng.integers(0, 2, n),
    rng.uniform(0, 4, n),
    rng.integers(0, 2, n),
]).astype(float)

w_true = np.array([2.0, 0.8, 0.3, -1.0])   # what the simulated people care about

# 2,000 comparisons between random pairs. People pick a over b with
# probability sigmoid(r(a) - r(b)) under the hidden weights.
m = 2000
a = rng.integers(0, n, m)
b = (a + rng.integers(1, n, m)) % n        # never compare a response with itself
a_wins = rng.random(m) < sigmoid(reward(F[a], w_true) - reward(F[b], w_true))
win = np.where(a_wins, a, b)
lose = np.where(a_wins, b, a)
D = F[win] - F[lose]                       # winner's features minus loser's

# Fit w by gradient descent on the average of -log sigmoid(r_w - r_l)
w = np.zeros(4)
eta = 0.5
for step in range(1001):
    margin = reward(D, w)                  # r_w - r_l for every comparison
    loss = -np.log(sigmoid(margin)).mean()
    grad = -((1 - sigmoid(margin))[:, None] * D).mean(axis=0)
    if step in (0, 10, 50, 200, 1000):
        print(f"step {step:4d}  loss {loss:.3f}  w = {np.round(w, 2)}")
    w -= eta * grad

print("hidden w =", w_true)
print("picks predicted, fitted w:", (reward(D, w) > 0).mean())
print("picks predicted, hidden w:", (reward(D, w_true) > 0).mean())
```

```
step    0  loss 0.693  w = [0. 0. 0. 0.]
step   10  loss 0.548  w = [ 0.68  0.26  0.22 -0.35]
step   50  loss 0.478  w = [ 1.6   0.63  0.31 -0.79]
step  200  loss 0.472  w = [ 2.    0.8   0.35 -0.98]
step 1000  loss 0.472  w = [ 2.02  0.81  0.35 -0.99]
hidden w = [ 2.   0.8  0.3 -1. ]
picks predicted, fitted w: 0.772
picks predicted, hidden w: 0.771
```

By step 200, three of the four fitted weights are within 0.02 of the hidden ones, and the biggest miss is length (0.35 against 0.3).
The fitted weights call 77.2% of the picks and the hidden ones 77.1%: the simulated people are noisy, and the fit edges past the truth by matching this sample's noise.

## Where this goes next

- Unit 12: step 3 pushes the policy toward whatever $r_\phi$ scores highest, and $r_\phi$ only learned from comparisons between $\pi_0$'s answers. Unit 12 follows the reward model's scores as the policy moves away from those answers, and looks at what the KL penalty does about it.
- My post [Your reward model is an attack surface](../../posts/your-reward-model-is-an-attack-surface/) covers the same ground from the security side.

## Self-test

<details>
<summary>A reward model scores answer A at 1.5 and answer D at −1.7. Under Bradley–Terry, how often do people pick A over D?</summary>

The gap is $1.5 - (-1.7) = 3.2$, and $\sigma(3.2) = 0.96$, so 96% of the time.

</details>

<details>
<summary>Why does a reward of 0 mean nothing on its own?</summary>

Bradley–Terry only uses the gap between two rewards.
Adding the same constant to every reward leaves every probability and every loss unchanged, so training can't pin down where 0 is.

</details>

<details>
<summary>In the first widget, where is the orange push biggest, and why does that make sense?</summary>

At the far left, where the gap is very negative.
There the model was nearly sure people would pick the other answer, so it was badly wrong, and the push approaches 1.
On the far right the model already expected the pick and the push is close to 0.

</details>

<details>
<summary>In the second widget, why doesn't even the hidden-rewards line call 100% of picks?</summary>

The simulated people pick at random with probability σ of the hidden gap.
B beats A 29% of the time even though A's hidden reward is higher, and any rewards that rank A above B get those picks wrong.
Averaged over the six pairs, the hidden rewards call about 83%.

</details>

<details>
<summary>Why is the KL penalty measured against π₀ and not some other model?</summary>

Every answer the reward model learned from was written by $\pi_0$, so answers that look like $\pi_0$'s are where its scores were checked against people.
The penalty keeps the policy near there.

</details>

<details>
<summary>What does DPO use in place of a reward, and why doesn't it need Z(x)?</summary>

It uses $\beta \log(\pi_\theta(y \mid x) / \pi_0(y \mid x))$.
$Z(x)$ depends only on the prompt, both answers in a comparison share the prompt, and Bradley–Terry only uses the difference, so $\beta \log Z(x)$ cancels.

</details>

::: done-when
- Given two rewards, you can compute the Bradley–Terry probability, the loss for either pick, and the size of the push.
- You can name InstructGPT's three steps and say, for each, which model it trains and on what data.
- You can explain why each comparison moves both rewards by the same amount, and what that amount depends on.
- You can write the DPO loss and say what plays the part of the reward.
:::

## Go deeper

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT), section 3.5: the reward model loss (equation 1) and the RL objective with its per-token KL penalty (equation 2).
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), section 4: the derivation sketched above, and the DPO gradient, which weights each comparison by how wrong the implied reward was, like the push in the first widget.
