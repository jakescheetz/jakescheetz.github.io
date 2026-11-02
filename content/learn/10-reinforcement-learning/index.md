---
title: Policies and reinforcement learning
subtitle: Learning from a score when nobody can write down the right answer.
unit: 10
minutes: 70
scripts: [widgets.js]
---

In unit 6 the model learned from text, so for every position the correct next token was right there to copy.
Reinforcement learning is for when there is no correct answer to copy, only a score for what the model did.
The simplest algorithm that learns from a score is REINFORCE, and a three-armed slot machine is enough to build it.
Unit 11 uses the same idea to train a language model against a reward model, which is the RL in RLHF.

## Agents, rewards and returns

Reinforcement learning describes a loop between two parts.
The agent is the part that learns and makes choices.
The environment is everything else.
It shows the agent a state $s$, the agent picks an action $a$, and the environment answers with a reward $r$ (a single number) and a new state.

One run of this loop from start to finish is an episode, and the return is the total reward collected along the way:

$$
G = r_1 + r_2 + \cdots + r_T
$$

where $T$ is the number of steps in the episode.
Many setups shrink later rewards by a discount factor a little below 1 before adding them up; I'll leave that out here.

In chess the state is the board, an action is a move, and an episode is one game.
The reward can be 0 after every move and +1 or −1 when the game ends.
Then the whole return arrives at the very end, and the agent has to work out which of its moves earned it.

The running example is the smallest version of all: a three-armed bandit.
The name comes from slot machines, nicknamed one-armed bandits.
There's one state, three arms A, B and C, and an episode is a single pull.
Each arm pays a random reward: on average 4 for A, 5 for B and 6 for C, plus noise from a normal distribution with standard deviation 1.
The agent doesn't know those averages.
All it ever sees is the arm it pulled and the number that came back.

## A policy picks actions with probabilities

The policy is the part of the agent that decides.
Written $\pi_\theta(a \mid s)$, it's the probability of taking action $a$ in state $s$, and $\theta$ is its parameters.
The agent acts by sampling from those probabilities, so the same state can lead to different actions.

The bandit has only one state, so I'll drop $s$ and write $\pi_\theta(a)$.
The parameters are three logits $\theta = (\theta_A, \theta_B, \theta_C)$, one per arm, and the policy is their softmax, as in unit 6:

$$
\pi_\theta(a) = \frac{e^{\theta_a}}{e^{\theta_A} + e^{\theta_B} + e^{\theta_C}}
$$

Starting from $\theta = (0, 0, 0)$, each arm gets probability 1/3.

A language model is a policy too.
The state is the prompt plus the tokens written so far, the action is the next token, and $\pi_\theta$ is the softmax output from unit 6.
An episode is one whole response $y = (y_1, \dots, y_m)$ to a prompt $x$.
Its probability is the product of the per-token probabilities, so its log is a sum:

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{m} \log \pi_\theta(y_t \mid x, y_1, \dots, y_{t-1})
$$

In RLHF the reward usually comes once, after the last token, from a reward model (unit 11).

Whatever the setting, the goal is to choose $\theta$ so the expected return is as high as possible:

$$
J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[\,r\,]
$$

For the bandit that's $J(\theta) = 4\,\pi_\theta(A) + 5\,\pi_\theta(B) + 6\,\pi_\theta(C)$.
At the start it's 5, and the best possible is 6, by always pulling C.

## Why you can't backprop through the reward

In unit 6 the loss was a formula in the logits, so the chain rule could carry its gradient back to every parameter.
Here two links in that chain are missing.

The first is the sampling.
The action is a discrete pick.
Nudge $\theta$ a tiny bit and the policy still picks B, or it suddenly picks C, and there's no slope in between to follow.

The second is the environment.
The reward comes from a slot machine, a game engine or a person, and none of them hand you a derivative.
Even when the reward is a neural network, as in unit 11, it scores a response made of sampled tokens, so the first problem is still there.

What does depend smoothly on $\theta$ is the probability of each action.
So instead of asking how the reward changes when $\theta$ changes, REINFORCE asks which probabilities to raise.

## REINFORCE: raise the probability of what did well

Write $R(a)$ for the average reward of action $a$ (4, 5 or 6 for the bandit).
Then the expected return and its gradient are

$$
J(\theta) = \sum_a \pi_\theta(a)\, R(a), \qquad \nabla J = \sum_a R(a)\, \nabla \pi_\theta(a)
$$

That's exact, but it needs $R(a)$ for every action, and the agent doesn't know any of them.
The way around it is one line of calculus.
The derivative of $\log \pi$ is $\nabla \pi / \pi$, so $\nabla \pi = \pi \, \nabla \log \pi$, and substituting that in gives

$$
\nabla J = \sum_a \pi_\theta(a)\, R(a)\, \nabla \log \pi_\theta(a) = \mathbb{E}_{a \sim \pi_\theta}\big[\, r\, \nabla \log \pi_\theta(a) \,\big]
$$

The sum is now weighted by the policy's own probabilities, which makes it an average over the actions the policy itself picks.
An average like that can be estimated by sampling.
Pull an arm, see the reward $r$, and $r\, \nabla \log \pi_\theta(a)$ is one noisy estimate of the gradient.
(Using the reward you saw in place of $R(a)$ is fine, because on average it equals $R(a)$.)

Unit 3 stepped downhill on a loss.
This time we want $J$ to go up, so the step goes uphill:

$$
\theta \leftarrow \theta + \eta\, r\, \nabla \log \pi_\theta(a)
$$

That's REINFORCE (Williams, 1992).

For a softmax policy, $\nabla \log \pi_\theta(a)$ is the one-hot vector for $a$ minus the probabilities $\pi_\theta$.
It's unit 6's cross-entropy gradient $p - \text{onehot}$ with the sign flipped, since cross-entropy is $-\log p$.
So the pulled arm's logit goes up by $\eta\, r\, (1 - \pi_\theta(a))$, and every other arm $k$ goes down by $\eta\, r\, \pi_\theta(k)$.

The same update also reads as ordinary next-token training from unit 6, on a sample the model wrote itself, with the loss multiplied by the reward.

One step with real numbers.
Start at equal probabilities with $\eta = 0.1$, pull B, and get a reward of 5.2.
$\theta_B$ goes up by $0.1 \times 5.2 \times 2/3 \approx 0.35$, $\theta_A$ and $\theta_C$ go down by $0.1 \times 5.2 \times 1/3 \approx 0.17$, and the probabilities become (0.27, 0.46, 0.27).

B isn't the best arm, but its probability went from 0.33 to 0.46 anyway.
Every arm pays around 5, so every pull pushes its own arm up.
REINFORCE still works on average, because pulls of C push a bit harder than pulls of A.
But each single step is mostly noise sitting on top of that small difference.

## A baseline: compare with the average

The fix is to subtract a baseline $b$ from the reward before using it:

$$
\theta \leftarrow \theta + \eta\, (r - b)\, \nabla \log \pi_\theta(a)
$$

The difference $r - b$ is called the advantage: how much better this pull did than usual.
With $b = 5$, the pull of B that paid 5.2 has an advantage of 0.2 and barely moves anything (the probabilities become (0.331, 0.338, 0.331)).
A pull of A that pays 4.1 has an advantage of −0.9 and pushes A down, and a pull of C that pays 6 pushes C up.

On average, the baseline adds nothing to the update.
Its share of the expected gradient is

$$
\mathbb{E}_{a \sim \pi_\theta}\big[\, b\, \nabla \log \pi_\theta(a) \,\big] = b \sum_a \nabla \pi_\theta(a) = b\, \nabla \Big( \sum_a \pi_\theta(a) \Big) = b\, \nabla 1 = 0
$$

because the probabilities always add up to 1.
So $b$ can be any number that doesn't depend on which action was taken, and the average direction of training stays the same.
What changes is how noisy each single estimate is.
In the worked example below, subtracting the average reward makes each one-pull estimate four to five times less spread out.

A simple choice, and the one used here, is a running average of recent rewards: after each pull, $b \leftarrow b + 0.1\,(r - b)$, starting from the first reward.
Larger setups usually train a second network instead, a value function, that predicts the expected return from the current state and serves as $b$.

<figure class="widget" id="bandit">
<figcaption>Each bar is the policy's probability of pulling that arm, with the arm's average payout underneath. The lines on the right follow those probabilities through one run of 300 pulls, then a new run starts. The number above the outlined bar is the advantage of the last pull. With the baseline off it's always positive and some runs settle on A or B; switch it on and watch the tally.</figcaption>
</figure>

Plain REINFORCE uses each batch of samples for one gradient step and then throws it away, and one unlucky large step can wreck a policy that took a long time to train.
PPO, proximal policy optimization ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)), keeps the same update direction and adds a clip.
Within one round of updates, the clipped objective stops rewarding any change that moves an action's probability more than about 20% away from where it was under the policy that collected the samples (0.2 is the setting the paper suggests).
That makes it safe to reuse each batch for several gradient steps.
InstructGPT used PPO for its RL step, and unit 11 picks up from there.

## Worked example

```python
import numpy as np

means = np.array([4.0, 5.0, 6.0])   # average reward of arms A, B, C

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

# 1. One-sample gradient estimates at the starting policy (all logits 0).
rng = np.random.default_rng(0)
p = softmax(np.zeros(3))
print("exact gradient:", (p * (means - p @ means)).round(3))
for b in (0.0, p @ means):
    a = rng.choice(3, size=100_000, p=p)
    r = means[a] + rng.normal(size=a.size)
    g = (r - b)[:, None] * (np.eye(3)[a] - p)    # (r - b) * grad log pi(a)
    print(f"b = {b}: mean {g.mean(0).round(3)}, spread {g.std(0).round(2)}")

# 2. REINFORCE: sample an arm, see its reward, nudge the logits.
def train(use_baseline, seed, steps=300, eta=0.1):
    rng = np.random.default_rng(seed)
    theta = np.zeros(3)
    b = None
    for _ in range(steps):
        p = softmax(theta)
        a = rng.choice(3, p=p)
        r = means[a] + rng.normal()
        if b is None:
            b = r
        advantage = r - b if use_baseline else r
        theta += eta * advantage * (np.eye(3)[a] - p)
        b += 0.1 * (r - b)                       # running average reward
    return softmax(theta)

print("one run, baseline:", train(True, seed=0).round(3))
for use_baseline in (False, True):
    final = np.array([train(use_baseline, seed) for seed in range(200)])
    best = (final[:, 2] > 0.9).mean()
    wrong = (final[:, 2] < 0.1).mean()
    print(f"baseline {use_baseline}: C above 0.9 in {best:.0%} of runs, C below 0.1 in {wrong:.0%}")
```

```
exact gradient: [-0.333  0.     0.333]
b = 0.0: mean [-0.343  0.009  0.334], spread [2.17 2.42 2.64]
b = 5.0: mean [-0.334 -0.001  0.334], spread [0.55 0.54 0.55]
one run, baseline: [0.005 0.015 0.98 ]
baseline False: C above 0.9 in 78% of runs, C below 0.1 in 18%
baseline True: C above 0.9 in 100% of runs, C below 0.1 in 0%
```

Both sets of one-pull estimates average out to the exact gradient $(-1/3, 0, 1/3)$, but with $b = 5$ each one has a spread of about 0.55 instead of 2.2 to 2.6.
Over 200 training runs of 300 pulls, every run with the baseline ended with more than 0.9 of its probability on C, while without it 18% of runs ended with C below 0.1, stuck on a worse arm.

## Where this shows up later

- Unit 11: the policy is a language model, an episode is one response, the reward comes from a reward model trained on people's comparisons, and the update is PPO with a KL penalty that keeps the policy close to $\pi_0$.
- Unit 12: REINFORCE raises the probability of whatever scores well. When the score comes from a learned reward model, that includes responses that only score well because the reward model is wrong about them.

## Self-test

<details>
<summary>Starting from equal probabilities with η = 0.1 and no baseline, you pull C and get a reward of 6. How do the three logits change?</summary>

$\theta_C$ goes up by $0.1 \times 6 \times 2/3 = 0.4$.
$\theta_A$ and $\theta_B$ each go down by $0.1 \times 6 \times 1/3 = 0.2$.

</details>

<details>
<summary>Why can't you backpropagate from the reward straight to the policy's parameters?</summary>

The action is sampled, which is a discrete pick with no slope, and the reward comes from an environment that gives you no derivative.
The probabilities of the actions do depend smoothly on $\theta$, so REINFORCE changes those instead.

</details>

<details>
<summary>In the widget with the baseline off, the number above the pulled bar is always positive. Why, and what does that do to a run?</summary>

Without a baseline the advantage is the reward itself, and every arm pays around 4 to 6.
So every pull raises the probability of the arm that was pulled.
If A or B gets lucky early, it gets pulled more, raised more, and C can end up pulled so rarely that it never catches up.
With the baseline on, the numbers are positive and negative, and only better-than-usual pulls get pushed up.

</details>

<details>
<summary>Why doesn't subtracting a baseline change the average update?</summary>

The baseline's share of the expected gradient is $b$ times the gradient of the sum of all the probabilities.
That sum is always 1, so its gradient is 0.

</details>

<details>
<summary>You add 100 to every reward. What happens to training with and without the running-average baseline?</summary>

Without the baseline, every update gets about 20 times bigger and still always points toward the pulled arm, so the difference of 1 between arms drowns in noise.
The expected gradient is unchanged, but you'd need a far smaller learning rate to get anywhere.
With the baseline, $b$ starts at the first reward and moves up with it, so the advantages come out exactly as before and training behaves the same.

</details>

<details>
<summary>For a language model writing a response, what are the state, the action, the episode and the reward?</summary>

The state is the prompt plus the tokens written so far, the action is the next token, the episode is the whole response, and the reward is usually a single score for the finished response.

</details>

::: done-when
- You can name the state, action, episode and reward for the bandit and for a language model writing a response.
- You can do one REINFORCE step on a softmax policy by hand, with and without a baseline.
- You can show in one line why a baseline leaves the average gradient unchanged.
- You've run the widget with the baseline off and on and can explain the difference in the tally.
:::

## Go deeper

- Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (2nd edition, free online). Section 2.8, "Gradient Bandit Algorithms", is this unit's bandit with an average-reward baseline, and its Figure 2.5 compares with and without the baseline on rewards near +4. Sections 13.3 and 13.4 do REINFORCE with and without a baseline for full episodes.
- OpenAI Spinning Up, [Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html): the same derivation for episodes with many steps, and why a learned value function is the usual baseline.
- [The PPO paper](https://arxiv.org/abs/1707.06347), sections 1 to 3: what goes wrong with plain policy gradient steps and how the clipped objective limits them.
