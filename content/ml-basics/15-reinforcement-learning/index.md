---
title: Policies and reinforcement learning
subtitle: Learning from a score when nobody can write down the right answer.
unit: 15
section: Alignment
minutes: 70
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'In reinforcement learning, what is the return $G$?'
    answer: 'The total reward collected over one episode'
    wrong: ['The reward from the last step only', 'The reward minus the baseline', 'The average reward per step across many episodes']
    why: '$G = r_1 + r_2 + \cdots + r_T$, summed over the $T$ steps of one episode (leaving out discounting). The reward minus the baseline is the advantage.'
  - kind: term
    q: 'What does $\pi_\theta(a \mid s)$ mean?'
    answer: 'The probability that the policy takes action $a$ in state $s$'
    wrong: ['The reward for taking action $a$ in state $s$', 'The state the environment moves to after action $a$', 'The expected return from state $s$']
    why: 'The policy is a probability distribution over actions, and the agent acts by sampling from it. The expected return from a state is what a value function predicts.'
  - kind: term
    q: 'What is the advantage in REINFORCE with a baseline?'
    answer: 'How much better the reward was than the baseline, $r - b$'
    wrong: ['The reward divided by the baseline, $r / b$', 'The gap between the best arm''s average and the pulled arm''s', 'The total reward collected over the episode']
    why: 'The update multiplies $\nabla \log \pi_\theta(a)$ by $r - b$, so better-than-usual actions get pushed up and worse ones get pushed down.'
  - kind: term
    q: 'What is an episode?'
    answer: 'One run of the agent-environment loop, from start to finish'
    wrong: ['One gradient step on the policy''s parameters', 'One state and the action taken in it', 'All the runs collected for one training batch']
    why: 'In chess it''s one game. For the bandit it''s a single pull, and for a language model it''s one whole response.'
  - kind: term
    q: 'What does PPO''s clipped objective limit?'
    answer: 'How far one round of updates can move an action''s probability from the policy that collected the samples'
    wrong: ['How large a single reward is allowed to be', 'How far the policy can drift from $\pi_0$ over the whole of training', 'How many actions an episode is allowed to have']
    why: 'Past about 20% (the paper''s 0.2), moving a probability further earns nothing, which makes it safe to reuse a batch for several steps. Staying close to $\pi_0$ is the KL penalty''s job in unit 16.'
  - kind: apply
    q: 'Equal probabilities over three arms, $\eta = 0.1$, no baseline. You pull A and get a reward of 3. How much does $\theta_A$ change?'
    answer: '$+0.2$'
    wrong: ['$+0.3$', '$+0.1$', '$-0.1$']
    why: 'The pulled arm moves by $\eta\, r\, (1 - \pi(A)) = 0.1 \cdot 3 \cdot 2/3 = 0.2$. The other two each drop by 0.1.'
  - kind: apply
    q: 'Same start ($\eta = 0.1$, equal probabilities), but now with baseline $b = 5$. You pull A and get 4. How much does $\theta_A$ change?'
    answer: '$-0.067$'
    wrong: ['$+0.267$', '$-0.033$', '$+0.067$']
    why: 'The advantage is $4 - 5 = -1$, so $\theta_A$ moves by $0.1 \cdot (-1) \cdot 2/3 \approx -0.067$. Without the baseline it would have gone up by 0.267.'
  - kind: apply
    q: 'The bandit''s arms pay 4, 5 and 6 on average. What is $J(\theta)$ for a policy with probabilities $(0.2, 0.3, 0.5)$?'
    answer: '$5.3$'
    wrong: ['$4.7$', '$5.0$', '$6.0$']
    why: '$J = 4 \cdot 0.2 + 5 \cdot 0.3 + 6 \cdot 0.5 = 0.8 + 1.5 + 3 = 5.3$. Swapping the probabilities of A and C gives 4.7.'
  - kind: apply
    q: 'A softmax policy has probabilities $(0.2, 0.3, 0.5)$ and you pull B. What is $\nabla \log \pi_\theta(B)$ with respect to the logits?'
    answer: '$(-0.2, 0.7, -0.5)$'
    wrong: ['$(0.2, -0.7, 0.5)$', '$(0, 1, 0)$', '$(-0.2, 0.3, -0.5)$']
    why: 'It''s the one-hot vector for B minus the probabilities: $(0, 1, 0) - (0.2, 0.3, 0.5)$. The flipped sign is unit 11''s cross-entropy gradient.'
  - kind: apply
    q: 'The running-average baseline is $b = 5$ and the next pull pays 7. What is $b$ after the update $b \leftarrow b + 0.1\,(r - b)$?'
    answer: '$5.2$'
    wrong: ['$7.0$', '$6.0$', '$5.02$']
    why: '$5 + 0.1 \cdot (7 - 5) = 5.2$. The baseline moves a tenth of the way toward each new reward.'
  - kind: apply
    q: 'A language model writes a three-token response with per-token probabilities 0.5, 0.2 and 0.1. What is $\pi_\theta(y \mid x)$?'
    answer: '$0.01$'
    wrong: ['$0.8$', '$0.267$', '$0.1$']
    why: 'A response''s probability is the product of its token probabilities, $0.5 \cdot 0.2 \cdot 0.1 = 0.01$, so its log is the sum of the three logs.'
  - kind: apply
    q: 'A language model is trained with RL, and the reward model scores each 50-token response once at the end. What counts as one action?'
    answer: 'Picking one token'
    wrong: ['Writing the whole response', 'Reading the prompt', 'The reward model''s score']
    why: 'The state is the prompt plus the tokens so far and the action is the next token. The whole response is the episode.'
---

::: why
You can teach a dog to sit without ever showing it the "right answer".
You wait for it to try something, and you hand over a treat when it does well.
Reinforcement learning trains a model the same way.
Unit 11 could copy the correct next token from the text, but here the model only gets a score for what it did.
The simplest method that learns from a score is called REINFORCE, and a slot machine with three arms is enough to build it.
Unit 16 uses the same idea to train a language model against a reward model, which is the RL in RLHF.
:::

## Agents, rewards and returns

Reinforcement learning (RL) describes a loop between two parts.
The *agent* is the part that learns and makes choices.
The *environment* is everything else.

The loop goes like this.
The environment shows the agent a *state* $s$, which is the situation right now.
The agent picks an *action* $a$.
The environment answers with a *reward* $r$ (a single number) and a new state.

One run of this loop from start to finish is an *episode*.
The *return* is the total reward collected along the way:

$$
G = r_1 + r_2 + \cdots + r_T
$$

::: formula
$r_1$ is the reward after the first step, $r_2$ after the second, and so on, and $T$ is the number of steps in the episode.
:::

Many setups shrink later rewards by a discount factor a little below 1 before adding them up.
I'll leave that out here.

In chess the state is the board, an action is a move, and an episode is one game.
The reward can be 0 after every move and +1 or −1 when the game ends.
Then the whole return arrives at the very end, and the agent has to work out which of its moves earned it.

The running example is the smallest version of all: a three-armed bandit.
The name comes from slot machines, nicknamed one-armed bandits.
There's one state and three arms, A, B and C.
An episode is a single pull.

Each arm pays a random reward.
The average is 4 for A, 5 for B and 6 for C, plus noise from a [normal distribution](../03-probability/#random-variables-and-distributions) with standard deviation 1.
The agent doesn't know those averages.
All it ever sees is the arm it pulled and the number that came back.

<figure class="widget demo" id="demo-rl-loop"></figure>

## A policy picks actions with probabilities

The *policy* is the part of the agent that decides.
It's written $\pi_\theta(a \mid s)$.
Read that as the [conditional probability](../03-probability/#conditional-probability-and-bayes-rule) of taking action $a$ given state $s$.
The $\theta$ stands for the policy's parameters, the numbers training will change.

The agent acts by sampling from those probabilities.
So the same state can lead to different actions.

The bandit has only one state, so I'll drop $s$ and write $\pi_\theta(a)$.
The parameters are three logits $\theta = (\theta_A, \theta_B, \theta_C)$, one per arm.
The policy is their softmax, as in [unit 11](../11-next-token-prediction/#softmax-turns-scores-into-probabilities):

$$
\pi_\theta(a) = \frac{e^{\theta_a}}{e^{\theta_A} + e^{\theta_B} + e^{\theta_C}}
$$

::: formula
The numerator is $e$ raised to the chosen arm's logit, and the denominator is the same for all three arms, so the three probabilities add up to 1.
:::

Starting from $\theta = (0, 0, 0)$, each arm gets probability 1/3.

A language model is a policy too.
The state is the prompt plus the tokens written so far.
The action is the next token.
And $\pi_\theta$ is the softmax output from unit 11.

An episode is one whole response $y = (y_1, \dots, y_m)$ to a prompt $x$.
Its probability is the product of the per-token probabilities.
A response of three tokens with probabilities 0.5, 0.2 and 0.1 has probability $0.5 \times 0.2 \times 0.1 = 0.01$.
The log of a product is a sum, as in [unit 3](../03-probability/#why-models-work-in-log-probabilities):

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{m} \log \pi_\theta(y_t \mid x, y_1, \dots, y_{t-1})
$$

::: formula
On the left is the log-probability of the whole response $y$ given the prompt $x$.
On the right, term $t$ is the log-probability of token $y_t$ given the prompt and the $t - 1$ tokens before it, and the sum runs over all $m$ tokens.
:::

In RLHF the reward usually comes once, after the last token, from a reward model (unit 16).

Whatever the setting, the goal is the same.
Choose $\theta$ so the [expected](../03-probability/#expectation-spread-and-sampling) return is as high as possible:

$$
J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[\,r\,]
$$

::: formula
$\mathbb{E}_{a \sim \pi_\theta}[\,r\,]$ is unit 3's expectation of the reward $r$, with the subscript $a \sim \pi_\theta$ saying which distribution to average over: actions $a$ drawn from the policy.
:::

$J(\theta)$ is the *objective*, the number training tries to raise.
For the bandit that's $J(\theta) = 4\,\pi_\theta(A) + 5\,\pi_\theta(B) + 6\,\pi_\theta(C)$.
At the start it's 5.
The best possible is 6, by always pulling C.

## Why you can't backprop through the reward

In unit 11 the loss was a formula in the logits.
So the [chain rule](../02-gradients/#the-rules-you-need) could carry its gradient back to every parameter.
Here two links in that chain are missing.

The first is the sampling.
The action is a discrete pick: B or C, with nothing in between.
Nudge $\theta$ a tiny bit and the policy still picks B, or it suddenly picks C.
There's no slope to follow.

The second is the environment.
The reward comes from a slot machine, a game engine or a person, and none of them hand you a derivative.
Even when the reward is a neural network, as in unit 16, it scores a response made of sampled tokens.
So the first problem is still there.

One thing does depend smoothly on $\theta$: the probability of each action.
So REINFORCE skips the question of how the reward changes when $\theta$ changes.
It asks which probabilities to raise.

## REINFORCE: raise the probability of what did well

Write $R(a)$ for the average reward of action $a$ (4, 5 or 6 for the bandit).
Then the expected return and its gradient are

$$
J(\theta) = \sum_a \pi_\theta(a)\, R(a), \qquad \nabla J = \sum_a R(a)\, \nabla \pi_\theta(a)
$$

::: formula
The left equation is $J$ written as unit 3's weighted sum: each action's average reward times the probability of picking it.
The right one takes the gradient $\nabla$ (the vector of derivatives with respect to each entry of $\theta$) of that sum; $R(a)$ doesn't depend on $\theta$, so only $\pi_\theta(a)$ gets differentiated.
:::

That's exact, but it needs $R(a)$ for every action, and the agent doesn't know any of them.
One line of calculus gets around this.
The derivative of $\log \pi$ is $\nabla \pi / \pi$, so $\nabla \pi = \pi \, \nabla \log \pi$.
Substitute that in:

$$
\begin{aligned}
\nabla J &= \sum_a \pi_\theta(a)\, R(a)\, \nabla \log \pi_\theta(a) \\
&= \mathbb{E}_{a \sim \pi_\theta}\big[\, r\, \nabla \log \pi_\theta(a) \,\big]
\end{aligned}
$$

::: formula
The first line is the exact sum after substituting.
Its weights are the policy's own probabilities $\pi_\theta(a)$, and the second line writes that weighted sum as an expectation.
:::

So the gradient is an average over the actions the policy itself picks.
You can estimate an average like that by sampling, the [Monte Carlo estimate](../03-probability/#expectation-spread-and-sampling) from unit 3.
Pull an arm and see the reward $r$.
Then $r\, \nabla \log \pi_\theta(a)$ is one noisy estimate of the gradient.
(Using the reward you saw in place of $R(a)$ is fine, because on average it equals $R(a)$.)

[Unit 4](../04-gradient-descent/#walk-downhill) stepped downhill on a loss with $\theta \leftarrow \theta - \eta \nabla L$.
This time we want $J$ to go up, so the step goes uphill:

$$
\theta \leftarrow \theta + \eta\, r\, \nabla \log \pi_\theta(a)
$$

::: formula
$\eta$ is the learning rate, as in unit 4, $r$ is the reward from this pull, and $\nabla \log \pi_\theta(a)$ points in the direction that raises the probability of the action just taken.
:::

Each pull raises the probability of the action just taken, in proportion to its reward.
A high reward pushes it up hard, and a small one barely moves it.
That's *REINFORCE* (Williams, 1992).

For a softmax policy, $\nabla \log \pi_\theta(a)$ is the one-hot vector for $a$ minus the probabilities $\pi_\theta$.
It's unit 11's [cross-entropy gradient](../11-next-token-prediction/#the-gradient-is-p-y) $p - \text{onehot}$ with the sign flipped, since cross-entropy is $-\log p$.
So the pulled arm's logit goes up by $\eta\, r\, (1 - \pi_\theta(a))$, and every other arm $k$ goes down by $\eta\, r\, \pi_\theta(k)$.

You can also read the same update as ordinary next-token training from unit 11.
The training text is a sample the model wrote itself, and the loss is multiplied by the reward.

Here is one step with real numbers.
Start at equal probabilities with the learning rate $\eta = 0.1$, pull B, and get a reward of 5.2.
$\theta_B$ goes up by $0.1 \times 5.2 \times 2/3 \approx 0.35$.
$\theta_A$ and $\theta_C$ go down by $0.1 \times 5.2 \times 1/3 \approx 0.17$.
The probabilities become (0.27, 0.46, 0.27).

<figure class="widget demo" id="demo-reinforce"></figure>

B isn't the best arm, but its probability went from 0.33 to 0.46 anyway.
Every arm pays around 5, so every pull pushes its own arm up.
REINFORCE still works on average, because pulls of C push a bit harder than pulls of A.
But each single step is mostly noise sitting on top of that small difference.

## A baseline: compare with the average

The fix is to subtract a number $b$, called the *baseline*, from the reward before using it:

$$
\theta \leftarrow \theta + \eta\, (r - b)\, \nabla \log \pi_\theta(a)
$$

It's the same update with $r$ replaced by $r - b$.
The difference $r - b$ is called the *advantage*: how much better this pull did than usual.

Take $b = 5$.
The pull of B that paid 5.2 has an advantage of 0.2 and barely moves anything (the probabilities become (0.331, 0.338, 0.331)).
A pull of A that pays 4.1 has an advantage of −0.9 and pushes A down.
A pull of C that pays 6 pushes C up.

<figure class="widget demo" id="demo-baseline"></figure>

On average, the baseline adds nothing to the update, because the probabilities always add up to 1:

$$
\begin{aligned}
\mathbb{E}_{a \sim \pi_\theta}\big[\, b\, \nabla \log \pi_\theta(a) \,\big]
&= b \sum_a \nabla \pi_\theta(a) \\
&= b\, \nabla \Big( \sum_a \pi_\theta(a) \Big) = b\, \nabla 1 = 0
\end{aligned}
$$

::: formula
Read it one step at a time.
The baseline's part of the update, averaged over actions, is $\sum_a \pi_\theta(a)\, b\, \nabla \log \pi_\theta(a)$, and $\pi\, \nabla \log \pi = \nabla \pi$ turns that into $b \sum_a \nabla \pi_\theta(a)$.
The gradient of a sum is the sum of the gradients, so that's $b$ times the gradient of the total probability.
The total is always 1, a constant, and the gradient of a constant is 0.
:::

So $b$ can be any number that doesn't depend on which action was taken.
The average direction of training stays the same.
What changes is how noisy each single estimate is, in [unit 3's](../03-probability/#expectation-spread-and-sampling) sense of standard deviation.
In the worked example below, subtracting the average reward makes each one-pull estimate four to five times less spread out.

A simple choice, and the one used here, is a running average of recent rewards.
After each pull, $b \leftarrow b + 0.1\,(r - b)$, starting from the first reward.
Larger setups usually train a second network instead, called a *value function*.
It predicts the expected return from the current state and serves as $b$.

The widget runs the whole thing for 300 pulls at a time, with the baseline on or off.

<figure class="widget" id="bandit">
<figcaption>Each bar is the policy's probability of pulling that arm, with the arm's average payout underneath. The lines on the right follow those probabilities through one run of 300 pulls, then a new run starts. The number above the outlined bar is the advantage of the last pull. With the baseline off it's always positive and some runs settle on A or B; switch it on and watch the tally.</figcaption>
</figure>

Plain REINFORCE uses each batch of samples for one gradient step and then throws it away.
One unlucky large step can also wreck a policy that took a long time to train.

*PPO*, proximal policy optimization ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)), keeps the same update direction and adds a clip.
Within one round of updates, the clipped objective stops rewarding any change that moves an action's probability more than about 20% away from where it started.
"Where it started" means its probability under the policy that collected the samples, and 0.2 is the setting the paper suggests.
That makes it safe to reuse each batch for several gradient steps.
InstructGPT used PPO for its RL step, and unit 16 picks up from there.

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

Both sets of one-pull estimates average out to the exact gradient $(-1/3, 0, 1/3)$.
With $b = 5$, each one has a spread of about 0.55, down from 2.2 to 2.6.

The second part runs 200 training runs of 300 pulls each.
Every run with the baseline ended with more than 0.9 of its probability on C.
Without it, 18% of runs ended with C below 0.1, stuck on a worse arm.

::: real-world
- **Game-playing systems.** DeepMind's AlphaGo improved by playing Go against itself. The reward was winning or losing the game, and policy gradients like the one in this unit were part of its training.
- **Chat assistants.** OpenAI's InstructGPT, a forerunner of ChatGPT, was tuned with PPO. People compared pairs of responses, a reward model learned from those comparisons, and RL raised the probability of high-scoring responses (unit 16).
- **Reward hacking.** RL raises whatever number you give it. In a well-known OpenAI example, a boat-racing agent learned to circle the same spot collecting bonus points and never finished the race. Security and safety teams check what a reward can be tricked into paying for.
:::

## Where this shows up later

- Unit 16: the policy is a language model, an episode is one response, the reward comes from a reward model trained on people's comparisons, and the update is PPO with a KL penalty that keeps the policy close to $\pi_0$.
- Unit 17: REINFORCE raises the probability of whatever scores well. When the score comes from a learned reward model, that includes responses that only score well because the reward model is wrong about them.

::: review
Agent and environment
:   The agent learns and picks actions. The environment is everything else, and it hands back rewards.

State, action, reward
:   The situation the agent sees, the choice it makes, and the single number it gets back.

Episode
:   One run of the loop from start to finish: one game, one pull, or one whole response.

Return, $G$
:   The total reward collected over one episode.

Policy, $\pi_\theta(a \mid s)$
:   The probability of taking action $a$ in state $s$. The agent acts by sampling from it.

Objective, $J(\theta)$
:   The expected return under the policy. Training tries to raise it.

REINFORCE
:   Sample an action, see the reward, and step $\theta$ by $\eta\, r\, \nabla \log \pi_\theta(a)$. Actions that scored well become more likely.

Baseline, $b$
:   A number subtracted from the reward, such as the average recent reward. It cuts the noise and leaves the average update unchanged.

Advantage, $r - b$
:   How much better the reward was than the baseline.

Value function
:   A second network that predicts the expected return from a state. Larger setups use it as the baseline.

PPO
:   REINFORCE's update direction plus a clip that limits how far one round of updates can move the policy.
:::

::: say
Reinforcement learning is for when there's a score but no right answer to copy.
A policy gives each action a probability, and the agent acts by sampling.
REINFORCE tries an action, looks at the reward, and raises that action's probability in proportion to the reward.
Subtracting a baseline means only better-than-usual actions get pushed up, which makes training far less noisy.
A language model is a policy whose actions are tokens, and unit 16 trains it this way.
:::

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
