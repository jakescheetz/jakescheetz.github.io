---
title: Sampling
subtitle: How a list of next-token probabilities becomes text, and the settings that decide which text.
unit: 9
minutes: 45
scripts: [widgets.js]
---

A language model doesn't output text.
At each position it outputs a probability for every token in its vocabulary (unit 6), and something outside the model has to pick one.
That picking step is the loop that turns one prediction into a whole response, and it's where the settings every chat API exposes (temperature, top-k, top-p) come in.
The last three units depend on it, because RL training learns from responses the model samples itself.

## The generation loop

Start with a prompt, split into tokens $x_1, \dots, x_n$.
Run the model on it and take the logits at the last position: $V$ numbers $z$, one per vocabulary token.
Softmax turns them into probabilities $p$.
Pick one token using $p$, append it as $x_{n+1}$, and run the model again on the longer sequence.
Repeat until the model picks an end-of-text token or you hit a length limit.

<figure>
<pre>step  context                             pick
 1    The cat sat on the              ──► <span class="orange">mat</span>
 2    The cat sat on the <span class="orange">mat</span>          ──► <span class="orange">.</span>
 3    The cat sat on the <span class="orange">mat</span><span class="orange">.</span>         ──► <span class="orange">It</span>
 4    The cat sat on the <span class="orange">mat</span><span class="orange">.</span> <span class="orange">It</span>      ──► <span class="orange">purred</span></pre>
<figcaption>Each pick is appended to the context for the next step. The picks here are made up for illustration.</figcaption>
</figure>

Each new token costs one forward pass, which is why long answers stream out a piece at a time.
Real implementations save the keys and values (unit 7) from earlier positions, so each step only processes the newest token and the output matches a full rerun.

The model's job is the same at every step: map a context to $p$.
The picking rule is a separate piece called the decoding strategy.
It has no trained parameters, and you choose it when you generate.

For the examples below, the context is "The cat sat on the", and I've cut the vocabulary down to eight candidates with made-up logits.

| token | logit $z$ | $p$ at $T = 1$ |
|---|---:|---:|
| mat | 3.0 | 0.444 |
| floor | 2.3 | 0.221 |
| bed | 1.9 | 0.148 |
| couch | 1.4 | 0.090 |
| roof | 0.8 | 0.049 |
| table | 0.3 | 0.030 |
| dog | −0.5 | 0.013 |
| moon | −1.5 | 0.005 |

A real model spreads its probability over its whole vocabulary (50,257 tokens for GPT-2); eight keeps the numbers readable.

## Greedy decoding, and why it repeats itself

The simplest rule is to always take the most likely token, $\arg\max_i p_i$.
This is greedy decoding, and for the example it picks "mat".

Greedy is deterministic: the same context always gets the same continuation.
That's handy when you want reproducible output and bad for long text.
Holtzman et al. found that each time a phrase repeats, the model gives the next repetition a higher probability.
Once a loop has started, repeating is likely to stay the top choice, and greedy always takes the top choice, so it stays in the loop.
The Hugging Face post linked at the end shows GPT-2 doing this with greedy decoding from the prompt "I enjoy walking with my cute dog".
The start of its output:

> I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to walk with my dog. I'm not sure if I'll ever be able to walk with my dog.

Holtzman et al. measured it on 5,000 continuations from GPT-2 Large, each up to 200 tokens.
With greedy decoding, 73.66% of them ended in a phrase repeated at least three times.
Human-written text did that 0.28% of the time.

Human text also isn't the most probable text.
Scored token by token, a human-written passage keeps landing on tokens the model rated unlikely, so a decoder that always takes the top token writes more predictably than any person.

## Temperature

The alternative is sampling: draw the next token at random, with probabilities $p$.
For the example, that gives "mat" 44.4% of the time, "floor" 22.1%, and "moon" about once in 200 draws.
Because it doesn't always take the top token, sampling can leave a loop.

Temperature $T$ controls how spread out $p$ is before the draw.
Divide every logit by $T$, then take the softmax as usual:

$$
p_i = \frac{e^{z_i / T}}{\sum_{j=1}^{V} e^{z_j / T}}
$$

At $T = 1$ you get the model's own distribution.
What the division does is easiest to see in the ratio of two probabilities, where the denominator cancels:

$$
\frac{p_i}{p_j} = e^{(z_i - z_j) / T}
$$

"mat" and "floor" are 0.7 apart in logit.
At $T = 1$, mat is $e^{0.7} \approx 2.01$ times as likely as floor.
At $T = 0.5$ the gap counts double, $e^{1.4} \approx 4.06$.
At $T = 2$ it counts half, $e^{0.35} \approx 1.42$.

So $T < 1$ sharpens the distribution toward the top token, and $T > 1$ flattens it.
As $T$ goes to 0, all the probability moves onto the top token and sampling turns into greedy decoding, which is why a temperature of 0 is usually taken to mean greedy.
As $T$ grows, every token heads toward probability $1/V$.
The order never changes: dividing by a positive number keeps the logits in the same order, so the most likely token stays the most likely at every temperature.

## Cutting off the tail: top-k and top-p

Lowering the temperature is a blunt way to avoid bad tokens, because it also shrinks the variety among the good ones.
Holtzman et al. found that sampling at temperatures below 0.9 sharply increased repetition.

The bad tokens live in the tail.
In the example, the four least likely tokens hold 0.097 of the probability at $T = 1$, so about one draw in ten lands on roof, table, dog or moon.
A real model has tens of thousands of tail tokens, and the paper found that together they get more probability than they deserve (it calls this the unreliable tail).
A bad draw stays in the context, and every later token is predicted with it there; in the paper, pure sampling from GPT-2 drifts away from the prompt and even invents a word, "umidauda".

Top-k sampling keeps only the $k$ most likely tokens, calls that set $S$, and renormalizes so the kept probabilities add up to 1 again.
The distribution you actually sample from is $q$:

$$
q_i = \frac{p_i}{\sum_{j \in S} p_j} \;\text{ for } i \in S, \qquad q_i = 0 \;\text{ otherwise}
$$

With $k = 3$ the example keeps mat, floor and bed, which hold 0.813 together, and renormalizing gives 0.547, 0.271 and 0.182.

A fixed $k$ ignores how sure the model is.
After "The United", one token deserves nearly all the probability, and $k = 40$ still keeps 39 that don't.
In a context with hundreds of reasonable continuations, $k = 40$ throws some of them away.

Top-p, which Holtzman et al. call nucleus sampling, picks the size of $S$ from the distribution itself.
Sort the tokens most likely first and keep the smallest set whose probabilities reach a threshold, here 0.9:

$$
\sum_{i \in S} p_i \ge 0.9
$$

then renormalize the same way.
For the example at $T = 1$ the running totals are 0.444, 0.665, 0.813, 0.903, and the fourth token takes the total past 0.9, so top-p 0.9 keeps mat, floor, bed and couch.
When the model is confident, one or two tokens reach 0.9 on their own and that's all that survives.
When it's unsure, the set grows; in the paper it ranged from one token to about a thousand.

The widget applies temperature first, then top-k, then top-p on what top-k left.
Order changes the result.
A higher temperature flattens $p$, so it takes more tokens to reach 0.9: at $T = 2$, top-p 0.9 keeps six of the eight.

<figure class="widget" id="sampler">
<figcaption>Blue bars are the distribution actually sampled from, after temperature, the top-k and top-p cuts, and renormalizing. Greyed tokens are cut, and their faint bars show the probability they lost. The tick on a blue bar marks where it ended before renormalizing. "Sample 20" draws 20 tokens from the blue bars; moving a slider clears the counts.</figcaption>
</figure>

## Same model, different behavior

Everything above happens after the model has produced $p$.
The weights are identical whether you decode greedily or sample at $T = 1.2$ with top-p 0.95; only the draw changes.
Holtzman et al. make this point in their abstract: decoding strategy alone can change the quality of the text a lot, from exactly the same model.

So any behavior you observe belongs to the model plus its decoding settings.
Red-teaming a model runs straight into this.
If you test a prompt once at $T = 0$, you've seen one path through the distribution.
If a bad answer has probability 0.02 per attempt at $T = 1$, greedy decoding may never show it, while someone who can resample will see it about twice per hundred tries.

## Worked example

```python
import numpy as np

tokens = ["mat", "floor", "bed", "couch", "roof", "table", "dog", "moon"]
z = np.array([3.0, 2.3, 1.9, 1.4, 0.8, 0.3, -0.5, -1.5])  # logits after "The cat sat on the"

def softmax(z, T=1.0):
    e = np.exp((z - z.max()) / T)  # subtracting the max changes nothing but avoids overflow
    return e / e.sum()

for T in [0.5, 1.0, 2.0]:
    print(f"T = {T}:", np.round(softmax(z, T), 3))

def top_p(p, threshold):
    order = np.argsort(p)[::-1]                    # most likely first
    cum = np.cumsum(p[order])
    n_keep = np.searchsorted(cum, threshold) + 1   # smallest set that reaches the threshold
    kept = np.zeros_like(p)
    kept[order[:n_keep]] = p[order[:n_keep]]
    return kept / kept.sum()

p = softmax(z, T=1.0)
q = top_p(p, 0.9)
print("running total:", np.round(np.cumsum(p), 3))
print("top-p 0.9:    ", np.round(q, 3))

rng = np.random.default_rng(0)
draws = rng.choice(len(tokens), size=20, p=q)
print(" ".join(tokens[i] for i in draws))
print(np.bincount(draws, minlength=len(tokens)))
print("greedy picks:", tokens[np.argmax(p)])
```

```
T = 0.5: [0.706 0.174 0.078 0.029 0.009 0.003 0.001 0.   ]
T = 1.0: [0.444 0.221 0.148 0.09  0.049 0.03  0.013 0.005]
T = 2.0: [0.278 0.196 0.16  0.125 0.092 0.072 0.048 0.029]
running total: [0.444 0.665 0.813 0.903 0.952 0.982 0.995 1.   ]
top-p 0.9:     [0.492 0.244 0.164 0.099 0.    0.    0.    0.   ]
floor mat mat mat bed couch floor floor floor couch bed mat bed mat floor mat bed floor mat mat
[8 6 4 2 0 0 0 0]
greedy picks: mat
```

From $T = 0.5$ to $T = 2$, "mat" falls from 0.706 to 0.278 and "moon" rises from below 0.001 (printed as 0) to 0.029, with the order unchanged.
Top-p 0.9 keeps four tokens and divides each by 0.903; in the 20 draws "mat" came up 8 times against an expected $20 \times 0.492 \approx 9.8$, ordinary noise for 20 samples.

## Where this shows up later

- Unit 10: a language model is a policy, and sampling is how it takes actions. The policy gradient needs samples drawn from the policy itself, so RL training samples rather than decoding greedily.
- Unit 11: people compare two responses sampled from $\pi_0$ for the same prompt. Greedy decoding would only ever give one.
- Unit 12: best-of-n is a decoding strategy too: sample $n$ responses and keep the one the reward model scores highest.

## Self-test

<details>
<summary>At T = 1, mat and floor have probabilities 0.444 and 0.221. How many times as likely is mat at T = 0.5, and why?</summary>

About 4.06 times.
The ratio is $e^{(z_i - z_j)/T}$, and halving $T$ doubles the logit gap from 0.7 to 1.4, so the ratio goes from $e^{0.7} \approx 2.01$ to $e^{1.4} \approx 4.06$.

</details>

<details>
<summary>Why does greedy decoding get stuck repeating itself?</summary>

It always takes the top token, so the same context always gives the same continuation.
The probability of a phrase goes up each time it repeats, so once a loop starts, repeating tends to stay the top choice and greedy keeps taking it.

</details>

<details>
<summary>The model puts 0.99 on one token and spreads the rest over thousands. What do top-k with k = 40 and top-p 0.9 each keep?</summary>

Top-p keeps only the one token, because it reaches 0.9 alone.
Top-k keeps 40 tokens no matter what, so 39 tail tokens stay in the running.

</details>

<details>
<summary>In the widget, set top-p to 0.90 at temperature 1, then raise the temperature to 2. Why do more tokens survive?</summary>

The higher temperature flattens the distribution, so the top tokens hold less of the probability and it takes more of them to reach 0.9.
At $T = 1$ four tokens get there (0.903); at $T = 2$ it takes six (0.923).

</details>

<details>
<summary>In the widget at the default settings, how often should "moon" show up in a batch of 20 samples?</summary>

Its probability is 0.005, so the expected count is about 0.1 per batch.
The chance of seeing it at least once in 20 draws is $1 - 0.995^{20} \approx 0.095$, so about one batch in ten.

</details>

::: done-when
- You can compute softmax at a given temperature for a short logit vector and say what happens as $T$ goes to 0 and as it grows.
- Given a list of probabilities and a top-p value, you can say which tokens survive and what their renormalized probabilities are.
- You can explain why greedy decoding falls into loops and why sampling from the full distribution drifts off topic.
- You can explain why a result about a model's behavior needs its decoding settings next to it.
:::

## Go deeper

- [Holtzman et al., The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). Section 3 defines top-k, temperature and nucleus sampling, and Table 1 compares them on GPT-2 Large.
- [Hugging Face, How to generate text](https://huggingface.co/blog/how-to-generate). Greedy, beam search, top-k and top-p run on real GPT-2, with the outputs printed so you can see each one's failure.
