---
title: Evaluating models
subtitle: Count what the model gets right and wrong, then decide which mistakes matter and how far to trust the count.
unit: 7
section: Classical ML
minutes: 55
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'In this unit''s phishing filter, what does precision measure?'
    answer: 'Of the emails it flagged, the share that really were phishing'
    wrong: ['Of the phishing emails, the share it flagged', 'Of all the emails, the share it labeled correctly', 'Of the legitimate emails, the share it left alone']
    why: 'Precision is $TP / (TP + FP)$: the denominator is everything the filter flagged. The share of phishing emails it flagged is recall.'
  - kind: term
    q: 'What does recall measure?'
    answer: 'Of the examples that really are positive, the share the model caught'
    wrong: ['Of the examples the model called positive, the share that were right', 'Of all the examples, the share the model got right', 'How close the model''s scores are to 0 or 1']
    why: 'Recall is $TP / (TP + FN)$: the denominator is every real positive, caught or missed.'
  - kind: term
    q: 'For a phishing filter, what is a false positive?'
    answer: 'A legitimate email that got flagged as phishing'
    wrong: ['A phishing email that got through unflagged', 'A phishing email that got flagged', 'A legitimate email that got through unflagged']
    why: '"Positive" is the filter''s call (phishing) and "false" means the call was wrong, so the email was legitimate.'
  - kind: term
    q: 'What does it mean for a classifier to be calibrated?'
    answer: 'Among examples it scores around $p$, about a fraction $p$ are positive'
    wrong: ['It ranks every positive example above every negative one', 'It scores the same accuracy on the validation and test sets', 'Its scores are spread evenly between 0 and 1']
    why: 'Calibration is about whether the probabilities mean what they say. Ranking well is a different property: the stretched filter in the unit ranks exactly like the calibrated one.'
  - kind: term
    q: 'What is F1?'
    answer: '$2PR / (P + R)$, which stays close to the smaller of precision and recall'
    wrong: ['$(P + R) / 2$, the plain average of precision and recall', '$P \cdot R$, the product of precision and recall', '$(TP + TN) / N$, the share of examples labeled correctly']
    why: 'F1 is the harmonic mean of precision $P$ and recall $R$. A filter that flags everything has $P = 0.10$, $R = 1$ and F1 $= 0.18$, where the plain average would say 0.55.'
  - kind: apply
    q: 'A test set has 200 emails and 4 of them are phishing. What accuracy does a filter that never flags anything get?'
    answer: '$0.98$'
    wrong: ['$0.02$', '$0.96$', '$0$']
    why: 'It''s right about all 196 legitimate emails and wrong about the 4 phishing ones, so $196 / 200 = 0.98$, while catching nothing.'
  - kind: apply
    q: 'A filter flags 20 emails and 12 of them are phishing. The test set has 30 phishing emails. What are its precision and recall?'
    answer: 'Precision 0.60, recall 0.40'
    wrong: ['Precision 0.40, recall 0.60', 'Precision 0.60, recall 0.67', 'Precision 0.40, recall 0.67']
    why: 'Precision is $12 / 20 = 0.60$ (of what it flagged) and recall is $12 / 30 = 0.40$ (of all the phishing).'
  - kind: apply
    q: 'A model has precision 0.5 and recall 1.0. What is its F1?'
    answer: 'About $0.67$'
    wrong: ['$0.75$', '$0.5$', '$1.0$']
    why: '$2 \cdot 0.5 \cdot 1.0 / (0.5 + 1.0) = 1 / 1.5 \approx 0.67$. The 0.75 is the plain average, which F1 deliberately isn''t.'
  - kind: apply
    q: 'You lower the threshold $t$ from 0.5 to 0.2 without retraining. What happens?'
    answer: 'Recall goes up or stays the same, and precision usually drops'
    wrong: ['Precision and recall both go up', 'Precision goes up and recall drops', 'Nothing changes, because the model''s scores are the same']
    why: 'A lower $t$ flags every email the old one did plus some more. Recall can only grow, and the extra flags are often legitimate emails, which pulls precision down.'
  - kind: apply
    q: 'Of 200 emails a filter scored around 0.9, 120 were phishing. What does that say about its calibration there?'
    answer: 'It''s overconfident: it says 0.9, and 0.6 are phishing'
    wrong: ['It''s underconfident: it says 0.9, and 0.6 are phishing', 'It''s calibrated, because most of them were phishing', 'You can''t tell without its accuracy']
    why: '$120 / 200 = 0.6$, well below the 0.9 it claimed. Scores further from 0.5 than the truth is what overconfident means.'
  - kind: apply
    q: 'A model scores 0.90 accuracy on 100 test examples. Roughly what is the standard error of that number?'
    answer: '$0.03$'
    wrong: ['$0.09$', '$0.003$', '$0.3$']
    why: '$\sqrt{0.9 \cdot 0.1 / 100} = \sqrt{0.0009} = 0.03$, so another 100 examples could easily give 0.87 or 0.93.'
  - kind: apply
    q: 'You want to halve the standard error of a test-set accuracy. How much bigger does the test set need to be?'
    answer: '4 times'
    wrong: ['2 times', '8 times', '$\sqrt{2}$ times']
    why: 'The standard error shrinks like $1 / \sqrt{n}$, so halving it takes $n$ times 4.'
---

A trained classifier gives you a probability for every example, and sooner or later someone asks how good it is.
The answer has to come from data the model never trained on ([unit 6](../06-generalization/#train-validation-and-test)), and it has to be a number that tracks what you care about.
The obvious number, the share of answers that are right, can give a model that does nothing a score of 90%.

## Accuracy, and the filter that never flags anything

Take a phishing filter built like the spam filter in [unit 5](../05-classification/#from-a-score-to-a-probability): it reads an email and outputs $p$, its probability that the email is phishing.
It calls an email phishing when $p \ge 0.5$.
The test set is 100 emails it never saw in training, and 10 of them are phishing.

*Accuracy* is the share of the test set it labels correctly:

$$
\text{accuracy} = \frac{\text{number labeled correctly}}{N}
$$

$N$ is the number of emails in the test set, here 100.
The filter gets 96 right, so its accuracy is 0.96.

Now take a filter that never flags anything.
It's right about all 90 legitimate emails and wrong about all 10 phishing ones, so its accuracy is 0.90.
It catches no phishing at all and still scores within 6 points of the real filter.
When one class is rare, accuracy mostly measures how you handle the common one.
With 1 phishing email in 1,000, the do-nothing filter would score 0.999.

## The confusion matrix

To see which mistakes a filter makes, sort its answers into four groups.
Phishing is the *positive* class here, since it's the one we're looking for.

- A *true positive* (TP) is phishing, and flagged.
- A *false positive* (FP) is legitimate, but flagged anyway.
- A *false negative* (FN) is phishing, but missed.
- A *true negative* (TN) is legitimate, and left alone.

Laid out as a grid, the four counts are the *confusion matrix*.
For the filter at $p \ge 0.5$:

<figure>
<pre>                     flagged        not flagged
<span class="orange">  phishing (10)      TP   7         FN   3</span>
  legit    (90)      FP   1         TN  89</pre>
<figcaption>The confusion matrix for the 100 test emails. Rows are what each email really is, columns are what the filter said.</figcaption>
</figure>

Accuracy is the two correct cells over everything, $(TP + TN) / N = (7 + 89) / 100 = 0.96$.
The two wrong cells cost different things.
A false negative is a phishing email sitting in someone's inbox.
A false positive is a real email stuck in quarantine.
Two numbers keep them apart:

$$
\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}
$$

Both have the true positives on top and differ in what they divide by.
Precision divides by everything the filter flagged, so it answers "when the filter flags an email, how often is it really phishing?"
Here that's $7 / (7 + 1) = 0.875$.
Recall divides by every email that really is phishing, so it answers "of the phishing emails, how many did the filter catch?"
Here that's $7 / (7 + 3) = 0.70$.
Neither one uses TN, so the 89 easy legitimate emails can't pad them.
The never-flag filter has recall $0 / 10 = 0$, and its precision is $0 / 0$, undefined, because it never flagged anything.

When you need a single number, the usual choice is F1:

$$
F_1 = \frac{2PR}{P + R}
$$

$P$ is precision and $R$ is recall.
This is their *harmonic mean*, and unlike the plain average $(P + R)/2$ it stays close to whichever of the two is smaller, so it's only high when both are.
For the filter, $2 \cdot 0.875 \cdot 0.70 / (0.875 + 0.70) = 0.78$.
A filter that flags every email has recall 1 and precision $10 / 100 = 0.10$.
The plain average of those is 0.55, which flatters a useless filter; F1 is 0.18.
A filter that catches nothing gets an F1 of 0.

## Moving the threshold

The 0.5 cutoff was a choice.
[Unit 5](../05-classification/#the-decision-boundary-is-a-line) puts it at $p = 0.5$ because that's where the score $w \cdot x + b$ crosses zero, but nothing else is special about it.
Call the cutoff $t$, the *threshold*, and flag an email when $p \ge t$.

Lowering $t$ flags more emails.
Phishing emails with middling scores get caught, so recall goes up.
More legitimate emails get flagged too, so precision usually goes down.
At $t = 0.2$ the filter catches all 10 phishing emails (recall 1.00), but 5 of its 15 flags are wrong (precision 0.67).
At $t = 0.8$ every flag is right (precision 1.00), and it catches only 3 (recall 0.30).

<figure class="widget" id="threshold">
<figcaption>Left: the 100 test emails sorted by the filter's score p, phishing in orange. Filled dots are flagged (p ≥ t). Right: precision (blue), recall (green) and F1 (dashed) at every threshold, drawn sideways so the threshold line crosses all three at the current t. Drag t and watch which dots change state.</figcaption>
</figure>

Which $t$ to use depends on what each mistake costs you.
If a phishing email in an inbox is much worse than a newsletter in quarantine, you want a low $t$ and accept the lower precision.
On these 100 emails F1 is highest, 0.90, for any $t$ from 0.32 to 0.36.
Choosing $t$ that way and then reporting 0.90 would be tuning on the test set, which turns it into a validation set ([unit 6](../06-generalization/#train-validation-and-test)).
Pick $t$ on the validation set, then report what the test set says at that $t$.

## Calibration

Precision and recall only look at which side of $t$ each score lands on.
Calibration looks at the scores themselves: when the filter says 0.8, is the email phishing about 80% of the time?
A model is *calibrated* if, among all the examples it scores near some value $p$, a fraction close to $p$ really are positive.

To check, put the examples into bins by score (0 to 0.1, 0.1 to 0.2, and so on) and compare each bin's average score with the share of its examples that are actually positive.
Plotting one against the other gives a *reliability diagram*.
A calibrated model's points sit on the diagonal, where the two are equal.
That needs many examples per bin, so the worked example below uses 10,000 emails from the same simulation.
The filter scored 122 of them between 0.8 and 0.9; their average score was 0.85, and 85% of them were phishing.

Now make the filter overconfident.
[Unit 5](../05-classification/#from-a-score-to-a-probability) computes $p = \sigma(z)$, where $z = w \cdot x + b$ is the raw score and $\sigma$ squashes it into a probability.
Report $\sigma(2z)$ instead: doubling $z$ pushes every probability further toward 0 or 1, while keeping the emails in the same order.
Because the order is the same, any threshold on the old scores has a matching threshold on the new ones that flags exactly the same emails, with the same precision and recall.
Only the probabilities changed.
In the 0.8 to 0.9 bin the stretched filter has 88 emails averaging 0.85, and only 64% of them are phishing.

One summary number for this is the *expected calibration error*:

$$
\text{ECE} = \sum_{m} \frac{n_m}{N} \, \big| \, \text{phish}_m - \text{score}_m \big|
$$

The sum runs over the bins $m$.
$\text{score}_m$ is the bin's average score and $\text{phish}_m$ is the share of its emails that are phishing, so the difference is that bin's gap (the orange bars in the widget below).
Each gap is weighted by $n_m / N$, the share of all $N$ emails that land in bin $m$, so a bin with 5 emails counts for little.
A perfectly calibrated model has ECE 0.

<figure class="widget" id="reliability">
<figcaption>Each blue dot is one bin of scores: across is the bin's average score, up is the share of its emails that were phishing. The dashed diagonal is perfect calibration and the orange bars are the gaps. The stretch k multiplies z before the sigmoid: k = 1 is the calibrated filter, k = 2 the overconfident one from the text. "New 10,000 emails" redraws the simulation, so you can see how much each bin moves by chance.</figcaption>
</figure>

Calibration matters whenever someone uses $p$ as a probability: a person deciding how much to trust "92% phishing", or a threshold worked out from the cost of each mistake.
[Guo et al.](https://arxiv.org/abs/1706.04599) found modern neural networks poorly calibrated and their ResNet overconfident, like the stretched filter.
Their fix, *temperature scaling*, divides every $z$ by one number fitted on the validation set; for the stretched filter that number would be 2.

## How much to trust one number

The 0.96 came from 100 emails, and a different 100 would have given a different number.
Each email is either labeled right (1) or wrong (0), so accuracy is an average of 100 samples.
[Unit 3](../03-probability/#expectation-spread-and-sampling) showed that an average like that misses the true value by about one sample's standard deviation divided by the square root of the number of samples.
A single right-or-wrong result with true accuracy $a$ is 1 with probability $a$ and 0 otherwise, so its variance is $a(1-a)$ and its standard deviation is $\sqrt{a(1-a)}$.
For a test set of $n$ examples that gives:

$$
\text{SE} = \sqrt{\frac{a(1-a)}{n}}
$$

SE is the *standard error*, the typical distance between the accuracy you measure and the one you'd get on unlimited data.
The $a(1-a)$ is largest at $a = 0.5$ and shrinks to 0 for a model that's always right or always wrong.
The $n$ underneath means a test set 4 times as big halves the error.

For $a = 0.94$ and $n = 100$ the SE is about 0.024.
The worked example checks this by splitting its 10,000 emails into 100 test sets of 100.
Their accuracies run from 0.87 to 0.98 with a standard deviation of 0.024, and 96 of the 100 land within two standard errors of the accuracy on all 10,000, which is 0.939.
So the test set's 0.96 was a lucky draw.
Two filters that score 0.94 and 0.96 on the same 100 emails differ by less than one standard error, and that test can't tell them apart.

Recall is noisier still, because only the 10 phishing emails count toward it.
With $a = 0.70$ and $n = 10$ the same formula gives about 0.14.

## A score needs a reference point

An accuracy of 0.96 or an F1 of 0.78 means little on its own.
The never-flag filter at 0.90 is one reference point, the score you get for doing nothing.
Others are the model you're replacing, and people doing the same job on the same examples.
A *benchmark* fixes the test set, the metric and the rules, so scores from different models can be compared, and it only works while its test examples stay out of everyone's training data ([unit 6](../06-generalization/#leakage-and-contamination)).

I built [EchoBench](https://www.netspi.com/blog/technical-blog/ai-ml-pentesting/introducing-echobench-a-human-calibrated-benchmark-for-autonomous-pentesting/) around that problem, for autonomous pentesting agents.
Part of its score is strict F1 on the findings an agent reports.
Precision is the share of its reported findings that hold up, recall is the share of the known findings it reproduces, and "strict" means a finding only counts with specific evidence such as the endpoint and role involved.
The reference is a cohort of associate pentesters who tested the same five web applications.
Their F1 was 71.2% on one application and 38.9% on another, so an agent with 50% F1 is behind the people on the first and ahead of them on the second.
EchoBench divides the agent's F1 by the people's F1 on the same application, so 1.0 means the agent matched them.

## Worked example

```python
import numpy as np

rng = np.random.default_rng(145)
z = rng.normal(-5.0, 3.5, 10_000)          # the model's logit for each email
p = 1 / (1 + np.exp(-z))                   # its probability that the email is phishing
y = (rng.random(10_000) < p).astype(int)   # true label, 1 = phishing (drawn so p is calibrated)

def counts(p, y, t):
    pred = p >= t
    return (np.sum(pred & (y == 1)), np.sum(pred & (y == 0)),
            np.sum(~pred & (y == 1)), np.sum(~pred & (y == 0)))

# The first 100 emails are the test set (and the widget's dots).
test_p, test_y = np.round(p[:100], 2), y[:100]
print("phishing:", test_y.sum(), "of 100   always-no accuracy:", np.mean(test_y == 0))
for t in (0.2, 0.5, 0.8):
    tp, fp, fn, tn = counts(test_p, test_y, t)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"t = {t}  TP {tp:2d}  FP {fp:2d}  FN {fn:2d}  TN {tn:2d}  accuracy {(tp + tn) / 100:.2f}"
          f"  precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}")

# The same model on 100 different test sets of 100 emails each.
accs = np.array([np.mean((p[i:i + 100] >= 0.5) == y[i:i + 100]) for i in range(0, 10_000, 100)])
print("accuracy over 100 test sets: min", accs.min(), " max", accs.max(), " std", accs.std().round(3))
print("accuracy on all 10,000:", np.mean((p >= 0.5) == y))

# Calibration: of the emails scored 0.8 to 0.9, how many are phishing?
over = 1 / (1 + np.exp(-2 * z))            # same ranking, twice the logit: overconfident
for name, q in (("calibrated", p), ("overconfident", over)):
    inbin = (q >= 0.8) & (q < 0.9)
    print(f"{name:13s}  emails scored 0.8-0.9: {inbin.sum():3d}  mean score {q[inbin].mean():.2f}"
          f"  actually phishing {y[inbin].mean():.2f}")
```

```
phishing: 10 of 100   always-no accuracy: 0.9
t = 0.2  TP 10  FP  5  FN  0  TN 85  accuracy 0.95  precision 0.67  recall 1.00  F1 0.80
t = 0.5  TP  7  FP  1  FN  3  TN 89  accuracy 0.96  precision 0.88  recall 0.70  F1 0.78
t = 0.8  TP  3  FP  0  FN  7  TN 90  accuracy 0.93  precision 1.00  recall 0.30  F1 0.46
accuracy over 100 test sets: min 0.87  max 0.98  std 0.024
accuracy on all 10,000: 0.939
calibrated     emails scored 0.8-0.9: 122  mean score 0.85  actually phishing 0.85
overconfident  emails scored 0.8-0.9:  88  mean score 0.85  actually phishing 0.64
```

Accuracy barely moves between $t = 0.2$ and $t = 0.8$ (0.95 to 0.93) while recall falls from 1.00 to 0.30, a change accuracy hides.
The labels were drawn from the model's own probabilities, so the calibrated line comes out at 0.85 and 0.85, and the spread from 0.87 to 0.98 is one model on test sets that differ only by chance.

## Where this shows up later

- Unit 11: cross-entropy and perplexity score a language model's probabilities directly, and cross-entropy is lowest, on average, when those probabilities are the true ones.
- Unit 14: temperature divides the logits by $T$ before the softmax, the same operation as temperature scaling here.
- Unit 16: a reward model is checked by how often it agrees with held-out human picks between two answers. InstructGPT's labelers agreed with each other 72.6% of the time, which is the reference point for that number.
- Unit 17: once a measured score becomes the thing you optimize, it stops tracking what it was built to measure.

## Self-test

<details>
<summary>A test set has 1,000 emails, 20 of them phishing. The filter flags 25 emails, and 15 of those are phishing. What are its accuracy, precision, recall and F1?</summary>

TP = 15, FP = 10, FN = 5, TN = 970.
Accuracy is $(15 + 970) / 1000 = 0.985$, precision is $15 / 25 = 0.60$, recall is $15 / 20 = 0.75$, and F1 is $2 \cdot 0.60 \cdot 0.75 / 1.35 \approx 0.67$.
The never-flag filter would get 0.98 accuracy on the same set.

</details>

<details>
<summary>Why do precision, recall and F1 leave out the true negatives?</summary>

When positives are rare, true negatives are most of the test set, and counting them would let a filter look good just by leaving the easy emails alone.
That's what happens to accuracy.

</details>

<details>
<summary>In the threshold widget, move t from 0.50 down to 0.20. Which dots change state, and what happens to precision and recall?</summary>

Three phishing emails (scores 0.22, 0.36 and 0.42) and four legitimate ones (0.20, 0.27, 0.29 and 0.31) become flagged.
TP goes from 7 to 10 and FP from 1 to 5, so recall rises from 0.70 to 1.00 and precision falls from 0.88 to 0.67.

</details>

<details>
<summary>In the calibration widget at k = 2, which side of the diagonal are the dots on, and why does precision at a matching threshold stay the same?</summary>

Toward the right, from about 0.7 up, the dots sit well below the diagonal (fewer phishing than claimed), and toward the left, below about 0.3, they sit above it.
Near 0.5 doubling $z$ barely moves the scores, so the gaps there are small and chance can put a dot on either side.
Doubling $z$ keeps the emails in the same order, so for any threshold there's a matching one that flags the same emails.

</details>

<details>
<summary>Filter A scores 0.95 accuracy and filter B scores 0.93, each on 200 test emails. Can you say A is better?</summary>

Not from that.
With $a \approx 0.94$ and $n = 200$, the standard error is $\sqrt{0.94 \cdot 0.06 / 200} \approx 0.017$, and the gap of 0.02 is barely more than one standard error.
You'd want a bigger test set before choosing.

</details>

::: done-when
- You can fill in a confusion matrix from a list of labels and predictions and compute accuracy, precision, recall and F1 from it.
- You can say which way precision and recall move when the threshold goes down, and why.
- You can check calibration by binning scores and say whether a model is over- or underconfident.
- You can estimate the standard error of an accuracy and say whether a gap between two models is bigger than it.
:::

## Go deeper

- Google's Machine Learning Crash Course on [accuracy, precision and recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) and [thresholds and the confusion matrix](https://developers.google.com/machine-learning/crash-course/classification/thresholding). A second short explanation of the same metrics, with exercises.
- [Guo et al., "On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599). Figure 1 and section 2 for reliability diagrams and ECE, section 4.2 for temperature scaling.
- [Miller, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640). The standard error from this unit applied to language model benchmarks, plus how to measure the difference between two models.
