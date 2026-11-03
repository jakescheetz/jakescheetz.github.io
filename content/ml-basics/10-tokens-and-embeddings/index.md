---
title: Tokens and embeddings
subtitle: How text turns into integers, and each integer into a vector.
unit: 10
section: Language models
minutes: 60
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'What is a token ID?'
    answer: 'The number of a piece of text in the tokenizer''s fixed vocabulary'
    wrong: ['The position of a token within the prompt', 'The number of characters inside a token', 'The vector the model looks up for a token']
    why: 'The vocabulary is a numbered list of $V$ pieces of text, and a tokenizer outputs the number of each piece it finds. The vector comes after, from the embedding table.'
  - kind: term
    q: 'What does $V$ stand for?'
    answer: 'The number of tokens in the vocabulary'
    wrong: ['The number of tokens in the prompt', 'The length of each embedding vector', 'The number of merges BPE ran']
    why: 'IDs run from $0$ to $V - 1$. The prompt length is $n$ and the vector length is $d$.'
  - kind: term
    q: 'What does one byte-pair encoding merge do?'
    answer: 'Adds the most frequent neighboring pair as a new token and replaces every occurrence of it'
    wrong: ['Splits the rarest token back into its single characters', 'Adds the most frequent whole word as a new token', 'Joins the two most frequent tokens wherever they appear in a word']
    why: 'BPE counts pairs of tokens that sit next to each other, glues the most frequent pair into one token, and rewrites the corpus with it.'
  - kind: term
    q: 'What is the embedding table $E$?'
    answer: 'A $V \times d$ matrix with one learned row per token'
    wrong: ['A $d \times d$ matrix applied to every token vector', 'A $V \times d$ matrix of random numbers that stay fixed', 'A $V \times V$ matrix with one row per pair of tokens']
    why: 'Row $v$ is the vector for token $v$. The rows start random and get trained along with every other weight.'
  - kind: term
    q: 'What is the one-hot vector for token $x_i$?'
    answer: '$V$ entries, all 0 except a 1 at position $x_i$'
    wrong: ['$d$ entries, all 0 except a 1 at position $x_i$', '$V$ entries, all 1 except a 0 at position $x_i$', '$n$ entries, with a 1 wherever the token appears']
    why: 'It has one slot per vocabulary token, so multiplying it by $E$ picks out row $x_i$.'
  - kind: apply
    q: 'A byte-level BPE tokenizer starts from the 256 byte values, learns 30,000 merges and adds 2 special tokens. What is $V$?'
    answer: '$30{,}258$'
    wrong: ['$30{,}256$', '$30{,}000$', '$30{,}002$']
    why: 'Each merge adds one token on top of the 256 bytes, and the special tokens add two more: $256 + 30{,}000 + 2$. It''s the same sum that gives GPT-2 its 50,257.'
  - kind: apply
    q: 'A corpus is 40 tokens long, and the next merge''s pair appears 6 times. How long is the corpus after the merge?'
    answer: '$34$'
    wrong: ['$46$', '$28$', '$39$']
    why: 'Every occurrence turns two tokens into one, so the corpus loses one token per occurrence: $40 - 6 = 34$.'
  - kind: apply
    q: 'Using the ten merges from the worked example (e + s first, then es + t, then l + o, ...), how does "nest" tokenize?'
    answer: '"n" + "est"'
    wrong: ['"ne" + "st"', '"nest"', '"n" + "e" + "s" + "t"']
    why: 'Merges apply in the order they were learned. Merges 1 and 2 turn e, s, t into "est" before merge 5 (n + e) comes up, and by then there''s no e left next to the n.'
  - kind: apply
    q: 'A model has $V = 1{,}000$ and $d = 64$. How many numbers does its embedding table hold?'
    answer: '$64{,}000$'
    wrong: ['$1{,}064$', '$640{,}000$', '$6{,}400$']
    why: '$E$ is $V \times d$, so it holds $1{,}000 \times 64 = 64{,}000$ numbers.'
  - kind: apply
    q: '$E$ has rows $(1, 2)$, $(3, 4)$, $(5, 6)$, $(7, 8)$, numbered from 0. What is $uE$ for the one-hot $u = (0, 0, 1, 0)$?'
    answer: '$(5, 6)$'
    wrong: ['$(3, 4)$', '$(16, 20)$', '$11$']
    why: 'The 1 sits at position 2, so every row gets weight 0 except row 2, which is $(5, 6)$. The result is a whole row, not one number.'
  - kind: apply
    q: 'A training batch contains only tokens 11 and 13. Which rows of $E$ get a gradient through the lookup?'
    answer: 'Only rows 11 and 13'
    wrong: ['Every row of $E$', 'Every row except 11 and 13', 'None, since a lookup has no gradient']
    why: 'The one-hot entry $u_v$ is 0 for every token that isn''t in the batch, so those rows get no gradient through the lookup.'
  - kind: apply
    q: 'In GPT-2''s embeddings, unrelated tokens like `" cat"` and `" however"` score around 0.2. How should you read 0.55 for `" cat"` and `" dog"`?'
    answer: 'As fairly similar, since it sits well above that 0.2 background'
    wrong: ['As barely related, since it''s only about half of 1', 'As unrelated, since only scores above 0.8 count as close', 'As identical, since any positive cosine means the same direction']
    why: 'All of GPT-2''s rows lean toward one shared direction, so unrelated pairs don''t land near 0. Compare scores against each other, not against 0 or 1.'
---

A language model never sees letters.
Before any text reaches it, a tokenizer cuts the text into pieces from a fixed list and replaces each piece with its number in that list.
The model's first layer then swaps each number for a vector by looking it up in a table.
The list comes from byte-pair encoding (the method GPT-2 uses), and the table is where the vectors from unit 1 come in.

## Text becomes a list of integers

A network only does arithmetic, so text has to become numbers before anything else can happen.
The first step is to fix a *vocabulary*: a list of $V$ pieces of text, numbered $0$ to $V - 1$.
A *tokenizer* splits a string into pieces from that list and outputs their numbers, which are called *token IDs*.
A prompt becomes a sequence $x_1, \dots, x_n$, where each $x_i$ is one ID and $n$ is the number of tokens.

The open question is what the pieces are.
The two obvious answers, characters and whole words, both have a problem.

Single characters give a tiny vocabulary (a few hundred symbols, or exactly 256 if you work with raw bytes), and any string at all can be written with them.
The cost is length.
Every character takes a position, so sequences get long, and the model has to assemble every word out of letters itself.
Long sequences are expensive: attention (unit 12) compares each position with every earlier one, and a model can only read a fixed number of positions at once (1,024 for GPT-2).

Whole words give short sequences, but the vocabulary gets enormous and still misses things.
Any word that wasn't in the training text, like a typo or a new product name, has no entry, so it gets replaced by one shared "unknown" token, and the model can't tell one unknown word from another.
Related forms like "run", "runs" and "running" become three separate entries that share nothing.

Sennrich, Haddow and Birch put numbers on this for a German training corpus (Table 1 of their paper, linked at the bottom).
Split into words, it was 100 million tokens with 1.75 million distinct types.
Split into characters, it was 550 million tokens with 3,000 types.
Their subword method landed at 112 million tokens with 63,000 types, and none of their test text came out as unknown.

Subwords keep frequent words whole and split rare ones into pieces that are common.
GPT-2 stores `" tokenization"` as two tokens, `" token"` and `"ization"`, and my last name, `" Scheetz"`, as `" Sche"` and `"etz"`.
The leading space is part of the token, which I'll come back to below.

## Byte-pair encoding

*Byte-pair encoding* (BPE) builds a subword vocabulary out of training text by repeating one rule:

1. Start with a vocabulary of single characters and split every word into characters.
2. Count every pair of neighboring tokens across the corpus. A word that appears 6 times adds 6 to each pair inside it.
3. Take the most frequent pair, add the two glued together as a new token, and replace every occurrence of the pair with it.
4. Go back to step 2 until you've done as many merges as you want.

Words are counted separately, so a merge never reaches across a space.

For a corpus I'll use the toy example from the Sennrich paper: "low" 5 times, "lower" twice, "newest" 6 times and "widest" 3 times.
That's 16 words and 79 characters, using 10 distinct letters.
In the first count, e + s shows up 6 times inside "newest" and 3 times inside "widest", 9 in total.
s + t also has 9, and a tie goes to whichever pair was counted first, so "es" joins the vocabulary as its eleventh token.
The next count finds es + t, again 9 times, and "est" joins as the twelfth.

Each merge adds one token to the vocabulary and shortens the corpus by the number of times the pair appeared: 79 tokens, then 70, then 61.

<figure class="widget" id="bpe-merges">
<figcaption>The four words of the corpus with their counts. The orange outline marks the most frequent pair, which the next merge glues together; blue text marks the tokens the last merge created. Step through by hand or press play, and watch the vocabulary bar grow while the corpus bar shrinks.</figcaption>
</figure>

The number of merges is the only setting BPE has.
The vocabulary size is the number of starting symbols plus the number of merges, so picking how many merges to run is picking where you land between characters and words.
After twelve merges every word in this corpus is a single token, and you're back to a word-level vocabulary.

Tokenizing new text uses the merge list, not the corpus.
Split the new word into characters, then apply the merges in the order they were learned.
"lowest" never appeared in the corpus, but after ten merges it comes out as "low" + "est", two pieces the corpus taught.
"newer" comes out as "new" + "e" + "r", because e + r appears only twice (inside "lower") and ten merges ran out before reaching it.

## What GPT-2 does

GPT-2 runs BPE on bytes instead of characters.
Text is stored as UTF-8, where each character takes one to four bytes, so starting from the 256 possible byte values means any string in any language, emoji included, can be encoded without an unknown token.
The GPT-2 paper points out that starting from Unicode characters instead would need a base vocabulary of over 130,000.

On top of the 256 bytes, GPT-2 learned 50,000 merges and added one special token, <code style="font-variant-ligatures: none">&lt;|endoftext|&gt;</code>, that marks where one document ends and the next begins.
That's where its vocabulary size comes from:

$$
V = 256 + 50{,}000 + 1 = 50{,}257
$$

::: formula
The first term is the byte values BPE starts from, the second is one new token per merge, and the third is the end-of-text token.
:::

Before counting pairs, GPT-2 cuts text into chunks with a regular expression: runs of letters, runs of digits and runs of other symbols, each allowed one leading space, plus a few English contractions like 's and 've.
Merges never cross a chunk boundary, which stops "dog.", "dog!" and "dog?" from each becoming a token of its own (the paper's example).
Because the space rides along at the front, `" the"` (ID 262) and `"the"` (ID 1169) are different tokens, and the first three merges GPT-2 learned were space + t, space + a, and h + e.

`" strawberry"` is a single token in GPT-2, ID 41236.
The model receives that one number and nothing that spells the word out, so whatever it knows about the letters inside a token, it had to pick up indirectly from training text.
So asking GPT-2 how many r's are in "strawberry" asks about something its input doesn't contain.

## Embeddings: one row per token

A token ID is just a label.
In the toy vocabulary, ID 13 is "low" and ID 14 is "ne", and being neighbors as numbers says nothing about them.
Feeding the raw ID into a layer like $Wx + b$ from [unit 8](../08-neural-networks/#layers-and-why-the-bend-matters) would treat "ne" as a little more than "low", which means nothing.

Instead, the model keeps an *embedding table* $E$, a $V \times d$ matrix with one row per token.
Row $v$, written $E_v$, is the vector for token $v$, and $d$ is the model width, the number of entries in each vector.
Embedding a token means taking its row:

$$
h_i = E_{x_i}
$$

::: formula
$h_i$ is the vector for position $i$ of the prompt, and $E_{x_i}$ is the row whose number is that position's token ID $x_i$.
:::

A sequence of $n$ IDs becomes $n$ vectors of length $d$, and everything in units 12 and 13 works on those vectors.
For the smallest GPT-2, $d = 768$, so $E$ holds $50{,}257 \times 768 = 38{,}597{,}376$ numbers, about 31% of the model's 124 million parameters.

The lookup can also be written as a matrix multiplication ([unit 1](../01-vectors/#a-matrix-is-a-stack-of-dot-products)), which is how training sees it.
Write token $x_i$ as a *one-hot vector* $u$: $V$ entries, all 0 except a 1 at position $x_i$.
Multiplying $u$ by $E$ adds up the rows of $E$ weighted by the entries of $u$, and every row but one gets weight 0, so the result is row $x_i$:

$$
uE = \sum_{v=0}^{V-1} u_v E_v = E_{x_i}
$$

::: formula
$u$ is written as a row vector here.
The sum runs over every token $v$ in the vocabulary.
Each term is the entry $u_v$ (a 0 or a 1) times row $E_v$, so the only term that survives is the one at $v = x_i$.
:::

<figure>
<pre> id  token    u       row v of E
  0  d        0  ×  ( 0.35   0.82   0.33  -1.30)
  1  e        0  ×  ( 0.91   0.45  -0.54   0.58)
  ⋮
 12  lo       0  ×  ( 0.59   0.89   0.32  -0.82)
 13  <span class="blue">low    </span>  <span class="orange">1</span>  ×  <span class="blue">( 0.73  -0.50   0.88  -1.07)</span>
 14  ne       0  ×  ( 0.91  -0.02  -1.25  -0.31)
  ⋮
 19  widest   0  ×  (-0.35  -1.48  -0.11  -0.45)
                    ────────────────────────────
              uE =  <span class="blue">( 0.73  -0.50   0.88  -1.07)</span></pre>
<figcaption>Looking up token 13 ("low") in the 20 × 4 table from the worked example. Each row of E gets multiplied by its entry of u and the results are added up; only row 13 has a 1, so uE is row 13.</figcaption>
</figure>

Real code does the lookup by indexing (GPT-2's code uses `tf.gather`, PyTorch has `nn.Embedding`), since multiplying by 50,256 zeros is wasted work.
The one-hot view tells you how the table learns.
It's a linear layer, so backpropagation ([unit 8](../08-neural-networks/#backpropagation)) reaches it like any other weights, and since $u_v = 0$ for every token not in the batch, only the rows of tokens that appeared get a gradient through the lookup.
(GPT-2 also reuses $E$ at the output end, which unit 13 covers, and that path reaches every row.)

## Similar tokens end up close

Nobody writes the rows of $E$ by hand.
They start as small random numbers and get trained by gradient descent ([unit 4](../04-gradient-descent/#walk-downhill)) along with every other weight in the model.

Training pulls tokens that get used the same way toward similar rows.
`" Monday"` and `" Tuesday"` show up in the same kinds of sentences, so the rest of the network should react to them in nearly the same way, and giving them rows that point in nearly the same direction is a direct way to get there.
[Cosine similarity](../01-vectors/#cosine-similarity) from unit 1 measures that: the dot product of two rows divided by both their lengths, which is 1 for rows pointing the same way and 0 for perpendicular ones.

I checked this on the released GPT-2 weights.
The rows for `" Monday"` and `" Tuesday"` have a cosine similarity of 0.89, and searching all 50,257 rows for the six closest to `" Monday"` returns the other six days of the week, at 0.80 to 0.89.
The closest row to `" lowest"` is `" highest"`, at 0.80.
Opposites fill the same slots in a sentence, so they end up close too.

<figure class="widget" id="embedding-neighbors">
<figcaption>Cosine similarities between twelve rows of GPT-2 small's embedding table (each token includes its leading space). Pick a token to sort the other eleven by how close their rows are to its row.</figcaption>
</figure>

In the widget, the pairs that aren't related land between 0.12 and 0.35, not near 0.
GPT-2's rows all lean a little toward one shared direction: if you subtract the average row from every row first, `" cat"` against `" however"` drops from 0.19 to 0.07.
So read these numbers against each other: 0.55 for `" cat"` and `" dog"` is high because the background is about 0.2.

## Worked example

```python
from collections import Counter
import numpy as np

corpus = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3

# Count each word once, then split it into characters.
word_counts = Counter(corpus.split())
splits = {word: list(word) for word in word_counts}

def count_pairs():
    pairs = Counter()
    for word, count in word_counts.items():
        tokens = splits[word]
        for pair in zip(tokens, tokens[1:]):
            pairs[pair] += count
    return pairs

def merge(tokens, pair):
    out, i = [], 0
    while i < len(tokens):
        if tokens[i:i + 2] == list(pair):
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out

def corpus_length():
    return sum(len(splits[w]) * c for w, c in word_counts.items())

vocab = sorted(set(corpus.replace(" ", "")))
print(f"start: {len(vocab)} tokens ({' '.join(vocab)}), corpus is {corpus_length()} tokens")

merges = []
for step in range(10):
    pairs = count_pairs()
    best = max(pairs, key=pairs.get)  # ties go to the pair seen first
    merges.append(best)
    for word in splits:
        splits[word] = merge(splits[word], best)
    print(f"merge {step + 1:2}: {best[0]!r:7} + {best[1]!r:6} "
          f"seen {pairs[best]}x  corpus is now {corpus_length()} tokens")

def tokenize(word):
    tokens = list(word)
    for pair in merges:  # apply merges in the order they were learned
        tokens = merge(tokens, pair)
    return tokens

vocab += ["".join(pair) for pair in merges]
ids = {token: i for i, token in enumerate(vocab)}
for word in ["lowest", "newer", "widen"]:
    tokens = tokenize(word)
    print(word, "->", tokens, [ids[t] for t in tokens])

# An embedding table: one row of d = 4 numbers per token (random until trained).
E = np.random.default_rng(1).normal(size=(len(vocab), 4)).round(2)
x = [ids[t] for t in tokenize("lowest")]
print(E.shape)
print(E[x])                                          # lookup: rows 13 and 11
print(np.allclose(np.eye(len(vocab))[x] @ E, E[x]))  # same as one-hot times E
```

```
start: 10 tokens (d e i l n o r s t w), corpus is 79 tokens
merge  1: 'e'     + 's'    seen 9x  corpus is now 70 tokens
merge  2: 'es'    + 't'    seen 9x  corpus is now 61 tokens
merge  3: 'l'     + 'o'    seen 7x  corpus is now 54 tokens
merge  4: 'lo'    + 'w'    seen 7x  corpus is now 47 tokens
merge  5: 'n'     + 'e'    seen 6x  corpus is now 41 tokens
merge  6: 'ne'    + 'w'    seen 6x  corpus is now 35 tokens
merge  7: 'new'   + 'est'  seen 6x  corpus is now 29 tokens
merge  8: 'w'     + 'i'    seen 3x  corpus is now 26 tokens
merge  9: 'wi'    + 'd'    seen 3x  corpus is now 23 tokens
merge 10: 'wid'   + 'est'  seen 3x  corpus is now 20 tokens
lowest -> ['low', 'est'] [13, 11]
newer -> ['new', 'e', 'r'] [15, 1, 6]
widen -> ['wid', 'e', 'n'] [18, 1, 4]
(20, 4)
[[ 0.73 -0.5   0.88 -1.07]
 [-0.1   0.1   0.04 -0.51]]
True
```

"est" comes out of the first two merges because it sits inside two different words, and that shared piece is what lets the unseen word "lowest" come out as IDs 13 and 11.
The rest of the output is the embedding step: `E[x]` picks rows 13 and 11 of the $20 \times 4$ table, and multiplying the one-hot rows by $E$ gives the same two rows.

## Where this shows up later

- Unit 11: the model ends with one score per vocabulary entry, so GPT-2 produces 50,257 logits at every position.
- Unit 12: attention mixes the $n$ token vectors that come out of the lookup.
- Unit 13: GPT-2 adds a learned position vector to each token's row, and that sum is where the residual stream starts. At the top of the model, GPT-2 turns vectors back into logits with $E$ itself.
- Unit 14: generation picks one ID at a time, and the tokenizer turns the IDs back into text.

## Self-test

<details>
<summary>In the BPE widget, why is the first merge e + s and not l + o?</summary>

e + s appears 9 times (6 in "newest", 3 in "widest") and l + o appears 7 times (5 in "low", 2 in "lower").
s + t also has 9, but e + s gets counted first, so it wins the tie.

</details>

<details>
<summary>Each merge in the widget shrinks the corpus bar by a different amount. What sets the amount?</summary>

The number of times the merged pair appeared.
Every occurrence turns two tokens into one, so merge 3 (l + o, 7 times) takes the corpus from 61 tokens to 54.

</details>

<details>
<summary>Using the ten merges from the worked example, how does "west" tokenize?</summary>

Start from w, e, s, t.
Merge 1 makes w, es, t and merge 2 makes w, est.
No later merge matches (w + i needs an i), so the answer is "w" + "est".

</details>

<details>
<summary>Why does GPT-2 start BPE from bytes instead of characters?</summary>

Every string is a sequence of the same 256 byte values, so nothing can ever be unknown, and the base vocabulary stays at 256.
Starting from Unicode characters would need over 130,000 base symbols before the first merge.

</details>

<details>
<summary>The table E is 20 × 4 and the token is ID 11. What does the lookup return, and which one-hot vector gives the same result?</summary>

Row 11 of $E$, which is (−0.10, 0.10, 0.04, −0.51) in the worked example.
The one-hot vector has 20 entries with a 1 at position 11, and multiplying it by $E$ gives the same row.

</details>

<details>
<summary>Why can't the model just feed token IDs into its first layer as numbers?</summary>

The IDs are arbitrary labels.
Their order only reflects how the vocabulary was built, so a layer doing arithmetic on them would treat ID 14 as "a bit more" than ID 13 when the two tokens have nothing to do with each other.
A lookup gives every token its own learned vector instead.

</details>

::: done-when
- You can run the first three BPE merges by hand on a small word list with counts, and say how much each one shortens the corpus.
- Given a list of merges, you can tokenize a word that wasn't in the training text.
- You can explain why characters give long sequences, why words give huge vocabularies with unknown tokens, and where subwords sit between them.
- You can say what shape $E$ has, what a lookup returns, and why it equals a one-hot vector times $E$.
:::

## Go deeper

- Sennrich, Haddow and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): section 3.2 has BPE in about 20 lines of Python with this unit's toy corpus, and Table 1 has the vocabulary vs. length numbers quoted above.
- Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (the GPT-2 paper), section 2.2: why byte-level BPE, and the rule that keeps merges inside one character category.
- Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE): builds a byte-level BPE tokenizer from scratch and walks through the odd model behaviors that come from tokenization.
