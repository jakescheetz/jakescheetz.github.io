---
title: Vectors and dot products
subtitle: Lists of numbers, and the one operation that compares them.
unit: 1
section: Math
minutes: 45
scripts: [widgets.js]
quiz:
  - kind: term
    q: 'What is a vector, the way this course uses the word?'
    answer: 'An ordered list of numbers'
    wrong: ['A single number with a direction attached', 'A grid of numbers with rows and columns', 'A set of numbers where order doesn''t matter']
    why: 'Order matters: $(3, 1)$ and $(1, 3)$ are different vectors. A grid of numbers is a matrix.'
  - kind: term
    q: 'What does the dot product of two vectors give you?'
    answer: 'One number: the products of matching entries, added up'
    wrong: ['A new vector of the products of matching entries', 'The angle between the two vectors', 'A matrix with one row per entry']
    why: '$a \cdot b = a_1 b_1 + \cdots + a_n b_n$ is a single number. The angle is related to it, but you have to divide the lengths out first.'
  - kind: term
    q: 'What does cosine similarity measure?'
    answer: 'How closely two vectors point the same way, ignoring their lengths'
    wrong: ['How far apart the tips of the two arrows are', 'The product of the two lengths', 'How many entries the two vectors share']
    why: 'It''s $a \cdot b$ divided by both lengths, so only the angle is left. It runs from −1 to 1.'
  - kind: term
    q: 'What does $\|a\|$ mean?'
    answer: 'The length of $a$'
    wrong: ['The absolute value of each entry of $a$', 'The number of entries in $a$', 'The sum of the entries of $a$']
    why: '$\|a\| = \sqrt{a_1^2 + \cdots + a_n^2}$, the length of the arrow, from Pythagoras.'
  - kind: term
    q: 'In the matrix-vector product $Wa$, what is each entry of the result?'
    answer: 'One row of $W$ dotted with $a$'
    wrong: ['One column of $W$ dotted with $a$', 'One entry of $W$ times one entry of $a$', 'The sum of all of $W$ times $a$']
    why: 'A matrix times a vector is a stack of dot products, one per row of the matrix.'
  - kind: apply
    q: 'What is $(2, -1) \cdot (3, 4)$?'
    answer: '$2$'
    wrong: ['$10$', '$-2$', '$(6, -4)$']
    why: '$2 \cdot 3 + (-1) \cdot 4 = 6 - 4 = 2$. The answer is one number, not a vector.'
  - kind: apply
    q: 'Two nonzero vectors have a dot product of 0. What is the angle between them?'
    answer: '90°'
    wrong: ['0°', '180°', '45°']
    why: 'The cosine is 0 only at 90°, so the vectors are perpendicular and neither casts a shadow on the other.'
  - kind: apply
    q: 'What is the length of $(3, 4)$?'
    answer: '$5$'
    wrong: ['$7$', '$25$', '$12$']
    why: '$\sqrt{3^2 + 4^2} = \sqrt{25} = 5$.'
  - kind: apply
    q: '$W$ is $3 \times 2$. What can you multiply it by, and what comes out?'
    answer: 'A vector with 2 entries, giving a vector with 3 entries'
    wrong: ['A vector with 3 entries, giving a vector with 2 entries', 'A vector with 3 entries, giving a vector with 3 entries', 'Any vector, giving 6 entries']
    why: 'The number of columns of $W$ (2) has to match the vector''s length, and there''s one output per row (3).'
  - kind: apply
    q: 'Vector $b$ points almost exactly opposite to $a$. What sign does $a \cdot b$ have?'
    answer: 'Negative'
    wrong: ['Positive', 'Zero', 'It depends only on the lengths']
    why: 'Past 90° the cosine is negative: $b$''s shadow lands behind the origin on the line through $a$.'
  - kind: apply
    q: 'What is $Wa$ for $W$ with rows $(1, 0)$ and $(0, 2)$, and $a = (3, 1)$?'
    answer: '$(3, 2)$'
    wrong: ['$(3, 1)$', '$(1, 6)$', '$5$']
    why: 'Row 1 gives $(1, 0) \cdot (3, 1) = 3$ and row 2 gives $(0, 2) \cdot (3, 1) = 2$.'
  - kind: apply
    q: 'Two pairs of word embeddings have cosine similarities 0.9 and 0.1. What does that tell you?'
    answer: 'The first pair points in much more similar directions'
    wrong: ['The first pair''s vectors are 9 times longer', 'The second pair''s vectors are closer in length', 'The first pair has more entries']
    why: 'Cosine similarity ignores length and only compares direction. Unit 10 shows real examples from GPT-2.'
---

Almost everything a language model does comes down to lists of numbers getting multiplied and added together.
Two operations show up in every unit after this one: the dot product, which compares two vectors, and matrix-vector multiplication, which is a pile of dot products done at once.

## A vector is a list of numbers

A vector is an ordered list of numbers, like $a = (3, 1)$.
The order matters, so $(3, 1)$ and $(1, 3)$ are different vectors.

There are two ways to picture one, and it helps to keep both in your head.
The first is an arrow: $(3, 1)$ is an arrow from the origin to the point 3 across and 1 up.
The second is a description, where each position holds one measurement of something.
A house could be (square feet, bedrooms, age).
In unit 10 every word gets a vector of a few thousand numbers, and nobody decides what each position means; training works that out.

You can add two vectors position by position, $(3, 1) + (1, 2) = (4, 3)$, and scale one by multiplying every entry, $2 \cdot (3, 1) = (6, 2)$.
The length of a vector, written $\|a\|$, is the length of its arrow, which comes from Pythagoras:

$$
\|a\| = \sqrt{a_1^2 + a_2^2 + \cdots + a_n^2}
$$

::: formula
Here $a_1, a_2, \dots, a_n$ are the entries of $a$ and $n$ is how many there are.
For two entries this is Pythagoras for a right triangle with sides $a_1$ and $a_2$, and the same recipe works for any number of entries.
:::

Square each entry, add the squares, and take the square root of the total.
So $\|(3, 1)\| = \sqrt{3^2 + 1^2} = \sqrt{10} \approx 3.16$.

## The dot product

The dot product multiplies two vectors position by position and adds up the results:

$$
a \cdot b = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n
$$

::: formula
Each term $a_i b_i$ is the $i$-th entry of $a$ times the $i$-th entry of $b$, and the sum runs over all $n$ positions.
:::

Both vectors need the same number of entries.
For $a = (3, 1)$ and $b = (1, 2)$ that's $3 \cdot 1 + 1 \cdot 2 = 5$, a single number, not a vector.

The formula is easy.
What it means took me longer.
The same number can also be written using the angle $\theta$ between the two arrows:

$$
a \cdot b = \|a\|\,\|b\|\cos\theta
$$

::: formula
The first two factors are the [lengths](#a-vector-is-a-list-of-numbers) of $a$ and $b$, and $\theta$ (theta) is the angle between the two arrows.
Lengths are never negative, so the sign of the dot product comes entirely from $\cos\theta$.
:::

Cosine is 1 at 0°, 0 at 90°, and −1 at 180°, so the dot product is large and positive when the vectors point the same way, zero when they're perpendicular, and negative when they point away from each other.

The picture I use: drop a line from the tip of $b$ straight down onto the line through $a$.
The stretch of that line that $b$ covers is $b$'s shadow on $a$, and the dot product is the shadow's length times the length of $a$.
If $b$ leans backward, the shadow lands behind the origin and the dot product goes negative.

<figure class="widget" id="dot-product">
<figcaption>The orange segment is b's shadow on the line through a. Its signed length times |a| is the dot product. Drag the sliders, or let b rotate and watch the sign flip as b passes 90° from a.</figcaption>
</figure>

## Cosine similarity

The dot product mixes two things together: how aligned the vectors are, and how long they are.
A very long vector gets a big dot product with almost anything.
When you only care about direction, divide the lengths out:

$$
\cos\theta = \frac{a \cdot b}{\|a\|\,\|b\|}
$$

::: formula
This is the second formula for $a \cdot b$ above, solved for $\cos\theta$.
The numerator is the dot product, and the denominator is the product of the two lengths, which cancels out how long the arrows are.
:::

This ratio is called *cosine similarity*.
It's always between −1 and 1, and it's the standard way to ask "do these two embeddings mean similar things?", which comes up in unit 10.

## A matrix is a stack of dot products

A matrix is a grid of numbers, and one with $m$ rows and $n$ columns is called $m \times n$.
Multiplying an $m \times n$ matrix by a vector with $n$ entries gives a vector with $m$ entries, and each entry is one row of the matrix dotted with the vector:

<figure>
<pre>        W           a        Wa
   ┌         ┐    ┌   ┐    ┌    ┐
   │ <span class="blue"> 2   0</span>  │    │ 3 │    │ <span class="blue"> 6</span> │  ◄─ <span class="blue">(2,  0)</span> · (3, 1)
   │ <span class="green"> 1   1</span>  │ ×  │ 1 │ =  │ <span class="green"> 4</span> │  ◄─ <span class="green">(1,  1)</span> · (3, 1)
   │ <span class="orange"> 0  -1</span>  │    └   ┘    │ <span class="orange">-1</span> │  ◄─ <span class="orange">(0, -1)</span> · (3, 1)
   └         ┘             └    ┘</pre>
<figcaption>Each output is one row of W dotted with a.</figcaption>
</figure>

So you can read $Wa$ as "score $a$ against every row of $W$."
Each row works like a detector, and its output says how much $a$ points in that row's direction.
A layer of a neural network (unit 8) is exactly this, followed by a simple function applied to each output.

The shapes have to line up: the number of columns in $W$ has to equal the length of $a$.
A $3 \times 2$ matrix times a 2-entry vector works and gives 3 entries.
A $3 \times 2$ matrix times a 3-entry vector doesn't work at all.
Most of the shape errors you'll hit in NumPy or PyTorch are this rule.

## Worked example

```python
import numpy as np

a = np.array([3.0, 1.0])
b = np.array([1.0, 2.0])

print(a @ b)                                  # dot product
print(np.linalg.norm(a), np.linalg.norm(b))   # lengths

cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(cos, np.degrees(np.arccos(cos)))        # cosine and the angle it implies

W = np.array([[2.0,  0.0],
              [1.0,  1.0],
              [0.0, -1.0]])
print(W @ a)                                  # one dot product per row of W
```

```
5.0
3.1622776601683795 2.23606797749979
0.7071067811865475 45.00000000000001
[ 6.  4. -1.]
```

The angle comes out to 45° (the trailing `...01` is floating-point rounding): the cosine is $5 / (\sqrt{10}\sqrt{5}) = 5 / \sqrt{50} = 1/\sqrt{2}$.
In NumPy, `@` is the dot product for two vectors and matrix multiplication when one side is a matrix.

## Where this shows up later

- Unit 8: a neural network layer is a matrix times a vector, plus a bias, passed through a nonlinearity.
- Unit 10: word embeddings are vectors, and cosine similarity is how you compare them.
- Unit 12: attention scores are dot products between a "query" vector and a set of "key" vectors. That unit is mostly this one plus a softmax.

## Self-test

<details>
<summary>What is (2, −1, 3) · (1, 4, 0)?</summary>

$2 \cdot 1 + (-1) \cdot 4 + 3 \cdot 0 = -2$.
It's negative, so the two vectors point more away from each other than toward.

</details>

<details>
<summary>Two nonzero vectors have a dot product of 0. What does that tell you?</summary>

They're perpendicular.
Neither one casts any shadow on the other.

</details>

<details>
<summary>Why does cosine similarity divide by both lengths?</summary>

So it only measures direction.
Without it, a long vector would look similar to everything just for being long.

</details>

<details>
<summary>W is 3 × 2 and x has 2 entries. What shape is Wx, and what is its second entry?</summary>

$Wx$ has 3 entries.
The second one is the second row of $W$ dotted with $x$.

</details>

<details>
<summary>In the widget, the dot product hits zero twice per lap. Where, and why does its sign flip there?</summary>

When $b$ is 90° away from $a$ on either side.
At that point the shadow shrinks to nothing, and as $b$ keeps rotating it starts landing on the other side of the origin.

</details>

::: done-when
- You can compute a dot product and a cosine similarity by hand for small vectors.
- You can explain with the shadow picture why the dot product goes negative when two vectors point apart.
- Given the shapes of a matrix and a vector, you can say whether they multiply and what shape comes out.
:::

## Go deeper

- 3Blue1Brown's Essence of Linear Algebra: [vectors](https://www.3blue1brown.com/lessons/vectors), [matrices as transformations](https://www.3blue1brown.com/lessons/linear-transformations), and [dot products](https://www.3blue1brown.com/lessons/dot-products). The best visual intuition for this material that I know of.
- [Mathematics for Machine Learning](https://mml-book.github.io/), chapter 3, sections 3.1 to 3.4: norms, inner products, lengths and angles, done more formally.
