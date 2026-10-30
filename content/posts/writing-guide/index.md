---
title: Writing guide
subtitle: Everything a post can use, rendered on the grid
date: 2026-10-30
draft: true
toc: true
bibliography: references.bib
abstract: |
  This post never ships: `draft: true` keeps it out of production builds.
  It shows what each Markdown feature renders to; preview it with `./build.py serve`.
---

## Front matter

A post is `content/posts/<slug>/index.md`; its URL is `/posts/<slug>/`.
Anything else in that directory (figures, a `.bib` file) is published next to it.

| Field          | Effect                                              |
| -------------- | --------------------------------------------------- |
| `title`        | Required. Header and post index.                    |
| `date`         | Required, `YYYY-MM-DD`. Sorts the index.            |
| `subtitle`     | One line under the title and in the index.          |
| `updated`      | Shown as Updated; defaults to `date`.               |
| `draft`        | `true` hides the post from production builds.       |
| `toc`          | `true` adds a table of contents.                    |
| `abstract`     | Boxed summary above the body, like this one.        |
| `bibliography` | A `.bib` file next to the post; enables citations.  |

## Math

Inline math like $p(x_t \mid x_{<t})$ sits in the line.
Display math is padded so the text after it stays on the grid:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

## Citations and footnotes

Cite keys from the post's `.bib` file: @vaswani2017attention introduced scaled dot-product attention, or parenthetically [@vaswani2017attention, sec. 3.2].
Footnotes collect at the end.[^end]

[^end]: Like this one.

## Code

```python
import numpy as np

def attention(q, k, v):
    scores = q @ k.T / np.sqrt(k.shape[-1])
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (weights / weights.sum(axis=-1, keepdims=True)) @ v
```

## Diagrams

Box-drawing characters line up because everything is monospace.
Wrap a block in `<figure>` to inset it and add a caption:

<figure>
```
┌────────┐   ┌────────────────────────┐   ┌────────┐
│ tokens │──►│ [ attention → MLP ] ×N │──►│ logits │
└────────┘   └────────────────────────┘   └────────┘
```
<figcaption>A decoder-only transformer.</figcaption>
</figure>

## Quotes and asides

> A blockquote, for when someone else said it better.

<details>
<summary>A collapsible derivation</summary>
<p>Hidden until the reader asks for it.</p>
</details>

## Images

Put image files next to `index.md` and reference them relatively, e.g. `![Caption](loss-curve.png)`.
Images and video are padded to the grid automatically.

## References

::: {#refs}
:::
