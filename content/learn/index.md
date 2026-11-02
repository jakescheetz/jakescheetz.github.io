---
title: From vectors to RLHF
subtitle: The basics behind language models, built up to training on human feedback and where it breaks.
---

I put this together as my own learning material.
Most material on this either starts with the math and never gets to language models, or starts with language models and waves at the math.
This route goes in order: each unit only uses what came before it, and it ends at the part I care most about, which is how models get trained on human feedback and where that goes wrong.

## How it works

There are twelve units.
Each one explains a single idea from the ground up, has at least one picture you can play with, and ends with a short self-test and a "you're done when" list.
If you can do everything on that list without looking back at the page, check the unit off and move on.
Progress is only saved in your browser.

You need high school algebra and enough Python to read a short script.
Everything else gets introduced when it's needed.
Code examples use NumPy.

<figure>
<pre>         <span class="blue">01</span> vectors ──► <span class="blue">02</span> gradients ──► <span class="green">03</span> descent ──► <span class="green">04</span> networks
                                                        │
 <span class="purple">08</span> transformers ◄── <span class="purple">07</span> attention ◄── <span class="purple">06</span> next token ◄── <span class="purple">05</span> tokens
 │
 <span class="purple">09</span> sampling ──► <span class="orange">10</span> RL ──► <span class="orange">11</span> RLHF ──► <span class="orange">12</span> reward hacking

 <span class="blue">■</span> math  <span class="green">■</span> learning  <span class="purple">■</span> language models  <span class="orange">■</span> feedback</pre>
<figcaption>The route. Each unit only depends on the ones before it.</figcaption>
</figure>
