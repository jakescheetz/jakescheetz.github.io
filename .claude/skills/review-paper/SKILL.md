---
name: review-paper
description: Review an AI alignment / ML paper end to end with the agent team. Use when Jake asks to review, critique, or summarize a paper (arXiv link, URL, or local PDF).
---

# Review a paper

Run by the **lead**. Input: a paper (URL, arXiv ID, or PDF path) and optionally what Jake cares about.

1. **Set up.** Make `research/reviews/<short-slug>/`. Add a board entry under In progress.
2. **Researcher: read and review.** Prompt the researcher (`herdr agent prompt researcher ...`) to write `review.md` there with these sections:
   - **Summary**: the claim in 3–5 sentences, in plain words.
   - **Contributions**: what is actually new.
   - **Method and evidence**: setup, data, baselines, metrics; is the evidence sufficient for the claim?
   - **Weaknesses and alternative explanations**: confounds, missing ablations, overclaiming, eval leakage, cherry-picking.
   - **Alignment relevance**: what it implies for training objectives, oversight, evals, or deployment.
   - **Security lens** (only if it adds something): adversarial incentives, misuse, attack surface.
   - **Related work**: 3–8 papers it should be read alongside, each verified, added to `sources.bib`.
   - **Reproducibility**: code/data availability; what a small replication would take.
   - **Verdict**: how much to trust it, and the one question we'd ask the authors.
3. **Optional replication.** If the review proposes a small replication and Jake wants it, hand it to the developer (`research/reviews/<slug>/code/`).
4. **Check.** Read `review.md` yourself: every claim about the paper traceable to a section/figure, every citation verified. Send back gaps.
5. **Optional post.** If Jake wants it public, prompt the writer for a draft post ("paper notes" style) from `review.md`.
6. **Report** to Jake: verdict in two lines, where the files are, what's waiting on him. Move the board entry to Done.
