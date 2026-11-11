# Rewrite: "Your reward model is an attack surface"

**Source (live, do not edit):** `content/posts/your-reward-model-is-an-attack-surface/index.md`
**New draft:** `content/posts/reward-hacking-from-the-ground-up/index.md` (`draft: true`, own `references.bib`). Working slug; final title/slug is Jake's call.

**Pass type:** restructure + reframe (more than a tightening pass), run via `/edit-paper`.

**What Jake said is wrong:**
1. It doesn't read well.
2. It doesn't build up and give the reader the context they need for each idea.
3. The frame leans too hard on security. Per AGENTS.md, the subject is alignment (reward hacking, Goodhart, overoptimization, oversight) and security is a lens we use loosely where it helps.

**Direction:**
- Lead with the alignment problem: we can't specify what we want, so we train a proxy for it and then optimize against that proxy. Build up in order: policy → why a learned reward is needed → RLHF recipe → why optimizing a proxy under distribution shift goes wrong (Goodhart) → toy simulation → evidence at scale (Gao et al.) → what mitigations do and don't fix → what's left out (deliberate gaming / deceptive alignment).
- Define every term before using it (KL, best-of-n, proxy vs true reward, distribution shift, overoptimization).
- Fuzzing becomes one short, clearly-labelled analogy ("if you've done security work…"), not the thesis or the title.
- Keep the toy simulation and chart, and keep every claim that has a source. Any claim whose meaning changes goes in `edit-notes.md` with before/after.
- Reader: technically literate, not necessarily ML or security.

**Done when:** new draft builds with `./build.py`, `edit-notes.md` lists substance + every substantive change, and Jake has a diff to review.

**Files:**
- `research/reward-model-post-rewrite/edit-notes.md`: **Substance** (researcher), **Changes** + **Open questions** (writer)
