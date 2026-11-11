---
name: edit-paper
description: Clean up, tighten, or restructure a paper or long draft (white paper, post, or Jake's own writing) without changing its claims. Use when Jake asks to edit, polish, clean up, or tighten a paper or draft.
---

# Clean up a paper or draft

Run by the **lead**. Input: a file (or pasted text) and what kind of pass Jake wants: light polish, tightening, or restructure. If he didn't say, default to tightening.

1. **Copy, don't overwrite.** If the source is outside the repo, copy it to `research/<topic>/` (or the post dir). Keep the original next to it as `*.orig.md` so the diff is reviewable.
2. **Researcher: substance pass.** Prompt the researcher to check claims against sources: unsupported claims, overstated results, missing caveats, citations that don't say what the text says. Findings go in `edit-notes.md` under **Substance**. The researcher does not rewrite prose.
3. **Writer: prose pass.** Prompt the writer to edit for clarity and flow: cut filler, one idea per paragraph, define terms before use, consistent notation, stronger opening and abstract, headings that say what the section shows. Keep Jake's voice. Apply the researcher's substance notes only where they are clear fixes; leave the rest as questions in `edit-notes.md`.
4. **Developer (if needed):** figures, tables, math rendering, citation formatting, `./build.py` passes.
5. **Never change a claim's meaning silently.** Every substantive change is listed in `edit-notes.md` with before/after and why.
6. **Report** to Jake: what kind of pass was done, the top 3 changes, open questions for him, and `git diff --no-index` / file paths to review.
