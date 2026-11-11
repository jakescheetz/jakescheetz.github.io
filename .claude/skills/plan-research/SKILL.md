---
name: plan-research
description: Turn a research question into a scoped plan with a reading list, experiment design, and tasks for the agent team. Use when Jake wants to start, plan, or build out a new line of research.
---

# Plan research

Run by the **lead**. Input: a question or area (e.g. "how robust are reward models to distribution shift?").

1. **Scope with Jake** (a few questions at most): what he wants to learn, what the output is (white paper, post series, experiment), how big. Skip questions he already answered.
2. **Create the thread.** Copy `research/_template/` to `research/<topic-slug>/`, fill `brief.md`: question, why it matters for alignment, scope, the security angle only if it is real, definition of done.
3. **Researcher: map the field.** Prompt the researcher for a first pass: 10–20 key papers (verified, in `sources.bib`), a short state-of-knowledge in `findings.md`, the open gaps, and 1–3 candidate experiments that are small enough to run on Jake's machine or a modest cloud budget.
4. **Developer: feasibility.** For the chosen experiment, prompt the developer to sketch `code/README.md`: models/data, compute estimate, what gets measured, how results become figures.
5. **Plan the outputs.** Agree with Jake on the sequence, typically: literature post → experiment → white paper (`paper.md`) → build-log post about how the team did it.
6. **Board.** Write the tasks to `research/BOARD.md` with owners and files, then kick off the first ones.
7. **Report** to Jake: the plan in under 10 lines and what you need from him.
