# Agent team

This repo is Jake Scheetz's personal site and research blog.

**Research focus:** AI alignment and safety: how models learn the wrong objective, reward hacking,
evaluation and oversight, interpretability, robustness, and how AI systems fail or get misused.
Security is an adjacent lens (threat models, adversarial behavior, attack surfaces of training and
deployment), not the main subject. Write for technically literate readers, not a cybersecurity audience.

It is worked on by a small team of coding agents running side by side in Herdr,
all in this one checkout and sharing this file as common context.
Read `README.md` for how the site builds; read this file for how the team works.

## Roles

| Agent        | Owns (writes to)                                         | Job |
| ------------ | -------------------------------------------------------- | --- |
| `lead`       | `research/BOARD.md`, `research/<topic>/brief.md`         | Talks with Jake, scopes work, splits it into tasks, hands tasks to the others, reviews what comes back. |
| `researcher` | `research/<topic>/` (notes, sources, findings, paper)    | Literature review, paper reviews, alignment analysis (with a security lens where it helps), experiment design, white-paper drafts. |
| `writer`     | `content/posts/<slug>/`                                  | Turns research into blog posts; also writes build-log posts about how this team works. |
| `developer`  | `build.py`, `templates/`, `assets/`, `filters/`, `.github/`, `research/<topic>/code/` | Site features, figures and charts, and the code behind experiments. |

Stay inside the paths you own. If you need a change elsewhere, add a handoff to the board instead of editing it.
All agents may read everything.

## Workflows

Jake starts these by asking the lead in plain words (or with the slash command). The lead runs the playbook in `.claude/skills/<name>/SKILL.md` and delegates the steps.

| Ask | Skill | Produces |
| --- | ----- | -------- |
| "Review this paper: <link or PDF>" | `/review-paper` | `research/reviews/<slug>/review.md`, optional post draft |
| "Clean up / tighten this paper or draft" | `/edit-paper` | Edited draft + `edit-notes.md` listing every substantive change |
| "Plan research on <question>" | `/plan-research` | `research/<topic>/` with brief, reading list, experiment plan, board tasks |

## Shared state

- `research/BOARD.md` is the single task board. Every agent reads it before starting and updates it when it starts, finishes, or hands off a task. Tag entries with your role, e.g. `[researcher]`.
- `research/<topic>/` is one folder per research thread, created from `research/_template/`.
- Hand off by writing what you produced and where it is to the board, then, if you are running in Herdr, tell the next agent directly (see below).

## Talking to each other in Herdr

When `HERDR_ENV=1`, the other agents are live and addressable by name (`lead`, `researcher`, `writer`, `developer`).
Use the `herdr` skill. In short:

```bash
herdr agent list                                    # who is running and their state
herdr agent prompt writer "Findings for <topic> are in research/<topic>/findings.md; draft a post." 
herdr agent read researcher --source recent-unwrapped --lines 80
```

Keep prompts to other agents short and point at files; put the substance in the files.

**Show what you're doing.** When you start a task, run `scripts/status "<verb>: <what>"` (e.g. `scripts/status "Reviewing: Sleeper Agents paper"`), keep it under ~40 characters, and update it when the task changes. Run `scripts/status --clear` when you are idle. It shows under your name in Jake's Herdr sidebar and does nothing outside Herdr.
Do not answer another agent's approval dialog; tell Jake.

## Standards

- **Never invent sources.** Every citation in a `.bib` must have a `url` or `doi` you actually opened. If you could not verify it, leave it out and say so.
- Separate what a source says from what we infer. Mark speculation as such.
- Where security comes up, stay analytical: threat models, failure classes, mitigations; no working exploits against real systems.
- New posts stay `draft: true` until Jake approves them. Only Jake publishes.
- **Git:** do not commit unless Jake asks; never push (a push to `main` deploys the live site).
- Preview with `./build.py serve` (http://127.0.0.1:8000, drafts included). `./build.py` must still pass before handing off site changes.
- Follow `content/posts/writing-guide/index.md` for every Markdown feature the site supports (math, citations, footnotes, code, diagrams, colored ASCII charts).
