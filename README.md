# personal-site

Personal site and research blog, styled with [The Monospace Web](https://github.com/owickstrom/the-monospace-web).
Markdown is rendered by [pandoc](https://pandoc.org/) through `build.py` (Python standard library only) and deployed to GitHub Pages by `.github/workflows/deploy.yml` on every push to `main`.

## Requirements

- Python 3.9+
- pandoc 3.x (`brew install pandoc`); CI pins 3.13

## Writing a post

```sh
./build.py new "On the geometry of attention"   # creates content/posts/on-the-geometry-of-attention/index.md as a draft
./build.py serve                                # http://127.0.0.1:8000, drafts included, rebuilds on save
```

Put figures and a `references.bib` next to the post's `index.md` and reference them relatively.
Citations (`@key`, `[@key, sec. 2]`) link straight to the paper's `url` (or `doi`) from the `.bib` entry.
To color an ASCII chart, write it as a raw `<pre>` and wrap series markers in `<span class="chart-proxy">` / `<span class="chart-true">` (see the reward-model post).
An ASCII diagram inside a `<figure>` shrinks to fit a narrow screen (`assets/site.js`), down to an 8px font; keep diagrams to about 65 columns so they stay readable on a phone.
Display equations wider than a phone (about 300px) scroll sideways, so split long ones over lines with `\begin{aligned} … \end{aligned}`.
`content/posts/writing-guide/` is a draft that demonstrates every supported element (math, citations, footnotes, code, diagrams) and documents the front matter fields.

To publish: delete the `draft: true` line, commit, push to `main`.
Drafts, including the files in their directories, are left out of production builds.

## ML basics

`content/ml-basics/` is the "ML basics" course at `/ml-basics/`.
`content/ml-basics/index.md` is its intro page and declares the `sections` (name and color); the build appends the grouped unit list, progress, and a review quiz.
Each unit is `content/ml-basics/NN-slug/index.md` with `unit: N` (order, unique), `section:` (one of the declared names; a section's units must be consecutive), `minutes: M`, optionally `scripts: [widgets.js]` for its interactive figures, and a `quiz:` bank.
Units get previous/next links, and a "done" checkbox whose state is kept in the reader's browser (`localStorage`).

A quiz bank is a list of questions, each with `kind` (`term` or `apply`), `q`, `answer`, a list of `wrong` options and a `why`; use single-quoted YAML strings and `$…$` for math.
Each unit page shows five random questions from its bank (mixing both kinds, options shuffled), and the review quiz on the course page draws ten from the units the reader has marked done.
The build fails with a message if a question is missing a field.

Put the term-by-term reading of a display equation in a `::: formula` fenced div right after it; it renders as a smaller, muted aside, so the main text has to make sense without it.
A `::: done-when` div holds the unit's "You're done when" checklist.

Widgets are plain JavaScript on top of the `window.Learn` helpers in `assets/learn.js` (SVG stage, sliders, buttons, readout, animation loop that respects reduced motion); styles and color classes are in `assets/learn.css`.
`content/ml-basics/01-vectors/` is the reference unit.

`./build.py build --out DIR` and `./build.py serve --port N --out DIR` build somewhere other than `_site/`, so several previews can run at once.

## Layout

| Path                  | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `content/index.md`    | Home page: bio and header facts; post index and course summary are appended |
| `content/posts/*/`    | One directory per post                                         |
| `content/ml-basics/`  | ML basics course: intro page plus one directory per unit       |
| `site.yaml`           | Site-wide metadata (author, site title, language)              |
| `templates/page.html` | Pandoc HTML template for every page; holds the pixel-cloud avatar |
| `filters/`            | `cite-links.lua` points citations at the paper; `math.lua` keeps punctuation with inline math and lets wide equations scroll |
| `assets/`             | Upstream theme (`reset.css`, `index.css`, `index.js`), `site.*` and `learn.*` additions, favicons |

`./build.py` writes the production site to `_site/`.

## License

The theme files `assets/reset.css`, `assets/index.css`, and `assets/index.js` are © Oskar Wickström, MIT licensed; see `assets/LICENSE-monospace-web.md`.
