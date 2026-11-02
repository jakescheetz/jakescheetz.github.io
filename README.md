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
`content/posts/writing-guide/` is a draft that demonstrates every supported element (math, citations, footnotes, code, diagrams) and documents the front matter fields.

To publish: delete the `draft: true` line, commit, push to `main`.
Drafts, including the files in their directories, are left out of production builds.

## Learning route

`content/learn/` is the "From vectors to RLHF" route at `/learn/`.
`content/learn/index.md` is the route's intro page; the unit table and progress bar are appended by the build.
Each unit is `content/learn/NN-slug/index.md` with `unit: N` (order, unique), `minutes: M`, and optionally `scripts: [widgets.js]` for its interactive figures.
Units get previous/next links, and a "done" checkbox whose state is kept in the reader's browser (`localStorage`).

Widgets are plain JavaScript on top of the `window.Learn` helpers in `assets/learn.js` (SVG stage, sliders, buttons, readout, animation loop that respects reduced motion); styles and color classes are in `assets/learn.css`.
`content/learn/01-vectors/` is the reference unit.

`./build.py build --out DIR` and `./build.py serve --port N --out DIR` build somewhere other than `_site/`, so several previews can run at once.

## Layout

| Path                  | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `content/index.md`    | Home page: bio and header facts; post index and route summary are appended |
| `content/posts/*/`    | One directory per post                                         |
| `content/learn/`      | Learning route: intro page plus one directory per unit          |
| `site.yaml`           | Site-wide metadata (author, site title, language)              |
| `templates/page.html` | Pandoc HTML template for every page; holds the pixel-cloud avatar |
| `filters/`            | `cite-links.lua` points citations at the paper; `math.lua` keeps punctuation with inline math and lets wide equations scroll |
| `assets/`             | Upstream theme (`reset.css`, `index.css`, `index.js`), `site.*` and `learn.*` additions, favicons |

`./build.py` writes the production site to `_site/`.

## License

The theme files `assets/reset.css`, `assets/index.css`, and `assets/index.js` are © Oskar Wickström, MIT licensed; see `assets/LICENSE-monospace-web.md`.
