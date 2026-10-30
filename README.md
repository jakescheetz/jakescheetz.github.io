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
`content/posts/writing-guide/` is a draft that demonstrates every supported element (math, citations, footnotes, code, diagrams) and documents the front matter fields.

To publish: delete the `draft: true` line, commit, push to `main`.
Drafts, including the files in their directories, are left out of production builds.

## Layout

| Path                  | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `content/index.md`    | Home page: bio and header facts; the post index is appended    |
| `content/posts/*/`    | One directory per post                                         |
| `site.yaml`           | Site-wide metadata (author, site title, language)              |
| `templates/page.html` | Pandoc HTML template for every page                            |
| `assets/`             | Upstream theme (`reset.css`, `index.css`, `index.js`) + `site.*` additions |

`./build.py` writes the production site to `_site/`.

## License

The theme files in `assets/` (except `site.css` and `site.js`) are © Oskar Wickström, MIT licensed; see `assets/LICENSE-monospace-web.md`.
