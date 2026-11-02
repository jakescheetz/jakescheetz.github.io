#!/usr/bin/env python3
"""Build the site: Markdown under content/ -> HTML under _site/, via pandoc.

  ./build.py                  production build (drafts excluded)
  ./build.py serve [--port N] dev build with drafts + debug grid, rebuilds on change
  ./build.py new "Post title" scaffold content/posts/<slug>/index.md as a draft

`build` and `serve` take --out DIR to write somewhere other than _site/.
Only the Python standard library and pandoc are required.
"""
import argparse
import datetime
import functools
import html
import http.server
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTENT = ROOT / "content"
POSTS = CONTENT / "posts"
LEARN = CONTENT / "learn"
ASSETS = ROOT / "assets"
TEMPLATES = ROOT / "templates"
FILTERS = ROOT / "filters"
SITE_META = ROOT / "site.yaml"
OUT = ROOT / "_site"

WORDS_PER_MINUTE = 230
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n(?:---|\.\.\.)\n", re.DOTALL)


class BuildError(Exception):
    pass


class Page:
    def __init__(self, src, meta):
        self.src = src
        self.meta = meta
        self.rel_out = src.relative_to(CONTENT).with_suffix(".html")
        depth = len(self.rel_out.parent.parts)
        self.root = "../" * depth if depth else "./"
        if self.rel_out.name == "index.html":
            self.url = self.rel_out.parent.as_posix() + "/" if depth else ""
        else:
            self.url = self.rel_out.as_posix()
        self.is_home = src == CONTENT / "index.md"
        self.is_post = POSTS in src.parents
        self.is_learn_index = src == LEARN / "index.md"
        self.is_unit = src.name == "index.md" and src.parent.parent == LEARN
        self.draft = meta.get("draft") is True

    @property
    def rel(self):
        return self.src.relative_to(ROOT).as_posix()

    @property
    def slug(self):
        return self.src.parent.name

    @property
    def unit_number(self):
        return int(self.meta["unit"])

    @property
    def minutes(self):
        return int(self.meta["minutes"])


def pandoc(args, cwd, label):
    result = subprocess.run(["pandoc", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(f"{label}: pandoc failed:\n{result.stderr.strip()}")
    if result.stderr.strip():
        print(f"{label}: {result.stderr.strip()}", file=sys.stderr)
    return result.stdout


def require_int(meta, field, label):
    value = meta.get(field)
    if not (isinstance(value, str) and value.isdigit() and int(value) > 0):
        raise BuildError(f"{label}: `{field}` must be a positive whole number, got {value!r}")


def load_page(src):
    label = src.relative_to(ROOT).as_posix()
    raw = pandoc([src.name, "--to=html5", f"--template={TEMPLATES / 'meta.json'}"], src.parent, label)
    page = Page(src, json.loads(raw))
    if page.is_post:
        for field in ("title", "date"):
            if not page.meta.get(field):
                raise BuildError(f"{label}: post front matter needs `{field}`")
        for field in ("date", "updated"):
            value = page.meta.get(field)
            if value is not None and not DATE_RE.match(str(value)):
                raise BuildError(f"{label}: `{field}` must be YYYY-MM-DD, got {value!r}")
    if page.is_unit:
        if not page.meta.get("title"):
            raise BuildError(f"{label}: unit front matter needs `title`")
        require_int(page.meta, "unit", label)
        require_int(page.meta, "minutes", label)
    return page


def reading_time(src):
    body = FRONT_MATTER_RE.sub("", src.read_text(encoding="utf-8"), count=1)
    minutes = max(1, round(len(body.split()) / WORDS_PER_MINUTE))
    return f"{minutes} min"


def total_time(units):
    hours = sum(u.minutes for u in units) / 60
    return f"about {max(1, round(hours))} h"


def plain_text(rendered):
    return html.unescape(re.sub(r"<[^>]+>", "", rendered)).strip()


def post_index(posts):
    if not posts:
        return "<p>Nothing published yet.</p>"
    rows = []
    for post in posts:
        date = html.escape(post.meta["date"])
        link = f'<a href="{html.escape(post.url)}">{post.meta["title"]}</a>'
        if post.draft:
            link += ' <span class="muted">[draft]</span>'
        subtitle = post.meta.get("subtitle")
        summary = f'<br /><span class="muted">{subtitle}</span>' if subtitle else ""
        rows.append(
            f'<tr><td class="width-min"><time datetime="{date}">{date}</time></td>'
            f'<td class="width-auto">{link}{summary}</td></tr>'
        )
    return (
        '<table class="post-index">\n'
        '<thead><tr><th class="width-min">Date</th><th class="width-auto">Title</th></tr></thead>\n'
        "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"
    )


def unit_index(units):
    if not units:
        return "<p>No units yet.</p>"
    rows = []
    for unit in units:
        link = f'<a href="{html.escape(unit.slug)}/">{unit.meta["title"]}</a>'
        if unit.draft:
            link += ' <span class="muted">[draft]</span>'
        subtitle = unit.meta.get("subtitle")
        summary = f'<br /><span class="muted">{subtitle}</span>' if subtitle else ""
        rows.append(
            f'<tr><td class="width-min">{unit.unit_number:02d}</td>'
            f'<td class="width-auto">{link}{summary}</td>'
            f'<td class="width-min">{unit.minutes} min</td>'
            f'<td class="width-min"><span data-unit-mark="{html.escape(unit.slug)}">·</span></td></tr>'
        )
    return (
        '<table class="unit-index">\n'
        '<thead><tr><th class="width-min">#</th><th class="width-auto">Unit</th>'
        '<th class="width-min">Time</th><th class="width-min">Done</th></tr></thead>\n'
        "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"
    )


def learn_summary(route, units):
    if route is None or not units:
        return ""
    subtitle = route.meta.get("subtitle")
    lead = f"{subtitle} " if subtitle else ""
    return (
        f'<p><a href="{html.escape(route.url)}">{route.meta.get("title", "Learn")}</a>: {lead}'
        f"{len(units)} units, {total_time(units)}.</p>"
    )


def render(page, out, site, today, dev):
    args = [
        page.src.name,
        "--from=markdown",
        "--to=html5",
        "--standalone",
        "--math-method=mathml",
        f"--lua-filter={FILTERS / 'math.lua'}",
        f"--template={TEMPLATES / 'page.html'}",
        f"--metadata-file={SITE_META}",
        f"--output={out / page.rel_out}",
        f"--variable=root:{page.root}",
        f"--variable=year:{today.year}",
    ]
    if page.meta.get("bibliography") or page.meta.get("references"):
        args += [
            "--citeproc",
            "--metadata=link-citations:true",
            f"--lua-filter={FILTERS / 'cite-links.lua'}",
        ]
    if page.meta.get("toc") is True:
        args.append("--toc")
    if not page.meta.get("description") and page.meta.get("subtitle"):
        args.append(f"--metadata=description:{plain_text(page.meta['subtitle'])}")
    if page.is_post:
        args += ["--variable=post:true", f"--variable=reading-time:{reading_time(page.src)}"]
    if page.is_home:
        args += [
            "--variable=home:true",
            f"--variable=build-date:{today.isoformat()}",
            f"--variable=post-index:{post_index(site['posts'])}",
            f"--variable=learn-summary:{learn_summary(site['route'], site['units'])}",
        ]
    units = site["units"]
    if page.is_learn_index:
        args += [
            "--variable=learn:true",
            "--variable=learn-index:true",
            f"--variable=unit-index:{unit_index(units)}",
            f"--variable=unit-count:{len(units)}",
            f"--variable=unit-slugs:{' '.join(u.slug for u in units)}",
            f"--variable=route-time:{total_time(units) if units else '-'}",
        ]
    if page.is_unit:
        route = site["route"]
        position = units.index(page)
        args += [
            "--variable=learn:true",
            "--variable=unit:true",
            f"--variable=unit-number:{page.unit_number:02d}",
            f"--variable=unit-count:{len(units):02d}",
            f"--variable=unit-slug:{page.slug}",
            f"--variable=unit-minutes:{page.minutes} min",
            "--variable=route-url:../",
            f"--variable=route-title:{route.meta.get('title', 'Learn') if route else 'Learn'}",
        ]
        for name, neighbor in (("prev", units[position - 1] if position else None),
                               ("next", units[position + 1] if position + 1 < len(units) else None)):
            if neighbor is not None:
                args += [
                    f"--variable={name}-url:../{neighbor.slug}/",
                    f"--variable={name}-number:{neighbor.unit_number:02d}",
                    f"--variable={name}-title:{neighbor.meta['title']}",
                ]
    if dev:
        args.append("--variable=dev:true")
    (out / page.rel_out).parent.mkdir(parents=True, exist_ok=True)
    pandoc(args, page.src.parent, page.rel)


def is_hidden(path, base):
    return any(part.startswith(".") for part in path.relative_to(base).parts)


def build(out=OUT, drafts=False, dev=False):
    if shutil.which("pandoc") is None:
        raise BuildError("pandoc not found on PATH (macOS: brew install pandoc)")
    today = datetime.date.today()
    pages = [load_page(src) for src in sorted(CONTENT.rglob("*.md")) if not is_hidden(src, CONTENT)]

    # A draft post's or unit's directory (images, scripts, bibliography, ...) stays private too.
    excluded_dirs = []
    if not drafts:
        excluded_dirs = [p.src.parent for p in pages if p.draft and (p.is_post or p.is_unit) and p.src.name == "index.md"]
        pages = [p for p in pages if not p.draft]
    posts = sorted((p for p in pages if p.is_post), key=lambda p: (p.meta["date"], p.url), reverse=True)
    units = sorted((p for p in pages if p.is_unit), key=lambda p: p.unit_number)
    numbers = [u.unit_number for u in units]
    if len(set(numbers)) != len(numbers):
        dupes = sorted({n for n in numbers if numbers.count(n) > 1})
        raise BuildError(f"duplicate unit numbers in content/learn: {dupes}")
    route = next((p for p in pages if p.is_learn_index), None)
    site = {"posts": posts, "units": units, "route": route}

    # Build into a staging directory and swap it in at the end, so a dev server never serves a
    # half-written page while a rebuild is running.
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(ASSETS, staging / "assets", ignore=shutil.ignore_patterns(".*"))
    for src in sorted(CONTENT.rglob("*")):
        if src.is_dir() or src.suffix == ".md" or is_hidden(src, CONTENT):
            continue
        if any(d in src.parents for d in excluded_dirs):
            continue
        dest = staging / src.relative_to(CONTENT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    for page in pages:
        render(page, staging, site, today, dev)
    previous = out.with_name(out.name + ".old")
    if previous.exists():
        shutil.rmtree(previous)
    if out.exists():
        out.rename(previous)
    staging.rename(out)
    if previous.exists():
        shutil.rmtree(previous)
    return pages, posts, units


def report(out, pages, posts, units, drafts):
    drafted = sum(1 for p in posts + units if p.draft)
    note = f" ({drafted} draft)" if drafts and drafted else ""
    where = out.relative_to(ROOT) if ROOT in out.parents else out
    print(f"Built {where}/: pages {len(pages)}, posts {len(posts)}, units {len(units)}{note}")


def snapshot():
    files = [SITE_META]
    for base in (CONTENT, ASSETS, TEMPLATES, FILTERS):
        files += [p for p in base.rglob("*") if p.is_file()]
    return {p: p.stat().st_mtime_ns for p in files if p.exists()}


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def serve(port, out=OUT):
    def rebuild():
        try:
            report(out, *build(out, drafts=True, dev=True), drafts=True)
        except BuildError as error:
            print(f"error: {error}", file=sys.stderr)

    rebuild()
    handler = functools.partial(QuietHandler, directory=str(out))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Serving http://127.0.0.1:{port}/ with drafts; rebuilding on change. Ctrl-C to stop.")
    state = snapshot()
    try:
        while True:
            time.sleep(0.5)
            current = snapshot()
            if current != state:
                state = current
                rebuild()
    except KeyboardInterrupt:
        server.shutdown()


def new_post(title):
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if not slug:
        raise BuildError(f"cannot derive a URL slug from {title!r}")
    path = POSTS / slug / "index.md"
    if path.exists():
        raise BuildError(f"{path.relative_to(ROOT)} already exists")
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        f"title: {json.dumps(title)}\n"
        'subtitle: ""                # one line; shown under the title and in the post index\n'
        f"date: {datetime.date.today().isoformat()}\n"
        "draft: true                 # delete this line to publish\n"
        "# updated: YYYY-MM-DD\n"
        "# toc: true\n"
        "# bibliography: references.bib\n"
        "---\n\n",
        encoding="utf-8",
    )
    print(f"Created {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")
    build_cmd = commands.add_parser("build", help="production build (default)")
    build_cmd.add_argument("--drafts", action="store_true", help="include drafts")
    build_cmd.add_argument("--out", type=Path, default=OUT, help="output directory (default: _site)")
    serve_cmd = commands.add_parser("serve", help="dev server with drafts and live rebuild")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.add_argument("--out", type=Path, default=OUT, help="output directory (default: _site)")
    new_cmd = commands.add_parser("new", help="scaffold a draft post")
    new_cmd.add_argument("title")
    args = parser.parse_args()

    try:
        if args.command == "serve":
            serve(args.port, args.out.resolve())
        elif args.command == "new":
            new_post(args.title)
        else:
            drafts = getattr(args, "drafts", False)
            out = getattr(args, "out", OUT).resolve()
            report(out, *build(out, drafts=drafts), drafts=drafts)
    except BuildError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
