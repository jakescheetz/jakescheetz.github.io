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
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTENT = ROOT / "content"
POSTS = CONTENT / "posts"
LEARN = CONTENT / "ml-basics"
QUIZ_KINDS = ("term", "apply")
SECTION_COLORS = ("blue", "green", "cyan", "purple", "orange")
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


def check_quiz(meta, label):
    quiz = meta.get("quiz")
    if quiz is None:
        return
    if not isinstance(quiz, list) or not quiz:
        raise BuildError(f"{label}: `quiz` must be a non-empty list of questions")
    for number, item in enumerate(quiz, 1):
        where = f"{label}: quiz question {number}"
        if not isinstance(item, dict):
            raise BuildError(f"{where} must be a mapping with q, answer, wrong, why, kind")
        for field in ("q", "answer", "why"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise BuildError(f"{where} needs a text `{field}` (quote values like true/false/yes)")
        wrong = item.get("wrong")
        if not isinstance(wrong, list) or len(wrong) < 2 or not all(isinstance(w, str) and w.strip() for w in wrong):
            raise BuildError(f"{where} needs `wrong`: a list of at least two text answers")
        if item.get("kind") not in QUIZ_KINDS:
            raise BuildError(f"{where}: `kind` must be one of {', '.join(QUIZ_KINDS)}")


def load_page(src):
    label = src.relative_to(ROOT).as_posix()
    raw = pandoc(
        [src.name, "--to=html5", "--math-method=mathml", f"--lua-filter={FILTERS / 'math.lua'}",
         f"--template={TEMPLATES / 'meta.json'}"],
        src.parent, label,
    )
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
        for field in ("title", "section"):
            if not page.meta.get(field):
                raise BuildError(f"{label}: unit front matter needs `{field}`")
        require_int(page.meta, "unit", label)
        require_int(page.meta, "minutes", label)
        check_quiz(page.meta, label)
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


def unit_rows(units):
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
    return "\n".join(rows)


def unit_index(sections):
    if not sections:
        return "<p>No units yet.</p>"
    parts = []
    for section in sections:
        units = section["units"]
        slugs = " ".join(u.slug for u in units)
        span = f"{units[0].unit_number:02d}" + (f"–{units[-1].unit_number:02d}" if len(units) > 1 else "")
        parts.append(
            f'<h3 class="unit-section"><span class="{section["color"]}">■</span> {section["name"]}'
            f' <span class="muted">· {span} · <span data-progress="{slugs}">0/{len(units)}</span> done</span></h3>\n'
            f'<table class="unit-index">\n<tbody>\n{unit_rows(units)}\n</tbody>\n</table>'
        )
    return "\n".join(parts)


def learn_summary(route, sections, units):
    """Home page block: the course's `home-title` as a linked heading, then its subtitle and size."""
    if route is None or not units:
        return ""
    title = route.meta.get("home-title") or route.meta.get("title", "Learn")
    subtitle = route.meta.get("subtitle")
    lead = f"{subtitle} " if subtitle else ""
    anchor = re.sub(r"[^a-z0-9]+", "-", plain_text(title).lower()).strip("-")
    return (
        f'<h2 id="{anchor}"><a href="{html.escape(route.url)}">{title} →</a></h2>\n'
        f"<p>{lead}{len(units)} units in {len(sections)} sections, {total_time(units)}.</p>"
    )


def quiz_bank(units):
    """Hidden question bank for assets/learn.js to draw quizzes from (one unit, or all of them)."""
    items = []
    for unit in units:
        tag = (
            f' data-unit="{html.escape(unit.slug)}" data-unit-number="{unit.unit_number:02d}"'
            f' data-unit-title="{html.escape(plain_text(unit.meta["title"]))}"'
        )
        for q in unit.meta.get("quiz") or []:
            options = [f'<div class="quiz-option" data-correct="">{q["answer"]}</div>']
            options += [f'<div class="quiz-option">{wrong}</div>' for wrong in q["wrong"]]
            items.append(
                f'<div class="quiz-q" data-kind="{q["kind"]}"{tag}>'
                f'<div class="quiz-prompt">{q["q"]}</div>{"".join(options)}'
                f'<div class="quiz-why">{q["why"]}</div></div>'
            )
    return f'<div class="quiz-bank" hidden="">{"".join(items)}</div>' if items else ""


def render(page, out, site, today, dev):
    # Template variables go through a defaults file, not -V arguments: the quiz
    # banks are hundreds of KB of HTML, and Linux caps one argument at 128 KB.
    variables = {"root": page.root, "year": str(today.year)}
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
        variables.update({"post": "true", "reading-time": reading_time(page.src)})
    if page.is_home:
        variables.update({
            "home": "true",
            "build-date": today.isoformat(),
            "post-index": post_index(site["posts"]),
            "learn-summary": learn_summary(site["route"], site["sections"], site["units"]),
        })
    units = site["units"]
    if page.is_learn_index or page.is_unit:
        args += [
            f"--lua-filter={FILTERS / 'unit-links.lua'}",
            f"--metadata=unit-slugs:{' '.join(u.slug for u in units)}",
            f"--metadata=unit-link-prefix:{'../' if page.is_unit else ''}",
            f"--metadata=unit-self:{page.slug if page.is_unit else ''}",
        ]
    if page.is_learn_index:
        variables.update({
            "learn": "true",
            "learn-index": "true",
            "unit-index": unit_index(site["sections"]),
            "unit-count": str(len(units)),
            "unit-slugs": " ".join(u.slug for u in units),
            "route-time": total_time(units) if units else "-",
            "review-bank": quiz_bank(units),
        })
    if page.is_unit:
        route = site["route"]
        position = units.index(page)
        variables.update({
            "learn": "true",
            "unit": "true",
            "unit-number": f"{page.unit_number:02d}",
            "unit-count": f"{len(units):02d}",
            "unit-slug": page.slug,
            "unit-minutes": f"{page.minutes} min",
            "section-name": page.meta["section"],
            "route-url": "../",
            "route-title": route.meta.get("title", "Learn") if route else "Learn",
            "quiz-bank": quiz_bank([page]),
        })
        for name, neighbor in (("prev", units[position - 1] if position else None),
                               ("next", units[position + 1] if position + 1 < len(units) else None)):
            if neighbor is not None:
                variables.update({
                    f"{name}-url": f"../{neighbor.slug}/",
                    f"{name}-number": f"{neighbor.unit_number:02d}",
                    f"{name}-title": neighbor.meta["title"],
                })
    if dev:
        variables["dev"] = "true"
    (out / page.rel_out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as defaults:
        json.dump({"variables": variables}, defaults, ensure_ascii=False)
        defaults.flush()
        pandoc([*args, f"--defaults={defaults.name}"], page.src.parent, page.rel)


def is_hidden(path, base):
    return any(part.startswith(".") for part in path.relative_to(base).parts)


def group_sections(route, units):
    """Units grouped into the route's `sections` (name + color), in unit order."""
    if not units:
        return []
    label = route.rel if route else LEARN.relative_to(ROOT).as_posix()
    declared = (route.meta.get("sections") if route else None) or []
    colors = {}
    for item in declared:
        if not isinstance(item, dict) or not item.get("name") or item.get("color") not in SECTION_COLORS:
            raise BuildError(f"{label}: each section needs a `name` and a `color` from {', '.join(SECTION_COLORS)}")
        colors[item["name"]] = item["color"]
    groups = []
    for unit in units:
        name = unit.meta["section"]
        if name not in colors:
            raise BuildError(f"{unit.rel}: section {name!r} isn't listed under `sections` in {label}")
        if groups and groups[-1]["name"] == name:
            groups[-1]["units"].append(unit)
        elif any(g["name"] == name for g in groups):
            raise BuildError(f"{unit.rel}: section {name!r} has units on both sides of another section")
        else:
            groups.append({"name": name, "color": colors[name], "units": [unit]})
    return groups


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
        raise BuildError(f"duplicate unit numbers in {LEARN.relative_to(ROOT)}: {dupes}")
    route = next((p for p in pages if p.is_learn_index), None)
    site = {"posts": posts, "units": units, "route": route, "sections": group_sections(route, units)}

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
