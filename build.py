#!/usr/bin/env python3
"""Build the site: Markdown under content/ -> HTML under _site/, via pandoc.

  ./build.py                  production build (drafts excluded)
  ./build.py serve [--port N] dev build with drafts + debug grid, rebuilds on change
  ./build.py new "Post title" scaffold content/posts/<slug>/index.md as a draft

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
ASSETS = ROOT / "assets"
# Game ROMs, save states and battery saves are copyrighted or ROM-derived:
# never copy them into _site, even if one is dropped into assets/ or content/.
NEVER_PUBLISH = ("*.gb", "*.gbc", "*.gba", "*.state", "*.sav", "*.srm")
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
        rel = src.relative_to(CONTENT).with_suffix(".html")
        self.out = OUT / rel
        depth = len(rel.parent.parts)
        self.root = "../" * depth if depth else "./"
        if rel.name == "index.html":
            self.url = rel.parent.as_posix() + "/" if depth else ""
        else:
            self.url = rel.as_posix()
        self.is_home = src == CONTENT / "index.md"
        self.is_post = POSTS in src.parents
        self.draft = meta.get("draft") is True

    @property
    def rel(self):
        return self.src.relative_to(ROOT).as_posix()


def pandoc(args, cwd, label):
    result = subprocess.run(["pandoc", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(f"{label}: pandoc failed:\n{result.stderr.strip()}")
    if result.stderr.strip():
        print(f"{label}: {result.stderr.strip()}", file=sys.stderr)
    return result.stdout


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
    return page


def reading_time(src):
    body = FRONT_MATTER_RE.sub("", src.read_text(encoding="utf-8"), count=1)
    minutes = max(1, round(len(body.split()) / WORDS_PER_MINUTE))
    return f"{minutes} min"


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


def render(page, posts, today, dev):
    args = [
        page.src.name,
        "--from=markdown",
        "--to=html5",
        "--standalone",
        "--math-method=mathml",
        f"--template={TEMPLATES / 'page.html'}",
        f"--metadata-file={SITE_META}",
        f"--output={page.out}",
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
            f"--variable=post-index:{post_index(posts)}",
        ]
    if dev:
        args.append("--variable=dev:true")
    page.out.parent.mkdir(parents=True, exist_ok=True)
    pandoc(args, page.src.parent, page.rel)


def is_hidden(path, base):
    return any(part.startswith(".") for part in path.relative_to(base).parts)


def build(drafts=False, dev=False):
    if shutil.which("pandoc") is None:
        raise BuildError("pandoc not found on PATH (macOS: brew install pandoc)")
    today = datetime.date.today()
    pages = [load_page(src) for src in sorted(CONTENT.rglob("*.md")) if not is_hidden(src, CONTENT)]

    # A draft post's directory (images, bibliography, ...) stays private too.
    excluded_dirs = []
    if not drafts:
        excluded_dirs = [p.src.parent for p in pages if p.draft and p.is_post and p.src.name == "index.md"]
        pages = [p for p in pages if not p.draft]
    posts = sorted((p for p in pages if p.is_post), key=lambda p: (p.meta["date"], p.url), reverse=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ASSETS, OUT / "assets", ignore=shutil.ignore_patterns(".*", *NEVER_PUBLISH))
    for src in sorted(CONTENT.rglob("*")):
        if src.is_dir() or src.suffix == ".md" or is_hidden(src, CONTENT):
            continue
        if any(src.match(pat) for pat in NEVER_PUBLISH):
            continue
        if any(d in src.parents for d in excluded_dirs):
            continue
        dest = OUT / src.relative_to(CONTENT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    for page in pages:
        render(page, posts, today, dev)
    return pages, posts


def report(pages, posts, drafts):
    drafted = sum(1 for p in posts if p.draft)
    note = f" ({drafted} draft)" if drafts and drafted else ""
    print(f"Built {OUT.relative_to(ROOT)}/: pages {len(pages)}, posts {len(posts)}{note}")


def snapshot():
    files = [SITE_META]
    for base in (CONTENT, ASSETS, TEMPLATES, FILTERS):
        files += [p for p in base.rglob("*") if p.is_file()]
    return {p: p.stat().st_mtime_ns for p in files if p.exists()}


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def serve(port):
    def rebuild():
        try:
            report(*build(drafts=True, dev=True), drafts=True)
        except BuildError as error:
            print(f"error: {error}", file=sys.stderr)

    rebuild()
    handler = functools.partial(QuietHandler, directory=str(OUT))
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
    build_cmd.add_argument("--drafts", action="store_true", help="include draft posts")
    serve_cmd = commands.add_parser("serve", help="dev server with drafts and live rebuild")
    serve_cmd.add_argument("--port", type=int, default=8000)
    new_cmd = commands.add_parser("new", help="scaffold a draft post")
    new_cmd.add_argument("title")
    args = parser.parse_args()

    try:
        if args.command == "serve":
            serve(args.port)
        elif args.command == "new":
            new_post(args.title)
        else:
            drafts = getattr(args, "drafts", False)
            report(*build(drafts=drafts), drafts=drafts)
    except BuildError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
