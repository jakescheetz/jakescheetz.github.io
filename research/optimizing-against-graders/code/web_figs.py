#!/usr/bin/env python3
"""Game-styled web graphics for the E1 post (brief.md, "Decision (Jake, 2026-11-11)"; BOARD D1).

  python3 web_figs.py [names...]       -> figures/web/<name>.svg + <name>.json (all figures if no names)

Style: a Game Boy-ish 4-shade green palette, square pixels on an 8 px grid, crisp edges, monospace labels, framed
like a dialogue box. Light and dark are both chosen palettes (prefers-color-scheme inside each SVG). With 4 shades,
identity never rides on shade alone: every series also has a direct label and a marker or dash pattern.
Everything drawn comes from our own result files or the detector definitions in pokered/detectors.py; no ROM art,
sprites, tiles or screenshots. Standard library only.

Figures:
  route-map                 schematic of the segment: Oak's Lab -> M1 Viridian -> M2 Route 2 -> M3 Viridian Forest,
                            the off-route side buildings (O) and Route 22 (D); Pewter onwards greyed (past the segment)
  score-vs-progress         batch 2, D1: proxy score P and true progress T per checkpoint (A0, A2'' oracle, random)
  score-vs-progress-b1      the same for batch 1 (no visited-squares observation)
"""
import json
import sys
from html import escape
from pathlib import Path

CODE = Path(__file__).resolve().parent
OUT = CODE / "figures" / "web"
SRC = {"b1": CODE / "results/b1/fig1.json", "b2": CODE / "results/b2/fig1.json"}
R0_P0_D1 = 29.689            # R0 (random) under P0 on D1, 24 episodes (R0/eval-dev-24/summary.json); T 0
SEG = ("D1→D1",)

# s0 lightest .. s3 darkest; bg = screen. Dark mode: its own steps (light ink on a dark screen), not a flip.
CSS = """
svg { font-family: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace; font-size: 12px; }
* { shape-rendering: crispEdges; }
.bg { fill: #e4ecc8; } .s0 { fill: #b9cc8c; } .s1 { fill: #8bac0f; } .s2 { fill: #306230; } .s3 { fill: #0f380f; }
.k0 { stroke: #b9cc8c; } .k1 { stroke: #8bac0f; } .k2 { stroke: #306230; } .k3 { stroke: #0f380f; }
.t3 { fill: #0f380f; } .t2 { fill: #306230; } .grey { fill: #c9cfb4; } .kgrey { stroke: #9aa287; } .tgrey { fill: #7d8570; }
.hatch { fill: url(#hatch); } .hatchline { stroke: #306230; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #121c12; } .s0 { fill: #24391f; } .s1 { fill: #4f7a26; } .s2 { fill: #9cbd3c; } .s3 { fill: #d4e6a0; }
  .k0 { stroke: #24391f; } .k1 { stroke: #4f7a26; } .k2 { stroke: #9cbd3c; } .k3 { stroke: #d4e6a0; }
  .t3 { fill: #d4e6a0; } .t2 { fill: #9cbd3c; } .grey { fill: #202620; } .kgrey { stroke: #4a5247; } .tgrey { fill: #7f8a78; }
  .hatchline { stroke: #9cbd3c; }
}
"""


class Svg:
    def __init__(self, w, h, title, desc):
        self.w, self.h = w, h
        self.el = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
                   f'aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
                   f'<style>{CSS}</style><defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse">'
                   f'<path class="hatchline" d="M0 8 L8 0" stroke-width="2"/></pattern></defs>',
                   f'<rect class="bg" width="{w}" height="{h}"/>']

    def add(self, s):
        self.el.append(s)

    def rect(self, x, y, w, h, cls, tip=None, extra=""):
        t = f"<title>{escape(tip)}</title>" if tip else ""
        self.add(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}"{extra}>{t}</rect>')

    def text(self, x, y, s, cls="t3", anchor="start", size=None, bold=False):
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        sz = f' font-size="{size}"' if size else ""
        b = ' font-weight="bold"' if bold else ""
        self.add(f'<text class="{cls}" x="{x}" y="{y}"{a}{sz}{b}>{escape(s)}</text>')

    def frame(self, x, y, w, h, cls="k3", fill=None, px=4):
        """Pixel 'dialogue box': square border with notched corners."""
        if fill:
            self.rect(x + px, y + px, w - 2 * px, h - 2 * px, fill)
        c = cls.replace("k", "s")
        for (xx, yy, ww, hh) in ((x + px, y, w - 2 * px, px), (x + px, y + h - px, w - 2 * px, px),
                                 (x, y + px, px, h - 2 * px), (x + w - px, y + px, px, h - 2 * px)):
            self.rect(xx, yy, ww, hh, c)

    def done(self):
        return "\n".join(self.el + ["</svg>"]) + "\n"


def flag(s, x, y, n, cls="s3", tcls="bg"):
    """Milestone flag: pole + pennant (pixels), number in the pennant."""
    s.rect(x, y, 4, 28, cls)
    s.rect(x + 4, y, 20, 14, cls)
    s.add(f'<text class="{tcls}" x="{x + 14}" y="{y + 11}" text-anchor="middle" font-size="11" font-weight="bold">'
          f'{n}</text>')


def write(name, svg, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.svg").write_text(svg)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote figures/web/{name}.svg, .json")


# ------------------------------------------------------------------ (1) route map
def route_map():
    sys.path.insert(0, str(CODE))
    from pokered import detectors as Dt, ram
    W, H = 640, 640
    s = Svg(W, H, "The route: Oak's Lab to Viridian Forest",
            "Schematic of the game segment the agents play, bottom to top: start in Oak's Lab in Pallet Town, Route 1, "
            "Viridian City (milestone 1), Route 2 (milestone 2), Viridian Forest (milestone 3). Side buildings that "
            "count as off-route exploration (O) and Route 22, a dead end (D), branch off. Pewter City, its gym and "
            "Brock come after the segment and are greyed out.")
    s.frame(8, 8, W - 16, H - 16, fill="bg")
    s.text(28, 40, "THE ROUTE", bold=True, size=16)
    s.text(28, 60, "one episode = 16,384 button presses from Oak's Lab", "t2", size=11)
    cx, cw = 232, 176                      # route column
    cards = [  # (label, y, h, milestone, map id)
        ("PALLET TOWN", 540, 64, None, ram.PALLET_TOWN),
        ("ROUTE 1", 452, 72, None, ram.ROUTE_1),
        ("VIRIDIAN CITY", 356, 80, 1, ram.VIRIDIAN_CITY),
        ("ROUTE 2", 268, 72, 2, ram.ROUTE_2),
        ("VIRIDIAN FOREST", 180, 72, 3, ram.VIRIDIAN_FOREST),
    ]
    data = {"route": [], "side_O": [], "dead_end_D": [], "neutral": [], "past_segment": []}
    for i, (lab, y, h, m, mid) in enumerate(cards):
        s.frame(cx, y, cw, h, fill="s0")
        s.text(cx + cw / 2, (y + 26) if mid == ram.PALLET_TOWN else (y + h / 2 + 5), lab, anchor="middle", bold=True)
        if i:                              # path between cards: pixel dashes
            for yy in range(y + h, cards[i - 1][1], 8):
                s.rect(cx + cw / 2 - 4, yy + 2, 8, 4, "s2")
        if m:
            flag(s, cx + cw + 12, y + 4, m)
            s.text(cx + cw + 44, y + 18, f"M{m}: reach {lab.title()}", "t3", size=11)
        data["route"].append({"map": lab.title(), "map_id": hex(mid), "milestone": f"M{m}" if m else None})
    # start marker inside Pallet: Oak's Lab
    s.rect(cx + 16, 574, 48, 22, "s2", tip="Oak's Lab: every training episode starts here, with Squirtle (or Charmander in the second start state)")
    s.text(cx + 40, 589, "LAB", "bg", anchor="middle", size=10, bold=True)
    s.text(cx + 72, 590, "★ start", "t3", size=11)
    data["start"] = "Oak's Lab (Pallet Town), just after receiving the Pokédex"
    # past the segment (greyed): Pewter, gym, Brock
    s.add(f'<rect class="grey" x="{cx}" y="92" width="{cw}" height="56"/>')
    s.add(f'<rect fill="none" class="kgrey" stroke-width="2" stroke-dasharray="4 4" x="{cx + 1}" y="93" '
          f'width="{cw - 2}" height="54"><title>Milestones M4 (Pewter City), M5 (gym) and M6 (Brock) are outside the '
          f'experiment\'s segment: no trained agent passed M3 (only the scripted planner got further).</title></rect>')
    s.text(cx + cw / 2, 116, "PEWTER · GYM · BROCK", "tgrey", anchor="middle", size=11, bold=True)
    s.text(cx + cw / 2, 134, "outside the segment", "tgrey", anchor="middle", size=10)
    s.text(cx + cw + 12, 116, "no trained agent", "tgrey", size=10)
    s.text(cx + cw + 12, 130, "passed M3", "tgrey", size=10)
    for yy in range(148, 180, 8):
        s.rect(cx + cw / 2 - 4, yy + 2, 8, 4, "grey")
    data["past_segment"] = ["Pewter City (M4)", "Pewter Gym (M5)", "Brock (M6)"]
    # off-route side buildings (O): doors off each card, from detectors.SIDE_MAPS
    sides = {ram.PALLET_TOWN: [0x25, 0x27], ram.VIRIDIAN_CITY: [0x2B, 0x2C, 0x2D], ram.ROUTE_2: [0x2E, 0x30]}
    name = lambda m: Dt.SIDE_MAPS[m].replace("Viridian ", "").replace("Route 2 ", "").replace("(Route 2 entrance)", "").replace("(entrance)", "")
    ys = {c[4]: (c[1], c[2]) for c in cards}
    for mid, doors in sides.items():
        y, h = ys[mid]
        for k, d in enumerate(doors):
            bx, by = cx + cw + 24 + k * 56, y + h - 32
            s.rect(cx + cw, by + 10, bx - cx - cw, 4, "s1")
            s.frame(bx, by, 48, 24, cls="k2", fill="s1", px=2)
            s.rect(bx + 20, by + 12, 8, 10, "s3", tip=f"{Dt.SIDE_MAPS[d]}: an off-route map (counts toward O)")
            data["side_O"].append({"map": Dt.SIDE_MAPS[d], "map_id": hex(d), "off_route_of": hex(mid)})
        labs = [name(d).strip().replace("Gym", "Gym (locked)") for d in doors]
        lines = [" · ".join(labs)] if len(" · ".join(labs)) <= 30 else [" · ".join(labs[:2]) + " ·", labs[2]]
        for j, ln in enumerate(lines):
            s.text(cx + cw + 24, y + h + 4 + 12 * j, ln, "t2", size=9)
    # Route 22 (D): west of Viridian, hatched dead end
    y, h = ys[ram.VIRIDIAN_CITY]
    s.rect(cx - 150, y + 20, 150 - 0, 4, "s1")
    s.add(f'<rect class="s0" x="{cx - 196}" y="{y + 4}" width="136" height="56"/>')
    s.add(f'<rect class="hatch" x="{cx - 196}" y="{y + 4}" width="136" height="56"><title>Route 22: a dead end west '
          f'of Viridian City (time spent here counts toward D)</title></rect>')
    s.frame(cx - 196, y + 4, 136, 56, cls="k3", px=4)
    s.rect(cx - 180, y + 22, 104, 20, "bg")
    s.text(cx - 128, y + 37, "ROUTE 22", anchor="middle", bold=True, size=11)
    s.text(cx - 128, y + 78, "dead end = D", "t2", anchor="middle", size=11)
    data["dead_end_D"] = [{"map": Dt.D_MAPS[ram.ROUTE_22], "map_id": hex(ram.ROUTE_22)}]
    # neutral: Pokécenter / Mart in Viridian
    s.rect(cx + 12, y + 8, 26, 16, "s2", tip="Viridian Pokécenter: neutral (heals; not counted as off-route)")
    s.text(cx + 25, y + 20, "PC", "bg", anchor="middle", size=9, bold=True)
    s.rect(cx + 42, y + 8, 34, 16, "s2", tip="Viridian Mart: neutral")
    s.text(cx + 59, y + 20, "MART", "bg", anchor="middle", size=9, bold=True)
    data["neutral"] = list(Dt.NEUTRAL_MAPS.values())
    # legend
    lx, ly = 28, 160
    s.text(lx, ly, "KEY", bold=True)
    items = [("flag", "milestone (true progress T)"), ("path", "the route"), ("door", "off-route map: O"),
             ("hatch", "dead end: D"), ("grey", "outside the segment")]
    for i, (k, lab) in enumerate(items):
        yy = ly + 18 + i * 24
        if k == "flag":
            s.rect(lx, yy - 10, 3, 16, "s3"); s.rect(lx + 3, yy - 10, 12, 8, "s3")
        elif k == "path":
            s.rect(lx + 4, yy - 6, 8, 4, "s2"); s.rect(lx + 4, yy + 2, 8, 4, "s2")
        elif k == "door":
            s.rect(lx, yy - 10, 16, 14, "s1"); s.rect(lx + 5, yy - 4, 6, 8, "s3")
        elif k == "hatch":
            s.add(f'<rect class="hatch" x="{lx}" y="{yy - 10}" width="16" height="14"/>')
        else:
            s.rect(lx, yy - 10, 16, 14, "grey")
        s.text(lx + 24, yy + 2, lab, "t3", size=10)
    s.text(lx, ly + 142, "O: off-route maps, and maps", "t2", size=9)
    s.text(lx, ly + 154, "behind you (e.g. Pallet once", "t2", size=9)
    s.text(lx, ly + 166, "you reach M1); PC, MART neutral", "t2", size=9)
    s.text(28, H - 26, "schematic, not to scale; map names from the game, drawn by us", "t2", size=10)
    data["note"] = ("Schematic. Route maps and milestones from pokered/ram.py (MILESTONES); side buildings (O), "
                    "Route 22 (D) and neutral maps from pokered/detectors.py (SIDE_MAPS, D_MAPS, NEUTRAL_MAPS). "
                    "Only a subset of SIDE_MAPS is drawn; the full list is in the detector.")
    data["side_O_all"] = {hex(k): v for k, v in Dt.SIDE_MAPS.items()}
    write("route-map", s.done(), data)


# ------------------------------------------------------------------ (2) score vs progress
def score_vs_progress(batch):
    d = json.loads(SRC[batch].read_text())["D1→D1"]
    W, H = 680, 536
    title = {"b2": "Batch 2: score and real progress climb together",
             "b1": "Batch 1: no learning curve"}[batch]
    sub = {"b2": "with the visited-squares observation", "b1": "before the visited-squares observation"}[batch]
    s = Svg(W, H, title, f"{title}, {sub}. Top: proxy score P per checkpoint for the proxy-trained agent (A0), the "
            f"oracle-trained agent (A2'') and random play. Bottom: true progress T (milestones reached). Trained and "
            f"tested from the first start state (Oak's Lab, Squirtle), 5 seeds each, 8 test episodes per seed per checkpoint; band = 95% CI over seeds.")
    s.frame(8, 8, W - 16, H - 16, fill="bg")
    s.text(28, 40, title.upper(), bold=True, size=14)
    s.text(28, 58, f"{sub}; start: Oak's Lab with Squirtle; 5 training runs each", "t2", size=11)
    s.text(28, 74, "*reference, unstable: some of its seeds collapse from 2M on (all 5 averaged here)", "t2", size=10)
    x0, w = 70, 450
    X = lambda v: x0 + v / 5e6 * w
    panels = [("P", "SCORE (what the agent is paid)", 108, 170, 140, [0, 35, 70, 105, 140]),
              ("T", "REAL PROGRESS (milestones reached, average)", 330, 130, 3, [0, 1, 2, 3])]
    arms = [("A0", "trained on the score", "s3", "k3", "", "sq", "score-trained"),
            ("A2pp", "trained on real progress (reference, unstable)", "s2", "k2", ' stroke-dasharray="8 4"', "dia",
             "progress-trained*")]
    data = {"batch": batch, "state": "D1", "source": str(SRC[batch].relative_to(CODE)), "R0": {"P": R0_P0_D1, "T": 0.0},
            "arms": {}}
    for key, lab, y0, h, top, ticks in panels:
        Y = lambda v, y0=y0, h=h, top=top: y0 + h - v / top * h
        s.text(x0, y0 - 10, lab, bold=True, size=11)
        for t in ticks:
            s.add(f'<line class="k0" stroke-width="2" x1="{x0}" x2="{x0 + w}" y1="{Y(t):.0f}" y2="{Y(t):.0f}"/>')
            s.text(x0 - 10, Y(t) + 4, f"{t}", "t2", anchor="end", size=10)
        s.add(f'<line class="k3" stroke-width="4" x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}"/>')
        for t in (0, 1e6, 2e6, 3e6, 4e6, 5e6):
            s.text(X(t), y0 + h + 16, "0" if not t else f"{t / 1e6:g}M", "t2", anchor="middle", size=10)
        # random play
        r0 = R0_P0_D1 if key == "P" else 0.0
        s.add(f'<line class="k1" stroke-width="4" stroke-dasharray="4 4" x1="{x0}" x2="{x0 + w}" y1="{Y(r0):.0f}" '
              f'y2="{Y(r0):.0f}"><title>random play: {key} {r0:g}</title></line>')
        ends = []
        for arm, alab, fcls, kcls, dash, mk, short in arms:
            ms = d[arm]["marks"]
            band = [(m["step"], m[f"{key}_ci"][0], m[f"{key}_ci"][1]) for m in ms]
            if arm == "A0":
                s.add('<polygon class="s0" points="' + " ".join(f"{X(a):.0f},{Y(hi):.0f}" for a, _, hi in band) + " " +
                      " ".join(f"{X(a):.0f},{Y(lo):.0f}" for a, lo, _ in reversed(band)) + '"/>')
            pts = [(X(m["step"]), Y(m[key])) for m in ms]
            s.add(f'<polyline fill="none" class="{kcls}" stroke-width="4" stroke-linejoin="miter"{dash} points="' +
                  " ".join(f"{a:.0f},{b:.0f}" for a, b in pts) + '"/>')
            for m, (a, b) in zip(ms, pts):
                tip = f"{alab}, {m['step'] / 1e6:.2f}M: {key} {m[key]:.2f} [95% CI {m[f'{key}_ci'][0]:.2f}, {m[f'{key}_ci'][1]:.2f}]"
                if mk == "sq":
                    s.rect(round(a) - 6, round(b) - 6, 12, 12, fcls, tip=tip)
                else:
                    s.add(f'<polygon class="{fcls}" points="{a:.0f},{b - 8:.0f} {a + 8:.0f},{b:.0f} {a:.0f},{b + 8:.0f} '
                          f'{a - 8:.0f},{b:.0f}"><title>{escape(tip)}</title></polygon>')
            ends.append([Y(ms[-1][key]) + 4, short])
            data["arms"].setdefault(arm, {"arm": arm, "label": alab, "runs": d[arm]["runs"], "marks": []})
            if key == "P":
                data["arms"][arm]["marks"] = [{k: m[k] for k in ("step", "P", "P_ci", "T", "T_ci", "n")} for m in ms]
        # end labels, nudged apart (>= 14 px) from each other and from the "random" label
        ends.append([Y(r0) + (4 if key == "P" else -6), "random"])
        ends.sort()
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < 14:
                ends[i][0] = ends[i - 1][0] + 14
        for yy, lab_ in ends:
            s.text(x0 + w + 12, yy, lab_, "t2" if lab_ == "random" else "t3", size=10, bold=lab_ != "random")
    s.text(x0 + w, H - 46, "training steps →", "t2", anchor="end", size=11)
    # legend
    lx, ly = 80, H - 30
    s.rect(lx, ly - 10, 12, 12, "s3"); s.text(lx + 18, ly, "trained on the score", "t3", size=11)
    s.add(f'<polygon class="s2" points="{lx + 200},{ly - 12} {lx + 208},{ly - 4} {lx + 200},{ly + 4} {lx + 192},{ly - 4}"/>')
    s.text(lx + 214, ly, "trained on real progress*", "t3", size=11)
    s.add(f'<line class="k1" stroke-width="4" stroke-dasharray="4 4" x1="{lx + 400}" x2="{lx + 428}" y1="{ly - 4}" y2="{ly - 4}"/>')
    s.text(lx + 434, ly, "random", "t3", size=11)
    data["caption"] = {"b2": "Batch 2. The agent trained on the score (A0) pushes the score far above random play, "
                             "and its real progress rises with it: score and progress climb together.",
                       "b1": "Batch 1. Without the visited-squares observation the score-trained agent (A0) sits on a "
                             "plateau from the first checkpoint and makes little real progress; the progress-trained "
                             "agent (A2″) ends ahead of it on progress."}[batch]
    write("score-vs-progress" + ("" if batch == "b2" else "-b1"), s.done(), data)



# ------------------------------------------------------------------ (3) where they went
RUNS = Path.home() / "roms" / "runs"
OW = {0x00: "Pallet Town", 0x0C: "Route 1", 0x01: "Viridian City", 0x0D: "Route 2", 0x21: "Route 22"}
OW_WH = {0x00: (20, 18), 0x0C: (20, 36), 0x01: (40, 36), 0x0D: (20, 72), 0x21: (40, 18)}   # tiles (map header sizes)
INDOOR = [("Oak's Lab", {0x28}), ("Red's/Blue's house", {0x25, 0x26, 0x27}),
          ("Pokécenter/Mart", {0x29, 0x2A}), ("Viridian houses", {0x2B, 0x2C}),
          ("Viridian Forest", {0x33})]
GROUPS = [("score-trained", "trained on the score", "batch 2",
           lambda: sorted(RUNS.glob("b2-A0-D1-s[1-5]/eval/traces/eval-5000192-D1-*.jsonl.gz"))),
          ("capped-score-trained", "trained on the capped score", "round 1",
           lambda: sorted(RUNS.glob("r1-A1-D1-s[1-3]/eval/traces/eval-5000192-D1-*.jsonl.gz"))),
          ("random", "random play", "",
           lambda: sorted((RUNS / "R0" / "eval-dev-24" / "traces").glob("random-0-D1-*.jsonl.gz")))]


def edge_steps(files):
    """Most common coordinate jump (dx, dy) = (x before - x after, y before - y after) for each pair of overworld
    maps, from consecutive rows of our traces (warps and blackouts are rare and lose the vote)."""
    import collections
    import gzip
    tr = collections.Counter()
    for f in files:
        with gzip.open(f, "rt") as g:
            g.readline()
            prev = None
            for line in g:
                r = json.loads(line)
                if prev and prev["map"] != r["map"] and prev["map"] in OW and r["map"] in OW:
                    tr[(prev["map"], r["map"], prev["x"] - r["x"], prev["y"] - r["y"])] += 1
                prev = r
    best = {}
    for (a, b, dx, dy), n in tr.items():
        if n > best.get((a, b), (0,))[0]:
            best[(a, b)] = (n, dx, dy)
    return best


def where_they_went():
    import gzip
    files = {k: f() for k, *_, f in GROUPS}
    # Stepping across an edge moves one tile in the global frame, so along the crossing axis
    # origin(next) = origin + jump + sign(jump); along the other axis origin(next) = origin + jump.
    best = edge_steps(files["score-trained"] + files["random"])
    steps = {f"{OW[a]}->{OW[b]}": list(v) for (a, b), v in best.items()}
    org = {0x00: (0, 0)}
    for a, b in [(0x00, 0x0C), (0x0C, 0x01), (0x01, 0x0D), (0x01, 0x21)]:
        n, dx, dy = best[(a, b)]
        sg = lambda v: (v > 0) - (v < 0)
        org[b] = ((org[a][0] + dx + sg(dx)) if abs(dx) > abs(dy) else org[a][0] + dx,
                  (org[a][1] + dy + sg(dy)) if abs(dy) > abs(dx) else org[a][1] + dy)
    for (a, b) in [(0x00, 0x0C), (0x0C, 0x01), (0x01, 0x0D)]:       # check: north neighbours abut exactly
        assert org[a][1] - org[b][1] == OW_WH[b][1], (OW[a], OW[b], org)
    assert org[0x01][0] - org[0x21][0] == OW_WH[0x21][0], org
    xs = [org[m][0] for m in OW] + [org[m][0] + OW_WH[m][0] for m in OW]
    ys = [org[m][1] for m in OW] + [org[m][1] + OW_WH[m][1] for m in OW]
    gx0, gy0, GW, GH = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    PX = 3
    pw, ph = GW * PX, GH * PX
    W, H = 3 * pw + 4 * 28, ph + 300
    s = Svg(W, H, "Where the agents went",
            "Three maps of the route from Pallet Town (bottom) to Route 2 (top), one per agent, at the 5M checkpoint: "
            "trained on the score (batch 2), trained on the capped score (round 1), and random play. Each square is "
            "shaded by the share of test episodes that stepped on it. Below each map: the share of episodes that "
            "entered each building and the forest.")
    s.frame(8, 8, W - 16, H - 16, fill="bg")
    s.text(28, 40, "WHERE THE AGENTS WENT", bold=True, size=16)
    s.text(28, 58, "start: Oak's Lab (Pallet Town, bottom); 5M checkpoint, one sample: scores swing between checkpoints",
           "t2", size=11)
    data = {"checkpoint": 5000192, "start": "Oak's Lab with Squirtle", "frame_from_our_traces": steps,
            "map_origins_tiles": {OW[m]: org[m] for m in OW}, "panels": {}}
    bins = [(0.75, "s3"), (0.5, "s2"), (0.25, "s1"), (0.0, "s0")]
    top = 92
    for gi, (key, lab, sub, _) in enumerate(GROUPS):
        fl = files[key]
        tiles, entered = {}, {n: 0 for n, _ in INDOOR}
        for f in fl:
            seen, maps = set(), set()
            with gzip.open(f, "rt") as g:
                g.readline()
                for line in g:
                    r = json.loads(line)
                    maps.add(r["map"])
                    if r["map"] in OW:
                        seen.add((r["map"], r["x"], r["y"]))
            for t in seen:
                tiles[t] = tiles.get(t, 0) + 1
            for n, ms in INDOOR:
                entered[n] += bool(maps & ms)
        n = len(fl)
        px0 = 28 + gi * (pw + 28)
        s.text(px0, top - 8, lab.upper(), bold=True, size=11)
        if sub:
            s.text(px0 + pw, top - 8, sub, "t2", anchor="end", size=10)
        s.rect(px0, top, pw, ph, "bg")
        for m in OW:                               # map outlines (sizes only) and names
            ox, oy = (org[m][0] - gx0) * PX + px0, (org[m][1] - gy0) * PX + top
            s.add(f'<rect fill="none" class="k0" stroke-width="1" x="{ox}" y="{oy}" width="{OW_WH[m][0] * PX}" '
                  f'height="{OW_WH[m][1] * PX}"/>')
        cells = []
        for (m, x, y), c in tiles.items():
            v = c / n
            cls = next(k for th, k in bins if v > th or th == 0.0)
            cells.append(f'<rect class="{cls}" x="{(org[m][0] + x - gx0) * PX + px0}" '
                         f'y="{(org[m][1] + y - gy0) * PX + top}" width="{PX}" height="{PX}"/>')
        s.add("<g>" + "".join(cells) + "</g>")
        for m, nm in OW.items():                  # labels in the empty space left of the route column
            ox, oy = (org[m][0] - gx0) * PX + px0, (org[m][1] - gy0) * PX + top
            if m == 0x21:
                s.text(ox + 4, oy - 4, nm, "t2", size=9)
            elif m == 0x01:
                s.text(px0 + 4, (org[0x21][1] + OW_WH[0x21][1] - gy0) * PX + top + 14, "Viridian City →", "t2", size=9)
            else:
                s.text(ox - 4, oy + OW_WH[m][1] * PX / 2, nm, "t2", anchor="end", size=9)
        # strip: buildings + forest
        sy = top + ph + 22
        s.text(px0, sy, f"entered (of {n} episodes):", "t3", size=10, bold=True)
        for j, (nm, _) in enumerate(INDOOR):
            v = entered[nm] / n
            yy = sy + 16 + j * 18
            s.rect(px0, yy - 9, 60, 10, "s0")
            s.rect(px0, yy - 9, int(round(v * 60)), 10, "s2", tip=f"{lab}: {nm} entered in {entered[nm]} of {n} episodes")
            s.text(px0 + 66, yy, f"{entered[nm]}/{n} {nm}", "t3", size=9)
        data["panels"][key] = {"label": lab, "batch": sub, "episodes": n, "tiles_visited": len(tiles),
                               "files": [str(f.relative_to(RUNS)) for f in fl], "entered": entered,
                               "tiles": sorted([[OW[m], x, y, c] for (m, x, y), c in tiles.items()])}
    # legend
    ly = H - 40
    s.text(28, ly, "share of episodes that stepped on the square:", "t3", size=10)
    for j, (lab_, cls) in enumerate((("up to 25%", "s0"), ("25-50%", "s1"), ("50-75%", "s2"), ("over 75%", "s3"))):
        x = 340 + j * 110
        s.rect(x, ly - 10, 12, 12, cls)
        s.text(x + 18, ly, lab_, "t3", size=10)
    s.text(28, ly + 16, "outlines: map sizes only; squares: where the agents walked outdoors (our position logs); "
           "indoor squares not drawn", "t2", size=9)
    data["caption"] = ("Where each agent walked outdoors at the 5M checkpoint, starting in Oak's Lab. The agent trained "
                       "on the score covers Pallet Town, Route 1 and Viridian City densely and reaches Route 22 and "
                       "Route 2; the capped-score agent walks a thin path north; random play barely leaves Pallet Town. "
                       "Indoor squares are not drawn; the strips show which buildings each agent entered.")
    data["not_drawn"] = "indoor maps (their tiles count toward the score too); only the five outdoor maps are drawn"
    write("where-they-went", s.done(), data)


# ------------------------------------------------------------------ (4) reward sources
def reward_sources():
    """Three rows of 50-block bars (1 block = 2%), each on a stated yardstick (researcher's fact-check, item 2):
    (a) what earned each agent's own score, by term; (b) share of the ORIGINAL score (P0) earned on off-route maps
    (O), with the capped agent re-scored under P0; (c) share of each agent's own score earned inside grass-grinding
    stretches (G). All pooled over the 7 tested checkpoints (0.25M-5M), first start state, dev."""
    import glob
    sys.path.insert(0, str(CODE))
    import b1_report
    eps = [json.loads(l) for f in sorted(glob.glob(str(RUNS / "b2-A0-D1-s[1-5]" / "eval-dev-primary" / "episodes.jsonl")))
           for l in open(f)]
    tot = sum(e["P"] for e in eps)
    a0_terms = {k: sum(e["P_terms"][k] for e in eps) / tot for k in ("explore", "level", "heal")}
    r1 = [c for c in json.loads((CODE / "results/r1/cells.json").read_text())["rows"]
          if c["eval_state"] == "D1" and c["mark"] > 0 and not c["interim"]]
    t1 = sum(c["P1"] * c["n"] for c in r1)
    a1_terms = {k: sum(c["P1_terms"][k] * c["n"] for c in r1) / t1 for k in ("explore", "level", "heal")}
    cat = {b: {x["type"]: x for x in json.loads((CODE / f"results/{b}/catalogue-fields.json").read_text())["types"]}
           for b in ("b2", "r1")}
    a1_p0_O = json.loads((CODE / "results/r1/p0-o-d-shares.json").read_text())["cells"]["D1|O|all"][-1][0]
    # batch-2 A0, first start state only (D1-trained, own state), from the traces: O and G share of P0
    at = ao = ag = 0.0
    for f in sorted(glob.glob(str(RUNS / "b2-A0-D1-s[1-5]" / "eval-dev-primary" / "episodes.jsonl"))):
        run = Path(f).parent.parent
        for l in open(f):
            e = json.loads(l)
            t, i = b1_report.p_inside(run / "eval" / "traces" / f"{Path(e['ckpt']).stem}-D1-{e['seed']}.jsonl.gz",
                                      e["events"])
            at, ao, ag = at + t, ao + i.get("O", 0.0), ag + i.get("G", 0.0)
    r0d = RUNS / "R0" / "eval-dev-24"
    rt = ri = 0.0
    for l in open(r0d / "episodes.jsonl"):
        e = json.loads(l)
        if e["state"] == "D1":
            t, i = b1_report.p_inside(r0d / "traces" / f"random-0-D1-{e['seed']}.jsonl.gz", e["events"])
            rt, ri = rt + t, ri + i.get("O", 0.0)
    rows = {
        "a": ("WHAT EARNED THE SCORE", "each agent's own score, by source",
              [("trained on the score", "batch 2", a0_terms), ("trained on the capped score", "round 1", a1_terms)]),
        "b": ("OFF-ROUTE EXPLORING", "share of the original score earned on off-route maps",
              [("trained on the score", "batch 2", {"O": ao / at}),
               ("trained on the capped score", "round 1, re-scored", {"O": a1_p0_O}),
               ("random play", "", {"O": ri / rt})]),
        "c": ("GRASS GRINDING", "share of each agent's own score earned while grinding battles",
              [("trained on the score", "batch 2", {"G": ag / at}),
               ("trained on the capped score", "round 1", {"G": cat["r1"]["G"]["share_of_P_pooled"]})]),
    }
    fills = {"explore": ("s1", "exploring new squares"), "level": ("s3", "levelling up"), "heal": ("s2", "healing"),
             "O": ("hatch", "off-route"), "G": ("hatch", "grinding")}
    W, H = 680, 590
    s = Svg(W, H, "Where the score came from, before and after the patch",
            "Bars of 50 blocks, one block = 2 percent. (a) What earned each agent's own score: almost all exploring "
            "for the score-trained agent; exploring plus about 12 percent levelling for the capped-score agent. (b) "
            "Share of the original score earned on off-route maps: about 17 percent for the score-trained agent, about "
            "17 percent for the capped-score agent re-scored on the original score, 12 percent for random play. (c) "
            "Share of each agent's own score earned while grinding battles: 0 before, 22 percent after.")
    s.frame(8, 8, W - 16, H - 16, fill="bg")
    s.text(28, 40, "WHERE THE SCORE CAME FROM", bold=True, size=16)
    s.text(28, 58, "before the patch (trained on the score) and after (trained on the capped score);", "t2", size=11)
    s.text(28, 74, "start: Oak's Lab with Squirtle; all 7 test checkpoints pooled", "t2", size=11)
    y = 112
    BX, BL = 246, 6                                    # bar x, block size (+1 gap)
    data = {"yardsticks": {"a": "each arm's own reward (A0: P0, A1: P1), pooled over 7 checkpoints",
                           "b": "P0 for all three (A1 re-scored under P0, §9 I14); window-based O ranges",
                           "c": "each arm's own reward; P inside merged G event ranges ÷ total P (catalogue)"},
            "rows": {}}
    for key, (title, sub, items) in rows.items():
        s.text(28, y, f"({key}) {title}", bold=True, size=12)
        s.text(28, y + 16, sub, "t2", size=10)
        y += 34
        for lab, tag, parts in items:
            s.text(28, y + 10, lab, "t3", size=11)
            if tag:
                s.text(28, y + 23, tag, "t2", size=9)
            blocks = []
            cnt = {k: max(int(round(v * 50)), 1 if v >= 0.005 else 0) for k, v in parts.items()}  # >= 1 block if named
            while sum(cnt.values()) > 50:                      # take the overflow from the largest slice
                cnt[max(cnt, key=cnt.get)] -= 1
            for k in parts:
                blocks += [k] * cnt[k]
            for b in range(50):
                k = blocks[b] if b < len(blocks) else None
                cls = fills[k][0] if k else "s0"
                tip = (f"{lab}: " + ", ".join(f"{fills[kk][1]} {vv:.1%}" for kk, vv in parts.items()))
                if cls == "hatch":
                    s.rect(BX + b * (BL + 1), y, BL, 16, "s0")
                s.rect(BX + b * (BL + 1), y, BL, 16, cls, tip=tip if b == 0 else None)
                if cls == "hatch":
                    s.add(f'<rect fill="none" class="k3" stroke-width="1" x="{BX + b * (BL + 1) + 0.5}" y="{y + 0.5}" '
                          f'width="{BL - 1}" height="15"/>')
            main = max(parts, key=parts.get) if key == "a" else next(iter(parts))
            txt = (" / ".join(f"{fills[k][1].split()[0]} {v:.1%}" if v > 0.995 else f"{fills[k][1].split()[0]} {v:.0%}"
                              for k, v in parts.items() if v >= 0.005)
                   if key == "a" else f"{parts[main]:.1%}")
            if key == "a":
                s.text(BX, y + 30, txt, "t2", size=9)
            else:
                s.text(BX + 50 * (BL + 1) + 8, y + 12, txt, "t3", size=10, bold=True)
            data["rows"].setdefault(key, []).append({"label": lab, "tag": tag, **{k: round(v, 4) for k, v in parts.items()}})
            y += 40 if key == "a" else 34
        y += 14
    # legend
    ly = H - 44
    for j, (k, lab) in enumerate((("explore", "exploring"), ("level", "levelling up"), ("heal", "healing"))):
        x = 28 + j * 150
        s.rect(x, ly - 10, 12, 12, fills[k][0])
        s.text(x + 18, ly, lab, "t3", size=10)

    s.rect(28, ly + 8, 12, 12, "s0"); s.rect(28, ly + 8, 12, 12, "hatch")
    s.text(46, ly + 18, "(b), (c): hatched blocks = the share named; faint = the rest; one block = 2%", "t2", size=9)
    data["caption"] = ("Before the patch nearly all the score came from exploring; after it, about an eighth came from "
                       "levelling up. Measured on the original score, the share earned off-route did not move "
                       "(17% before and after; random play 12%). Grinding battles went from none to 22% of the "
                       "capped agent's score.")
    data["sources"] = ["~/roms/runs/b2-A0-D1-s*/eval-dev-primary/episodes.jsonl", "results/r1/cells.json",
                       "~/roms/runs/b2-A0-D1-s*/eval/traces (batch-2 O and G, first start state only)",
                       "results/r1/catalogue-fields.json",
                       "results/r1/p0-o-d-shares.json", "~/roms/runs/R0/eval-dev-24 (traces + events)"]
    write("reward-sources", s.done(), data)


# ------------------------------------------------------------------ (5) round-1 seeds
SEED_NOTE = {1: "progress only at 0.5M, then grinding", 2: "no progress at any checkpoint; grinds late",
             3: "progress at 4 checkpoints, no grinding"}          # findings.md F-R1.4


def round1_seeds():
    cells = [c for c in json.loads((CODE / "results/r1/cells.json").read_text())["rows"]
             if c["eval_state"] == "D1" and not c["interim"]]
    a1 = {(c["seed"], c["mark"]): c for c in cells}
    seeds, marks = sorted({c["seed"] for c in cells}), sorted({c["mark"] for c in cells})
    a0 = {m["step"]: m["T"] for m in json.loads(SRC["b2"].read_text())["D1→D1"]["A0"]["marks"]}
    W, H = 720, 470
    s = Svg(W, H, "Round 1: three runs on the capped score",
            "One panel per training run on the capped score. Bars: real progress (milestones reached, average of 8 "
            "test episodes) at each checkpoint; dotted line: the run trained on the uncapped score in batch 2. Strip "
            "under each panel: the share of that checkpoint's capped score that came from levelling up.")
    s.frame(8, 8, W - 16, H - 16, fill="bg")
    s.text(28, 40, "ROUND 1: THREE RUNS, THREE STORIES", bold=True, size=16)
    s.text(28, 58, "trained on the capped score; start: Oak's Lab with Squirtle; 8 test episodes per checkpoint",
           "t2", size=11)
    s.text(28, 74, "x: checkpoint (millions of training steps); y: milestones reached, average", "t2", size=10)
    pw, gap, top, ph = 196, 26, 132, 156           # panel; 12 blocks of 13 px = 3 milestones
    BW, BH = 18, 12
    data = {"marks": marks, "runs": {}, "score_trained_batch2_T": {str(k): v for k, v in a0.items()}}
    for k, sd in enumerate(seeds):
        px = 44 + k * (pw + gap)
        s.frame(px - 24, top - 40, pw + 32, ph + 144, cls="k2", px=4)
        s.text(px - 12, top - 18, f"RUN {sd}", bold=True, size=12)
        for j, ln in enumerate(SEED_NOTE[sd].replace("; ", ";\n").split("\n")):
            s.text(px - 12, top - 4 + 11 * j, ln, "t2", size=9)
        Y = lambda v: top + 14 + ph - v / 3 * ph
        for t in (1, 2, 3):
            s.add(f'<line class="k0" stroke-width="1" x1="{px}" x2="{px + pw}" y1="{Y(t):.0f}" y2="{Y(t):.0f}"/>')
            s.text(px - 4, Y(t) + 4, str(t), "t2", anchor="end", size=9)
        s.add(f'<line class="k3" stroke-width="2" x1="{px}" x2="{px + pw}" y1="{Y(0):.0f}" y2="{Y(0):.0f}"/>')
        step = pw / len(marks)
        pts = []
        for i, m in enumerate(marks):
            c = a1[(sd, m)]
            bx = px + i * step + (step - BW) / 2
            nb = int(round(c["T"] / 0.25))
            for b in range(nb):
                s.rect(round(bx), round(Y(0) - (b + 1) * (BH + 1)), BW, BH, "s3",
                       tip=f"run {sd}, {fm_(m)}: {c['T']} milestones on average ({c['T_gt0']}/8 episodes above 0)" if b == 0 else None)
            if nb == 0:
                s.rect(round(bx), round(Y(0) - 3), BW, 2, "s1",
                       tip=f"run {sd}, {fm_(m)}: no milestone in any of the 8 episodes")
            s.text(bx + BW / 2, Y(0) + 14, (fm_(m).rstrip("M").lstrip("0") or "0") if m else "0", "t2",
                   anchor="middle", size=9)
            if m in a0:
                pts.append((bx + BW / 2, Y(a0[m])))
            # levelling strip: 5 blocks = 0-100%
            lv = c["P1_terms"]["level"] / c["P1"] if c["P1"] else 0.0
            ly = Y(0) + 52
            for b in range(5):
                s.rect(round(bx), ly - b * 7, BW, 6, "s3" if b < int(round(lv * 5)) else "s0",
                       tip=f"run {sd}, {fm_(m)}: levelling {lv:.0%} of the capped score" if b == 0 else None)
            data["runs"].setdefault(f"run {sd}", []).append({"mark": m, "T": c["T"], "T_gt0": c["T_gt0"],
                                                              "level_share_of_P1": round(lv, 4), "P1": c["P1"]})
        s.add('<polyline fill="none" class="k2" stroke-width="2" stroke-dasharray="3 3" points="' +
              " ".join(f"{a:.0f},{b:.0f}" for a, b in pts) + '"><title>trained on the uncapped score (batch 2, 5 runs)'
              '</title></polyline>')
        s.text(px, Y(0) + 72, "▲ share of score from levelling", "t2", size=8)
    ly = H - 34
    s.rect(28, ly - 10, 12, 12, "s3"); s.text(46, ly, "real progress (milestones, average)", "t3", size=10)
    s.add(f'<line class="k2" stroke-width="2" stroke-dasharray="3 3" x1="300" x2="328" y1="{ly - 4}" y2="{ly - 4}"/>')
    s.text(334, ly, "trained on the uncapped score (batch 2)", "t3", size=10)
    s.text(28, ly + 16, "each checkpoint is one sample: progress swings between checkpoints", "t2", size=9)
    data["caption"] = ("Round 1, one panel per training run on the capped score. Run 1 made progress only at 0.5M and "
                       "then earned more of its score by levelling up; run 2 never reached a milestone and levelled late; run 3 made "
                       "progress at four checkpoints and never levelled much. The dotted line is the uncapped-score "
                       "agent of batch 2.")
    write("round1-seeds", s.done(), data)


def fm_(step):
    return "0" if step == 0 else f"{step / 1e6:.2f}".rstrip("0").rstrip(".") + "M"

FIGS = {"route-map": route_map, "score-vs-progress": lambda: score_vs_progress("b2"),
        "score-vs-progress-b1": lambda: score_vs_progress("b1"), "where-they-went": where_they_went, "reward-sources": reward_sources, "round1-seeds": round1_seeds}

if __name__ == "__main__":
    for n in (sys.argv[1:] or FIGS):
        FIGS[n]()
