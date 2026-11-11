#!/usr/bin/env python3
"""Game-styled web graphics for the E1 post (brief.md, "Decision (Jake, 2026-11-11)"; BOARD D1).

  python3 web_figs.py [names...]       -> figures/web/<name>.svg + <name>.json (all figures if no names)

Style: square pixels on an 8 px grid, crisp edges, monospace labels, framed like a dialogue box, in full color.
Colors follow the thing, not the figure: score-trained agent = blue, progress-trained reference = orange,
capped-score agent = aqua, random play = grey; reward terms: exploring = yellow, levelling = violet, healing =
green; visit share = one magenta ramp. Light and dark are both chosen palettes (prefers-color-scheme inside each
SVG), checked with the dataviz palette validator. Every series also has a direct label, and text stays in ink.
If the run traces under ~/roms/runs are not on this machine, where-they-went and reward-sources redraw from the
numbers saved in figures/web/<name>.json.
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

# Ink and surface, then one color per thing (see the module docstring). Dark mode has its own steps, not a flip.
CSS = """
svg { font-family: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace; font-size: 12px; }
* { shape-rendering: crispEdges; }
.bg { fill: #fcfcfb; } .ink { fill: #0b0b0b; } .t3 { fill: #0b0b0b; } .t2 { fill: #52514e; } .tgrey { fill: #898781; }
.kink { stroke: #0b0b0b; } .kgrid { stroke: #e1e0d9; } .kaxis { stroke: #c3c2b7; } .empty { fill: #e9e8e2; }
.onfill { fill: #ffffff; } .onflag { fill: #0b0b0b; }
.a0 { fill: #2a78d6; } .ka0 { stroke: #2a78d6; } .a0band { fill: #2a78d6; fill-opacity: 0.16; }
.ref { fill: #eb6834; } .kref { stroke: #eb6834; }
.a1 { fill: #1baf7a; } .ka1 { stroke: #1baf7a; }
.rnd { fill: #898781; } .krnd { stroke: #898781; }
.explore { fill: #eda100; } .level { fill: #6250d6; } .heal { fill: #008300; }
.h0 { fill: #e896b6; } .h1 { fill: #dd6795; } .h2 { fill: #b3306a; } .h3 { fill: #6b183c; }
.route { fill: #008300; } .flag { fill: #eda100; } .dead { fill: #e34948; } .kdead { stroke: #e34948; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #1a1a19; } .ink { fill: #f0efec; } .t3 { fill: #f0efec; } .t2 { fill: #c3c2b7; } .tgrey { fill: #898781; }
  .kink { stroke: #f0efec; } .kgrid { stroke: #2c2c2a; } .kaxis { stroke: #4a4a46; } .empty { fill: #2c2c2a; }
  .a0 { fill: #3987e5; } .ka0 { stroke: #3987e5; } .a0band { fill: #3987e5; fill-opacity: 0.22; }
  .ref { fill: #d95926; } .kref { stroke: #d95926; }
  .a1 { fill: #199e70; } .ka1 { stroke: #199e70; }
  .explore { fill: #c98500; } .level { fill: #9085e9; } .heal { fill: #008300; }
  .h0 { fill: #a23266; } .h1 { fill: #cc4f82; } .h2 { fill: #e98bb0; } .h3 { fill: #f9d8e5; }
  .flag { fill: #c98500; } .dead { fill: #e66767; } .kdead { stroke: #e66767; }
}
"""


class Svg:
    def __init__(self, w, h, title, desc):
        self.w, self.h = w, h
        self.el = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
                   f'aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>'
                   f'<style>{CSS}</style>',
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

    def frame(self, x, y, w, h, cls="ink", fill=None, px=4):
        """Pixel 'dialogue box': square border with notched corners."""
        if fill:
            self.rect(x + px, y + px, w - 2 * px, h - 2 * px, fill)
        for (xx, yy, ww, hh) in ((x + px, y, w - 2 * px, px), (x + px, y + h - px, w - 2 * px, px),
                                 (x, y + px, px, h - 2 * px), (x + w - px, y + px, px, h - 2 * px)):
            self.rect(xx, yy, ww, hh, cls)

    def done(self):
        return "\n".join(self.el + ["</svg>"]) + "\n"


def flag(s, x, y, n, cls="flag", tcls="onflag"):
    """Milestone flag: pole + pennant (pixels), number in the pennant."""
    s.rect(x, y, 4, 28, "ink")
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
    """A trail map, bottom to top: five stops on one path, three milestone flags, one dead end. Side buildings and
    the maps after the segment are left to the text and to the .json (redrawn 2026-11-11: the first version was
    too busy)."""
    sys.path.insert(0, str(CODE))
    from pokered import detectors as Dt, ram
    W, H = 640, 568
    s = Svg(W, H, "The route: Oak's Lab to Viridian Forest",
            "Schematic of the game segment the agents play, bottom to top: start in Oak's Lab in Pallet Town, Route 1, "
            "Viridian City (milestone 1), Route 2 (milestone 2), Viridian Forest (milestone 3). Route 22 branches "
            "west off Viridian City and is a dead end.")
    s.frame(8, 8, W - 16, H - 16, fill="bg")
    s.text(28, 40, "THE ROUTE", bold=True, size=16)
    s.text(28, 60, "read it from the bottom up: the agent starts in Oak's Lab and heads north", "t2", size=11)
    cx, cw, ch = 232, 176, 48              # route column
    stops = [  # (label, y, milestone, what the milestone asks for, map id)
        ("PALLET TOWN", 462, None, None, ram.PALLET_TOWN),
        ("ROUTE 1", 374, None, None, ram.ROUTE_1),
        ("VIRIDIAN CITY", 286, 1, "reach Viridian City", ram.VIRIDIAN_CITY),
        ("ROUTE 2", 198, 2, "reach Route 2", ram.ROUTE_2),
        ("VIRIDIAN FOREST", 110, 3, "enter Viridian Forest", ram.VIRIDIAN_FOREST),
    ]
    data = {"route": [], "dead_end_D": [], "not_drawn": {}}
    mx = cx + cw // 2
    for i, (lab, y, m, ask, mid) in enumerate(stops):
        if i:                              # the path up from the stop below, with an arrowhead halfway
            y0, y1 = y + ch, stops[i - 1][1]
            s.rect(mx - 4, y0, 8, y1 - y0, "route")
            my = (y0 + y1) // 2
            s.add(f'<polygon class="route" points="{mx},{my - 10} {mx + 14},{my + 6} {mx - 14},{my + 6}"/>')
        s.frame(cx, y, cw, ch, fill="bg")
        s.text(mx, y + ch / 2 + 5, lab, anchor="middle", bold=True, size=13)
        if m:
            flag(s, cx + cw + 16, y + 8, m)
            s.text(cx + cw + 50, y + 20, f"MILESTONE {m}", "t3", size=12, bold=True)
            s.text(cx + cw + 50, y + 36, ask, "t2", size=11)
        data["route"].append({"map": lab.title(), "map_id": hex(mid), "milestone": f"M{m}" if m else None})
    # the game goes on past the forest: a faint stub, no card
    for yy in range(84, 108, 8):
        s.rect(mx - 4, yy + 2, 8, 4, "empty")
    s.text(mx + 14, 96, "the game goes on from here", "tgrey", size=10)
    # start: Oak's Lab, in Pallet Town
    y = stops[0][1]
    s.rect(cx - 56, y + ch // 2 - 2, 56, 4, "a0")
    s.rect(cx - 144, y + 8, 88, 32, "a0",
           tip="Oak's Lab: every training episode starts here, just after the agent gets the Pokédex")
    s.text(cx - 100, y + 29, "START", "onfill", anchor="middle", size=13, bold=True)
    s.text(cx - 100, y + 58, "in Oak's Lab", "t2", anchor="middle", size=11)
    data["start"] = "Oak's Lab (Pallet Town), just after receiving the Pokédex"
    # Route 22: a dead end west of Viridian City
    y = stops[2][1]
    s.add(f'<line class="kdead" stroke-width="4" stroke-dasharray="8 6" x1="{cx - 56}" x2="{cx}" y1="{y + ch // 2}" '
          f'y2="{y + ch // 2}"/>')
    s.frame(cx - 184, y + 4, 128, 40, cls="dead", fill="bg")
    s.add(f'<title>Route 22: a dead end west of Viridian City (time spent here counts toward the detour)</title>')
    s.text(cx - 120, y + 29, "ROUTE 22", anchor="middle", bold=True, size=12)
    s.text(cx - 120, y + 62, "a dead end", "t2", anchor="middle", size=11)
    data["dead_end_D"] = [{"map": Dt.D_MAPS[ram.ROUTE_22], "map_id": hex(ram.ROUTE_22)}]
    s.text(28, H - 26, "a schematic, not to scale; map names from the game", "t2", size=10)
    data["not_drawn"] = {"side_O_all": {hex(k): v for k, v in Dt.SIDE_MAPS.items()},
                         "neutral": list(Dt.NEUTRAL_MAPS.values()),
                         "past_segment": ["Pewter City (M4)", "Pewter Gym (M5)", "Brock (M6)"]}
    data["note"] = ("Schematic. Route maps and milestones from pokered/ram.py (MILESTONES); Route 22 (D) from "
                    "pokered/detectors.py (D_MAPS). The off-route side buildings (SIDE_MAPS), the neutral maps and "
                    "the maps past the segment are listed under not_drawn; the post explains off-route in the text.")
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
    arms = [("A0", "trained on the score", "a0", "ka0", "", "sq", "score-trained"),
            ("A2pp", "trained on real progress (reference, unstable)", "ref", "kref", ' stroke-dasharray="8 4"', "dia",
             "progress-trained*")]
    data = {"batch": batch, "state": "D1", "source": str(SRC[batch].relative_to(CODE)), "R0": {"P": R0_P0_D1, "T": 0.0},
            "arms": {}}
    for key, lab, y0, h, top, ticks in panels:
        Y = lambda v, y0=y0, h=h, top=top: y0 + h - v / top * h
        s.text(x0, y0 - 10, lab, bold=True, size=11)
        for t in ticks:
            s.add(f'<line class="kgrid" stroke-width="2" x1="{x0}" x2="{x0 + w}" y1="{Y(t):.0f}" y2="{Y(t):.0f}"/>')
            s.text(x0 - 10, Y(t) + 4, f"{t}", "t2", anchor="end", size=10)
        s.add(f'<line class="kink" stroke-width="4" x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}"/>')
        for t in (0, 1e6, 2e6, 3e6, 4e6, 5e6):
            s.text(X(t), y0 + h + 16, "0" if not t else f"{t / 1e6:g}M", "t2", anchor="middle", size=10)
        # random play
        r0 = R0_P0_D1 if key == "P" else 0.0
        s.add(f'<line class="krnd" stroke-width="4" stroke-dasharray="4 4" x1="{x0}" x2="{x0 + w}" y1="{Y(r0):.0f}" '
              f'y2="{Y(r0):.0f}"><title>random play: {key} {r0:g}</title></line>')
        ends = []
        for arm, alab, fcls, kcls, dash, mk, short in arms:
            ms = d[arm]["marks"]
            band = [(m["step"], m[f"{key}_ci"][0], m[f"{key}_ci"][1]) for m in ms]
            if arm == "A0":
                s.add('<polygon class="a0band" points="' + " ".join(f"{X(a):.0f},{Y(hi):.0f}" for a, _, hi in band) + " " +
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
    s.rect(lx, ly - 10, 12, 12, "a0"); s.text(lx + 18, ly, "trained on the score", "t3", size=11)
    s.add(f'<polygon class="ref" points="{lx + 200},{ly - 12} {lx + 208},{ly - 4} {lx + 200},{ly + 4} {lx + 192},{ly - 4}"/>')
    s.text(lx + 214, ly, "trained on real progress*", "t3", size=11)
    s.add(f'<line class="krnd" stroke-width="4" stroke-dasharray="4 4" x1="{lx + 400}" x2="{lx + 428}" y1="{ly - 4}" y2="{ly - 4}"/>')
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


def _where_counts():
    """(origins, edge steps, panels) for where-they-went. From the eval traces when they are on this machine;
    otherwise from the numbers saved in figures/web/where-they-went.json by the run that had them."""
    import gzip
    files = {k: f() for k, *_, f in GROUPS}
    if not all(files.values()):
        saved = json.loads((OUT / "where-they-went.json").read_text())
        ids = {v: k for k, v in OW.items()}
        org = {ids[n]: tuple(v) for n, v in saved["map_origins_tiles"].items()}
        panels = {k: {"n": v["episodes"], "entered": v["entered"], "files": v["files"],
                      "tiles": {(ids[n], x, y): c for n, x, y, c in v["tiles"]}}
                  for k, v in saved["panels"].items()}
        print("where-they-went: traces not found under ~/roms/runs; redrawing from the saved .json")
        return org, saved["frame_from_our_traces"], panels
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
    panels = {}
    for key, *_ in GROUPS:
        tiles, entered = {}, {n: 0 for n, _ in INDOOR}
        for f in files[key]:
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
        panels[key] = {"n": len(files[key]), "entered": entered, "tiles": tiles,
                       "files": [str(f.relative_to(RUNS)) for f in files[key]]}
    return org, steps, panels


def where_they_went():
    org, steps, panels = _where_counts()
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
            "map_origins_tiles": {OW[m]: list(org[m]) for m in OW}, "panels": {}}
    bins = [(0.75, "h3"), (0.5, "h2"), (0.25, "h1"), (0.0, "h0")]
    top = 92
    for gi, (key, lab, sub, _) in enumerate(GROUPS):
        n, tiles, entered = panels[key]["n"], panels[key]["tiles"], panels[key]["entered"]
        px0 = 28 + gi * (pw + 28)
        s.text(px0, top - 8, lab.upper(), bold=True, size=11)
        if sub:
            s.text(px0 + pw, top - 8, sub, "t2", anchor="end", size=10)
        s.rect(px0, top, pw, ph, "bg")
        for m in OW:                               # map outlines (sizes only) and names
            ox, oy = (org[m][0] - gx0) * PX + px0, (org[m][1] - gy0) * PX + top
            s.add(f'<rect fill="none" class="kaxis" stroke-width="1" x="{ox}" y="{oy}" width="{OW_WH[m][0] * PX}" '
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
            s.rect(px0, yy - 9, 60, 10, "empty")
            s.rect(px0, yy - 9, int(round(v * 60)), 10, "ink", tip=f"{lab}: {nm} entered in {entered[nm]} of {n} episodes")
            s.text(px0 + 66, yy, f"{entered[nm]}/{n} {nm}", "t3", size=9)
        data["panels"][key] = {"label": lab, "batch": sub, "episodes": n, "tiles_visited": len(tiles),
                               "files": panels[key]["files"], "entered": entered,
                               "tiles": sorted([[OW[m], x, y, c] for (m, x, y), c in tiles.items()])}
    # legend
    ly = H - 40
    s.text(28, ly, "share of episodes that stepped on the square:", "t3", size=10)
    for j, (lab_, cls) in enumerate((("up to 25%", "h0"), ("25-50%", "h1"), ("50-75%", "h2"), ("over 75%", "h3"))):
        x = 340 + j * 110
        s.rect(x, ly - 10, 12, 12, cls)
        s.text(x + 18, ly, lab_, "t3", size=10)
    s.text(28, ly + 16, "outlines: map sizes only; squares: where the agents walked outdoors (from position logs); "
           "indoor squares not drawn", "t2", size=9)
    data["caption"] = ("Where each agent walked outdoors at the 5M checkpoint, starting in Oak's Lab. The agent trained "
                       "on the score covers Pallet Town, Route 1 and Viridian City densely and reaches Route 22 and "
                       "Route 2; the capped-score agent walks a thin path north; random play barely leaves Pallet Town. "
                       "Indoor squares are not drawn; the strips show which buildings each agent entered.")
    data["not_drawn"] = "indoor maps (their tiles count toward the score too); only the five outdoor maps are drawn"
    write("where-they-went", s.done(), data)


# ------------------------------------------------------------------ (4) reward sources
def _reward_rows():
    """Row values for reward-sources: {"a": [(label, tag, {term: share})], "b": ..., "c": ...}. From the run
    folders when they are on this machine; otherwise from the numbers saved in figures/web/reward-sources.json."""
    import glob
    eps_files = sorted(glob.glob(str(RUNS / "b2-A0-D1-s[1-5]" / "eval-dev-primary" / "episodes.jsonl")))
    if not eps_files:
        saved = json.loads((OUT / "reward-sources.json").read_text())["rows"]
        print("reward-sources: run folders not found under ~/roms/runs; redrawing from the saved .json")
        return {k: [(r["label"], r["tag"], {t: v for t, v in r.items() if t not in ("label", "tag")}) for r in rows]
                for k, rows in saved.items()}
    sys.path.insert(0, str(CODE))
    import b1_report
    eps = [json.loads(l) for f in eps_files for l in open(f)]
    tot = sum(e["P"] for e in eps)
    a0_terms = {k: sum(e["P_terms"][k] for e in eps) / tot for k in ("explore", "level", "heal")}
    r1 = [c for c in json.loads((CODE / "results/r1/cells.json").read_text())["rows"]
          if c["eval_state"] == "D1" and c["mark"] > 0 and not c["interim"]]
    t1 = sum(c["P1"] * c["n"] for c in r1)
    a1_terms = {k: sum(c["P1_terms"][k] * c["n"] for c in r1) / t1 for k in ("explore", "level", "heal")}
    cat = {x["type"]: x for x in json.loads((CODE / "results/r1/catalogue-fields.json").read_text())["types"]}
    a1_p0_O = json.loads((CODE / "results/r1/p0-o-d-shares.json").read_text())["cells"]["D1|O|all"][-1][0]
    # batch-2 A0, first start state only (D1-trained, own state), from the traces: O and G share of P0
    at = ao = ag = 0.0
    for f in eps_files:
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
    return {"a": [("trained on the score", "batch 2", a0_terms), ("trained on the capped score", "round 1", a1_terms)],
            "b": [("trained on the score", "batch 2", {"O": ao / at}),
                  ("trained on the capped score", "round 1, re-scored", {"O": a1_p0_O}),
                  ("random play", "", {"O": ri / rt})],
            "c": [("trained on the score", "batch 2", {"G": ag / at}),
                  ("trained on the capped score", "round 1", {"G": cat["G"]["share_of_P_pooled"]})]}


def reward_sources():
    """Three small bar charts, each on a stated yardstick (researcher's fact-check, item 2): (a) what earned each
    agent's own score, by term, as one full-width bar per agent; (b) share of the ORIGINAL score (P0) earned on
    off-route maps (O), with the capped agent re-scored under P0; (c) share of each agent's own score earned inside
    grass-grinding stretches (G). All pooled over the 7 tested checkpoints (0.25M-5M), first start state, dev.
    (b) and (c) share one 0-30% axis so their bars can be compared. Redrawn 2026-11-11 (Jake: the 50-block version
    was hard to read): solid bars, plain row names, every value written at the end of its bar."""
    values = _reward_rows()
    heads = {"a": ("(a) WHAT EARNED THE SCORE", "each agent's own score, split by where the points came from"),
             "b": ("(b) EARNED OFF-ROUTE", "share of the original score earned on off-route maps (all three scored the same way)"),
             "c": ("(c) EARNED BY GRINDING", "share of each agent's own score earned while grinding battles in grass")}
    names = {"explore": "exploring new squares", "level": "levelling up", "heal": "healing",
             "O": "off-route", "G": "grinding"}
    short = {"explore": "exploring", "level": "levelling up", "heal": "healing"}
    agent = {"trained on the score": ("a0", "before the patch"), "trained on the capped score": ("a1", "after the patch"),
             "random play": ("rnd", "random play")}
    W, H = 680, 552
    s = Svg(W, H, "Where the score came from, before and after the patch",
            "Three bar charts. (a) What earned each agent's own score: almost all exploring before the patch; "
            "exploring plus about 12 percent levelling up after it. (b) Share of the original score earned on "
            "off-route maps: about 17 percent before the patch, about 17 percent after it, 12 percent for random "
            "play. (c) Share of each agent's own score earned while grinding battles: 0 before, 22 percent after.")
    s.frame(8, 8, W - 16, H - 16, fill="bg")
    s.text(28, 40, "WHERE THE SCORE CAME FROM", bold=True, size=16)
    s.text(28, 58, "the agent before the patch and the agent after it; all 7 test checkpoints pooled", "t2", size=11)
    BX, BW = 190, 400                                  # bar x and full width
    TOP = 0.30                                         # (b), (c): the axis runs 0 to 30%
    data = {"yardsticks": {"a": "each arm's own reward (A0: P0, A1: P1), pooled over 7 checkpoints",
                           "b": "P0 for all three (A1 re-scored under P0, §9 I14); window-based O ranges",
                           "c": "each arm's own reward; P inside merged G event ranges ÷ total P (catalogue)"},
            "axis": {"a": "0 to 100% of the agent's own score", "b": "0 to 30%", "c": "0 to 30%"}, "rows": {}}
    y = 98
    for key, (title, sub) in heads.items():
        s.text(28, y, title, bold=True, size=12)
        s.text(28, y + 16, sub, "t2", size=10)
        y += 36
        rows = values[key]
        if key != "a":                                 # light gridlines behind the rows of (b) and (c)
            for t in (0.1, 0.2, 0.3):
                gx = BX + t / TOP * BW
                s.add(f'<line class="kgrid" stroke-width="2" x1="{gx:.0f}" x2="{gx:.0f}" y1="{y - 6}" '
                      f'y2="{y + 30 * len(rows) - 4}"/>')
        for lab, tag, parts in rows:
            cls, name = agent[lab]
            tip = f"{name} ({lab}{', ' + tag if tag else ''}): " + ", ".join(f"{names[k]} {v:.1%}" for k, v in parts.items())
            s.rect(28, y + 4, 10, 10, cls)             # the agent's color chip, same in every figure
            s.text(46, y + 13, name, "t3", size=11)
            if key == "a":
                x = float(BX)
                for k, v in parts.items():             # one full-width bar, a 2 px gap between terms
                    w = v * BW
                    if w >= 1:
                        s.rect(round(x), y, max(round(w) - 2, 1), 18, k, tip=tip)
                    x += w
                lx = BX
                for k, v in parts.items():             # what each color is, written under the bar
                    if v < 0.005:
                        continue
                    s.rect(lx, y + 26, 8, 8, k)
                    t = f"{short[k]} {v:.1%}" if v > 0.995 else f"{short[k]} {v:.0%}"
                    s.text(lx + 13, y + 34, t, "t2", size=10)
                    lx += 13 + len(t) * 6.1 + 18
                y += 52
            else:
                v = next(iter(parts.values()))
                w = v / TOP * BW
                s.rect(BX, y, max(round(w), 2), 18, cls, tip=tip)
                s.text(BX + max(round(w), 2) + 8, y + 13, f"{v:.1%}", "t3", size=11, bold=True)
                y += 30
            data["rows"].setdefault(key, []).append({"label": lab, "tag": tag, **{k: round(v, 5) for k, v in parts.items()}})
        if key != "a":
            s.add(f'<line class="kink" stroke-width="2" x1="{BX}" x2="{BX}" y1="{y - 30 * len(rows) - 6}" y2="{y - 4}"/>')
            for t in (0, 0.1, 0.2, 0.3):
                s.text(BX + t / TOP * BW, y + 8, f"{t:.0%}", "t2", anchor="middle", size=9)
            y += 12
        y += 24
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


SEED_LINES = {1: ["progress at 0.5M only,", "then grinding"], 2: ["never makes progress,", "grinds late"],
              3: ["progress at 4 of 7", "checkpoints, no grinding"]}          # SEED_NOTE, wrapped for the row label


def round1_seeds():
    """A grid: one row per run, two columns. Left: real progress (milestones, 0 to 3) at each checkpoint, with the
    batch-2 average as a blue tick. Right: the share of that checkpoint's capped score that came from levelling up
    (0 to 50%). Every non-zero bar carries its value. Redrawn 2026-11-11 (Jake: the stacked blocks with a strip
    underneath were hard to read). The untrained step-0 mark stays in the .json but is not drawn."""
    cells = [c for c in json.loads((CODE / "results/r1/cells.json").read_text())["rows"]
             if c["eval_state"] == "D1" and not c["interim"]]
    a1 = {(c["seed"], c["mark"]): c for c in cells}
    seeds, marks = sorted({c["seed"] for c in cells}), sorted({c["mark"] for c in cells})
    a0 = {m["step"]: m["T"] for m in json.loads(SRC["b2"].read_text())["D1→D1"]["A0"]["marks"]}
    drawn = [m for m in marks if m > 0]
    W, H = 720, 520
    s = Svg(W, H, "Round 1: three runs on the capped score",
            "A grid with one row per training run on the capped score and two columns. Left column: real progress "
            "(milestones reached, average of 8 test episodes) at each checkpoint, with the average of the runs "
            "trained before the patch marked for comparison. Right column: the share of that checkpoint's capped "
            "score that came from levelling up.")
    s.frame(8, 8, W - 16, H - 16, fill="bg")
    s.text(28, 40, "ROUND 1: THREE RUNS ON THE PATCHED SCORE", bold=True, size=16)
    s.text(28, 58, "one bar per checkpoint (8 test episodes each); start: Oak's Lab with Squirtle", "t2", size=11)
    cols = [("T", 206, 230, 3.0, [1, 2, 3], "REAL PROGRESS", "milestones reached, average (0 to 3)"),
            ("L", 480, 212, 0.5, [0.25, 0.5], "SCORE FROM LEVELLING UP", "share of that checkpoint's score")]
    for _, x0, w, _, _, head, sub in cols:
        s.text(x0, 90, head, bold=True, size=11)
        s.text(x0, 105, sub, "t2", size=10)
    top, ph, pitch, BW = 136, 72, 110, 18
    num = lambda v: f"{v:.3f}".rstrip("0").rstrip(".")
    data = {"marks": marks, "marks_drawn": drawn, "runs": {}, "score_trained_batch2_T": {str(k): v for k, v in a0.items()}}
    for k, sd in enumerate(seeds):
        y0 = top + k * pitch
        s.text(28, y0 + 14, f"RUN {sd}", bold=True, size=13)
        for j, ln in enumerate(SEED_LINES[sd]):
            s.text(28, y0 + 32 + 13 * j, ln, "t2", size=10)
        for key, x0, w, vmax, ticks, _, _ in cols:
            Y = lambda v: y0 + ph - v / vmax * ph
            for t in ticks:
                s.add(f'<line class="kgrid" stroke-width="2" x1="{x0}" x2="{x0 + w}" y1="{Y(t):.0f}" y2="{Y(t):.0f}"/>')
                s.text(x0 - 6, Y(t) + 3, f"{t:g}" if key == "T" else f"{t:.0%}", "t2", anchor="end", size=9)
            s.add(f'<line class="kink" stroke-width="2" x1="{x0}" x2="{x0 + w}" y1="{Y(0):.0f}" y2="{Y(0):.0f}"/>')
            step = w / len(drawn)
            for i, m in enumerate(drawn):
                c = a1[(sd, m)]
                bx = round(x0 + i * step + (step - BW) / 2)
                lv = c["P1_terms"]["level"] / c["P1"] if c["P1"] else 0.0
                v = c["T"] if key == "T" else lv
                tip = (f"run {sd}, {fm_(m)}: {c['T']} milestones on average ({c['T_gt0']}/8 episodes above 0)"
                       if key == "T" else f"run {sd}, {fm_(m)}: levelling {lv:.0%} of the capped score")
                hgt = v / vmax * ph
                if hgt >= 1:
                    s.rect(bx, round(Y(0) - hgt), BW, round(hgt), "a1" if key == "T" else "level", tip=tip)
                    ty = round(Y(0) - hgt) - 4
                    if key == "T" and m in a0 and -4 <= Y(0) - hgt - Y(a0[m]) <= 16:
                        ty = round(Y(a0[m])) - 6       # the batch-2 tick sits just above this bar: label above the tick
                    if key == "T" or v >= 0.01:
                        s.text(bx + BW / 2, ty, num(v) if key == "T" else f"{v:.0%}", "t3",
                               anchor="middle", size=9, bold=True)
                else:
                    s.rect(bx, round(Y(0)) - 3, BW, 2, "rnd", tip=tip)
                if key == "T" and m in a0:              # the batch-2 average at this checkpoint: a blue tick
                    s.rect(bx - 4, round(Y(a0[m])) - 1, BW + 8, 3, "a0",
                           tip=f"before the patch (batch 2, 5 runs), {fm_(m)}: {a0[m]:.2f} milestones on average")
                s.text(bx + BW / 2, Y(0) + 13, fm_(m).rstrip("M"), "t2", anchor="middle", size=9)
        for i, m in enumerate(marks):
            c = a1[(sd, m)]
            lv = c["P1_terms"]["level"] / c["P1"] if c["P1"] else 0.0
            data["runs"].setdefault(f"run {sd}", []).append({"mark": m, "T": c["T"], "T_gt0": c["T_gt0"],
                                                              "level_share_of_P1": round(lv, 4), "P1": c["P1"]})
    for _, x0, w, *_ in cols:
        s.text(x0 + w / 2, top + 2 * pitch + ph + 30, "training steps (millions)", "t2", anchor="middle", size=10)
    ly = H - 40
    s.rect(28, ly - 10, 12, 12, "a1"); s.text(46, ly, "real progress", "t3", size=10)
    s.rect(160, ly - 6, 24, 3, "a0"); s.text(192, ly, "before the patch (batch 2, average of 5 runs)", "t3", size=10)
    s.rect(500, ly - 10, 12, 12, "level"); s.text(518, ly, "score from levelling up", "t3", size=10)
    s.text(28, ly + 16, "each checkpoint is one sample: progress swings between checkpoints", "t2", size=9)
    data["caption"] = ("Round 1, one row per training run on the capped score. Run 1 made progress only at 0.5M and "
                       "then earned more of its score by levelling up; run 2 never reached a milestone and levelled late; run 3 made "
                       "progress at four checkpoints and never levelled much. The blue ticks are the uncapped-score "
                       "agent of batch 2.")
    write("round1-seeds", s.done(), data)


def fm_(step):
    return "0" if step == 0 else f"{step / 1e6:.2f}".rstrip("0").rstrip(".") + "M"

FIGS = {"route-map": route_map, "score-vs-progress": lambda: score_vs_progress("b2"),
        "score-vs-progress-b1": lambda: score_vs_progress("b1"), "where-they-went": where_they_went, "reward-sources": reward_sources, "round1-seeds": round1_seeds}

if __name__ == "__main__":
    for n in (sys.argv[1:] or FIGS):
        FIGS[n]()
