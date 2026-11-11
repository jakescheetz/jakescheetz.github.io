#!/usr/bin/env python3
"""Fig. 5 (paper.md "Figures" item 5): round 1, A1 under P1, D1-trained, own state D1, dev only.

  python3 fig5_r1.py [--out results/r1]      -> OUT/fig5.svg, OUT/fig5.md (table view + ASCII chart), OUT/fig5.json

(a) True progress T per checkpoint, one line per A1 seed (8 eval episodes each), with A0 batch 2 (D1-trained, 5 seeds
    pooled, 95% bootstrap band from results/b2/fig1.json) for comparison. R0 is 0 at every mark.
(b) P1 split by term (explore / level / heal; seen is 0 everywhere), one small panel per seed, R0 under P1 as a
    dashed line.
Inputs: results/r1/cells.json (r1_dev.py), results/b2/fig1.json + cells.json (batch-2 report), R0 under P1
(~/roms/runs/R0/eval-dev-24-p1/summary.json, r0_p1.py). The 2.5M interim read is left out (not a tested mark).
Standard library only. Palette: the dataviz reference categorical slots (A0 = slot 1 blue as in Fig. 1-4, A1 = slot 3
aqua; terms = slots 4, 5, 7), validated light and dark with validate_palette.js (light-mode contrast WARN on the
aqua, yellow and magenta: relieved by the legend, direct labels and the table in fig5.md).
"""
import argparse
import json
from html import escape
from pathlib import Path

CODE = Path(__file__).resolve().parent
R0SUM = Path.home() / "roms" / "runs" / "R0" / "eval-dev-24-p1" / "summary.json"
TERMS = ("explore", "level", "heal")
SEED_MARK = {1: "circle", 2: "square", 3: "triangle"}

CSS = """
svg { font-family: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace; font-size: 11px; }
.bg { fill: #fcfcfb; } .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .grid { stroke: #e4e3df; }
.axis { stroke: #52514e; } .r0 { stroke: #52514e; stroke-dasharray: 4 3; }
.A0 { stroke: #2a78d6; } .A0f { fill: #2a78d6; } .band { fill: #2a78d6; opacity: .14; stroke: none; }
.A1 { stroke: #1baf7a; } .A1f { fill: #1baf7a; } .ring { stroke: #fcfcfb; }
.explore { fill: #eda100; } .level { fill: #e87ba4; } .heal { fill: #6250d6; } .gap { stroke: #fcfcfb; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #1a1a19; } .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; } .grid { stroke: #383835; }
  .axis { stroke: #c3c2b7; } .r0 { stroke: #c3c2b7; }
  .A0 { stroke: #3987e5; } .A0f { fill: #3987e5; } .band { fill: #3987e5; }
  .A1 { stroke: #199e70; } .A1f { fill: #199e70; } .ring { stroke: #1a1a19; }
  .explore { fill: #c98500; } .level { fill: #d55181; } .heal { fill: #9085e9; } .gap { stroke: #1a1a19; }
}
"""


def fm(step):
    return "0" if step == 0 else f"{step / 1e6:.2f}".rstrip("0").rstrip(".") + "M"


def load():
    cells = [c for c in json.loads((CODE / "results/r1/cells.json").read_text())["rows"]
             if c["eval_state"] == "D1" and not c["interim"]]
    seeds = sorted({c["seed"] for c in cells})
    marks = sorted({c["mark"] for c in cells})
    a1 = {(c["seed"], c["mark"]): c for c in cells}
    f1 = json.loads((CODE / "results/b2/fig1.json").read_text())["D1→D1"]["A0"]
    a0 = {m["step"]: (m["T"], m["T_ci"][0], m["T_ci"][1]) for m in f1["marks"]}
    b2c = [c for c in json.loads((CODE / "results/b2/cells.json").read_text())
           if c["arm"] == "A0" and c["train_state"] == "D1" and c["eval_state"] == "D1" and c["mark"] == 0]
    if b2c:                                           # ckpt-0 (untrained) mark, if the batch-2 report has it
        a0[0] = (sum(c["T_mean"] for c in b2c) / len(b2c), None, None)
    r0 = json.loads(R0SUM.read_text())["summary"]["random-0.pt|D1"]
    return seeds, marks, a1, a0, r0, f1["runs"]


def svg_fig(seeds, marks, a1, a0, r0):
    W, H = 720, 580
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-labelledby="t5 d5"><title id="t5">Fig. 5: round 1, true progress and patched reward by term</title>'
         f'<desc id="d5">(a) T per checkpoint for A1 seeds 1-3 and A0 batch 2 on D1. (b) P1 by term per checkpoint, '
         f'one panel per A1 seed, R0 dashed. Values in fig5.md.</desc><style>{CSS}</style>',
         f'<rect class="bg" width="{W}" height="{H}"/>']
    # ---- (a) T vs training steps, linear x ----
    x0, y0, w, h = 60, 52, 470, 190
    X = lambda v: x0 + v / 5e6 * w
    Y = lambda v: y0 + h - v / 3.0 * h
    s.append(f'<text class="ink" x="{x0 - 50}" y="22" font-weight="bold">(a) True progress T on D1 (own training '
             f'state), per checkpoint</text>')
    for t in (0, 1, 2, 3):
        s.append(f'<line class="grid" x1="{x0}" x2="{x0 + w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="ink2" x="{x0 - 14}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for t in (0, 1e6, 2e6, 3e6, 4e6, 5e6):
        s.append(f'<text class="ink2" x="{X(t):.1f}" y="{y0 + h + 16}" text-anchor="middle">{fm(int(t))}</text>')
    s.append(f'<line class="axis" x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}"/>'
             f'<text class="ink2" x="{x0 + w}" y="{y0 + h + 32}" text-anchor="end">training steps</text>'
             f'<text class="ink2" x="{x0 - 36}" y="{y0 + h / 2}" transform="rotate(-90 {x0 - 36} {y0 + h / 2})" '
             f'text-anchor="middle">T (milestones, mean of 8 episodes)</text>')
    band = [(m, lo, hi) for m, (_, lo, hi) in sorted(a0.items()) if lo is not None]
    s.append('<polygon class="band" points="' + " ".join(f"{X(m):.1f},{Y(hi):.1f}" for m, _, hi in band) + " " +
             " ".join(f"{X(m):.1f},{Y(lo):.1f}" for m, lo, _ in reversed(band)) + '"/>')
    pts = sorted(a0.items())
    s.append('<polyline class="A0" fill="none" stroke-width="2" points="' +
             " ".join(f"{X(m):.1f},{Y(v[0]):.1f}" for m, v in pts) + '"/>')
    for m, (t, lo, hi) in pts:
        ci = f" [95% CI {lo}, {hi}]" if lo is not None else ""
        s.append(f'<circle class="A0f ring" stroke-width="2" cx="{X(m):.1f}" cy="{Y(t):.1f}" r="4">'
                 f'<title>A0 batch 2 (5 seeds), {fm(m)}: T {t:.3f}{ci}</title></circle>')
    off = {1: -5, 2: 0, 3: 5}                         # small x jitter so equal values stay visible
    for sd in seeds:
        p = [(m + off[sd] * 5e6 / w, a1[(sd, m)]["T"], a1[(sd, m)]) for m in marks]
        s.append(f'<polyline class="A1" fill="none" stroke-width="{1.5}" opacity=".9" '
                 f'{"stroke-dasharray=\"6 3\" " if sd == 2 else "stroke-dasharray=\"2 2\" " if sd == 3 else ""}'
                 'points="' + " ".join(f"{X(x):.1f},{Y(t):.1f}" for x, t, _ in p) + '"/>')
        for x, t, c in p:
            tip = (f"A1 s{sd}, {fm(c['mark'])}: T {c['T']} ({c['T_gt0']}/{c['n']} episodes &gt; 0), "
                   f"P1 {c['P1']}")
            cx, cy = X(x), Y(t)
            shape = SEED_MARK[sd]
            if shape == "circle":
                el = f'<circle class="A1f ring" stroke-width="2" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5">'
                end = "</circle>"
            elif shape == "square":
                el = f'<rect class="A1f ring" stroke-width="2" x="{cx - 4:.1f}" y="{cy - 4:.1f}" width="8" height="8" rx="1">'
                end = "</rect>"
            else:
                el = (f'<polygon class="A1f ring" stroke-width="2" points="{cx:.1f},{cy - 5.5:.1f} '
                      f'{cx + 5:.1f},{cy + 3.5:.1f} {cx - 5:.1f},{cy + 3.5:.1f}">')
                end = "</polygon>"
            s.append(f"{el}<title>{tip}</title>{end}")
    # legend (right of panel a)
    lx, ly = x0 + w + 24, y0 + 6
    s.append(f'<text class="ink" x="{lx}" y="{ly}">Legend</text>')
    items = [("A0", "A0, batch 2", None), ("A1", "A1 seed 1", "circle"),
             ("A1", "A1 seed 2", "square"), ("A1", "A1 seed 3", "triangle")]
    for i, (arm, lab, shape) in enumerate(items):
        yy = ly + 20 + i * 20
        dash = {"square": ' stroke-dasharray="6 3"', "triangle": ' stroke-dasharray="2 2"'}.get(shape, "")
        s.append(f'<line class="{arm}" stroke-width="2" x1="{lx}" x2="{lx + 26}" y1="{yy}" y2="{yy}"{dash}/>')
        if shape == "square":
            s.append(f'<rect class="A1f" x="{lx + 9}" y="{yy - 4}" width="8" height="8" rx="1"/>')
        elif shape == "triangle":
            s.append(f'<polygon class="A1f" points="{lx + 13},{yy - 5.5} {lx + 18},{yy + 3.5} {lx + 8},{yy + 3.5}"/>')
        else:
            s.append(f'<circle class="{arm}f" cx="{lx + 13}" cy="{yy}" r="4"/>')
        s.append(f'<text class="ink" x="{lx + 34}" y="{yy + 4}">{escape(lab)}</text>')
    for i, t in enumerate(("A0: 5 seeds, P0;", "band = 95% CI", "A1: P1, 8 eps", "R0 (random):", "T 0 everywhere")):
        s.append(f'<text class="ink2" x="{lx}" y="{ly + 106 + 16 * i}">{t}</text>')
    # ---- (b) P1 by term, one panel per seed ----
    top = 330
    s.append(f'<text class="ink" x="10" y="{top - 30}" font-weight="bold">(b) Patched reward P1 by term on D1, '
             f'per checkpoint (mean of 8 episodes)</text>')
    pw, ph, gap = 196, 150, 34
    ymax = 20
    r0p = r0["P_mean"]
    for k, sd in enumerate(seeds):
        px = 60 + k * (pw + gap)
        Yb = lambda v: top + ph - v / ymax * ph
        s.append(f'<text class="ink" x="{px}" y="{top - 8}">A1 seed {sd}</text>')
        for t in (0, 5, 10, 15, 20):
            s.append(f'<line class="grid" x1="{px}" x2="{px + pw}" y1="{Yb(t):.1f}" y2="{Yb(t):.1f}"/>')
            if k == 0:
                s.append(f'<text class="ink2" x="{px - 8}" y="{Yb(t) + 4:.1f}" text-anchor="end">{t}</text>')
        bw = pw / len(marks)
        for i, m in enumerate(marks):
            c = a1[(sd, m)]
            bx, acc = px + i * bw + 3, 0.0
            for t in TERMS:
                v = c["P1_terms"][t]
                if v <= 0:
                    continue
                y1, y2 = Yb(acc + v), Yb(acc)
                s.append(f'<rect class="{t} gap" stroke-width="1" x="{bx:.1f}" y="{y1:.1f}" width="{bw - 6:.1f}" '
                         f'height="{max(y2 - y1, 0.5):.1f}"><title>A1 s{sd}, {fm(m)}: {t} {v} of P1 {c["P1"]}'
                         f'</title></rect>')
                acc += v
            s.append(f'<text class="ink2" x="{bx + (bw - 6) / 2:.1f}" y="{top + ph + 14}" text-anchor="middle" '
                     f'font-size="10">{fm(m).rstrip("M") or "0"}</text>')
        s.append(f'<line class="r0" stroke-width="1.5" x1="{px}" x2="{px + pw}" y1="{Yb(r0p):.1f}" y2="{Yb(r0p):.1f}">'
                 f'<title>R0 under P1 (24 episodes): {r0p}</title></line>')
        s.append(f'<line class="axis" x1="{px}" x2="{px + pw}" y1="{top + ph}" y2="{top + ph}"/>'
                 f'<text class="ink2" x="{px + pw}" y="{top + ph + 28}" text-anchor="end">steps (M)</text>')
    s.append(f'<text class="ink2" x="{24}" y="{top + ph / 2}" transform="rotate(-90 24 {top + ph / 2})" '
             f'text-anchor="middle">P1</text>')
    ly2 = top + ph + 56
    for i, (t, lab) in enumerate((("explore", "explore (capped 20 new tiles/map)"), ("level", "level"),
                                  ("heal", "heal"))):
        lx2 = (60, 340, 430)[i]
        s.append(f'<rect class="{t}" x="{lx2}" y="{ly2 - 9}" width="12" height="12" rx="2"/>'
                 f'<text class="ink" x="{lx2 + 18}" y="{ly2 + 1}">{escape(lab)}</text>')
    s.append(f'<line class="r0" stroke-width="1.5" x1="510" x2="536" y1="{ly2 - 3}" y2="{ly2 - 3}"/>'
             f'<text class="ink" x="542" y="{ly2 + 1}">R0 under P1 ({r0p})</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def ascii_fig(seeds, marks, a1, a0, r0, html):
    """Monospace chart for the post: (a) T per mark, equal-spaced checkpoints; (b) P1 stacked bars per seed × mark.
    html=True wraps markers in the site's chart-* spans (assets/site.css)."""
    sp = (lambda cls, ch: f'<span class="chart-{cls}">{ch}</span>') if html else (lambda cls, ch: ch)
    col = 8
    rows = [round(v * 4) / 4 for v in (3.0 - i * 0.25 for i in range(13))]
    grid = {r: [" "] * (col * len(marks)) for r in rows}
    q = lambda v: min(rows, key=lambda r: abs(r - v))
    for i, m in enumerate(marks):
        if m in a0:
            grid[q(a0[m][0])][i * col + 6] = sp("a0", "◆")
        for sd, ch in zip(seeds, "123"):
            grid[q(a1[(sd, m)]["T"])][i * col + 1 + sd] = sp("a1", ch)
    L = ["(a) True progress T on D1, per checkpoint", ""]
    for r in rows:
        lab = f"{r:4.1f} ┤" if r == int(r) else "     │"
        line = "".join(grid[r])
        if r == 0:
            line = "".join(c if c != " " else "─" for c in grid[r])
        L.append(lab + line.rstrip())
    L.append("      " + "".join(fm(m).center(col) for m in marks).rstrip())
    L.append("      checkpoints (equally spaced; not to scale)")
    L += ["", f"      {sp('a1', '1')} {sp('a1', '2')} {sp('a1', '3')}  A1 seeds 1-3 (P1)    "
          f"{sp('a0', '◆')} A0 batch 2 (P0), 5 seeds    R0: 0 everywhere", ""]
    # (b) bars: 1 char = 0.5 P1
    unit = 0.5
    r0c = int(round(r0["P_mean"] / unit))
    ch = {"explore": "█", "level": "▓", "heal": "░"}
    L += ["(b) P1 by term on D1 (1 char = 0.5; ┆ = R0 under P1, " + f"{r0['P_mean']})", ""]
    for sd in seeds:
        for m in marks:
            c = a1[(sd, m)]
            bar, acc = [], 0.0
            for t in TERMS:
                n = int(round((acc + c["P1_terms"][t]) / unit)) - int(round(acc / unit))
                acc += c["P1_terms"][t]
                bar += [(t, ch[t])] * max(n, 0)
            cells = [sp(t, x) for t, x in bar]
            while len(cells) <= r0c:
                cells.append(" ")
            if len(bar) <= r0c:
                cells[r0c] = "┆"
            L.append(f"s{sd} {fm(m):>5} │" + "".join(cells).rstrip() + f"  {c['P1']:.2f}")
        L.append("")
    L.append(f"      {sp('explore', '█')} explore   {sp('level', '▓')} level   {sp('heal', '░')} heal")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CODE / "results" / "r1"))
    a = ap.parse_args()
    out = Path(a.out)
    seeds, marks, a1, a0, r0, a0runs = load()
    (out / "fig5.svg").write_text(svg_fig(seeds, marks, a1, a0, r0))
    plain, html = ascii_fig(seeds, marks, a1, a0, r0, False), ascii_fig(seeds, marks, a1, a0, r0, True)
    md = ["# Fig. 5: round 1 (A1 under P1), true progress and P1 by term, D1, dev", "",
          "A1 = MapCapProxy(map_new_tile_cap=20), D1-trained, 3 seeds, 8 eval episodes per seed per checkpoint on D1 "
          "(own training state). A0 = batch 2, D1-trained, 5 seeds pooled (" + ", ".join(a0runs) + "), 95% "
          "bootstrap CI over seeds; A0 is trained and scored under P0, so only T is compared. R0 (random) T is 0 "
          f"at every mark; R0 under P1 = {r0['P_mean']}. The 2.5M interim read is left out. Generated by "
          "`fig5_r1.py` from `results/r1/cells.json` and `results/b2/fig1.json`. SVG: `fig5.svg`.", "",
          "## Table view", "",
          "| mark | A0 b2 T [95% CI] | " + " | ".join(f"A1 s{sd} T (eps > 0)" for sd in seeds) + " | " +
          " | ".join(f"A1 s{sd} P1 = explore / level / heal" for sd in seeds) + " |",
          "|---|---|" + "---|" * (2 * len(seeds))]
    for m in marks:
        t0 = a0.get(m)
        a0s = "" if t0 is None else (f"{t0[0]:.3f}" + (f" [{t0[1]}, {t0[2]}]" if t0[1] is not None else ""))
        md.append(f"| {fm(m)} | {a0s} | " + " | ".join(
            f"{a1[(sd, m)]['T']} ({a1[(sd, m)]['T_gt0']}/8)" for sd in seeds) + " | " + " | ".join(
            f"{a1[(sd, m)]['P1']} = " + " / ".join(str(a1[(sd, m)]['P1_terms'][t]) for t in TERMS) for sd in seeds) + " |")
    md += ["", "## ASCII chart (plain)", "", "```"] + plain + ["```", "",
           "## For the post (raw HTML; needs the chart-a0 / chart-a1 / chart-explore / chart-level / chart-heal "
           "classes in assets/site.css)", "", "```html", "<figure>", "<pre>" + "\n".join(html) + "</pre>",
           "<figcaption>Round 1: true progress per checkpoint for each seed trained on the capped reward, against "
           "the uncapped run of batch 2; and the capped reward split by term.</figcaption>", "</figure>", "```"]
    (out / "fig5.md").write_text("\n".join(md) + "\n")
    (out / "fig5.json").write_text(json.dumps(
        {"marks": marks, "A0_b2_T": {str(m): v for m, v in a0.items()}, "A0_runs": a0runs, "R0_P1": r0["P_mean"],
         "A1": {f"s{sd}": {fm(m): {k: a1[(sd, m)][k] for k in ("T", "T_gt0", "P1", "P1_terms")} for m in marks}
                for sd in seeds}}, indent=1))
    print("\n".join(plain))
    print(f"wrote {out / 'fig5.svg'}, fig5.md, fig5.json")


if __name__ == "__main__":
    main()
