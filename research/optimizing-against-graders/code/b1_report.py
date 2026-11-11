"""Batch-1 report parts called by analyze_b1.py (experiment-e1.md §9 "Batch-1 analysis plan" C, D, E):

  fig1(...)        Fig. 1 (§9 C) as stdlib SVG + a table view (fig1.md / fig1.json)
  h0_tests(...)    H0 (i)–(iii), verdict per training state, leg progress as secondary (§9 D)
  catalogue(...)   per type × mark: prevalence, share of P, first appearance pooled + per seed (§9 E)

All inputs are episode rows (dev views, or the sealed held-out rows when analyze_b1 builds the sealed part)
plus r0_compare rows. Nothing here reads files except the traces (share of P) and nothing prints values.

Operational choices, as ruled in experiment-e1.md §9 B′ (2026-11-07):
- **(a) No dual axis.** A P panel stacked over a T panel with the same x; leg progress (same units as T) is the
  dashed line in the T panel. The KL-x version uses the same layout.
- **(b) Collapsed A2″ seed-checkpoint** (§9 (1)): that seed's mean T on its **own training state** at that mark
  is ≥ 1 below its best mean T at any earlier mark. It is dropped from the A2″ reference in every cell of that
  checkpoint (primary and cross): Fig. 1 hollow, H0 (ii), and attribution / A2″-vs-R0 (analyze_b1 passes
  r0_compare --drop). A cell with no A2″ seed left is labelled "undetermined (reference collapsed)".
- **(c) Per-seed first appearance (descriptive):** the seed's 8 episodes vs R0's 24, same MC permutation, Holm
  per detector over that seed's 7 own-state cells; no tick = not detected. Never makes a type present.
- **(d) (iii):** "not met (degenerate)" if A0's pooled P first reaches 50% of its 5M value only at 5M;
  "undefined" if pooled P at 5M is ≤ 0 or ≤ its 0.25M value. Either way H0 can't be supported on that state.
- **(e) Monotone within CI** (falsifier) = for every pair of consecutive marks the 95% seed-bootstrap CI of
  T(next) − T(prev) reaches 0 or above; no multiplicity correction (conservative against the falsifier).
- **(f) Share of P** = Σ per-step p inside the type's merged event ranges ÷ Σ p, over the A0 own-state eval
  episodes (from the saved traces); a ratio of pooled sums. Ranking pools all marks and both training states;
  ties by prevalence. (From round 1, if P_r has negative terms: positive part, net alongside; not needed in P0.)
- **(g) A2″-HA runs** are in no dev or Holm family; they enter only the held-out analysis after unsealing.
"""
import gzip
import json
from html import escape
from math import log10
from pathlib import Path

import numpy as np

import r0_compare as R

SEED = 20261106
B_CI = 10_000
ALPHA = R.ALPHA
HACKABLE = ("L", "HF", "G", "B", "M", "D", "O")          # K = stuck (not a hack), U = review queue (never counted)
ARMS = ("A0", "A2pp")
ARM_LABEL = {"A0": "A0", "A1": "A1", "A2pp": "A2″", "A2ppHA": "A2″ HA-trained"}   # A2ppHA: §9 (g) reference line, HA panels only


# ---------------------------------------------------------------- stats helpers
def seed_boot(groups, b=B_CI, seed=SEED):
    """95% CI of the pooled mean: resample seeds with replacement, then episodes within each drawn seed."""
    gs = [np.asarray(g, float) for g in groups if len(g)]
    if not gs:
        return None, None
    rng = np.random.default_rng(seed)
    s = len(gs)
    if len({len(g) for g in gs}) == 1:
        a, n = np.stack(gs), len(gs[0])
        si = rng.integers(0, s, (b, s))
        ei = rng.integers(0, n, (b, s, n))
        means = a[si[:, :, None], ei].mean(axis=(1, 2))
    else:
        means = np.array([np.concatenate([gs[i][rng.integers(0, len(gs[i]), len(gs[i]))]
                                          for i in rng.integers(0, s, s)]).mean() for _ in range(b)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def seed_boot_diff(ga, gb, b=B_CI, seed=SEED):
    """95% CI of mean(b) − mean(a) for per-seed groups at two marks: the same seeds are drawn for both."""
    ga = [np.asarray(g, float) for g in ga]
    gb = [np.asarray(g, float) for g in gb]
    rng = np.random.default_rng(seed)
    s, out = len(ga), np.empty(b)
    if s and len({len(g) for g in ga + gb}) == 1:
        a, c, n = np.stack(ga), np.stack(gb), len(ga[0])
        si = rng.integers(0, s, (b, s))[:, :, None]
        d = c[si, rng.integers(0, n, (b, s, n))].mean(axis=(1, 2)) - a[si, rng.integers(0, n, (b, s, n))].mean(axis=(1, 2))
        lo, hi = np.quantile(d, [0.025, 0.975])
        return float(lo), float(hi)
    for k in range(b):
        si = rng.integers(0, s, s)
        xa = np.concatenate([ga[i][rng.integers(0, len(ga[i]), len(ga[i]))] for i in si])
        xb = np.concatenate([gb[i][rng.integers(0, len(gb[i]), len(gb[i]))] for i in si])
        out[k] = xb.mean() - xa.mean()
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi)


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else None


def rnd(x, k=4):
    return None if x is None else round(x, k)


def by_seed(info, rows, arm, train_state, eval_state, step, field, drop=()):
    """[[values of one seed's episodes], ...] for the arm's runs trained on train_state, at one mark."""
    out = []
    for run in sorted(r for r, d in info.items() if d["arm"] == arm and d["state"] == train_state):
        if (run, step) in drop:
            continue
        v = [e[field] for e in rows[run] if e["step"] == step and e["state"] == eval_state
             and e.get(field) is not None]
        if v:
            out.append(v)
    return out


def collapsed(info, rows):
    """{(run, step)} A2″ seed-checkpoints collapsed on their own training state (§9 (1))."""
    out = set()
    for run, d in info.items():
        if d["arm"] != "A2pp":
            continue
        best = None
        for step in sorted({e["step"] for e in rows[run]}):
            t = mean(e["T"] for e in rows[run] if e["step"] == step and e["state"] == d["state"])
            if t is None:
                continue
            if best is not None and t <= best - 1:
                out.add((run, step))
            best = t if best is None else max(best, t)
    return out


def first_appearance(r0_rows, train_states):
    """Pooled: {(train_state, type): first mark present after Holm} from the arm's primary vs-R0 rows."""
    out = {}
    for r in r0_rows:
        if r.get("counts_after_holm") and r["state"] in train_states:
            step = int(r["ckpt"].split("-")[1].split(".")[0])
            k = (r["state"], r["detector"])
            out[k] = min(out.get(k, step), step)
    return out


def per_seed_first(info, primary_dir, r0_dir, arm="A0"):
    """{(run, type): first mark present} with each seed alone vs R0 (Holm over that seed's 7 cells)."""
    out = {}
    for run, d in sorted(info.items()):
        if d["arm"] != arm:
            continue
        for r in R.compare([primary_dir(run)], r0_dir):
            if r.get("counts_after_holm"):
                step = int(r["ckpt"].split("-")[1].split(".")[0])
                out[(run, r["detector"])] = min(out.get((run, r["detector"]), step), step)
    return out


# ---------------------------------------------------------------- SVG
CSS = """
svg { font-family: ui-monospace, "JetBrains Mono", Menlo, Consolas, monospace; font-size: 11px; }
polyline { fill: none; }
.bg { fill: #fcfcfb; } .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .grid { stroke: #e4e3df; }
.axis { stroke: #52514e; } .A0 { stroke: #2a78d6; } .A2pp { stroke: #eb6834; } .A2ppHA { stroke: #b8336a; }
circle.A2ppHA { fill: #b8336a; } polygon.band.A2ppHA { fill: #b8336a; }
circle.A0 { fill: #2a78d6; } circle.A2pp { fill: #eb6834; } circle.hollow, circle.axis { fill: #fcfcfb; }
polygon.band { stroke: none; opacity: .16; } polygon.band.A0 { fill: #2a78d6; } polygon.band.A2pp { fill: #eb6834; }
.tick1 { stroke: #0b0b0b; } .tick2 { stroke: #52514e; opacity: .55; }
@media (prefers-color-scheme: dark) {
  .bg, circle.hollow, circle.axis { fill: #1a1a19; } .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; }
  .grid { stroke: #383835; } .axis { stroke: #c3c2b7; } .A0 { stroke: #3987e5; } .A2pp { stroke: #d95926; }
  circle.A0 { fill: #3987e5; } circle.A2pp { fill: #d95926; } circle.hollow { fill: #1a1a19; }
  polygon.band.A0 { fill: #3987e5; } polygon.band.A2pp { fill: #d95926; }
  .A2ppHA { stroke: #f39aa5; } circle.A2ppHA { fill: #f39aa5; } polygon.band.A2ppHA { fill: #f39aa5; }
  .tick1 { stroke: #f0efec; } .tick2 { stroke: #c3c2b7; }
}
"""


class Plot:
    """One small panel: x log or linear, y linear, recessive grid, 2px pooled lines, thin per-seed lines."""

    def __init__(self, svg, x, y, w, h, xdom, ydom, xlog, title, ylabel, xlabel):
        self.s, self.x, self.y, self.w, self.h = svg, x, y, w, h
        self.xdom, self.ydom, self.xlog = xdom, ydom, xlog
        f = (lambda v: log10(v)) if xlog else (lambda v: v)
        self.fx = f
        a, b = f(xdom[0]), f(xdom[1])
        self.X = lambda v: x + (f(v) - a) / ((b - a) or 1) * w
        self.Y = lambda v: y + h - (v - ydom[0]) / ((ydom[1] - ydom[0]) or 1) * h
        s = svg
        s.append(f'<text class="ink" x="{x}" y="{y - 8}" font-weight="bold">{escape(title)}</text>')
        for t in nice(ydom):
            s.append(f'<line class="grid" x1="{x}" x2="{x + w}" y1="{self.Y(t):.1f}" y2="{self.Y(t):.1f}"/>')
            s.append(f'<text class="ink2" x="{x - 6}" y="{self.Y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
        xt = [2.5e5, 5e5, 1e6, 2e6, 3e6, 5e6] if xlog else nice(xdom)
        for t in xt:
            if xdom[0] <= t <= xdom[1]:
                lab = (f"{t / 1e6:g}M" if t >= 1e6 else f"{t / 1e3:g}k") if xlog else f"{t:g}"
                s.append(f'<text class="ink2" x="{self.X(t):.1f}" y="{y + h + 14}" text-anchor="middle">{lab}</text>')
        s.append(f'<line class="axis" x1="{x}" x2="{x + w}" y1="{y + h}" y2="{y + h}"/>')
        s.append(f'<text class="ink2" x="{x + w}" y="{y + h + 28}" text-anchor="end">{escape(xlabel)}</text>')
        s.append(f'<text class="ink2" x="{x - 34}" y="{y + h / 2}" transform="rotate(-90 {x - 34} {y + h / 2})" '
                 f'text-anchor="middle">{escape(ylabel)}</text>')

    def band(self, arm, pts):
        pts = [p for p in pts if p[2] is not None]
        if len(pts) < 2:
            return
        up = " ".join(f"{self.X(x):.1f},{self.Y(hi):.1f}" for x, _, _, hi in pts)
        dn = " ".join(f"{self.X(x):.1f},{self.Y(lo):.1f}" for x, _, lo, _ in reversed(pts))
        self.s.append(f'<polygon class="band {arm}" points="{up} {dn}"/>')

    def line(self, arm, pts, width=2, dash=False, opacity=1, r=4, tip=""):
        pts = [p for p in pts if p[1] is not None]
        if not pts:
            return
        d = " ".join(f"{self.X(p[0]):.1f},{self.Y(p[1]):.1f}" for p in pts)
        self.s.append(f'<polyline class="{arm}" fill="none" stroke-width="{width}" opacity="{opacity}" '
                      f'{"stroke-dasharray=\"5 3\" " if dash else ""}points="{d}"/>')
        for p in pts:
            hollow = len(p) > 2 and p[2]
            self.s.append(f'<circle class="{arm}{" hollow" if hollow else ""}" stroke-width="1.5" '
                          f'opacity="{opacity}" cx="{self.X(p[0]):.1f}" cy="{self.Y(p[1]):.1f}" r="{r}">'
                          f'<title>{escape(tip)} x={p[0]:,.4g} y={p[1]:.3f}{" (collapsed)" if hollow else ""}</title>'
                          f'</circle>')

    def ticks(self, items):
        """items: [(x, label, level)] level 1 = pooled (ink), 2 = per seed (light); stacked above the panel."""
        rows = {}
        for x, lab, lvl in sorted(items):
            k = (lvl, x)
            rows[k] = rows.get(k, -1) + 1
            yy = self.y + 4 + rows[k] * 11 + (0 if lvl == 1 else 26)
            cls = "tick1" if lvl == 1 else "tick2"
            self.s.append(f'<line class="{cls}" x1="{self.X(x):.1f}" x2="{self.X(x):.1f}" y1="{yy - 4}" y2="{yy + 4}"/>')
            self.s.append(f'<text class="{"ink" if lvl == 1 else "ink2"}" x="{self.X(x) + 3:.1f}" y="{yy + 4}" '
                          f'font-size="9">{escape(lab)}</text>')


def nice(dom, n=4):
    lo, hi = dom
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** int(np.floor(np.log10(raw)))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    t, out = np.ceil(lo / step) * step, []
    while t <= hi + 1e-9:
        out.append(round(float(t), 10))
        t += step
    return out


CHAR_W = 6.8                                   # 11 px monospace advance (≈ 0.62 em)


def legend_items(extra=()):
    """(kind, class, label): kind line / dash / hollow / thin; extra = [(class, label)] series."""
    return ([("line", "A0", "A0"), ("line", "A2pp", "A2″")] + [("line", c, l) for c, l in extra] +
            [("dash", "axis", "leg progress (T panel)"), ("hollow", "axis", "collapsed seed-ckpt"),
             ("thin", "axis", "one seed")])


def legend_layout(x, width, extra=()):
    """Flow the legend items into rows that fit width: [(row, x, item)]."""
    out, row, xx = [], 0, x
    for it in legend_items(extra):
        w = 28 + len(it[2]) * CHAR_W + 22
        if xx > x and xx + w > x + width:
            row, xx = row + 1, x
        out.append((row, xx, it))
        xx += w
    return out


def legend(svg, x, y, width=10_000, extra=()):
    """Draws the legend; returns the number of rows used (18 px each)."""
    lay = legend_layout(x, width, extra)
    for row, xx, (kind, cls, lab) in lay:
        yy = y + 18 * row
        if kind == "hollow":
            svg.append(f'<circle class="axis hollow" cx="{xx + 6}" cy="{yy}" r="3.5" stroke-width="1.5"/>')
            svg.append(f'<text class="ink" x="{xx + 16}" y="{yy + 4}">{escape(lab)}</text>')
            continue
        attr = {"line": 'stroke-width="2"', "dash": 'stroke-width="2" stroke-dasharray="5 3"',
                "thin": 'stroke-width="1" opacity=".5"'}[kind]
        svg.append(f'<line class="{cls}" x1="{xx}" x2="{xx + 22}" y1="{yy}" y2="{yy}" {attr}/>')
        svg.append(f'<text class="ink" x="{xx + 28}" y="{yy + 4}">{escape(lab)}</text>')
    return 1 + max(r for r, _, _ in lay)


def wrap(lines, width, char_w=CHAR_W):
    import textwrap
    n = max(20, int(width / char_w))
    return [w for line in lines for w in (textwrap.wrap(line, n) or [""])]


# ---------------------------------------------------------------- Fig. 1
def fig1_data(info, rows, cells, kl_bad, coll):
    """cells: [(train_state, eval_state)]. Returns {cell: {arm: {...}}} with pooled means, CIs, per-seed lines."""
    steps = sorted({e["step"] for r in rows.values() for e in r})
    out = {}
    for ts, es in cells:
        key = f"{ts}→{es}"
        out[key] = {}
        for arm in ARMS:
            runs = sorted(r for r, d in info.items() if d["arm"] == arm and d["state"] == ts)
            if not runs:
                continue
            a = {"runs": runs, "marks": []}
            for step in steps:
                m = {"step": step}
                for f in ("P", "T", "leg_progress", "kl_init"):
                    g = by_seed(info, rows, arm, ts, es, step, f,
                                drop={(r, s) for r in kl_bad for s in steps} if f == "kl_init" else ())
                    flat = [x for v in g for x in v]
                    m[f] = rnd(mean(flat))
                    if f != "kl_init" and g:
                        m[f + "_ci"] = [rnd(c) for c in seed_boot(g)]
                m["n"] = sum(len(v) for v in by_seed(info, rows, arm, ts, es, step, "T"))
                m["seeds"] = {}
                for run in runs:
                    ev = [e for e in rows[run] if e["step"] == step and e["state"] == es]
                    if ev:
                        m["seeds"][run] = {"P": rnd(mean(e["P"] for e in ev)), "T": rnd(mean(e["T"] for e in ev)),
                                           "kl_init": None if run in kl_bad else
                                           rnd(mean(e["kl_init"] for e in ev if "kl_init" in e)),
                                           "collapsed": (run, step) in coll}
                a["marks"].append(m)
            out[key][arm] = a
    return out


def fig1_svg(data, ticks, title, caption, with_kl=True, extra_legend=()):
    """ticks: {cell: [(step, label, level)]}. One row-pair (P over T) per cell; columns x = step [, x = KL]."""
    cells = list(data)
    cols = 2 if with_kl else 1
    pw, ph, gap = 330, 130, 64
    W = 70 + cols * (pw + 80)
    caption = wrap(caption, W - 32)
    tlines = wrap([title], W - 32, 8.1)                # 13 px bold
    lrows = 1 + max(r for r, _, _ in legend_layout(20, W - 40, extra_legend))
    top = 22 + 17 * (len(tlines) - 1)
    H = 70 + top - 22 + 18 * (lrows - 1) + len(cells) * (2 * ph + 2 * gap + 30) + 20 + 14 * len(caption)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">',
         f"<title>{escape(title)}</title><style>{CSS}</style>", f'<rect class="bg" width="{W}" height="{H}"/>']
    s += [f'<text class="ink" x="16" y="{22 + 17 * i}" font-size="13" font-weight="bold">{escape(t)}</text>'
          for i, t in enumerate(tlines)]
    legend(s, 20, top + 22, W - 40, extra_legend)
    y = top + 68 + 18 * (lrows - 1)
    for cell in cells:
        arms = data[cell]
        marks = [m["step"] for a in arms.values() for m in a["marks"]]
        if not marks:
            continue
        xd = (min(marks) * 0.9, max(marks) * 1.1)
        pmax = max([m.get("P_ci", [0, 0])[1] or 0 for a in arms.values() for m in a["marks"]] +
                   [v["P"] or 0 for a in arms.values() for m in a["marks"] for v in m["seeds"].values()] + [1])
        tmax = 3.0
        kls = [v["kl_init"] for a in arms.values() for m in a["marks"] for v in m["seeds"].values()
               if v["kl_init"] is not None]
        for c in range(cols):
            x0 = 70 + c * (pw + 80)
            xlog = c == 0
            dom = xd if xlog else ((0, max(kls) * 1.05) if kls else (0, 1))
            xl = "training step (log)" if xlog else "KL(π_ckpt ‖ π_init), mean per step"
            pp = Plot(s, x0, y, pw, ph, dom, (0, pmax * 1.05), xlog, f"{cell}: proxy P" +
                      ("" if xlog else " vs KL"), "P", xl)
            tp = Plot(s, x0, y + ph + gap, pw, ph, dom, (0, tmax), xlog, f"{cell}: true T, leg progress" +
                      ("" if xlog else " vs KL"), "milestones", xl)
            for arm, a in arms.items():
                ms = a["marks"]
                for f, plot in (("P", pp), ("T", tp)):
                    if xlog:
                        plot.band(arm, [(m["step"], m[f], *(m.get(f + "_ci") or (None, None))) for m in ms])
                    for run in a["runs"]:
                        pts = [((m["step"] if xlog else m["seeds"][run]["kl_init"]), m["seeds"][run][f],
                                m["seeds"][run]["collapsed"]) for m in ms if run in m["seeds"]]
                        pts = [p for p in pts if p[0] is not None]
                        plot.line(arm, pts, width=1, opacity=.5, r=2.5, tip=f"{run} {f}")
                    px = [(m["step"] if xlog else m["kl_init"], m[f]) for m in ms]
                    plot.line(arm, [p for p in px if p[0] is not None], tip=f"{ARM_LABEL[arm]} pooled {f}")
                lp = [(m["step"] if xlog else m["kl_init"], m["leg_progress"]) for m in ms]
                tp.line(arm, [p for p in lp if p[0] is not None], dash=True, r=0, tip=f"{ARM_LABEL[arm]} leg progress")
            if xlog and ticks.get(cell):
                pp.ticks(ticks[cell])
        y += 2 * ph + 2 * gap + 30
    for i, line in enumerate(caption):
        s.append(f'<text class="ink2" x="16" y="{y + 14 * i}">{escape(line)}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def fig1_md(data, title, notes):
    lines = [f"# {title}", "", *notes, "",
             "| panel | arm | mark | n | P [95% CI] | T [95% CI] | leg progress [95% CI] | KL init | collapsed seeds |",
             "|---|---|---|---|---|---|---|---|---|"]
    ci = lambda m, f: "" if m.get(f) is None else f"{m[f]} [{m.get(f + '_ci', ['', ''])[0]}, {m.get(f + '_ci', ['', ''])[1]}]"
    for cell, arms in data.items():
        for arm, a in arms.items():
            for m in a["marks"]:
                col = ", ".join(r for r, v in m["seeds"].items() if v["collapsed"])
                lines.append(f"| {cell} | {ARM_LABEL[arm]} | {m['step']:,} | {m['n']} | {ci(m, 'P')} | {ci(m, 'T')} | "
                             f"{ci(m, 'leg_progress')} | {'' if m['kl_init'] is None else m['kl_init']} | {col} |")
    return "\n".join(lines) + "\n"


def fig1(info, rows, out, name, cells, kl_bad, coll, ticks, with_kl, caption_extra=()):
    """Write OUT/NAME.svg, .md, .json. Returns the data."""
    data = fig1_data(info, rows, cells, kl_bad, coll)
    caption = [
        "Means pooled over seeds (8 eval episodes per seed per mark); band = 95% bootstrap over seeds, episodes",
        "resampled within seeds (B = 10,000). Thin lines = one seed; hollow = collapsed A2″ seed-ckpt (§9 (1)).",
        "Ticks above P: first mark a hack type is present in A0 above R0 after Holm (dark = seeds pooled,",
        "light = one seed alone, labelled type·seed). P and T are stacked panels, not two axes.",
        *caption_extra]
    title = "Fig. 1: proxy P and true T for A0 and A2″, by training state"
    Path(out / f"{name}.svg").write_text(fig1_svg(data, ticks, title, caption, with_kl))
    Path(out / f"{name}.md").write_text(fig1_md(data, title, caption))
    Path(out / f"{name}.json").write_text(json.dumps(data, indent=1))
    return data


# ---------------------------------------------------------------- H0 (§9 D)
def h0_tests(info, rows, coll, states=("D1", "D2"), metrics=(("T", "primary"), ("leg_progress", "secondary"))):
    """metrics: (field, role) pairs for (ii)–(iii); the default is the pre-registered T + leg progress. The §9 H10
    training series passes T only (training logs no leg progress)."""
    steps = sorted({e["step"] for r in rows.values() for e in r})
    s5 = steps[-1]
    s05 = min(steps, key=lambda s: abs(s - 500_000))
    late = [s for s in steps if s >= 1_900_000]                        # marks ≥ 2M (2, 3, 4, 5M)
    res = {"marks": steps, "mark_5M": s5, "mark_0.5M": s05, "marks_ge_2M": late, "B": R.P_B, "states": {}}
    flat = lambda g: [x for v in g for x in v]
    for ts in states:
        if not any(d["state"] == ts for d in info.values()):
            continue
        A0 = lambda step, f: by_seed(info, rows, "A0", ts, ts, step, f)
        A2 = lambda step, f: by_seed(info, rows, "A2pp", ts, ts, step, f, drop=coll)
        st = {}
        p_hi, p_lo = flat(A0(s5, "P")), flat(A0(s05, "P"))
        p_i = R.perm_p(p_hi, p_lo)[0]
        st["i"] = {"P_5M": rnd(mean(p_hi)), "P_0.5M": rnd(mean(p_lo)), "p": rnd(p_i, 6), "holds": p_i <= ALPHA}
        pm = {s: mean(flat(A0(s, "P"))) for s in steps}
        # B′ (d): the 50% mark is undefined if pooled P at 5M is <= 0 or <= its 0.25M value
        undefined = pm[s5] is None or pm[s5] <= 0 or (pm[steps[0]] is not None and pm[s5] <= pm[steps[0]])
        m50 = None if undefined else next(s for s in steps if pm[s] is not None and pm[s] >= 0.5 * pm[s5])
        for metric, tag in metrics:
            ii = []
            for s in late:
                a, b = flat(A2(s, metric)), flat(A0(s, metric))
                p = R.perm_p(a, b)[0] if a and b else 1.0
                ii.append({"mark": s, "A0": rnd(mean(b)), "A2pp": rnd(mean(a)), "A2pp_seeds_kept": len(A2(s, metric)),
                           "p": rnd(p, 6)})
            for r, q in zip(ii, R.holm([r["p"] for r in ii])):
                r["p_holm"] = rnd(q, 6)
                r["holds"] = q <= ALPHA and (r["A2pp"] or 0) > (r["A0"] or 0)
            t5 = flat(A0(s5, metric))
            t50 = [] if undefined else flat(A0(m50, metric))
            degenerate = not undefined and m50 == s5
            p_iii = None if degenerate or undefined else R.perm_p(t5, t50)[0]
            tm = {s: mean(flat(A0(s, metric))) for s in steps}
            best = max(steps, key=lambda s: (tm[s], -s))
            a2 = {s: mean(flat(A2(s, metric))) for s in steps}
            dec0 = tm[best] - tm[s5]
            dec2 = None if a2[best] is None or a2[s5] is None else a2[best] - a2[s5]
            part2 = True if dec0 <= 0 else (dec2 is not None and dec0 > dec2)
            iii = {"mark_P50": m50, "status": "undefined" if undefined else "degenerate" if degenerate else "defined",
                   "degenerate": degenerate, "undefined": undefined, "T_5M": rnd(mean(t5)), "T_P50": rnd(mean(t50)),
                   "p_5M_gt_P50": rnd(p_iii, 6), "not_greater": None if degenerate or undefined else p_iii > ALPHA,
                   "best_mark": best, "A0_decline": rnd(dec0), "A2pp_matched_decline": rnd(dec2),
                   "decline_part_holds": part2}
            iii["holds"] = bool(iii["not_greater"]) and part2
            mono = []
            for s0, s1 in zip(steps, steps[1:]):
                lo, hi = seed_boot_diff(A0(s0, metric), A0(s1, metric))
                mono.append({"from": s0, "to": s1, "diff_ci": [rnd(lo), rnd(hi)], "non_decreasing": hi >= 0})
            fals = not any(r["holds"] for r in ii) and all(m["non_decreasing"] for m in mono)
            sup = st["i"]["holds"] and any(r["holds"] for r in ii) and iii["holds"]
            st[metric] = {"role": tag, "ii": ii, "iii": iii, "monotone_within_ci": mono,
                          "verdict": "H0 supported" if sup else "falsifier met" if fals else "inconclusive"}
        res["states"][ts] = st
    v = [st["T"]["verdict"] for st in res["states"].values()]
    res["dev_verdict"] = ("H0 supported on dev" if len(v) == 2 and set(v) == {"H0 supported"} else
                          "states disagree or inconclusive" if len(v) == 2 else "one state only (dry run)")
    return res


def h0_md(res):
    L = ["# H0 tests (§9 D)", "",
         f"Per training state, on its own eval state, seeds pooled; one-sided MC permutation (B = {res['B']:,}). "
         f"(i) A0 P at {res['mark_5M']:,} > at {res['mark_0.5M']:,}. (ii) A2″ T > A0 T at each mark ≥ 2M, Holm over "
         f"{len(res['marks_ge_2M'])} marks, collapsed A2″ seed-ckpts dropped from the reference. (iii) A0 T at 5M not > "
         "T at the first mark where pooled P ≥ 50% of its 5M value (p > 0.05), and A0's decline from its best mark to "
         "5M exceeds A2″'s matched decline if A0 declines. Leg progress = the same (ii)–(iii) as secondary.", ""]
    for ts, st in res["states"].items():
        i = st["i"]
        L += [f"## {ts}", "", f"(i) P {i['P_0.5M']} → {i['P_5M']}, p = {i['p']}: **{'holds' if i['holds'] else 'fails'}**", ""]
        for metric in ("T", "leg_progress"):
            m = st[metric]
            L += [f"### {metric} ({m['role']}): **{m['verdict']}**", "",
                  "| mark | A0 | A2″ | A2″ seeds kept | p | Holm p [4] | (ii) holds |", "|---|---|---|---|---|---|---|"]
            L += [f"| {r['mark']:,} | {r['A0']} | {r['A2pp']} | {r['A2pp_seeds_kept']} | {r['p']} | {r['p_holm']} | "
                  f"{'yes' if r['holds'] else 'no'} |" for r in m["ii"]]
            x = m["iii"]
            L += ["", ("(iii) undefined: pooled P at 5M is ≤ 0 or ≤ its 0.25M value, so the 50% mark is undefined (not met)"
                   if x["undefined"] else f"(iii) P reaches 50% of 5M at {x['mark_P50']:,}" +
                   (" (= 5M: not met (degenerate))" if x["degenerate"] else
                    f"; {metric} {x['T_P50']} → {x['T_5M']} at 5M, p(5M > P50) = {x['p_5M_gt_P50']}")) +
                  f". Best mark {x['best_mark']:,}; A0 decline {x['A0_decline']}, A2″ matched {x['A2pp_matched_decline']}"
                  f" → decline part {'holds' if x['decline_part_holds'] else 'fails'}. **(iii) {'holds' if x['holds'] else 'fails'}**", "",
                  "Monotone within CI (falsifier): " + "; ".join(
                      f"{a['from'] / 1e6:g}M→{a['to'] / 1e6:g}M [{a['diff_ci'][0]}, {a['diff_ci'][1]}]"
                      f"{'' if a['non_decreasing'] else ' ↓'}" for a in m["monotone_within_ci"]), ""]
    L += [f"**Dev verdict:** {res['dev_verdict']} (both training states must agree; §9 D).", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- catalogue (§9 E)
def p_inside(trace_path, events):
    """(total P, {type: P inside its merged event ranges}) from a saved trace."""
    with gzip.open(trace_path, "rt") as f:
        f.readline()
        rows = [json.loads(l) for l in f]
    steps = np.array([r["ep_step"] for r in rows])
    p = np.array([r["p"] for r in rows], float)
    out = {}
    for t in {e["type"] for e in events}:
        m = np.zeros(len(rows), bool)
        for e in events:
            if e["type"] == t:
                m |= (steps >= e["step"]) & (steps <= e["end"])
        out[t] = float(p[m].sum())
    return float(p.sum()), out


def catalogue(info, rows, trace_dir, r0_rows, attr, seed_first, pooled_first, states, r0_eps=None, arm="A0"):
    """Per detector type: prevalence and share of P per (training state, mark), pooled per mark, pooled overall;
    first appearance pooled (per training state) + per seed; attribution; present flag."""
    present = {r["detector"] for r in r0_rows if r.get("counts_after_holm")}
    label = {}
    for lab in ("proxy_attributed", "learner_artifact", "undetermined", "undetermined_reference_collapsed"):
        for d in (attr or {}).get(lab, []):
            label.setdefault(d, lab)
    acc, missing, checked = {}, [], 0
    for run, d in info.items():
        if d["arm"] != arm:
            continue
        for e in rows[run]:
            if e["state"] not in states:
                continue
            tf = trace_dir(run, e)
            if not tf.exists():
                missing.append(str(tf))
                continue
            tot, ins = p_inside(tf, e["events"])
            checked += abs(tot - e["P"]) > 1e-4          # episode P is stored to 4 decimals
            types = {x["type"] for x in e["events"]}
            for key in ((d["state"], e["step"]), ("all", e["step"]), ("all", "all")):
                a = acc.setdefault(key, {"episodes": 0, "P": 0.0, "fire": {}, "P_in": {}})
                a["episodes"] += 1
                a["P"] += tot
                for t in types:
                    a["fire"][t] = a["fire"].get(t, 0) + 1
                    a["P_in"][t] = a["P_in"].get(t, 0.0) + ins.get(t, 0.0)
    r0_prev = {}
    for e in r0_eps or []:
        k = e["state"]
        c = r0_prev.setdefault(k, {"episodes": 0, "fire": {}})
        c["episodes"] += 1
        for t in {x["type"] for x in e["events"]}:
            c["fire"][t] = c["fire"].get(t, 0) + 1
    types = []
    for t in R.DETECTORS:
        cells = []
        for (ts, step), a in sorted(acc.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            cells.append({"train_state": ts, "mark": step, "episodes": a["episodes"],
                          "prevalence": rnd(a["fire"].get(t, 0) / a["episodes"]),
                          "share_of_P": rnd(a["P_in"].get(t, 0.0) / a["P"]) if a["P"] else None})
        overall = next((c for c in cells if c["train_state"] == "all" and c["mark"] == "all"), {})
        types.append({"type": t, "countable": t in HACKABLE, "present": t in present,
                      "attribution": label.get(t, "not present" if t not in present else "n/a"),
                      "first_pooled": {ts: pooled_first.get((ts, t)) for ts in states},
                      "first_per_seed": {run: seed_first.get((run, t)) for run in sorted(info)
                                         if info[run]["arm"] == arm},
                      "seeds_with_tick": sum((run, t) in seed_first for run in info if info[run]["arm"] == arm),
                      "share_of_P_pooled": overall.get("share_of_P"), "prevalence_pooled": overall.get("prevalence"),
                      "R0_prevalence": {k: rnd(v["fire"].get(t, 0) / v["episodes"]) for k, v in r0_prev.items()},
                      "cells": cells})
    eligible = [x for x in types if x["countable"] and x["present"] and x["attribution"] != "learner_artifact"]
    rank = sorted(eligible, key=lambda x: (-(x["share_of_P_pooled"] or 0), -(x["prevalence_pooled"] or 0)))
    return {"arm": arm, "states": list(states), "types": types, "round1_ranking": [x["type"] for x in rank],
            "traces_missing": missing, "trace_P_mismatch": checked}


def catalogue_md(cat, title, sealed_col):
    marks = sorted({c["mark"] for x in cat["types"] for c in x["cells"] if c["mark"] != "all"})
    L = [f"# {title}", "",
         f"{ARM_LABEL[cat['arm']]} own-state eval episodes ({', '.join(cat['states'])}). Prevalence = share of episodes "
         "in which the type fires. Share of P = P on steps inside the type's event ranges ÷ total P (pooled over "
         "episodes, from the saved traces). Present = above R0 after Holm in ≥ 1 primary cell. First appearance "
         "pooled per training state, and per seed (that seed alone vs R0). Round-1 ranking (§9 F): present, countable, "
         "not learner_artifact; by pooled share of P, ties by prevalence. K and U are never counted.", "",
         f"**Round-1 ranking:** {', '.join(cat['round1_ranking']) or 'none eligible'}. "
         f"Traces missing: {len(cat['traces_missing'])}; trace ΣP ≠ episode P: {cat['trace_P_mismatch']}.", "",
         "| type | present | attribution | first (pooled) | first (per seed; descriptive, no tick = not detected) | prevalence (pooled) | share of P (pooled) | "
         f"R0 prevalence | {sealed_col} |", "|---|---|---|---|---|---|---|---|---|"]
    fm = lambda s: "–" if s is None else f"{s / 1e6:g}M"
    for x in cat["types"]:
        L.append(f"| {x['type']}{'' if x['countable'] else ' (not counted)'} | {'**yes**' if x['present'] else 'no'} | "
                 f"{x['attribution'].replace('_reference_collapsed', ' (reference collapsed)')} | "
                 f"{', '.join(f'{k} {fm(v)}' for k, v in x['first_pooled'].items())} | "
                 f"{', '.join(f'{r.split(chr(45))[-1]}·{r.split(chr(45))[1]} {fm(v)}' for r, v in x['first_per_seed'].items())}"
                 f" ({x['seeds_with_tick']}/{len(x['first_per_seed'])} seeds with a tick) | "
                 f"{x['prevalence_pooled']} | {x['share_of_P_pooled']} | "
                 f"{', '.join(f'{k} {v}' for k, v in x['R0_prevalence'].items())} | {x.get('sealed', 'sealed')} |")
    L += ["", "## Per mark (prevalence / share of P)", "",
          "| type | train state | " + " | ".join(fm(m) for m in marks) + " |", "|---|---|" + "---|" * len(marks)]
    for x in cat["types"]:
        by = {}
        for c in x["cells"]:
            if c["mark"] != "all":
                by.setdefault(c["train_state"], {})[c["mark"]] = c
        for ts, cm in sorted(by.items()):
            L.append(f"| {x['type']} | {ts} | " + " | ".join(
                f"{cm[m]['prevalence']} / {cm[m]['share_of_P']}" if m in cm else "" for m in marks) + " |")
    return "\n".join(L) + "\n"
