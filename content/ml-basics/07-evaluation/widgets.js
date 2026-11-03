// Unit 07: a threshold over 100 scored emails (confusion matrix, precision, recall, F1),
// and a reliability diagram for a model whose logits get stretched.
(function () {
  const { svg, stage, controls, slider, buttons, button, readout, fmt } = Learn;
  const sigmoid = (t) => 1 / (1 + Math.exp(-t));

  (function () {
    const figure = document.getElementById("threshold");
    if (!figure) return;

    // The worked example's test set: the first 100 emails from default_rng(145), scores in
    // hundredths (round(p, 2) * 100) and labels (1 = phishing).
    const S = [0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 6, 0, 0, 1, 1, 1, 1, 0, 0, 2, 3, 17, 86, 3, 0, 0, 3, 56, 8, 0,
      0, 1, 19, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 1, 1, 3, 1, 5, 0, 0, 2, 31, 8, 27, 17, 54, 90,
      0, 0, 0, 0, 0, 0, 36, 13, 73, 1, 0, 0, 2, 0, 20, 0, 0, 6, 0, 96, 4, 0, 3, 19, 22, 2, 0, 1, 13, 1,
      29, 1, 42, 1, 0, 51, 1, 0, 0, 0];
    const Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
      0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 1, 0, 0, 0, 0];
    const emails = S.map((s, i) => ({ s, y: Y[i] })).sort((a, b) => a.s - b.s || a.y - b.y);

    // Thresholds are whole hundredths too, so "p ≥ t" compares integers and matches NumPy exactly.
    function counts(t) {
      let tp = 0, fp = 0, fn = 0, tn = 0;
      for (const e of emails) {
        const flagged = e.s >= t;
        if (flagged && e.y) tp++; else if (flagged) fp++; else if (e.y) fn++; else tn++;
      }
      const precision = tp + fp ? tp / (tp + fp) : null;
      const recall = tp / (tp + fn);
      const f1 = tp ? 2 * precision * recall / (precision + recall) : 0;
      return { tp, fp, fn, tn, precision, recall, f1 };
    }

    const width = 720, rows = 15, height = rows * Learn.ROW;
    const top = 14, bottom = height - 40;
    const Yt = (v) => bottom - v * (bottom - top);          // score or threshold, 0..1, upward
    const left0 = 64, left1 = 456;                          // sorted emails
    const Xr = (i) => left0 + (i + 0.5) * (left1 - left0) / emails.length;
    const right0 = 512, right1 = 700;                       // metric value, 0..1, across
    const Xm = (v) => right0 + v * (right1 - right0);

    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "100 emails sorted by the model's phishing score, a threshold line, and precision, recall and F1 as the threshold moves");

    // Axes.
    svg("line", { x1: left0, y1: bottom, x2: left1, y2: bottom, class: "muted thin" }, root);
    svg("line", { x1: left0, y1: top, x2: left0, y2: bottom, class: "muted thin" }, root);
    svg("line", { x1: right0, y1: bottom, x2: right1, y2: bottom, class: "muted thin" }, root);
    svg("line", { x1: right0, y1: top, x2: right0, y2: bottom, class: "muted thin" }, root);
    svg("line", { x1: right1, y1: top, x2: right1, y2: bottom, class: "muted faint thin" }, root);
    for (const v of [0, 0.5, 1]) {
      svg("text", { x: left0 - 8, y: Yt(v) + 4, "text-anchor": "end", class: "muted", text: String(v) }, root);
    }
    svg("text", { x: (left0 + left1) / 2, y: height - 8, "text-anchor": "middle", class: "muted", text: "emails, by score" }, root);
    svg("text", { x: right0, y: height - 8, "text-anchor": "middle", class: "muted", text: "0" }, root);
    svg("text", { x: right1, y: height - 8, "text-anchor": "middle", class: "muted", text: "1" }, root);

    // Precision, recall and F1 at every threshold, drawn sideways so they share the score axis.
    const curve = (key) => {
      const pts = [];
      for (let t = 0; t <= 100; t++) {
        const v = counts(t)[key];
        if (v !== null) pts.push(`${Xm(v)},${Yt(t / 100)}`);
      }
      return pts.join(" ");
    };
    svg("polyline", { points: curve("precision"), class: "blue" }, root);
    svg("polyline", { points: curve("recall"), class: "green" }, root);
    svg("polyline", { points: curve("f1"), class: "ink dashed" }, root);

    const dots = emails.map((e, i) => svg("circle", {
      cx: Xr(i), cy: Yt(e.s / 100), r: 3.4, "stroke-width": 1.5,
      class: e.y ? "orange" : "muted", stroke: "currentColor",
    }, root));

    const line = svg("line", { x1: left0, x2: right1, class: "ink dashed thin" }, root);
    const label = svg("text", { x: left1 + 8, class: "ink", text: "t" }, root);
    const marks = ["precision", "recall", "f1"].map((key) => ({
      key, node: svg("circle", { r: 4.5, fill: "currentColor", class: key === "precision" ? "blue" : key === "recall" ? "green" : "ink" }, root),
    }));

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);

    const pad = (n) => String(n).padStart(2);
    function draw(t) {
      const c = counts(t);
      const y = Yt(t / 100);
      line.setAttribute("y1", y); line.setAttribute("y2", y);
      label.setAttribute("y", y - 5);
      emails.forEach((e, i) => dots[i].setAttribute("fill", e.s >= t ? "currentColor" : "none"));
      for (const m of marks) {
        const v = c[m.key];
        m.node.style.display = v === null ? "none" : "";
        if (v !== null) { m.node.setAttribute("cx", Xm(v)); m.node.setAttribute("cy", y); }
      }
      const tt = fmt(t / 100);
      out.innerHTML =
        `flag an email when p ≥ ${tt}\n` +
        `                 flagged   not flagged\n` +
        `<span class="orange">phishing (10)</span>    TP ${pad(c.tp)}     FN ${pad(c.fn)}\n` +
        `legit    (90)    FP ${pad(c.fp)}     TN ${pad(c.tn)}\n` +
        `accuracy  (${c.tp} + ${c.tn}) / 100 = ${fmt((c.tp + c.tn) / 100)}\n` +
        (c.precision === null
          ? `<span class="blue">precision 0 / 0, undefined (nothing flagged)</span>\n`
          : `<span class="blue">precision ${c.tp} / (${c.tp} + ${c.fp}) = ${fmt(c.precision)}</span>\n`) +
        `<span class="green">recall    ${c.tp} / (${c.tp} + ${c.fn}) = ${fmt(c.recall)}</span>\n` +
        `F1        ${c.tp ? `2PR / (P + R) = ${fmt(c.f1)}` : "0 (no phishing caught)"}`;
    }

    const tSlider = slider(ctl, { label: "threshold t", min: 0, max: 1, step: 0.01, value: 0.5, format: (v) => v.toFixed(2) }, (v) => draw(Math.round(v * 100)));
    for (const v of [0.2, 0.5, 0.8]) button(btns, v.toFixed(2), () => tSlider.set(v));
    button(btns, "Flag nothing", () => tSlider.set(1));
  })();

  (function () {
    const figure = document.getElementById("reliability");
    if (!figure) return;

    // 10,000 simulated emails drawn the same way as the worked example (not the same draws):
    // logit z ~ N(-5, 3.5²), phishing with probability σ(z). The model reports σ(k z).
    const N = 10000, BINS = 10;
    let seed = 145, z, y;
    function draw10k() {
      let a = seed >>> 0;
      const uniform = () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
      z = new Float64Array(N); y = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        z[i] = -5 + 3.5 * normal();
        y[i] = uniform() < sigmoid(z[i]) ? 1 : 0;
      }
    }

    const width = 720, rows = 13, height = rows * Learn.ROW;
    const x0 = 64, x1 = 700, top = 12, bottom = height - 40;
    const X = (v) => x0 + v * (x1 - x0);
    const Yv = (v) => bottom - v * (bottom - top);

    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "Reliability diagram: for each band of reported scores, the share of emails that were actually phishing");
    svg("line", { x1: x0, y1: bottom, x2: x1, y2: bottom, class: "muted thin" }, root);
    svg("line", { x1: x0, y1: top, x2: x0, y2: bottom, class: "muted thin" }, root);
    for (let m = 1; m < BINS; m++) svg("line", { x1: X(m / BINS), y1: top, x2: X(m / BINS), y2: bottom, class: "muted faint thin" }, root);
    svg("line", { x1: X(0), y1: Yv(0), x2: X(1), y2: Yv(1), class: "muted dashed thin" }, root);
    for (const v of [0, 0.5, 1]) {
      svg("text", { x: x0 - 8, y: Yv(v) + 4, "text-anchor": "end", class: "muted", text: String(v) }, root);
      svg("text", { x: X(v), y: height - 8, "text-anchor": v === 0 ? "start" : v === 1 ? "end" : "middle", class: "muted", text: String(v) }, root);
    }
    svg("text", { x: X(0.25), y: height - 8, "text-anchor": "middle", class: "muted", text: "score" }, root);
    svg("text", { x: x0 + 10, y: top + 12, class: "blue", text: "share phishing" }, root);

    const gaps = svg("g", { class: "orange" }, root);
    const gapLines = Array.from({ length: BINS }, () => svg("line", { class: "thick" }, gaps));
    const path = svg("polyline", { class: "blue" }, root);
    const dotGroup = svg("g", { class: "blue" }, root);
    const binDots = Array.from({ length: BINS }, () => svg("circle", { r: 4.5, fill: "currentColor" }, dotGroup));

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);
    let k = 1;

    function render() {
      const n = new Array(BINS).fill(0), sumQ = new Array(BINS).fill(0), pos = new Array(BINS).fill(0);
      for (let i = 0; i < N; i++) {
        const q = sigmoid(k * z[i]);
        const m = Math.min(BINS - 1, Math.floor(q * BINS));
        n[m]++; sumQ[m] += q; pos[m] += y[i];
      }
      const pts = [];
      let ece = 0;
      const lines = [`stretch k = ${fmt(k)}, reported score σ(k z)`, `score     emails  mean score  phishing`];
      for (let m = 0; m < BINS; m++) {
        const show = n[m] > 0;
        binDots[m].style.display = gapLines[m].style.display = show ? "" : "none";
        const band = `${fmt(m / BINS, 1)}–${fmt((m + 1) / BINS, 1)}`;
        if (!show) { lines.push(`${band}       0`); continue; }
        const conf = sumQ[m] / n[m], acc = pos[m] / n[m];
        ece += n[m] / N * Math.abs(acc - conf);
        binDots[m].setAttribute("cx", X(conf)); binDots[m].setAttribute("cy", Yv(acc));
        gapLines[m].setAttribute("x1", X(conf)); gapLines[m].setAttribute("y1", Yv(conf));
        gapLines[m].setAttribute("x2", X(conf)); gapLines[m].setAttribute("y2", Yv(acc));
        pts.push(`${X(conf)},${Yv(acc)}`);
        lines.push(`${band}  ${String(n[m]).padStart(6)}  ${fmt(conf).padStart(10)}  ${fmt(acc).padStart(8)}`);
      }
      path.setAttribute("points", pts.join(" "));
      lines.push(`<span class="orange">ECE (weighted average gap) = ${fmt(ece, 3)}</span>`);
      out.innerHTML = lines.join("\n");
    }

    const kSlider = slider(ctl, { label: "stretch k", min: 0.25, max: 3, step: 0.05, value: 1, format: (v) => v.toFixed(2) }, (v) => {
      k = v;
      if (z) render();
    });
    button(btns, "Underconfident (0.5)", () => kSlider.set(0.5));
    button(btns, "Calibrated (1)", () => kSlider.set(1));
    button(btns, "Overconfident (2)", () => kSlider.set(2));
    button(btns, "New 10,000 emails", () => { seed++; draw10k(); render(); });
    draw10k();
    render();
  })();
})();
