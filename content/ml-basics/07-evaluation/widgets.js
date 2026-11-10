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

// Step-through: 100 test emails, the never-flag filter, then the four confusion-matrix groups.
(function () {
  const G = { x: 30, y: 44, cell: 26 };
  // Cells 0..9 are the phishing emails. The real filter flags cells 0..6 (TP) and cell 10 (FP).
  const flagged = (i) => i < 7 || i === 10;
  const TP = 7, FN = 3, FP = 1, TN = 89;
  Learn.demo(document.getElementById("demo-confusion"), {
    height: 340,
    label: "A grid of 100 test emails, 10 of them phishing, sorted into true and false positives and negatives",
    steps: [
      { s: { never: 0, flag: 0, sort: 0, prec: 0, rec: 0 }, cap: "The test set: 100 emails. The 10 orange squares are phishing. The 90 blue squares are legitimate." },
      { s: { never: 1 }, cap: "A filter that never flags anything is right about all 90 legitimate emails and wrong about all 10 phishing ones. Accuracy: 90 ÷ 100 = 0.90, with zero phishing caught." },
      { s: { never: 0, flag: 1 }, cap: "The real filter flags 8 emails (dark outline). 7 are phishing and 1 is legitimate. It misses 3 phishing emails. It gets 96 right, so accuracy is 0.96." },
      { s: { sort: 1 }, cap: "Sort the 100 answers into four groups. TP 7: phishing, flagged. FN 3: phishing, missed. FP 1: legitimate, flagged. TN 89: legitimate, left alone." },
      { s: { prec: 1 }, cap: "Precision looks only at the flagged column. Of the 8 flagged emails, 7 are phishing: 7 ÷ 8 = 0.875." },
      { s: { prec: 0, rec: 1 }, cap: "Recall looks only at the phishing row. Of the 10 phishing emails, the filter caught 7: 7 ÷ 10 = 0.70." },
    ],
    draw(ctx, s, c, d) {
      d.text("100 test emails", G.x, 24, { c: c.muted, s: 13, w: 700 });
      for (let i = 0; i < 100; i++) {
        const x = G.x + (i % 10) * G.cell, y = G.y + Math.floor(i / 10) * G.cell;
        const color = i < 10 ? c.orange : c.blue;
        ctx.globalAlpha = 0.3;
        d.rect(x + 3, y + 3, G.cell - 6, G.cell - 6, color);
        ctx.globalAlpha = 1;
        d.rect(x + 3, y + 3, G.cell - 6, G.cell - 6, null, color, 2);
        if (flagged(i) && s.flag > 0.01) {
          ctx.globalAlpha = s.flag;
          d.rect(x + 3, y + 3, G.cell - 6, G.cell - 6, color, c.ink, 3);
        }
        const wrong = (i < 10 ? s.never : 0) + (flagged(i) !== (i < 10) ? s.flag : 0);
        if (wrong > 0.01) {
          ctx.globalAlpha = d.clamp(wrong, 0, 1);
          d.line(x + 5, y + 5, x + G.cell - 5, y + G.cell - 5, c.red, 3);
          d.line(x + G.cell - 5, y + 5, x + 5, y + G.cell - 5, c.red, 3);
        }
        ctx.globalAlpha = 1;
      }
      ctx.globalAlpha = d.clamp(s.never + s.flag, 0, 1);
      d.text("red × = wrong answer", G.x, 322, { c: c.red, s: 12, a: "left" });
      ctx.globalAlpha = 1;

      d.panel(c, 320, 30, 300, 280);
      const count = 1 - s.sort;
      if (count > 0.01) {
        const right = 100 - 10 * s.never - 4 * s.flag, wrong = 10 * s.never + 4 * s.flag;
        const some = s.never + s.flag;
        ctx.globalAlpha = count;
        d.text(s.never > s.flag ? "never-flag filter" : s.flag > 0.01 ? "real filter, p ≥ 0.5" : "no filter yet", 336, 56, { c: c.muted, s: 13, w: 700 });
        ctx.globalAlpha = count * d.clamp(some, 0, 1);
        d.text(`right   ${Math.round(right)}`, 336, 100, { c: c.ink, s: 16 });
        d.text(`wrong   ${Math.round(wrong)}`, 336, 130, { c: c.red, s: 16 });
        d.text("accuracy", 336, 180, { c: c.muted, s: 13 });
        d.text(`${Math.round(right)} ÷ 100 = ${d.fmt(Math.round(right) / 100, 2)}`, 336, 208, { c: c.ink, s: 20, w: 700 });
        d.text(`phishing caught: ${Math.round(7 * s.flag)} of 10`, 336, 262, { c: c.orange, s: 14, w: 700 });
        ctx.globalAlpha = 1;
      }
      if (s.sort > 0.01) {
        const cx = [472, 566], ry = [124, 184];
        if (s.prec > 0.01) { ctx.globalAlpha = s.sort * s.prec * 0.3; d.rect(430, 96, 84, 116, c.blue); }
        if (s.rec > 0.01) { ctx.globalAlpha = s.sort * s.rec * 0.3; d.rect(430, 96, 178, 56, c.orange); }
        ctx.globalAlpha = s.sort;
        d.text("flagged", cx[0], 62, { c: c.muted, s: 13, a: "center", w: 700 });
        d.text("not", cx[1], 54, { c: c.muted, s: 13, a: "center", w: 700 });
        d.text("flagged", cx[1], 70, { c: c.muted, s: 13, a: "center", w: 700 });
        d.text("phishing", 336, ry[0], { c: c.orange, s: 13, w: 700 });
        d.text("legit", 336, ry[1], { c: c.blue, s: 13, w: 700 });
        d.line(430, 96, 608, 96, c.muted, 1.5); d.line(430, 154, 608, 154, c.muted, 1.5);
        d.line(430, 212, 608, 212, c.muted, 1.5); d.line(430, 96, 430, 212, c.muted, 1.5);
        d.line(516, 96, 516, 212, c.muted, 1.5); d.line(608, 96, 608, 212, c.muted, 1.5);
        const cell = (name, n, x, y, color) => {
          d.text(name, x, y - 12, { c: c.muted, s: 12, a: "center" });
          d.text(String(n), x, y + 10, { c: color, s: 20, a: "center", w: 700 });
        };
        cell("TP", TP, cx[0], ry[0], c.ink); cell("FN", FN, cx[1], ry[0], c.red);
        cell("FP", FP, cx[0], ry[1], c.red); cell("TN", TN, cx[1], ry[1], c.ink);
        const line = (alpha, a, b) => {
          if (alpha < 0.05) return;
          ctx.globalAlpha = s.sort * alpha;
          d.text(a, 336, 246, { c: c.muted, s: 13 });
          d.text(b, 336, 276, { c: c.ink, s: 17, w: 700 });
        };
        line(s.prec, "precision = TP ÷ (TP + FP)", `${TP} ÷ (${TP} + ${FP}) = ${d.fmt(TP / (TP + FP), 3)}`);
        line(s.rec, "recall = TP ÷ (TP + FN)", `${TP} ÷ (${TP} + ${FN}) = ${d.fmt(TP / (TP + FN), 2)}`);
        line(1 - Math.max(s.prec, s.rec), "accuracy = (TP + TN) ÷ 100", `(${TP} + ${TN}) ÷ 100 = ${d.fmt((TP + TN) / 100, 2)}`);
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: slide the threshold across the scored emails and watch precision and recall trade.
(function () {
  // Scores in hundredths, from the worked example's 100 test emails. The other 79 legitimate
  // emails score under 0.10 and are summarized, not drawn.
  const phish = [22, 36, 42, 51, 54, 56, 73, 86, 90, 96];
  const legit = [13, 13, 17, 17, 19, 19, 20, 27, 29, 31, 63];
  const X = (v) => 60 + v * 520, AX = 150;
  Learn.demo(document.getElementById("demo-threshold"), {
    height: 320,
    label: "Test emails placed along a line by score, with a threshold that moves and the precision and recall at each position",
    steps: [
      { s: { t: 0.5, show: 0 }, cap: "Each dot is one test email, placed by its score p. The 10 phishing emails sit above the line. Legitimate ones sit below. 79 more legitimate emails score under 0.10 and are not drawn." },
      { s: { show: 1 }, cap: "Set the threshold t = 0.5. Filled dots are flagged. The filter catches 7 of 10 phishing emails (recall 0.70). 7 of its 8 flags are right (precision 0.88)." },
      { s: { t: 0.2 }, cap: "Lower t to 0.2. Now all 10 phishing emails are flagged, so recall is 1.00. But 5 legitimate emails are flagged too. Precision drops to 10 ÷ 15 = 0.67." },
      { s: { t: 0.8 }, cap: "Raise t to 0.8. Only 3 emails are flagged and all 3 are phishing, so precision is 1.00. The filter now misses 7 phishing emails. Recall is 0.30." },
      { s: { t: 0.34 }, cap: "At t = 0.34 the filter flags 9 phishing emails and 1 legitimate one. Precision and recall are both 0.90, and so is F1, its highest value on this test set." },
    ],
    draw(ctx, s, c, d) {
      const tx = X(s.t), cut = s.t * 100 - 1e-6;
      if (s.show > 0.01) {
        ctx.globalAlpha = s.show;
        d.rect(tx, 50, X(1) - tx + 20, 190, c.soft);
        ctx.globalAlpha = 1;
      }
      d.line(X(0), AX, X(1) + 20, AX, c.muted, 1.5);
      for (let k = 0; k <= 10; k++) {
        d.line(X(k / 10), AX - 4, X(k / 10), AX + 4, c.muted, 1.5);
        if (k % 2 === 0) d.text(d.fmt(k / 10, 1), X(k / 10), 252, { c: c.muted, s: 12, a: "center" });
      }
      d.text("score p", 636, AX - 12, { c: c.muted, s: 12, a: "right" });
      d.text("phishing", 8, AX - 30, { c: c.orange, s: 12, w: 700 });
      d.text("legit", 8, AX + 30, { c: c.blue, s: 12, w: 700 });
      const mark = (x, y, color, on) => {
        d.dot(x, y, 7, color);
        if (on < 0.99) { ctx.globalAlpha = 1 - on; d.dot(x, y, 4.5, c.bg); ctx.globalAlpha = 1; }
      };
      let tp = 0, fp = 0;
      for (const v of phish) { const on = v >= cut; if (on) tp++; mark(X(v / 100), AX - 30, c.orange, on ? s.show : 0); }
      let prev = -1, stack = 0;
      for (const v of legit) {
        stack = v === prev ? stack + 1 : 0; prev = v;
        const on = v >= cut; if (on) fp++;
        mark(X(v / 100), AX + 30 + stack * 18, c.blue, on ? s.show : 0);
      }
      d.text("+79", X(0.045), AX + 30, { c: c.blue, s: 13, a: "center", w: 700 });
      if (s.show > 0.01) {
        ctx.globalAlpha = s.show;
        d.line(tx, 44, tx, 240, c.ink, 2, [6, 5]);
        d.text(`t = ${s.t.toFixed(2)}`, d.clamp(tx, 70, 570), 30, { c: c.ink, a: "center", w: 700 });
        d.text("flagged →", Math.min(tx + 8, 540), 62, { c: c.muted, s: 12 });
        const fn = 10 - tp, P = tp / (tp + fp), R = tp / 10, F = 2 * P * R / (P + R);
        d.panel(c, 20, 270, 600, 40);
        d.text(`TP ${tp}  FP ${fp}  FN ${fn}`, 34, 290, { c: c.ink, s: 13 });
        d.text(`precision ${P.toFixed(2)}`, 208, 290, { c: c.blue, s: 13, w: 700 });
        d.text(`recall ${R.toFixed(2)}`, 354, 290, { c: c.green, s: 13, w: 700 });
        d.text(`F1 ${F.toFixed(2)}`, 480, 290, { c: c.ink, s: 13, w: 700 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: one bin of scores, calibrated and then stretched (overconfident).
(function () {
  const X = (v) => 70 + v * 440, rowA = 96, rowB = 216, H = 34;
  Learn.demo(document.getElementById("demo-calibration"), {
    height: 310,
    label: "Two bars comparing what a filter claims for emails scored 0.8 to 0.9 with the share that really were phishing",
    steps: [
      { s: { fillA: 0, b: 0, fillB: 0, gap: 0 }, cap: "The filter scored 122 emails between 0.8 and 0.9. Their average score is 0.85. So the filter claims about 85% of these emails are phishing." },
      { s: { fillA: 0.85 }, cap: "Check the labels. 85% of them really are phishing. The claim and the count match, so the filter is calibrated in this bin." },
      { s: { b: 1 }, cap: "Now double every raw score z before the sigmoid. The stretched filter puts 88 emails in the same bin, again with an average score of 0.85." },
      { s: { fillB: 0.64 }, cap: "Check the labels again. Only 64% of these 88 emails are phishing." },
      { s: { gap: 1 }, cap: "The gap is 0.85 − 0.64 = 0.21. The stretched filter is overconfident: it claims more than the labels support. ECE averages gaps like this one over all the bins." },
    ],
    draw(ctx, s, c, d) {
      const row = (y, title, fill, color) => {
        d.text(title, X(0), y - 44, { c: c.ink, s: 13, w: 700 });
        d.rect(X(0), y - H / 2, X(1) - X(0), H, c.soft, c.muted, 2);
        if (fill > 0.005) {
          d.rect(X(0), y - H / 2, X(fill) - X(0), H, color);
          d.text(`${Math.round(fill * 100)}% phishing`, X(0) + 10, y, { c: c.bg, s: 13, w: 700 });
        }
        d.line(X(0.85), y - H / 2 - 8, X(0.85), y + H / 2 + 8, c.ink, 3);
        d.text("claims 0.85", X(0.85) + 8, y - H / 2 - 12, { c: c.ink, s: 12, w: 700 });
        d.text("0", X(0), y + H / 2 + 14, { c: c.muted, s: 12, a: "center" });
        d.text("1", X(1), y + H / 2 + 14, { c: c.muted, s: 12, a: "center" });
      };
      row(rowA, "calibrated filter: 122 emails scored 0.8 to 0.9", s.fillA, c.blue);
      if (s.fillA > 0.84) d.text("match", X(1) + 12, rowA, { c: c.blue, s: 13, w: 700 });
      if (s.b > 0.01) {
        ctx.globalAlpha = s.b;
        row(rowB, "stretched filter (2z): 88 emails scored 0.8 to 0.9", s.fillB, c.orange);
        if (s.gap > 0.01) {
          ctx.globalAlpha = s.b * s.gap * 0.45;
          d.rect(X(0.64), rowB - H / 2, X(0.85) - X(0.64), H, c.red);
          ctx.globalAlpha = s.b * s.gap;
          d.text("gap", (X(0.64) + X(0.85)) / 2, rowB - 7, { c: c.ink, s: 12, a: "center", w: 700 });
          d.text("0.21", (X(0.64) + X(0.85)) / 2, rowB + 8, { c: c.ink, s: 12, a: "center", w: 700 });
          d.text("overconfident", X(0.85), rowB + H / 2 + 32, { c: c.red, s: 13, a: "center", w: 700 });
        }
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: the standard error of an accuracy, and what a 4 times bigger test set buys.
(function () {
  const X = (v) => 70 + (v - 0.9) * 5000, A = 0.96, B = 0.94;
  const se = (a, n) => Math.sqrt(a * (1 - a) / n);
  Learn.demo(document.getElementById("demo-stderr"), {
    height: 300,
    label: "Two filters' accuracies on an axis with error bars that shrink as the test set grows",
    steps: [
      { s: { bar: 0, b: 0, k: 1 }, cap: "Filter A scores 0.96 on 100 test emails. A different 100 emails would give a slightly different number." },
      { s: { bar: 1 }, cap: "The standard error says how different. For 100 emails it is about 0.02. The bar reaches one standard error to each side of 0.96." },
      { s: { b: 1 }, cap: "Filter B scores 0.94 on the same 100 emails. The gap between A and B is 0.02, about one standard error. The bars overlap, so this test can't tell which filter is better." },
      { s: { k: 2 }, cap: "Suppose the scores held on 400 emails. 4 times the emails halves each standard error, to about 0.01. The bars still touch." },
      { s: { k: 4 }, cap: "On 1,600 emails each standard error halves again, to about 0.005. The bars no longer overlap. A gap of 0.02 on a test set this size is worth believing." },
    ],
    draw(ctx, s, c, d) {
      const n = 100 * s.k * s.k, AX = 236;
      d.line(X(0.9), AX, X(1), AX, c.muted, 1.5);
      for (let k = 0; k <= 10; k++) {
        const v = 0.9 + k / 100;
        d.line(X(v), AX - 4, X(v), AX + 4, c.muted, 1.5);
        if (k % 2 === 0) d.text(v.toFixed(2), X(v), AX + 18, { c: c.muted, s: 12, a: "center" });
      }
      d.text("accuracy", X(0.95), AX + 42, { c: c.muted, s: 12, a: "center" });
      d.panel(c, 20, 16, 226, 64);
      d.text(`test set: ${Math.round(n).toLocaleString("en-US")} emails`, 34, 36, { c: c.ink, s: 13, w: 700 });
      d.text(`standard error ≈ ${se(0.95, n).toFixed(3)}`, 34, 60, { c: c.muted, s: 13 });
      const row = (y, name, a, color, alpha) => {
        if (alpha < 0.01) return;
        const e = se(a, n);
        ctx.globalAlpha = alpha;
        d.line(X(a), y + 12, X(a), AX, color, 1, [3, 4]);
        if (s.bar > 0.01) {
          ctx.globalAlpha = alpha * s.bar * 0.3;
          d.rect(X(a - e), y - 10, X(a + e) - X(a - e), 20, color);
          ctx.globalAlpha = alpha * s.bar;
          d.line(X(a - e), y - 10, X(a - e), y + 10, color, 2);
          d.line(X(a + e), y - 10, X(a + e), y + 10, color, 2);
          d.line(X(a - e), y, X(a + e), y, color, 2);
        }
        ctx.globalAlpha = alpha;
        d.dot(X(a), y, 7, color);
        d.text(`${name}: ${a.toFixed(2)}`, X(a), y - 24, { c: color, s: 13, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      };
      row(130, "filter A", A, c.blue, 1);
      row(190, "filter B", B, c.orange, s.b);
    },
  });
})();
