// Unit 05: logistic regression trained by gradient descent on 22 emails.
// Left: the emails, the decision boundary, the p = 0.1 and 0.9 lines and probability shading.
// Right: every email's score z placed on the sigmoid curve, with its miss p − y dashed.
(function () {
  const figure = document.getElementById("logistic");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, arrow, loop, fmt } = Learn;
  const num = (v, digits = 2) => fmt(v, digits).replace("-", "−");
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const LN9 = Math.log(9); // z where p = 0.9 (and −LN9 where p = 0.1)

  // (links, exclamation marks), the same 22 emails as the worked example.
  const ham = [[0, 0], [1, 0], [0, 2], [2, 1], [1, 3], [3, 0], [2, 4], [4, 2], [3, 3], [5, 1], [6, 3]];
  const spam = [[5, 6], [7, 3], [6, 5], [8, 7], [4, 7], [9, 4], [6, 8], [7, 6], [3, 6], [8, 2], [3, 2]];
  const all = ham.map((x) => ({ x, y: 0 })).concat(spam.map((x) => ({ x, y: 1 })));
  // Without these two, a straight line can split the classes perfectly.
  const stray = (e) => (e.x[0] === 6 && e.x[1] === 3) || (e.x[0] === 3 && e.x[1] === 2);
  const datasets = { overlapping: all, separable: all.filter((e) => !stray(e)) };

  const width = 720, rows = 18, height = rows * Learn.ROW;
  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Left: emails by links and exclamation marks, with the decision boundary and probability shading. Right: each email's score on the sigmoid curve.");

  // Left panel: feature space, equal scale on both axes so w looks perpendicular to the line.
  const lo = -0.5, hi = 10, S = 30;
  const DX = (v) => 30 + (v - lo) * S;
  const DY = (v) => height - 22 - (v - lo) * S;
  const cell = 0.5, cells = Math.round((hi - lo) / cell);
  const shade = svg("g", {}, root);
  const tiles = [];
  for (let i = 0; i < cells; i++) {
    for (let j = 0; j < cells; j++) {
      const cx = lo + (i + 0.5) * cell, cy = lo + (j + 0.5) * cell;
      const rect = svg("rect", {
        x: DX(lo + i * cell), y: DY(lo + (j + 1) * cell), width: cell * S + 0.5, height: cell * S + 0.5,
        fill: "currentColor", "fill-opacity": 0,
      }, shade);
      tiles.push({ rect, cx, cy, cls: "" });
    }
  }
  svg("line", { x1: DX(lo), y1: DY(0), x2: DX(hi), y2: DY(0), class: "muted faint thin" }, root);
  svg("line", { x1: DX(0), y1: DY(lo), x2: DX(0), y2: DY(hi), class: "muted faint thin" }, root);
  for (const t of [5, 10]) {
    svg("text", { x: DX(t) - 5, y: DY(lo) + 16, class: "muted", text: String(t) }, root);
    svg("text", { x: DX(lo) - 24, y: DY(t) + 4, class: "muted", text: String(t) }, root);
  }
  svg("text", { x: DX(lo) - 20, y: DY(8) + 4, class: "muted", text: "!" }, root);
  const band = [svg("line", { class: "muted dashed thin" }, root), svg("line", { class: "muted dashed thin" }, root)];
  const boundary = svg("line", { class: "ink", "stroke-width": 2 }, root);
  const wArrow = arrow(root, "ink");
  const wLabel = svg("text", { class: "ink", text: "w" }, root);
  svg("text", { x: DX(6.3), y: DY(lo) + 16, class: "muted", text: "links" }, root);

  // Spam: filled orange dots. Not spam: hollow blue rings.
  function marker(parent, e) {
    return e.y
      ? svg("circle", { r: 5, fill: "currentColor", class: "orange" }, parent)
      : svg("circle", { r: 4.5, fill: "none", stroke: "currentColor", "stroke-width": 2, class: "blue" }, parent);
  }
  const dataG = svg("g", {}, root);

  // Right panel: the sigmoid over z in [−10, 10].
  const px0 = 392, px1 = 708;
  const ZX = (z) => (px0 + px1) / 2 + Math.max(-10, Math.min(10, z)) * (px1 - px0) / 20;
  const SY = (p) => 26 + (1 - p) * (height - 56);
  svg("line", { x1: 368, y1: 0, x2: 368, y2: height, class: "muted faint thin" }, root);
  svg("line", { x1: px0, y1: SY(1), x2: px1, y2: SY(1), class: "muted dashed thin" }, root);
  svg("line", { x1: px0, y1: SY(0.5), x2: px1, y2: SY(0.5), class: "muted faint thin" }, root);
  svg("line", { x1: px0, y1: SY(0), x2: px1, y2: SY(0), class: "muted thin" }, root);
  svg("line", { x1: ZX(0), y1: SY(0), x2: ZX(0), y2: SY(1), class: "muted faint thin" }, root);
  svg("text", { x: px0 - 14, y: SY(1) + 4, class: "muted", text: "1" }, root);
  svg("text", { x: px0 - 14, y: SY(0) + 4, class: "muted", text: "0" }, root);
  svg("text", { x: ZX(0) + 6, y: 16, class: "muted", text: "σ(z)" }, root);
  svg("text", { x: px1 - 10, y: SY(0) - 8, class: "muted", text: "z" }, root);
  svg("text", { x: ZX(-10) - 4, y: SY(0) + 26, class: "muted", text: "−10" }, root);
  svg("text", { x: ZX(10) - 30, y: SY(0) + 26, class: "muted", text: "10" }, root);
  const curve = [];
  for (let z = -10; z <= 10.001; z += 0.25) curve.push(`${ZX(z).toFixed(1)},${SY(sigmoid(z)).toFixed(1)}`);
  svg("polyline", { points: curve.join(" "), class: "ink" }, root);
  const sigG = svg("g", {}, root);

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  const MAX_STEPS = 10000, HOLD = 3;
  let eta = 0.1, which = "overlapping";
  let data, dots, w, b, k, wait, acc;

  function load() {
    data = datasets[which];
    while (dataG.firstChild) dataG.removeChild(dataG.firstChild);
    while (sigG.firstChild) sigG.removeChild(sigG.firstChild);
    dots = data.map((e) => {
      const miss = svg("line", { class: "muted dashed thin" }, sigG);
      const onCurve = marker(sigG, e);
      const inData = marker(dataG, e);
      inData.setAttribute("cx", DX(e.x[0]));
      inData.setAttribute("cy", DY(e.x[1]));
      return { miss, onCurve };
    });
    reset();
  }

  function reset() {
    w = [0, 0];
    b = 0;
    k = 0;
    wait = 0;
    acc = 0;
    draw();
  }

  // One full-batch gradient step: w ← w − η · mean((p − y) x), b ← b − η · mean(p − y).
  function step() {
    let g0 = 0, g1 = 0, gb = 0;
    for (const e of data) {
      const r = sigmoid(w[0] * e.x[0] + w[1] * e.x[1] + b) - e.y;
      g0 += r * e.x[0]; g1 += r * e.x[1]; gb += r;
    }
    const n = data.length;
    w = [w[0] - eta * g0 / n, w[1] - eta * g1 / n];
    b -= eta * gb / n;
    k += 1;
  }

  // The part of the line w · x + b = z inside the plotted square, or null.
  function segment(z) {
    const c = z - b, pts = [];
    if (Math.abs(w[1]) > 1e-9) for (const x of [lo, hi]) { const v = (c - w[0] * x) / w[1]; if (v >= lo && v <= hi) pts.push([x, v]); }
    if (Math.abs(w[0]) > 1e-9) for (const v of [lo, hi]) { const x = (c - w[1] * v) / w[0]; if (x >= lo && x <= hi) pts.push([x, v]); }
    let best = null, far = 1e-6;
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
        if (d > far) { far = d; best = [pts[i], pts[j]]; }
      }
    }
    return best;
  }

  function placeLine(line, z) {
    const s = segment(z);
    line.style.display = s ? "" : "none";
    if (!s) return s;
    line.setAttribute("x1", DX(s[0][0])); line.setAttribute("y1", DY(s[0][1]));
    line.setAttribute("x2", DX(s[1][0])); line.setAttribute("y2", DY(s[1][1]));
    return s;
  }

  function draw() {
    for (const t of tiles) {
      const p = sigmoid(w[0] * t.cx + w[1] * t.cy + b);
      const cls = p >= 0.5 ? "orange" : "blue";
      if (cls !== t.cls) { t.rect.setAttribute("class", cls); t.cls = cls; }
      t.rect.setAttribute("fill-opacity", (0.4 * Math.abs(2 * p - 1)).toFixed(3));
    }

    const s = placeLine(boundary, 0);
    placeLine(band[0], LN9);
    placeLine(band[1], -LN9);
    const len = Math.hypot(w[0], w[1]);
    if (s && len > 1e-9) {
      // w drawn from the middle of the visible boundary, pointing to the spam side.
      const mx = DX((s[0][0] + s[1][0]) / 2), my = DY((s[0][1] + s[1][1]) / 2);
      const ux = w[0] / len, uy = -w[1] / len;
      wArrow.update(mx, my, mx + 44 * ux, my + 44 * uy);
      wLabel.setAttribute("x", mx + 58 * ux - 4);
      wLabel.setAttribute("y", my + 58 * uy + 4);
      wArrow.group.style.display = "";
      wLabel.style.display = "";
    } else {
      wArrow.group.style.display = "none";
      wLabel.style.display = "none";
    }

    let loss = 0, right = 0;
    data.forEach((e, i) => {
      const z = w[0] * e.x[0] + w[1] * e.x[1] + b;
      const p = sigmoid(z);
      // −log of the probability given to the true label, computed stably from z.
      const m = e.y ? z : -z;
      loss += m > 0 ? Math.log1p(Math.exp(-m)) : -m + Math.log1p(Math.exp(m));
      if ((p >= 0.5) === (e.y === 1)) right += 1;
      const { miss, onCurve } = dots[i];
      onCurve.setAttribute("cx", ZX(z));
      onCurve.setAttribute("cy", SY(p));
      miss.setAttribute("x1", ZX(z)); miss.setAttribute("y1", SY(p));
      miss.setAttribute("x2", ZX(z)); miss.setAttribute("y2", SY(e.y));
    });
    loss /= data.length;

    out.innerHTML =
      `step ${k}   η = ${fmt(eta)}   ${which}\n` +
      `w = (${num(w[0])}, ${num(w[1])})   b = ${num(b)}\n` +
      `|w| = ${fmt(len)}   loss = ${fmt(loss, 4)}\n` +
      `right: ${right} of ${data.length}  <span class="orange">● spam</span>  <span class="blue">○ not spam</span>`;
  }

  slider(ctl, { label: "learning rate η", min: 0.02, max: 0.4, step: 0.02, value: eta, format: (v) => v.toFixed(2) }, (v) => {
    eta = v;
    if (data) reset();
  });

  // Steps per second grow with the step count, so the first few steps are slow enough to
  // follow and the long crawl toward 10,000 takes a few seconds.
  const player = loop(figure, (dt) => {
    if (!dt) return;
    if (k >= MAX_STEPS) {
      wait += dt;
      if (wait > HOLD) reset();
      return;
    }
    acc += dt * (4 + 1.5 * k);
    let moved = false;
    while (acc >= 1 && k < MAX_STEPS) { acc -= 1; step(); moved = true; }
    if (moved) draw();
  });

  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
  const stop = () => { player.pause(); toggle.textContent = "Play"; };
  button(btns, "Step", () => { stop(); step(); draw(); });
  button(btns, "+1000 steps", () => { stop(); for (let i = 0; i < 1000; i++) step(); draw(); });
  button(btns, "Restart", () => reset());
  button(btns, "overlapping data", () => { which = "overlapping"; load(); });
  button(btns, "separable data", () => { which = "separable"; load(); });

  load();
})();

// Step-through: the sigmoid turns a score into a probability.
(function () {
  const X = (z) => 330 + z * 44, Y = (p) => 270 - p * 210;
  const sig = (z) => 1 / (1 + Math.exp(-z));
  Learn.demo(document.getElementById("demo-sigmoid"), {
    height: 320,
    label: "The S-shaped sigmoid curve, with one email's score on the horizontal axis and its probability on the vertical axis",
    steps: [
      { s: { z: -1, cur: 0, pt: 0 }, cap: "Email A has 4 links and 2 exclamation marks. Its score is z = 0.5 × 4 + 1 × 2 − 5 = −1. A score can be any number, so it is not a probability yet." },
      { s: { cur: 1 }, cap: "This S-shaped curve is the sigmoid. It takes any score (across) and returns a number between 0 and 1 (up)." },
      { s: { pt: 1 }, cap: "Put email A on the curve at z = −1 and read across: p = 0.27. The model gives A a 27% chance of being spam." },
      { s: { z: 3 }, cap: "Email B has 6 links and 5 exclamation marks, so z = 3 + 5 − 5 = 3. The sigmoid gives p = 0.95. B is very likely spam." },
      { s: { z: 0 }, cap: "A score of exactly 0 gives p = 0.5. Here the model can't decide either way." },
      { s: { z: -5 }, cap: "Far to the left the curve flattens out. At z = −5, p is about 0.007. The curve gets close to 0 and 1 but never reaches them." },
    ],
    draw(ctx, s, c, d) {
      d.line(X(-6), Y(1), X(6), Y(1), c.line);
      d.line(X(-6), Y(0.5), X(6), Y(0.5), c.line);
      d.line(X(-6), Y(0), X(6), Y(0), c.muted, 1.5);
      d.line(X(-6), Y(1) - 14, X(-6), Y(0), c.muted, 1.5);
      for (let z = -6; z <= 6; z += 2) d.text(d.fmt(z, 0), X(z), Y(0) + 16, { c: c.muted, s: 12, a: "center" });
      for (const p of [0, 0.5, 1]) d.text(String(p), X(-6) - 8, Y(p), { c: c.muted, s: 12, a: "right" });
      d.text("score z", X(6), Y(0) + 36, { c: c.muted, s: 13, a: "right" });
      d.text("p", X(-6), Y(1) - 24, { c: c.muted, s: 13, a: "center" });
      const p = sig(s.z);
      if (s.cur > 0.01) {
        ctx.globalAlpha = s.cur;
        ctx.strokeStyle = c.blue; ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) { const z = -6 + i * 0.1; if (i) ctx.lineTo(X(z), Y(sig(z))); else ctx.moveTo(X(z), Y(sig(z))); }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      // The score on the z axis is there from the start; the probability appears with the point.
      d.line(X(s.z), Y(0) - 7, X(s.z), Y(0) + 7, c.orange, 3);
      if (s.pt > 0.01) {
        ctx.globalAlpha = s.pt;
        d.line(X(s.z), Y(0), X(s.z), Y(p), c.orange, 1.5, [4, 4]);
        d.line(X(-6), Y(p), X(s.z), Y(p), c.orange, 1.5, [4, 4]);
        d.dot(X(s.z), Y(p), 7, c.orange);
        ctx.globalAlpha = 1;
      }
      d.panel(c, 84, 84, 188, s.pt > 0.01 ? 70 : 40);
      d.text(`z = ${d.fmt(s.z, 2)}`, 98, 104, { c: c.ink, s: 17, w: 700 });
      if (s.pt > 0.01) {
        ctx.globalAlpha = s.pt;
        d.text(`p = σ(z) = ${d.fmt(p, 3)}`, 98, 134, { c: c.orange, w: 700 });
      }
    },
  });
})();

// Step-through: the decision boundary, the direction of w, and what the length of w does.
(function () {
  const X = (x) => 40 + x * 34, Y = (y) => 334 - y * 34;
  const normal = [[0, 0], [1, 0], [0, 2], [2, 1], [1, 3], [3, 0], [2, 4], [4, 2], [3, 3], [5, 1], [6, 3]];
  const spam = [[5, 6], [7, 3], [6, 5], [8, 7], [4, 7], [9, 4], [6, 8], [7, 6], [3, 6], [8, 2], [3, 2]];
  const LEN = Math.hypot(0.5, 1), Z9 = Math.log(9); // score where p = 0.9 (about 2.2)
  Learn.demo(document.getElementById("demo-boundary"), {
    height: 360,
    label: "Emails plotted by links and exclamation marks, split by a straight decision boundary with the weight vector crossing it",
    steps: [
      { s: { ln: 0, sh: 0, wv: 0, band: 0, sc: 1 }, cap: "Each mark is one email, placed by its links (across) and exclamation marks (up). Orange dots are spam. Blue rings are normal email. A and B are the two emails from the text." },
      { s: { ln: 1 }, cap: "With w = (0.5, 1) and b = −5, the score is zero on the line 0.5 x₁ + x₂ = 5. It runs from (0, 5) to (10, 0). This line is the decision boundary." },
      { s: { sh: 1 }, cap: "Above the line the score is positive, so the model says spam. Below it the score is negative, so the model says not spam. B is above. A is below." },
      { s: { wv: 1 }, cap: "The arrow is w = (0.5, 1). It crosses the boundary at a right angle and points to the spam side. Moving along w raises the score fastest." },
      { s: { band: 1 }, cap: "The dashed lines mark p = 0.1 and p = 0.9, where the score is −2.2 and 2.2. They sit 2.2 ÷ 1.12 = 1.97 units from the boundary. Between them the model is unsure." },
      { s: { sc: 2 }, cap: "Double w and b. The boundary stays put, because the same points score zero. The dashed lines move in to 0.98 units, so the model is now surer about every email." },
    ],
    draw(ctx, s, c, d) {
      for (let g = 0; g <= 10; g += 2) d.line(X(g), Y(0), X(g), Y(9), c.line);
      for (let g = 0; g <= 9; g += 3) d.line(X(0), Y(g), X(10), Y(g), c.line);
      d.line(X(0), Y(0), X(10), Y(0), c.muted, 1.5);
      d.line(X(0), Y(0), X(0), Y(9), c.muted, 1.5);
      for (let g = 0; g <= 10; g += 2) d.text(String(g), X(g), Y(0) + 15, { c: c.muted, s: 12, a: "center" });
      for (let g = 3; g <= 9; g += 3) d.text(String(g), X(0) - 9, Y(g), { c: c.muted, s: 12, a: "right" });
      d.text("links x₁", X(10) + 22, Y(0) + 15, { c: c.muted, s: 12 });
      d.text("exclamation marks x₂", X(0) + 4, Y(9) - 12, { c: c.muted, s: 12 });
      ctx.save();
      ctx.beginPath(); ctx.rect(X(0), Y(9), X(10) - X(0), Y(0) - Y(9)); ctx.clip();
      if (s.sh > 0.01) {
        ctx.globalAlpha = s.sh * 0.13;
        ctx.fillStyle = c.orange;
        ctx.beginPath(); ctx.moveTo(X(0), Y(5)); ctx.lineTo(X(10), Y(0)); ctx.lineTo(X(10), Y(9)); ctx.lineTo(X(0), Y(9)); ctx.closePath(); ctx.fill();
        ctx.fillStyle = c.blue;
        ctx.beginPath(); ctx.moveTo(X(0), Y(5)); ctx.lineTo(X(10), Y(0)); ctx.lineTo(X(0), Y(0)); ctx.closePath(); ctx.fill();
      }
      if (s.band > 0.01) {
        ctx.globalAlpha = s.band;
        const off = Z9 / s.sc; // the line 0.5 x1 + x2 = 5 ± off
        d.line(X(0), Y(5 + off), X(10), Y(off), c.ink, 1.5, [6, 5]);
        d.line(X(0), Y(5 - off), X(10), Y(-off), c.ink, 1.5, [6, 5]);
      }
      if (s.ln > 0.01) {
        ctx.globalAlpha = s.ln;
        d.line(X(0), Y(5), X(10), Y(0), c.ink, 3);
      }
      ctx.restore();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2; ctx.strokeStyle = c.blue;
      for (const [x, y] of normal) { ctx.beginPath(); ctx.arc(X(x), Y(y), 5, 0, 7); ctx.stroke(); }
      for (const [x, y] of spam) d.dot(X(x), Y(y), 5.5, c.orange);
      d.text("A", X(4) + 9, Y(2) + 12, { c: c.blue, w: 700 });
      d.text("B", X(6) + 9, Y(5) + 12, { c: c.orange, w: 700 });
      if (s.wv > 0.01) {
        ctx.globalAlpha = s.wv;
        d.arrow(X(4), Y(3), X(4 + 0.5 * s.sc), Y(3 + s.sc), c.purple, 3.5);
        d.text("w", X(4 + 0.5 * s.sc) + 8, Y(3 + s.sc) - 4, { c: c.purple, w: 700 });
      }
      if (s.sh > 0.01) {
        ctx.globalAlpha = s.sh;
        d.text("spam side", X(10) - 6, Y(8.4), { c: c.orange, s: 13, w: 700, a: "right" });
        d.text("not spam", X(0.4), Y(0.5), { c: c.blue, s: 13, w: 700 });
      }
      if (s.ln > 0.01) {
        ctx.globalAlpha = s.ln;
        d.panel(c, 420, 40, 208, s.band > 0.01 ? 226 : s.wv > 0.01 ? 132 : 104);
        d.text(`w = (${d.fmt(0.5 * s.sc, 2)}, ${d.fmt(s.sc, 2)})`, 434, 64, { c: c.purple, w: 700 });
        d.text(`b = ${d.fmt(-5 * s.sc, 1)}`, 434, 88, { c: c.ink, w: 700 });
        d.text("boundary: z = 0", 434, 120, { c: c.muted, s: 13 });
        if (s.wv > 0.01) {
          ctx.globalAlpha = s.wv;
          d.text(`length of w = ${d.fmt(LEN * s.sc, 2)}`, 434, 148, { c: c.ink, s: 13 });
        }
        if (s.band > 0.01) {
          ctx.globalAlpha = s.band;
          d.text("p = 0.1 and p = 0.9", 434, 184, { c: c.muted, s: 13 });
          d.text("lines sit", 434, 204, { c: c.muted, s: 13 });
          d.text(`${d.fmt(2.2 / (LEN * s.sc), 2)} units`, 434, 228, { c: c.ink, s: 17, w: 700 });
          d.text("from the boundary", 434, 250, { c: c.muted, s: 13 });
        }
      }
    },
  });
})();

// Step-through: log loss against the probability the model gave to the true label.
(function () {
  const X = (q) => 70 + q * 380, Y = (L) => 280 - L * 50;
  Learn.demo(document.getElementById("demo-logloss"), {
    height: 330,
    label: "The log loss curve: zero when the true label gets probability 1, rising without limit as that probability falls toward 0",
    steps: [
      { s: { q: 0.95, pt: 0 }, cap: "Across: the probability the model gave to what really happened. Up: the loss, minus the log of that probability. The loss is 0 at probability 1 and climbs as the probability falls." },
      { s: { pt: 1 }, cap: "Email B is spam and the model said p = 0.95. The true label got 0.95, so the loss is −log 0.95 = 0.05. Right and confident costs almost nothing." },
      { s: { q: 0.73 }, cap: "Email A is normal and the model said p = 0.27 for spam. That leaves 0.73 on the true label. The loss is −log 0.73 = 0.31." },
      { s: { q: 0.182 }, cap: "The spam email at (3, 2) got p = 0.182. The model is wrong and fairly sure of itself. The loss is 1.70, more than A and B put together." },
      { s: { q: 0.01 }, cap: "Push the probability on the truth down to 0.01 and the loss is 4.61. Each drop by a factor of 10 adds another 2.30. There is no ceiling." },
    ],
    draw(ctx, s, c, d) {
      for (let L = 1; L <= 5; L++) d.line(X(0), Y(L), X(1), Y(L), c.line);
      d.line(X(0), Y(0), X(1), Y(0), c.muted, 1.5);
      d.line(X(0), Y(5) - 10, X(0), Y(0), c.muted, 1.5);
      for (const q of [0, 0.25, 0.5, 0.75, 1]) d.text(String(q), X(q), Y(0) + 16, { c: c.muted, s: 12, a: "center" });
      for (let L = 1; L <= 5; L++) d.text(String(L), X(0) - 9, Y(L), { c: c.muted, s: 12, a: "right" });
      d.text("probability on the true label", X(1), Y(0) + 36, { c: c.muted, s: 13, a: "right" });
      d.text("loss", X(0) - 4, 14, { c: c.muted, s: 13, a: "right" });
      ctx.strokeStyle = c.blue; ctx.lineWidth = 3;
      ctx.beginPath();
      for (let i = 0; i <= 200; i++) {
        const q = 0.0055 + (1 - 0.0055) * i / 200, y = Y(-Math.log(q));
        if (i) ctx.lineTo(X(q), y); else ctx.moveTo(X(q), y);
      }
      ctx.stroke();
      const L = -Math.log(s.q);
      if (s.pt > 0.01) {
        ctx.globalAlpha = s.pt;
        d.line(X(s.q), Y(0), X(s.q), Y(L), c.orange, 1.5, [4, 4]);
        d.line(X(0), Y(L), X(s.q), Y(L), c.orange, 1.5, [4, 4]);
        d.dot(X(s.q), Y(L), 7, L > 1 ? c.red : c.orange);
        d.panel(c, 476, 40, 152, 100);
        d.text("on the truth", 488, 62, { c: c.muted, s: 13 });
        d.text(d.fmt(s.q, 3), 488, 84, { c: c.ink, w: 700 });
        d.text(`loss ${d.fmt(L, 2)}`, 488, 116, { c: L > 1 ? c.red : c.ink, s: 18, w: 700 });
      }
    },
  });
})();

// Step-through: how hard log loss and squared error push on the score of a misjudged spam email.
(function () {
  const X = (z) => 196 + z * 25, Y = (p) => 262 - p * 190;
  const sig = (z) => 1 / (1 + Math.exp(-z));
  Learn.demo(document.getElementById("demo-push"), {
    height: 320,
    label: "A spam email sliding left along the sigmoid, with bars comparing the gradient of log loss and of squared error",
    steps: [
      { s: { z: 0, bars: 0 }, cap: "A spam email (y = 1) sits on the sigmoid at score z = 0, so p = 0.5. Training should push its score to the right. The short line through the dot shows the curve's slope there." },
      { s: { bars: 1 }, cap: "The bars show how hard each loss pushes on the score. Log loss pushes with p − y = −0.5. Squared error pushes with 2(p − y) × p(1 − p) = −0.25." },
      { s: { z: -2 }, cap: "Move the email to z = −2, so p = 0.12. The model is more wrong now. Log loss pushes harder: −0.88. Squared error pushes less: −0.18, because the curve is flatter here." },
      { s: { z: -6 }, cap: "At z = −6, p = 0.0025. The model is badly wrong. Log loss pushes with −0.9975. Squared error pushes with only −0.0049, about 200 times weaker." },
    ],
    draw(ctx, s, c, d) {
      d.line(X(-7), Y(1), X(7), Y(1), c.line);
      d.line(X(-7), Y(0), X(7), Y(0), c.muted, 1.5);
      d.line(X(0), Y(1) - 10, X(0), Y(0), c.line);
      for (let z = -6; z <= 6; z += 3) d.text(d.fmt(z, 0), X(z), Y(0) + 16, { c: c.muted, s: 12, a: "center" });
      d.text("score z", X(7), Y(0) + 36, { c: c.muted, s: 13, a: "right" });
      d.text("p = 1", X(-7), Y(1) - 12, { c: c.muted, s: 12 });
      ctx.strokeStyle = c.blue; ctx.lineWidth = 3;
      ctx.beginPath();
      for (let i = 0; i <= 140; i++) { const z = -7 + i * 0.1; if (i) ctx.lineTo(X(z), Y(sig(z))); else ctx.moveTo(X(z), Y(sig(z))); }
      ctx.stroke();
      const p = sig(s.z), slope = p * (1 - p);
      const push = p - 1, pushSq = 2 * (p - 1) * slope;
      // Tangent: a 3-unit-wide piece of the line with the sigmoid's slope at this score.
      d.line(X(s.z - 1.5), Y(p - 1.5 * slope), X(s.z + 1.5), Y(p + 1.5 * slope), c.ink, 2);
      d.dot(X(s.z), Y(p), 7, c.orange);
      d.text("spam email", Math.max(24, X(s.z) - 90), Y(p) - 18, { c: c.orange, s: 13, w: 700 });
      d.panel(c, 392, 24, 236, s.bars > 0.01 ? 250 : 84);
      d.text(`z = ${d.fmt(s.z, 1)}   p = ${d.fmt(p, 4)}`, 404, 46, { c: c.ink, w: 700 });
      d.text(`slope p(1 − p) = ${d.fmt(slope, 4)}`, 404, 78, { c: c.muted, s: 13 });
      if (s.bars > 0.01) {
        ctx.globalAlpha = s.bars;
        d.text("push on the score", 404, 114, { c: c.muted, s: 13, w: 700 });
        d.text(`log loss  ${d.fmt(push, 4)}`, 404, 142, { c: c.ink });
        d.rect(404, 156, 200, 16, c.soft);
        d.rect(404, 156, Math.max(1.5, 200 * -push), 16, c.orange);
        d.text(`squared error  ${d.fmt(pushSq, 4)}`, 404, 198, { c: c.ink });
        d.rect(404, 212, 200, 16, c.soft);
        d.rect(404, 212, Math.max(1.5, 200 * -pushSq), 16, c.red);
        d.text(`ratio: ${d.fmt(push / pushSq, 0)} to 1`, 404, 252, { c: c.muted, s: 13, w: 700 });
      }
    },
  });
})();
