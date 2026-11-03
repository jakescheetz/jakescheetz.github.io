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
  svg("text", { x: ZX(-10) - 4, y: SY(0) + 18, class: "muted", text: "−10" }, root);
  svg("text", { x: ZX(10) - 30, y: SY(0) + 18, class: "muted", text: "10" }, root);
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
