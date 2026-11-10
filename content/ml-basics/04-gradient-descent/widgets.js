// Unit 04: gradient descent on the line-fit loss. Left: the data and the current line.
// Right: contours of the loss over (w, b) and the path gradient descent takes.
(function () {
  const figure = document.getElementById("descent");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  // The four points from the text, and the loss L(w, b) = mean((w x + b - y)^2).
  const xs = [0, 1, 2, 3], ys = [1, 3, 4, 7];
  const N = xs.length;
  const loss = (w, b) => xs.reduce((s, x, i) => s + (w * x + b - ys[i]) ** 2, 0) / N;
  const grad = (w, b) => {
    let gw = 0, gb = 0;
    xs.forEach((x, i) => { const e = w * x + b - ys[i]; gw += e * x; gb += e; });
    return [2 * gw / N, 2 * gb / N];
  };

  // Exact minimum and the bowl's shape. The second derivatives of L are constant:
  // H = (2/N) [[sum x^2, sum x], [sum x, N]]. Its eigenvalues are the curvatures along the
  // ellipse axes (steep and flat), its eigenvectors the axis directions.
  const sx = xs.reduce((s, x) => s + x, 0), sxx = xs.reduce((s, x) => s + x * x, 0);
  const sy = ys.reduce((s, y) => s + y, 0), sxy = xs.reduce((s, x, i) => s + x * ys[i], 0);
  const wStar = (N * sxy - sx * sy) / (N * sxx - sx * sx), bStar = (sy - wStar * sx) / N;
  const lossStar = loss(wStar, bStar);
  const h11 = 2 * sxx / N, h12 = 2 * sx / N, h22 = 2;
  const mid = (h11 + h22) / 2, gap = Math.sqrt(((h11 - h22) / 2) ** 2 + h12 * h12);
  const steep = mid + gap, flat = mid - gap;
  const flatDir = [h12, flat - h11]; // eigenvector for the flat curvature (unnormalized)

  const width = 720, rows = 18, height = rows * Learn.ROW;
  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Left: four data points and the current line. Right: loss contours over w and b with the gradient descent path.");
  const defs = svg("defs", {}, root);

  // Left panel: data space.
  const left = { x0: 0, x1: 230 };
  const dx = (x) => 30 + x * 56;           // x in [-0.5, 3.5]
  const dy = (y) => height - 26 - y * 30;  // y in [-0.8, 9]
  const clipData = svg("clipPath", { id: "descent-clip-data" }, defs);
  svg("rect", { x: left.x0, y: 0, width: left.x1 - left.x0, height }, clipData);
  const dataG = svg("g", { "clip-path": "url(#descent-clip-data)" }, root);
  svg("line", { x1: dx(-0.5), y1: dy(0), x2: dx(3.6), y2: dy(0), class: "muted faint thin" }, dataG);
  svg("line", { x1: dx(0), y1: dy(-0.8), x2: dx(0), y2: dy(9.5), class: "muted faint thin" }, dataG);
  svg("text", { x: dx(3.6) - 12, y: dy(0) - 8, class: "muted", text: "x" }, root);
  svg("text", { x: dx(0) + 8, y: 16, class: "muted", text: "y" }, root);
  const residuals = xs.map(() => svg("line", { class: "muted dashed thin" }, dataG));
  const fitLine = svg("line", { class: "orange" }, dataG);
  xs.forEach((x, i) => svg("circle", { cx: dx(x), cy: dy(ys[i]), r: 4, fill: "currentColor", class: "ink" }, dataG));

  // Right panel: parameter space (w across, b up), equal scale on both axes.
  const panel = { x0: 250, x1: width };
  const scale = 64, cw = 1.6, cb = 1.8;
  const PX = (w) => (panel.x0 + panel.x1) / 2 + (w - cw) * scale;
  const PY = (b) => height / 2 - (b - cb) * scale;
  const clipBowl = svg("clipPath", { id: "descent-clip-bowl" }, defs);
  svg("rect", { x: panel.x0, y: 0, width: panel.x1 - panel.x0, height }, clipBowl);
  const bowl = svg("g", { "clip-path": "url(#descent-clip-bowl)" }, root);
  svg("line", { x1: panel.x0, y1: PY(0), x2: panel.x1, y2: PY(0), class: "muted faint thin" }, bowl);
  svg("line", { x1: PX(0), y1: 0, x2: PX(0), y2: height, class: "muted faint thin" }, bowl);
  svg("line", { x1: panel.x0, y1: 0, x2: panel.x0, y2: height, class: "muted faint thin" }, root);
  svg("text", { x: panel.x1 - 14, y: PY(0) + 18, class: "muted", text: "w" }, root);
  svg("text", { x: PX(0) + 8, y: 16, class: "muted", text: "b" }, root);

  // Contours: L - L* = c is an ellipse with radius sqrt(2c / curvature) along each axis.
  const tilt = Math.atan2(-flatDir[1], flatDir[0]) * 180 / Math.PI; // screen y points down
  const contours = svg("g", { class: "muted" }, bowl);
  for (const c of [0.05, 0.2, 0.8, 3.2, 12.8]) {
    svg("ellipse", {
      cx: PX(wStar), cy: PY(bStar),
      rx: Math.sqrt(2 * c / flat) * scale, ry: Math.sqrt(2 * c / steep) * scale,
      transform: `rotate(${tilt} ${PX(wStar)} ${PY(bStar)})`,
      fill: "none", stroke: "currentColor", "stroke-width": 1,
    }, contours);
  }
  const best = svg("g", { class: "green" }, bowl);
  svg("line", { x1: PX(wStar) - 6, y1: PY(bStar), x2: PX(wStar) + 6, y2: PY(bStar) }, best);
  svg("line", { x1: PX(wStar), y1: PY(bStar) - 6, x2: PX(wStar), y2: PY(bStar) + 6 }, best);
  const trail = svg("polyline", { class: "blue", "stroke-width": 1.5, "stroke-linejoin": "round" }, bowl);
  const dots = svg("g", { class: "blue" }, bowl);
  const current = svg("circle", { r: 6, fill: "currentColor", class: "orange" }, bowl);

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  const MAX_STEPS = 100, INTERVAL = 0.12, HOLD = 2.5;
  let eta = 0.1;
  let start = [-1, 3];
  let w, b, k, points, status, wait;

  function reset() {
    [w, b] = start;
    k = 0;
    points = [[w, b]];
    status = "";
    wait = 0;
    while (dots.firstChild) dots.removeChild(dots.firstChild);
    dot(w, b);
    draw();
  }

  function dot(w, b) {
    svg("circle", { cx: PX(w), cy: PY(b), r: 2.5, fill: "currentColor" }, dots);
  }

  function step() {
    if (status) return;
    const [gw, gb] = grad(w, b);
    w -= eta * gw;
    b -= eta * gb;
    k += 1;
    // Keep drawn coordinates bounded once the path has left the panel.
    if (Math.abs(w) < 1e3 && Math.abs(b) < 1e3) {
      points.push([w, b]);
      dot(w, b);
    }
    const L = loss(w, b);
    if (!isFinite(L) || L > 1e5) status = "diverged";
    else if (L - lossStar < 1e-4) status = "reached the bottom";
    else if (k >= MAX_STEPS) status = `stopped after ${MAX_STEPS} steps`;
    draw();
  }

  function draw() {
    trail.setAttribute("points", points.map(([pw, pb]) => `${PX(pw)},${PY(pb)}`).join(" "));
    const shown = Math.abs(w) < 1e3 && Math.abs(b) < 1e3;
    current.style.display = shown ? "" : "none";
    if (shown) {
      current.setAttribute("cx", PX(w));
      current.setAttribute("cy", PY(b));
      fitLine.setAttribute("x1", dx(-0.5)); fitLine.setAttribute("y1", dy(b - 0.5 * w));
      fitLine.setAttribute("x2", dx(3.6)); fitLine.setAttribute("y2", dy(b + 3.6 * w));
      xs.forEach((x, i) => {
        residuals[i].setAttribute("x1", dx(x)); residuals[i].setAttribute("y1", dy(ys[i]));
        residuals[i].setAttribute("x2", dx(x)); residuals[i].setAttribute("y2", dy(w * x + b));
      });
    }
    const L = loss(w, b);
    const big = (v) => (Math.abs(v) < 1e5 ? fmt(v, 4) : v.toExponential(2));
    const fs = 1 - eta * steep, ff = 1 - eta * flat;
    out.innerHTML =
      `η = ${fmt(eta, 3)}   step ${k}${status ? `   <span class="${status === "diverged" ? "orange" : "green"}">${status}</span>` : ""}\n` +
      `<span class="orange">w = ${big(w)}   b = ${big(b)}</span>\n` +
      `loss = ${big(L)}   (lowest ${fmt(lossStar, 4)})\n` +
      `distance to the bottom, each step:\n` +
      `  steep  × (1 − η·${fmt(steep)}) = <span class="${Math.abs(fs) >= 1 ? "orange" : "blue"}">${fmt(fs)}</span>\n` +
      `  flat   × (1 − η·${fmt(flat)}) = <span class="blue">${fmt(ff)}</span>`;
  }

  const etaControl = slider(ctl, { label: "learning rate η", min: 0.01, max: 0.26, step: 0.005, value: eta, format: (v) => v.toFixed(3) }, (v) => {
    eta = v;
    reset();
  });

  let acc = 0;
  const player = loop(figure, (dt) => {
    if (!dt) return;
    acc += dt;
    if (status) {
      wait += dt;
      if (wait > HOLD) { reset(); acc = 0; }
      return;
    }
    while (acc > INTERVAL && !status) {
      acc -= INTERVAL;
      step();
    }
  });

  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
  const stop = () => { player.pause(); toggle.textContent = "Play"; };
  button(btns, "Step", () => { stop(); if (status) reset(); else step(); });
  button(btns, "Restart", () => { start = [-1, 3]; reset(); });
  for (const [label, value] of [["slow", 0.02], ["steady", 0.1], ["zig-zag", 0.2], ["diverge", 0.25]]) {
    button(btns, label, () => { etaControl.set(value); });
  }

  // Click or tap in the bowl to start from that (w, b) instead.
  root.addEventListener("click", (event) => {
    const ctm = root.getScreenCTM();
    if (!ctm) return;
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    if (p.x < panel.x0) return;
    start = [cw + (p.x - (panel.x0 + panel.x1) / 2) / scale, cb - (p.y - height / 2) / scale];
    reset();
  });

  reset();
})();

// Step-through: the loss of a line is the average of its squared misses.
(function () {
  const xs = [0, 1, 2, 3], ys = [1, 3, 4, 7];
  const X = (x) => 60 + x * 100, Y = (y) => 300 - y * 34;
  Learn.demo(document.getElementById("demo-loss"), {
    height: 340,
    label: "Four data points, a straight line, the miss at each point and the mean squared error",
    steps: [
      { s: { line: 0, w: 2, b: 0, miss: 0, sq: 0 }, cap: "Here are four points: (0, 1), (1, 3), (2, 4) and (3, 7). We want one straight line that comes close to all of them." },
      { s: { line: 1 }, cap: "Try the line ŷ = 2x + 0, so w = 2 and b = 0. It predicts 0, 2, 4 and 6 at the four x values." },
      { s: { miss: 1 }, cap: "A miss is the prediction minus the target. The dashed bars are the misses: −1, −1, 0 and −1. This line sits below three of the points." },
      { s: { sq: 1 }, cap: "Square each miss and average: (1 + 1 + 0 + 1) ÷ 4 = 0.75. That one number is the loss of this line." },
      { s: { w: -1, b: 3 }, cap: "Now a bad guess: w = −1, b = 3. The line slopes the wrong way. The misses are 2, −1, −3 and −7, and the loss jumps to 63 ÷ 4 = 15.75." },
      { s: { w: 1.9, b: 0.9 }, cap: "This is the best line: w = 1.9, b = 0.9. Its loss is 0.175. No straight line scores lower on these four points." },
    ],
    draw(ctx, s, c, d) {
      for (let gx = 0; gx <= 3; gx++) d.line(X(gx), Y(8), X(gx), Y(-0.6), c.line);
      for (let gy = 0; gy <= 8; gy += 2) d.line(X(-0.3), Y(gy), X(3.4), Y(gy), c.line);
      d.line(X(-0.3), Y(0), X(3.4), Y(0), c.muted, 1.5);
      d.line(X(0), Y(8), X(0), Y(-0.6), c.muted, 1.5);
      for (let gx = 1; gx <= 3; gx++) d.text(String(gx), X(gx), Y(0) + 14, { c: c.muted, s: 12, a: "center" });
      for (let gy = 2; gy <= 8; gy += 2) d.text(String(gy), X(0) - 10, Y(gy), { c: c.muted, s: 12, a: "right" });
      d.text("x", X(3.4) + 8, Y(0), { c: c.muted, s: 13 });
      d.text("y", X(0), Y(8) - 12, { c: c.muted, s: 13, a: "center" });
      const e = xs.map((x, i) => s.w * x + s.b - ys[i]);
      const loss = e.reduce((t, v) => t + v * v, 0) / 4;
      if (s.line > 0.01) {
        ctx.globalAlpha = s.line;
        d.line(X(-0.3), Y(s.b - 0.3 * s.w), X(3.4), Y(s.b + 3.4 * s.w), c.blue, 3);
      }
      if (s.miss > 0.01) {
        ctx.globalAlpha = s.miss;
        xs.forEach((x, i) => {
          d.line(X(x), Y(ys[i]), X(x), Y(ys[i] + e[i]), c.orange, 3, [5, 4]);
          d.text(d.fmt(e[i], 1), X(x) + 9, Y(ys[i] + e[i] / 2), { c: c.orange, s: 13, w: 700 });
        });
      }
      ctx.globalAlpha = 1;
      xs.forEach((x, i) => d.dot(X(x), Y(ys[i]), 5.5, c.ink));
      if (s.line > 0.01) {
        ctx.globalAlpha = s.line;
        d.panel(c, 428, 24, 200, s.sq > 0.01 ? 240 : 56);
        d.text(`ŷ = ${d.fmt(s.w, 1)}x + ${d.fmt(s.b, 1)}`.replace("+ −", "− "), 442, 52, { c: c.blue, w: 700 });
        if (s.sq > 0.01) {
          ctx.globalAlpha = s.sq * s.line;
          d.text(" miss   miss²", 442, 88, { c: c.muted });
          e.forEach((v, i) => d.text(d.fmt(v, 1).padStart(5) + d.fmt(v * v, 2).padStart(8), 442, 112 + i * 22, { c: c.ink }));
          d.line(442, 200, 614, 200, c.muted, 1.5);
          d.text("average of miss²", 442, 218, { c: c.muted, s: 13 });
          d.text(`loss = ${d.fmt(loss, 3)}`, 442, 244, { c: loss > 5 ? c.red : c.ink, s: 20, w: 700 });
        }
      }
    },
  });
})();

// Step-through: gradient descent on the line fit, from the bad guess toward the best line.
(function () {
  const xs = [0, 1, 2, 3], ys = [1, 3, 4, 7], eta = 0.1;
  const X = (x) => 44 + x * 72, Y = (y) => 300 - y * 32;
  const loss = (w, b) => xs.reduce((t, x, i) => t + (w * x + b - ys[i]) ** 2, 0) / 4;
  const grad = (w, b) => {
    let gw = 0, gb = 0;
    xs.forEach((x, i) => { const e = w * x + b - ys[i]; gw += e * x; gb += e; });
    return [gw / 2, gb / 2]; // (2 / N) times the sums, with N = 4
  };
  const path = [[-1, 3]];
  for (let k = 0; k < 50; k++) { const [w, b] = path[k], [gw, gb] = grad(w, b); path.push([w - eta * gw, b - eta * gb]); }
  const at = (k) => {
    const i = Math.min(49, Math.floor(k)), t = k - i;
    return [path[i][0] + (path[i + 1][0] - path[i][0]) * t, path[i][1] + (path[i + 1][1] - path[i][1]) * t];
  };
  const L0 = loss(-1, 3);
  Learn.demo(document.getElementById("demo-step"), {
    height: 340,
    label: "A line through four points improving step by step under gradient descent, with its gradient and loss",
    steps: [
      { s: { k: 0, g: 0 }, cap: "Start from the bad guess w = −1, b = 3. The loss is 15.75. The red bar shows how big that is." },
      { s: { g: 1 }, cap: "Compute the gradient here: ∂L/∂w = −14 and ∂L/∂b = −4.5. Both are negative, so raising w or raising b lowers the loss." },
      { s: { k: 1 }, cap: "Take one step with η = 0.1: w = −1 + 1.4 = 0.4 and b = 3 + 0.45 = 3.45. The loss drops from 15.75 to 3.08." },
      { s: { k: 2 }, cap: "Compute the gradient again at the new spot, then step. Now w = 0.685, b = 3.39 and the loss is 2.47. The gradient is smaller, so the steps are smaller." },
      { s: { k: 10 }, cap: "After 10 steps: w = 1.18, b = 2.44 and the loss is 1.03. The line now tilts the right way." },
      { s: { k: 50 }, cap: "After 50 steps: w = 1.84, b = 1.03 and the loss is 0.18. The dashed green line is the best one, w = 1.9, b = 0.9, with loss 0.175." },
    ],
    draw(ctx, s, c, d) {
      const [w, b] = at(s.k), [gw, gb] = grad(w, b), L = loss(w, b);
      for (let gx = 0; gx <= 3; gx++) d.line(X(gx), Y(8), X(gx), Y(-0.6), c.line);
      for (let gy = 0; gy <= 8; gy += 2) d.line(X(-0.3), Y(gy), X(3.4), Y(gy), c.line);
      d.line(X(-0.3), Y(0), X(3.4), Y(0), c.muted, 1.5);
      d.line(X(0), Y(8), X(0), Y(-0.6), c.muted, 1.5);
      d.line(X(-0.3), Y(0.9 - 0.3 * 1.9), X(3.4), Y(0.9 + 3.4 * 1.9), c.green, 1.5, [5, 5]);
      xs.forEach((x, i) => d.line(X(x), Y(ys[i]), X(x), Y(w * x + b), c.orange, 2, [4, 4]));
      d.line(X(-0.3), Y(b - 0.3 * w), X(3.4), Y(b + 3.4 * w), c.blue, 3);
      xs.forEach((x, i) => d.dot(X(x), Y(ys[i]), 5.5, c.ink));
      d.panel(c, 330, 24, 298, 292);
      d.text(`step ${Math.round(s.k)}`, 346, 48, { c: c.muted, s: 13, w: 700 });
      d.text(`w = ${d.fmt(w, 3)}`, 346, 78, { c: c.blue, s: 17, w: 700 });
      d.text(`b = ${d.fmt(b, 3)}`, 346, 104, { c: c.blue, s: 17, w: 700 });
      if (s.g > 0.01) {
        ctx.globalAlpha = s.g;
        d.text("gradient (uphill direction)", 346, 140, { c: c.muted, s: 13 });
        d.text(`∂L/∂w = ${d.fmt(gw, 2)}`, 346, 162, { c: c.orange, w: 700 });
        d.text(`∂L/∂b = ${d.fmt(gb, 2)}`, 346, 184, { c: c.orange, w: 700 });
        d.text("step: subtract 0.1 × gradient", 346, 208, { c: c.muted, s: 13 });
        ctx.globalAlpha = 1;
      }
      d.text(`loss = ${d.fmt(L, 3)}`, 346, 246, { c: c.ink, s: 20, w: 700 });
      d.rect(346, 268, 266, 20, c.soft);
      d.rect(346, 268, Math.max(2, 266 * L / L0), 20, L > 5 ? c.red : c.orange);
      d.line(346 + 266 * 0.175 / L0, 262, 346 + 266 * 0.175 / L0, 294, c.green, 2);
      d.text("green mark: lowest possible", 346, 304, { c: c.muted, s: 12 });
    },
  });
})();

// Step-through: the learning rate on the one-parameter bowl. Each step multiplies θ by 1 − ηλ.
(function () {
  const X = (t) => 240 + t * 95, Y = (L) => 300 - L * 100;
  const H = (t) => t * t / 2; // the bowl ½λθ² drawn with λ = 1
  Learn.demo(document.getElementById("demo-rate"), {
    height: 340,
    label: "A bowl-shaped loss with the path gradient descent takes for four different learning rates",
    steps: [
      { s: { k: 0.1, n: 0 }, cap: "The loss is a bowl with its bottom at θ = 0. The ball starts at θ = 1. Every step multiplies θ by the factor 1 − ηλ." },
      { s: { n: 5 }, cap: "With ηλ = 0.1 the factor is 0.9. Five steps only move θ from 1 to 0.59. This is safe but slow." },
      { s: { k: 0.9 }, cap: "With ηλ = 0.9 the factor is 0.1. θ goes 1, 0.1, 0.01. The ball is at the bottom after two or three steps." },
      { s: { k: 1.5 }, cap: "With ηλ = 1.5 the factor is −0.5. Each step jumps over the bottom to the other side but lands half as far away: 1, −0.5, 0.25, −0.125." },
      { s: { k: 2.5 }, cap: "With ηλ = 2.5 the factor is −1.5. Each jump lands farther out than the last: 1, −1.5, 2.25, −3.375. The ball leaves the bowl. Training has diverged." },
    ],
    draw(ctx, s, c, d) {
      const f = 1 - s.k, th = [1];
      for (let j = 1; j <= 5; j++) th.push(th[j - 1] * f);
      d.line(X(-2.3), Y(0), X(2.3), Y(0), c.muted, 1.5);
      d.line(X(0), Y(0) - 4, X(0), Y(0) + 6, c.muted, 1.5);
      for (const t of [-2, -1, 0, 1, 2]) d.text(d.fmt(t, 0), X(t), Y(0) + 18, { c: c.muted, s: 12, a: "center" });
      d.text("θ", X(2.3) + 4, Y(0) + 18, { c: c.muted, s: 13 });
      d.text("loss", X(0), 24, { c: c.muted, s: 13, a: "center" });
      ctx.save();
      ctx.beginPath(); ctx.rect(X(-2.3), 12, X(2.3) - X(-2.3), Y(0) - 12 + 3); ctx.clip();
      ctx.strokeStyle = c.ink; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= 92; i++) { const t = -2.3 + i * 0.05; if (i) ctx.lineTo(X(t), Y(H(t))); else ctx.moveTo(X(t), Y(H(t))); }
      ctx.stroke();
      // Clamp far-off points so the drawn coordinates stay small once θ has left the bowl.
      const P = (t) => { const u = d.clamp(t, -9, 9); return [X(u), Y(H(u))]; };
      let ball = P(th[0]);
      for (let j = 0; j < 5; j++) {
        const part = d.clamp(s.n - j, 0, 1);
        if (part <= 0) break;
        const [x1, y1] = P(th[j]), [x2, y2] = P(th[j + 1]);
        ball = [x1 + (x2 - x1) * part, y1 + (y2 - y1) * part];
        d.line(x1, y1, ball[0], ball[1], c.blue, 2);
        d.dot(x1, y1, 4, c.blue);
      }
      d.dot(ball[0], ball[1], 7, Math.abs(f) > 1 ? c.red : c.orange);
      ctx.restore();
      d.text("start", X(1) + 10, Y(H(1)) + 14, { c: c.muted, s: 12 });
      d.panel(c, 474, 24, 154, 250);
      d.text(`ηλ = ${d.fmt(s.k, 2)}`, 486, 46, { c: c.ink, w: 700 });
      d.text(`factor ${d.fmt(f, 2)}`, 486, 70, { c: Math.abs(f) > 1 ? c.red : c.blue, w: 700 });
      d.text("θ after step", 486, 100, { c: c.muted, s: 13 });
      for (let j = 0; j <= 5; j++) {
        ctx.globalAlpha = j === 0 ? 1 : d.clamp(s.n - j + 1, 0, 1);
        d.text(`${j}  ${d.fmt(th[j], 3).padStart(7)}`, 486, 124 + j * 22, { c: Math.abs(th[j]) > 2.3 ? c.red : c.ink });
      }
      ctx.globalAlpha = 1;
      const note = Math.abs(f) > 1 ? "diverges" : f < 0 ? "zig-zag" : f < 0.5 ? "fast" : "slow";
      if (s.n > 0.5) d.text(note, 486, 258, { c: Math.abs(f) > 1 ? c.red : c.muted, s: 13, w: 700 });
    },
  });
})();
