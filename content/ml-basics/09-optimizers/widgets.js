// Unit 09: optimizers on unit 04's stretched bowl.
// Widget 1 races gradient descent, momentum and Adam on a bowl whose stretch and tilt you set.
// Widget 2 runs mini-batch SGD on unit 04's four points under three learning-rate schedules.
(function () {
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  // Unit 04's line fit: points (0, 1), (1, 3), (2, 4), (3, 7). Its loss is a bowl with its
  // bottom at w = 1.9, b = 0.9 and constant second derivatives H = [[7, 3], [3, 2]].
  const STAR = [1.9, 0.9], START = [-1, 3], LOWEST = 0.175;
  const STEEP = 4.5 + Math.sqrt(15.25);     // curvatures of H (its eigenvalues): 8.41 and 0.59
  const FLAT = 4.5 - Math.sqrt(15.25);
  const TILT = Math.atan2(STEEP - 7, 3) * 180 / Math.PI; // steep axis vs the w axis: 25.1°
  const LEVELS = [0.05, 0.2, 0.8, 3.2, 12.8];             // contour heights above the lowest loss

  // H = R diag(steep, flat) R^T, with R rotating the w axis onto the steep axis.
  function bowl(steep, flat, tiltDeg) {
    const a = tiltDeg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
    return {
      steep, flat, tilt: tiltDeg,
      h11: steep * c * c + flat * s * s, h12: (steep - flat) * c * s, h22: steep * s * s + flat * c * c,
    };
  }
  const grad = (H, d) => [H.h11 * d[0] + H.h12 * d[1], H.h12 * d[0] + H.h22 * d[1]];
  const excess = (H, d) => 0.5 * (d[0] * (H.h11 * d[0] + H.h12 * d[1]) + d[1] * (H.h12 * d[0] + H.h22 * d[1]));
  const big = (v) => (!isFinite(v) ? "∞" : Math.abs(v) < 1e4 ? fmt(v, 4) : v.toExponential(1));

  // Contour ellipses, the green cross at the bottom, and faint w and b axes.
  function drawBowl(group, H, PX, PY) {
    while (group.firstChild) group.removeChild(group.firstChild);
    const cx = PX(STAR[0]), cy = PY(STAR[1]), scale = PX(1) - PX(0);
    const contours = svg("g", { class: "muted" }, group);
    for (const c of LEVELS) {
      svg("ellipse", {
        cx, cy,
        rx: Math.min(1e5, Math.sqrt(2 * c / H.steep) * scale), ry: Math.min(1e5, Math.sqrt(2 * c / H.flat) * scale),
        transform: `rotate(${-H.tilt} ${cx} ${cy})`, // screen y points down
        fill: "none", stroke: "currentColor", "stroke-width": 1,
      }, contours);
    }
    const best = svg("g", { class: "green" }, group);
    svg("line", { x1: cx - 6, y1: cy, x2: cx + 6, y2: cy }, best);
    svg("line", { x1: cx, y1: cy - 6, x2: cx, y2: cy + 6 }, best);
  }

  function axes(root, clip, PX, PY, x0, x1, height) {
    svg("line", { x1: x0, y1: PY(0), x2: x1, y2: PY(0), class: "muted faint thin" }, clip);
    svg("line", { x1: PX(0), y1: 0, x2: PX(0), y2: height, class: "muted faint thin" }, clip);
    svg("text", { x: x1 - 14, y: PY(0) + 18, class: "muted", text: "w" }, root);
    svg("text", { x: PX(0) + 8, y: 16, class: "muted", text: "b" }, root);
  }

  function clipped(root, id, x, width, height) {
    const defs = svg("defs", {}, root);
    const clip = svg("clipPath", { id }, defs);
    svg("rect", { x, y: 0, width, height }, clip);
    return svg("g", { "clip-path": `url(#${id})` }, root);
  }

  function screenToStage(root, event) {
    const ctm = root.getScreenCTM();
    return ctm ? new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse()) : null;
  }

  // Play/Pause plus Step, wired the same way as unit 04's widget.
  function playControls(btns, player, onStep) {
    const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
      if (player.playing) player.pause(); else player.play();
      toggle.textContent = player.playing ? "Pause" : "Play";
    });
    button(btns, "Step", () => { player.pause(); toggle.textContent = "Play"; onStep(); });
  }

  // Widget 1: the race ---------------------------------------------------------------------
  (function () {
    const figure = document.getElementById("optimizer-race");
    if (!figure) return;

    const width = 720, rows = 16, height = rows * Learn.ROW;
    const scale = 60, cw = 0.45, cb = 1.95;
    const PX = (w) => width / 2 + (w - cw) * scale;
    const PY = (b) => height / 2 - (b - cb) * scale;
    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "Loss contours over w and b with the paths of gradient descent, momentum and Adam from the same start.");
    const view = clipped(root, "race-clip", 0, width, height);
    axes(root, view, PX, PY, 0, width, height);
    const bowlG = svg("g", {}, view);

    const RUNNERS = [
      { name: "gradient descent", short: "GD", cls: "blue" },
      { name: "momentum", short: "momentum", cls: "orange" },
      { name: "Adam", short: "Adam", cls: "purple" },
    ];
    for (const r of RUNNERS) {
      r.trail = svg("polyline", { class: r.cls, "stroke-width": 1.5, "stroke-linejoin": "round" }, view);
      r.dot = svg("circle", { r: 5, fill: "currentColor", class: r.cls }, view);
    }

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);

    const MAX_STEPS = 600, INTERVAL = 0.04, HOLD = 3;
    const B1 = 0.9, B2 = 0.999, EPS = 1e-8;
    let eta = 0.1, beta = 0.5, stretch = STEEP / FLAT, tilt = TILT;
    let start = START.slice();
    let H, k, wait;

    function reset() {
      H = bowl(STEEP, STEEP / stretch, tilt);
      drawBowl(bowlG, H, PX, PY);
      k = 0;
      wait = 0;
      for (const r of RUNNERS) {
        r.d = [start[0] - STAR[0], start[1] - STAR[1]];
        r.v = [0, 0]; r.m = [0, 0]; r.s = [0, 0];
        r.points = [[start[0], start[1]]];
        r.status = "";
        r.L = excess(H, r.d);
      }
      draw();
    }

    function update(r, g) {
      if (r.name === "gradient descent") {
        for (let i = 0; i < 2; i++) r.d[i] -= eta * g[i];
      } else if (r.name === "momentum") {
        for (let i = 0; i < 2; i++) { r.v[i] = beta * r.v[i] + g[i]; r.d[i] -= eta * r.v[i]; }
      } else {
        for (let i = 0; i < 2; i++) {
          r.m[i] = B1 * r.m[i] + (1 - B1) * g[i];
          r.s[i] = B2 * r.s[i] + (1 - B2) * g[i] * g[i];
          const mHat = r.m[i] / (1 - B1 ** k), sHat = r.s[i] / (1 - B2 ** k);
          r.d[i] -= eta * mHat / (Math.sqrt(sHat) + EPS);
        }
      }
    }

    const done = () => RUNNERS.every((r) => r.status) || k >= MAX_STEPS;

    function step() {
      if (done()) return;
      k += 1;
      for (const r of RUNNERS) {
        if (r.status) continue;
        update(r, grad(H, r.d));
        r.L = excess(H, r.d);
        const w = r.d[0] + STAR[0], b = r.d[1] + STAR[1];
        if (Math.abs(w) < 100 && Math.abs(b) < 100) r.points.push([w, b]);
        if (!isFinite(r.L) || r.L > 1e4) r.status = "diverged";
        else if (r.L < 1e-4) r.status = `at the bottom, ${k} steps`;
      }
      draw();
    }

    function draw() {
      for (const r of RUNNERS) {
        r.trail.setAttribute("points", r.points.map(([w, b]) => `${PX(w)},${PY(b)}`).join(" "));
        const [w, b] = r.points[r.points.length - 1];
        r.dot.setAttribute("cx", PX(w));
        r.dot.setAttribute("cy", PY(b));
        r.dot.style.display = r.status === "diverged" ? "none" : "";
      }
      const line = (r) => {
        const state = r.status
          ? `<span class="${r.status === "diverged" ? "orange" : "green"}">${r.status}</span>`
          : big(r.L);
        return `<span class="${r.cls}">${r.short.padEnd(10)}</span>${state}`;
      };
      out.innerHTML =
        `η = ${fmt(eta, 3)}   β = ${fmt(beta, 2)}   step ${k}\n` +
        `stretch ${Math.round(stretch)}×   tilt ${Math.round(tilt)}°\n` +
        `loss above the lowest:\n` +
        RUNNERS.map(line).join("\n");
    }

    slider(ctl, { label: "η", min: 0.01, max: 0.23, step: 0.005, value: eta, format: (v) => v.toFixed(3) }, (v) => { eta = v; reset(); });
    slider(ctl, { label: "β", min: 0, max: 0.95, step: 0.05, value: beta, format: (v) => v.toFixed(2) }, (v) => { beta = v; reset(); });
    // Stretch on a log scale, from a round bowl (1×) to 1000×. "any" so the presets land exactly.
    const stretchC = slider(ctl, { label: "stretch", min: 0, max: 3, step: "any", value: Math.log10(stretch), format: (v) => `${Math.round(10 ** v)}×` }, (v) => { stretch = 10 ** v; reset(); });
    const tiltC = slider(ctl, { label: "tilt", min: 0, max: 90, step: "any", value: tilt, format: (v) => `${Math.round(v)}°` }, (v) => { tilt = v; reset(); });

    let acc = 0;
    const player = loop(figure, (dt) => {
      if (!dt) return;
      acc += dt;
      if (done()) {
        wait += dt;
        if (wait > HOLD) { reset(); acc = 0; }
        return;
      }
      while (acc > INTERVAL && !done()) { acc -= INTERVAL; step(); }
    });

    playControls(btns, player, () => { if (done()) reset(); else step(); });
    button(btns, "Restart", () => { start = START.slice(); reset(); });
    const preset = (label, s, t) => button(btns, label, () => {
      stretchC.show(Math.log10(s)); tiltC.show(t);
      stretch = s; tilt = t;
      reset();
    });
    preset("unit 04's bowl", STEEP / FLAT, TILT);
    preset("1000× along w", 1000, 0);
    preset("1000× tilted 45°", 1000, 45);

    root.addEventListener("click", (event) => {
      const p = screenToStage(root, event);
      if (!p) return;
      start = [cw + (p.x - width / 2) / scale, cb - (p.y - height / 2) / scale];
      reset();
    });

    reset();
  })();

  // Widget 2: mini-batch noise and schedules ----------------------------------------------
  (function () {
    const figure = document.getElementById("sgd-schedule");
    if (!figure) return;

    const xs = [0, 1, 2, 3], ys = [1, 3, 4, 7];
    const lossAll = (w, b) => xs.reduce((s, x, i) => s + (w * x + b - ys[i]) ** 2, 0) / xs.length;

    const width = 720, rows = 14, height = rows * Learn.ROW;
    const panelW = 460, scale = 60, cw = 0.45, cb = 1.95;
    const PX = (w) => panelW / 2 + (w - cw) * scale;
    const PY = (b) => height / 2 - (b - cb) * scale;
    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "Left: mini-batch gradient descent on unit 04's loss bowl. Right: the learning rate over the run.");
    const view = clipped(root, "sgd-clip", 0, panelW, height);
    axes(root, view, PX, PY, 0, panelW, height);
    drawBowl(svg("g", {}, view), bowl(STEEP, FLAT, TILT), PX, PY);
    const trail = svg("polyline", { class: "blue", "stroke-width": 1.2, "stroke-linejoin": "round" }, view);
    const dot = svg("circle", { r: 5, fill: "currentColor", class: "orange" }, view);

    // Right: η over the run, scaled to the peak.
    const T = 200, WARMUP = 20;
    const sx0 = 500, sx1 = 700, sy0 = height - 40, sy1 = 40;
    const SX = (t) => sx0 + (sx1 - sx0) * t / T;
    const SY = (f) => sy0 - (sy0 - sy1) * f;
    svg("line", { x1: panelW + 10, y1: 0, x2: panelW + 10, y2: height, class: "muted faint thin" }, root);
    svg("line", { x1: sx0, y1: sy0, x2: sx1, y2: sy0, class: "muted thin" }, root);
    svg("line", { x1: sx0, y1: sy0, x2: sx0, y2: sy1 - 10, class: "muted thin" }, root);
    svg("text", { x: sx0 - 6, y: sy1 - 16, class: "muted", text: "η" }, root);
    svg("text", { x: sx1, y: sy0 + 20, class: "muted", "text-anchor": "end", text: "step 200" }, root);
    svg("text", { x: sx0, y: sy0 + 20, class: "muted", "text-anchor": "middle", text: "0" }, root);
    const curve = svg("polyline", { class: "purple", "stroke-width": 1.5 }, root);
    const marker = svg("circle", { r: 4.5, fill: "currentColor", class: "orange" }, root);

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);

    const SCHEDULES = {
      constant: () => 1,
      cosine: (t) => 0.5 * (1 + Math.cos(Math.PI * t / T)),
      "warmup + cosine": (t) => (t < WARMUP ? (t + 1) / WARMUP : 0.5 * (1 + Math.cos(Math.PI * (t - WARMUP) / (T - WARMUP)))),
    };
    let schedule = "constant", etaMax = 0.05, batch = 1;
    let w, b, t, points, wait, finals = [];

    function drawCurve() {
      const f = SCHEDULES[schedule];
      const pts = [];
      for (let s = 0; s <= T; s++) pts.push(`${SX(s)},${SY(f(Math.min(s, T - 1)))}`);
      curve.setAttribute("points", pts.join(" "));
    }

    function reset(keepHistory) {
      [w, b] = START;
      t = 0;
      wait = 0;
      points = [[w, b]];
      if (!keepHistory) finals = [];
      drawCurve();
      draw();
    }

    // One step on a random batch of `batch` distinct points out of the four.
    function step() {
      if (t >= T) return;
      const idx = [0, 1, 2, 3];
      for (let i = 3; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      let gw = 0, gb = 0;
      for (const i of idx.slice(0, batch)) { const e = w * xs[i] + b - ys[i]; gw += 2 * e * xs[i]; gb += 2 * e; }
      const lr = etaMax * SCHEDULES[schedule](t);
      w -= lr * gw / batch;
      b -= lr * gb / batch;
      t += 1;
      if (Math.abs(w) < 100 && Math.abs(b) < 100) points.push([w, b]);
      if (t === T) finals.push(lossAll(w, b) - LOWEST);
      draw();
    }

    function draw() {
      trail.setAttribute("points", points.map(([pw, pb]) => `${PX(pw)},${PY(pb)}`).join(" "));
      const [pw, pb] = points[points.length - 1];
      dot.setAttribute("cx", PX(pw)); dot.setAttribute("cy", PY(pb));
      const f = SCHEDULES[schedule](Math.min(t, T - 1));
      marker.setAttribute("cx", SX(t)); marker.setAttribute("cy", SY(f));
      const recent = finals.slice(-10);
      const mean = recent.length ? recent.reduce((s, v) => s + v, 0) / recent.length : 0;
      out.innerHTML =
        `${schedule}, batch ${batch}\n` +
        `step ${t} of ${T}, η = ${fmt(etaMax * f, 4)}\n` +
        `<span class="orange">loss ${big(lossAll(w, b))}, ${big(lossAll(w, b) - LOWEST)} above lowest</span>\n` +
        (recent.length
          ? `last ${recent.length} run${recent.length > 1 ? "s" : ""} ended on average\n<span class="blue">${big(mean)}</span> above lowest`
          : "no finished runs yet\n ");
    }

    slider(ctl, { label: "batch", min: 1, max: 4, step: 1, value: batch }, (v) => { batch = v; reset(); });
    slider(ctl, { label: "peak η", min: 0.01, max: 0.1, step: 0.005, value: etaMax, format: (v) => v.toFixed(3) }, (v) => { etaMax = v; reset(); });

    const INTERVAL = 1 / 30, HOLD = 1.5;
    let acc = 0;
    const player = loop(figure, (dt) => {
      if (!dt) return;
      acc += dt;
      if (t >= T) {
        wait += dt;
        if (wait > HOLD) { reset(true); acc = 0; }
        return;
      }
      while (acc > INTERVAL && t < T) { acc -= INTERVAL; step(); }
    });
    playControls(btns, player, () => { if (t >= T) reset(true); else step(); });
    for (const name of Object.keys(SCHEDULES)) {
      button(btns, name, () => { schedule = name; reset(); });
    }

    reset();
  })();
})();

// Step-through: the four single-point gradients against their average, at the start and at the bottom.
(function () {
  const xs = [0, 1, 2, 3], ys = [1, 3, 4, 7];
  const START = [-1, 3], BEST = [1.9, 0.9], OY = 116;
  Learn.demo(document.getElementById("demo-noise"), {
    height: 340,
    label: "Four arrows, one gradient per data point, and a thicker arrow for their average, first at the starting line and then at the best line",
    steps: [
      { s: { n: 1, avg: 0, pos: 0, zoom: 9, ox: 500 }, cap: "Start at w = −1, b = 3. Each arrow is the gradient from one data point. Point 1 alone gives (0, 4). A step goes against the arrow, so point 1 says: lower b." },
      { s: { n: 4 }, cap: "The other three points give (−2, −2), (−12, −6) and (−42, −14). Four points, four different arrows." },
      { s: { avg: 1 }, cap: "Their average is the full gradient, (−14, −4.5). A batch of one steps along a single blue arrow. Any one of them can be far off. Averaged over many steps, they add up to the orange one." },
      { s: { pos: 1, zoom: 60, ox: 320 }, cap: "Now move to the best line, w = 1.9, b = 0.9, and zoom in. The average is exactly (0, 0). But single arrows as long as about 3 remain, so a batch of one keeps getting kicked off the bottom." },
    ],
    draw(ctx, s, c, d) {
      const w = START[0] + (BEST[0] - START[0]) * s.pos, b = START[1] + (BEST[1] - START[1]) * s.pos;
      const grads = xs.map((x, i) => { const e = w * x + b - ys[i]; return [2 * e * x, 2 * e]; });
      const mean = [0, 1].map((k) => grads.reduce((a, g) => a + g[k], 0) / 4);
      const PX = (v) => s.ox + v * s.zoom, PY = (v) => OY - v * s.zoom;
      d.line(20, OY, 620, OY, c.line, 1.5);
      d.line(s.ox, 16, s.ox, 250, c.line, 1.5);
      d.text("w part", 616, OY + 14, { c: c.muted, s: 12, a: "right" });
      d.text("b part", s.ox + 8, 22, { c: c.muted, s: 12 });
      const unit = s.pos < 0.5 ? 10 : 1;
      d.line(24, 42, 24 + unit * s.zoom, 42, c.muted, 2);
      d.line(24, 37, 24, 47, c.muted, 2);
      d.line(24 + unit * s.zoom, 37, 24 + unit * s.zoom, 47, c.muted, 2);
      d.text(`length ${unit}`, 24, 24, { c: c.muted, s: 12 });
      // Mid-zoom the arrows overshoot; keep them out of the readout below.
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, 640, 256); ctx.clip();
      grads.forEach((g, i) => {
        const alpha = d.clamp(s.n - i, 0, 1);
        if (alpha < 0.01) return;
        ctx.globalAlpha = alpha;
        d.arrow(PX(0), PY(0), PX(g[0]), PY(g[1]), c.blue, 3);
        const len = Math.hypot(g[0], g[1]) || 1, off = 14;
        d.text(String(i + 1), d.clamp(PX(g[0]) + g[0] / len * off, 12, 628), d.clamp(PY(g[1]) - g[1] / len * off, 14, 246), { c: c.blue, s: 13, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      });
      if (s.avg > 0.01) {
        ctx.globalAlpha = s.avg;
        d.arrow(PX(0), PY(0), PX(mean[0]), PY(mean[1]), c.orange, 5);
        d.dot(PX(mean[0]), PY(mean[1]), 5, c.orange);
        ctx.globalAlpha = 1;
      }
      d.dot(PX(0), PY(0), 4, c.ink);
      ctx.restore();
      d.panel(c, 20, 262, 600, 66);
      const xsLabel = [34, 150, 280, 420];
      grads.forEach((g, i) => {
        ctx.globalAlpha = d.clamp(s.n - i, 0, 1);
        d.text(`${i + 1}: (${d.fmt(g[0], 1)}, ${d.fmt(g[1], 1)})`, xsLabel[i], 282, { c: c.blue, s: 13, w: 700 });
      });
      ctx.globalAlpha = s.avg;
      d.text(`average = full gradient: (${d.fmt(mean[0], 1)}, ${d.fmt(mean[1], 1)})`, 34, 308, { c: c.orange, s: 13, w: 700 });
      ctx.globalAlpha = 1;
      d.text(`w = ${d.fmt(w, 1)}, b = ${d.fmt(b, 1)}`, 606, 308, { c: c.muted, s: 13, a: "right" });
    },
  });
})();

// Step-through: momentum's velocity when the gradient is steady, then when it flips every step.
(function () {
  const BETA = 0.5, BASE = 176, U = 56;
  const X = (k) => 96 + (k - 1) * 78;
  Learn.demo(document.getElementById("demo-momentum"), {
    height: 320,
    label: "Bars for the gradient and the momentum velocity over six steps, first with a steady gradient and then with one that flips sign",
    steps: [
      { s: { t: 1, flip: 0 }, cap: "Momentum keeps a velocity v. Each step it halves the old v (β = 0.5) and adds the new gradient. Here the gradient is 1 on every step. After step 1, v = 1." },
      { s: { t: 2 }, cap: "Step 2: half of the old velocity plus the new gradient. 0.5 × 1 + 1 = 1.5." },
      { s: { t: 6 }, cap: "Keep going: 1.75, 1.88, 1.94, 1.97. The velocity levels off at 2, which is 1 ÷ (1 − 0.5). A steady gradient doubles the step." },
      { s: { t: 1, flip: 1 }, cap: "Now the gradient flips sign on every step: +1, −1, +1, −1. After step 1, v = 1 again." },
      { s: { t: 2 }, cap: "Step 2: 0.5 × 1 + (−1) = −0.5. The leftover velocity cancels half of the new gradient." },
      { s: { t: 6 }, cap: "Keep going: 0.75, −0.63, 0.69, −0.66. The size settles at 0.67, which is 1 ÷ (1 + 0.5). A flipping gradient shrinks the step." },
    ],
    draw(ctx, s, c, d) {
      const flip = s.flip > 0.5;
      d.rect(24, 18, 14, 14, c.blue); d.text("gradient g", 46, 25, { c: c.blue, s: 13, w: 700 });
      d.rect(160, 18, 14, 14, c.orange); d.text("velocity v", 182, 25, { c: c.orange, s: 13, w: 700 });
      d.text(flip ? "gradient flips: +1, −1, +1, ..." : "gradient steady: 1, 1, 1, ...", 616, 25, { c: c.ink, s: 13, a: "right", w: 700 });
      d.line(60, BASE, 560, BASE, c.muted, 1.5);
      const limit = flip ? 1 / (1 + BETA) : 1 / (1 - BETA);
      for (const sign of flip ? [1, -1] : [1]) d.line(60, BASE - sign * limit * U, 560, BASE - sign * limit * U, c.muted, 1.5, [6, 5]);
      d.text(flip ? "±0.67" : "2", 568, BASE - limit * U, { c: c.muted, s: 13, w: 700 });
      d.text("0", 568, BASE, { c: c.muted, s: 13 });
      let v = 0;
      for (let k = 1; k <= 6; k++) {
        const g = flip && k % 2 === 0 ? -1 : 1;
        v = BETA * v + g;
        d.text(`step ${k}`, X(k) + 26, 256, { c: c.muted, s: 12, a: "center" });
        const alpha = d.clamp(s.t - k + 1, 0, 1);
        if (alpha < 0.01) continue;
        ctx.globalAlpha = alpha;
        d.rect(X(k), Math.min(BASE, BASE - g * U), 22, U, c.blue);
        d.rect(X(k) + 30, Math.min(BASE, BASE - v * U), 22, Math.abs(v) * U, c.orange);
        d.text(d.fmt(v, 2), X(k) + 41, v > 0 ? BASE - v * U - 12 : BASE - v * U + 12, { c: c.orange, s: 13, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      }
      d.panel(c, 24, 276, 592, 34);
      d.text("each step:  v ← 0.5 × v + g", 38, 293, { c: c.ink, s: 13, w: 700 });
      d.text(flip ? "settles at 1 ÷ (1 + 0.5) = 0.67" : "settles at 1 ÷ (1 − 0.5) = 2", 602, 293, { c: c.muted, s: 13, a: "right" });
    },
  });
})();

// Step-through: Adam's first step on unit 4's line, one quantity at a time.
(function () {
  const g = [-14, -4.5], ETA = 0.1, B1 = 0.9, B2 = 0.999;
  const m = g.map((v) => (1 - B1) * v), sq = g.map((v) => (1 - B2) * v * v);
  const mh = m.map((v) => v / (1 - B1)), sh = sq.map((v) => v / (1 - B2));
  const ratio = mh.map((v, i) => v / Math.sqrt(sh[i]));
  const rows = [
    ["gradient g", g, 1],
    ["GD step: −η × g", g.map((v) => -ETA * v), 2],
    ["m = 0.1 × g", m, 2],
    ["s = 0.001 × g²", sq, 5],
    ["m̂ = m ÷ 0.1", mh, 1],
    ["ŝ = s ÷ 0.001", sh, 2],
    ["m̂ ÷ √ŝ", ratio, 2],
    ["Adam step: −η × that", ratio.map((v) => -ETA * v), 2],
  ];
  const RY = (i) => 62 + i * 31;
  Learn.demo(document.getElementById("demo-adam"), {
    height: 330,
    label: "A table that builds Adam's first step for w and b from the gradient, next to bars comparing it with gradient descent's step",
    steps: [
      { s: { r: 2, h0: 0, h1: 1, adam: 0 }, cap: "The start of unit 4's line. The gradient is −14 for w and −4.5 for b. Plain gradient descent with η = 0.1 moves w by 1.4 and b by 0.45." },
      { s: { r: 3, h0: 2, h1: 2 }, cap: "Adam keeps m, a running average of the gradient. It starts at 0, so after one step m = 0.1 × g. That is −1.4 for w and −0.45 for b." },
      { s: { r: 4, h0: 3, h1: 3 }, cap: "It also keeps s, a running average of the squared gradient. After one step s = 0.001 × g², which is 0.196 for w and 0.02025 for b." },
      { s: { r: 6, h0: 4, h1: 5 }, cap: "Both averages started at zero, so both are too small. Bias correction divides m by 0.1 and s by 0.001. That gives m̂ = −14 and −4.5, and ŝ = 196 and 20.25." },
      { s: { r: 7, h0: 6, h1: 6 }, cap: "Divide m̂ by the square root of ŝ. For w: −14 ÷ 14 = −1. For b: −4.5 ÷ 4.5 = −1. The size of the gradient cancels. Only its sign is left." },
      { s: { r: 8, h0: 7, h1: 7, adam: 1 }, cap: "Adam's step is −η times that: 0.1 for w and 0.1 for b. Gradient descent moved w 14 times as far and b 4.5 times as far. Adam moves every parameter the same distance." },
    ],
    draw(ctx, s, c, d) {
      d.text("w", 300, 30, { c: c.muted, s: 13, a: "right", w: 700 });
      d.text("b", 400, 30, { c: c.muted, s: 13, a: "right", w: 700 });
      d.line(20, 44, 410, 44, c.muted, 1.5);
      d.rect(20, RY(s.h0) - 15, 390, RY(s.h1) - RY(s.h0) + 30, c.soft);
      rows.forEach(([name, vals, digits], i) => {
        const alpha = d.clamp(s.r - i, 0, 1);
        if (alpha < 0.01) return;
        ctx.globalAlpha = alpha;
        const color = i === 1 ? c.blue : i === 7 ? c.orange : c.ink;
        d.text(name, 30, RY(i), { c: i === 1 || i === 7 ? color : c.muted, s: 13, w: 700 });
        d.text(d.fmt(vals[0], digits), 300, RY(i), { c: color, s: 14, a: "right", w: 700 });
        d.text(d.fmt(vals[1], digits), 400, RY(i), { c: color, s: 14, a: "right", w: 700 });
        ctx.globalAlpha = 1;
      });
      // Bars: how far each parameter moves on the first step.
      const BX = 470, U = 90;
      const bar = (y, label, v, color) => {
        d.text(label, BX - 10, y, { c: c.muted, s: 13, a: "right", w: 700 });
        d.rect(BX, y - 9, v * U, 18, color);
        d.text(d.fmt(v, 2), BX + v * U + 8, y, { c: color, s: 13, w: 700 });
      };
      d.text("first step, gradient", 440, 62, { c: c.blue, s: 13, w: 700 });
      d.text("descent", 440, 80, { c: c.blue, s: 13, w: 700 });
      bar(108, "w", -ETA * g[0], c.blue);
      bar(136, "b", -ETA * g[1], c.blue);
      ctx.globalAlpha = s.adam;
      d.text("first step, Adam", 440, 196, { c: c.orange, s: 13, w: 700 });
      bar(224, "w", -ETA * ratio[0], c.orange);
      bar(252, "b", -ETA * ratio[1], c.orange);
      ctx.globalAlpha = 1;
      d.line(BX, 96, BX, 264, c.muted, 1.5);
    },
  });
})();

// Step-through: warmup then cosine decay over a 200-step run.
(function () {
  const T = 200, W = 20, MAX = 0.05, MIN = 0.005;
  const eta = (t) => (t <= W ? MAX * t / W : MIN + 0.5 * (MAX - MIN) * (1 + Math.cos(Math.PI * (t - W) / (T - W))));
  const X = (t) => 80 + t * 2.6, Y = (v) => 250 - v * 4000;
  Learn.demo(document.getElementById("demo-schedule"), {
    height: 320,
    label: "The learning rate over a 200-step run: a straight climb for 20 steps, then a cosine-shaped fall to a floor",
    steps: [
      { s: { t: 0 }, cap: "An example run: 200 steps, 20 of them warmup, a peak learning rate of 0.05 and a floor of 0.005 (10% of the peak). At step 0 the learning rate is 0." },
      { s: { t: 10 }, cap: "Step 10 is halfway through warmup, so the learning rate is half the peak: 0.05 × 10 ÷ 20 = 0.025." },
      { s: { t: 20 }, cap: "Step 20: warmup ends and the learning rate is at its peak, 0.05. From here the cosine formula takes over." },
      { s: { t: 110 }, cap: "Step 110 is halfway through the decay. The cosine is 0 there, so the learning rate sits halfway between peak and floor: 0.0275. This is where it falls fastest." },
      { s: { t: 200 }, cap: "Step 200: the curve levels off at the floor, 0.005. Small steps at the end let the parameters settle instead of bouncing." },
    ],
    draw(ctx, s, c, d) {
      d.line(X(0), Y(0), X(T), Y(0), c.muted, 1.5);
      d.line(X(0), Y(0), X(0), Y(0.056), c.muted, 1.5);
      for (const t of [0, 20, 110, 200]) {
        d.line(X(t), Y(0) - 4, X(t), Y(0) + 4, c.muted, 1.5);
        d.text(String(t), X(t), Y(0) + 18, { c: c.muted, s: 12, a: "center" });
      }
      d.text("step t", X(T), Y(0) + 42, { c: c.muted, s: 12, a: "right" });
      for (const [v, name] of [[MAX, "peak 0.05"], [MIN, "floor 0.005"]]) {
        d.line(X(0), Y(v), X(T), Y(v), c.line, 1.5, [5, 5]);
        d.text(name, X(T) + 30, Y(v) - 16, { c: c.muted, s: 12, a: "right" });
      }
      d.text("η", X(0) - 14, Y(0.056) + 4, { c: c.muted, s: 13, a: "center", w: 700 });
      ctx.globalAlpha = 0.35;
      d.rect(X(0), Y(0.056), X(W) - X(0), Y(0) - Y(0.056), c.soft);
      ctx.globalAlpha = 1;
      d.text("warmup", X(W) + 6, Y(0) - 12, { c: c.muted, s: 12 });
      ctx.strokeStyle = c.blue; ctx.lineWidth = 3.5; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(X(0), Y(0));
      for (let t = 1; t <= s.t; t++) ctx.lineTo(X(t), Y(eta(t)));
      ctx.lineTo(X(s.t), Y(eta(s.t)));
      ctx.stroke();
      const px = X(s.t), py = Y(eta(s.t));
      d.line(px, Y(0), px, py, c.orange, 1.5, [3, 4]);
      d.dot(px, py, 7, c.orange);
      d.panel(c, 330, 16, 220, 34);
      d.text(`step ${Math.round(s.t)}   η = ${eta(s.t).toFixed(4)}`, 344, 33, { c: c.ink, s: 13, w: 700 });
    },
  });
})();
