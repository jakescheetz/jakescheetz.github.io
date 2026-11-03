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
