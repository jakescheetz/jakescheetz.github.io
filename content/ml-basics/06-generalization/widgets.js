// Unit 06: fitting polynomials to ten noisy points.
// Widget 1 (#capacity): degree and L2-penalty sliders, the fit, and training vs validation
// error for every degree. Widget 2 (#early-stopping): gradient descent on the degree-9 fit,
// with training and validation loss plotted against the step number.
(function () {
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  // Data from the page's worked example: the truth is sin(pi x), plus noise with std 0.25.
  const xTrain = [-1, -0.78, -0.56, -0.33, -0.11, 0.11, 0.33, 0.56, 0.78, 1];
  const yTrain = [-0.23, -0.28, -1.19, -1.31, -0.44, 0.75, 0.64, 1.24, 0.51, -0.21];
  const xVal = [0.63, -0.63, -0.73, -0.08, 0.88, 0.2, -0.16, -0.69, 0.12, 0.32, -0.09, -0.98, -0.92, 0.88, -0.35, -0.05, 0.68, -0.91, 0.94, 0.22];
  const yVal = [1.22, -0.86, -0.65, -0.3, 0.22, 0.36, -0.46, -0.88, 0.46, 0.88, -0.2, -0.6, -0.08, 0.1, -0.61, 0.3, 0.94, -0.74, 0.41, 0.96];
  const xMore = [0.87, 0.31, 0.8, 0.08, 0.19, 0.79, 0.75, 0.49, -0.55, 0.89, 0.81, -0.71, 0.76, -0.61, 0.09, 0.21, -0.44, 0.05, 0.86, -0.45, -0.88, 0.15, 0.14, -0.37, 0.41, 0.19, -0.14, -0.29, 0.64, 0.46];
  const yMore = [0.02, 0.54, 0.75, 0.17, 0.66, 0.72, 0.62, 1.24, -0.91, 0.11, 0.72, -1.18, 0.57, -0.41, 0.1, 0.39, -1.13, 0.43, 0.49, -1.1, -0.57, 0.32, 0.85, -1.04, 0.97, 0.56, -0.24, -0.79, 0.79, 1.03];
  const NOISE = 0.0625; // 0.25 squared: the validation MSE even the true curve gets, on average

  // Math ----------------------------------------------------------------------------------

  const powers = (x, k) => { const row = [1]; for (let j = 1; j <= k; j++) row.push(row[j - 1] * x); return row; };
  const evalPoly = (w, x) => { let s = 0; for (let j = w.length - 1; j >= 0; j--) s = s * x + w[j]; return s; };
  const mse = (w, xs, ys) => xs.reduce((s, x, i) => s + (evalPoly(w, x) - ys[i]) ** 2, 0) / xs.length;

  // Degree-k polynomial minimizing MSE + (lam/2)(w_1^2 + ... + w_k^2), like the page's fit():
  // least squares on the data rows plus k penalty rows, solved with Householder QR.
  function fit(xs, ys, k, lam) {
    const n = k + 1, c = Math.sqrt(lam * xs.length / 2);
    const A = xs.map((x) => powers(x, k));
    const b = ys.slice();
    for (let j = 1; j <= k; j++) { const row = new Array(n).fill(0); row[j] = c; A.push(row); b.push(0); }
    const m = A.length;
    for (let j = 0; j < n; j++) {
      let norm = 0;
      for (let i = j; i < m; i++) norm += A[i][j] ** 2;
      norm = Math.sqrt(norm);
      if (norm === 0) continue;
      const alpha = A[j][j] > 0 ? -norm : norm;
      const v = new Array(m).fill(0);
      for (let i = j; i < m; i++) v[i] = A[i][j];
      v[j] -= alpha;
      let vv = 0;
      for (let i = j; i < m; i++) vv += v[i] * v[i];
      if (vv === 0) continue;
      for (let col = j; col < n; col++) {
        let s = 0;
        for (let i = j; i < m; i++) s += v[i] * A[i][col];
        s = 2 * s / vv;
        for (let i = j; i < m; i++) A[i][col] -= s * v[i];
      }
      let s = 0;
      for (let i = j; i < m; i++) s += v[i] * b[i];
      s = 2 * s / vv;
      for (let i = j; i < m; i++) b[i] -= s * v[i];
    }
    const w = new Array(n).fill(0);
    for (let j = n - 1; j >= 0; j--) {
      let s = b[j];
      for (let col = j + 1; col < n; col++) s -= A[j][col] * w[col];
      w[j] = s / A[j][j];
    }
    return w;
  }

  // Drawing helpers shared by both widgets -------------------------------------------------

  const width = 720, rows = 18, height = rows * Learn.ROW;
  const DX = (x) => 205 + x * 170;           // data panel: x in [-1.05, 1.05]
  const DY = (y) => height / 2 - y * 72;     // y in about [-2.3, 2.3]
  const PANEL = 430;                         // the right panel starts here
  const ETOP = 16, EBOT = height - 40;       // error axis, log scale from 2 down to 0.005
  const LOG_HI = Math.log10(2), LOG_LO = Math.log10(0.005);
  const EY = (m) => ETOP + (LOG_HI - Math.log10(Math.max(m, 0.005))) / (LOG_HI - LOG_LO) * (EBOT - ETOP);
  const PX0 = 482, PX1 = 704;                // horizontal extent of the error plot

  let clipCount = 0;
  function dataPanel(root) {
    const id = `u06-clip-${++clipCount}`;
    const clip = svg("clipPath", { id }, svg("defs", {}, root));
    svg("rect", { x: 0, y: 0, width: PANEL - 20, height }, clip);
    const g = svg("g", { "clip-path": `url(#${id})` }, root);
    svg("line", { x1: 0, y1: DY(0), x2: PANEL - 20, y2: DY(0), class: "muted faint thin" }, g);
    svg("line", { x1: DX(0), y1: 0, x2: DX(0), y2: height, class: "muted faint thin" }, g);
    svg("text", { x: PANEL - 36, y: DY(0) + 18, class: "muted", text: "x" }, root);
    svg("text", { x: DX(0) + 8, y: 16, class: "muted", text: "y" }, root);
    svg("line", { x1: PANEL, y1: 0, x2: PANEL, y2: height, class: "muted faint thin" }, root);
    const truth = [];
    for (let i = 0; i <= 100; i++) { const x = -1.05 + 2.1 * i / 100; truth.push(`${DX(x)},${DY(Math.sin(Math.PI * x))}`); }
    svg("polyline", { points: truth.join(" "), class: "muted dashed thin" }, g);
    const curve = svg("polyline", { class: "ink", "stroke-width": 2, "stroke-linejoin": "round" }, g);
    const valDots = svg("g", { class: "orange" }, g);
    xVal.forEach((x, i) => svg("circle", { cx: DX(x), cy: DY(yVal[i]), r: 3.5, fill: "none", stroke: "currentColor", "stroke-width": 1.5 }, valDots));
    const trainDots = svg("g", { class: "blue" }, g);
    const moreDots = svg("g", { class: "blue" }, g);
    xTrain.forEach((x, i) => svg("circle", { cx: DX(x), cy: DY(yTrain[i]), r: 4, fill: "currentColor" }, trainDots));
    xMore.forEach((x, i) => svg("circle", { cx: DX(x), cy: DY(yMore[i]), r: 3, fill: "currentColor" }, moreDots));
    moreDots.style.display = "none";
    return {
      setCurve(w) {
        const pts = [];
        for (let i = 0; i <= 160; i++) {
          const x = -1.05 + 2.1 * i / 160;
          pts.push(`${DX(x)},${DY(Math.max(-4, Math.min(4, evalPoly(w, x))))}`);
        }
        curve.setAttribute("points", pts.join(" "));
      },
      showMore(on) { moreDots.style.display = on ? "" : "none"; },
    };
  }

  function errorAxes(root) {
    for (const [m, label] of [[1, "1"], [0.1, "0.1"], [0.01, "0.01"]]) {
      svg("line", { x1: PX0 - 6, y1: EY(m), x2: PX1, y2: EY(m), class: "muted faint thin" }, root);
      svg("text", { x: PX0 - 10, y: EY(m) + 4, "text-anchor": "end", class: "muted", text: label }, root);
    }
    svg("line", { x1: PX0, y1: EY(NOISE), x2: PX1, y2: EY(NOISE), class: "muted dashed thin" }, root);
    svg("line", { x1: PX0, y1: EBOT, x2: PX1, y2: EBOT, class: "muted thin" }, root);
    svg("text", { x: PX0 - 10, y: ETOP - 2, "text-anchor": "end", class: "muted", text: "MSE" }, root);
  }

  const showLam = (lam) => (lam === 0 ? "0" : lam >= 0.001 ? String(Number(lam.toPrecision(1))) : lam.toExponential(0));

  // Widget 1: degree and penalty ---------------------------------------------------------

  (function () {
    const figure = document.getElementById("capacity");
    if (!figure) return;
    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "Left: noisy training and validation points with a polynomial fit. Right: training and validation error for degrees 0 to 9.");
    const panel = dataPanel(root);
    errorAxes(root);
    const KX = (k) => PX0 + 8 + k * 23;
    for (const k of [0, 3, 6, 9]) svg("text", { x: KX(k), y: EBOT + 18, "text-anchor": "middle", class: "muted", text: String(k) }, root);
    const marker = svg("line", { y1: ETOP, y2: EBOT, class: "muted thin" }, root);
    const trainLine = svg("polyline", { class: "blue", "stroke-width": 1.5 }, root);
    const valLine = svg("polyline", { class: "orange", "stroke-width": 1.5 }, root);
    const trainDot = svg("circle", { r: 4.5, fill: "currentColor", class: "blue" }, root);
    const valDot = svg("circle", { r: 4.5, fill: "currentColor", class: "orange" }, root);

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);
    let degree = 3, lam = 0, more = false;

    function draw() {
      const xs = more ? xTrain.concat(xMore) : xTrain;
      const ys = more ? yTrain.concat(yMore) : yTrain;
      const tr = [], va = [];
      let current;
      for (let k = 0; k <= 9; k++) {
        const w = fit(xs, ys, k, lam);
        tr.push(mse(w, xs, ys));
        va.push(mse(w, xVal, yVal));
        if (k === degree) current = w;
      }
      panel.setCurve(current);
      panel.showMore(more);
      trainLine.setAttribute("points", tr.map((m, k) => `${KX(k)},${EY(m)}`).join(" "));
      valLine.setAttribute("points", va.map((m, k) => `${KX(k)},${EY(m)}`).join(" "));
      marker.setAttribute("x1", KX(degree)); marker.setAttribute("x2", KX(degree));
      trainDot.setAttribute("cx", KX(degree)); trainDot.setAttribute("cy", EY(tr[degree]));
      valDot.setAttribute("cx", KX(degree)); valDot.setAttribute("cy", EY(va[degree]));
      const best = va.indexOf(Math.min(...va));
      const biggest = Math.max(...current.map(Math.abs));
      out.innerHTML =
        `degree ${degree}   λ = ${showLam(lam)}   ${xs.length} training points\n` +
        `<span class="blue">training MSE   ${fmt(tr[degree], 4)}</span>\n` +
        `<span class="orange">validation MSE ${fmt(va[degree], 4)}</span>  (noise 0.0625)\n` +
        `largest |w| ${fmt(biggest, 1)}\n` +
        `lowest validation MSE at degree ${best}`;
    }

    slider(ctl, { label: "degree k", min: 0, max: 9, step: 1, value: degree }, (v) => { degree = v; draw(); });
    // 0 means no penalty; otherwise λ = 10^(-7 + v/4), so the slider spans 1e-7 to 1.
    slider(ctl, { label: "penalty λ", min: 0, max: 28, step: 1, value: 0, format: (v) => showLam(v === 0 ? 0 : 10 ** (-7 + v / 4)) }, (v) => {
      lam = v === 0 ? 0 : 10 ** (-7 + v / 4);
      draw();
    });
    const toggle = button(btns, "Add 30 training points", () => {
      more = !more;
      toggle.textContent = more ? "Back to 10 training points" : "Add 30 training points";
      draw();
    });
    draw();
  })();

  // Widget 2: early stopping ---------------------------------------------------------------

  (function () {
    const figure = document.getElementById("early-stopping");
    if (!figure) return;
    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "Left: the degree-9 fit changing during gradient descent. Right: training and validation loss against the step number, log scale.");
    const panel = dataPanel(root);
    errorAxes(root);
    const MAX_LOG = 6; // run to a million steps
    const SX = (t) => PX0 + Math.log10(Math.max(1, t)) / MAX_LOG * (PX1 - PX0);
    for (const [t, label] of [[1, "1"], [100, "100"], [1e4, "10k"], [1e6, "1M"]]) {
      svg("text", { x: SX(t), y: EBOT + 18, "text-anchor": "middle", class: "muted", text: label }, root);
    }
    const bestLine = svg("line", { y1: ETOP, y2: EBOT, class: "green thin" }, root);
    const trainLine = svg("polyline", { class: "blue", "stroke-width": 1.5 }, root);
    const valLine = svg("polyline", { class: "orange", "stroke-width": 1.5 }, root);

    // Plain gradient descent on the training MSE, from all-zero weights.
    const K = 9, N = xTrain.length, ETA = 0.3;
    const X = xTrain.map((x) => powers(x, K));
    const Xv = xVal.map((x) => powers(x, K));
    const valLoss = (w) => Xv.reduce((s, row, i) => { let p = 0; for (let j = 0; j <= K; j++) p += row[j] * w[j]; return s + (p - yVal[i]) ** 2; }, 0) / Xv.length;
    let w, t, trainPts, valPts, best, lastTrain, lastVal, done, wait;

    function gdStep() {
      const g = new Array(K + 1).fill(0);
      for (let i = 0; i < N; i++) {
        let e = -yTrain[i];
        for (let j = 0; j <= K; j++) e += X[i][j] * w[j];
        for (let j = 0; j <= K; j++) g[j] += e * X[i][j];
      }
      for (let j = 0; j <= K; j++) w[j] -= ETA * 2 * g[j] / N;
      t += 1;
    }

    function reset() {
      w = new Array(K + 1).fill(0);
      t = 0;
      lastTrain = mse(w, xTrain, yTrain);
      lastVal = valLoss(w);
      trainPts = [`${SX(1)},${EY(lastTrain)}`];
      valPts = [`${SX(1)},${EY(lastVal)}`];
      best = { loss: lastVal, step: 0 };
      done = false;
      wait = 0;
      draw();
    }

    // Run until step `target`. The validation minimum comes early, so check every step only
    // for the first 10,000 and at frame ends after that.
    function runTo(target) {
      while (t < target) {
        gdStep();
        if (t <= 1e4) {
          const v = valLoss(w);
          if (v < best.loss) best = { loss: v, step: t };
        }
      }
      lastTrain = mse(w, xTrain, yTrain);
      lastVal = valLoss(w);
      if (lastVal < best.loss) best = { loss: lastVal, step: t };
      trainPts.push(`${SX(t)},${EY(lastTrain)}`);
      valPts.push(`${SX(t)},${EY(lastVal)}`);
      if (t >= 10 ** MAX_LOG) done = true;
      draw();
    }

    function draw() {
      panel.setCurve(w);
      trainLine.setAttribute("points", trainPts.join(" "));
      valLine.setAttribute("points", valPts.join(" "));
      bestLine.style.display = best.step ? "" : "none";
      bestLine.setAttribute("x1", SX(best.step)); bestLine.setAttribute("x2", SX(best.step));
      out.innerHTML =
        `degree 9, η = 0.3   step ${t.toLocaleString("en-US")}${done ? " (end)" : ""}\n` +
        `<span class="blue">training MSE   ${fmt(lastTrain, 4)}</span>\n` +
        `<span class="orange">validation MSE ${fmt(lastVal, 4)}</span>\n` +
        `<span class="green">lowest validation ${fmt(best.loss, 4)} (step ${best.step.toLocaleString("en-US")})</span>`;
    }

    const out = readout(figure);
    const btns = buttons(figure);
    const SPEED = 0.6, HOLD = 3; // decades of steps per second; seconds to hold at the end
    let progress = 0;
    const player = loop(figure, (dt) => {
      if (!dt) return;
      if (done) {
        wait += dt;
        if (wait > HOLD) { reset(); progress = 0; }
        return;
      }
      progress = Math.min(MAX_LOG, progress + SPEED * dt);
      runTo(Math.floor(10 ** progress));
    });
    const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
      if (player.playing) player.pause(); else player.play();
      toggle.textContent = player.playing ? "Pause" : "Play";
    });
    button(btns, "Restart", () => { reset(); progress = 0; });
    button(btns, "Jump to step 305", () => {
      player.pause(); toggle.textContent = "Play";
      if (t > 305) { reset(); progress = 0; }
      // Step there through the same log-spaced checkpoints the animation uses, so the plot fills in.
      const end = Math.log10(305);
      while (progress < end) { progress = Math.min(end, progress + 0.02); runTo(Math.floor(10 ** progress)); }
      runTo(305);
    });
    reset();
  })();
})();
