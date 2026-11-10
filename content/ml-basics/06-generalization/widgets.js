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

// Step-through demos -------------------------------------------------------------------------
// Shared by the three demos below: the training and validation points from the worked example,
// and a small solver for the degree-k fit with an L2 penalty.
const u06demo = (function () {
  const xTrain = [-1, -0.78, -0.56, -0.33, -0.11, 0.11, 0.33, 0.56, 0.78, 1];
  const yTrain = [-0.23, -0.28, -1.19, -1.31, -0.44, 0.75, 0.64, 1.24, 0.51, -0.21];
  const xVal = [0.63, -0.63, -0.73, -0.08, 0.88, 0.2, -0.16, -0.69, 0.12, 0.32, -0.09, -0.98, -0.92, 0.88, -0.35, -0.05, 0.68, -0.91, 0.94, 0.22];
  const yVal = [1.22, -0.86, -0.65, -0.3, 0.22, 0.36, -0.46, -0.88, 0.46, 0.88, -0.2, -0.6, -0.08, 0.1, -0.61, 0.3, 0.94, -0.74, 0.41, 0.96];
  const N = xTrain.length;
  const poly = (w, x) => { let s = 0; for (let j = w.length - 1; j >= 0; j--) s = s * x + w[j]; return s; };
  const mse = (w, xs, ys) => xs.reduce((s, x, i) => s + (poly(w, x) - ys[i]) ** 2, 0) / xs.length;
  // The training MSE is a bowl: its gradient is H w − g, with H = (2/N) XᵀX and g = (2/N) Xᵀy.
  function normal(k) {
    const n = k + 1, H = [], g = new Array(n).fill(0);
    for (let a = 0; a < n; a++) {
      H.push(new Array(n).fill(0));
      for (let b = 0; b < n; b++) for (const x of xTrain) H[a][b] += 2 * x ** (a + b) / N;
      xTrain.forEach((x, i) => { g[a] += 2 * x ** a * yTrain[i] / N; });
    }
    return [H, g];
  }
  // Minimize MSE + (lam/2)(w_1² + ... + w_k²): solve (H + lam·D) w = g by elimination, where D
  // is the identity without its first entry (w_0 is not penalized).
  function fit(k, lam) {
    const [H, g] = normal(k), n = k + 1;
    for (let j = 1; j < n; j++) H[j][j] += lam;
    const A = H.map((row, i) => [...row, g[i]]);
    for (let col = 0; col < n; col++) {
      let p = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[p][col])) p = r;
      [A[col], A[p]] = [A[p], A[col]];
      for (let r = col + 1; r < n; r++) { const f = A[r][col] / A[col][col]; for (let q = col; q <= n; q++) A[r][q] -= f * A[col][q]; }
    }
    const w = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = A[r][n]; for (let q = r + 1; q < n; q++) s -= A[r][q] * w[q]; w[r] = s / A[r][r]; }
    return w;
  }
  const info = (w) => ({ w, train: mse(w, xTrain, yTrain), val: mse(w, xVal, yVal), big: Math.max(...w.map(Math.abs)) });

  // The data plot the demos share: axes, the true curve, the points, and one fitted curve.
  // `curve(x)` gives the fit's height; `valAlpha` fades the validation points in.
  function plot(ctx, c, d, g, curve, valAlpha) {
    const X = (x) => g.cx + x * g.sx, Y = (y) => g.cy - y * g.sy;
    d.line(X(-1.05), Y(0), X(1.05), Y(0), c.muted, 1.5);
    d.line(X(0), Y(-2.3), X(0), Y(2.3), c.muted, 1.5);
    for (const x of [-1, 1]) d.text(d.fmt(x, 0), X(x), Y(0) - 15, { c: c.muted, s: 12, a: "center" });
    ctx.save();
    ctx.beginPath(); ctx.rect(X(-1.05), Y(2.3), 2.1 * g.sx, 4.6 * g.sy); ctx.clip();
    ctx.strokeStyle = c.muted; ctx.lineWidth = 1.5; ctx.setLineDash([5, 5]);
    ctx.beginPath();
    for (let i = 0; i <= 105; i++) { const x = -1.05 + i * 0.02; if (i) ctx.lineTo(X(x), Y(Math.sin(Math.PI * x))); else ctx.moveTo(X(x), Y(Math.sin(Math.PI * x))); }
    ctx.stroke(); ctx.setLineDash([]);
    if (curve) {
      ctx.strokeStyle = c.ink; ctx.lineWidth = 3; ctx.lineJoin = "round";
      ctx.beginPath();
      for (let i = 0; i <= 210; i++) { const x = -1.05 + i * 0.01, y = d.clamp(curve(x), -4, 4); if (i) ctx.lineTo(X(x), Y(y)); else ctx.moveTo(X(x), Y(y)); }
      ctx.stroke();
    }
    ctx.restore();
    if (valAlpha > 0.01) {
      const before = ctx.globalAlpha;
      ctx.globalAlpha = before * valAlpha;
      ctx.strokeStyle = c.orange; ctx.lineWidth = 2;
      xVal.forEach((x, i) => { ctx.beginPath(); ctx.arc(X(x), Y(yVal[i]), 4.5, 0, 7); ctx.stroke(); });
      ctx.globalAlpha = before;
    }
    xTrain.forEach((x, i) => d.dot(X(x), Y(yTrain[i]), 5.5, c.blue));
  }
  // Height of a curve part-way between two fits (m = 1.5 is halfway from fit 1 to fit 2).
  const between = (fits, m) => {
    const i = Math.max(0, Math.min(fits.length - 2, Math.floor(m))), t = m - i;
    return (x) => poly(fits[i].w, x) * (1 - t) + poly(fits[i + 1].w, x) * t;
  };
  return { xTrain, yTrain, xVal, yVal, poly, mse, normal, fit, info, plot, between };
})();

// Step-through: a straight line underfits, degree 3 fits, degree 9 overfits.
(function () {
  const U = u06demo, degrees = [1, 3, 9];
  const fits = degrees.map((k) => U.info(U.fit(k, 0)));
  const notes = ["underfits", "follows the trend", "overfits"];
  const G = { cx: 215, cy: 180, sx: 176, sy: 66 };
  Learn.demo(document.getElementById("demo-fit"), {
    height: 360,
    label: "Ten noisy training points with polynomial fits of degree 1, 3 and 9, then 20 validation points the fit never saw",
    steps: [
      { s: { fit: 0, m: 0, val: 0 }, cap: "The dashed curve is the hidden rule, sin(πx). The ten blue dots are the training points. Noise puts each one about 0.25 above or below the curve." },
      { s: { fit: 1 }, cap: "Fit a straight line (degree 1). It can't bend, so it misses the dip and the hump. Its training loss is 0.4364. This is underfitting." },
      { s: { m: 1 }, cap: "Degree 3 can turn twice. It follows the dip and the hump and ignores the small wobbles. Its training loss is 0.0979." },
      { s: { m: 2 }, cap: "Degree 9 has ten weights for ten points. It passes through every dot, so its training loss is 0.0000. Between the dots it swings far away from the hidden rule." },
      { s: { val: 1 }, cap: "Now show 20 new points the fit never saw (orange rings). Degree 9 misses them badly: validation loss 1.4732. It learned the noise in its ten points. This is overfitting." },
      { s: { m: 1 }, cap: "Back to degree 3. Its validation loss is 0.0699, close to the noise floor of 0.0625. The simpler curve does better on new data." },
    ],
    draw(ctx, s, c, d) {
      U.plot(ctx, c, d, G, s.fit > 0.5 ? U.between(fits, s.m) : null, s.val);
      d.text("- - hidden rule", 36, 20, { c: c.muted, s: 12 });
      if (s.fit <= 0.5) return;
      const i = Math.round(s.m), f = fits[i];
      d.panel(c, 420, 40, 208, s.val > 0.01 ? 200 : 136);
      d.text(`degree ${degrees[i]}`, 434, 64, { c: c.ink, s: 17, w: 700 });
      d.text(`${degrees[i] + 1} weights`, 434, 88, { c: c.muted, s: 13 });
      d.text("training loss", 434, 120, { c: c.blue, s: 13, w: 700 });
      d.text(f.train.toFixed(4), 434, 144, { c: c.ink, s: 18, w: 700 });
      if (s.val > 0.01) {
        ctx.globalAlpha = s.val;
        d.text("validation loss", 434, 180, { c: c.orange, s: 13, w: 700 });
        d.text(f.val.toFixed(4), 434, 204, { c: f.val > 0.5 ? c.red : c.ink, s: 18, w: 700 });
        d.text(notes[i], 434, 228, { c: i === 1 ? c.muted : c.red, s: 13, w: 700 });
      }
    },
  });
})();

// Step-through: an L2 penalty calms the degree-9 fit, and too much penalty flattens it.
(function () {
  const U = u06demo, lams = [0, 0.0001, 0.01, 1];
  const fits = lams.map((lam) => U.info(U.fit(9, lam)));
  const G = { cx: 215, cy: 180, sx: 176, sy: 66 };
  Learn.demo(document.getElementById("demo-penalty"), {
    height: 360,
    label: "The degree-9 fit for four penalty strengths, from wild swings at zero penalty to a nearly flat curve at a large penalty",
    steps: [
      { s: { m: 0 }, cap: "Degree 9 with no penalty (λ = 0). The curve hits every blue dot but swings wildly. Its largest weight is 270.4 and its validation loss is 1.4732." },
      { s: { m: 1 }, cap: "Add a tiny penalty, λ = 0.0001. Big weights now cost something. The largest weight falls to 5.0, the swings calm down and validation loss drops to 0.1325." },
      { s: { m: 2 }, cap: "At λ = 0.01 the curve is smooth. Training loss rises to 0.0954, because the curve no longer hits every dot. Validation loss is 0.0663, the lowest of the four." },
      { s: { m: 3 }, cap: "At λ = 1 the penalty wins. The largest weight is only 0.3, so the curve goes nearly flat. Training loss is 0.4667 and validation loss is 0.2629. Too much penalty underfits." },
      { s: { m: 2 }, cap: "So λ is a setting you pick by comparing validation losses, the same way you pick the degree. Here that picks λ = 0.01." },
    ],
    draw(ctx, s, c, d) {
      U.plot(ctx, c, d, G, U.between(fits, s.m), 1);
      const i = Math.round(s.m), f = fits[i];
      d.panel(c, 420, 40, 208, 256);
      d.text(`λ = ${lams[i]}`, 434, 64, { c: c.ink, s: 17, w: 700 });
      d.text("degree 9, ten weights", 434, 88, { c: c.muted, s: 13 });
      d.text("largest weight", 434, 120, { c: c.purple, s: 13, w: 700 });
      d.text(f.big.toFixed(1), 434, 144, { c: f.big > 100 ? c.red : c.ink, s: 18, w: 700 });
      d.text("training loss", 434, 178, { c: c.blue, s: 13, w: 700 });
      d.text(f.train.toFixed(4), 434, 202, { c: c.ink, s: 18, w: 700 });
      d.text("validation loss", 434, 236, { c: c.orange, s: 13, w: 700 });
      d.text(f.val.toFixed(4), 434, 260, { c: f.val > 0.5 ? c.red : c.ink, s: 18, w: 700 });
      d.text("blue dots: training   orange rings: validation", 36, 346, { c: c.muted, s: 12 });
    },
  });
})();

// Step-through: gradient descent on the degree-9 fit, and the step where validation loss turns.
(function () {
  const U = u06demo, ETA = 0.3, n = 10;
  // Gradient descent on a bowl has a closed form. Along each axis of the bowl (an eigenvector
  // of H, with curvature lam) the weights close the gap to the exact fit by a factor of
  // 1 − η·lam per step. Jacobi rotations find those axes.
  const [H, g] = U.normal(9);
  const A = H.map((row) => row.slice()), V = A.map((_, i) => A.map((_, j) => +(i === j)));
  for (let sweep = 0; sweep < 60; sweep++) for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
    if (Math.abs(A[p][q]) < 1e-300) continue;
    const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
    const t = (th < 0 ? -1 : 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), co = 1 / Math.sqrt(t * t + 1), si = t * co;
    for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = co * a - si * b; A[k][q] = si * a + co * b; }
    for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = co * a - si * b; A[q][k] = si * a + co * b; }
    for (let k = 0; k < n; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = co * a - si * b; V[k][q] = si * a + co * b; }
  }
  const lam = A.map((row, i) => row[i]);
  const target = lam.map((l, i) => V.reduce((s, row, k) => s + row[i] * g[k], 0) / l); // the exact fit, along each axis
  const weightsAt = (step) => {
    const w = new Array(n).fill(0);
    lam.forEach((l, i) => { const a = target[i] * (1 - (1 - ETA * l) ** step); for (let k = 0; k < n; k++) w[k] += a * V[k][i]; });
    return w;
  };
  // Both losses at 241 step counts spread evenly on a log scale from 1 to 1,000,000.
  const curve = [];
  for (let i = 0; i <= 240; i++) { const w = weightsAt(10 ** (i / 40)); curve.push([U.mse(w, U.xTrain, U.yTrain), U.mse(w, U.xVal, U.yVal)]); }
  let bestStep = 1, bestVal = Infinity;
  for (let step = 1; step <= 5000; step++) { const v = U.mse(weightsAt(step), U.xVal, U.yVal); if (v < bestVal) { bestVal = v; bestStep = step; } }

  const G = { cx: 172, cy: 180, sx: 140, sy: 62 };
  const SX = (u) => 424 + u / 6 * 196;
  const EY = (m) => 110 + (Math.log10(2) - Math.log10(Math.max(m, 0.005))) / (Math.log10(2) - Math.log10(0.005)) * 190;
  const L10 = (v) => Math.log10(v);
  Learn.demo(document.getElementById("demo-stop"), {
    height: 360,
    label: "Left: the degree-9 curve during gradient descent. Right: training and validation loss against the step number, where validation loss turns back up after step 305",
    steps: [
      { s: { u: 0, seen: 0, mk: 0 }, cap: "Train the degree-9 curve with gradient descent, starting from all weights at zero. After 1 step the curve has barely left zero. The chart on the right tracks both losses." },
      { s: { u: L10(30), seen: L10(30) }, cap: "After 30 steps the curve has the rough shape of the data. Both losses are falling together: training 0.1933, validation 0.0945." },
      { s: { u: L10(305), seen: L10(305) }, cap: "At step 305 the validation loss reaches its lowest value, 0.0679. Training loss is 0.0836. The curve follows the dip and the hump." },
      { s: { u: 3, seen: 3 }, cap: "By step 1,000 training loss is down to 0.0373, but validation loss is back up to 0.1020. The extra steps are fitting noise in the ten blue dots." },
      { s: { u: 6, seen: 6 }, cap: "After a million steps training loss is 0.0114 and validation loss is 0.1542. The curve now bends to reach each blue dot." },
      { s: { u: L10(305), mk: 1 }, cap: "Early stopping: keep the weights from step 305, where validation loss was lowest (green line). You get the smooth curve back without choosing a smaller model." },
    ],
    draw(ctx, s, c, d) {
      const w = weightsAt(10 ** s.u);
      U.plot(ctx, c, d, G, (x) => U.poly(w, x), 1);
      const train = U.mse(w, U.xTrain, U.yTrain), val = U.mse(w, U.xVal, U.yVal);
      d.text(`step ${Math.round(10 ** s.u).toLocaleString("en-US")}`, 424, 24, { c: c.ink, s: 17, w: 700 });
      d.text(`training loss    ${train.toFixed(4)}`, 424, 52, { c: c.blue, s: 13, w: 700 });
      d.text(`validation loss  ${val.toFixed(4)}`, 424, 74, { c: c.orange, s: 13, w: 700 });
      for (const [m, label] of [[1, "1"], [0.1, "0.1"], [0.01, "0.01"]]) {
        d.line(SX(0), EY(m), SX(6), EY(m), c.line);
        d.text(label, SX(0) - 8, EY(m), { c: c.muted, s: 12, a: "right" });
      }
      d.line(SX(0), EY(0.005), SX(6), EY(0.005), c.muted, 1.5);
      d.line(SX(0), EY(2), SX(0), EY(0.005), c.muted, 1.5);
      for (const [u, label] of [[0, "1"], [2, "100"], [4, "10k"], [6, "1M"]]) d.text(label, SX(u), EY(0.005) + 15, { c: c.muted, s: 12, a: u === 6 ? "right" : "center" });
      d.text("steps (log scale)", SX(6), EY(0.005) + 36, { c: c.muted, s: 12, a: "right" });
      if (s.mk > 0.01) {
        ctx.globalAlpha = s.mk;
        d.line(SX(L10(bestStep)), EY(2), SX(L10(bestStep)), EY(0.005), c.green, 2);
        ctx.globalAlpha = 1;
      }
      const last = Math.round(d.clamp(s.seen, 0, 6) * 40);
      for (const [k, color] of [[0, c.blue], [1, c.orange]]) {
        ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.lineJoin = "round";
        ctx.beginPath();
        for (let i = 0; i <= last; i++) { if (i) ctx.lineTo(SX(i / 40), EY(curve[i][k])); else ctx.moveTo(SX(i / 40), EY(curve[i][k])); }
        ctx.stroke();
      }
      d.dot(SX(s.u), EY(train), 5, c.blue);
      d.dot(SX(s.u), EY(val), 5, c.orange);
    },
  });
})();
