// Unit 17: best-of-n against a reward model that keeps rewarding x after people stop liking it.
// Same toy as the blog post: x ~ N(0, 3²), reward model = x + N(0, 1), true quality = x − x²/10.
(function () {
  const figure = document.getElementById("best-of-n");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  const width = 720, rows = 15, height = rows * Learn.ROW;
  const K = 12; // n runs over 1, 2, 4, ..., 2^K
  const ns = Array.from({ length: K + 1 }, (_, k) => 2 ** k);
  const klOf = (n) => Math.log(n) - (n - 1) / n;
  const trueReward = (x) => x - x * x / 10;

  // Standard normal samples (Box-Muller, keeping the second value for the next call).
  let spare = null;
  function gauss() {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0;
    while (u === 0) u = Math.random();
    const r = Math.sqrt(-2 * Math.log(u)), t = 2 * Math.PI * Math.random();
    spare = r * Math.sin(t);
    return r * Math.cos(t);
  }

  // Normal CDF via the Abramowitz-Stegun erf approximation (error below 2e-7).
  function phi(z) {
    const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return z >= 0 ? (1 + y) / 2 : (1 - y) / 2;
  }

  // Running averages for every n at once, plus the last few picks for drawing.
  const KEEP = 24;
  const stats = ns.map(() => ({ trials: 0, x: 0, proxy: 0, true: 0, recent: [] }));
  function trial(k) {
    let bestScore = -Infinity, bestX = 0;
    for (let i = 0; i < ns[k]; i++) {
      const x = 3 * gauss();
      const score = x + gauss();
      if (score > bestScore) { bestScore = score; bestX = x; }
    }
    const s = stats[k];
    s.trials++;
    s.x += bestX;
    s.proxy += bestScore;
    s.true += trueReward(bestX);
    s.recent.push([bestX, bestScore]);
    if (s.recent.length > KEEP) s.recent.shift();
  }
  for (let r = 0; r < 200; r++) for (let k = 0; k <= K; k++) trial(k);

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Left: the responses best-of-n picks, scored by the reward model and by people. Right: average scores against KL from the starting policy.");

  // Left panel: response space. x across, score up.
  const L = { x0: 30, x1: 380, y0: 8, y1: 250, xmin: -8, xmax: 13, vmin: -8, vmax: 13 };
  const LX = (x) => L.x0 + (x - L.xmin) / (L.xmax - L.xmin) * (L.x1 - L.x0);
  const LY = (v) => L.y1 - (v - L.vmin) / (L.vmax - L.vmin) * (L.y1 - L.y0);
  const density = [];
  for (let x = L.xmin; x <= L.xmax + 1e-9; x += 0.25) {
    density.push([LX(x), L.y1 - 70 * Math.exp(-x * x / 18)]);
  }
  svg("polygon", { class: "muted faint", points: [[LX(L.xmin), L.y1], ...density, [LX(L.xmax), L.y1]].map((p) => p.join(",")).join(" ") }, root);
  svg("text", { class: "muted", x: LX(6.2), y: L.y1 - 10, text: "π₀" }, root);
  svg("line", { class: "muted faint thin", x1: L.x0, y1: LY(0), x2: L.x1, y2: LY(0) }, root);
  svg("line", { class: "muted thin", x1: L.x0, y1: L.y1, x2: L.x1, y2: L.y1 }, root);
  svg("line", { class: "orange faint thin dashed", x1: LX(5), y1: L.y0, x2: LX(5), y2: L.y1 }, root);
  for (const x of [-5, 0, 5]) {
    svg("text", { class: "muted", x: LX(x), y: L.y1 + 20, "text-anchor": "middle", text: String(x) }, root);
  }
  svg("text", { class: "muted", x: L.x1, y: L.y1 + 20, "text-anchor": "end", text: "x" }, root);
  svg("line", { class: "blue dashed", x1: LX(L.vmin), y1: LY(L.vmin), x2: LX(L.vmax), y2: LY(L.vmax) }, root);
  svg("text", { class: "blue", x: LX(-2), y: LY(3.5), "text-anchor": "end", text: "proxy" }, root);
  const curve = [];
  for (let x = -5.2; x <= L.xmax + 1e-9; x += 0.2) curve.push([LX(x), LY(trueReward(x))].join(","));
  svg("polyline", { class: "orange", points: curve.join(" ") }, root);
  svg("text", { class: "orange", x: LX(11.4), y: LY(-5.6), "text-anchor": "middle", text: "true" }, root);
  const picks = svg("g", {}, root);
  const pickDots = Array.from({ length: KEEP }, () => [
    svg("circle", { r: 3, class: "blue", fill: "currentColor" }, picks),
    svg("circle", { r: 3, class: "orange", fill: "currentColor" }, picks),
  ]);

  // Right panel: average scores at every n, placed by KL.
  const R = { x0: 450, x1: 706, y0: 8, y1: 250, kmax: 7.5, vmin: -2, vmax: 12 };
  const RX = (kl) => R.x0 + kl / R.kmax * (R.x1 - R.x0);
  const RY = (v) => R.y1 - (v - R.vmin) / (R.vmax - R.vmin) * (R.y1 - R.y0);
  svg("line", { class: "muted thin", x1: R.x0, y1: R.y0, x2: R.x0, y2: R.y1 }, root);
  svg("line", { class: "muted thin", x1: R.x0, y1: R.y1, x2: R.x1, y2: R.y1 }, root);
  svg("line", { class: "muted faint thin", x1: R.x0, y1: RY(0), x2: R.x1, y2: RY(0) }, root);
  for (const v of [0, 5, 10]) {
    svg("text", { class: "muted", x: R.x0 - 8, y: RY(v) + 4, "text-anchor": "end", text: String(v) }, root);
  }
  for (const kl of [0, 2, 4]) {
    svg("text", { class: "muted", x: RX(kl), y: R.y1 + 20, "text-anchor": "middle", text: String(kl) }, root);
  }
  svg("text", { class: "muted", x: R.x1, y: R.y1 + 20, "text-anchor": "end", text: "KL" }, root);
  const marker = svg("line", { class: "muted dashed thin", y1: R.y0, y2: R.y1 }, root);
  const proxyLine = svg("polyline", { class: "blue" }, root);
  const trueLine = svg("polyline", { class: "orange" }, root);
  const proxyDots = ns.map(() => svg("circle", { r: 3, class: "blue", fill: "currentColor" }, root));
  const trueDots = ns.map(() => svg("circle", { r: 3, class: "orange", fill: "currentColor" }, root));

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  let k = 4; // n = 16, where true quality peaks
  function place(dot, x, y, r) {
    dot.setAttribute("cx", x);
    dot.setAttribute("cy", y);
    dot.setAttribute("r", r);
  }

  function draw() {
    const s = stats[k], n = ns[k];
    pickDots.forEach(([blue, orange], i) => {
      const pick = s.recent[s.recent.length - 1 - i];
      const shown = pick && pick[0] >= L.xmin && pick[0] <= L.xmax;
      blue.style.display = orange.style.display = shown ? "" : "none";
      if (!shown) return;
      const r = i === 0 ? 5 : 3;
      const vs = [pick[1], trueReward(pick[0])];
      [blue, orange].forEach((dot, j) => {
        const inside = vs[j] >= L.vmin && vs[j] <= L.vmax;
        dot.style.display = inside ? "" : "none";
        place(dot, LX(pick[0]), LY(vs[j]), r);
        dot.style.opacity = i === 0 ? 1 : 0.55;
      });
    });

    const proxyPoints = [], truePoints = [];
    ns.forEach((m, j) => {
      const t = stats[j], x = RX(klOf(m)), r = j === k ? 6 : 3;
      const yp = RY(t.proxy / t.trials), yt = RY(t.true / t.trials);
      proxyPoints.push(`${x},${yp}`);
      truePoints.push(`${x},${yt}`);
      place(proxyDots[j], x, yp, r);
      place(trueDots[j], x, yt, r);
    });
    proxyLine.setAttribute("points", proxyPoints.join(" "));
    trueLine.setAttribute("points", truePoints.join(" "));
    marker.setAttribute("x1", RX(klOf(n)));
    marker.setAttribute("x2", RX(klOf(n)));

    const meanX = s.x / s.trials;
    const tail = 100 * (1 - phi(meanX / 3));
    const pct = tail >= 10 ? tail.toFixed(0) : tail >= 1 ? tail.toFixed(1) : tail.toPrecision(2);
    out.innerHTML =
      `best of ${n} draws from π₀\n` +
      `KL = log ${n} − ${n - 1}/${n} = ${fmt(klOf(n))} nats\n` +
      `${s.trials} trials, average pick x = ${fmt(meanX)}\n` +
      `π₀ draws an x that large ${pct}% of the time\n` +
      `<span class="blue">reward model score ${fmt(s.proxy / s.trials).padStart(6)}</span>\n` +
      `<span class="orange">true quality       ${fmt(s.true / s.trials).padStart(6)}</span>`;
  }

  slider(ctl, { label: "n, log scale", min: 0, max: K, step: 1, value: k, format: (v) => String(2 ** v) }, (v) => {
    k = v;
    draw();
  });

  // Every 0.1 s, run one more best-of-n trial at every n and redraw.
  let clock = 0;
  const player = loop(figure, (dt) => {
    clock += dt;
    if (clock < 0.1) return;
    clock = 0;
    for (let j = 0; j <= K; j++) trial(j);
    draw();
  });
  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
})();

// Step-through: the toy's true reward curve, the reward model's straight line, and the error between them.
(function () {
  const X = (v) => 50 + (v + 6) * 22, Y = (v) => 290 - (v + 4) * 16.25;
  const truth = (x) => x - x * x / 10;
  Learn.demo(document.getElementById("demo-proxy"), {
    height: 340,
    label: "The true reward curve, the reward model's straight line, and the gap between them at x = 2, 5 and 10",
    steps: [
      { s: { tr: 0, ln: 0, mk: 0, mx: 2 }, cap: "Each response is one number x, something like length. π0 writes x near 0, with a standard deviation of 3. About 95% of its answers have x below 5." },
      { s: { tr: 1 }, cap: "The orange curve is the true reward, what people think: x − x²/10. It rises to 2.5 at x = 5, then falls." },
      { s: { ln: 1 }, cap: "The blue line is the reward model's score. It learned \"more x is better\" from π0's answers, so it is a straight line: score = x (plus noise, left out here)." },
      { s: { mk: 1 }, cap: "At x = 2 the score is 2 and the true reward is 1.6. The error is 0.4. Near π0's answers the line and the curve agree." },
      { s: { mx: 5 }, cap: "At x = 5 the score is 5 and the true reward is 2.5. The error has grown to 2.5." },
      { s: { mx: 10 }, cap: "At x = 10 the score is 10 and the true reward is 0. The error is 10. π0 almost never writes here, so no comparison ever corrected the line." },
    ],
    draw(ctx, s, c, d) {
      d.line(X(-6), Y(0), X(12), Y(0), c.muted, 1.5);
      d.line(X(0), 30, X(0), 290, c.muted, 1.5);
      for (const v of [-5, 5, 10]) { d.line(X(v), Y(0) - 4, X(v), Y(0) + 4, c.muted, 1.5); d.text(String(v).replace("-", "−"), X(v), 304, { c: c.muted, s: 12, a: "center" }); }
      d.text("0", X(0), 304, { c: c.muted, s: 12, a: "center" });
      d.text("x", X(12), 322, { c: c.muted, s: 12, a: "right" });
      // π0's bell, drawn up from the bottom edge
      ctx.fillStyle = c.soft; ctx.strokeStyle = c.muted; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X(-6), 290);
      for (let v = -6; v <= 12.01; v += 0.25) ctx.lineTo(X(v), 290 - 60 * Math.exp(-v * v / 18));
      ctx.lineTo(X(12), 290); ctx.closePath(); ctx.fill(); ctx.stroke();
      d.text("π0's answers", X(0), 274, { c: c.muted, s: 12, a: "center" });
      ctx.save(); ctx.beginPath(); ctx.rect(X(-6), 30, X(12) - X(-6), 260); ctx.clip();
      if (s.tr > 0.01) {
        ctx.globalAlpha = s.tr; ctx.strokeStyle = c.orange; ctx.lineWidth = 3; ctx.beginPath();
        for (let v = -3.2; v <= 12.01; v += 0.2) { if (v === -3.2) ctx.moveTo(X(v), Y(truth(v))); else ctx.lineTo(X(v), Y(truth(v))); }
        ctx.stroke();
      }
      if (s.ln > 0.01) { ctx.globalAlpha = s.ln; d.line(X(-4), Y(-4), X(12), Y(12), c.blue, 3, [8, 5]); }
      ctx.restore();
      if (s.tr > 0.01) { ctx.globalAlpha = s.tr; d.text("true reward", X(5), 209, { c: c.orange, s: 13, a: "center", w: 700 }); }
      if (s.ln > 0.01) { ctx.globalAlpha = s.ln; d.text("reward model score", 368, 66, { c: c.blue, s: 13, a: "right", w: 700 }); }
      if (s.mk > 0.01) {
        ctx.globalAlpha = s.mk;
        const x = s.mx, t = truth(x);
        d.line(X(x), Y(x), X(x), Y(t), c.red, 4);
        d.dot(X(x), Y(x), 6, c.blue); d.dot(X(x), Y(t), 6, c.orange);
        d.panel(c, 462, 30, 166, 130);
        d.text(`x = ${d.fmt(x, 1)}`, 476, 52, { c: c.ink, w: 700 });
        d.text(`score  ${d.fmt(x, 1)}`, 476, 82, { c: c.blue });
        d.text(`true   ${d.fmt(t, 1)}`, 476, 106, { c: c.orange });
        d.text(`error  ${d.fmt(x - t, 1)}`, 476, 136, { c: c.red, w: 700 });
      }
      ctx.globalAlpha = 1;
    },
  });
})();

// Step-through: best-of-n on the toy with a fixed set of draws, from n = 4 up to n = 4096.
(function () {
  const X = (v) => 50 + (v + 6) * 22, Y = (v) => 290 - (v + 4) * 15;
  const truth = (x) => x - x * x / 10;
  // Seeded draws (mulberry32 + Box–Muller): x from π0 = N(0, 3²), score = x + N(0, 1).
  let a = 1796;
  const rnd = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const norm = () => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
  const P = [], best = [];
  for (let i = 0; i < 4096; i++) { const x = 3 * norm(); P.push([x, x + norm()]); best.push(i === 0 || P[i][1] > P[best[i - 1]][1] ? i : best[i - 1]); }
  Learn.demo(document.getElementById("demo-pick"), {
    height: 340,
    label: "Draws from π0 plotted by x and reward model score, with the top scorer marked, for n = 4, 16, 256 and 4096",
    steps: [
      { s: { ln: 2, sc: 0, win: 0, cv: 0 }, cap: "Best-of-4. Draw 4 responses from π0. Their x values are 2.9, −3.4, −1.9 and −2." },
      { s: { sc: 1 }, cap: "The reward model scores each one: x plus a little noise. The scores are 3.4, −2.6, −1.4 and −1.8." },
      { s: { win: 1 }, cap: "Keep the top scorer and throw the rest away. Best-of-4 returns the response with x = 2.9 and score 3.4." },
      { s: { cv: 1 }, cap: "The orange curve is what people think. They would give this pick 2.0. The score and the true reward are still close." },
      { s: { ln: 4 }, cap: "Best-of-16: 12 more draws. The top scorer now has x = 5.3 and score 6.7. Its true reward is 2.5, right at the top of the curve." },
      { s: { ln: 8 }, cap: "Best-of-256. The top scorer has x = 7.5 and score 9.2. Its true reward has dropped to 1.9. The pick is past the peak." },
      { s: { ln: 12 }, cap: "Best-of-4096. The top scorer has x = 10.9 and score 11.5, the highest yet. People would give it −1.0. The search found where the reward model is most wrong." },
    ],
    draw(ctx, s, c, d) {
      const n = Math.max(1, Math.min(4096, Math.round(Math.pow(2, s.ln)))), w = P[best[n - 1]];
      d.line(X(-6), Y(0), X(12), Y(0), c.muted, 1.5);
      d.line(X(0), 35, X(0), 290, c.muted, 1.5);
      for (const v of [-5, 5, 10]) { d.line(X(v), Y(0) - 4, X(v), Y(0) + 4, c.muted, 1.5); d.text(String(v).replace("-", "−"), X(v), 304, { c: c.muted, s: 12, a: "center" }); }
      d.text("x", X(12), 322, { c: c.muted, s: 12, a: "right" });
      d.text("score", X(0) - 8, 44, { c: c.muted, s: 12, a: "right" });
      ctx.save(); ctx.beginPath(); ctx.rect(X(-6), 30, X(12) - X(-6), 262); ctx.clip();
      if (s.cv > 0.01) {
        ctx.globalAlpha = s.cv; ctx.strokeStyle = c.orange; ctx.lineWidth = 3; ctx.beginPath();
        for (let v = -3.2; v <= 12.01; v += 0.2) { if (v === -3.2) ctx.moveTo(X(v), Y(truth(v))); else ctx.lineTo(X(v), Y(truth(v))); }
        ctx.stroke(); ctx.globalAlpha = 1;
      }
      const r = n <= 16 ? 5 : n <= 256 ? 3 : 1.5;
      ctx.fillStyle = c.blue; ctx.globalAlpha = n <= 16 ? 1 : 0.6;
      for (let i = 0; i < n; i++) ctx.fillRect(X(P[i][0]) - r, Y(P[i][1] * s.sc) - r, 2 * r, 2 * r);
      ctx.globalAlpha = 1;
      ctx.restore();
      if (s.win > 0.01) {
        ctx.globalAlpha = s.win;
        const wx = X(w[0]), wy = Y(w[1] * s.sc);
        if (s.cv > 0.01) { ctx.globalAlpha = s.win * s.cv; d.line(wx, wy, wx, Y(truth(w[0])), c.red, 3); d.dot(wx, Y(truth(w[0])), 6, c.orange); ctx.globalAlpha = s.win; }
        d.rect(wx - 9, wy - 9, 18, 18, null, c.ink, 2);
        d.panel(c, 462, 30, 166, 130);
        d.text(`n = ${n}`, 476, 52, { c: c.ink, w: 700 });
        d.text(`pick x ${d.fmt(w[0], 1)}`, 476, 82, { c: c.ink });
        d.text(`score  ${d.fmt(w[1], 1)}`, 476, 106, { c: c.blue });
        ctx.globalAlpha = s.win * s.cv;
        d.text(`true   ${d.fmt(truth(w[0]), 1)}`, 476, 136, { c: c.orange, w: 700 });
      }
      ctx.globalAlpha = 1;
    },
  });
})();

// Step-through: the worked example's averages at each n, placed by KL: score rises, true reward peaks and falls.
(function () {
  const N = [1, 4, 16, 64, 256, 1024, 4096];
  const KL = N.map((n) => Math.log(n) - (n - 1) / n);
  const PROXY = [-0.09, 3.22, 5.59, 7.42, 8.97, 10.29, 11.47];   // from the worked example's output
  const TRUE = [-0.99, 1.60, 2.17, 1.97, 1.34, 0.50, -0.50];
  const NOISE = [-0.80, 1.52, 2.03, 2.20, 2.27, 2.30, 2.33];
  const X = (kl) => 60 + kl * 50, Y = (v) => 280 - (v + 1.5) * 18.5;
  Learn.demo(document.getElementById("demo-curve"), {
    height: 330,
    label: "Average reward model score and true reward for best-of-n, plotted against KL from π0",
    steps: [
      { s: { k: 0, no: 0 }, cap: "Best-of-1 is π0 itself, 0 nats away. In the worked example's run its average score is −0.09 and its average true reward is −0.99." },
      { s: { k: 1 }, cap: "Best-of-4 is 0.64 nats from π0. The average pick is x = 2.90. The score rises to 3.22 and the true reward to 1.60. Both go up together." },
      { s: { k: 2 }, cap: "Best-of-16 is 1.84 nats away. The average pick is x = 5.01, the top of the true curve. The score is 5.59 and the true reward is 2.17, its peak." },
      { s: { k: 3 }, cap: "Best-of-64 is 3.17 nats away. The score climbs to 7.42, but the true reward slips to 1.97. The search has gone past the peak." },
      { s: { k: 6 }, cap: "Best-of-4096 is 7.32 nats away. The score is 11.47, the best yet. The true reward is −0.50. The reward model says these are great answers, and people would rate them below 0." },
      { s: { no: 1 }, cap: "Green: a reward model whose only error is noise. Its true reward rises at every n, to 2.33 at n = 4096 (the maximum is 2.5). Noise slows the search down. The straight line is what made the true reward fall." },
    ],
    draw(ctx, s, c, d) {
      const k = d.clamp(s.k, 0, 6), i0 = Math.round(k);
      d.line(X(0), 30, X(0), 290, c.muted, 1.5);
      d.line(X(0), Y(0), X(7.5), Y(0), c.muted, 1.5);
      for (const v of [5, 10]) { d.line(X(0) - 4, Y(v), X(0), Y(v), c.muted, 1.5); d.text(String(v), X(0) - 8, Y(v), { c: c.muted, s: 12, a: "right" }); }
      d.text("0", X(0) - 8, Y(0), { c: c.muted, s: 12, a: "right" });
      N.forEach((n, i) => { d.line(X(KL[i]), 286, X(KL[i]), 294, c.muted, 1.5); d.text(String(n), X(KL[i]), 304, { c: i === i0 ? c.ink : c.muted, s: 12, a: "center", w: i === i0 ? 700 : 500 }); });
      d.text("n, placed by KL from π0", X(7.5), 322, { c: c.muted, s: 12, a: "right" });
      const series = (vals, color, upto) => {
        ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(KL[0]), Y(vals[0]));
        for (let i = 1; i <= 6; i++) {
          if (i <= upto) ctx.lineTo(X(KL[i]), Y(vals[i]));
          else if (i - 1 < upto) { const f = upto - (i - 1); ctx.lineTo(X(KL[i - 1] + (KL[i] - KL[i - 1]) * f), Y(vals[i - 1] + (vals[i] - vals[i - 1]) * f)); }
        }
        ctx.stroke();
        for (let i = 0; i <= 6; i++) if (i <= upto + 0.01) d.dot(X(KL[i]), Y(vals[i]), i === i0 ? 6 : 4, color);
      };
      if (s.no > 0.01) { ctx.globalAlpha = s.no; series(NOISE, c.green, 6); ctx.globalAlpha = 1; }
      series(PROXY, c.blue, k);
      series(TRUE, c.orange, k);
      d.panel(c, 462, 30, 166, s.no > 0.01 ? 160 : 130);
      d.text(`n = ${N[i0]}`, 476, 52, { c: c.ink, w: 700 });
      d.text(`KL    ${d.fmt(KL[i0], 2)}`, 476, 78, { c: c.muted });
      d.text(`score ${d.fmt(PROXY[i0], 2)}`, 476, 104, { c: c.blue, w: 700 });
      d.text(`true  ${d.fmt(TRUE[i0], 2)}`, 476, 130, { c: c.orange, w: 700 });
      if (s.no > 0.01) { ctx.globalAlpha = s.no; d.text(`noise only ${d.fmt(NOISE[i0], 2)}`, 476, 162, { c: c.green, s: 13, w: 700 }); ctx.globalAlpha = 1; }
    },
  });
})();
