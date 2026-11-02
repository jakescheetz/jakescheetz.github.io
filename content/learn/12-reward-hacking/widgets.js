// Unit 12: best-of-n against a reward model that keeps rewarding x after people stop liking it.
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
