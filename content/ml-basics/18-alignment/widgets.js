// Unit 18: goal misgeneralization in a corridor. During training the coin sits at the end of the
// level, so "go to the coin" and "go right" earn the same reward. Same runs as the worked example:
// 20 REINFORCE runs per share, seeded with mulberry32, scored exactly on coin-on-the-left levels.
(function () {
  const figure = document.getElementById("coin-corridor");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  const N = 9, START = 4, VIEW = 4, STEPS = 12, RUNS = 20;
  const SHARES = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2];

  // mulberry32, identical to the worked example's rng().
  function rng(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), s | 1) >>> 0;
      t = (t ^ (t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0))) >>> 0;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  // +1 coin in view to the right, -1 in view to the left, 0 out of view.
  function sees(pos, coin) {
    const d = coin - pos;
    return Math.abs(d) > VIEW ? 0 : (d > 0 ? 1 : -1);
  }
  const pRight = (w, pos, coin) => sigmoid(w[0] * sees(pos, coin) + w[1]);

  // REINFORCE with a running-average baseline (unit 15), 3,000 levels, learning rate 0.2.
  function train(share, seed) {
    const rand = rng(seed);
    let wCoin = 0, wEnd = 0, b = null;
    for (let e = 0; e < 3000; e++) {
      let coin = N - 1;
      if (rand() < share) {
        coin = Math.floor(rand() * (N - 1));
        if (coin >= START) coin += 1;
      }
      let pos = START, gCoin = 0, gEnd = 0, r = 0;
      for (let t = 0; t < STEPS; t++) {
        const c = sees(pos, coin);
        const p = sigmoid(wCoin * c + wEnd);
        const right = rand() < p ? 1 : 0;
        gCoin += (right - p) * c;
        gEnd += right - p;
        pos = right ? Math.min(N - 1, pos + 1) : Math.max(0, pos - 1);
        if (pos === coin) { r = 1; break; }
      }
      b = b === null ? r : b + 0.1 * (r - b);
      wCoin += 0.2 * (r - b) * gCoin;
      wEnd += 0.2 * (r - b) * gEnd;
    }
    return [wCoin, wEnd];
  }

  // Exact chance of getting the coin within STEPS, and of standing at the end when time runs out.
  function outcome(w, coin) {
    let here = new Map([[START, 1]]);
    let got = 0;
    for (let t = 0; t < STEPS; t++) {
      const next = new Map();
      for (const [pos, q] of here) {
        const p = pRight(w, pos, coin);
        const moves = [[Math.min(N - 1, pos + 1), p], [Math.max(0, pos - 1), 1 - p]];
        for (const [to, pr] of moves) next.set(to, (next.get(to) || 0) + q * pr);
      }
      got += next.get(coin) || 0;
      next.delete(coin);
      here = next;
    }
    return [got, here.get(N - 1) || 0];
  }

  const results = SHARES.map((share) => {
    const runs = [];
    for (let seed = 1; seed <= RUNS; seed++) runs.push(train(share, seed));
    let train_ = 0, coin = 0, end = 0;
    for (const w of runs) {
      train_ += outcome(w, N - 1)[0];
      for (let c = 0; c < START; c++) {
        const [got, atEnd] = outcome(w, c);
        coin += got;
        end += atEnd;
      }
    }
    const tests = RUNS * START;
    return {
      runs,
      wins: runs.filter((w) => w[0] > w[1]).length,
      train: train_ / RUNS,
      coin: coin / tests,
      end: end / tests,
      wCoin: runs.reduce((a, w) => a + w[0], 0) / RUNS,
      wEnd: runs.reduce((a, w) => a + w[1], 0) / RUNS,
    };
  });

  const root = stage(figure, { rows: 17, width: 720 });
  root.setAttribute("aria-label", "Top: a corridor of nine cells with the agent, the coin and the end of the level. Bottom: for each share of training levels with a randomly placed coin, how often the trained agents get a coin placed to their left, and how often they end up at the end of the level instead.");

  // Corridor.
  const C = { x0: 90, w: 60, y: 22, h: 44 };
  const CX = (i) => C.x0 + (i + 0.5) * C.w;
  for (let i = 0; i < N; i++) {
    svg("rect", { class: "muted thin", x: C.x0 + i * C.w, y: C.y, width: C.w, height: C.h, fill: "none", stroke: "currentColor" }, root);
  }
  svg("rect", { class: "blue faint", x: C.x0 + (N - 1) * C.w + 2, y: C.y + 2, width: C.w - 4, height: C.h - 4, fill: "currentColor" }, root);
  svg("text", { class: "blue", x: CX(N - 1), y: C.y - 7, "text-anchor": "middle", text: "end" }, root);
  svg("text", { class: "muted", x: CX(START), y: C.y + C.h + 17, "text-anchor": "middle", text: "start" }, root);
  const view = svg("rect", { class: "ink faint dashed thin", y: C.y - 4, height: C.h + 8, fill: "none", stroke: "currentColor", rx: 4 }, root);
  const coinDot = svg("circle", { class: "orange", r: 9, cy: C.y + C.h / 2, fill: "currentColor" }, root);
  const agentDot = svg("circle", { class: "ink", r: 11, cy: C.y + C.h / 2, fill: "none", stroke: "currentColor", "stroke-width": 3 }, root);

  // Chart: one pair of bars per share.
  const G = { x0: 90, x1: 600, y0: 118, y1: 246 };
  const slot = (G.x1 - G.x0) / SHARES.length;
  const GY = (v) => G.y1 - v * (G.y1 - G.y0);
  svg("line", { class: "muted thin", x1: G.x0, y1: G.y1, x2: G.x1, y2: G.y1 }, root);
  svg("line", { class: "muted faint thin dashed", x1: G.x0, y1: GY(1), x2: G.x1, y2: GY(1) }, root);
  for (const v of [0, 0.5, 1]) {
    svg("text", { class: "muted", x: G.x0 - 8, y: GY(v) + 4, "text-anchor": "end", text: String(v) }, root);
  }
  const highlight = svg("rect", { class: "muted faint", y: G.y0 - 6, width: slot - 6, height: G.y1 - G.y0 + 6, fill: "currentColor", rx: 3 }, root);
  const pct = (s) => `${+(s * 100).toFixed(1)}%`;
  SHARES.forEach((s, j) => {
    const x = G.x0 + j * slot;
    const r = results[j];
    svg("rect", { class: "orange", x: x + slot / 2 - 18, y: GY(r.coin), width: 16, height: G.y1 - GY(r.coin), fill: "currentColor" }, root);
    svg("rect", { class: "blue", x: x + slot / 2 + 2, y: GY(r.end), width: 16, height: Math.max(0.5, G.y1 - GY(r.end)), fill: "currentColor" }, root);
    svg("text", { class: "muted", x: x + slot / 2, y: G.y1 + 24, "text-anchor": "middle", text: pct(s) }, root);
  });
  svg("text", { class: "orange", x: G.x1 + 8, y: G.y0 + 14, text: "coin" }, root);
  svg("text", { class: "blue", x: G.x1 + 8, y: G.y0 + 44, text: "at end" }, root);
  svg("text", { class: "muted", x: G.x1, y: G.y1 + 58, "text-anchor": "end", text: "share of levels with a random coin" }, root);

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  // One animated test level at a time: a random run at this share, the coin in a random cell.
  let k = 0, ep = null, clock = 0;
  function newLevel() {
    const run = Math.floor(Math.random() * RUNS);
    let coin = Math.floor(Math.random() * (N - 1));
    if (coin >= START) coin += 1;
    ep = { run, w: results[k].runs[run], coin, pos: START, t: 0, done: "" };
    clock = 0;
  }
  function step() {
    if (ep.done) return;
    const right = Math.random() < pRight(ep.w, ep.pos, ep.coin);
    ep.pos = right ? Math.min(N - 1, ep.pos + 1) : Math.max(0, ep.pos - 1);
    ep.t += 1;
    if (ep.pos === ep.coin) ep.done = "got the coin";
    else if (ep.t === STEPS) ep.done = ep.pos === N - 1 ? "out of time, at the end" : "out of time";
  }

  function draw() {
    const r = results[k];
    highlight.setAttribute("x", G.x0 + k * slot + 3);
    const lo = Math.max(0, ep.pos - VIEW), hi = Math.min(N - 1, ep.pos + VIEW);
    view.setAttribute("x", C.x0 + lo * C.w + 3);
    view.setAttribute("width", (hi - lo + 1) * C.w - 6);
    coinDot.setAttribute("cx", CX(ep.coin));
    coinDot.style.display = ep.done === "got the coin" ? "none" : "";
    agentDot.setAttribute("cx", CX(ep.pos));
    const side = ep.coin < START ? "left" : "right";
    out.innerHTML =
      `random coin in ${pct(SHARES[k])} of training levels\n` +
      `runs with w_coin > w_end: ${r.wins} of ${RUNS}\n` +
      `average w_coin ${fmt(r.wCoin)}, w_end ${fmt(r.wEnd)}\n` +
      `training, coin at end: gets it ${fmt(r.train)}\n` +
      `test, coin on left:    <span class="orange">gets it ${fmt(r.coin)}</span>\n` +
      `                       <span class="blue">at the end ${fmt(r.end)}</span>\n` +
      `run ${ep.run + 1}: w_coin ${fmt(ep.w[0])}, w_end ${fmt(ep.w[1])}\n` +
      `coin to the ${side}\n` +
      `step ${ep.t} of ${STEPS}${ep.done ? `: ${ep.done}` : ""}`;
  }

  slider(ctl, { label: "random coins", min: 0, max: SHARES.length - 1, step: 1, value: 0, format: (v) => pct(SHARES[v]) }, (v) => {
    k = v;
    newLevel();
    draw();
  });

  // One step every 0.35 s; after an episode ends, wait 1.2 s and start a new test level.
  const player = loop(figure, (dt) => {
    clock += dt;
    if (ep.done ? clock < 1.2 : clock < 0.35) return;
    if (ep.done) newLevel();
    else { step(); clock = 0; }
    draw();
  });
  button(btns, "New level", () => {
    newLevel();
    draw();
  });
  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
})();
