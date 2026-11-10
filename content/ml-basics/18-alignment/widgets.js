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

// Step-through: the corridor. In training two goals give the same walk; on a test level they split.
(function () {
  const CX = (cell) => 68 + 56 * cell + 28;
  Learn.demo(document.getElementById("demo-corridor"), {
    height: 310,
    label: "A corridor of 9 cells with an agent and a coin, first on a training level and then on a test level",
    steps: [
      { s: { pos: 4, coin: 8, test: 0, goals: 0, r0: 0, r2: 0 }, cap: "The level is a corridor of 9 cells. The agent (blue square) starts at cell 4, and the level ends at cell 8. In training the coin (orange) sits at cell 8." },
      { s: { pos: 8 }, cap: "The agent steps right four times and reaches the coin. Its reward is 1." },
      { s: { goals: 1 }, cap: "Two different goals explain this walk: \"go to the coin\" and \"go to the end\". Here they are the same walk and earn the same reward, so the reward can't show which one the agent learned." },
      { s: { pos: 4, coin: 1, test: 1 }, cap: "A test level: the coin moves to cell 1, left of the start. Now the two goals point in opposite directions." },
      { s: { pos: 8, r0: 1 }, cap: "Agents trained with the coin always at the end mostly walk right anyway. Across 20 runs they get a left-side coin 25% of the time and are standing at the end 60% of the time." },
      { s: { pos: 1, r0: 0, r2: 1 }, cap: "Put the coin in a random cell on just 2% of training levels. Now the 20 runs get a left-side coin 92% of the time. A few levels where the goals disagree are enough." },
    ],
    draw(ctx, s, c, d) {
      const test = s.test > 0.5;
      d.text(test ? "test level: coin on the left" : "training level: coin at the end", 68, 44, { c: c.ink, w: 700 });
      for (let i = 0; i < 9; i++) {
        d.rect(68 + 56 * i, 120, 56, 56, c.bg, c.muted, 1.5);
        d.text(String(i), CX(i), 192, { c: c.muted, s: 12, a: "center" });
      }
      d.rect(68 + 56 * 8, 120, 56, 56, null, c.blue, 3);
      d.text("end", CX(8), 104, { c: c.blue, s: 13, a: "center", w: 700 });
      d.text("start", CX(4), 210, { c: c.muted, s: 12, a: "center" });
      d.rect(CX(s.pos) - 16, 132, 32, 32, c.blue);
      d.dot(CX(s.coin), 148, 10, c.orange);
      ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(CX(s.coin), 148, 10, 0, 7); ctx.stroke();
      if (s.goals > 0.01) {
        ctx.globalAlpha = s.goals;
        d.text(`goal "go to the coin":  walk ${s.coin < 4 ? "left" : "right"}`, 68, 238, { c: c.orange, s: 13, w: 700 });
        d.text(`goal "go to the end":   walk right`, 68, 260, { c: c.blue, s: 13, w: 700 });
      }
      const result = (alpha, str, color) => { if (alpha > 0.02) { ctx.globalAlpha = alpha; d.text(str, 68, 290, { c: color, s: 13, w: 700 }); } };
      result(s.r0, "coin always at the end in training: coin 25%, at end 60%", c.red);
      result(s.r2, "2% random coins in training: coin 92%, at end 6%", c.green);
      ctx.globalAlpha = 1;
    },
  });
})();

// Step-through: the two-weight policy as a tug of war between w_end and w_coin × c.
(function () {
  const X = (z) => 60 + (z + 5) * 36, sig = (t) => 1 / (1 + Math.exp(-t));
  Learn.demo(document.getElementById("demo-weights"), {
    height: 300,
    label: "The policy's score as two arrows added on a number line, and the resulting chance of stepping right",
    steps: [
      { s: { wc: 1.63, we: 2.24, c: 1, ar: 0 }, cap: "After training with the coin always at the end, the average weights over 20 runs are w_coin = 1.63 and w_end = 2.24. The push toward the end is the bigger one." },
      { s: { ar: 1 }, cap: "Coin in view to the right, so c = 1. Both arrows point right: 2.24 + 1.63 = 3.87, and σ(3.87) = 0.98. The agent steps right 98% of the time." },
      { s: { c: -1 }, cap: "Coin in view to the left, so c = −1. The coin arrow flips, but it is the shorter one: 2.24 − 1.63 = 0.61, and σ(0.61) = 0.65. The agent still steps right, away from the coin, 65% of the time." },
      { s: { wc: 3.07, we: 1.49 }, cap: "Train with 2% random coins instead. Now w_coin = 3.07 and w_end = 1.49. With the coin on the left: 1.49 − 3.07 = −1.58, and σ(−1.58) = 0.17. The agent steps left, toward the coin, 83% of the time." },
      { s: { c: 1 }, cap: "Same weights, coin on the right: 1.49 + 3.07 = 4.56, and σ(4.56) = 0.99. On levels like the training ones, both sets of weights walk right. Only the coin-left levels tell them apart." },
    ],
    draw(ctx, s, c, d) {
      const score = s.wc * s.c + s.we, p = sig(score);
      d.text("score = w_end + w_coin × c", 60, 30, { c: c.muted, s: 12 });
      d.line(X(-5), 150, X(5), 150, c.muted, 1.5);
      for (let z = -4; z <= 4; z += 2) { d.line(X(z), 145, X(z), 155, c.muted, 1.5); d.text(d.fmt(z, 0), X(z), 168, { c: c.muted, s: 12, a: "center" }); }
      if (s.ar > 0.01) {
        ctx.globalAlpha = s.ar;
        d.arrow(X(0), 122, X(s.we), 122, c.blue, 4);
        d.text(`w_end = ${d.fmt(s.we, 2)}`, X(0) + 2, 106, { c: c.blue, s: 12, w: 700 });
        d.arrow(X(s.we), 84, X(score), 84, c.orange, 4);
        d.text(`w_coin × c = ${d.fmt(s.wc * s.c, 2)}`, (X(s.we) + X(score)) / 2, 66, { c: c.orange, s: 12, a: "center", w: 700 });
        d.line(X(score), 84, X(score), 150, c.muted, 1.5, [3, 4]);
        d.dot(X(score), 150, 6, c.ink);
        d.text(`step left ${d.fmt(1 - p, 2)}`, 60, 198, { c: c.purple, s: 13, w: 700 });
        d.text(`step right ${d.fmt(p, 2)}`, 420, 198, { c: c.blue, s: 13, w: 700, a: "right" });
        d.rect(60, 212, 360 * (1 - p), 30, c.purple);
        d.rect(60 + 360 * (1 - p), 212, 360 * p, 30, c.blue);
        d.rect(60, 212, 360, 30, null, c.ink, 2);
        d.text(s.c > 0 ? "coin in view to the right (c = 1)" : "coin in view to the left (c = −1)", 240, 270, { c: c.ink, s: 13, a: "center" });
      }
      ctx.globalAlpha = 1;
      d.panel(c, 444, 30, 184, s.ar > 0.01 ? 190 : 100);
      d.text(s.wc < 2.35 ? "coin always at end" : "2% random coins", 458, 50, { c: c.muted, s: 12 });
      d.text(`w_coin   ${d.fmt(s.wc, 2)}`, 458, 76, { c: c.orange, w: 700 });
      d.text(`w_end    ${d.fmt(s.we, 2)}`, 458, 102, { c: c.blue, w: 700 });
      if (s.ar > 0.01) {
        ctx.globalAlpha = s.ar;
        d.text(`c        ${d.fmt(s.c, 0)}`, 458, 128, { c: c.ink });
        d.line(458, 146, 614, 146, c.muted, 1);
        d.text(`score    ${d.fmt(score, 2)}`, 458, 166, { c: c.ink });
        d.text(`P(right) ${d.fmt(p, 2)}`, 458, 196, { c: c.ink, w: 700 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: performance gap recovered (PGR) on a number line, with made-up round scores.
(function () {
  const X = (v) => 80 + (v - 50) * 10, WEAK = 55, CEIL = 95;
  Learn.demo(document.getElementById("demo-pgr"), {
    height: 280,
    label: "A score line from 50% to 100% with a weak supervisor at 55%, a strong ceiling at 95%, and a student in between",
    steps: [
      { s: { ce: 0, stu: 0, st: 75 }, cap: "Say a weak supervisor scores 55% on a task. It labels the training data, mistakes included." },
      { s: { ce: 1 }, cap: "Fine-tuned on correct labels, the strong model scores 95%. That is its ceiling. The gap between the supervisor and the ceiling is 40 points." },
      { s: { stu: 1 }, cap: "Now fine-tune the strong model on the weak supervisor's labels. Say this student scores 75%. It closed 20 of the 40 points, so PGR = 20 ÷ 40 = 0.5." },
      { s: { st: 55 }, cap: "If the student scored 55%, the same as its supervisor, it closed 0 of the 40 points. PGR = 0." },
      { s: { st: 95 }, cap: "If the student scored 95%, it did as well as it would with correct labels. PGR = 1." },
      { s: { st: 87 }, cap: "A student at 87% closed 32 of the 40 points, so PGR = 0.8. Burns et al.'s extra loss reached nearly this on their language tasks." },
    ],
    draw(ctx, s, c, d) {
      d.line(X(50), 150, X(100), 150, c.muted, 1.5);
      for (let v = 50; v <= 100; v += 10) { d.line(X(v), 145, X(v), 155, c.muted, 1.5); d.text(v + "%", X(v), 170, { c: c.muted, s: 12, a: "center" }); }
      d.line(X(WEAK), 108, X(WEAK), 150, c.blue, 3);
      d.text("weak supervisor", X(WEAK), 78, { c: c.blue, s: 12, a: "center", w: 700 });
      d.text("55%", X(WEAK), 96, { c: c.blue, s: 13, a: "center", w: 700 });
      if (s.ce > 0.01) {
        ctx.globalAlpha = s.ce;
        d.line(X(CEIL), 108, X(CEIL), 150, c.ink, 3);
        d.text("strong ceiling", X(CEIL), 78, { c: c.ink, s: 12, a: "center", w: 700 });
        d.text("95%", X(CEIL), 96, { c: c.ink, s: 13, a: "center", w: 700 });
        d.line(X(WEAK) + 6, 126, X(CEIL) - 6, 126, c.muted, 1.5, [5, 4]);
        d.text("gap: 40 points", (X(WEAK) + X(CEIL)) / 2, 112, { c: c.muted, s: 12, a: "center" });
      }
      if (s.stu > 0.01) {
        ctx.globalAlpha = s.stu;
        d.line(X(WEAK), 150, X(s.st), 150, c.orange, 8);
        d.dot(X(s.st), 150, 8, c.orange);
        d.text(`student ${d.fmt(s.st, 0)}%`, X(s.st), 196, { c: c.orange, s: 13, a: "center", w: 700 });
        d.text(`PGR = (${d.fmt(s.st, 0)} − 55) ÷ (95 − 55) = ${d.fmt((s.st - WEAK) / (CEIL - WEAK), 2)}`, 320, 244, { c: c.ink, s: 16, a: "center", w: 700 });
      }
      ctx.globalAlpha = 1;
    },
  });
})();
