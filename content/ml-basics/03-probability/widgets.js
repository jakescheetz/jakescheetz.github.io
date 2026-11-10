// Unit 03: a die's running mean closing in on E[X], and Bayes' rule counted out on 1,000 emails.
(function () {
  const figure = document.getElementById("dice");
  if (!figure) return;
  const { svg, stage, buttons, button, readout, loop, fmt } = Learn;

  const width = 720, rows = 14, height = rows * Learn.ROW;
  const top = 40, bottom = 214;
  const maxN = 10000;

  const dice = {
    fair: { name: "fair die", p: [1, 1, 1, 1, 1, 1].map((v) => v / 6) },
    loaded: { name: "loaded die", p: [0.1, 0.1, 0.1, 0.1, 0.1, 0.5] },
  };
  for (const d of Object.values(dice)) {
    d.mean = d.p.reduce((s, q, i) => s + q * (i + 1), 0);
    d.sd = Math.sqrt(d.p.reduce((s, q, i) => s + q * (i + 1 - d.mean) ** 2, 0));
  }
  let die = dice.fair;

  // Left: share of rolls per face (blue bars) against the true probabilities (orange ticks).
  const barW = 30, barGap = 14, hx0 = 40;
  const pMax = 0.6;
  const BX = (i) => hx0 + i * (barW + barGap);
  const HY = (q) => bottom - (q / pMax) * (bottom - top);
  // Right: running mean against the number of rolls, on a log scale from 1 to 10,000.
  const rx0 = 390, rx1 = 700;
  const RX = (n) => rx0 + (Math.log10(n) / 4) * (rx1 - rx0);
  const RY = (m) => bottom - ((m - 1) / 5) * (bottom - top);

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Histogram of die rolls next to the running mean of the rolls, which closes in on the expected value as rolls pile up");

  svg("text", { x: hx0, y: 18, class: "muted", text: "share of rolls" }, root);
  svg("line", { x1: hx0 - 8, y1: bottom, x2: BX(5) + barW + 8, y2: bottom, class: "muted thin" }, root);
  const bars = [], ticks = [];
  for (let i = 0; i < 6; i++) {
    bars.push(svg("rect", { x: BX(i), width: barW, y: bottom, height: 0, class: "blue", fill: "currentColor" }, root));
    ticks.push(svg("line", { x1: BX(i) - 4, x2: BX(i) + barW + 4, class: "orange" }, root));
    svg("text", { x: BX(i) + barW / 2, y: bottom + 22, "text-anchor": "middle", text: String(i + 1) }, root);
  }

  svg("text", { x: rx0, y: 18, class: "muted", text: "running mean" }, root);
  svg("line", { x1: rx0, y1: bottom, x2: rx1, y2: bottom, class: "muted thin" }, root);
  svg("line", { x1: rx0, y1: top, x2: rx0, y2: bottom, class: "muted thin" }, root);
  [[1, "1"], [10, "10"], [100, "100"], [1000, "1k"], [10000, "10k"]].forEach(([n, label]) => {
    svg("line", { x1: RX(n), y1: bottom, x2: RX(n), y2: bottom + 5, class: "muted thin" }, root);
    svg("text", { x: RX(n), y: bottom + 22, "text-anchor": "middle", class: "muted", text: label }, root);
  });
  // Set back from the axis so this "1" doesn't run into the x-axis "1" when labels scale up on phones.
  [1, 6].forEach((m) => svg("text", { x: rx0 - 14, y: RY(m) + 4, "text-anchor": "end", class: "muted", text: String(m) }, root));
  const meanLine = svg("line", { x1: rx0, x2: rx1, class: "orange dashed" }, root);
  const meanLabel = svg("text", { x: rx1, "text-anchor": "end", class: "orange", text: "E[X]" }, root);
  const funnel = [svg("polyline", { class: "muted dashed thin" }, root), svg("polyline", { class: "muted dashed thin" }, root)];
  const path = svg("polyline", { class: "blue" }, root);
  const dot = svg("circle", { r: 4, class: "blue", fill: "currentColor" }, root);

  const out = readout(figure);
  const btns = buttons(figure);

  let counts, N, sum, points, pending;
  function reset() {
    counts = [0, 0, 0, 0, 0, 0];
    N = 0; sum = 0; points = []; pending = 0;
  }
  reset();

  function drawFunnel() {
    const clamp = (m) => Math.max(1, Math.min(6, m));
    for (const [k, sign] of [[0, 1], [1, -1]]) {
      const pts = [];
      for (let e = 0; e <= 4.0001; e += 0.05) {
        const n = 10 ** e;
        pts.push(`${RX(n).toFixed(1)},${RY(clamp(die.mean + sign * die.sd / Math.sqrt(n))).toFixed(1)}`);
      }
      funnel[k].setAttribute("points", pts.join(" "));
    }
    meanLine.setAttribute("y1", RY(die.mean));
    meanLine.setAttribute("y2", RY(die.mean));
    meanLabel.setAttribute("y", RY(die.mean) - 8);
    die.p.forEach((q, i) => { ticks[i].setAttribute("y1", HY(q)); ticks[i].setAttribute("y2", HY(q)); });
  }

  function rollOnce() {
    let u = Math.random(), face = 5;
    for (let i = 0; i < 6; i++) {
      if (u < die.p[i]) { face = i; break; }
      u -= die.p[i];
    }
    counts[face]++;
    N++;
    sum += face + 1;
    // Keep a point whenever the path has moved at least a unit to the right, so 10,000 rolls stay cheap to draw.
    const x = RX(N);
    if (!points.length || x - points[points.length - 1][0] >= 1) points.push([x, RY(sum / N)]);
  }

  function roll(k) {
    for (let j = 0; j < k && N < maxN; j++) rollOnce();
  }

  function draw() {
    counts.forEach((c, i) => {
      const share = N ? c / N : 0;
      bars[i].setAttribute("y", HY(share));
      bars[i].setAttribute("height", bottom - HY(share));
    });
    const current = N ? [RX(N), RY(sum / N)] : null;
    const pts = current ? points.concat([current]) : [];
    path.setAttribute("points", pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" "));
    dot.style.display = current ? "" : "none";
    if (current) { dot.setAttribute("cx", current[0]); dot.setAttribute("cy", current[1]); }

    const mean = N ? sum / N : NaN;
    const err = mean - die.mean;
    out.innerHTML =
      `${die.name}   <span class="orange">E[X] = ${fmt(die.mean)}</span>   σ = ${fmt(die.sd)}\n` +
      `rolls N = ${N.toLocaleString("en-US")}\n` +
      (N ? `<span class="blue">running mean = ${fmt(mean, 3)}</span>   miss = ${err >= 0 ? "+" : ""}${fmt(err, 3)}` : "no rolls yet") + "\n" +
      (N ? `typical miss σ/√N = ${fmt(die.sd / Math.sqrt(N), 3)}\n= ${fmt(die.sd)}/√${N.toLocaleString("en-US")}` : "typical miss σ/√N appears\nafter the first roll");
  }

  drawFunnel();

  const player = loop(figure, (dt) => {
    if (dt) {
      // Roll faster as N grows, so each tenfold step on the log axis takes about the same time.
      pending += Math.max(5, N * 0.75) * dt;
      const k = Math.floor(pending);
      pending -= k;
      roll(k);
      if (N >= maxN) {
        player.pause();
        toggle.textContent = "Play";
      }
    }
    draw();
  });

  const pause = () => { player.pause(); toggle.textContent = "Play"; };
  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) { pause(); return; }
    if (N >= maxN) reset();
    player.play();
    toggle.textContent = "Pause";
  });
  button(btns, "Roll 1", () => { pause(); roll(1); draw(); });
  button(btns, "Roll 100", () => { pause(); roll(100); draw(); });
  button(btns, "Reset", () => { reset(); draw(); });
  const which = button(btns, "Use loaded die", () => {
    die = die === dice.fair ? dice.loaded : dice.fair;
    which.textContent = die === dice.fair ? "Use loaded die" : "Use fair die";
    reset();
    drawFunnel();
    draw();
  });
})();

(function () {
  const figure = document.getElementById("bayes");
  if (!figure) return;
  const { svg, stage, controls, slider, readout, fmt } = Learn;

  const width = 720, rows = 14;
  const cols = 50, gridRows = 20, cell = 12, total = cols * gridRows;
  const x0 = (width - cols * cell) / 2, y0 = 18;

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "1,000 emails as squares, colored by whether they are phishing and whether the filter flagged them");

  // Column by column: missed phishing, flagged phishing, flagged legit, then the unflagged legit rest.
  // That keeps every flagged email in one run, so you can compare orange to blue inside it.
  const cells = [];
  for (let i = 0; i < total; i++) {
    const c = Math.floor(i / gridRows), r = i % gridRows;
    cells.push(svg("rect", { x: x0 + c * cell + 1, y: y0 + r * cell + 1, width: cell - 2, height: cell - 2, fill: "currentColor" }, root));
  }

  const out = readout(figure);
  const ctl = controls(figure);
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  let prior = 0.01, hit = 0.9, falseAlarm = 0.05;

  function draw() {
    const nPhish = Math.round(total * prior);
    const nCaught = Math.min(nPhish, Math.round(total * prior * hit));
    const nLegit = total - nPhish;
    const nFalse = Math.min(nLegit, Math.round(nLegit * falseAlarm));
    const missed = nPhish - nCaught;
    cells.forEach((rect, i) => {
      let cls;
      if (i < missed) cls = "orange faint";
      else if (i < nPhish) cls = "orange";
      else if (i < nPhish + nFalse) cls = "blue";
      else cls = "muted faint";
      rect.setAttribute("class", cls);
    });

    const flagPhish = hit * prior, flagLegit = falseAlarm * (1 - prior);
    const pFlag = flagPhish + flagLegit;
    // The sliders keep the base rate and catch rate above zero, so P(flag) never is.
    const posterior = flagPhish / pFlag;
    out.innerHTML =
      `<span class="orange">phishing: ${nPhish} of 1,000, ${nCaught} flagged</span>\n` +
      `legit:    ${nLegit} of 1,000, <span class="blue">${nFalse} flagged</span>\n` +
      `P(flag) = ${fmt(hit)} × ${fmt(prior, 3)}\n        + ${fmt(falseAlarm, 3)} × ${fmt(1 - prior, 3)} = ${fmt(pFlag, 4)}\n` +
      `P(phish|flag) = ${fmt(flagPhish, 4)} / ${fmt(pFlag, 4)} = <span class="orange">${fmt(posterior, 3)}</span>`;
  }

  slider(ctl, { label: "base rate", min: 0.005, max: 0.5, step: 0.005, value: prior, format: pct }, (v) => { prior = v; draw(); });
  slider(ctl, { label: "catch rate", min: 0.5, max: 1, step: 0.01, value: hit, format: pct }, (v) => { hit = v; draw(); });
  slider(ctl, { label: "false alarms", min: 0, max: 0.3, step: 0.005, value: falseAlarm, format: pct }, (v) => { falseAlarm = v; draw(); });
})();

// A small seeded generator, so the step-through pictures are the same on every visit.
function demoRng(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Step-through: the running average of die rolls settling toward 3.5 inside the σ/√N funnel.
(function () {
  const random = demoRng(15), N = 10000, avg = new Float64Array(N + 1), sigma = Math.sqrt(17.5 / 6);
  let sum = 0;
  for (let i = 1; i <= N; i++) { sum += 1 + Math.floor(random() * 6); avg[i] = sum / i; }
  const X = (lg) => 70 + lg / 4 * 520, Y = (v) => 300 - (v - 2) / 3 * 265;
  const miss = (n) => (sigma / Math.sqrt(n)).toFixed(3);
  Learn.demo(document.getElementById("demo-montecarlo"), {
    height: 350,
    label: "The running average of die rolls settling toward 3.5 inside a narrowing funnel",
    steps: [
      { s: { lg: 1, fun: 0 }, cap: `Roll a fair die and keep a running average. The dashed line is the true mean, 3.5. After 10 rolls the average is ${avg[10].toFixed(2)}, and it still jumps around a lot.` },
      { s: { lg: 2 }, cap: `After 100 rolls the average is ${avg[100].toFixed(2)}. At this point it usually lands within about 0.17 of 3.5.` },
      { s: { lg: 3, fun: 1 }, cap: `After 1,000 rolls it reads ${avg[1000].toFixed(2)}. The shaded funnel shows the typical miss, σ ÷ √N. It narrows as N grows.` },
      { s: { lg: 4 }, cap: `After 10,000 rolls it reads ${avg[10000].toFixed(3)}. The typical miss is now only about 0.017.` },
      { s: { lg: 4.0001 }, cap: "Compare 100 rolls with 10,000 rolls. 100 times more rolls gave only 10 times more accuracy. That is the square-root rule." },
    ],
    draw(ctx, s, c, d) {
      const n = Math.min(N, Math.max(1, Math.round(Math.pow(10, s.lg))));
      ctx.save(); ctx.beginPath(); ctx.rect(X(0), Y(5), X(4) - X(0), Y(2) - Y(5)); ctx.clip();
      if (s.fun > 0.01) {
        ctx.globalAlpha = s.fun * 0.3; ctx.fillStyle = c.blue; ctx.beginPath();
        for (let lg = 0; lg <= 4.0001; lg += 0.02) { const e = sigma / Math.sqrt(Math.pow(10, lg)); if (lg === 0) ctx.moveTo(X(lg), Y(3.5 + e)); else ctx.lineTo(X(lg), Y(3.5 + e)); }
        for (let lg = 4; lg >= -0.0001; lg -= 0.02) { const l = Math.max(0, lg); ctx.lineTo(X(l), Y(3.5 - sigma / Math.sqrt(Math.pow(10, l)))); }
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      }
      d.line(X(0), Y(3.5), X(4), Y(3.5), c.ink, 1.5, [6, 5]);
      ctx.strokeStyle = c.orange; ctx.lineWidth = 2; ctx.beginPath();
      for (let k = 1; k <= n; k++) { const x = X(Math.log10(k)), y = Y(avg[k]); if (k === 1) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.stroke();
      ctx.restore();
      d.dot(X(Math.log10(n)), Y(avg[n]), 5, c.orange);
      d.line(X(0), Y(2), X(4), Y(2), c.muted, 1.5);
      d.line(X(0), Y(2), X(0), Y(5), c.muted, 1.5);
      ["1", "10", "100", "1,000", "10,000"].forEach((name, k) => { d.line(X(k), Y(2), X(k), Y(2) + 5, c.muted); d.text(name, X(k), Y(2) + 17, { c: c.muted, s: 12, a: "center" }); });
      for (const v of [2, 3, 3.5, 4, 5]) d.text(String(v), X(0) - 8, Y(v), { c: v === 3.5 ? c.ink : c.muted, s: 12, a: "right", w: v === 3.5 ? 700 : 500 });
      d.text("number of rolls (N)", X(2), 338, { c: c.muted, s: 12, a: "center" });
      d.panel(c, 388, 40, 240, 88);
      d.text(`rolls N = ${n.toLocaleString("en-US")}`, 402, 60, { c: c.ink, s: 13 });
      d.text(`average = ${avg[n].toFixed(3)}`, 402, 84, { c: c.orange, s: 13, w: 700 });
      d.text(`typical miss = ${miss(n)}`, 402, 108, { c: c.blue, s: 13 });
    },
  });
})();

// Step-through: Bayes' rule as counts on a grid of 10,000 emails.
(function () {
  // 0 normal, 1 normal and flagged, 2 phishing and missed, 3 phishing and caught.
  const type = new Uint8Array(10000), normals = [];
  for (let r = 0; r < 100; r++) for (let k = 0; k < 100; k++) {
    const i = r * 100 + k;
    if (r < 10 && k < 10) type[i] = r < 9 ? 3 : 2; else normals.push(i);
  }
  const random = demoRng(5);
  for (let i = normals.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [normals[i], normals[j]] = [normals[j], normals[i]]; }
  for (let f = 0; f < 495; f++) type[normals[f]] = 1;
  const CELL = 3.1, GX = 14, GY = 20;
  Learn.demo(document.getElementById("demo-bayes"), {
    height: 350,
    label: "10,000 emails as a grid: the phishing ones, the ones the filter caught, the false alarms, and the share of flagged emails that are phishing",
    steps: [
      { s: { pa: 0, ca: 0, fa: 0, dim: 0, res: 0 }, cap: "Each tiny square is one email. There are 10,000 of them." },
      { s: { pa: 1 }, cap: "1% of email is phishing. That is 100 emails, shown in red in the top-left corner." },
      { s: { ca: 1 }, cap: "The filter catches 90% of phishing, so 90 phishing emails get flagged. The 10 it missed fade out." },
      { s: { fa: 1 }, cap: "The filter also flags 5% of normal email by mistake. 5% of 9,900 is 495 false alarms, shown in orange." },
      { s: { dim: 1 }, cap: "Now keep only the flagged emails and ignore the rest. That leaves 90 + 495 = 585 flagged emails." },
      { s: { res: 1 }, cap: "Of the 585 flagged emails, only 90 are phishing. 90 ÷ 585 ≈ 15%. That is P(phish | flag). Most flags are false alarms because phishing is rare." },
    ],
    draw(ctx, s, c, d) {
      const cells = (which, color, alpha) => {
        if (alpha < 0.01) return;
        ctx.globalAlpha = alpha; ctx.fillStyle = color;
        for (let k = 0; k < 10000; k++) if (type[k] === which) ctx.fillRect(GX + (k % 100) * CELL, GY + Math.floor(k / 100) * CELL, CELL - 0.6, CELL - 0.6);
        ctx.globalAlpha = 1;
      };
      const keep = 1 - 0.85 * s.dim;
      cells(0, c.muted, 0.4 * keep);
      cells(1, c.muted, 0.4 * (1 - s.fa));
      cells(1, c.orange, s.fa);
      cells(2, c.muted, 0.4 * (1 - s.pa) * keep);
      cells(2, c.red, s.pa * (1 - 0.75 * s.ca) * keep);
      cells(3, c.muted, 0.4 * (1 - s.pa));
      cells(3, c.red, s.pa);
      const x = 340;
      let y = 36;
      const row = (alpha, swatch, name, value) => {
        ctx.globalAlpha = alpha;
        if (swatch) d.rect(x, y - 6, 12, 12, swatch);
        d.text(name, x + 20, y, { c: c.ink, s: 13 });
        d.text(value, 630, y, { c: c.ink, s: 14, a: "right", w: 700 });
        ctx.globalAlpha = 1; y += 34;
      };
      row(1, c.muted, "All emails", "10,000");
      row(Math.max(0.3, s.pa), c.red, "Phishing (1%)", "100");
      row(Math.max(0.3, s.ca), c.red, "Caught (90%)", "90");
      row(Math.max(0.3, s.fa), c.orange, "False alarms (5%)", "495");
      row(Math.max(0.3, s.dim), null, "All flagged", "585");
      if (s.res > 0.01) {
        ctx.globalAlpha = s.res;
        d.panel(c, 340, 214, 290, 104);
        d.text("P(phish | flag)", 354, 238, { c: c.muted, s: 13 });
        d.text("= 90 ÷ 585", 354, 264, { c: c.ink, s: 16 });
        d.text("≈ 15%", 354, 294, { c: c.red, s: 24, w: 700 });
      }
    },
  });
})();

// Step-through: multiplying probabilities underflows; adding log-probabilities doesn't.
(function () {
  const LOG10_HALF = Math.log10(0.5), LN_HALF = Math.log(0.5);
  const superscript = (n) => String(n).split("").map((ch) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+ch]).join("");
  const product = (n) => {
    if (n <= 3) return String(Math.pow(0.5, n));
    const e = n * LOG10_HALF, exp = Math.floor(e);
    return `${Math.pow(10, e - exp).toFixed(1)} × 10⁻${superscript(-exp)}`;
  };
  Learn.demo(document.getElementById("demo-logprob"), {
    height: 340,
    label: "A product of probabilities shrinking to zero next to a sum of log-probabilities growing steadily",
    steps: [
      { s: { lg: 0 }, cap: "One token with a 0.5 chance. The probability is 0.5. Its log is −0.69." },
      { s: { lg: 1 }, cap: "10 tokens. Multiply 0.5 ten times and you get about 0.001. Add ten logs instead and you get about −6.9." },
      { s: { lg: 2 }, cap: "100 tokens. The product is about 7.9 × 10⁻³¹, far too small for the bar to show. The log sum is −69.3, a normal number." },
      { s: { lg: 3 }, cap: "1,000 tokens. The product is about 9.3 × 10⁻³⁰². That is near the smallest number a computer can store. The log sum is −693.1." },
      { s: { lg: Math.log10(2000) }, cap: "2,000 tokens. The product is too small to store, so the computer rounds it to 0. Every long text would tie at 0. The log sum, −1386.3, still works and ranks texts the same way." },
    ],
    draw(ctx, s, c, d) {
      const n = Math.max(1, Math.round(Math.pow(10, s.lg))), under = n > 1074;
      d.text("Multiply the chances", 160, 22, { c: c.ink, s: 15, w: 700, a: "center" });
      d.text("Add the logs", 480, 22, { c: c.ink, s: 15, w: 700, a: "center" });
      d.text(`tokens: ${n.toLocaleString("en-US")}`, 320, 322, { c: c.muted, a: "center" });
      d.line(320, 14, 320, 300, c.line);
      const base = 290, tall = 210, p = Math.pow(0.5, n);
      d.line(90, base, 230, base, c.muted, 1.5);
      d.rect(110, base - tall * p, 100, tall * p, c.orange);
      ctx.setLineDash([4, 4]); d.rect(110, base - tall, 100, tall, null, c.muted, 1); ctx.setLineDash([]);
      d.text("1.0", 104, base - tall, { c: c.muted, s: 12, a: "right" });
      d.text(under ? "stored as 0" : product(n), 160, base - tall - 18, { c: under ? c.red : c.ink, s: 16, a: "center", w: 700 });
      if (under) d.text("underflow!", 160, base - 40, { c: c.red, s: 16, w: 700, a: "center" });
      const top = 80, deep = 200;
      d.line(410, top, 550, top, c.muted, 1.5);
      d.rect(430, top, 100, deep * n / 2000, c.blue);
      ctx.setLineDash([4, 4]); d.rect(430, top, 100, deep, null, c.muted, 1); ctx.setLineDash([]);
      d.text("0", 402, top, { c: c.muted, s: 12, a: "right" });
      d.text("−1386", 424, top + deep, { c: c.muted, s: 12, a: "right" });
      d.text(`sum = ${d.fmt(n * LN_HALF, 1)}`, 480, top - 22, { c: c.blue, s: 16, a: "center", w: 700 });
    },
  });
})();
