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
      `rolls N = ${N.toLocaleString("en-US")}   ` +
      (N ? `<span class="blue">running mean = ${fmt(mean, 3)}</span>   miss = ${err >= 0 ? "+" : ""}${fmt(err, 3)}` : "no rolls yet") + "\n" +
      (N ? `typical miss σ/√N = ${fmt(die.sd)}/√${N.toLocaleString("en-US")} = ${fmt(die.sd / Math.sqrt(N), 3)}` : "typical miss σ/√N appears after the first roll");
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
      `1,000 emails: <span class="orange">${nPhish} phishing, ${nCaught} flagged</span>; ` +
      `${nLegit} legit, <span class="blue">${nFalse} flagged</span>\n` +
      `P(flag) = ${fmt(hit)} × ${fmt(prior, 3)} + ${fmt(falseAlarm, 3)} × ${fmt(1 - prior, 3)} = ${fmt(pFlag, 4)}\n` +
      `P(phish | flag) = ${fmt(flagPhish, 4)} / ${fmt(pFlag, 4)} = <span class="orange">${fmt(posterior, 3)}</span>`;
  }

  slider(ctl, { label: "base rate", min: 0.005, max: 0.5, step: 0.005, value: prior, format: pct }, (v) => { prior = v; draw(); });
  slider(ctl, { label: "catch rate", min: 0.5, max: 1, step: 0.01, value: hit, format: pct }, (v) => { hit = v; draw(); });
  slider(ctl, { label: "false alarms", min: 0, max: 0.3, step: 0.005, value: falseAlarm, format: pct }, (v) => { falseAlarm = v; draw(); });
})();
