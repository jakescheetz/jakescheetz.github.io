// Unit 09: temperature, top-k and top-p over a fixed next-token distribution, plus sampling.
(function () {
  const figure = document.getElementById("sampler");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, fmt, reducedMotion } = Learn;

  // The same eight candidates and logits as the text, after "The cat sat on the".
  const tokens = ["mat", "floor", "bed", "couch", "roof", "table", "dog", "moon"];
  const logits = [3.0, 2.3, 1.9, 1.4, 0.8, 0.3, -0.5, -1.5];
  const SAMPLES = 20;

  const width = 720, rows = 16, ROW = Learn.ROW;
  const header = 1.5 * ROW;          // band for the column labels
  const pitch = 1.75 * ROW;          // one token per 1.75 lines, so phone-sized labels don't touch
  const barX = 110, barMax = 300;    // a bar of length barMax is probability 1
  const countX = 520, cell = 10;     // one square per sample, up to 20 in a row
  const rowY = (i) => header + pitch * (i + 0.5);

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Probability bars for eight candidate next tokens, with cut tokens greyed out and sample counts on the right");
  svg("text", { x: 0, y: header / 2, "dominant-baseline": "central", class: "muted", text: "token" }, root);
  svg("text", { x: barX, y: header / 2, "dominant-baseline": "central", class: "muted", text: "probability" }, root);
  svg("text", { x: countX, y: header / 2, "dominant-baseline": "central", class: "muted", text: "samples" }, root);

  const rowsUI = tokens.map((token, i) => {
    const y = rowY(i);
    const group = svg("g", {}, root);
    const label = svg("text", { x: 0, y, "dominant-baseline": "central", text: token }, group);
    // Faint bar: probability after temperature, before any cut. Solid bar: what actually gets sampled.
    // The tick on a kept token's bar marks where it ended before renormalizing.
    const before = svg("rect", { x: barX, y: y - 7, height: 14, class: "muted faint", fill: "currentColor" }, group);
    const after = svg("rect", { x: barX, y: y - 7, height: 14, class: "blue", fill: "currentColor" }, group);
    const tick = svg("line", { y1: y - 10, y2: y + 10, class: "ink thin" }, group);
    const value = svg("text", { y, "dominant-baseline": "central" }, group);
    const counts = svg("g", { class: "orange" }, group);
    return { y, label, before, after, tick, value, counts };
  });

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  let T = 1, k = 8, topP = 1;
  let draws = [];        // indices sampled so far in the current batch
  let pending = 0;       // timer of a batch still being drawn

  function softmax(values) {
    const m = Math.max(...values);
    const e = values.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  // Temperature, then top-k, then top-p on what top-k left. Tokens are already sorted by logit.
  function distribution() {
    const p = softmax(logits.map((z) => z / T));
    const keep = p.map((_, i) => i < k);
    let mass = 0;
    for (let i = 0; i < k; i++) mass += p[i];
    let running = 0;
    for (let i = 0; i < k; i++) {
      if (running >= topP - 1e-9) keep[i] = false;   // the set already reached p
      running += p[i] / mass;
    }
    const kept = p.map((v, i) => (keep[i] ? v : 0));
    const keptMass = kept.reduce((a, b) => a + b, 0);
    return { p, keep, keptMass, q: kept.map((v) => v / keptMass) };
  }

  function pick(q) {
    let u = Math.random(), i = 0;
    while (i < q.length - 1 && (u -= q[i]) >= 0) i++;
    while (q[i] === 0) i--;            // guard against rounding landing on a cut token
    return i;
  }

  function drawCounts() {
    const tally = tokens.map(() => 0);
    for (const i of draws) tally[i]++;
    rowsUI.forEach((row, i) => {
      row.counts.replaceChildren();
      for (let c = 0; c < tally[i]; c++) {
        svg("rect", { x: countX + c * cell, y: row.y - 7, width: cell - 2, height: 14, fill: "currentColor" }, row.counts);
      }
    });
    return tally;
  }

  function draw() {
    const { p, keep, keptMass, q } = distribution();
    rowsUI.forEach((row, i) => {
      row.before.setAttribute("width", p[i] * barMax);
      row.after.setAttribute("width", q[i] * barMax);
      const tickX = barX + p[i] * barMax;
      row.tick.setAttribute("x1", tickX);
      row.tick.setAttribute("x2", tickX);
      // Hide the tick when renormalizing barely moves the bar (nothing cut, or a sliver).
      row.tick.style.display = keep[i] && (q[i] - p[i]) * barMax > 2 ? "" : "none";
      row.value.setAttribute("x", barX + Math.max(p[i], q[i]) * barMax + 8);
      row.value.textContent = keep[i] ? `${fmt(100 * q[i], q[i] < 0.095 ? 1 : 0)}%` : "cut";
      row.value.setAttribute("class", keep[i] ? "ink" : "muted");
      row.label.setAttribute("class", keep[i] ? "ink" : "muted faint");
    });
    const tally = drawCounts();
    const nKept = keep.filter(Boolean).length;
    const sequence = draws.map((i) => tokens[i]).join(" ");
    out.innerHTML =
      `T = ${fmt(T)}   top-k = ${k}   top-p = ${fmt(topP)}\n` +
      `<span class="blue">kept ${nKept} of 8 tokens</span>, holding ${fmt(keptMass, 3)} of the probability before renormalizing\n` +
      (draws.length
        ? `<span class="orange">${sequence}</span>\n` +
          tokens.map((t, i) => `${t} ${tally[i]}`).join("  ")
        : `press "Sample ${SAMPLES}" to draw ${SAMPLES} tokens from the blue bars`);
  }

  function stopBatch() {
    clearTimeout(pending);
    pending = 0;
  }

  function sampleBatch() {
    stopBatch();
    const { q } = distribution();
    draws = [];
    if (reducedMotion) {
      for (let s = 0; s < SAMPLES; s++) draws.push(pick(q));
      draw();
      return;
    }
    // One draw every 60 ms so you can watch the counts build up.
    const next = () => {
      draws.push(pick(q));
      draw();
      pending = draws.length < SAMPLES ? setTimeout(next, 60) : 0;
    };
    next();
  }

  // Changing the distribution makes old counts meaningless, so clear them.
  function changed() {
    stopBatch();
    draws = [];
    draw();
  }

  const tControl = slider(ctl, { label: "temperature", min: 0.1, max: 3, step: 0.05, value: T, format: (v) => v.toFixed(2) }, (v) => { T = v; changed(); });
  const kControl = slider(ctl, { label: "top-k", min: 1, max: 8, step: 1, value: k, format: (v) => String(v) }, (v) => { k = v; changed(); });
  const pControl = slider(ctl, { label: "top-p", min: 0.05, max: 1, step: 0.01, value: topP, format: (v) => v.toFixed(2) }, (v) => { topP = v; changed(); });

  button(btns, `Sample ${SAMPLES}`, sampleBatch);
  button(btns, "Reset", () => {
    tControl.set(1);
    kControl.set(8);
    pControl.set(1);
  });
})();
