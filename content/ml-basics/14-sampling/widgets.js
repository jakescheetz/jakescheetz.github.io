// Unit 14: temperature, top-k and top-p over a fixed next-token distribution, plus sampling.
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
    // Break the sampled tokens into lines of at most 38 characters, so they fit a phone-width readout.
    const sequence = [];
    draws.forEach((i) => {
      const last = sequence.length - 1;
      if (last >= 0 && sequence[last].length + 1 + tokens[i].length <= 38) sequence[last] += ` ${tokens[i]}`;
      else sequence.push(tokens[i]);
    });
    const counts = tokens.map((t, i) => `${t} ${tally[i]}`);
    out.innerHTML =
      `T = ${fmt(T)}   top-k = ${k}   top-p = ${fmt(topP)}\n` +
      `<span class="blue">kept ${nKept} of 8 tokens</span>, holding ${fmt(keptMass, 3)} of\n` +
      `the probability before renormalizing\n` +
      (draws.length
        ? `<span class="orange">${sequence.join("\n")}</span>\n` +
          `${counts.slice(0, 4).join("  ")}\n${counts.slice(4).join("  ")}`
        : `press "Sample ${SAMPLES}" to draw ${SAMPLES} tokens\nfrom the blue bars`);
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

// Step-through: the generation loop. Run the model, pick a token, append it, run again.
(function () {
  const context = ["The", "cat", "sat", "on", "the"];
  const top = [["mat", 0.444], ["floor", 0.221], ["bed", 0.148], ["couch", 0.090]]; // from the unit's table
  const slotX = (i) => 20 + i * 72;
  Learn.demo(document.getElementById("demo-loop"), {
    height: 320,
    label: "A prompt goes into the model, the model gives probabilities, one token is picked and appended to the prompt",
    steps: [
      { s: { run: 0, pick: 0, mv: 0, again: 0, dot: 0 }, cap: "The prompt is 5 tokens: The cat sat on the. The model has not run yet." },
      { s: { run: 1 }, cap: "Run the model on all 5 tokens. Softmax turns the last position's logits into one probability per candidate: mat 0.444, floor 0.221, bed 0.148, and so on." },
      { s: { pick: 1 }, cap: "The decoding strategy picks one token from that list. Here it picks mat. The model only made the list. The picking rule sits outside the model." },
      { s: { mv: 1 }, cap: "Append mat to the context. The context is now 6 tokens long." },
      { s: { again: 1 }, cap: "Run the model again on all 6 tokens. It makes a fresh list of probabilities for the token after mat." },
      { s: { dot: 1 }, cap: "The strategy picks again (say, a period) and appends it. Each new token costs one forward pass. The loop ends at an end-of-text token or a length limit." },
    ],
    draw(ctx, s, c, d) {
      const box = (label, x, y, stroke, fill) => {
        d.rect(x, y, 64, 34, fill || c.bg, stroke, 2);
        d.text(label, x + 32, y + 17, { c: stroke, a: "center", w: 700 });
      };
      d.text("context", 20, 18, { c: c.muted, s: 13 });
      context.forEach((token, i) => box(token, slotX(i), 32, c.ink));
      // The model and the arrows in and out of it.
      d.rect(60, 130, 140, 60, c.soft, c.ink, 2);
      d.text("model", 130, 152, { c: c.ink, s: 16, a: "center", w: 700 });
      d.text("(unit 13)", 130, 173, { c: c.muted, s: 12, a: "center" });
      if (s.run > 0.02) {
        ctx.globalAlpha = s.run;
        d.arrow(130, 70, 130, 126, c.blue, 2.5);
        d.text("all tokens in", 142, 98, { c: c.muted, s: 12 });
        d.arrow(204, 160, 254, 160, c.blue, 2.5);
        d.text("p for the next token", 262, 92, { c: c.muted, s: 13 });
        d.panel(c, 260, 104, 360, 156);
        const first = s.run * (1 - s.again);
        if (first > 0.02) {
          ctx.globalAlpha = first;
          top.forEach(([token, p], j) => {
            const y = 132 + j * 28;
            d.text(token, 276, y, { c: c.ink });
            d.rect(340, y - 8, 240 * p, 16, c.blue);
            d.text(d.fmt(p, 3), 348 + 240 * p, y, { c: c.ink, s: 13 });
          });
          d.text("… and 4 more", 276, 243, { c: c.muted, s: 13 });
          if (s.pick > 0.02) {
            ctx.globalAlpha = first * s.pick;
            d.rect(266, 118, 348, 28, null, c.orange, 2);
          }
        }
        if (s.again > 0.02) {
          ctx.globalAlpha = s.again;
          d.text("a new list of probabilities,", 276, 160, { c: c.ink });
          d.text("this time for token 7", 276, 186, { c: c.ink });
        }
        ctx.globalAlpha = 1;
      }
      // The picked token flies from the list up to the end of the context.
      if (s.pick > 0.02) {
        ctx.globalAlpha = s.mv > 0.02 ? 1 : s.pick;
        box("mat", 270 + (slotX(5) - 270) * s.mv, 115 + (32 - 115) * s.mv, c.orange, c.bg);
        ctx.globalAlpha = 1;
      }
      if (s.dot > 0.02) {
        ctx.globalAlpha = s.dot;
        box(".", slotX(6), 32, c.orange, c.bg);
        ctx.globalAlpha = 1;
      }
      const passes = Math.round(s.run) + Math.round(s.again);
      const length = 5 + Math.round(s.mv) + Math.round(s.dot);
      d.text(`forward passes: ${passes}    context length: ${length} tokens`, 20, 296, { c: c.muted, s: 13 });
    },
  });
})();

// Step-through: temperature stretches or shrinks the gaps between the bars; the order stays.
(function () {
  // The unit's eight candidates and a softmax with temperature.
  const demoTokens = ["mat", "floor", "bed", "couch", "roof", "table", "dog", "moon"];
  const demoLogits = [3.0, 2.3, 1.9, 1.4, 0.8, 0.3, -0.5, -1.5];
  const demoSoftmax = (T) => {
    const e = demoLogits.map((z) => Math.exp((z - demoLogits[0]) / T));
    const total = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / total);
  };
  const base = demoSoftmax(1), X0 = 96, W = 420;
  Learn.demo(document.getElementById("demo-temperature"), {
    height: 330,
    label: "Probability bars for eight candidate tokens as the temperature changes",
    steps: [
      { s: { T: 1, ord: 0 }, cap: "At T = 1 you get the model's own distribution. mat has 0.444, floor has 0.221 and moon has 0.005. mat is 2.01 times as likely as floor." },
      { s: { T: 0.5 }, cap: "Lower T to 0.5. Dividing every logit by 0.5 doubles every gap. mat climbs to 0.706 and floor drops to 0.174. mat is now 4.06 times as likely as floor." },
      { s: { T: 2 }, cap: "Raise T to 2. Every gap is cut in half, so the bars even out. mat falls to 0.278 and moon rises to 0.029. mat is only 1.42 times as likely as floor." },
      { s: { T: 0.1 }, cap: "Push T down to 0.1. mat takes 0.999 of the probability, so nearly every draw picks it. That is why T = 0 is taken to mean greedy decoding." },
      { s: { T: 1, ord: 1 }, cap: "Back to T = 1. Through every change the bars stayed in the same order, top to bottom. Temperature changes the gaps between tokens. It can't change which one is ahead." },
    ],
    draw(ctx, s, c, d) {
      const p = demoSoftmax(s.T);
      d.text(`T = ${d.fmt(s.T, 2)}`, 20, 28, { c: c.orange, s: 20, w: 700 });
      d.text(`mat ÷ floor = ${d.fmt(Math.exp(0.7 / s.T), 2)}`, 620, 28, { c: c.ink, a: "right" });
      demoTokens.forEach((token, i) => {
        const y = 68 + i * 30;
        if (s.ord > 0.02) {
          ctx.globalAlpha = s.ord;
          d.text(String(i + 1), 14, y, { c: c.orange, s: 13, w: 700 });
          ctx.globalAlpha = 1;
        }
        d.text(token, 36, y, { c: c.ink });
        d.rect(X0, y - 9, W * p[i], 18, c.blue);
        d.line(X0 + W * base[i], y - 12, X0 + W * base[i], y + 12, c.ink, 2);
        d.text(d.fmt(p[i], 3), X0 + W * Math.max(p[i], base[i]) + 10, y, { c: c.ink, s: 13 });
      });
      d.line(X0, 50, X0, 296, c.muted, 1.5);
      d.text("tick = the bar at T = 1", 620, 312, { c: c.muted, s: 13, a: "right" });
      d.text(s.T < 0.95 ? "T < 1: sharper" : s.T > 1.05 ? "T > 1: flatter" : "T = 1: the model's own", 20, 312, { c: c.muted, s: 13, w: 700 });
    },
  });
})();

// Step-through: top-k and top-p cut the tail, then the kept bars are renormalized.
(function () {
  // The unit's eight candidates and a softmax with temperature.
  const demoTokens = ["mat", "floor", "bed", "couch", "roof", "table", "dog", "moon"];
  const demoLogits = [3.0, 2.3, 1.9, 1.4, 0.8, 0.3, -0.5, -1.5];
  const demoSoftmax = (T) => {
    const e = demoLogits.map((z) => Math.exp((z - demoLogits[0]) / T));
    const total = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / total);
  };
  const X0 = 96, W = 520, THRESHOLD = 0.9, K = 3;
  Learn.demo(document.getElementById("demo-cut"), {
    height: 360,
    label: "Eight probability bars with the least likely tokens cut by top-k or top-p, and the kept bars rescaled",
    steps: [
      { s: { T: 1, rule: "none", cut: 0, norm: 0 }, cap: "Start from the model's distribution at T = 1, sorted most likely first. The right column is the running total. All 8 tokens can be drawn." },
      { s: { rule: "k", cut: 1 }, cap: "Top-k with k = 3 keeps the 3 most likely tokens: mat, floor and bed. Together they hold 0.813. The other five are cut." },
      { s: { norm: 1 }, cap: "Renormalize: divide each kept probability by 0.813 so they add up to 1 again. The bars grow to 0.547, 0.271 and 0.182. The ratios between them stay the same." },
      { s: { rule: "p", norm: 0 }, cap: "Now use top-p with a threshold of 0.9. Walk down the running total: 0.444, 0.665, 0.813, 0.903. The fourth token takes it past 0.9, so top-p keeps four." },
      { s: { norm: 1 }, cap: "Renormalize by the kept total, 0.903. The four bars become 0.492, 0.244, 0.164 and 0.099." },
      { s: { T: 2 }, cap: "Raise the temperature to 2. The bars flatten, so it now takes six tokens to reach 0.9 (their total is 0.923). Top-p keeps six. Top-k would still keep exactly 3." },
    ],
    draw(ctx, s, c, d) {
      const p = demoSoftmax(s.T);
      const cum = [];
      p.reduce((sum, v, i) => (cum[i] = sum + v), 0);
      let keep = 8;
      if (s.rule === "k") keep = K;
      if (s.rule === "p") keep = cum.findIndex((v) => v >= THRESHOLD - 1e-9) + 1;
      const total = cum[keep - 1];
      d.text(`T = ${d.fmt(s.T, 1)}`, 20, 30, { c: c.ink, w: 700 });
      d.text(s.rule === "k" ? "top-k, k = 3" : s.rule === "p" ? "top-p, threshold 0.9" : "no cut", X0, 30, { c: c.orange, w: 700 });
      d.text("running total", 620, 30, { c: c.muted, s: 13, a: "right" });
      demoTokens.forEach((token, i) => {
        const y = 64 + i * 30, kept = i < keep;
        const q = kept ? p[i] * (1 + s.norm * (1 / total - 1)) : p[i];
        ctx.globalAlpha = kept ? 1 : 1 - 0.7 * s.cut;
        d.text(token, 20, y, { c: kept ? c.ink : c.muted });
        d.rect(X0, y - 9, W * q, 18, kept ? c.blue : c.muted);
        if (kept && s.norm > 0.02) d.line(X0 + W * p[i], y - 12, X0 + W * p[i], y + 12, c.ink, 2);
        const shown = kept ? d.fmt(q, 3) : s.cut > 0.5 ? "cut" : d.fmt(q, 3);
        d.text(shown, X0 + W * q + 10, y, { c: kept ? c.ink : c.muted, s: 13 });
        const edge = s.rule !== "none" && i === keep - 1;
        d.text(d.fmt(cum[i], 3), 620, y, { c: edge ? c.orange : c.muted, s: 13, a: "right", w: edge ? 700 : 500 });
        ctx.globalAlpha = 1;
        if (edge && s.rule === "p") d.text("≥ 0.9", 566, y, { c: c.orange, s: 13, a: "right", w: 700 });
      });
      d.line(X0, 46, X0, 292, c.muted, 1.5);
      if (keep < 8) d.line(20, 64 + keep * 30 - 15, 620, 64 + keep * 30 - 15, c.orange, 1.5, [6, 5]);
      d.text(keep === 8 ? "kept: all 8 tokens, total 1" : `kept: ${keep} tokens, total ${d.fmt(total, 3)}`, 20, 318, { c: c.ink, w: 700 });
      if (s.norm > 0.02 && keep < 8) {
        ctx.globalAlpha = s.norm;
        d.text(`each kept bar ÷ ${d.fmt(total, 3)}  (tick = before)`, 20, 342, { c: c.muted, s: 13 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();
