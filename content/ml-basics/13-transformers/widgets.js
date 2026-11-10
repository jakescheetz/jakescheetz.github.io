// Unit 13: the residual stream. A tiny two-block, pre-norm transformer with random weights runs
// on four tokens in this script; each row shows every token's vector after one more sublayer
// has added its contribution.
(function () {
  const figure = document.getElementById("residual-stream");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  // The model ----------------------------------------------------------------------------
  const tokens = ["the", "cat", "sat", "on"];
  const n = tokens.length, d = 8, dff = 32, blocks = 2;

  // Seeded random numbers so every reader sees the same model.
  let seed = 8;
  const uniform = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  const matrix = (rows, cols, std) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => std * normal()));

  // x is a vector (length = rows of W); returns x W.
  const times = (x, W) => W[0].map((_, c) => x.reduce((s, xi, r) => s + xi * W[r][c], 0));
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const norm = (a) => Math.hypot(...a);
  const layerNorm = (x) => {
    const mean = x.reduce((s, v) => s + v, 0) / x.length;
    const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
    return x.map((v) => (v - mean) / Math.sqrt(variance + 1e-5)); // gamma = 1, beta = 0
  };

  const E = matrix(n, d, 0.6); // one embedding row per token in this sentence
  const P = matrix(n, d, 0.6); // one position row per position
  const layers = Array.from({ length: blocks }, () => ({
    WQ: matrix(d, d, 0.5), WK: matrix(d, d, 0.5), WV: matrix(d, d, 0.35), WO: matrix(d, d, 0.22),
    W1: matrix(d, dff, 0.35), W2: matrix(dff, d, 0.15),
  }));

  // states[k][i] = token i's vector after k sublayers; deltas[k][i] = what sublayer k added.
  const states = [E.map((e, i) => add(e, P[i]))];
  const deltas = [null];
  const weights = [null]; // attention weights for attention sublayers, weights[k][i][j]
  for (const L of layers) {
    let h = states[states.length - 1];
    const x = h.map(layerNorm);
    const q = x.map((v) => times(v, L.WQ)), k = x.map((v) => times(v, L.WK)), v = x.map((u) => times(u, L.WV));
    const w = q.map((qi, i) => {
      const s = k.slice(0, i + 1).map((kj) => qi.reduce((sum, a, c) => sum + a * kj[c], 0) / Math.sqrt(d));
      const top = Math.max(...s);
      const e = s.map((z) => Math.exp(z - top));
      const total = e.reduce((a, b) => a + b, 0);
      return e.map((z) => z / total);
    });
    const attn = w.map((wi) => times(wi.reduce((acc, a, j) => add(acc, v[j].map((u) => a * u)), new Array(d).fill(0)), L.WO));
    deltas.push(attn); weights.push(w);
    h = h.map((hi, i) => add(hi, attn[i]));
    states.push(h);

    const mlp = h.map((hi) => times(times(layerNorm(hi), L.W1).map((u) => Math.max(0, u)), L.W2));
    deltas.push(mlp); weights.push(null);
    states.push(h.map((hi, i) => add(hi, mlp[i])));
  }
  const stages = states.length; // 5: embeddings, then 2 blocks × (attention, MLP)
  const names = ["embed", "1 attn", "1 MLP", "2 attn", "2 MLP"];
  const longNames = ["embeddings", "block 1 attention", "block 1 MLP", "block 2 attention", "block 2 MLP"];
  const colorOf = (k) => (k === 0 ? "blue" : weights[k] ? "orange" : "green");

  // The drawing ----------------------------------------------------------------------------
  const width = 720, rows = 19, height = rows * Learn.ROW;
  const left = 112, colWidth = (width - left) / n;
  const pitch = 15, barWidth = 11;
  const rowGap = 60, amp = 22;
  const rowY = (k) => height - 46 - k * rowGap; // baseline of stage row k
  const colX = (i) => left + colWidth * (i + 0.5);
  const barX = (i, j) => colX(i) - (pitch * d) / 2 + j * pitch + (pitch - barWidth) / 2;
  const maxAbs = Math.max(...states.flat(2).map(Math.abs));
  const scale = amp / maxAbs;

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Four token vectors drawn as bar charts, one row per sublayer, each row adding to the one below");

  for (let i = 0; i < n; i++) {
    // The lane each token's vector travels up, ending in its logits.
    svg("line", { x1: colX(i), y1: rowY(0) + amp, x2: colX(i), y2: 34, class: "muted faint thin" }, root);
    svg("text", { x: colX(i), y: 22, "text-anchor": "middle", class: "muted", text: "logits" }, root);
  }
  for (let k = 0; k < stages; k++) {
    svg("text", { x: 0, y: rowY(k) + 4, class: colorOf(k), text: names[k] }, root);
    svg("line", { x1: left, y1: rowY(k), x2: width, y2: rowY(k), class: "muted faint thin" }, root);
  }
  const linkLayer = svg("g", { class: "orange" }, root);

  // Bars: for each stage row, a faint bar for the incoming value, a colored segment for what this
  // sublayer added, and an ink tick at the new value.
  const bars = [];
  for (let k = 0; k < stages; k++) {
    const row = svg("g", {}, root);
    bars.push(Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => ({
      before: svg("rect", { x: barX(i, j), width: barWidth, class: "muted faint", fill: "currentColor" }, row),
      added: svg("rect", { x: barX(i, j), width: barWidth, class: colorOf(k), fill: "currentColor" }, row),
      tick: svg("line", { x1: barX(i, j) - 1, x2: barX(i, j) + barWidth + 1, class: "ink thin" }, row),
    }))));
  }

  const labels = tokens.map((t, i) => svg("text", { x: colX(i), y: height - 6, "text-anchor": "middle", text: t }, root));
  // Clicking a column picks that token.
  tokens.forEach((_, i) => {
    const hit = svg("rect", { x: colX(i) - colWidth / 2, y: 0, width: colWidth, height, fill: "transparent", style: "cursor: pointer" }, root);
    hit.addEventListener("click", () => tokenControl.set(i + 1));
  });

  const setBar = (rect, from, to) => {
    rect.setAttribute("y", Math.min(from, to));
    rect.setAttribute("height", Math.abs(to - from));
  };
  // Progress value that shows rows 0..k fully drawn.
  const fullRow = (k) => (k >= stages - 1 ? stages : k + 0.99);

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  let progress = 0; // integer part = row being added, fraction = how far; `stages` = all done
  let selected = 2;
  let hold = 0; // seconds spent showing the finished stack before the animation repeats

  function draw() {
    const current = Math.min(stages - 1, Math.floor(progress));
    const raw = Math.min(1, (progress - current) * 1.6);
    const frac = progress >= stages ? 1 : raw * raw * (3 - 2 * raw);

    for (let k = 0; k < stages; k++) {
      const shown = k < current ? 1 : k === current ? frac : 0;
      const base = rowY(k);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
          const b = bars[k][i][j];
          const before = k === 0 ? 0 : states[k - 1][i][j];
          const after = before + (states[k][i][j] - before) * shown;
          const visible = k <= current && (k < current || frac > 0);
          b.before.style.display = b.added.style.display = b.tick.style.display = visible ? "" : "none";
          setBar(b.before, base, base - before * scale);
          setBar(b.added, base - before * scale, base - after * scale);
          b.tick.setAttribute("y1", base - after * scale);
          b.tick.setAttribute("y2", base - after * scale);
          b.before.style.opacity = i === selected ? "" : "0.2";
          b.added.style.opacity = i === selected ? "" : "0.45";
        }
      }
    }
    labels.forEach((label, i) => label.setAttribute("class", i === selected ? "ink" : "muted"));

    // Attention lines into the selected token, one per earlier token, opacity = weight.
    linkLayer.replaceChildren();
    for (let k = 1; k <= current; k++) {
      if (!weights[k] || (k === current && frac === 0)) continue;
      weights[k][selected].forEach((w, j) => {
        svg("line", {
          x1: colX(j), y1: rowY(k - 1) - amp - 1, x2: colX(selected), y2: rowY(k) + amp + 1,
          style: `opacity: ${(0.15 + 0.85 * w) * (k === current ? frac : 1)}; stroke-width: ${1 + 3 * w}`,
        }, linkLayer);
      });
    }

    // Four lines in every state so the layout below doesn't jump.
    const i = selected, t = `"${tokens[i]}"`;
    const pad = (s) => s.padEnd(24);
    let text = `${t} at position ${i + 1}, ${longNames[current]}\n`;
    if (current === 0) {
      text += `<span class="blue">${pad(`h = E[${t}] + P[${i + 1}]`)}</span>|h| = ${fmt(norm(states[0][i]))}\n`;
      text += `nothing added yet\n `;
    } else {
      const color = colorOf(current);
      const isAttn = Boolean(weights[current]);
      text += `${pad("h before")}|h| = ${fmt(norm(states[current - 1][i]))}\n`;
      text += `<span class="${color}">${pad(`${isAttn ? "attention" : "MLP"} adds Δ`)}</span>|Δ| = <span class="${color}">${fmt(norm(deltas[current][i]))}</span>\n`;
      text += `${pad("h after = h before + Δ")}|h| = ${fmt(norm(states[current][i]))}\n`;
      text += isAttn
        ? `weights: ` + weights[current][i].map((w, j) => `${tokens[j]} <span class="orange">${fmt(w)}</span>`).join("  ")
        : `MLP: this token's own vector only`;
    }
    out.innerHTML = text;
  }

  const currentRow = () => Math.min(stages - 1, Math.floor(progress));
  const stageControl = slider(ctl, { label: "rows shown", min: 1, max: stages, step: 1, value: stages, format: (v) => names[Math.round(v) - 1] }, (v) => {
    progress = fullRow(v - 1);
    hold = 0;
    draw();
  });
  const tokenControl = slider(ctl, { label: "token", min: 1, max: n, step: 1, value: selected + 1, format: (v) => tokens[Math.round(v) - 1] }, (v) => {
    selected = v - 1;
    draw();
  });

  // Animated: start from an empty stage and add one row at a time, then hold and repeat.
  if (!Learn.reducedMotion) progress = 0;
  const player = loop(figure, (dt) => {
    if (dt) {
      if (progress >= stages) {
        hold += dt;
        if (hold > 2.5) { hold = 0; progress = 0; }
      } else {
        progress = Math.min(stages, progress + 0.55 * dt);
      }
    }
    stageControl.show(currentRow() + 1);
    draw();
  });
  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
  button(btns, "Next row", () => {
    player.pause();
    toggle.textContent = "Play";
    hold = 0;
    const next = progress >= stages ? 0 : currentRow() + 1;
    progress = fullRow(next);
    stageControl.show(Math.min(next, stages - 1) + 1);
    draw();
  });
  // Grabbing the row slider stops the animation so it doesn't fight the reader.
  stageControl.input.addEventListener("input", () => {
    player.pause();
    toggle.textContent = "Play";
  });
})();

// Step-through: one token's lane through two blocks. Every sublayer adds a vector to the stream.
(function () {
  const h0 = [1, -2, 0.5, 3];
  const adds = [
    { name: "block 1 attn", sym: "a1", v: [0.5, 0, -1, 0.5], attn: true },
    { name: "block 1 MLP", sym: "m1", v: [-1, 1.5, 0, 0.5], attn: false },
    { name: "block 2 attn", sym: "a2", v: [0, 1, 0.5, -1], attn: true },
    { name: "block 2 MLP", sym: "m2", v: [1, 0, 1.5, 0], attn: false },
  ];
  const stream = [h0];
  for (const add of adds) stream.push(stream[stream.length - 1].map((v, k) => v + add.v[k]));
  const rowY = (i) => 312 - i * 58;
  Learn.demo(document.getElementById("demo-residual"), {
    height: 376,
    label: "One token's vector moving up through two transformer blocks, with each sublayer's output added to it",
    steps: [
      { s: { pc: 0, n: 0, sum: 0 }, cap: "One token's lane starts at the bottom with its embedding, h0 = (1, −2, 0.5, 3). The lane runs upward." },
      { s: { pc: 1 }, cap: "Block 1's attention reads the lane and computes a vector, a1 = (0.5, 0, −1, 0.5). It has not touched the lane yet." },
      { s: { n: 1 }, cap: "Add a1 to the lane, entry by entry: (1, −2, 0.5, 3) + (0.5, 0, −1, 0.5) = (1.5, −2, −0.5, 3.5). The old numbers were added to, not replaced." },
      { s: { pc: 2, n: 2 }, cap: "Block 1's MLP reads the new lane and adds m1 = (−1, 1.5, 0, 0.5). The lane is now (0.5, −0.5, −0.5, 4). That finishes block 1." },
      { s: { pc: 4, n: 4 }, cap: "Block 2 does the same two moves. Its attention adds a2 and its MLP adds m2. The lane ends at (1.5, 0.5, 1.5, 3)." },
      { s: { sum: 1 }, cap: "The top vector is the embedding plus everything the sublayers added. Check the first entry: 1 + 0.5 − 1 + 0 + 1 = 1.5." },
    ],
    draw(ctx, s, c, d) {
      d.text("sublayer adds", 222, 34, { c: c.muted, s: 13, a: "center" });
      d.text("stream h (one token)", 497, 34, { c: c.muted, s: 13, a: "center" });
      const cells = (v, x0, pitch, w, y, stroke, bold) => {
        for (let k = 0; k < 4; k++) {
          d.rect(x0 + k * pitch, y - 17, w, 34, c.bg, stroke, 2);
          d.text(d.fmt(v[k], 1), x0 + k * pitch + w / 2, y, { c: c.ink, a: "center", w: bold ? 700 : 500 });
        }
      };
      for (let i = 0; i <= 4; i++) {
        const y = rowY(i);
        const rowAlpha = i === 0 ? 1 : d.clamp(s.n - (i - 1), 0, 1);
        if (i > 0) {
          const add = adds[i - 1], color = add.attn ? c.orange : c.green;
          const pieceAlpha = d.clamp(s.pc - (i - 1), 0, 1);
          if (pieceAlpha > 0.02) {
            ctx.globalAlpha = pieceAlpha;
            d.text(add.name, 14, y - 8, { c: color, s: 13, w: 700 });
            d.text(add.sym, 14, y + 10, { c: c.muted, s: 13 });
            cells(add.v, 124, 50, 46, y, color, false);
          }
          if (rowAlpha > 0.02) {
            ctx.globalAlpha = rowAlpha;
            d.text("+", 348, y, { c: color, s: 20, a: "center", w: 700 });
            d.arrow(497, rowY(i - 1) - 19, 497, y + 19, c.blue, 2);
          }
        } else {
          d.text("embedding", 14, y - 8, { c: c.blue, s: 13, w: 700 });
          d.text("h0", 14, y + 10, { c: c.muted, s: 13 });
        }
        if (rowAlpha > 0.02) {
          ctx.globalAlpha = rowAlpha;
          cells(stream[i], 384, 58, 54, y, c.blue, true);
        }
        ctx.globalAlpha = 1;
      }
      if (s.sum > 0.02) {
        ctx.globalAlpha = s.sum;
        d.text("h_top = h0 + a1 + m1 + a2 + m2", 320, 356, { c: c.ink, s: 16, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: layer norm on one token's vector, then on the same vector made 10 times longer.
(function () {
  const x = [2, 4, 4, 10], MEAN = 5, STD = 3; // mean and standard deviation of x
  const y0 = 230, U = 20;
  Learn.demo(document.getElementById("demo-layernorm"), {
    height: 340,
    label: "Layer norm on a four-entry vector: subtract the mean, then divide by the standard deviation",
    steps: [
      { s: { k: 1, mean: 0, sub: 0, div: 0, gb: 0 }, cap: "One token's vector has 4 entries: (2, 4, 4, 10). Layer norm works on these 4 numbers and ignores every other token." },
      { s: { mean: 1 }, cap: "Find the mean of the entries: (2 + 4 + 4 + 10) ÷ 4 = 5. The dashed line marks it." },
      { s: { sub: 1 }, cap: "Subtract the mean from every entry: (−3, −1, −1, 5). The entries now average 0." },
      { s: { div: 1 }, cap: "Find the standard deviation: square the entries, average them, take the square root. √((9 + 1 + 1 + 25) ÷ 4) = 3. Divide every entry by 3: (−1, −0.33, −0.33, 1.67)." },
      { s: { k: 10 }, cap: "Now suppose many blocks have added to the stream and the vector is 10 times longer: (20, 40, 40, 100). Its mean is 50 and its standard deviation is 30. The output is the same four numbers." },
      { s: { gb: 1 }, cap: "Last, layer norm multiplies each entry by a learned γ and adds a learned β. Here γ = 1 and β = 0, so nothing moves. Training can change them." },
    ],
    draw(ctx, s, c, d) {
      const shrink = 1 + s.div * (1 / (STD * s.k) - 1);
      const w = x.map((v) => (s.k * v - s.sub * s.k * MEAN) * shrink);
      const curMean = s.k * MEAN * (1 - s.sub) * shrink;
      // Left: the numbers.
      d.text("input h", 20, 50, { c: c.muted, s: 13 });
      d.text(`(${x.map((v) => d.fmt(v * s.k, 0)).join(", ")})`, 20, 74, { c: c.blue, w: 700 });
      if (s.mean > 0.02) {
        ctx.globalAlpha = s.mean;
        d.text(`mean = ${d.fmt(MEAN * s.k, 0)}`, 20, 116, { c: c.ink });
      }
      if (s.div > 0.02) {
        ctx.globalAlpha = s.div;
        d.text(`std  = ${d.fmt(STD * s.k, 0)}`, 20, 142, { c: c.ink });
        d.text("LN(h)", 20, 192, { c: c.muted, s: 13 });
        d.text("(−1, −0.33, −0.33, 1.67)", 20, 216, { c: c.orange, s: 13, w: 700 });
      }
      if (s.gb > 0.02) {
        ctx.globalAlpha = s.gb;
        d.text("then × γ, + β (learned)", 20, 258, { c: c.muted, s: 13 });
      }
      ctx.globalAlpha = 1;
      // Right: the working values as bars around zero.
      d.line(270, y0, 620, y0, c.muted, 1.5);
      d.text("0", 262, y0, { c: c.muted, s: 13, a: "right" });
      const color = s.div > 0.5 ? c.orange : c.blue;
      for (let i = 0; i < 4; i++) {
        const bx = 290 + i * 76, top = y0 - w[i] * U;
        d.rect(bx, Math.min(y0, top), 50, Math.abs(w[i] * U), color);
        d.text(d.fmt(w[i], 2), bx + 25, w[i] >= 0 ? top - 12 : top + 14, { c: c.ink, s: 13, a: "center", w: 700 });
      }
      if (s.mean > 0.02) {
        ctx.globalAlpha = s.mean;
        const my = y0 - curMean * U;
        d.line(270, my, 620, my, c.ink, 1.5, [6, 5]);
        d.text(`mean ${d.fmt(curMean, 1)}`, 632, my - 12, { c: c.ink, s: 13, a: "right" });
        ctx.globalAlpha = 1;
      }
      d.text(s.div > 0.5 ? "after dividing by the std" : s.sub > 0.5 ? "after subtracting the mean" : "the entries of h", 445, 326, { c: c.muted, s: 13, a: "center" });
    },
  });
})();

// Step-through: attention at the final "." in two sentences, without and with position vectors.
(function () {
  const E = { dog: [2, 0], bites: [0, 2], man: [1, 1], ".": [1, 0] };
  const P = [[0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5]];
  const sentences = [["dog", "bites", "man", "."], ["man", "bites", "dog", "."]];
  const tops = [44, 204];
  function attend(sentence, pos) {
    const v = sentence.map((t, i) => [E[t][0] + pos * P[i][0], E[t][1] + pos * P[i][1]]);
    const q = v[3];
    const e = v.map((k) => Math.exp(q[0] * k[0] + q[1] * k[1]));
    const total = e.reduce((a, b) => a + b, 0);
    const w = e.map((x) => x / total);
    const out = [0, 1].map((j) => v.reduce((sum, vec, i) => sum + w[i] * vec[j], 0));
    return { v, w, out };
  }
  Learn.demo(document.getElementById("demo-position"), {
    height: 360,
    label: "Attention weights and output at the last token of two sentences, before and after adding position vectors",
    steps: [
      { s: { w: 0, out: 0, pos: 0, hl: 0 }, cap: "Two sentences with the same four tokens. Only dog and man trade places. Under each token is its vector, and the same token has the same vector in both sentences." },
      { s: { w: 1 }, cap: "Attention at the final \".\" scores each token with a dot product, and softmax turns the scores into weights (orange bars). dog gets 0.53 and man gets 0.2 in both sentences, wherever they sit." },
      { s: { out: 1 }, cap: "The output is the weighted sum of the vectors. Both sentences give (1.46, 0.34). From this output, the model can't tell who bit whom." },
      { s: { pos: 1 }, cap: "Now add a position vector to each token first. dog in position 1 becomes (2.5, 0). dog in position 3 becomes (1.5, 0). The same word now has a different vector in each sentence." },
      { s: { hl: 1 }, cap: "The weights change with it: dog gets 0.72 in the top sentence and 0.41 in the bottom one. The outputs are (2.03, 0) and (1.3, 0.15), so later layers can tell the sentences apart." },
    ],
    draw(ctx, s, c, d) {
      const results = sentences.map((sentence) => attend(sentence, s.pos));
      sentences.forEach((sentence, r) => {
        const T = tops[r], { v, w, out } = results[r];
        sentence.forEach((token, i) => {
          const bx = 20 + i * 112;
          if (s.pos > 0.02) {
            ctx.globalAlpha = s.pos;
            d.text(`position ${i + 1}`, bx + 50, T - 12, { c: c.purple, s: 12, a: "center" });
            ctx.globalAlpha = 1;
          }
          d.rect(bx, T, 100, 34, i === 3 ? c.soft : c.bg, i === 3 ? c.blue : c.ink, 2);
          d.text(token, bx + 50, T + 17, { c: i === 3 ? c.blue : c.ink, s: 15, a: "center", w: 700 });
          d.text(`(${d.fmt(v[i][0], 1)}, ${d.fmt(v[i][1], 1)})`, bx + 50, T + 48, { c: c.muted, s: 12, a: "center" });
          if (s.w > 0.02) {
            ctx.globalAlpha = s.w;
            d.rect(bx, T + 60, 100, 12, c.soft);
            d.rect(bx, T + 60, 100 * w[i], 12, c.orange);
            d.text(d.fmt(w[i], 2), bx + 50, T + 86, { c: c.ink, s: 13, a: "center", w: 700 });
            ctx.globalAlpha = 1;
          }
        });
        if (s.out > 0.02) {
          ctx.globalAlpha = s.out;
          d.panel(c, 480, T, 148, 72);
          d.text("output at \".\"", 492, T + 20, { c: c.muted, s: 13 });
          d.text(`(${d.fmt(out[0], 2)}, ${d.fmt(out[1], 2)})`, 492, T + 48, { c: c.blue, s: 15, w: 700 });
          ctx.globalAlpha = 1;
        }
      });
      if (s.out > 0.02) {
        const same = s.pos < 0.02;
        ctx.globalAlpha = s.out * (same ? 1 : Math.max(s.pos, 0.02));
        d.text(same ? "no position vectors: the two outputs match" : "with position vectors: the two outputs differ", 320, 332, { c: same ? c.red : c.blue, s: 14, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      }
      if (s.hl > 0.02) {
        ctx.globalAlpha = s.hl;
        for (const T of tops) d.rect(476, T - 4, 156, 80, null, c.orange, 2);
        ctx.globalAlpha = 1;
      }
    },
  });
})();
