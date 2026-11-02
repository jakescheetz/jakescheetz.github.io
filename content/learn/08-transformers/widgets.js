// Unit 08: the residual stream. A tiny two-block, pre-norm transformer with random weights runs
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
