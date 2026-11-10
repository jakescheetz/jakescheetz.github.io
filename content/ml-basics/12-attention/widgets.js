// Unit 12: causal attention weights for one query token, and why scores are divided by √d_k.
(function () {
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;
  const ROW = Learn.ROW;
  const PITCH = 1.6 * ROW; // vertical spacing of list rows, roomy enough for phone-sized labels

  const softmax = (z) => {
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  };
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const vec = (v) => `(${v.map((x) => String(x).replace("-", "−")).join(", ")})`;

  // Widget 1: pick a query token, see its weights over the tokens up to and including it.
  (function () {
    const figure = document.getElementById("attention-picker");
    if (!figure) return;

    const tokens = "the cat sat on the mat because it was tired".split(" ");
    // Hand-made 4-number vectors (d_k = 4). Positions mean roughly:
    // [animal noun, thing noun, verb, pronoun]. Not taken from a trained model.
    const keys = [
      [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0],
      [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 0], [0, 0, 1, 0],
    ];
    const queries = [
      [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0],
      [0, 0, 1.5, 0], [0, 0, 2, 0], [3, -1, 0, 0], [0, 0, 0, 3], [1.5, 0, 0, 1.5],
    ];
    const dk = 4;
    const n = tokens.length;

    const width = 720, rows = 19;
    const labelX = 140, barX = 165, barMax = 400;
    const top = 2 * ROW;
    const rowY = (j) => top + (j + 0.5) * PITCH;

    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "Attention weights from one query token to every token up to it in the sentence");
    svg("text", { x: labelX, y: ROW, "text-anchor": "end", "dominant-baseline": "central", class: "muted", text: "key" }, root);
    svg("text", { x: barX, y: ROW, "dominant-baseline": "central", class: "muted", text: "weight from the query" }, root);
    svg("line", { x1: barX, y1: top, x2: barX, y2: top + n * PITCH, class: "muted thin" }, root);

    const marker = svg("polygon", { class: "blue" }, root);
    const items = tokens.map((tok, j) => {
      const y = rowY(j);
      const label = svg("text", { x: labelX, y, "text-anchor": "end", "dominant-baseline": "central", text: tok, style: "cursor: pointer" }, root);
      const bar = svg("rect", { x: barX, y: y - 7, height: 14, fill: "currentColor", class: "orange" }, root);
      const value = svg("text", { y, "dominant-baseline": "central" }, root);
      label.addEventListener("click", () => {
        stop();
        position.set(j + 1);
      });
      return { label, bar, value };
    });

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);

    let query = 7; // 0-based index of "it"
    let scale = 1;

    function draw() {
      const q = queries[query];
      const raw = keys.slice(0, query + 1).map((k) => dot(q, k));
      const scores = raw.map((s) => scale * s / Math.sqrt(dk));
      const w = softmax(scores);
      let best = 0;
      w.forEach((v, j) => { if (v > w[best] + 1e-9) best = j; });

      items.forEach((item, j) => {
        const visible = j <= query;
        item.label.setAttribute("class", j === query ? "blue" : visible ? "ink" : "muted faint");
        item.bar.style.display = visible ? "" : "none";
        if (visible) {
          item.bar.setAttribute("width", Math.max(0.5, w[j] * barMax));
          item.value.setAttribute("x", barX + w[j] * barMax + 8);
          item.value.setAttribute("class", "ink");
          item.value.textContent = fmt(w[j]);
        } else {
          item.value.setAttribute("x", barX + 8);
          item.value.setAttribute("class", "muted faint");
          item.value.textContent = "masked";
        }
      });
      const y = rowY(query);
      marker.setAttribute("points", `8,${y - 7} 20,${y} 8,${y + 7}`);

      const tok = tokens[query], bestTok = tokens[best];
      const sum = w.reduce((a, b) => a + b, 0);
      out.innerHTML =
        `<span class="blue">query "${tok}" (position ${query + 1})\n` +
        `q = ${vec(q)}</span>\n` +
        `top weight on "${bestTok}" (position ${best + 1})\n` +
        `k = ${vec(keys[best])}\n` +
        `score = q · k / √4 = ${fmt(raw[best])} / 2 = ${fmt(raw[best] / 2)}\n` +
        `× scale ${scale.toFixed(1)} = ${fmt(scores[best])}\n` +
        `<span class="orange">weight on "${bestTok}" = ${fmt(w[best])}</span>\n` +
        `weights on 1 to ${query + 1} sum to ${fmt(sum)}\n` +
        (query < n - 1 ? `positions ${query + 2} to ${n} masked` : "no positions masked");
    }

    const position = slider(ctl, { label: "query token", min: 1, max: n, step: 1, value: query + 1, format: (v) => tokens[v - 1] }, (v) => {
      query = v - 1;
      draw();
    });
    slider(ctl, { label: "score scale", min: 0, max: 3, step: 0.1, value: 1, format: (v) => `×${v.toFixed(1)}` }, (v) => {
      scale = v;
      draw();
    });

    let wait = 0;
    const player = loop(figure, (dt) => {
      if (!dt) { draw(); return; }
      wait += dt;
      if (wait < 1.6) return;
      wait = 0;
      query = (query + 1) % n;
      position.show(query + 1);
      draw();
    });
    const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
      if (player.playing) player.pause(); else player.play();
      toggle.textContent = player.playing ? "Pause" : "Play";
    });
    function stop() {
      player.pause();
      toggle.textContent = "Play";
    }
    ctl.addEventListener("input", stop);
  })();

  // Widget 2: dot products of random vectors grow like √d_k and saturate the softmax.
  (function () {
    const figure = document.getElementById("scaling");
    if (!figure) return;

    const MAXD = 1024, NKEYS = 8;
    // Small seeded generator (mulberry32) so a given sample looks the same every time.
    function rng(seed) {
      let a = seed >>> 0;
      const uniform = () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      return () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
    }
    const draws = (normal, count) => Float64Array.from({ length: count }, normal);

    const width = 720, rows = 17;
    const axisX = 250, half = 170, limit = 20; // the score axis shows −20 to 20
    const weightX = 470, weightMax = 220;
    const top = 2 * ROW;
    const rowY = (j) => top + (j + 0.5) * PITCH;
    const bottom = top + NKEYS * PITCH;

    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "Scores of one query against eight random keys, and their softmax weights");
    svg("text", { x: axisX, y: ROW, "text-anchor": "middle", "dominant-baseline": "central", class: "muted", text: "score" }, root);
    svg("text", { x: weightX, y: ROW, "dominant-baseline": "central", class: "muted", text: "softmax weight" }, root);
    svg("line", { x1: axisX, y1: top, x2: axisX, y2: bottom, class: "muted thin" }, root);
    svg("line", { x1: weightX, y1: top, x2: weightX, y2: bottom, class: "muted thin" }, root);
    for (const s of [-limit, limit]) {
      const x = axisX + s / limit * half;
      svg("line", { x1: x, y1: top, x2: x, y2: bottom, class: "muted faint dashed thin" }, root);
      svg("text", { x, y: bottom + ROW, "text-anchor": "middle", "dominant-baseline": "central", class: "muted", text: String(s).replace("-", "−") }, root);
    }

    const items = Array.from({ length: NKEYS }, (_, j) => {
      const y = rowY(j);
      svg("text", { x: 30, y, "dominant-baseline": "central", text: `k${j + 1}` }, root);
      return {
        score: svg("rect", { y: y - 7, height: 14, fill: "currentColor", class: "blue" }, root),
        clip: svg("polygon", { class: "blue" }, root),
        weight: svg("rect", { x: weightX, y: y - 7, height: 14, fill: "currentColor", class: "orange" }, root),
      };
    });

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);

    let logD = 6;
    let scaled = false;
    let sample = 1;
    let q, ks;

    function resample() {
      const normal = rng(sample);
      q = draws(normal, MAXD);
      ks = Array.from({ length: NKEYS }, () => draws(normal, MAXD));
    }

    // Spread of q · k over many fresh random pairs, measured rather than assumed.
    const spreadCache = new Map();
    function spread(d) {
      if (!spreadCache.has(d)) {
        const normal = rng(12345 + d);
        const pairs = 1000;
        let sum = 0, sumSq = 0;
        for (let p = 0; p < pairs; p++) {
          let s = 0;
          for (let i = 0; i < d; i++) s += normal() * normal();
          sum += s; sumSq += s * s;
        }
        const mean = sum / pairs;
        spreadCache.set(d, Math.sqrt(sumSq / pairs - mean * mean));
      }
      return spreadCache.get(d);
    }

    function draw() {
      const d = 2 ** logD;
      const div = scaled ? Math.sqrt(d) : 1;
      const scores = ks.map((k) => {
        let s = 0;
        for (let i = 0; i < d; i++) s += q[i] * k[i];
        return s / div;
      });
      const w = softmax(scores);
      let best = 0;
      w.forEach((v, j) => { if (v > w[best]) best = j; });

      items.forEach((item, j) => {
        const s = Math.max(-limit, Math.min(limit, scores[j]));
        const x = axisX + s / limit * half;
        const y = rowY(j);
        item.score.setAttribute("x", Math.min(axisX, x));
        item.score.setAttribute("width", Math.max(0.5, Math.abs(x - axisX)));
        const clipped = Math.abs(scores[j]) > limit;
        item.clip.style.display = clipped ? "" : "none";
        if (clipped) {
          const sgn = Math.sign(scores[j]);
          item.clip.setAttribute("points", `${x},${y - 11} ${x + sgn * 12},${y} ${x},${y + 11}`);
        }
        item.weight.setAttribute("width", Math.max(0.5, w[j] * weightMax));
      });

      const lo = Math.min(...scores), hi = Math.max(...scores);
      out.innerHTML =
        `d_k = ${d}, ${scaled ? "scores are q · k / √d_k" : "scores are raw q · k"}\n` +
        `raw q · k over 1,000 random pairs:\n` +
        `spread (std) ${fmt(spread(d), 1)}   √d_k = ${fmt(Math.sqrt(d), 1)}\n` +
        `<span class="blue">the 8 scores run from ${fmt(lo, 1).replace("-", "−")} to ${fmt(hi, 1).replace("-", "−")}</span>\n` +
        `<span class="orange">largest weight: ${fmt(w[best])} on k${best + 1}</span>`;
    }

    slider(ctl, { label: "d_k", min: 0, max: 10, step: 1, value: logD, format: (v) => String(2 ** v) }, (v) => {
      logD = v;
      if (q) draw();
    });
    const scaleButton = button(btns, "Scaling: off", () => {
      scaled = !scaled;
      scaleButton.textContent = `Scaling: ${scaled ? "on" : "off"}`;
      draw();
    });
    button(btns, "New sample", () => {
      sample += 1;
      resample();
      draw();
    });

    resample();
    draw();
  })();
})();

// Step-through: one query ("it") scored against every key up to it, then softmax, then the blend.
(function () {
  // The same hand-made vectors as the attention-picker widget (d_k = 4), for the tokens up to "it".
  const tokens = "the cat sat on the mat because it".split(" ");
  const keys = [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]];
  const q = [3, -1, 0, 0], dk = 4;
  const minus = (v) => String(v).replace("-", "−");
  const vec = (v) => `(${v.map(minus).join(", ")})`;
  const dots = keys.map((k) => k.reduce((sum, v, m) => sum + v * q[m], 0));
  const scores = dots.map((v) => v / Math.sqrt(dk));
  const ex = scores.map(Math.exp), total = ex.reduce((a, b) => a + b, 0), w = ex.map((v) => v / total);
  const rowY = (j) => 62 + 28 * j;
  Learn.demo(document.getElementById("demo-qkv"), {
    height: 325,
    label: "The query of the token it, the keys of the eight tokens up to it, their dot products, scores and attention weights",
    steps: [
      { s: { k: 0, dp: 0, sc: 0, w: 0, out: 0 }, cap: `The token "it" has to find out what it refers to. Its query is q = ${vec(q)}. With my hand-picked meanings that reads: looking for an animal, not a thing.` },
      { s: { k: 1 }, cap: `Every token up to and including "it" has a key. cat's key is ${vec(keys[1])}, an animal. mat's key is ${vec(keys[5])}, a thing. Most other keys are all zeros.` },
      { s: { dp: 1 }, cap: `Dot the query with each key. cat: 3 × 2 = ${minus(dots[1])}. mat: −1 × 2 = ${minus(dots[5])}. The other six come out to 0.` },
      { s: { sc: 1 }, cap: `Divide by √d_k = √4 = 2. The scores are ${minus(scores[1])} for cat, ${minus(scores[5])} for mat and 0 for the rest.` },
      { s: { w: 1 }, cap: `Softmax turns the eight scores into weights that add up to 1. cat gets ${w[1].toFixed(2)}, mat gets ${w[5].toFixed(2)}, and each of the other six gets ${w[0].toFixed(2)}.` },
      { s: { out: 1 }, cap: `The new vector for "it" is the weighted sum of the values: ${w[1].toFixed(2)} of cat's value plus small amounts of the others. The vector for "it" now carries information about the cat.` },
    ],
    draw(ctx, s, c, d) {
      d.text("token", 24, 30, { c: c.muted, s: 13 });
      const head = (alpha, str, x, align) => { if (alpha > 0.01) { ctx.globalAlpha = alpha; d.text(str, x, 30, { c: c.muted, s: 13, a: align }); ctx.globalAlpha = 1; } };
      head(s.k, "key k", 110, "left");
      head(s.dp, "q · k", 300, "right");
      head(s.sc, "÷ √4", 380, "right");
      head(s.w, "weight", 410, "left");
      tokens.forEach((token, j) => {
        const y = rowY(j), isQuery = j === tokens.length - 1, strong = dots[j] !== 0;
        d.text(token, 24, y, { c: isQuery ? c.blue : c.ink, s: 15, w: isQuery ? 700 : 500 });
        if (s.k > 0.01) {
          ctx.globalAlpha = s.k;
          d.text(vec(keys[j]), 110, y, { c: strong ? c.ink : c.muted });
        }
        if (s.dp > 0.01) {
          ctx.globalAlpha = s.dp;
          d.text(minus(dots[j]), 300, y, { c: strong ? c.ink : c.muted, a: "right", w: strong ? 700 : 500 });
        }
        if (s.sc > 0.01) {
          ctx.globalAlpha = s.sc;
          d.text(minus(scores[j]), 380, y, { c: strong ? c.ink : c.muted, a: "right", w: strong ? 700 : 500 });
        }
        if (s.w > 0.01) {
          ctx.globalAlpha = s.w;
          d.rect(410, y - 8, 190 * w[j] * s.w, 16, c.orange);
          d.text(w[j].toFixed(2), 418 + 190 * w[j] * s.w, y, { c: c.ink });
        }
        ctx.globalAlpha = 1;
      });
      d.line(24, 278, 616, 278, c.muted, 2);
      d.text(`query of "it":  q = ${vec(q)}`, 24, 302, { c: c.blue, w: 700 });
      if (s.out > 0.01) {
        ctx.globalAlpha = s.out;
        d.text(`output ≈ ${w[1].toFixed(2)} × v(cat) + ...`, 340, 302, { c: c.orange, w: 700 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: why big scores stall the softmax, and what dividing by √d_k does about it.
(function () {
  const X = (gap) => 70 + 34 * gap, Y = (weight) => 280 - 400 * (weight - 0.5);
  const top = (gap) => 1 / (1 + Math.exp(-gap)); // softmax weight on key 1 for scores (gap, 0)
  const f2 = (v) => v.toFixed(2), f4 = (v) => v.toFixed(4);
  const dk = 64, root = Math.sqrt(dk);
  const modes = ["two keys", `d_k = ${dk}, raw scores`, `d_k = ${dk}, scores ÷ ${root}`];
  Learn.demo(document.getElementById("demo-scale"), {
    height: 330,
    label: "The softmax weight on the top key as a curve against the gap between two scores, with a dot at the current gap",
    steps: [
      { s: { a: 1, m: 0 }, cap: `Two keys with scores 1 and 0. Softmax gives weights ${f2(top(1))} and ${f2(1 - top(1))}. The curve shows the weight on key 1 for every gap between the two scores.` },
      { s: { a: root, m: 1 }, cap: `With d_k = ${dk}, a raw dot product is typically about √${dk} = ${root} in size. Scores ${root} and 0 give weights ${f4(top(root))} and ${f4(1 - top(root))}. One key takes almost everything.` },
      { s: { a: root - 1 }, cap: `Nudge the top score down from ${root} to ${root - 1}. Its weight only moves from ${f4(top(root))} to ${f4(top(root - 1))}. The curve is flat here, so the gradient that reaches W_Q and W_K is tiny.` },
      { s: { a: 1, m: 2 }, cap: `Now divide every score by √d_k = ${root}. Scores ${root} and 0 become 1 and 0, and the weights go back to ${f2(top(1))} and ${f2(1 - top(1))}.` },
      { s: { a: (root - 1) / root }, cap: `The same nudge after scaling: ${root - 1} ÷ ${root} = ${(root - 1) / root}. The weight moves from ${f2(top(1))} to ${f2(top((root - 1) / root))}. The curve is steep here, so training gets a signal it can use.` },
    ],
    draw(ctx, s, c, d) {
      for (let k = 0; k <= 5; k++) {
        const weight = 0.5 + 0.1 * k;
        d.line(X(0), Y(weight), X(10), Y(weight), c.line);
        d.text(weight.toFixed(1), X(0) - 10, Y(weight), { c: c.muted, s: 13, a: "right" });
      }
      for (let gap = 0; gap <= 10; gap += 2) {
        d.line(X(gap), Y(0.5), X(gap), Y(1), c.line);
        d.text(String(gap), X(gap), 298, { c: c.muted, s: 13, a: "center" });
      }
      d.line(X(0), Y(0.5), X(10), Y(0.5), c.muted, 1.5);
      d.line(X(0), Y(0.5), X(0), Y(1), c.muted, 1.5);
      d.text("weight on key 1", X(0) + 12, 56, { c: c.muted, s: 13 });
      d.text("score of key 1 minus score of key 2", X(5), 318, { c: c.muted, s: 13, a: "center" });
      ctx.strokeStyle = c.blue; ctx.lineWidth = 3;
      ctx.beginPath();
      for (let k = 0; k <= 100; k++) {
        const gap = k / 10;
        if (k === 0) ctx.moveTo(X(gap), Y(top(gap))); else ctx.lineTo(X(gap), Y(top(gap)));
      }
      ctx.stroke();
      const weight = top(s.a), px = X(s.a), py = Y(weight);
      d.line(px, py, px, Y(0.5), c.orange, 1.5, [4, 4]);
      d.dot(px, py, 7, c.orange);
      d.panel(c, 430, 50, 198, 176);
      d.text(modes[d.clamp(Math.round(s.m), 0, 2)], 444, 72, { c: c.muted, s: 12 });
      d.text("scores", 444, 100, { c: c.muted, s: 13 });
      d.text(`(${d.fmt(s.a, 3)}, 0)`, 444, 124, { c: c.ink, s: 16, w: 700 });
      d.text("weights", 444, 154, { c: c.muted, s: 13 });
      d.text(`key 1  ${f4(weight)}`, 444, 178, { c: c.blue, s: 15, w: 700 });
      d.text(`key 2  ${f4(1 - weight)}`, 444, 202, { c: c.muted, s: 15 });
    },
  });
})();

// Step-through: the causal mask on a 4 × 4 grid of scores, then softmax along each row.
(function () {
  const tokens = ["the", "cat", "sat", "on"], n = 4;
  // Made-up scores. Row i is the query at position i + 1, column j the key at position j + 1.
  const S = [[0.5, 1.2, 0.3, 2.0], [0.8, 0.2, 2.5, 0.4], [0.3, 1.4, 0.6, 1.8], [0.2, 1.6, 0.9, 0.5]];
  // Causal weights: softmax over the visible part of each row (columns up to the diagonal).
  const A = S.map((row, i) => {
    const e = row.map((v, j) => (j <= i ? Math.exp(v) : 0)), t = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / t);
  });
  const masked = n * (n - 1) / 2;
  const f2 = (v) => v.toFixed(2);
  const CW = 88, CH = 40, X0 = 170, Y0 = 70;
  Learn.demo(document.getElementById("demo-mask"), {
    height: 280,
    label: "A four by four grid of attention scores. The cells above the diagonal are masked, then each row becomes weights that add up to one",
    steps: [
      { s: { r2: 0, mk: 0, inf: 0, sm: 0 }, cap: "Four tokens give a 4 × 4 grid of scores (made-up numbers). Row i belongs to the query at position i. Column j belongs to the key at position j." },
      { s: { r2: 1 }, cap: `Row 2 is "cat". In training it has to predict the next token, "sat". Its biggest score, ${S[1][2]}, sits in column 3, which is "sat" itself. It could just copy the answer.` },
      { s: { r2: 0, mk: 1 }, cap: `Every cell above the diagonal has the same problem: its key comes later than its query. There are 3 + 2 + 1 = ${masked} of them.` },
      { s: { inf: 1 }, cap: `Overwrite those ${masked} scores with −∞. e raised to −∞ is 0, so softmax will give them zero weight.` },
      { s: { sm: 1 }, cap: `Run softmax along each row. Row 1 can only see itself, so its weight is ${f2(A[0][0])}. Row 2 splits its weight ${f2(A[1][0])} and ${f2(A[1][1])} between "the" and "cat".` },
      { s: { mk: 0 }, cap: "This grid is the weight matrix A. Nothing is left above the diagonal, and every row adds up to 1." },
    ],
    draw(ctx, s, c, d) {
      tokens.forEach((token, j) => d.text(`k${j + 1} ${token}`, X0 + CW * j + CW / 2, 48, { c: c.muted, s: 13, a: "center" }));
      for (let i = 0; i < n; i++) {
        const cy = Y0 + CH * i + CH / 2;
        d.text(`q${i + 1} ${tokens[i]}`, X0 - 12, cy, { c: c.muted, s: 13, a: "right" });
        for (let j = 0; j < n; j++) {
          const x = X0 + CW * j, y = Y0 + CH * i, cx = x + CW / 2;
          if (j <= i) {
            if (s.sm > 0.01) { ctx.globalAlpha = 0.55 * s.sm * A[i][j]; d.rect(x, y, CW, CH, c.orange); }
            ctx.globalAlpha = 1 - s.sm;
            d.text(f2(S[i][j]), cx, cy, { c: c.blue, s: 15 , a: "center" });
            ctx.globalAlpha = s.sm;
            d.text(f2(A[i][j]), cx, cy, { c: c.ink, s: 15, a: "center", w: 700 });
          } else {
            if (s.mk > 0.01) { ctx.globalAlpha = 0.22 * s.mk; d.rect(x, y, CW, CH, c.red); }
            ctx.globalAlpha = 1 - s.inf;
            d.text(f2(S[i][j]), cx, cy, { c: c.blue, s: 15, a: "center" });
            ctx.globalAlpha = s.inf * (1 - s.sm);
            d.text("−∞", cx, cy, { c: c.red, s: 15, a: "center", w: 700 });
            ctx.globalAlpha = s.sm;
            d.text("0", cx, cy, { c: c.muted, s: 15, a: "center" });
          }
          ctx.globalAlpha = 1;
          d.rect(x, y, CW, CH, null, c.muted, 1);
        }
        if (s.sm > 0.01) {
          ctx.globalAlpha = s.sm;
          d.text(`sum ${f2(A[i].reduce((a, b) => a + b, 0))}`, X0 + CW * n + 14, cy, { c: c.muted });
          ctx.globalAlpha = 1;
        }
      }
      d.rect(X0, Y0, CW * n, CH * n, null, c.ink, 2);
      if (s.r2 > 0.01) {
        ctx.globalAlpha = s.r2;
        d.rect(X0, Y0 + CH, CW * n, CH, null, c.orange, 3);
        d.rect(X0 + CW * 2 + 5, Y0 + CH + 5, CW - 10, CH - 10, null, c.red, 2);
        ctx.globalAlpha = 1;
      }
      const note = s.sm > 0.5 ? "weights: softmax of each row" : s.inf > 0.5 ? "scores, with later positions set to −∞" : "scores: row = query, column = key";
      d.text(note, X0, 254, { c: c.muted, s: 13 });
    },
  });
})();
