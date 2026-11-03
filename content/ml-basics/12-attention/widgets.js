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
        `<span class="blue">query "${tok}" (position ${query + 1}), q = ${vec(q)}</span>\n` +
        `biggest weight on "${bestTok}" (position ${best + 1}), k = ${vec(keys[best])}\n` +
        `score = q · k / √4 = ${fmt(raw[best])} / 2 = ${fmt(raw[best] / 2)}, × scale ${scale.toFixed(1)} = ${fmt(scores[best])}\n` +
        `<span class="orange">weight on "${bestTok}" = ${fmt(w[best])}</span>, ` +
        `weights on positions 1 to ${query + 1} sum to ${fmt(sum)}` +
        (query < n - 1 ? `, ${query + 2} to ${n} masked` : "");
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
        `spread (std) of raw q · k over 1,000 random pairs: ${fmt(spread(d), 1)}   √d_k = ${fmt(Math.sqrt(d), 1)}\n` +
        `<span class="blue">these 8 scores run from ${fmt(lo, 1).replace("-", "−")} to ${fmt(hi, 1).replace("-", "−")}</span>\n` +
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
