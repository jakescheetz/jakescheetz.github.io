// Unit 10: byte-pair encoding one merge at a time, and nearest neighbors in GPT-2's embedding table.
(function () {
  const { svg, stage, buttons, button, readout, loop } = Learn;
  const R = Learn.ROW;

  // Label metrics follow the stage's font size, which Learn.stage scales up on narrow screens.
  const fontSize = (root) => parseFloat(getComputedStyle(root).fontSize) || 13;
  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

  // BPE --------------------------------------------------------------------------------------

  (function () {
    const figure = document.getElementById("bpe-merges");
    if (!figure) return;

    // The toy corpus from the worked example (and from Sennrich et al.'s paper).
    const corpus = [["low", 5], ["lower", 2], ["newest", 6], ["widest", 3]];

    // Same rules as the Python: pairs are counted word by word, left to right, weighted by how
    // often the word appears, and a tie goes to the pair that was counted first.
    function countPairs(splits) {
      const pairs = new Map();
      corpus.forEach(([word, count], w) => {
        const t = splits[w];
        for (let i = 0; i + 1 < t.length; i++) {
          const key = `${t[i]} ${t[i + 1]}`;
          if (!pairs.has(key)) pairs.set(key, { pair: [t[i], t[i + 1]], count: 0, parts: new Map() });
          const entry = pairs.get(key);
          entry.count += count;
          entry.parts.set(word, (entry.parts.get(word) || 0) + count);
        }
      });
      return [...pairs.values()];
    }

    function merge(tokens, [a, b]) {
      const out = [], fresh = [];
      for (let i = 0; i < tokens.length; i++) {
        if (i + 1 < tokens.length && tokens[i] === a && tokens[i + 1] === b) {
          out.push(a + b); fresh.push(true); i++;
        } else {
          out.push(tokens[i]); fresh.push(false);
        }
      }
      return { out, fresh };
    }

    // Run BPE to the end once and keep every intermediate state.
    const states = [];
    let splits = corpus.map(([word]) => [...word]);
    let fresh = splits.map((t) => t.map(() => false));
    let last = null;
    for (;;) {
      const pairs = countPairs(splits);
      let next = null;
      for (const p of pairs) if (!next || p.count > next.count) next = p;
      const ties = next ? pairs.filter((p) => p !== next && p.count === next.count) : [];
      const length = splits.reduce((sum, t, w) => sum + t.length * corpus[w][1], 0);
      states.push({ splits, fresh, last, next, ties, length });
      if (!next) break;
      const merged = splits.map((t) => merge(t, next.pair));
      splits = merged.map((m) => m.out);
      fresh = merged.map((m) => m.fresh);
      last = next.pair;
    }
    const baseVocab = new Set(corpus.flatMap(([word]) => [...word])).size;
    const maxVocab = baseVocab + states.length - 1;
    const maxLength = states[0].length;

    const width = 720, rows = 11;
    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "The four words of the corpus split into tokens, with the most frequent adjacent pair outlined");
    const words = svg("g", {}, root);
    const bars = svg("g", {}, root);
    const out = readout(figure);
    const btns = buttons(figure);

    let k = 0;

    function drawWords(fs) {
      clear(words);
      const charW = 0.6 * fs, pad = 0.45 * fs, gap = 0.4 * fs, bh = 1.5 * fs;
      const state = states[k];
      const pair = state.next && state.next.pair;
      corpus.forEach(([, count], w) => {
        const cy = R * (1.75 + 2.5 * w);
        svg("text", { x: 0, y: cy + 0.35 * fs, class: "muted", text: `${count}×` }, words);
        const tokens = state.splits[w];
        // Mark the occurrences the next merge will join, left to right without overlaps.
        const inPair = tokens.map(() => false);
        for (let i = 0; pair && i + 1 < tokens.length; i++) {
          if (tokens[i] === pair[0] && tokens[i + 1] === pair[1]) { inPair[i] = inPair[i + 1] = true; i++; }
        }
        let x = 3 * charW + 0.8 * fs;
        const lefts = [];
        tokens.forEach((token, i) => {
          const bw = token.length * charW + 2 * pad;
          lefts.push([x, x + bw]);
          svg("rect", {
            x, y: cy - bh / 2, width: bw, height: bh,
            class: inPair[i] ? "orange" : "muted",
            fill: "none", stroke: "currentColor", "stroke-width": inPair[i] ? 2.5 : 1,
          }, words);
          svg("text", {
            x: x + bw / 2, y: cy + 0.35 * fs, "text-anchor": "middle",
            class: state.fresh[w][i] ? "blue" : "ink", text: token,
          }, words);
          x += bw + gap;
        });
        for (let i = 0; i + 1 < tokens.length; i++) {
          if (inPair[i] && inPair[i + 1]) {
            const y = cy + bh / 2 + 4;
            svg("line", { x1: lefts[i][0], y1: y, x2: lefts[i + 1][1], y2: y, class: "orange" }, words);
            i++;
          }
        }
      });
    }

    function drawBars(fs) {
      clear(bars);
      const x0 = 400, full = 300, bh = 0.8 * R;
      const vocab = baseVocab + k;
      const length = states[k].length;
      const bar = (labelY, label, value, max, cls) => {
        svg("text", { x: x0, y: labelY, class: cls, text: label }, bars);
        const y = labelY + 0.35 * fs;
        svg("rect", { x: x0, y, width: full, height: bh, class: "muted faint", fill: "none", stroke: "currentColor", "stroke-width": 1 }, bars);
        svg("rect", { x: x0, y, width: full * value / max, height: bh, class: cls, fill: "currentColor" }, bars);
      };
      bar(R * 2.5, `vocabulary: ${vocab}`, vocab, maxVocab, "blue");
      bar(R * 6.5, `corpus: ${length} tokens`, length, maxLength, "ink");
    }

    function draw() {
      const fs = fontSize(root);
      drawWords(fs);
      drawBars(fs);
      const state = states[k];
      const total = states.length - 1;
      const lines = [state.last
        ? `merge ${k} of ${total}: ${state.last[0]} + ${state.last[1]} → <span class="blue">${state.last.join("")}</span>`
        : `start: ${baseVocab} single characters`];
      if (state.next) {
        const parts = [...state.next.parts];
        const seen = parts.length > 1
          ? `${parts.map(([word, n]) => `${n} (${word})`).join(" + ")} = ${state.next.count} times`
          : `${state.next.count} times (${parts[0][0]})`;
        lines.push(`next: <span class="orange">${state.next.pair.join(" + ")}</span>, ${seen}`);
        if (state.ties.length) {
          const tied = state.ties.map((p) => p.pair.join(" + ")).join(", ");
          lines.push(`tie with ${tied}: the pair counted first wins`);
        }
      } else {
        lines.push(`next: none, every word is one token`);
      }
      // Always three lines, so the buttons below don't jump when a tie line comes and goes.
      while (lines.length < 3) lines.push("\u00a0");
      out.innerHTML = lines.join("\n");
    }

    let wait = 0;
    const player = loop(figure, (dt) => {
      if (dt) {
        wait += dt;
        // One merge every 1.4 s, then a longer pause on the finished state before starting over.
        if (wait > (k === states.length - 1 ? 3.5 : 1.4)) {
          wait = 0;
          k = k === states.length - 1 ? 0 : k + 1;
          draw();
        }
      }
    });
    const stop = () => { player.pause(); toggle.textContent = "Play"; };
    button(btns, "Back", () => { stop(); k = Math.max(0, k - 1); draw(); });
    button(btns, "Next merge", () => { stop(); k = Math.min(states.length - 1, k + 1); draw(); });
    const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
      if (player.playing) player.pause(); else { wait = 0; player.play(); }
      toggle.textContent = player.playing ? "Pause" : "Play";
    });
    button(btns, "Reset", () => { stop(); k = 0; draw(); });
    window.addEventListener("resize", draw);
    draw();
  })();

  // Embedding neighbors ----------------------------------------------------------------------

  (function () {
    const figure = document.getElementById("embedding-neighbors");
    if (!figure) return;

    // Cosine similarities between rows of GPT-2 small's token embedding table (wte), for these
    // twelve tokens, each with its leading space. Computed once from the released weights.
    const tokens = ["Monday", "Tuesday", "seven", "eight", "red", "blue", "cat", "dog", "lowest", "highest", "the", "however"];
    const ids = [3321, 3431, 3598, 3624, 2266, 4171, 3797, 3290, 9016, 4511, 262, 2158];
    const sim = [
      [1.00, 0.89, 0.35, 0.31, 0.16, 0.22, 0.22, 0.26, 0.25, 0.23, 0.22, 0.23],
      [0.89, 1.00, 0.35, 0.33, 0.21, 0.25, 0.19, 0.27, 0.26, 0.21, 0.21, 0.24],
      [0.35, 0.35, 1.00, 0.86, 0.31, 0.31, 0.28, 0.26, 0.29, 0.30, 0.30, 0.25],
      [0.31, 0.33, 0.86, 1.00, 0.28, 0.31, 0.22, 0.26, 0.31, 0.31, 0.30, 0.24],
      [0.16, 0.21, 0.31, 0.28, 1.00, 0.64, 0.27, 0.24, 0.23, 0.25, 0.30, 0.21],
      [0.22, 0.25, 0.31, 0.31, 0.64, 1.00, 0.25, 0.28, 0.28, 0.26, 0.25, 0.21],
      [0.22, 0.19, 0.28, 0.22, 0.27, 0.25, 1.00, 0.55, 0.20, 0.19, 0.21, 0.19],
      [0.26, 0.27, 0.26, 0.26, 0.24, 0.28, 0.55, 1.00, 0.21, 0.19, 0.19, 0.19],
      [0.25, 0.26, 0.29, 0.31, 0.23, 0.28, 0.20, 0.21, 1.00, 0.80, 0.12, 0.23],
      [0.23, 0.21, 0.30, 0.31, 0.25, 0.26, 0.19, 0.19, 0.80, 1.00, 0.18, 0.22],
      [0.22, 0.21, 0.30, 0.30, 0.30, 0.25, 0.21, 0.19, 0.12, 0.18, 1.00, 0.30],
      [0.23, 0.24, 0.25, 0.24, 0.21, 0.21, 0.19, 0.19, 0.23, 0.22, 0.30, 1.00],
    ];

    const width = 720, pitch = 1.6 * R, others = tokens.length - 1;
    const rows = Math.ceil(others * 1.6 + 0.5);
    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "Cosine similarity between the chosen token's embedding and eleven others, sorted");
    const x0 = 170, full = 480;
    const zero = svg("line", { x1: x0, y1: 0, x2: x0, y2: rows * R, class: "muted thin" }, root);
    zero.setAttribute("aria-hidden", "true");

    // One group per token, moved into place with a CSS transform so reordering slides.
    const rowsById = tokens.map((token) => {
      const g = svg("g", { style: "transition: transform 0.45s ease" }, root);
      const label = svg("text", { x: x0 - 10, "text-anchor": "end", text: token }, g);
      const bar = svg("rect", { x: x0, fill: "currentColor", height: 0.7 * R }, g);
      const value = svg("text", {}, g);
      return { g, label, bar, value };
    });

    const out = readout(figure);
    const btns = buttons(figure);
    const chooser = tokens.map((token, i) => button(btns, token, () => choose(i)));

    let current = 0;
    function choose(i) {
      current = i;
      const fs = fontSize(root);
      const order = tokens.map((_, j) => j).filter((j) => j !== i).sort((a, b) => sim[i][b] - sim[i][a]);
      rowsById.forEach((row, j) => {
        if (j === i) { row.g.style.display = "none"; return; }
        row.g.style.display = "";
        const rank = order.indexOf(j);
        const cy = pitch * (rank + 0.5) + 0.25 * R;
        row.g.style.transform = `translate(0px, ${cy}px)`;
        const w = full * sim[i][j];
        row.label.setAttribute("y", 0.35 * fs);
        row.bar.setAttribute("y", -0.35 * R);
        row.bar.setAttribute("width", w);
        row.value.setAttribute("x", x0 + w + 8);
        row.value.setAttribute("y", 0.35 * fs);
        row.value.textContent = sim[i][j].toFixed(2);
        row.g.setAttribute("class", rank === 0 ? "blue" : "muted");
        row.label.setAttribute("class", rank === 0 ? "blue" : "ink");
      });
      chooser.forEach((b, j) => {
        b.setAttribute("aria-pressed", j === i ? "true" : "false");
        // The chosen token's button takes the same blue as its name in the readout.
        b.style.color = j === i ? "var(--accent-blue)" : "";
        b.style.borderColor = j === i ? "var(--accent-blue)" : "";
      });
      const rest = order.slice(1).map((j) => sim[i][j]);
      out.innerHTML =
        `<span class="blue">" ${tokens[i]}"</span> is token ${ids[i]}, row ${ids[i]} of E\n` +
        `nearest here: <span class="blue">" ${tokens[order[0]]}"</span> at ${sim[i][order[0]].toFixed(2)}\n` +
        `the other ten: ${Math.min(...rest).toFixed(2)} to ${Math.max(...rest).toFixed(2)}`;
    }
    window.addEventListener("resize", () => choose(current));
    choose(0);
  })();
})();
