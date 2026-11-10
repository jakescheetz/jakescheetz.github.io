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
        lines.push(`next: <span class="orange">${state.next.pair.join(" + ")}</span>, ${state.next.count} times`);
        lines.push(parts.length > 1
          ? `= ${parts.map(([word, n]) => `${n} (${word})`).join(" + ")}`
          : `all in ${parts[0][0]}`);
        if (state.ties.length) {
          const tied = state.ties.map((p) => p.pair.join(" + ")).join(", ");
          lines.push(`tied: ${tied} (first wins)`);
        }
      } else {
        lines.push(`next: none, every word is one token`);
      }
      // Always four lines, so the buttons below don't jump when a tie line comes and goes.
      while (lines.length < 4) lines.push("\u00a0");
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

// Step-through: the first four BPE merges on the toy corpus, with the pair counts behind them.
(function () {
  const corpus = [["low", 5], ["lower", 2], ["newest", 6], ["widest", 3]];
  const glue = (t, [a, b]) => {
    const out = [];
    for (let i = 0; i < t.length; i++) {
      if (t[i] === a && t[i + 1] === b) { out.push(a + b); i++; } else out.push(t[i]);
    }
    return out;
  };
  // states[m] is the corpus after m merges: its tokens, its pairs ranked by count (ties keep
  // the order they were counted in, like the Python), its length and the token just made.
  const states = [];
  let splits = corpus.map(([word]) => [...word]), made = "";
  for (let m = 0; m <= 4; m++) {
    const pairs = new Map();
    corpus.forEach(([, count], w) => {
      const t = splits[w];
      for (let i = 0; i + 1 < t.length; i++) {
        const key = `${t[i]} + ${t[i + 1]}`;
        if (!pairs.has(key)) pairs.set(key, { key, pair: [t[i], t[i + 1]], count: 0 });
        pairs.get(key).count += count;
      }
    });
    const ranked = [...pairs.values()].sort((p, q) => q.count - p.count);
    const length = splits.reduce((sum, t, w) => sum + t.length * corpus[w][1], 0);
    states.push({ splits, ranked, length, made });
    made = ranked[0].pair.join("");
    splits = splits.map((t) => glue(t, ranked[0].pair));
  }
  const S = states;
  Learn.demo(document.getElementById("demo-bpe"), {
    height: 320,
    label: "Four words split into tokens, a table of pair counts, and the vocabulary and corpus sizes after each merge",
    steps: [
      { s: { m: 0, cnt: 0, hl: 0 }, cap: `The corpus is four words with their counts. Split every word into letters. That gives 10 distinct letters and ${S[0].length} tokens in total.` },
      { s: { cnt: 1, hl: 1 }, cap: `Count every pair of neighbors. e + s sits in newest (6 times) and widest (3 times), so it scores ${S[0].ranked[0].count}. s + t ties at ${S[0].ranked[1].count}, but e + s was counted first.` },
      { s: { m: 1, hl: 0 }, cap: `Merge 1: glue e + s into the new token es everywhere. The vocabulary grows to 11. The corpus shrinks by ${S[0].ranked[0].count}, from ${S[0].length} to ${S[1].length} tokens.` },
      { s: { hl: 1 }, cap: `Count again. Now ${S[1].ranked[0].key} is the top pair, also ${S[1].ranked[0].count} times.` },
      { s: { m: 2, hl: 0 }, cap: `Merge 2: es + t becomes est. Vocabulary 12, corpus ${S[2].length} tokens. The ending est now takes one token in both newest and widest.` },
      { s: { m: 4 }, cap: `Two more merges: ${S[2].ranked[0].key}, then ${S[3].ranked[0].key}, ${S[3].ranked[0].count} times each. Now low is a single token. Vocabulary 14, corpus ${S[4].length} tokens.` },
    ],
    draw(ctx, s, c, d) {
      const m = d.clamp(Math.round(s.m), 0, 4), state = states[m], next = state.ranked[0].pair;
      d.text("corpus", 24, 30, { c: c.muted, s: 13 });
      corpus.forEach(([, count], w) => {
        const y = 78 + 58 * w, tokens = state.splits[w];
        d.text(`${count}×`, 24, y, { c: c.muted, s: 15 });
        const inPair = tokens.map(() => false);
        for (let i = 0; i + 1 < tokens.length; i++) {
          if (tokens[i] === next[0] && tokens[i + 1] === next[1]) { inPair[i] = inPair[i + 1] = true; i++; }
        }
        let x = 64;
        tokens.forEach((token, i) => {
          const bw = token.length * 9.6 + 18;
          d.rect(x, y - 17, bw, 34, c.bg, c.muted, 2);
          if (inPair[i] && s.hl > 0.01) {
            ctx.globalAlpha = s.hl;
            d.rect(x, y - 17, bw, 34, null, c.orange, 3);
            ctx.globalAlpha = 1;
          }
          const fresh = token === state.made;
          d.text(token, x + bw / 2, y, { c: fresh ? c.blue : c.ink, s: 16, a: "center", w: fresh ? 700 : 500 });
          x += bw + 6;
        });
      });
      if (s.cnt > 0.01) {
        ctx.globalAlpha = s.cnt;
        d.panel(c, 330, 50, 290, 146);
        d.text("pair counts (top 4)", 346, 72, { c: c.muted, s: 13 });
        state.ranked.slice(0, 4).forEach((p, i) => {
          const top = i === 0;
          d.text(p.key, 346, 100 + 24 * i, { c: top ? c.orange : c.ink, w: top ? 700 : 500 });
          d.text(String(p.count), 600, 100 + 24 * i, { c: top ? c.orange : c.ink, w: top ? 700 : 500, a: "right" });
        });
        ctx.globalAlpha = 1;
      }
      d.text(`vocabulary: ${10 + m} tokens`, 346, 232, { c: c.blue, s: 15, w: 700 });
      d.text(`corpus: ${state.length} tokens`, 346, 260, { c: c.ink, s: 15, w: 700 });
      if (m > 0) {
        const last = states[m - 1].ranked[0];
        d.text(`last merge: ${last.key} → ${state.made}`, 346, 292, { c: c.muted });
      }
    },
  });
})();

// Step-through: tokenizing the unseen word "lowest" by replaying the merge list in order.
(function () {
  const merges = [["e", "s"], ["es", "t"], ["l", "o"], ["lo", "w"], ["n", "e"], ["ne", "w"], ["new", "est"], ["w", "i"], ["wi", "d"], ["wid", "est"]];
  const vocab = [..."deilnorstw", ...merges.map((p) => p.join(""))];
  const glue = (t, [a, b]) => {
    const out = [];
    for (let i = 0; i < t.length; i++) {
      if (t[i] === a && t[i + 1] === b) { out.push(a + b); i++; } else out.push(t[i]);
    }
    return out;
  };
  // seq[p] is the word after the first p merges have been tried.
  const seq = [[..."lowest"]];
  merges.forEach((pair) => seq.push(glue(seq[seq.length - 1], pair)));
  const hit = merges.map((_, i) => seq[i + 1].length < seq[i].length);
  const final = seq[10], finalIds = final.map((t) => vocab.indexOf(t));
  Learn.demo(document.getElementById("demo-tokenize"), {
    height: 330,
    label: "The ten learned merges in order, and the word lowest shrinking from six letters to two tokens as they are applied",
    steps: [
      { s: { p: 0, ids: 0 }, cap: `The word lowest never appeared in the corpus. Start the same way training did: split it into letters. ${seq[0].length} tokens.` },
      { s: { p: 1 }, cap: `Apply merge 1, e + s. The word has an e next to an s, so they become es. ${seq[1].length} tokens.` },
      { s: { p: 2 }, cap: `Merge 2, es + t, gives est. ${seq[2].length} tokens.` },
      { s: { p: 3 }, cap: `Merge 3, l + o, gives lo. ${seq[3].length} tokens.` },
      { s: { p: 4 }, cap: `Merge 4, lo + w, gives low. ${seq[4].length} tokens.` },
      { s: { p: 10 }, cap: `Merges 5 to 10 find no matching pair in this word, so nothing changes. The result is ${final.join(" + ")}.` },
      { s: { ids: 1 }, cap: `Replace each token with its number in the vocabulary: low is ID ${finalIds[0]} and est is ID ${finalIds[1]}. The model receives ${finalIds.join(", ")} and never sees the letters.` },
    ],
    draw(ctx, s, c, d) {
      const p = d.clamp(Math.round(s.p), 0, 10), tokens = seq[p];
      d.text("merge list, in the order learned", 20, 24, { c: c.muted, s: 13 });
      merges.forEach((pair, i) => {
        const y = 52 + 26 * i, tried = i < p, now = i === p - 1;
        if (now) d.rect(16, y - 13, 250, 26, c.soft, c.orange, 2);
        d.text(String(i + 1).padStart(2, " "), 28, y, { c: c.muted });
        d.text(pair.join(" + "), 62, y, { c: tried ? c.ink : c.muted, w: now ? 700 : 500 });
        if (tried) d.text(hit[i] ? "applied" : "no match", 176, y, { c: hit[i] ? c.blue : c.muted, s: 13, w: hit[i] ? 700 : 500 });
      });
      d.text("\"lowest\"", 450, 44, { c: c.ink, s: 18, w: 700, a: "center" });
      const widths = tokens.map((t) => t.length * 12 + 20);
      const total = widths.reduce((a, b) => a + b, 0) + 8 * (tokens.length - 1);
      const made = p > 0 && hit[p - 1] ? merges[p - 1].join("") : "";
      let x = 450 - total / 2;
      tokens.forEach((token, i) => {
        const fresh = token === made;
        d.rect(x, 88, widths[i], 44, c.bg, fresh ? c.orange : c.muted, fresh ? 3 : 2);
        d.text(token, x + widths[i] / 2, 110, { c: fresh ? c.orange : c.ink, s: 20, a: "center", w: 700 });
        if (s.ids > 0.01 && p === 10) {
          ctx.globalAlpha = s.ids;
          d.arrow(x + widths[i] / 2, 138, x + widths[i] / 2, 186, c.muted, 2);
          d.text(`ID ${vocab.indexOf(token)}`, x + widths[i] / 2, 204, { c: c.blue, s: 16, a: "center", w: 700 });
          ctx.globalAlpha = 1;
        }
        x += widths[i] + 8;
      });
      if (s.ids < 0.99) {
        ctx.globalAlpha = 1 - s.ids;
        d.text(`${tokens.length} tokens`, 450, 160, { c: c.muted, a: "center" });
        ctx.globalAlpha = 1;
      }
      if (s.ids > 0.01) {
        ctx.globalAlpha = s.ids;
        d.text(`the model receives [${finalIds.join(", ")}]`, 450, 252, { c: c.ink, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: token IDs 13 and 11 each pull one row out of the 20 × 4 embedding table.
(function () {
  // Rows of the table from the worked example (E = default_rng(1).normal(size=(20, 4)).round(2)).
  const rows = [
    [0, "d", [0.35, 0.82, 0.33, -1.30]], [1, "e", [0.91, 0.45, -0.54, 0.58]], null,
    [11, "est", [-0.10, 0.10, 0.04, -0.51]], [12, "lo", [0.59, 0.89, 0.32, -0.82]],
    [13, "low", [0.73, -0.50, 0.88, -1.07]], [14, "ne", [0.91, -0.02, -1.25, -0.31]], null,
    [19, "widest", [-0.35, -1.48, -0.11, -0.45]],
  ];
  const num = (v) => (v < 0 ? "−" : " ") + Math.abs(v).toFixed(2);
  const vec = (v) => `(${v.map(num).join("  ")})`;
  const plain = (v) => `(${v.map((x) => num(x).trim()).join(", ")})`;
  const LOW = 5, EST = 3; // where rows 13 and 11 sit in the list above
  const rowY = (k) => 62 + 26 * k;
  Learn.demo(document.getElementById("demo-lookup"), {
    height: 350,
    label: "Two token IDs on the left, the embedding table in the middle, and the two rows the lookup returns below it",
    steps: [
      { s: { tab: 0, hl: 0, pick: LOW, h1: 0, h2: 0, oh: 0, tr: 0 }, cap: "The tokenizer turned lowest into two IDs: 13 and 11. An ID is only a label, like a locker number. Arithmetic on it would mean nothing." },
      { s: { tab: 1 }, cap: "The model keeps an embedding table E with one row per token. This toy table has 20 rows of 4 numbers each. The dots stand for rows left out of the picture." },
      { s: { hl: 1, h1: 1 }, cap: `Look up ID 13: take row 13 of E. That row, ${plain(rows[LOW][2])}, is now the vector for low.` },
      { s: { pick: EST, h2: 1 }, cap: `Look up ID 11 the same way. Row 11, ${plain(rows[EST][2])}, is now the vector for est.` },
      { s: { pick: LOW, oh: 1 }, cap: "The same lookup as math. Write ID 13 as a one-hot vector u: 20 entries, all 0 except a 1 at position 13. Multiply each row by its entry of u and add them up. Only row 13 survives." },
      { s: { oh: 0, hl: 0, tr: 1 }, cap: "2 IDs went in and 2 vectors of 4 numbers came out. In training, only the rows that were looked up (13 and 11) get a gradient through the lookup." },
    ],
    draw(ctx, s, c, d) {
      d.text("\"lowest\"", 24, 40, { c: c.ink, s: 16, w: 700 });
      [["low", 13], ["est", 11]].forEach(([token, id], i) => {
        const x = 24 + 68 * i;
        d.rect(x, 66, 58, 34, c.bg, c.muted, 2);
        d.text(token, x + 29, 83, { c: c.ink, s: 16, a: "center" });
        d.arrow(x + 29, 104, x + 29, 130, c.muted, 2);
        d.text(String(id), x + 29, 146, { c: c.blue, s: 18, a: "center", w: 700 });
      });
      d.text("token IDs", 24, 174, { c: c.muted, s: 13 });
      if (s.tab > 0.01) {
        const pick = Math.round(s.pick);
        ctx.globalAlpha = s.tab;
        d.text("id", 240, 34, { c: c.muted, s: 13, a: "right" });
        d.text("token", 256, 34, { c: c.muted, s: 13 });
        d.text("row of E", 362, 34, { c: c.muted, s: 13 });
        if (s.hl > 0.01) {
          ctx.globalAlpha = s.tab * s.hl;
          d.rect(210, 49 + 26 * s.pick, 404, 26, c.soft, c.blue, 2);
        }
        if (s.tr > 0.01) {
          ctx.globalAlpha = s.tab * s.tr;
          d.rect(210, rowY(LOW) - 13, 404, 26, c.soft, c.orange, 2);
          d.rect(210, rowY(EST) - 13, 404, 26, c.soft, c.orange, 2);
        }
        rows.forEach((row, k) => {
          const y = rowY(k);
          ctx.globalAlpha = s.tab;
          if (!row) { d.text("⋮", 232, y, { c: c.muted, a: "center" }); return; }
          const on = (s.hl > 0.5 && k === pick) || (s.tr > 0.5 && (k === LOW || k === EST));
          const color = on ? (s.tr > 0.5 ? c.orange : c.blue) : c.ink;
          d.text(String(row[0]), 240, y, { c: on ? color : c.muted, a: "right", w: on ? 700 : 500 });
          d.text(row[1], 256, y, { c: color, w: on ? 700 : 500 });
          d.text(vec(row[2]), 362, y, { c: color, w: on ? 700 : 500 });
          if (s.oh > 0.01) {
            ctx.globalAlpha = s.tab * s.oh;
            d.text(k === LOW ? "1" : "0", 326, y, { c: k === LOW ? c.orange : c.muted, a: "center", w: k === LOW ? 700 : 500 });
            d.text("×", 346, y, { c: c.muted, a: "center" });
          }
        });
        if (s.oh > 0.01) {
          ctx.globalAlpha = s.tab * s.oh;
          d.text("u", 326, 34, { c: c.orange, s: 13, a: "center", w: 700 });
        }
        ctx.globalAlpha = s.tab;
        d.line(210, 288, 614, 288, c.muted, 2);
        ctx.globalAlpha = 1;
      }
      [[s.h1, "vector for low", rows[LOW][2], 308], [s.h2, "vector for est", rows[EST][2], 332]].forEach(([alpha, name, v, y]) => {
        if (alpha < 0.01) return;
        ctx.globalAlpha = alpha;
        d.text(name, 350, y, { c: c.muted, a: "right" });
        d.text(vec(v), 362, y, { c: c.blue, w: 700 });
        ctx.globalAlpha = 1;
      });
    },
  });
})();
