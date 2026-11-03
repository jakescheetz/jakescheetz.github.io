// Unit 08: stepping through one forward and backward pass, and a network learning XOR.
(function () {
  const { svg, stage, buttons, button, readout, loop, fmt } = Learn;
  const ROW = Learn.ROW;
  // fmt() with a real minus sign, and no trailing zeros on whole numbers.
  const num = (v, digits = 2) => {
    const s = Number(fmt(v, digits)).toString();
    return s.startsWith("-") ? "−" + s.slice(1) : s;
  };

  // Widget 1: forward values, then gradients, on a 2-2-1 network for the input (1, 0).
  (function () {
    const figure = document.getElementById("backprop-steps");
    if (!figure) return;
    const width = 720, rows = 16;

    const pos = {
      x1: [70, 58], x2: [70, 246],
      z1: [230, 58], z2: [230, 246],
      h1: [385, 58], h2: [385, 246],
      yh: [530, 152], L: [660, 152],
    };
    const names = { x1: "x₁", x2: "x₂", z1: "z₁", z2: "z₂", h1: "h₁", h2: "h₂", yh: "ŷ", L: "L" };
    const value = { x1: 1, x2: 0, z1: 0.5, z2: -0.5, h1: 0.5, h2: 0, yh: 0.5, L: 0.25 };
    const grad = { L: 1, yh: -1, h1: -1, h2: -0.5, z1: -1, z2: 0 };
    const edgeList = ["x1-z1", "x2-z1", "x1-z2", "x2-z2", "z1-h1", "z2-h2", "h1-yh", "h2-yh", "yh-L"];

    // One entry per click. `node` gets its value (forward) or gradient (backward) at this step;
    // `reuse` lists forward values the backward step reads again.
    const steps = [
      { node: ["x1", "x2"], text: "Inputs x = (1, 0). The target is y = 1." },
      { node: ["z1"], edges: ["x1-z1", "x2-z1"], text: "z₁ = 0.5·x₁ + (−0.5)·x₂ + 0 = 0.5" },
      { node: ["z2"], edges: ["x1-z2", "x2-z2"], text: "z₂ = (−1)·x₁ + 1·x₂ + 0.5 = −0.5" },
      { node: ["h1"], edges: ["z1-h1"], text: "h₁ = ReLU(z₁) = max(0, 0.5) = 0.5" },
      { node: ["h2"], edges: ["z2-h2"], text: "h₂ = ReLU(z₂) = max(0, −0.5) = 0" },
      { node: ["yh"], edges: ["h1-yh", "h2-yh"], text: "ŷ = 1·h₁ + 0.5·h₂ + 0 = 0.5" },
      { node: ["L"], edges: ["yh-L"], text: "L = (ŷ − y)² = (0.5 − 1)² = 0.25" },
      { back: true, node: ["L"], text: "∂L/∂L = 1. The backward pass starts at the loss." },
      { back: true, node: ["yh"], edges: ["yh-L"], reuse: ["yh"], text: "∂L/∂ŷ = ∂L/∂L · 2(ŷ − y) = 2(0.5 − 1) = −1" },
      { back: true, node: ["h1"], edges: ["h1-yh"], text: "∂L/∂h₁ = ∂L/∂ŷ · 1 = −1   (1 is h₁'s weight into ŷ)" },
      { back: true, node: ["h2"], edges: ["h2-yh"], text: "∂L/∂h₂ = ∂L/∂ŷ · 0.5 = −0.5   (0.5 is h₂'s weight)" },
      { back: true, node: ["z1"], edges: ["z1-h1"], reuse: ["z1"], text: "∂L/∂z₁ = ∂L/∂h₁ · 1 = −1   (z₁ > 0, so ReLU passes it on)" },
      { back: true, node: ["z2"], edges: ["z2-h2"], reuse: ["z2"], text: "∂L/∂z₂ = ∂L/∂h₂ · 0 = 0   (z₂ < 0, so ReLU blocks it)" },
      {
        back: true, node: [], edges: ["x1-z1", "x2-z1", "x1-z2", "x2-z2", "h1-yh", "h2-yh"], reuse: ["x1", "x2", "h1", "h2"],
        text: "∂L/∂W₂ = ∂L/∂ŷ · h = (−0.5, 0)   ∂L/∂b₂ = −1\n∂L/∂W₁ = ∂L/∂z times x: row 1 (−1, 0), row 2 (0, 0)   ∂L/∂b₁ = (−1, 0)",
      },
    ];
    // The step at which each node's value and gradient first appear.
    const valueAt = {}, gradAt = {};
    steps.forEach((s, i) => s.node.forEach((n) => { (s.back ? gradAt : valueAt)[n] = i; }));

    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "A network with inputs x1 and x2, weighted sums z1 and z2, ReLU outputs h1 and h2, output y-hat and loss L");
    const R = 24;
    const edges = {};
    for (const id of edgeList) {
      const [a, b] = id.split("-");
      const [x1, y1] = pos[a], [x2, y2] = pos[b];
      const d = Math.hypot(x2 - x1, y2 - y1);
      const ux = (x2 - x1) / d, uy = (y2 - y1) / d;
      edges[id] = svg("line", { x1: x1 + ux * R, y1: y1 + uy * R, x2: x2 - ux * R, y2: y2 - uy * R, class: "muted thin" }, root);
    }
    const nodes = {}, valueText = {}, gradText = {};
    for (const [id, [x, y]] of Object.entries(pos)) {
      const g = svg("g", { class: "muted" }, root);
      nodes[id] = svg("circle", { cx: x, cy: y, r: R, fill: "none", stroke: "currentColor", "stroke-width": 2 }, g);
      svg("text", { x, y, "text-anchor": "middle", "dominant-baseline": "central", text: names[id], class: "ink" }, g);
      // Values sit on a baseline just above the circle; gradients hang just below it, so both
      // clear the circle at any label size.
      valueText[id] = svg("text", { x, y: y - R - 7, "text-anchor": "middle", class: "blue" }, root);
      gradText[id] = svg("text", { x, y: y + R + 6, "text-anchor": "middle", "dominant-baseline": "hanging", class: "orange" }, root);
    }
    svg("text", { x: 230, y: 152, "text-anchor": "middle", "dominant-baseline": "central", class: "muted", text: "Wx+b" }, root);
    svg("text", { x: 385, y: 152, "text-anchor": "middle", "dominant-baseline": "central", class: "muted", text: "ReLU" }, root);

    const out = readout(figure);
    const btns = buttons(figure);
    let current = 0;

    function draw() {
      const s = steps[current];
      for (const id of edgeList) {
        const on = s.edges && s.edges.includes(id);
        edges[id].setAttribute("class", on ? (s.back ? "orange" : "blue") : "muted thin");
      }
      for (const id of Object.keys(pos)) {
        const active = s.node.includes(id);
        nodes[id].parentNode.setAttribute("class", active ? (s.back ? "orange" : "blue") : "muted");
        nodes[id].setAttribute("stroke-width", active ? 3 : 2);
        valueText[id].textContent = current >= valueAt[id] ? num(value[id]) : "";
        valueText[id].setAttribute("font-weight", s.reuse && s.reuse.includes(id) ? "bold" : "normal");
        valueText[id].setAttribute("text-decoration", s.reuse && s.reuse.includes(id) ? "underline" : "none");
        gradText[id].textContent = id in gradAt && current >= gradAt[id] ? num(grad[id]) : "";
      }
      const phase = s.back ? '<span class="orange">backward</span>' : '<span class="blue">forward</span>';
      const lines = s.text.split("\n");
      out.innerHTML =
        `W₁ = [[0.5, −0.5], [−1, 1]]  b₁ = (0, 0.5)  W₂ = (1, 0.5)  b₂ = 0\n` +
        `step ${current + 1} of ${steps.length}, ${phase}` +
        (s.reuse && s.back ? ", reusing the underlined forward values" : "") + "\n" +
        `${lines[0]}\n${lines[1] || " "}`;
      back.disabled = current === 0;
      next.disabled = current === steps.length - 1;
    }

    const back = button(btns, "Back", () => { current = Math.max(0, current - 1); draw(); });
    const next = button(btns, "Next", () => { current = Math.min(steps.length - 1, current + 1); draw(); });
    button(btns, "Start over", () => { current = 0; draw(); });
    draw();
  })();

  // Widget 2: a 2-H-1 ReLU network trained on XOR by full-batch gradient descent, live.
  (function () {
    const figure = document.getElementById("xor-training");
    if (!figure) return;
    const { controls, slider } = Learn;
    const width = 720, rows = 15, height = rows * ROW;

    const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const Y = [0, 1, 1, 0];
    const eta = 0.1, maxSteps = 1000, stepsPerSecond = 50;

    // Seeded random numbers so "New start" is repeatable from the same seed.
    function mulberry32(seed) {
      return function () {
        seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }
    function normal(rand) {
      const u = 1 - rand(), v = rand();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    let H = 8, seed = 1, net, step, history;
    function init() {
      const rand = mulberry32(seed * 7919);
      net = {
        W1: Array.from({ length: H }, () => [normal(rand), normal(rand)]),
        b1: new Array(H).fill(0),
        W2: Array.from({ length: H }, () => 0.5 * normal(rand)),
        b2: 0,
      };
      step = 0;
      history = [lossNow()];
    }
    function predict(x1, x2) {
      let y = net.b2;
      for (let j = 0; j < H; j++) {
        const z = net.W1[j][0] * x1 + net.W1[j][1] * x2 + net.b1[j];
        if (z > 0) y += net.W2[j] * z;
      }
      return y;
    }
    function lossNow() {
      let L = 0;
      for (let i = 0; i < 4; i++) L += (predict(X[i][0], X[i][1]) - Y[i]) ** 2;
      return L / 4;
    }
    // Same forward and backward pass as the Python in the worked example.
    function train() {
      const dW1 = Array.from({ length: H }, () => [0, 0]), db1 = new Array(H).fill(0), dW2 = new Array(H).fill(0);
      let db2 = 0;
      for (let i = 0; i < 4; i++) {
        const [x1, x2] = X[i];
        const z = net.W1.map((w, j) => w[0] * x1 + w[1] * x2 + net.b1[j]);
        const h = z.map((v) => Math.max(0, v));
        let y = net.b2;
        for (let j = 0; j < H; j++) y += net.W2[j] * h[j];
        const dy = 2 * (y - Y[i]) / 4;
        db2 += dy;
        for (let j = 0; j < H; j++) {
          dW2[j] += dy * h[j];
          const dz = z[j] > 0 ? dy * net.W2[j] : 0;
          dW1[j][0] += dz * x1; dW1[j][1] += dz * x2; db1[j] += dz;
        }
      }
      for (let j = 0; j < H; j++) {
        net.W1[j][0] -= eta * dW1[j][0]; net.W1[j][1] -= eta * dW1[j][1];
        net.b1[j] -= eta * db1[j]; net.W2[j] -= eta * dW2[j];
      }
      net.b2 -= eta * db2;
      step++;
      history.push(lossNow());
    }
    // A hidden unit is dead when its weighted sum is ≤ 0 on all four inputs: no gradient reaches it.
    function deadUnits() {
      let dead = 0;
      for (let j = 0; j < H; j++) {
        if (X.every(([x1, x2]) => net.W1[j][0] * x1 + net.W1[j][1] * x2 + net.b1[j] <= 0)) dead++;
      }
      return dead;
    }

    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "The network's output over the input square, the four XOR points, and the loss curve");

    // Left: output map over x1, x2 in [-0.5, 1.5]. Blue where the output is above 0.5, orange below.
    const mapSize = 250, mapX = 40, mapY = 14, lo = -0.5, hi = 1.5, cells = 25, cell = mapSize / cells;
    const MX = (v) => mapX + (v - lo) / (hi - lo) * mapSize;
    const MY = (v) => mapY + mapSize - (v - lo) / (hi - lo) * mapSize;
    const clipId = "xor-map-clip";
    const defs = svg("defs", {}, root);
    const clip = svg("clipPath", { id: clipId }, defs);
    svg("rect", { x: mapX, y: mapY, width: mapSize, height: mapSize }, clip);
    const cellRects = [];
    const blue = svg("g", { class: "blue" }, root), orange = svg("g", { class: "orange" }, root);
    for (let r = 0; r < cells; r++) {
      for (let c = 0; c < cells; c++) {
        const x1 = lo + (c + 0.5) / cells * (hi - lo), x2 = hi - (r + 0.5) / cells * (hi - lo);
        cellRects.push({ x1, x2, rect: svg("rect", { x: mapX + c * cell, y: mapY + r * cell, width: cell + 0.3, height: cell + 0.3, fill: "currentColor", stroke: "none" }) });
      }
    }
    const folds = svg("g", { class: "ink", opacity: 0.6, "clip-path": `url(#${clipId})` }, root);
    svg("rect", { x: mapX, y: mapY, width: mapSize, height: mapSize, fill: "none", stroke: "currentColor", class: "muted", "stroke-width": 1 }, root);
    for (let i = 0; i < 4; i++) {
      const g = svg("g", { class: Y[i] ? "blue" : "orange" }, root);
      svg("circle", { cx: MX(X[i][0]), cy: MY(X[i][1]), r: 8, fill: "currentColor", stroke: "var(--background-color)", "stroke-width": 2 }, g);
    }
    svg("text", { x: MX(0), y: mapY + mapSize + 20, "text-anchor": "middle", class: "muted", text: "0" }, root);
    svg("text", { x: MX(1), y: mapY + mapSize + 20, "text-anchor": "middle", class: "muted", text: "1" }, root);
    svg("text", { x: mapX + mapSize, y: mapY + mapSize + 20, "text-anchor": "end", class: "muted", text: "x₁" }, root);
    svg("text", { x: mapX - 8, y: MY(0) + 5, "text-anchor": "end", class: "muted", text: "0" }, root);
    svg("text", { x: mapX - 8, y: MY(1) + 5, "text-anchor": "end", class: "muted", text: "1" }, root);
    svg("text", { x: mapX - 8, y: mapY + 14, "text-anchor": "end", class: "muted", text: "x₂" }, root);

    // Right: loss on a log scale, from 1 down to 1e-6, over maxSteps steps.
    const px = 400, pw = 300, py = mapY, ph = mapSize;
    const PX = (s) => px + s / maxSteps * pw;
    const PY = (L) => py + Math.min(1, Math.max(0, -Math.log10(Math.max(L, 1e-6)) / 6)) * ph;
    svg("line", { x1: px, y1: py, x2: px, y2: py + ph, class: "muted thin" }, root);
    svg("line", { x1: px, y1: py + ph, x2: px + pw, y2: py + ph, class: "muted thin" }, root);
    for (const [L, label] of [[1, "1"], [1e-3, "1e-3"], [1e-6, "1e-6"]]) {
      svg("line", { x1: px - 4, y1: PY(L), x2: px, y2: PY(L), class: "muted thin" }, root);
      svg("text", { x: px - 8, y: PY(L) + 5, "text-anchor": "end", class: "muted", text: label }, root);
    }
    svg("text", { x: px + 10, y: py + 14, class: "muted", text: "loss" }, root);
    svg("text", { x: px + pw, y: py + ph + 20, "text-anchor": "end", class: "muted", text: `step ${maxSteps}` }, root);
    const curve = svg("polyline", { class: "purple" }, root);

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);

    function draw() {
      for (const { x1, x2, rect } of cellRects) {
        const y = predict(x1, x2);
        const strength = Math.min(1, Math.abs(y - 0.5) * 2);
        const parent = y >= 0.5 ? blue : orange;
        if (rect.parentNode !== parent) parent.appendChild(rect);
        rect.setAttribute("fill-opacity", (0.08 + 0.5 * strength).toFixed(3));
      }
      // Each hidden unit's fold: the line where its weighted sum crosses zero.
      folds.replaceChildren();
      for (let j = 0; j < H; j++) {
        const [a, b] = net.W1[j], c = net.b1[j];
        let p, q;
        if (Math.abs(b) > Math.abs(a)) {
          p = [lo, -(a * lo + c) / b]; q = [hi, -(a * hi + c) / b];
        } else {
          if (Math.abs(a) < 1e-9) continue;
          p = [-(b * lo + c) / a, lo]; q = [-(b * hi + c) / a, hi];
        }
        svg("line", { x1: MX(p[0]), y1: MY(p[1]), x2: MX(q[0]), y2: MY(q[1]), class: "dashed thin" }, folds);
      }
      const stride = Math.max(1, Math.floor(history.length / 400));
      const pts = [];
      for (let s = 0; s < history.length; s += stride) pts.push(`${PX(s).toFixed(1)},${PY(history[s]).toFixed(1)}`);
      pts.push(`${PX(history.length - 1).toFixed(1)},${PY(history[history.length - 1]).toFixed(1)}`);
      curve.setAttribute("points", pts.join(" "));

      const preds = X.map(([a, b]) => num(predict(a, b)));
      const L = history[history.length - 1];
      out.innerHTML =
        `step ${step}   <span class="purple">loss ${L < 1e-4 ? L.toExponential(1) : fmt(L, 4)}</span>\n` +
        `ŷ at (0,0) ${preds[0]}  (0,1) ${preds[1]}  (1,0) ${preds[2]}  (1,1) ${preds[3]}\n` +
        `hidden units: ${H}, dead (z ≤ 0 on all four inputs): ${deadUnits()}`;
    }

    init();
    let carry = 0, player = null, toggle = null;
    player = loop(figure, (dt) => {
      if (dt && step < maxSteps) {
        carry += dt * stepsPerSecond;
        while (carry >= 1 && step < maxSteps) { train(); carry--; }
      }
      draw();
      if (step >= maxSteps && player && player.playing) { player.pause(); toggle.textContent = "Play"; }
    });
    toggle = button(btns, player.playing ? "Pause" : "Play", () => {
      if (step >= maxSteps) init();
      if (player.playing) player.pause(); else player.play();
      toggle.textContent = player.playing ? "Pause" : "Play";
      draw();
    });
    button(btns, "+100 steps", () => {
      for (let i = 0; i < 100 && step < maxSteps; i++) train();
      draw();
    });
    button(btns, "New start", () => { seed++; init(); draw(); });
    slider(ctl, { label: "hidden units", min: 1, max: 8, step: 1, value: H, format: (v) => String(v) }, (v) => {
      H = v; init(); draw();
    });
  })();
})();
