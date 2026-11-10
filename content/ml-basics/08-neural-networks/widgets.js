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
      { back: true, node: ["L"], text: "∂L/∂L = 1.\nThe backward pass starts at the loss." },
      { back: true, node: ["yh"], edges: ["yh-L"], reuse: ["yh"], text: "∂L/∂ŷ = ∂L/∂L · 2(ŷ − y)\n      = 2(0.5 − 1) = −1" },
      { back: true, node: ["h1"], edges: ["h1-yh"], text: "∂L/∂h₁ = ∂L/∂ŷ · 1 = −1\n(1 is h₁'s weight into ŷ)" },
      { back: true, node: ["h2"], edges: ["h2-yh"], text: "∂L/∂h₂ = ∂L/∂ŷ · 0.5 = −0.5\n(0.5 is h₂'s weight)" },
      { back: true, node: ["z1"], edges: ["z1-h1"], reuse: ["z1"], text: "∂L/∂z₁ = ∂L/∂h₁ · 1 = −1\n(z₁ > 0, so ReLU passes it on)" },
      { back: true, node: ["z2"], edges: ["z2-h2"], reuse: ["z2"], text: "∂L/∂z₂ = ∂L/∂h₂ · 0 = 0\n(z₂ < 0, so ReLU blocks it)" },
      {
        back: true, node: [], edges: ["x1-z1", "x2-z1", "x1-z2", "x2-z2", "h1-yh", "h2-yh"], reuse: ["x1", "x2", "h1", "h2"],
        text: "∂L/∂W₂ = ∂L/∂ŷ · h = (−0.5, 0)\n∂L/∂b₂ = −1   ∂L/∂b₁ = (−1, 0)\n∂L/∂W₁ = ∂L/∂z times x:\n  row 1 (−1, 0), row 2 (0, 0)",
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
      // Always four text lines, so the buttons below don't jump between steps.
      const lines = s.text.split("\n");
      while (lines.length < 4) lines.push(" ");
      out.innerHTML =
        `W₁ = [[0.5, −0.5], [−1, 1]]\n` +
        `b₁ = (0, 0.5)  W₂ = (1, 0.5)  b₂ = 0\n` +
        `step ${current + 1} of ${steps.length}, ${phase}\n` +
        (s.reuse && s.back ? "reusing the underlined forward values" : " ") + "\n" +
        lines.join("\n");
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
        `ŷ at (0,0) ${preds[0]}   (0,1) ${preds[1]}\n` +
        `     (1,0) ${preds[2]}   (1,1) ${preds[3]}\n` +
        `hidden units: ${H}, dead: ${deadUnits()}\n` +
        `(dead: z ≤ 0 on all four inputs)`;
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

// Step-through: one neuron. Weighted sum, then ReLU, for three inputs.
(function () {
  const W = [0.5, -0.5];
  const O = { x: 500, y: 232 }, U = 90; // the ReLU graph: origin and pixels per unit
  Learn.demo(document.getElementById("demo-neuron"), {
    height: 320,
    label: "One neuron: two inputs with weights, a weighted sum z, and a ReLU graph showing the output h",
    steps: [
      { s: { x1: 1, x2: 0, sum: 0, relu: 0 }, cap: "A neuron with two inputs. Each input has its own weight: 0.5 for x₁ and −0.5 for x₂. The bias b is 0." },
      { s: { sum: 1 }, cap: "Feed in x = (1, 0). The weighted sum is 0.5 × 1 + (−0.5) × 0 + 0 = 0.5. That number is z." },
      { s: { relu: 1 }, cap: "ReLU keeps a positive number as it is. z = 0.5, so the output h is 0.5." },
      { s: { x1: 0, x2: 1 }, cap: "Now feed in x = (0, 1). The sum is 0.5 × 0 + (−0.5) × 1 + 0 = −0.5. ReLU turns a negative number into zero, so h = 0." },
      { s: { x1: 2, x2: 0 }, cap: "x = (2, 0) points further along w. Now z = 1 and h = 1. Once z is past zero, the output grows in a straight line." },
    ],
    draw(ctx, s, c, d) {
      const z = W[0] * s.x1 + W[1] * s.x2, h = Math.max(0, z);
      const box = (x, y, name, v, color) => {
        d.rect(x - 34, y - 24, 68, 48, c.bg, color, 2);
        d.text(name, x, y - 36, { c: c.muted, s: 13, a: "center", w: 700 });
        d.text(d.fmt(v, 2), x, y, { c: color, s: 18, a: "center", w: 700 });
      };
      d.arrow(98, 90, 206, 148, c.muted, 2);
      d.arrow(98, 220, 206, 172, c.muted, 2);
      d.text("× 0.5", 150, 100, { c: c.ink, s: 13, w: 700 });
      d.text("× (−0.5)", 150, 212, { c: c.ink, s: 13, w: 700 });
      box(60, 90, "x₁", s.x1, c.blue);
      box(60, 220, "x₂", s.x2, c.blue);
      d.rect(210, 132, 96, 56, c.bg, c.ink, 2);
      d.text("sum + b", 258, 118, { c: c.muted, s: 13, a: "center", w: 700 });
      d.text("z =", 222, 160, { c: c.muted, s: 14 });
      ctx.globalAlpha = s.sum;
      d.text(d.fmt(z, 2), 296, 160, { c: c.ink, s: 18, a: "right", w: 700 });
      d.text(`0.5×${d.fmt(s.x1, 1)} + (−0.5)×${d.fmt(s.x2, 1)} + 0`, 30, 290, { c: c.muted, s: 13 });
      ctx.globalAlpha = 1;
      d.arrow(308, 160, 372, 160, c.muted, 2);

      // ReLU graph.
      d.text("h = ReLU(z)", 500, 30, { c: c.muted, s: 13, a: "center", w: 700 });
      d.line(O.x - 1.2 * U, O.y, O.x + 1.4 * U, O.y, c.muted, 1.5);
      d.line(O.x, O.y + 16, O.x, O.y - 1.4 * U, c.muted, 1.5);
      for (const v of [-1, 1]) {
        d.line(O.x + v * U, O.y - 4, O.x + v * U, O.y + 4, c.muted, 1.5);
        d.text(d.fmt(v, 0), O.x + v * U, O.y + 18, { c: c.muted, s: 12, a: "center" });
      }
      d.line(O.x - 4, O.y - U, O.x + 4, O.y - U, c.muted, 1.5);
      d.text("1", O.x - 12, O.y - U, { c: c.muted, s: 12, a: "center" });
      d.text("z", O.x + 1.4 * U - 6, O.y + 18, { c: c.muted, s: 12, a: "center" });
      d.line(O.x - 1.2 * U, O.y, O.x, O.y, c.blue, 3.5);
      d.line(O.x, O.y, O.x + 1.3 * U, O.y - 1.3 * U, c.blue, 3.5);
      if (s.relu > 0.01) {
        ctx.globalAlpha = s.relu;
        const px = O.x + z * U, py = O.y - h * U;
        d.line(px, O.y, px, py, c.orange, 2, [4, 4]);
        d.dot(px, py, 8, c.orange);
        d.rect(404, 262, 192, 40, c.bg, c.orange, 2);
        d.text(`z = ${d.fmt(z, 2)}  →  h = ${d.fmt(h, 2)}`, 500, 282, { c: c.ink, s: 14, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: the hand-built XOR network drawn along the one axis both hidden units read.
(function () {
  const X = (v) => 120 + v * 190, Y = (v) => 240 - v * 85;
  const targets = [0, 1, 0], names = ["(0, 0)", "(0, 1) (1, 0)", "(1, 1)"];
  Learn.demo(document.getElementById("demo-xor"), {
    height: 340,
    label: "XOR targets plotted against x1 plus x2, with the two ReLU hidden units and the output that bends to fit them",
    steps: [
      { s: { lin: 0, h1: 0, h2: 0, out: 0 }, cap: "Place the four XOR inputs by their sum x₁ + x₂. The targets (squares) are 0 at sum 0, 1 at sum 1, and 0 at sum 2. They go up, then back down." },
      { s: { lin: 1 }, cap: "A linear function can't rise and then fall. The best it can do is a flat 0.5 everywhere, which is off by 0.5 at every input." },
      { s: { lin: 0, h1: 1 }, cap: "Hidden unit h₁ = ReLU(x₁ + x₂). It rises with the sum: 0, 1, 2." },
      { s: { h2: 1 }, cap: "Hidden unit h₂ = ReLU(x₁ + x₂ − 1). It stays at 0 until the sum passes 1, then rises: 0, 0, 1." },
      { s: { out: 1 }, cap: "The output is ŷ = h₁ − 2 × h₂. Up to sum 1 it follows h₁. After that h₂ pulls it down twice as fast: 0, 1, 0. The bend at 1 lets it hit every target." },
    ],
    draw(ctx, s, c, d) {
      for (const v of [0, 1, 2]) {
        d.line(X(0), Y(v), X(2), Y(v), v ? c.line : c.muted, v ? 1 : 1.5);
        d.text(String(v), X(0) - 16, Y(v), { c: c.muted, s: 12, a: "center" });
        d.line(X(v), Y(0) - 4, X(v), Y(0) + 4, c.muted, 1.5);
        d.text(String(v), X(v), Y(0) + 20, { c: c.muted, s: 12, a: "center" });
        d.text(names[v], X(v), Y(0) + 42, { c: c.ink, s: 12, a: "center" });
      }
      d.line(X(0), Y(0) + 6, X(0), Y(2) - 10, c.muted, 1.5);
      d.text("sum x₁ + x₂", X(1), 320, { c: c.muted, s: 12, a: "center" });
      d.text("inputs", 14, Y(0) + 42, { c: c.muted, s: 12 });
      const path = (pts, color, alpha, width, dash) => {
        if (alpha < 0.01) return;
        ctx.globalAlpha = alpha;
        for (let i = 0; i + 1 < pts.length; i++) d.line(X(pts[i][0]), Y(pts[i][1]), X(pts[i + 1][0]), Y(pts[i + 1][1]), color, width, dash);
        ctx.globalAlpha = 1;
      };
      path([[0, 0.5], [2, 0.5]], c.red, s.lin, 3, [8, 6]);
      path([[0, 0], [2, 2]], c.blue, s.h1 * (1 - 0.6 * s.out), 3);
      path([[0, 0], [1, 0], [2, 1]], c.purple, s.h2 * (1 - 0.6 * s.out), 3);
      path([[0, 0], [1, 1], [2, 0]], c.orange, s.out, 5);
      const tag = (str, x, y, color, alpha) => { if (alpha > 0.01) { ctx.globalAlpha = alpha; d.text(str, x, y, { c: color, s: 13, w: 700 }); ctx.globalAlpha = 1; } };
      tag("linear: 0.5", X(2) + 12, Y(0.5), c.red, s.lin);
      tag("h₁", X(2) + 12, Y(2), c.blue, s.h1);
      tag("h₂", X(2) + 12, Y(1), c.purple, s.h2);
      tag("ŷ = h₁ − 2h₂", X(2) + 12, Y(0) - 14, c.orange, s.out);
      for (const v of [0, 1, 2]) d.rect(X(v) - 9, Y(targets[v]) - 9, 18, 18, c.bg, c.ink, 3);
      d.rect(16, 16, 16, 16, c.bg, c.ink, 3);
      d.text("target", 42, 24, { c: c.ink, s: 13, w: 700 });
    },
  });
})();

// Step-through: a finite-difference check of one backprop gradient, then one training step.
(function () {
  // The backprop network with only W1[0][0] = w free: z1 = w, h1 = ReLU(w), h2 = 0, so y-hat = h1.
  const L = (w) => (Math.max(0, w) - 1) ** 2;
  const X = (w) => 70 + (w + 0.2) * 300, Y = (v) => 286 - v * 190;
  const w0 = 0.5, eps = 0.1, eta = 0.1, grad = 2 * (w0 - 1);
  const slope = (L(w0 + eps) - L(w0 - eps)) / (2 * eps);
  Learn.demo(document.getElementById("demo-gradcheck"), {
    height: 340,
    label: "The loss as one weight changes, with two nudged points and the line through them whose slope is the gradient",
    steps: [
      { s: { up: 0, dn: 0, sec: 0, step: 0 }, cap: "The loss of the backprop network as its first weight changes and everything else stays fixed. The weight is 0.5 and the loss is 0.25. Backprop says the gradient here is −1." },
      { s: { up: 1 }, cap: "Nudge the weight up by ε = 0.1, to 0.6, and run a forward pass. The loss is 0.16." },
      { s: { dn: 1 }, cap: "Nudge it down by 0.1, to 0.4, and run another forward pass. The loss is 0.36." },
      { s: { sec: 1 }, cap: "Divide the change in loss by the distance between the nudges: (0.16 − 0.36) ÷ 0.2 = −1. That matches backprop. Real checks use a far smaller ε, like 0.000001." },
      { s: { up: 0, dn: 0, sec: 0, step: 1 }, cap: "Training uses that gradient. With η = 0.1 the update is 0.5 − 0.1 × (−1) = 0.6. The weight moves right and the loss drops from 0.25 to 0.16." },
    ],
    draw(ctx, s, c, d) {
      d.line(X(-0.2), Y(0), X(1.4), Y(0), c.muted, 1.5);
      d.line(X(0), Y(0) + 6, X(0), Y(1.1), c.muted, 1.5);
      for (const v of [0, 0.5, 1]) {
        d.line(X(v), Y(0) - 4, X(v), Y(0) + 4, c.muted, 1.5);
        d.text(String(v), X(v), Y(0) + 18, { c: c.muted, s: 12, a: "center" });
      }
      for (const v of [0.5, 1]) {
        d.line(X(0) - 4, Y(v), X(0) + 4, Y(v), c.muted, 1.5);
        d.text(String(v), X(0) - 10, Y(v), { c: c.muted, s: 12, a: "right" });
      }
      d.text("first weight of W₁", X(1.4), Y(0) + 40, { c: c.muted, s: 12, a: "right" });
      d.text("loss L", X(0) + 10, Y(1.1) + 2, { c: c.muted, s: 12 });
      ctx.strokeStyle = c.blue; ctx.lineWidth = 3.5; ctx.setLineDash([]);
      ctx.beginPath();
      for (let i = 0; i <= 160; i++) { const w = -0.2 + i * 0.01; if (i) ctx.lineTo(X(w), Y(L(w))); else ctx.moveTo(X(w), Y(L(w))); }
      ctx.stroke();
      if (s.sec > 0.01) {
        ctx.globalAlpha = s.sec;
        const a = w0 - 0.4, b = w0 + 0.25, mid = (L(w0 + eps) + L(w0 - eps)) / 2;
        d.line(X(a), Y(mid + slope * (a - w0)), X(b), Y(mid + slope * (b - w0)), c.orange, 2, [7, 5]);
        ctx.globalAlpha = 1;
      }
      const nudge = (w, alpha, side) => {
        if (alpha < 0.01) return;
        ctx.globalAlpha = alpha;
        d.line(X(w), Y(0), X(w), Y(L(w)), c.orange, 1.5, [3, 4]);
        d.dot(X(w), Y(L(w)), 6, c.orange);
        d.text(d.fmt(L(w), 2), X(w) + side * 12, Y(L(w)) - side * 13, { c: c.orange, s: 13, a: side > 0 ? "left" : "right", w: 700 });
        ctx.globalAlpha = 1;
      };
      nudge(w0 + eps, s.up, 1);
      nudge(w0 - eps, s.dn, -1);
      const w = w0 - eta * grad * s.step;
      d.line(X(w), Y(0), X(w), Y(L(w)), c.ink, 1.5, [3, 4]);
      d.dot(X(w), Y(L(w)), 8, c.ink);
      d.panel(c, 356, 20, 270, 122);
      d.text(`weight ${d.fmt(w, 2)}   loss ${d.fmt(L(w), 2)}`, 370, 42, { c: c.ink, s: 13, w: 700 });
      d.text(`backprop gradient: ${d.fmt(grad, 2)}`, 370, 68, { c: c.muted, s: 13 });
      ctx.globalAlpha = s.sec;
      d.text("finite difference:", 370, 96, { c: c.orange, s: 13, w: 700 });
      d.text(`(0.16 − 0.36) ÷ 0.2 = ${d.fmt(slope, 2)}`, 370, 120, { c: c.orange, s: 13, w: 700 });
      ctx.globalAlpha = s.step;
      d.text("update: w ← w − η × gradient", 370, 96, { c: c.ink, s: 13 });
      d.text(`0.5 − 0.1 × (−1) = ${d.fmt(w0 - eta * grad, 2)}`, 370, 120, { c: c.ink, s: 13, w: 700 });
      ctx.globalAlpha = 1;
    },
  });
})();
