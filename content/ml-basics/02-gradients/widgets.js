// Unit 02: a secant line closing in on the tangent of x², and the gradient of x² + 3y² on its contour map.
(function () {
  const figure = document.getElementById("tangent");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  const width = 720, rows = 14, height = rows * Learn.ROW;
  const X = (x) => width / 2 + x * 80;      // x from -4.5 to 4.5
  const Y = (v) => height - 28 - v * 22;    // f from about -1 to 10.9
  const f = (x) => x * x;

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "The curve f(x) = x squared with a tangent line and a secant line at x");
  const clip = svg("clipPath", { id: "tangent-clip" }, svg("defs", {}, root));
  svg("rect", { x: 0, y: 0, width, height }, clip);
  svg("line", { x1: 0, y1: Y(0), x2: width, y2: Y(0), class: "muted faint thin" }, root);
  svg("line", { x1: X(0), y1: 0, x2: X(0), y2: height, class: "muted faint thin" }, root);
  svg("text", { x: 12, y: Y(0) - 10, class: "muted", text: "f(x) = x²" }, root);

  const plot = svg("g", { "clip-path": "url(#tangent-clip)" }, root);
  const curve = [];
  for (let x = -4.5; x <= 4.5001; x += 0.05) curve.push(`${X(x).toFixed(1)},${Y(f(x)).toFixed(1)}`);
  svg("polyline", { points: curve.join(" "), class: "ink" }, plot);
  const run = svg("line", { class: "muted dashed thin" }, plot);
  const rise = svg("line", { class: "muted dashed thin" }, plot);
  const secant = svg("line", { class: "orange" }, plot);
  const tangent = svg("line", { class: "blue" }, plot);
  const q = svg("circle", { r: 4.5, class: "orange", fill: "currentColor" }, plot);
  const p = svg("circle", { r: 4.5, class: "blue", fill: "currentColor" }, plot);

  // Draw the straight line through (x0, y0) with the given slope across the whole stage.
  function lineThrough(node, x0, y0, slope) {
    node.setAttribute("x1", X(-5)); node.setAttribute("y1", Y(y0 + slope * (-5 - x0)));
    node.setAttribute("x2", X(5)); node.setAttribute("y2", Y(y0 + slope * (5 - x0)));
  }

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  let x = 1.5, h = 0.5;
  let phase = Math.asin(x / 2.5);

  function draw() {
    const fx = f(x), fq = f(x + h);
    const sec = (fq - fx) / h, tan = 2 * x;
    lineThrough(secant, x, fx, sec);
    lineThrough(tangent, x, fx, tan);
    run.setAttribute("x1", X(x)); run.setAttribute("y1", Y(fx));
    run.setAttribute("x2", X(x + h)); run.setAttribute("y2", Y(fx));
    rise.setAttribute("x1", X(x + h)); rise.setAttribute("y1", Y(fx));
    rise.setAttribute("x2", X(x + h)); rise.setAttribute("y2", Y(fq));
    p.setAttribute("cx", X(x)); p.setAttribute("cy", Y(fx));
    q.setAttribute("cx", X(x + h)); q.setAttribute("cy", Y(fq));

    // Lines stay under about 38 characters so they don't wrap on a phone.
    out.innerHTML =
      `x = ${fmt(x)}   h = ${fmt(h)}\n` +
      `f(x) = ${fmt(fx, 4)}   f(x + h) = ${fmt(fq, 4)}\n` +
      `<span class="orange">secant slope = ${fmt(fq - fx, 4)} / ${fmt(h)} = ${fmt(sec)}</span>\n` +
      `<span class="blue">tangent slope = 2x = ${fmt(tan)}</span>\n` +
      `difference = ${fmt(sec - tan)}   (always h for x²)`;
  }

  const xControl = slider(ctl, { label: "x", min: -3, max: 3, step: 0.01, value: x, format: (v) => v.toFixed(2) }, (v) => {
    x = v;
    phase = Math.asin(Math.max(-1, Math.min(1, x / 2.5)));
    draw();
  });
  slider(ctl, { label: "h", min: 0.01, max: 1.5, step: 0.01, value: h, format: (v) => v.toFixed(2) }, (v) => {
    h = v;
    draw();
  });

  const player = loop(figure, (dt) => {
    if (dt) {
      phase += 0.5 * dt;
      x = 2.5 * Math.sin(phase);
      xControl.show(x.toFixed(2));
    }
    draw();
  });
  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
  ctl.addEventListener("input", () => {
    player.pause();
    toggle.textContent = "Play";
  });
})();

(function () {
  const figure = document.getElementById("gradient");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, arrow, loop, fmt } = Learn;

  const width = 720, rows = 18, height = rows * Learn.ROW;
  const S = 48; // SVG units per unit of x or y
  const ox = width / 2, oy = height / 2;
  const X = (x) => ox + x * S;
  const Y = (y) => oy - y * S;
  const f = (x, y) => x * x + 3 * y * y;
  const grad = (x, y) => [2 * x, 6 * y];
  const ARROW_SCALE = 0.2; // the gradient arrow is drawn at a fifth of its length
  // Keep the point where the gradient arrow and its label stay inside the stage.
  const LIMIT_X = 5, LIMIT_Y = 1.4;
  const PATH_X = 2.8, PATH_Y = 1.4; // the animation follows (PATH_X cos t, PATH_Y sin t)

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Contour lines of f(x, y) = x squared plus 3 y squared, with the gradient arrow at a point");
  svg("line", { x1: 0, y1: oy, x2: width, y2: oy, class: "muted faint thin" }, root);
  svg("line", { x1: ox, y1: 0, x2: ox, y2: height, class: "muted faint thin" }, root);
  for (const c of [1, 2, 4, 7, 11, 16, 22]) {
    svg("ellipse", { cx: ox, cy: oy, rx: Math.sqrt(c) * S, ry: Math.sqrt(c / 3) * S, class: "muted thin", fill: "none", stroke: "currentColor" }, root);
  }
  const contour = svg("ellipse", { cx: ox, cy: oy, class: "orange", fill: "none", stroke: "currentColor", "stroke-width": 2 }, root);
  const gradArrow = arrow(root, "blue");
  const step = arrow(root, "green"); // drawn after the gradient so it stays visible when they line up
  const gradLabel = svg("text", { class: "blue", text: "∇f" }, root);
  const stepLabel = svg("text", { class: "green", text: "s" }, root);
  const dot = svg("circle", { r: 5, class: "ink", fill: "currentColor" }, root);
  // A larger invisible target so the point is easy to grab, also focusable for arrow keys.
  const handle = svg("circle", { r: 18, fill: "transparent", tabindex: 0, "aria-label": "Point; drag it or use the arrow keys" }, root);
  handle.style.cursor = "grab";
  handle.style.touchAction = "none";

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  let px = 1, py = 1, angle = 45; // the point and the direction of the step s, in degrees
  let t = Math.atan2(py / PATH_Y, px / PATH_X); // animation parameter along the path

  function draw() {
    const [gx, gy] = grad(px, py);
    const value = f(px, py);
    const g = Math.hypot(gx, gy);
    const rad = angle * Math.PI / 180;
    const s = [Math.cos(rad), Math.sin(rad)];
    const rate = gx * s[0] + gy * s[1];

    contour.setAttribute("rx", Math.sqrt(value) * S);
    contour.setAttribute("ry", Math.sqrt(value / 3) * S);
    step.update(X(px), Y(py), X(px + s[0]), Y(py + s[1]));
    // Label s beside the middle of its shaft, on the side away from ∇f, so the two never overlap.
    const side = -s[1] * gx + s[0] * gy > 0 ? -1 : 1;
    stepLabel.setAttribute("x", X(px + 0.6 * s[0] - side * 0.4 * s[1]) - 4);
    stepLabel.setAttribute("y", Y(py + 0.6 * s[1] + side * 0.4 * s[0]) + 5);
    const tip = [px + ARROW_SCALE * gx, py + ARROW_SCALE * gy];
    gradArrow.update(X(px), Y(py), X(tip[0]), Y(tip[1]));
    const away = g ? [gx / g, gy / g] : [0, 0];
    gradLabel.setAttribute("x", X(tip[0] + 0.45 * away[0]) - 8);
    gradLabel.setAttribute("y", Y(tip[1] + 0.45 * away[1]) + 5);
    gradLabel.style.display = g < 0.5 ? "none" : "";
    for (const node of [dot, handle]) {
      node.setAttribute("cx", X(px));
      node.setAttribute("cy", Y(py));
    }

    let last;
    if (g < 1e-9) {
      last = `∇f = 0: flat in every direction`;
    } else {
      const between = Math.acos(Math.max(-1, Math.min(1, rate / g))) * 180 / Math.PI;
      last = `∇f · s = ${fmt(rate)}   (s is ${fmt(between, 0)}° from ∇f)`;
    }
    // Lines stay under about 38 characters so they don't wrap on a phone.
    out.innerHTML =
      `(x, y) = (${fmt(px)}, ${fmt(py)})   <span class="orange">f = ${fmt(value)}</span>\n` +
      `<span class="blue">∇f = (${fmt(gx)}, ${fmt(gy)})   |∇f| = ${fmt(g)}</span>\n` +
      `<span class="green">s = (${fmt(s[0])}, ${fmt(s[1])})</span>, a step of length 1\n` +
      last;
  }

  const angleControl = slider(ctl, { label: "direction", min: 0, max: 359, step: 1, value: angle, format: (v) => `${Math.round(v)}°` }, (v) => {
    angle = v;
    draw();
  });

  const player = loop(figure, (dt) => {
    if (dt) {
      t += 0.4 * dt;
      px = PATH_X * Math.cos(t);
      py = PATH_Y * Math.sin(t);
    }
    draw();
  });
  player.pause(); // start still, at (1, 1), so the readout matches the text
  const toggle = button(btns, "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
  button(btns, "Point s along ∇f", () => {
    const [gx, gy] = grad(px, py);
    if (gx === 0 && gy === 0) return;
    angleControl.set(Math.round(((Math.atan2(gy, gx) * 180 / Math.PI) + 360) % 360));
  });

  function stop() {
    player.pause();
    toggle.textContent = "Play";
  }
  function moveTo(x, y) {
    px = Math.max(-LIMIT_X, Math.min(LIMIT_X, x));
    py = Math.max(-LIMIT_Y, Math.min(LIMIT_Y, y));
    // Resume the animation at the path angle that matches the new point.
    t = Math.atan2(py / PATH_Y, px / PATH_X);
    draw();
  }
  function toData(event) {
    const pt = root.createSVGPoint();
    pt.x = event.clientX; pt.y = event.clientY;
    const local = pt.matrixTransform(root.getScreenCTM().inverse());
    return [(local.x - ox) / S, (oy - local.y) / S];
  }

  let dragging = false;
  handle.addEventListener("pointerdown", (event) => {
    dragging = true;
    handle.setPointerCapture(event.pointerId);
    handle.style.cursor = "grabbing";
    stop();
    event.preventDefault();
  });
  handle.addEventListener("pointermove", (event) => {
    if (dragging) moveTo(...toData(event));
  });
  const release = () => {
    dragging = false;
    handle.style.cursor = "grab";
  };
  handle.addEventListener("pointerup", release);
  handle.addEventListener("pointercancel", release);
  // Clicking anywhere else on the map moves the point there.
  root.addEventListener("click", (event) => {
    if (event.target === handle) return;
    stop();
    moveTo(...toData(event));
  });
  handle.addEventListener("keydown", (event) => {
    const moves = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] };
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    stop();
    moveTo(px + move[0], py + move[1]);
  });
  ctl.addEventListener("input", stop);
})();

// Step-through: a secant line on x² turning into the tangent as the gap h shrinks.
(function () {
  const X = (x) => 60 + x * 100, Y = (y) => 320 - y * 10;
  Learn.demo(document.getElementById("demo-secant"), {
    height: 350,
    label: "A secant line on the curve x squared turning into the tangent line at x = 3 as the gap shrinks",
    steps: [
      { s: { lh: Math.log10(2), tri: 1, tan: 0 }, cap: "The curve is f(x) = x². The blue point sits at x = 3. The orange point sits a gap h = 2 to the right. The line through both is a secant. Its slope is rise ÷ run = 16 ÷ 2 = 8." },
      { s: { lh: 0 }, cap: "Shrink the gap to h = 1. The slope drops to 7." },
      { s: { lh: Math.log10(0.5) }, cap: "Shrink it to h = 0.5. The slope is now 6.5." },
      { s: { lh: -1 }, cap: "At h = 0.1 the slope is 6.1. The slope keeps closing in on one number." },
      { s: { lh: -2, tan: 1, tri: 0 }, cap: "At h = 0.01 the two points nearly touch. The secant has become the tangent line (dashed). Its slope, 6, is the derivative of x² at x = 3." },
    ],
    draw(ctx, s, c, d) {
      const h = Math.pow(10, s.lh), m = 6 + h;
      d.line(X(0), Y(0), X(5.6), Y(0), c.muted, 1.5);
      d.line(X(0), Y(0), X(0), Y(30), c.muted, 1.5);
      for (let i = 1; i <= 5; i++) { d.line(X(i), Y(0), X(i), Y(0) + 5, c.muted); d.text(String(i), X(i), Y(0) + 16, { c: c.muted, s: 12, a: "center" }); }
      for (let j = 10; j <= 30; j += 10) { d.line(X(0) - 5, Y(j), X(0), Y(j), c.muted); d.text(String(j), X(0) - 9, Y(j), { c: c.muted, s: 12, a: "right" }); }
      d.text("x", X(5.6) + 8, Y(0), { c: c.muted, s: 13 });
      ctx.strokeStyle = c.ink; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let x = 0; x <= 5.45; x += 0.05) { if (x === 0) ctx.moveTo(X(x), Y(x * x)); else ctx.lineTo(X(x), Y(x * x)); }
      ctx.stroke();
      d.text("f(x) = x²", X(4.45), Y(25), { c: c.ink, w: 700, a: "right" });
      ctx.save(); ctx.beginPath(); ctx.rect(X(0), Y(30), X(5.6) - X(0), Y(0) - Y(30)); ctx.clip();
      if (s.tan > 0.01) { ctx.globalAlpha = s.tan; d.line(X(1), Y(9 - 12), X(5.5), Y(9 + 6 * 2.5), c.blue, 2, [7, 6]); ctx.globalAlpha = 1; }
      d.line(X(0.8), Y(9 + m * (0.8 - 3)), X(5.6), Y(9 + m * 2.6), c.orange, 2.5);
      ctx.restore();
      const qx = 3 + h, qy = qx * qx;
      if (s.tri > 0.01) {
        ctx.globalAlpha = s.tri;
        d.line(X(3), Y(9), X(qx), Y(9), c.muted, 1.5, [4, 4]);
        d.line(X(qx), Y(9), X(qx), Y(qy), c.muted, 1.5, [4, 4]);
        if (h > 0.3) {
          d.text("run = h", (X(3) + X(qx)) / 2, Y(9) + 14, { c: c.muted, s: 12, a: "center" });
          d.text("rise", X(qx) + 8, (Y(9) + Y(qy)) / 2, { c: c.muted, s: 12 });
        }
        ctx.globalAlpha = 1;
      }
      d.dot(X(3), Y(9), 6, c.blue);
      d.dot(X(qx), Y(qy), 6, c.orange);
      d.panel(c, 80, 24, 210, 82);
      d.text(`gap h = ${h >= 0.1 ? d.fmt(h, 2) : d.fmt(h, 3)}`, 96, 48, { c: c.ink });
      d.text(`slope = ${m.toFixed(2)}`, 96, 80, { c: c.orange, s: 19, w: 700 });
      if (s.tan > 0.01) { ctx.globalAlpha = s.tan; d.text("tangent slope = 6", 300, 150, { c: c.blue, w: 700, a: "right" }); ctx.globalAlpha = 1; }
    },
  });
})();

// Step-through: the chain rule on L = (2w − 3)², values forward and rates backward.
(function () {
  const nodes = [{ x: 90, name: "w", value: "1", role: "knob" }, { x: 320, name: "u", value: "−1", role: "middle step" }, { x: 550, name: "L", value: "1", role: "loss" }];
  const Yn = 130;
  Learn.demo(document.getElementById("demo-chain"), {
    height: 330,
    label: "The chain rule: w flows to u flows to L, then the two rates multiply going backward",
    steps: [
      { s: { v: 0, fw: 0, r1: 0, r2: 0, prod: 0, bw: 0, chk: 0 }, cap: "The loss L depends on the knob w, but not directly. First, w turns into a middle value u = 2w − 3. Then u turns into the loss L = u²." },
      { s: { v: 1, fw: 1 }, cap: "Push a value forward. Start at w = 1. Then u = 2(1) − 3 = −1. Then L = (−1)² = 1." },
      { s: { r1: 1 }, cap: "Find the rate of link 1. When w moves by 1, u = 2w − 3 moves by 2. So du/dw = 2." },
      { s: { r2: 1 }, cap: "Find the rate of link 2. For L = u², the power rule gives dL/du = 2u = 2 × (−1) = −2. If u grows a little, L shrinks twice as much." },
      { s: { prod: 1, bw: 1 }, cap: "Now go backward and multiply the rates: dL/dw = (−2) × 2 = −4. That is the chain rule." },
      { s: { chk: 1 }, cap: "Check it. Nudge w from 1 to 1.01. L drops from 1 to 0.9604, a change of about −0.04 = −4 × 0.01. The negative slope says: raise w to lower the loss." },
    ],
    draw(ctx, s, c, d) {
      d.arrow(nodes[0].x + 42, Yn, nodes[1].x - 44, Yn, c.muted, 2);
      d.arrow(nodes[1].x + 42, Yn, nodes[2].x - 44, Yn, c.muted, 2);
      d.text("u = 2w − 3", (nodes[0].x + nodes[1].x) / 2, Yn - 22, { c: c.ink, s: 15, a: "center" });
      d.text("L = u²", (nodes[1].x + nodes[2].x) / 2, Yn - 22, { c: c.ink, s: 15, a: "center" });
      nodes.forEach((node, k) => {
        d.rect(node.x - 38, Yn - 38, 76, 76, c.bg, k === 2 ? c.red : c.blue, 2.5);
        d.text(node.name, node.x, Yn - (s.v > 0.01 ? 10 : 0), { c: c.ink, s: 22, w: 700, a: "center" });
        if (s.v > 0.01) { ctx.globalAlpha = s.v; d.text(`= ${node.value}`, node.x, Yn + 14, { c: c.ink, s: 15, a: "center" }); ctx.globalAlpha = 1; }
        d.text(node.role, node.x, Yn + 54, { c: c.muted, s: 12, a: "center" });
      });
      const span = nodes[2].x - nodes[0].x;
      if (s.fw > 0.01 && s.fw < 0.99) d.dot(nodes[0].x + span * s.fw, Yn, 8, c.blue);
      if (s.bw > 0.01 && s.bw < 0.99) d.dot(nodes[2].x - span * s.bw, Yn, 8, c.orange);
      const show = (alpha, fn) => { if (alpha > 0.01) { ctx.globalAlpha = alpha; fn(); ctx.globalAlpha = 1; } };
      show(s.r1, () => d.text("du/dw = 2", (nodes[0].x + nodes[1].x) / 2, 218, { c: c.orange, s: 16, a: "center", w: 700 }));
      show(s.r2, () => d.text("dL/du = 2u = −2", (nodes[1].x + nodes[2].x) / 2, 218, { c: c.orange, s: 16, a: "center", w: 700 }));
      show(s.prod, () => { d.panel(c, 150, 244, 340, 40); d.text("dL/dw = (−2) × 2 = −4", 320, 264, { c: c.ink, s: 18, a: "center", w: 700 }); });
      show(s.chk, () => d.text("Check: w 1 → 1.01 moves L 1 → 0.9604 (about −0.04)", 320, 308, { c: c.muted, s: 13, a: "center" }));
    },
  });
})();

// Step-through: the gradient on a contour map of x² + 3y², then a few steps downhill.
(function () {
  const O = { x: 330, y: 180 }, S = 100;
  const X = (x) => O.x + x * S, Y = (y) => O.y - y * S;
  const path = [[1, 1]];
  for (let k = 0; k < 6; k++) { const q = path[path.length - 1]; path.push([q[0] - 0.1 * 2 * q[0], q[1] - 0.1 * 6 * q[1]]); }
  const at = (kf) => {
    const a = Math.floor(kf), b = Math.min(path.length - 1, a + 1), t = kf - a;
    return [path[a][0] + (path[b][0] - path[a][0]) * t, path[a][1] + (path[b][1] - path[a][1]) * t];
  };
  Learn.demo(document.getElementById("demo-gradient"), {
    height: 360,
    label: "A contour map of a bowl, the gradient arrow at the point (1, 1), and steps downhill toward the center",
    steps: [
      { s: { k: 0, pt: 0, ga: 0, da: 0 }, cap: "This is a top-down map of the bowl f(x, y) = x² + 3y². Each ring is a contour: every point on one ring has the same height. The bottom of the bowl is the center, (0, 0)." },
      { s: { pt: 1 }, cap: "Stand at the point (1, 1). Its height is 1² + 3 × 1² = 4." },
      { s: { ga: 1 }, cap: "The gradient here is (2x, 6y) = (2, 6). The red arrow shows it. It points uphill. It leans toward y because the bowl is steeper in the y direction." },
      { s: { da: 1 }, cap: "To go downhill, step the opposite way. Use a small step: new point = (1, 1) − 0.1 × (2, 6) = (0.8, 0.4). The blue arrow is that step." },
      { s: { k: 1 }, cap: "Take the step. The height drops from 4 to 1.12. At the new point the gradient is different, so work it out again." },
      { s: { k: 2 }, cap: "Repeat from the new point with its own gradient. The height drops to about 0.49." },
      { s: { k: 5 }, cap: "Three more steps. The point is close to the bottom, with a height of about 0.11. This loop is called gradient descent. Unit 4 covers it." },
    ],
    draw(ctx, s, c, d) {
      d.line(20, O.y, 620, O.y, c.line);
      d.line(O.x, 15, O.x, 345, c.line);
      for (const level of [0.25, 1, 2, 4, 6]) {
        ctx.beginPath(); ctx.ellipse(O.x, O.y, Math.sqrt(level) * S, Math.sqrt(level / 3) * S, 0, 0, 7);
        ctx.strokeStyle = c.muted; ctx.globalAlpha = 0.55; ctx.lineWidth = 1.3; ctx.stroke(); ctx.globalAlpha = 1;
        if (level === 1 || level === 4 || level === 6) d.text(`height ${level}`, O.x, O.y - Math.sqrt(level / 3) * S - 9, { c: c.muted, s: 12, a: "center" });
      }
      d.dot(O.x, O.y, 4, c.ink);
      d.text("bottom (0, 0)", O.x + 8, O.y + 14, { c: c.ink, s: 12 });
      if (s.pt < 0.01) return;
      ctx.globalAlpha = s.pt;
      const p = at(s.k), done = Math.floor(s.k);
      ctx.strokeStyle = c.blue; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(X(path[0][0]), Y(path[0][1]));
      for (let i = 1; i <= done; i++) ctx.lineTo(X(path[i][0]), Y(path[i][1]));
      ctx.lineTo(X(p[0]), Y(p[1])); ctx.stroke(); ctx.setLineDash([]);
      ctx.globalAlpha = s.pt * 0.5;
      for (let j = 0; j <= done; j++) d.dot(X(path[j][0]), Y(path[j][1]), 3.5, c.blue);
      const gx = 2 * p[0], gy = 6 * p[1];
      if (s.ga > 0.01) {
        ctx.globalAlpha = s.ga * s.pt;
        d.arrow(X(p[0]), Y(p[1]), X(p[0] + 0.07 * gx), Y(p[1] + 0.07 * gy), c.red, 3);
        if (s.k < 0.5) d.text("gradient (uphill)", X(p[0] + 0.07 * gx) + 10, Y(p[1] + 0.07 * gy) + 4, { c: c.red, s: 13, w: 700 });
      }
      if (s.da > 0.01) {
        ctx.globalAlpha = s.da * s.pt;
        d.arrow(X(p[0]), Y(p[1]), X(p[0] - 0.1 * gx), Y(p[1] - 0.1 * gy), c.blue, 3);
        if (s.k < 0.5) d.text("step (downhill)", X(p[0] - 0.1 * gx) - 12, Y(p[1] - 0.1 * gy) + 16, { c: c.blue, s: 13, w: 700, a: "right" });
      }
      ctx.globalAlpha = s.pt;
      d.dot(X(p[0]), Y(p[1]), 7, c.orange);
      d.panel(c, 14, 14, 218, 92);
      d.text(`point (${d.fmt(p[0], 2)}, ${d.fmt(p[1], 2)})`, 26, 36, { c: c.ink, s: 13 });
      d.text(`height ${d.fmt(p[0] * p[0] + 3 * p[1] * p[1], 2)}`, 26, 60, { c: c.ink, s: 13 });
      d.text(`gradient (${d.fmt(gx, 2)}, ${d.fmt(gy, 2)})`, 26, 84, { c: c.red, s: 13 });
    },
  });
})();
