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
