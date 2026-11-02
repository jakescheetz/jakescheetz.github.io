// Unit 03: gradient descent on the line-fit loss. Left: the data and the current line.
// Right: contours of the loss over (w, b) and the path gradient descent takes.
(function () {
  const figure = document.getElementById("descent");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  // The four points from the text, and the loss L(w, b) = mean((w x + b - y)^2).
  const xs = [0, 1, 2, 3], ys = [1, 3, 4, 7];
  const N = xs.length;
  const loss = (w, b) => xs.reduce((s, x, i) => s + (w * x + b - ys[i]) ** 2, 0) / N;
  const grad = (w, b) => {
    let gw = 0, gb = 0;
    xs.forEach((x, i) => { const e = w * x + b - ys[i]; gw += e * x; gb += e; });
    return [2 * gw / N, 2 * gb / N];
  };

  // Exact minimum and the bowl's shape. The second derivatives of L are constant:
  // H = (2/N) [[sum x^2, sum x], [sum x, N]]. Its eigenvalues are the curvatures along the
  // ellipse axes (steep and flat), its eigenvectors the axis directions.
  const sx = xs.reduce((s, x) => s + x, 0), sxx = xs.reduce((s, x) => s + x * x, 0);
  const sy = ys.reduce((s, y) => s + y, 0), sxy = xs.reduce((s, x, i) => s + x * ys[i], 0);
  const wStar = (N * sxy - sx * sy) / (N * sxx - sx * sx), bStar = (sy - wStar * sx) / N;
  const lossStar = loss(wStar, bStar);
  const h11 = 2 * sxx / N, h12 = 2 * sx / N, h22 = 2;
  const mid = (h11 + h22) / 2, gap = Math.sqrt(((h11 - h22) / 2) ** 2 + h12 * h12);
  const steep = mid + gap, flat = mid - gap;
  const flatDir = [h12, flat - h11]; // eigenvector for the flat curvature (unnormalized)

  const width = 720, rows = 18, height = rows * Learn.ROW;
  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Left: four data points and the current line. Right: loss contours over w and b with the gradient descent path.");
  const defs = svg("defs", {}, root);

  // Left panel: data space.
  const left = { x0: 0, x1: 230 };
  const dx = (x) => 30 + x * 56;           // x in [-0.5, 3.5]
  const dy = (y) => height - 26 - y * 30;  // y in [-0.8, 9]
  const clipData = svg("clipPath", { id: "descent-clip-data" }, defs);
  svg("rect", { x: left.x0, y: 0, width: left.x1 - left.x0, height }, clipData);
  const dataG = svg("g", { "clip-path": "url(#descent-clip-data)" }, root);
  svg("line", { x1: dx(-0.5), y1: dy(0), x2: dx(3.6), y2: dy(0), class: "muted faint thin" }, dataG);
  svg("line", { x1: dx(0), y1: dy(-0.8), x2: dx(0), y2: dy(9.5), class: "muted faint thin" }, dataG);
  svg("text", { x: dx(3.6) - 12, y: dy(0) + 18, class: "muted", text: "x" }, root);
  svg("text", { x: dx(0) + 8, y: 16, class: "muted", text: "y" }, root);
  const residuals = xs.map(() => svg("line", { class: "muted dashed thin" }, dataG));
  const fitLine = svg("line", { class: "orange" }, dataG);
  xs.forEach((x, i) => svg("circle", { cx: dx(x), cy: dy(ys[i]), r: 4, fill: "currentColor", class: "ink" }, dataG));

  // Right panel: parameter space (w across, b up), equal scale on both axes.
  const panel = { x0: 250, x1: width };
  const scale = 64, cw = 1.6, cb = 1.8;
  const PX = (w) => (panel.x0 + panel.x1) / 2 + (w - cw) * scale;
  const PY = (b) => height / 2 - (b - cb) * scale;
  const clipBowl = svg("clipPath", { id: "descent-clip-bowl" }, defs);
  svg("rect", { x: panel.x0, y: 0, width: panel.x1 - panel.x0, height }, clipBowl);
  const bowl = svg("g", { "clip-path": "url(#descent-clip-bowl)" }, root);
  svg("line", { x1: panel.x0, y1: PY(0), x2: panel.x1, y2: PY(0), class: "muted faint thin" }, bowl);
  svg("line", { x1: PX(0), y1: 0, x2: PX(0), y2: height, class: "muted faint thin" }, bowl);
  svg("line", { x1: panel.x0, y1: 0, x2: panel.x0, y2: height, class: "muted faint thin" }, root);
  svg("text", { x: panel.x1 - 14, y: PY(0) + 18, class: "muted", text: "w" }, root);
  svg("text", { x: PX(0) + 8, y: 16, class: "muted", text: "b" }, root);

  // Contours: L - L* = c is an ellipse with radius sqrt(2c / curvature) along each axis.
  const tilt = Math.atan2(-flatDir[1], flatDir[0]) * 180 / Math.PI; // screen y points down
  const contours = svg("g", { class: "muted" }, bowl);
  for (const c of [0.05, 0.2, 0.8, 3.2, 12.8]) {
    svg("ellipse", {
      cx: PX(wStar), cy: PY(bStar),
      rx: Math.sqrt(2 * c / flat) * scale, ry: Math.sqrt(2 * c / steep) * scale,
      transform: `rotate(${tilt} ${PX(wStar)} ${PY(bStar)})`,
      fill: "none", stroke: "currentColor", "stroke-width": 1,
    }, contours);
  }
  const best = svg("g", { class: "green" }, bowl);
  svg("line", { x1: PX(wStar) - 6, y1: PY(bStar), x2: PX(wStar) + 6, y2: PY(bStar) }, best);
  svg("line", { x1: PX(wStar), y1: PY(bStar) - 6, x2: PX(wStar), y2: PY(bStar) + 6 }, best);
  const trail = svg("polyline", { class: "blue", "stroke-width": 1.5, "stroke-linejoin": "round" }, bowl);
  const dots = svg("g", { class: "blue" }, bowl);
  const current = svg("circle", { r: 6, fill: "currentColor", class: "orange" }, bowl);

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  const MAX_STEPS = 100, INTERVAL = 0.12, HOLD = 2.5;
  let eta = 0.1;
  let start = [-1, 3];
  let w, b, k, points, status, wait;

  function reset() {
    [w, b] = start;
    k = 0;
    points = [[w, b]];
    status = "";
    wait = 0;
    while (dots.firstChild) dots.removeChild(dots.firstChild);
    dot(w, b);
    draw();
  }

  function dot(w, b) {
    svg("circle", { cx: PX(w), cy: PY(b), r: 2.5, fill: "currentColor" }, dots);
  }

  function step() {
    if (status) return;
    const [gw, gb] = grad(w, b);
    w -= eta * gw;
    b -= eta * gb;
    k += 1;
    // Keep drawn coordinates bounded once the path has left the panel.
    if (Math.abs(w) < 1e3 && Math.abs(b) < 1e3) {
      points.push([w, b]);
      dot(w, b);
    }
    const L = loss(w, b);
    if (!isFinite(L) || L > 1e5) status = "diverged";
    else if (L - lossStar < 1e-4) status = "reached the bottom";
    else if (k >= MAX_STEPS) status = `stopped after ${MAX_STEPS} steps`;
    draw();
  }

  function draw() {
    trail.setAttribute("points", points.map(([pw, pb]) => `${PX(pw)},${PY(pb)}`).join(" "));
    const shown = Math.abs(w) < 1e3 && Math.abs(b) < 1e3;
    current.style.display = shown ? "" : "none";
    if (shown) {
      current.setAttribute("cx", PX(w));
      current.setAttribute("cy", PY(b));
      fitLine.setAttribute("x1", dx(-0.5)); fitLine.setAttribute("y1", dy(b - 0.5 * w));
      fitLine.setAttribute("x2", dx(3.6)); fitLine.setAttribute("y2", dy(b + 3.6 * w));
      xs.forEach((x, i) => {
        residuals[i].setAttribute("x1", dx(x)); residuals[i].setAttribute("y1", dy(ys[i]));
        residuals[i].setAttribute("x2", dx(x)); residuals[i].setAttribute("y2", dy(w * x + b));
      });
    }
    const L = loss(w, b);
    const big = (v) => (Math.abs(v) < 1e5 ? fmt(v, 4) : v.toExponential(2));
    const fs = 1 - eta * steep, ff = 1 - eta * flat;
    out.innerHTML =
      `η = ${fmt(eta, 3)}   step ${k}${status ? `   <span class="${status === "diverged" ? "orange" : "green"}">${status}</span>` : ""}\n` +
      `<span class="orange">w = ${big(w)}   b = ${big(b)}</span>\n` +
      `loss = ${big(L)}   (lowest ${fmt(lossStar, 4)})\n` +
      `distance to the bottom, each step:\n` +
      `  steep  × (1 − η·${fmt(steep)}) = <span class="${Math.abs(fs) >= 1 ? "orange" : "blue"}">${fmt(fs)}</span>\n` +
      `  flat   × (1 − η·${fmt(flat)}) = <span class="blue">${fmt(ff)}</span>`;
  }

  const etaControl = slider(ctl, { label: "learning rate η", min: 0.01, max: 0.26, step: 0.005, value: eta, format: (v) => v.toFixed(3) }, (v) => {
    eta = v;
    reset();
  });

  let acc = 0;
  const player = loop(figure, (dt) => {
    if (!dt) return;
    acc += dt;
    if (status) {
      wait += dt;
      if (wait > HOLD) { reset(); acc = 0; }
      return;
    }
    while (acc > INTERVAL && !status) {
      acc -= INTERVAL;
      step();
    }
  });

  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
  const stop = () => { player.pause(); toggle.textContent = "Play"; };
  button(btns, "Step", () => { stop(); if (status) reset(); else step(); });
  button(btns, "Restart", () => { start = [-1, 3]; reset(); });
  for (const [label, value] of [["slow", 0.02], ["steady", 0.1], ["zig-zag", 0.2], ["diverge", 0.25]]) {
    button(btns, label, () => { etaControl.set(value); });
  }

  // Click or tap in the bowl to start from that (w, b) instead.
  root.addEventListener("click", (event) => {
    const ctm = root.getScreenCTM();
    if (!ctm) return;
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    if (p.x < panel.x0) return;
    start = [cw + (p.x - (panel.x0 + panel.x1) / 2) / scale, cb - (p.y - height / 2) / scale];
    reset();
  });

  reset();
})();
