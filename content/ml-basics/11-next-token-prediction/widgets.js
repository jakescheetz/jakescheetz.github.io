// Unit 11: logits -> softmax -> cross-entropy loss, and the p − y gradient step.
(function () {
  const figure = document.getElementById("softmax-loss");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  const tokens = ["mat", "floor", "sofa", "dog", "ran"];
  const correct = 0; // "the cat sat on the ___": the right answer is "mat"
  const start = [2, 1, 0.5, -1, -2];
  const zMax = 4; // slider range is −4 to 4
  const eta = 1; // learning rate for the Step button

  const width = 720, rows = 13;
  const rowY = (i) => 48 + 38.4 * i;
  const barH = 16;

  // Column geometry, in SVG units.
  const labelX = 85; // token labels are right-aligned here
  const zMid = 185, zScale = 80 / zMax; // logit bars grow left or right from zMid
  const pLeft = 295, pWidth = 140; // probability bars grow right from pLeft
  const gMid = 515, gScale = 60; // gradient bars, −1 to 1
  const cLeft = 610, cRight = 715, cTop = 40, cBottom = 210, lossMax = 5; // −log p curve
  const cX = (p) => cLeft + p * (cRight - cLeft);
  const cY = (loss) => cBottom - Math.min(loss, lossMax) / lossMax * (cBottom - cTop);

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Logit bars, probability bars, gradient bars and the loss curve for five candidate next tokens");

  const header = (x, text) => svg("text", { x, y: 18, "text-anchor": "middle", class: "muted", text }, root);
  header(zMid, "logit z");
  header(pLeft + pWidth / 2, "prob p");
  header(gMid, "p − y");
  header((cLeft + cRight) / 2, "−log p");

  const top = rowY(0) - barH, bottom = rowY(tokens.length - 1) + barH;
  svg("line", { x1: zMid, y1: top, x2: zMid, y2: bottom, class: "muted thin" }, root);
  svg("line", { x1: pLeft, y1: top, x2: pLeft, y2: bottom, class: "muted thin" }, root);
  svg("line", { x1: pLeft + pWidth, y1: top, x2: pLeft + pWidth, y2: bottom, class: "muted dashed thin faint" }, root);
  svg("line", { x1: gMid, y1: top, x2: gMid, y2: bottom, class: "muted thin" }, root);

  const bars = tokens.map((token, i) => {
    const cls = i === correct ? "blue" : "muted";
    const y = rowY(i);
    svg("text", { x: labelX, y, "text-anchor": "end", "dominant-baseline": "central", class: cls, text: token }, root);
    const rect = (c) => svg("rect", { y: y - barH / 2, height: barH, fill: "currentColor", class: c }, root);
    return { z: rect(cls), p: rect(cls), g: rect("orange") };
  });

  // The −log p curve for p from e^−5 to 1, with a dot at p(mat).
  svg("line", { x1: cLeft, y1: cTop, x2: cLeft, y2: cBottom, class: "muted thin" }, root);
  svg("line", { x1: cLeft, y1: cBottom, x2: cRight, y2: cBottom, class: "muted thin" }, root);
  // p = 0 and p = 1 under the curve's axis, hanging below it so larger phone labels clear the line.
  svg("text", { x: cLeft, y: cBottom + 6, "text-anchor": "middle", "dominant-baseline": "hanging", class: "muted", text: "0" }, root);
  svg("text", { x: cRight, y: cBottom + 6, "text-anchor": "middle", "dominant-baseline": "hanging", class: "muted", text: "1" }, root);
  const curve = [];
  for (let k = 0; k <= 60; k++) {
    const p = Math.exp(-lossMax * (1 - k / 60));
    curve.push(`${cX(p)},${cY(-Math.log(p))}`);
  }
  svg("polyline", { points: curve.join(" "), class: "orange" }, root);
  const guide = svg("line", { class: "muted dashed thin" }, root);
  const dot = svg("circle", { r: 5, fill: "currentColor", class: "blue" }, root);

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  const z = start.slice();
  const clamp = (v) => Math.max(-zMax, Math.min(zMax, v));
  const softmax = (v) => {
    const m = Math.max(...v);
    const e = v.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  };
  const gradient = (p) => p.map((pi, i) => pi - (i === correct ? 1 : 0));

  // Width and x of a bar from `mid` to `mid + v * scale` (either direction).
  function placeBar(rect, mid, v, scale) {
    const w = Math.abs(v) * scale;
    rect.setAttribute("x", v < 0 ? mid - w : mid);
    rect.setAttribute("width", w);
  }

  const signed = (v) => (fmt(v) === "0.00" ? " 0.00" : (v < 0 ? "−" : "+") + fmt(Math.abs(v)));

  function draw() {
    const p = softmax(z);
    const g = gradient(p);
    const loss = -Math.log(p[correct]);
    bars.forEach((bar, i) => {
      placeBar(bar.z, zMid, z[i], zScale);
      placeBar(bar.p, pLeft, p[i], pWidth);
      placeBar(bar.g, gMid, g[i], gScale);
    });
    const dx = cX(p[correct]), dy = cY(loss);
    dot.setAttribute("cx", dx); dot.setAttribute("cy", dy);
    guide.setAttribute("x1", dx); guide.setAttribute("y1", dy);
    guide.setAttribute("x2", dx); guide.setAttribute("y2", cBottom);

    const e = z.map(Math.exp);
    const sum = e.reduce((a, b) => a + b, 0);
    out.innerHTML =
      `Σ e^z  = ${e.map((v) => fmt(v)).join(" + ")} = ${fmt(sum)}\n` +
      `<span class="blue">p(mat)</span> = ${fmt(e[correct])} / ${fmt(sum)} = <span class="blue">${fmt(p[correct], 3)}</span>\n` +
      `loss   = −log p(mat) = <span class="orange">${fmt(loss, 3)}</span>\n` +
      `p − y  = (${g.map(signed).join(", ")})`;
  }

  const sliders = tokens.map((token, i) =>
    slider(ctl, { label: `z(${token})`, min: -zMax, max: zMax, step: 0.1, value: z[i], format: (v) => fmt(v, 1) }, (v) => {
      z[i] = v;
      draw();
    })
  );

  // One gradient-descent step on the logits: z ← z − η (p − y), held inside the slider range.
  function step(size) {
    const g = gradient(softmax(z));
    z.forEach((v, i) => {
      z[i] = clamp(v - size * g[i]);
      sliders[i].show(z[i]);
    });
    draw();
  }

  // "Train" runs the same update continuously, about two steps per second, until paused.
  // Logits stop at the edge of the slider range, so mat tops out at 4.
  const player = loop(figure, (dt) => {
    if (dt) step(2 * eta * dt);
  });
  player.pause();

  function stop() {
    player.pause();
    train.textContent = "Train";
  }

  button(btns, "Step", () => {
    stop();
    step(eta);
  });
  const train = button(btns, "Train", () => {
    if (player.playing) stop();
    else {
      player.play();
      train.textContent = "Pause";
    }
  });
  button(btns, "Reset", () => {
    stop();
    start.forEach((v, i) => sliders[i].set(v));
  });
  // Grabbing a slider stops training so it doesn't fight the reader.
  ctl.addEventListener("input", stop);

  draw();
})();
