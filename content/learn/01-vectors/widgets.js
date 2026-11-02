// Unit 01: the dot product as b's shadow on a.
(function () {
  const figure = document.getElementById("dot-product");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, arrow, loop, fmt } = Learn;

  const width = 720, rows = 14, height = rows * Learn.ROW;
  const scale = 44; // SVG units per unit of vector length
  const ox = width / 2, oy = height / 2;
  const X = (x) => ox + x * scale;
  const Y = (y) => oy - y * scale;

  const a = [3, 1];
  const aLength = Math.hypot(a[0], a[1]);
  const aUnit = [a[0] / aLength, a[1] / aLength];

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Vector a, vector b, and the shadow b casts on the line through a");
  svg("line", { x1: 0, y1: oy, x2: width, y2: oy, class: "muted faint thin" }, root);
  svg("line", { x1: ox, y1: 0, x2: ox, y2: height, class: "muted faint thin" }, root);
  // The line through a, extended both ways so a shadow can land behind the origin.
  svg("line", { x1: X(-8 * aUnit[0]), y1: Y(-8 * aUnit[1]), x2: X(8 * aUnit[0]), y2: Y(8 * aUnit[1]), class: "muted dashed thin" }, root);
  const shadow = svg("line", { class: "orange thick" }, root);
  const drop = svg("line", { class: "muted dashed thin" }, root);
  arrow(root, "ink").update(ox, oy, X(a[0]), Y(a[1]));
  svg("text", { x: X(a[0]) + 10, y: Y(a[1]) + 4, text: "a" }, root);
  const bArrow = arrow(root, "blue");
  const bLabel = svg("text", { class: "blue", text: "b" }, root);

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  let angle = Math.atan2(2, 1) * 180 / Math.PI; // b starts at (1, 2), the vector from the text
  let length = Math.hypot(1, 2);

  function draw() {
    const rad = angle * Math.PI / 180;
    const b = [length * Math.cos(rad), length * Math.sin(rad)];
    const dot = a[0] * b[0] + a[1] * b[1];
    const along = dot / aLength; // signed length of b's shadow on a
    const p = [along * aUnit[0], along * aUnit[1]];

    bArrow.update(ox, oy, X(b[0]), Y(b[1]));
    bLabel.setAttribute("x", X(b[0] * 1.12) - 4);
    bLabel.setAttribute("y", Y(b[1] * 1.12) + 4);
    shadow.setAttribute("x1", ox); shadow.setAttribute("y1", oy);
    shadow.setAttribute("x2", X(p[0])); shadow.setAttribute("y2", Y(p[1]));
    drop.setAttribute("x1", X(b[0])); drop.setAttribute("y1", Y(b[1]));
    drop.setAttribute("x2", X(p[0])); drop.setAttribute("y2", Y(p[1]));

    const cos = dot / (aLength * length);
    const degrees = Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
    out.innerHTML =
      `a = (3.00, 1.00)   <span class="blue">b = (${fmt(b[0])}, ${fmt(b[1])})</span>\n` +
      `a · b = 3.00 × ${fmt(b[0])} + 1.00 × ${fmt(b[1])} = <span class="orange">${fmt(dot)}</span>\n` +
      `shadow of b on a = ${fmt(along)}, times |a| (${fmt(aLength)}) = <span class="orange">${fmt(dot)}</span>\n` +
      `cosine similarity = ${fmt(cos)}   (angle between them ${fmt(degrees, 0)}°)`;
  }

  const angleControl = slider(ctl, { label: "angle of b", min: 0, max: 359, step: 1, value: Math.round(angle), format: (v) => `${Math.round(v)}°` }, (v) => {
    angle = v;
    draw();
  });
  slider(ctl, { label: "length of b", min: 0.5, max: 3, step: 0.1, value: length.toFixed(1), format: (v) => v.toFixed(1) }, (v) => {
    length = v;
    draw();
  });

  const player = loop(figure, (dt) => {
    if (dt) {
      angle = (angle + 24 * dt) % 360;
      angleControl.show(angle);
    }
    draw();
  });
  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
  // Grabbing a slider stops the animation so it doesn't fight the reader.
  ctl.addEventListener("input", () => {
    player.pause();
    toggle.textContent = "Play";
  });
})();
