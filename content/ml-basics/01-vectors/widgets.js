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
      `a = (3.00, 1.00)  <span class="blue">b = (${fmt(b[0])}, ${fmt(b[1])})</span>\n` +
      `a · b = 3.00 × ${fmt(b[0])} + 1.00 × ${fmt(b[1])}\n` +
      `      = <span class="orange">${fmt(dot)}</span>\n` +
      `shadow of b on a = ${fmt(along)}\n` +
      `shadow × |a| = ${fmt(along)} × ${fmt(aLength)} = <span class="orange">${fmt(dot)}</span>\n` +
      `cosine similarity = ${fmt(cos)} (${fmt(degrees, 0)}°)`;
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

// Step-through: the dot product, then the shadow picture, then cosine similarity.
(function () {
  const O = { x: 235, y: 255 }, U = 55;
  const X = (v) => O.x + v * U, Y = (v) => O.y - v * U;
  const angA = Math.atan2(1, 3), angB = Math.atan2(2, 1), lenB = Math.sqrt(5);
  Learn.demo(document.getElementById("demo-dot"), {
    height: 360,
    label: "Two arrows a and b, the shadow of b on the line through a, and the dot product",
    steps: [
      { s: { a: 1, b: 0, calc: 0, sh: 0, cs: 0, ang: angB }, cap: "This is vector a = (3, 1). Read it as directions from the origin: 3 steps right, then 1 step up." },
      { s: { b: 1 }, cap: "This is vector b = (1, 2): 1 step right, then 2 steps up." },
      { s: { calc: 1 }, cap: "The dot product: multiply the matching parts, then add. 3 × 1 + 1 × 2 = 5. The answer is one number, not a list." },
      { s: { sh: 1 }, cap: "The shadow picture. Shine a light straight down onto a's line (dashed). The thick orange bar is b's shadow. A long shadow in a's direction means a big positive dot product." },
      { s: { ang: angA + Math.PI / 2 }, cap: "Turn b until it sits at a right angle (90°) to a. Now b casts no shadow on a's line. The dot product is 0." },
      { s: { ang: angA + Math.PI * 0.83 }, cap: "Keep turning. Now b points away from a. Its shadow falls behind the origin, so the dot product is negative." },
      { s: { ang: angB, cs: 1 }, cap: "Back to b = (1, 2). The angle between a and b is 45°. Cosine similarity divides the dot product by both lengths: 5 ÷ (3.16 × 2.24) ≈ 0.71." },
    ],
    draw(ctx, s, c, d) {
      for (let gx = -4; gx <= 7; gx++) d.line(X(gx), 30, X(gx), 340, c.line);
      for (let gy = -1; gy <= 4; gy++) d.line(X(-4), Y(gy), X(7), Y(gy), c.line);
      d.line(X(-4), O.y, X(7), O.y, c.muted, 1.5);
      d.line(O.x, 30, O.x, 340, c.muted, 1.5);
      const bx = lenB * Math.cos(s.ang), by = lenB * Math.sin(s.ang);
      const dot = 3 * bx + by, t = dot / 10, px = 3 * t, py = t;
      if (s.sh > 0.01) {
        ctx.globalAlpha = s.sh;
        d.line(X(-3.9), Y(-1.3), X(7), Y(7 / 3), c.muted, 1.5, [6, 6]);
        d.line(X(bx), Y(by), X(px), Y(py), c.muted, 1.5, [3, 4]);
        ctx.globalAlpha = s.sh * 0.55;
        d.line(X(0), Y(0), X(px), Y(py), c.orange, 9);
        ctx.globalAlpha = 1;
      }
      if (s.cs > 0.01) {
        ctx.globalAlpha = s.cs;
        ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(O.x, O.y, 34, -angA, -s.ang, true); ctx.stroke();
        d.text("45°", O.x + 44, O.y - 34, { c: c.ink, w: 700 });
      }
      ctx.globalAlpha = s.a;
      d.arrow(X(0), Y(0), X(3), Y(1), c.blue, 3.5);
      d.text("a = (3, 1)", X(3) + 6, Y(1) + 20, { c: c.blue, w: 700 });
      ctx.globalAlpha = s.b;
      d.arrow(X(0), Y(0), X(bx), Y(by), c.orange, 3.5);
      d.text(`b = (${d.fmt(bx, 1)}, ${d.fmt(by, 1)})`, bx >= 0 ? X(bx) + 10 : Math.max(136, X(bx) - 10), Y(by) - 14, { c: c.orange, w: 700, a: bx >= 0 ? "left" : "right" });
      ctx.globalAlpha = 1;
      if (s.calc > 0.01) {
        ctx.globalAlpha = s.calc;
        d.panel(c, 420, 30, 208, s.cs > 0.01 ? 180 : 140);
        d.text("dot product a · b", 434, 52, { c: c.muted, s: 13 });
        d.text(`3×${d.fmt(bx, 1)} + 1×${d.fmt(by, 1)}`, 434, 78, { c: c.ink });
        d.text(`= ${d.fmt(dot, 1)}`, 434, 108, { c: c.ink, s: 24, w: 700 });
        const note = Math.abs(dot) < 0.06 ? "zero: right angle" : dot > 0 ? "positive: same side" : "negative: apart";
        d.text(note, 434, 142, { c: dot < -0.06 ? c.red : c.muted, s: 13, w: 700 });
        if (s.cs > 0.01) {
          ctx.globalAlpha = s.cs * s.calc;
          d.text("cosine similarity", 434, 170, { c: c.muted, s: 13 });
          d.text(`= ${d.fmt(dot / (Math.sqrt(10) * lenB), 2)}`, 434, 192, { c: c.blue, s: 17, w: 700 });
        }
      }
    },
  });
})();

// Step-through: a 3 × 2 matrix times a vector, one row at a time.
(function () {
  const W = [[2, 0], [1, 1], [0, -1]], out = [6, 4, -1], rowsY = [100, 160, 220];
  const paren = (v) => (v < 0 ? `(−${Math.abs(v)})` : String(v));
  const minus = (v) => (v < 0 ? `−${Math.abs(v)}` : String(v));
  Learn.demo(document.getElementById("demo-matrix"), {
    height: 340,
    label: "A 3 by 2 matrix multiplying a vector, one row at a time",
    steps: [
      { s: { row: 0, hl: 0, filled: 0, shape: 1, done: 0 }, cap: "W is a grid with 3 rows and 2 columns. The vector a has 2 numbers. W's column count (2) matches a's length (2), so they can multiply." },
      { s: { hl: 1, filled: 1, shape: 0 }, cap: "Take row 1 of W and dot it with a: 2 × 3 + 0 × 1 = 6. That is the first output number." },
      { s: { row: 1, filled: 2 }, cap: "Next, row 2 dotted with a: 1 × 3 + 1 × 1 = 4. That is the second output number." },
      { s: { row: 2, filled: 3 }, cap: "Then row 3 dotted with a: 0 × 3 + (−1) × 1 = −1. That is the third output number." },
      { s: { hl: 0, done: 1 }, cap: "Done. 2 numbers went in and 3 came out. Each row scored a against its own direction, like a detector." },
    ],
    draw(ctx, s, c, d) {
      const bracket = (x1, x2, y1, y2, color) => {
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x1 + 10, y1); ctx.lineTo(x1, y1); ctx.lineTo(x1, y2); ctx.lineTo(x1 + 10, y2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x2 - 10, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y2); ctx.lineTo(x2 - 10, y2); ctx.stroke();
      };
      const ry = 100 + s.row * 60;
      if (s.hl > 0.01) {
        ctx.globalAlpha = s.hl;
        d.rect(92, ry - 24, 156, 48, c.soft);
        d.rect(290, 104, 64, 112, c.soft);
        ctx.globalAlpha = s.hl * 0.3;
        d.rect(432, ry - 24, 76, 48, c.orange);
        ctx.globalAlpha = 1;
      }
      bracket(96, 244, 70, 250, c.ink);
      for (let r = 0; r < 3; r++) for (let k = 0; k < 2; k++) d.text(minus(W[r][k]), 140 + k * 60, rowsY[r], { c: c.ink, s: 22, a: "center" });
      d.text("W (3 × 2)", 170, 276, { c: c.muted, a: "center", w: 700 });
      d.text("×", 270, 160, { c: c.muted, s: 24, a: "center" });
      bracket(296, 348, 115, 205, c.blue);
      d.text("3", 322, 135, { c: c.blue, s: 22, a: "center" });
      d.text("1", 322, 185, { c: c.blue, s: 22, a: "center" });
      d.text("a (2)", 322, 276, { c: c.muted, a: "center", w: 700 });
      d.text("=", 390, 160, { c: c.muted, s: 24, a: "center" });
      bracket(436, 504, 70, 250, c.orange);
      for (let j = 0; j < 3; j++) {
        ctx.globalAlpha = d.clamp(s.filled - j, 0, 1);
        d.text(minus(out[j]), 470, rowsY[j], { c: c.ink, s: 22, a: "center", w: 700 });
        ctx.globalAlpha = 1;
        if (s.filled - j < 0.5) d.text("?", 470, rowsY[j], { c: c.muted, s: 22, a: "center" });
      }
      d.text("Wa (3)", 470, 276, { c: c.muted, a: "center", w: 700 });
      const r = Math.round(s.row);
      const line = (alpha, str, color) => { if (alpha > 0.05) { ctx.globalAlpha = alpha; d.text(str, 320, 316, { c: color, s: 16, a: "center", w: 700 }); ctx.globalAlpha = 1; } };
      line(s.hl, `Row ${r + 1}:  ${W[r][0]}×3 + ${paren(W[r][1])}×1 = ${minus(out[r])}`, c.ink);
      line(s.shape, "columns of W = 2 = length of a  ✓", c.blue);
      line(s.done, "2 numbers in  →  3 numbers out", c.blue);
    },
  });
})();
