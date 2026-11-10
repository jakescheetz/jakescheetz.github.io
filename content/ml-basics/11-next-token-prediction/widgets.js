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
      `Σ e^z  = ${e.slice(0, 3).map((v) => fmt(v)).join(" + ")}\n` +
      `       + ${e.slice(3).map((v) => fmt(v)).join(" + ")} = ${fmt(sum)}\n` +
      `<span class="blue">p(mat)</span> = ${fmt(e[correct])} / ${fmt(sum)} = <span class="blue">${fmt(p[correct], 3)}</span>\n` +
      `loss   = −log p(mat) = <span class="orange">${fmt(loss, 3)}</span>\n` +
      `p − y  = (${g.slice(0, 3).map(signed).join(", ")},\n` +
      `          ${g.slice(3).map(signed).join(", ")})`;
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

// Step-through: softmax on the five example logits, one move at a time.
(function () {
  const names = ["mat", "floor", "sofa", "dog", "ran"], z0 = [2, 1, 0.5, -1, -2];
  const fix = (v, n = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(n);
  const ex0 = z0.map(Math.exp), sum0 = ex0.reduce((a, b) => a + b, 0), p0 = ex0.map((v) => v / sum0);
  const rowY = (i) => 70 + 38 * i;
  Learn.demo(document.getElementById("demo-softmax"), {
    height: 320,
    label: "Five tokens with their logits, the exponential of each logit, and the probabilities softmax gives them",
    steps: [
      { s: { ex: 0, sm: 0, pr: 0, sh: 0, rt: 0 }, cap: "After \"the cat sat on the\", the model gives five logits. mat has the highest, 2.0. Logits can be negative, and they don't add up to 1." },
      { s: { ex: 1 }, cap: `Raise e (about 2.718) to each logit. For mat that is ${fix(ex0[0])}, and for ran it is ${fix(ex0[4])}. Every number is now positive, and the order has not changed.` },
      { s: { sm: 1 }, cap: `Add the five numbers. The total is ${fix(sum0)}.` },
      { s: { pr: 1 }, cap: `Divide each number by the total. mat gets ${fix(ex0[0])} ÷ ${fix(sum0)} = ${fix(p0[0])}. The five probabilities add up to 1.` },
      { s: { sh: 3 }, cap: `Add 3 to every logit. Every exponential grows by the same factor, ${fix(Math.exp(3))}, and so does the total. The factor cancels in the division, so the probabilities do not move.` },
      { s: { sh: 0, rt: 1 }, cap: `Back to the original logits. mat and floor differ by 1, so mat is e times as likely: ${fix(p0[0], 3)} ÷ ${fix(p0[1], 3)} = ${fix(p0[0] / p0[1])}. Only the gaps between logits matter.` },
    ],
    draw(ctx, s, c, d) {
      const ex = z0.map((v) => Math.exp(v + s.sh)), sum = ex.reduce((a, b) => a + b, 0);
      d.text("token", 24, 34, { c: c.muted, s: 13 });
      d.text("logit z", 200, 34, { c: c.muted, s: 13, a: "right" });
      if (s.rt > 0.01) {
        ctx.globalAlpha = s.rt;
        d.rect(14, rowY(0) - 18, 600, 74, null, c.orange, 2);
        d.text(`${fix(p0[0], 3)} ÷ ${fix(p0[1], 3)} = ${fix(p0[0] / p0[1])}`, 24, 296, { c: c.orange, w: 700 });
        ctx.globalAlpha = 1;
      }
      names.forEach((name, i) => {
        const y = rowY(i), main = i === 0;
        d.text(name, 24, y, { c: main ? c.blue : c.ink, s: 15, w: main ? 700 : 500 });
        d.text(fix(z0[i] + s.sh, 1), 200, y, { c: c.ink, s: 15, a: "right" });
        if (s.ex > 0.01) {
          ctx.globalAlpha = s.ex;
          d.text(fix(ex[i]), 320, y, { c: c.ink, s: 15, a: "right" });
        }
        if (s.pr > 0.01) {
          const p = ex[i] / sum;
          ctx.globalAlpha = s.pr;
          d.rect(360, y - 10, 240 * p * s.pr, 20, main ? c.blue : c.muted);
          d.text(fix(p), 368 + 240 * p * s.pr, y, { c: main ? c.blue : c.ink, w: main ? 700 : 500 });
        }
        ctx.globalAlpha = 1;
      });
      if (s.ex > 0.01) {
        ctx.globalAlpha = s.ex;
        d.text("e^z", 320, 34, { c: c.muted, s: 13, a: "right" });
        ctx.globalAlpha = 1;
      }
      if (s.sm > 0.01) {
        ctx.globalAlpha = s.sm;
        d.line(236, 244, 324, 244, c.ink, 2);
        d.text(`total ${fix(sum)}`, 320, 264, { c: c.orange, s: 15, a: "right", w: 700 });
        ctx.globalAlpha = 1;
      }
      if (s.pr > 0.01) {
        ctx.globalAlpha = s.pr;
        d.text("p = e^z ÷ total", 360, 34, { c: c.muted, s: 13 });
        d.text("adds up to 1.00", 360, 264, { c: c.muted });
        ctx.globalAlpha = 1;
      }
      const grown = d.clamp(s.sh / 3, 0, 1);
      if (grown > 0.01) {
        ctx.globalAlpha = grown;
        d.text(`every e^z and the total: ×${fix(Math.exp(s.sh))}`, 340, 296, { c: c.muted, s: 13 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: the loss −log p for a few probabilities on the right answer.
(function () {
  const X = (p) => 80 + 400 * p, Y = (loss) => 290 - 48 * loss;
  const L = (p) => -Math.log(p);
  const f = (v) => v.toFixed(2);
  Learn.demo(document.getElementById("demo-loss"), {
    height: 340,
    label: "The curve of minus log p, with a dot showing the loss for the probability the model put on the right token",
    steps: [
      { s: { p: 0.6 }, cap: `The right answer is mat, and the model gave it probability 0.60. The loss is −log 0.60 = ${f(L(0.6))}. The curve shows the loss for every possible probability.` },
      { s: { p: 0.9 }, cap: `Suppose the model were more sure, and right: p = 0.90. The loss drops to ${f(L(0.9))}. Being sure and right costs almost nothing.` },
      { s: { p: 0.5 }, cap: `At p = 0.50, a coin flip on the right answer, the loss is ${f(L(0.5))}.` },
      { s: { p: 0.1 }, cap: `At p = 0.10 the loss is ${f(L(0.1))}. The curve gets steeper as p gets smaller.` },
      { s: { p: 0.01 }, cap: `At p = 0.01 the loss is ${f(L(0.01))}. Each time p drops by a factor of 10, the loss rises by another ${f(L(0.1))}, with no ceiling. Confident mistakes are the expensive ones.` },
    ],
    draw(ctx, s, c, d) {
      for (let k = 0; k <= 5; k++) {
        d.line(X(0), Y(k), X(1), Y(k), c.line);
        d.text(String(k), X(0) - 10, Y(k), { c: c.muted, s: 13, a: "right" });
      }
      [0, 0.25, 0.5, 0.75, 1].forEach((p) => {
        d.line(X(p), Y(0), X(p), Y(5), c.line);
        d.text(p.toFixed(2), X(p), 306, { c: c.muted, s: 13, a: "center" });
      });
      d.line(X(0), Y(0), X(1), Y(0), c.muted, 1.5);
      d.line(X(0), Y(0), X(0), Y(5), c.muted, 1.5);
      d.text("loss = −log p", X(0) + 14, 38, { c: c.muted, s: 13 });
      d.text("probability on the right token", X(0.5), 328, { c: c.muted, s: 13, a: "center" });
      ctx.strokeStyle = c.blue; ctx.lineWidth = 3;
      ctx.beginPath();
      for (let k = 0; k <= 200; k++) {
        const p = Math.exp(-5 + 5 * k / 200); // even steps in log p, so the steep end stays smooth
        if (k === 0) ctx.moveTo(X(p), Y(L(p))); else ctx.lineTo(X(p), Y(L(p)));
      }
      ctx.stroke();
      const loss = L(s.p), px = X(s.p), py = Y(Math.min(loss, 5));
      d.line(px, py, px, Y(0), c.orange, 1.5, [4, 4]);
      d.line(X(0), py, px, py, c.orange, 1.5, [4, 4]);
      d.dot(px, py, 7, c.orange);
      d.panel(c, 496, 60, 132, 124);
      d.text("p(mat)", 510, 82, { c: c.muted, s: 13 });
      d.text(f(s.p), 510, 106, { c: c.ink, s: 20, w: 700 });
      d.text("loss", 510, 136, { c: c.muted, s: 13 });
      d.text(f(loss), 510, 160, { c: c.orange, s: 22, w: 700 });
    },
  });
})();

// Step-through: the gradient p − y on the example logits, then gradient steps on the logits.
(function () {
  const names = ["mat", "floor", "sofa", "dog", "ran"], y = [1, 0, 0, 0, 0];
  const softmax = (z) => { const e = z.map(Math.exp), t = e.reduce((a, b) => a + b, 0); return e.map((v) => v / t); };
  const f = (v) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(2);
  const signed = (v) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);
  // zs[k] holds the logits after k gradient steps with learning rate 1 (the worked example).
  const zs = [[2, 1, 0.5, -1, -2]];
  for (let k = 0; k < 3; k++) { const p = softmax(zs[k]); zs.push(zs[k].map((v, i) => v - (p[i] - y[i]))); }
  const ps = zs.map(softmax), loss = ps.map((p) => -Math.log(p[0])), g = ps[0].map((v, i) => v - y[i]);
  const rowY = (i) => 74 + 38 * i;
  Learn.demo(document.getElementById("demo-gradient"), {
    height: 310,
    label: "Five tokens with their logits, probabilities, one-hot target and the gradient p minus y, before and after gradient steps",
    steps: [
      { s: { k: 0, y: 0, g: 0, ar: 0 }, cap: `The bars are the model's probabilities p. mat is the right answer and has ${f(ps[0][0])}, so the loss is ${f(loss[0])}.` },
      { s: { y: 1 }, cap: "The target y is one-hot: 1 for mat and 0 for every other token. It says where all the probability should have gone." },
      { s: { g: 1 }, cap: `The gradient is p − y. For mat that is ${f(ps[0][0])} − 1 = ${f(g[0])}. For every other token it is just that token's own probability. The five entries add up to 0.` },
      { s: { ar: 1 }, cap: `Gradient descent subtracts the gradient from the logits. mat's logit goes up by ${f(-g[0])}. floor's goes down by ${f(g[1])}, the most of any wrong token. ran barely moves.` },
      { s: { k: 1, ar: 0 }, cap: `After one step with learning rate 1, mat's logit is ${f(zs[1][0])} and floor's is ${f(zs[1][1])}. p(mat) rose from ${f(ps[0][0])} to ${f(ps[1][0])}, and the loss fell from ${f(loss[0])} to ${f(loss[1])}.` },
      { s: { k: 3 }, cap: `After three steps p(mat) is ${f(ps[3][0])} and the loss is ${f(loss[3])}. The gradient shrinks as p gets closer to y, so each step moves the logits less.` },
    ],
    draw(ctx, s, c, d) {
      const k = d.clamp(s.k, 0, 3), lo = Math.min(2, Math.floor(k)), t = k - lo;
      const z = zs[lo].map((v, i) => v + (zs[lo + 1][i] - v) * t), p = softmax(z);
      d.text("token", 24, 36, { c: c.muted, s: 13 });
      d.text("logit z", 176, 36, { c: c.muted, s: 13, a: "right" });
      d.text("probability p", 230, 36, { c: c.muted, s: 13 });
      names.forEach((name, i) => {
        const yy = rowY(i), main = i === 0, grad = p[i] - y[i];
        d.text(name, 24, yy, { c: main ? c.blue : c.ink, s: 15, w: main ? 700 : 500 });
        d.text(f(z[i]), 176, yy, { c: c.ink, s: 15, a: "right" });
        d.rect(230, yy - 10, 180 * p[i], 20, main ? c.blue : c.muted);
        d.text(f(p[i]), 238 + 180 * p[i], yy, { c: main ? c.blue : c.ink, w: main ? 700 : 500 });
        if (s.ar > 0.01) {
          const len = 50 * Math.abs(grad), dir = grad < 0 ? -1 : 1; // a negative gradient pushes the logit up
          ctx.globalAlpha = s.ar;
          d.arrow(198, yy - dir * len / 2, 198, yy + dir * len / 2, grad < 0 ? c.blue : c.orange, 3);
        }
        if (s.y > 0.01) {
          ctx.globalAlpha = s.y;
          d.text(String(y[i]), 470, yy, { c: main ? c.blue : c.muted, s: 15, a: "center", w: main ? 700 : 500 });
        }
        if (s.g > 0.01) {
          ctx.globalAlpha = s.g;
          d.text(signed(grad), 600, yy, { c: c.orange, s: 15, a: "right", w: 700 });
        }
        ctx.globalAlpha = 1;
      });
      if (s.y > 0.01) {
        ctx.globalAlpha = s.y;
        d.text("target y", 470, 36, { c: c.muted, s: 13, a: "center" });
      }
      if (s.g > 0.01) {
        ctx.globalAlpha = s.g;
        d.text("p − y", 600, 36, { c: c.orange, s: 13, a: "right", w: 700 });
      }
      ctx.globalAlpha = 1;
      d.line(24, 252, 616, 252, c.muted, 2);
      const done = Math.round(k);
      d.text(done === 0 ? "before any step" : `after ${done} step${done === 1 ? "" : "s"}`, 24, 280, { c: c.muted });
      d.text(`loss = −log p(mat) = ${f(-Math.log(p[0]))}`, 230, 280, { c: c.ink, s: 15, w: 700 });
    },
  });
})();
