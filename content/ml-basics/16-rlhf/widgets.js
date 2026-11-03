// Unit 16: the Bradley–Terry curve, and a reward model fitting four answers from comparisons.
(function () {
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;
  const sigmoid = (t) => 1 / (1 + Math.exp(-t));
  const num = (v) => fmt(v).replace("-", "−");

  // Widget 1: P(y_w beats y_l) = sigmoid(reward gap), and the push 1 − P each comparison gives.
  (function () {
    const figure = document.getElementById("bt-curve");
    if (!figure) return;

    const width = 720, rows = 12;
    const left = 70, right = 690, top = 20, bottom = 180;
    const X = (d) => left + (d + 6) / 12 * (right - left);
    const Y = (p) => bottom - p * (bottom - top);

    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "The sigmoid curve of reward gap against probability, split at the current gap into P and 1 − P");
    svg("line", { x1: left, y1: Y(1), x2: right, y2: Y(1), class: "muted dashed thin" }, root);
    svg("line", { x1: left, y1: Y(0.5), x2: right, y2: Y(0.5), class: "muted faint thin" }, root);
    svg("line", { x1: left, y1: Y(0), x2: right, y2: Y(0), class: "muted thin" }, root);
    svg("line", { x1: X(0), y1: Y(0), x2: X(0), y2: Y(1), class: "muted faint thin" }, root);
    for (const [p, label] of [[1, "1"], [0.5, "0.5"], [0, "0"]]) {
      svg("text", { x: left - 10, y: Y(p) + 4, "text-anchor": "end", class: "muted", text: label }, root);
    }
    for (const d of [-6, 0, 6]) {
      svg("text", { x: X(d), y: bottom + 30, "text-anchor": "middle", class: "muted", text: d < 0 ? `−${-d}` : String(d) }, root);
    }
    svg("text", { x: X(3), y: bottom + 30, "text-anchor": "middle", class: "muted", text: "reward gap" }, root);

    const points = [];
    for (let d = -6; d <= 6.001; d += 0.1) points.push(`${X(d).toFixed(1)},${Y(sigmoid(d)).toFixed(1)}`);
    svg("polyline", { points: points.join(" "), class: "ink" }, root);

    const pBar = svg("line", { class: "blue thick" }, root);
    const pushBar = svg("line", { class: "orange thick" }, root);
    const dot = svg("circle", { r: 7, fill: "currentColor", class: "blue" }, root);
    const pLabel = svg("text", { class: "blue", text: "P" }, root);
    const pushLabel = svg("text", { class: "orange", text: "1 − P" }, root);

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);
    // The curve rises left to right, so above the point it lies to the right and below it to the
    // left: put the 1 − P label on the left and the P label on the right, unless near an edge.
    function place(label, x, y1, y2, onLeft) {
      label.setAttribute("x", onLeft ? x - 12 : x + 12);
      label.setAttribute("text-anchor", onLeft ? "end" : "start");
      label.setAttribute("y", (y1 + y2) / 2 + 4);
      label.style.display = Math.abs(y2 - y1) < 22 ? "none" : "";
    }

    function draw(d) {
      const p = sigmoid(d);
      const x = X(d), y = Y(p);
      pBar.setAttribute("x1", x); pBar.setAttribute("y1", Y(0));
      pBar.setAttribute("x2", x); pBar.setAttribute("y2", y);
      pushBar.setAttribute("x1", x); pushBar.setAttribute("y1", y);
      pushBar.setAttribute("x2", x); pushBar.setAttribute("y2", Y(1));
      dot.setAttribute("cx", x); dot.setAttribute("cy", y);
      place(pLabel, x, Y(0), y, d > 4.5);
      place(pushLabel, x, y, Y(1), d > -4.5);

      out.innerHTML =
        `r(y_w) − r(y_l) = ${num(d)}\n` +
        `P(y_w ≻ y_l) = σ(${num(d)}) = <span class="blue">${fmt(p)}</span>\n` +
        `loss = −log P = ${fmt(-Math.log(p))}\n` +
        `push = 1 − P = <span class="orange">${fmt(1 - p)}</span>, r(y_w) up and r(y_l) down`;
    }

    const gap = slider(ctl, { label: "reward gap", min: -6, max: 6, step: 0.1, value: 0.9, format: (v) => num(v) }, draw);
    button(btns, "People pick A over B", () => gap.set(0.9));
    button(btns, "People pick B over A", () => gap.set(-0.9));
  })();

  // Widget 2: four answers, rewards learned one comparison at a time.
  (function () {
    const figure = document.getElementById("bt-fit");
    if (!figure) return;

    const names = ["A", "B", "C", "D"];
    const hidden = [1.5, 0.6, -0.4, -1.7]; // what the simulated people go by
    const width = 720, rows = 11;
    const left = 80, right = 680, axisY = 176;
    const X = (r) => left + (Math.max(-3.4, Math.min(3.4, r)) + 3) / 6 * (right - left);
    const rowY = (i) => 24 + i * 38;

    const root = stage(figure, { rows, width });
    root.setAttribute("aria-label", "Four answers on a reward axis: hollow rings at the hidden rewards, filled dots at the learned estimates");
    svg("line", { x1: left, y1: axisY, x2: right, y2: axisY, class: "muted thin" }, root);
    for (let r = -3; r <= 3; r++) {
      svg("line", { x1: X(r), y1: axisY - 4, x2: X(r), y2: axisY + 4, class: "muted thin" }, root);
    }
    for (const r of [-3, 0, 3]) {
      svg("text", { x: X(r), y: axisY + 30, "text-anchor": "middle", class: "muted", text: r < 0 ? `−${-r}` : String(r) }, root);
    }
    svg("line", { x1: X(0), y1: 8, x2: X(0), y2: axisY, class: "muted faint thin dashed" }, root);

    const dots = names.map((name, i) => {
      const y = rowY(i);
      svg("line", { x1: left, y1: y, x2: right, y2: y, class: "muted faint thin" }, root);
      svg("text", { x: 40, y: y + 5, "text-anchor": "middle", text: name }, root);
      svg("circle", { cx: X(hidden[i]), cy: y, r: 9, fill: "none", stroke: "currentColor", "stroke-width": 2, class: "muted" }, root);
      return svg("circle", { cy: y, r: 7, fill: "currentColor", class: "blue" }, root);
    });

    const out = readout(figure);
    const ctl = controls(figure);
    const btns = buttons(figure);

    let r, count, last, history, carry;
    let eta = 0.1, speed = 6;

    function reset() {
      r = [0, 0, 0, 0];
      count = 0;
      last = null;
      history = [];
      carry = 0;
    }

    function compare() {
      const i = Math.floor(Math.random() * 4);
      const j = (i + 1 + Math.floor(Math.random() * 3)) % 4;
      const iWins = Math.random() < sigmoid(hidden[i] - hidden[j]);
      const w = iWins ? i : j, l = iWins ? j : i;
      const p = sigmoid(r[w] - r[l]); // the model's probability for the pick people made
      const push = eta * (1 - p);
      r[w] += push;
      r[l] -= push;
      history.push([p > 0.5 ? 1 : p === 0.5 ? 0.5 : 0, hidden[w] > hidden[l] ? 1 : 0]);
      if (history.length > 100) history.shift();
      last = { w, l, p, push };
      count++;
    }

    function draw() {
      dots.forEach((dot, i) => {
        dot.setAttribute("cx", X(r[i]));
        dot.setAttribute("class", last && i === last.w ? "green" : last && i === last.l ? "orange" : "blue");
      });
      const cell = (v) => num(v).padStart(7);
      const right100 = (k) => Math.round(history.reduce((s, h) => s + h[k], 0) / Math.max(1, history.length) * 100);
      const latest = last
        ? `<span class="green">${names[last.w]}</span> beat <span class="orange">${names[last.l]}</span>. ` +
          `Model gave that ${fmt(last.p)}, push ${fmt(last.push, 3)}`
        : "none yet";
      out.innerHTML =
        `comparisons: ${count}\n` +
        `latest: ${latest}\n` +
        `       ${names.map((n) => n.padStart(7)).join("")}\n` +
        `model  <span class="blue">${r.map(cell).join("")}</span>\n` +
        `hidden ${hidden.map(cell).join("")}\n` +
        `picks called right, last ${history.length}:\n` +
        `  model ${right100(0)}%, hidden rewards ${right100(1)}%`;
    }

    slider(ctl, { label: "step size η", min: 0.02, max: 0.5, step: 0.02, value: eta, format: (v) => v.toFixed(2) }, (v) => { eta = v; });
    slider(ctl, { label: "comparisons/s", min: 1, max: 40, step: 1, value: speed, format: String }, (v) => { speed = v; });

    reset();
    const player = loop(figure, (dt) => {
      carry += dt * speed;
      while (carry >= 1) { compare(); carry -= 1; }
      draw();
    });
    const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
      if (player.playing) player.pause(); else player.play();
      toggle.textContent = player.playing ? "Pause" : "Play";
    });
    button(btns, "1 comparison", () => { compare(); draw(); });
    button(btns, "+100", () => { for (let k = 0; k < 100; k++) compare(); draw(); });
    button(btns, "Reset", () => { reset(); draw(); });
  })();
})();
