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
        `push = 1 − P = <span class="orange">${fmt(1 - p)}</span>\n` +
        `r(y_w) up and r(y_l) down`;
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
        ? `<span class="green">${names[last.w]}</span> beat <span class="orange">${names[last.l]}</span>\n` +
          `model gave that ${fmt(last.p)}, push ${fmt(last.push, 3)}`
        : "none yet\n";
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

// Step-through: the three RLHF steps as one diagram.
(function () {
  Learn.demo(document.getElementById("demo-steps"), {
    height: 330,
    label: "The three RLHF steps: fine-tune a model, train a reward model on people's picks, then train the policy against the reward model",
    steps: [
      { s: { p2: 0, p3: 0, upd: 0, leash: 0 }, cap: "Step 1. Take a pretrained model and fine-tune it on answers written by people. Call the result π0. It is already a decent chatbot." },
      { s: { p2: 1 }, cap: "Step 2. π0 writes a few answers to one prompt, and people say which they prefer. A second model, the reward model, learns to predict those picks." },
      { s: { p3: 1 }, cap: "Step 3. The policy starts as a copy of π0. It writes an answer, and the reward model scores it with one number. Say this answer scores 1.5." },
      { s: { upd: 1 }, cap: "RL (the PPO algorithm) updates the policy toward answers that score higher. No person is in this loop. The reward model stands in for them." },
      { s: { leash: 1 }, cap: "A penalty, the KL term, charges the policy for moving away from π0. It works like a leash, and a larger β makes the leash shorter." },
    ],
    draw(ctx, s, c, d) {
      const box = (x, y, w, h, str, color) => { d.rect(x, y, w, h, c.bg, color, 2); d.text(str, x + w / 2, y + h / 2, { c: color, a: "center", w: 700 }); };
      d.text("step 1", 20, 50, { c: c.muted, s: 12 });
      box(80, 30, 150, 40, "pretrained model", c.ink);
      d.arrow(230, 50, 290, 50, c.muted, 2);
      d.text("fine-tune", 259, 22, { c: c.muted, s: 12, a: "center" });
      box(290, 30, 70, 40, "π0", c.blue);
      if (s.p2 > 0.01) {
        ctx.globalAlpha = s.p2;
        d.arrow(360, 50, 420, 50, c.muted, 2);
        d.text("writes", 389, 22, { c: c.muted, s: 12, a: "center" });
        box(420, 32, 50, 36, "A", c.green);
        box(478, 32, 50, 36, "B", c.ink);
        d.arrow(474, 68, 474, 113, c.muted, 2);
        d.text("people pick A", 484, 92, { c: c.green, s: 12 });
        d.text("step 2", 630, 92, { c: c.muted, s: 12, a: "right" });
        box(399, 115, 150, 36, "reward model", c.orange);
      }
      if (s.leash > 0.01) {
        ctx.globalAlpha = s.leash;
        d.line(325, 70, 325, 230, c.purple, 2, [6, 5]);
        d.text("KL penalty (β)", 315, 150, { c: c.purple, s: 13, a: "right", w: 700 });
        d.text("stay near π0", 315, 170, { c: c.purple, s: 12, a: "right" });
      }
      if (s.p3 > 0.01) {
        ctx.globalAlpha = s.p3;
        d.text("step 3", 20, 250, { c: c.muted, s: 12 });
        box(250, 230, 150, 40, "policy πθ", c.blue);
        d.arrow(400, 250, 450, 250, c.muted, 2);
        box(450, 230, 90, 40, "answer y", c.ink);
        d.arrow(495, 230, 495, 153, c.muted, 2);
        d.text("score 1.5", 505, 192, { c: c.orange, w: 700 });
      }
      if (s.upd > 0.01) {
        ctx.globalAlpha = s.upd;
        d.line(549, 133, 606, 133, c.orange, 2);
        d.line(605, 133, 605, 300, c.orange, 2);
        d.line(606, 300, 325, 300, c.orange, 2);
        d.arrow(325, 300, 325, 272, c.orange, 2);
        d.text("PPO update: more like the high scorers", 465, 314, { c: c.orange, s: 12, a: "center" });
      }
      ctx.globalAlpha = 1;
    },
  });
})();

// Step-through: Bradley–Terry on the four example answers: gap, probability, loss and push.
(function () {
  const R = [["A", 1.5], ["B", 0.6], ["C", -0.4], ["D", -1.7]];
  const X = (v) => 240 + 72 * v, sig = (t) => 1 / (1 + Math.exp(-t));
  Learn.demo(document.getElementById("demo-bt"), {
    height: 290,
    label: "Four answers on a reward number line, and the Bradley–Terry probability for a pair of them",
    steps: [
      { s: { cmp: 0, other: 0.6, sh: 0, loss: 0, flip: 0 }, cap: "A reward model gives each answer one number. Here A (correct, with code) gets 1.5, B gets 0.6, C gets −0.4 and D (wrong) gets −1.7." },
      { s: { cmp: 1 }, cap: "Compare A and B. The gap is 1.5 − 0.6 = 0.9. Bradley–Terry puts the gap through the sigmoid: σ(0.9) = 0.71. So the model says people pick A 71% of the time." },
      { s: { other: -1.7 }, cap: "Compare A and D. The gap is 1.5 − (−1.7) = 3.2, and σ(3.2) = 0.96. A bigger gap means the model is surer of the pick." },
      { s: { sh: 10 }, cap: "Add 10 to every reward. The gap is still 3.2, so the probability is still 0.96. Only the gap matters, so a reward of 0 means nothing on its own." },
      { s: { sh: 0, other: 0.6, loss: 1 }, cap: "Back to A and B. People pick A. The model gave that pick 0.71, so the loss is −log 0.71 = 0.34. The push on the two rewards is what's left: 0.29." },
      { s: { flip: 1 }, cap: "Now say people pick B. The model gave that pick only 0.29, so the loss is −log 0.29 = 1.24 and the push is 0.71. A surprise moves the rewards more." },
    ],
    draw(ctx, s, c, d) {
      const name = Math.abs(s.other - 0.6) < Math.abs(s.other + 1.7) ? "B" : "D";
      const gap = 1.5 - s.other, p = sig(gap), pickB = s.flip > 0.5;
      d.text("reward", 60, 22, { c: c.muted, s: 12 });
      d.line(X(-2.5), 90, X(2.5), 90, c.muted, 1.5);
      for (let v = -2; v <= 2; v++) {
        d.line(X(v), 85, X(v), 95, c.muted, 1.5);
        d.text(d.fmt(v + s.sh, 0), X(v), 108, { c: c.muted, s: 12, a: "center" });
      }
      for (const [n, r] of R) {
        const on = s.cmp > 0.5 && (n === "A" || n === name);
        d.dot(X(r), 90, 7, on ? (n === "A" ? c.blue : c.orange) : c.ink);
        d.text(n, X(r), 68, { c: on ? (n === "A" ? c.blue : c.orange) : c.ink, a: "center", w: 700 });
        d.text(d.fmt(r + s.sh, 1), X(r), 48, { c: c.muted, s: 12, a: "center" });
      }
      if (s.cmp > 0.01) {
        ctx.globalAlpha = s.cmp;
        const x1 = X(s.other), x2 = X(1.5);
        d.line(x1, 126, x2, 126, c.ink, 2); d.line(x1, 120, x1, 132, c.ink, 2); d.line(x2, 120, x2, 132, c.ink, 2);
        d.text(`gap ${d.fmt(gap, 1)}`, (x1 + x2) / 2, 144, { c: c.ink, s: 13, a: "center", w: 700 });
        d.text(`pick A  ${d.fmt(p, 2)}`, 60, 186, { c: c.blue, s: 13, w: 700 });
        d.text(`pick ${name}  ${d.fmt(1 - p, 2)}`, 420, 186, { c: c.orange, s: 13, w: 700, a: "right" });
        d.rect(60, 200, 360 * p, 32, c.blue);
        d.rect(60 + 360 * p, 200, 360 * (1 - p), 32, c.orange);
        d.rect(60, 200, 360, 32, null, c.ink, 2);
        if (s.loss > 0.01) {
          ctx.globalAlpha = s.cmp * s.loss;
          d.text(pickB ? `people pick ${name}: a surprise` : "people pick A: as expected", 240, 256, { c: pickB ? c.red : c.ink, s: 13, a: "center", w: 700 });
        }
        ctx.globalAlpha = s.cmp;
        d.panel(c, 444, 30, 184, 226);
        d.text("reward gap", 458, 52, { c: c.muted, s: 13 });
        d.text(`${d.fmt(1.5 + s.sh, 1)} − ${s.other + s.sh < 0 ? "(" + d.fmt(s.other + s.sh, 1) + ")" : d.fmt(s.other + s.sh, 1)}`, 458, 74, { c: c.ink });
        d.text(`= ${d.fmt(gap, 1)}`, 458, 98, { c: c.ink, s: 18, w: 700 });
        d.text("σ(gap)", 458, 128, { c: c.muted, s: 13 });
        d.text(`= ${d.fmt(p, 2)}`, 458, 150, { c: c.blue, s: 18, w: 700 });
        if (s.loss > 0.01) {
          ctx.globalAlpha = s.cmp * s.loss;
          const q = pickB ? 1 - p : p;
          d.line(458, 172, 614, 172, c.muted, 1);
          d.text(`loss = −log ${d.fmt(q, 2)}`, 458, 192, { c: c.ink, s: 13 });
          d.text(`     = ${d.fmt(-Math.log(q), 2)}`, 458, 212, { c: c.ink, s: 13, w: 700 });
          d.text(`push = ${d.fmt(1 - q, 2)}`, 458, 238, { c: pickB ? c.red : c.orange, w: 700 });
        }
      }
      ctx.globalAlpha = 1;
    },
  });
})();

// Step-through: the best policy is π0 reweighted by exp(reward / β), then divided by Z.
(function () {
  const R = [["A", 1.5], ["B", 0.6], ["C", -0.4], ["D", -1.7]], P0 = 0.25, BASE = 250, SC = 180;
  Learn.demo(document.getElementById("demo-dpo"), {
    height: 320,
    label: "Bars for four answers: π0's probabilities next to the best policy's, for different values of β",
    steps: [
      { s: { w: 0, z: 0, beta: 1, back: 0 }, cap: "Pretend a prompt has only four possible answers and π0 gives each one 25%. The rewards are the same as before: 1.5, 0.6, −0.4 and −1.7." },
      { s: { w: 1 }, cap: "Multiply each 25% by e to the power of reward ÷ β. With β = 1, A's multiplier is e^1.5 = 4.48 and D's is e^−1.7 = 0.18. High rewards grow, low rewards shrink." },
      { s: { z: 1 }, cap: "The four products add up to 1.79, not 1. Divide each by that sum, called Z, and they are probabilities again. A now gets 63% and D gets 3%." },
      { s: { beta: 0.5 }, cap: "Lower β to 0.5. Leaving π0 now costs less, so the best policy leans harder on the reward. A gets 84% and D gets 0.1%." },
      { s: { beta: 4 }, cap: "Raise β to 4. Leaving π0 now costs a lot, so the best policy stays close to 25% each. A gets 35% and D gets 16%." },
      { s: { beta: 1, back: 1 }, cap: "DPO runs this backward, here with β = 1. A went from 25% to 62.6%, and log(0.626 ÷ 0.25) = 0.92. B went to 25.5%, and log(0.255 ÷ 0.25) = 0.02. The difference is 0.9, the reward gap between A and B. Z dropped out." },
    ],
    draw(ctx, s, c, d) {
      const prod = R.map(([, r]) => P0 * Math.exp(r / s.beta));
      const Z = prod.reduce((a, b) => a + b, 0);
      const prob = prod.map((v) => v / Z);
      const pct = (v) => (v < 0.01 ? d.fmt(v * 100, 1) : d.fmt(v * 100, 0)) + "%";
      d.line(40, BASE, 480, BASE, c.muted, 1.5);
      R.forEach(([n, r], i) => {
        const x0 = 40 + i * 112;
        d.rect(x0 + 10, BASE - P0 * SC, 36, P0 * SC, c.bg, c.blue, 2);
        d.text("25%", x0 + 28, BASE - P0 * SC - 12, { c: c.blue, s: 12, a: "center" });
        if (s.w > 0.01) {
          const v = P0 + (prod[i] - P0) * s.w + (prob[i] - prod[i]) * s.z, h = v * SC;
          ctx.globalAlpha = s.w;
          d.rect(x0 + 52, BASE - h, 36, h, c.orange);
          d.text(s.z < 0.5 ? "×" + d.fmt(Math.exp(r / s.beta), 2) : pct(prob[i]), x0 + 70, BASE - h - 12, { c: c.orange, s: 12, a: "center", w: 700 });
          ctx.globalAlpha = 1;
        }
        d.text(n, x0 + 49, 268, { c: c.ink, a: "center", w: 700 });
        d.text(`r = ${d.fmt(r, 1)}`, x0 + 49, 286, { c: c.muted, s: 12, a: "center" });
      });
      d.panel(c, 496, 30, 132, 150);
      d.text(`β = ${d.fmt(s.beta, 1)}`, 510, 54, { c: c.ink, s: 16, w: 700 });
      ctx.globalAlpha = s.z;
      d.text(`Z = ${d.fmt(Z, 2)}`, 510, 82, { c: c.ink });
      ctx.globalAlpha = 1;
      d.rect(510, 112, 12, 12, c.bg, c.blue, 2);
      d.text("π0", 530, 118, { c: c.blue, s: 12 });
      d.rect(510, 142, 12, 12, c.orange);
      d.text("best policy", 530, 148, { c: c.orange, s: 12 });
      if (s.back > 0.01) {
        ctx.globalAlpha = s.back;
        const la = Math.log(prob[0] / P0), lb = Math.log(prob[1] / P0);
        d.text(`β log(policy ÷ π0):  A ${d.fmt(la, 2)},  B ${d.fmt(lb, 2)},  gap ${d.fmt(s.beta * (la - lb), 2)}`, 260, 308, { c: c.ink, s: 12, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();
