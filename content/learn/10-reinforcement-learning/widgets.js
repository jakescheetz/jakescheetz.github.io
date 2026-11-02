// Unit 10: REINFORCE on a three-armed bandit, with the baseline switched on or off.
(function () {
  const figure = document.getElementById("bandit");
  if (!figure) return;
  const { svg, stage, controls, slider, buttons, button, readout, loop, fmt } = Learn;

  const width = 720, rows = 14;
  const means = [4, 5, 6]; // average reward of arms A, B, C (each pull adds N(0, 1) noise)
  const names = ["A", "B", "C"];
  const colors = ["blue", "orange", "green"];
  const runLength = 300;

  // Left: one bar per arm, height = the policy's probability of pulling it.
  const barBottom = 200, barTop = 40, barWidth = 50;
  const barX = [60, 150, 240];
  const BY = (p) => barBottom - (barBottom - barTop) * p;
  // Right: each arm's probability over the 300 pulls of the current run.
  const cx0 = 390, cx1 = 700;
  const CX = (i) => cx0 + (cx1 - cx0) * i / runLength;

  const root = stage(figure, { rows, width });
  root.setAttribute("aria-label", "Policy probabilities for three slot-machine arms, and how they change over one training run");

  svg("line", { x1: 15, y1: barBottom, x2: 285, y2: barBottom, class: "muted thin" }, root);
  const bars = barX.map((x, i) => {
    const group = svg("g", { class: colors[i] }, root);
    const rect = svg("rect", { x: x - barWidth / 2, width: barWidth, fill: "currentColor" }, group);
    svg("text", { x, y: 224, "text-anchor": "middle", text: names[i] }, group);
    svg("text", { x, y: 252, "text-anchor": "middle", class: "muted", text: `~${means[i]}` }, root);
    return rect;
  });
  // Outline on the arm just pulled, and its advantage above it.
  const pulled = svg("rect", { width: barWidth + 8, fill: "none", stroke: "currentColor", "stroke-width": 2, class: "ink" }, root);
  const pulledText = svg("text", { "text-anchor": "middle", class: "ink" }, root);

  svg("line", { x1: cx0, y1: barTop, x2: cx0, y2: barBottom, class: "muted thin" }, root);
  svg("line", { x1: cx0, y1: barBottom, x2: cx1, y2: barBottom, class: "muted thin" }, root);
  svg("line", { x1: cx0, y1: barTop, x2: cx1, y2: barTop, class: "muted faint dashed thin" }, root);
  svg("text", { x: cx0 - 8, y: barTop + 5, "text-anchor": "end", class: "muted", text: "1" }, root);
  svg("text", { x: cx0 - 8, y: barBottom + 5, "text-anchor": "end", class: "muted", text: "0" }, root);
  svg("text", { x: (cx0 + cx1) / 2, y: 224, "text-anchor": "middle", class: "muted", text: "pulls" }, root);
  svg("text", { x: cx1, y: 224, "text-anchor": "end", class: "muted", text: "300" }, root);
  const traces = colors.map((c) => svg("polyline", { class: c }, root));

  const out = readout(figure);
  const ctl = controls(figure);
  const btns = buttons(figure);

  const softmax = (z) => {
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e[0] + e[1] + e[2];
    return e.map((v) => v / s);
  };
  const gauss = () => {
    const u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const sample = (p) => {
    const u = Math.random();
    return u < p[0] ? 0 : u < p[0] + p[1] ? 1 : 2;
  };

  let eta = 0.1, speed = 20, useBaseline = false;
  let theta, b, pull, last, history, hold = 0, acc = 0;
  let run = 0, finished = 0, tally = [0, 0, 0];

  function newRun() {
    theta = [0, 0, 0];
    b = null;
    pull = 0;
    last = null;
    history = [softmax(theta)];
    hold = 0;
    acc = 0;
    run++;
  }

  function resetAll() {
    run = 0; finished = 0; tally = [0, 0, 0];
    newRun();
  }

  // One REINFORCE step: sample an arm, get its reward, theta += eta * advantage * (onehot(a) - p).
  function step() {
    const p = softmax(theta);
    const a = sample(p);
    const r = means[a] + gauss();
    if (b === null) b = r;
    const used = useBaseline ? b : 0;
    const advantage = r - used;
    for (let k = 0; k < 3; k++) theta[k] += eta * advantage * ((k === a ? 1 : 0) - p[k]);
    b += 0.1 * (r - b); // running average of rewards
    pull++;
    const q = softmax(theta);
    history.push(q);
    last = { a, r, used, advantage, age: 0 };
    if (pull === runLength) {
      finished++;
      tally[q.indexOf(Math.max(...q))]++;
      hold = 1.5; // show the finished run briefly before starting the next one
    }
  }

  function draw() {
    const p = softmax(theta);
    bars.forEach((rect, i) => {
      rect.setAttribute("y", BY(p[i]));
      rect.setAttribute("height", barBottom - BY(p[i]));
    });
    traces.forEach((line, i) => {
      line.setAttribute("points", history.map((q, t) => `${CX(t).toFixed(1)},${BY(q[i]).toFixed(1)}`).join(" "));
    });

    if (last) {
      const x = barX[last.a];
      const opacity = Math.max(0.3, 1 - last.age);
      pulled.setAttribute("x", x - barWidth / 2 - 4);
      pulled.setAttribute("y", BY(p[last.a]) - 4);
      pulled.setAttribute("height", barBottom - BY(p[last.a]) + 4);
      pulled.style.opacity = opacity;
      pulledText.setAttribute("x", x);
      pulledText.setAttribute("y", BY(p[last.a]) - 12);
      pulledText.textContent = (last.advantage >= 0 ? "+" : "") + fmt(last.advantage, 1);
      pulledText.style.opacity = opacity;
    }
    pulled.style.display = pulledText.style.display = last ? "" : "none";

    const probs = p.map((v, i) => `<span class="${colors[i]}">π(${names[i]}) ${fmt(v)}</span>`).join("  ");
    const lastLine = last
      ? `pulled <span class="${colors[last.a]}">${names[last.a]}</span>: reward ${fmt(last.r)}, ` +
        (useBaseline ? `baseline ${fmt(last.used)}, ` : "") +
        `advantage ${fmt(last.advantage)}`
      : "no pulls yet";
    out.innerHTML =
      `run ${run}, pull ${pull} of ${runLength}, baseline ${useBaseline ? "on" : "off"}\n` +
      `${lastLine}\n` +
      `θ = (${theta.map((v) => fmt(v)).join(", ")})\n` +
      `${probs}\n` +
      `finished runs ${finished}, most likely arm at the end: A ${tally[0]}, B ${tally[1]}, C ${tally[2]}`;
  }

  slider(ctl, { label: "learn rate η", min: 0.02, max: 0.4, step: 0.02, value: eta, format: (v) => v.toFixed(2) }, (v) => {
    eta = v;
  });
  slider(ctl, { label: "pulls/sec", min: 1, max: 60, step: 1, value: speed, format: (v) => String(v) }, (v) => {
    speed = v;
  });

  resetAll();

  const player = loop(figure, (dt) => {
    if (dt) {
      if (last) last.age += dt;
      if (hold > 0) {
        hold -= dt;
        if (hold <= 0) newRun();
      } else {
        acc += dt * speed;
        while (acc >= 1 && pull < runLength) { step(); acc -= 1; }
      }
    }
    draw();
  });

  const toggle = button(btns, player.playing ? "Pause" : "Play", () => {
    if (player.playing) player.pause(); else player.play();
    toggle.textContent = player.playing ? "Pause" : "Play";
  });
  button(btns, "Pull once", () => {
    player.pause();
    toggle.textContent = "Play";
    if (pull >= runLength) newRun();
    step();
    draw();
  });
  const baselineButton = button(btns, "Baseline: off", () => {
    useBaseline = !useBaseline;
    baselineButton.textContent = `Baseline: ${useBaseline ? "on" : "off"}`;
    resetAll();
    draw();
  });
  button(btns, "Reset", () => {
    resetAll();
    draw();
  });
})();
