// Unit 15: REINFORCE on a three-armed bandit, with the baseline switched on or off.
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
      ? `pulled <span class="${colors[last.a]}">${names[last.a]}</span>: reward ${fmt(last.r)}\n` +
        (useBaseline ? `baseline ${fmt(last.used)}, ` : "") +
        `advantage ${fmt(last.advantage)}`
      : "no pulls yet\n ";
    out.innerHTML =
      `run ${run}, pull ${pull}/${runLength}, baseline ${useBaseline ? "on" : "off"}\n` +
      `${lastLine}\n` +
      `θ = (${theta.map((v) => fmt(v)).join(", ")})\n` +
      `${probs}\n` +
      `finished runs: ${finished}\n` +
      `top arm at the end: A ${tally[0]}, B ${tally[1]}, C ${tally[2]}`;
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

// Step-through: the agent-environment loop, on one pull of the bandit.
(function () {
  Learn.demo(document.getElementById("demo-rl-loop"), {
    height: 300,
    label: "An agent and an environment trading a state, an action and a reward",
    steps: [
      { s: { st: 0, act: 0, rew: 0, G: 0, chess: 0 }, cap: "Two parts. The agent learns and makes choices. The environment is everything else. Here the environment is a slot machine with three arms." },
      { s: { st: 1 }, cap: "The environment shows the agent a state. The bandit has only one state: three arms waiting to be pulled." },
      { s: { act: 1 }, cap: "The agent picks an action by sampling from its policy. Each arm has probability 1/3, and this time the draw comes up B." },
      { s: { rew: 1 }, cap: "The environment answers with a reward, a single number. Arm B pays 5 on average plus noise, and this pull pays 5.2." },
      { s: { G: 1 }, cap: "For the bandit, one pull is a whole episode. The return is the total reward in the episode, so G = 5.2." },
      { s: { chess: 1 }, cap: "In chess an episode is a whole game. The reward can be 0 after every move and +1 for a win at the end, so G = 1. The agent has to work out which moves earned it." },
    ],
    draw(ctx, s, c, d) {
      d.rect(40, 80, 170, 130, c.soft, c.blue, 2);
      d.text("agent", 125, 112, { c: c.blue, s: 17, a: "center", w: 700 });
      d.text("policy π", 125, 140, { c: c.ink, a: "center" });
      d.text("A, B, C: 1/3 each", 125, 172, { c: c.muted, s: 12, a: "center" });
      d.rect(430, 80, 170, 130, c.soft, c.ink, 2);
      d.text("environment", 515, 112, { c: c.ink, s: 17, a: "center", w: 700 });
      d.text("slot machine", 515, 140, { c: c.ink, a: "center" });
      d.text("pays A~4 B~5 C~6", 515, 172, { c: c.muted, s: 12, a: "center" });
      const link = (alpha, y, toAgent, label, color) => {
        if (alpha < 0.02) return;
        ctx.globalAlpha = alpha;
        if (toAgent) d.arrow(424, y, 216, y, color, 2.5); else d.arrow(216, y, 424, y, color, 2.5);
        d.text(label, 320, y - 13, { c: color, s: 13, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      };
      link(s.st, 108, true, "state: three arms", c.muted);
      link(s.act, 150, false, "action: pull B", c.blue);
      link(s.rew, 192, true, "reward: 5.2", c.orange);
      if (s.G > 0.02) {
        ctx.globalAlpha = s.G;
        d.text("bandit episode = 1 pull     return G = 5.2", 320, 244, { c: c.ink, a: "center", w: 700 });
      }
      if (s.chess > 0.02) {
        ctx.globalAlpha = s.chess;
        d.text("chess episode = 1 game      G = 0 + 0 + … + 0 + 1 = 1", 320, 274, { c: c.muted, a: "center" });
      }
      ctx.globalAlpha = 1;
    },
  });
})();

// Step-through: one REINFORCE step with no baseline. Pull B, get 5.2, move the logits.
(function () {
  const names = ["A", "B", "C"], means = [4, 5, 6], barX = [30, 160, 290], BW = 56, BASE = 250, H = 380;
  const softmax = (z) => { const e = z.map(Math.exp), t = e.reduce((a, b) => a + b, 0); return e.map((v) => v / t); };
  const eta = 0.1, r = 5.2, pulled = 1;
  const p0 = softmax([0, 0, 0]);
  const delta = p0.map((p, i) => eta * r * ((i === pulled ? 1 : 0) - p)); // (-0.173, 0.347, -0.173)
  const p1 = softmax(delta);                                               // (0.272, 0.457, 0.272)
  Learn.demo(document.getElementById("demo-reinforce"), {
    height: 340,
    label: "Three bars for the policy's probabilities of arms A, B and C, before and after one REINFORCE step",
    steps: [
      { s: { pull: 0, push: 0, upd: 0, hl: 0 }, cap: "The policy starts with all three logits at 0, so each arm has probability 0.33. The agent doesn't know that the arms pay 4, 5 and 6 on average." },
      { s: { pull: 1 }, cap: "Sample an arm from the policy. The draw comes up B, and the machine pays r = 5.2." },
      { s: { push: 1 }, cap: "REINFORCE moves each logit by η × r × (one-hot − probability), with η = 0.1. B's logit goes up by 0.1 × 5.2 × 2/3 ≈ 0.35. A's and C's each go down by 0.1 × 5.2 × 1/3 ≈ 0.17." },
      { s: { upd: 1 }, cap: "Take the softmax of the new logits (−0.17, 0.35, −0.17). The probabilities become 0.27, 0.46 and 0.27. The policy will now pull B more often." },
      { s: { hl: 1 }, cap: "B isn't the best arm, though. C is. Every arm pays around 5, so every pull pushes its own arm up. One step like this is mostly noise." },
    ],
    draw(ctx, s, c, d) {
      d.line(16, BASE, 404, BASE, c.muted, 1.5);
      names.forEach((name, i) => {
        const p = p0[i] + s.upd * (p1[i] - p0[i]), top = BASE - H * p, x = barX[i];
        const active = i === pulled && s.pull > 0.5;
        d.rect(x, top, BW, H * p, active ? c.orange : c.blue);
        d.text(d.fmt(p, 2), x + BW / 2, top - 13, { c: c.ink, a: "center", w: 700 });
        d.text(name, x + BW / 2, BASE + 18, { c: c.ink, a: "center", w: 700 });
        d.text(`pays ~${means[i]}`, x + BW / 2, BASE + 40, { c: c.muted, s: 12, a: "center" });
        d.text(`θ = ${d.fmt(s.upd * delta[i], 2)}`, x + BW / 2, BASE + 64, { c: c.muted, s: 13, a: "center" });
        const arrow = s.push * (1 - s.upd);
        if (arrow > 0.02) {
          ctx.globalAlpha = arrow;
          const y0 = BASE - H * p0[i], len = delta[i] * 150, color = delta[i] > 0 ? c.orange : c.muted;
          d.arrow(x + BW + 12, y0, x + BW + 12, y0 - len, color, 3);
          d.text((delta[i] > 0 ? "+" : "") + d.fmt(delta[i], 2), x + BW + 22, y0 - len / 2, { c: color, s: 13, w: 700 });
          ctx.globalAlpha = 1;
        }
      });
      if (s.hl > 0.02) {
        ctx.globalAlpha = s.hl;
        d.rect(barX[2] - 4, BASE - H * p1[2] - 4, BW + 8, H * p1[2] + 4, null, c.green, 2);
        d.text("best arm", barX[2] + BW / 2, BASE - H * p1[2] - 44, { c: c.green, s: 13, a: "center", w: 700 });
        ctx.globalAlpha = 1;
      }
      if (s.pull > 0.02) {
        ctx.globalAlpha = s.pull;
        d.panel(c, 420, 40, 208, 236);
        d.text("pulled: B", 434, 64, { c: c.orange, w: 700 });
        d.text("reward r = 5.2", 434, 90, { c: c.ink });
        if (s.push > 0.02) {
          ctx.globalAlpha = s.pull * s.push;
          d.text("change in each logit", 434, 126, { c: c.muted, s: 13 });
          d.text(`B: +${d.fmt(delta[1], 2)}`, 434, 150, { c: c.orange, w: 700 });
          d.text(`A, C: ${d.fmt(delta[0], 2)}`, 434, 174, { c: c.ink });
        }
        if (s.upd > 0.02) {
          ctx.globalAlpha = s.pull * s.upd;
          d.text("new probabilities", 434, 212, { c: c.muted, s: 13 });
          d.text(p1.map((p) => d.fmt(p, 2)).join("  "), 434, 238, { c: c.blue, w: 700 });
        }
        ctx.globalAlpha = 1;
      }
    },
  });
})();

// Step-through: the same update with a baseline of 5. Three pulls, each scaled by r - b.
(function () {
  const names = ["A", "B", "C"], means = [4, 5, 6], barX = [30, 160, 290], BW = 56, BASE = 262, H = 500;
  const softmax = (z) => { const e = z.map(Math.exp), t = e.reduce((a, b) => a + b, 0); return e.map((v) => v / t); };
  const eta = 0.1, b = 5;
  const pulls = [{ arm: 1, r: 5.2 }, { arm: 0, r: 4.1 }, { arm: 2, r: 6 }];
  // probs[t] is the policy after t pulls; each pull moves the logits by eta * (r - b) * (one-hot - p).
  let theta = [0, 0, 0];
  const probs = [softmax(theta)];
  for (const pull of pulls) {
    const p = softmax(theta);
    pull.adv = pull.r - b;
    pull.delta = eta * pull.adv * (1 - p[pull.arm]);
    theta = theta.map((v, i) => v + eta * pull.adv * ((i === pull.arm ? 1 : 0) - p[i]));
    probs.push(softmax(theta));
  }
  const noBaseline = softmax(probs[0].map((p, i) => eta * 5.2 * ((i === 1 ? 1 : 0) - p)))[1]; // 0.457
  const signed = (v, digits) => (v > 0 ? "+" : "") + v.toFixed(digits).replace("-", "−");
  Learn.demo(document.getElementById("demo-baseline"), {
    height: 330,
    label: "Three bars for the policy's probabilities over three pulls, each update scaled by the reward minus a baseline of 5",
    steps: [
      { s: { t: 0, pull: -1, info: 0, cmp: 0 }, cap: "Same start: each arm at 0.33. This time the agent keeps a baseline b = 5, about the average reward. Each update uses r − b, the advantage." },
      { s: { t: 1, pull: 0, info: 1 }, cap: "Pull B and get 5.2 again. The advantage is 5.2 − 5 = 0.2, so B's logit rises by only 0.1 × 0.2 × 2/3 ≈ 0.013. The probabilities barely move: 0.331, 0.338, 0.331." },
      { s: { t: 2, pull: 1 }, cap: "Pull A and get 4.1. The advantage is 4.1 − 5 = −0.9. A negative advantage pushes the pulled arm down, and A drops to 0.311." },
      { s: { t: 3, pull: 2 }, cap: "Pull C and get 6. The advantage is 6 − 5 = 1, so C rises to 0.363. After three pulls the arms rank C, B, A, the same order as their real payouts." },
      { s: { cmp: 1 }, cap: "Without the baseline, that first pull alone took B from 0.33 to 0.46 (dashed outline). With it, an arm only goes up when it beats the usual reward." },
    ],
    draw(ctx, s, c, d) {
      const lo = Math.min(2, Math.floor(s.t)), f = s.t - lo;
      const cur = Math.round(s.pull), pull = cur >= 0 ? pulls[cur] : null;
      d.line(16, BASE, 404, BASE, c.muted, 1.5);
      names.forEach((name, i) => {
        const p = probs[lo][i] + f * (probs[lo + 1][i] - probs[lo][i]), top = BASE - H * p, x = barX[i];
        const active = pull && pull.arm === i;
        d.rect(x, top, BW, H * p, active ? c.orange : c.blue);
        d.text(d.fmt(p, 3), x + BW / 2, top - 13, { c: c.ink, a: "center", w: 700 });
        d.text(name, x + BW / 2, BASE + 18, { c: c.ink, a: "center", w: 700 });
        d.text(`pays ~${means[i]}`, x + BW / 2, BASE + 40, { c: c.muted, s: 12, a: "center" });
        if (active && s.info > 0.02) {
          ctx.globalAlpha = s.info;
          const color = pull.adv < 0 ? c.red : c.orange, y0 = BASE - H / 3;
          d.arrow(x + BW + 12, y0, x + BW + 12, y0 - pull.adv * 40, color, 3);
          d.text(signed(pull.adv, 1), x + BW + 22, y0 - pull.adv * 20, { c: color, s: 13, w: 700 });
          ctx.globalAlpha = 1;
        }
      });
      if (s.cmp > 0.02) {
        ctx.globalAlpha = s.cmp;
        ctx.strokeStyle = c.muted; ctx.lineWidth = 2; ctx.setLineDash([6, 5]);
        ctx.strokeRect(barX[1], BASE - H * noBaseline, BW, H * noBaseline);
        ctx.setLineDash([]);
        d.text(`no baseline: ${d.fmt(noBaseline, 2)}`, barX[1] + BW / 2, BASE - H * noBaseline - 14, { c: c.muted, s: 12, a: "center" });
        ctx.globalAlpha = 1;
      }
      d.panel(c, 420, 40, 208, 236);
      d.text("baseline b = 5", 434, 64, { c: c.ink, w: 700 });
      if (pull && s.info > 0.02) {
        ctx.globalAlpha = s.info;
        d.text(`pulled: ${names[pull.arm]}`, 434, 100, { c: c.orange, w: 700 });
        d.text(`reward r = ${d.fmt(pull.r, 1)}`, 434, 126, { c: c.ink });
        d.text("advantage r − b", 434, 162, { c: c.muted, s: 13 });
        d.text(`= ${signed(pull.adv, 1)}`, 434, 190, { c: pull.adv < 0 ? c.red : c.orange, s: 22, w: 700 });
        d.text(`logit of ${names[pull.arm]}: ${signed(pull.delta, 3)}`, 434, 232, { c: c.ink, s: 13 });
        d.text(pull.adv < 0 ? "pushed down" : "pushed up", 434, 254, { c: c.muted, s: 13 });
        ctx.globalAlpha = 1;
      }
    },
  });
})();
