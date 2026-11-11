// Step-through animations for this post, drawn with Learn.demo (assets/learn.js).
// Colors match the charts: blue is the agent trained on the score, orange is real progress or the thing being
// pointed at. Every number in a caption is from the post; the first two demos are sketches and say so.

// 1. The textbook reward-hacking curve: what I went looking for. A sketch, not data.
(function () {
  const X = (t) => 70 + t * 500, Y = (v) => 270 - v * 220;
  const reward = (t) => 0.95 * (1 - Math.exp(-2.4 * t));
  const goal = (t) => reward(t) - 1.1 * Math.pow(Math.max(0, t - 0.3), 1.6);
  Learn.demo(document.getElementById("demo-curve"), {
    height: 320,
    label: "A sketch of the textbook reward-hacking curve: the reward keeps rising while the real goal peaks and falls",
    steps: [
      { s: { p: 0.42, gap: 0 }, cap: "Train an agent on a reward that stands in for what you want. Early on, the reward (blue) and the real goal (orange) rise together." },
      { s: { p: 1 }, cap: "Keep optimizing. The reward keeps rising, but the real goal peaks and then falls. This is the textbook reward-hacking curve." },
      { s: { gap: 1 }, cap: "The gap between the lines is the part of the reward that no longer tracks what you wanted. This is a sketch of the usual picture, not my data. It's the shape I went looking for." },
    ],
    draw(ctx, s, c, d) {
      d.line(X(0), Y(0), X(1) + 20, Y(0), c.muted, 1.5);
      d.line(X(0), Y(0), X(0), 36, c.muted, 1.5);
      d.text("how hard you optimize →", X(1) + 20, Y(0) + 20, { c: c.muted, s: 12, a: "right" });
      d.text("a sketch, not data", X(0) + 12, 44, { c: c.muted, s: 12 });
      const curve = (fn, color) => {
        ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath();
        for (let t = 0; t <= s.p + 1e-9; t += 0.01) { if (t === 0) ctx.moveTo(X(t), Y(fn(t))); else ctx.lineTo(X(t), Y(fn(t))); }
        ctx.stroke();
        d.rect(X(s.p) - 5, Y(fn(s.p)) - 5, 10, 10, color);
      };
      curve(goal, c.orange);
      curve(reward, c.blue);
      const apart = s.p > 0.7;
      d.text("the reward", X(s.p) - 12, Y(reward(s.p)) - 16, { c: c.blue, w: 700, a: "right" });
      d.text("the real goal", X(s.p) - 12, Y(goal(s.p)) + (apart ? 20 : 34), { c: c.orange, w: 700, a: "right" });
      if (s.gap > 0.01) {
        ctx.globalAlpha = s.gap;
        const x = X(1) + 22;
        d.line(x, Y(reward(1)), x, Y(goal(1)), c.ink, 2);
        d.line(x - 6, Y(reward(1)), x + 6, Y(reward(1)), c.ink, 2);
        d.line(x - 6, Y(goal(1)), x + 6, Y(goal(1)), c.ink, 2);
        d.text("the gap", x - 10, (Y(reward(1)) + Y(goal(1))) / 2, { c: c.ink, w: 700, a: "right" });
      }
    },
  });
})();

// 2. The score and the hidden scoreboard, on a toy map: 10 squares of route and 3 in a side building.
(function () {
  const route = (i) => ({ x: 40 + 40 * i, y: 150, id: `r${i}` });
  const side = (k) => ({ x: 240, y: 150 + 40 * k, id: `s${k}` });
  const cells = [...Array(10).keys()].map(route).concat([1, 2, 3].map(side));
  const path = [0, 1, 2, 3, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5].map(route).concat([1, 2, 3].map(side));
  const REACHED = 13; // index in `path` where the agent first stands on the milestone square
  Learn.demo(document.getElementById("demo-score"), {
    height: 330,
    label: "A toy map: the agent earns score for each new square, and a hidden scoreboard counts milestones",
    steps: [
      { s: { pos: 0 }, cap: "A toy version of the setup. The agent (blue) starts on the left. The route runs right to the first milestone, and a side building hangs off it." },
      { s: { pos: 4 }, cap: "The agent walks onto 4 new squares. Each new square pays 0.1, so its score is 0.4." },
      { s: { pos: 6 }, cap: "It walks back over 2 of them. Old squares pay nothing, so the score stays at 0.4." },
      { s: { pos: 13 }, cap: "It heads right and reaches the milestone. That's 9 new squares so far, so the score is 0.9. The scoreboard ticks to 1 of 3. The agent never sees the scoreboard." },
      { s: { pos: 20 }, cap: "Now it walks back and into the side building: 3 more new squares, and the score rises to 1.2. The scoreboard doesn't move. That gap is where a score can be gamed." },
    ],
    draw(ctx, s, c, d) {
      const at = Math.floor(s.pos + 1e-6), frac = Math.min(1, s.pos - at);
      const seen = new Set(path.slice(1, at + 1).map((cell) => cell.id));
      seen.delete("r0");
      d.text("a toy map: 10 squares of route, 3 in a side building", 40, 40, { c: c.muted, s: 12 });
      for (const cell of cells) {
        if (seen.has(cell.id)) { ctx.globalAlpha = 0.3; d.rect(cell.x, cell.y, 36, 36, c.blue); ctx.globalAlpha = 1; }
        d.rect(cell.x, cell.y, 36, 36, null, cell.id[0] === "s" ? c.muted : c.ink, 2);
      }
      d.text("start", 58, 204, { c: c.muted, s: 12, a: "center" });
      d.text("side building", 288, 232, { c: c.muted, s: 12 });
      d.text("(off-route)", 288, 250, { c: c.muted, s: 12 });
      // milestone flag on the last route square
      d.rect(416, 100, 3, 50, c.ink);
      d.rect(419, 100, 24, 16, c.orange);
      d.text("1", 431, 108, { c: c.bg, s: 12, w: 800, a: "center" });
      d.text("milestone 1", 450, 108, { c: c.ink, s: 12, w: 700 });
      // the agent, sliding between squares
      const a = path[at], b = path[Math.min(path.length - 1, at + 1)];
      d.rect(a.x + (b.x - a.x) * frac + 8, a.y + (b.y - a.y) * frac + 8, 20, 20, c.blue);
      // readout
      d.panel(c, 410, 196, 216, 116);
      d.text("the score (what it's paid)", 422, 216, { c: c.muted, s: 12 });
      d.text(d.fmt(seen.size * 0.1, 1), 422, 242, { c: c.blue, s: 22, w: 700 });
      d.text(`${seen.size} new squares × 0.1`, 478, 244, { c: c.muted, s: 12 });
      d.text("the scoreboard (hidden)", 422, 270, { c: c.muted, s: 12 });
      d.text(`${s.pos >= REACHED - 0.5 ? 1 : 0} of 3 milestones`, 422, 294, { c: c.orange, s: 16, w: 700 });
    },
  });
})();

// 3. What the cap did, in order. The 40-square map is an example; the percentages are measured.
(function () {
  Learn.demo(document.getElementById("demo-cap"), {
    height: 340,
    label: "The patch: only the first 20 new squares on a map pay, which removes most reward of every kind",
    steps: [
      { s: { cap: 0, off: 0, on: 0, lvl: 0, res: 0 }, cap: "Before the patch, every new square on a map pays 0.1. On this example map the agent finds 40 new squares and earns 4.0 points." },
      { s: { cap: 1 }, cap: "The patch: only the first 20 new squares on each map pay. The same walk now earns 2.0." },
      { s: { off: 1 }, cap: "On the agent's real behavior, the cap removed 85.7% of the reward it earned off-route. That was the goal." },
      { s: { on: 1 }, cap: "It also removed 82.4% of the reward for everything else. The cap can't tell on-route squares from off-route ones. Total reward fell to 0.17 of what it was." },
      { s: { lvl: 1 }, cap: "The level reward didn't change: still 1 point per level. By my offline estimate, that made a level about six times more valuable relative to exploring." },
      { s: { res: 1 }, cap: "After retraining on the patched score, real progress at the end of training fell from 1.43 milestones to 0.33. Grinding went from none to 22% of the patched reward." },
    ],
    draw(ctx, s, c, d) {
      d.text("one map, 40 new squares (an example)", 30, 26, { c: c.muted, s: 12 });
      for (let i = 0; i < 40; i++) {
        const x = 30 + (i % 10) * 20, y = 44 + Math.floor(i / 10) * 20;
        ctx.globalAlpha = i < 20 ? 1 : 1 - s.cap;
        d.rect(x, y, 16, 16, c.blue);
        ctx.globalAlpha = i < 20 ? 0 : s.cap;
        d.rect(x + 1, y + 1, 14, 14, null, c.muted, 1.5);
        ctx.globalAlpha = 1;
      }
      ctx.globalAlpha = 1 - s.cap;
      d.text("every new square pays 0.1", 250, 58, { c: c.ink, s: 13 });
      ctx.globalAlpha = s.cap;
      d.text("only the first 20 pay", 250, 58, { c: c.red, s: 13, w: 700 });
      ctx.globalAlpha = 1;
      d.text("this walk earns", 250, 84, { c: c.muted, s: 12 });
      d.text(`${(4 - 2 * s.cap).toFixed(1)} points`, 250, 110, { c: c.blue, s: 22, w: 700 });
      const bar = (y, name, share, reveal, color) => {
        d.text(name, 30, y + 8, { c: c.ink, s: 13 });
        d.rect(260, y, 300, 16, c.soft);
        if (reveal > 0.01) {
          d.rect(260, y, 300 * share * reveal, 16, color);
          ctx.globalAlpha = reveal;
          d.text(`${d.fmt(share * 100 * reveal, 1)}%`, 570, y + 8, { c: c.ink, s: 13, w: 700 });
          ctx.globalAlpha = 1;
        }
      };
      bar(152, "off-route reward removed", 0.857, s.off, c.orange);
      bar(184, "everything else removed", 0.824, s.on, c.blue);
      const show = (alpha, fn) => { if (alpha > 0.01) { ctx.globalAlpha = alpha; fn(); ctx.globalAlpha = 1; } };
      show(s.on, () => d.text("total reward on the same behavior: 0.17 of before", 30, 220, { c: c.muted, s: 12 }));
      show(s.lvl, () => {
        d.rect(30, 240, 12, 12, c.purple);
        d.text("1 level = 1 point, unchanged: about 6× more valuable next to exploring", 50, 246, { c: c.ink, s: 12 });
      });
      show(s.res, () => {
        d.panel(c, 30, 272, 580, 50);
        d.text("real progress  1.43 → 0.33 milestones", 44, 297, { c: c.ink, s: 13, w: 700 });
        d.text("grinding  0% → 22% of reward", 372, 297, { c: c.ink, s: 13, w: 700 });
      });
    },
  });
})();
