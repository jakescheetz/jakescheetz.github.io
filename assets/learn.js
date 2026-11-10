// Helpers for /learn widgets, plus unit progress saved in this browser (localStorage).
// Widgets live in each unit's own script (front matter `scripts:`) and build on window.Learn.
(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const DONE_KEY = "learn:done:";
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // One grid row in SVG user units: stages use 1 unit = 1px at the desktop font size.
  const ROW = 19.2;

  function el(tag, attrs, parent) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  // Insert before the figcaption so captions stay last.
  function place(figure, node) {
    figure.insertBefore(node, figure.querySelector("figcaption"));
    return node;
  }

  // An SVG drawing area `width` × `rows` text lines in user units. It always fills the panel's
  // width; fitStage() sets its height to the matching aspect ratio, rounded up to whole lines.
  function stage(figure, { rows, width = 720 }) {
    const node = svg("svg", {
      class: "stage",
      viewBox: `0 0 ${width} ${rows * ROW}`,
      style: `height: calc(var(--line-height) * ${rows})`,
      role: "img",
    });
    place(figure, node);
    fitStage(node);
    return node;
  }

  function fitStage(node) {
    const [, , w, h] = node.getAttribute("viewBox").split(" ").map(Number);
    const line = gridCellDimensions().height; // from index.js
    if (!node.clientWidth || !line) return;
    const rows = Math.max(1, Math.ceil(node.clientWidth * h / w / line - 0.01));
    node.style.height = `${rows * line}px`;
    // Labels are in user units; scale them up when the stage renders smaller than its viewBox
    // so they still read at about 13px (capped, so they don't crowd a phone-sized drawing).
    node.style.fontSize = `${Math.min(26, Math.max(13, 13 * w / node.clientWidth))}px`;
  }
  window.addEventListener("resize", () => document.querySelectorAll("svg.stage").forEach(fitStage));

  function controls(figure) {
    return place(figure, el("div", { class: "controls" }));
  }

  let sliderCount = 0;
  function slider(parent, { label, min, max, step, value, format = String }, onInput) {
    const id = `learn-slider-${++sliderCount}`;
    const row = el("div", { class: "control" }, parent);
    el("label", { for: id, text: label }, row);
    const input = el("input", { type: "range", id, min, max, step, value }, row);
    const output = el("output", { for: id }, row);
    const update = () => {
      const v = Number(input.value);
      output.textContent = format(v);
      onInput(v);
    };
    input.addEventListener("input", update);
    update();
    return {
      input,
      get value() { return Number(input.value); },
      // Move the slider and fire onInput, as if the reader had dragged it.
      set(v) { input.value = v; update(); },
      // Move the slider and its label without firing onInput (for animations driving it).
      show(v) { input.value = v; output.textContent = format(Number(v)); },
    };
  }

  function button(parent, label, onClick) {
    const node = el("button", { type: "button", text: label }, parent);
    node.addEventListener("click", onClick);
    return node;
  }

  function buttons(figure) {
    return place(figure, el("div", { class: "buttons" }));
  }

  // Not a live region: animated widgets rewrite it every frame, which would flood screen readers.
  // Slider values are announced through each slider's own <output>.
  function readout(figure) {
    return place(figure, el("pre", { class: "readout" }));
  }

  // A line with a triangular head at (x2, y2). `cls` colors both (e.g. "blue").
  function arrow(parent, cls) {
    const group = svg("g", { class: `arrow ${cls || ""}` }, parent);
    const line = svg("line", {}, group);
    const head = svg("polygon", {}, group);
    return {
      group,
      update(x1, y1, x2, y2, size = 9) {
        line.setAttribute("x1", x1); line.setAttribute("y1", y1);
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const back = (a) => [x2 - size * Math.cos(angle + a), y2 - size * Math.sin(angle + a)];
        const length = Math.hypot(x2 - x1, y2 - y1);
        // Stop the shaft at the base of the head so the tip stays sharp.
        const shaft = Math.max(0, length - size * 0.8) / (length || 1);
        line.setAttribute("x2", x1 + (x2 - x1) * shaft); line.setAttribute("y2", y1 + (y2 - y1) * shaft);
        head.setAttribute("points", [[x2, y2], back(0.42), back(-0.42)].map((p) => p.join(",")).join(" "));
        group.style.display = length < 1 ? "none" : "";
      },
    };
  }

  // Calls step(dt, t) every frame while `figure` is on screen. Under reduced motion it draws
  // one frame and stops. Returns { playing, play(), pause() } for widgets with a play button.
  function loop(figure, step) {
    let raf = 0, last = 0, t = 0, visible = false, wanted = !reducedMotion;
    const tick = (now) => {
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
      last = now;
      t += dt;
      step(dt, t);
      // step() may have called pause(); only keep going if it didn't.
      if (visible && wanted) raf = requestAnimationFrame(tick);
    };
    const sync = () => {
      cancelAnimationFrame(raf);
      last = 0;
      if (visible && wanted) raf = requestAnimationFrame(tick);
    };
    new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); }).observe(figure);
    step(0, 0);
    return {
      get playing() { return wanted; },
      play() { wanted = true; sync(); },
      pause() { wanted = false; sync(); },
    };
  }

  const fmt = (v, digits = 2) => (Object.is(Math.round(v * 10 ** digits) / 10 ** digits, -0) ? 0 : v).toFixed(digits);

  // Step-through animations ------------------------------------------------------------------
  // demo(figure, { height, label, steps, draw }) turns an empty <figure class="widget demo">
  // into a canvas with a caption and Back / Next / Play buttons.
  //   steps: [{ s: { name: number, ... }, cap: "one or two plain sentences" }, ...]
  //          Each step's `s` only lists what changes; values carry over from earlier steps.
  //   draw(ctx, s, c, d): paint one frame on a 640 x `height` drawing area. `s` is the current
  //          state (numbers are eased between steps), `c` the theme colors, `d` the helpers
  //          below (text, line, arrow, panel, dot, rect, clamp, fmt).
  const DEMO_W = 640;
  const demoPainters = [];

  function demoColors() {
    const css = getComputedStyle(document.documentElement);
    const get = (name) => css.getPropertyValue(name).trim();
    return {
      ink: get("--text-color"), muted: get("--text-color-alt"), bg: get("--background-color"),
      soft: get("--background-color-alt"), line: get("--demo-line"),
      blue: get("--accent-blue"), orange: get("--accent-orange"), green: get("--accent-green"),
      purple: get("--accent-purple"), cyan: get("--accent-cyan"), red: get("--accent-red"),
    };
  }

  // Numbers with a real minus sign and no "-0".
  function demoFmt(v, digits = 2) {
    let r = Number(v.toFixed(digits));
    if (Math.abs(r) < 10 ** -digits) r = 0;
    return r < 0 ? "−" + Math.abs(r) : String(r);
  }

  function demoTools(ctx, fontScale) {
    const tools = {
      clamp: (v, a, b) => Math.max(a, Math.min(b, v)),
      fmt: demoFmt,
      // text(str, x, y, { c: color, s: size, w: weight, a: align, b: baseline })
      text(str, x, y, o = {}) {
        ctx.font = `${o.w || 500} ${(o.s || 14) * fontScale}px "JetBrains Mono", ui-monospace, monospace`;
        ctx.fillStyle = o.c || "#888";
        ctx.textAlign = o.a || "left";
        ctx.textBaseline = o.b || "middle";
        ctx.fillText(str, x, y);
      },
      line(x1, y1, x2, y2, color, width = 1, dash) {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "butt";
        ctx.setLineDash(dash || []);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.setLineDash([]);
      },
      arrow(x1, y1, x2, y2, color, width = 3) {
        const angle = Math.atan2(y2 - y1, x2 - x1), length = Math.hypot(x2 - x1, y2 - y1);
        if (length < 2) return;
        const head = Math.min(13, length * 0.45);
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width; ctx.lineCap = "butt";
        ctx.beginPath(); ctx.moveTo(x1, y1);
        ctx.lineTo(x2 - Math.cos(angle) * head * 0.7, y2 - Math.sin(angle) * head * 0.7); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - head * Math.cos(angle - 0.42), y2 - head * Math.sin(angle - 0.42));
        ctx.lineTo(x2 - head * Math.cos(angle + 0.42), y2 - head * Math.sin(angle + 0.42));
        ctx.closePath(); ctx.fill();
      },
      // A filled rectangle; pass `stroke` for an outline as well.
      rect(x, y, w, h, fill, stroke, width = 2) {
        if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); }
        if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.strokeRect(x, y, w, h); }
      },
      // A boxed area for numbers, in the site's square bordered style.
      panel(c, x, y, w, h) { tools.rect(x, y, w, h, c.bg, c.ink, 2); },
      dot(x, y, r, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); },
    };
    return tools;
  }

  function demo(figure, { height, label, steps, draw }) {
    if (!figure) return;
    const canvas = el("canvas", { role: "img", "aria-label": label || "" }, figure);
    const ctx = canvas.getContext("2d");
    const cap = el("p", { class: "demo-cap", "aria-live": "polite" }, figure);
    const row = el("div", { class: "buttons demo-buttons" }, figure);
    const count = el("span", { class: "demo-count muted" }, row);
    const back = button(row, "Back", () => { stop(); go(index - 1); });
    const next = button(row, "Next", () => { stop(); go(index + 1); });
    const play = button(row, "Play", () => {
      if (timer) { stop(); return; }
      go(index === steps.length - 1 ? 0 : index + 1);
      timer = setInterval(() => { if (index >= steps.length - 1) stop(); else go(index + 1); }, 3600);
      ui();
    });

    // Each step's full state: its own values on top of everything before it.
    const full = [];
    let acc = {};
    for (const step of steps) { acc = { ...acc, ...step.s }; full.push(acc); }
    let index = 0, cur = { ...full[0] }, from = cur, to = cur, t0 = 0, raf = 0, timer = 0;
    let scale = 1, offsetY = 0, fontScale = 1;

    function paint() {
      const ratio = window.devicePixelRatio || 1;
      ctx.setTransform(scale * ratio, 0, 0, scale * ratio, 0, offsetY * ratio);
      ctx.clearRect(0, -offsetY / scale, DEMO_W, canvas.height / (scale * ratio));
      ctx.globalAlpha = 1;
      draw(ctx, cur, demoColors(), demoTools(ctx, fontScale));
      ctx.globalAlpha = 1;
    }
    // Fill the panel's width; round the height up to whole text lines and center the drawing.
    function size() {
      const width = canvas.clientWidth;
      const lineHeight = gridCellDimensions().height; // from index.js
      if (!width || !lineHeight) return;
      scale = width / DEMO_W;
      const rows = Math.max(1, Math.ceil(height * scale / lineHeight - 0.01));
      offsetY = (rows * lineHeight - height * scale) / 2;
      // Labels shrink with the drawing; give some of that back on a phone-sized canvas.
      fontScale = Math.min(1.15, Math.max(1, 0.72 / scale));
      const ratio = window.devicePixelRatio || 1;
      canvas.style.height = `${rows * lineHeight}px`;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(rows * lineHeight * ratio);
      // Reserve room for the longest caption so the buttons don't jump between steps.
      cap.style.minHeight = "";
      let tallest = 0;
      for (const step of steps) { cap.textContent = step.cap; tallest = Math.max(tallest, cap.offsetHeight); }
      cap.style.minHeight = `${Math.ceil(tallest / lineHeight - 0.01) * lineHeight}px`;
      cap.textContent = steps[index].cap;
      paint();
    }
    function ui() {
      cap.textContent = steps[index].cap;
      count.textContent = `Step ${index + 1}/${steps.length}`;
      back.disabled = index === 0;
      next.disabled = index === steps.length - 1;
      play.textContent = timer ? "Pause" : index === steps.length - 1 ? "Replay" : "Play";
    }
    const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
    function blend(a, b, t) {
      const out = {};
      for (const key in b) out[key] = typeof b[key] === "number" && typeof a[key] === "number" ? a[key] + (b[key] - a[key]) * t : b[key];
      return out;
    }
    function frame(now) {
      const t = Math.max(0, Math.min(1, (now - t0) / 950));
      cur = blend(from, to, ease(t));
      paint();
      raf = t < 1 ? requestAnimationFrame(frame) : 0;
    }
    function go(n) {
      if (n < 0 || n >= steps.length) return;
      cancelAnimationFrame(raf);
      from = { ...cur }; to = full[n]; index = n; ui();
      if (reducedMotion) { cur = { ...to }; paint(); return; }
      t0 = performance.now(); raf = requestAnimationFrame(frame);
    }
    function stop() { clearInterval(timer); timer = 0; ui(); }

    ui();
    new ResizeObserver(size).observe(canvas);
    size();
    demoPainters.push(paint);
  }

  const repaintDemos = () => demoPainters.forEach((paint) => paint());
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(repaintDemos);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", repaintDemos);

  // Progress -------------------------------------------------------------------------------

  function isDone(slug) {
    try { return localStorage.getItem(DONE_KEY + slug) === "1"; } catch (e) { return false; }
  }

  function setDone(slug, done) {
    try {
      if (done) localStorage.setItem(DONE_KEY + slug, "1");
      else localStorage.removeItem(DONE_KEY + slug);
    } catch (e) { /* storage unavailable: progress just isn't remembered */ }
  }

  function renderProgress() {
    for (const box of document.querySelectorAll("input[data-unit-done]")) box.checked = isDone(box.dataset.unitDone);
    for (const mark of document.querySelectorAll("[data-unit-mark]")) {
      const done = isDone(mark.dataset.unitMark);
      mark.textContent = done ? "✓" : "·";
      mark.classList.toggle("done", done);
    }
    for (const node of document.querySelectorAll("[data-progress]")) {
      const slugs = node.dataset.progress.split(" ").filter(Boolean);
      node.textContent = `${slugs.filter(isDone).length}/${slugs.length}`;
    }
    for (const node of document.querySelectorAll("[data-progress-bar]")) {
      const slugs = node.dataset.progressBar.split(" ").filter(Boolean);
      const done = slugs.filter(isDone).length;
      node.replaceChildren(
        el("span", { class: "done", text: "█".repeat(done) }),
        el("span", { class: "muted", text: "░".repeat(slugs.length - done) }),
        ` ${done} of ${slugs.length} done`,
      );
    }
  }

  document.addEventListener("change", (event) => {
    const slug = event.target.dataset && event.target.dataset.unitDone;
    if (slug) {
      setDone(slug, event.target.checked);
      renderProgress();
    }
  });
  renderProgress();

  // Quizzes --------------------------------------------------------------------------------
  // A <section class="quiz" data-pick="N"> holds a hidden .quiz-bank (built by build.py from
  // front matter). Each draw takes N random questions, alternating "term" and "apply" kinds
  // when both exist, and shuffles each question's options.

  function shuffled(items) {
    const a = items.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function mixedDraw(pool, count) {
    const byKind = {};
    for (const q of shuffled(pool)) (byKind[q.dataset.kind] = byKind[q.dataset.kind] || []).push(q);
    const queues = Object.values(byKind);
    const picked = [];
    while (picked.length < count && queues.some((q) => q.length)) {
      for (const queue of queues) if (queue.length && picked.length < count) picked.push(queue.shift());
    }
    return shuffled(picked);
  }

  let quizCount = 0;
  function setupQuiz(section) {
    const id = ++quizCount;
    const pick = Number(section.dataset.pick) || 5;
    const bank = [...section.querySelectorAll(".quiz-bank .quiz-q")];
    if (!bank.length) return;
    const filter = section.querySelector("input[data-quiz-filter]");
    const note = el("p", { class: "quiz-note muted" }, section);
    const list = el("ol", { class: "quiz-list" }, section);
    const result = el("p", { class: "quiz-result" }, section);
    const row = el("div", { class: "buttons" }, section);
    let items = [];

    function pool() {
      if (!filter || !filter.checked) return { questions: bank, note: "" };
      const done = bank.filter((q) => isDone(q.dataset.unit));
      if (done.length) return { questions: done, note: "" };
      return { questions: bank, note: "No units are marked done yet, so these come from all of them." };
    }

    function draw() {
      const { questions, note: message } = pool();
      note.textContent = message;
      note.hidden = !message;
      result.textContent = "";
      list.replaceChildren();
      items = mixedDraw(questions, pick).map((q, i) => {
        const li = el("li", { class: "quiz-item" }, list);
        el("div", { class: "quiz-prompt" }, li).innerHTML = q.querySelector(".quiz-prompt").innerHTML;
        const choices = shuffled([...q.querySelectorAll(".quiz-option")]).map((option) => {
          const label = el("label", { class: "quiz-choice" }, li);
          const input = el("input", { type: "radio", name: `quiz-${id}-${i}` }, label);
          el("span", {}, label).innerHTML = option.innerHTML;
          return { label, input, correct: option.hasAttribute("data-correct") };
        });
        const mark = el("p", { class: "quiz-mark", hidden: "" }, li);
        const why = el("div", { class: "quiz-why", hidden: "" }, li);
        why.innerHTML = q.querySelector(".quiz-why").innerHTML;
        if (filter && q.dataset.unit) {
          const from = el("p", { class: "quiz-from muted" }, why);
          from.append("From ");
          el("a", { href: `${q.dataset.unit}/`, text: `${q.dataset.unitNumber} ${q.dataset.unitTitle}` }, from);
        }
        return { choices, mark, why };
      });
      check.disabled = false;
    }

    function grade() {
      let right = 0, blank = 0;
      for (const item of items) {
        const chosen = item.choices.find((c) => c.input.checked);
        if (!chosen) blank++;
        const ok = Boolean(chosen && chosen.correct);
        if (ok) right++;
        for (const c of item.choices) {
          c.label.classList.toggle("is-correct", c.correct);
          c.label.classList.toggle("is-wrong", c.input.checked && !c.correct);
          c.input.disabled = true;
        }
        item.mark.hidden = false;
        item.mark.className = `quiz-mark ${ok ? "done" : "missed"}`;
        item.mark.textContent = ok ? "✓ Right." : chosen ? "✗ Not quite." : "✗ Left blank.";
        item.why.hidden = false;
      }
      result.textContent = `${right} of ${items.length} right` + (blank ? ` (${blank} left blank)` : "") + ".";
      check.disabled = true;
    }

    const check = button(row, "Check answers", grade);
    button(row, "New questions", draw);
    if (filter) filter.addEventListener("change", draw);
    draw();
  }
  document.querySelectorAll("section.quiz").forEach(setupQuiz);

  window.Learn = { reducedMotion, ROW, el, svg, stage, controls, slider, button, buttons, readout, arrow, loop, fmt, demo };
})();
