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

  window.Learn = { reducedMotion, ROW, el, svg, stage, controls, slider, button, buttons, readout, arrow, loop, fmt };
})();
