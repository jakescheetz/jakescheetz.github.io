// Pad display equations so the content after them stays on the line grid,
// the same way index.js pads images and video.
function adjustMathPadding() {
  const lineHeight = gridCellDimensions().height;
  for (const equation of document.querySelectorAll('math[display="block"]')) {
    equation.style.paddingBottom = "0px";
    const remainder = equation.getBoundingClientRect().height % lineHeight;
    const padding = remainder > 0.5 ? lineHeight - remainder : 0;
    equation.style.paddingBottom = `${padding}px`;
  }
}

// ASCII diagrams (a <pre> in a plain <figure>) can't wrap. When one is wider than the
// page, as on a phone, shrink its font until it fits instead of cutting it off, down to a
// floor where it still reads; past that it scrolls. Line height shrinks with the font so
// box-drawing lines stay joined, and padding rounds the block back up to the line grid.
const DIAGRAM_MIN_FONT_PX = 8;
function fitDiagrams() {
  const lineHeight = gridCellDimensions().height;
  for (const pre of document.querySelectorAll("figure:not(.widget) > pre")) {
    pre.style.fontSize = "";
    pre.style.lineHeight = "";
    pre.style.paddingBottom = "";
    if (pre.scrollWidth <= pre.clientWidth) continue;
    const base = parseFloat(getComputedStyle(pre).fontSize);
    const size = Math.max(DIAGRAM_MIN_FONT_PX, Math.floor(base * pre.clientWidth / pre.scrollWidth * 10) / 10);
    pre.style.fontSize = `${size}px`;
    pre.style.lineHeight = `${(size * lineHeight) / base}px`;
    const remainder = pre.getBoundingClientRect().height % lineHeight;
    if (remainder > 0.5) pre.style.paddingBottom = `${lineHeight - remainder}px`;
  }
}

function layout() {
  fitDiagrams();
  adjustMathPadding();
}
layout();
window.addEventListener("load", layout);
window.addEventListener("resize", layout);
document.fonts?.ready.then(layout);
