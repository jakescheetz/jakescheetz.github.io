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

adjustMathPadding();
window.addEventListener("load", adjustMathPadding);
window.addEventListener("resize", adjustMathPadding);
