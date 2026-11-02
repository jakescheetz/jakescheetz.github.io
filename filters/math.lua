-- Layout fixes for MathML output.
-- 1. Keep punctuation on the same line as the inline math before it ("$x$," never wraps
--    between the math and the comma); browsers treat inline MathML as a breakable unit.
-- 2. Wrap display math in <span class="math-display">, a block that can scroll sideways on
--    narrow screens (a <math> element itself ignores max-width and overflow).
function Inlines(inlines)
  local out = pandoc.Inlines({})
  local i = 1
  while i <= #inlines do
    local el, next = inlines[i], inlines[i + 1]
    if el.t == "Math" and el.mathtype == "DisplayMath" then
      out:insert(pandoc.Span({ el }, { class = "math-display" }))
      i = i + 1
      goto continue
    end
    if el.t == "Math" and next and next.t == "Str" then
      local punct, rest = next.text:match("^([%.,;:!%?%)%]’”]+)(.*)$")
      if punct then
        out:insert(pandoc.Span({ el, pandoc.Str(punct) }, { class = "nowrap" }))
        if rest ~= "" then out:insert(pandoc.Str(rest)) end
        i = i + 2
        goto continue
      end
    end
    out:insert(el)
    i = i + 1
    ::continue::
  end
  return out
end
