-- Layout fixes for MathML output.
-- 1. Keep punctuation on the same line as the inline math before it ("$x$," never wraps
--    between the math and the comma), and keep quote marks around inline math with it;
--    browsers treat inline MathML as a breakable unit.
-- 2. Wrap display math in <span class="math-display">, a block that can scroll sideways on
--    narrow screens (a <math> element itself ignores max-width and overflow).
-- 3. Put TeX's thin space after operator names ("\log p" reads "log p", not "logp"):
--    texmath emits an invisible function-application operator that browsers draw with no
--    width.
-- build.py runs this on front matter too, so quiz text gets the same treatment.

local OPERATORS = {
  log = true, ln = true, exp = true, sin = true, cos = true, tan = true,
  max = true, min = true, det = true, Pr = true,
}
-- Commands that open a delimiter or add their own space: no thin space before these.
local NO_SPACE_BEFORE = {
  left = true, right = true, big = true, Big = true, bigg = true, Bigg = true,
  bigl = true, bigr = true, Bigl = true, Bigr = true, biggl = true, biggr = true,
  Biggl = true, Biggr = true, langle = true, lvert = true, lVert = true, lbrace = true,
  lfloor = true, lceil = true, quad = true, qquad = true, cdot = true, times = true,
}

-- Index just past the sub/superscript starting at tex[i] ("_" or "^").
local function after_script(tex, i)
  local c = tex:sub(i + 1, i + 1)
  if c == "{" then
    local depth = 0
    for j = i + 1, #tex do
      local ch = tex:sub(j, j)
      if ch == "{" then
        depth = depth + 1
      elseif ch == "}" then
        depth = depth - 1
        if depth == 0 then return j + 1 end
      end
    end
    return #tex + 1
  elseif c == "\\" then
    local _, e = tex:find("^\\%a+", i + 1)
    return (e or i + 1) + 1
  end
  return i + 2
end

local function space_operators(tex)
  local out, i = {}, 1
  while true do
    local s, e, name = tex:find("\\(%a+)", i)
    if not s then
      out[#out + 1] = tex:sub(i)
      break
    end
    out[#out + 1] = tex:sub(i, e)
    i = e + 1
    if OPERATORS[name] then
      while tex:sub(i, i) == "_" or tex:sub(i, i) == "^" do
        local j = after_script(tex, i)
        out[#out + 1] = tex:sub(i, j - 1)
        i = j
      end
      local ws = tex:match("^%s*", i)
      local k = i + #ws
      local nxt = tex:sub(k, k)
      local ordinary = nxt:match("%w") ~= nil
      if nxt == "\\" then
        local cmd = tex:match("^\\(%a+)", k)
        ordinary = cmd ~= nil and not NO_SPACE_BEFORE[cmd]
      end
      if ordinary then out[#out + 1] = "\\," end
      out[#out + 1] = ws
      i = k
    end
  end
  return table.concat(out)
end

function Math(el)
  el.text = space_operators(el.text)
  return el
end

-- Inline math, or a quotation that is nothing but inline math ("$x$").
local function is_inline_math(el)
  if el.t == "Math" then return el.mathtype == "InlineMath" end
  return el.t == "Quoted" and #el.content == 1 and el.content[1].t == "Math"
end

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
    if is_inline_math(el) then
      local punct, rest
      if next and next.t == "Str" then
        punct, rest = next.text:match("^([%.,;:!%?%)%]’”]+)(.*)$")
      end
      if punct then
        out:insert(pandoc.Span({ el, pandoc.Str(punct) }, { class = "nowrap" }))
        if rest ~= "" then out:insert(pandoc.Str(rest)) end
        i = i + 2
        goto continue
      end
      if el.t == "Quoted" then
        out:insert(pandoc.Span({ el }, { class = "nowrap" }))
        i = i + 1
        goto continue
      end
    end
    out:insert(el)
    i = i + 1
    ::continue::
  end
  return out
end
