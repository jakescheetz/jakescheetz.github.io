-- Turn "unit 04", "Unit 4's", "units 12 and 13" in course pages into links to those units.
-- build.py passes `unit-slugs` (space-separated directory names, e.g. "01-vectors 02-gradients"),
-- `unit-link-prefix` ("../" on a unit page, "" on the course page) and `unit-self` (the current
-- unit, which is never linked). Text already inside a link or a heading is left alone.

local slugs, prefix, self = {}, "", ""

local function read_meta(meta)
  if meta["unit-slugs"] then
    for slug in pandoc.utils.stringify(meta["unit-slugs"]):gmatch("%S+") do
      local number = tonumber(slug:match("^(%d+)%-"))
      if number then slugs[number] = slug end
    end
  end
  prefix = meta["unit-link-prefix"] and pandoc.utils.stringify(meta["unit-link-prefix"]) or ""
  self = meta["unit-self"] and pandoc.utils.stringify(meta["unit-self"]) or ""
  return meta
end


-- "04’s" -> 4, "04", "’s"; "4)" -> 4, "4", ")"
local function split_number(text)
  local digits, rest = text:match("^(%d%d?)(.*)$")
  if digits and (rest == "" or rest:match("^[^%d]")) then return tonumber(digits), digits, rest end
end

local function target(number)
  local slug = slugs[number]
  if slug and slug ~= self then return prefix .. slug .. "/" end
end

local function link_units(inlines)
  if next(slugs) == nil then return inlines end
  local out = pandoc.Inlines({})
  local i = 1
  while i <= #inlines do
    local el = inlines[i]
    local lead, word = "", nil
    if el.t == "Str" then lead, word = el.text:match("^(%(?)([Uu]nits?)$") end
    local space, num = inlines[i + 1], inlines[i + 2]
    if word and space and space.t == "Space" and num and num.t == "Str" and split_number(num.text) then
      local number, digits, rest = split_number(num.text)
      local url = target(number)
      if lead ~= "" then out:insert(pandoc.Str(lead)) end
      if url then
        out:insert(pandoc.Link({ pandoc.Str(word), pandoc.Space(), pandoc.Str(digits) }, url))
      else
        out:extend({ pandoc.Str(word), pandoc.Space(), pandoc.Str(digits) })
      end
      if rest ~= "" then out:insert(pandoc.Str(rest)) end
      i = i + 3
      -- Further numbers in a list: "units 12 and 13", "units 2, 4 and 8".
      while rest:match("^,?$") do
        local a, b, c = inlines[i], inlines[i + 1], inlines[i + 2]
        local joined = a and a.t == "Space" and b and b.t == "Str" and (b.text == "and" or b.text == "or" or b.text == "to")
          and c and c.t == "Space" and inlines[i + 3] and inlines[i + 3].t == "Str" and split_number(inlines[i + 3].text)
        local listed = a and a.t == "Space" and b and b.t == "Str" and split_number(b.text) and rest == ","
        if joined then
          out:extend({ a, b, c })
          number, digits, rest = split_number(inlines[i + 3].text)
          i = i + 4
        elseif listed then
          out:insert(a)
          number, digits, rest = split_number(b.text)
          i = i + 2
        else
          break
        end
        url = target(number)
        out:insert(url and pandoc.Link({ pandoc.Str(digits) }, url) or pandoc.Str(digits))
        if rest ~= "" then out:insert(pandoc.Str(rest)) end
      end
    else
      out:insert(el)
      i = i + 1
    end
  end
  return out
end

-- Read the metadata first, then walk the document top-down so links and headings can be skipped.
return {
  { Meta = read_meta },
  {
    traverse = "topdown",
    Link = function(el) return el, false end,
    Header = function(el) return el, false end,
    Inlines = link_units,
  },
}
