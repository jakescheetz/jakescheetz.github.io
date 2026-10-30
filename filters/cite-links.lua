-- Point in-text citations at the cited work itself instead of the reference list.
-- Runs after --citeproc with `link-citations`, which renders each citation as a link to
-- `#ref-<id>`. A single citation becomes one link, author names included; grouped
-- citations keep one link each. Entries without a URL or DOI keep linking to the list.
function Pandoc(doc)
  local urls = {}
  for _, ref in ipairs(pandoc.utils.references(doc)) do
    if ref.url then
      urls[ref.id] = pandoc.utils.stringify(ref.url)
    elseif ref.doi then
      urls[ref.id] = "https://doi.org/" .. pandoc.utils.stringify(ref.doi)
    end
  end

  return doc:walk({
    Cite = function(cite)
      local single = #cite.citations == 1 and urls[cite.citations[1].id]
      if single then
        local text = cite.content:walk({ Link = function(link) return link.content end })
        cite.content = { pandoc.Link(text, single) }
        return cite
      end
      return cite:walk({
        Link = function(link)
          local id = link.target:match("^#ref%-(.+)$")
          if id and urls[id] then
            link.target = urls[id]
          end
          return link
        end,
      })
    end,
  })
end
