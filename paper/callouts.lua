-- Fenced divs ::: {.keybox title="..."} become tcolorbox environments in LaTeX output.
local boxes = { keybox = "keybox", note = "notebox", hedge = "hedgebox" }
function Div(el)
  for cls, env in pairs(boxes) do
    if el.classes:includes(cls) and FORMAT:match("latex") then
      local title = el.attributes["title"] or ""
      table.insert(el.content, 1, pandoc.RawBlock("latex", "\\begin{" .. env .. "}{" .. title .. "}"))
      table.insert(el.content, pandoc.RawBlock("latex", "\\end{" .. env .. "}"))
      return el.content
    end
  end
end
