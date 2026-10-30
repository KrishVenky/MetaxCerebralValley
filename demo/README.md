# Drawing Lazarus demo

Static, dependency-free page (vanilla JS + SVG). Build the data bundle, then serve this folder:

    python scripts/export_demo.py          # from the repo root: writes demo/data/ (sheet images + index.json)
    cd demo && python -m http.server 8000  # open http://localhost:8000

What it shows, per drawing:

- the whole sheet, with every detected symbol (coloured by kind), the pipes routed along the traced
  line work, and the verifier's findings. Green-outlined symbols were missed on the first pass and
  recovered by the re-inspection loop.
- **Needs a human** lists findings a person should look at. **Open ends** (pipes that just stop or
  leave the sheet) are informational and hidden on the drawing until you tick the box.
- the excluded areas (title block, notes) and, where the sheet has one, the legend Muse Spark read.
- **dashed links** (purple dotted lines): instrument/signal lines the pipe tracer cannot see. The ones a
  deterministic pixel check confirmed are drawn along the line it found; hover any symbol to see its other
  dashed links (inferred, not confirmed). Dashed links never count as connections in the isolation
  simulation.
- **a human review loop**, no API needed. Click a numbered flag (or a row under *Needs a human*) to open the
  review card: confirm the finding as drawn, change the symbol's type, fix its tag, connect it to another
  symbol (click it on the drawing), disconnect a pipe, or remove it as not a symbol. The graph and the
  isolation analysis update immediately, and a flag disappears when you act on it or when the verifier's own
  rule (isolated symbol, wrong connection count, missing tag) stops holding after your edit. *Undo last
  review* steps back; progress is kept in the browser (localStorage) and *Download corrections* saves it as
  JSON.
- an isolation simulation: click a valve to close it, click any other symbol (or shift+click a valve)
  to set the source; Esc reopens all valves. Isolation is connectivity analysis
  (`lazarus/simulate.py`, ported in `app.js`), not hydraulics.

Data sources (re-run the matching script after any pipeline change, then re-export):

| Group | Script | Graph files |
|---|---|---|
| Benchmark sheets (held out of the legend) | `python -m scripts.run_sheet 14.jpg 129.jpg ...` | `.cache/out/sheets/` |
| Real scans (no ground truth) | `python -m scripts.real_test ri_sample_iso10628.png omre_pid.jpg` | `.cache/out/real/` |

Real scans are shown as they come out of the pipeline. Their symbols use a fixed taxonomy, and
connectivity is rougher than on the benchmark sheets. The page says so.

Limits: pipe routes are for display only (the graph and simulation do not use them); the page needs a
local web server because it loads `data/index.json`.
