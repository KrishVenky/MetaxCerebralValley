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
