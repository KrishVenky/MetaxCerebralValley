# Drawing Lazarus

An agent that turns a scanned legacy engineering drawing (a P&ID) into a **verified, simulatable
connectivity graph**: which symbols exist, what they are, what joins to what, and which parts a
person still has to look at. Then you can close a valve on the drawing and see what gets cut off.

Built for the Meta Global AI Developer Hackathon (Cerebral Valley): Indie Developer track, "Push the
Frontier". Team AxeCap.

## Why it matters

Plants run on drawings that only exist as scans. Nobody can query a scan, so "what is isolated if
valve V-104 is shut?" is answered by a person reading paper. Reading a drawing is easy to get
*mostly* right and dangerous to get quietly wrong, so Lazarus does not stop at detection: a
deterministic verifier checks the result, a re-inspection loop re-looks at what it flags, and
whatever is still doubtful is handed to a human instead of being guessed.

## What it does

1. **Find symbols.** Meta SAM 3.1, prompted with "symbol", over overlapping tiles of the whole sheet.
   Small scans are upscaled first so symbols land at a size SAM handles.
2. **Find what SAM cannot.** Muse Spark (Meta API) looks at the whole sheet for large outlines SAM
   never returns (vessels, exchangers), for non-drawing areas (title block, notes), and for the
   legend, which it then reads into `{item, meaning}` entries.
3. **Classify.** Muse Spark Open (via Baseten) types each symbol and reads its tag. On the
   benchmark it matches against the drawing's legend; on real scans it uses a fixed taxonomy plus the
   vocabulary the legend defines (instrument codes, valve names).
4. **Trace pipes.** Classical vision: erase symbols, skeletonise the line work, flood from each
   symbol, find junctions. Output is a graph of symbols, junctions and pipe edges.
5. **Verify.** No model in the loop. Dangling pipe ends, isolated symbols, valves with the wrong
   number of connections, tags that are not codes the legend defines, tags that disagree with their
   type, low-confidence classifications.
6. **Re-inspect.** Actionable flags get a closer look: SAM re-runs zoomed in around a dangling end,
   Spark re-reads a bad tag. A found symbol is kept only if it is confidently typed, wired into the
   graph, and removes the flag that prompted the search.
7. **Hand off.** What is still flagged is reported as needing a human. Pipes that just stop are
   reported separately as open ends.
8. **Simulate.** Connectivity and isolation: close valves, pick a source, see what stays reachable.
   This is graph analysis, not hydraulics; flow direction is unknown for these drawings.

## Results

Six held-out Digitize-PID sheets (the legend reference set uses six other drawings), full sheets
of 7168 x 4561 px:

| Measure | Result |
|---|---|
| Symbols found | 684 / 812 = **84%** recall |
| Detection precision | 684 / 719 = **95%** |
| Legend type exactly right (of found) | 641 / 684 = **94%** |
| Pipe links recovered, solid pipes (3 hand-labelled sheets) | 111 / 120 = **92%** |
| Pipe links recovered, all labelled links | 111 / 155 = 72% (85% when both end symbols were detected) |
| Link precision | 111 / 116 = **96%** |

Two things moved the numbers most: giving SAM more than one prompt phrase when a tile is sparse
(recall 54% to 78% on tiles), and reading types from a legend rather than a free taxonomy.
Dashed signal lines are mostly not recovered yet, which is most of the gap between 92% and 72%.

### Real scans (no ground truth)

Two public-domain real drawings are included in the demo, shown as the pipeline produces them: a
clean ISO 10628 render and a dense 1958 reactor-plant scan. Detection generalises reasonably
(instrument bubbles, pumps, motors, vessels via Spark; the plant scan's own legend is read into 60
entries and excluded from tracing). **Connectivity on dense hand-drawn scans is not trustworthy
yet**: most small in-line valves are missed and the plant scan still has many spurious junctions.
We measured these changes on the ISO render: 3x upscaling raised symbols found from 23 to 33, and
Spark-found vessel boxes cut false junctions from 34 to 14.

Legend-aware classification on the plant scan: instrument tags that parse as legend codes went from
36 of 97 to 36 of 37, and 59 circles holding a letter and number (reference markers, not
instruments) are no longer typed as instruments. This is a consistency check against the legend, not
accuracy against labels. We also tried showing the legend image to the classifier next to each
symbol; it fixed two answers and broke two valve types, so it is off by default.

## Meta tooling

| Model | Used for |
|---|---|
| SAM 3.1 (Meta Model API) | symbol detection over tiles, zoomed re-inspection |
| Muse Spark 1.3 (Meta Model API) | vessel and exchanger boxes, non-drawing regions, finding and reading the legend |
| Muse Spark Open (Baseten) | symbol typing against the legend, tag reading and re-reading |

**Muse Code** (Meta's coding agent CLI, model `muse-spark-1.3-contributor`) wrote the first draft of
`tests/test_codes.py` and `tests/test_routes.py`, the unit tests for the legend-grammar and
pipe-routing modules. It ran headless in a scratch copy of `lazarus/` with shell and web tools
disabled, so it could read and write files but not run anything. We ran pytest, and all of its tests
passed against the unmodified code. We added one test of our own (a pipe that does not join the two
nodes), noted in `tests/test_routes.py`. The rest of the project was written with Claude Code.

Meta API spend for the whole project so far: about $8 of the $150 credit. Every call is cached on
disk and logged to a spend ledger, with a hard stop (`SPEND_GUARD_USD`).

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # add META_API_KEY and BASETEN_API_KEY
python -m scripts.fetch_data               # the 12 benchmark drawings (not redistributed here)
python -m scripts.run_sheet 14.jpg 129.jpg 118.jpg 112.jpg 114.jpg 122.jpg
python -m scripts.real_test ri_sample_iso10628.png omre_pid.jpg
python -m scripts.connection_eval .cache/out/sheets
python scripts/export_demo.py
cd demo && python -m http.server 8000      # http://localhost:8000
python -m pytest tests
```

The demo: click a valve to close it, click another symbol to make it the source, tick "Open ends"
to see pipes that stop, hover a green-outlined symbol to see what re-inspection recovered. The
committed demo bundle contains the real scans; benchmark sheets draw their graphs without the
background image unless you run `scripts.fetch_data` and `scripts/export_demo.py` locally.

## Layout

```
lazarus/            pipeline: sheet tiling, SAM + Spark clients, classify/legend/codes,
                    pipe tracing, verifier, re-inspection loop, routes, simulation
scripts/            benchmarks and experiments (run_sheet, real_test, connection_eval, ...)
demo/               static page (vanilla JS + SVG) over exported graphs
data/labels/        hand-labelled connections used by connection_eval
data/real/          public-domain real test scans and their sources
tests/              simulation tests; legend-grammar and pipe-route tests (drafted by Muse Code)
```

## Data and licences

- **Code** (`lazarus/`, `scripts/`, `demo/`, `tests/`) is MIT licensed, see `LICENSE`. The licence
  covers our code only; the data below keeps its own terms.
- **Digitize-PID** (Paliwal et al., 2021, arXiv:2109.03794), via `hamzas/digitize-pid-symbols` on
  Hugging Face. It declares no licence, so its images are fetched by `scripts/fetch_data.py` and
  not committed. The hand-labelled connections in `data/labels/` are our own annotations.
- **Real scans** in `data/real/` are public domain; sources are in `data/real/SOURCES.md`. A third,
  CC BY-SA scan used in experiments is not included.
- Nothing in this repo contains API keys. `.env` is ignored.

## Limitations

- Connection graph quality drops on dense or hand-drawn scans (see above).
- Dashed signal lines are mostly not recovered; ISA instrument-loop connections are therefore missed.
- Re-inspection is bounded (two rounds) and recovers few symbols on the benchmark; most of its value
  today is the triage into "needs a human" versus "open end".
- The isolation simulation answers connectivity questions only.
