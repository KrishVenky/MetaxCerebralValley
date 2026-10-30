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
   A second trace closes gaps along the horizontal and vertical axes, which turns a dashed line into
   one component; symbol pairs only this trace joins are kept as **dashed links**, separate from the
   pipe edges (a signal line is not a process line, so isolation analysis never crosses one). Each is
   then checked against the pixels by a deterministic checker (drafted by Muse Code); the ones it
   confirms are marked pixel-verified, the rest stay "inferred".
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
| Links recovered, solid pipes (3 hand-labelled sheets) | 111 / 120 = **92%** |
| Links recovered, all labelled links (solid and dashed) | 130 / 155 = **84%** (100% when both end symbols were detected) |
| Link precision, all links | 130 / 144 = **90%** (solid pipes alone: 111 / 116 = 96%) |

Three things moved the numbers most: giving SAM more than one prompt phrase when a tile is sparse
(recall 54% to 78% on tiles), reading types from a legend rather than a free taxonomy, and the
gap-closing second trace for dashed lines (all-link recall 72% to 84%, link precision 96% to 90%; the
solid-pipe result is unchanged because the second trace is separate). Of the 35 labelled links that
involve a dashed line, 19 are recovered. The dashed pass found 28 links on these tiles: 19 right, 9
wrong (mostly pairs joined only through a crossing of unrelated lines). The checker confirms 7 of
them, and all 7 are right; the other 21 are shown as inferred. The 21 px gap size was tuned on this
dataset's dash pitch (21 and 24 give the same recall, 36 drops precision to 63%), so it may not carry
over to other drawings.

### Against one plain model call

Same three hand-labelled tiles, scored the same way (`scripts/baseline_eval.py`). The baseline is a
single Muse Spark call shown the tile with the **ground-truth symbols already boxed and numbered**,
asked to list which symbols a pipe joins. Lazarus has to find the symbols itself.

| Measure | One Muse Spark call (symbols given) | Lazarus (finds its own symbols) |
|---|---|---|
| Recall, all labelled links | 110 / 155 = 71% | 130 / 155 = **84%** |
| Recall, solid pipes | 94 / 120 = 78% | 111 / 120 = **92%** |
| Precision, all links | 110 / 117 = **94%** | 130 / 144 = 90% |
| Dashed links recovered | 16 / 35 | 19 / 35 |
| Output flagged for a human | none | yes, every finding is listed |

Lazarus is ahead on recall, on solid pipes and (narrowly) on dashed links, while also doing the symbol
detection the baseline is handed; the plain call is a little more precise. When we first measured
this, Lazarus recovered none of the dashed links (72% against 71%); the gap-closing pass is what
changed that.
One run per sheet (the baseline is not deterministic), and the three baseline calls used 19k to 34k
reasoning tokens each (about $0.35 for the three tiles).

### Real scans (no ground truth)

Two public-domain real drawings are included in the demo, shown as the pipeline produces them: a
clean ISO 10628 render and a dense 1958 reactor-plant scan. Detection generalises reasonably
(instrument bubbles, pumps, motors, vessels via Spark; the plant scan's own legend is read into 60
entries and excluded from tracing). **Connectivity on dense hand-drawn scans is not trustworthy
yet**: most small in-line valves are missed and the plant scan still has many spurious junctions.
We measured these changes on the ISO render: 3x upscaling raised symbols found from 23 to 33, and
Spark-found vessel boxes cut false junctions from 34 to 14.

Two things we tried for real scans and did not ship. (1) The dashed-link pass is switched off on them:
on the plant scan it returned 1,132 links for 135 symbols (the benchmark gives about one per symbol),
because text and hatching merge into the line work once gaps are closed. (2) Small in-line valves are
missed on both scans (on the ISO render, V-102, V-105, V-109 and the check valves) and re-inspection never
looks for them, since a valve on a continuous pipe leaves no dangling end. SAM 3.1 returns nothing for
"valve" prompts on that render, Muse Spark finds the valves but its boxes are 70 to 100 px off a symbol
that is 40 by 70 px, and a classical candidate finder (compact ink on a pipe that continues both sides)
found them together with about a dozen signal-line arrowheads that the classifier types as valves at the
same confidence (0.55 to 0.78), so the output would have been noise for a reviewer. Still open.

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
`tests/test_codes.py` and `tests/test_routes.py` (unit tests for the legend-grammar and pipe-routing
modules) and, in two rounds, `lazarus/dashed.py` with `tests/test_dashed.py`: a deterministic checker
that decides whether a dashed line really joins two symbols by walking straight, L and Z routes between
them. It ran headless in a scratch copy of `lazarus/` with shell and web tools disabled, so it could
read and write files but not run anything; we ran pytest and measured the checker, and used its output
unmodified. On the labelled sheets it accepts 16 of 35 true dashed links and none of 90 unlinked
pairs when given the true symbol boxes. Its limit is that it only models direct routes, so it
confirms 7 of the 19 dashed links the pipeline finds, all of them right. We added one test of our own
to `tests/test_routes.py` (noted there). The rest of the project, including the gap-closing trace
that finds the dashed links, was written with Claude Code.

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
tests/              simulation and signal-link tests; legend-grammar, pipe-route and
                    dashed-checker tests (drafted by Muse Code)
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

- Connection graph quality drops on dense or hand-drawn scans (see above), and the dashed-link pass is
  not run on them.
- Dashed links: 19 of 35 recovered, 9 wrong ones on the same tiles, only 7 pixel-verified; the rest
  are marked inferred. The gap size is tuned on the benchmark's dash pitch and is untested on other
  drawings. Dashed links are not used by the isolation simulation.
- Re-inspection is bounded (two rounds) and recovers few symbols on the benchmark; most of its value
  today is the triage into "needs a human" versus "open end".
- The isolation simulation answers connectivity questions only.
