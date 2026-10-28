# Real P&ID test images (data/real/)

All three files were downloaded from Wikimedia Commons (original-resolution files, unmodified) on 2026-10-28.
Redistribution-relevant notes: omre_pid.jpg and ri_sample_iso10628.png carry no attribution requirement;
schema_pid_isa_s51.jpg is CC BY-SA 3.0 (attribution + share-alike required if you redistribute it or derivatives).

## 1. omre_pid.jpg
- Source URL: https://commons.wikimedia.org/wiki/File:Organic_Moderated_Reactor_Experiment_Piping_and_Instrumentation_Diagram.jpg
- Direct file: https://upload.wikimedia.org/wikipedia/commons/5/5b/Organic_Moderated_Reactor_Experiment_Piping_and_Instrumentation_Diagram.jpg
- Original document: https://www.osti.gov/biblio/4307822 (Fig. 9, OMRE report, Feb 1958)
- License: Public domain (US federal government work). Commons page: "This work is in the public domain in the United States because it is a work prepared by an officer or employee of the United States Federal Government as part of that person's official duties" (17 U.S.C. 105).
- Author: US Atomic Energy Commission
- Size: 4000 x 3266 px, 3,367,661 bytes (JPEG, 300 dpi tag)
- Description: Full-sheet P&ID of the Organic Moderated Reactor Experiment: reactor vessel, tanks, pumps, heat exchanger, air-blast cooler, vacuum still, ~100+ valves, many boxed instrument tags (BPI, BTIM, etc.), line-number bubbles, handwritten OSTI stamp. HAS a legend (instrument function table + valve symbol key, bottom-left). Genuine scanned look (paper fold crease, grey scan tone, 1950s symbology, very dense). Hardest and most realistic test; non-ISA symbol shapes.

## 2. ri_sample_iso10628.png
- Source URL: https://commons.wikimedia.org/wiki/File:RI_Sample_ISO.png
- Direct file: https://upload.wikimedia.org/wikipedia/commons/d/d6/RI_Sample_ISO.png
- License: Public domain (PD-self). Commons page: "I, the copyright holder of this work, release this work into the public domain. This applies worldwide."
- Author: Geichler (Commons user), uploaded 28 Aug 2009
- Size: 2201 x 1701 px, 25,248 bytes (1-bit PNG)
- Description: Clean CAD-style sample P&ID per EN ISO 10628: naphtha tank, shell-and-tube exchanger, two pump sets with motors, control valves, instrument bubbles (FC, TIRC, LIRC, PDT...), full line-number labels, off-page arrows. No legend. Binary clean render, no scan noise; good "easy/clean" baseline with ISO symbology and moderate density (~25 valves/instruments).

## 3. schema_pid_isa_s51.jpg
- Source URL: https://commons.wikimedia.org/wiki/File:Sch%C3%A9ma_P%26ID.jpg
- Direct file: https://upload.wikimedia.org/wikipedia/commons/1/14/Sch%C3%A9ma_P%26ID.jpg
- License: CC BY-SA 3.0 Unported. Commons page: "This file is licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license."
- Author: Evinfo (Commons user, "own work"), 7 Mar 2012 (scanned, Canon MG6100)
- Size: 3344 x 2012 px, 801,794 bytes
- Description: Scan of a hand/CAD-drawn ISA S5.1 instrumentation diagram (French labels): water tank, ~12 valves of several types, large instrument bubbles (HV, TV, LAHH, LSH, TC, LIC...), dashed signal lines, flow-element symbols. No legend; low symbol count (sparse) but real scanner noise (speckle, slight skew), so good for testing robustness to scan artefacts and dashed-line signal connectivity. Caveat: it appears to be the left part of a larger drawing (right edge is cut off).

## Rejected / not included
- File:P&ID.JPG (evaporative crystallizer, https://commons.wikimedia.org/wiki/File:P%26ID.JPG): Commons says CC BY-SA 3.0 "own work" by Ub, but the drawing's title block reads that it is "the property of Bewdley Technical Services Limited", so provenance/licence is doubtful. Deleted.
- File:Schéma P&ID1.jpg: near-duplicate scan of the same drawing as #3 (same author/licence); dropped to avoid train/test leakage.
- File:P&ID Lab Theory.pdf (CC BY-SA 3.0): a 28-page text document on piping codes, no drawings. Deleted.
- File:Detail of Instrumentation and Control Tanks ... HAER AL-129-A (public domain, LOC): isometric cutaway illustration, not a P&ID. Deleted.
- File:P&ID drawing template.jpg (1180x610), Pig launcher.jpg (775x327): below resolution target; symbol-only SVGs (P&ID X8081 etc.) and P&ID-*.png (<500 px): single symbols, not diagrams.
- ISO 10628-2 2012 Symbols Sheet 1-7.pdf (CC BY-SA 4.0): symbol legend sheets, not P&IDs (could be useful as legend reference only).
- Web search for US-gov (DOE/EPA/NRC/OSHA/USACE) standalone P&IDs found only Scribd/Academia/course re-uploads or non-drawing documents; none usable with clear licensing.
