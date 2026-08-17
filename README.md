# PhasePick Refiner

PhasePick Refiner improves machine-learning P/S arrival picks within
waveform-similarity clusters. It is standalone: it does not import Waveview or
require Waveview-formatted CSV/PKL files.

## What It Does

1. Reads a user pick CSV, event catalog, cluster JSON, and eventwise MiniSEEDs.
2. Selects one common P/S master for each station-cluster using phase scores,
   SNR, and consistency with the cluster's median S-P time.
3. Optionally lets the user review only those master picks in a small PyQt
   window.
4. Uses normalized ObsPy cross-correlation to refine existing picks and detect
   missing P/S pairs. P uses Z; S compares matching E/E, N/N, 1/1, or 2/2
   components and keeps the strongest match.
5. Writes a refined CSV with exactly the input column names/order, a provenance
   CSV, an attempt audit, and summary plots/tables.

Automatic picks are accepted only when both P and S exceed the CC threshold and
their corrected S-P remains reasonable. The original input file is never
overwritten.

## Setup

Python 3.10+ and ObsPy are required. From the repository:

```bash
conda activate obspy
pip install -e .
cp config.example.json config.json
```

Edit the input paths and output directory in `config.json`. Required pick
meanings are:

- event ID
- station ID in `NET.STA.LOC.CHANNEL_PREFIX` form
- P pick time
- S pick time

Phase scores and SNR are optional in the file format, but strongly recommended
for reliable automatic master selection. The catalog must contain event ID and
origin time. Cluster JSON must map cluster IDs to event-ID lists.

To see the configured column contract:

```bash
python run_phasepick_refiner.py columns config.json
```

## Workflow

For your own project, run:

```bash
python run_phasepick_refiner.py validate config.json
python run_phasepick_refiner.py select-masters config.json
python run_phasepick_refiner.py review-masters config.json  # optional
python run_phasepick_refiner.py refine config.json
python run_phasepick_refiner.py report config.json
```

For a completely automatic run:

```bash
MPLCONFIGDIR=/tmp python run_phasepick_refiner.py all config.json
```

## Included Example

The repository includes six real event waveforms in `example/`. Run the same
workflow without copying or changing a configuration file:

```bash
python run_phasepick_refiner.py validate config.example.json
python run_phasepick_refiner.py select-masters config.example.json
python run_phasepick_refiner.py review-masters config.example.json  # optional
python run_phasepick_refiner.py refine config.example.json
python run_phasepick_refiner.py report config.example.json
```

The example writes only to ignored `example/output/`.
If the configuration is expanded from a test subset to more clusters,
`review-masters` appends their automatic master selections while preserving
existing reviewed rows.

The master reviewer shows one cluster master event across five catalogued
stations per page, ordered from nearest to farthest epicentral distance, with
fixed Z plus two-horizontal rows for each station. An asterisk marks stations
that use the displayed event as their local CC master. Stations without ML
picks still show their waveforms. Dotted theoretical P/S arrivals use ObsPy
TauP and the configurable `viewer.taup_model` (default `iasp91`). They are
rough visual guides; a project-specific local velocity model is preferable for
location work.

Press `P` or `S`, then left-click any station band to set its arrival. A
complete manual P/S pair can promote the displayed event to that station's
local master. `Esc` cancels picking, Left/Right changes cluster, Page Up/Down
changes station page, and the Matplotlib toolbar pans/zooms. The master-event
dropdown handles clusters with several station-local masters. The right
sidebar controls filtering, gain, and origin-relative X limits. Picking keeps
the current horizontal zoom. Closing the window saves `master_selections.csv`.

## Outputs

- `phasepicks_refined.csv`: same schema and column order as the input pick CSV
- `phasepicks_refined_sources.csv`: one row per final phase with `O`, `CC`, or
  manual-master `C` provenance and CC metadata
- `master_selections.csv`: selected and reviewed station-cluster masters
- `cc_attempts.csv`: every attempted event/station pair and rejection reason
- `report/`: comparison CSVs, summary text, and figures

For a different pick-file convention, change only the mappings under
`columns.picks` in `config.json`. Code changes are normally unnecessary.

## Why Classes?

Each class owns one part of the workflow:

- `PickDataset` owns input tables and column meanings.
- `WaveformArchive` owns MiniSEED indexing and channel access.
- `MasterSelector` owns master-selection rules.
- `CrossCorrelationRefiner` owns waveform matching.
- `PickOutputWriter` owns schema preservation and provenance.
- `ReportGenerator` owns analysis products.
- `PhasePickRefinementProject` coordinates them in workflow order.

An object is simply one of these class definitions brought to life with its own
data. Functions are still used inside the classes; the class keeps related data
and functions together.
