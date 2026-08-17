# Included Example Dataset

This is a compact, non-synthetic subset of the Ferndale aftershock working
dataset used while developing PhasePick Refiner.

It contains:

- clusters `4` and `1121`
- three events from each cluster
- 49 original ML/Gamma event-station pick rows
- six eventwise, 60-second MiniSEED files
- the corresponding six catalog rows
- the 21-row station table

The CSV values and MiniSEED contents were copied without scientific
modification. The cluster JSON was reduced to the six included event IDs. The
example is intended for learning and software verification, and is not an
authoritative waveform or earthquake catalog distribution.

The repository's MIT license covers the PhasePick Refiner software. It does not
relicense the underlying earthquake catalog or seismic-network waveform data;
users should obtain and cite authoritative scientific data for research use.

Run the ordinary PhasePick Refiner workflow from the repository root:

```bash
python run_phasepick_refiner.py validate config.example.json
python run_phasepick_refiner.py select-masters config.example.json
python run_phasepick_refiner.py review-masters config.example.json  # optional
python run_phasepick_refiner.py refine config.example.json
python run_phasepick_refiner.py report config.example.json
```

Generated files go to `example/output/`, which is ignored by Git.

The included fixture currently produces 16 selected station-cluster masters,
28 accepted CC P/S pairs, and six newly detected phases with the default test
configuration. Small numerical differences may occur across library versions.
