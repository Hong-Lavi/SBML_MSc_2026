# SBML_MSc_2026

Canonical, reproducibility-first repo for SBML onboarding (12-day program).

## Quickstart (Day 1 sanity)
```bash
conda env create -f environment.yml
conda activate sbml-msc
bash scripts/run_sanity.sh
```

## Repo structure
- `src/sbml_msc_2026/`: python package (datasets/features/models/trainers/utils)
- `scripts/`: entrypoints (train/infer/eval + sanity)
- `configs/`: YAML configs
- `data/raw`, `data/processed`: local data (gitignored)
- `results/`: outputs (gitignored)
- `docs/`: snapshots, logs
