#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "=== SBML_MSc_2026 | Day 1 sanity ==="

echo "[1/5] Python"
which python || true
python -V

echo "[2/5] Key imports"
python - <<'PY'
import importlib
pkgs = ['numpy','pandas','sklearn','yaml']
for p in pkgs:
    m = importlib.import_module(p)
    print(p, 'OK', getattr(m, '__version__', ''))
try:
    import torch
    print('torch OK', torch.__version__, 'cuda_available=', torch.cuda.is_available())
except Exception as e:
    print('torch not available:', e)
PY

echo "[3/5] Local package import"
python -c "import sbml_msc_2026; print('sbml_msc_2026 OK')"

echo "[4/5] Run sanity entrypoint"
python scripts/sanity_python.py --config configs/base.yaml --out results/sanity

echo "[5/5] Pytest"
pytest -q

echo "✅ Done. See results/sanity/ and docs/logs/day1_sanity.log"
