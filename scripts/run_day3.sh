#!/usr/bin/env bash
# ============================================================
# Day3: DDI MLP Training Pipeline Runner
# ============================================================
# 사용법: bash scripts/run_day3.sh
# 전제: 프로젝트 루트에서 실행, conda 환경 활성화 상태
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "============================================================"
echo "Day3: DDI MLP Training Pipeline"
echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH:   $PYTHONPATH"
echo "============================================================"

# ── 1) Unit Tests ──────────────────────────────────────────────
echo ""
echo "--- [Step 1/3] Running Day3 unit tests ---"
python -m pytest "${PROJECT_ROOT}/tests/test_day3_model.py" -v --tb=short
python -m pytest "${PROJECT_ROOT}/tests/test_day3_trainer.py" -v --tb=short
echo "✅ All Day3 tests passed."

# ── 2) Training ────────────────────────────────────────────────
echo ""
echo "--- [Step 2/3] Running DDI MLP training ---"
python "${PROJECT_ROOT}/scripts/train_ddi.py" \
    --config "${PROJECT_ROOT}/configs/day3_ddi.yaml"

# ── 3) Verify outputs ─────────────────────────────────────────
echo ""
echo "--- [Step 3/3] Verifying outputs ---"
RESULTS_DIR="${PROJECT_ROOT}/results/day3"
CKPT_DIR="${RESULTS_DIR}/checkpoints"

if [ -f "${CKPT_DIR}/best_model.pt" ]; then
    echo "✅ Checkpoint: ${CKPT_DIR}/best_model.pt"
else
    echo "❌ Missing: best_model.pt"
    exit 1
fi

if [ -f "${RESULTS_DIR}/metrics.json" ]; then
    echo "✅ Metrics: ${RESULTS_DIR}/metrics.json"
else
    echo "❌ Missing: metrics.json"
    exit 1
fi

if [ -f "${CKPT_DIR}/training_log.json" ]; then
    echo "✅ Training log: ${CKPT_DIR}/training_log.json"
else
    echo "❌ Missing: training_log.json"
    exit 1
fi

echo ""
echo "============================================================"
echo "Day3 complete. Results in: ${RESULTS_DIR}/"
echo "============================================================"
