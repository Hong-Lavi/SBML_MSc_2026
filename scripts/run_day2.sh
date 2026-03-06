#!/usr/bin/env bash
# ===========================================================================
# Day2 실행 스크립트
# ===========================================================================
#
# 사용법:
#   cd SBML_MSc_2026
#   bash scripts/run_day2.sh
#
# 이 스크립트가 하는 일:
#   1. PYTHONPATH 설정 (src/ 패키지를 import 가능하게)
#   2. Day2 파이프라인 실행 (FP 계산 → split → DataLoader shape 검증)
#   3. Day2 관련 테스트 실행
#
# set -euo pipefail 설명:
#   -e: 어떤 명령이든 에러(exit code ≠ 0)나면 즉시 스크립트 중단
#   -u: 정의되지 않은 변수 사용 시 에러 (오타 방지)
#   -o pipefail: 파이프 중 어느 단계든 실패하면 전체 실패 처리
# ===========================================================================
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "=========================================="
echo " SBML_MSc_2026 | Day 2 Pipeline"
echo "=========================================="

echo ""
echo "[1/3] Running Day2 pipeline..."
python scripts/run_day2_pipeline.py --config configs/day2_ddi.yaml

echo ""
echo "[2/3] Running Day2 tests..."
pytest tests/test_day2_fingerprint.py tests/test_day2_dataset.py -v

echo ""
echo "[3/3] Done!"
echo "Results: results/day2/day2_summary.json"
echo "FP cache: data/processed/fingerprints/fp_cache.npz"
echo "✅ Day2 complete!"
