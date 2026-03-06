"""
===========================================================================
Day2 Entrypoint — 전체 파이프라인 통합 실행
===========================================================================

[실행 방법]
  cd SBML_MSc_2026
  export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
  python scripts/run_day2_pipeline.py --config configs/day2_ddi.yaml

[이 스크립트가 하는 일]
  1. Config 로드 + seed 설정
  2. FP 계산 (또는 캐시 로드)
  3. Data split (drug_aware)
  4. Train/Test Dataset + DataLoader 생성
  5. 첫 배치 shape sanity check
  6. 결과 요약 JSON 저장

  이 스크립트는 "모델 학습"은 하지 않는다. Day3 범위.
  Day2의 목표는 "모델에 넣을 수 있는 형태의 배치가 나오는가?"까지 확인.

[scripts/ 폴더의 역할]
  src/의 모듈은 직접 실행하지 않는다.
  scripts/에서 모듈들을 import하여 조합하는 "접착제(glue)" 역할을 한다.
  이렇게 분리하면:
  - 같은 모듈을 다른 스크립트에서 재사용 가능
  - 스크립트만 바꿔서 다른 실험 구성 가능
  - 테스트 코드가 스크립트에 의존하지 않음
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
import torch

# --- src/ 패키지 import ---
# PYTHONPATH에 src/가 포함되어 있어야 한다.
# run_sanity.sh처럼 export PYTHONPATH="$(pwd)/src:..." 로 설정.
from sbml_msc_2026.utils.config import load_yaml
from sbml_msc_2026.utils.seed import set_seed
from sbml_msc_2026.features.fingerprint import get_or_compute_fp_dict
from sbml_msc_2026.datasets.ddi_dataset import DDIPairDataset, build_ddi_dataloader
from sbml_msc_2026.datasets.splitter import split_ddi_pairs

import pandas as pd

logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    # ==================================================================
    # Step 1: Config 로드 + Seed 설정
    # ==================================================================
    # 왜 config를 코드 바깥에 두는가?
    # → 같은 코드로 seed/batch_size/split방식만 바꿔가며 실험 가능.
    # → git diff로 "이번 실험에서 뭘 바꿨는지" 추적 가능.
    cfg = load_yaml(config_path)
    seed_info = set_seed(cfg["seed"], cfg["reproducibility"]["deterministic"])
    logger.info(f"Config loaded from {config_path}")
    logger.info(f"Seed info: {seed_info}")

    # ==================================================================
    # Step 2: Raw 데이터 로드 + FP 계산/캐시
    # ==================================================================
    raw_csv = cfg["paths"]["raw_ddi_csv"]
    fp_cache_dir = Path(cfg["paths"]["fp_cache_dir"])
    fp_cache_path = fp_cache_dir / "fp_cache.npz"

    df = pd.read_csv(raw_csv)
    logger.info(f"Loaded {len(df)} DDI pairs from {raw_csv}")

    # 전체 약물 SMILES 추출 → FP dict
    all_smiles = df["drug_a_smiles"].tolist() + df["drug_b_smiles"].tolist()
    fp_dict = get_or_compute_fp_dict(
        smiles_list=all_smiles,
        cache_path=fp_cache_path,
        radius=cfg["fingerprint"]["radius"],
        n_bits=cfg["fingerprint"]["n_bits"],
    )
    logger.info(f"FP dict: {len(fp_dict)} unique drugs")

    # ==================================================================
    # Step 3: Train/Test Split
    # ==================================================================
    splits = split_ddi_pairs(
        df=df,
        method=cfg["split"]["method"],
        train_ratio=cfg["split"]["train_ratio"],
        seed=cfg["seed"],
    )
    logger.info(
        f"Split: train={len(splits['train'])} pairs, "
        f"test={len(splits['test'])} pairs"
    )

    # ==================================================================
    # Step 4: Dataset + DataLoader 생성
    # ==================================================================
    train_ds = DDIPairDataset(
        csv_path=raw_csv,
        fp_cache_path=fp_cache_path,
        pair_method=cfg["pair_feature"]["method"],
        fp_radius=cfg["fingerprint"]["radius"],
        fp_n_bits=cfg["fingerprint"]["n_bits"],
        indices=splits["train"],
    )
    test_ds = DDIPairDataset(
        csv_path=raw_csv,
        fp_cache_path=fp_cache_path,
        pair_method=cfg["pair_feature"]["method"],
        fp_radius=cfg["fingerprint"]["radius"],
        fp_n_bits=cfg["fingerprint"]["n_bits"],
        indices=splits["test"],
    )

    dl_cfg = cfg["dataloader"]
    train_loader = build_ddi_dataloader(
        train_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
    )
    test_loader = build_ddi_dataloader(
        test_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
    )

    # ==================================================================
    # Step 5: Shape Sanity Check
    # ==================================================================
    # 이것이 Day2의 최종 검증 포인트다.
    # 모델에 넣을 수 있는 shape이 나오는지 확인한다.
    print("\n" + "=" * 60)
    print("Day2 Shape Sanity Check")
    print("=" * 60)

    pair_method = cfg["pair_feature"]["method"]
    n_bits = cfg["fingerprint"]["n_bits"]
    expected_dim = 2 * n_bits if pair_method in ("concat", "symmetric") else n_bits

    for split_name, loader in [("train", train_loader), ("test", test_loader)]:
        for batch_feat, batch_label in loader:
            B, D = batch_feat.shape
            print(f"[{split_name}] batch shape: features={batch_feat.shape}, "
                  f"labels={batch_label.shape}")
            assert D == expected_dim, (
                f"Feature dim mismatch! Expected {expected_dim}, got {D}. "
                f"pair_method={pair_method}, n_bits={n_bits}"
            )
            assert batch_label.dtype == torch.int64, (
                f"Label dtype should be int64, got {batch_label.dtype}"
            )
            break  # 첫 배치만 확인

    print(f"\nExpected feature dim: {expected_dim} ✓")
    print(f"Pair method: {pair_method}")
    print(f"Train pairs: {len(train_ds)}, Test pairs: {len(test_ds)}")

    # ==================================================================
    # Step 6: 결과 요약 JSON 저장
    # ==================================================================
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "day": 2,
        "config": config_path,
        "seed": cfg["seed"],
        "n_total_pairs": len(df),
        "n_unique_drugs": len(fp_dict),
        "split_method": cfg["split"]["method"],
        "n_train_pairs": len(splits["train"]),
        "n_test_pairs": len(splits["test"]),
        "pair_method": pair_method,
        "feature_dim": expected_dim,
        "fp_n_bits": n_bits,
        "fp_radius": cfg["fingerprint"]["radius"],
        "batch_size": dl_cfg["batch_size"],
    }

    summary_path = results_dir / "day2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print("✅ Day2 pipeline sanity check PASSED!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Day2 DDI feature pipeline")
    parser.add_argument("--config", type=str, default="configs/day2_ddi.yaml")
    args = parser.parse_args()

    main(args.config)
