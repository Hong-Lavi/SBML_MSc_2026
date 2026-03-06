"""
Day 3: DDI MLP 학습 Entry Point

Pipeline 위치: [최상위] — config 로드 → seed → data → model → train → save
Input:  configs/day3_ddi.yaml
Output: results/day3/ (checkpoint, metrics.json, training_log.json)

TODO 개수: 6개 (TODO S-1 ~ S-6)
난이도: ★★☆
예상 소요: ~30min

[이 스크립트가 하는 일]
  1. Config 로드 + seed 고정
  2. Day2 모듈로 train/val/test DataLoader 구성
  3. MLP 모델 생성
  4. Weighted CE Loss + optimizer 생성
  5. fit() 호출 (training + early stopping)
  6. Best checkpoint 로드 + test set 최종 평가 + 결과 저장

[OOM 리스크]
  dummy data 16행에서는 문제없지만, 실제 DDI 데이터셋 (수십만 pair) 시:
  - DataLoader num_workers를 2~4로 올리되, 메모리 사용량 모니터링 필요
  - model hidden_dims가 크면 GPU 메모리 부족 가능 → batch_size 줄이기
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ── PYTHONPATH 설정 (src-layout) ──
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch

from sbml_msc_2026.utils.config import load_yaml
from sbml_msc_2026.utils.seed import set_seed

logger = logging.getLogger(__name__)


def main(config_path: str = "configs/day3_ddi.yaml") -> None:
    """Day3 DDI MLP 학습 파이프라인 entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── Step 1: Config & Seed ──────────────────────────────────
    # TODO S-1: config를 로드하고 seed를 설정하라.
    #   목표: load_yaml()로 config 로드, set_seed()로 시드 고정.
    #   힌트: cfg = load_yaml(config_path)
    #         set_seed(cfg["seed"], cfg["reproducibility"]["deterministic"])
    #   생각: seed 설정은 반드시 데이터 로딩/모델 초기화 전에 해야 한다.
    #         그래야 split 결과, weight 초기화, dropout mask 등이 모두 재현 가능.

    pass  # TODO S-1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Step 2: Data Loading ───────────────────────────────────
    # TODO S-2: Day2 모듈을 사용하여 train/val/test DataLoader를 구성하라.
    #   목표:
    #     (a) CSV 로드 + split_ddi_pairs()로 train/test 인덱스 분리
    #     (b) train 인덱스를 다시 train/val로 분리 (val_fraction 사용)
    #     (c) DDIPairDataset 3개 생성 (train, val, test)
    #     (d) build_ddi_dataloader()로 DataLoader 3개 생성
    #   힌트:
    #     import pandas as pd
    #     from sbml_msc_2026.datasets.ddi_dataset import DDIPairDataset, build_ddi_dataloader
    #     from sbml_msc_2026.datasets.splitter import split_ddi_pairs
    #
    #     df = pd.read_csv(cfg["paths"]["raw_ddi_csv"])
    #     splits = split_ddi_pairs(df, method=..., train_ratio=..., seed=...)
    #
    #     # val split: train 인덱스를 val_fraction 비율로 추가 분리
    #     train_all_idx = splits["train"]
    #     rng = np.random.RandomState(cfg["seed"])
    #     rng.shuffle(train_all_idx)
    #     n_val = int(len(train_all_idx) * cfg["training"]["val_fraction"])
    #     val_idx = train_all_idx[:n_val]
    #     train_idx = train_all_idx[n_val:]
    #     test_idx = splits["test"]
    #
    #     # Dataset 생성 (공통 인자는 config에서)
    #     # DataLoader 생성 (train: shuffle=True, val/test: shuffle=False)
    #
    #   생각: val split을 test에서 떼오면 안 된다!
    #         test set은 "최종 보고 전용"이므로 학습/모델 선택에 관여 불가.
    #         "val은 train에서 떼온다"가 원칙.
    #         dummy data가 매우 작아서 (16행) val/train이 각 1~5행일 수 있지만,
    #         파이프라인 동작 검증이 목적이므로 문제없다.

    pass  # TODO S-2

    logger.info(
        f"Data splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    # ── Step 3: Model ──────────────────────────────────────────
    # TODO S-3: DDI_MLP 모델을 config 설정으로 생성하라.
    #   목표: config에서 hidden_dims, dropout, num_classes를 읽어
    #         DDI_MLP를 생성하고 device로 이동.
    #   힌트:
    #     from sbml_msc_2026.models.mlp import DDI_MLP
    #
    #     # input_dim은 하드코딩하지 않고 실제 데이터에서 추론
    #     input_dim = train_dataset[0][0].shape[0]
    #     # num_classes도 데이터에서 추론 가능
    #     num_classes = len(df[config에서 label 컬럼명].unique())
    #
    #     model = DDI_MLP(
    #         input_dim=input_dim,
    #         hidden_dims=cfg["model"]["hidden_dims"],
    #         num_classes=num_classes,
    #         dropout=cfg["model"]["dropout"],
    #     ).to(device)
    #   생각: input_dim을 config에 하드코딩하면 pair_method 변경 시 불일치 발생.
    #         데이터에서 동적으로 추론하는 것이 robust하다.

    pass  # TODO S-3

    logger.info(f"Model parameters: {model.count_parameters():,}")

    # ── Step 4: Loss & Optimizer ───────────────────────────────
    # TODO S-4: Weighted CrossEntropyLoss와 optimizer를 생성하라.
    #   목표:
    #     (a) train labels에서 class weight를 계산 (compute_class_weights)
    #     (b) weight를 torch.Tensor로 변환 + device 이동
    #     (c) nn.CrossEntropyLoss(weight=...) 생성
    #     (d) Adam optimizer 생성
    #   힌트:
    #     from sbml_msc_2026.trainers.metrics import compute_class_weights
    #
    #     # train labels 접근: train_dataset의 df에서 label 컬럼의 .values
    #     train_labels = train_dataset.df[label 컬럼명].values
    #     weights = compute_class_weights(train_labels, num_classes)
    #     weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    #     criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    #
    #     optimizer = torch.optim.Adam(
    #         model.parameters(),
    #         lr=cfg["training"]["lr"],
    #         weight_decay=cfg["training"]["weight_decay"],
    #     )
    #   생각: weight tensor의 device가 model과 다르면 RuntimeError.
    #         반드시 .to(device)를 확인할 것.
    #         cfg["training"]["class_weight"] == "none"이면 weight=None으로 처리.

    pass  # TODO S-4

    # ── Step 5: Training ───────────────────────────────────────
    # TODO S-5: fit()을 호출하여 학습을 수행하라.
    #   목표: fit() 호출 후 결과를 results 변수에 저장.
    #   힌트:
    #     from sbml_msc_2026.trainers.ddi_trainer import fit
    #
    #     results = fit(
    #         model=model,
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         optimizer=optimizer,
    #         criterion=criterion,
    #         device=device,
    #         num_classes=num_classes,
    #         max_epochs=cfg["training"]["max_epochs"],
    #         patience=cfg["training"]["early_stopping"]["patience"],
    #         checkpoint_dir=cfg["paths"]["checkpoint_dir"],
    #     )
    #   생각: 여기서 test_loader는 넘기지 않는다!
    #         test 평가는 학습 완료 후 별도로 수행해야 한다.
    #         학습 중 test 성능을 보면 → 무의식적으로 test에 맞추게 됨 → data snooping.

    pass  # TODO S-5

    # ── Step 6: Final Test Evaluation ──────────────────────────
    # TODO S-6: best checkpoint를 로드하고 test set에서 최종 평가하라.
    #   목표:
    #     (a) best model state_dict 로드
    #     (b) test_loader로 evaluate() 호출
    #     (c) 결과를 JSON으로 저장 (results_dir / "metrics.json")
    #   힌트:
    #     from sbml_msc_2026.trainers.ddi_trainer import evaluate
    #
    #     ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pt"
    #     model.load_state_dict(torch.load(ckpt_path, map_location=device))
    #     test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    #
    #     # JSON 저장 시 ndarray는 .tolist()로 변환
    #     results_dir = Path(cfg["paths"]["results_dir"])
    #     results_dir.mkdir(parents=True, exist_ok=True)
    #     save_dict = {
    #         "best_epoch": results["best_epoch"],
    #         "test_macro_f1": test_metrics["macro_f1"],
    #         "test_per_class_f1": test_metrics["per_class_f1"].tolist(),
    #         "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    #         "test_classification_report": test_metrics["classification_report"],
    #     }
    #     with open(results_dir / "metrics.json", "w") as f:
    #         json.dump(save_dict, f, indent=2)
    #   생각: 왜 best checkpoint를 다시 로드하는가?
    #         fit() 종료 시점의 model은 마지막 epoch 상태인데,
    #         early stopping으로 이전 epoch이 best일 수 있다.
    #         best checkpoint를 로드해야 정확한 최종 성능을 보고할 수 있다.

    pass  # TODO S-6

    # ── 결과 출력 ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Best epoch: {results['best_epoch']}")
    logger.info(f"Best val macro-F1: {results['best_metrics']['macro_f1']:.4f}")
    logger.info(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Classification Report:\n{test_metrics['classification_report']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Day3 DDI MLP Training")
    parser.add_argument(
        "--config", type=str, default="configs/day3_ddi.yaml",
        help="Path to Day3 config YAML",
    )
    args = parser.parse_args()
    main(config_path=args.config)
