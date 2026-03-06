"""
Day 3: DDI MLP 학습/평가 Trainer

Pipeline 위치: DataLoader + Model → [이 모듈] → checkpoint + metrics + training_log
Input:
    model       : DDI_MLP (nn.Module)
    train_loader: DataLoader → (batch_feat [B, D], batch_label [B])
    val_loader  : DataLoader → (batch_feat [B, D], batch_label [B])
Output:
    best checkpoint  : results/day3/checkpoints/best_model.pt
    metrics dict     : {macro_f1, confusion_matrix, classification_report, ...}
    training_log     : epoch별 train_loss, val_loss, val_macro_f1 기록 (JSON)

TODO 개수: 7개 (TODO T-1 ~ T-7)
난이도: ★★★
예상 소요: ~60min

[핵심 개념: Early Stopping]
  Overfitting 방지를 위한 정규화 기법.
  Validation metric이 patience epoch 연속 개선되지 않으면 학습 중단.

  알고리즘:
    best_score ← -∞  (mode=max 기준)
    counter ← 0
    for epoch = 1, 2, ..., max_epochs:
        train(...)
        score ← evaluate(val_loader)["macro_f1"]
        if score > best_score:
            best_score ← score
            counter ← 0
            save_checkpoint(model)
        else:
            counter ← counter + 1
            if counter ≥ patience:
                STOP

  물리적 비유:
    학습은 "산을 오르는 과정"이다.
    Early stopping은 "정상에 가까워진 것 같으면 멈추는 판단"이다.
    patience가 너무 작으면 → 산 중턱에서 멈춤 (underfitting).
    patience가 너무 크면 → 정상을 지나 반대편으로 내려감 (overfitting).

[왜 val loss가 아닌 macro_f1을 monitor하는가]
  Weighted CE loss는 class weight에 의해 스케일이 바뀌므로
  "loss 값의 절대적 크기"가 모델 품질과 직결되지 않는다.
  macro_F1은 class weight와 무관하게 예측 성능을 직접 반영한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sbml_msc_2026.trainers.metrics import compute_metrics

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    1 epoch 학습을 수행하고 평균 loss를 반환한다.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader → (feat [B, D], label [B])
    optimizer : torch.optim.Optimizer
    criterion : nn.Module (e.g., CrossEntropyLoss)
    device    : torch.device

    Returns
    -------
    avg_loss : float
        epoch 전체의 sample-weighted 평균 loss

    Notes
    -----
    Gradient descent 1 step의 수식:
        θ_{t+1} = θ_t − η · ∇_θ L(θ_t)

    코드에서 이것이 어떻게 대응되는가:
        optimizer.zero_grad()   → ∇_θ 초기화
        loss.backward()         → ∇_θ L 계산 (backpropagation)
        optimizer.step()        → θ_{t+1} = θ_t − η · ∇_θ L
    """
    # TODO T-1: 학습 모드 설정 + epoch loss 추적 변수 초기화.
    #   목표: model을 학습 모드로 전환하고 loss 누적 변수를 초기화.
    #   힌트: model.train()
    #         total_loss = 0.0
    #         n_samples = 0
    #   생각: model.train()을 빼먹으면?
    #         → evaluate()에서 model.eval() 후 복귀하지 않으면
    #           dropout이 꺼진 채로 다음 epoch이 학습됨.
    #           BN도 running stats 모드로 동작 → 학습 불안정.

    pass  # TODO T-1

    # TODO T-2: 배치 순회 + forward/backward/step.
    #   목표: loader의 모든 batch를 순회하며 gradient descent 수행.
    #         batch별 loss를 sample 수로 가중 누적.
    #   힌트:
    #     for feat, label in loader:
    #         feat, label = feat.to(device), label.to(device)
    #         optimizer.zero_grad()
    #         logits = model(feat)              # (B, num_classes)
    #         loss = criterion(logits, label)    # scalar
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item() * feat.size(0)
    #         n_samples += feat.size(0)
    #   생각: loss.item() * feat.size(0)로 누적하는 이유?
    #         마지막 batch가 batch_size보다 작을 수 있다 (drop_last=False).
    #         단순 loss 합산은 마지막 batch의 기여도가 부정확.
    #         sample-weighted 합산 후 total / n_samples로 정확한 평균을 구한다.

    pass  # TODO T-2

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    """
    모델을 평가하고 metrics dict를 반환한다.

    Parameters
    ----------
    model, loader, criterion, device : 학습 함수와 동일
    num_classes : int
        confusion matrix 크기 결정용

    Returns
    -------
    dict with keys:
        "loss"                   : float
        "macro_f1"               : float
        "per_class_f1"           : ndarray (C,)
        "confusion_matrix"       : ndarray (C, C)
        "classification_report"  : str

    Notes
    -----
    @torch.no_grad() 데코레이터:
        evaluate 시 gradient 계산이 불필요하므로 비활성화.
        → 메모리 절약 (activation 저장 불필요) + 속도 향상.
        with torch.no_grad(): 블록과 동일한 효과.
    """
    # TODO T-3: 평가 모드 전환 + 예측/정답 수집 + loss 누적.
    #   목표: model.eval() 호출 후, 모든 batch의 예측과 정답을
    #         리스트에 수집하고 loss를 누적.
    #   힌트:
    #     model.eval()
    #     all_preds: list = []
    #     all_labels: list = []
    #     total_loss = 0.0
    #     n_samples = 0
    #     for feat, label in loader:
    #         feat, label = feat.to(device), label.to(device)
    #         logits = model(feat)                   # (B, num_classes)
    #         loss = criterion(logits, label)         # scalar
    #         preds = logits.argmax(dim=1)            # (B,)
    #         all_preds.append(preds.cpu().numpy())
    #         all_labels.append(label.cpu().numpy())
    #         total_loss += loss.item() * feat.size(0)
    #         n_samples += feat.size(0)
    #   생각: .cpu().numpy()가 왜 필요한가?
    #         sklearn metrics는 numpy array만 받는다.
    #         GPU tensor를 넘기면 TypeError 발생.
    #         argmax(dim=1)은 각 sample에서 가장 높은 logit의 index를 반환.

    pass  # TODO T-3

    # TODO T-4: 수집된 예측/정답을 합쳐서 compute_metrics()를 호출하라.
    #   목표: all_preds, all_labels를 하나의 배열로 합치고
    #         compute_metrics()에 전달하여 결과 dict를 구성.
    #   힌트:
    #     y_true = np.concatenate(all_labels)  # (N,)
    #     y_pred = np.concatenate(all_preds)   # (N,)
    #     metrics = compute_metrics(y_true, y_pred, num_classes)
    #     metrics["loss"] = total_loss / max(n_samples, 1)
    #   생각: concatenate가 아니라 np.hstack도 가능하지만,
    #         1D 배열에서는 동일하게 동작한다.
    #         빈 loader (n_samples=0)에 대한 방어로 max(n_samples, 1) 사용.

    pass  # TODO T-4

    return metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    max_epochs: int,
    patience: int,
    checkpoint_dir: str | Path,
) -> Dict[str, Any]:
    """
    전체 학습 루프: train → val evaluate → early stopping → checkpoint.

    Parameters
    ----------
    patience : int
        val metric이 개선되지 않는 연속 epoch 수. 초과 시 학습 중단.
    checkpoint_dir : Path
        best model checkpoint 저장 경로.

    Returns
    -------
    dict with keys:
        "best_metrics"  : 최고 val metrics dict
        "best_epoch"    : int
        "training_log"  : List[dict] — epoch별 {epoch, train_loss, val_loss, val_macro_f1}
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TODO T-5: early stopping 변수 초기화.
    #   목표: best_score, best_epoch, counter, best_metrics, training_log 초기화.
    #   힌트:
    #     best_score = -float("inf")   # mode=max이므로 -∞에서 시작
    #     best_epoch = 0
    #     counter = 0
    #     best_metrics = {}
    #     training_log: List[dict] = []
    #   생각: mode="min" (loss 모니터) 이라면 best_score = +float("inf").
    #         이 프로젝트에서는 macro_f1 (higher is better)이므로 max.

    pass  # TODO T-5

    # TODO T-6: epoch 루프 + early stopping 로직.
    #   목표: max_epochs 동안 아래를 반복:
    #     1) train_one_epoch() → train_loss
    #     2) evaluate(val_loader) → val_metrics
    #     3) score = val_metrics["macro_f1"]
    #     4) score > best_score이면: best 갱신 + checkpoint 저장 + counter 리셋
    #        아니면: counter 증가 → patience 도달 시 break
    #     5) epoch_log를 training_log에 append
    #   힌트:
    #     for epoch in range(max_epochs):
    #         train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    #         val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
    #         score = val_metrics["macro_f1"]
    #
    #         epoch_log = {
    #             "epoch": epoch,
    #             "train_loss": train_loss,
    #             "val_loss": val_metrics["loss"],
    #             "val_macro_f1": score,
    #         }
    #         training_log.append(epoch_log)
    #
    #         if score > best_score:
    #             best_score = score
    #             best_epoch = epoch
    #             best_metrics = val_metrics
    #             counter = 0
    #             torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
    #             logger.info(f"Epoch {epoch}: new best macro_f1={score:.4f}, saved checkpoint")
    #         else:
    #             counter += 1
    #             logger.info(f"Epoch {epoch}: no improvement ({counter}/{patience})")
    #             if counter >= patience:
    #                 logger.info(f"Early stopping at epoch {epoch}")
    #                 break
    #   생각: checkpoint에 model.state_dict()만 저장하는 것이 관례.
    #         전체 model을 pickle로 저장하면 class 정의가 바뀔 때 로드 실패.
    #         optimizer state도 저장하면 학습 재개가 가능하지만 Day3에서는 생략.

    pass  # TODO T-6

    # TODO T-7: training_log를 JSON으로 저장하라.
    #   목표: training_log를 checkpoint_dir / "training_log.json"에 저장.
    #   힌트:
    #     with open(checkpoint_dir / "training_log.json", "w") as f:
    #         json.dump(training_log, f, indent=2)
    #   생각: training_log의 각 entry에는 scalar만 들어있으므로
    #         json.dump가 바로 동작한다.
    #         confusion_matrix (ndarray)는 best_metrics에만 있고
    #         training_log에는 포함하지 않아야 직렬화 문제가 없다.

    pass  # TODO T-7

    return {
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "training_log": training_log,
    }
