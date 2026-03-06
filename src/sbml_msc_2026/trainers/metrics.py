"""
Day 3: Research-grade 평가 메트릭 모듈

Pipeline 위치: Trainer.evaluate() → [이 모듈] → metrics dict → JSON 저장
Input:  y_true (N,), y_pred (N,) — numpy arrays of true/predicted labels
Output: dict {macro_f1, confusion_matrix, classification_report, per_class_f1}

TODO 개수: 6개 (TODO E-1 ~ E-6)
난이도: ★★☆
예상 소요: ~20min

[핵심 개념: macro-F1]
  각 class c에 대해:
    precision_c = TP_c / (TP_c + FP_c)
    recall_c    = TP_c / (TP_c + FN_c)
    F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)

  macro-F1 = (1/C) * sum_{c=1}^{C} F1_c

  물리적 비유:
    micro-F1은 "전체 정답률"과 비슷해서 다수 class가 지배한다.
    macro-F1은 "각 class를 동등하게 평가하는 민주적 평균"이다.
    class imbalance가 심할수록 macro와 micro의 괴리가 커진다.
    → 논문에서 macro-F1을 보고하면 소수 class도 잘 맞추는지 드러난다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    분류 평가 메트릭을 계산하여 dict로 반환한다.

    Parameters
    ----------
    y_true : ndarray, shape (N,)
        정답 레이블
    y_pred : ndarray, shape (N,)
        예측 레이블 (argmax 후)
    num_classes : int, optional
        클래스 수. None이면 y_true에서 추론.

    Returns
    -------
    dict with keys:
        "macro_f1"               : float
        "per_class_f1"           : ndarray, shape (C,)
        "confusion_matrix"       : ndarray, shape (C, C)
        "classification_report"  : str (sklearn 형식)
    """
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1

    # TODO E-1: sklearn.metrics에서 필요한 함수들을 import하라.
    #   목표: f1_score, confusion_matrix, classification_report를 가져오기.
    #   힌트: from sklearn.metrics import f1_score, confusion_matrix, classification_report
    #   생각: import를 함수 내부에 두는 이유?
    #         → 모듈 전체의 import 시간을 줄이고,
    #           sklearn이 없는 환경에서 다른 함수는 사용 가능하게 함.
    #         → 다만 실무에서는 파일 상단 import가 더 일반적.
    #           여기서는 학습 목적으로 내부 import 패턴을 연습.

    pass  # TODO E-1

    # TODO E-2: macro-F1을 계산하라.
    #   목표: 전체 class의 F1을 구하고 단순 평균 (macro).
    #   힌트: f1_score(y_true, y_pred, average="macro", zero_division=0)
    #   생각: zero_division=0 → 해당 class에 예측/정답이 없으면 F1=0 처리.
    #         이걸 안 넣으면 UndefinedMetricWarning이 발생한다.

    macro_f1 = ...  # TODO E-2

    # TODO E-3: per-class F1을 계산하라.
    #   목표: 각 class별 F1 score를 ndarray로 반환.
    #   힌트: f1_score(y_true, y_pred, average=None, zero_division=0)
    #   생각: average=None → class별 F1을 배열로 반환.
    #         "어떤 interaction type을 잘/못 맞추는지" 분석의 출발점.

    per_class_f1 = ...  # TODO E-3

    # TODO E-4: confusion matrix를 계산하라.
    #   목표: (C, C) 행렬. cm[i][j] = "실제 i인데 j로 예측한 횟수".
    #   힌트: confusion_matrix(y_true, y_pred, labels=range(num_classes))
    #   생각: labels 인자를 명시하는 이유?
    #         → y_pred에 특정 class가 한 번도 안 나올 수 있다.
    #           labels를 안 주면 해당 class 행/열이 누락되어
    #           matrix shape이 (C, C)보다 작아진다.

    cm = ...  # TODO E-4

    # TODO E-5: classification_report를 생성하라.
    #   목표: precision, recall, F1, support를 class별로 보여주는 문자열.
    #   힌트: classification_report(y_true, y_pred, zero_division=0)
    #   생각: 이 리포트를 로그에 저장하면 논문 appendix에 바로 쓸 수 있다.
    #         output_dict=True로 하면 dict로도 받을 수 있다.

    report = ...  # TODO E-5

    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Class imbalance 보정을 위한 가중치를 계산한다.

    수식:
        w_c = N_total / (C * N_c)

        N_total : 전체 샘플 수
        C       : 클래스 수
        N_c     : 클래스 c의 샘플 수

    물리적 비유:
        저울에서 가벼운 쪽에 추를 더 올리는 것과 같다.
        loss 함수가 "모든 class를 동등하게 중요하게" 취급하도록 보정.
        N_c가 작은 소수 class → w_c가 커짐 → loss 기여도 증가.

    Parameters
    ----------
    labels : ndarray, shape (N,)
        학습 데이터의 레이블 배열 (0-indexed 정수)
    num_classes : int
        전체 클래스 수

    Returns
    -------
    weights : ndarray, shape (C,), dtype=float64
        각 class의 가중치. torch.nn.CrossEntropyLoss(weight=...)에 전달용.

    Notes
    -----
    - N_c = 0인 class가 있으면 division by zero.
      → 해당 class weight를 0으로 설정하거나 epsilon을 더한다.
    - sklearn.utils.class_weight.compute_class_weight도 같은 역할이지만,
      직접 구현하면 내부 동작을 정확히 이해할 수 있다.
    """
    # TODO E-6: 위 수식대로 class weight를 계산하라.
    #   목표: labels에서 각 class의 빈도를 세고, w_c = N / (C * N_c)로 가중치 계산.
    #   힌트:
    #     counts = np.bincount(labels, minlength=num_classes)
    #     → counts[c] = class c의 샘플 수
    #     N = len(labels)
    #     weights = N / (num_classes * counts)
    #   생각:
    #     - bincount는 음수 입력에서 ValueError. labels가 0-indexed인지 확인 필수.
    #     - counts에 0이 있으면 inf 발생. np.where(counts > 0, ..., 0.0)로 안전 처리.
    #     - 또는 counts = np.maximum(counts, 1)로 clip하는 방법도 있다.

    pass  # TODO E-6
