"""
Day 3: Trainer + Sanity Overfit 테스트

Pipeline 위치: trainers/ddi_trainer.py + trainers/metrics.py 통합 검증
검증 대상:
  1. Sanity overfit: 1 batch를 반복 학습 → loss가 0에 수렴하는가?
  2. Metrics: compute_metrics() 반환 형식이 올바른가?
  3. Class weights: compute_class_weights() 계산이 수식과 일치하는가?

TODO 개수: 5개 (TODO TT-1 ~ TT-5)
난이도: ★★☆
예상 소요: ~25min

[핵심: Sanity Overfit Test란?]
  "모델이 1개 batch를 완벽하게 외울 수 있는가?"를 검증한다.
  충분한 epoch 동안 같은 batch만 반복 학습하면 loss → 0이어야 한다.

  이것이 실패하면:
    - forward path에 버그가 있거나
    - loss function 설정이 잘못되었거나
    - gradient가 올바르게 흐르지 않거나
    - learning rate가 너무 작은 것.

  물리적 비유:
    "교과서 1페이지를 100번 읽어도 못 외운다면,
     독해 능력 자체에 문제가 있는 것이다."
    → 대규모 학습 전에 이 테스트로 기본 동작을 확인해야 한다.

  주의: 이 테스트는 "올바른 학습"이 아니라 "학습 능력 존재"만 검증한다.
        실제 generalization은 별도 평가가 필요하다.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sbml_msc_2026.models.mlp import DDI_MLP
from sbml_msc_2026.trainers.metrics import compute_metrics, compute_class_weights


class TestSanityOverfit:
    """1 batch sanity overfit 테스트."""

    # TODO TT-1: sanity overfit 테스트를 작성하라.
    #   목표: 작은 dummy data 1 batch를 생성하고,
    #         해당 batch만 반복 학습했을 때 loss가 충분히 작아지는지 검증.
    #   힌트:
    #     (a) dummy data 생성:
    #         input_dim = ...  # 적당한 값
    #         num_classes = ... # 적당한 값
    #         B = ...  # batch size
    #         x = torch.randn(B, input_dim)
    #         y = torch.randint(0, num_classes, (B,))
    #         dataset = TensorDataset(x, y)
    #         loader = DataLoader(dataset, batch_size=B, shuffle=False)
    #
    #     (b) 모델 + 학습 설정:
    #         model = DDI_MLP(input_dim, hidden_dims=[...], num_classes=num_classes, dropout=0.0)
    #         criterion = nn.CrossEntropyLoss()
    #         optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    #         device = torch.device("cpu")
    #
    #     (c) 반복 학습:
    #         from sbml_msc_2026.trainers.ddi_trainer import train_one_epoch
    #         for epoch in range(적절한_epoch_수):
    #             loss = train_one_epoch(model, loader, optimizer, criterion, device)
    #
    #     (d) 검증:
    #         assert loss < 작은_threshold, f"Sanity overfit failed: loss={loss}"
    #
    #   생각: dropout=0.0으로 해야 한다!
    #         dropout이 켜져 있으면 동일 입력이라도 매번 다른 뉴런이 꺼져서
    #         loss가 완전히 0에 수렴하기 어렵다.
    #         lr을 충분히 크게 (1e-2 ~ 1e-1) 설정해야 빠르게 수렴.
    #         seed를 고정해야 테스트가 deterministic.

    def test_sanity_overfit(self):
        pass  # TODO TT-1


class TestMetrics:
    """compute_metrics() 단위 테스트."""

    # TODO TT-2: compute_metrics()의 반환 형식을 검증하라.
    #   목표: 직접 구성한 y_true, y_pred를 넣고
    #         반환 dict에 필수 key들이 존재하고 type이 올바른지 확인.
    #   힌트: np.array()로 num_classes개의 클래스가 포함된 작은 배열 구성.
    #         compute_metrics() 호출 후 다음을 assert:
    #         - "macro_f1" key 존재 + float 타입
    #         - "confusion_matrix" shape == (num_classes, num_classes)
    #         - "classification_report"가 str 타입
    #   생각: 정답값을 직접 계산해서 비교하면 더 엄밀하지만,
    #         여기서는 "형식이 올바른가"를 검증하는 것이 주 목적.

    def test_metrics_format(self):
        pass  # TODO TT-2

    # TODO TT-3: perfect prediction에서 macro_f1 = 1.0인지 검증하라.
    #   목표: y_true == y_pred일 때 macro_f1이 정확히 1.0.
    #   힌트: 모든 class가 포함된 배열을 만들고, 동일한 배열을 y_pred로 전달.
    #         pytest.approx()로 부동소수점 비교.
    #   생각: pytest.approx는 부동소수점 비교에서 미세한 차이를 허용.
    #         == 1.0 대신 approx(1.0)을 써야 안정적.

    def test_perfect_prediction(self):
        pass  # TODO TT-3


class TestClassWeights:
    """compute_class_weights() 단위 테스트."""

    # TODO TT-4: 균등 분포에서 모든 weight가 1.0인지 검증하라.
    #   목표: 각 class가 동일 빈도일 때 w_c = N/(C*N_c) = 1.0.
    #   힌트: num_classes개의 클래스가 균등하게 포함된 labels 배열을 직접 구성.
    #         compute_class_weights() 호출 후 np.testing.assert_allclose()로
    #         모든 weight가 1.0인지 검증.
    #   생각: 이것은 수식의 sanity check다.
    #         균등 분포에서 모든 weight가 동일해야 한다는 건 직관적으로도 맞다.
    #         수식 대입: N=총샘플수, C=클래스수, N_c=각클래스샘플수 → w_c = 1.0.

    def test_uniform_weights(self):
        pass  # TODO TT-4

    # TODO TT-5: 불균등 분포에서 소수 class의 weight가 더 큰지 검증하라.
    #   목표: 소수 class의 weight > 다수 class의 weight.
    #   힌트: 하나의 class가 다수를 차지하고 나머지가 소수인 labels 배열을 구성.
    #         compute_class_weights() 호출 후 소수 class의 weight가
    #         다수 class의 weight보다 큰지 assert.
    #   생각: 수식으로 직접 대입해서 예상 weight를 계산해본 뒤 비교하면 더 엄밀.
    #         w_c = N/(C*N_c)이므로 N_c가 작을수록 w_c가 커진다.
    #         이 직관이 맞는지를 테스트가 확인하는 것.

    def test_imbalanced_weights(self):
        pass  # TODO TT-5
