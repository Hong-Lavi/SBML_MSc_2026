"""
Day 3: MLP 모델 단위 테스트

Pipeline 위치: models/mlp.py 검증
검증 대상: forward shape, 파라미터 수, gradient 흐름

TODO 개수: 4개 (TODO TM-1 ~ TM-4)
난이도: ★☆☆
예상 소요: ~15min

[왜 모델 단위 테스트를 먼저 하는가]
  Trainer에 모델을 넣기 전에, 모델 자체가 올바른 shape을 뱉는지 확인한다.
  Forward shape이 틀리면 CrossEntropyLoss에서 차원 불일치 에러가 발생하는데,
  에러 메시지가 모호해서 원인 파악이 어렵다.
  여기서 미리 잡으면 디버깅 시간을 크게 절약할 수 있다.
"""

from __future__ import annotations

import pytest
import torch

from sbml_msc_2026.models.mlp import DDI_MLP


class TestDDI_MLP:
    """DDI_MLP 모델 단위 테스트."""

    # TODO TM-1: forward output shape을 검증하는 테스트를 작성하라.
    #   목표: 임의의 (B, input_dim) 텐서를 모델에 넣고,
    #         출력이 (B, num_classes)인지 assert.
    #   힌트:
    #     def test_forward_shape(self):
    #         model = DDI_MLP(input_dim=..., hidden_dims=[..., ...],
    #                         num_classes=..., dropout=...)
    #         x = torch.randn(B, input_dim)   # dummy input
    #         logits = model(x)
    #         assert logits.shape == (B, num_classes)
    #   생각: input_dim, num_classes는 구체적 숫자를 직접 정하되,
    #         config에서 참조할 필요 없다 (단위 테스트는 config 독립).
    #         여러 (B, input_dim, num_classes) 조합을 parametrize하면 더 robust.

    def test_forward_shape(self):
        pass  # TODO TM-1

    # TODO TM-2: 파라미터 수가 양수인지 검증하는 테스트를 작성하라.
    #   목표: model.count_parameters() > 0 확인.
    #   힌트: 위에서 만든 model 재사용.
    #   생각: count_parameters()가 0을 반환하면
    #         layer 정의가 잘못되었거나 nn.Sequential 구성이 틀린 것.

    def test_count_parameters(self):
        pass  # TODO TM-2

    # TODO TM-3: gradient가 올바르게 흐르는지 검증하라.
    #   목표: forward → loss → backward 후 모든 파라미터에 .grad가 존재하는지 확인.
    #   힌트:
    #     logits = model(x)
    #     loss = logits.sum()  # dummy loss
    #     loss.backward()
    #     for name, param in model.named_parameters():
    #         assert param.grad is not None, f"No grad for {name}"
    #   생각: grad가 None인 파라미터가 있으면
    #         해당 layer가 forward path에 연결되지 않은 것.
    #         nn.Sequential 구성에서 빠뜨린 layer가 있을 수 있다.

    def test_gradient_flow(self):
        pass  # TODO TM-3

    # TODO TM-4: eval 모드에서 출력이 deterministic한지 검증하라.
    #   목표: model.eval() 후 같은 입력에 대해 2회 forward 결과가 동일한지 확인.
    #   힌트:
    #     model.eval()
    #     out1 = model(x)
    #     out2 = model(x)
    #     assert torch.allclose(out1, out2)
    #   생각: train 모드에서는 dropout 때문에 매번 다른 결과가 나올 수 있지만,
    #         eval 모드에서는 dropout이 비활성화되므로 결과가 동일해야 한다.
    #         이 테스트가 실패하면 model.eval()이 dropout에 영향을 못 주고 있는 것.

    def test_eval_deterministic(self):
        pass  # TODO TM-4
