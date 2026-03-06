"""
Day 3: DDI 예측용 Multi-Layer Perceptron (MLP)

Pipeline 위치: DataLoader → [이 모듈] → Trainer
Input:  batch_feat (B, input_dim) — pair feature tensor
Output: logits (B, num_classes) — raw class scores (softmax 전)

TODO 개수: 6개 (TODO M-1 ~ M-6)
난이도: ★★☆
예상 소요: ~30min

[왜 MLP인가]
  DDI pair feature는 두 약물의 Morgan FP를 concat한 고정 길이 벡터다.
  입력이 고정 차원 tabular feature이므로 MLP가 자연스러운 baseline이다.
  CNN은 공간/시계열 구조, RNN은 순서 구조가 있을 때 유리하지만,
  fingerprint concat에는 그런 구조가 없다.

[Architecture]
  input (B, D) → [Linear → BN → ReLU → Dropout] × L → Linear → logits (B, C)
  - BatchNorm: 각 layer 출력 분포를 정규화 → 학습 안정성 향상
  - Dropout: 뉴런을 확률적으로 비활성화 → 과적합 방지
  - 마지막 layer에는 softmax 없음 (CrossEntropyLoss가 내부 처리)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DDI_MLP(nn.Module):
    """
    DDI 예측용 MLP.

    Parameters
    ----------
    input_dim : int
        pair feature 차원 (D). pair_method에 따라 결정.
    hidden_dims : List[int]
        각 hidden layer의 출력 차원. 예: [512, 128] → 2-layer hidden.
    num_classes : int
        DDI interaction type 수 (C).
    dropout : float
        Dropout 확률. 0이면 dropout 비활성화.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # TODO M-1: hidden layers를 nn.Sequential로 구성하라.
        #   목표: input_dim → hidden_dims[0] → ... → hidden_dims[-1] 순서로
        #         Linear → BatchNorm1d → ReLU → Dropout 블록을 반복 쌓기.
        #   힌트: layers: list = [] 에 nn.Linear, nn.BatchNorm1d, nn.ReLU,
        #         nn.Dropout을 순서대로 append한 뒤 nn.Sequential(*layers).
        #         첫 layer의 in_features는 input_dim,
        #         이후 layer는 이전 hidden_dim이 in_features가 된다.
        #   생각: BatchNorm을 왜 넣는가?
        #         → Internal Covariate Shift 완화: 각 layer 입력 분포가
        #           학습 중 계속 바뀌면 수렴이 느려진다.
        #           BN이 평균=0, 분산=1로 정규화하여 안정화.
        #         → eval 모드에서는 running mean/var 사용. model.eval() 필수.
        #   shape 추적:
        #     Layer 0: (B, input_dim) → (B, hidden_dims[0])
        #     Layer k: (B, hidden_dims[k-1]) → (B, hidden_dims[k])

        self.hidden = ...  # TODO M-1

        # TODO M-2: 마지막 hidden → num_classes로 매핑하는 출력 layer 정의.
        #   목표: hidden_dims[-1] → num_classes Linear layer 하나.
        #   힌트: nn.Linear(in_features, out_features)
        #   생각: 여기에 softmax를 넣지 않는 이유?
        #         CrossEntropyLoss = log_softmax + NLLLoss 를 내부적으로 수행.
        #         모델이 softmax를 또 하면 이중 적용 → gradient가 약해진다.

        self.output_layer = ...  # TODO M-2

        # TODO M-3: 모든 Linear layer에 Xavier uniform 초기화를 적용하라.
        #   목표: self.apply(self._init_weights) 호출.
        #   힌트: nn.Module.apply()는 모든 sub-module에 재귀적으로 함수를 적용.
        #   생각: 왜 Xavier?
        #         Xavier: Var(output) = Var(input)이 되도록 설계.
        #         fan_in, fan_out을 모두 고려 → sigmoid/tanh에 최적.
        #         He init은 ReLU에 최적이지만, BN이 있으면 초기화 민감도가 낮아져서
        #         Xavier도 잘 동작한다. 실무에서는 둘 다 실험하고 비교하는 것이 정석.

        pass  # TODO M-3

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Xavier uniform 초기화. apply()에서 호출됨."""
        # TODO M-4: module이 nn.Linear인 경우에만 초기화를 적용하라.
        #   힌트: isinstance(module, nn.Linear)
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None: nn.init.zeros_(module.bias)
        #   생각: bias=False인 Linear도 있을 수 있으므로
        #         module.bias is not None 체크 필수.

        pass  # TODO M-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, input_dim)
            pair feature batch

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
            raw class scores
        """
        # TODO M-5: hidden → output_layer 순서로 forward를 구성하라.
        #   목표: x를 hidden에 통과시키고, 결과를 output_layer에 넘겨 logits 반환.
        #   힌트: h = self.hidden(x)  # (B, hidden_dims[-1])
        #         logits = self.output_layer(h)  # (B, num_classes)
        #   생각: nn.Sequential이 내부 layer를 순차 실행하므로 2줄이면 끝.

        pass  # TODO M-5

    def count_parameters(self) -> int:
        """학습 가능한 파라미터 총 수를 반환한다."""
        # TODO M-6: requires_grad=True인 파라미터의 numel() 합을 반환하라.
        #   힌트: sum(p.numel() for p in self.parameters() if p.requires_grad)
        #   생각: 이 값을 로그에 기록하면 모델 크기를 추적할 수 있다.
        #         논문 Table에 "# params" 항목으로 자주 보고된다.

        pass  # TODO M-6
