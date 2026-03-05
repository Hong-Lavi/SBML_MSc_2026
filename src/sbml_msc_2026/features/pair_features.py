"""
===========================================================================
Stage 2: Pair Feature 생성
===========================================================================

[이 모듈이 하는 일]
  개별 약물의 fingerprint(fp_a, fp_b)를 받아서
  약물 "쌍(pair)"을 표현하는 하나의 feature vector를 생성한다.
  이 때 , fingerprint는 앞선 fingerprint.py 모듈에서 계산된 2048차원 벡터이다.
  fingerprint.py에서 한 일은 "개별 약물의 특징을 벡터로 표현"하는 것이고,
    이 모듈에서 하는 일은 "두 약물의 특징을 하나의 벡터로 합치는 것"이다.
[왜 필요한가]
  DDI 예측 모델의 입력은 "약물 쌍"이다.
  MLP 같은 모델에 넣으려면 두 약물의 정보를 하나의 고정 길이 벡터로
  합쳐야 한다. 이때 "어떻게 합치느냐"가 성능과 대칭성에 직접 영향을 준다.

[합치는 전략들과 그 의미]

  1) concat(fp_a, fp_b) → shape: (2 * n_bits,) = (4096,)
     가장 단순. fp_a를 앞, fp_b를 뒤에 이어붙임.
     문제: concat(A,B) ≠ concat(B,A)
     → (아스피린, 카페인)과 (카페인, 아스피린)이 다른 입력이 됨.
     → DDI는 본질적으로 대칭(A와 B의 상호작용 = B와 A의 상호작용)이므로
       이 비대칭성이 노이즈가 된다.

  2) sum(fp_a, fp_b) → shape: (n_bits,)
     원소별 덧셈. 자동으로 대칭: sum(A,B) = sum(B,A).
     문제: 정보 손실이 크다. A와 B가 동일한 substructure를 공유하면
     그 차이가 사라진다 (1+1=2, 0+1=1 → 어느 쪽이 가진 건지 모름).

  3) diff(fp_a, fp_b) = |fp_a - fp_b| → shape: (n_bits,)
     절대값 차이. 대칭이고, 두 약물의 "구조적 차이"에 집중.
     문제: 공통 구조 정보가 사라진다.
     예시는 sum과 반대. A와 B가 동일한 substructure를 공유하면
        그 부분은 0이 되어 버린다 (1-1=0) 

  4) symmetric concat: concat(sorted(fp_a, fp_b)) → shape: (2 * n_bits,)
     두 FP를 정렬 기준(예: 벡터 합계 or SMILES 사전순)으로 순서를 정하고
     항상 같은 순서로 concat. 대칭 + 정보 보존.
     DeepDDI 논문에서 사용한 방식에 가까움.

[이 파이프라인에서의 선택]
  config의 pair_feature.method로 전략을 선택한다.
  Day2에서는 concat과 symmetric을 구현하고, Day4에서 augmentation과 비교한다.
  config는 뭘 하는것인가?
    - config는 모델 학습과 평가에 필요한 모든 설정값을 담는 객체다.
    매 실험마다 다르게 할 수 있는 하이퍼파라미터나 옵션들을 config로 관리한다. 
    이렇게 하면 코드가 더 깔끔해지고, 실험 설정을 쉽게 변경할 수 있다.
    ex) pair_feature.method, fingerprint.n_bits, training.batch_size 등.
[출력 shape 정리]
  method      | 출력 shape
  ------------|------------------
  concat      | (2 * n_bits,) = (4096,)  ← n_bits=2048 기준
  sum         | (n_bits,)    = (2048,)
  diff        | (n_bits,)    = (2048,)
  symmetric   | (2 * n_bits,) = (4096,)
===========================================================================
"""

from __future__ import annotations
# __future__ import는 Python 2에서 Python 3의 기능을 사용할 수 있게 해주는 문법이지만, Python 3.7 이상에서는 from __future__ import annotations가 기본적으로 활성화되어 있다. 
# 이 구문은 함수 어노테이션에서 타입 힌트를 문자열로 처리하여, 순환 참조 문제를 해결하는 데 도움을 준다. 따라서 이 프로젝트가 Python 3.7 이상을 대상으로 한다면, 이 구문은 없어도 무방하다.
import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# 지원하는 pair feature 생성 전략
PairMethod = Literal["concat", "sum", "diff", "symmetric"]


def make_pair_feature(
    fp_a: np.ndarray,
    #np.array와 np.ndarray의 차이점: np.array는 NumPy 라이브러리에서 제공하는 배열 객체로, 동일한 타입의 요소를 담으며 고정된 크기를 가집니다. 
    #np.ndarray는 np.array의 공식 이름으로, NumPy에서 배열을 나타내는 기본 클래스입니다. 일반적으로 np.array라는 이름으로 사용되지만, 실제로는 np.ndarray가 그 구현체입니다.
    fp_b: np.ndarray,
    method: PairMethod = "concat",
) -> np.ndarray:
    """
    두 약물의 fingerprint를 합쳐 하나의 pair feature vector를 생성한다.

    Parameters
    ----------
    fp_a : np.ndarray, shape (D,)
        약물 A의 fingerprint. D = n_bits (예: 2048)
    fp_b : np.ndarray, shape (D,)
        약물 B의 fingerprint.
    method : str
        합치는 전략. "concat" | "sum" | "diff" | "symmetric"

    Returns
    -------
    np.ndarray
        pair feature vector.
        - concat/symmetric → shape (2D,)
        - sum/diff          → shape (D,)

    Raises
    ------
    ValueError
        fp_a, fp_b의 shape이 다르거나, 지원하지 않는 method일 때.
    """

    # --- 입력 검증 ---
    # 연구 코드에서 shape mismatch는 흔한 버그.
    # 여기서 잡지 않으면 모델 학습 중에 cryptic한 에러가 난다.
    if fp_a.shape != fp_b.shape:
        raise ValueError(
            f"Shape mismatch: fp_a={fp_a.shape}, fp_b={fp_b.shape}. "
            f"두 FP의 n_bits 설정이 동일한지 확인하라."
        )

    if method == "concat":
        # TODO 5-1: np.concatenate로 fp_a, fp_b를 이어붙여 반환하라.
        #   예상 shape: (2 * D,)
        #   힌트: np.concatenate([fp_a, fp_b])
        return np.concatenate([fp_a, fp_b]) 
    # 이 문법에서 np.concatenate는 리스트 형태로 여러 배열을 받아서 하나의 배열로 이어붙이는 함수다. 
    # [fp_a, fp_b]는 fp_a와 fp_b를 담은 리스트로, 이 두 배열이 순서대로 이어붙여진 결과가 반환된다. 
    # 따라서 concat 전략에서는 fp_a가 앞에, fp_b가 뒤에 붙어서 (2 * D,) 형태의 벡터가 만들어진다.
    # 원래 fp_a의 자료형은 np.ndarray이고, np.concatenate의 결과도 np.ndarray이므로, 반환값의 자료형은 np.ndarray가 된다.


    elif method == "sum":
        # TODO 5-2: 원소별 합 (fp_a + fp_b)을 반환하라.
        #   예상 shape: (D,)
        return fp_a+fp_b  # ← 이 줄을 수정

    elif method == "diff":
        # TODO 5-3: 원소별 절대값 차이 |fp_a - fp_b|를 반환하라.
        #   예상 shape: (D,)
        #   힌트: np.abs(fp_a - fp_b)
        return abs(fp_a-fp_b)  # ← 이 줄을 수정

    elif method == "symmetric":
        # TODO 5-4: 대칭 concat을 구현하라.
        #   (1) 두 벡터의 합(sum)을 비교하여 순서를 결정한다.
        #       → fp_a.sum() <= fp_b.sum() 이면 (fp_a, fp_b) 순서
        #       → 아니면 (fp_b, fp_a) 순서
        #   (2) 결정된 순서로 np.concatenate 한다.
        #
        #   왜 sum 기준인가?
        #   SMILES 문자열 사전순보다 수치 기준이 안정적이다.
        #   동일 분자도 SMILES 표현이 여러 개 (canonical vs non-canonical)
        #   있을 수 있지만, FP는 canonical SMILES에서 계산하므로
        #   sum 값은 유일하게 결정된다.
        #
        #   Edge case: fp_a.sum() == fp_b.sum()이면?
        #   → 순서가 어느 쪽이든 상관없다. 동일 pair이므로.
        #
        #   예상 shape: (2 * D,)
        if fp_a.sum() <= fp_b.sum():
            return np.concatenate([fp_a, fp_b])
        else:
            return np.concatenate([fp_b, fp_a])

    else:
        raise ValueError(f"Unknown pair method: {method}. "
                         f"Supported: concat, sum, diff, symmetric")


# -----------------------------------------------------------------------
# 단독 실행 테스트
# -----------------------------------------------------------------------
if __name__ == "__main__":
    D = 8  # 테스트용 작은 차원
    a = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
    b = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)

    for method in ["concat", "sum", "diff", "symmetric"]:
        result = make_pair_feature(a, b, method=method)
        print(f"{method:12s} → shape={result.shape}, values={result}")

    # 대칭성 검증: symmetric이면 (a,b) == (b,a) 여야 한다
    sym_ab = make_pair_feature(a, b, method="symmetric")
    sym_ba = make_pair_feature(b, a, method="symmetric")
    assert np.array_equal(sym_ab, sym_ba), "Symmetric pair feature is NOT symmetric!"
    print("✅ Symmetry test passed!")

    # concat은 대칭이 아님을 확인
    cat_ab = make_pair_feature(a, b, method="concat")
    cat_ba = make_pair_feature(b, a, method="concat")
    if not np.array_equal(cat_ab, cat_ba):
        print("⚠️  concat is NOT symmetric (expected behavior)")
