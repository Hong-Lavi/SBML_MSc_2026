"""
===========================================================================
Stage 4: Data Leakage를 방지하는 Train/Test Split
===========================================================================

[이 모듈이 하는 일]
  DDI pair 데이터를 train/test 인덱스로 분할한다.
  분할 방식을 "random"과 "drug_aware" 중 선택할 수 있다.

[왜 split 로직을 별도 모듈로 분리하는가]
  1. Dataset 클래스는 "데이터 서빙"만 책임져야 한다. (Single Responsibility)
  2. Split 전략을 바꿀 때 Dataset 코드를 건드리지 않아도 된다.
  3. Split 재현성 검증 테스트를 독립적으로 작성할 수 있다.

[핵심 개념: Data Leakage in DDI]

  ★★★ 이것이 Day2에서 가장 중요한 개념이다 ★★★

  DDI에서 pair 단위 random split은 leakage를 일으킨다.

  예시:
    Train set: (약물A, 약물B) → interaction 1
    Test set:  (약물A, 약물C) → interaction 2

    모델이 test에서 약물A를 본 적 없는 것처럼 평가해야 하는데,
    train에서 이미 약물A의 패턴을 학습했다.
    → test 성능이 실제보다 과대평가된다.
    → 논문에 보고하면 reviewer가 바로 잡아낸다.

  Drug-aware split:
    약물 집합 자체를 train/test로 나눈다.
    Train drugs: {A, B, C}
    Test drugs:  {D, E, F}
    → Test set의 pair에는 train에 없는 약물만 포함.
    → "새로운 약물"에 대한 일반화 성능을 정직하게 평가.

  현실적 타협:
    - 완전 cold-start (양쪽 약물 다 새것)은 너무 엄격해서 pair가 부족.
    - warm-start (한쪽만 새것)이 실용적. 하지만 이것도 drug-level split.

  이 구현에서는 단순 drug-level split을 사용한다:
    고유 약물 집합을 train/test로 나누고,
    train drugs끼리만 이루어진 pair → train
    나머지 → test

[데이터 흐름]
  DataFrame(drug_a_smiles, drug_b_smiles, interaction_type)
    → 고유 약물 집합 추출
    → 약물 집합을 train/test로 분할
    → 각 pair가 어느 split에 속하는지 인덱스 리스트로 반환
    → DDIPairDataset(indices=train_indices) / DDIPairDataset(indices=test_indices)
===========================================================================
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SplitMethod = Literal["random", "drug_aware"]


def split_ddi_pairs(
    df: pd.DataFrame,
    method: SplitMethod = "drug_aware",
    train_ratio: float = 0.75,
    seed: int = 123,
) -> Dict[str, List[int]]:
    """
    DDI pair DataFrame을 train/test 인덱스로 분할한다.

    Parameters
    ----------
    df : pd.DataFrame
        컬럼: drug_a_smiles, drug_b_smiles, interaction_type
    method : str
        "random": pair 단위 무작위 분할 (leakage 위험, 비교용)
        "drug_aware": 약물 단위 분할 (leakage 방지)
    train_ratio : float
        train에 할당할 비율 (약물 수 기준 or pair 수 기준, method에 따라)
    seed : int
        재현성을 위한 RNG seed

    Returns
    -------
    Dict with keys "train", "test", each containing List[int] (row indices)

    Notes
    -----
    반환된 인덱스는 df의 원본 인덱스(0-based)다.
    DDIPairDataset(indices=splits["train"])으로 바로 넘길 수 있다.
    """

    rng = np.random.RandomState(seed)
    # 왜 np.random.RandomState인가?
    # np.random.seed()는 global state를 바꿔서 다른 코드에 영향을 줌.
    # RandomState는 독립적인 RNG 인스턴스라서 이 함수 안에서만 영향.

    if method == "random":
        return _random_split(df, train_ratio, rng)
    elif method == "drug_aware":
        return _drug_aware_split(df, train_ratio, rng)
    else:
        raise ValueError(f"Unknown split method: {method}")


def _random_split(
    df: pd.DataFrame,
    train_ratio: float,
    rng: np.random.RandomState,
) -> Dict[str, List[int]]:
    """
    Pair 단위 무작위 분할.

    ⚠️  DDI에서는 leakage 위험이 있다. 비교 실험용으로만 사용.
    실제 평가에서는 drug_aware를 써야 한다.
    """

    # TODO 8-1: 전체 인덱스를 shuffle한 뒤 train_ratio 비율로 분할하라.
    #
    #   (a) indices = np.arange(len(df))
    # np.arrange(len(df))는 0부터 len(df)-1까지의 정수 배열을 생성하는 함수다.
    # 예를 들어, df의 길이가 5라면 np.arange(len(df))는 array([0, 1, 2, 3, 4])를 반환한다
    #   (b) rng.shuffle(indices)  — in-place shuffle
    # rng.shuffle(indices)는 indices 배열을 무작위로 섞는 함수다. 이 함수는 배열 자체를 변경하며, 반환값은 None이다. 예를 들어, indices가 array([0, 1, 2, 3, 4])였다면, shuffle 후에는 array([3, 0, 4, 1, 2])와 같이 순서가 바뀔 수 있다.
    #   (c) n_train = int(len(indices) * train_ratio)
    # n_train은 train set에 할당할 인덱스의 개수를 계산하는 변수다. 예를 들어, len(indices)가 100이고 train_ratio가 0.75라면, n_train은 int(100 * 0.75) = 75가 된다. 즉, 처음 75개의 인덱스는 train set에 할당되고, 나머지 25개는 test set에 할당된다.
    #   (d) train_idx = indices[:n_train].tolist()
    # indices[:n_train]는 shuffle된 indices 배열에서 처음 n_train 개의 요소를 선택하는 슬라이싱 연산이다. 예를 들어, indices가 array([3, 0, 4, 1, 2])이고 n_train이 3이라면, indices[:n_train]는 array([3, 0, 4])가 된다. .tolist()는 numpy 배열을 Python 리스트로 변환하는 메서드다. 따라서 train_idx는 [3, 0, 4]와 같은 Python 리스트가 된다.
    #   (e) test_idx  = indices[n_train:].tolist()
    #   (f) return {"train": train_idx, "test": test_idx}
    #
    #   .tolist()를 하는 이유:
    #   numpy array 인덱스를 pandas iloc에 넘기면 때때로 경고가 뜸.
    #   python list로 변환하면 안전.
    indicies=np.arange(len(df))
    rng.shuffle(indicies)
    n_train = int(len(indicies) * train_ratio)
    train_idx = indicies[:n_train].tolist()
    test_idx = indicies[n_train:].tolist()
    return {"train": train_idx, "test": test_idx}


def _drug_aware_split(
    df: pd.DataFrame,
    train_ratio: float,
    rng: np.random.RandomState,
) -> Dict[str, List[int]]:
    """
    약물 단위 분할 — leakage 방지.

    알고리즘:
      1. df에서 등장하는 모든 고유 약물(SMILES)을 추출
      2. 고유 약물을 shuffle
      3. train_ratio 비율만큼 train drugs, 나머지를 test drugs로 분할
      4. pair의 양쪽 약물이 모두 train drugs에 속하면 → train
         그 외 → test (한쪽이라도 test drug이면 test에 할당)

    이렇게 하면 test set에서 평가하는 pair에는
    train에서 한 번도 보지 못한 약물이 최소 하나 포함된다.
    """

    # TODO 8-2: 고유 약물 집합을 추출하라.
    #   힌트: set(df["drug_a_smiles"]) | set(df["drug_b_smiles"])
    #   → list로 변환 후 정렬 (재현성을 위해 순서 고정)
    all_drugs = sorted(set(df["drug_a_smiles"]) | set(df["drug_b_smiles"]))  # ← 이 줄을 수정

    # TODO 8-3: all_drugs를 rng.shuffle로 섞고 train/test로 분할하라.
    #   n_train_drugs = int(len(all_drugs) * train_ratio)
    #   train_drugs = set(all_drugs[:n_train_drugs])
    #   test_drugs  = set(all_drugs[n_train_drugs:])
    rng.shuffle(all_drugs)
    n_train_drugs = int(len(all_drugs) * train_ratio)
    train_drugs = set(all_drugs[:n_train_drugs])
    test_drugs  = set(all_drugs[n_train_drugs:])
    # test_drugs는 명시적으로 만들지 않아도 됨. train에 없으면 test.

    logger.info(
        f"Drug-aware split: {len(train_drugs)} train drugs, "
        f"{len(all_drugs) - len(train_drugs)} test drugs"
    )

    # TODO 8-4: 각 pair를 순회하며 train/test 인덱스를 할당하라.
    #   조건: 양쪽 약물 모두 train_drugs에 속할 때만 train.
    #
    #   for i, row in df.iterrows():
    # df.iterrows()는 pandas DataFrame의 각 행을 반복하는 메서드다. 이 메서드는 (index, Series) 튜플을 반환한다. 예를 들어, df가 다음과 같다면:
    #   drug_a_smiles  drug_b_smiles  interaction_type
    # 0  C1=CC=CC=C1  C2=CC=
    # 1  C3=CC=CC=C3  C4=CC=CC=C4
    # df.iterrows()를 사용하면 첫 번째 반복에서 i는 0이 되고 row는 Series로서 drug_a_smiles가 "C1=CC=CC=C1", drug_b_smiles가 "C2=CC=CC=C2", interaction_type이 1이 된다. 두 번째 반복에서는 i가 1이 되고 row는 drug_a_smiles가 "C3=CC=CC=C3", drug_b_smiles가 "C4=CC=CC=C4", interaction_type이 0이 된다.
    #       a_in_train = row["drug_a_smiles"] in train_drugs
    #       b_in_train = row["drug_b_smiles"] in train_drugs
    #       if a_in_train and b_in_train:
    #           train_indices.append(i)
    #       else:
    #           test_indices.append(i)
    train_indices: List[int] = []
    test_indices: List[int] = []
    for i, row in df.iterrows():  # ← 이 블록을 구현
        a_in_train = row["drug_a_smiles"] in train_drugs
        b_in_train = row["drug_b_smiles"] in train_drugs
        if a_in_train and b_in_train:
            train_indices.append(i)
        else:
            test_indices.append(i)

    logger.info(
        f"Split result: {len(train_indices)} train pairs, "
        f"{len(test_indices)} test pairs"
    )

    # --- Leakage 검증 ---
    # TODO 8-5: (선택) train과 test 사이에 공유되는 약물이 있는지 확인하라.
    #   train set에 등장하는 약물 집합과 test set에 등장하는 약물 집합의
    #   교집합을 구하여 로그로 출력하라.
    #   Drug-aware split에서는 test pair의 최소 한 쪽 약물이
    #   train에 없어야 하므로, 완전 교집합은 아닐 수 있다.
    #   (warm-start 상황에서는 부분적 overlap 허용)

    return {"train": train_indices, "test": test_indices}



# -----------------------------------------------------------------------
# 단독 실행 테스트
# -----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    csv_path = "data/raw/dummy_ddi_pairs.csv"
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print(f"Total pairs: {len(df)}")
    print("=" * 60)

    for method in ["random", "drug_aware"]:
        splits = split_ddi_pairs(df, method=method, train_ratio=0.75, seed=123)
        print(f"\n--- {method} ---")
        print(f"  Train indices: {splits['train']}")
        print(f"  Test  indices: {splits['test']}")
        print(f"  Train size: {len(splits['train'])}, Test size: {len(splits['test'])}")

        # Leakage 체크 (약물 단위)
        train_drugs = set(df.iloc[splits["train"]]["drug_a_smiles"]) | \
                      set(df.iloc[splits["train"]]["drug_b_smiles"])
        test_drugs  = set(df.iloc[splits["test"]]["drug_a_smiles"]) | \
                      set(df.iloc[splits["test"]]["drug_b_smiles"])
        overlap = train_drugs & test_drugs
        print(f"  Drug overlap between train/test: {len(overlap)} drugs")
        if method == "drug_aware" and overlap:
            print(f"  ⚠️  Overlapping drugs (warm-start): {overlap}")


# Overlap은 왜 생기는가?
# - 완전 cold-start (양쪽 약물 다 새것)은 너무 엄격해서 pair가 부족할 수 있다.
# - warm-start (한쪽만 새것)이 실용적이므로, 약물 단위로 완전히 분리하지 않고 일부 overlap을 허용하는 경우가 있다.