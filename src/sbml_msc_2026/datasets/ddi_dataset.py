"""
===========================================================================
Stage 3: DDI Pair Dataset + DataLoader
===========================================================================

[이 모듈이 하는 일]
  CSV(drug_a_smiles, drug_b_smiles, interaction_type=0,1,2)를 읽어서
  interaction_type이 0,1,2인 의미는?
    0: no interaction
    1: pharmacodynamic interaction (PD)는 약물의 효과에 영향을 주는 상호작용
    2: pharmacokinetic interaction (PK)는 약물의 흡수, 분포, 대사, 배설에 영향을 주는 상호작용
  PyTorch Dataset/DataLoader로 변환한다.
  DataLoader가 배치를 뱉을 때 shape은 [B, D]여야 한다.
  여기서 B는 batch_size, D는 pair feature 차원 (concat이면 4096, sum이면 2048).
  ex) batch_size=256, pair_method="concat" → features shape: [256, 4096], labels shape: [256]
  근데 batch_size가 너무 크면 GPU 메모리가 부족할 수 있다. 적절한 batch_size를 선택해야 한다.
  Batch size란 한 번에 모델에 넣는 데이터 포인트 수다. GPU 메모리와 학습 속도에 영향을 준다.
    DDI 예측 모델의 입력은 "약물 쌍"이므로,
    각 데이터 포인트는 (pair feature, label) 형태로 표현된다.
[왜 필요한가]
    DDI 예측 모델을 학습하려면, 약물 쌍과 상호작용 유형(label)을 모델에 공급할 수 있는 형태로 데이터를 준비해야 한다.
    PyTorch의 Dataset과 DataLoader는 모델 학습에 필요한 데이터 로딩과 배치 생성을 효율적으로 처리하는 도구다.
    Dataset은 데이터셋의 크기와 개별 데이터 포인트에 접근하는 방법을 정의하고,
    DataLoader는 Dataset에서 배치를 생성하고, 학습 루프에서 모델에 공급할 수 있도록 데이터를 준비한다.
    이 모듈에서는 CSV 파일에서 DDI 데이터를 읽어서, 각 약물 쌍에 대해 pair feature를 생성하고, 이를 PyTorch Tensor로 변환하여 모델에 공급할 수 있는 형태로 준비한다.


[왜 별도 모듈인가]
  기존 dataset.py는 "단일 약물 → FP → label" 구조였다.
  DDI는 "약물 쌍 → pair feature → interaction type" 구조로,
  데이터 로딩 로직이 근본적으로 다르다.
  - FP를 매번 계산하지 않고 캐시된 dict에서 lookup한다.
  - pair feature 생성 전략(concat/symmetric 등)을 외부에서 주입받는다.

[데이터 흐름]
  1. CSV 로드 → DataFrame (drug_a_smiles, drug_b_smiles, interaction_type)
  2. 전체 SMILES 추출 → fingerprint.get_or_compute_fp_dict() → fp_dict
  3. __getitem__(idx):
     a. idx번째 row에서 smiles_a, smiles_b, label 추출
     b. fp_dict[smiles_a], fp_dict[smiles_b] 조회 (O(1) lookup)
     c. pair_features.make_pair_feature(fp_a, fp_b, method) → pair_vec
     d. pair_vec → torch.Tensor, label → torch.LongTensor
     e. return (pair_tensor, label_tensor)

  DataLoader가 이 Dataset을 감싸면:
     배치 출력: (features: [B, D], labels: [B])
     - B = batch_size
     - D = pair feature 차원 (concat이면 4096, sum이면 2048)

[OOM 리스크]
  이 설계에서 fp_dict는 메모리에 상주한다.
  약물 10만 개 × 2048 × float32 = ~800MB → 대부분의 머신에서 OK.
  하지만 pair feature를 미리 전부 계산해서 저장하면:
  100만 pairs × 4096 × float32 = ~16GB → OOM 위험.
  → 그래서 pair feature는 __getitem__ 시점에 on-the-fly로 생성한다.

[Data Leakage 주의]
  이 모듈은 split을 하지 않는다. Split은 splitter.py가 담당.
  여기서는 "주어진 인덱스 리스트"에 해당하는 데이터만 서빙한다.
===========================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sbml_msc_2026.features.fingerprint import get_or_compute_fp_dict
from sbml_msc_2026.features.pair_features import make_pair_feature, PairMethod

logger = logging.getLogger(__name__)


class DDIPairDataset(Dataset):
    """
    약물-약물 상호작용(DDI) 쌍 데이터셋.

    __init__에서 CSV를 로드하고 FP dict를 준비한 뒤,
    __getitem__에서 pair feature를 on-the-fly로 생성한다.

    Parameters
    ----------
    csv_path : str or Path
        DDI pair CSV 경로. 컬럼: drug_a_smiles, drug_b_smiles, interaction_type
    fp_cache_path : str or Path
        FP 캐시 .npz 경로
    pair_method : str
        pair feature 합성 전략 ("concat", "sum", "diff", "symmetric")
    fp_radius : int
        Morgan FP 반경
    fp_n_bits : int
        Morgan FP bit 수
    indices : Optional[List[int]]
        사용할 row 인덱스. None이면 전체 사용.
        splitter.py가 train/test 인덱스를 넘겨줄 때 사용.
    """

    def __init__(
        self,
        csv_path: str | Path,
        fp_cache_path: str | Path,
        pair_method: PairMethod = "concat",
        fp_radius: int = 2,
        fp_n_bits: int = 2048,
        indices: Optional[List[int]] = None,
    ) -> None:
        # ----- (a) CSV 로드 -----
        # TODO 6-1: pd.read_csv로 csv_path를 읽어서 self.df에 저장하라.
        #   필수 컬럼: "drug_a_smiles", "drug_b_smiles", "interaction_type"
        #   읽은 후 필수 컬럼이 모두 있는지 assert로 검증하라.
        #
        #   왜 assert인가?
        #   컬럼 이름이 틀리면 이후 모든 단계에서 KeyError가 나는데,
        #   에러 메시지가 원인을 바로 알려주지 않는다.
        #   여기서 일찍 잡아야 디버깅 시간을 아낀다.
        self.df = pd.read_csv(csv_path)  # ← 이 줄을 수정
        required_cols = {"drug_a_smiles", "drug_b_smiles", "interaction_type"}
        assert required_cols.issubset(self.df.columns), f"Missing columns: ..."

        # ----- (b) 인덱스 필터링 -----
        # TODO 6-2: indices가 주어지면 self.df를 해당 인덱스로 필터링하라.
        #   힌트: self.df = self.df.iloc[indices].reset_index(drop=True)
        #
        #   reset_index(drop=True)를 하는 이유:
        #   iloc로 슬라이싱하면 원래 인덱스가 유지되어
        #   __getitem__에서 idx로 접근할 때 혼란이 생긴다.
        # 어떤 혼란이냐면 예를 들어, 원래 df가 1000행이 있고, indices=[10, 20, 30]이면,
        # iloc로 슬라이싱한 후에도 인덱스는 10, 20, 30으로 남아있다. 그런데 __getitem__(0)을 호출하면
        #  idx는 0이지만, df.iloc[0]은 원래 df의 10번째 행이 된다. 그래서 reset_index(drop=True)로 인덱스를 0, 1, 2로 재설정해야 한다.
        #   drop=True는 원래 인덱스를 컬럼으로 보존하지 않겠다는 뜻.
        # 꼭 if indices is not None: 조건으로 감싸서, indices가 None이면 전체 데이터를 사용하도록 해야 하나?
        # 
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        # ----- (c) FP dict 준비 -----
        # TODO 6-3: drug_a_smiles와 drug_b_smiles 컬럼의 모든 SMILES를
        #   합쳐서 하나의 리스트로 만든 후,
        #   get_or_compute_fp_dict()를 호출하여 self.fp_dict에 저장하라.
        #
        #   힌트:
        #   all_smiles = self.df["drug_a_smiles"].tolist() + self.df["drug_b_smiles"].tolist()
        #   self.fp_dict = get_or_compute_fp_dict(all_smiles, fp_cache_path, ...)
        all_smiles=self.df["drug_a_smiles"].tolist() + self.df["drug_b_smiles"].tolist()
        # 왜 sum을 하는가?
        #  - drug_a_smiles와 drug_b_smiles는 각각 약물 A와 B의 SMILES를 담고 있다.
        # - 이 둘을 합쳐야 전체 약물의 SMILES 리스트가 된다.
        
        self.fp_dict: Dict[str, np.ndarray] = get_or_compute_fp_dict(all_smiles, fp_cache_path, fp_radius, fp_n_bits)  # ← 이 줄을 수정
        # 이 구문의 의미는? self.fp.dict의 자료형은 Dict[str, np.ndarray]이다. 
        #즉, 문자열(약물의 SMILES)을 키로 하고, NumPy 배열(해당 약물의 fingerprint)을 값으로 하는 딕셔너리다. 
        # get_or_compute_fp_dict 함수는 주어진 SMILES 리스트에 대해 FP를 계산하거나 캐시에서 불러와서 이 딕셔너리를 반환한다.
        # 만약 get_or_compute_fp_dict가 실패해서 빈 딕셔너리를 반환한다면, 이후 __getitem__에서 모든 SMILES에 대해 fp_dict.get(smiles, zero_fp) 패턴으로 처리할 수 있다.
        # ----- (d) pair method 저장 -----
        self.pair_method = pair_method

        logger.info(
            f"DDIPairDataset: {len(self.df)} pairs, "
            f"{len(self.fp_dict)} unique drugs, "
            f"pair_method={pair_method}"
        )

    def __len__(self) -> int:
        """Dataset의 전체 pair 수를 반환."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        idx번째 약물 쌍의 pair feature와 label을 반환한다.

        Returns
        -------
        pair_tensor : torch.Tensor, shape (D,)
            D = 2*n_bits (concat/symmetric) or n_bits (sum/diff)
        label_tensor : torch.Tensor, shape () — scalar, dtype=int64
        return의 차원은 항상 (D차원 텐서와, 스칼라 텐서)가 된다.
        배치 차원은 DataLoader가 만들어 줄 것이다.
        Notes
        -----
        __getitem__에서 pair feature를 on-the-fly로 생성하는 이유:
        - 전체 pair feature matrix를 미리 만들면 메모리 폭발 위험
        - DataLoader가 num_workers로 병렬 호출하므로 I/O 병목도 적음
        - pair_method를 바꿀 때 캐시를 다시 만들 필요 없음
        """
        # TODO 6-4: self.df에서 idx번째 행의 smiles_a, smiles_b, label을 가져오라.
        #label은 interaction_type 컬럼에서 가져와야 한다.
        row = self.df.iloc[idx]
        smiles_a = row["drug_a_smiles"]  # ← 수정
        smiles_b = row["drug_b_smiles"]  # ← 수정
        label = row["interaction_typed"]      # ← 수정

        # TODO 6-5: self.fp_dict에서 fp_a, fp_b를 lookup하라.
        #   주의: 불량 SMILES로 fp_dict에 없는 경우를 처리해야 한다.
        #   → fp_dict.get(smiles, default_zero_vector) 패턴 사용
        #
        #   왜 KeyError를 그냥 내지 않는가?
        #   DataLoader가 멀티프로세스로 __getitem__을 호출하면
        #   KeyError의 traceback이 매우 읽기 어려워진다.
        #   여기서 default로 처리하고 로그를 남기는 게 디버깅에 유리하다.
        n_bits = next(iter(self.fp_dict.values())).shape[0] if self.fp_dict else 2048
        # 위 코드의 역할은 fp_dict에서 임의의 FP 벡터를 가져와서 그 차원 수(n_bits)를 알아내는 것이다.
        zero_fp = np.zeros(n_bits, dtype=np.float32)
        fp_a = self.fp_dict.get(smiles_a, zero_fp)  # ← 수정: self.fp_dict.get(smiles_a, zero_fp)
        fp_b = self.fp_dict.get(smiles_b, zero_fp)  # ← 수정: self.fp_dict.get(smiles_b, zero_fp)

        # TODO 6-6: make_pair_feature(fp_a, fp_b, self.pair_method)로
        #   pair feature를 생성하라.
        pair_vec = make_pair_feature(fp_a, fp_b, self.pair_method)  # ← 수정

        # TODO 6-7: pair_vec와 label을 torch.Tensor로 변환하여 반환하라.
        #   pair_tensor: torch.float32
        #   label_tensor: torch.int64 (CrossEntropyLoss가 요구하는 dtype)
        #
        #   torch.from_numpy() vs torch.tensor():
        #   - from_numpy: 메모리 공유 (빠르지만 원본 수정 시 위험)
        #   - tensor: 복사 생성 (안전하지만 약간 느림)
        #   DataLoader가 batch를 만들 때 어차피 복사하므로, 여기서는 어느 쪽이든 OK.
        pair_tensor = torch.from_numpy(pair_vec).float()   # ← 수정
        label_tensor = torch.tensor(label, dtype=torch.int64) # ← 수정

        return pair_tensor, label_tensor


# -----------------------------------------------------------------------
# DataLoader 팩토리 함수
# -----------------------------------------------------------------------
def build_ddi_dataloader(
    dataset: DDIPairDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    DDIPairDataset을 감싸는 DataLoader를 생성한다.

    Parameters
    ----------
    dataset : DDIPairDataset
    batch_size : int
        한 번에 모델에 넣을 pair 수.
        GPU 메모리 기준: pair_dim(4096) × batch_size × float32
        batch=256이면 4096 × 256 × 4 = ~4MB → GPU에서 매우 가벼움.
    shuffle : bool
        학습 시 True, 평가 시 False.
    num_workers : int
        데이터 로딩 병렬 프로세스 수.
        0 = 메인 프로세스에서 로딩 (디버깅 시 권장).
        2~4 = 학습 속도 향상 (DataLoader가 미리 배치를 준비).
        너무 높으면 CPU/메모리 경쟁으로 오히려 느려짐.
    pin_memory : bool
        True면 CPU → GPU 전송 시 pinned memory 사용 (전송 속도 향상).
        GPU가 없으면 False.

    Returns
    -------
    DataLoader
        각 배치: (features: [B, D], labels: [B])
    """

    # TODO 7-1: DataLoader를 생성하여 반환하라.
    #   힌트: DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, ...)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)  # ← 이 줄을 수정


# -----------------------------------------------------------------------
# 단독 실행 테스트
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # --- smoke test ---
    csv_path = "data/raw/dummy_ddi_pairs.csv"
    cache_path = "data/processed/fingerprints/dummy_cache.npz"

    print("=" * 60)
    print("DDIPairDataset smoke test")
    print("=" * 60)

    # 전체 데이터로 Dataset 생성
    ds = DDIPairDataset(
        csv_path=csv_path,
        fp_cache_path=cache_path,
        pair_method="concat",
    )
    print(f"Dataset length: {len(ds)}")

    # 첫 번째 아이템 확인
    feat, label = ds[0]
    print(f"First item → feature shape: {feat.shape}, label: {label}")

    # DataLoader 테스트
    loader = build_ddi_dataloader(ds, batch_size=4, shuffle=False)
    for batch_idx, (batch_feat, batch_label) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  features shape: {batch_feat.shape}")  # 기대: [4, 4096]
        print(f"  labels shape:   {batch_label.shape}")  # 기대: [4]
        print(f"  labels:         {batch_label.tolist()}")
        if batch_idx >= 1:
            break

    print("\n✅ DDIPairDataset smoke test passed!")
