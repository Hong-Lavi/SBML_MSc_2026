"""
===========================================================================
Stage 1: Fingerprint 계산 + 디스크 캐싱
===========================================================================

[이 모듈이 하는 일]
  SMILES 문자열 → RDKit Mol 객체 → Morgan Fingerprint (bit vector)
  그리고 계산된 FP를 .npz 파일로 캐싱하여 재실행 시 RDKit 연산을 스킵한다.

[왜 필요한가]
  DDI 파이프라인에서 동일한 약물이 수십~수백 개의 pair에 반복 등장한다.
  예: 약물 A가 100개의 pair에 포함되면, FP를 100번 계산하는 건 낭비.
  → 약물 단위로 FP를 한 번 계산 → dict에 저장 → .npz로 디스크 캐시
  → 다음 실행부터는 캐시를 로드하여 O(1)로 접근.

[데이터 흐름]
  CSV 컬럼("drug_a_smiles", "drug_b_smiles")
    → 고유 SMILES 집합 추출 (set)
    → 각 SMILES에 대해 Morgan FP 계산
    → {smiles_string: np.ndarray(2048,)} dict
    → np.savez_compressed("cache.npz", **dict)  # 디스크 캐시

[핵심 개념: Morgan Fingerprint]
  원자 주변 반경 r 내의 substructure를 해싱하여 고정 길이 bit vector로 인코딩.
  radius=2이면 각 원자에서 2-hop 이웃까지의 원형 환경(circular environment)을
  해시 함수로 bit 위치에 매핑한다. ECFP4와 동일한 알고리즘이다.
  nBits=2048이면 2048개의 bit 중 해당 substructure가 존재하는 위치만 1로 켜진다.
  → 결국 분자의 구조적 특징을 길이 2048의 binary vector로 압축한 것.

[메모리 고려]
  약물 10만 개 × 2048 bits → float32 기준 약 800MB.
  이 프로젝트에서는 dummy 데이터(6개 약물)이므로 문제없지만,
  실제 데이터에서는 bit vector를 np.packbits로 압축하거나
  sparse matrix로 저장하는 것을 고려해야 한다.
===========================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# 1) 단일 SMILES → Morgan FP numpy array
# -----------------------------------------------------------------------
def smiles_to_morgan_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """
    하나의 SMILES 문자열을 Morgan Fingerprint numpy array로 변환한다.

    Parameters
    ----------
    smiles : str
        분자의 SMILES 표현. 예: "CCO" (에탄올), "C1=CC=CC=C1" (벤젠)
    radius : int
        Morgan FP의 원자 이웃 탐색 반경. 2이면 ECFP4와 동등.
    n_bits : int
        해싱된 bit vector의 길이. 2048이 DDI 논문에서 가장 흔한 설정.

    Returns
    -------
    np.ndarray of shape (n_bits,) dtype=float32  — 성공 시
    None                                          — 파싱 실패 시

    Notes
    -----
    - Chem.MolFromSmiles()가 None을 반환하면 불량 SMILES다.
      실제 데이터에서 0.1~5% 정도 불량이 섞여있는 것이 일반적.
    - 불량 SMILES를 무시하면 안 되고, 로그를 남겨서 나중에 추적해야 한다.
    """
    # TODO 1-1: RDKit으로 SMILES를 Mol 객체로 파싱하라.
    #   힌트: Chem.MolFromSmiles(smiles)
    #   불량이면 None이 반환된다.
    mol = Chem.MolFromSmiles(smiles)

    # TODO 1-2: mol이 None이면 경고 로그를 남기고 None을 반환하라.
    #   힌트: logger.warning(f"Invalid SMILES: {smiles}")
    if mol is None:
        logger.warning(f"Invalid SMILES: {smiles}")
        return None


    # TODO 1-3: AllChem.GetMorganFingerprintAsBitVect()로 FP를 생성하라.
    #   파라미터: mol, radius, nBits=n_bits
    #   반환값: RDKit ExplicitBitVect 객체
    fp_bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    # TODO 1-4: RDKit BitVect → numpy array 변환
    #   (a) np.zeros((n_bits,), dtype=np.float32)로 빈 배열 생성
    #   (b) DataStructs.ConvertToNumpyArray(fp_bitvect, arr)로 채움
    #   (c) arr를 반환
    #
    #   왜 이렇게 두 단계로 하는가?
    #   RDKit의 BitVect는 자체 C++ 객체라서 numpy로 직접 캐스팅이 안 된다.
    #   ConvertToNumpyArray가 내부적으로 비트를 풀어서 numpy 배열에 복사한다.
    arr = np.zeros((n_bits,), dtype=np.float32)
    # ← 여기에 ConvertToNumpyArray 호출 추가
    DataStructs.ConvertToNumpyArray(fp_bitvect, arr)
    return arr


# -----------------------------------------------------------------------
# 2) 고유 SMILES 집합에 대해 일괄 FP 계산
#  - 입력: 전체 SMILES 리스트 (중복 포함)
#  - 출력: {smiles: fp_array} dict (고유 SMILES만, 불량 SMILES는 제외)
#  - 로그: 고유 SMILES 수, FP 계산 성공/실패 수
#  - 실제 데이터에서는 고유 약물이 수천 개 이상이 될 수 있으므로, FP 계산은 가볍지만 캐싱이 중요하다.

# -----------------------------------------------------------------------
def compute_fp_dict(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> Dict[str, np.ndarray]:
    """
    SMILES 리스트에서 중복을 제거하고, 각 고유 SMILES에 대해
    Morgan FP를 계산하여 {smiles: fp_array} dict로 반환한다.

    Parameters
    ----------
    smiles_list : List[str]
        CSV에서 읽은 전체 SMILES (drug_a + drug_b 합친 것).
        중복이 있을 수 있다.

    Returns
    -------
    Dict[str, np.ndarray]
        key: SMILES 문자열, value: shape (n_bits,) float32 array

    Notes
    -----
    실제 데이터에서의 규모감:
    - DrugBank DDI: ~2,500 약물, ~200K pairs
    - 고유 약물 수가 적으므로 FP 계산은 가볍지만, pair 수가 많아 캐싱이 중요
    """

    # TODO 2-1: smiles_list에서 고유한 SMILES만 추출하라.
    #   힌트: set()을 사용
    unique_smiles: Set[str] = set(smiles_list)  # ← 이 줄을 수정

    logger.info(f"Computing FPs for {len(unique_smiles)} unique SMILES "
                f"(from {len(smiles_list)} total)")

    fp_dict: Dict[str, np.ndarray] = {}

    # TODO 2-2: unique_smiles를 순회하며 smiles_to_morgan_fp()를 호출하고,
    #   결과가 None이 아닌 경우에만 fp_dict에 추가하라.
    #   불량 SMILES가 몇 개인지 카운트하여 최종 로그를 남겨라.
    n_failed = 0
    for smi in unique_smiles:
        fp = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            fp_dict[smi] = fp
        else:
            n_failed += 1

    logger.info(f"FP computation done: {len(fp_dict)} success, {n_failed} failed")
    return fp_dict


# -----------------------------------------------------------------------
# 3) 캐시 저장 / 로드
# 캐시 저장과 로드의 이유는 다음과 같다:
# - FP 계산은 RDKit의 C++ 연산이므로 Python에서 반복적으로 호출하면 상당히 느릴 수 있다.
# - 특히 실제 데이터에서는 고유 약물이 수천 개 이상이 될 수 있으므로, 매번 FP를 계산하는 것은 비효율적이다
#  - 따라서 한 번 계산한 FP를 디스크에 저장하여 다음 실행 시 빠르게 로드하는 것이 중요하다.

# -----------------------------------------------------------------------
def save_fp_cache(fp_dict: Dict[str, np.ndarray], cache_path: str | Path) -> None:
    """
    FP dict를 .npz로 디스크에 저장한다.

    .npz 파일이란?
      numpy의 압축 아카이브 포맷. 여러 array를 이름(key)으로 묶어 저장한다.
      np.savez_compressed("file.npz", arr1=a, arr2=b)
      → np.load("file.npz")["arr1"] 으로 접근

    왜 npz인가?
      - parquet은 tabular 데이터에 적합하지만, FP는 고정 길이 dense vector라
        numpy native 포맷이 I/O가 가장 빠르다.
      - pickle은 보안/호환성 문제가 있어 numpy 공식 포맷을 권장.

    Parameters
    ----------
    fp_dict : Dict[str, np.ndarray]
    cache_path : str or Path
        저장할 .npz 파일 경로. 부모 디렉토리가 없으면 생성한다.
    """

    # TODO 3-1: cache_path의 부모 디렉토리가 없으면 생성하라.
    #   힌트: Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    #   parents=True는 중간 디렉토리도 함께 생성하라는 의미, exist_ok=True는 이미 존재해도 에러 내지 말라는 의미다.
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    #cache_path가 "data/processed/fingerprints/cache.npz"라면, 부모 디렉토리는 "data/processed/fingerprints"가 된다. 이 디렉토리가 존재하지 않으면 mkdir()이 생성한다.
    #cache_path와 fp_dict의 관계: cache_path는 fp_dict를 저장할 .npz 파일의 경로를 나타낸다. fp_dict는 {smiles: np.ndarray} 형태의 딕셔너리로, 각 SMILES 문자열에 대응하는 Morgan FP 배열을 담고 있다. save_fp_cache() 함수는 이 fp_dict를 cache_path에 .npz 형식으로 저장하는 역할을 한다.
    #save_fp_cache() 함수의 역할: fp_dict를 .npz 파일로 저장하는 함수다. cache_path가 가리키는 위치에 .npz 파일을 생성하여, fp_dict의 내용을 압축된 형태로 디스크에 저장한다. 이렇게 하면 다음 실행 시 빠르게 로드할 수 있다.
    # 현재까지 fp_dict는 {smiles: np.ndarray} 형태이므로, np.savez_compressed()로 저장하기 전에
    # 별도의 smiles_keys 배열과 fp_matrix로 변환해야 한다.
    # TODO 3-2: np.savez_compressed()로 fp_dict를 저장하라.
    #   주의: npz의 key에 특수문자('=', '(', ')' 등)가 포함되면
    #   np.load 시 문제가 될 수 있다.
    #   → SMILES를 key로 직접 쓰는 대신, 별도의 smiles 목록 array와
    #     FP matrix를 분리하여 저장하는 것이 안전하다.
    #
    #   저장 전략:
    #     smiles_keys = np.array(list(fp_dict.keys()))       # shape: (N,) N은 고유 SMILES 수
    #     fp_matrix   = np.stack(list(fp_dict.values()))     # shape: (N, n_bits) n)bits는 FP 길이 = 2048
    #     np.savez_compressed(cache_path, smiles=smiles_keys, fps=fp_matrix)
    #list와 np.array의 차이점: list는 Python의 일반적인 리스트 자료형으로, 다양한 타입의 요소를 담을 수 있고 크기가 가변적입니다. 반면 np.array는 NumPy 라이브러리에서 제공하는 배열 객체로, 동일한 타입의 요소를 담으며 고정된 크기를 가집니다. np.array는 벡터화 연산과 브로드캐스팅을 지원하여 대규모 데이터 처리에 효율적입니다.
    smiles_keys=np.array(list(fp_dict.keys()))
    fp_matrix=np.stack(list(fp_dict.values()))  # ← 이 블록을 구현
    np.savez_compressed(cache_path, smiles=smiles_keys, fps=fp_matrix)

    logger.info(f"FP cache saved to {cache_path}")


def load_fp_cache(cache_path: str | Path) -> Optional[Dict[str, np.ndarray]]:
    """
    .npz 캐시에서 FP dict를 복원한다.

    Returns
    -------
    Dict[str, np.ndarray] — 캐시 존재 시
    None                   — 캐시 파일이 없을 때
    """

    # TODO 3-3: cache_path가 존재하는지 확인하고,
    #   존재하면 np.load()로 읽어서 {smiles: fp_array} dict를 복원하라.
    #   존재하지 않으면 None을 반환하라.
    #
    #   복원 로직:
    #     data = np.load(cache_path, allow_pickle=False)
    #     smiles_keys = data["smiles"]
    #     fp_matrix = data["fps"]
    #     return {smi: fp_matrix[i] for i, smi in enumerate(smiles_keys)}
    #
    #   allow_pickle=False를 반드시 써야 한다.
    #   pickle이 허용되면 악성 .npz 파일로 임의 코드 실행이 가능하다.
    #try/except FileNotFoundError로 감싸서 None 반환
    #네 코드 스타일 보면 try/except 쓰는 편이니까, np.load 호출부를 try/except FileNotFoundError로 감싸고 except 블록에서 return None 하면 된다. 직접 고쳐봐 — skeleton 버그가 아니라 TODO 구현 범위 내 로직이다.
    try:
        data = np.load(cache_path, allow_pickle=False)
        smiles_keys = data["smiles"]
        fp_matrix = data["fps"]
        return {smi: fp_matrix[i] for i, smi in enumerate(smiles_keys)}  # ← 이 블록을 구현
    except FileNotFoundError:
        return None

# -----------------------------------------------------------------------
# 4) 통합 함수: 캐시 있으면 로드, 없으면 계산 후 저장
# 캐시가 없다는 것은 첫 실행이거나, 캐시가 삭제된 경우(오류 혹은 사용자 지정)다. 이 때는 FP를 계산하여 캐시에 저장한 후 반환한다.

# -----------------------------------------------------------------------
def get_or_compute_fp_dict(
    smiles_list: List[str],
    cache_path: str | Path,
    radius: int = 2,
    n_bits: int = 2048,
) -> Dict[str, np.ndarray]:
    """
    "캐시 히트면 로드, 미스면 계산 후 저장" 패턴.
    이 함수가 외부에서 호출하는 유일한 진입점이 되어야 한다.

    호출 흐름:
      get_or_compute_fp_dict()
        ├─ load_fp_cache() 시도
        │   ├─ 캐시 존재 → dict 반환 (빠름)
        │   └─ 캐시 없음 → None
        └─ compute_fp_dict() → save_fp_cache() → dict 반환
        #캐시 없는 경우에만 compute_fp_dict()가 호출되고, 그 결과가 save_fp_cache()로 저장된 후 반환되는 것이다.
        

    Parameters
    ----------
    smiles_list : List[str]
        전체 SMILES (drug_a + drug_b). 중복 허용.
        # drug_a_smiles과 drug_b_smiles 컬럼에서 추출한 SMILES 리스트를 합친 것이다. 중복이 있을 수 있다.
        # 우리가 이 프로젝트에서 하고 싶은 것이 정확히 뭐지?
        # 우리가 이 프로젝트에서 하고 싶은 것은 DDI 예측 모델을 구축하는 것이다. DDI 예측 모델은 약물 간 상호작용을 예측하는 모델로, 약물의 구조적 특징을 입력으로 사용한다. 
        # Morgan FP는 약물의 구조적 특징을 고정 길이의 binary vector로 표현하는 방법이므로, DDI 예측 모델의 입력으로 사용할 수 있다. 
        # 따라서 우리는 SMILES 문자열을 Morgan FP로 변환하여 DDI 예측 모델에 활용하려는 것이다.
        #그런데 왜 drug_a_smiles과 drug_b_smiles 컬럼에서 추출한 SMILES 리스트를 합치는가?
        # drug_a_smiles과 drug_b_smiles 컬럼에서 추출한 SMILES 리스트를 합치는 이유는, DDI 예측 모델에서 약물 A와 약물 B
        # 간의 상호작용을 예측하기 위해서는 두 약물의 구조적 특징이 모두 필요하기 때문이다.
        # 예를 들어, 약물 A와 약물 B가 상호작용하는지 예측하려면, 약물 A의 FP와 약물 B의 FP가 모두 필요하다. 
        # 따라서 drug_a_smiles과 drug_b_smiles 컬럼에서 추출한 SMILES 리스트를 합쳐서 
        #고유한 약물의 SMILES 집합을 만들어야 한다. 이렇게 하면 각 약물에 대해 한 번만 FP를 계산할 수 있고, DDI 예측 모델에서 해당 FP를 재사용할 수 있다.
        #근데 합친다는게 정확히 어떤 원리로 합친다는거지?
        # drug_a_smiles과 drug_b_smiles 컬럼에서 추출한 SMILES 리스트를 합친다는 것은, 두 컬럼에서 추출한 SMILES 문자열을 하나의 리스트로 합치는 것을 의미한다. 
        # 예를 들어, drug_a_smiles 컬럼에서 ["CCO", "C1=CC=CC=C1"]라는 SMILES가 추출되고, drug_b_smiles 컬럼에서 ["CC(=O)O", "C1=CC=CC=C1"]라는 SMILES가 추출된다고 가정해보자.
        # 이 때, drug_a_smiles과 drug_b_smiles 컬럼에서 추출한 SMILES 리스트를 합치면 ["CCO", "C1=CC=CC=C1", "CC(=O)O", "C1=CC=CC=C1"]라는 리스트가 된다. 
        # 이렇게 합친 리스트에는 중복된 SMILES가 있을 수 있다. 따라서 이 리스트에서 고유한 SMILES 집합을 추출하여 FP를 계산해야 한다. 
        
    cache_path : str or Path
        .npz 캐시 파일 경로
    radius, n_bits : FP 하이퍼파라미터
    """

    # TODO 4-1: load_fp_cache()를 호출하여 캐시를 시도하고,
    #   성공하면 로그를 남기고 반환하라.
    #   실패(None)하면 compute_fp_dict() → save_fp_cache() → 반환하라.
    fp_dict = load_fp_cache(cache_path)
    if fp_dict is None:          # ← except 대신 이걸로
        fp_dict = compute_fp_dict(smiles_list, radius, n_bits)
        save_fp_cache(fp_dict, cache_path)
    return fp_dict


# -----------------------------------------------------------------------
# 5) 단독 실행 테스트 (python -m sbml_msc_2026.features.fingerprint)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 간단한 smoke test
    test_smiles = ["CCO", "C1=CC=CC=C1", "INVALID_SMILES", "CC(=O)O"]

    # 개별 FP 테스트
    for smi in test_smiles:
        result = smiles_to_morgan_fp(smi)
        if result is not None:
            print(f"{smi:40s} → shape={result.shape}, sum={result.sum():.0f}")
        else:
            print(f"{smi:40s} → FAILED (invalid SMILES)")

    # 캐시 라운드트립 테스트
    fp_dict = compute_fp_dict(test_smiles)
    cache_file = Path("data/processed/fingerprints/_test_cache.npz")
    save_fp_cache(fp_dict, cache_file)
    loaded = load_fp_cache(cache_file)
    if loaded is not None:
        assert set(loaded.keys()) == set(fp_dict.keys()), "Cache roundtrip key mismatch!"
        for k in fp_dict:
            assert np.array_equal(fp_dict[k], loaded[k]), f"Cache mismatch for {k}"
        print("✅ Cache roundtrip test passed!")
