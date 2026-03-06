"""
Day2 테스트 — Fingerprint 계산 + 캐싱 검증
===========================================================================

[이 테스트가 검증하는 것]
  1. 정상 SMILES → 올바른 shape의 FP 반환
  2. 불량 SMILES → None 반환 (crash하지 않음)
  3. FP 값이 binary (0 또는 1)인지 확인
  4. 캐시 저장 → 로드 라운드트립이 동일한지 확인
  5. pair feature의 shape과 대칭성 검증

실행: pytest tests/test_day2_fingerprint.py -v
===========================================================================
"""

import numpy as np
import pytest
from pathlib import Path

from sbml_msc_2026.features.fingerprint import (
    smiles_to_morgan_fp,
    compute_fp_dict,
    save_fp_cache,
    load_fp_cache,
)
from sbml_msc_2026.features.pair_features import make_pair_feature


# -----------------------------------------------------------------------
# Fingerprint 기본 테스트
# -----------------------------------------------------------------------
class TestMorganFP:
    """Morgan Fingerprint 계산 관련 테스트."""

    def test_valid_smiles_returns_correct_shape(self):
        """정상 SMILES → shape (2048,) float32 반환."""
        fp = smiles_to_morgan_fp("CCO", radius=2, n_bits=2048)
        # TODO T-1: fp가 None이 아닌지 assert하라.
        # TODO T-2: fp.shape이 (2048,)인지 assert하라.
        # TODO T-3: fp.dtype이 np.float32인지 assert하라.
        pass  # ← 이 블록을 구현

    def test_invalid_smiles_returns_none(self):
        """불량 SMILES → None 반환, 예외 없음."""
        fp = smiles_to_morgan_fp("NOT_A_MOLECULE")
        # TODO T-4: fp가 None인지 assert하라.
        pass  # ← 이 블록을 구현

    def test_fp_values_are_binary(self):
        """FP 값은 0.0 또는 1.0만 포함해야 한다."""
        fp = smiles_to_morgan_fp("C1=CC=CC=C1")  # 벤젠
        # TODO T-5: fp의 모든 원소가 {0.0, 1.0}에 속하는지 확인하라.
        #   힌트: set(fp.tolist()).issubset({0.0, 1.0})
        pass  # ← 이 블록을 구현

    def test_different_smiles_different_fp(self):
        """서로 다른 분자는 다른 FP를 가져야 한다."""
        fp1 = smiles_to_morgan_fp("CCO")         # 에탄올
        fp2 = smiles_to_morgan_fp("C1=CC=CC=C1")  # 벤젠
        # TODO T-6: fp1과 fp2가 동일하지 않음을 확인하라.
        #   힌트: assert not np.array_equal(fp1, fp2)
        pass  # ← 이 블록을 구현


# -----------------------------------------------------------------------
# 캐시 라운드트립 테스트
# -----------------------------------------------------------------------
class TestFPCache:
    """FP dict 캐시 저장/로드 테스트."""

    def test_cache_roundtrip(self, tmp_path: Path):
        """저장한 fp_dict와 로드한 fp_dict가 동일해야 한다.

        tmp_path는 pytest가 자동으로 제공하는 임시 디렉토리.
        테스트 끝나면 자동 삭제된다.
        """
        smiles_list = ["CCO", "C1=CC=CC=C1", "CC(=O)O"]
        fp_dict = compute_fp_dict(smiles_list)

        cache_file = tmp_path / "test_fp.npz"
        save_fp_cache(fp_dict, cache_file)
        loaded = load_fp_cache(cache_file)

        # TODO T-7: loaded가 None이 아닌지 확인하라.
        # TODO T-8: loaded의 key 집합이 fp_dict와 동일한지 확인하라.
        # TODO T-9: 각 key에 대해 값(array)이 동일한지 확인하라.
        pass  # ← 이 블록을 구현

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        """존재하지 않는 경로 → None 반환."""
        result = load_fp_cache(tmp_path / "nonexistent.npz")
        # TODO T-10: result가 None인지 확인하라.
        pass  # ← 이 블록을 구현


# -----------------------------------------------------------------------
# Pair Feature 테스트
# -----------------------------------------------------------------------
class TestPairFeature:
    """Pair feature 생성 로직 테스트."""

    def setup_method(self):
        """각 테스트 전에 실행되는 fixture."""
        self.D = 16  # 테스트용 작은 차원
        self.fp_a = np.random.RandomState(42).rand(self.D).astype(np.float32)
        self.fp_b = np.random.RandomState(99).rand(self.D).astype(np.float32)

    def test_concat_shape(self):
        """concat → shape (2D,)."""
        result = make_pair_feature(self.fp_a, self.fp_b, method="concat")
        # TODO T-11: result.shape이 (2 * self.D,)인지 확인하라.
        pass  # ← 이 블록을 구현

    def test_sum_shape(self):
        """sum → shape (D,)."""
        result = make_pair_feature(self.fp_a, self.fp_b, method="sum")
        # TODO T-12: result.shape이 (self.D,)인지 확인하라.
        pass  # ← 이 블록을 구현

    def test_symmetric_is_symmetric(self):
        """symmetric: (A,B) == (B,A)."""
        ab = make_pair_feature(self.fp_a, self.fp_b, method="symmetric")
        ba = make_pair_feature(self.fp_b, self.fp_a, method="symmetric")
        # TODO T-13: ab와 ba가 동일한지 확인하라.
        #   힌트: np.array_equal(ab, ba)
        pass  # ← 이 블록을 구현

    def test_concat_is_not_symmetric(self):
        """concat: (A,B) ≠ (B,A) — 비대칭을 확인."""
        ab = make_pair_feature(self.fp_a, self.fp_b, method="concat")
        ba = make_pair_feature(self.fp_b, self.fp_a, method="concat")
        # TODO T-14: ab와 ba가 다른지 확인하라.
        pass  # ← 이 블록을 구현

    def test_shape_mismatch_raises(self):
        """FP shape이 다르면 ValueError."""
        short = np.zeros(8, dtype=np.float32)
        # TODO T-15: pytest.raises(ValueError)로 에러 발생을 확인하라.
        #   힌트: with pytest.raises(ValueError):
        #             make_pair_feature(self.fp_a, short, method="concat")
        pass  # ← 이 블록을 구현
