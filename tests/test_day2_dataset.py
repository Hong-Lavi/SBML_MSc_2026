"""
Day2 테스트 — DDI Dataset + Splitter 통합 검증
===========================================================================

[이 테스트가 검증하는 것]
  1. DDIPairDataset의 __len__이 올바른 값을 반환하는가
  2. __getitem__이 올바른 shape의 tensor를 반환하는가
  3. DataLoader 배치 shape이 [B, D]인가
  4. Drug-aware split에서 train/test 인덱스가 겹치지 않는가
  5. Split 인덱스의 합이 전체 데이터와 일치하는가

실행: pytest tests/test_day2_dataset.py -v
===========================================================================
"""

import pandas as pd
import numpy as np
import torch
import pytest
from pathlib import Path

from sbml_msc_2026.datasets.ddi_dataset import DDIPairDataset, build_ddi_dataloader
from sbml_msc_2026.datasets.splitter import split_ddi_pairs


# -----------------------------------------------------------------------
# 공통 fixture
# -----------------------------------------------------------------------
@pytest.fixture
def dummy_csv(tmp_path: Path) -> Path:
    """테스트용 DDI pair CSV를 임시 디렉토리에 생성."""
    csv_path = tmp_path / "test_ddi.csv"
    csv_path.write_text(
        "drug_a_smiles,drug_b_smiles,interaction_type\n"
        "CCO,C1=CC=CC=C1,0\n"
        "CCO,CC(=O)O,1\n"
        "C1=CC=CC=C1,CC(=O)O,2\n"
        "CC(=O)O,CCO,0\n"
    )
    return csv_path


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    """FP 캐시 경로."""
    return tmp_path / "fp_cache.npz"


# -----------------------------------------------------------------------
# DDIPairDataset 테스트
# -----------------------------------------------------------------------
class TestDDIPairDataset:

    def test_len(self, dummy_csv, cache_path):
        """Dataset length = CSV 행 수."""
        ds = DDIPairDataset(dummy_csv, cache_path, pair_method="concat")
        # TODO T-16: len(ds)가 4인지 assert하라.
        pass  # ← 이 블록을 구현

    def test_getitem_shape_concat(self, dummy_csv, cache_path):
        """concat method → feature shape (4096,)."""
        ds = DDIPairDataset(dummy_csv, cache_path, pair_method="concat", fp_n_bits=2048)
        feat, label = ds[0]
        # TODO T-17: feat.shape이 (4096,)인지 확인하라.
        # TODO T-18: feat.dtype이 torch.float32인지 확인하라.
        # TODO T-19: label.dtype이 torch.int64인지 확인하라.
        pass  # ← 이 블록을 구현

    def test_getitem_shape_sum(self, dummy_csv, cache_path):
        """sum method → feature shape (2048,)."""
        ds = DDIPairDataset(dummy_csv, cache_path, pair_method="sum", fp_n_bits=2048)
        feat, label = ds[0]
        # TODO T-20: feat.shape이 (2048,)인지 확인하라.
        pass  # ← 이 블록을 구현

    def test_dataloader_batch_shape(self, dummy_csv, cache_path):
        """DataLoader 배치 shape = [B, D]."""
        ds = DDIPairDataset(dummy_csv, cache_path, pair_method="concat")
        loader = build_ddi_dataloader(ds, batch_size=2, shuffle=False)

        for batch_feat, batch_label in loader:
            B, D = batch_feat.shape
            # TODO T-21: B가 2 이하인지 확인하라. (마지막 배치는 작을 수 있음)
            # TODO T-22: D가 4096인지 확인하라.
            # TODO T-23: batch_label.shape이 (B,)인지 확인하라.
            pass  # ← 이 블록을 구현
            break


# -----------------------------------------------------------------------
# Splitter 테스트
# -----------------------------------------------------------------------
class TestSplitter:

    def test_random_split_covers_all(self, dummy_csv):
        """Random split: train + test = 전체."""
        df = pd.read_csv(dummy_csv)
        splits = split_ddi_pairs(df, method="random", train_ratio=0.75, seed=42)
        all_idx = sorted(splits["train"] + splits["test"])
        # TODO T-24: all_idx가 list(range(len(df)))와 동일한지 확인하라.
        pass  # ← 이 블록을 구현

    def test_no_index_overlap(self, dummy_csv):
        """Train과 test 인덱스가 겹치지 않아야 한다."""
        df = pd.read_csv(dummy_csv)
        splits = split_ddi_pairs(df, method="drug_aware", train_ratio=0.75, seed=42)
        overlap = set(splits["train"]) & set(splits["test"])
        # TODO T-25: overlap이 빈 집합인지 확인하라.
        pass  # ← 이 블록을 구현

    def test_split_reproducibility(self, dummy_csv):
        """동일 seed → 동일 split."""
        df = pd.read_csv(dummy_csv)
        s1 = split_ddi_pairs(df, method="drug_aware", seed=123)
        s2 = split_ddi_pairs(df, method="drug_aware", seed=123)
        # TODO T-26: s1["train"]과 s2["train"]이 동일한지 확인하라.
        # TODO T-27: s1["test"]와 s2["test"]가 동일한지 확인하라.
        pass  # ← 이 블록을 구현

    def test_dataset_with_split_indices(self, dummy_csv, cache_path):
        """Split 인덱스로 생성한 Dataset의 길이가 올바른가."""
        df = pd.read_csv(dummy_csv)
        splits = split_ddi_pairs(df, method="random", train_ratio=0.75, seed=42)

        train_ds = DDIPairDataset(
            dummy_csv, cache_path, pair_method="concat",
            indices=splits["train"],
        )
        test_ds = DDIPairDataset(
            dummy_csv, cache_path, pair_method="concat",
            indices=splits["test"],
        )

        # TODO T-28: len(train_ds) + len(test_ds) == len(df) 확인
        pass  # ← 이 블록을 구현
