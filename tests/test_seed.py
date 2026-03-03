from __future__ import annotations

import numpy as np

from sbml_msc_2026.utils.seed import set_seed


def test_numpy_seed_reproducible() -> None:
    set_seed(123, deterministic=True)
    a = np.random.randn(10)

    set_seed(123, deterministic=True)
    b = np.random.randn(10)

    assert np.allclose(a, b)


def test_numpy_seed_changes() -> None:
    set_seed(123, deterministic=True)
    a = np.random.randn(10)

    set_seed(124, deterministic=True)
    b = np.random.randn(10)

    assert not np.allclose(a, b)
