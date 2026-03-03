from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """Set RNG seeds for reproducibility.

    Notes:
    - Determinism on GPU can be tricky (CUDA kernels, cuDNN, etc.).
    - We keep torch optional; install later (Day 3).

    TODO:
    - add JAX seed if needed
    - add CUBLAS_WORKSPACE_CONFIG handling if we move to strict CUDA determinism
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    info: Dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "torch_available": torch is not None,
    }

    if torch is not None:
        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

    return info
