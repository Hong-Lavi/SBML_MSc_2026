from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
        return out.strip()
    except Exception:
        return ""


def collect_env_info() -> Dict[str, Any]:
    """Collect lightweight environment info for reproducibility."""
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_status": _run(["git", "status", "--porcelain"]),
    }

    # Optional torch info
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
        )
    except Exception:
        info.update({"torch": None, "cuda_available": None})

    return info


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
