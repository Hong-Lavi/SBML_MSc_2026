from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sbml_msc_2026.utils.config import load_yaml
from sbml_msc_2026.utils.env import collect_env_info, save_json
from sbml_msc_2026.utils.logger import get_logger
from sbml_msc_2026.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(cfg["paths"]["log_dir"])
    log_path = log_dir / "day1_sanity.log"
    logger = get_logger("sanity", log_path=log_path)

    # 1) seed
    seed = int(cfg["seed"])
    deterministic = bool(cfg["reproducibility"]["deterministic"])
    seed_info = set_seed(seed, deterministic)
    save_json(seed_info, out_dir / "seed_info.json")

    # 2) env info
    env_info = collect_env_info()
    save_json(env_info, out_dir / "env_info.json")

    # 3) tiny artifact
    x = np.random.randn(5).tolist()
    save_json({"sample": x}, out_dir / "sample.json")

    logger.info("Sanity done")
    logger.info("out_dir=%s", str(out_dir))
    logger.info("seed=%s deterministic=%s", seed, deterministic)


if __name__ == "__main__":
    main()
