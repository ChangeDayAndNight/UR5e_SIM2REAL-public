#!/usr/bin/env python
"""Fail-closed free-space gate for long distillation launches."""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path


GIB = 1024 ** 3
DISTILL_MINIMUM_FREE_GIB = 25.0


def require_free_space(path: Path | str, minimum_gib: float) -> float:
    target = Path(path).resolve()
    if isinstance(minimum_gib, bool) or not isinstance(minimum_gib, (int, float)) \
            or not math.isfinite(float(minimum_gib)) or float(minimum_gib) <= 0:
        raise ValueError("minimum_gib must be a finite positive number")
    free_gib = shutil.disk_usage(target).free / GIB
    if free_gib < float(minimum_gib):
        raise RuntimeError(
            f"free-space gate failed: {free_gib:.2f} GiB available; "
            f"{float(minimum_gib):.2f} GiB required"
        )
    return free_gib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--minimum-gib", type=float, default=DISTILL_MINIMUM_FREE_GIB)
    args = parser.parse_args()
    try:
        available = require_free_space(args.path, args.minimum_gib)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[disk][error] {exc}")
        return 1
    print(
        f"[disk] FREE_SPACE_OK available_gib={available:.2f} "
        f"required_gib={args.minimum_gib:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
