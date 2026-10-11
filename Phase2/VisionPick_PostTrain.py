#!/usr/bin/env python
"""Canonical Student adoption: final + loss-ranked four, 4 seeds x 512 episodes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

from scripts.adoption_pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Vision Distill final+4 loss candidates x seeds 42..45 x 512ep "
            "fail-closed adoption"
        )
    )
    parser.add_argument("--run", required=True, help="canonical completed Vision run directory")
    parser.add_argument("--max_iterations", type=int, default=20_000)
    parser.add_argument("--wait_pid", type=int, default=0)
    parser.add_argument("--wait_create_time", type=float, default=None)
    parser.add_argument("--poll_seconds", type=int, default=30)
    args = parser.parse_args()
    return run_pipeline(
        role="student",
        run=args.run,
        max_iterations=args.max_iterations,
        wait_pid=args.wait_pid,
        wait_create_time=args.wait_create_time,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
