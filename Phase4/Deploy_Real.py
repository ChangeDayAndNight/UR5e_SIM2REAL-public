#!/usr/bin/env python
"""Receive-only release guard for the future real-policy entry point.

No real robot motion is authorized in this release.  The former command-line
options and unreachable motion implementation are intentionally absent so they
cannot be mistaken for an approved commissioning workflow.  Future motion work
requires a separately reviewed authorization design and release.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.contracts import ContractError, require_motion_authorization  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real motion is not authorized in this release; use the canonical "
            "deployment verifier in receive-only no-motion mode."
        )
    )
    return parser.parse_args()


def main() -> int:
    _parse_args()  # Preserve a useful --help path without accepting motion options.
    require_motion_authorization("policy")
    raise AssertionError("unreachable")  # pragma: no cover


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"[deploy][blocked] {exc}", file=sys.stderr)
        raise SystemExit(2)
