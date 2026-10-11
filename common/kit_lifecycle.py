"""Fail-closed lifecycle wrapper for command-line Kit applications."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Protocol


KIT_CLOSE_FAILURE_EXIT_CODE = 2


class _ClosableApp(Protocol):
    def close(self) -> None: ...


def _report_close_failure(exc: BaseException) -> None:
    """Best-effort diagnostics that can never replace the primary failure."""

    try:
        print(
            f"[kit][close-fail] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    except BaseException:
        # stdout/stderr can already be unavailable during native Kit teardown.
        pass


def run_main_with_app_close(main: Callable[[], int], app: _ClosableApp) -> int:
    """Run ``main`` and attempt ``app.close`` on every Python exit path.

    A primary exception (including ``SystemExit``/``KeyboardInterrupt``) is
    always re-raised even if Kit shutdown also fails.  For normal integer
    returns, a non-zero application result is likewise preserved; a shutdown
    failure after an otherwise successful run becomes exit code 2.
    """

    try:
        result = main()
    except BaseException:
        try:
            app.close()
        except BaseException as close_exc:
            _report_close_failure(close_exc)
        raise

    try:
        app.close()
    except BaseException as close_exc:
        _report_close_failure(close_exc)
        if type(result) is int and result != 0:
            return result
        return KIT_CLOSE_FAILURE_EXIT_CODE

    if type(result) is not int:
        raise TypeError(f"Kit CLI main must return int, got {type(result).__name__}")
    return result
