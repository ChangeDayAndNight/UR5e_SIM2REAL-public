from __future__ import annotations

from pathlib import Path

import pytest

from common.kit_lifecycle import KIT_CLOSE_FAILURE_EXIT_CODE, run_main_with_app_close


class _App:
    def __init__(self, close_error: BaseException | None = None):
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def test_nonzero_main_result_survives_close_system_exit_zero():
    app = _App(SystemExit(0))
    assert run_main_with_app_close(lambda: 7, app) == 7
    assert app.close_calls == 1


def test_success_becomes_nonzero_when_close_fails():
    app = _App(RuntimeError("close failed"))
    assert run_main_with_app_close(lambda: 0, app) == KIT_CLOSE_FAILURE_EXIT_CODE
    assert app.close_calls == 1


def test_primary_system_exit_is_re_raised_after_close_attempt():
    app = _App(SystemExit(0))

    def fail() -> int:
        raise SystemExit(9)

    with pytest.raises(SystemExit) as caught:
        run_main_with_app_close(fail, app)
    assert caught.value.code == 9
    assert app.close_calls == 1


def test_primary_exception_identity_survives_secondary_close_failure():
    primary = RuntimeError("primary")
    app = _App(RuntimeError("secondary"))

    def fail() -> int:
        raise primary

    with pytest.raises(RuntimeError) as caught:
        run_main_with_app_close(fail, app)
    assert caught.value is primary
    assert app.close_calls == 1


def test_phase3_source_routes_top_level_through_lifecycle_wrapper():
    source = (
        Path(__file__).resolve().parents[1] / "Phase3/SimDeploy_Rehearsal.py"
    ).read_text(encoding="utf-8")
    guard = source.index('if __name__ == "__main__":')
    tail = source[guard:]
    assert "raise SystemExit(run_main_with_app_close(main, app))" in tail
    assert "rc = main()" not in tail
    assert "app.close()" not in tail
