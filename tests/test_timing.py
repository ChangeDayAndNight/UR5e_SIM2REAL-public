"""요소3(dt 정합 + RateLimiter) 단위테스트 — 시뮬레이터 불필요.

derive_decimation의 파생/차단 규칙과 RateLimiter의 주기/오버런 카운팅을
검증한다. 실시간 의존은 최소화(짧은 주기 + 부등호 검정)한다.

실행: python -m pytest tests/test_timing.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import time

import pytest

from common.timing import RateLimiter, derive_decimation

# ------------------------------------------------------- derive_decimation ----


@pytest.mark.parametrize(
    "physics_hz,control_hz,expected",
    [
        (120.0, 20.0, 6),   # configs/common.yaml 기본값
        (100.0, 20.0, 5),
        (120.0, 120.0, 1),
        (240.0, 20.0, 12),
    ],
)
def test_derive_decimation_exact(physics_hz, control_hz, expected):
    """정수배 조합 → decimation 파생값 검증."""
    assert derive_decimation(physics_hz, control_hz) == expected


def test_derive_decimation_non_integer_ratio_raises():
    """비정합(100/30 = 3.33...)은 학습 시작 차단 — ValueError."""
    with pytest.raises(ValueError):
        derive_decimation(100.0, 30.0)


@pytest.mark.parametrize(
    "physics_hz,control_hz",
    [(0.0, 20.0), (120.0, 0.0), (-120.0, 20.0), (120.0, -20.0)],
)
def test_derive_decimation_nonpositive_raises(physics_hz, control_hz):
    """음수/0 주파수 → ValueError."""
    with pytest.raises(ValueError):
        derive_decimation(physics_hz, control_hz)


def test_derive_decimation_sub_unit_ratio_raises():
    """physics < control(ratio<1 → decimation<1)도 차단."""
    with pytest.raises(ValueError):
        derive_decimation(20.0, 120.0)


# ------------------------------------------------------------- RateLimiter ----


@pytest.mark.parametrize("hz,period", [(20.0, 0.05), (1000.0, 0.001), (50.0, 0.02)])
def test_ratelimiter_period(hz, period):
    """period_s = 1/hz."""
    assert RateLimiter(hz).period_s == pytest.approx(period, rel=1e-12)


def test_ratelimiter_first_sleep_returns_period():
    """첫 sleep도 한 주기를 기다려 초기 command burst를 만들지 않는다."""
    rl = RateLimiter(100.0)
    started = time.perf_counter()
    elapsed = rl.sleep()
    wall = time.perf_counter() - started
    assert elapsed >= rl.period_s * 0.8
    assert wall >= rl.period_s * 0.8
    assert rl.overruns == 0


def test_ratelimiter_overrun_counting():
    """An overrun reschedules from now and never produces a catch-up burst."""
    rl = RateLimiter(1000.0)  # 1ms 주기 — 테스트 시간 최소화
    rl.sleep()  # 초기화
    time.sleep(0.05)  # 주기 대비 ~50배 지연
    rl.sleep()
    assert rl.overruns == 1
    rl.sleep()
    assert rl.overruns == 1
    assert rl.consecutive_overruns == 0


def test_ratelimiter_no_overrun_when_on_time():
    """여유 있는 주기(50ms)로 제때 돌면 overruns == 0 (총 ~0.1s)."""
    rl = RateLimiter(20.0)
    for _ in range(3):
        dt = rl.sleep()
        assert isinstance(dt, float) and dt > 0.0
    assert rl.overruns == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
