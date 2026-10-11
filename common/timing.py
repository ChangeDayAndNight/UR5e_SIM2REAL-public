# ★요소3: sim dt ↔ 실기 control rate 정합★
# decimation은 어디에도 저장하지 않는다 — 항상 여기서 파생·검증한다.
# (세 값을 따로 저장하면 언젠가 서로 어긋난다 — 단일 소스 + 파생이 원칙)
from __future__ import annotations

import time


def derive_decimation(physics_hz: float, control_hz: float) -> int:
    """decimation = physics_hz / control_hz. 정수가 아니면 ValueError.

    sim: ManagerBasedRLEnvCfg에 sim.dt=1/physics_hz, decimation=반환값 주입
    real: 1/control_hz가 servoJ duration이자 RateLimiter 주기
    → 정합이 안 맞으면 '학습 시작 자체'가 막힌다 (조용한 불일치 방지).
    """
    if physics_hz <= 0 or control_hz <= 0:
        raise ValueError(f"주파수는 양수여야 함: physics={physics_hz}, control={control_hz}")
    ratio = physics_hz / control_hz
    decimation = round(ratio)
    if decimation < 1 or abs(ratio - decimation) > 1e-6:
        raise ValueError(
            f"physics_hz({physics_hz})가 control_hz({control_hz})의 정수배가 아님 "
            f"(ratio={ratio}) — configs/common.yaml control 섹션을 수정할 것"
        )
    return decimation


class RateLimiter:
    """실기 루프 주기 유지기 — 오버런(주기 초과)을 세고 경고한다.

    ⚠️ Windows 타이머 해상도(기본 ~15.6ms)로 20Hz 유지가 흔들릴 수 있음 —
      배포는 별도 PC(사용자 결정)이므로 그쪽 OS에서 Deploy_DryRun의
      지터 히스토그램으로 실측 검증한다.
    """

    def __init__(self, hz: float):
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.hz = hz
        self.period_s = 1.0 / hz
        self.overruns = 0
        self.consecutive_overruns = 0
        self.max_consecutive_overruns = 0
        self._next_t: float | None = None
        self._last_t: float | None = None

    def sleep(self) -> float:
        """다음 주기까지 대기. 실제 경과 주기[s]를 반환 (dump/로그용)."""
        now = time.perf_counter()
        if self._next_t is None:
            self._next_t = now + self.period_s
            self._last_t = now
            # The rollout arms this limiter immediately before each command.
            # Waiting one full period on initialization prevents command 0
            # from bypassing the cadence gate.
            time.sleep(self.period_s)
            after_wait = time.perf_counter()
            self._last_t = after_wait
            self._next_t = after_wait + self.period_s
            return after_wait - now
        remain = self._next_t - now
        if remain > 0:
            time.sleep(remain)
            self.consecutive_overruns = 0
        else:
            self.overruns += 1
            self.consecutive_overruns += 1
            self.max_consecutive_overruns = max(
                self.max_consecutive_overruns, self.consecutive_overruns
            )
            print(f"[timing][warn] 주기 오버런 {-remain*1000:.1f}ms (#{self.overruns})")
        after_wait = time.perf_counter()
        assert self._last_t is not None
        elapsed = after_wait - self._last_t
        self._last_t = after_wait
        # Never run catch-up bursts after an overrun.  A stale control stream is
        # rescheduled from now; sending several commands back-to-back is unsafe.
        self._next_t = after_wait + self.period_s
        return elapsed
