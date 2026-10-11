# ★요소5: 안전 계층★ — 워크스페이스 바운드 / 속도·힘 리밋 / e-stop hook.
# 배포 루프의 "모든" 명령이 SafetyGuard를 경유한다 (Deploy_Real에서 우회 금지).
# sim 리허설(Phase3)도 같은 값으로 통과시켜 실기 전에 임계값을 검증한다.
# ⚠️ 소프트웨어 안전은 하드웨어 e-stop 버튼을 대체하지 않는다.
from __future__ import annotations

from typing import Callable

import numpy as np

from common.types import RobotState, SafetyCfg, SafetyVerdict


class SafetyGuard:
    """per-step 안전 검사 + clamp + trip latch.

    사용 계약 (rollout.py):
      1) q_target = guard.clamp_joint_target(q_cur, q_target, dt)  # 속도 한계 clamp
      2) verdict = guard.check(state)                              # AABB/힘 판정
      3) verdict.ok가 아니면 guard.trip(reason) → estop_hook(=RobotInterface.emergency_stop) 호출
      4) trip 이후 clamp/check는 전부 거부(latch) — 재기동 전까지 명령 불가
    """

    def __init__(
        self, cfg: SafetyCfg, estop_hook: Callable[[str], None] | None = None
    ):
        self.cfg = cfg
        self._estop_hook = estop_hook
        self._tripped = False
        self._trip_reason = ""

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def trip(self, reason: str) -> None:
        """latch + e-stop hook 호출 (중복 trip은 무시)."""
        if self._tripped:
            return
        self._tripped = True
        self._trip_reason = reason
        print(f"[safety] ★TRIP★ {reason}")
        if self._estop_hook is not None:
            try:
                self._estop_hook(reason)
            except Exception as exc:  # latch must survive a failed software stop call
                self._trip_reason += f"; emergency-stop hook failed: {exc}"
                print(f"[safety][error] emergency-stop hook failed: {exc}")

    def clamp_joint_target(
        self, q_cur: np.ndarray, q_target: np.ndarray, dt: float
    ) -> np.ndarray:
        """스텝당 관절 이동량을 joint_vel_max*dt로 clamp (초과분은 잘라낸다)."""
        if self._tripped:
            raise RuntimeError(f"SafetyGuard tripped: {self._trip_reason}")
        q_cur = np.asarray(q_cur, dtype=np.float64)
        q_target = np.asarray(q_target, dtype=np.float64)
        max_step = self.cfg.joint_vel_max_rad_s * dt
        delta = np.clip(q_target - q_cur, -max_step, max_step)
        return q_cur + delta

    def clamp_task_delta(
        self,
        delta_pose: np.ndarray,
        dt: float,
        current_tcp_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        """Clamp Cartesian translation and axis-angle magnitudes per control step.

        ★★2026-08-12(감사 B1) z-floor clamp 추가★★ — sim 액션 래퍼(delayed_actions.py)의
        `actions[:,2] = max(actions[:,2], floor − z_cur)`와 ★같은 의미론★을 실기 경로에 복원한다.
        [감사 실측] place 타겟 z(−0.0125)가 패드-벨트 접촉점(−0.0101)보다 아래 →
          sim에서 정책은 클램프에 기대 하강을 명령하는 습관을 학습한다. 실기에 등가물이
          없으면 ★매 정상 place 시도★가 벨트 압착 경로를 밟고, AABB 하한(−0.0375)은 벨트
          상면이라 위치 기반 안전층이 구조적으로 무력했다. 리허설은 sim 액션텀을 경유해
          클램프가 있으므로 ★리허설로는 재현되지 않는 종류★의 갭이었다.
        trip이 아니라 clamp인 이유: sim과 같은 의미론이어야 정책 거동이 보존된다(§8-1).
        current_tcp_pose가 None이면 z-floor는 건너뛴다(호출측이 반드시 전달할 것 —
        run_rollout은 state.tcp_pose를 전달한다).
        """
        if self._tripped:
            raise RuntimeError(f"SafetyGuard tripped: {self._trip_reason}")
        delta = np.asarray(delta_pose, dtype=np.float64).reshape(6).copy()
        for sl, max_norm in (
            (slice(0, 3), self.cfg.tcp_linear_vel_max_m_s * dt),
            (slice(3, 6), self.cfg.tcp_angular_vel_max_rad_s * dt),
        ):
            norm = float(np.linalg.norm(delta[sl]))
            if norm > max_norm > 0.0:
                delta[sl] *= max_norm / norm
        if self.cfg.tcp_z_floor_m is not None and current_tcp_pose is not None:
            z_cur = float(np.asarray(current_tcp_pose, dtype=np.float64).reshape(-1)[2])
            min_dz = self.cfg.tcp_z_floor_m - z_cur
            if delta[2] < min_dz:
                delta[2] = min_dz  # 바닥 아래로 내려가는 성분만 잘라낸다(sim과 동일: 상승 강제 포함)
        return delta

    def check_task_target(self, current_tcp_pose: np.ndarray, delta_pose: np.ndarray) -> SafetyVerdict:
        """Reject a Cartesian command whose predicted target leaves the workspace."""
        target = np.asarray(current_tcp_pose, dtype=np.float64)[:3] + np.asarray(delta_pose)[:3]
        lo = np.asarray(self.cfg.workspace_min)
        hi = np.asarray(self.cfg.workspace_max)
        if np.any(target < lo) or np.any(target > hi):
            return SafetyVerdict(
                ok=False, reason=f"명령 TCP workspace 이탈: {np.round(target, 3).tolist()}"
            )
        return SafetyVerdict(ok=True)

    def check(self, state: RobotState) -> SafetyVerdict:
        """TCP workspace AABB / 관절 속도 / TCP 힘 판정 (trip은 호출자가 결정)."""
        if self._tripped:
            return SafetyVerdict(ok=False, reason=f"tripped: {self._trip_reason}")

        tcp_pos = np.asarray(state.tcp_pose[:3], dtype=np.float64)
        lo = np.asarray(self.cfg.workspace_min)
        hi = np.asarray(self.cfg.workspace_max)
        if np.any(tcp_pos < lo) or np.any(tcp_pos > hi):
            return SafetyVerdict(
                ok=False, reason=f"workspace 이탈: TCP {np.round(tcp_pos, 3).tolist()}"
            )

        vel_max = float(np.max(np.abs(np.asarray(state.joint_vel))))
        if vel_max > self.cfg.joint_vel_max_rad_s:
            return SafetyVerdict(
                ok=False,
                reason=f"관절 속도 초과: {vel_max:.3f} > {self.cfg.joint_vel_max_rad_s}",
            )

        if state.tcp_force is not None:
            f_norm = float(np.linalg.norm(np.asarray(state.tcp_force[:3])))
            if f_norm > self.cfg.tcp_force_max_n:
                return SafetyVerdict(
                    ok=False,
                    reason=f"TCP 힘 초과: {f_norm:.1f}N > {self.cfg.tcp_force_max_n}N",
                )

        return SafetyVerdict(ok=True)
