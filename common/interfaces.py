# 요소6: Sim/Real 공통 추상화 — 이 두 ABC가 "쌍둥이의 유일한 계약"이다.
#   RobotInterface : sim=sim/sim_robot.py(Isaac articulation) / real=real/real_robot.py(RTDE)
#   CameraInterface: sim=sim/sim_camera.py(Isaac camera)      / real=real/real_camera.py(pyrealsense2)
# 배포 루프(common/rollout.py)는 이 타입만 알고, 구현이 sim인지 real인지 모른다.
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from common.types import AppliedArmCommandReceipt, CameraFrame, Intrinsics, RobotState


class RobotInterface(ABC):
    """로봇 추상화. 좌표/단위 계약은 RobotState 참조 (quat=wxyz, base frame).

    ⚠️ real 구현 주의: UR RTDE의 getActualTCPPose는 rotation-vector를 주므로
       real_robot.py가 wxyz quat으로 변환해서 RobotState 계약을 지켜야 한다.
    """

    @abstractmethod
    def get_state(self) -> RobotState:
        """현재 상태 스냅샷 (blocking 최소화)."""

    @abstractmethod
    def send_joint_target(self, q_target: np.ndarray, duration_s: float) -> None:
        """스트리밍 추종 명령 (제어 루프 전용).

        sim: joint position target 기록 → physics가 추종
        real: servoJ(q_target, time=duration_s, ...) — duration_s = 1/control_hz
        ⚠️ 캘리브레이션/홈 이동에 쓰지 말 것 → move_to_joints 사용.
        """

    def send_task_delta(
        self,
        delta_pose: np.ndarray,
        current_tcp_pose: np.ndarray,
        duration_s: float,
        *,
        mapped_raw_delta: np.ndarray,
    ) -> AppliedArmCommandReceipt:
        """Stream a base-frame TCP delta ``[dxyz, axis-angle]``.

        ``mapped_raw_delta`` is the original ActionMapper result while
        ``delta_pose`` is its SafetyGuard output.  A successful return must
        delay those two commands as one lineage and bind the selected original,
        selected guarded/pre-floor, and actually applied commands.  Missing or
        malformed receipts are never eligible for Phase 3 clamp evidence.
        Implementations that only support legacy joint policies fail explicitly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support ik_rel")

    @abstractmethod
    def move_to_joints(self, q: np.ndarray, speed_scale: float = 0.1) -> None:
        """블로킹 절대 이동 (캘리브레이션 포즈 순회/홈 복귀 전용, real=moveJ)."""

    @abstractmethod
    def set_gripper(self, closed: bool) -> None:
        """그리퍼 개폐. real 드라이버 채널은 명시적 real.local.yaml 계약을 따른다."""

    @abstractmethod
    def is_ok(self) -> bool:
        """헬스체크 — real: protective stop / 통신 단절 감지. False면 루프가 trip한다."""

    @abstractmethod
    def emergency_stop(self) -> None:
        """요소5: SafetyGuard e-stop hook이 호출. 즉시 정지 + 이후 명령 무효."""

    @abstractmethod
    def close(self) -> None:
        """자원 해제 (연결 종료)."""


class CameraInterface(ABC):
    """depth 카메라 추상화 — RGB 없음 (E2E depth 정책 계약)."""

    @abstractmethod
    def start(self) -> None:
        """스트림 시작 (real: pipeline.start + 워밍업 프레임 버리기)."""

    @abstractmethod
    def get_frame(self) -> CameraFrame:
        """최신 depth 프레임 [m]. 결측(0/NaN)은 그대로 반환 — 전처리가 처리한다."""

    @abstractmethod
    def get_intrinsics(self) -> Intrinsics:
        """원본 해상도 기준 intrinsics (캘리브레이션/역투영용 — 정책 경로에는 불필요)."""

    @abstractmethod
    def stop(self) -> None:
        """스트림 종료."""
