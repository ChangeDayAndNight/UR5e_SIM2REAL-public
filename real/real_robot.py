# RealRobot — ur_rtde 기반 RobotInterface 실기 어댑터 (요소6).
"""RealRobot: UR5e 실기를 common.interfaces.RobotInterface 계약으로 감싼다.

목적: 배포 루프(common/rollout.py)가 sim/real을 구분하지 않도록, UR RTDE의
      rotation-vector 포즈를 wxyz quat + TCP offset 보정(요소4)까지 끝낸
      RobotState로 노출한다.
사용: env_ur_real(배포 PC) 전용 — Deploy_Real.py / Deploy_DryRun.py /
      Calib_*.py가 생성한다. 라이브러리 모듈이므로 직접 실행하지 않는다.
산출: 없음 (라이브러리). 순수 수학(rotvec_to_quat 등)은 tests/에서 검증.

⚠️ ur_rtde 호출 경로는 구현되었지만 배포 PC의 exact 1.6.2 API·실기
   검증이 남은 P0이다. 현 릴리스는 수신 전용이며 명령 경로는 조건 없이 차단된다.
⚠️ isaaclab / isaacsim import 금지 (env_ur_real에는 Isaac이 없다).
   ur_rtde도 모듈 최상위에서 import하지 않는다 — __init__에서 지연 import
   (SDK 없는 학습 PC에서도 이 모듈의 순수 수학을 pytest하기 위함).
"""
from __future__ import annotations

import time
import threading
from pathlib import Path

import numpy as np

from common.calibration import (
    apply_tcp_offset,
    load_tcp_offset,
    mat_to_pose7,
    pose7_to_mat,
    ur_base_to_base_link_pose7,
    ur_base_to_base_link_wrench,
)
from common.interfaces import RobotInterface
from common.gripper_command import (
    GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S,
    GRIPPER_COMMAND_OPEN,
    GRIPPER_COMMAND_TRANSITION_S,
    advance_gripper_command_state,
)
from common.types import AppliedArmCommandReceipt, RobotState

_NUM_JOINTS = 6

# servoJ의 speed/accel 자리 (⚠️ TODO(확인 필요): ur_rtde 문서상 servoJ에서
# speed/accel 인자는 "현재 무시된다(reserved)"고 기억함 — 공식 문서로 확인 후
# 값/주석 확정. 무시되더라도 위치 인자 자리는 채워야 한다.)
_SERVOJ_SPEED = 0.5  # [rad/s] placeholder
_SERVOJ_ACCEL = 0.5  # [rad/s^2] placeholder

# moveJ 공칭 속도/가속 — speed_scale과 곱해 사용 (UR5e 관절 최대 π rad/s보다 보수적)
_MOVEJ_NOMINAL_SPEED = 1.0  # [rad/s]
_MOVEJ_ACCEL = 0.8  # [rad/s^2] ⚠️ TODO(확인 필요): moveJ acceleration 단위/기본값

class _GripperPulseTicket:
    """Thread-safe completion record for one physical DO pulse request."""

    def __init__(self, closed: bool, generation: int, dispatch_timeout_s: float) -> None:
        self.closed = bool(closed)
        self.generation = int(generation)
        # Dispatch deadlines use a private monotonic clock.  Observation
        # timestamps keep perf_counter semantics and must not affect safety.
        self.reserved_at_s = time.monotonic()
        self.dispatch_deadline_s = self.reserved_at_s + float(dispatch_timeout_s)
        self.dispatch_resolved = threading.Event()
        self.finished = threading.Event()
        self.cancelled = threading.Event()
        self.status = "pending"
        self.high_dispatched_at_s: float | None = None
        self.completed_at_s: float | None = None
        self.error: BaseException | None = None
        self.pulse_error: BaseException | None = None
        self.worker: threading.Thread | None = None


# ---------- 순수 수학 헬퍼 (하드웨어 불필요 — tests/에서 단위 검증) ----------


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """UR rotation-vector (axis*angle [rad], (3,)) → quaternion (4,) wxyz.

    성질 (단위 테스트 계약):
      - 영벡터 → [1, 0, 0, 0] (항등 회전)
      - [pi, 0, 0] → [0, 1, 0, 0] (x축 180°)
      - [pi/2, 0, 0] → [cos(pi/4), sin(pi/4), 0, 0]
      - 반환 quat은 단위 노름 & w >= 0 정규화 (common.calibration.mat_to_quat 관례와 동일)
      - 왕복 일관성: quat_to_mat(rotvec_to_quat(r)) == Rodrigues(r)
    """
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = r / angle
    half = 0.5 * angle
    q = np.concatenate([[np.cos(half)], np.sin(half) * axis])
    return q if q[0] >= 0 else -q


def ur_pose_to_pose7(pose_ur: np.ndarray) -> np.ndarray:
    """UR 포즈 (6,)=[x,y,z,rx,ry,rz] (rotation-vector) → (7,)=pos+quat(wxyz).

    RobotState.tcp_pose 계약(quat=wxyz, base frame)으로 가는 변환의 전반부.
    (후반부 = common.calibration.apply_tcp_offset — 요소4)
    """
    p = np.asarray(pose_ur, dtype=np.float64).reshape(6)
    return np.concatenate([p[:3], rotvec_to_quat(p[3:])])


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Quaternion ``wxyz`` to the shortest UR rotation vector."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    half = float(np.arctan2(np.linalg.norm(q[1:]), np.clip(q[0], -1.0, 1.0)))
    if half < 1e-12:
        return np.zeros(3)
    return q[1:] / np.linalg.norm(q[1:]) * (2.0 * half)


def apply_task_delta_pose7(pose7: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    """NumPy equivalent of Isaac Lab ``apply_delta_pose`` for ``ik_rel``."""
    pose = np.asarray(pose7, dtype=np.float64).reshape(7)
    delta = np.asarray(delta_pose, dtype=np.float64).reshape(6)
    angle = float(np.linalg.norm(delta[3:]))
    if angle < 1e-12:
        r_delta = np.eye(3)
    else:
        axis = delta[3:] / angle
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        r_delta = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    target = pose7_to_mat(pose)
    target[:3, 3] = pose[:3] + delta[:3]
    target[:3, :3] = r_delta @ target[:3, :3]
    return mat_to_pose7(target)


# ---------- 실기 어댑터 ----------


def pulse_digital_out(
    io_set,
    pin_fire: int,
    pin_other: int,
    pulse_s: float,
    sleep=None,
    stop_event: threading.Event | None = None,
    on_high_dispatched=None,
    on_completed=None,
) -> None:
    """★2FG7 DO 펄스 시퀀스(순수 로직)★ — README에 보존한 2026-08-12 실측 계약.

    시퀀스: ①양쪽 라인 LOW(open/close 동시 assert 방지) → ②대상 핀 HIGH → ③pulse_s 대기
            → ④대상 핀 LOW. 실측에서 0.30s 펄스로 정상 트리거 확인(씹히면 0.5s로 상향).

    io_set/sleep을 주입받는 순수 함수라 ur_rtde 없이 pytest로 시퀀스를 고정한다(§4-2 정신).
    ⚠️ 이 함수는 pin_fire·pin_other ★두 핀만★ 만진다 — DO0(항상 ON·용도 미확인)을 포함한
      다른 출력을 건드리면 툴 체인저 낙하 위험. 호출측이 핀 번호를 검증할 것.
    """
    import time as _time

    slp = sleep if sleep is not None else _time.sleep
    io_set(pin_other, False)
    io_set(pin_fire, False)
    if stop_event is not None and stop_event.is_set():
        return
    io_set(pin_fire, True)
    try:
        if on_high_dispatched is not None:
            on_high_dispatched()
        if stop_event is None:
            slp(float(pulse_s))
        else:
            stop_event.wait(float(pulse_s))
    finally:
        io_set(pin_fire, False)
        if stop_event is not None:
            io_set(pin_other, False)
    if on_completed is not None:
        on_completed()


class RealRobot(RobotInterface):
    """UR5e + OnRobot 2FG7(DO pulse) 수신 전용 검증 어댑터.

    생성 인자:
      real_cfg        : 명시적으로 선택한 configs/real.local.yaml 전체 dict
      tcp_offset_path : configs/calib/tcp_offset.yaml 경로 (요소4 — 실측 파일)

    e-stop latch: emergency_stop() 이후 모든 명령 메서드는 RuntimeError.
    (SafetyGuard의 latch와 이중 안전 — 어느 한쪽만 우회해도 명령이 나가지 않는다)
    """

    def __init__(
        self,
        real_cfg: dict,
        tcp_offset_path: Path | str,
        *,
        receive_only: bool = False,
        motion_scope: str = "policy",
        operator_ack: bool = False,
    ) -> None:
        # Defense in depth.  This runs before configuration access, ur_rtde
        # imports, or any receive/control/IO socket.
        # ★F74★ 호출측이 필요한 최소 범위를 선언한다 — 기본값은 가장 엄격한 "policy"라
        #   범위를 명시하지 않은 호출은 캘리브레이션 완료까지 계속 막힌다(안전 기본값).
        if not receive_only:
            from deployment.contracts import require_motion_authorization

            require_motion_authorization(motion_scope, operator_ack=operator_ack)
        robot_cfg = real_cfg["robot"]
        self._ip: str = str(robot_cfg["ip"])
        self._servoj_lookahead: float = float(robot_cfg["servoj_lookahead_time"])
        self._servoj_gain: float = float(robot_cfg["servoj_gain"])
        self._rtde_frequency_hz = float(robot_cfg.get("rtde_frequency_hz", 500.0))
        self._movej_timeout_s = float(robot_cfg.get("movej_timeout_s", 30.0))
        if self._rtde_frequency_hz != 500.0:
            raise ValueError("[real] UR5e RTDE frequency must be explicitly fixed at 500 Hz")
        self._tcp_command_verified = bool(robot_cfg.get("tcp_command_verified", False))
        self._home_q = np.asarray(robot_cfg["home_joints_rad"], dtype=np.float64)
        self._receive_only = bool(receive_only)
        self._grip_stop_event = threading.Event()
        self._grip_io_lock = threading.Lock()
        self._grip_threads: set[threading.Thread] = set()
        self._grip_worker_error: BaseException | None = None
        self._grip_stop_incomplete = False
        self._grip_watchdog_shutdown = threading.Event()
        self._grip_watchdog_wake = threading.Event()
        self._grip_watchdog_ready = threading.Event()
        self._grip_watchdog_thread: threading.Thread | None = None
        self._grip_watchdog_error: BaseException | None = None
        # Linearizes every control/IO command with publication of a background
        # gripper fault and the software-stop sequence.  RLock is required
        # because command failure paths call emergency_stop while holding it.
        self._command_lock = threading.RLock()

        gripper_cfg = real_cfg.get("gripper", {})
        # ★★2026-08-12 채널 확정(README에 보존한 실기 실측)★★: 2FG7은 Compute Box
        #   (XML-RPC/Modbus)도 URCap도 아닌 ★로봇 컨트롤러 표준 디지털 출력 펄스★로 여닫힌다
        #   (포트스캔·URScript probe로 대안 부재 확인). 열기=DO6, 닫기=DO5, 팔과 같은 RTDE 연결.
        self._gripper_driver: str = str(gripper_cfg.get("driver", "do_pulse"))
        self._grip_do_open: int = int(gripper_cfg.get("do_open", 6))
        self._grip_do_close: int = int(gripper_cfg.get("do_close", 5))
        self._grip_pulse_s: float = float(gripper_cfg.get("pulse_s", 0.30))
        self._grip_dispatch_ack_timeout_s: float = float(
            gripper_cfg.get(
                "dispatch_ack_timeout_s", GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S
            )
        )
        if (
            not np.isfinite(self._grip_dispatch_ack_timeout_s)
            or not np.isclose(
                self._grip_dispatch_ack_timeout_s,
                GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "[real] gripper dispatch_ack_timeout_s must be exactly 0.040"
            )
        # ★안전(report §07)★: DO0은 모든 실측에서 항상 ON·용도 미확인(툴 체인저 잠금 후보) —
        #   개폐 핀이 실측 매핑(5/6) 밖이면 배선이 바뀐 것이므로 기동을 거부한다(fail-closed).
        if {self._grip_do_open, self._grip_do_close} != {5, 6}:
            raise ValueError(
                f"[real] gripper DO 핀 {self._grip_do_open}/{self._grip_do_close}이 실측 매핑"
                "(open=6, close=5)과 다르다 — 배선 재확인 후 real.local.yaml과 이 검사를 함께 갱신할 것"
            )
        self._gripper_closed_state: bool | None = None  # 마지막 명령 캐시 (중복 전송 방지)
        # Motion startup has no trustworthy physical width or command history.
        # It remains invalid until an explicit open pulse has been accepted,
        # its worker has completed, and the full 1.20 s transition has elapsed.
        self._gripper_command_state: float | None = None
        self._gripper_open_initialization_deadline_s: float | None = None
        self._gripper_open_initialization_ticket: _GripperPulseTicket | None = None
        self._gripper_pending_dispatch: _GripperPulseTicket | None = None
        self._gripper_dispatch_generation = 0
        self._arm_command_generation = 0
        self._arm_command_inflight: int | None = None
        self._stop_sequence_active = False

        # 요소4: flange→TCP 변환 — sim ee_frame과 같은 파일을 읽는다.
        self._T_flange_tcp = load_tcp_offset(tcp_offset_path)

        # ★base frame 정합(2026-07-28, 감사 C-2)★: RTDE는 UR 컨트롤러 Base 기준으로 보고하고
        #   sim은 URDF base_link 기준이다 — URDF가 둘의 차이를 z축 180°로 명시한다.
        #   변환 없이 쓰면 x·y 부호가 반대가 되어 정책이 거울상 좌표를 받는다.
        _conv = str(robot_cfg.get("tcp_frame_convention", "ur_base")).strip().lower()
        if _conv not in ("ur_base", "base_link"):
            raise ValueError(
                f"real.local.yaml robot.tcp_frame_convention='{_conv}' — 'ur_base' 또는 'base_link'만 허용"
            )
        self._needs_base_rot: bool = _conv == "ur_base"
        print(f"[real] tcp_frame_convention={_conv} → base_link 변환 "
              f"{'적용(Rz π)' if self._needs_base_rot else '없음'}  "
              f"⚠️ 첫 실기 세션에서 홈 자세 getActualTCPPose()를 sim FK와 대조해 확인할 것")

        self._estopped = False
        # Initialize every SDK handle before opening the first socket.  This
        # makes all constructor failure paths safe to clean up, including the
        # case where control connects but receive/IO/TCP validation fails.
        self._rtde_c = None
        self._rtde_r = None
        self._rtde_io = None

        # --- ur_rtde 지연 import + 연결 ---
        # ⚠️ TODO(확인 필요): pip 패키지명 ur-rtde의 파이썬 모듈이
        #   rtde_control / rtde_receive, 클래스가 RTDEControlInterface /
        #   RTDEReceiveInterface가 맞는지 공식 문서로 확인.
        try:
            from rtde_receive import RTDEReceiveInterface  # type: ignore
            if not self._receive_only:
                from rtde_control import RTDEControlInterface  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "[real] ur_rtde 미설치 — env_ur_real 활성화 후 bundle root에서 "
                "`python -m pip install --no-index --find-links "
                "deployment/wheelhouse --require-hashes -r "
                "deployment/requirements-win11.lock.txt` 후 사용할 것"
            ) from e

        try:
            # ⚠️ TODO(확인 필요): 생성자 추가 인자(frequency, flags 등) 필요 여부.
            self._rtde_c = (
                None
                if self._receive_only
                else RTDEControlInterface(self._ip, self._rtde_frequency_hz)
            )
            self._rtde_r = RTDEReceiveInterface(self._ip, self._rtde_frequency_hz)
        except Exception as e:
            self._disconnect_rtde_resources()
            raise RuntimeError(
                f"[real] UR5e 연결 실패 ({self._ip}) — 전원/네트워크/원격제어 모드 확인: {e}"
            ) from e
        # ★★2026-08-12(게이트 1b 실측) 그리퍼 IO를 ★연결 시점에★ 초기화★★
        #   [실측] 첫 set_gripper 호출이 ★817.9 ms★ 걸렸다(이후 6회는 0.3 ms —
        #     과거 실기 기록; 신규 기록은 deployment/evidence/.../report.json). 원인은 펄스 경로 안의
        #     지연 import + RTDEIOInterface 연결이었다.
        #   [왜 문제인가] 배포 루프는 20Hz(50 ms 예산)다. 그 비용을 ★제어 루프 첫 파지 순간★에
        #     내면 한 스텝이 16배 초과해 servoL 스트리밍이 끊긴다 — 로봇이 실제로 움직이는
        #     중에 발생하는 지연이라 위험하다. 연결 시점으로 옮기면 루프는 스레드 시작(0.3 ms)만 낸다.
        #   [부수 이득] ur_rtde의 rtde_io 미설치·IO 연결 실패를 ★기동 시점★에 드러낸다
        #     (fail-closed — 루프 도중 예외보다 안전).
        if self._gripper_driver == "do_pulse" and not self._receive_only:
            try:
                from rtde_io import RTDEIOInterface  # type: ignore

                self._rtde_io = RTDEIOInterface(self._ip)
                print(f"[real] 그리퍼 IO 준비 완료 (do_pulse · open=DO{self._grip_do_open}"
                      f" · close=DO{self._grip_do_close} · pulse={self._grip_pulse_s}s)")
            except Exception as e:
                self._disconnect_rtde_resources()
                raise RuntimeError(
                    f"[real] 그리퍼 IO(RTDEIOInterface) 초기화 실패 ({self._ip}): {e}" \
                    "\n"
                    "  점검: ① env_ur_real에 ur_rtde 설치 ② 펜던트 ★Remote Control 모드★"
                    " ③ 다른 클라이언트의 IO 점유"
                ) from e
        print(
            f"[real] UR5e 연결 완료: {self._ip} "
            f"(mode={'receive-only' if self._receive_only else 'control'}, "
            f"driver={self._gripper_driver})"
        )

        try:
            tcp_source = self._rtde_c if self._rtde_c is not None else self._rtde_r
            if not hasattr(tcp_source, "getTCPOffset"):
                raise RuntimeError("[real] RTDE interface cannot verify the active controller TCP")
            active_tcp = np.asarray(tcp_source.getTCPOffset(), dtype=np.float64).reshape(6)
            if not np.isfinite(active_tcp).all() or np.max(np.abs(active_tcp)) > 1.0e-6:
                raise RuntimeError(
                    "[real] controller TCP must be exactly flange/zero; "
                    f"getTCPOffset={active_tcp.tolist()}"
                )
        except BaseException:
            self._disconnect_rtde_resources()
            raise

        if self._gripper_driver == "do_pulse" and not self._receive_only:
            try:
                self._start_gripper_dispatch_watchdog()
            except BaseException:
                self._disconnect_rtde_resources()
                raise

        # ⚠️ TODO(확인 필요 — 이중 offset 함정): getActualTCPPose는 "컨트롤러에
        #   설정된 TCP" 기준 포즈를 준다. 이 클래스는 요소4의 T_flange_tcp를 직접
        #   곱하므로, 컨트롤러 측 TCP는 반드시 0(플랜지)이어야 한다.
        #   배포 PC에서 setTcp([0]*6) 또는 펜던트 설정 확인 후 이 TODO 해소.
        #   (대안: getForwardKinematics로 플랜지 포즈를 직접 받는 방법 확인)

    # ----- 내부 유틸 -----

    def _get_command_lock(self) -> threading.RLock:
        """Return the command/fault serialization lock (lazy for pure test doubles)."""
        lock = getattr(self, "_command_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._command_lock = lock
        return lock

    def _start_gripper_dispatch_watchdog(self) -> None:
        """Pre-start the independent 40 ms logical dispatch-deadline monitor."""
        with self._get_command_lock():
            if getattr(self, "_grip_watchdog_thread", None) is not None:
                if getattr(self, "_grip_watchdog_error", None) is not None:
                    raise RuntimeError(
                        f"[real] gripper dispatch watchdog failed: {self._grip_watchdog_error}"
                    ) from self._grip_watchdog_error
                if getattr(self, "_grip_watchdog_ready", threading.Event()).is_set():
                    return
                raise RuntimeError("[real] gripper dispatch watchdog is not ready")
            self._grip_watchdog_shutdown = getattr(
                self, "_grip_watchdog_shutdown", threading.Event()
            )
            self._grip_watchdog_wake = getattr(
                self, "_grip_watchdog_wake", threading.Event()
            )
            self._grip_watchdog_ready = getattr(
                self, "_grip_watchdog_ready", threading.Event()
            )
            self._grip_watchdog_error = None

        def _watch() -> None:
            self._grip_watchdog_ready.set()
            try:
                while not self._grip_watchdog_shutdown.is_set():
                    with self._get_command_lock():
                        ticket = getattr(self, "_gripper_pending_dispatch", None)
                    if ticket is None or ticket.status != "pending":
                        self._grip_watchdog_wake.wait(timeout=0.05)
                        self._grip_watchdog_wake.clear()
                        continue

                    remaining_s = ticket.dispatch_deadline_s - time.monotonic()
                    if remaining_s > 0.0:
                        self._grip_watchdog_wake.wait(timeout=min(remaining_s, 0.05))
                        self._grip_watchdog_wake.clear()
                        continue

                    error = TimeoutError(
                        "[real] gripper HIGH dispatch acknowledgement exceeded "
                        f"{float(getattr(self, '_grip_dispatch_ack_timeout_s', GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S)):.3f}s"
                    )
                    published = False
                    with self._get_command_lock():
                        if (
                            getattr(self, "_gripper_pending_dispatch", None) is ticket
                            and ticket.status == "pending"
                            and time.monotonic() >= ticket.dispatch_deadline_s
                        ):
                            published = self._resolve_gripper_dispatch_locked(
                                ticket, "timeout", error
                            )
                            if published:
                                self._grip_worker_error = error
                                self._latch_emergency_stop_locked(error)
                    if published:
                        # The logical latch is already visible. Physical stop
                        # traffic is deliberately detached from this monitor.
                        self._finish_gripper_dispatch_timeout_stop()
            except BaseException as exc:
                with self._get_command_lock():
                    self._grip_watchdog_error = exc
                    if getattr(self, "_grip_worker_error", None) is None:
                        self._grip_worker_error = exc
                    self._latch_emergency_stop_locked(exc)
            finally:
                self._grip_watchdog_ready.set()

        worker = threading.Thread(
            target=_watch,
            name="gripper-dispatch-watchdog",
            daemon=True,
        )
        with self._get_command_lock():
            self._grip_watchdog_thread = worker
        try:
            # Thread.start()/ready wait are initialization work and never hold
            # the command lock. No command-capable adapter is published yet.
            worker.start()
        except BaseException as exc:
            with self._get_command_lock():
                self._grip_watchdog_thread = None
                self._grip_watchdog_error = exc
                self._latch_emergency_stop_locked(exc)
            raise RuntimeError("[real] gripper dispatch watchdog failed to start") from exc
        if not self._grip_watchdog_ready.wait(timeout=0.2):
            error = TimeoutError("[real] gripper dispatch watchdog startup timed out")
            self._grip_watchdog_shutdown.set()
            self._grip_watchdog_wake.set()
            worker.join(timeout=0.2)
            with self._get_command_lock():
                self._grip_watchdog_error = error
                self._latch_emergency_stop_locked(error)
            raise error
        with self._get_command_lock():
            startup_error = self._grip_watchdog_error
        if startup_error is not None:
            raise RuntimeError("[real] gripper dispatch watchdog startup failed") from startup_error

    def _stop_gripper_dispatch_watchdog(self) -> None:
        """Stop/join the pre-started watchdog without holding the command lock."""
        shutdown = getattr(self, "_grip_watchdog_shutdown", None)
        wake = getattr(self, "_grip_watchdog_wake", None)
        worker = getattr(self, "_grip_watchdog_thread", None)
        if shutdown is not None:
            shutdown.set()
        if wake is not None:
            wake.set()
        if worker is None or worker is threading.current_thread():
            return
        try:
            worker.join(timeout=0.2)
        except RuntimeError:
            return
        if worker.is_alive():
            self._grip_stop_incomplete = True

    @staticmethod
    def _resolve_gripper_dispatch_locked(
        ticket: _GripperPulseTicket,
        status: str,
        error: BaseException | None = None,
    ) -> bool:
        """Resolve one dispatch generation exactly once while its state lock is held."""
        if ticket.status != "pending":
            return False
        ticket.status = str(status)
        ticket.error = error
        if status != "acked":
            ticket.cancelled.set()
        ticket.dispatch_resolved.set()
        return True

    def _reserve_arm_command(self) -> int:
        """Publish one arm SDK generation, mutually exclusive with gripper dispatch."""
        with self._get_command_lock():
            self._assert_alive()
            generation = int(getattr(self, "_arm_command_generation", 0)) + 1
            self._arm_command_generation = generation
            self._arm_command_inflight = generation
            return generation

    def _finish_arm_command(
        self,
        generation: int,
        *,
        sdk_result: object = None,
        error: BaseException | None = None,
    ) -> BaseException | None:
        """Commit one arm SDK result under the state lock; never performs SDK I/O."""
        with self._get_command_lock():
            if getattr(self, "_arm_command_inflight", None) != generation:
                return error or RuntimeError("[real] arm command generation was invalidated")
            self._arm_command_inflight = None
            failure = error
            if failure is None and sdk_result is not True:
                failure = RuntimeError("[real] arm SDK command was not acknowledged")
            if failure is None and (
                getattr(self, "_estopped", False)
                or getattr(self, "_grip_worker_error", None) is not None
                or getattr(self, "_gripper_pending_dispatch", None) is not None
            ):
                failure = RuntimeError("[real] arm command completed after a stop/fault")
            if failure is not None:
                self._latch_emergency_stop_locked(failure)
            return failure

    def _finish_gripper_dispatch_timeout_stop(self) -> None:
        """Start best-effort physical stopping without extending the 40 ms caller bound."""
        worker = threading.Thread(
            target=self.emergency_stop,
            name="gripper-dispatch-timeout-stop",
            daemon=True,
        )
        worker.start()

    def _latch_emergency_stop_locked(self, error: BaseException | None = None) -> None:
        """Publish the logical stop atomically; this helper performs no SDK I/O."""
        pending = getattr(self, "_gripper_pending_dispatch", None)
        if pending is not None:
            terminal_error = error or RuntimeError("gripper dispatch cancelled by emergency stop")
            self._resolve_gripper_dispatch_locked(pending, "cancelled", terminal_error)
            pending.cancelled.set()
        self._arm_command_inflight = None
        self._estopped = True
        self._gripper_closed_state = None
        self._invalidate_gripper_command_state()
        stop_event = getattr(self, "_grip_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        wake = getattr(self, "_grip_watchdog_wake", None)
        if wake is not None:
            wake.set()

    def _invalidate_gripper_command_state(self) -> None:
        with self._get_command_lock():
            pending = getattr(self, "_gripper_pending_dispatch", None)
            if pending is not None:
                self._resolve_gripper_dispatch_locked(
                    pending,
                    "cancelled",
                    pending.error or RuntimeError("gripper command state invalidated"),
                )
                pending.cancelled.set()
            self._gripper_command_state = None
            self._gripper_open_initialization_deadline_s = None
            self._gripper_open_initialization_ticket = None
            self._gripper_pending_dispatch = None

    def _refresh_gripper_command_state(self) -> None:
        with self._get_command_lock():
            self._refresh_gripper_command_state_locked()

    def _refresh_gripper_command_state_locked(self) -> None:
        """Publish known-open only after pulse completion plus 1.20 seconds."""
        if getattr(self, "_gripper_command_state", None) is not None:
            return
        if (
            getattr(self, "_estopped", False)
            or getattr(self, "_grip_worker_error", None) is not None
            or getattr(self, "_grip_stop_incomplete", False)
        ):
            return
        ticket = getattr(self, "_gripper_open_initialization_ticket", None)
        if ticket is None or not ticket.finished.is_set():
            return
        if (
            ticket.error is not None
            or ticket.pulse_error is not None
            or ticket.status != "acked"
            or ticket.high_dispatched_at_s is None
            or ticket.completed_at_s is None
        ):
            self._invalidate_gripper_command_state()
            return
        # The deadline is derived from successful pulse completion, never
        # enqueue/API-call time.  A worker delayed by scheduling or an earlier
        # pulse therefore cannot become ready immediately when it finally runs.
        deadline = ticket.completed_at_s + GRIPPER_COMMAND_TRANSITION_S
        self._gripper_open_initialization_deadline_s = deadline
        if time.perf_counter() >= deadline and getattr(self, "_gripper_closed_state", None) is False:
            self._gripper_command_state = GRIPPER_COMMAND_OPEN
            self._gripper_open_initialization_deadline_s = None
            self._gripper_open_initialization_ticket = None

    def _assert_alive(self) -> None:
        if getattr(self, "_grip_worker_error", None) is not None:
            # A background DO failure is a command-path failure, not merely a
            # health warning for the next loop iteration.  Publish the latch
            # without running any possibly blocking stop SDK here: arm entry
            # checks must remain bounded even if RTDE-IO is wedged.
            error = self._grip_worker_error
            self._estopped = True
            self._gripper_closed_state = None
            self._invalidate_gripper_command_state()
            stop_event = getattr(self, "_grip_stop_event", None)
            if stop_event is not None:
                stop_event.set()
            raise RuntimeError(f"[real] gripper worker failed: {error}") from error
        if self._estopped:
            raise RuntimeError("[real] e-stop latch 상태 — 재기동 전까지 명령 불가")
        if self._receive_only:
            raise RuntimeError("[real] receive-only adapter refuses all commands")
        if getattr(self, "_gripper_pending_dispatch", None) is not None:
            raise RuntimeError("[real] gripper HIGH dispatch acknowledgement is pending")
        if getattr(self, "_arm_command_inflight", None) is not None:
            raise RuntimeError("[real] another arm command SDK call is in flight")

    def _disconnect_rtde_resources(self) -> None:
        """Best-effort cleanup for both partial initialization and normal close."""
        for name, attr in (
            ("IO", "_rtde_io"),
            ("control", "_rtde_c"),
            ("receive", "_rtde_r"),
        ):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                if hasattr(obj, "disconnect"):
                    obj.disconnect()
            except Exception as exc:
                print(f"[real][warn] RTDE {name} disconnect 실패: {exc}")
            finally:
                setattr(self, attr, None)

    @property
    def home_joints_rad(self) -> np.ndarray:
        """real.local.yaml의 홈 포즈 사본 (Deploy/Calib 스크립트용)."""
        return self._home_q.copy()

    # ----- RobotInterface 구현 -----

    def _capture_rtde_snapshot(self) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float
    ]:
        """Capture one newly-arrived and internally coherent RTDE package.

        ``RTDEReceiveInterface`` getters expose the latest package cached by its
        receive thread.  A host timestamp taken around arbitrary getters cannot
        establish freshness.  Start from the current controller timestamp,
        wait until it advances after our request, then accept the field reads
        only if a second timestamp read is identical.  The monotonic request /
        observation bounds are retained for conservative age/skew evidence.
        """
        timeout_s = max(0.05, 10.0 / self._rtde_frequency_hz)
        deadline = time.perf_counter() + timeout_s
        while True:
            request_lower_s = time.perf_counter()
            baseline = float(self._rtde_r.getTimestamp())
            selected = baseline
            while selected <= baseline:
                if time.perf_counter() >= deadline:
                    raise RuntimeError("[real] timed out waiting for a fresh RTDE package")
                time.sleep(0)
                selected = float(self._rtde_r.getTimestamp())
            observed_upper_s = time.perf_counter()
            if not np.isfinite(selected) or selected < 0.0:
                raise RuntimeError(f"[real] invalid RTDE controller timestamp: {selected!r}")

            q = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64).reshape(_NUM_JOINTS)
            qd = np.asarray(self._rtde_r.getActualQd(), dtype=np.float64).reshape(_NUM_JOINTS)
            pose_ur = np.asarray(self._rtde_r.getActualTCPPose(), dtype=np.float64).reshape(6)
            tcp_force = np.asarray(
                self._rtde_r.getActualTCPForce(), dtype=np.float64
            ).reshape(6)
            confirmed = float(self._rtde_r.getTimestamp())
            if confirmed == selected:
                return (
                    q,
                    qd,
                    pose_ur,
                    tcp_force,
                    selected,
                    request_lower_s,
                    observed_upper_s,
                )
            # A 500 Hz package crossed the multi-get sequence. Retry against a
            # later package rather than silently composing fields from two.
            if time.perf_counter() >= deadline:
                raise RuntimeError("[real] could not capture one coherent RTDE package")

    def get_state(self) -> RobotState:
        """RTDEReceive 스냅샷 → RobotState (quat=wxyz, TCP offset 적용 — 요소4)."""
        # ⚠️ TODO(확인 필요): getActualQ()/getActualQd()가 각각 관절각[rad] /
        #   관절속도[rad/s]의 길이 6 list를 반환하는지 (문서: RTDEReceiveInterface).
        try:
            (
                q,
                qd,
                pose_ur,
                tcp_force,
                source_timestamp_s,
                source_lower_s,
                source_upper_s,
            ) = self._capture_rtde_snapshot()
        except BaseException:
            self.emergency_stop()
            raise

        # ⚠️ TODO(확인 필요): getActualTCPPose() 반환이 [x,y,z,rx,ry,rz]
        #   (m, rotation-vector[rad], base frame)인지 확인.
        flange_pose7 = ur_pose_to_pose7(pose_ur)  # rotvec → wxyz quat
        # ★base frame 정합(감사 C-2)★: 컨트롤러 Base → URDF base_link (Rz π).
        #   ★TCP offset 적용 '전에' 변환한다★ — offset은 flange 로컬 프레임 기준 합성이라
        #   base 회전과 순서를 바꾸면 결과가 달라진다(먼저 base를 맞추고 그 위에 offset).
        if self._needs_base_rot:
            flange_pose7 = ur_base_to_base_link_pose7(flange_pose7)
        tcp_pose = apply_tcp_offset(flange_pose7, self._T_flange_tcp)  # ★요소4★

        # ⚠️ TODO(확인 필요): getActualTCPForce()가 (6,) wrench [Fx,Fy,Fz,Tx,Ty,Tz]
        #   [N, Nm]를 "base frame"으로 주는지 (RobotState.tcp_force 계약은 base frame).
        try:
            # ★force도 같은 base frame 계약★ — pose만 돌리고 force를 두면 안전판정이
            #   뒤집힌 축으로 이뤄진다(감사 C-2 지적: tcp_force도 동일 회전 필요).
            if self._needs_base_rot:
                tcp_force = ur_base_to_base_link_wrench(tcp_force)
        except BaseException:
            self.emergency_stop()
            raise

        # Student state is a deterministic command-history signal, never a
        # DO/DI echo or fabricated physical width.  Hardware/software stops
        # invalidate it.  Motion startup stays unknown until an explicit-open
        # command has fully settled for 1.20 s.
        try:
            stopped = bool(
                getattr(self._rtde_r, "isProtectiveStopped", lambda: False)()
            ) or bool(
                getattr(self._rtde_r, "isEmergencyStopped", lambda: False)()
            )
            if hasattr(self._rtde_r, "isSafeguardStopped"):
                stopped = stopped or bool(self._rtde_r.isSafeguardStopped())
        except BaseException:
            self.emergency_stop()
            raise
        if (
            stopped
            or getattr(self, "_estopped", False)
            or getattr(self, "_grip_worker_error", None) is not None
        ):
            self._invalidate_gripper_command_state()
        else:
            self._refresh_gripper_command_state()

        # Do not relabel a host timestamp as sensor time.  ur_rtde exposes the
        # controller timestamp carried by the current RTDE data package; it is
        # a controller-clock value (seconds), not time.perf_counter().  Record
        # the host arrival separately, immediately after the last receive call.
        host_arrival_timestamp = time.perf_counter()

        return RobotState(
            joint_pos=q,
            joint_vel=qd,
            tcp_pose=tcp_pose,
            tcp_force=tcp_force,
            timestamp=host_arrival_timestamp,
            gripper_command_state=getattr(self, "_gripper_command_state", None),
            source_timestamp_s=source_timestamp_s,
            source_timestamp_domain="ur_rtde_controller_clock",
            source_acquisition_host_lower_s=source_lower_s,
            source_acquisition_host_upper_s=source_upper_s,
        )

    def send_joint_target(self, q_target: np.ndarray, duration_s: float) -> None:
        """servoJ 스트리밍 추종 — duration_s = 1/control_hz (rollout이 넘겨줌).

        ⚠️ 캘리브레이션/홈 이동에 쓰지 말 것 (interfaces.py 계약) → move_to_joints.
        """
        self._assert_alive()
        q = np.asarray(q_target, dtype=np.float64).reshape(_NUM_JOINTS)
        duration = float(duration_s)
        if not np.isfinite(q).all() or not np.isfinite(duration) or duration <= 0.0:
            self.emergency_stop()
            raise ValueError("[real] non-finite/invalid servoJ target or duration")
        # ⚠️ TODO(확인 필요): servoJ 정확한 시그니처 —
        #   servoJ(q, speed, acceleration, time, lookahead_time, gain) 추정
        #   (real.local.yaml robot 섹션 주석과 동일 가정). speed/accel이 실제로 무시되는
        #   reserved 인자인지, lookahead_time [0.03,0.2] / gain [100,2000] 범위인지
        #   공식 문서로 확인 후 확정.
        generation = self._reserve_arm_command()
        try:
            ok = self._rtde_c.servoJ(
                q.tolist(),
                _SERVOJ_SPEED,
                _SERVOJ_ACCEL,
                duration,
                self._servoj_lookahead,
                self._servoj_gain,
            )
        except BaseException as exc:
            self._finish_arm_command(generation, error=exc)
            self.emergency_stop()
            raise
        failure = self._finish_arm_command(generation, sdk_result=ok)
        if failure is not None:
            self.emergency_stop()
            raise failure

    def send_task_delta(
        self,
        delta_pose: np.ndarray,
        current_tcp_pose: np.ndarray,
        duration_s: float,
        *,
        mapped_raw_delta: np.ndarray,
    ) -> AppliedArmCommandReceipt:
        """Apply the Isaac Lab relative-pose convention and stream it with ``servoL``."""
        self._assert_alive()
        if not self._tcp_command_verified:
            raise RuntimeError(
                "[real] ik_rel 전송 차단: real.local.yaml robot.tcp_command_verified=false. "
                "controller TCP=flange 및 base-frame 왕복 검증 후에만 활성화할 것"
            )
        # Own immutable-by-convention snapshots for the entire SDK call.  A
        # caller mutating its arrays concurrently cannot change the returned
        # receipt after the servo target was derived.
        delta = np.array(delta_pose, dtype=np.float64, copy=True)
        mapped_raw = np.array(mapped_raw_delta, dtype=np.float64, copy=True)
        current = np.array(current_tcp_pose, dtype=np.float64, copy=True)
        duration = float(duration_s)
        if (
            delta.shape != (6,)
            or mapped_raw.shape != (6,)
            or current.shape != (7,)
            or not np.isfinite(delta).all()
            or not np.isfinite(mapped_raw).all()
            or not np.isfinite(current).all()
            or not np.isfinite(duration)
            or duration <= 0.0
        ):
            self.emergency_stop()
            raise ValueError("[real] non-finite/invalid servoL input or duration")
        tcp_target_b = apply_task_delta_pose7(current, delta)
        flange_target_b = mat_to_pose7(
            pose7_to_mat(tcp_target_b) @ np.linalg.inv(self._T_flange_tcp)
        )
        flange_target_ur = (
            ur_base_to_base_link_pose7(flange_target_b)
            if self._needs_base_rot
            else flange_target_b
        )
        pose_ur = np.concatenate(
            [flange_target_ur[:3], quat_to_rotvec(flange_target_ur[3:])]
        )
        generation = self._reserve_arm_command()
        try:
            ok = self._rtde_c.servoL(
                pose_ur.tolist(),
                _SERVOJ_SPEED,
                _SERVOJ_ACCEL,
                duration,
                self._servoj_lookahead,
                self._servoj_gain,
            )
        except BaseException as exc:
            self._finish_arm_command(generation, error=exc)
            self.emergency_stop()
            raise
        failure = self._finish_arm_command(generation, sdk_result=ok)
        if failure is not None:
            self.emergency_stop()
            raise failure
        # The real adapter has no additional command delay or final workspace
        # clamp, so the guarded servoL delta is an identity inner stage.  The
        # separately supplied mapped raw command still preserves any outer
        # SafetyGuard intervention.  The generation is reserved under the
        # command lock and is monotonic across this adapter instance.
        return AppliedArmCommandReceipt.from_commands(
            sequence=generation,
            delayed_mapped_raw=mapped_raw,
            post_delay_pre_floor=delta,
            applied=delta,
        )

    def move_to_joints(self, q: np.ndarray, speed_scale: float = 0.1) -> None:
        """Interruptible asynchronous moveJ with health/timeout supervision."""
        self._assert_alive()
        q = np.asarray(q, dtype=np.float64).reshape(_NUM_JOINTS)
        scale = float(speed_scale)
        if not np.isfinite(q).all() or not np.isfinite(scale) or not 0.01 <= scale <= 1.0:
            self.emergency_stop()
            raise ValueError("[real] non-finite/invalid moveJ target or speed_scale")
        speed = scale * _MOVEJ_NOMINAL_SPEED
        # ⚠️ TODO(확인 필요): moveJ 시그니처 —
        #   moveJ(q, speed, acceleration, asynchronous=False) 추정. 블로킹(동기)
        #   호출이 기본인지, 반환값(bool)으로 실패를 알리는지 확인.
        generation = self._reserve_arm_command()
        try:
            ok = self._rtde_c.moveJ(q.tolist(), speed, _MOVEJ_ACCEL, True)
            if ok is not True:
                raise RuntimeError(f"[real] async moveJ rejected: q={np.round(q, 3).tolist()}")
            deadline = time.monotonic() + self._movej_timeout_s
            while True:
                if not self.is_ok():
                    raise RuntimeError("robot health failed during moveJ")
                q_now = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64).reshape(_NUM_JOINTS)
                if not np.isfinite(q_now).all():
                    raise RuntimeError("non-finite joint state during moveJ")
                if float(np.max(np.abs(q_now - q))) <= 0.01:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"moveJ exceeded {self._movej_timeout_s:g}s timeout")
                time.sleep(0.02)
        except BaseException as exc:
            self._finish_arm_command(generation, error=exc)
            self.emergency_stop()
            raise
        failure = self._finish_arm_command(generation, sdk_result=True)
        if failure is not None:
            self.emergency_stop()
            raise failure

    def set_gripper(self, closed: bool) -> None:
        """그리퍼 개폐 — 2FG7 DO 펄스(★2026-08-12 실측, README 배포 게이트에 보존★).

        edge-trigger: 같은 상태 재명령은 생략한다(rollout이 매 스텝 호출하므로 필수 —
        상태가 바뀔 때만 0.3s 펄스가 백그라운드로 나간다). 힘·속도·폭은 프리셋 고정
        (DO 방식의 한계 — 조절하려면 Compute Box Web Client에서 프리셋 변경).
        """
        closed = bool(closed)
        ticket: _GripperPulseTicket | None = None
        pre_dispatch_error: BaseException | None = None

        # Phase 1 validates the requested edge while holding only the state
        # lock. Thread.start() waits for CPython's bootstrap event, so worker
        # creation/start is also deferred until after this block.
        with self._get_command_lock():
            self._assert_alive()
            self._refresh_gripper_command_state()

            # Fail closed on a close-first startup.  Without width feedback we
            # cannot choose a valid starting point for the 24-step trace.
            if self._gripper_command_state is None and closed:
                error = RuntimeError(
                    "[real] gripper command state unknown — explicit open and "
                    "1.20 s settling are required before close"
                )
                self._latch_emergency_stop_locked(error)
                pre_dispatch_error = error
            else:
                edge = (
                    self._gripper_closed_state is None
                    or closed != self._gripper_closed_state
                )
                if not edge:
                    if self._gripper_command_state is not None:
                        # A prior target-pin HIGH was already acknowledged. Each
                        # subsequent accepted 20 Hz command advances once.
                        self._gripper_command_state = advance_gripper_command_state(
                            self._gripper_command_state, closed
                        )
                    return

                if self._gripper_driver != "do_pulse":
                    raise ValueError(
                        f"[real] 알 수 없는 gripper.driver: {self._gripper_driver!r} "
                        "(real.local.yaml: do_pulse — 2026-08-12 채널 확정)"
                    )
                if getattr(self, "_grip_threads", set()):
                    error = RuntimeError(
                        "[real] gripper reversal cannot dispatch while the previous pulse is active"
                    )
                    self._grip_worker_error = error
                    self._latch_emergency_stop_locked(error)
                    pre_dispatch_error = error

        if pre_dispatch_error is not None:
            self._finish_gripper_dispatch_timeout_stop()
            raise pre_dispatch_error

        # The helper publishes its pending generation before Thread.start(),
        # so arm calls still fail closed during worker bootstrap without an
        # outer command-lock owner waiting for that bootstrap.
        ticket = self._set_gripper_do_pulse(closed)

        assert ticket is not None
        timeout_s = float(
            getattr(
                self,
                "_grip_dispatch_ack_timeout_s",
                GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S,
            )
        )
        remaining_s = max(0.0, ticket.dispatch_deadline_s - time.monotonic())
        if not ticket.dispatch_resolved.wait(timeout=remaining_s):
            error = TimeoutError(
                f"[real] gripper HIGH dispatch acknowledgement exceeded {timeout_s:.3f}s"
            )
            # The worker never holds the command lock across an SDK call.  The
            # timeout can therefore publish one atomic failed generation here
            # without waiting for the blocked RTDE-IO transport to return.
            timed_out = False
            with self._get_command_lock():
                if (
                    ticket.status == "pending"
                    and getattr(self, "_gripper_pending_dispatch", None) is ticket
                ):
                    timed_out = self._resolve_gripper_dispatch_locked(
                        ticket, "timeout", error
                    )
                    self._grip_worker_error = error
                    self._latch_emergency_stop_locked(error)
                elif ticket.status != "acked":
                    error = ticket.error or RuntimeError(
                        "[real] gripper dispatch resolved unsuccessfully at deadline"
                    )
                    timed_out = True
            if timed_out:
                self._finish_gripper_dispatch_timeout_stop()
                if isinstance(error, TimeoutError):
                    raise error
                raise RuntimeError(str(error)) from error

        # Phase 2 commits the logical target/state only after the exact target
        # pin HIGH SDK call returned successfully.  A worker fault or stop that
        # won the race invalidates the ticket and cannot update observation.
        commit_error: BaseException | None = None
        with self._get_command_lock():
            if (
                getattr(self, "_gripper_pending_dispatch", None) is not ticket
                or ticket.status != "acked"
            ):
                error = ticket.error or RuntimeError(
                    "[real] gripper dispatch ticket was invalidated before commit"
                )
                self._grip_worker_error = error
                self._latch_emergency_stop_locked(error)
                commit_error = error
            else:
                self._gripper_pending_dispatch = None
                if ticket.error is not None or ticket.high_dispatched_at_s is None:
                    error = ticket.error or RuntimeError(
                        "[real] gripper HIGH dispatch resolved without success"
                    )
                    self._grip_worker_error = error
                    self._latch_emergency_stop_locked(error)
                    commit_error = error
                else:
                    self._assert_alive()
                    self._gripper_closed_state = closed
                    if self._gripper_command_state is None:
                        # Explicit-open startup remains invalid until this exact pulse
                        # finishes successfully and another 1.20 seconds have elapsed.
                        self._gripper_open_initialization_ticket = ticket
                    else:
                        self._gripper_command_state = advance_gripper_command_state(
                            self._gripper_command_state, closed
                        )

        if commit_error is not None:
            self._finish_gripper_dispatch_timeout_stop()
            if isinstance(commit_error, TimeoutError):
                raise commit_error
            raise RuntimeError(str(commit_error)) from commit_error

    def _set_gripper_do_pulse(self, closed: bool) -> _GripperPulseTicket:
        """★2FG7 DO 펄스 드라이버(2026-08-12 채널 확정)★ — 열기 DO6 / 닫기 DO5.

        [비블로킹 계약] 펄스(0.3s)는 20Hz 제어 루프(50ms 예산)를 막으면
        안 되므로 ★백그라운드 스레드★에서 실행한다. set_gripper의 edge 검사(같은 상태 재명령
        생략)와 아래 lock이 겹침 펄스(양 라인 동시 HIGH)를 이중으로 차단한다.

        [실측 검증 상태] setStandardDigitalOut(pin, level) 시그니처와 DO5/6 트리거는
        2026-08-12 legacy 세션에서 확인됨(PolyScope 5.14.6, 개폐 중 관절 불변 5.7e-05 rad).
        현재 고정 `ur-rtde==1.6.2`와 ★이 클래스 경로★로 재검증하기 전까지
        real.local.yaml driver_verified=false 유지(readiness gate가 막는 것이 정상).

        ⚠️ 전제: 펜던트 Remote Control 모드. DO5·6 외 출력은 절대 건드리지 않는다.
        """
        if self._rtde_io is None:
            # ★연결 시점에 준비된다★(__init__) — 여기서 초기화하면 제어 루프 첫 파지에서
            #   800ms급 블로킹이 발생한다(게이트 1b 실측). None이면 계약 위반이다.
            raise RuntimeError("[real] 그리퍼 IO 미초기화 — RealRobot 생성 경로 확인")

        # Reserve a unique generation before the worker can enter any SDK.
        # Arm commands observe this flag both before and after acquiring the
        # command lock, so they fail without waiting for unresolved RTDE-IO.
        with self._get_command_lock():
            watchdog_ready = getattr(self, "_grip_watchdog_ready", None)
            if watchdog_ready is None or not watchdog_ready.is_set():
                raise RuntimeError("[real] gripper dispatch watchdog is not ready")
            if getattr(self, "_grip_watchdog_error", None) is not None:
                raise RuntimeError(
                    f"[real] gripper dispatch watchdog failed: {self._grip_watchdog_error}"
                ) from self._grip_watchdog_error
            if getattr(self, "_gripper_pending_dispatch", None) is not None:
                raise RuntimeError("[real] another gripper dispatch generation is pending")
            if getattr(self, "_arm_command_inflight", None) is not None:
                raise RuntimeError("[real] arm command is in flight; gripper dispatch refused")
            if getattr(self, "_grip_threads", set()):
                raise RuntimeError("[real] another gripper pulse worker is active")
            self._assert_alive()
            generation = int(getattr(self, "_gripper_dispatch_generation", 0)) + 1
            self._gripper_dispatch_generation = generation
            ticket = _GripperPulseTicket(
                closed,
                generation,
                float(
                    getattr(
                        self,
                        "_grip_dispatch_ack_timeout_s",
                        GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S,
                    )
                ),
            )
            self._gripper_pending_dispatch = ticket
            self._grip_watchdog_wake.set()
        pin_fire = self._grip_do_close if closed else self._grip_do_open
        pin_other = self._grip_do_open if closed else self._grip_do_close

        def _attempt_safe_low_after_late_high() -> None:
            """Best-effort only: an already-entered SDK may have applied HIGH."""
            ok = self._rtde_io.setStandardDigitalOut(pin_fire, False)
            if ok is not True:
                self._grip_stop_incomplete = True
                raise RuntimeError("gripper late-HIGH safe-low was not acknowledged")

        def _io_set_with_fault_barrier(pin: int, level: bool) -> None:
            # Only the logical precondition check is serialized.  The SDK call
            # itself must remain outside the global command lock because the
            # transport has no bounded-call guarantee.
            if level:
                with self._get_command_lock():
                    if (
                        ticket.status != "pending"
                        or
                        ticket.cancelled.is_set()
                        or getattr(self, "_estopped", False)
                        or self._grip_stop_event.is_set()
                        or getattr(self, "_gripper_pending_dispatch", None) is not ticket
                        or int(getattr(self, "_gripper_dispatch_generation", -1))
                        != ticket.generation
                    ):
                        raise RuntimeError(
                            "gripper HIGH suppressed before SDK dispatch"
                        )
            ok = self._rtde_io.setStandardDigitalOut(pin, level)
            if ok is not True:
                raise RuntimeError(
                    f"gripper DO{pin} {'HIGH' if level else 'LOW'} was not acknowledged"
                )
            if level and (
                ticket.status != "pending"
                or ticket.cancelled.is_set()
                or getattr(self, "_estopped", False)
                or self._grip_stop_event.is_set()
            ):
                # The HIGH SDK call had already been entered and can physically
                # apply late.  We can only attempt LOW after it returns; this is
                # not an independent stop-channel guarantee.
                try:
                    _attempt_safe_low_after_late_high()
                finally:
                    raise RuntimeError(
                        "gripper HIGH returned after dispatch cancellation/stop"
                    )

        def _on_high_dispatched() -> None:
            error: BaseException | None = None
            with self._get_command_lock():
                acknowledged_at_s = time.perf_counter()
                dispatch_now_s = time.monotonic()
                if (
                    ticket.status != "pending"
                    or
                    ticket.cancelled.is_set()
                    or getattr(self, "_estopped", False)
                    or self._grip_stop_event.is_set()
                    or getattr(self, "_gripper_pending_dispatch", None) is not ticket
                    or int(getattr(self, "_gripper_dispatch_generation", -1))
                    != ticket.generation
                    or dispatch_now_s > ticket.dispatch_deadline_s
                ):
                    error = ticket.error or (
                        TimeoutError("gripper HIGH SDK success arrived after dispatch deadline")
                        if dispatch_now_s > ticket.dispatch_deadline_s
                        else RuntimeError("gripper HIGH acknowledgement cancelled or stale")
                    )
                    resolved = self._resolve_gripper_dispatch_locked(
                        ticket,
                        "timeout" if dispatch_now_s > ticket.dispatch_deadline_s else "cancelled",
                        error,
                    )
                    ticket.cancelled.set()
                    if resolved:
                        self._grip_worker_error = error
                        self._latch_emergency_stop_locked(error)
                else:
                    ticket.high_dispatched_at_s = acknowledged_at_s
                    self._resolve_gripper_dispatch_locked(ticket, "acked")
            if error is not None:
                try:
                    _attempt_safe_low_after_late_high()
                finally:
                    raise error

        def _on_completed() -> None:
            with self._get_command_lock():
                if (
                    ticket.cancelled.is_set()
                    or getattr(self, "_estopped", False)
                    or self._grip_stop_event.is_set()
                    or ticket.status != "acked"
                    or ticket.high_dispatched_at_s is None
                ):
                    raise RuntimeError("gripper pulse completion cancelled")
                ticket.completed_at_s = time.perf_counter()

        def _worker() -> None:
            try:
                acquired = self._grip_io_lock.acquire(
                    timeout=float(
                        getattr(
                            self,
                            "_grip_dispatch_ack_timeout_s",
                            GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S,
                        )
                    )
                )
                if not acquired:
                    raise TimeoutError(
                        "gripper pulse could not acquire the IO lock before dispatch deadline"
                    )
                try:
                    if self._grip_stop_event.is_set():
                        raise RuntimeError("gripper pulse cancelled before dispatch")
                    pulse_digital_out(
                        _io_set_with_fault_barrier,
                        pin_fire,
                        pin_other,
                        self._grip_pulse_s,
                        stop_event=self._grip_stop_event,
                        on_high_dispatched=_on_high_dispatched,
                        on_completed=_on_completed,
                    )
                finally:
                    self._grip_io_lock.release()
            except BaseException as exc:
                # Publish the failed generation before any later arm command
                # can pass its in-lock recheck.  Physical stop traffic runs only
                # after releasing this lock.
                publish_global_fault = False
                with self._get_command_lock():
                    if ticket.status == "pending":
                        self._resolve_gripper_dispatch_locked(ticket, "fault", exc)
                        primary_error = exc
                    else:
                        ticket.pulse_error = exc
                        primary_error = ticket.error or exc
                    ticket.cancelled.set()
                    publish_global_fault = (
                        int(getattr(self, "_gripper_dispatch_generation", -1))
                        == ticket.generation
                    )
                    if publish_global_fault:
                        if getattr(self, "_grip_worker_error", None) is None:
                            self._grip_worker_error = primary_error
                        self._latch_emergency_stop_locked(primary_error)
                if publish_global_fault:
                    self.emergency_stop()
            finally:
                ticket.finished.set()
                with self._get_command_lock():
                    self._grip_threads.discard(threading.current_thread())

        worker = threading.Thread(target=_worker, name="grip-pulse", daemon=True)
        ticket.worker = worker
        with self._get_command_lock():
            if getattr(self, "_gripper_pending_dispatch", None) is not ticket:
                raise RuntimeError("[real] gripper dispatch reservation was invalidated")
            self._grip_threads.add(worker)
        try:
            # CPython Thread.start() waits for the child's bootstrap event and
            # therefore must never execute while the global state lock is held.
            worker.start()
        except BaseException as exc:
            with self._get_command_lock():
                self._grip_threads.discard(worker)
                self._resolve_gripper_dispatch_locked(ticket, "fault", exc)
                self._grip_worker_error = exc
                self._latch_emergency_stop_locked(exc)
            raise
        return ticket

    def wait_for_gripper(self, timeout_s: float) -> None:
        """Wait for all pulse workers and surface background I/O failures."""
        deadline = time.perf_counter() + float(timeout_s)
        for worker in tuple(self._grip_threads):
            worker.join(timeout=max(0.0, deadline - time.perf_counter()))
        if any(worker.is_alive() for worker in tuple(self._grip_threads)):
            raise TimeoutError("[real] gripper pulse worker did not finish before timeout")
        if self._grip_worker_error is not None:
            error = self._grip_worker_error
            self._grip_worker_error = None
            raise RuntimeError(f"[real] gripper pulse worker failed: {error}") from error

    def get_health_snapshot(self) -> dict[str, int | bool]:
        """Read robot/safety/runtime state without exposing a command channel."""
        required = ("getRobotMode", "getSafetyMode", "getRuntimeState")
        missing = [name for name in required if not hasattr(self._rtde_r, name)]
        if missing:
            raise RuntimeError(f"[real] RTDE receive health API missing: {missing}")
        return {
            "connected": bool(self._rtde_r.isConnected()),
            "protective_stopped": bool(self._rtde_r.isProtectiveStopped()),
            "emergency_stopped": bool(self._rtde_r.isEmergencyStopped()),
            "safeguard_stopped": bool(
                self._rtde_r.isSafeguardStopped()
                if hasattr(self._rtde_r, "isSafeguardStopped") else False
            ),
            "robot_mode": int(self._rtde_r.getRobotMode()),
            "safety_mode": int(self._rtde_r.getSafetyMode()),
            "runtime_state": int(self._rtde_r.getRuntimeState()),
        }

    def get_controller_version(self) -> str:
        """Return the exact dotted controller version through receive-only RTDE."""
        if not hasattr(self._rtde_r, "getControllerVersion"):
            raise RuntimeError("[real] RTDE controller-version API unavailable")
        raw = self._rtde_r.getControllerVersion()
        if not isinstance(raw, (tuple, list)) or len(raw) < 3:
            raise RuntimeError(f"[real] invalid controller-version payload: {raw!r}")
        return ".".join(str(int(value)) for value in raw)

    def is_ok(self) -> bool:
        """protective stop / 통신 단절 감지 — False면 rollout이 trip한다 (요소5)."""
        if self._estopped:
            self._invalidate_gripper_command_state()
            return False
        if self._grip_worker_error is not None:
            self._invalidate_gripper_command_state()
            print(f"[real][warn] gripper worker failure: {self._grip_worker_error}")
            return False
        try:
            # ⚠️ TODO(확인 필요): isConnected()가 RTDEReceiveInterface에 있는지,
            #   isProtectiveStopped()/isEmergencyStopped()가 receive/control 어느
            #   인터페이스 소속인지 (문서 기억상 receive 쪽이나 확정 필요).
            #   추가로 getSafetyMode()/getRobotMode() 판정 병용 여부도 검토.
            if not self._rtde_r.isConnected():
                self._invalidate_gripper_command_state()
                print("[real][warn] ⚠️ RTDE 수신 연결 끊김")
                return False
            if self._rtde_r.isProtectiveStopped():
                self._invalidate_gripper_command_state()
                print("[real][warn] ⚠️ protective stop 감지")
                return False
            if self._rtde_r.isEmergencyStopped():
                self._invalidate_gripper_command_state()
                print("[real][warn] ⚠️ 하드웨어 e-stop 감지")
                return False
            if hasattr(self._rtde_r, "isSafeguardStopped") and self._rtde_r.isSafeguardStopped():
                self._invalidate_gripper_command_state()
                print("[real][warn] safeguard stop 감지")
                return False
            if self._rtde_c is not None:
                if not self._rtde_c.isConnected():
                    self._invalidate_gripper_command_state()
                    print("[real][warn] RTDE control connection lost")
                    return False
                if not self._rtde_c.isProgramRunning():
                    self._invalidate_gripper_command_state()
                    print("[real][warn] RTDE control program is not running")
                    return False
        except Exception as e:
            self._invalidate_gripper_command_state()
            print(f"[real][warn] ⚠️ 상태 조회 실패 (통신 이상 간주): {e}")
            return False
        return True

    def emergency_stop(self) -> None:
        """Latch atomically, then execute every potentially blocking SDK call unlocked."""
        execute = False
        with self._get_command_lock():
            self._latch_emergency_stop_locked()
            if not getattr(self, "_stop_sequence_active", False):
                self._stop_sequence_active = True
                execute = True
        if not execute:
            return
        try:
            self._execute_emergency_stop_sequence()
        finally:
            with self._get_command_lock():
                self._stop_sequence_active = False

    def _execute_emergency_stop_sequence(self) -> None:
        """요소5: SafetyGuard e-stop hook. API가 허용하는 가장 즉각적인 정지.

        순서(스트리밍 중단 → 감속 정지 → 스크립트 종료), 각 단계는 독립
        try/except — 앞 단계가 실패해도 다음 정지 수단을 반드시 시도한다.
        latch를 '먼저' 걸어 이후 어떤 명령도 나가지 않게 한다.
        ⚠️ 소프트웨어 정지는 하드웨어 e-stop 버튼을 대체하지 않는다.
        """
        # The latch/event were published under _command_lock before entering
        # this method.  No join or RTDE control/IO SDK call below may hold that
        # global state lock; a wedged transport must not block command rejection.
        current_worker = threading.current_thread()
        with self._get_command_lock():
            workers = tuple(self._grip_threads)
        for worker in workers:
            if worker is not current_worker:
                try:
                    worker.join(timeout=0.2)
                except RuntimeError:
                    # A generation can be published immediately before its
                    # Thread.start() call. The stop latch/event still prevents
                    # that not-yet-started worker from dispatching HIGH.
                    pass
        live_workers = [worker for worker in workers if worker.is_alive()]
        if live_workers:
            self._grip_stop_incomplete = True
            print(
                f"[real][warn] {len(live_workers)} gripper worker(s) did not stop within 0.2s"
            )
        if self._rtde_io is not None:
            # Never let a wedged RTDE-IO worker block arm stop commands. Use a
            # bounded lock attempt; the latch/event already prevent new work,
            # and a failure remains visible through _grip_stop_incomplete.
            acquired = self._grip_io_lock.acquire(timeout=0.05)
            if acquired:
                try:
                    for pin in (self._grip_do_open, self._grip_do_close):
                        try:
                            ok = self._rtde_io.setStandardDigitalOut(pin, False)
                            if ok is not True:
                                raise RuntimeError("safe-low was not acknowledged")
                        except Exception as exc:
                            self._grip_stop_incomplete = True
                            print(f"[real][warn] gripper safe-low DO{pin} failed: {exc}")
                finally:
                    self._grip_io_lock.release()
            else:
                self._grip_stop_incomplete = True
                print("[real][warn] gripper IO lock timed out; continuing arm emergency stop")
        print("[real] ★긴급 정지★ servoStop → speedStop → stopScript")
        if self._rtde_c is None:
            print("[real][warn] receive-only connection cannot issue a software stop")
            return
        # ⚠️ TODO(확인 필요): servoStop()/speedStop()이 감속 인자(a)를 받는지,
        #   즉시성 순서가 이 배열이 최선인지 공식 문서로 확인.
        try:
            self._rtde_c.servoStop()
        except Exception as e:
            print(f"[real][warn] servoStop 실패: {e}")
        try:
            self._rtde_c.speedStop()
        except Exception as e:
            print(f"[real][warn] speedStop 실패: {e}")
        for name, stop in (("stopJ", self._rtde_c.stopJ), ("stopL", self._rtde_c.stopL)):
            try:
                stop(1.0)
            except Exception as e:
                print(f"[real][warn] {name} 실패: {e}")
        # ⚠️ TODO(확인 필요): stopScript() 존재 여부 — 컨트롤러의 RTDE 제어
        #   스크립트를 종료하므로 이후 재기동은 재연결(reconnect/reuploadScript)
        #   필요. safety.yaml estop 주석의 열린 질문과 동일 항목.
        try:
            self._rtde_c.stopScript()
        except Exception as e:
            print(f"[real][warn] stopScript 실패: {e}")

    def close(self) -> None:
        """자원 해제 — 스트리밍 정지 후 연결 종료 (e-stop 아님, 정상 종료 경로)."""
        self._invalidate_gripper_command_state()
        self._grip_stop_event.set()
        self._stop_gripper_dispatch_watchdog()
        for worker in tuple(self._grip_threads):
            try:
                worker.join(timeout=0.2)
            except RuntimeError:
                # Thread registered but not yet started by the other caller.
                pass
        live_workers = [worker for worker in tuple(self._grip_threads) if worker.is_alive()]
        if live_workers:
            self._grip_stop_incomplete = True
            print(
                f"[real][warn] {len(live_workers)} gripper worker(s) remained alive at close"
            )
            # Do not disconnect an IO transport beneath a live pulse worker.
            # Escalate through the bounded fail-safe path; it records the
            # incomplete join and continues arm-stop attempts if IO is wedged.
            self.emergency_stop()
        if not self._estopped and self._rtde_c is not None:
            try:
                # 정상 종료 시에도 servo 스트리밍은 명시적으로 끊는다.
                self._rtde_c.servoStop()  # ⚠️ TODO(확인 필요): 위와 동일
            except Exception as e:
                print(f"[real][warn] 종료 중 servoStop 실패: {e}")
        self._disconnect_rtde_resources()
        print("[real] UR5e 연결 종료")
