# 공용 값 타입 — sim/real 어느 쪽 어댑터도 이 타입으로만 대화한다.
# 단위/프레임 계약을 필드 주석에 고정한다 (여기서 어긋나면 전부 어긋남).
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ARM_CLAMP_COMPARISON_ATOL = 1.0e-12
ARM_COMMAND_CHANNELS = 6


@dataclass(frozen=True)
class Intrinsics:
    """카메라 내부 파라미터 (전처리 해상도가 아니라 '원본 프레임' 기준)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class RobotState:
    """로봇 상태 스냅샷 — sim: articulation 읽기 / real: RTDEReceive 읽기."""

    joint_pos: np.ndarray  # (6,) [rad]
    joint_vel: np.ndarray  # (6,) [rad/s]
    tcp_pose: np.ndarray  # (7,) pos[m]+quat(wxyz), base frame, TCP offset 적용 후
    tcp_force: np.ndarray | None  # (6,) wrench [N, Nm] base frame — 미지원 시 None
    # Host-monotonic arrival timestamp recorded only after the adapter has
    # acquired the sample.  It is intentionally not the sensor timestamp.
    timestamp: float  # [s] host time.perf_counter domain
    # Student observation contract: delayed/successfully-dispatched binary
    # command state in [0,1], where 0=open and 1=close.  This is explicitly
    # not physical opening/width feedback.  A real motion adapter starts at
    # None (unknown); faults and every e-stop invalidate it back to None.
    gripper_command_state: float | None = None
    # Timestamp supplied by the physical source.  RealRobot uses the UR
    # controller timestamp from RTDEReceiveInterface.getTimestamp().  Sim and
    # lightweight test doubles may leave these unset, but receive-only HIL
    # evidence requires both fields and rejects host-clock substitutes.
    source_timestamp_s: float | None = None
    source_timestamp_domain: str | None = None
    # Host-monotonic interval in which the selected RTDE package arrived.
    # Both are required for absolute freshness/skew evidence because the UR
    # controller clock has an arbitrary epoch.
    source_acquisition_host_lower_s: float | None = None
    source_acquisition_host_upper_s: float | None = None


@dataclass
class CameraFrame:
    """카메라 프레임 — depth 단위는 반드시 meter (sim/real 공통 계약).

    ★2026-07-17 RGB-D 확장(B안)★: rgb는 (H,W,3) uint8 ★RGB 채널 순서★(BGR 금지).
    depth와 rgb는 ★같은 intrinsics로 정합된 프레임★이어야 한다 — real은
    rs.align(depth→color) 필수(hand_eye v7 캘리브가 color optical 기준),
    sim은 단일 TiledCamera가 두 annotator를 같은 렌더로 출력하므로 자동 정합.
    use_rgb=False 계약(depth-only ablation)에서는 None 허용.
    """

    depth_m: np.ndarray  # (H, W) float32 [m]; 결측=0 또는 NaN 허용(전처리가 처리)
    timestamp: float  # [s] host-monotonic arrival time, after frame conversion
    rgb: np.ndarray | None = None  # (H, W, 3) uint8 RGB — use_rgb 계약일 때 필수
    sequence_id: int | None = None  # monotonic sensor frame number; real requires it
    # RealSense SDK frame metadata, independent of ``timestamp`` above.
    source_timestamp_s: float | None = None
    source_timestamp_domain: str | None = None
    # Host wall-clock reading paired with ``timestamp`` at adapter return.
    # RealSense system/global timestamps share this epoch; hardware-clock
    # frames cannot support absolute age/skew evidence.
    host_wall_timestamp_s: float | None = None


@dataclass(frozen=True)
class PreprocessCfg:
    """요소1: 공유 전처리 파라미터 — configs/common.yaml에서 생성 (config_io.py)."""

    depth_near_m: float
    depth_far_m: float
    out_height: int
    out_width: int
    joint_pos_scale: float
    joint_vel_scale: float
    tcp_pos_scale: float
    action_dim: int  # last_action 슬롯 크기 (= ActionCfg의 action 차원)
    # ★2026-07-14 유효 FOV 크롭(실셀 갠트리 카메라 대응)★ — 정사각 crop 변을
    #   min(h,w)×crop_fraction으로 축소(단, out_size 미만으로는 축소 안 함 — 이미
    #   협FOV로 렌더된 sim 프레임은 그대로 통과). real D435i(848×480)는 이 비율로
    #   크롭 = 유효 vfov×fraction, sim 카메라는 같은 유효 FOV로 직접 렌더(등가).
    crop_fraction: float = 1.0
    # ★2026-07-17 RGB-D 계약(B안 — 사용자 결정)★: True면 flat 레이아웃에 rgb
    #   (out_h*out_w*3, [0,1] float, C-H-W 평탄화) 블록이 depth 다음에 추가된다.
    #   값 변경 = obs_layout/obs_dim/cfg_hash 변경 = 계약 변경(재학습+재export).
    #   sim/real '분기'가 아니라 '계약 파라미터'다 — 양쪽이 같은 값을 읽는다(§4-3).
    use_rgb: bool = False
    # ★2026-07-18 프레임 기하 계약★ — "square"(구: min변 정사각 center-crop) |
    #   "full"(전체 프레임을 out 크기로 비등방 리사이즈 — 픽셀 종횡비 왜곡을
    #   sim/real이 '동일하게' 가지므로 계약 성립). full 도입 근거: 42.5° 정사각에서
    #   홈 자세 큐브 시야 내 비율 18.3%(기하 프로브 실측) → 증류 실패(eval 0.8%,
    #   접근 불능 med 13.9cm). full은 D435i color 가로 69.4°를 그대로 써 48.6%.
    frame_mode: str = "square"
    # Temporal semantics of the 1D gripper_command_state observation.  These
    # fields are part of the exported config hash even though obs_dim is
    # unchanged, so old physical-gripper-position specs are rejected.
    gripper_command_transition_s: float = 1.20
    gripper_command_transition_steps: int = 24


@dataclass(frozen=True)
class ActionCfg:
    """action 계약 — sim ActionTerm 상수와 real ActionMapper가 같은 값을 쓴다.

    현행 ``ik_rel``은 base-frame TCP 위치 delta와 axis-angle delta를 뜻한다.
    sim DifferentialIK와 real servoL 경로가 동일 scale을 쓴다. 관절 모드는 legacy 호환용이다.
    """

    mode: str  # "ik_rel" | "joint_absolute"(legacy) | "joint_delta"(legacy)
    num_arm_joints: int
    action_scale_rad: float  # 절대: home±(action×scale)[rad] / delta(legacy): 스텝당 delta 상한
    gripper_close_threshold: float
    task_pos_scale_m: float = 0.2
    task_rot_scale_rad: float = 0.2
    # 절대 모드 offset = sim default_joint_pos(home). robot_spec 단일 소스에서 config_io가 채운다.
    home_joint_pos: tuple[float, ...] = ()
    # ★F78 수직 파지★ — ik_rel에서 axis-angle 델타의 x·y(기울임)를 0으로 고정하고 yaw만 남긴다.
    #   sim 등가물: delayed_actions.DelayedDifferentialInverseKinematicsActionCfg.vertical_grasp_only
    #   두 값은 ★반드시 함께★ 켜고 끈다(§4-3 sim=real 제1원칙).
    vertical_grasp_only: bool = False

    @property
    def action_dim(self) -> int:
        return self.num_arm_joints + 1  # +1 = 그리퍼 채널


@dataclass(frozen=True)
class SafetyCfg:
    """요소5: 안전 한계 — configs/safety.yaml에서 생성."""

    workspace_min: tuple[float, float, float]  # TCP AABB [m]
    workspace_max: tuple[float, float, float]
    joint_vel_max_rad_s: float
    tcp_force_max_n: float
    tcp_linear_vel_max_m_s: float = 0.25
    tcp_angular_vel_max_rad_s: float = 0.5
    # ★★2026-08-12(감사 B1) TCP 하강 바닥 [m]★★ — sim 액션 래퍼의 tcp_z_floor
    #   (binpick_env_cfg: TABLE_HEIGHT+0.030 = −0.0075)와 ★같은 의미의 실기 등가물★.
    #   [필요성] place 타겟 z(−0.0125)가 패드-벨트 접촉점보다 아래라, 정책은 sim에서
    #     '클램프가 잡아 준다'를 전제로 하강을 명령한다. 실기에 이 클램프가 없으면
    #     매 정상 place 시도가 벨트 압착 경로를 밟고, AABB 하한(−0.0375)은 벨트 상면이라
    #     위치 기반 안전층이 이 시나리오를 못 잡는다(감사 실측). trip이 아니라 ★clamp★인
    #     이유도 sim과 동일해야 하기 때문(§8-1: 리허설=실기 같은 의미론).
    #   None이면 비활성(구 동작). 값은 configs/safety.yaml workspace.tcp_z_floor_m 단일 소스 —
    #   sim 값과의 동기는 tests/test_safety_zfloor.py 드리프트 가드가 강제한다.
    tcp_z_floor_m: float | None = None


@dataclass
class SafetyVerdict:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class AppliedArmCommandReceipt:
    """One successfully applied, full-pipeline arm-command lineage.

    ``delayed_mapped_raw`` is the original mapped command selected by the exact
    same delay operation as ``post_delay_pre_floor``.  The latter is the
    SafetyGuard output after that delay and immediately before the action
    term's final floor/clip processing; ``applied`` is the command actually
    accepted after that processing.  Keeping all three stages makes an outer
    SafetyGuard clamp, an inner adapter clamp, and a canceling/reversing pair
    independently auditable.  Adapters without an internal delay or final
    clamp still bind the original mapped input and SafetyGuard output as two
    distinct stages.  ``sequence`` is adapter-local and strictly increases at
    every successful control step so stale evidence cannot be replayed.
    """

    sequence: int
    delayed_mapped_raw: tuple[float, ...]
    post_delay_pre_floor: tuple[float, ...]
    applied: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) \
                or self.sequence <= 0:
            raise ValueError("applied arm command receipt sequence must be a positive integer")
        for name in ("delayed_mapped_raw", "post_delay_pre_floor", "applied"):
            command = np.asarray(getattr(self, name), dtype=np.float64)
            if command.shape != (ARM_COMMAND_CHANNELS,) or not np.isfinite(command).all():
                raise ValueError(f"{name} must be a finite 6D command")
            object.__setattr__(self, name, tuple(float(value) for value in command))

    @classmethod
    def from_commands(
        cls,
        *,
        sequence: int,
        delayed_mapped_raw,
        post_delay_pre_floor,
        applied,
    ) -> "AppliedArmCommandReceipt":
        return cls(
            sequence=sequence,
            delayed_mapped_raw=delayed_mapped_raw,
            post_delay_pre_floor=post_delay_pre_floor,
            applied=applied,
        )


@dataclass
class RolloutSafetyStats:
    """Arm clamp evidence for commands that were actually issued.

    ``denominator`` deliberately excludes receive-only/dry-run iterations and
    commands that raised before the arm command completed.  Each per-channel
    count is the union of (a) original mapped command versus SafetyGuard output
    and (b) that guarded command versus the exact applied command.  All three
    values belong to one delay-selected lineage, so neither an action delay nor
    two clamps that numerically cancel can hide an intervention.
    """

    denominator: int = 0
    arm_clamp_counts: tuple[int, ...] = (0, 0, 0, 0, 0, 0)

    def __post_init__(self) -> None:
        if isinstance(self.denominator, bool) or not isinstance(self.denominator, int) \
                or self.denominator < 0:
            raise ValueError("rollout safety denominator must be a non-negative integer")
        if len(self.arm_clamp_counts) != ARM_COMMAND_CHANNELS:
            raise ValueError("rollout safety requires exactly six arm clamp counts")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            or not 0 <= value <= self.denominator
            for value in self.arm_clamp_counts
        ):
            raise ValueError("rollout safety clamp counts must be integers in [0, denominator]")

    @property
    def arm_clamp_fractions(self) -> tuple[float, ...]:
        if self.denominator <= 0:
            return (0.0,) * len(self.arm_clamp_counts)
        return tuple(count / self.denominator for count in self.arm_clamp_counts)

    def as_manifest(self, *, maximum_fraction_exclusive: float = 0.01) -> dict:
        fractions = self.arm_clamp_fractions
        return {
            "arm_channel_count": len(self.arm_clamp_counts),
            "denominator": int(self.denominator),
            "clamp_counts": [int(value) for value in self.arm_clamp_counts],
            "clamp_fractions": [float(value) for value in fractions],
            "maximum_fraction_exclusive": float(maximum_fraction_exclusive),
            "all_channels_below_limit": bool(
                self.denominator > 0
                and all(value < maximum_fraction_exclusive for value in fractions)
            ),
        }


@dataclass
class RolloutResult:
    """run_rollout 1회 실행 요약 (파일 산출과 별개의 반환값)."""

    steps: int
    tripped: bool
    trip_reason: str
    overruns: int  # 제어 주기 오버런 횟수 (real 타이밍 품질 지표)
    terminated: bool = False
    success: bool = False
    rollout_safety: RolloutSafetyStats = field(default_factory=RolloutSafetyStats)
