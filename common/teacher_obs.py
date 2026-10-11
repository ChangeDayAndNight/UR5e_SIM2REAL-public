# ★모듈형 배포 — 인식 모듈 + Teacher v2 34D 관측 조립★ [배포 PC][제어 루프 안]
#
# 구조
#   D435i(eye-in-hand) depth ──▶ common.cube_depth_locator(작업면 위 높이 띠의 큐브 덩어리) ──▶ 큐브 중심(base)
#   RTDE 관절/TCP ────────────────────────────────────────────────────────────────────────────▶ 34D teacher 관측
#   그리퍼 명령 상태 ──▶ held latch(폐합 전이 시점 추정치-TCP 근접) ───────────────────────────▶
#
# Teacher v2 관측(common.teacher_contract, 34D)과 실기 신호의 대응 — ★근사 항이 없다★
#   joint_pos(6)              RobotState.joint_pos [rad]                                    ← 실측
#   joint_vel(6)              RobotState.joint_vel [rad/s]                                  ← 실측
#   gripper_command_state(1)  지연·전이 적분된 명령 상태(common/gripper_command)              ← 계약
#   tcp_pose(7)               RobotState.tcp_pose (base, pos+quat wxyz, TCP offset 적용)     ← 실측
#   target_pos(3)             사용자 지정                                                     ← 입력
#   object_pos(3)             인식 모듈(큐브 중심, base) / held_offset / last_known 합성       ← 추정
#   held(1)                   폐합 전이 시점에 추정치가 TCP 반경 안이면 latch                  ← 규칙
#   last_action(7)            직전 정책 출력의 ★실행 가능 정규화★(clamp ±1, 기울임 3·4=0)     ← 계약
#
# ★sim과 같은 규칙★: 학습측 mdp/perception_model.py가 이 파일과 같은 합성·latch 규칙으로 관측을
#   만든다(상수는 common/teacher_contract 단일 소스). 그래서 정책은 실기 신호의 오차·결측을
#   학습에서 이미 경험했다. 구 v1(42D)이 접촉력·손가락 위치를 상수로 근사하던 방식은 폐기했다.
#
# ★정직한 한계★
#   · held는 파지 성공의 증거가 아니라 '닫을 때 큐브가 손 근처였다'는 기록이다. 헛집기는 큐브가
#     다시 보이는 순간(멀리서 검출) 해제되어 정책이 재시도할 수 있다.
#   · 추정치가 갑자기 10 cm 넘게 튀면(오검출) 2프레임 연속 확인 전에는 받지 않는다.
#   · 큐브가 안 보이고 쥐고 있지도 않은 채 max_missed_frames_before_hold를 넘기면 hold를 요청해
#     팔을 세운다(run_rollout의 hold 계약). 학습은 last_known 유지로 이 구간을 경험하므로 기본값을
#     2 s(40프레임)로 둔다 — 그리퍼 아래 큐브는 실기 FOV 밖이라(hand-eye 기하) 짧은 결측은 정상이다.
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np
import torch

from common.calibration import pose7_to_mat
from common.cube_depth_locator import DepthLocatorCfg, locate_cube_from_depth
from common.teacher_contract import (
    GRIPPER_CLOSED_THRESHOLD,
    HELD_OFFSET_FROM_TCP_M,
    HELD_RADIUS_M,
    TEACHER_OBS_DIM,
    teacher_obs_slice,
)
from common.types import CameraFrame, Intrinsics, RobotState

DEFAULT_OBJECT_SIZE_M = 0.05
DEFAULT_TABLE_HEIGHT_M = -0.0375


@dataclass(frozen=True)
class TeacherObsCfg:
    """Teacher 관측 조립 상수 — 번들 manifest에 기록되어 실기와 sim 리허설이 같은 값을 쓴다."""

    object_size_m: float = DEFAULT_OBJECT_SIZE_M
    table_height_m: float = DEFAULT_TABLE_HEIGHT_M
    # depth 추정기 — 작업면 위 높이 띠, 큐브 형상 조건(common.cube_depth_locator)
    z_band_m: tuple[float, float] = (0.025, 0.150)
    footprint_min_frac: float = 0.35
    footprint_max_frac: float = 1.60
    # 쥐지 않은 큐브는 작업면 위에 있다 → z를 table_height + size/2로 고정(깊이 잡음 제거)
    snap_z_on_table: bool = True
    # "쥐었다" latch: 폐합 전이 시점에 추정치가 TCP에서 이 반경 안 (sim/real 단일 소스 상수)
    held_radius_m: float = HELD_RADIUS_M
    # 쥐고 있는데 검출이 끊기면(손가락 가림·FOV 밖) 큐브 중심 = TCP + 이 오프셋(base)
    held_offset_from_tcp_m: tuple[float, float, float] = HELD_OFFSET_FROM_TCP_M
    # 검출 실패가 이 스텝 수를 넘으면 hold(팔 정지) 요청 — ★F93★ 4 s(40→80). 학습은 last_known 유지를
    #   경험했고 hold는 sim에 없는 실기 전용 안전망이다. 현 브라켓(hand_eye v7)에서 그리퍼 아래 큐브는
    #   세로 FOV 밖이라 최종 하강(≥20스텝)+폐합 전이(12스텝)는 ★항상★ 블라인드다(리뷰 T-01 기하 계산
    #   ≥32프레임) — 40이면 정상 파지 중에 팔이 영구 정지할 수 있었다. 그래서 (a) 임계 80, (b) 마지막
    #   추정치 바로 위(held_radius+object_size 안)나 그리퍼가 완전 개방이 아닌 동안은 결측을 세지 않는다
    #   (예상된 블라인드 콘). hold는 '큐브를 잃고 엉뚱한 곳에 있다'만 잡는다.
    max_missed_frames_before_hold: int = 80
    # 급점프 억제 — 마지막 채택 추정치에서 이 거리 넘게 튀면 연속 확인 후에만 채택
    max_jump_m: float = 0.10
    jump_confirm_frames: int = 2
    vertical_grasp_only: bool = True
    gripper_closed_threshold: float = GRIPPER_CLOSED_THRESHOLD   # gripper_command_state 폐합 판정 문턱


@dataclass
class TeacherObsDebug:
    """직전 스텝의 조립 근거 — 로그·리허설 통계용."""

    detected: bool = False
    reason: str = ""
    held: bool = False
    object_pos: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    object_source: str = "none"          # "detected" | "held_offset" | "last_known" | "none"
    raw_center: np.ndarray | None = None
    tcp_pos: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))


def executable_last_action(last_action: np.ndarray, *, vertical_grasp_only: bool) -> np.ndarray:
    """observations.executable_last_action의 numpy 미러 — clamp ±1, 기울임 채널 3·4 = 0."""
    a = np.clip(np.asarray(last_action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    if vertical_grasp_only and a.shape[0] > 5:
        a[3:5] = 0.0
    return a


def T_base_cam_from_tcp(tcp_pose7: np.ndarray, T_flange_tcp: np.ndarray, T_flange_cam: np.ndarray) -> np.ndarray:
    """base←cam = base←tcp · tcp←flange · flange←cam. RobotState.tcp_pose는 TCP offset이 적용된 값."""
    T_base_tcp = pose7_to_mat(np.asarray(tcp_pose7, dtype=np.float64))
    T_tcp_flange = np.linalg.inv(np.asarray(T_flange_tcp, dtype=np.float64))
    return T_base_tcp @ T_tcp_flange @ np.asarray(T_flange_cam, dtype=np.float64)


def make_depth_locator_cfg(cfg: TeacherObsCfg, workspace_min, workspace_max) -> DepthLocatorCfg:
    return DepthLocatorCfg(object_size_m=cfg.object_size_m, table_height_m=cfg.table_height_m,
                           z_band_m=tuple(cfg.z_band_m), workspace_min=tuple(map(float, workspace_min)),
                           workspace_max=tuple(map(float, workspace_max)),
                           footprint_min_frac=cfg.footprint_min_frac, footprint_max_frac=cfg.footprint_max_frac)


def table_center_z(cfg: TeacherObsCfg) -> float:
    return cfg.table_height_m + cfg.object_size_m / 2.0


class TeacherObsBuilder:
    """run_rollout(obs_builder=...) 훅 — (frame, state, target_pos, last_action) → [1, 34] 텐서.

    `hold_requested`는 run_rollout이 매 스텝 읽는다(큐브 미검출·미보유 = 팔 정지·그리퍼 유지).
    `locate(depth_m, intrinsics, T_base_cam)`는 테스트/리허설에서 교체 가능한 인식 함수
    (기본: common.cube_depth_locator.locate_cube_from_depth; .present/.reason/.center_base 계약).
    """

    def __init__(self, intrinsics: Intrinsics, T_flange_cam: np.ndarray, T_flange_tcp: np.ndarray,
                 cfg: TeacherObsCfg, dt_s: float, *, workspace_min=(-1.0, -1.0, -0.2),
                 workspace_max=(1.0, 1.0, 0.6), locate: Callable | None = None, device: str = "cpu"):
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        self.intrinsics = intrinsics
        self.T_flange_cam = np.asarray(T_flange_cam, dtype=np.float64)
        self.T_flange_tcp = np.asarray(T_flange_tcp, dtype=np.float64)
        if self.T_flange_cam.shape != (4, 4) or self.T_flange_tcp.shape != (4, 4):
            raise ValueError("calibration transforms must be 4x4")
        self.cfg = cfg
        self.dt_s = float(dt_s)
        self.device = device
        self.locator_cfg = make_depth_locator_cfg(cfg, workspace_min, workspace_max)
        self._locate = locate if locate is not None else partial(locate_cube_from_depth, cfg=self.locator_cfg)
        self.hold_requested = False
        self.debug = TeacherObsDebug()
        self.steps = 0
        self.detected_steps = 0
        self.held_steps = 0
        self.hold_steps = 0
        self.jump_rejects = 0
        self.max_blind_streak = 0      # ★F93★ 연속 (미검출 ∧ ¬held) 최대 길이 — hold 임계 근거용 증거
        self.reset()

    def reset(self) -> None:
        self._held = False
        self._closed_prev = False
        self._last_object: np.ndarray | None = None
        self._missed = 0
        self._blind_streak = 0
        self._pending: tuple[np.ndarray, int] | None = None
        self.hold_requested = False
        self.debug = TeacherObsDebug()

    # ── 인식 ──────────────────────────────────────────────────────────────────
    def perceive(self, frame: CameraFrame, tcp_pose7: np.ndarray):
        """큐브 중심(base) 또는 None — 급점프는 연속 확인 전까지 거부한다."""
        T_base_cam = T_base_cam_from_tcp(tcp_pose7, self.T_flange_tcp, self.T_flange_cam)
        est = self._locate(np.asarray(frame.depth_m), self.intrinsics, T_base_cam)
        if not getattr(est, "present", False) or getattr(est, "center_base", None) is None:
            self._pending = None
            return None, getattr(est, "reason", "위치 없음") or "위치 없음"
        center = np.asarray(est.center_base, dtype=np.float64)
        if self._last_object is not None and np.linalg.norm(center - self._last_object) > self.cfg.max_jump_m:
            if self._pending is not None and np.linalg.norm(center - self._pending[0]) <= self.cfg.max_jump_m:
                self._pending = (center, self._pending[1] + 1)
            else:
                self._pending = (center, 1)
            if self._pending[1] < self.cfg.jump_confirm_frames:
                self.jump_rejects += 1
                return None, f"급점프 {np.linalg.norm(center - self._last_object):.2f} m — 확인 대기"
        self._pending = None
        return center, ""

    # ── 조립 ──────────────────────────────────────────────────────────────────
    def __call__(self, frame: CameraFrame, state: RobotState, target_pos: np.ndarray,
                 last_action: np.ndarray) -> torch.Tensor:
        cfg = self.cfg
        tcp_pose = np.asarray(state.tcp_pose, dtype=np.float64).reshape(7)
        tcp_pos = tcp_pose[:3]
        if state.gripper_command_state is None:
            raise ValueError("gripper command state unknown — run_rollout이 먼저 걸러야 한다")
        gripper_state = float(state.gripper_command_state)
        closed_cmd = gripper_state >= cfg.gripper_closed_threshold

        center, reason = self.perceive(frame, tcp_pose)
        detected = center is not None
        held_offset = np.asarray(cfg.held_offset_from_tcp_m, dtype=np.float64)
        fallback = np.array([tcp_pos[0], tcp_pos[1], table_center_z(cfg)])
        est_prev = self._last_object if self._last_object is not None else fallback

        # ★held latch — sim(perception_model._update)과 같은 규칙★
        #   폐합 전이 시점에 마지막 추정치가 TCP 반경 안이면 참. 닫힌 채 검출되면 그 근접으로 갱신
        #   (멀리서 보이면 헛집기 → 해제). 열리면 해제.
        held = self._held
        if closed_cmd and not self._closed_prev:
            held = float(np.linalg.norm(est_prev - tcp_pos)) <= cfg.held_radius_m
            self._missed = 0                       # ★F93★ 폐합 전이 = 블라인드가 예상되는 국면
        if detected and closed_cmd:
            held = float(np.linalg.norm(center - tcp_pos)) <= cfg.held_radius_m
        held = held and closed_cmd
        self._held = held
        self._closed_prev = closed_cmd

        if detected:
            object_pos = center.copy()
            if cfg.snap_z_on_table and not held:
                object_pos[2] = table_center_z(cfg)
            source = "detected"
            self._missed = 0
        elif held:
            object_pos = tcp_pos + held_offset
            source = "held_offset"
            self._missed = 0
        else:
            # ★F93(리뷰 P2/T-01/D7)★ 예상된 블라인드는 '분실'이 아니다: 마지막 추정치 바로 위에 있거나
            #   (최종 하강·폐합·개방 국면 = FOV 밖) 그리퍼가 완전 개방이 아닌 동안은 세지 않는다.
            in_blind_cone = (
                self._last_object is not None
                and float(np.linalg.norm(self._last_object[:2] - tcp_pos[:2]))
                <= cfg.held_radius_m + cfg.object_size_m
            )
            if not (in_blind_cone or gripper_state > 0.0):
                self._missed += 1
            if self._last_object is not None:
                object_pos = self._last_object.copy()
                source = "last_known"
            else:
                object_pos = fallback
                source = "none"
        if (not detected) and (not held):
            self._blind_streak += 1
            self.max_blind_streak = max(self.max_blind_streak, self._blind_streak)
        else:
            self._blind_streak = 0
        self._last_object = object_pos.copy()
        self.hold_requested = (not detected) and (not held) and (self._missed > cfg.max_missed_frames_before_hold)

        obs = np.zeros(TEACHER_OBS_DIM, dtype=np.float32)
        obs[teacher_obs_slice("joint_pos")] = np.asarray(state.joint_pos, dtype=np.float64).reshape(6)
        obs[teacher_obs_slice("joint_vel")] = np.asarray(state.joint_vel, dtype=np.float64).reshape(6)
        obs[teacher_obs_slice("gripper_command_state")] = gripper_state
        obs[teacher_obs_slice("tcp_pose")] = tcp_pose
        obs[teacher_obs_slice("target_pos")] = np.asarray(target_pos, dtype=np.float64).reshape(3)
        obs[teacher_obs_slice("object_pos")] = object_pos
        obs[teacher_obs_slice("held")] = float(held)
        obs[teacher_obs_slice("last_action")] = executable_last_action(
            last_action, vertical_grasp_only=cfg.vertical_grasp_only)
        if not np.isfinite(obs).all():
            raise ValueError("non-finite teacher observation")

        self.steps += 1
        self.detected_steps += int(detected)
        self.held_steps += int(held)
        self.hold_steps += int(self.hold_requested)
        self.debug = TeacherObsDebug(detected=detected, reason=reason, held=held, object_pos=object_pos.copy(),
                                     object_source=source, raw_center=center, tcp_pos=tcp_pos.copy())
        return torch.from_numpy(obs).unsqueeze(0).to(self.device)

    def stats(self) -> dict:
        n = max(1, self.steps)
        return {"steps": self.steps, "detection_rate": self.detected_steps / n,
                "held_fraction": self.held_steps / n, "hold_fraction": self.hold_steps / n,
                "jump_rejects": self.jump_rejects, "max_blind_streak": self.max_blind_streak}
