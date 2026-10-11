# 정책 출력 → 로봇 명령 매핑. sim ActionTerm과 real 배포 루프가
# "같은 수식·같은 상수(configs/common.yaml action 섹션)"를 쓰도록 여기로 단일화.
# action 계약: (action_dim=7,) = [arm 6 | gripper 1].
#
# 현행 = ik_rel: xyz base-frame delta + axis-angle delta. Legacy 관절 모드는 과거 체크용으로 보존.
from __future__ import annotations

import numpy as np

from common.types import ActionCfg


class ActionMapper:
    """정책 출력 → (팔 명령 6, 그리퍼 개폐). sim ActionTerm과 항등인 순수 수식.

    ik_rel(현행): [dxyz × task_pos_scale, axis-angle × task_rot_scale]
    joint_absolute(legacy): q_target = home + action[:6] * action_scale_rad
        · sim JointPositionActionCfg(use_default_offset=True)와 항등. clip 없음
          (sim은 clip_actions=None + 액션텀 무클립) — 관절 한계는 실기 컨트롤러/PhysX가 강제.
        · q_cur는 쓰지 않는다(절대 목표). 시그니처는 배포 루프 계약상 유지.
    joint_delta(legacy): q_target = q_cur + clip(action[:6], -1, 1) * action_scale_rad
    gripper_closed = action[-1] < gripper_close_threshold (strict <)
      ★2026-07-09 부호 관례 정정★: Isaac Lab BinaryJointPositionAction은 action<0=close
      (vendored binary_joint_actions.py L124-129: binary_mask=actions<0 → close_command).
      기존 '>0=close' 계약은 sim과 정반대여서 정책이 그리퍼를 의도적으로 닫을 수 없었다
      (게이트 프로브로 실증: close=+1 명령에 손가락 q→0/열림). sim=real 일치가 제1원칙
      (§4-3)이므로 upstream 관례를 전체 계약으로 채택.

    ⚠️ 안전 clamp는 하지 않는다 — 속도/워크스페이스 clamp는 SafetyGuard의 책임
      (책임 분리: 매핑은 순수 수식, 안전은 safety.py).
    """

    def __init__(self, cfg: ActionCfg):
        if cfg.mode not in ("ik_rel", "joint_absolute", "joint_delta"):
            raise NotImplementedError(f"action mode 미구현: {cfg.mode}")
        if cfg.mode == "joint_absolute" and len(cfg.home_joint_pos) != cfg.num_arm_joints:
            raise ValueError(
                f"joint_absolute home_joint_pos 길이 불일치: 기대 {cfg.num_arm_joints}, "
                f"실제 {len(cfg.home_joint_pos)} (config_io.build_action_cfg가 robot_spec에서 주입)"
            )
        self.cfg = cfg
        self._home = np.asarray(cfg.home_joint_pos, dtype=np.float64)

    def map(self, action: np.ndarray, q_cur: np.ndarray) -> tuple[np.ndarray, bool]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != self.cfg.action_dim:
            raise ValueError(
                f"action 차원 불일치: 기대 {self.cfg.action_dim}, 실제 {action.shape[0]}"
            )
        q_cur = np.asarray(q_cur, dtype=np.float64).reshape(-1)
        if q_cur.shape[0] != self.cfg.num_arm_joints:
            raise ValueError(
                f"q_cur 차원 불일치: 기대 {self.cfg.num_arm_joints}, 실제 {q_cur.shape[0]}"
            )
        arm = action[: self.cfg.num_arm_joints]
        if self.cfg.mode == "ik_rel":
            # Isaac Lab DifferentialIKController.apply_delta_pose와 동일:
            # xyz는 base-frame 변위, 회전 3-vector는 axis-angle 변위.
            rot = arm[3:6].copy()
            # ★★F78 수직 파지 — sim 액션텀(delayed_actions.vertical_grasp_only)과 항등★★
            #   큐브 상면 법선(월드 +z)에 수직으로 내려가야 두 손가락이 마주보는 면을 문다.
            #   기울임(axis-angle x·y)을 0으로 막고 ★yaw만 남긴다★ — 5cm 큐브 대각선 70.7mm
            #   vs 그리퍼 개구 74mm(여유 3mm)라 yaw가 없으면 돌아간 큐브를 못 집는다.
            #   ⚠️ 이 분기는 sim과 ★반드시 같아야★ 한다(§4-3 sim=real 제1원칙).
            #     한쪽만 고치면 정책이 학습한 자세와 실기 거동이 조용히 갈라진다.
            if getattr(self.cfg, "vertical_grasp_only", False):
                rot[0:2] = 0.0
            arm_command = np.concatenate(
                [
                    arm[:3] * self.cfg.task_pos_scale_m,
                    rot * self.cfg.task_rot_scale_rad,
                ]
            )
        elif self.cfg.mode == "joint_absolute":
            # ★절대 목표: home + action×scale (clip 없음 — sim과 항등). q_cur 미사용★
            arm_command = self._home + arm * self.cfg.action_scale_rad
        else:  # joint_delta (legacy)
            arm_command = q_cur + np.clip(arm, -1.0, 1.0) * self.cfg.action_scale_rad
        # ★upstream 관례(action<0=close, binary_joint_actions.py L127)와 항등★
        gripper_closed = bool(action[-1] < self.cfg.gripper_close_threshold)
        return arm_command, gripper_closed
