"""★sim 학습 액션 스킴 단일 소스★ — 정책 출력의 '의미'.

목적: `obs_spec.json`의 `action_scheme` 필드에 넣을 값을 ★한 곳★에서 정한다.
      이 값은 "정책이 내놓는 7차원 벡터가 무엇을 뜻하는가"이며, 차원이 같아도 의미가
      다르면 실기에서 ★조용히 오역★된다(TCP pose delta를 관절 절대목표로 해석 등).

⚠️ **이 모듈은 isaaclab을 import하지 않는다** — `Phase2/Export_Policy.py`가 순수 torch
   스크립트(AppLauncher 없음)라서, 학습 확장과 export 스크립트가 ★같은 상수★를 보려면
   isaaclab-free여야 한다. 여기에 isaaclab import를 추가하지 말 것.

배경(감사 P7-1, 2026-08-06): 이 상수가 없던 시절 `Export_Policy`는 `make_obs_spec`에
   `action_scheme`을 넘기지 않았고, 기본값이 ★당시 배포측 계약값★(`configs/common.yaml`의
   `joint_absolute`)으로 폴백했다. 그 결과 기록값과 배포측 기댓값이 같아져 ★검증이 통과★했다 —
   sim은 `ik_rel`로 학습했는데도. 막으려던 사고를 기본값이 되레 허용한 과거 결함이며,
   현행 sim/real 계약은 둘 다 `ik_rel`이다.

실행: 직접 실행 없음 — `binpick_env_cfg.py`(sim)와 `Phase2/Export_Policy.py`(export)가 import.
산출: `SIM_ACTION_SCHEME`
"""

from __future__ import annotations

# ★현행 sim 학습 액션★ = task-space 상대 IK.
#   binpick_env_cfg.ActionsCfg: arm_action = DifferentialInverseKinematicsActionCfg(
#       command_type="pose", use_relative_mode=True, ...)  → 6차원 TCP pose delta
#     + gripper_action = BinaryJointPositionActionCfg                → 1차원 개폐
#   ⇒ 총 7차원. ★관절 목표가 아니다.★
#   `tests/test_action_scheme_contract.py`가 이 상수와 실제 ActionsCfg 배선의 일치를 강제한다.
SIM_ACTION_SCHEME: str = "ik_rel"

# 배포측도 `configs/common.yaml action.mode=ik_rel`이며 ActionMapper→servoL 경로가
# 같은 위치/axis-angle 스케일을 사용한다. `validate_obs_spec`이 이후 드리프트를 거부한다.
