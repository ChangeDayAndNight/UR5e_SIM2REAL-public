"""F78 수직 파지 제약 — sim과 real이 ★같은 성분★을 막는지 강제한다."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from common.action_mapping import ActionMapper
from common.config_io import build_action_cfg, configs_dir, load_yaml

ROOT = Path(__file__).resolve().parents[1]
_SIM_ACTION = (
    ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
    / "mdp" / "delayed_actions.py"
)


def _cfg(vertical: bool):
    import dataclasses

    base = build_action_cfg(load_yaml(configs_dir() / "common.yaml"))
    return dataclasses.replace(base, vertical_grasp_only=vertical)


def test_tilt_components_are_zeroed_and_yaw_survives():
    """기울임(axis-angle x·y)은 죽고 yaw(z)는 살아야 한다.

    yaw를 함께 막으면 45° 돌아간 큐브를 못 집는다 — 5cm 큐브 대각선 70.7mm에
    그리퍼 개구가 74mm뿐이라 여유가 3mm다.
    """
    mapper = ActionMapper(_cfg(True))
    action = np.array([0.1, -0.2, 0.3, 0.9, -0.8, 0.7, -1.0])
    arm, _ = mapper.map(action, np.zeros(6))
    assert arm[3] == 0.0 and arm[4] == 0.0, "기울임 성분이 남아 있다"
    assert arm[5] != 0.0, "yaw가 함께 죽었다 — 돌아간 큐브를 집을 수 없게 된다"
    # 병진은 손대지 않는다.
    assert arm[:3] == pytest.approx(action[:3] * mapper.cfg.task_pos_scale_m)


def test_disabled_flag_leaves_rotation_untouched():
    mapper = ActionMapper(_cfg(False))
    action = np.array([0.0, 0.0, 0.0, 0.9, -0.8, 0.7, 1.0])
    arm, _ = mapper.map(action, np.zeros(6))
    assert arm[3] != 0.0 and arm[4] != 0.0, "비활성인데 기울임이 죽었다"


def test_sim_and_real_zero_the_same_slice():
    """★sim=real 제1원칙★ — 두 경로가 같은 인덱스를 막는지 소스로 강제한다.

    real은 `rot[0:2] = 0.0`, sim은 `actions[:, 3:5] = 0.0`이다. 회전은 real이
    이미 잘라낸 3-vector(rot)이고 sim은 7채널 원본이라 오프셋 3이 붙는다 —
    ★같은 물리 성분(기울임 x·y)★을 가리켜야 한다. 한쪽만 바뀌면 정책이 학습한
    자세와 실기 거동이 조용히 갈라진다.
    """
    sim_src = _SIM_ACTION.read_text(encoding="utf-8")
    assert "vertical_grasp_only" in sim_src, "sim 액션텀에 수직 파지 분기가 없다"
    assert "actions[:, 3:5] = 0.0" in sim_src, (
        "sim이 막는 슬라이스가 3:5(=기울임)가 아니다"
    )
    real_src = (ROOT / "common" / "action_mapping.py").read_text(encoding="utf-8")
    assert "rot[0:2] = 0.0" in real_src, "real이 막는 슬라이스가 0:2(=기울임)가 아니다"


def test_sim_and_real_flags_are_enabled_together():
    """설정 두 곳이 따로 놀면 안 된다 — 켜려면 양쪽 다 켠다."""
    real_on = bool(load_yaml(configs_dir() / "common.yaml")["action"]
                   .get("vertical_grasp_only", False))
    env_src = (
        ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
        / "binpick_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert 'vertical_grasp_only=bool(_COMMON["action"].get("vertical_grasp_only", False))' in env_src
    assert _cfg(real_on).vertical_grasp_only == real_on


def test_sim_constraint_runs_before_the_z_floor_clamp():
    """순서 계약 — 기울임 제거가 z-floor 클램프보다 ★먼저★ 와야 한다.

    z-floor는 actions[:, 2](병진 z)를 현재 높이 기준으로 조정한다. 그 뒤에 회전을
    건드리는 것은 무해하지만, 반대로 회전 제거가 나중에 오면 클램프가 계산한 델타와
    실제 적용 델타가 달라져 receipt(포스트-클램프 기록)와 어긋난다.
    """
    tree = ast.parse(_SIM_ACTION.read_text(encoding="utf-8"))
    src = _SIM_ACTION.read_text(encoding="utf-8")
    assert src.index("actions[:, 3:5] = 0.0") < src.index('getattr(self.cfg, "tcp_z_floor_m"'), (
        "수직 파지 제약이 z-floor 클램프보다 뒤에 있다"
    )
    assert tree is not None


def test_distillation_target_masks_the_same_channels_as_the_environment():
    """★F79 회귀★ — 환경이 무시하는 채널은 ★증류 타깃에서도★ 0이어야 한다.

    [사고] F78이 환경에서 액션 3·4(기울임)를 0으로 만들자 teacher 정책의 그 채널에
    gradient가 걸리지 않게 됐다. 값이 제약 없이 표류해 실측상 사용 채널의 ★3.1배★
    (기울임 7.19 vs 사용 2.31)까지 커졌고, 증류는 7채널 전부에 MSE를 걸어 student가
    그 노이즈를 회귀하려 했다 — Loss/behavior가 기준선의 3.7배(0.826 vs 0.221).

    환경 마스크와 타깃 마스크가 어긋나면 같은 사고가 재발한다. 소스에서 강제한다.
    """
    policy_src = (
        ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
        / "agents" / "vision_policy.py"
    ).read_text(encoding="utf-8")
    assert "def evaluate" in policy_src, "증류 타깃 오버라이드가 없다"
    assert "teacher_actions[..., 3:5] = 0.0" in policy_src, (
        "증류 타깃이 막는 슬라이스가 3:5(=환경과 동일)가 아니다"
    )
    sim_src = _SIM_ACTION.read_text(encoding="utf-8")
    assert "actions[:, 3:5] = 0.0" in sim_src, "환경 마스크 슬라이스가 바뀌었다"


def test_distillation_mask_follows_the_same_config_flag():
    """마스크는 환경과 ★같은 설정 키★를 읽어야 한다 — 따로 놀면 안 된다."""
    policy_src = (
        ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
        / "agents" / "vision_policy.py"
    ).read_text(encoding="utf-8")
    assert 'get("vertical_grasp_only"' in policy_src, (
        "증류 마스크가 common.yaml action.vertical_grasp_only를 읽지 않는다"
    )

def test_distillation_target_does_not_clip_executed_ik_commands():
    """The simulator has no +/-1 IK clip; only the gripper is binary."""
    policy_src = (
        ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
        / "agents" / "vision_policy.py"
    ).read_text(encoding="utf-8")
    assert "_TARGET_CLAMP" not in policy_src
    assert "distillation_teacher_actions(super().evaluate(obs), _VERTICAL_GRASP_ONLY)" in policy_src


def test_distillation_targets_preserve_mapped_teacher_commands():
    """Compare actual mapped commands, rather than an assumed saturation bound."""
    import importlib.util
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "source" / "ur5e_binpick"))
    spec = importlib.util.find_spec("ur5e_binpick.tasks.binpick.agents.vision_policy")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch

    action_cfg = build_action_cfg(load_yaml(configs_dir() / "common.yaml"))
    mapper = ActionMapper(action_cfg)
    raw = torch.tensor([[2.2, -1.7, 3.0, 8.0, -7.0, 4.5, -12.0]])
    original = raw.clone()
    target = module.distillation_teacher_actions(raw, action_cfg.vertical_grasp_only)
    teacher_arm, teacher_close = mapper.map(raw[0].numpy(), np.zeros(6))
    target_arm, target_close = mapper.map(target[0].numpy(), np.zeros(6))
    np.testing.assert_array_equal(teacher_arm, target_arm)
    assert teacher_close == target_close
    assert torch.equal(raw, original)
    assert target[0, -1] == -1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
