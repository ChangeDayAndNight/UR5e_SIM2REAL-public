"""Behavioral regressions for the September distillation audit (no simulator)."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.action_mapping import ActionMapper
from common.config_io import build_action_cfg, configs_dir, load_yaml
from common.depth_noise import apply_depth_noise
from common.noise_parameters import resample_episode_parameters
from common.observation_delay import delay_policy_observation
from common.rgb_noise import apply_rgb_noise

ROOT = Path(__file__).resolve().parents[1]
BINPICK = ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick"


@pytest.fixture(scope="module")
def vision():
    sys.path.insert(0, str(ROOT / "source/ur5e_binpick"))
    return importlib.import_module("ur5e_binpick.tasks.binpick.agents.vision_policy")


@pytest.mark.parametrize("grip", [-20.0, -0.00001, 0.0, 0.00001, 20.0])
@pytest.mark.parametrize("vertical", [False, True])
def test_teacher_target_has_identical_action_semantics(vision, grip, vertical):
    from dataclasses import replace

    cfg = replace(build_action_cfg(load_yaml(configs_dir() / "common.yaml")),
                  vertical_grasp_only=vertical)
    mapper = ActionMapper(cfg)
    raw = torch.tensor([[2.5, -4.0, 1.1, -8.0, 7.0, 3.5, grip]], dtype=torch.float64)
    target = vision.distillation_teacher_actions(raw, vertical)
    teacher_arm, teacher_close = mapper.map(raw.numpy(), np.zeros(6))
    student_arm, student_close = mapper.map(target.numpy(), np.zeros(6))
    np.testing.assert_array_equal(student_arm, teacher_arm)
    assert teacher_close == student_close
    assert target[0, -1].item() == (-1.0 if grip < 0 else 1.0)


@pytest.mark.parametrize("mode", ["deterministic", "sampled"])
def test_collection_mode_and_runner_logging_are_valid(vision, mode):
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict

    obs = TensorDict({"policy": torch.zeros(2, 50206), "teacher": torch.zeros(2, 42)},
                     batch_size=[2])
    policy = vision.VisionStudentTeacher(
        obs, {"policy": ["policy"], "teacher": ["teacher"]}, 7,
        student_hidden_dims=[16], teacher_hidden_dims=[8], collection_mode=mode,
    )
    with torch.inference_mode():
        rng_before = torch.random.get_rng_state().clone()
        a = policy.act(obs)
        rng_after = torch.random.get_rng_state().clone()
        b = policy.act(obs)
        mean = policy.act_inference(obs)
        # DistillationRunner.log reads distribution-backed action_std.
        assert policy.action_std.shape == (2, 7)
        if mode == "deterministic":
            assert torch.equal(a, b) and torch.equal(a, mean)
            assert torch.equal(rng_before, rng_after)
            assert not policy.action_std.any()
        else:
            assert not torch.equal(a, b)
            assert torch.allclose(policy.action_std, torch.full_like(a, 0.1))
    policy.load_state_dict(policy.state_dict())  # checkpoint schema remains loadable


def test_invalid_collection_mode_fails_before_network_allocation(vision):
    if not hasattr(vision, "VisionStudentTeacher"):
        pytest.skip("rsl_rl not installed")
    with pytest.raises(ValueError, match="collection_mode"):
        vision.VisionStudentTeacher(None, None, 7, collection_mode="typo")


def test_per_environment_depth_dropout():
    depth = torch.ones(2, 8, 8, 1)
    result = apply_depth_noise(depth, 0.0, 0.0, torch.tensor([0.0, 1.0]))
    assert torch.equal(result[0], depth[0])
    assert not result[1].any()


def test_per_environment_rgb_jitter():
    rgb = torch.full((2, 8, 8, 3), 0.4)
    result = apply_rgb_noise(rgb, torch.tensor([0.5, 1.5]), torch.ones(2), torch.zeros(2))
    torch.testing.assert_close(result[0], torch.full_like(result[0], 0.2))
    torch.testing.assert_close(result[1], torch.full_like(result[1], 0.6))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), torch.tensor([0.1, float("nan")])])
def test_nonfinite_noise_parameters_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        apply_depth_noise(torch.ones(2, 4, 4), 0.0, 0.0, bad)
    with pytest.raises(ValueError, match="finite"):
        apply_rgb_noise(torch.ones(2, 4, 4, 3), bad, 1.0, 0.0)


def test_noise_parameter_shape_is_not_silently_broadcast_over_pixels():
    with pytest.raises(ValueError, match="scalar or"):
        apply_depth_noise(torch.ones(2, 4, 4), 0.0, 0.0, torch.zeros(4))
    with pytest.raises(ValueError, match="require"):
        apply_rgb_noise(torch.ones(4, 4, 3), torch.ones(4), 1.0, 0.0)


@pytest.mark.parametrize("ids", [[1], slice(1, 2), torch.tensor([1])])
def test_partial_reset_never_changes_other_environments(ids):
    env = SimpleNamespace(num_envs=3, device="cpu")
    resample_episode_parameters(env, "noise", None, {"p": (0.0, 0.0)})
    resample_episode_parameters(env, "noise", ids, {"p": (1.0, 1.0)})
    torch.testing.assert_close(env.noise["p"], torch.tensor([0.0, 1.0, 0.0]))
    before = env.noise["p"].clone()
    resample_episode_parameters(env, "noise", [], {"p": (0.3, 0.8)})
    assert torch.equal(env.noise["p"], before)


@pytest.mark.parametrize("function,attribute,kwargs,key", [
    ("resample_depth_noise_params", "_binpick_depth_noise",
     dict(sigma_base_m=0.002, sigma_scale_per_m2=0.004, dropout_prob_range=(0.1, 0.1)), "dropout_prob"),
    ("resample_rgb_noise_params", "_binpick_rgb_noise",
     dict(brightness_range=(0.8, 0.8), contrast_range=(1.0, 1.0), pixel_sigma_range=(0.0, 0.0)),
     "brightness_scale"),
])
def test_actual_event_resamplers_honor_env_ids(function, attribute, kwargs, key):
    # Execute the actual function AST, without initializing Isaac/RTX just to
    # test an event's per-env indexing. No replacement implementation in tests.
    tree = ast.parse((BINPICK / "mdp/events_dr.py").read_text(encoding="utf-8"))
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(BINPICK / "mdp/events_dr.py"), "exec"), ns)
    env = SimpleNamespace(num_envs=3, device="cpu")
    ns[function](env, None, **kwargs)
    getattr(env, attribute)[key][[0, 2]] = 0.4321
    ns[function](env, torch.tensor([1]), **kwargs)
    torch.testing.assert_close(getattr(env, attribute)[key][[0, 2]], torch.full((2,), 0.4321))


class _OneStepDelay:
    def __init__(self):
        self.previous = None

    def compute(self, value):
        previous = value if self.previous is None else self.previous
        self.previous = value.clone()
        return previous


@pytest.mark.parametrize("delay_proprio", [False, True])
def test_image_latency_and_latest_proprio_are_separate(delay_proprio):
    first = torch.tensor([[1.0, 2.0, 3.0, 4.0, 50.0, 60.0]])
    second = first + 100
    buf = _OneStepDelay()
    delay_policy_observation(first, buf, 4, delay_proprio=delay_proprio)
    result = delay_policy_observation(second, buf, 4, delay_proprio=delay_proprio)
    assert torch.equal(result[:, :4], first[:, :4])
    assert torch.equal(result[:, 4:], (first if delay_proprio else second)[:, 4:])
    assert torch.equal(second, first + 100)


def test_sim_ik_scale_uses_both_shared_config_scales():
    tree = ast.parse((BINPICK / "binpick_env_cfg.py").read_text(encoding="utf-8"))
    scale = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_IK_SCALE" for t in n.targets))
    value = eval(compile(ast.Expression(scale), "IK scale", "eval"),
                 {"_COMMON": {"action": {"task_pos_scale_m": 0.07, "task_rot_scale_rad": 0.11}}})
    assert value == (0.07, 0.07, 0.07, 0.11, 0.11, 0.11)


def test_vision_resets_render_before_new_observations():
    tree = ast.parse((BINPICK / "binpick_env_cfg.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BinPickVisionEnvCfg")
    assert any(isinstance(n, ast.Assign) and ast.literal_eval(n.value) == 1
               for n in ast.walk(cls)
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute)
                   and t.attr == "num_rerenders_on_reset" for t in n.targets))


def _channel_shares(loss_fn, pred, target):
    """채널별 기여율 — loss_fn이 무엇을 어떻게 가중하는지 실제로 잰다."""
    import torch as _t
    shares = []
    for c in range(target.shape[-1]):
        p = target.clone()
        p[:, c] = pred[:, c]                      # 그 채널에만 오차를 남긴다
        shares.append(float(loss_fn(p, target)))
    total = sum(shares)
    return [s / total for s in shares] if total > 0 else shares


def test_f87_distillation_loss_normalizes_channel_scales(vision):
    """★F87 — 병진 3채널이 MSE의 9.5%뿐이던 문제의 회귀 고정★

    [측정] student가 실제 방문하는 상태에서 teacher 타깃의 MSE 기여율
    (model_24999 · 100배치 × 128env): dx 0.8% / dy 1.9% / dz 6.8% / rz 55.0% /
    grip 35.6% ⇒ 3cm 배달을 결정하는 병진 합이 9.5%. ±1 클램프(F80)를 걸어도
    12.5%로 사실상 같았다 — F80의 '클램프 시 병진 52%'는 무작위 OOD 관측에서
    잰 값이라 실제 분포에서는 성립하지 않았다.

    계약: 같은 크기의 채널별 오차가 ★비슷한 크기로★ 손실에 반영되어야 한다.
    타깃 스케일이 12배 차이나도(rz 0.98 vs dx 0.08) 기여율 차이는 완만해야 한다.
    """
    pytest.importorskip("rsl_rl")
    torch.manual_seed(0)
    # 실측 스케일을 그대로 재현한 타깃: dx 0.08 · dy 0.14 · dz 0.28 · rz 0.98 · grip ±1
    scales = torch.tensor([0.08, 0.14, 0.28, 0.0, 0.0, 0.98, 1.0])
    target = torch.randn(256, 7) * scales
    target[:, 6] = torch.where(target[:, 6] < 0, -1.0, 1.0)
    target[:, 3:5] = 0.0
    pred = target + 0.05                      # 모든 채널에 동일한 절대 오차

    plain = torch.nn.functional.mse_loss
    balanced = vision.channel_balanced_mse      # ★복제본이 아니라 실제 손실 함수★

    plain_share = _channel_shares(plain, pred, target)
    bal_share = _channel_shares(balanced, pred, target)
    plain_translation = sum(plain_share[0:3])
    bal_translation = sum(bal_share[0:3])

    # 무가중 MSE는 채널마다 같은 오차에 같은 기여를 주지만(1/7씩), 실제 학습에서
    # 문제가 되는 것은 ★타깃 크기에 비례해 커지는 오차★다. 정규화는 타깃이 작은
    # 채널의 상대 오차를 키워 준다 — 아래는 그 방향성을 고정한다.
    assert bal_translation > plain_translation, (
        f"정규화가 병진 비중을 키우지 못한다: plain={plain_translation:.3f} "
        f"balanced={bal_translation:.3f}"
    )
    # 타깃 비례 오차(현실적인 회귀 잔차 모델)에서는 격차가 훨씬 크게 벌어진다.
    prop_pred = target * 1.1
    prop_plain = sum(_channel_shares(plain, prop_pred, target)[0:3])
    prop_bal = sum(_channel_shares(balanced, prop_pred, target)[0:3])
    assert prop_plain < 0.20, f"실측 재현 실패: 무가중 병진 기여율 {prop_plain:.3f}"
    assert prop_bal > 0.40, f"정규화 후에도 병진이 묻힌다: {prop_bal:.3f}"


def test_f87_algorithm_class_is_wired_and_switchable():
    """cfg가 기본으로 채널 정규화를 고르고, plain 대조군 경로가 살아 있어야 한다."""
    src = (ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/agents"
           / "rsl_rl_ppo_cfg.py").read_text(encoding="utf-8")
    assert "DistillationChannelBalanced" in src, "cfg가 F87 알고리즘을 고르지 않는다"
    assert 'BINPICK_DISTILL_LOSS' in src, "plain 대조군 스위치가 없다"
    policy_src = (ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/agents"
                  / "vision_policy.py").read_text(encoding="utf-8")
    assert "_m.DistillationChannelBalanced = DistillationChannelBalanced" in policy_src, (
        "알고리즘이 rsl_rl 네임스페이스에 등록되지 않아 러너가 찾지 못한다"
    )
