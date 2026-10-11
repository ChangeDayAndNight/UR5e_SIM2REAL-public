"""배포 v2 — 시야각 레터박스와 이력 student 러너 회귀테스트 (Isaac·GPU 불필요)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from common.fov_match import FovMatchedCamera, FovSpec, canvas_shape, letterbox_frame
from common.types import CameraFrame, Intrinsics

SIM_FOV = FovSpec(69.4, 69.4)      # Isaac은 등방 렌더 — 가로 화각이 세로에도 적용된다
REAL_FOV = FovSpec(69.4, 42.5)     # D435i color 848x480


def make_frame(h=480, w=848, rgb=True):
    depth = np.full((h, w), 0.5, dtype=np.float32)
    colour = np.full((h, w, 3), 200, dtype=np.uint8) if rgb else None
    return CameraFrame(depth_m=depth, timestamp=1.0, rgb=colour, sequence_id=7)


def test_canvas_matches_the_tangent_ratio_and_only_pads_the_short_axis():
    canvas_h, canvas_w, top, left = canvas_shape(480, 848, REAL_FOV, SIM_FOV)
    scale = math.tan(math.radians(42.5) / 2) / math.tan(math.radians(69.4) / 2)
    assert canvas_w == 848 and left == 0            # 가로 화각이 같으므로 손대지 않는다
    assert canvas_h == round(480 / scale) and top == (canvas_h - 480) // 2
    # 캔버스를 112²로 눌렀을 때 내용이 차지하는 세로 비율 = tan 비
    assert 112 * 480 / canvas_h == pytest.approx(112 * scale, abs=0.6)


def test_identical_fov_is_an_identity_transform():
    frame = make_frame(112, 112)
    assert letterbox_frame(frame, SIM_FOV, SIM_FOV) is frame
    assert canvas_shape(112, 112, SIM_FOV, SIM_FOV) == (112, 112, 0, 0)


def test_letterbox_centres_content_and_pads_with_missing_data():
    frame = make_frame()
    out = letterbox_frame(frame, REAL_FOV, SIM_FOV)
    canvas_h, _, top, _ = canvas_shape(480, 848, REAL_FOV, SIM_FOV)
    assert out.depth_m.shape == (canvas_h, 848) and out.rgb.shape == (canvas_h, 848, 3)
    assert np.all(out.depth_m[top:top + 480] == 0.5) and np.all(out.rgb[top:top + 480] == 200)
    assert np.all(out.depth_m[:top] == 0.0) and np.all(out.rgb[:top] == 0)   # depth 0 = 결측 → 전처리가 far
    assert np.all(out.depth_m[top + 480:] == 0.0)
    assert out.timestamp == frame.timestamp and out.sequence_id == frame.sequence_id
    assert out.depth_m.dtype == frame.depth_m.dtype and out.rgb.dtype == frame.rgb.dtype


def test_source_wider_than_training_is_refused_rather_than_cropped():
    with pytest.raises(ValueError, match="exceeds the training FOV"):
        canvas_shape(480, 848, FovSpec(80.0, 80.0), SIM_FOV)


def test_fov_spec_from_intrinsics_round_trips():
    f = (112 / 2) / math.tan(math.radians(69.4) / 2)
    spec = FovSpec.from_intrinsics(Intrinsics(fx=f, fy=f, cx=56, cy=56, width=112, height=112))
    assert spec.horizontal_deg == pytest.approx(69.4, abs=1e-6)
    assert spec.vertical_deg == pytest.approx(69.4, abs=1e-6)
    with pytest.raises(ValueError):
        FovSpec(0.0, 42.5)


def test_camera_wrapper_letterboxes_every_frame_and_keeps_raw_intrinsics():
    class Fake:
        def __init__(self):
            self.started = self.stopped = 0

        def start(self): self.started += 1
        def stop(self): self.stopped += 1
        def get_frame(self): return make_frame()
        def get_intrinsics(self):
            fx = (848 / 2) / math.tan(math.radians(69.4) / 2)
            fy = (480 / 2) / math.tan(math.radians(42.5) / 2)
            return Intrinsics(fx=fx, fy=fy, cx=424, cy=240, width=848, height=480)

    inner = Fake()
    cam = FovMatchedCamera(inner, SIM_FOV)
    cam.start()
    out = cam.get_frame()
    canvas_h, _, _, _ = canvas_shape(480, 848, REAL_FOV, SIM_FOV)
    assert out.depth_m.shape[0] == canvas_h
    assert cam.get_intrinsics().height == 480          # 원본 intrinsics를 그대로 노출
    assert cam.describe()["frames"] == 1 and inner.started == 1
    cam.stop(); assert inner.stopped == 1


# ── 이력 러너 ────────────────────────────────────────────────────────────────
class _StatelessHistoryStub(torch.nn.Module):
    """action = sum(history) + obs 평균 — 이력이 실제로 이어지는지 확인하기 쉬운 형태."""

    def forward(self, obs, history):
        feature = obs.mean(dim=1, keepdim=True).expand(-1, history.shape[2])
        window = torch.cat((history, feature.unsqueeze(1)), dim=1)
        action = window.sum(dim=(1, 2), keepdim=False).unsqueeze(1).expand(-1, 7)
        return action, window[:, 1:]


def make_runner(tmp_path, obs_dim, action_dim, pre_cfg, action_cfg, steps=4, feature_dim=3):
    from common.policy_io import make_obs_spec, save_obs_spec
    from common.student_policy_io import StudentHistoryPolicyRunner
    scripted = torch.jit.script(_StatelessHistoryStub())
    policy = tmp_path / "policy.pt"
    scripted.save(str(policy))
    # action_scheme은 배포측 계약값과 대조된다(common/policy_io.validate_obs_spec) —
    # 테스트는 sim 확장 없이 그 계약값을 그대로 쓴다.
    spec = make_obs_spec(pre_cfg, action_cfg, action_scheme=action_cfg.mode)
    save_obs_spec(tmp_path / "obs_spec.json", spec)
    return StudentHistoryPolicyRunner(policy, tmp_path / "obs_spec.json", pre_cfg, action_cfg,
                                      history_steps=steps, feature_dim=feature_dim)


@pytest.fixture(scope="module")
def cfgs():
    from common.config_io import build_action_cfg, build_preprocess_cfg, configs_dir, load_yaml
    from common.obs_preprocess import obs_dim
    common = load_yaml(configs_dir() / "common.yaml")
    pre_cfg, action_cfg = build_preprocess_cfg(common), build_action_cfg(common)
    return pre_cfg, action_cfg, int(obs_dim(pre_cfg))


def test_history_runner_accumulates_and_reset_clears_the_buffer(tmp_path, cfgs):
    pre_cfg, action_cfg, obs_dim_value = cfgs
    runner = make_runner(tmp_path, obs_dim_value, action_cfg.action_dim, pre_cfg, action_cfg)
    obs = torch.full((1, obs_dim_value), 1.0)
    first = runner.act(obs)[0, 0]
    second = runner.act(obs)[0, 0]
    assert second > first                      # 이력이 쌓이면 합이 커진다
    runner.reset()
    assert runner.steps == 0
    assert runner.act(obs)[0, 0] == pytest.approx(first)   # 초기화하면 첫 스텝과 같다


def test_history_runner_rejects_wrong_shapes(tmp_path, cfgs):
    pre_cfg, action_cfg, obs_dim_value = cfgs
    runner = make_runner(tmp_path, obs_dim_value, action_cfg.action_dim, pre_cfg, action_cfg)
    with pytest.raises(ValueError, match="obs 형상 오류"):
        runner.act(torch.zeros(1, obs_dim_value - 1))
    runner.act(torch.zeros(1, obs_dim_value))
    with pytest.raises(ValueError, match="history batch changed"):
        runner.act(torch.zeros(2, obs_dim_value))       # reset 없이 배치가 바뀌면 거부


def test_history_runner_refuses_a_stateful_single_output_module(tmp_path, cfgs):
    pre_cfg, action_cfg, obs_dim_value = cfgs
    from common.policy_io import make_obs_spec, save_obs_spec
    from common.student_policy_io import StudentHistoryPolicyRunner

    class OneOutput(torch.nn.Module):
        def forward(self, obs, history):
            return obs[:, :7]

    scripted = torch.jit.script(OneOutput())
    policy = tmp_path / "single.pt"
    scripted.save(str(policy))
    save_obs_spec(tmp_path / "obs_spec.json",
                  make_obs_spec(pre_cfg, action_cfg, action_scheme=action_cfg.mode))
    runner = StudentHistoryPolicyRunner(policy, tmp_path / "obs_spec.json", pre_cfg, action_cfg,
                                        history_steps=4, feature_dim=3)
    with pytest.raises(RuntimeError, match="\\(action, next_history\\)"):
        runner.act(torch.zeros(1, obs_dim_value))
