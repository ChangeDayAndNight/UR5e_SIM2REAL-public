"""모듈형 배포 v1 — Teacher 관측 조립기·계약서·rollout 훅 회귀테스트 (Isaac 불필요)."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from common.teacher_contract import TEACHER_OBS_DIM, teacher_obs_slice
from common.teacher_obs import (
    T_base_cam_from_tcp, TeacherObsBuilder, TeacherObsCfg, executable_last_action, table_center_z,
)
from common.types import CameraFrame, Intrinsics, RobotState


@dataclass
class FakeDetection:
    present: bool
    reason: str = ""
    center_base: tuple | None = None


def make_state(tcp_xyz, gripper_cmd=0.0):
    return RobotState(
        joint_pos=np.linspace(-1, 1, 6), joint_vel=np.zeros(6),
        tcp_pose=np.array([*tcp_xyz, 1.0, 0.0, 0.0, 0.0]), tcp_force=None, timestamp=0.0,
        gripper_command_state=gripper_cmd,
    )


def make_frame():
    return CameraFrame(depth_m=np.full((8, 8), 0.3, dtype=np.float32), timestamp=0.0,
                       rgb=np.zeros((8, 8, 3), dtype=np.uint8), sequence_id=1)


INTR = Intrinsics(fx=80.0, fy=140.0, cx=56.0, cy=56.0, width=112, height=112)
EYE = np.eye(4)


def builder_with(detections, **cfg_kwargs):
    queue = list(detections)

    def locate(depth, intrinsics, T_base_cam):
        return queue.pop(0) if queue else FakeDetection(False, "empty")

    return TeacherObsBuilder(INTR, EYE, EYE, TeacherObsCfg(**cfg_kwargs), dt_s=0.05, locate=locate)


def test_executable_last_action_clamps_and_zeroes_tilt():
    a = executable_last_action(np.array([3.0, -3.0, 0.5, 0.7, -0.7, 0.2, -9.0]), vertical_grasp_only=True)
    assert a.tolist() == [1.0, -1.0, 0.5, 0.0, 0.0, 0.2, -1.0]
    b = executable_last_action(np.array([0.0, 0.0, 0.0, 0.7, -0.7, 0.0, 0.0]), vertical_grasp_only=False)
    assert b[3] == 0.7 and b[4] == -0.7


def test_camera_chain_reduces_to_tcp_pose_with_identity_calibration():
    tcp = np.array([0.4, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0])
    T = T_base_cam_from_tcp(tcp, EYE, EYE)
    assert np.allclose(T[:3, 3], tcp[:3]) and np.allclose(T[:3, :3], np.eye(3))
    # TCP offset 0.2425 m along flange z: camera sits at flange, i.e. 0.2425 m above the TCP
    T_flange_tcp = np.eye(4); T_flange_tcp[2, 3] = 0.2425
    T2 = T_base_cam_from_tcp(tcp, T_flange_tcp, EYE)
    assert np.isclose(T2[2, 3], tcp[2] - 0.2425)


def test_table_center_and_depth_locator_on_synthetic_cube():
    from common.cube_depth_locator import DepthLocatorCfg, locate_cube_from_depth
    cfg = TeacherObsCfg(table_height_m=-0.0375, object_size_m=0.05)
    assert table_center_z(cfg) == pytest.approx(-0.0125)
    # 합성 장면: 카메라가 base +z를 내려다본다(광축 -z). 작업면 0.40 m 아래, 5 cm 큐브 상면 0.35 m 아래.
    side = 112; f = 80.87
    intr = Intrinsics(fx=f, fy=f, cx=side / 2, cy=side / 2, width=side, height=side)
    depth = np.full((side, side), 0.40, dtype=np.float32)
    cube_px = int(round(0.05 * f / 0.35))                       # 상면 폭 ≈ 11 px
    r0, c0 = 40, 60
    depth[r0:r0 + cube_px, c0:c0 + cube_px] = 0.35
    T_base_cam = np.array([[1, 0, 0, 0.5], [0, -1, 0, 0.0], [0, 0, -1, 0.3625], [0, 0, 0, 1.0]], dtype=np.float64)
    loc_cfg = DepthLocatorCfg(table_height_m=-0.0375, object_size_m=0.05, workspace_min=(-2, -2, -1), workspace_max=(2, 2, 1))
    est = locate_cube_from_depth(depth, intr, T_base_cam, loc_cfg)
    assert est.present, est.reason
    # 상면 중심 픽셀 → 카메라 x,y → base: x = 0.5 + xc, y = -yc, z = 0.3625 - 0.35 - 0.025 = -0.0125
    uc, vc = c0 + cube_px / 2 - 0.5, r0 + cube_px / 2 - 0.5
    xc, yc = (uc - side / 2) * 0.35 / f, (vc - side / 2) * 0.35 / f
    assert np.allclose(est.center_base, [0.5 + xc, -yc, -0.0125], atol=0.006)
    # 손가락 같은 길쭉한 조각은 거부된다
    finger = np.full((side, side), 0.40, dtype=np.float32); finger[50:53, 20:45] = 0.30
    assert not locate_cube_from_depth(finger, intr, T_base_cam, loc_cfg).present


def test_builder_layout_held_latch_and_hold_request():
    tcp = (0.5, 0.0, 0.03)
    b = builder_with([
        FakeDetection(True, center_base=(0.5, 0.0, 0.00)),     # step 1: open, cube on table under TCP
        FakeDetection(False, "occluded"),                      # step 2: close command → latch on last estimate
        FakeDetection(False, "occluded"),                      # step 3: closed, lifted, occluded → held_offset
        FakeDetection(False, "gone"),                          # 4-6: open, no cube → last_known, no hold yet
        FakeDetection(False, "gone"), FakeDetection(False, "gone"),
    ], max_missed_frames_before_hold=2)
    target = np.array([0.4, 0.2, 0.0]); last = np.array([0.1, 0.2, 0.3, 0.9, 0.9, 0.1, -2.0])
    obs = b(make_frame(), make_state(tcp, 0.0), target, last)
    assert obs.shape == (1, TEACHER_OBS_DIM) and torch.isfinite(obs).all()
    o = obs[0].numpy()
    assert np.allclose(o[teacher_obs_slice("target_pos")], target)
    assert o[teacher_obs_slice("gripper_command_state")][0] == 0.0
    assert o[teacher_obs_slice("held")][0] == 0.0
    assert o[teacher_obs_slice("object_pos")][2] == pytest.approx(-0.0375 + 0.025)   # snapped to table
    assert np.allclose(o[teacher_obs_slice("last_action")], [0.1, 0.2, 0.3, 0.0, 0.0, 0.1, -1.0])
    assert not b.hold_requested and b.debug.object_source == "detected"

    # 폐합 전이: 큐브가 안 보여도 마지막 추정치가 TCP 반경 안 → held latch (배포·sim 공통 규칙)
    o2 = b(make_frame(), make_state(tcp, 1.0), target, last)[0].numpy()
    assert b.debug.held and o2[teacher_obs_slice("held")][0] == 1.0
    assert o2[teacher_obs_slice("gripper_command_state")][0] == 1.0
    assert b.debug.object_source == "held_offset"

    o3 = b(make_frame(), make_state((0.5, 0.0, 0.08), 1.0), target, last)[0].numpy()
    assert b.debug.held and b.debug.object_source == "held_offset" and not b.hold_requested
    assert np.allclose(o3[teacher_obs_slice("object_pos")], np.array([0.5, 0.0, 0.08]) + TeacherObsCfg().held_offset_from_tcp_m)

    # ★F93★ hold는 마지막 추정치에서 ★멀리★(held_radius+object_size 밖) 떨어져 열린 채 못 볼 때만 센다.
    for k in range(3):
        b(make_frame(), make_state((0.7, 0.0, 0.08), 0.0), target, last)
    assert b.hold_requested and b.debug.object_source == "last_known" and not b.debug.held
    s = b.stats(); assert s["steps"] == 6 and s["held_fraction"] == pytest.approx(2 / 6)
    assert s["max_blind_streak"] == 3


def test_blind_descent_over_last_estimate_never_requests_hold():
    """★F93(리뷰 T-01/P2)★ 최종 하강·폐합 전이는 현 브라켓에서 항상 블라인드다 — 마지막 추정치 바로
    위에서의 결측(그리고 폐합 전이 중의 결측)은 hold를 유발하면 안 된다(팔 영구 정지 = 파지 실패)."""
    n_blind = TeacherObsCfg().max_missed_frames_before_hold + 20
    dets = [FakeDetection(True, center_base=(0.5, 0.0, 0.00))] + [FakeDetection(False, "fov")] * n_blind
    b = builder_with(dets)
    target = np.array([0.4, 0.2, 0.0]); last = np.zeros(7)
    b(make_frame(), make_state((0.5, 0.0, 0.20), 0.0), target, last)          # 큐브를 한 번 보고
    z = 0.20
    for k in range(n_blind):                                                    # 바로 위로 내려간다(블라인드)
        z = max(0.02, z - 0.0125)
        state_val = 0.0 if k < n_blind - 30 else min(1.0, (k - (n_blind - 30)) / 24.0)  # 마지막 30스텝 폐합 전이
        b(make_frame(), make_state((0.5, 0.0, z), state_val), target, last)
        assert not b.hold_requested, f"step {k}: 예상된 블라인드 콘에서 hold가 걸렸다"
    assert b.stats()["max_blind_streak"] >= 1
    assert TeacherObsCfg().max_missed_frames_before_hold >= 80


def test_builder_releases_held_when_cube_is_seen_far_from_tcp():
    """헛집기 복구: 닫힌 채 큐브가 TCP에서 멀리 검출되면 held가 풀려 정책이 재시도할 수 있다."""
    b = builder_with([
        FakeDetection(True, center_base=(0.5, 0.0, 0.0)),      # open, cube under TCP
        FakeDetection(False, "occluded"),                      # close → latch
        FakeDetection(True, center_base=(0.5, 0.0, 0.0)),      # closed, TCP lifted 20 cm, cube still on table
    ])
    t = np.zeros(3); la = np.zeros(7)
    b(make_frame(), make_state((0.5, 0.0, 0.03), 0.0), t, la)
    b(make_frame(), make_state((0.5, 0.0, 0.03), 1.0), t, la)
    assert b.debug.held
    o = b(make_frame(), make_state((0.5, 0.0, 0.23), 1.0), t, la)[0].numpy()
    assert not b.debug.held and o[teacher_obs_slice("held")][0] == 0.0
    assert o[teacher_obs_slice("object_pos")][2] == pytest.approx(-0.0125)          # snapped back to table


def test_builder_does_not_latch_when_closing_far_from_cube():
    b = builder_with([FakeDetection(True, center_base=(0.5, 0.0, 0.0)), FakeDetection(False, "occluded")])
    t = np.zeros(3); la = np.zeros(7)
    b(make_frame(), make_state((0.5, 0.0, 0.30), 0.0), t, la)
    o = b(make_frame(), make_state((0.5, 0.0, 0.30), 1.0), t, la)[0].numpy()
    assert not b.debug.held and o[teacher_obs_slice("held")][0] == 0.0


def test_builder_rejects_jumps_until_confirmed():
    b = builder_with([
        FakeDetection(True, center_base=(0.5, 0.0, 0.0)),
        FakeDetection(True, center_base=(0.9, 0.0, 0.0)),    # 40 cm jump → rejected (pending 1)
        FakeDetection(True, center_base=(0.9, 0.0, 0.0)),    # confirmed on the 2nd frame → accepted
    ])
    st = make_state((0.5, 0.0, 0.3), 0.0); t = np.zeros(3); la = np.zeros(7)
    o1 = b(make_frame(), st, t, la)[0].numpy(); assert o1[teacher_obs_slice("object_pos")][0] == pytest.approx(0.5)
    o2 = b(make_frame(), st, t, la)[0].numpy(); assert o2[teacher_obs_slice("object_pos")][0] == pytest.approx(0.5)
    assert not b.debug.detected and "급점프" in b.debug.reason and b.jump_rejects == 1
    o3 = b(make_frame(), st, t, la)[0].numpy(); assert o3[teacher_obs_slice("object_pos")][0] == pytest.approx(0.9)
    assert b.debug.detected


def test_builder_rejects_unknown_gripper_state():
    b = builder_with([FakeDetection(True, center_base=(0.5, 0.0, 0.0))])
    with pytest.raises(ValueError, match="gripper command state"):
        b(make_frame(), make_state((0.5, 0.0, 0.03), None), np.zeros(3), np.zeros(7))


def test_teacher_spec_roundtrip_and_tamper_detection():
    from common.config_io import build_action_cfg, configs_dir, load_yaml
    from common.teacher_policy_io import make_teacher_obs_spec, validate_teacher_obs_spec
    action_cfg = build_action_cfg(load_yaml(configs_dir() / "common.yaml"))
    spec = make_teacher_obs_spec(action_cfg, action_scheme=action_cfg.mode, vertical_grasp_only=True)
    validate_teacher_obs_spec(spec, action_cfg)
    assert spec["obs_dim"] == 34 and spec["action_dim"] == 7 and spec["policy_kind"] == "teacher_state"
    bad = copy.deepcopy(spec); bad["cfg_hash"] = "0" * 16
    with pytest.raises(ValueError, match="cfg_hash"):
        validate_teacher_obs_spec(bad, action_cfg)
    student = {"obs_dim": 50206, "action_dim": 7, "layout": [], "cfg_hash": "x"}
    with pytest.raises(ValueError, match="teacher_state"):
        validate_teacher_obs_spec(student, action_cfg)
    wrong = copy.deepcopy(spec); wrong["action_scheme"] = "joint_delta"
    with pytest.raises(ValueError):
        validate_teacher_obs_spec(wrong, action_cfg)


def test_rollout_uses_obs_builder_and_honours_its_hold():
    from common.rollout import run_rollout
    from common.action_mapping import ActionMapper
    from common.config_io import build_action_cfg, build_preprocess_cfg, build_safety_cfg, configs_dir, load_yaml
    from common.safety import SafetyGuard
    from common.timing import RateLimiter
    from test_rollout_mock import MockCamera, MockRobot
    common = load_yaml(configs_dir() / "common.yaml")
    pre_cfg, action_cfg = build_preprocess_cfg(common), build_action_cfg(common)
    safety_cfg = build_safety_cfg(load_yaml(configs_dir() / "safety.yaml"))
    robot, camera = MockRobot(), MockCamera()
    calls = []

    class Builder:
        hold_requested = False

        def __call__(self, frame, state, target_pos, last_action):
            calls.append(np.asarray(last_action).copy())
            self.hold_requested = len(calls) >= 3
            return torch.zeros(1, TEACHER_OBS_DIM)

    class Policy:
        def act(self, obs):
            assert obs.shape == (1, TEACHER_OBS_DIM)
            return np.full((1, action_cfg.action_dim), 0.5)

    builder = Builder()
    result = run_rollout(robot, camera, Policy(), ActionMapper(action_cfg), SafetyGuard(safety_cfg),
                         RateLimiter(20.0), pre_cfg, np.zeros(3), max_steps=4, dry_run=True, obs_builder=builder)
    assert result.steps == 4 and not result.tripped
    assert len(calls) == 4 and np.allclose(calls[1], 0.5)      # last_action fed back into the builder
    assert builder.hold_requested
