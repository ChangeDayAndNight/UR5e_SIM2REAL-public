"""The current deployment release must remain receive-only at every layer."""
from __future__ import annotations

import sys
import types

import pytest

from deployment.contracts import ContractError, require_motion_authorization


def _must_not_load_config(*_args, **_kwargs):
    raise AssertionError("configuration was loaded before the release motion block")


def test_motion_is_blocked_without_an_authorization_file(tmp_path):
    """★기본값은 차단★ — 인가 파일이 없으면 F74 이전과 똑같이 거부된다."""
    with pytest.raises(ContractError, match="motion authorization absent"):
        require_motion_authorization("freedrive", project_root=tmp_path)


def test_unknown_scope_is_rejected_before_reading_any_file(tmp_path):
    with pytest.raises(ContractError, match="unknown motion scope"):
        require_motion_authorization("teleport", project_root=tmp_path)


@pytest.mark.parametrize(
    ("module_name", "argv"),
    [
        ("deployment.Verify_JointParity", ["tool", "--run", "--real-config", "missing.yaml"]),
        ("deployment.Verify_Gripper2FG7", ["tool", "--real-config", "missing.yaml"]),
        ("deployment.Calib_HandEye", ["tool", "--real-config", "missing.yaml"]),
        ("deployment.Calib_TcpOffset", ["tool", "--real-config", "missing.yaml"]),
    ],
)
def test_command_capable_tools_block_before_config_or_constructor(monkeypatch, module_name, argv):
    module = __import__(module_name, fromlist=["main"])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(module, "load_real_config", _must_not_load_config)
    with pytest.raises(ContractError, match="motion authorization"):
        module.main()


def test_deploy_real_is_a_minimal_release_block_before_any_bundle_load(monkeypatch):
    from Phase4 import Deploy_Real

    monkeypatch.setattr(sys, "argv", ["Deploy_Real.py"])
    with pytest.raises(ContractError, match="motion authorization"):
        Deploy_Real.main()
    source = open(Deploy_Real.__file__, encoding="utf-8").read()
    assert "load_deploy_bundle" not in source
    assert "RealRobot" not in source
    assert "--no-gripper" not in source
    assert "--speed-scale" not in source


def test_real_robot_lower_layer_blocks_before_rtde_control_and_io(monkeypatch):
    from real.real_robot import RealRobot

    calls = {"control": 0, "io": 0}

    class Control:
        def __init__(self, *_args, **_kwargs):
            calls["control"] += 1

    class IO:
        def __init__(self, *_args, **_kwargs):
            calls["io"] += 1

    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=Control))
    monkeypatch.setitem(sys.modules, "rtde_io", types.SimpleNamespace(RTDEIOInterface=IO))
    with pytest.raises(ContractError, match="motion authorization"):
        RealRobot({}, "unused.yaml", receive_only=False)
    assert calls == {"control": 0, "io": 0}


def test_rtde_connect_tool_exposes_receive_only_path_only():
    from deployment import rtde_connect_test

    source = open(rtde_connect_test.__file__, encoding="utf-8").read()
    assert "--tiny-move" not in source
    assert "receive_only=True" in source
    assert "RTDEControlInterface" not in source
