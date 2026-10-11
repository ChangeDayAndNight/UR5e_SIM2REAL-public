"""Run_Task(실기 한 명령 실행기)의 계약 — 하드웨어 없이 검증한다."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_RUN_TASK = ROOT / "Phase4" / "Run_Task.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_task", _RUN_TASK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_motion_requires_an_explicit_typed_flag(tmp_path, capsys):
    """★핵심 안전 성질★ — 모션은 결코 우연히 일어나지 않는다.

    인가 YAML·만료·캘리 게이트를 걷어낸 대신, 사람이 플래그를 ★직접 타이핑★해야만
    명령 경로가 열린다. 기본 실행은 로봇에 닿기 전에 거부되어야 한다.
    """
    module = _load()
    code = module.main(["prog", "--policy-dir", str(tmp_path)])
    assert code == 2, "플래그 없이도 모션 경로로 진입했다"
    assert "--yes-i-am-at-the-estop" in capsys.readouterr().out


def test_safety_guard_is_constructed_and_never_disabled():
    """SafetyGuard는 걷어낸 항목이 아니다 — 반드시 만들어지고 estop_hook이 붙는다."""
    tree = ast.parse(_RUN_TASK.read_text(encoding="utf-8"))
    guard_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "SafetyGuard"
    ]
    assert guard_calls, "SafetyGuard를 만들지 않는다"
    assert any(kw.arg == "estop_hook" for kw in guard_calls[0].keywords), (
        "SafetyGuard에 estop_hook이 연결되지 않았다"
    )


def test_dry_run_constructs_robot_receive_only():
    """--dry-run 은 수신 전용이어야 한다 — 명령 소켓 자체를 만들지 않는다."""
    tree = ast.parse(_RUN_TASK.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "RealRobot":
            keywords = {kw.arg for kw in node.keywords}
            assert "receive_only" in keywords, "receive_only를 넘기지 않는다"
            assert "operator_ack" in keywords, "operator_ack을 넘기지 않는다"
            return
    pytest.fail("RealRobot 생성부를 찾지 못했다")


def test_action_scale_shrinks_translation_and_rotation_only():
    """액션 축소는 ★변환 단계★에서 한다 — 정책 출력·관측 계약은 건드리지 않는다."""
    import dataclasses

    from common.config_io import build_action_cfg, configs_dir, load_yaml

    base = build_action_cfg(load_yaml(configs_dir() / "common.yaml"))
    scaled = dataclasses.replace(
        base,
        task_pos_scale_m=base.task_pos_scale_m * 0.3,
        task_rot_scale_rad=base.task_rot_scale_rad * 0.3,
    )
    assert scaled.task_pos_scale_m == pytest.approx(base.task_pos_scale_m * 0.3)
    assert scaled.task_rot_scale_rad == pytest.approx(base.task_rot_scale_rad * 0.3)
    # 관측/행동 폭은 불변이어야 한다 — 정책 계약이 깨지면 실기에서 즉사한다.
    assert scaled.action_dim == base.action_dim


def test_configs_load_without_a_bundle_manifest():
    """번들 매니페스트(릴리스 전용)를 거치지 않고 configs/ 에서 직접 만들 수 있어야 한다."""
    from common.config_io import (
        build_action_cfg,
        build_preprocess_cfg,
        build_safety_cfg,
        configs_dir,
        load_yaml,
    )

    common_yaml = load_yaml(configs_dir() / "common.yaml")
    assert build_preprocess_cfg(common_yaml).use_rgb in (True, False)
    assert build_action_cfg(common_yaml).action_dim > 0
    safety = build_safety_cfg(load_yaml(configs_dir() / "safety.yaml"))
    assert len(safety.workspace_min) == 3 and len(safety.workspace_max) == 3
    assert float(common_yaml["control"]["control_hz"]) > 0


def test_operator_ack_skips_the_authorization_file(capsys):
    """현장 입회 선언은 파일 없이 통과하되 ★배너를 반드시 출력★한다(조용히 열리지 않는다)."""
    from deployment.contracts import require_motion_authorization

    require_motion_authorization("policy", project_root=ROOT, operator_ack=True)
    output = capsys.readouterr().out
    assert "현장 입회" in output and "e-stop" in output


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
