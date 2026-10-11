"""배포 PC 전수 점검기의 계약 — ★절대 로봇을 움직이지 않는다★."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_RUNBOOK = ROOT / "deployment" / "DeployPC_Runbook.py"


def _load():
    spec = importlib.util.spec_from_file_location("deploy_runbook", _RUNBOOK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runbook_never_commands_motion():
    """★핵심 안전 계약★ — 점검기는 명령 경로를 호출하지 않는다.

    RealRobot을 만들더라도 receive_only=True 여야 하고, servoJ/moveJ/teachMode 같은
    명령 메서드를 부르는 코드가 있어서는 안 된다. 점검기가 조용히 움직이면 사람이
    준비되지 않은 상태에서 로봇이 튀는 최악의 시나리오가 된다.
    """
    tree = ast.parse(_RUNBOOK.read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    forbidden = {"servoJ", "moveJ", "speedJ", "teachMode", "send_joint_target",
                 "emergency_stop", "run_rollout"}
    assert not (called & forbidden), f"점검기가 명령 경로를 호출한다: {called & forbidden}"

    # RealRobot 생성은 반드시 receive_only=True 여야 한다.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "RealRobot":
            receive_only = [kw for kw in node.keywords if kw.arg == "receive_only"]
            assert receive_only, "RealRobot을 receive_only 없이 생성한다"
            assert receive_only[0].value.value is True, "receive_only가 True가 아니다"


def test_runbook_reports_every_stage_without_hardware(tmp_path, capsys):
    """하드웨어가 없어도 죽지 않고 각 단계 상태를 보고해야 한다(학습 PC에서도 돈다)."""
    module = _load()
    exit_code = module.main(["prog", "--policy-dir", str(tmp_path)])
    output = capsys.readouterr().out
    # 산출물이 없으므로 실패로 끝나야 하고, 그 사유가 화면에 있어야 한다.
    assert exit_code != 0
    assert "매니페스트 없음" in output
    assert "할 일" in output


def test_stage_helper_tracks_worst_status():
    module = _load()
    stage = module.Stage(1, "테스트")
    stage.ok("정상")
    stage.done()
    assert stage.status == module._PASS
    stage.warn("주의")
    assert stage.status == module._WARN
    stage.fail("실패", "조치")
    assert stage.status == module._FAIL
    assert stage.next_action == "조치"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
