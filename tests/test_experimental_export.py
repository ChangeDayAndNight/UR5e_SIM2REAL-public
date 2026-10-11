"""실험 export 레인의 계약 — ★릴리스로 둔갑할 수 없어야 한다★."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TOOL = ROOT / "scripts" / "tools" / "export_experimental_policy.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("experimental_export", _TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_machinery_is_untouched_by_the_experimental_lane():
    """★핵심 계약★ — 실험 레인은 릴리스 검증기를 호출하거나 우회하지 않는다.

    실험 산출물이 릴리스로 오인·승격되는 경로가 없어야 한다. 그래서 이 도구는
    `verify_release`를 ★호출하지 않고★ 릴리스 매니페스트를 ★쓰지도 않는다★.

    문자열 검색이 아니라 **AST**로 본다 — docstring이 "왜 별도 레인인가"를 설명하며
    `verify_release()`를 언급하는 것은 정상이고, 금지 대상은 ★실제 호출/기록★이다.
    """
    import ast

    tree = ast.parse(_TOOL.read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "verify_release" not in called, "실험 exporter가 릴리스 검증기를 호출한다"

    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    assert not {"verify_release", "RELEASE_MANIFEST_NAME"} & imported, (
        "실험 exporter가 릴리스 계약 심볼을 import한다"
    )

    written = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "release_manifest.json" not in written, "실험 exporter가 릴리스 매니페스트를 쓴다"


# 실험 레인에 존재하는 산출물 종류. ★새 종류를 추가하면 여기 등록해야★ 아래 계약 검사가
# 그 종류까지 덮는다(등록을 잊으면 테스트가 조용히 통과하는 대신 실패한다).
EXPERIMENTAL_ARTIFACT_TYPES = {
    "experimental_policy",           # 증류 student(단일 입력)
    "experimental_teacher_policy",   # 모듈형 인식 + teacher (배포 v1)
    "experimental_student_policy",   # end-to-end 이력 student (배포 v2)
}


def test_experimental_manifest_declares_itself_not_a_release():
    """산출물이 스스로 '릴리스 아님'을 선언하는지 — 사람이 파일만 보고 알 수 있어야 한다.

    ★종류와 무관하게★ 성립해야 하는 계약이다. 실험 레인에 새 종류(v1 teacher, v2 student)를
    들이면서 이 검사가 한 종류만 보게 두면, 새 산출물은 아무 선언 없이도 통과한다.
    """
    tool = _load_tool()
    assert tool.OUT_ROOT.name == "exported_experimental"
    for produced in sorted(tool.OUT_ROOT.glob("*/experimental_manifest.json")):
        manifest = json.loads(produced.read_text(encoding="utf-8"))
        assert manifest["artifact_type"] in EXPERIMENTAL_ARTIFACT_TYPES, (
            f"{produced.parent.name}: 등록되지 않은 실험 산출물 종류 "
            f"{manifest['artifact_type']!r} — EXPERIMENTAL_ARTIFACT_TYPES에 추가할 것")
        assert manifest["release"] is False
        assert manifest["adopted"] is False
        assert manifest["not_for_release_reason"]
        # 학습/export 커밋을 ★둘 다★ 남기고 불일치를 숨기지 않는다.
        assert {"training_commit", "export_commit", "commits_match"} <= set(manifest["git"]), (
            f"{produced.parent.name}: git 블록에 학습 커밋이 없다 — '어느 코드가 이 정책을 "
            f"만들었는가'가 기록되지 않았다")


def test_experimental_artifacts_load_standalone_on_a_deploy_pc():
    """배포 PC엔 ur5e_binpick 패키지가 없다 — TorchScript 단독으로 추론되어야 한다."""
    torch = pytest.importorskip("torch")
    tool = _load_tool()
    produced = sorted(tool.OUT_ROOT.glob("*/experimental_manifest.json"))
    if not produced:
        pytest.skip("실험 산출물이 아직 생성되지 않았다")
    for manifest_path in produced:
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest["contract"]
        policy = torch.jit.load(str(directory / "policy.pt"), map_location="cpu")
        policy.eval()
        obs = torch.zeros(2, contract["obs_dim"])
        history_cfg = manifest.get("history")
        with torch.no_grad():
            if history_cfg is None:
                action = policy(obs)
            else:
                # ★무상태 이력 정책★: 버퍼가 입출력으로 드러나 있어야 배포측이 에피소드
                #   경계에서 비울 수 있다. 서명이 (obs, history)인지까지 여기서 고정한다.
                assert history_cfg["stateful"] is False, (
                    f"{directory.name}: 상태를 내부에 숨긴 정책은 에피소드 경계에서 "
                    f"초기화할 수 없다")
                history = torch.zeros(2, int(history_cfg["steps"]) - 1, int(history_cfg["feature_dim"]))
                action, next_history = policy(obs, history)
                assert next_history.shape == history.shape
                assert bool(torch.isfinite(next_history).all())
        assert action.shape == (2, contract["action_dim"])
        assert bool(torch.isfinite(action).all())


def test_experimental_export_requires_a_sealed_training_receipt(tmp_path):
    """봉인되지 않은 checkpoint(진행 중 run)는 거부되어야 한다 — 완주본 한정 원칙."""
    tool = _load_tool()
    orphan = tmp_path / "model_1.pt"
    orphan.write_bytes(b"not a checkpoint")
    assert tool.main(["prog", "--checkpoint", str(orphan)]) != 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
