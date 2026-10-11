from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_policy_dryrun_guard_precedes_config_release_and_socket_paths():
    source = (ROOT / "Phase4/Deploy_DryRun.py").read_text(encoding="utf-8")
    guard = source.index("require_policy_dryrun_authorization()", source.index("def main"))
    config = source.index("dc.load_deploy_bundle(", guard)
    release = source.index("verify_release(", guard)
    socket = source.index("dc.build_real_robot(", guard)
    assert guard < config < release < socket


def test_policy_dryrun_guard_is_unconditional():
    from deployment.contracts import ContractError, require_policy_dryrun_authorization

    try:
        require_policy_dryrun_authorization(project_root=Path(__file__).resolve().parents[1])
    except ContractError as exc:
        # ★F74★ 무조건 거부가 아니라 "인가 없으면 거부"다. 저장소에는 인가 파일이
        #   커밋되지 않으므로(.gitignore) 평시 테스트에서는 여전히 거부되어야 한다.
        assert "motion authorization" in str(exc)
    else:
        raise AssertionError("policy dry-run authorization unexpectedly opened")
