from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_repository", ROOT / "scripts" / "validate_repository.py"
)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_current_repository_static_contract_is_clean():
    report = validator.validate_repository(ROOT)
    assert report.errors == []
    assert report.counts["python"] >= 160
    # ★F74★ 13 → 14: configs/motion_authorization.example.yaml 추가(실제 인가 파일은
    #   .gitignore라 추적되지 않는다 — 예시만 계약에 포함된다).
    # ★F97★ 배포 저장소는 같은 트리에 정책 번들(exported_experimental/<sha>/configs/)을 함께 싣는다.
    #   번들 설정은 export 시점의 ★증거 사본★이지 저장소가 소유한 설정이 아니므로, 저장소 소유분만
    #   14로 못 박고 번들 사본 수는 검증기가 실제로 본 파일에서 센다(학습 저장소에서는 0이다).
    #   추가로 real.local.yaml(현장 설정)은 학습 저장소에서 .gitignore이고 배포 저장소에서는
    #   '배포 PC 시작점'으로 추적된다 — 둘 다 저장소 소유 계약 14에 포함되지 않는다.
    def _shipped_extra(suffixes: set[str]) -> int:
        """저장소 소유가 아닌 '동봉물' 수 — 정책 번들 사본과 현장 설정."""
        return sum(
            1 for path in validator.effective_repository_files(ROOT)
            if path.suffix.casefold() in suffixes
            and ("exported_experimental" in path.parts or path.name == "real.local.yaml")
        )

    assert report.counts["yaml"] == 14 + _shipped_extra({".yaml", ".yml"})
    assert report.counts["json"] == 25 + _shipped_extra({".json"})
    assert report.counts["toml"] == 2
    assert report.counts["urdf"] == 4
    # ★F97★ 배포 저장소의 문서 세트: README.md + DEPLOY_GUIDE.md + PROJECT_LOG.md + 번들별 DEPLOY_README.md
    assert report.counts["markdown"] == 1 + sum(
        1 for path in validator.effective_repository_files(ROOT)
        if path.suffix.casefold() == ".md" and path.name != "README.md"
    )


def test_sensitive_scanner_catches_real_machine_identity_and_credentials():
    private_ip = "192" + ".168.4.20"
    secret_assignment = "api_key" + " = 'not-a-placeholder-secret'"
    user_path = "C:" + "\\Users\\actual_operator\\capture.json"
    json_serial = '"serial"' + ': "839112073111"'
    text = "\n".join((private_ip, secret_assignment, user_path, json_serial))
    findings = validator.sensitive_text_findings("fixture.txt", text)
    assert "private IPv4 address" in findings
    assert "non-placeholder credential assignment" in findings
    assert "user-specific absolute path" in findings
    assert "hard-coded device serial" in findings


def test_sensitive_scanner_allows_test_net_and_placeholder_paths():
    text = "server=192.0.2.10\nasset=@/home/name/asset.usd@\npassword='placeholder'"
    assert validator.sensitive_text_findings("fixture.txt", text) == []


def test_generated_artifact_policy_blocks_duplicate_bundle_and_runtime_outputs():
    duplicate = "Real" + "_Verify/common/config.py"
    assert validator.generated_artifact_reason(duplicate) == "legacy duplicate deployment bundle"
    assert validator.generated_artifact_reason("Phase1/Result/run/report.txt")
    assert validator.generated_artifact_reason("Phase1/logs/train.log")
    assert validator.generated_artifact_reason("deployment/wheelhouse/pkg.whl")
    assert validator.generated_artifact_reason("calibration/captures_final/results/summary.txt") is None


def test_repository_validator_reports_format_policy_size_and_git_symlink(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("# root\n", encoding="utf-8")
    (tmp_path / "extra.md").write_text("duplicate\n", encoding="utf-8")
    (tmp_path / "bad.json").write_text("{broken\n", encoding="utf-8")
    (tmp_path / "stale.txt").write_text("Phase0/rtde_connect_test.py\n", encoding="utf-8")
    (tmp_path / "large.bin").write_bytes(b"x" * 64)
    (tmp_path / "target.txt").write_text("ok\n", encoding="utf-8")
    (tmp_path / "tracked-link").write_text("target.txt", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    blob = subprocess.run(
        ["git", "hash-object", "-w", "tracked-link"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"120000,{blob},tracked-link"],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setattr(validator, "MAX_TRACKED_BYTES", 32)

    report = validator.validate_repository(tmp_path)
    joined = "\n".join(report.errors)
    assert "project Markdown policy violated" in joined
    assert "parse failed: bad.json" in joined
    assert "stale deleted-path reference" in joined
    assert "file exceeds 100 MiB: large.bin" in joined
    assert "symlink is forbidden: tracked-link" in joined
