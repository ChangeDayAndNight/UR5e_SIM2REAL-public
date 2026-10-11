"""Canonical generated-bundle source contract tests."""
import json
from pathlib import Path
import hashlib
import subprocess
import types

import numpy as np
import pytest

from common.file_snapshot import RegularFileSnapshot
from deployment import build_bundle
from deployment.contracts import ContractError
from deployment.verify_deployment import verify_bundle_manifest
from deployment import verify_deployment

ROOT = Path(__file__).resolve().parents[1]


class _VerifiedRelease(dict):
    def __init__(self, value: dict, digest: str) -> None:
        super().__init__(value)
        self._verified_manifest_sha256 = digest


def _write_minimal_verified_bundle(root: Path) -> dict:
    exported = root / "exported"
    deployment = root / "deployment"
    wheelhouse = deployment / "wheelhouse"
    exported.mkdir(parents=True)
    wheelhouse.mkdir(parents=True)
    policy = exported / "policy.pt"
    spec = exported / "obs_spec.json"
    policy.write_bytes(b"trusted-policy")
    spec.write_text("{}", encoding="utf-8")
    release = {
        "source": {"git_commit": "a" * 40},
        "files": {
            "policy.pt": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "obs_spec.json": hashlib.sha256(spec.read_bytes()).hexdigest(),
        },
    }
    release_path = exported / "release_manifest.json"
    release_path.write_text(json.dumps(release, sort_keys=True), encoding="utf-8")
    provenance = deployment / "wheelhouse.manifest.json"
    provenance.write_text("{}", encoding="utf-8")
    lock = deployment / "requirements-win11.lock.txt"
    lock.write_text("--require-hashes\n", encoding="utf-8")
    wheel = wheelhouse / "trusted.whl"
    wheel.write_bytes(b"trusted-wheel")
    paths = (policy, spec, release_path, provenance, lock, wheel)
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    wheel_files = {
        "trusted.whl": {
            "bytes": wheel.stat().st_size,
            "sha256": files["deployment/wheelhouse/trusted.whl"],
        }
    }
    manifest = {
        "schema_version": 2,
        "source_git_commit": "a" * 40,
        "files": files,
        "wheelhouse": {
            "provenance_manifest_sha256": files["deployment/wheelhouse.manifest.json"],
            "lock_sha256": files["deployment/requirements-win11.lock.txt"],
            "aggregate_sha256": hashlib.sha256(
                json.dumps(wheel_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "files": wheel_files,
        },
    }
    (root / "bundle_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return manifest


def test_legacy_real_verify_tree_is_removed():
    path = ROOT / "Real_Verify"
    assert not path.exists() or not any(item.is_file() for item in path.rglob("*"))


def test_canonical_deployment_sources_have_no_markdown_generators():
    for source in (ROOT / "deployment").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert 'write_text("#' not in text
        assert '"report.md"' not in text


def test_phase4_readiness_reuses_frozen_bundle_authority_hashes():
    source = (ROOT / "Phase4/_deploy_common.py").read_text(encoding="utf-8")
    assert "manifest = verify_bundle_manifest(ROOT_DIR)" in source
    assert "expected_manifest_sha256=bundle.release_manifest_sha256" in source
    assert "release_manifest_sha256=bundle.release_manifest_sha256" in source
    assert "bundle_manifest_sha256=bundle.bundle_manifest_sha256" in source
    assert "sha256_file(release_manifest)" not in source
    assert "sha256_file(bundle_manifest)" not in source


def test_generated_tree_excludes_markdown_and_cache():
    assert ".md" in build_bundle.EXCLUDED_SUFFIXES
    assert "__pycache__" in build_bundle.EXCLUDED_PARTS
    assert "deployment" in build_bundle.CANONICAL_TREES


def test_bundle_legal_files_are_copied_hashed_and_drift_rejected(tmp_path):
    output = tmp_path / "bundle"
    release = tmp_path / "release"
    wheelhouse = tmp_path / "wheelhouse"
    output.mkdir()
    release.mkdir()
    wheelhouse.mkdir()
    wheelhouse_record = {
        "target": "test",
        "method": "test",
        "provenance_manifest_sha256": "a" * 64,
        "lock_sha256": "b" * 64,
        "aggregate_sha256": "c" * 64,
        "files": {},
    }

    manifest = build_bundle._populate_bundle(
        release,
        output,
        wheelhouse,
        wheelhouse_record,
        source_git_commit="a" * 40,
    )
    assert manifest["source_git_commit"] == "a" * 40
    for relative in build_bundle.BUNDLE_LEGAL_FILES:
        bundled = output / relative
        assert bundled.read_bytes() == (ROOT / relative).read_bytes()
        assert manifest["files"][relative] == build_bundle.sha256_file(bundled)

    license_path = output / "assets/ur5e_2fg7/LICENSE-BSD-3-Clause.txt"
    license_path.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ContractError, match="bundle hash mismatch"):
        verify_bundle_manifest(output)


def test_wheelhouse_rejects_any_non_wheel_entry(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "unexpected.exe").write_bytes(b"not a wheel")
    with pytest.raises(ContractError, match="only regular .whl"):
        build_bundle._wheelhouse_record(wheelhouse)


def test_wheelhouse_provenance_parse_and_digest_share_one_snapshot(monkeypatch, tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    deployment = tmp_path / "deployment"
    wheelhouse.mkdir()
    deployment.mkdir()
    wheel = wheelhouse / "trusted.whl"
    wheel.write_bytes(b"trusted-wheel")
    file_record = {
        wheel.name: {
            "bytes": wheel.stat().st_size,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
    }
    trusted = json.dumps({
        "schema_version": 1,
        "target": "trusted-target",
        "method": "trusted-method",
        "files": file_record,
    }).encode("utf-8")
    swapped = json.dumps({
        "schema_version": 1,
        "target": "swapped-target",
        "method": "swapped-method",
        "files": file_record,
    }).encode("utf-8")
    provenance = deployment / "wheelhouse.manifest.json"
    provenance.write_bytes(trusted)
    (deployment / "requirements-win11.lock.txt").write_text(
        "--require-hashes\n", encoding="utf-8"
    )
    real_snapshot = build_bundle.read_regular_file_snapshot

    def swap_after_open(path, **kwargs):
        snapshot = real_snapshot(path, **kwargs)
        if Path(path) == provenance:
            provenance.write_bytes(swapped)
        return snapshot

    monkeypatch.setattr(build_bundle, "ROOT", tmp_path)
    monkeypatch.setattr(build_bundle, "read_regular_file_snapshot", swap_after_open)
    record = build_bundle._wheelhouse_record(wheelhouse)
    assert record["target"] == "trusted-target"
    assert record["method"] == "trusted-method"
    assert record["provenance_manifest_sha256"] == hashlib.sha256(trusted).hexdigest()
    assert provenance.read_bytes() == swapped


def test_copy_tree_rejects_symlink_input(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    external = tmp_path / "external.py"
    external.write_text("outside", encoding="utf-8")
    link = source / "linked.py"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is not available on this host")
    with pytest.raises(ContractError, match="symlink/reparse"):
        build_bundle._copy_tree(source, tmp_path / "destination")


def _pretend_windows_os(monkeypatch) -> None:
    """verify_deployment가 보는 os.name만 'nt'로 바꾼다.

    전역 os.name을 바꾸면 Linux(CI)에서 pathlib이 WindowsPath를 만들다 pytest까지 죽는다.
    """
    fake_os = types.ModuleType("os")
    fake_os.__dict__.update(vars(verify_deployment.os))
    fake_os.name = "nt"
    monkeypatch.setattr(verify_deployment, "os", fake_os)


def test_windows_runtime_uses_supported_win11_x64_baseline():
    runtime = json.loads((ROOT / "deployment/runtime-win11.json").read_text(encoding="utf-8"))
    assert runtime["os_product"] == "Windows 11"
    assert runtime["minimum_build"] == 22000
    assert runtime["architecture"] == "AMD64"
    assert runtime["python"] == "3.11.15"
    assert "24H2" not in runtime["notes"]
    assert runtime["allowed_bootstrap_packages"] == []


def test_runtime_rejects_python_distributions_outside_lock_policy(monkeypatch, tmp_path):
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "runtime-win11.json").write_text(json.dumps({
        "os": "Windows",
        "python_implementation": "CPython",
        "python": "3.11.15",
        "architecture": "AMD64",
        "minimum_build": 22000,
        "allowed_bootstrap_packages": [],
    }), encoding="utf-8")
    (deployment / "requirements-win11.lock.txt").write_text(
        "--require-hashes\n"
        "foo==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n"
        "pip==25.3 \\\n    --hash=sha256:" + "b" * 64 + "\n",
        encoding="utf-8",
    )

    class Dist:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(verify_deployment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(verify_deployment.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(verify_deployment.platform, "python_version", lambda: "3.11.15")
    monkeypatch.setattr(verify_deployment.platform, "machine", lambda: "AMD64")
    _pretend_windows_os(monkeypatch)
    monkeypatch.setattr(
        verify_deployment.sys,
        "getwindowsversion",
        lambda: type("Win", (), {"build": 22631})(),
        raising=False,
    )
    monkeypatch.setattr(
        verify_deployment.importlib.metadata,
        "distributions",
        lambda: [Dist("foo", "1.0"), Dist("pip", "25.3"), Dist("evil-plugin", "1")],
    )
    with pytest.raises(ContractError, match="outside the exact lock"):
        verify_deployment.verify_runtime(tmp_path)

    monkeypatch.setattr(
        verify_deployment.importlib.metadata,
        "distributions",
        lambda: [Dist("foo", "1.0"), Dist("pip", "25.3")],
    )
    result = verify_deployment.verify_runtime(tmp_path)
    assert result["additional_packages"] == []


def test_bundle_output_cannot_be_nested_in_canonical_source(monkeypatch, tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    monkeypatch.setattr(build_bundle, "ROOT", tmp_path)
    with pytest.raises(ContractError, match="source/release/wheelhouse"):
        build_bundle.build_bundle(release, tmp_path / "deployment" / "nested")


def test_staged_source_commit_binding_uses_checkout_filters_for_crlf(monkeypatch, tmp_path):
    staging = tmp_path / "staging"
    staged = staging / "common/example.py"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"print('ok')\r\n")
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    seen: list[list[str]] = []

    def filtered(command, **_kwargs):
        seen.append(command)
        if command[1] == "hash-object":
            return ("a" * 40 + "\n").encode("ascii")
        return "a" * 40 + "\n"

    monkeypatch.setattr(subprocess, "check_output", filtered)
    build_bundle._verify_staged_sources_at_commit(
        staging,
        {"files": {"common/example.py": digest}},
        "a" * 40,
    )
    assert seen and seen[0][1] == "hash-object"
    assert "--path=common/example.py" in seen[0]
    assert seen[1][1] == "rev-parse"


def test_staged_git_oid_and_manifest_hash_use_identical_opened_bytes(monkeypatch, tmp_path):
    staging = tmp_path / "staging"
    staged = staging / "common/example.py"
    staged.parent.mkdir(parents=True)
    trusted = b"print('trusted')\n"
    swapped = b"print('swapped')\n"
    staged.write_bytes(trusted)
    trusted_sha256 = hashlib.sha256(trusted).hexdigest()
    real_snapshot = build_bundle.read_regular_file_snapshot
    git_inputs: list[bytes] = []

    def swap_after_open(path, **kwargs):
        snapshot = real_snapshot(path, **kwargs)
        if Path(path) == staged:
            staged.write_bytes(swapped)
        return snapshot

    def git_result(command, **kwargs):
        if command[1] == "hash-object":
            git_inputs.append(kwargs["input"])
            return ("a" * 40 + "\n").encode("ascii")
        return "a" * 40 + "\n"

    monkeypatch.setattr(build_bundle, "read_regular_file_snapshot", swap_after_open)
    monkeypatch.setattr(subprocess, "check_output", git_result)
    build_bundle._verify_staged_sources_at_commit(
        staging,
        {"files": {"common/example.py": trusted_sha256}},
        "a" * 40,
    )
    assert git_inputs == [trusted]
    assert staged.read_bytes() == swapped


def test_bundle_and_embedded_release_are_parsed_from_manifest_anchored_bytes(
    monkeypatch, tmp_path
):
    trusted_manifest = _write_minimal_verified_bundle(tmp_path)
    manifest_path = tmp_path / "bundle_manifest.json"
    release_path = tmp_path / "exported/release_manifest.json"
    trusted_manifest_bytes = manifest_path.read_bytes()
    trusted_release_bytes = release_path.read_bytes()
    swapped_manifest = {
        **trusted_manifest,
        "source_git_commit": "b" * 40,
    }
    swapped_release = json.loads(trusted_release_bytes.decode("utf-8"))
    swapped_release["source"]["git_commit"] = "b" * 40
    real_snapshot = verify_deployment.read_regular_file_snapshot

    def swap_after_open(path, **kwargs):
        snapshot = real_snapshot(path, **kwargs)
        if Path(path) == manifest_path:
            manifest_path.write_text(
                json.dumps(swapped_manifest, sort_keys=True), encoding="utf-8"
            )
        elif Path(path) == release_path:
            release_path.write_text(
                json.dumps(swapped_release, sort_keys=True), encoding="utf-8"
            )
        return snapshot

    monkeypatch.setattr(
        verify_deployment, "read_regular_file_snapshot", swap_after_open
    )
    verified = verify_bundle_manifest(tmp_path)
    assert verified["source_git_commit"] == "a" * 40
    assert verified._verified_manifest_sha256 == hashlib.sha256(
        trusted_manifest_bytes
    ).hexdigest()
    assert verified._verified_release_manifest["source"]["git_commit"] == "a" * 40
    assert verified._verified_release_manifest_sha256 == hashlib.sha256(
        trusted_release_bytes
    ).hexdigest()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["source_git_commit"] == "b" * 40
    assert json.loads(release_path.read_text(encoding="utf-8"))["source"]["git_commit"] == "b" * 40


def test_runtime_and_lock_parse_and_hash_the_same_opened_payloads(monkeypatch, tmp_path):
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    runtime_path = deployment / "runtime-win11.json"
    trusted_runtime = json.dumps({
        "os": "Windows",
        "python_implementation": "CPython",
        "python": "3.11.15",
        "architecture": "AMD64",
        "minimum_build": 22000,
        "allowed_bootstrap_packages": [],
    }).encode("utf-8")
    runtime_path.write_bytes(trusted_runtime)
    lock_path = deployment / "requirements-win11.lock.txt"
    trusted_lock = (
        "--require-hashes\n"
        "foo==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\n"
    ).encode("utf-8")
    lock_path.write_bytes(trusted_lock)
    real_snapshot = verify_deployment.read_regular_file_snapshot

    def swap_after_open(path, **kwargs):
        snapshot = real_snapshot(path, **kwargs)
        if Path(path) == runtime_path:
            runtime_path.write_text('{"os":"Swapped"}', encoding="utf-8")
        elif Path(path) == lock_path:
            lock_path.write_text("evil==9\n", encoding="utf-8")
        return snapshot

    class Dist:
        metadata = {"Name": "foo"}
        version = "1.0"

    monkeypatch.setattr(
        verify_deployment, "read_regular_file_snapshot", swap_after_open
    )
    monkeypatch.setattr(verify_deployment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(verify_deployment.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(verify_deployment.platform, "python_version", lambda: "3.11.15")
    monkeypatch.setattr(verify_deployment.platform, "machine", lambda: "AMD64")
    _pretend_windows_os(monkeypatch)
    monkeypatch.setattr(
        verify_deployment.sys,
        "getwindowsversion",
        lambda: type("Win", (), {"build": 22631})(),
        raising=False,
    )
    monkeypatch.setattr(
        verify_deployment.importlib.metadata, "distributions", lambda: [Dist()]
    )
    result = verify_deployment.verify_runtime(
        tmp_path,
        expected_runtime_sha256=hashlib.sha256(trusted_runtime).hexdigest(),
        expected_lock_sha256=hashlib.sha256(trusted_lock).hexdigest(),
    )
    assert result["runtime_sha256"] == hashlib.sha256(trusted_runtime).hexdigest()
    assert result["lock_sha256"] == hashlib.sha256(trusted_lock).hexdigest()
    assert result["locked_packages"] == {"foo": "1.0"}


def test_offline_forwards_bundle_authority_and_rejects_release_swap_before_policy(
    monkeypatch, tmp_path
):
    digests = {
        "runtime": "1" * 64,
        "lock": "2" * 64,
        "common": "3" * 64,
        "release": "4" * 64,
        "policy": "5" * 64,
        "spec": "6" * 64,
        "bundle": "7" * 64,
    }
    files = {
        "deployment/runtime-win11.json": digests["runtime"],
        "deployment/requirements-win11.lock.txt": digests["lock"],
        "configs/common.yaml": digests["common"],
        "exported/release_manifest.json": digests["release"],
        "exported/policy.pt": digests["policy"],
        "exported/obs_spec.json": digests["spec"],
    }
    manifest = verify_deployment._VerifiedBundleManifest(
        {"schema_version": 2, "files": files},
        manifest_sha256=digests["bundle"],
        release_manifest={"files": {}},
        release_manifest_sha256=digests["release"],
    )
    common = verify_deployment.load_yaml(ROOT / "configs/common.yaml")
    calls: dict[str, object] = {"runner_count": 0, "release_digest": digests["release"]}

    def runtime(_root, **kwargs):
        calls["runtime"] = kwargs
        return {"verified": True}

    def yaml_snapshot(_path, **kwargs):
        calls["common"] = kwargs
        return common, RegularFileSnapshot(
            payload=b"", sha256=digests["common"], size=0
        )

    class VerifiedRelease(dict):
        def __init__(self, digest):
            super().__init__({
                "files": {"policy.pt": digests["policy"], "obs_spec.json": digests["spec"]}
            })
            self._verified_manifest_sha256 = digest

    def release(_root, **kwargs):
        calls["release"] = kwargs
        return VerifiedRelease(calls["release_digest"])

    class Runner:
        def __init__(self, *_args, **kwargs):
            calls["runner_count"] = int(calls["runner_count"]) + 1
            calls["runner"] = kwargs

        def act(self, _obs):
            return np.zeros((1, 7), dtype=np.float32)

    monkeypatch.setattr(verify_deployment, "verify_bundle_manifest", lambda _root: manifest)
    monkeypatch.setattr(verify_deployment, "verify_runtime", runtime)
    monkeypatch.setattr(verify_deployment, "load_yaml_snapshot", yaml_snapshot)
    monkeypatch.setattr(verify_deployment, "verify_release", release)
    monkeypatch.setattr(verify_deployment, "PolicyRunner", Runner)
    monkeypatch.setattr(verify_deployment, "obs_dim", lambda _cfg: 11)

    result = verify_deployment.verify_offline(tmp_path)
    assert calls["runtime"] == {
        "expected_runtime_sha256": digests["runtime"],
        "expected_lock_sha256": digests["lock"],
    }
    assert calls["common"]["expected_sha256"] == digests["common"]
    assert calls["release"]["expected_manifest_sha256"] == digests["release"]
    assert calls["runner"]["expected_policy_sha256"] == digests["policy"]
    assert calls["runner"]["expected_obs_spec_sha256"] == digests["spec"]
    assert result["authority"]["bundle_manifest_sha256"] == digests["bundle"]

    calls["release_digest"] = "8" * 64
    with pytest.raises(ContractError, match="differs from bundle authority"):
        verify_deployment.verify_offline(tmp_path)
    assert calls["runner_count"] == 1


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_staged_source_commit_binding_accepts_lf_and_crlf_clean_forms(
    monkeypatch, tmp_path, line_ending
):
    repo = tmp_path / "repo"
    tracked = repo / "common/example.py"
    tracked.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=repo, check=True)
    tracked.write_bytes(b"print('ok')\n")
    subprocess.run(["git", "add", "common/example.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    staging = tmp_path / ("stage-crlf" if line_ending == b"\r\n" else "stage-lf")
    staged = staging / "common/example.py"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"print('ok')" + line_ending)
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    monkeypatch.setattr(build_bundle, "ROOT", repo)
    build_bundle._verify_staged_sources_at_commit(
        staging, {"files": {"common/example.py": digest}}, commit
    )


def test_bundle_build_uses_sibling_staging_and_cleans_failure(monkeypatch, tmp_path):
    release = tmp_path / "exported"
    wheelhouse = tmp_path / "wheelhouse"
    release.mkdir()
    wheelhouse.mkdir()
    output = tmp_path / "dist" / "bundle"
    monkeypatch.setattr(build_bundle, "ROOT", tmp_path)
    monkeypatch.setattr(
        build_bundle,
        "verify_release",
        lambda *_args, **_kwargs: _VerifiedRelease(
            {"source": {"git_commit": "a" * 40}}, "1" * 64
        ),
    )
    monkeypatch.setattr(build_bundle, "require_clean_git_commit", lambda *_args: "a" * 40)
    monkeypatch.setattr(build_bundle, "_wheelhouse_record", lambda *_args: {"files": {}})
    monkeypatch.setattr(
        build_bundle,
        "_populate_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ContractError("injected")),
    )
    with pytest.raises(ContractError, match="injected"):
        build_bundle.build_bundle(release, output, wheelhouse)
    assert not output.exists()
    assert not list(output.parent.glob(".bundle.staging-*"))


def test_bundle_build_rejects_semantically_equal_release_with_different_bytes(
    monkeypatch, tmp_path
):
    release = tmp_path / "exported"
    wheelhouse = tmp_path / "wheelhouse"
    release.mkdir()
    wheelhouse.mkdir()
    output = tmp_path / "dist" / "bundle"
    trusted_digest = "1" * 64
    swapped_digest = "2" * 64
    calls: list[dict] = []

    def verify(*_args, **kwargs):
        calls.append(kwargs)
        digest = trusted_digest if len(calls) == 1 else swapped_digest
        return _VerifiedRelease(
            {"source": {"git_commit": "a" * 40}}, digest
        )

    def populate(_release, staging, *_args, **_kwargs):
        (staging / "exported").mkdir()
        return {"schema_version": 2, "files": {}}

    monkeypatch.setattr(build_bundle, "ROOT", tmp_path)
    monkeypatch.setattr(build_bundle, "verify_release", verify)
    monkeypatch.setattr(build_bundle, "require_clean_git_commit", lambda *_args: "a" * 40)
    monkeypatch.setattr(build_bundle, "_wheelhouse_record", lambda *_args: {"files": {}})
    monkeypatch.setattr(build_bundle, "_populate_bundle", populate)

    with pytest.raises(ContractError, match="staged release differs"):
        build_bundle.build_bundle(release, output, wheelhouse)
    assert calls[1]["expected_manifest_sha256"] == trusted_digest
    assert not output.exists()
    assert not list(output.parent.glob(".bundle.staging-*"))


def test_bundle_manifest_postpublish_swap_is_quarantined(monkeypatch, tmp_path):
    staging = tmp_path / ".bundle.staging-test"
    final = tmp_path / "bundle"
    exported = staging / "exported"
    exported.mkdir(parents=True)
    release_path = exported / "release_manifest.json"
    release_path.write_bytes(b"trusted-release")
    payload_path = staging / "payload.bin"
    payload_path.write_bytes(b"trusted-payload")
    release_digest = hashlib.sha256(release_path.read_bytes()).hexdigest()
    manifest = {
        "files": {
            "exported/release_manifest.json": release_digest,
            "payload.bin": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        }
    }
    manifest_payload = build_bundle._bundle_manifest_payload(manifest)
    (staging / "bundle_manifest.json").write_bytes(manifest_payload)
    manifest_digest = hashlib.sha256(manifest_payload).hexdigest()
    verified_release = _VerifiedRelease({"anchor": "release"}, release_digest)

    def verify_then_swap(path, **_kwargs):
        if Path(path).parent == final:
            # The first public manifest snapshot was trusted.  Mutate it while
            # the embedded release (B) is checked; the final manifest A read
            # must catch this rather than returning success.
            (final / "bundle_manifest.json").write_bytes(b"{}\n")
        return verified_release

    monkeypatch.setattr(
        build_bundle, "verify_release", verify_then_swap
    )
    with pytest.raises(ContractError, match="bundle manifest.*sha256 mismatch"):
        build_bundle._publish_verified_bundle(
            staging,
            final,
            expected_manifest=manifest,
            expected_manifest_sha256=manifest_digest,
            expected_release=verified_release,
            expected_release_manifest_sha256=release_digest,
        )
    assert not final.exists()
    quarantined = list(tmp_path.glob(".bundle.failed-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "bundle_manifest.json").read_bytes() == b"{}\n"


def test_bundle_rejects_release_from_different_git_revision(monkeypatch, tmp_path):
    release = tmp_path / "exported"
    release.mkdir()
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    monkeypatch.setattr(build_bundle, "ROOT", tmp_path)
    monkeypatch.setattr(build_bundle, "require_clean_git_commit", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        build_bundle,
        "verify_release",
        lambda *_args, **_kwargs: {"source": {"git_commit": "b" * 40}},
    )
    with pytest.raises(ContractError, match="HEAD differs"):
        build_bundle.build_bundle(release, tmp_path / "dist/bundle", wheelhouse)
