from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pytest

from common.runtime_authority import (
    RuntimeAuthorityError,
    RuntimeAuthoritySession,
    _canonical_mapping_sha256,
    _validate_manifest,
    activate_runtime_authority_child,
    runtime_authority_session_contract,
    validate_runtime_authority_execution_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _runtime_repo(
    root: Path, *, autocrlf: bool = False
) -> tuple[Path, Path, Path, Path]:
    common = root / "common"
    scripts = root / "scripts"
    configs = root / "configs"
    scratch = root / "scratch"
    for directory in (common, scripts, configs, scratch):
        directory.mkdir(parents=True)
    for name in ("runtime_authority.py", "file_snapshot.py"):
        shutil.copyfile(ROOT / "common" / name, common / name)
    (common / "__init__.py").write_text("", encoding="utf-8")
    marker = common / "marker.py"
    marker.write_text("VALUE = 'trusted-source'\n", encoding="utf-8")
    config = configs / "common.yaml"
    config.write_text("value: trusted-config\n", encoding="utf-8")
    runtime_assets = runtime_authority_session_contract()["tracked_runtime_assets"]
    for relative in runtime_assets:
        asset_path = root / relative
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(f"trusted-asset:{relative}\n".encode("utf-8"))
    asset = root / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.usd"
    optional_nurec = root / "assets" / "Scene" / "Workplace.usdz"
    optional_nurec.parent.mkdir(parents=True)
    optional_nurec.write_bytes(b"ignored-local-nurec")
    (root / ".gitignore").write_text("scratch/\nassets/Scene/\n", encoding="utf-8")
    child = scripts / "authority_child.py"
    child.write_text(
        """from pathlib import Path
import json
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from common.runtime_authority import (
    activate_runtime_authority_child,
    emit_runtime_authority_sentinel,
    runtime_authority_execution_receipt,
    runtime_yaml_snapshot,
)

activate_runtime_authority_child(root, allowed_purposes={"test-session"})
scratch = root / "scratch"
(scratch / "ready").write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 10.0
while not (scratch / "proceed").exists():
    if time.monotonic() >= deadline:
        raise SystemExit("parent did not release child")
    time.sleep(0.01)
snapshot = runtime_yaml_snapshot(root / "configs/common.yaml")
(scratch / "observed").write_bytes(snapshot.payload)
(scratch / "observed-asset").write_bytes(
    (root / "assets/ur5e_2fg7/ur5e_tcw1_2fg7.usd").read_bytes()
)
emit_runtime_authority_sentinel(
    "child", required_configs={"configs/common.yaml"}
)
receipt = runtime_authority_execution_receipt(
    "child", required_configs={"configs/common.yaml"}
)
(scratch / "execution-receipt.json").write_text(
    json.dumps(receipt, sort_keys=True), encoding="utf-8"
)
""",
        encoding="utf-8",
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "runtime-authority@example.invalid")
    _git(root, "config", "user.name", "Runtime Authority Test")
    if autocrlf:
        _git(root, "config", "core.autocrlf", "true")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "runtime authority fixture")
    return child, marker, config, asset


def _wait_for(path: Path, process: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"authorized child exited before readiness: {process.returncode}\n{stdout}\n{stderr}"
            )
        if time.monotonic() >= deadline:
            process.kill()
            raise AssertionError("timed out waiting for authorized child")
        time.sleep(0.01)


def test_child_consumes_frozen_yaml_and_windows_blocks_source_config_asset_swap_back(tmp_path):
    child, marker, config, asset = _runtime_repo(tmp_path)
    scratch = tmp_path / "scratch"
    with RuntimeAuthoritySession(tmp_path, purpose="test-session") as session:
        process = subprocess.Popen(
            [sys.executable, "-u", str(child)],
            cwd=tmp_path,
            env=session.child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        _wait_for(scratch / "ready", process)

        blocked: list[bool] = []
        blocked_errors: list[OSError | None] = []
        for target, attacker_payload in (
            (marker, b"VALUE = 'transient-source'\n"),
            (config, b"value: transient-config\n"),
            (asset, b"transient-runtime-asset"),
        ):
            attacker = scratch / f"{target.name}.attacker"
            attacker.write_bytes(attacker_payload)
            try:
                os.replace(attacker, target)
            except OSError as exc:
                blocked.append(True)
                blocked_errors.append(exc)
            else:
                blocked.append(False)
                blocked_errors.append(None)
                # POSIX fallback has no deny-share primitive.  Restore A before
                # releasing the child; the in-memory YAML snapshot must still
                # ensure that the child consumes A.
                if target == marker:
                    target.write_text("VALUE = 'trusted-source'\n", encoding="utf-8")
                elif target == config:
                    target.write_text("value: trusted-config\n", encoding="utf-8")
                else:
                    target.write_bytes(
                        b"trusted-asset:assets/ur5e_2fg7/ur5e_tcw1_2fg7.usd\n"
                    )

        (scratch / "proceed").write_text("go", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, (stdout, stderr)
        binding = session.verify_child_sentinel("child")
        assert binding["purpose"] == "test-session"
        receipt = json.loads(
            (scratch / "execution-receipt.json").read_text(encoding="utf-8")
        )
        validated = validate_runtime_authority_execution_receipt(
            receipt,
            expected_stage="child",
            expected_purpose="test-session",
            expected_git_commit=session.commit,
            expected_configs={"configs/common.yaml": binding["canonical_yaml"]["configs/common.yaml"]},
            required_configs={"configs/common.yaml"},
            project_root=tmp_path,
        )
        assert validated["binding"] == binding
        assert (scratch / "observed").read_text(encoding="utf-8") == "value: trusted-config\n"
        assert (scratch / "observed-asset").read_bytes() == (
            b"trusted-asset:assets/ur5e_2fg7/ur5e_tcw1_2fg7.usd\n"
        )
        manifest_assets = {
            relative
            for relative in validated["runtime_manifest"]["runtime_files"]
            if relative.startswith("assets/")
        }
        assert manifest_assets == set(
            runtime_authority_session_contract()["tracked_runtime_assets"]
        )
        if os.name == "nt":
            assert blocked == [True, True, True]
            assert isinstance(blocked_errors[2], PermissionError) or getattr(
                blocked_errors[2], "winerror", None
            ) in {5, 32}


def test_runtime_session_rejects_tracked_bytes_not_at_head(tmp_path):
    _child, marker, _config, _asset = _runtime_repo(tmp_path)
    marker.write_text("VALUE = 'dirty'\n", encoding="utf-8")
    with pytest.raises(RuntimeAuthorityError, match="clean Git worktree"):
        RuntimeAuthoritySession(tmp_path, purpose="test-session").__enter__()


def test_runtime_asset_manifest_rejects_rebound_missing_asset(tmp_path):
    _child, _marker, _config, _asset = _runtime_repo(tmp_path)
    with RuntimeAuthoritySession(tmp_path, purpose="test-session") as session:
        manifest = copy.deepcopy(session.manifest)
        manifest["runtime_files"].pop("assets/d435i/d435i.usd")
        manifest["runtime_files_sha256"] = _canonical_mapping_sha256(
            manifest["runtime_files"]
        )
        with pytest.raises(RuntimeAuthorityError, match="asset inventory differs"):
            _validate_manifest(manifest, root=tmp_path.resolve(), purpose="test-session")


def test_runtime_asset_clean_filter_projection_is_git_blob_exact(tmp_path):
    _child, _marker, _config, _asset = _runtime_repo(tmp_path, autocrlf=True)
    urdf = tmp_path / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf"
    clean_bytes = urdf.read_bytes()
    assert b"\r\n" not in clean_bytes
    worktree_bytes = clean_bytes.replace(b"\n", b"\r\n")
    urdf.write_bytes(worktree_bytes)
    # Refresh the temporary index stat cache.  The clean filter maps these
    # CRLF worktree bytes to the existing LF blob, so this stages no change.
    _git(tmp_path, "add", "--", urdf.relative_to(tmp_path).as_posix())
    assert subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=tmp_path, check=False
    ).returncode == 0
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", urdf.relative_to(tmp_path).as_posix()],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert status == b""

    with RuntimeAuthoritySession(tmp_path, purpose="test-session") as session:
        record = session.manifest["runtime_files"][
            "assets/ur5e_2fg7/ur5e_tcw1_2fg7.urdf"
        ]
        expected_blob = subprocess.run(
            ["git", "rev-parse", "HEAD:assets/ur5e_2fg7/ur5e_tcw1_2fg7.urdf"],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
            encoding="ascii",
        ).stdout.strip()
        assert record["git_blob"] == expected_blob
        assert record["sha256"] == hashlib.sha256(worktree_bytes).hexdigest()


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment not found: {name}")


def test_runtime_asset_allowlist_matches_task_preflight_and_urdf_mesh_closure():
    locked_assets = set(runtime_authority_session_contract()["tracked_runtime_assets"])
    assert len(locked_assets) == 30
    assert "assets/asset_manifest.json" in locked_assets

    manifest = json.loads((ROOT / "assets" / "asset_manifest.json").read_text(encoding="utf-8"))
    canonical_usd = {entry["path"] for entry in manifest["bundled_assets"]}
    preflight_usd = set(_literal_assignment(
        ROOT / "scripts" / "preflight_training.py", "EXPECTED_ASSET_PATHS"
    ))
    assert canonical_usd == preflight_usd
    assert canonical_usd <= locked_assets

    task_asset_sources = {
        ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
        / "mdp" / "ur5e_assets.py": "ur5e_tcw1_2fg7.usd",
        ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
        / "mdp" / "scene.py": "d435i.usd",
    }
    for source, asset_name in task_asset_sources.items():
        assert asset_name in source.read_text(encoding="utf-8")
    assert {
        "assets/ur5e_2fg7/ur5e_tcw1_2fg7.usd",
        "assets/d435i/d435i.usd",
        "assets/ur5e_2fg7/ur5e_tcw1_2fg7.urdf",
        "assets/d435i/meshes/d435.dae",
    } <= locked_assets

    urdf = ROOT / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf"
    mesh_closure = {
        (urdf.parent / mesh.attrib["filename"]).resolve().relative_to(ROOT).as_posix()
        for mesh in ET.parse(urdf).getroot().iter("mesh")
    }
    assert len(mesh_closure) == 19
    assert mesh_closure <= locked_assets
    assert "assets/d435i/D435i_Bracket.mtl" in locked_assets
    bracket_obj = (ROOT / "assets" / "d435i" / "D435i_Bracket.obj").read_text(
        encoding="utf-8"
    )
    assert "mtllib Part Studio 1 - qc_plate_base_stick_realsense.mtl" in bracket_obj
    assert all((ROOT / relative).is_file() for relative in locked_assets)

    tracked = set(subprocess.run(
        ["git", "ls-files", "assets"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines())
    assert locked_assets <= tracked
    assert "assets/Scene/Workplace.usdz" not in locked_assets
    assert not any(
        relative.startswith("assets/Workspace/")
        or relative.endswith((".png", ".jpg", ".jpeg"))
        or "/LICENSE" in relative
        or relative.endswith("/NOTICE")
        for relative in locked_assets
    )


def test_low_level_child_activation_requires_parent_manifest(monkeypatch, tmp_path):
    for key in (
        "UR5E_RUNTIME_AUTHORITY_MANIFEST",
        "UR5E_RUNTIME_AUTHORITY_MANIFEST_SHA256",
        "UR5E_RUNTIME_AUTHORITY_SENTINEL_DIR",
        "UR5E_RUNTIME_AUTHORITY_PURPOSE",
        "UR5E_RUNTIME_AUTHORITY_NONCE",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeAuthorityError, match="not launched"):
        activate_runtime_authority_child(tmp_path)


def test_scripts_package_is_regular_and_bound_to_repository_root():
    spec = importlib.util.find_spec("scripts")
    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).resolve() == (ROOT / "scripts" / "__init__.py").resolve()
    assert [Path(path).resolve() for path in spec.submodule_search_locations or ()] == [
        (ROOT / "scripts").resolve()
    ]
