#!/usr/bin/env python
"""Build a generated, hash-manifested deployment directory from canonical sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Building is a source-repository operation and must use the frozen isaac51
# interpreter even when bootstrap ran in a previous PowerShell process.  Do
# not re-exec when imported by tests or by the generated deployment bundle.
if __name__ == "__main__":
    from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

    reexec_in_training_python(__file__)
    configure_utf8_stdio()

from deployment.contracts import (  # noqa: E402
    ContractError,
    is_link_or_reparse,
    path_contains_link_or_reparse,
    sha256_file,
)
from common.file_snapshot import (  # noqa: E402
    FileSnapshotError,
    RegularFileSnapshot,
    read_regular_file_snapshot,
)
from deployment.release import (  # noqa: E402
    _directory_identity,
    _remove_owned_published_directory,
    _rename_directory_no_replace,
    require_clean_git_commit,
    verify_release,
)

CANONICAL_TREES = ("common", "real", "Phase4", "deployment")
CONFIG_FILES = ("common.yaml", "dr.yaml", "safety.yaml", "sim.yaml", "real.example.yaml")
BUNDLE_LEGAL_FILES = (
    "LICENSE",
    "assets/ur5e_2fg7/NOTICE",
    "assets/ur5e_2fg7/LICENSE-BSD-3-Clause.txt",
    "assets/ur5e_2fg7/LICENSE-URDF-Dataset-MIT.txt",
    "assets/ur5e_2fg7/LICENSE-OnRobot-MIT.txt",
    "assets/onrobot_2fg7/NOTICE",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "dist", "evidence", "wheelhouse"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".md"}


def _snapshot(path: Path, *, label: str, expected_sha256: str | None = None,
              expected_size: int | None = None) -> RegularFileSnapshot:
    """Capture one authority-bearing file and normalize snapshot errors."""
    if path_contains_link_or_reparse(path):
        raise ContractError(f"{label} contains a symlink/reparse path")
    try:
        return read_regular_file_snapshot(
            path,
            label=label,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
    except FileSnapshotError as exc:
        raise ContractError(str(exc)) from exc


def _json_snapshot(path: Path, *, label: str,
                   expected_sha256: str | None = None) -> tuple[object, RegularFileSnapshot]:
    """Parse JSON from the exact bytes whose digest is returned."""
    snapshot = _snapshot(path, label=label, expected_sha256=expected_sha256)
    try:
        value = json.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode {label}: {exc}") from exc
    return value, snapshot


def _verified_manifest_sha256(value: dict, *, label: str) -> str:
    """Return the private digest attached by an exact manifest verifier."""

    digest = getattr(value, "_verified_manifest_sha256", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ContractError(f"{label} has no exact verified manifest SHA-256")
    return digest


def _bundle_manifest_payload(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _verify_bundle_tree(
    root: Path,
    *,
    expected_manifest: dict,
    expected_manifest_sha256: str,
    expected_release: dict,
    expected_release_manifest_sha256: str,
) -> None:
    """Verify exact bundle and embedded-release bytes at one concrete path."""

    manifest, snapshot = _json_snapshot(
        root / "bundle_manifest.json",
        label="published bundle manifest",
        expected_sha256=expected_manifest_sha256,
    )
    if manifest != expected_manifest or snapshot.sha256 != expected_manifest_sha256:
        raise ContractError("bundle manifest differs from its publication authority")
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise ContractError("bundle publication manifest file inventory is invalid")
    entries = list(root.rglob("*"))
    if any(is_link_or_reparse(path) for path in entries):
        raise ContractError("bundle publication tree contains a symlink/reparse point")
    actual = {
        path.relative_to(root).as_posix()
        for path in entries
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    if actual != set(files):
        raise ContractError("bundle publication file set differs from its manifest")
    for relative, digest in sorted(files.items()):
        _snapshot(
            root / relative,
            label=f"bundle publication file {relative}",
            expected_sha256=digest,
        )
    if files.get("exported/release_manifest.json") \
            != expected_release_manifest_sha256:
        raise ContractError("bundle manifest differs from embedded release authority")
    embedded_release = verify_release(
        root / "exported",
        config_root=root,
        expected_manifest_sha256=expected_release_manifest_sha256,
    )
    if (
        _verified_manifest_sha256(embedded_release, label="bundle embedded release")
        != expected_release_manifest_sha256
        or embedded_release != expected_release
    ):
        raise ContractError("bundle embedded release differs from publication authority")
    final_manifest, final_snapshot = _json_snapshot(
        root / "bundle_manifest.json",
        label="rechecked bundle manifest",
        expected_sha256=expected_manifest_sha256,
    )
    if final_manifest != manifest or final_snapshot.sha256 != snapshot.sha256:
        raise ContractError("bundle manifest changed during tree verification")


def _publish_verified_bundle(
    staging: Path,
    final_output: Path,
    *,
    expected_manifest: dict,
    expected_manifest_sha256: str,
    expected_release: dict,
    expected_release_manifest_sha256: str,
) -> None:
    """Publish one exact bundle and retain rollback ownership through postverify."""

    identity = _directory_identity(staging, label="bundle staging")
    _verify_bundle_tree(
        staging,
        expected_manifest=expected_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_release=expected_release,
        expected_release_manifest_sha256=expected_release_manifest_sha256,
    )
    if _directory_identity(staging, label="bundle staging") != identity:
        raise ContractError("bundle staging directory changed during prepublication verify")
    if final_output.exists() or is_link_or_reparse(final_output):
        raise ContractError(f"output appeared during build (refusing overwrite): {final_output}")
    _rename_directory_no_replace(staging, final_output)
    try:
        if _directory_identity(final_output, label="published bundle") != identity:
            raise ContractError("published bundle is not the renamed staging directory")
        _verify_bundle_tree(
            final_output,
            expected_manifest=expected_manifest,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_release=expected_release,
            expected_release_manifest_sha256=expected_release_manifest_sha256,
        )
    except BaseException as exc:
        try:
            _remove_owned_published_directory(
                final_output, expected_identity=identity
            )
        except BaseException as cleanup_exc:
            raise ContractError(
                f"bundle postverification failed and owned cleanup failed: {cleanup_exc}"
            ) from exc
        raise


def _copy_regular_file(source: Path, destination: Path) -> None:
    if path_contains_link_or_reparse(source) or not source.is_file():
        raise ContractError(f"bundle source must be a regular non-symlink file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if path_contains_link_or_reparse(source) or not source.is_dir():
        raise ContractError(f"bundle source must be a regular non-symlink tree: {source}")
    for path in sorted(source.rglob("*")):
        if is_link_or_reparse(path):
            raise ContractError(f"bundle source tree contains a symlink/reparse path: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        target = destination / relative
        _copy_regular_file(path, target)


def _copy_bundle_legal_files(destination_root: Path) -> None:
    """Copy the project license and every notice/license applicable to bundled assets."""

    for relative in BUNDLE_LEGAL_FILES:
        _copy_regular_file(ROOT / relative, destination_root / relative)


def _wheelhouse_record(wheelhouse: Path) -> dict:
    if not wheelhouse.is_dir():
        raise ContractError(
            f"wheelhouse is missing: {wheelhouse}; build it with pip download --require-hashes"
        )
    entries = sorted(wheelhouse.iterdir())
    invalid = [
        path.name for path in entries
        if is_link_or_reparse(path) or not path.is_file() or path.suffix.lower() != ".whl"
    ]
    if invalid:
        raise ContractError(
            "wheelhouse may contain only regular .whl files; unexpected entries: "
            + ", ".join(invalid)
        )
    files = {}
    for path in entries:
        snapshot = _snapshot(path, label=f"wheelhouse distribution {path.name}")
        files[path.name] = {"bytes": snapshot.size, "sha256": snapshot.sha256}
    if not files:
        raise ContractError("wheelhouse contains no distributions")
    provenance_path = ROOT / "deployment/wheelhouse.manifest.json"
    provenance, provenance_snapshot = _json_snapshot(
        provenance_path, label="wheelhouse provenance manifest"
    )
    if not isinstance(provenance, dict):
        raise ContractError("wheelhouse provenance manifest must be an object")
    if provenance.get("schema_version") != 1 or provenance.get("files") != files:
        raise ContractError("wheelhouse file set/size/hash differs from wheelhouse.manifest.json")
    lock_snapshot = _snapshot(
        ROOT / "deployment/requirements-win11.lock.txt",
        label="wheelhouse dependency lock",
    )
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "target": provenance["target"],
        "method": provenance["method"],
        "provenance_manifest_sha256": provenance_snapshot.sha256,
        "lock_sha256": lock_snapshot.sha256,
        "aggregate_sha256": aggregate,
        "files": files,
    }


def build_bundle(release_dir: Path, output_dir: Path, wheelhouse: Path | None = None) -> dict:
    release_raw = Path(release_dir).absolute()
    output_raw = Path(output_dir).absolute()
    wheelhouse_raw = Path(wheelhouse or ROOT / "deployment/wheelhouse").absolute()
    for label, path in (("release", release_raw), ("wheelhouse", wheelhouse_raw)):
        if path_contains_link_or_reparse(path):
            raise ContractError(f"bundle {label} path must not contain symlinks/reparse points: {path}")
    if path_contains_link_or_reparse(output_raw):
        raise ContractError(f"bundle output path must not contain symlinks/reparse points: {output_raw}")
    release_dir = release_raw.resolve()
    output_dir = output_raw.resolve()
    final_output = output_dir
    if output_raw.exists():
        raise ContractError(f"output already exists (refusing overwrite): {output_dir}")
    wheelhouse = wheelhouse_raw.resolve()
    protected = [
        (ROOT / name).resolve() for name in CANONICAL_TREES
    ] + [
        (ROOT / "configs").resolve(), (ROOT / "assets").resolve(),
        (ROOT / "scripts").resolve(), release_dir, wheelhouse,
    ]
    if any(output_dir == item or item in output_dir.parents for item in protected):
        raise ContractError("bundle output may not be inside a source/release/wheelhouse tree")
    canonical_release = (ROOT / "exported").resolve()
    if release_dir != canonical_release or release_dir.parent != ROOT.resolve():
        raise ContractError(f"bundle release source must be exactly {canonical_release}")
    # No bundle can outlive or silently diverge from the exact clean source
    # revision that trained/exported the student policy.
    source_git_commit = require_clean_git_commit(ROOT)
    release_manifest = verify_release(release_dir, config_root=ROOT)
    release_git_commit = release_manifest.get("source", {}).get("git_commit")
    if release_git_commit != source_git_commit:
        raise ContractError(
            "clean bundle-builder HEAD differs from release source.git_commit: "
            f"head={source_git_commit}, release={release_git_commit!r}"
        )
    release_manifest_sha256 = _verified_manifest_sha256(
        release_manifest, label="source release"
    )
    wheelhouse_record = _wheelhouse_record(wheelhouse)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    )).absolute()
    try:
        if (
            is_link_or_reparse(staging)
            or staging.resolve().parent != output_dir.parent
            or not staging.name.startswith(f".{output_dir.name}.staging-")
        ):
            raise ContractError("bundle staging directory validation failed")
        manifest = _populate_bundle(
            release_dir,
            staging,
            wheelhouse,
            wheelhouse_record,
            source_git_commit=source_git_commit,
        )
        bundle_manifest_sha256 = hashlib.sha256(
            _bundle_manifest_payload(manifest)
        ).hexdigest()
        # Self-verify every file against the manifest before atomic publication.
        for relative, expected_hash in manifest["files"].items():
            try:
                _snapshot(
                    staging / relative,
                    label=f"staged bundle file {relative}",
                    expected_sha256=expected_hash,
                )
            except ContractError as exc:
                raise ContractError(f"staged bundle hash mismatch: {relative}") from exc
        # Close copy-time TOCTOU windows: the staged release must verify against
        # its copied configs, copied wheel bytes must still match the frozen
        # source record, and HEAD/source release must remain unchanged/clean.
        staged_release = verify_release(
            staging / "exported",
            config_root=staging,
            expected_manifest_sha256=release_manifest_sha256,
        )
        if (
            _verified_manifest_sha256(staged_release, label="staged release")
            != release_manifest_sha256
            or staged_release != release_manifest
        ):
            raise ContractError("staged release differs from the frozen source release")
        staged_wheelhouse = _wheelhouse_record_from_bundle(
            staging / "deployment/wheelhouse", wheelhouse_record
        )
        if staged_wheelhouse != wheelhouse_record:
            raise ContractError("staged wheelhouse differs from the frozen source record")
        if _wheelhouse_record(wheelhouse) != wheelhouse_record:
            raise ContractError("source wheelhouse changed while bundle was built")
        current_release = verify_release(
            release_dir,
            config_root=ROOT,
            expected_manifest_sha256=release_manifest_sha256,
        )
        if (
            _verified_manifest_sha256(current_release, label="rechecked source release")
            != release_manifest_sha256
            or current_release != release_manifest
        ):
            raise ContractError("source release changed while bundle was built")
        _verify_staged_sources_at_commit(staging, manifest, source_git_commit)
        if require_clean_git_commit(ROOT) != source_git_commit:
            raise ContractError("clean Git HEAD changed while bundle was built")
        _publish_verified_bundle(
            staging,
            final_output,
            expected_manifest=manifest,
            expected_manifest_sha256=bundle_manifest_sha256,
            expected_release=release_manifest,
            expected_release_manifest_sha256=release_manifest_sha256,
        )
        return manifest
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        try:
            if staging.exists() and staging.parent == final_output.parent \
                    and staging.name.startswith(f".{final_output.name}.staging-"):
                shutil.rmtree(staging)
        except BaseException as cleanup_exc:
            cleanup_error = cleanup_exc
        if cleanup_error is not None:
            raise ContractError(
                f"bundle publication failed and owned cleanup failed: {cleanup_error}"
            ) from exc
        raise


def _wheelhouse_record_from_bundle(wheelhouse: Path, expected: dict) -> dict:
    """Recompute staged wheel bytes without consulting source-side provenance paths."""
    entries = sorted(wheelhouse.iterdir()) if wheelhouse.is_dir() else []
    if any(is_link_or_reparse(path) or not path.is_file() for path in entries):
        raise ContractError("staged wheelhouse contains a non-regular entry")
    files = {}
    for path in entries:
        expected_file = expected.get("files", {}).get(path.name, {})
        snapshot = _snapshot(
            path,
            label=f"staged wheelhouse distribution {path.name}",
            expected_sha256=expected_file.get("sha256"),
            expected_size=expected_file.get("bytes"),
        )
        files[path.name] = {"bytes": snapshot.size, "sha256": snapshot.sha256}
    if files != expected.get("files"):
        raise ContractError("staged wheelhouse file set/size/hash mismatch")
    deployment_dir = wheelhouse.parent
    provenance_path = deployment_dir / "wheelhouse.manifest.json"
    lock_path = deployment_dir / "requirements-win11.lock.txt"
    if not provenance_path.is_file() or not lock_path.is_file():
        raise ContractError("staged wheelhouse provenance/lock differs from frozen record")
    provenance, provenance_snapshot = _json_snapshot(
        provenance_path,
        label="staged wheelhouse provenance manifest",
        expected_sha256=expected.get("provenance_manifest_sha256"),
    )
    lock_snapshot = _snapshot(
        lock_path,
        label="staged wheelhouse dependency lock",
        expected_sha256=expected.get("lock_sha256"),
    )
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != 1
        or provenance.get("files") != files
        or provenance.get("target") != expected.get("target")
        or provenance.get("method") != expected.get("method")
    ):
        raise ContractError("staged wheelhouse provenance/lock differs from frozen record")
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "target": expected["target"],
        "method": expected["method"],
        "provenance_manifest_sha256": provenance_snapshot.sha256,
        "lock_sha256": lock_snapshot.sha256,
        "aggregate_sha256": aggregate,
        "files": files,
    }


def _verify_staged_sources_at_commit(staging: Path, manifest: dict, commit: str) -> None:
    """Bind every copied source byte directly to the claimed Git tree, not the worktree."""
    skipped_prefixes = ("exported/", "deployment/wheelhouse/")
    for relative, staged_hash in sorted(manifest.get("files", {}).items()):
        if relative == "RUNBOOK.txt" or relative.startswith(skipped_prefixes):
            continue
        staged_path = staging / relative
        try:
            # The Git clean-filter object ID and the bundle-manifest SHA are
            # both derived from this one opened payload.  Never hash the path
            # again after feeding a potentially different payload to Git.
            snapshot = _snapshot(
                staged_path,
                label=f"staged source {relative}",
                expected_sha256=staged_hash,
            )
            # Compare Git object identities after applying the repository's
            # *clean* filter to copied bytes.  This accepts both an LF worktree
            # left by apply_patch and a CRLF checkout under core.autocrlf=true,
            # while still binding the exact logical content to ``commit``.
            staged_oid = subprocess.check_output(
                ["git", "hash-object", f"--path={relative}", "--stdin"],
                input=snapshot.payload,
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                timeout=20,
                text=False,
            ).decode("ascii").strip()
            committed_oid = subprocess.check_output(
                ["git", "rev-parse", f"{commit}:{relative}"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                timeout=20,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ContractError(
                f"bundled source is absent from claimed Git commit: {relative}"
            ) from exc
        if staged_oid != committed_oid or snapshot.sha256 != staged_hash:
            raise ContractError(f"staged source differs from Git commit: {relative}")


def _populate_bundle(
    release_dir: Path,
    output_dir: Path,
    wheelhouse: Path,
    wheelhouse_record: dict,
    *,
    source_git_commit: str,
) -> dict:
    for name in CANONICAL_TREES:
        _copy_tree(ROOT / name, output_dir / name)
    config_dst = output_dir / "configs"
    config_dst.mkdir()
    for name in CONFIG_FILES:
        _copy_regular_file(ROOT / "configs" / name, config_dst / name)
    _copy_tree(ROOT / "configs" / "calib", config_dst / "calib")
    _copy_bundle_legal_files(output_dir)
    asset = ROOT / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf"
    asset_dst = output_dir / "assets" / "ur5e_2fg7" / asset.name
    _copy_regular_file(asset, asset_dst)
    fk = ROOT / "scripts" / "tools" / "urdf_fk.py"
    fk_dst = output_dir / "scripts" / "tools" / "urdf_fk.py"
    _copy_regular_file(fk, fk_dst)
    _copy_tree(release_dir, output_dir / "exported")
    wheelhouse_dst = output_dir / "deployment" / "wheelhouse"
    wheelhouse_dst.mkdir(parents=True, exist_ok=True)
    # Copy only the exact set whose names/sizes/hashes matched provenance above.
    for filename in sorted(wheelhouse_record["files"]):
        _copy_regular_file(wheelhouse / filename, wheelhouse_dst / filename)
    runbook = output_dir / "RUNBOOK.txt"
    runbook.write_text(
        "UR5e deployment bundle (generated; do not edit in place)\n"
        "1. Install CPython/runtime exactly as deployment/runtime-win11.json.\n"
        "2. python -m pip install --no-index --find-links deployment/wheelhouse "
        "--require-hashes -r deployment/requirements-win11.lock.txt\n"
        "3. Copy configs/real.example.yaml to configs/real.local.yaml (the sole allowed "
        "unmanifested local config).\n"
        "4. Run deployment/verify_deployment.py --mode offline --bundle-root .\n"
        "5. Run deployment/verify_deployment.py --mode no-motion --bundle-root . --real-config "
        "configs/real.local.yaml --duration-s 600 (receive-only diagnostic; >=10,000 "
        "cycles; expected exit 2/status=timestamp_sync_unverified; it never issues "
        "kind=no_motion evidence).\n"
        "6. This release authorizes no motion; command-capable commissioning modes remain "
        "disabled even if evidence is present.\n",
        encoding="utf-8",
    )
    files = {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    manifest = {
        "schema_version": 2,
        "source_git_commit": source_git_commit,
        "files": files,
        "wheelhouse": wheelhouse_record,
    }
    (output_dir / "bundle_manifest.json").write_bytes(
        _bundle_manifest_payload(manifest)
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wheelhouse", type=Path, default=ROOT / "deployment" / "wheelhouse")
    args = parser.parse_args()
    try:
        manifest = build_bundle(args.release_dir, args.output, args.wheelhouse)
    except ContractError as exc:
        print(f"[bundle][error] {exc}")
        return 1
    print(f"[bundle] wrote {args.output.resolve()} ({len(manifest['files'])} hashed files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
