"""Frozen tracked-runtime authority for long-running simulation processes.

The ordinary start/end Git checks cannot detect a swap-back attack where a
child imports transient bytes and the pathname is restored before the parent
seals its receipt.  This module closes that window on Windows by keeping every
tracked runtime source/config file and the exact task asset closure open with
write/delete sharing denied for the complete child-process lifetime.

Public entry points use :func:`authorize_entrypoint` before importing Isaac or
project task modules.  The outer bootstrap owns the locks and launches a fresh
inner interpreter with an exclusive pycache and an immutable manifest.  The
inner process serves canonical YAML from the manifest bytes instead of opening
the mutable pathname and publishes a parent-verified consumption sentinel.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import BinaryIO, Iterable, Mapping

from common.file_snapshot import (
    FileSnapshotError,
    RegularFileSnapshot,
    read_regular_file_snapshot,
)


RUNTIME_AUTHORITY_SCHEMA_VERSION = 1
RUNTIME_AUTHORITY_RECORD_TYPE = "ur5e_tracked_runtime_authority"
RUNTIME_AUTHORITY_SENTINEL_TYPE = "ur5e_runtime_authority_child"
RUNTIME_AUTHORITY_EXECUTION_TYPE = "ur5e_runtime_authority_execution"

ENV_MANIFEST = "UR5E_RUNTIME_AUTHORITY_MANIFEST"
ENV_MANIFEST_SHA256 = "UR5E_RUNTIME_AUTHORITY_MANIFEST_SHA256"
ENV_SENTINEL_DIR = "UR5E_RUNTIME_AUTHORITY_SENTINEL_DIR"
ENV_PURPOSE = "UR5E_RUNTIME_AUTHORITY_PURPOSE"
ENV_NONCE = "UR5E_RUNTIME_AUTHORITY_NONCE"

_RUNTIME_PREFIXES = (
    "Phase1/",
    "Phase2/",
    "Phase3/",
    "common/",
    "configs/",
    "deployment/",
    "real/",
    "sim/",
    "scripts/",
    "source/ur5e_binpick/",
)

# Tracked assets opened by the standard Teacher/Vision task tree or consumed by
# its mandatory preflight.  Keep this an exact allowlist rather than accepting
# ``assets/`` wholesale: licenses, source images, retired workspace layers, and
# the ignored NuRec scan are not standard runtime authority.  In particular,
# ``assets/Scene/Workplace.usdz`` has its own sealed NuRec provenance and must
# not silently become a prerequisite for Teacher/Vision runs.
_CANONICAL_RUNTIME_USD_PATHS = frozenset({
    "assets/ur5e_2fg7/ur5e_tcw1_2fg7.usd",
    "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_base.usd",
    "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_physics.usd",
    "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_robot.usd",
    "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_sensor.usd",
    "assets/d435i/d435i.usd",
    "assets/d435i/Props/instanceable_meshes.usd",
})
_RUNTIME_URDF_MESH_PATHS = frozenset({
    # The bracket OBJ retains its exporter-era ``mtllib Part Studio ...``
    # token while the repository carries the equivalent material under this
    # canonical name and the URDF supplies its own material override.  The
    # converted runtime USD is self-contained, but lock the tracked MTL with
    # the source mesh closure so neither source representation can drift.
    "assets/d435i/D435i_Bracket.mtl",
    "assets/d435i/D435i_Bracket.obj",
    "assets/magbot_tcw1/magbot_tcw1_visual.obj",
    "assets/ur5e_2fg7/meshes/onrobot_2fg7/base_link.stl",
    "assets/ur5e_2fg7/meshes/onrobot_2fg7/left_finger_link.stl",
    "assets/ur5e_2fg7/meshes/onrobot_2fg7/right_finger_link.stl",
    "assets/ur5e_2fg7/meshes/ur5e/collision/base.stl",
    "assets/ur5e_2fg7/meshes/ur5e/collision/forearm.stl",
    "assets/ur5e_2fg7/meshes/ur5e/collision/shoulder.stl",
    "assets/ur5e_2fg7/meshes/ur5e/collision/upperarm.stl",
    "assets/ur5e_2fg7/meshes/ur5e/collision/wrist1.stl",
    "assets/ur5e_2fg7/meshes/ur5e/collision/wrist2.stl",
    "assets/ur5e_2fg7/meshes/ur5e/collision/wrist3.stl",
    "assets/ur5e_2fg7/meshes/ur5e/visual/base.dae",
    "assets/ur5e_2fg7/meshes/ur5e/visual/forearm.dae",
    "assets/ur5e_2fg7/meshes/ur5e/visual/shoulder.dae",
    "assets/ur5e_2fg7/meshes/ur5e/visual/upperarm.dae",
    "assets/ur5e_2fg7/meshes/ur5e/visual/wrist1.dae",
    "assets/ur5e_2fg7/meshes/ur5e/visual/wrist2.dae",
    "assets/ur5e_2fg7/meshes/ur5e/visual/wrist3.dae",
})
_RUNTIME_ASSET_PATHS = frozenset({
    "assets/asset_manifest.json",
    "assets/ur5e_2fg7/ur5e_tcw1_2fg7.urdf",
    "assets/d435i/meshes/d435.dae",
    *_CANONICAL_RUNTIME_USD_PATHS,
    *_RUNTIME_URDF_MESH_PATHS,
})
_PROJECT_IMPORT_PREFIXES = ("common", "deployment", "real", "sim", "scripts", "ur5e_binpick")
_PROJECT_IMPORT_EXACT = {"cli_args"}
_STAGE_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_HEX40_RE = re.compile(r"[0-9a-f]{40}")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")


def runtime_authority_session_contract() -> dict:
    """Return the exact source/import/session boundary asserted by a receipt."""
    return {
        "clean_head": "required-before-lock-and-after-child",
        "git_byte_projection": "hash-object --path=<tracked-path> --stdin",
        "tracked_runtime_prefixes": list(_RUNTIME_PREFIXES),
        "tracked_runtime_assets": sorted(_RUNTIME_ASSET_PATHS),
        "optional_nurec_assets": "separate-sealed-provenance-excluded-from-default-lock",
        "untracked_and_ignored_outputs": "excluded",
        "windows_file_lock": "CreateFileW-FILE_SHARE_READ-deny-write-delete",
        "windows_directory_lock": "CreateFileW-share-read-write-deny-delete-rename",
        "lock_lifetime": "outer-parent-through-child-process-exit",
        "python_cache": "fresh-exclusive-PYTHONPYCACHEPREFIX",
        "project_import_origins": "tracked-runtime-files-under-locked-project-root",
        "canonical_yaml": "embedded-parent-runtime-manifest-bytes",
    }


class RuntimeAuthorityError(RuntimeError):
    """Tracked runtime bytes could not be frozen or proven."""


def _read_authority_snapshot(
    path: Path | str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> RegularFileSnapshot:
    """Keep filesystem failures inside the public authority error contract."""
    try:
        return read_regular_file_snapshot(
            path,
            label=label,
            expected_sha256=expected_sha256,
        )
    except FileSnapshotError as exc:
        raise RuntimeAuthorityError(str(exc)) from exc


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _canonical_mapping_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _path_contains_link_or_reparse(path: Path) -> bool:
    """Reject symlinks/junctions in an existing lexical path."""
    absolute = path.absolute()
    parts = absolute.parts
    if not parts:
        return True
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            continue
        try:
            info = os.lstat(current)
        except OSError:
            return True
        attributes = int(getattr(info, "st_file_attributes", 0))
        if stat.S_ISLNK(info.st_mode) or attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            return True
    return False


def _run_git(
    root: Path,
    arguments: Iterable[str],
    *,
    input_bytes: bytes | None = None,
) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeAuthorityError(f"Git runtime-authority query failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeAuthorityError(
            f"Git runtime-authority query failed ({' '.join(arguments)}): {detail}"
        )
    return completed.stdout


def _git_head(root: Path) -> str:
    value = _run_git(root, ("rev-parse", "--verify", "HEAD")).decode("ascii").strip()
    if _HEX40_RE.fullmatch(value) is None:
        raise RuntimeAuthorityError("runtime authority requires an exact Git HEAD")
    return value


def _require_clean_worktree(root: Path) -> None:
    status = _run_git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"),
    )
    if status:
        first = status.decode("utf-8", errors="replace").splitlines()[0]
        raise RuntimeAuthorityError(f"runtime authority requires a clean Git worktree: {first}")


def _runtime_tree(root: Path, commit: str) -> list[tuple[str, str]]:
    raw = _run_git(root, ("ls-tree", "-r", "-z", "--full-tree", commit))
    rows: list[tuple[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split()
            relative = encoded_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RuntimeAuthorityError("Git runtime tree contains an invalid record") from exc
        if kind != "blob" or mode not in {"100644", "100755"}:
            continue
        if any(relative.startswith(prefix) for prefix in _RUNTIME_PREFIXES) \
                or relative in _RUNTIME_ASSET_PATHS:
            if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
                raise RuntimeAuthorityError("Git runtime tree contains an unsafe path")
            rows.append((relative, oid))
    if not rows:
        raise RuntimeAuthorityError("tracked runtime allowlist is empty")
    present_assets = {relative for relative, _oid in rows if relative.startswith("assets/")}
    if present_assets != _RUNTIME_ASSET_PATHS:
        raise RuntimeAuthorityError(
            "tracked runtime asset allowlist differs from clean HEAD: "
            f"missing={sorted(_RUNTIME_ASSET_PATHS - present_assets)} "
            f"extra={sorted(present_assets - _RUNTIME_ASSET_PATHS)}"
        )
    return sorted(rows)


def _hash_git_projection(root: Path, relative: str, payload: bytes) -> str:
    value = _run_git(
        root,
        ("hash-object", f"--path={relative}", "--stdin"),
        input_bytes=payload,
    ).decode("ascii").strip()
    if _HEX40_RE.fullmatch(value) is None:
        raise RuntimeAuthorityError(f"invalid Git projection for tracked runtime file: {relative}")
    return value


@dataclass
class _LockedFile:
    path: Path
    stream: BinaryIO
    payload: bytes

    def close(self) -> None:
        self.stream.close()


def _open_locked_file(path: Path) -> _LockedFile:
    """Open a regular file while denying concurrent write/delete on Windows."""
    if _path_contains_link_or_reparse(path):
        raise RuntimeAuthorityError(f"tracked runtime path traverses a link/reparse point: {path}")
    if os.name != "nt":
        stream = path.open("rb")
    else:
        import ctypes
        from ctypes import wintypes
        import msvcrt

        create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(path),
            0x80000000,  # GENERIC_READ
            0x00000001,  # FILE_SHARE_READ: explicitly deny write/delete sharing
            None,
            3,  # OPEN_EXISTING
            0x00200000 | 0x08000000,  # OPEN_REPARSE_POINT | SEQUENTIAL_SCAN
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle == invalid:
            raise RuntimeAuthorityError(
                f"cannot deny write/delete access to tracked runtime file {path}: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        try:
            fd = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
        except BaseException:
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
            raise
        stream = os.fdopen(fd, "rb", closefd=True)
    try:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeAuthorityError(f"tracked runtime input is not regular: {path}")
        payload = stream.read()
        stream.seek(0)
        after = os.fstat(stream.fileno())
        if (
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(payload) != opened.st_size
        ):
            raise RuntimeAuthorityError(f"tracked runtime input changed while locking: {path}")
    except BaseException:
        stream.close()
        raise
    return _LockedFile(path=path, stream=stream, payload=payload)


def _open_locked_directory(path: Path):
    """Hold a directory against rename/delete while allowing output creation."""
    if _path_contains_link_or_reparse(path) or not path.is_dir():
        raise RuntimeAuthorityError(f"runtime directory is missing or unsafe: {path}")
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x00000080,  # FILE_READ_ATTRIBUTES
        0x00000001 | 0x00000002,  # share read/write, deny delete/rename
        None,
        3,
        0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise RuntimeAuthorityError(
            f"cannot deny delete/rename access to runtime directory {path}: "
            f"{ctypes.WinError(ctypes.get_last_error())}"
        )
    return handle


def _close_locked_directory(handle) -> None:
    if handle is None or os.name != "nt":
        return
    import ctypes

    ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)


@dataclass
class _ActiveAuthority:
    root: Path
    purpose: str
    manifest: dict
    manifest_sha256: str
    consumed_yaml: set[str] = field(default_factory=set)
    import_guard: object | None = None


_ACTIVE: _ActiveAuthority | None = None


def _validate_manifest(value: object, *, root: Path, purpose: str) -> dict:
    keys = {
        "schema_version", "record_type", "project_root", "purpose", "git_commit",
        "runtime_files", "runtime_files_sha256", "canonical_yaml",
        "canonical_yaml_sha256", "nonce",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeAuthorityError("runtime authority manifest has the wrong schema")
    if value.get("schema_version") != RUNTIME_AUTHORITY_SCHEMA_VERSION \
            or value.get("record_type") != RUNTIME_AUTHORITY_RECORD_TYPE:
        raise RuntimeAuthorityError("runtime authority manifest type/version mismatch")
    if Path(str(value.get("project_root", ""))).resolve() != root \
            or value.get("purpose") != purpose:
        raise RuntimeAuthorityError("runtime authority root/purpose mismatch")
    if _HEX40_RE.fullmatch(str(value.get("git_commit", ""))) is None \
            or _HEX64_RE.fullmatch(str(value.get("runtime_files_sha256", ""))) is None \
            or _HEX64_RE.fullmatch(str(value.get("canonical_yaml_sha256", ""))) is None \
            or _HEX64_RE.fullmatch(str(value.get("nonce", ""))) is None:
        raise RuntimeAuthorityError("runtime authority manifest contains a malformed digest")
    files = value.get("runtime_files")
    yaml_files = value.get("canonical_yaml")
    if not isinstance(files, dict) or not files or not isinstance(yaml_files, dict) or not yaml_files:
        raise RuntimeAuthorityError("runtime authority file inventories are empty")
    if _canonical_mapping_sha256(files) != value["runtime_files_sha256"] \
            or _canonical_mapping_sha256(yaml_files) != value["canonical_yaml_sha256"]:
        raise RuntimeAuthorityError("runtime authority inventory aggregate mismatch")
    for relative, record in files.items():
        pure = PurePosixPath(str(relative))
        if pure.is_absolute() or ".." in pure.parts or not isinstance(record, dict) \
                or not (
                    any(str(relative).startswith(prefix) for prefix in _RUNTIME_PREFIXES)
                    or str(relative) in _RUNTIME_ASSET_PATHS
                ) \
                or set(record) != {"git_blob", "sha256", "size"} \
                or _HEX40_RE.fullmatch(str(record.get("git_blob", ""))) is None \
                or _HEX64_RE.fullmatch(str(record.get("sha256", ""))) is None \
                or isinstance(record.get("size"), bool) \
                or not isinstance(record.get("size"), int) or record["size"] < 0:
            raise RuntimeAuthorityError("runtime authority file inventory is malformed")
    asset_files = {str(relative) for relative in files if str(relative).startswith("assets/")}
    if asset_files != _RUNTIME_ASSET_PATHS:
        raise RuntimeAuthorityError(
            "runtime authority asset inventory differs from the required runtime closure"
        )
    expected_yaml = {
        relative
        for relative in files
        if relative.startswith("configs/") and relative.endswith((".yaml", ".yml"))
    }
    if set(yaml_files) != expected_yaml:
        raise RuntimeAuthorityError("runtime authority YAML inventory differs from tracked configs")
    for relative, record in yaml_files.items():
        if relative not in files or not relative.startswith("configs/") \
                or not relative.endswith((".yaml", ".yml")) \
                or not isinstance(record, dict) \
                or set(record) != {"payload_b64", "sha256", "size"}:
            raise RuntimeAuthorityError("runtime authority YAML inventory is malformed")
        try:
            payload = base64.b64decode(record["payload_b64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise RuntimeAuthorityError("runtime authority YAML payload is invalid") from exc
        if len(payload) != record["size"] \
                or hashlib.sha256(payload).hexdigest() != record["sha256"] \
                or files[relative]["sha256"] != record["sha256"]:
            raise RuntimeAuthorityError("runtime authority YAML payload/hash mismatch")
    return value


class _ProjectImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self, active: _ActiveAuthority) -> None:
        self.active = active

    @staticmethod
    def _is_project_name(fullname: str) -> bool:
        top = fullname.split(".", 1)[0]
        return top in _PROJECT_IMPORT_PREFIXES or fullname in _PROJECT_IMPORT_EXACT

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if not self._is_project_name(fullname):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None:
            return None
        _validate_module_spec(fullname, spec, self.active)
        # Returning the checked PathFinder spec prevents a later finder from
        # substituting a different origin after this guard approved the name.
        return spec


def _relative_origin(origin: str, root: Path) -> str:
    try:
        path = Path(origin).absolute().resolve()
        return path.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise RuntimeAuthorityError(f"project import escaped the frozen root: {origin}") from exc


def _validate_module_spec(fullname: str, spec, active: _ActiveAuthority) -> None:  # noqa: ANN001
    origin = getattr(spec, "origin", None)
    if origin in (None, "namespace"):
        locations = list(getattr(spec, "submodule_search_locations", ()) or ())
        if not locations:
            raise RuntimeAuthorityError(f"project namespace import has no origin: {fullname}")
        for location in locations:
            try:
                Path(location).absolute().resolve().relative_to(active.root)
            except (OSError, ValueError) as exc:
                raise RuntimeAuthorityError(
                    f"project namespace import escaped the frozen root: {fullname}"
                ) from exc
        return
    if origin in {"built-in", "frozen"}:
        raise RuntimeAuthorityError(f"project import resolved to a non-file origin: {fullname}")
    relative = _relative_origin(origin, active.root)
    if relative not in active.manifest["runtime_files"]:
        raise RuntimeAuthorityError(
            f"project import origin is not in the frozen tracked allowlist: {fullname} -> {relative}"
        )


def verify_project_import_origins() -> None:
    active = _ACTIVE
    if active is None:
        raise RuntimeAuthorityError("runtime authority is not active")
    for name, module in sorted(sys.modules.items()):
        if module is None or not _ProjectImportGuard._is_project_name(name):
            continue
        spec = getattr(module, "__spec__", None)
        if spec is None:
            # Entry scripts execute as __main__; project modules must have specs.
            if name != "__main__":
                raise RuntimeAuthorityError(f"loaded project module has no import spec: {name}")
            continue
        _validate_module_spec(name, spec, active)


def _activate(value: dict, *, root: Path, purpose: str, manifest_sha256: str) -> dict:
    global _ACTIVE
    if _ACTIVE is not None:
        if _ACTIVE.manifest_sha256 != manifest_sha256 or _ACTIVE.purpose != purpose:
            raise RuntimeAuthorityError("a different runtime authority is already active")
        return runtime_authority_binding()
    manifest = _validate_manifest(value, root=root, purpose=purpose)
    active = _ActiveAuthority(
        root=root,
        purpose=purpose,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    guard = _ProjectImportGuard(active)
    active.import_guard = guard
    _ACTIVE = active
    sys.meta_path.insert(0, guard)
    verify_project_import_origins()
    return runtime_authority_binding()


def _deactivate(expected_sha256: str) -> None:
    global _ACTIVE
    active = _ACTIVE
    if active is None:
        return
    if active.manifest_sha256 != expected_sha256:
        raise RuntimeAuthorityError("runtime authority deactivation identity mismatch")
    if active.import_guard in sys.meta_path:
        sys.meta_path.remove(active.import_guard)
    _ACTIVE = None


def runtime_yaml_snapshot(path: Path | str) -> RegularFileSnapshot | None:
    """Return frozen YAML bytes for a canonical config path when active."""
    active = _ACTIVE
    if active is None:
        return None
    source = Path(path).expanduser().absolute()
    try:
        relative = source.resolve().relative_to(active.root).as_posix()
    except (OSError, ValueError):
        return None
    if not relative.startswith("configs/"):
        return None
    record = active.manifest["canonical_yaml"].get(relative)
    if record is None:
        raise RuntimeAuthorityError(
            f"canonical config is absent from the frozen runtime manifest: {relative}"
        )
    payload = base64.b64decode(record["payload_b64"], validate=True)
    active.consumed_yaml.add(relative)
    return RegularFileSnapshot(payload=payload, sha256=record["sha256"], size=record["size"])


def runtime_authority_binding() -> dict:
    active = _ACTIVE
    if active is None:
        raise RuntimeAuthorityError("runtime authority is not active")
    return _binding_from_manifest(
        active.manifest,
        purpose=active.purpose,
        manifest_sha256=active.manifest_sha256,
    )


def _binding_from_manifest(
    manifest: Mapping[str, object],
    *,
    purpose: str,
    manifest_sha256: str,
) -> dict:
    yaml_hashes = {
        relative: record["sha256"]
        for relative, record in sorted(manifest["canonical_yaml"].items())
    }
    return {
        "schema_version": RUNTIME_AUTHORITY_SCHEMA_VERSION,
        "record_type": RUNTIME_AUTHORITY_RECORD_TYPE,
        "purpose": purpose,
        "manifest_sha256": manifest_sha256,
        "git_commit": manifest["git_commit"],
        "runtime_file_count": len(manifest["runtime_files"]),
        "runtime_files_sha256": manifest["runtime_files_sha256"],
        "canonical_yaml_sha256": manifest["canonical_yaml_sha256"],
        "canonical_yaml": yaml_hashes,
    }


def validate_runtime_authority_binding(
    value: object,
    *,
    expected_git_commit: str | None = None,
    expected_configs: Mapping[str, str] | None = None,
    expected_purpose: str | None = None,
) -> dict:
    keys = {
        "schema_version", "record_type", "purpose", "manifest_sha256", "git_commit",
        "runtime_file_count", "runtime_files_sha256", "canonical_yaml_sha256",
        "canonical_yaml",
    }
    if not isinstance(value, dict) or set(value) != keys \
            or value.get("schema_version") != RUNTIME_AUTHORITY_SCHEMA_VERSION \
            or value.get("record_type") != RUNTIME_AUTHORITY_RECORD_TYPE:
        raise RuntimeAuthorityError("runtime authority receipt binding has the wrong schema")
    for key in ("manifest_sha256", "runtime_files_sha256", "canonical_yaml_sha256"):
        if _HEX64_RE.fullmatch(str(value.get(key, ""))) is None:
            raise RuntimeAuthorityError(f"runtime authority binding has invalid {key}")
    if _HEX40_RE.fullmatch(str(value.get("git_commit", ""))) is None \
            or isinstance(value.get("runtime_file_count"), bool) \
            or not isinstance(value.get("runtime_file_count"), int) \
            or value["runtime_file_count"] <= 0 \
            or not isinstance(value.get("canonical_yaml"), dict):
        raise RuntimeAuthorityError("runtime authority receipt binding is malformed")
    if any(
        not isinstance(relative, str)
        or not relative.startswith("configs/")
        or _HEX64_RE.fullmatch(str(digest)) is None
        for relative, digest in value["canonical_yaml"].items()
    ):
        raise RuntimeAuthorityError("runtime authority canonical YAML binding is malformed")
    if expected_git_commit is not None and value["git_commit"] != expected_git_commit:
        raise RuntimeAuthorityError("runtime authority Git commit differs from the receipt source")
    if expected_purpose is not None and value["purpose"] != expected_purpose:
        raise RuntimeAuthorityError("runtime authority purpose differs from the execution role")
    if expected_configs is not None:
        for relative, digest in expected_configs.items():
            if value["canonical_yaml"].get(relative) != digest:
                raise RuntimeAuthorityError(
                    f"runtime authority config differs from the receipt source: {relative}"
                )
    return dict(value)


def _verify_manifest_against_current_tree(manifest: Mapping[str, object], root: Path) -> None:
    """Recompute the deterministic manifest inventories from the recorded HEAD."""
    commit = str(manifest["git_commit"])
    if _git_head(root) != commit:
        raise RuntimeAuthorityError("runtime authority source tree is at a different Git HEAD")
    _require_clean_worktree(root)
    files: dict[str, dict] = {}
    yaml_files: dict[str, dict] = {}
    for relative, expected_oid in _runtime_tree(root, commit):
        locked = _open_locked_file(root / PurePosixPath(relative))
        try:
            projection = _hash_git_projection(root, relative, locked.payload)
            if projection != expected_oid:
                raise RuntimeAuthorityError(
                    f"runtime authority current bytes differ from HEAD: {relative}"
                )
            digest = hashlib.sha256(locked.payload).hexdigest()
            files[relative] = {
                "git_blob": expected_oid,
                "sha256": digest,
                "size": len(locked.payload),
            }
            if relative.startswith("configs/") and relative.endswith((".yaml", ".yml")):
                yaml_files[relative] = {
                    "payload_b64": base64.b64encode(locked.payload).decode("ascii"),
                    "sha256": digest,
                    "size": len(locked.payload),
                }
        finally:
            locked.close()
    if _git_head(root) != commit:
        raise RuntimeAuthorityError("runtime authority source HEAD changed during verification")
    _require_clean_worktree(root)
    if manifest["runtime_files"] != files \
            or manifest["runtime_files_sha256"] != _canonical_mapping_sha256(files) \
            or manifest["canonical_yaml"] != yaml_files \
            or manifest["canonical_yaml_sha256"] != _canonical_mapping_sha256(yaml_files):
        raise RuntimeAuthorityError(
            "runtime authority manifest differs from the current clean HEAD projection"
        )


def runtime_authority_execution_receipt(
    stage: str,
    *,
    required_configs: Iterable[str],
) -> dict:
    """Freeze the exact child sentinel and parent manifest into a durable receipt."""
    active = _ACTIVE
    if active is None:
        raise RuntimeAuthorityError("runtime authority is not active")
    if _STAGE_RE.fullmatch(stage) is None:
        raise RuntimeAuthorityError("runtime authority execution stage is invalid")
    required = sorted(set(required_configs))
    if any(
        not isinstance(relative, str)
        or not relative.startswith("configs/")
        or relative not in active.manifest["canonical_yaml"]
        for relative in required
    ):
        raise RuntimeAuthorityError("runtime authority required config set is invalid")
    missing = set(required) - active.consumed_yaml
    if missing:
        raise RuntimeAuthorityError(
            "runtime authority execution receipt is missing consumed canonical configs: "
            + ", ".join(sorted(missing))
        )
    sentinel_dir_raw = os.environ.get(ENV_SENTINEL_DIR)
    if not sentinel_dir_raw:
        raise RuntimeAuthorityError("runtime authority sentinel directory is missing")
    sentinel_path = (
        Path(sentinel_dir_raw).expanduser().absolute().resolve() / f"{stage}.json"
    )
    snapshot = _read_authority_snapshot(
        sentinel_path, label=f"runtime authority {stage} sentinel"
    )
    try:
        sentinel = json.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeAuthorityError(f"cannot decode runtime authority sentinel: {exc}") from exc
    binding = verify_runtime_authority_sentinel(stage)
    if sentinel.get("binding") != binding \
            or sentinel.get("consumed_canonical_yaml") != sorted(active.consumed_yaml):
        raise RuntimeAuthorityError("runtime authority execution sentinel content drift")
    # Round-trip through the canonical encoder so callers cannot mutate the
    # live in-process manifest after this function returns.
    manifest = json.loads(_canonical_json_bytes(active.manifest).decode("utf-8"))
    sentinel = json.loads(_canonical_json_bytes(sentinel).decode("utf-8"))
    return {
        "schema_version": RUNTIME_AUTHORITY_SCHEMA_VERSION,
        "record_type": RUNTIME_AUTHORITY_EXECUTION_TYPE,
        "stage": stage,
        "required_canonical_yaml": required,
        "consumed_canonical_yaml": sorted(active.consumed_yaml),
        "binding": binding,
        "runtime_manifest": manifest,
        "runtime_manifest_sha256": active.manifest_sha256,
        "sentinel": sentinel,
        "sentinel_sha256": snapshot.sha256,
        "session_contract": runtime_authority_session_contract(),
    }


def validate_runtime_authority_execution_receipt(
    value: object,
    *,
    expected_stage: str,
    expected_purpose: str,
    expected_git_commit: str,
    expected_configs: Mapping[str, str],
    required_configs: Iterable[str],
    project_root: Path | str | None = None,
) -> dict:
    """Validate a durable execution receipt and, in-source, its clean HEAD tree."""
    keys = {
        "schema_version", "record_type", "stage", "required_canonical_yaml",
        "consumed_canonical_yaml", "binding", "runtime_manifest",
        "runtime_manifest_sha256", "sentinel", "sentinel_sha256",
        "session_contract",
    }
    if not isinstance(value, dict) or set(value) != keys \
            or value.get("schema_version") != RUNTIME_AUTHORITY_SCHEMA_VERSION \
            or value.get("record_type") != RUNTIME_AUTHORITY_EXECUTION_TYPE \
            or value.get("stage") != expected_stage:
        raise RuntimeAuthorityError("runtime authority execution receipt has the wrong schema")
    required = sorted(set(required_configs))
    consumed = value.get("consumed_canonical_yaml")
    if value.get("required_canonical_yaml") != required \
            or not isinstance(consumed, list) \
            or consumed != sorted(set(consumed)) \
            or not set(required).issubset(consumed):
        raise RuntimeAuthorityError("runtime authority execution YAML consumption is invalid")
    if value.get("session_contract") != runtime_authority_session_contract():
        raise RuntimeAuthorityError("runtime authority source/import/session contract differs")
    manifest_sha256 = value.get("runtime_manifest_sha256")
    if _HEX64_RE.fullmatch(str(manifest_sha256 or "")) is None:
        raise RuntimeAuthorityError("runtime authority execution manifest digest is invalid")
    raw_manifest = value.get("runtime_manifest")
    if not isinstance(raw_manifest, dict):
        raise RuntimeAuthorityError("runtime authority execution manifest is missing")
    recorded_root = Path(str(raw_manifest.get("project_root", ""))).expanduser().absolute().resolve()
    manifest = _validate_manifest(
        raw_manifest,
        root=recorded_root,
        purpose=expected_purpose,
    )
    if hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest() != manifest_sha256:
        raise RuntimeAuthorityError("runtime authority execution manifest hash mismatch")
    expected_binding = _binding_from_manifest(
        manifest,
        purpose=expected_purpose,
        manifest_sha256=manifest_sha256,
    )
    binding = validate_runtime_authority_binding(
        value.get("binding"),
        expected_git_commit=expected_git_commit,
        expected_configs=expected_configs,
        expected_purpose=expected_purpose,
    )
    if binding != expected_binding:
        raise RuntimeAuthorityError("runtime authority binding differs from its exact manifest")
    sentinel = value.get("sentinel")
    sentinel_keys = {
        "schema_version", "record_type", "stage", "pid", "nonce", "binding",
        "consumed_canonical_yaml",
    }
    if not isinstance(sentinel, dict) or set(sentinel) != sentinel_keys \
            or sentinel.get("schema_version") != RUNTIME_AUTHORITY_SCHEMA_VERSION \
            or sentinel.get("record_type") != RUNTIME_AUTHORITY_SENTINEL_TYPE \
            or sentinel.get("stage") != expected_stage \
            or isinstance(sentinel.get("pid"), bool) \
            or not isinstance(sentinel.get("pid"), int) or sentinel["pid"] <= 0 \
            or sentinel.get("nonce") != manifest["nonce"] \
            or sentinel.get("binding") != binding \
            or sentinel.get("consumed_canonical_yaml") != consumed:
        raise RuntimeAuthorityError("runtime authority execution sentinel is invalid")
    sentinel_sha256 = value.get("sentinel_sha256")
    if _HEX64_RE.fullmatch(str(sentinel_sha256 or "")) is None \
            or hashlib.sha256(_canonical_json_bytes(sentinel)).hexdigest() != sentinel_sha256:
        raise RuntimeAuthorityError("runtime authority execution sentinel hash mismatch")
    if manifest["git_commit"] != expected_git_commit:
        raise RuntimeAuthorityError("runtime authority execution Git commit differs")
    if any(
        manifest["canonical_yaml"].get(relative, {}).get("sha256") != digest
        for relative, digest in expected_configs.items()
    ):
        raise RuntimeAuthorityError("runtime authority execution canonical configs differ")
    if project_root is not None:
        current_root = Path(project_root).expanduser().absolute().resolve()
        # Source-repository verification replays the clean-filter projection.
        # A copied offline bundle intentionally has no .git and instead relies
        # on the full internally hashed manifest sealed into the release.
        if current_root == recorded_root and (current_root / ".git").exists():
            _verify_manifest_against_current_tree(manifest, current_root)
    return dict(value)


def activate_runtime_authority_child(
    project_root: Path | str,
    *,
    allowed_purposes: Iterable[str] | None = None,
) -> dict:
    """Activate the parent-owned manifest before project/task imports."""
    root = Path(project_root).expanduser().absolute().resolve()
    manifest_raw = os.environ.get(ENV_MANIFEST)
    expected_sha256 = os.environ.get(ENV_MANIFEST_SHA256)
    purpose = os.environ.get(ENV_PURPOSE)
    expected_pycache = os.environ.get("PYTHONPYCACHEPREFIX")
    if not manifest_raw or _HEX64_RE.fullmatch(str(expected_sha256 or "")) is None \
            or not purpose or not expected_pycache:
        raise RuntimeAuthorityError("child was not launched by a tracked runtime authority")
    allowed = None if allowed_purposes is None else set(allowed_purposes)
    if allowed is not None and purpose not in allowed:
        raise RuntimeAuthorityError("runtime authority purpose is invalid for this child")
    if sys.pycache_prefix is None \
            or Path(sys.pycache_prefix).resolve() != Path(expected_pycache).resolve():
        raise RuntimeAuthorityError("child did not start with the exclusive authority pycache")
    manifest_path = Path(manifest_raw).expanduser().absolute()
    snapshot = _read_authority_snapshot(
        manifest_path,
        label="runtime authority manifest",
        expected_sha256=expected_sha256,
    )
    try:
        value = json.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeAuthorityError(f"cannot decode runtime authority manifest: {exc}") from exc
    if os.environ.get(ENV_NONCE) != value.get("nonce"):
        raise RuntimeAuthorityError("runtime authority child nonce mismatch")
    return _activate(value, root=root, purpose=purpose, manifest_sha256=snapshot.sha256)


def emit_runtime_authority_sentinel(
    stage: str,
    *,
    required_configs: Iterable[str] = (),
) -> dict:
    if _STAGE_RE.fullmatch(stage) is None:
        raise RuntimeAuthorityError("runtime authority sentinel stage is invalid")
    active = _ACTIVE
    if active is None:
        raise RuntimeAuthorityError("runtime authority is not active")
    required = set(required_configs)
    missing = required - active.consumed_yaml
    if missing:
        raise RuntimeAuthorityError(
            "runtime authority sentinel is missing consumed canonical configs: "
            + ", ".join(sorted(missing))
        )
    verify_project_import_origins()
    binding = runtime_authority_binding()
    payload = {
        "schema_version": RUNTIME_AUTHORITY_SCHEMA_VERSION,
        "record_type": RUNTIME_AUTHORITY_SENTINEL_TYPE,
        "stage": stage,
        "pid": os.getpid(),
        "nonce": active.manifest["nonce"],
        "binding": binding,
        "consumed_canonical_yaml": sorted(active.consumed_yaml),
    }
    sentinel_dir_raw = os.environ.get(ENV_SENTINEL_DIR)
    if not sentinel_dir_raw:
        raise RuntimeAuthorityError("runtime authority sentinel directory is missing")
    sentinel_dir = Path(sentinel_dir_raw).expanduser().absolute().resolve()
    destination = sentinel_dir / f"{stage}.json"
    encoded = _canonical_json_bytes(payload)
    try:
        with destination.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise RuntimeAuthorityError(f"cannot publish runtime authority sentinel: {exc}") from exc
    print(
        f"RUNTIME_AUTHORITY_OK stage={stage} "
        f"manifest_sha256={binding['manifest_sha256']}",
        flush=True,
    )
    return binding


def verify_runtime_authority_sentinel(stage: str) -> dict:
    """Verify a descendant sentinel against the active child manifest."""
    if _STAGE_RE.fullmatch(stage) is None:
        raise RuntimeAuthorityError("runtime authority sentinel stage is invalid")
    active = _ACTIVE
    sentinel_dir_raw = os.environ.get(ENV_SENTINEL_DIR)
    if active is None or not sentinel_dir_raw:
        raise RuntimeAuthorityError("runtime authority is not active")
    path = Path(sentinel_dir_raw).expanduser().absolute().resolve() / f"{stage}.json"
    snapshot = _read_authority_snapshot(
        path, label=f"runtime authority {stage} sentinel"
    )
    try:
        value = json.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeAuthorityError(f"cannot decode runtime authority sentinel: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "record_type", "stage", "pid", "nonce", "binding",
        "consumed_canonical_yaml",
    } or value.get("schema_version") != RUNTIME_AUTHORITY_SCHEMA_VERSION \
            or value.get("record_type") != RUNTIME_AUTHORITY_SENTINEL_TYPE \
            or value.get("stage") != stage \
            or isinstance(value.get("pid"), bool) \
            or not isinstance(value.get("pid"), int) or value["pid"] <= 0 \
            or value.get("nonce") != active.manifest["nonce"] \
            or not isinstance(value.get("consumed_canonical_yaml"), list) \
            or value["consumed_canonical_yaml"] != sorted(set(value["consumed_canonical_yaml"])):
        raise RuntimeAuthorityError("runtime authority child sentinel is malformed")
    binding = validate_runtime_authority_binding(
        value.get("binding"),
        expected_git_commit=active.manifest["git_commit"],
        expected_purpose=active.purpose,
    )
    if binding != runtime_authority_binding():
        raise RuntimeAuthorityError("runtime authority child sentinel differs from active authority")
    return binding


class RuntimeAuthoritySession:
    """Parent-owned lock/manifest session spanning one process tree."""

    def __init__(self, project_root: Path | str, *, purpose: str) -> None:
        if not isinstance(purpose, str) or not purpose or not re.fullmatch(
            r"[a-z][a-z0-9_-]{0,63}", purpose
        ):
            raise RuntimeAuthorityError("runtime authority purpose is invalid")
        self.root = Path(project_root).expanduser().absolute().resolve()
        self.purpose = purpose
        self.commit: str | None = None
        self.manifest: dict | None = None
        self.manifest_sha256: str | None = None
        self.session_dir: Path | None = None
        self.pycache_dir: Path | None = None
        self.sentinel_dir: Path | None = None
        self.manifest_path: Path | None = None
        self._locked_files: list[_LockedFile] = []
        self._locked_directories: list[object] = []
        self._entered = False

    def __enter__(self) -> "RuntimeAuthoritySession":
        if self._entered or _ACTIVE is not None:
            raise RuntimeAuthorityError("runtime authority sessions cannot be nested")
        if _path_contains_link_or_reparse(self.root) or not self.root.is_dir():
            raise RuntimeAuthorityError("runtime authority project root is missing or unsafe")
        self._entered = True
        try:
            commit = _git_head(self.root)
            _require_clean_worktree(self.root)
            tree = _runtime_tree(self.root, commit)
            directories = {self.root}
            for relative, _oid in tree:
                current = (self.root / PurePosixPath(relative)).parent
                while current != self.root:
                    directories.add(current)
                    current = current.parent
            for directory in sorted(directories, key=lambda item: (len(item.parts), str(item))):
                self._locked_directories.append(_open_locked_directory(directory))

            files: dict[str, dict] = {}
            yaml_files: dict[str, dict] = {}
            for relative, expected_oid in tree:
                locked = _open_locked_file(self.root / PurePosixPath(relative))
                self._locked_files.append(locked)
                projection = _hash_git_projection(self.root, relative, locked.payload)
                if projection != expected_oid:
                    raise RuntimeAuthorityError(
                        f"tracked runtime bytes differ from clean HEAD: {relative}"
                    )
                digest = hashlib.sha256(locked.payload).hexdigest()
                files[relative] = {
                    "git_blob": expected_oid,
                    "sha256": digest,
                    "size": len(locked.payload),
                }
                if relative.startswith("configs/") and relative.endswith((".yaml", ".yml")):
                    yaml_files[relative] = {
                        "payload_b64": base64.b64encode(locked.payload).decode("ascii"),
                        "sha256": digest,
                        "size": len(locked.payload),
                    }
            if _git_head(self.root) != commit:
                raise RuntimeAuthorityError("Git HEAD changed while runtime inputs were locked")
            _require_clean_worktree(self.root)

            session_dir = Path(tempfile.mkdtemp(prefix="ur5e-runtime-authority-"))
            pycache_dir = session_dir / "pycache"
            sentinel_dir = session_dir / "sentinels"
            pycache_dir.mkdir()
            sentinel_dir.mkdir()
            manifest = {
                "schema_version": RUNTIME_AUTHORITY_SCHEMA_VERSION,
                "record_type": RUNTIME_AUTHORITY_RECORD_TYPE,
                "project_root": str(self.root),
                "purpose": self.purpose,
                "git_commit": commit,
                "runtime_files": files,
                "runtime_files_sha256": _canonical_mapping_sha256(files),
                "canonical_yaml": yaml_files,
                "canonical_yaml_sha256": _canonical_mapping_sha256(yaml_files),
                "nonce": os.urandom(32).hex(),
            }
            encoded = _canonical_json_bytes(manifest)
            manifest_path = session_dir / "runtime_manifest.json"
            with manifest_path.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            manifest_lock = _open_locked_file(manifest_path)
            self._locked_files.append(manifest_lock)

            self.commit = commit
            self.manifest = manifest
            self.manifest_sha256 = hashlib.sha256(encoded).hexdigest()
            self.session_dir = session_dir
            self.pycache_dir = pycache_dir
            self.sentinel_dir = sentinel_dir
            self.manifest_path = manifest_path
            return self
        except BaseException:
            self._close_resources()
            raise

    def child_environment(self, source: Mapping[str, str] | None = None) -> dict[str, str]:
        if not self._entered or self.manifest is None or self.manifest_path is None \
                or self.manifest_sha256 is None or self.pycache_dir is None \
                or self.sentinel_dir is None:
            raise RuntimeAuthorityError("runtime authority session is not active")
        environment = {str(key): str(value) for key, value in (os.environ if source is None else source).items()}
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONPYCACHEPREFIX"] = str(self.pycache_dir)
        environment[ENV_MANIFEST] = str(self.manifest_path)
        environment[ENV_MANIFEST_SHA256] = self.manifest_sha256
        environment[ENV_SENTINEL_DIR] = str(self.sentinel_dir)
        environment[ENV_PURPOSE] = self.purpose
        environment[ENV_NONCE] = self.manifest["nonce"]
        return environment

    def verify_child_sentinel(self, stage: str) -> dict:
        if _STAGE_RE.fullmatch(stage) is None or self.sentinel_dir is None \
                or self.manifest is None or self.manifest_sha256 is None:
            raise RuntimeAuthorityError("runtime authority sentinel request is invalid")
        path = self.sentinel_dir / f"{stage}.json"
        snapshot = _read_authority_snapshot(
            path, label=f"runtime authority {stage} sentinel"
        )
        try:
            value = json.loads(snapshot.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeAuthorityError(f"cannot decode runtime authority sentinel: {exc}") from exc
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "record_type", "stage", "pid", "nonce", "binding",
            "consumed_canonical_yaml",
        } or value.get("schema_version") != RUNTIME_AUTHORITY_SCHEMA_VERSION \
                or value.get("record_type") != RUNTIME_AUTHORITY_SENTINEL_TYPE \
                or value.get("stage") != stage \
                or isinstance(value.get("pid"), bool) \
                or not isinstance(value.get("pid"), int) or value["pid"] <= 0 \
                or value.get("nonce") != self.manifest["nonce"] \
                or not isinstance(value.get("consumed_canonical_yaml"), list) \
                or value["consumed_canonical_yaml"] != sorted(set(value["consumed_canonical_yaml"])):
            raise RuntimeAuthorityError("runtime authority child sentinel is malformed")
        binding = validate_runtime_authority_binding(
            value.get("binding"),
            expected_git_commit=self.commit,
            expected_purpose=self.purpose,
        )
        if binding["manifest_sha256"] != self.manifest_sha256 \
                or binding != self.binding():
            raise RuntimeAuthorityError("runtime authority child sentinel differs from parent authority")
        return binding

    def binding(self) -> dict:
        if self.manifest is None or self.manifest_sha256 is None:
            raise RuntimeAuthorityError("runtime authority session is not active")
        yaml_hashes = {
            relative: record["sha256"]
            for relative, record in sorted(self.manifest["canonical_yaml"].items())
        }
        return {
            "schema_version": RUNTIME_AUTHORITY_SCHEMA_VERSION,
            "record_type": RUNTIME_AUTHORITY_RECORD_TYPE,
            "purpose": self.purpose,
            "manifest_sha256": self.manifest_sha256,
            "git_commit": self.manifest["git_commit"],
            "runtime_file_count": len(self.manifest["runtime_files"]),
            "runtime_files_sha256": self.manifest["runtime_files_sha256"],
            "canonical_yaml_sha256": self.manifest["canonical_yaml_sha256"],
            "canonical_yaml": yaml_hashes,
        }

    def verify_unchanged(self) -> None:
        if not self._entered or self.commit is None or self.manifest is None:
            raise RuntimeAuthorityError("runtime authority session is not active")
        if _git_head(self.root) != self.commit:
            raise RuntimeAuthorityError("Git HEAD changed while the authorized child was active")
        _require_clean_worktree(self.root)
        manifest_files = self.manifest["runtime_files"]
        for locked in self._locked_files:
            try:
                relative = locked.path.resolve().relative_to(self.root).as_posix()
            except ValueError:
                continue  # private manifest lock
            expected = manifest_files.get(relative)
            if expected is None:
                continue
            locked.stream.seek(0)
            payload = locked.stream.read()
            locked.stream.seek(0)
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise RuntimeAuthorityError(
                    f"tracked runtime bytes changed while child was active: {relative}"
                )

    def _close_resources(self) -> None:
        for locked in reversed(self._locked_files):
            try:
                locked.close()
            except OSError:
                pass
        self._locked_files.clear()
        for handle in reversed(self._locked_directories):
            _close_locked_directory(handle)
        self._locked_directories.clear()
        if self.session_dir is not None:
            shutil.rmtree(self.session_dir, ignore_errors=True)
        self._entered = False

    def __exit__(self, exc_type, exc, traceback) -> bool:  # noqa: ANN001
        verification_error: BaseException | None = None
        try:
            if exc_type is None:
                self.verify_unchanged()
        except BaseException as caught:  # fail closed after child success
            verification_error = caught
        finally:
            self._close_resources()
        if verification_error is not None:
            raise verification_error
        return False


def authorize_entrypoint(
    project_root: Path | str,
    script_path: Path | str,
    *,
    purpose: str,
    sentinel_stage: str,
) -> dict:
    """Outer-bootstrap or activate an already-authorized inner entry point.

    Call this under ``if __name__ == '__main__'`` before importing Isaac or
    project task modules.  The outer call never reaches those imports.
    """
    root = Path(project_root).expanduser().absolute().resolve()
    if os.environ.get(ENV_MANIFEST):
        return activate_runtime_authority_child(root, allowed_purposes={purpose})
    try:
        with RuntimeAuthoritySession(root, purpose=purpose) as session:
            command = [sys.executable, "-u", str(Path(script_path).absolute()), *sys.argv[1:]]
            completed = subprocess.run(
                command,
                cwd=root,
                env=session.child_environment(),
                check=False,
            )
            if completed.returncode == 0:
                session.verify_child_sentinel(sentinel_stage)
            return_code = int(completed.returncode)
    except RuntimeAuthorityError as exc:
        print(f"[runtime-authority][error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(3) from exc
    raise SystemExit(return_code)
