#!/usr/bin/env python
"""Build a sealed representative-observation dataset from one sealed sim capture.

This is the only supported way to create ``--parity-obs`` input.  It performs
no Isaac/GPU work and issues no hardware command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The documented source-side CLI may be launched from a shell that was not
# activated by bootstrap_training.ps1.  Re-exec before importing NumPy/Torch
# dependent release code, while keeping normal module imports side-effect free.
if __name__ == "__main__":
    from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

    reexec_in_training_python(__file__)
    configure_utf8_stdio()

import numpy as np

from common.policy_io import read_anchored_regular_file  # noqa: E402
from deployment.contracts import ContractError, is_link_or_reparse, path_contains_link_or_reparse  # noqa: E402
from deployment.release import (  # noqa: E402
    MIN_PARITY_SAMPLES,
    CAPTURE_MANIFEST_NAME,
    PARITY_SIDECAR_SCHEMA_VERSION,
    PARITY_SIDECAR_TYPE,
    _read_authority_regular_file_once,
    _remove_owned_published_file,
    candidate_receipt_binding,
    close_parity_observations,
    load_parity_observations,
    parity_sidecar_path,
    publish_regular_file_no_replace,
    read_verified_capture_observation_block,
    scan_parity_source_dumps,
    verified_capture_records_from_manifest_bytes,
    verify_candidate,
)


def _sha256_size_regular_file(path: Path | str, *, label: str) -> tuple[str, int]:
    """Stream hash and size from one stable regular-file description.

    Parity datasets can be multiple GiB, so the byte-returning snapshot helper
    is intentionally not used here.  The digest, byte count, and stability
    checks nevertheless come from the same already-open file description.
    """

    source = Path(path).absolute()
    if path_contains_link_or_reparse(source):
        raise ContractError(f"{label} contains a symlink/reparse path")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError(f"{label} is not a regular file")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
            after = os.fstat(stream.fileno())
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if byte_count != before.st_size or any(
        getattr(before, field, None) != getattr(after, field, None)
        for field in stable_fields
    ):
        raise ContractError(f"{label} changed while it was hashed")
    return digest.hexdigest(), byte_count


def _canonical_output(path: Path | str) -> Path:
    raw = Path(path).expanduser().absolute()
    if path_contains_link_or_reparse(raw):
        raise ContractError("parity output path contains a symlink/reparse point")
    allowed = (ROOT / "logs/parity").resolve()
    output = raw.resolve()
    if output.parent != allowed or re.fullmatch(r"[A-Za-z0-9_.-]+\.npy", output.name) is None:
        raise ContractError(f"parity output must be one safe .npy directly under {allowed}")
    sidecar = parity_sidecar_path(output)
    if output.exists() or sidecar.exists():
        raise ContractError("parity output/sidecar already exists; refusing overwrite")
    allowed.mkdir(parents=True, exist_ok=True)
    if path_contains_link_or_reparse(allowed):
        raise ContractError("parity output directory became a symlink/reparse point")
    return output


def _verify_published_pair(
    dataset: Path,
    sidecar: Path,
    *,
    expected_dataset_sha256: str,
    expected_dataset_bytes: int,
    expected_sidecar_payload: bytes,
    expected_sidecar_value: dict,
) -> None:
    """Verify one coherent sidecar -> dataset -> sidecar publication session."""

    expected_sidecar_sha256 = hashlib.sha256(expected_sidecar_payload).hexdigest()
    first_sidecar, first_digest = _read_authority_regular_file_once(
        sidecar,
        label="published parity sidecar",
        expected_sha256=expected_sidecar_sha256,
    )
    try:
        decoded = json.loads(first_sidecar.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode published parity sidecar: {exc}") from exc
    if decoded != expected_sidecar_value or first_digest != expected_sidecar_sha256:
        raise ContractError("published parity sidecar differs from its authority")
    dataset_record = _sha256_size_regular_file(
        dataset, label="published parity dataset"
    )
    if dataset_record != (expected_dataset_sha256, expected_dataset_bytes):
        raise ContractError("published parity dataset differs from its sidecar record")
    second_sidecar, second_digest = _read_authority_regular_file_once(
        sidecar,
        label="rechecked published parity sidecar",
        expected_sha256=expected_sidecar_sha256,
    )
    if second_sidecar != first_sidecar or second_digest != first_digest:
        raise ContractError("published parity sidecar changed during pair verification")


def build_dataset(
    candidate_dir: Path | str,
    capture_manifest_path: Path | str,
    output_path: Path | str,
) -> tuple[Path, Path]:
    candidate_raw = Path(candidate_dir).expanduser().absolute()
    if path_contains_link_or_reparse(candidate_raw):
        raise ContractError("candidate CLI path contains a symlink/reparse point")
    capture_raw = Path(capture_manifest_path).expanduser().absolute()
    if path_contains_link_or_reparse(capture_raw):
        raise ContractError("capture-manifest CLI path contains a symlink/reparse point")
    output = _canonical_output(output_path)
    sidecar = parity_sidecar_path(output)
    candidate_path = candidate_raw.resolve()
    candidate = verify_candidate(
        candidate_path,
        project_root=ROOT,
        require_current_context=True,
    )
    binding = candidate_receipt_binding(candidate_path, candidate, project_root=ROOT)
    candidate_observation = candidate["observation_contract"]
    expected_dim = int(candidate_observation["observation_dim"])
    try:
        spec_payload = read_anchored_regular_file(
            candidate_path / "obs_spec.json",
            expected_sha256=candidate["files"]["obs_spec.json"],
            label="candidate obs_spec.json",
        )
        spec = json.loads(spec_payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot consume anchored candidate obs_spec.json: {exc}") from exc
    if (
        not isinstance(spec, dict)
        or int(spec.get("obs_dim", -1)) != expected_dim
        or spec.get("action_scheme") != candidate_observation["action_mode"]
        or spec.get("layout") != candidate_observation["layout"]
        or spec.get("cfg_hash") != candidate_observation["cfg_hash"]
        or candidate_observation["action_mode"] != "ik_rel"
    ):
        raise ContractError("candidate observation/action contract is not deployable ik_rel")
    # From here onward the manifest contract is the authority.  The decoded
    # spec above merely proves that the exact manifest-hashed bytes agree.
    layout = candidate_observation["layout"]
    obs_cfg_hash = candidate_observation["cfg_hash"]
    capture_path = capture_raw.resolve()
    if capture_path.name != CAPTURE_MANIFEST_NAME:
        raise ContractError("capture manifest filename is non-canonical")
    before, files = scan_parity_source_dumps(
        capture_path,
        expected_dim=expected_dim,
        candidate_dir=candidate_path,
        project_root=ROOT,
    )
    if before["total_rows"] < MIN_PARITY_SAMPLES:
        raise ContractError(
            f"source dumps contain {before['total_rows']} rows; {MIN_PARITY_SAMPLES} required"
        )
    source_records = verified_capture_records_from_manifest_bytes(
        capture_path,
        expected_sha256=before["capture_manifest_sha256"],
        ordered_files=files,
    )

    staging = Path(tempfile.mkdtemp(prefix=".parity-build-", dir=output.parent)).absolute()
    staged_dataset = staging / output.name
    staged_sidecar = staging / sidecar.name
    published_dataset_identity: tuple[int, int] | None = None
    published_sidecar_identity: tuple[int, int] | None = None
    try:
        if is_link_or_reparse(staging) or staging.parent != output.parent:
            raise ContractError("parity staging directory validation failed")
        destination = None
        offset = 0
        try:
            destination = np.lib.format.open_memmap(
                staged_dataset,
                mode="w+",
                dtype=np.float32,
                shape=(int(before["total_rows"]), expected_dim),
            )
            for path, expected_record in zip(files, source_records):
                block = read_verified_capture_observation_block(
                    path,
                    expected_record,
                    expected_dim=expected_dim,
                )
                destination[offset : offset + block.shape[0]] = block
                offset += block.shape[0]
            destination.flush()
        finally:
            if destination is not None:
                memory_map = getattr(destination, "_mmap", None)
                if memory_map is not None:
                    memory_map.close()
                destination = None
        if offset != before["total_rows"]:
            raise ContractError("parity dataset row assembly count drifted")
        after, after_files = scan_parity_source_dumps(
            capture_path,
            expected_dim=expected_dim,
            candidate_dir=candidate_path,
            project_root=ROOT,
        )
        if after != before:
            raise ContractError("parity source dumps changed while the dataset was built")
        if [path.resolve() for path in after_files] != [path.resolve() for path in files]:
            raise ContractError("parity capture file order/set changed while building")
        # Re-open through the same diversity/finite gate used by export.
        observations = None
        try:
            observations = load_parity_observations(
                staged_dataset, expected_dim, layout=layout
            )
        finally:
            close_parity_observations(observations)
            del observations
        dataset_sha256, dataset_bytes = _sha256_size_regular_file(
            staged_dataset, label="staged parity dataset"
        )
        dataset_record = {
            "path": output.relative_to(ROOT).as_posix(),
            "sha256": dataset_sha256,
            "bytes": dataset_bytes,
            "format": "npy",
            "rows": int(before["total_rows"]),
            "observation_dim": expected_dim,
        }
        payload = {
            "schema_version": PARITY_SIDECAR_SCHEMA_VERSION,
            "record_type": PARITY_SIDECAR_TYPE,
            "dataset": dataset_record,
            "observation_contract": {
                "cfg_hash": obs_cfg_hash,
                "layout": layout,
                "action_mode": "ik_rel",
            },
            "training": {
                "receipt_sha256": candidate["source"]["training_receipt_sha256"],
                "checkpoint_sha256": candidate["source"]["checkpoint_sha256"],
                "checkpoint_name": candidate["source"]["checkpoint_name"],
                "run_root": candidate["source"]["run_root"],
            },
            "candidate": binding,
            "source_dumps": before,
        }
        sidecar_payload = (
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with staged_sidecar.open("xb") as stream:
            stream.write(sidecar_payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Dataset first, sidecar last: the sidecar is the commit marker and
        # consumers reject an orphan dataset. Hard-link publication is atomic
        # and no-replace, so a concurrent builder can never be overwritten.
        published_dataset_identity = publish_regular_file_no_replace(
            staged_dataset, output
        )
        published_sha256, published_bytes = _sha256_size_regular_file(
            output, label="published parity dataset"
        )
        if (published_sha256, published_bytes) != (dataset_sha256, dataset_bytes):
            raise ContractError("published parity dataset differs from its sidecar record")
        published_sidecar_identity = publish_regular_file_no_replace(
            staged_sidecar, sidecar
        )
        _verify_published_pair(
            output,
            sidecar,
            expected_dataset_sha256=dataset_sha256,
            expected_dataset_bytes=dataset_bytes,
            expected_sidecar_payload=sidecar_payload,
            expected_sidecar_value=payload,
        )
        return output, sidecar
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        try:
            if published_sidecar_identity is not None:
                _remove_owned_published_file(
                    sidecar, expected_identity=published_sidecar_identity
                )
            if published_dataset_identity is not None:
                _remove_owned_published_file(
                    output, expected_identity=published_dataset_identity
                )
        except BaseException as cleanup_exc:
            cleanup_error = cleanup_exc
        if cleanup_error is not None:
            raise ContractError(
                f"parity pair publication failed and owned cleanup failed: {cleanup_error}"
            ) from exc
        raise
    finally:
        if staging.exists() and staging.parent == output.parent \
                and staging.name.startswith(".parity-build-"):
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        dataset, sidecar = build_dataset(
            args.candidate_dir,
            args.capture_manifest,
            args.output,
        )
    except ContractError as exc:
        print(f"[parity-build][error] {exc}")
        return 1
    print(f"[parity-build] dataset={dataset}")
    print(f"[parity-build] provenance={sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
