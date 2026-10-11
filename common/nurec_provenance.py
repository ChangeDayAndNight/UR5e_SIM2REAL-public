"""Pure validation for the ignored, optional NuRec workcell asset lineage."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import re

from deployment.contracts import ContractError, path_contains_link_or_reparse, sha256_file


NUREC_USDZ = "assets/Scene/Workplace.usdz"
NUREC_SOURCE = "assets/Scene/Workplace.ply"
NUREC_PROVENANCE = "assets/Scene/Workplace.provenance.json"
_PROVENANCE_KEYS = {
    "schema_version",
    "asset_path",
    "asset_sha256",
    "asset_bytes",
    "source_path",
    "source_sha256",
    "source_bytes",
    "capture_id",
    "conversion_tool",
    "created_at",
}


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_nurec_record(value: object) -> dict:
    """Validate a self-contained, path-sanitized NuRec provenance record."""
    if not isinstance(value, dict) or set(value) != _PROVENANCE_KEYS:
        raise ContractError("NuRec provenance has an incomplete/unexpected field set")
    if value.get("schema_version") != 1:
        raise ContractError("NuRec provenance schema must be 1")
    if value.get("asset_path") != NUREC_USDZ or value.get("source_path") != NUREC_SOURCE:
        raise ContractError("NuRec provenance paths must be canonical project-relative paths")
    for label in ("asset_sha256", "source_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(value.get(label, ""))) is None:
            raise ContractError(f"NuRec provenance {label} is invalid")
    if not _positive_int(value.get("asset_bytes")) or not _positive_int(value.get("source_bytes")):
        raise ContractError("NuRec provenance sizes must be positive integers")
    for label in ("capture_id", "conversion_tool"):
        text = value.get(label)
        if not isinstance(text, str) or re.fullmatch(r"[A-Za-z0-9_.:+ -]{1,128}", text) is None:
            raise ContractError(f"NuRec provenance {label} is missing or unsafe")
    if not _timestamp(value.get("created_at")):
        raise ContractError("NuRec provenance created_at must be timezone-aware ISO-8601")
    return value


def load_nurec_provenance(project_root: Path | str) -> dict:
    """Read and byte-verify the local PLY→USDZ lineage, failing closed."""
    root = Path(project_root).resolve()
    paths = {
        NUREC_USDZ: root / NUREC_USDZ,
        NUREC_SOURCE: root / NUREC_SOURCE,
        NUREC_PROVENANCE: root / NUREC_PROVENANCE,
    }
    for relative, path in paths.items():
        if path_contains_link_or_reparse(path.absolute()) or not path.is_file():
            raise ContractError(f"NuRec local provenance file is missing/unsafe: {relative}")
    try:
        decoded = json.loads(paths[NUREC_PROVENANCE].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode NuRec provenance: {exc}") from exc
    record = validate_nurec_record(decoded)
    checks = (
        (NUREC_USDZ, "asset_sha256", "asset_bytes"),
        (NUREC_SOURCE, "source_sha256", "source_bytes"),
    )
    for relative, hash_field, size_field in checks:
        path = paths[relative]
        if record[hash_field] != sha256_file(path) or record[size_field] != path.stat().st_size:
            raise ContractError(f"NuRec provenance byte binding mismatch: {relative}")
    return {
        "metadata": record,
        "provenance_sha256": sha256_file(paths[NUREC_PROVENANCE]),
        "provenance_bytes": paths[NUREC_PROVENANCE].stat().st_size,
    }


def validate_sealed_nurec(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "metadata", "provenance_sha256", "provenance_bytes"
    }:
        raise ContractError("sealed NuRec provenance is incomplete")
    validate_nurec_record(value.get("metadata"))
    if re.fullmatch(r"[0-9a-f]{64}", str(value.get("provenance_sha256", ""))) is None:
        raise ContractError("sealed NuRec provenance digest is invalid")
    if not _positive_int(value.get("provenance_bytes")):
        raise ContractError("sealed NuRec provenance size is invalid")
    return value
