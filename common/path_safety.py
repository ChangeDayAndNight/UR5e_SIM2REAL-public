"""Narrow helpers for destructive cleanup of generated evidence directories."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from deployment.contracts import is_link_or_reparse, path_contains_link_or_reparse


def reset_episode_dump(dump_root: Path | str, episode: int, *, repo_root: Path | str) -> Path:
    """Delete one episode only inside a candidate/capture-keyed generated root."""
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise ValueError("episode must be a non-negative integer")
    raw_root = Path(dump_root).expanduser()
    if path_contains_link_or_reparse(raw_root.absolute()):
        raise ValueError("dump root may not contain a symlink/reparse point")
    root = raw_root.resolve()
    repository = Path(repo_root).resolve()
    allowed_root = (repository / "Phase3" / "logs" / "obs_dump_sim").resolve()
    anchor = Path(root.anchor).resolve()
    broad_roots = {anchor, repository, Path.home().resolve()}
    if root in broad_roots:
        raise ValueError(f"refusing broad dump root: {root}")
    try:
        relative_root = root.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"dump root must be below the canonical generated directory {allowed_root}; got {root}"
        ) from exc
    if (
        len(relative_root.parts) != 2
        or re.fullmatch(r"[0-9a-f]{16}", relative_root.parts[0]) is None
        or re.fullmatch(
            r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", relative_root.parts[1]
        ) is None
    ):
        raise ValueError(
            "dump root must be candidate-id/capture-id below the canonical generated directory"
        )
    owner_path = root / ".capture_owner.json"
    if is_link_or_reparse(owner_path) or not owner_path.is_file():
        raise ValueError("dump root has no safe capture ownership marker")
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid capture ownership marker: {exc}") from exc
    if (
        not isinstance(owner, dict)
        or owner.get("record_type") != "ur5e_parity_capture_owner"
        or owner.get("candidate_id") != relative_root.parts[0]
        or owner.get("capture_id") != relative_root.parts[1]
        or not isinstance(owner.get("candidate"), dict)
        or owner["candidate"].get("manifest_sha256", "")[:16] != relative_root.parts[0]
    ):
        raise ValueError("capture ownership marker does not match its candidate/capture path")
    if root.exists() and not root.is_dir():
        raise ValueError(f"dump root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    child_name = f"ep{episode:03d}"
    if re.fullmatch(r"ep[0-9]{3,}", child_name) is None:
        raise ValueError("invalid episode directory name")
    lexical_child = root / child_name
    if is_link_or_reparse(lexical_child) or path_contains_link_or_reparse(lexical_child.absolute()):
        raise ValueError("episode dump may not contain a symlink/reparse point")
    child = lexical_child.resolve()
    if child == root or child.parent != root or root not in child.parents:
        raise ValueError(f"episode dump escapes its resolved root: {child}")
    if child.exists():
        if not child.is_dir():
            raise ValueError(f"episode dump is not a directory: {child}")
        shutil.rmtree(child)
    return child
