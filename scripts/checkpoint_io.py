"""Single-open, SHA-anchored checkpoint deserialization helpers."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager

from common.file_snapshot import FileSnapshotError, read_regular_file_snapshot
from deployment.contracts import ContractError, path_contains_link_or_reparse


@contextmanager
def _open_anchored_checkpoint(
    checkpoint: Path | str,
    *,
    expected_sha256: str,
    allowed_root: Path | str,
) -> Iterator[io.BytesIO]:
    """Yield an in-process stream over exactly the SHA-anchored bytes.

    The source pathname is opened once.  Consumers receive the already-open
    immutable payload through a pathless ``BytesIO``, so neither ``torch.load``
    nor an RSL-RL runner can resolve or race a mutable checkpoint pathname
    after the integrity decision.
    """
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ContractError("checkpoint load requires an exact lowercase SHA-256 anchor")
    root_raw = Path(allowed_root).expanduser().absolute()
    source_raw = Path(checkpoint).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw) or path_contains_link_or_reparse(source_raw):
        raise ContractError("checkpoint/root contains a symlink or reparse point")
    root = root_raw.resolve()
    source = source_raw.resolve()
    if not root.is_dir() or not source.is_file() or root not in source.parents:
        raise ContractError("checkpoint is outside its canonical allowed root")

    try:
        snapshot = read_regular_file_snapshot(
            source,
            label="checkpoint opened bytes",
            expected_sha256=expected_sha256,
        )
    except FileSnapshotError as exc:
        raise ContractError(
            "checkpoint opened bytes differ from the trusted checkpoint "
            f"or could not be verified: {exc}"
        ) from exc

    # ``RegularFileSnapshot.payload`` is immutable ``bytes`` derived from the
    # same open handle as the digest.  BytesIO gives torch/RSL-RL the seekable
    # binary interface they require without publishing a post-hash pathname
    # that another process could overwrite before deserialization.
    stream = io.BytesIO(snapshot.payload)
    try:
        yield stream
    finally:
        stream.close()


def load_anchored_checkpoint(
    checkpoint: Path | str,
    *,
    expected_sha256: str,
    allowed_root: Path | str,
    torch_module: Any,
) -> dict:
    """Deserialize only bytes proven to match ``expected_sha256``.

    The caller-controlled source is opened once.  The exact verified bytes are
    retained in process and passed to ``torch.load`` without exposing or
    reopening any pathname after authorization.
    """
    with _open_anchored_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        allowed_root=allowed_root,
    ) as dst:
        loaded = torch_module.load(dst, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ContractError("checkpoint payload must be a mapping")
    return loaded


def load_anchored_runner_checkpoint(
    checkpoint: Path | str,
    *,
    expected_sha256: str,
    allowed_root: Path | str,
    runner: Any,
) -> Any:
    """Run the installed RSL-RL ``runner.load`` on an anchored pathless stream.

    Delegating to the runner, instead of reproducing its state-dict handling,
    preserves the pinned RSL-RL 3.1.2 policy/optimizer/iteration/RND semantics.
    ``runner.load`` ultimately calls ``torch.load`` and accepts the binary file
    object even though its public annotation says ``str``.
    """
    load = getattr(runner, "load", None)
    if not callable(load):
        raise ContractError("checkpoint consumer does not provide runner.load")
    with _open_anchored_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        allowed_root=allowed_root,
    ) as stream:
        return load(stream)
