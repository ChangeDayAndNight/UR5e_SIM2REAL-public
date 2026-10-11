from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.path_safety import reset_episode_dump


def _write_owner(root: Path) -> None:
    candidate_id, capture_id = root.parts[-2:]
    root.mkdir(parents=True, exist_ok=True)
    (root / ".capture_owner.json").write_text(
        json.dumps({
            "record_type": "ur5e_parity_capture_owner",
            "candidate_id": candidate_id,
            "capture_id": capture_id,
            "candidate": {"manifest_sha256": candidate_id + "0" * 48},
        }),
        encoding="utf-8",
    )


def test_episode_cleanup_deletes_only_validated_direct_child(tmp_path):
    repo = tmp_path / "repo"
    dump_root = (
        repo / "Phase3" / "logs" / "obs_dump_sim"
        / ("a" * 16) / "20260814T010203Z-deadbeef"
    )
    episode = dump_root / "ep000"
    episode.mkdir(parents=True)
    _write_owner(dump_root)
    (episode / "stale.npz").write_bytes(b"stale")
    sibling = dump_root / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")

    target = reset_episode_dump(dump_root, 0, repo_root=repo)

    assert target == episode.resolve()
    assert not target.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("selector", ["repo", "home", "anchor"])
def test_episode_cleanup_rejects_broad_roots(tmp_path, monkeypatch, selector):
    repo = tmp_path / "repo"
    repo.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    roots = {
        "repo": repo,
        "home": fake_home,
        "anchor": Path(tmp_path.anchor),
    }
    with pytest.raises(ValueError, match="broad dump root"):
        reset_episode_dump(roots[selector], 0, repo_root=repo)


def test_episode_cleanup_rejects_symlink_target(tmp_path):
    repo = tmp_path / "repo"
    dump_root = (
        repo / "Phase3" / "logs" / "obs_dump_sim"
        / ("a" * 16) / "20260814T010203Z-deadbeef"
    )
    dump_root.mkdir(parents=True)
    _write_owner(dump_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = dump_root / "ep000"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink privilege is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        reset_episode_dump(dump_root, 0, repo_root=repo)


def test_episode_cleanup_rejects_arbitrary_external_directory(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    important = tmp_path / "important"
    episode = important / "ep000"
    episode.mkdir(parents=True)
    sentinel = episode / "do-not-delete.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical generated directory"):
        reset_episode_dump(important, 0, repo_root=repo)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "relative",
    [
        "ep000",
        "aaaaaaaaaaaaaaaa/not-a-capture",
        "aaaaaaaaaaaaaaaa/20260814T010203Z-deadbeef/nested",
        "not-a-candidate/20260814T010203Z-deadbeef",
    ],
)
def test_episode_cleanup_rejects_non_owned_canonical_descendants(tmp_path, relative):
    repo = tmp_path / "repo"
    dump_root = repo / "Phase3/logs/obs_dump_sim" / relative
    dump_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="candidate-id/capture-id"):
        reset_episode_dump(dump_root, 0, repo_root=repo)


def test_episode_cleanup_rejects_matching_path_without_owner_marker(tmp_path):
    repo = tmp_path / "repo"
    dump_root = (
        repo / "Phase3/logs/obs_dump_sim" / ("a" * 16)
        / "20260814T010203Z-deadbeef"
    )
    dump_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="ownership marker"):
        reset_episode_dump(dump_root, 0, repo_root=repo)


def test_episode_cleanup_rejects_reparse_helper_detection(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    dump_root = (
        repo / "Phase3" / "logs" / "obs_dump_sim"
        / ("a" * 16) / "20260814T010203Z-deadbeef"
    )
    dump_root.mkdir(parents=True)
    _write_owner(dump_root)
    monkeypatch.setattr("common.path_safety.path_contains_link_or_reparse", lambda _path: True)
    with pytest.raises(ValueError, match="reparse"):
        reset_episode_dump(dump_root, 0, repo_root=repo)
