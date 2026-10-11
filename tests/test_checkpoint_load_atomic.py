"""Regression guards for parent-authorized RSL-RL checkpoint consumption."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _load_checkpoint_io():
    path = ROOT / "scripts" / "checkpoint_io.py"
    spec = importlib.util.spec_from_file_location("checkpoint_io_runner_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pure_train_contract() -> dict:
    parsed = ast.parse(_source("scripts/rsl_rl/train.py"))
    wanted = {
        "_normalize_expected_checkpoint_sha256",
        "_validate_checkpoint_load_contract",
    }
    nodes = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    namespace = {"Path": Path}
    isolated = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, "scripts/rsl_rl/train.py", "exec"), namespace)
    return namespace


def test_runner_load_receives_only_the_open_sha_anchored_stream(tmp_path: Path) -> None:
    module = _load_checkpoint_io()
    store = tmp_path / "logs" / "rsl_rl" / "ur5e_binpick_vision"
    run = store / "_distill-warmstart-test"
    run.mkdir(parents=True)
    checkpoint = run / "model_0.pt"
    payload = b"trusted-rsl-rl-checkpoint"
    checkpoint.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    class Runner:
        def __init__(self) -> None:
            self.calls: list[tuple[object, bytes]] = []

        def load(self, stream):
            assert not isinstance(stream, (str, Path))
            assert isinstance(stream, io.BytesIO)
            assert not hasattr(stream, "name")
            assert stream.readable()
            loaded = stream.read()
            self.calls.append((stream, loaded))
            return {"infos": "rsl-rl-return-value"}

    runner = Runner()
    result = module.load_anchored_runner_checkpoint(
        checkpoint,
        expected_sha256=digest,
        allowed_root=store,
        runner=runner,
    )

    assert result == {"infos": "rsl-rl-return-value"}
    assert len(runner.calls) == 1
    assert runner.calls[0][1] == payload
    assert list(run.iterdir()) == [checkpoint]


def test_runner_load_rejects_swap_back_bytes_before_state_application(
    monkeypatch, tmp_path: Path,
) -> None:
    module = _load_checkpoint_io()
    store = tmp_path / "logs" / "rsl_rl" / "ur5e_binpick_vision"
    run = store / "_nurec-warmstart-test"
    run.mkdir(parents=True)
    checkpoint = run / "model_0.pt"
    trusted = b"adopted-student-checkpoint"
    checkpoint.write_bytes(trusted)
    digest = hashlib.sha256(trusted).hexdigest()
    real_open = Path.open

    class Runner:
        calls = 0

        def load(self, _stream):
            self.calls += 1

    runner = Runner()

    def raced_open(self, mode="r", *args, **kwargs):
        if self == checkpoint and mode == "rb":
            # The mutable pathname is restored before any possible path-level
            # recheck, while the already-open stream contains unauthorized data.
            checkpoint.write_bytes(trusted)
            return io.BytesIO(b"unadopted-swap-bytes")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raced_open)
    with pytest.raises(module.ContractError, match="opened bytes differ"):
        module.load_anchored_runner_checkpoint(
            checkpoint,
            expected_sha256=digest,
            allowed_root=store,
            runner=runner,
        )

    assert runner.calls == 0
    assert list(run.iterdir()) == [checkpoint]


def test_post_hash_path_mutation_cannot_change_runner_consumed_bytes(
    tmp_path: Path,
) -> None:
    module = _load_checkpoint_io()
    store = tmp_path / "logs" / "rsl_rl" / "ur5e_binpick_vision"
    run = store / "_distill-warmstart-test"
    run.mkdir(parents=True)
    checkpoint = run / "model_0.pt"
    trusted = b"trusted-rsl-rl-checkpoint"
    checkpoint.write_bytes(trusted)
    digest = hashlib.sha256(trusted).hexdigest()

    class Runner:
        seen: bytes | None = None

        def load(self, stream):
            # Replacing the authorized source after the hash decision used to
            # be able to target the helper's still-linked temporary pathname.
            # The consumer now receives only an in-process, pathless snapshot.
            assert isinstance(stream, io.BytesIO)
            assert not hasattr(stream, "name")
            checkpoint.write_bytes(b"unadopted-swap-bytes")
            stream.seek(0)
            self.seen = stream.read()

    runner = Runner()
    module.load_anchored_runner_checkpoint(
        checkpoint,
        expected_sha256=digest,
        allowed_root=store,
        runner=runner,
    )

    assert runner.seen == trusted
    assert checkpoint.read_bytes() == b"unadopted-swap-bytes"
    assert list(run.iterdir()) == [checkpoint]


@pytest.mark.parametrize("resumed_training", [False, True])
def test_pinned_distillation_runner_preserves_load_semantics_on_pathless_bytes(
    tmp_path: Path,
    resumed_training: bool,
) -> None:
    torch = pytest.importorskip("torch")
    runners = pytest.importorskip("rsl_rl.runners")
    module = _load_checkpoint_io()
    store = tmp_path / "logs" / "rsl_rl" / "ur5e_binpick_vision"
    run = store / "_nurec-warmstart-test"
    run.mkdir(parents=True)
    checkpoint = run / "model_0.pt"
    model_state = {"weight": torch.arange(3)}
    optimizer_state = {"state": {"step": 7}}
    torch.save(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "iter": 37,
            "infos": {"mode": "resume" if resumed_training else "teacher"},
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    class StateSink:
        def __init__(self, result: bool = False) -> None:
            self.result = result
            self.seen = None

        def load_state_dict(self, state):
            self.seen = state
            return self.result

    policy = StateSink(resumed_training)
    optimizer = StateSink()

    class FakeRunner:
        def __init__(self) -> None:
            self.alg = type("Algorithm", (), {"policy": policy, "optimizer": optimizer})()
            self.current_learning_iteration = 0

        def load(self, stream):
            assert isinstance(stream, io.BytesIO)
            assert not hasattr(stream, "name")
            return runners.DistillationRunner.load(
                self, stream, map_location="cpu"
            )

    runner = FakeRunner()
    infos = module.load_anchored_runner_checkpoint(
        checkpoint,
        expected_sha256=digest,
        allowed_root=store,
        runner=runner,
    )

    assert torch.equal(policy.seen["weight"], model_state["weight"])
    assert infos == {"mode": "resume" if resumed_training else "teacher"}
    if resumed_training:
        assert optimizer.seen == optimizer_state
        assert runner.current_learning_iteration == 37
    else:
        assert optimizer.seen is None
        assert runner.current_learning_iteration == 0


def test_low_level_checkpoint_modes_are_fail_closed_without_breaking_resume() -> None:
    namespace = _load_pure_train_contract()
    normalize = namespace["_normalize_expected_checkpoint_sha256"]
    validate = namespace["_validate_checkpoint_load_contract"]
    digest = "a" * 64

    assert normalize(digest) == digest
    with pytest.raises(ValueError, match="lowercase hex"):
        normalize("A" * 64)

    assert not validate(
        is_resume=False,
        is_distillation=False,
        expected_sha256=None,
        load_run=None,
        checkpoint_name=None,
    )
    with pytest.raises(ValueError, match="scratch"):
        validate(
            is_resume=False,
            is_distillation=False,
            expected_sha256=digest,
            load_run=None,
            checkpoint_name=None,
        )
    with pytest.raises(ValueError, match="parent-verified"):
        validate(
            is_resume=False,
            is_distillation=True,
            expected_sha256=None,
            load_run="_distill-warmstart-launch",
            checkpoint_name="model_0.pt",
        )
    with pytest.raises(ValueError, match="wrapper-owned seed"):
        validate(
            is_resume=False,
            is_distillation=True,
            expected_sha256=digest,
            load_run="ordinary-run",
            checkpoint_name="model_0.pt",
        )
    assert validate(
        is_resume=False,
        is_distillation=True,
        expected_sha256=digest,
        load_run="_distill-warmstart-launch",
        checkpoint_name="model_0.pt",
    )
    # Canonical explicit resume obtains its trusted digest from the immutable
    # schema-2 receipt later in train.py; it does not require a parent CLI hash.
    assert validate(
        is_resume=True,
        is_distillation=False,
        expected_sha256=None,
        load_run="2026-08-15_00-00-00",
        checkpoint_name="model_100.pt",
    )


def test_phase2_binds_each_exclusive_seed_sha_into_the_child_command() -> None:
    source = _source("Phase2/VisionPick_Train.py")
    builder = source[source.index("def _build_cmd(") : source.index(
        "def _validate_forwarded_args"
    )]
    assert builder.count('"--expected_checkpoint_sha256"') == 3
    assert '"--expected_checkpoint_sha256", distill_checkpoint_sha256' in builder
    assert '"--expected_checkpoint_sha256", nurec_checkpoint_sha256' in builder
    assert 'reserved.add("--expected_checkpoint_sha256")' in source


def test_low_level_load_order_never_reopens_the_authorized_path() -> None:
    source = _source("scripts/rsl_rl/train.py")
    receipt = source.index("resume_binding = verify_training_checkpoint_receipt")
    runner = source.index("runner = OnPolicyRunner")
    anchored = source.index("load_anchored_runner_checkpoint(", runner)
    learn = source.index("runner.learn(", anchored)
    assert receipt < runner < anchored < learn
    assert "runner.load(resume_path)" not in source
    assert "load(stream)" in _source("scripts/checkpoint_io.py")
