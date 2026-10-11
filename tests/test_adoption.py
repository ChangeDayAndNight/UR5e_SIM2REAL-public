from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from deployment.contracts import ContractError
from scripts import adoption
from scripts import adoption_pipeline
from scripts import evaluation_provenance as eval_prov
from scripts.adoption_pipeline import (
    _publish_adoption_no_replace,
    _verify_published_adoption,
    select_student_candidates,
    select_teacher_candidates,
)
from scripts.adoption_selection import load_scalar_series, scalar_at_or_before


def _evaluation_runtime_authority(context: dict) -> dict:
    return {
        "schema_version": 1,
        "record_type": "ur5e_tracked_runtime_authority",
        "purpose": "evaluation",
        "manifest_sha256": "1" * 64,
        "git_commit": context["project_git_commit"],
        "runtime_file_count": 100,
        "runtime_files_sha256": "2" * 64,
        "canonical_yaml_sha256": "3" * 64,
        "canonical_yaml": dict(context["canonical_configs"]),
    }


def _scalar_event_bytes(tag: str, values: list[tuple[int, float]]) -> bytes:
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter

    stream = io.BytesIO()
    writer = RecordWriter(stream)
    writer.write(event_pb2.Event(file_version="brain.Event:2").SerializeToString())
    for step, value in values:
        summary = summary_pb2.Summary(value=[
            summary_pb2.Summary.Value(tag=tag, simple_value=value),
        ])
        writer.write(event_pb2.Event(step=step, summary=summary).SerializeToString())
    writer.flush()
    return stream.getvalue()


def _write_eval(
    root: Path,
    name: str,
    *,
    checkpoint_name: str,
    checkpoint_path: Path,
    training: dict,
    seed: int,
    role: str = "student",
    successes: int = 492,
    infeasible: int = 0,
    official_override: tuple[str, str] | None = None,
) -> Path:
    directory = root / "Phase2" / "Result" / name
    directory.mkdir(parents=True)
    fields = [
        "episode", "grasp_success", "carry_latch_success", "placed_success",
        "target_infeasible", "ep_steps",
    ]
    with (directory / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(adoption.EPISODES_PER_SEED):
            success = int(index < successes)
            row = {
                "episode": index,
                "grasp_success": success,
                "carry_latch_success": success,
                "placed_success": success,
                "target_infeasible": int(index < infeasible),
                "ep_steps": 100,
            }
            if index == 0 and official_override is not None:
                row[official_override[0]] = official_override[1]
            writer.writerow(row)
    rate = successes / adoption.EPISODES_PER_SEED
    stats = {
        "runs": [{
            "n_episodes": adoption.EPISODES_PER_SEED,
            "metrics": {
                metric: {
                    "k": successes,
                    "n": adoption.EPISODES_PER_SEED,
                    "rate": rate,
                    "ci95": [max(0.0, rate - 0.02), min(1.0, rate + 0.02)],
                }
                for metric in adoption.METRICS
            },
        }],
    }
    (directory / "eval_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    task_id = adoption._role_contract(role)["eval_task"]
    (directory / "summary.txt").write_text(
        "\n".join([
            f"task           : {task_id}",
            f"checkpoint     : C:/sealed/{checkpoint_name}",
            f"seed           : {seed}",
        ]) + "\n",
        encoding="utf-8",
    )
    context = training["source"]
    receipt = {
        "schema_version": eval_prov.EVALUATION_RECEIPT_SCHEMA_VERSION,
        "receipt_type": eval_prov.EVALUATION_RECEIPT_TYPE,
        "checkpoint": {
            "path": checkpoint_path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": training["checkpoint_sha256"],
            "training_run_root": training["run_root"],
            "training_receipt": {
                "path": f"{training['run_root']}/run_manifest.json",
                "sha256": training["receipt_sha256"],
            },
        },
        "protocol": adoption._evaluation_protocol(role, seed),
        "evaluation_context": {
            "sha256": eval_prov.canonical_json_sha256(context),
            "value": context,
        },
        "runtime_authority": _evaluation_runtime_authority(context),
        "environment": {
            "forbidden_prefixes": list(eval_prov.FORBIDDEN_ENV_PREFIXES),
            "forbidden_keys": list(eval_prov.FORBIDDEN_ENV_KEYS),
            "observed_forbidden": {},
            "allowed_override_map": {},
        },
        "artifacts": eval_prov.evaluation_artifact_inventory(
            directory, project_root=root, task_id=task_id
        ),
        "created_at": "2026-08-14T00:00:00+00:00",
    }
    (directory / eval_prov.EVALUATION_RECEIPT_NAME).write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def _parse_evaluation(eval_dir: Path, **kwargs) -> dict:
    kwargs["expected_receipt_sha256"] = hashlib.sha256(
        (eval_dir / eval_prov.EVALUATION_RECEIPT_NAME).read_bytes()
    ).hexdigest()
    return adoption.parse_evaluation(eval_dir, **kwargs)


def _training_binding(root: Path, checkpoint: Path, checkpoint_hashes: dict[str, str]) -> dict:
    run_root = checkpoint.parent.resolve().relative_to(root.resolve()).as_posix()
    return {
        "run_root": run_root,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": checkpoint_hashes[checkpoint.name],
        "receipt_sha256": "e" * 64,
        "task_id": "Isaac-BinPick-UR5e-Vision-Distill-v0",
        "mode": "distillation",
        "source": {
            "schema_version": 1,
            "project_git_commit": "a" * 40,
            "canonical_configs": {"configs/common.yaml": "c" * 64},
            "test_context": True,
        },
        "tensorboard_events": {
            "events.out.tfevents.test": {"sha256": "d" * 64, "size": 1},
        },
        "checkpoints": {
            name: {"sha256": digest, "size": (checkpoint.parent / name).stat().st_size}
            for name, digest in checkpoint_hashes.items()
        },
    }


def _training_context(*, test_context: bool = True) -> dict:
    return {
        "schema_version": 1,
        "project_git_commit": "a" * 40,
        "canonical_configs": {"configs/common.yaml": "c" * 64},
        "test_context": test_context,
    }


def _seed_rows(counts: tuple[int, int, int, int]) -> list[dict]:
    return [{
        "seed": seed,
        "episodes": adoption.EPISODES_PER_SEED,
        "metrics": {metric: count for metric in adoption.METRICS},
        "target_infeasible_count": 0,
        "finite": True,
        "stage_invariants": True,
        "artifact": {},
    } for seed, count in zip(adoption.EVALUATION_SEEDS, counts)]


def _candidate(iteration: int, counts: tuple[int, int, int, int], selection: float) -> dict:
    return {
        "checkpoint_name": f"model_{iteration}.pt",
        "checkpoint_sha256": f"{iteration % 16:x}" * 64,
        "iteration": iteration,
        "selection_value": selection,
        "evaluations": _seed_rows(counts),
    }


def test_strict_gate_requires_each_seed_and_pooled_wilson_lower() -> None:
    # 487/512 passes the point threshold, but 1948/2048 does not have a
    # Wilson lower bound of 95%, so the candidate remains rejected.
    pooled_fail = adoption.finalize_candidate(_candidate(100, (487, 487, 487, 487), 1.0))
    assert pooled_fail["passed"] is False
    assert pooled_fail["pooled"]["placed"]["wilson_lower_95"] < 0.95

    accepted = adoption.finalize_candidate(_candidate(200, (492, 492, 492, 492), 2.0))
    assert accepted["passed"] is True
    one_seed_fail = adoption.finalize_candidate(_candidate(300, (486, 510, 510, 510), 3.0))
    assert one_seed_fail["passed"] is False


def test_winner_prefers_worst_seed_before_pooled_and_selection_metric() -> None:
    candidates = [
        _candidate(100, (490, 512, 512, 512), 0.1),
        _candidate(200, (491, 491, 491, 492), 9.0),
        _candidate(300, (490, 500, 500, 500), 8.0),
        _candidate(400, (490, 501, 501, 501), 7.0),
        _candidate(500, (490, 502, 502, 502), 6.0),
    ]
    receipt = adoption.make_adoption_receipt(
        role="teacher",
        run_root="logs/rsl_rl/ur5e_binpick_teacher/run",
        training_receipt_sha256="f" * 64,
        candidates=candidates,
        created_at="2026-08-14T00:00:00+00:00",
    )
    assert receipt["selection"]["checkpoint_name"] == "model_200.pt"
    assert receipt["protocol"]["winner_order"][:3] == [
        "min_seed_placed", "min_seed_carry", "min_seed_grasp",
    ]


def test_student_selection_is_final_plus_four_separated_loss_minima(tmp_path: Path) -> None:
    checkpoints = []
    for iteration in range(199, 3200, 250):
        path = tmp_path / f"model_{iteration}.pt"
        path.touch()
        checkpoints.append(path)
    steps = list(range(3200))
    values = [float((step - 1600) ** 2) for step in steps]
    selected = select_student_candidates(checkpoints, steps, values)
    iterations = [row["iteration"] for row in selected]
    assert iterations[0] == max(int(path.stem.split("_")[1]) for path in checkpoints)
    assert len(iterations) == 5
    assert all(abs(left - right) >= 500 for i, left in enumerate(iterations)
               for right in iterations[i + 1:])


def test_student_selection_accepts_real_checkpoint_inventory_starting_at_zero(
    tmp_path: Path,
) -> None:
    iterations = [*range(0, 4000, 500), 3999]
    checkpoints = []
    for iteration in iterations:
        path = tmp_path / f"model_{iteration}.pt"
        path.touch()
        checkpoints.append(path)
    steps = list(range(4000))
    values = [float((step - 2000) ** 2) for step in steps]

    selected = select_student_candidates(checkpoints, steps, values)

    selected_iterations = [row["iteration"] for row in selected]
    assert selected_iterations[0] == 3999
    assert len(selected_iterations) == 5
    assert all(
        abs(left - right) >= 500
        for index, left in enumerate(selected_iterations)
        for right in selected_iterations[index + 1:]
    )
    model_zero = next(
        row for row in select_student_candidates(checkpoints[:1], steps, values, count=1)
        if row["iteration"] == 0
    )
    assert model_zero["selection_value"] == values[0]


@pytest.mark.parametrize(
    ("steps", "values", "match"),
    [
        ([0, 1, 3], [1.0, 2.0, 3.0], "missing, duplicated, or unordered"),
        ([0, 1, 1, 2], [1.0, 2.0, 3.0, 4.0], "missing, duplicated, or unordered"),
        ([0, 1], [1.0], "length mismatch"),
        ([0, 1], [1.0, float("nan")], "non-finite"),
    ],
)
def test_student_trailing_window_is_exact_gap_free_and_finite(
    tmp_path: Path, steps: list[int], values: list[float], match: str,
) -> None:
    checkpoint = tmp_path / f"model_{steps[-1] if steps else 0}.pt"
    checkpoint.touch()
    with pytest.raises(ContractError, match=match):
        select_student_candidates([checkpoint], steps, values, count=1)


def test_teacher_selection_uses_trailing_median_not_instantaneous_spike(
    tmp_path: Path,
) -> None:
    """★F70 회귀 테스트★ — 순간 스파이크가 아니라 유지되는 성공 구간을 골라야 한다.

    2026-08-23 실사고: 순간 mean_reward 선발이 점화 중간 checkpoint 5개만 게이트에
    올려 채택이 전멸했다(74~81% vs 95.1%). 이 테스트는 스파이크 checkpoint가 후기
    안정 구간을 이기지 못함과, 최종 checkpoint 고정·separation을 함께 회귀한다.
    """
    steps = list(range(4000))
    values = [0.0] * 4000
    values[1500] = 1.0                       # 순간 스파이크 — 옛 선발이 1순위로 고르던 함정
    for step in range(2000, 4000):
        values[step] = 0.9                   # 유지되는 성공 구간
    checkpoints = []
    for iteration in range(0, 4000, 250):
        path = tmp_path / f"model_{iteration}.pt"
        path.touch()
        checkpoints.append(path)
    # 옛 선발 기준(순간값)이라면 스파이크가 전체 1위였음을 명시해 둔다.
    assert scalar_at_or_before(steps, values, 1500) == pytest.approx(1.0)

    selected = select_teacher_candidates(checkpoints, steps, values)

    iterations = [row["iteration"] for row in selected]
    assert iterations[0] == 3750             # 최종 checkpoint는 항상 포함·선두
    assert 1500 not in iterations            # 스파이크 checkpoint는 탈락한다
    assert len(iterations) == len(set(iterations)) == 5
    assert all(
        abs(left - right) >= 500
        for index, left in enumerate(iterations) for right in iterations[index + 1:]
    )
    plateau = [iteration for iteration in iterations if iteration >= 2250]
    assert len(plateau) == 4                 # 유지 구간이 separation 한도까지 상위권을 채운다
    values_by_iteration = {row["iteration"]: row["selection_value"] for row in selected}
    assert values_by_iteration[3750] == pytest.approx(0.9)


def test_scalar_selection_parses_only_receipt_bound_event_bytes(tmp_path: Path) -> None:
    tag = "Loss/behavior"
    event_path = tmp_path / "events.out.tfevents.test"
    payload = _scalar_event_bytes(tag, [(0, 3.0), (1, 2.0), (2, 1.0)])
    event_path.write_bytes(payload)
    inventory = {
        event_path.name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        },
    }

    steps, values = load_scalar_series(tmp_path, tag, inventory)

    assert steps == [0, 1, 2]
    assert values == [3.0, 2.0, 1.0]


def test_scalar_selection_rejects_swap_back_event_handle(
    tmp_path: Path, monkeypatch,
) -> None:
    tag = "Train/mean_reward"
    event_path = tmp_path / "events.out.tfevents.test"
    trusted = _scalar_event_bytes(tag, [(0, 1.0), (1, 2.0)])
    malicious = _scalar_event_bytes(tag, [(0, 9.0), (1, 8.0)])
    assert len(malicious) == len(trusted)
    alternate = tmp_path / "attacker-open-handle"
    alternate.write_bytes(malicious)
    event_path.write_bytes(malicious)
    inventory = {
        event_path.name: {
            "sha256": hashlib.sha256(trusted).hexdigest(),
            "size": len(trusted),
        },
    }
    real_open = Path.open
    restored = False

    def swap_back_open(path: Path, *args, **kwargs):
        nonlocal restored
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == event_path and mode == "rb" and not restored:
            opened_malicious = real_open(alternate, "rb")
            with real_open(event_path, "wb") as stream:
                stream.write(trusted)
            restored = True
            return opened_malicious
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_open)
    with pytest.raises(ContractError, match="bytes differ from their training receipt"):
        load_scalar_series(tmp_path, tag, inventory)

    assert restored is True
    assert hashlib.sha256(event_path.read_bytes()).hexdigest() == inventory[event_path.name]["sha256"]


def test_parse_student_evaluation_rejects_infeasible_target(tmp_path: Path) -> None:
    checkpoint = tmp_path / "logs/rsl_rl/ur5e_binpick_vision/run/model_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training = _training_binding(tmp_path, checkpoint, {checkpoint.name: digest})
    directory = _write_eval(
        tmp_path,
        "eval_bad",
        checkpoint_name="model_100.pt",
        checkpoint_path=checkpoint,
        training=training,
        seed=42,
        infeasible=1,
    )
    with pytest.raises(ContractError, match="target-infeasible"):
        _parse_evaluation(
            directory,
            project_root=tmp_path,
            checkpoint_name="model_100.pt",
            checkpoint_sha256=digest,
            checkpoint_path=checkpoint,
            training=training,
            seed=42,
            role="student",
        )


@pytest.mark.parametrize(
    ("field", "raw", "message"),
    [
        ("episode", "0.9", "exact integer"),
        ("ep_steps", "1.1", "exact integer"),
        ("grasp_success", "1.9", "exact integer"),
        ("target_infeasible", "", "official field is blank"),
    ],
)
def test_parse_evaluation_rejects_fractional_or_blank_official_fields(
    tmp_path: Path, field: str, raw: str, message: str,
) -> None:
    checkpoint = tmp_path / "logs/rsl_rl/ur5e_binpick_vision/run/model_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training = _training_binding(tmp_path, checkpoint, {checkpoint.name: digest})
    directory = _write_eval(
        tmp_path,
        "eval_official_field",
        checkpoint_name=checkpoint.name,
        checkpoint_path=checkpoint,
        training=training,
        seed=42,
        official_override=(field, raw),
    )
    with pytest.raises(ContractError, match=message):
        _parse_evaluation(
            directory,
            project_root=tmp_path,
            checkpoint_name=checkpoint.name,
            checkpoint_sha256=digest,
            checkpoint_path=checkpoint,
            training=training,
            seed=42,
            role="student",
        )


def test_post_receipt_consistent_csv_and_stats_tamper_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "logs/rsl_rl/ur5e_binpick_vision/run/model_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training = _training_binding(tmp_path, checkpoint, {checkpoint.name: digest})
    directory = _write_eval(
        tmp_path,
        "eval_consistent_tamper",
        checkpoint_name=checkpoint.name,
        checkpoint_path=checkpoint,
        training=training,
        seed=42,
        successes=100,
    )
    receipt = json.loads(
        (directory / eval_prov.EVALUATION_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert set(receipt["artifacts"]) == set(eval_prov.OFFICIAL_EVALUATION_ARTIFACTS)
    assert all(set(item) == {"path", "size", "sha256"} for item in receipt["artifacts"].values())

    csv_path = directory / "episodes.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for index, row in enumerate(rows):
        success = str(int(index < 492))
        for field in ("grasp_success", "carry_latch_success", "placed_success"):
            row[field] = success
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    stats_path = directory / "eval_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for metric in adoption.METRICS:
        stats["runs"][0]["metrics"][metric].update({
            "k": 492,
            "rate": 492 / adoption.EPISODES_PER_SEED,
            "ci95": [0.94, 0.98],
        })
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ContractError, match="official artifact path/size/hash drift"):
        _parse_evaluation(
            directory,
            project_root=tmp_path,
            checkpoint_name=checkpoint.name,
            checkpoint_sha256=digest,
            checkpoint_path=checkpoint,
            training=training,
            seed=42,
            role="student",
        )


def test_evaluation_metrics_and_artifact_hashes_share_same_swap_back_snapshot(
    tmp_path: Path, monkeypatch,
) -> None:
    checkpoint = tmp_path / "logs/rsl_rl/ur5e_binpick_vision/run/model_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training = _training_binding(tmp_path, checkpoint, {checkpoint.name: digest})
    directory = _write_eval(
        tmp_path,
        "eval_swap_back",
        checkpoint_name=checkpoint.name,
        checkpoint_path=checkpoint,
        training=training,
        seed=42,
        successes=492,
    )
    receipt_path = directory / eval_prov.EVALUATION_RECEIPT_NAME
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    official_names = ("episodes.csv", "eval_stats.json", "summary.txt")
    consumed = {name: (directory / name).read_bytes() for name in official_names}
    consumed_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in consumed.items()
    }

    alternate = _write_eval(
        tmp_path,
        "eval_swap_back_alternate",
        checkpoint_name=checkpoint.name,
        checkpoint_path=checkpoint,
        training=training,
        seed=42,
        successes=500,
    )
    path_visible = {name: (alternate / name).read_bytes() for name in official_names}
    for name, payload in path_visible.items():
        (directory / name).write_bytes(payload)

    real_sha256_file = adoption.sha256_file
    switched_to_consumed = False

    def switch_before_semantic_snapshot(path):
        nonlocal switched_to_consumed
        if Path(path) == checkpoint and not switched_to_consumed:
            switched_to_consumed = True
            for name, payload in consumed.items():
                (directory / name).write_bytes(payload)
        return real_sha256_file(path)

    # This hook reproduces the obsolete two path-hash passes if they ever
    # return: A is bound first, B supplies the metrics, then A is restored.
    real_artifact_binding = adoption.evaluation_artifact_binding
    binding_calls = 0

    def old_split_artifact_binding(*args, **kwargs):
        nonlocal binding_calls
        binding_calls += 1
        if binding_calls == 1:
            result = real_artifact_binding(*args, **kwargs)
            for name, payload in consumed.items():
                (directory / name).write_bytes(payload)
            return result
        for name, payload in path_visible.items():
            (directory / name).write_bytes(payload)
        return real_artifact_binding(*args, **kwargs)

    monkeypatch.setattr(adoption, "sha256_file", switch_before_semantic_snapshot)
    monkeypatch.setattr(adoption, "evaluation_artifact_binding", old_split_artifact_binding)
    parsed = adoption.parse_evaluation(
        directory,
        project_root=tmp_path,
        checkpoint_name=checkpoint.name,
        checkpoint_sha256=digest,
        checkpoint_path=checkpoint,
        training=training,
        seed=42,
        role="student",
        expected_receipt_sha256=receipt_sha256,
    )

    assert switched_to_consumed is True
    assert binding_calls == 0
    assert parsed["metrics"] == {name: 492 for name in adoption.METRICS}
    for name in official_names:
        assert parsed["artifact"]["files"][name] == consumed_hashes[name]


def test_adoption_verification_reparses_hash_bound_5x4_matrix(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "logs/rsl_rl/ur5e_binpick_vision/2026-08-14_00-00-00"
    run.mkdir(parents=True)
    candidates = []
    checkpoint_hashes = {}
    selection_values = {2200: 5.0, 200: 1.0, 700: 2.0, 1200: 3.0, 1700: 4.0}
    for iteration in selection_values:
        checkpoint = run / f"model_{iteration}.pt"
        checkpoint.write_bytes(f"checkpoint-{iteration}".encode())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checkpoint_hashes[checkpoint.name] = digest
    for iteration, selection_value in selection_values.items():
        checkpoint = run / f"model_{iteration}.pt"
        digest = checkpoint_hashes[checkpoint.name]
        training = _training_binding(tmp_path, checkpoint, checkpoint_hashes)
        evaluations = []
        for seed in adoption.EVALUATION_SEEDS:
            directory = _write_eval(
                tmp_path,
                f"eval_{iteration}_{seed}",
                checkpoint_name=checkpoint.name,
                checkpoint_path=checkpoint,
                training=training,
                seed=seed,
            )
            evaluations.append(_parse_evaluation(
                directory,
                project_root=tmp_path,
                checkpoint_name=checkpoint.name,
                checkpoint_sha256=digest,
                checkpoint_path=checkpoint,
                training=training,
                seed=seed,
                role="student",
            ))
        candidates.append({
            "checkpoint_name": checkpoint.name,
            "checkpoint_sha256": digest,
            "iteration": iteration,
            "selection_value": selection_value,
            "evaluations": evaluations,
        })
    training_receipt_sha = "e" * 64
    receipt = adoption.make_adoption_receipt(
        role="student",
        run_root="logs/rsl_rl/ur5e_binpick_vision/2026-08-14_00-00-00",
        training_receipt_sha256=training_receipt_sha,
        candidates=candidates,
        created_at="2026-08-14T00:00:00+00:00",
    )
    output = tmp_path / "Phase2/Result/2026-08-14_00-00-00/posttrain_evaluation/adoption.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    def fake_training(checkpoint, **_kwargs):
        path = Path(checkpoint)
        return _training_binding(tmp_path, path, checkpoint_hashes)

    monkeypatch.setattr(adoption, "verify_training_checkpoint_receipt", fake_training)
    monkeypatch.setattr(
        adoption,
        "capture_training_context",
        lambda *_args, **_kwargs: _training_context(),
    )
    steps = list(range(2201))
    values = [99.0] * len(steps)
    for iteration, selection_value in selection_values.items():
        for index in range(iteration - 199, iteration + 1):
            values[index] = selection_value
    monkeypatch.setattr(adoption, "load_scalar_series", lambda *_args: (steps, values))
    selected = run / receipt["selection"]["checkpoint_name"]
    binding = adoption.verify_adopted_checkpoint(
        selected, project_root=tmp_path, role="student"
    )
    assert binding["checkpoint_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()

    # Reproduce a path swap-back: the verifier consumes one valid adoption
    # byte representation while the public path is restored to another valid
    # representation before any later path-based hash could run.  The returned
    # authority digest must describe the bytes that were actually parsed.
    trusted_adoption_bytes = output.read_bytes()
    swapped_receipt = json.loads(trusted_adoption_bytes.decode("utf-8"))
    swapped_receipt["created_at"] = "2026-08-14T00:00:01+00:00"
    swapped_adoption_bytes = json.dumps(
        swapped_receipt, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    alternate_adoption = tmp_path / "alternate-adoption-receipt"
    alternate_adoption.write_bytes(swapped_adoption_bytes)
    output.write_bytes(swapped_adoption_bytes)
    real_open = Path.open
    adoption_attacked = False

    def swap_back_adoption_open(path: Path, *args, **kwargs):
        nonlocal adoption_attacked
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == output and mode in {"r", "rb"} and not adoption_attacked:
            adoption_attacked = True
            opened = real_open(alternate_adoption, *args, **kwargs)
            with real_open(output, "wb") as stream:
                stream.write(trusted_adoption_bytes)
            return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_adoption_open)
    swapped_binding = adoption.verify_adopted_checkpoint(
        selected, project_root=tmp_path, role="student"
    )
    assert swapped_binding["sha256"] == hashlib.sha256(swapped_adoption_bytes).hexdigest()
    assert adoption_attacked is True
    assert output.read_bytes() == trusted_adoption_bytes

    monkeypatch.setattr(
        adoption,
        "capture_training_context",
        lambda *_args, **_kwargs: _training_context(test_context=False),
    )
    with pytest.raises(ContractError, match="current Git/config"):
        adoption.verify_adopted_checkpoint(selected, project_root=tmp_path, role="student")
    # ★F95★ 명시적 commit-drift 허용: config가 다르면 commit을 맞게 줘도 거부한다.
    with pytest.raises(ContractError, match="limited to the Git commit"):
        adoption.verify_adopted_checkpoint(
            selected, project_root=tmp_path, role="student", accept_commit_drift_from="a" * 40
        )
    # commit만 다른 현재 컨텍스트: 플래그 없으면 거부, 틀린/짧은 commit이면 거부, 정확한 commit이면 통과+기록.
    monkeypatch.setattr(
        adoption,
        "capture_training_context",
        lambda *_args, **_kwargs: {**_training_context(), "project_git_commit": "b" * 40},
    )
    with pytest.raises(ContractError, match="current Git/config"):
        adoption.verify_adopted_checkpoint(selected, project_root=tmp_path, role="student")
    for bad in ("b" * 40, "a" * 12, ""):
        with pytest.raises(ContractError, match="exact 40-hex commit"):
            adoption.verify_adopted_checkpoint(
                selected, project_root=tmp_path, role="student", accept_commit_drift_from=bad
            )
    drift_binding = adoption.verify_adopted_checkpoint(
        selected, project_root=tmp_path, role="student", accept_commit_drift_from="a" * 40
    )
    # binding ★형상★은 드리프트 여부와 무관하게 7키 고정이다(배포/봉인 검증이 key set을 엄격히 본다).
    assert drift_binding == binding
    assert adoption.last_accepted_context_drift == {
        "adopted_commit": "a" * 40, "current_commit": "b" * 40,
        "accepted_by": "explicit accept_commit_drift_from (F95)",
    }
    monkeypatch.setattr(
        adoption,
        "capture_training_context",
        lambda *_args, **_kwargs: _training_context(),
    )
    assert adoption.verify_adopted_checkpoint(
        selected, project_root=tmp_path, role="student"
    ) == binding

    forged_selection = json.loads(output.read_text(encoding="utf-8"))
    forged_selection["candidates"][1]["selection_value"] = -123.0
    output.write_text(json.dumps(forged_selection, indent=2), encoding="utf-8")
    with pytest.raises(ContractError, match="TensorBoard selection evidence"):
        adoption.verify_adopted_checkpoint(selected, project_root=tmp_path, role="student")
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    forged = json.loads(output.read_text(encoding="utf-8"))
    forged["candidates"][0]["evaluations"][0]["metrics"]["placed"] -= 1
    output.write_text(json.dumps(forged, indent=2), encoding="utf-8")
    with pytest.raises(ContractError, match="metrics differ"):
        adoption.verify_adopted_checkpoint(selected, project_root=tmp_path, role="student")
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    first_eval = tmp_path / receipt["candidates"][0]["evaluations"][0]["artifact"]["directory"]
    with (first_eval / "episodes.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(ContractError, match="changed after adoption"):
        adoption.verify_adopted_checkpoint(selected, project_root=tmp_path, role="student")


def test_evaluation_receipt_rejects_ambient_override_or_checkpoint_rebinding(tmp_path: Path) -> None:
    checkpoint = tmp_path / "logs/rsl_rl/ur5e_binpick_vision/run/model_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training = _training_binding(tmp_path, checkpoint, {checkpoint.name: digest})
    directory = _write_eval(
        tmp_path,
        "eval_receipt_tamper",
        checkpoint_name=checkpoint.name,
        checkpoint_path=checkpoint,
        training=training,
        seed=42,
    )
    receipt_path = directory / eval_prov.EVALUATION_RECEIPT_NAME
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered = json.loads(json.dumps(original))
    tampered["environment"]["allowed_override_map"] = {"BINPICK_STAGE": "2"}
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ContractError, match="ambient override"):
        _parse_evaluation(
            directory,
            project_root=tmp_path,
            checkpoint_name=checkpoint.name,
            checkpoint_sha256=digest,
            checkpoint_path=checkpoint,
            training=training,
            seed=42,
            role="student",
        )
    tampered = json.loads(json.dumps(original))
    tampered["checkpoint"]["path"] = tampered["checkpoint"]["path"].replace(
        "model_100.pt", "model_200.pt"
    )
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ContractError, match="checkpoint/run binding mismatch"):
        _parse_evaluation(
            directory,
            project_root=tmp_path,
            checkpoint_name=checkpoint.name,
            checkpoint_sha256=digest,
            checkpoint_path=checkpoint,
            training=training,
            seed=42,
            role="student",
        )
    tampered = json.loads(json.dumps(original))
    tampered["evaluation_context"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ContractError, match="context/hash differs"):
        _parse_evaluation(
            directory,
            project_root=tmp_path,
            checkpoint_name=checkpoint.name,
            checkpoint_sha256=digest,
            checkpoint_path=checkpoint,
            training=training,
            seed=42,
            role="student",
        )


def test_evaluation_child_environment_removes_every_distribution_override() -> None:
    source = {
        "PATH": "safe",
        "BINPICK_STAGE": "2",
        "BINPICK_TEACHER_CKPT": "forged.pt",
        "BP_CURRICULUM_STEP_OFFSET": "500",
        "PYTHONPATH": "unsafe",
    }
    sanitized = eval_prov.sanitized_evaluation_environment(source)
    assert sanitized["PATH"] == "safe"
    assert sanitized["PYTHONNOUSERSITE"] == "1"
    assert sanitized["OMNI_KIT_ACCEPT_EULA"] == "YES"
    assert "PYTHONPATH" not in sanitized
    assert not any(key.startswith("BINPICK_") for key in sanitized)
    assert "BP_CURRICULUM_STEP_OFFSET" not in sanitized


def _fake_evaluation_process(
    monkeypatch, tmp_path: Path, sentinel_lines: list[str], *, receipt_payload: bytes,
) -> tuple[Path, list[dict]]:
    result_root = tmp_path / "Phase2" / "Result"
    result_root.mkdir(parents=True)
    eval_dir = result_root / "eval_new"
    parsed: list[dict] = []

    class Process:
        def __init__(self) -> None:
            eval_dir.mkdir()
            (eval_dir / eval_prov.EVALUATION_RECEIPT_NAME).write_bytes(receipt_payload)
            self.stdout = iter([line + "\n" for line in sentinel_lines])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(adoption_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(adoption_pipeline, "EVAL_SCRIPT", tmp_path / "Phase1/BinPick_Eval.py")
    monkeypatch.setattr(adoption_pipeline.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(
        adoption_pipeline,
        "parse_evaluation",
        lambda *_args, **kwargs: parsed.append(kwargs) or {"parsed": True},
    )
    return eval_dir, parsed


def test_evaluation_parent_requires_and_anchors_one_receipt_sentinel(
    tmp_path: Path, monkeypatch,
) -> None:
    payload = b'{"sealed":true}\n'
    digest = hashlib.sha256(payload).hexdigest()
    relative = "Phase2/Result/eval_new/evaluation_receipt.json"
    _, parsed = _fake_evaluation_process(
        monkeypatch,
        tmp_path,
        [f"EVALUATION_RECEIPT_OK sha256={digest} path={relative}"],
        receipt_payload=payload,
    )
    result = adoption_pipeline._run_evaluation(
        checkpoint=tmp_path / "model_1.pt",
        checkpoint_sha256="a" * 64,
        training={},
        seed=42,
        role="student",
        logger=adoption_pipeline.Logger(tmp_path / "pipeline.log"),
    )
    assert result == {"parsed": True}
    assert parsed[0]["expected_receipt_sha256"] == digest


def _fake_degenerate_then_ok_process(
    monkeypatch, tmp_path: Path, *, degenerate_attempts: int, receipt_payload: bytes,
) -> tuple[list[str], list[float]]:
    """attempt N까지 EVAL_RENDER_DEGENERATE로 죽고 그다음 성공하는 평가 대역.

    실제 열화 시도의 흔적을 그대로 재현한다: 죽는 시도도 eval_* 디렉터리를 남기고
    (BinPick_Eval은 관측 이전에 mkdir 한다), receipt sentinel 없이 exit 1로 끝난다.
    """
    result_root = tmp_path / "Phase2" / "Result"
    result_root.mkdir(parents=True)
    digest = hashlib.sha256(receipt_payload).hexdigest()
    attempts: list[str] = []
    sleeps: list[float] = []

    class Process:
        def __init__(self) -> None:
            attempt = len(attempts) + 1
            eval_dir = result_root / f"eval_attempt{attempt}"
            eval_dir.mkdir()
            if attempt <= degenerate_attempts:
                attempts.append("degenerate")
                self.stdout = iter([
                    "RuntimeError: EVAL_RENDER_DEGENERATE: depth에 공간 구조가 없다\n",
                ])
                self._code = 1
            else:
                attempts.append("ok")
                (eval_dir / eval_prov.EVALUATION_RECEIPT_NAME).write_bytes(receipt_payload)
                relative = f"Phase2/Result/eval_attempt{attempt}/evaluation_receipt.json"
                self.stdout = iter([f"EVALUATION_RECEIPT_OK sha256={digest} path={relative}\n"])
                self._code = 0

        def wait(self) -> int:
            return self._code

    monkeypatch.setattr(adoption_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(adoption_pipeline, "EVAL_SCRIPT", tmp_path / "Phase1/BinPick_Eval.py")
    monkeypatch.setattr(adoption_pipeline.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(adoption_pipeline.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        adoption_pipeline, "parse_evaluation", lambda *_args, **_kwargs: {"parsed": True}
    )
    return attempts, sleeps


def test_evaluation_retries_after_render_degeneration(tmp_path: Path, monkeypatch) -> None:
    """★F70 동반 회귀★ — 열화 시도는 수치를 버리고 재기동하며, 백오프와
    시도별 before 재스캔(열화 시도가 남긴 eval_* 흡수)이 함께 동작해야 한다."""
    payload = b'{"sealed":true}\n'
    attempts, sleeps = _fake_degenerate_then_ok_process(
        monkeypatch, tmp_path, degenerate_attempts=1, receipt_payload=payload
    )
    result = adoption_pipeline._run_evaluation(
        checkpoint=tmp_path / "model_1.pt",
        checkpoint_sha256="a" * 64,
        training={},
        seed=42,
        role="student",
        logger=adoption_pipeline.Logger(tmp_path / "pipeline.log"),
    )
    assert result == {"parsed": True}
    assert attempts == ["degenerate", "ok"]
    # 열화 시도의 exit 1 / RuntimeError 마커가 최종 판정을 오염시키지 않았고,
    # 재시도 직전 백오프가 정확히 한 번(1회차에는 없음) 걸렸다.
    assert sleeps == [pytest.approx(adoption_pipeline._RENDER_RETRY_BACKOFF_S)]


def test_evaluation_render_degeneration_exhaustion_fails_closed(
    tmp_path: Path, monkeypatch,
) -> None:
    payload = b'{"sealed":true}\n'
    attempts, sleeps = _fake_degenerate_then_ok_process(
        monkeypatch, tmp_path,
        degenerate_attempts=adoption_pipeline._RENDER_RETRY_MAX,
        receipt_payload=payload,
    )
    with pytest.raises(ContractError, match="persisted across"):
        adoption_pipeline._run_evaluation(
            checkpoint=tmp_path / "model_1.pt",
            checkpoint_sha256="a" * 64,
            training={},
            seed=42,
            role="student",
            logger=adoption_pipeline.Logger(tmp_path / "pipeline.log"),
        )
    assert attempts == ["degenerate"] * adoption_pipeline._RENDER_RETRY_MAX
    assert len(sleeps) == adoption_pipeline._RENDER_RETRY_MAX - 1


@pytest.mark.parametrize("mode", ["missing", "duplicate", "mismatch"])
def test_evaluation_parent_rejects_bad_receipt_sentinel(
    tmp_path: Path, monkeypatch, mode: str,
) -> None:
    payload = b'{"sealed":true}\n'
    digest = hashlib.sha256(payload).hexdigest()
    line = (
        "EVALUATION_RECEIPT_OK sha256="
        + (("0" * 64) if mode == "mismatch" else digest)
        + " path=Phase2/Result/eval_new/evaluation_receipt.json"
    )
    lines = [] if mode == "missing" else ([line, line] if mode == "duplicate" else [line])
    _fake_evaluation_process(
        monkeypatch, tmp_path, lines, receipt_payload=payload
    )
    with pytest.raises(ContractError, match="sentinel"):
        adoption_pipeline._run_evaluation(
            checkpoint=tmp_path / "model_1.pt",
            checkpoint_sha256="a" * 64,
            training={},
            seed=42,
            role="student",
            logger=adoption_pipeline.Logger(tmp_path / "pipeline.log"),
        )


def test_each_eval_does_not_rehash_the_full_checkpoint_inventory() -> None:
    provenance = (
        Path(__file__).parents[1] / "scripts" / "evaluation_provenance.py"
    ).read_text(encoding="utf-8")
    begin = provenance[provenance.index("def begin_evaluation"):provenance.index(
        "def finalize_evaluation"
    )]
    finalize = provenance[provenance.index("def finalize_evaluation"):provenance.index(
        "def write_evaluation_receipt"
    )]
    assert "require_all_checkpoints=True" not in begin
    assert "require_all_checkpoints=True" not in finalize
    pipeline = (
        Path(__file__).parents[1] / "scripts" / "adoption_pipeline.py"
    ).read_text(encoding="utf-8")
    verify_run = pipeline[pipeline.index("def _verify_run"):pipeline.index("def run_pipeline")]
    assert "require_all_checkpoints=True" in verify_run


def test_canonical_adoption_output_rejects_result_junction_alias(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "logs/rsl_rl/ur5e_binpick_teacher/run/model_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    real_check = adoption.path_contains_link_or_reparse
    monkeypatch.setattr(
        adoption,
        "path_contains_link_or_reparse",
        lambda path: "top5_evaluation" in str(path) or real_check(path),
    )
    with pytest.raises(ContractError, match="output traverses"):
        adoption.canonical_adoption_path(tmp_path, checkpoint, "teacher")


def test_evaluation_parser_checks_raw_alias_before_resolve(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "logs/rsl_rl/ur5e_binpick_vision/run/model_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training = _training_binding(tmp_path, checkpoint, {checkpoint.name: digest})
    directory = _write_eval(
        tmp_path,
        "eval_alias",
        checkpoint_name="model_100.pt",
        checkpoint_path=checkpoint,
        training=training,
        seed=42,
    )
    real_check = adoption.path_contains_link_or_reparse
    monkeypatch.setattr(
        adoption,
        "path_contains_link_or_reparse",
        lambda path: "eval_alias" in str(path) or real_check(path),
    )
    with pytest.raises(ContractError, match="evaluation root/directory"):
        _parse_evaluation(
            directory,
            project_root=tmp_path,
            checkpoint_name="model_100.pt",
            checkpoint_sha256=digest,
            checkpoint_path=checkpoint,
            training=training,
            seed=42,
            role="student",
        )


def test_final_adoption_publish_is_idempotent_but_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "adoption.json"
    value = {"status": "adopted", "selection": {"checkpoint_name": "model_1.pt"}}
    _publish_adoption_no_replace(path, value)
    original = path.read_bytes()
    _publish_adoption_no_replace(path, value)
    assert path.read_bytes() == original
    with pytest.raises(ContractError, match="refusing overwrite"):
        _publish_adoption_no_replace(path, {"status": "rejected"})
    assert path.read_bytes() == original


def test_final_adoption_postpublish_swap_is_quarantined(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "adoption.json"
    value = {"status": "adopted", "selection": {"checkpoint_name": "model_1.pt"}}
    real_publish = adoption_pipeline.publish_regular_file_no_replace

    def publish_then_swap(source, destination):
        identity = real_publish(source, destination)
        Path(destination).write_bytes(b'{"status":"attacker"}\n')
        return identity

    monkeypatch.setattr(
        adoption_pipeline, "publish_regular_file_no_replace", publish_then_swap
    )
    with pytest.raises(ContractError, match="published adoption receipt hash mismatch"):
        _publish_adoption_no_replace(path, value)
    assert not path.exists()
    quarantined = list(tmp_path.glob(".adoption.json.failed-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b'{"status":"attacker"}\n'


def test_final_adoption_full_postverify_failure_is_quarantined(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "adoption.json"
    value = {"status": "adopted", "selection": {"checkpoint_name": "model_1.pt"}}
    identity = _publish_adoption_no_replace(path, value)
    assert identity is not None
    monkeypatch.setattr(
        adoption_pipeline,
        "verify_adopted_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ContractError("injected full adoption postverify failure")
        ),
    )
    with pytest.raises(ContractError, match="full adoption postverify failure"):
        _verify_published_adoption(
            path,
            value,
            published_identity=identity,
            checkpoint=tmp_path / "model_1.pt",
            project_root=tmp_path,
            role="student",
        )
    assert not path.exists()
    assert list(tmp_path.glob(".adoption.json.failed-*"))


def test_distill_revalidates_teacher_adoption_after_training_before_receipt() -> None:
    source = (Path(__file__).parents[1] / "Phase2" / "VisionPick_Train.py").read_text(
        encoding="utf-8"
    )
    run_call = source.index("rc, run_dir = _run")
    end_check = source.index("end_teacher_adoption = verify_adopted_checkpoint", run_call)
    receipt = source.index("receipt = finalize_training_receipt", end_check)
    assert run_call < end_check < receipt
    assert "end_teacher_adoption != teacher_adoption" in source[end_check:receipt]


def test_distill_revalidates_teacher_source_immediately_before_launch() -> None:
    source = (Path(__file__).parents[1] / "Phase2" / "VisionPick_Train.py").read_text(
        encoding="utf-8"
    )
    pre_context = source.index("pre_run_context = capture_training_context")
    source_gate = source.index('teacher_input["source"] != start_context', pre_context)
    adoption_gate = source.index("pre_run_teacher != teacher_input", source_gate)
    run_call = source.index("rc, run_dir = _run", adoption_gate)
    assert pre_context < source_gate < adoption_gate < run_call


def test_candidate_export_checks_student_adoption_before_loading_weights() -> None:
    source = (Path(__file__).parents[1] / "Phase2" / "Export_Policy.py").read_text(
        encoding="utf-8"
    )
    gate = source.index("student_adoption = verify_adopted_checkpoint")
    load = source.index("loaded = load_anchored_checkpoint", gate)
    manifest = source.index("student_adoption=student_adoption", load)
    assert gate < load < manifest


def test_candidate_export_loads_only_same_opened_adopted_snapshot() -> None:
    root = Path(__file__).parents[1]
    source = (root / "Phase2" / "Export_Policy.py").read_text(
        encoding="utf-8"
    )
    main = source.index("def _main_impl()")
    provenance = source.index("source_training_run = collect_training_run_provenance", main)
    adoption = source.index("student_adoption = verify_adopted_checkpoint", provenance)
    snapshot = source.index("loaded = load_anchored_checkpoint(", adoption)
    helper = (root / "scripts" / "checkpoint_io.py").read_text(encoding="utf-8")
    load = helper.index("torch_module.load(dst")
    manifest = source.index("make_candidate_manifest(", snapshot)
    assert provenance < adoption < snapshot < manifest
    assert load >= 0
    assert "torch.load(ckpt_path" not in source[main:]


def test_evaluation_runner_uses_only_sha_anchored_checkpoint_state() -> None:
    root = Path(__file__).parents[1]
    source = (root / "Phase1" / "BinPick_Eval.py").read_text(encoding="utf-8")
    begin = source.index("evaluation_session = begin_evaluation")
    anchored = source.index("loaded_checkpoint = load_anchored_checkpoint", begin)
    apply_state = source.index("runner.alg.policy.load_state_dict(model_state)", anchored)
    inference = source.index("runner.get_inference_policy", apply_state)
    assert begin < anchored < apply_state < inference
    assert "runner.load(resume_path)" not in source


def test_phase3_policy_runner_is_bound_to_candidate_file_hashes() -> None:
    root = Path(__file__).parents[1]
    rehearsal = (root / "Phase3" / "SimDeploy_Rehearsal.py").read_text(
        encoding="utf-8"
    )
    runner = rehearsal.index("policy = PolicyRunner(")
    assert 'expected_policy_sha256=candidate_manifest["files"]["policy.pt"]' in rehearsal[runner:]
    assert 'expected_obs_spec_sha256=candidate_manifest["files"]["obs_spec.json"]' in rehearsal[runner:]
    policy_io = (root / "common" / "policy_io.py").read_text(encoding="utf-8")
    constructor = policy_io.index("class PolicyRunner:")
    body = policy_io[constructor:]
    assert "torch.jit.load(io.BytesIO(policy_bytes)" in body
    assert "policy_path.read_bytes()" not in body


def test_student_gate_is_the_user_deployment_bar_and_teacher_gate_is_unchanged():
    """★F90(2026-09-19)★ 역할별 게이트: teacher 487/0.95 유지, student 384/0.75(사용자 배포 기준)."""
    assert adoption.ROLE_GATES["teacher"] == (487, 0.95)
    assert adoption.ROLE_GATES["student"] == (384, 0.75)
    assert adoption._protocol("teacher")["min_successes_per_seed"] == 487
    assert adoption._protocol("student")["min_successes_per_seed"] == 384
    assert adoption._protocol("student")["min_wilson_lower"] == 0.75
    # 80% student: teacher 게이트로는 탈락, student 게이트로는 통과
    eighty = _candidate(100, (410, 410, 410, 410), 1.0)
    assert not adoption.finalize_candidate(eighty)["passed"]
    assert adoption.finalize_candidate(eighty, "student")["passed"]
    # 한 seed가 384 미만이면 student도 탈락
    assert not adoption.finalize_candidate(_candidate(200, (383, 500, 500, 500), 1.0), "student")["passed"]
    # 각 seed 384 정확히면 pooled 1536/2048 = 0.75, Wilson 하한 < 0.75 → 탈락(통계적 확신 요구)
    assert not adoption.finalize_candidate(_candidate(300, (384, 384, 384, 384), 1.0), "student")["passed"]
