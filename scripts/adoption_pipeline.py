"""Shared, exact post-training evaluation pipeline for Teacher and Student."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time

from deployment.contracts import ContractError, path_contains_link_or_reparse, sha256_file
from deployment.release import (
    _read_authority_regular_file_once,
    _remove_owned_published_file,
    publish_regular_file_no_replace,
)
from scripts.adoption import (
    ADOPTION_NAME,
    EPISODES_PER_SEED,
    EVALUATION_SEEDS,
    canonical_adoption_path,
    make_adoption_receipt,
    parse_evaluation,
    verify_adopted_checkpoint,
)
from scripts.training_provenance import (
    capture_training_context,
    verify_training_checkpoint_receipt,
)
from scripts.evaluation_provenance import (
    EVALUATION_RECEIPT_NAME,
    sanitized_evaluation_environment,
)
from scripts.adoption_selection import (
    checkpoint_iteration,
    load_scalar_series as _load_scalar_series,
    select_student_candidates,
    select_teacher_candidates,
)


ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = ROOT / "Phase1" / "BinPick_Eval.py"
# ★F85 — `\bnan\b`를 제거했다 (2026-09-07 채택 오탐 사고)★
#   [사고] model_24999/seed43 평가가 receipt 봉인까지 정상 완료(exit 0)했는데,
#     BinPick_Eval의 '무조건부 진단 요약'(P13 설계 — 분모 0이면 의도적으로 nan을
#     찍는다: "체류 스텝 중앙 nan" 등)에 `\bnan\b`(IGNORECASE)가 걸려 fatal로
#     오판, 채택 전체가 중단됐다. 성공률 낮은 student(존 진입 0회 시드)에서
#     구조적으로 재발한다.
#   [왜 빼도 되는가] 진짜 NaN 병리는 이 라인 스캔이 아니라 adoption.py의 finite
#     게이트(protocol.require_finite=True)가 receipt에 봉인된 실측 지표값으로
#     검증한다 — 이중 안전망의 정본은 그쪽이다. 여기 남는 패턴은 '프로세스가
#     죽는' 부류(OOM·Traceback 등)만 담당한다.
_ERROR_RE = re.compile(
    r"CUDA out of memory|Traceback|Error executing job|FloatingPointError",
    re.IGNORECASE,
)
_EVALUATION_RECEIPT_OK_RE = re.compile(
    r"^EVALUATION_RECEIPT_OK sha256=([0-9a-f]{64}) path=([^\r\n]+)$"
)

_ROLES = {
    "teacher": {
        "experiment": "ur5e_binpick_teacher",
        "task": "Isaac-BinPick-UR5e-Teacher-Eval-v0",
        # ★F70★ 게이트가 재는 지표로 선발한다. mean_reward는 성공의 조기 종료 때문에
        #   성공률과 ★역상관★한다(선발기 docstring 참조) — 되돌리면 채택이 다시 전멸한다.
        "tag": "Episode_Termination/placed_success",
        "num_envs": 512,
        "default_max_iterations": 25_000,
        "phase": "Phase1",
        "post_dir": "top5_evaluation",
    },
    "student": {
        "experiment": "ur5e_binpick_vision",
        "task": "Isaac-BinPick-UR5e-Vision-Eval-v0",
        "tag": "Loss/behavior",
        "num_envs": 128,
        "default_max_iterations": 20_000,
        "phase": "Phase2",
        "post_dir": "posttrain_evaluation",
    },
}


def _atomic_json(path: Path, value: object) -> None:
    raw = path.absolute()
    if path_contains_link_or_reparse(raw):
        raise ContractError("adoption output path traverses a symlink or reparse point")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path_contains_link_or_reparse(raw):
        raise ContractError("adoption output parent became a symlink or reparse point")
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _adoption_payload(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish_adoption_no_replace(
    path: Path,
    value: dict,
) -> tuple[int, int] | None:
    """Publish the final authority once; never replace an accepted/rejected decision."""
    raw = path.absolute()
    if path_contains_link_or_reparse(raw):
        raise ContractError("adoption publication path traverses a symlink or reparse point")
    payload = _adoption_payload(value)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    if raw.exists():
        try:
            current, _ = _read_authority_regular_file_once(
                raw,
                label="existing adoption receipt",
                expected_sha256=expected_sha256,
            )
        except ContractError as exc:
            raise ContractError(
                "adoption.json already exists; refusing overwrite"
            ) from exc
        if current != payload:
            raise ContractError("adoption.json already exists; refusing overwrite")
        return None
    handle, temporary_name = tempfile.mkstemp(prefix=".adoption-publish-", suffix=".json", dir=raw.parent)
    temporary = Path(temporary_name)
    published_identity: tuple[int, int] | None = None
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        published_identity = publish_regular_file_no_replace(temporary, raw)
        published, _ = _read_authority_regular_file_once(
            raw,
            label="published adoption receipt",
            expected_sha256=expected_sha256,
        )
        if published != payload:
            raise ContractError("published adoption receipt differs from authority")
    except BaseException as exc:
        temporary.unlink(missing_ok=True)
        if published_identity is not None:
            try:
                _remove_owned_published_file(
                    raw, expected_identity=published_identity
                )
            except BaseException as cleanup_exc:
                raise ContractError(
                    "adoption publication failed and owned cleanup failed: "
                    f"{cleanup_exc}"
                ) from exc
        raise
    return published_identity


def _verify_published_adoption(
    path: Path,
    value: dict,
    *,
    published_identity: tuple[int, int] | None,
    checkpoint: Path,
    project_root: Path,
    role: str,
) -> dict:
    """Retain cleanup ownership until full adoption evidence postverification."""

    try:
        verified = verify_adopted_checkpoint(
            checkpoint, project_root=project_root, role=role
        )
        expected_sha256 = hashlib.sha256(_adoption_payload(value)).hexdigest()
        if verified.get("sha256") != expected_sha256:
            raise ContractError(
                "published adoption receipt digest differs from its authority"
            )
        return verified
    except BaseException as exc:
        if published_identity is not None:
            try:
                _remove_owned_published_file(
                    path, expected_identity=published_identity
                )
            except BaseException as cleanup_exc:
                raise ContractError(
                    "adoption postverification failed and owned cleanup failed: "
                    f"{cleanup_exc}"
                ) from exc
        raise


class Logger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def emit(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def subprocess_line(self, line: str) -> None:
        print(line, end="", flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)


def wait_for_pid(pid: int, create_time: float | None, poll_seconds: int, logger: Logger) -> int | None:
    if pid <= 0:
        return None
    import psutil

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.emit(f"training wrapper PID {pid} already exited")
        return None
    if create_time is not None and abs(process.create_time() - create_time) > 2.0:
        raise ContractError("training wrapper PID create time mismatch")
    while True:
        try:
            code = process.wait(timeout=max(1, poll_seconds))
            return int(code) if code is not None else None
        except psutil.TimeoutExpired:
            logger.emit(f"training wrapper PID {pid} is still active")
        except psutil.NoSuchProcess:
            return None


# ── 렌더 열화 대응 (2026-08-22) ─────────────────────────────────────────────
#   vision 평가는 ★연속 기동 시 두 번째부터 depth가 비는★ 계측 결함이 있다(실측: 동일
#   checkpoint/seed가 1회차 placed 34.2% → 2회차 0.0%). BinPick_Eval이 이를 감지해
#   EVAL_RENDER_DEGENERATE로 죽지만, 채택 게이트는 20회 연속 기동이라 그대로는 완주할 수 없다.
#   ⇒ 열화된 시도는 ★수치를 버리고 다시 기동★한다. 열화 수치는 채택 근거가 될 수 없다.
_RENDER_DEGENERATE_MARKER = "EVAL_RENDER_DEGENERATE"
_RENDER_RETRY_MAX = 3
# 재시도 전 대기(초). 직전 Isaac이 GPU를 반납하기 전에 다음 기동을 올리면 렌더러가 열화된
# 상태로 뜨는 것으로 ★추정★된다(기전 미규명 — 실측은 '연속 기동 시 2회차부터 depth 空'뿐).
# 그래서 이 대기는 보증이 아니라 완화책이고, 무효 계측을 실제로 막는 것은 BinPick_Eval의
# 렌더 가드다. 시도마다 대기를 늘린다(30s, 60s).
_RENDER_RETRY_BACKOFF_S = 30.0


def _find_new_eval_dir(result_root: Path, before: set[Path]) -> Path:
    if path_contains_link_or_reparse(result_root.absolute()):
        raise ContractError("evaluation result root traverses a symlink or reparse point")
    raw_candidates = [path.absolute() for path in result_root.glob("eval_*") if path.is_dir()]
    if any(path_contains_link_or_reparse(path) for path in raw_candidates):
        raise ContractError("evaluation result contains a symlink or reparse directory")
    after = {path.resolve() for path in raw_candidates}
    created = after - before
    if len(created) != 1:
        raise ContractError(f"evaluation must create exactly one result directory; found {len(created)}")
    return next(iter(created))


def _run_evaluation(
    *, checkpoint: Path, checkpoint_sha256: str, training: dict,
    seed: int, role: str, logger: Logger,
) -> dict:
    cfg = _ROLES[role]
    result_root = ROOT / cfg["phase"] / "Result"
    if path_contains_link_or_reparse(result_root.absolute()):
        raise ContractError("evaluation result root traverses a symlink or reparse point")
    command = [
        sys.executable,
        "-u",
        str(EVAL_SCRIPT),
        "--task",
        cfg["task"],
        "--num_envs",
        str(cfg["num_envs"]),
        "--n_episodes",
        str(EPISODES_PER_SEED),
        "--seed",
        str(seed),
        "--headless",
        "--checkpoint",
        str(checkpoint),
    ]
    environment = sanitized_evaluation_environment(os.environ)
    code = -1
    saw_error = False
    saw_degenerate = False
    before: set = set()
    receipt_sentinels: list[tuple[str, str]] = []
    for attempt in range(1, _RENDER_RETRY_MAX + 1):
        # 대기는 ★재시도 직전에만★ 한다 — 1회차(정상 경로)는 어떤 지연·외부 조회도 없어야
        #   테스트 대역·20회 연속 기동에 부작용을 얹지 않는다.
        if attempt > 1:
            settle = _RENDER_RETRY_BACKOFF_S * (attempt - 1)
            logger.emit(f"render retry backoff: {settle:.0f}s 대기 후 재기동")
            time.sleep(settle)
        # ★매 시도마다 다시 스캔★한다 — 열화로 버린 시도도 eval_<ts> 디렉터리를 이미
        #   만들어 놓기 때문에(BinPick_Eval은 관측 이전에 mkdir 한다), 루프 밖에서 한 번만
        #   찍어 두면 _find_new_eval_dir이 신규 디렉터리를 둘 이상 보게 된다.
        raw_before = [path.absolute() for path in result_root.glob("eval_*") if path.is_dir()]
        if any(path_contains_link_or_reparse(path) for path in raw_before):
            raise ContractError("evaluation result contains a symlink or reparse directory")
        before = {path.resolve() for path in raw_before}
        logger.emit(
            f"evaluation launch (attempt {attempt}/{_RENDER_RETRY_MAX}): " + " ".join(command)
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        saw_error = False
        saw_degenerate = False
        receipt_sentinels = []
        assert process.stdout is not None
        for line in process.stdout:
            logger.subprocess_line(line)
            saw_degenerate |= _RENDER_DEGENERATE_MARKER in line
            saw_error |= bool(_ERROR_RE.search(line))
            match = _EVALUATION_RECEIPT_OK_RE.fullmatch(line.rstrip("\r\n"))
            if match is not None:
                receipt_sentinels.append((match.group(1), match.group(2)))
        code = process.wait()
        if not saw_degenerate:
            break
        logger.emit(
            f"evaluation attempt {attempt}/{_RENDER_RETRY_MAX}: {_RENDER_DEGENERATE_MARKER} "
            "— 렌더가 열화됐다. ★이 시도의 수치는 버린다★"
        )
    if saw_degenerate:
        raise ContractError(
            f"{_RENDER_DEGENERATE_MARKER} persisted across {_RENDER_RETRY_MAX} attempts: "
            f"{checkpoint.name}/seed{seed} — 이 실행의 계측은 무효다"
        )
    if code != 0 or saw_error:
        raise ContractError(
            f"evaluation failed or emitted a fatal marker: {checkpoint.name}/seed{seed}/exit{code}"
        )
    if len(receipt_sentinels) != 1:
        raise ContractError(
            "evaluation must emit exactly one EVALUATION_RECEIPT_OK sentinel; "
            f"found {len(receipt_sentinels)}"
        )
    eval_dir = _find_new_eval_dir(result_root, before)
    receipt_sha256, receipt_relative = receipt_sentinels[0]
    expected_receipt = (eval_dir / EVALUATION_RECEIPT_NAME).resolve()
    expected_relative = expected_receipt.relative_to(ROOT.resolve()).as_posix()
    if receipt_relative != expected_relative:
        raise ContractError("evaluation receipt sentinel path differs from the new result directory")
    receipt_raw = ROOT / PurePosixPath(receipt_relative)
    if path_contains_link_or_reparse(receipt_raw.absolute()) \
            or not receipt_raw.is_file() or receipt_raw.resolve() != expected_receipt:
        raise ContractError("evaluation receipt sentinel resolves to an unsafe/noncanonical file")
    if sha256_file(receipt_raw) != receipt_sha256:
        raise ContractError("evaluation receipt sentinel SHA differs from the published file")
    return parse_evaluation(
        eval_dir,
        project_root=ROOT,
        checkpoint_name=checkpoint.name,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_path=checkpoint,
        training=training,
        seed=seed,
        role=role,
        expected_receipt_sha256=receipt_sha256,
    )


def _load_state(path: Path, *, role: str, run_root: str, receipt_sha256: str) -> dict:
    if not path.exists():
        return {
            "schema_version": 1,
            "role": role,
            "run_root": run_root,
            "training_receipt_sha256": receipt_sha256,
            "evaluations": {},
        }
    if path_contains_link_or_reparse(path.absolute()):
        raise ContractError("adoption state traverses a symlink or reparse point")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode adoption state: {exc}") from exc
    expected = {
        "schema_version": 1,
        "role": role,
        "run_root": run_root,
        "training_receipt_sha256": receipt_sha256,
    }
    if not isinstance(value, dict) or any(value.get(key) != expected_value for key, expected_value in expected.items()) \
            or not isinstance(value.get("evaluations"), dict):
        raise ContractError("existing adoption state belongs to another run/protocol")
    return value


def _reuse_evaluation(
    stored: object, *, checkpoint: Path, checkpoint_sha256: str, training: dict,
    seed: int, role: str,
) -> dict | None:
    if not isinstance(stored, dict):
        return None
    artifact = stored.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("directory"), str):
        return None
    receipt_sha256 = artifact.get("evaluation_receipt_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(receipt_sha256)) is None:
        return None
    try:
        reparsed = parse_evaluation(
            ROOT / PurePosixPath(artifact["directory"]),
            project_root=ROOT,
            checkpoint_name=checkpoint.name,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_path=checkpoint,
            training=training,
            seed=seed,
            role=role,
            expected_receipt_sha256=str(receipt_sha256),
        )
    except (ContractError, OSError, ValueError):
        return None
    return reparsed if reparsed == stored else None


def _verify_run(
    run_dir: Path, *, role: str, max_iterations: int,
) -> tuple[dict, list[Path], list[int], list[float]]:
    cfg = _ROLES[role]
    if path_contains_link_or_reparse(run_dir.absolute()):
        raise ContractError("posttrain run traverses a symlink or reparse point")
    expected_parent = ROOT / "logs" / "rsl_rl" / cfg["experiment"]
    if run_dir.parent != expected_parent.resolve():
        raise ContractError("posttrain run is outside the canonical experiment store")
    checkpoints = sorted(run_dir.glob("model_[0-9]*.pt"), key=checkpoint_iteration)
    if any(path_contains_link_or_reparse(path.absolute()) for path in checkpoints):
        raise ContractError("training checkpoint set contains a symlink or reparse point")
    if not checkpoints or checkpoint_iteration(checkpoints[-1]) != max_iterations - 1:
        raise ContractError(
            f"run final checkpoint must be exactly model_{max_iterations - 1}.pt"
        )
    receipt = verify_training_checkpoint_receipt(
        checkpoints[-1],
        project_root=ROOT,
        expected_experiment=cfg["experiment"],
        require_tensorboard_events=True,
        require_all_checkpoints=True,
    )
    if capture_training_context(ROOT) != receipt["source"]:
        raise ContractError(
            "post-training evaluation Git/config/Isaac context differs from the "
            "sealed training run"
        )
    if role == "student" and (
        receipt.get("task_id") != "Isaac-BinPick-UR5e-Vision-Distill-v0"
        or receipt.get("mode") != "distillation"
    ):
        raise ContractError("Student adoption accepts only sealed plain distillation runs")
    if role == "teacher" and (
        receipt.get("task_id") != "Isaac-BinPick-UR5e-Teacher-v0"
        or receipt.get("mode") != "scratch"
    ):
        raise ContractError("Teacher adoption accepts only sealed scratch runs")
    steps, values = _load_scalar_series(
        run_dir, cfg["tag"], receipt["tensorboard_events"]
    )
    if steps[-1] < max_iterations - 1:
        raise ContractError("TensorBoard scalar series does not prove full training completion")
    return receipt, checkpoints, steps, values


def run_pipeline(
    *,
    role: str,
    run: Path | str,
    max_iterations: int,
    wait_pid: int = 0,
    wait_create_time: float | None = None,
    poll_seconds: int = 30,
) -> int:
    if role not in _ROLES:
        raise ValueError(role)
    run_raw = Path(run).expanduser().absolute()
    if path_contains_link_or_reparse(run_raw):
        print("[adoption][error] run path traverses a symlink or reparse point")
        return 1
    run_dir = run_raw.resolve()
    cfg = _ROLES[role]
    post_raw = ROOT / cfg["phase"] / "Result" / run_dir.name / cfg["post_dir"]
    if path_contains_link_or_reparse(post_raw.absolute()):
        print("[adoption][error] posttrain output traverses a symlink or reparse point")
        return 1
    post_dir = post_raw.resolve()
    post_dir.mkdir(parents=True, exist_ok=True)
    if path_contains_link_or_reparse(post_raw.absolute()):
        print("[adoption][error] posttrain output became a symlink or reparse point")
        return 1
    logger = Logger(post_dir / "pipeline.log")
    state_path = post_dir / "adoption_state.json"
    try:
        wrapper_code = wait_for_pid(wait_pid, wait_create_time, poll_seconds, logger)
        if wrapper_code not in (None, 0):
            raise ContractError(f"training wrapper failed: exit={wrapper_code}")
        receipt, checkpoints, steps, values = _verify_run(
            run_dir, role=role, max_iterations=max_iterations
        )
        run_manifest = run_dir / "run_manifest.json"
        immutable_manifest_sha256 = sha256_file(run_manifest)
        existing_adoption = canonical_adoption_path(ROOT, checkpoints[-1], role)
        if existing_adoption.exists():
            # A final rejected decision is also immutable.  An adopted decision
            # may be reused only after full raw-evidence revalidation.
            try:
                existing = json.loads(existing_adoption.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ContractError(f"existing adoption.json is unreadable: {exc}") from exc
            selection_existing = existing.get("selection") if isinstance(existing, dict) else None
            if not isinstance(existing, dict) or existing.get("status") != "adopted" \
                    or not isinstance(selection_existing, dict):
                raise ContractError("adoption.json already contains a final non-adopted decision")
            verify_adopted_checkpoint(
                run_dir / str(selection_existing.get("checkpoint_name", "")),
                project_root=ROOT,
                role=role,
            )
            logger.emit(
                f"ADOPTION_OK role={role} checkpoint={selection_existing['checkpoint_name']} reused=true"
            )
            return 0
        selection = (
            select_teacher_candidates(checkpoints, steps, values)
            if role == "teacher"
            else select_student_candidates(checkpoints, steps, values)
        )
        if len(selection) != 5:
            raise ContractError(f"selection produced {len(selection)} candidates; exactly five required")
        state = _load_state(
            state_path,
            role=role,
            run_root=receipt["run_root"],
            receipt_sha256=receipt["receipt_sha256"],
        )
        _atomic_json(post_dir / "selection.json", [{
            "checkpoint_name": row["checkpoint"].name,
            "iteration": row["iteration"],
            "selection_value": row["selection_value"],
        } for row in selection])
        candidate_records = []
        for candidate_index, candidate in enumerate(selection, start=1):
            checkpoint = candidate["checkpoint"].absolute()
            verified = verify_training_checkpoint_receipt(
                checkpoint,
                project_root=ROOT,
                expected_experiment=cfg["experiment"],
                require_tensorboard_events=True,
            )
            checkpoint_sha256 = verified["checkpoint_sha256"]
            seed_rows = []
            state_key = checkpoint.name
            stored_seeds = state["evaluations"].setdefault(state_key, {})
            if not isinstance(stored_seeds, dict):
                raise ContractError("adoption state checkpoint entry is malformed")
            for seed in EVALUATION_SEEDS:
                if sha256_file(run_manifest) != immutable_manifest_sha256:
                    raise ContractError("schema-2 run_manifest.json changed before evaluation")
                if capture_training_context(ROOT) != receipt["source"]:
                    raise ContractError(
                        "Git/config/Isaac context changed or differs from training before evaluation"
                    )
                reused = _reuse_evaluation(
                    stored_seeds.get(str(seed)),
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    training=verified,
                    seed=seed,
                    role=role,
                )
                if reused is not None:
                    logger.emit(f"candidate {candidate_index}/5 {checkpoint.name} seed {seed}: reuse verified")
                    row = reused
                else:
                    logger.emit(f"candidate {candidate_index}/5 {checkpoint.name} seed {seed}: start")
                    row = _run_evaluation(
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_sha256,
                        training=verified,
                        seed=seed,
                        role=role,
                        logger=logger,
                    )
                    stored_seeds[str(seed)] = row
                    _atomic_json(state_path, state)
                seed_rows.append(row)
                if sha256_file(run_manifest) != immutable_manifest_sha256:
                    raise ContractError("schema-2 run_manifest.json changed during evaluation")
                if capture_training_context(ROOT) != receipt["source"]:
                    raise ContractError(
                        "Git/config/Isaac context changed or differs from training during evaluation"
                    )
            candidate_records.append({
                "checkpoint_name": checkpoint.name,
                "checkpoint_sha256": checkpoint_sha256,
                "iteration": candidate["iteration"],
                "selection_value": candidate["selection_value"],
                "evaluations": seed_rows,
            })
        adoption = make_adoption_receipt(
            role=role,
            run_root=receipt["run_root"],
            training_receipt_sha256=receipt["receipt_sha256"],
            candidates=candidate_records,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        adoption_path = canonical_adoption_path(ROOT, selection[0]["checkpoint"], role)
        if adoption_path != post_dir / ADOPTION_NAME:
            raise ContractError("derived adoption output differs from canonical posttrain directory")
        if sha256_file(run_manifest) != immutable_manifest_sha256:
            raise ContractError("schema-2 run_manifest.json changed before adoption publish")
        if capture_training_context(ROOT) != receipt["source"]:
            raise ContractError(
                "Git/config/Isaac context changed or differs from training before adoption publish"
            )
        final_lineage = verify_training_checkpoint_receipt(
            checkpoints[-1],
            project_root=ROOT,
            expected_experiment=cfg["experiment"],
            require_tensorboard_events=True,
            require_all_checkpoints=True,
        )
        if final_lineage != receipt:
            raise ContractError("training checkpoint/event inventory changed during evaluation")
        adoption_identity = _publish_adoption_no_replace(adoption_path, adoption)
        if adoption["status"] != "adopted":
            logger.emit("no checkpoint passed every per-seed and pooled Wilson gate")
            return 1
        selected = adoption["selection"]
        assert isinstance(selected, dict)
        _verify_published_adoption(
            adoption_path,
            adoption,
            published_identity=adoption_identity,
            checkpoint=run_dir / selected["checkpoint_name"],
            project_root=ROOT,
            role=role,
        )
        logger.emit(f"ADOPTION_OK role={role} checkpoint={selected['checkpoint_name']}")
        return 0
    except (ContractError, OSError, ValueError, RuntimeError) as exc:
        logger.emit(f"ADOPTION_FAILED {type(exc).__name__}: {exc}")
        return 1
