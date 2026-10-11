"""Phase 2 — vision student 학습 래퍼 (공식 scripts/rsl_rl/train.py에 위임).

세 경로:
  (기본, stock PPO)  python Phase2\\VisionPick_Train.py
      → --task Isaac-BinPick-UR5e-Vision-v0 (BinPickVisionRunnerCfg)
  (증류, 권장)       python Phase2\\VisionPick_Train.py --distill-from <Phase1 teacher model_*.pt>
      → teacher ckpt를 logs/rsl_rl/ur5e_binpick_vision/_warmstart/model_0.pt로 시드하고
        --task Isaac-BinPick-UR5e-Vision-Distill-v0 (BinPickDistillRunnerCfg)로 실행.
        train.py는 algorithm.class_name=="Distillation"이면 ckpt를 자동 로드한다.
  (NuRec 파인튠)     python Phase2\\VisionPick_Train.py --nurec-finetune <증류 student model_*.pt>
      → ★2026-07-17 B안★ student(StudentTeacher) ckpt를 _nurec_warmstart/model_0.pt로
        시드하고 --task Isaac-BinPick-UR5e-Vision-NuRec-v0(러너 = Distill 재사용)로 실행.
        train.py의 같은 Distillation 자동로드 경로(train.py L236-237/L266-269 —
        get_checkpoint_path(load_run, checkpoint))가 StudentTeacher 전체(teacher 가중치
        포함)를 resume → 같은 teacher로 증류가 '연속'된다. num_envs 기본 4
        (NuRec 볼륨 env 간 무공유 — VRAM 사다리 4→2→1, env cfg 주석 참조).

실제 학습·env 재실행 가드는 train.py(공식 복사본)가 담당한다 — 이 래퍼는 순수 런처.
사전: pip install -e source/ur5e_binpick (학습 PC, isaac51).
산출: logs/rsl_rl/ur5e_binpick_vision/<ts>/ (ckpt+TB) + Phase2/logs/vision_*.log/.csv/.xlsx
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.training_env import configure_utf8_stdio, reexec_in_training_python
reexec_in_training_python(__file__)
configure_utf8_stdio()

from common.runtime_authority import (
    RuntimeAuthorityError,
    authorize_entrypoint,
    verify_runtime_authority_sentinel,
)

if __name__ == "__main__":
    authorize_entrypoint(
        _REPO_ROOT,
        __file__,
        purpose="vision-training",
        sentinel_stage="training",
    )

from scripts.training_provenance import (
    capture_training_context,
    finalize_training_receipt,
    utc_now,
    verify_training_checkpoint_receipt,
)
from scripts.adoption import verify_adopted_checkpoint
from scripts.check_training_disk import DISTILL_MINIMUM_FREE_GIB, require_free_space
from scripts.validate_training_run import TrainingArtifactError, validate_training_run
from deployment.contracts import ContractError, path_contains_link_or_reparse, sha256_file

try:  # cp949 콘솔/파이프에서도 한국어 print가 죽지 않도록
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TASK_PPO = "Isaac-BinPick-UR5e-Vision-v0"
TASK_DISTILL = "Isaac-BinPick-UR5e-Vision-Distill-v0"
TASK_NUREC = "Isaac-BinPick-UR5e-Vision-NuRec-v0"  # ★NuRec 파인튠(2026-07-17 B안)★
EXPERIMENT = "ur5e_binpick_vision"  # PPO/Distill/NuRec 러너 cfg 공용 experiment_name
METRIC_PREFIX = "vision_metrics"
LOG_PREFIX = "vision_train"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "rsl_rl", "train.py")
LOG_DIR = os.path.join(HERE, "logs")
RUNS_ROOT = os.path.join(ROOT, "logs", "rsl_rl", EXPERIMENT)
WARMSTART_CKPT = "model_0.pt"
NUREC_NUM_ENVS = "4"  # ★인자 미지정 시 강제 기본★ — NuRec 볼륨 VRAM(env당 +0.4GB, 무공유)
_FORWARD_SWITCHES = {"--video", "--export_io_descriptors", "--headless", "--enable_cameras"}
_FORWARD_VALUES = {
    "--video_length", "--video_interval", "--run_name", "--num_envs",
    "--max_iterations", "--seed",
}


def _require_full_preflight(*, require_nurec: bool) -> None:
    args = [sys.executable, os.path.join(ROOT, "scripts", "preflight_training.py")]
    if require_nurec:
        args.append("--require-nurec")
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONNOUSERSITE": "1"},
        check=False,
    )
    if proc.returncode != 0:
        raise ContractError(f"mandatory training preflight failed: exit={proc.returncode}")


def _default_num_envs() -> str:
    """기본 num_envs = configs/sim.yaml training.student_num_envs (단일 소스)."""
    try:
        import yaml

        with open(os.path.join(ROOT, "configs", "sim.yaml"), encoding="utf-8") as f:
            return str(int(yaml.safe_load(f)["training"]["student_num_envs"]))
    except Exception as e:
        print(f"[train][warn] configs/sim.yaml에서 student_num_envs 읽기 실패({e}) → 128 사용")
        return "128"


def _pop_flag_value(args: list[str], flag: str) -> tuple[list[str], str | None]:
    args = list(args)
    if flag in args:
        i = args.index(flag)
        val = args[i + 1] if i + 1 < len(args) else None
        del args[i : i + 2]
        return args, val
    return args, None


def _effective_max_iterations(args: list[str]) -> int:
    """Resolve the wrapper-forwarded iteration count without invoking Isaac."""
    values: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--max_iterations":
            if index + 1 >= len(args):
                raise ContractError("--max_iterations requires a value")
            values.append(args[index + 1])
            index += 2
            continue
        if token.startswith("--max_iterations="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise ContractError("--max_iterations may be specified only once")
    if values:
        raw = values[0]
    else:
        try:
            import yaml

            with open(os.path.join(ROOT, "configs", "sim.yaml"), encoding="utf-8") as stream:
                raw = str(yaml.safe_load(stream)["training"]["max_iterations"])
        except Exception as exc:
            raise ContractError(f"cannot resolve default max_iterations: {exc}") from exc
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ContractError("max_iterations must be an integer") from exc
    if str(value) != str(raw).strip() or value <= 0:
        raise ContractError("max_iterations must be a positive canonical integer")
    return value


def _verify_teacher_ckpt(teacher_ckpt: str) -> bool:
    """teacher ckpt의 42D 입력·hidden·7D 출력을 현행 계약과 대조한다.

    검증 자체가 불가능하면 진행하지 않는 fail-closed 게이트다. 구 39D Teacher는 hidden
    구조가 같아도 새 Vision env의 42D teacher 관측과 호환되지 않으므로 반드시 거부한다.
    """
    try:
        for p in (ROOT, os.path.join(ROOT, "source", "ur5e_binpick")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import torch
        from common.teacher_contract import TEACHER_ACTION_DIM, TEACHER_OBS_DIM
        from ur5e_binpick.tasks.binpick.agents.vision_policy import (
            TEACHER_ACTOR_HIDDEN_DIMS,
            infer_mlp_dims_from_state_dict,
        )
    except Exception as e:
        print(f"[distill][error] 사전 형상검증 불가 — 기동 거부: {e}")
        return False
    try:
        loaded = torch.load(teacher_ckpt, map_location="cpu", weights_only=True)
        actor_sd = {
            k[len("actor."):]: v
            for k, v in loaded["model_state_dict"].items()
            if k.startswith("actor.")
        }
        if not actor_sd:
            print(f"[distill][error] ckpt에 actor.* 키가 없음(rsl_rl PPO model_*.pt인지 확인): {teacher_ckpt}")
            return False
        t_obs, t_hidden, t_act = infer_mlp_dims_from_state_dict(actor_sd)
    except Exception as e:
        print(f"[distill][error] teacher ckpt 검사 실패 — 기동 거부: {e}")
        return False
    expected = (TEACHER_OBS_DIM, list(TEACHER_ACTOR_HIDDEN_DIMS), TEACHER_ACTION_DIM)
    if (t_obs, t_hidden, t_act) != expected:
        print("[distill][error] ⚠️ Teacher actor 계약 불일치 — 기동 거부:")
        print(f"  ckpt actor 구조 : {t_obs} → {t_hidden} → {t_act}")
        print(f"  현행 기대 구조  : {TEACHER_OBS_DIM} → {list(TEACHER_ACTOR_HIDDEN_DIMS)} "
              f"→ {TEACHER_ACTION_DIM}")
        print("  → 구 39D checkpoint 사용 금지. 현행 42D Teacher 평가 승자를 지정할 것.")
        return False
    print(f"[distill] teacher ckpt 형상 검증 통과: {t_obs} → {t_hidden} → {t_act}")
    return True


def _seed_warmstart_ckpt(
    ckpt: str, tag: str, *, expected_sha256: str,
) -> tuple[str, Path]:
    """Copy one verified parent stream into an exclusive per-launch seed."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise ContractError("warm-start requires an exact lowercase SHA-256 anchor")
    source_raw = Path(ckpt).expanduser().absolute()
    root_raw = Path(RUNS_ROOT).absolute()
    if path_contains_link_or_reparse(source_raw) or path_contains_link_or_reparse(root_raw):
        raise ContractError("warm-start source/root contains a symlink or reparse point")
    source = source_raw.resolve()
    if not source.is_file():
        raise ContractError(f"warm-start checkpoint is missing: {source}")
    root_raw.mkdir(parents=True, exist_ok=True)
    root = root_raw.resolve()
    if path_contains_link_or_reparse(root_raw):
        raise ContractError("warm-start root became a symlink or reparse point")
    seed_dir = Path(tempfile.mkdtemp(prefix=f"_{tag}-warmstart-", dir=root))
    if seed_dir.parent != root or path_contains_link_or_reparse(seed_dir):
        raise ContractError("exclusive warm-start directory escaped the canonical log root")
    destination = seed_dir / WARMSTART_CKPT
    try:
        with source.open("rb") as src, destination.open("xb") as dst:
            copied_sha256 = hashlib.sha256()
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                copied_sha256.update(block)
                dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        if copied_sha256.hexdigest() != expected_sha256:
            raise ContractError("warm-start opened bytes differ from the verified parent checkpoint")
        if sha256_file(destination) != expected_sha256:
            raise ContractError("warm-start post-copy hash differs from the verified parent checkpoint")
    except BaseException:
        destination.unlink(missing_ok=True)
        seed_dir.rmdir()
        raise
    print(f"[{tag}] exclusive ckpt seed: {destination}")
    return seed_dir.name, seed_dir


def _verify_warmstart_seed(seed_dir: Path, expected_sha256: str) -> Path:
    """Revalidate the exclusive copied checkpoint at the child-launch boundary."""
    root = Path(RUNS_ROOT).absolute().resolve()
    raw = seed_dir.absolute()
    checkpoint = raw / WARMSTART_CKPT
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
        or path_contains_link_or_reparse(raw)
        or path_contains_link_or_reparse(checkpoint)
        or raw.parent.resolve() != root
        or not raw.name.startswith(("_distill-warmstart-", "_nurec-warmstart-"))
        or not checkpoint.is_file()
        or len(list(raw.iterdir())) != 1
        or sha256_file(checkpoint) != expected_sha256
    ):
        raise ContractError("warm-start seed changed or differs from its verified parent")
    return checkpoint


def _remove_warmstart_seed(seed_dir: Path | None) -> None:
    if seed_dir is None:
        return
    root = Path(RUNS_ROOT).absolute().resolve()
    raw = seed_dir.absolute()
    if (
        path_contains_link_or_reparse(raw)
        or raw.parent.resolve() != root
        or not raw.name.startswith(("_distill-warmstart-", "_nurec-warmstart-"))
    ):
        raise ContractError("refusing unsafe warm-start cleanup path")
    children = list(raw.iterdir())
    if len(children) != 1 or children[0].name != WARMSTART_CKPT or not children[0].is_file():
        raise ContractError("warm-start directory contents changed; refusing cleanup")
    children[0].unlink()
    raw.rmdir()


def _verify_student_ckpt(student_ckpt: str) -> bool:
    """NuRec 파인튠 입력이 '증류 student(StudentTeacher)' ckpt인지 best-effort 검사.

    Phase1 PPO ckpt(actor.*)를 잘못 넣으면 DistillationRunner.load에서 늦게 죽는다 —
    여기서 'student.*' 키 존재(rsl_rl StudentTeacher.self.student — vision_policy.py의
    VisionStudentTeacher도 동일 속성)를 미리 확인해 친절히 거부. torch 임포트 불가 시
    생략(최종 검증은 train.py runner.load가 담당).
    """
    try:
        import torch
    except Exception as e:
        print(f"[nurec][warn] 사전 검사 생략(torch 임포트 실패: {e}) — train.py가 최종 검증한다.")
        return True
    try:
        loaded = torch.load(student_ckpt, map_location="cpu", weights_only=True)
        sd = loaded.get("model_state_dict", {})
    except Exception as e:
        print(f"[nurec][error] student ckpt 검사 실패 — 기동 거부: {e}")
        return False
    if not any(k.startswith("student") for k in sd):
        print("[nurec][error] ⚠️ ckpt에 student.* 키가 없음 — 증류 run의 model_*.pt(StudentTeacher)가"
              " 필요하다 (Phase1 teacher PPO ckpt는 --distill-from에 넣을 것). 기동 거부.")
        return False
    return True


def _build_cmd(
    user_args: list[str], distill_from: str | None, nurec_from: str | None, *,
    distill_checkpoint_sha256: str | None = None,
    nurec_checkpoint_sha256: str | None = None,
) -> tuple[list[str] | None, Path | None]:
    args = list(user_args)
    _validate_forwarded_args(args)
    reserved = {"--task", "--experiment_name", "--resume", "--load_run", "--checkpoint"}
    reserved.add("--expected_checkpoint_sha256")
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in reserved:
            raise ContractError(
                f"Vision wrapper owns task/experiment/warm-start flag: {flag}"
            )
    task = TASK_PPO
    if distill_from is not None and nurec_from is not None:
        print("[error] --distill-from과 --nurec-finetune은 동시 사용 불가 — 하나만 지정.")
        return None, None
    seed_dir: Path | None = None
    if distill_from is not None:
        if distill_checkpoint_sha256 is None:
            raise ContractError("Distill warm-start lacks a verified Teacher SHA-256 anchor")
        if not os.path.isfile(distill_from):
            print(f"[distill][error] --distill-from ckpt가 없음: {distill_from}")
            return None, None
        try:
            seed_run, seed_dir = _seed_warmstart_ckpt(
                distill_from, "distill", expected_sha256=distill_checkpoint_sha256
            )
            seeded_checkpoint = _verify_warmstart_seed(
                seed_dir, distill_checkpoint_sha256
            )
            # Inspect the exact adopted bytes that the child will consume.  The
            # original pathname is caller-controlled and may be replaced after
            # provenance verification, so validating it before the anchored
            # copy would leave a swap-back TOCTOU gap.
            if not _verify_teacher_ckpt(str(seeded_checkpoint)):
                _remove_warmstart_seed(seed_dir)
                return None, None
        except BaseException:
            _remove_warmstart_seed(seed_dir)
            raise
        task = TASK_DISTILL
        args += [
            "--load_run", seed_run,
            "--checkpoint", WARMSTART_CKPT,
            "--expected_checkpoint_sha256", distill_checkpoint_sha256,
        ]
    if nurec_from is not None:
        if nurec_checkpoint_sha256 is None:
            raise ContractError("NuRec warm-start lacks a verified Student SHA-256 anchor")
        # ★NuRec 파인튠(--distill-from 미러)★: student ckpt를 시드하고 NuRec 태스크로 전환.
        #   러너 cfg = BinPickDistillRunnerCfg(class_name=="Distillation") → train.py가
        #   --resume 없이도 시드 ckpt를 자동 로드(train.py L236-237/L266-269 소스 확인) —
        #   StudentTeacher 전체(teacher 가중치 포함)가 resume되어 ★증류가 연속★된다.
        if not os.path.isfile(nurec_from):
            print(f"[nurec][error] --nurec-finetune ckpt가 없음: {nurec_from}")
            return None, None
        try:
            seed_run, seed_dir = _seed_warmstart_ckpt(
                nurec_from, "nurec", expected_sha256=nurec_checkpoint_sha256
            )
            seeded_checkpoint = _verify_warmstart_seed(
                seed_dir, nurec_checkpoint_sha256
            )
            if not _verify_student_ckpt(str(seeded_checkpoint)):
                _remove_warmstart_seed(seed_dir)
                return None, None
        except BaseException:
            _remove_warmstart_seed(seed_dir)
            raise
        task = TASK_NUREC
        args += [
            "--load_run", seed_run,
            "--checkpoint", WARMSTART_CKPT,
            "--expected_checkpoint_sha256", nurec_checkpoint_sha256,
        ]
    if "--task" not in args:
        args += ["--task", task]
    if "--num_envs" not in args:
        # NuRec 파인튠은 기본 4 강제(볼륨 VRAM 사다리 4→2→1) — 그 외엔 sim.yaml 단일 소스.
        args += ["--num_envs", NUREC_NUM_ENVS if nurec_from is not None else _default_num_envs()]
    if "--headless" not in args:
        args.append("--headless")
    if "--enable_cameras" not in args:
        # vision 태스크(TiledCamera)는 카메라 렌더 필수 — AppLauncher --enable_cameras 자동 부여
        # (Phase1/BinPick_Progression.py·Phase3/SimDeploy_Rehearsal.py의 in-script 강제와 동일 취지)
        args.append("--enable_cameras")
    return [sys.executable, "-u", TRAIN_SCRIPT, *args], seed_dir


def _validate_forwarded_args(args: list[str]) -> None:
    index = 0
    while index < len(args):
        token = args[index]
        if token in _FORWARD_SWITCHES:
            index += 1
            continue
        flag, separator, inline = token.partition("=")
        if flag not in _FORWARD_VALUES:
            raise ContractError(f"unsupported/Hydra Vision argument: {token}")
        if separator:
            if not inline:
                raise ContractError(f"Vision argument requires a value: {flag}")
            index += 1
            continue
        if index + 1 >= len(args) or args[index + 1].startswith("--"):
            raise ContractError(f"Vision argument requires a value: {flag}")
        index += 2


def _run(cmd: list[str], log_path: str) -> tuple[int, str | None]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    print(f"[train] 실행: {' '.join(cmd)}\n[train] cwd={ROOT}\n[train] log={log_path}")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        run_markers: list[str] = []
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            if line.startswith("RSL_RL_RUN_DIR="):
                run_markers.append(line.split("=", 1)[1].strip())
        proc.wait()
    return proc.returncode, run_markers[0] if len(run_markers) == 1 else None


def export_metrics(run_dir: str, out_dir: str, name: str) -> None:
    """run 폴더의 TB 스칼라를 CSV(utf-8-sig)+XLSX로 내보낸다(best-effort)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import pandas as pd
    except Exception as e:
        print(f"[export] tensorboard/pandas 미설치 → 메트릭 export 생략: {e}")
        return
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"[export] TB 스칼라가 없습니다: {run_dir}")
        return
    data: dict[int, dict[str, float]] = {}
    for tag in tags:
        for ev in ea.Scalars(tag):
            data.setdefault(ev.step, {})[tag] = ev.value
    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    df.index.name = "iteration"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{METRIC_PREFIX}_{name}")
    df.to_csv(base + ".csv", encoding="utf-8-sig")
    print(f"[export] CSV  저장: {base}.csv  ({df.shape[0]} iter × {df.shape[1]} 항목)")
    try:
        df.to_excel(base + ".xlsx", sheet_name="metrics")
        print(f"[export] XLSX 저장: {base}.xlsx")
    except Exception as e:
        print(f"[export] XLSX 저장 실패(openpyxl 필요): {e}")


def main() -> int:
    if not os.path.isfile(TRAIN_SCRIPT):
        print(f"[error] 학습 스크립트가 없습니다: {TRAIN_SCRIPT}")
        return 1
    raw_args = sys.argv[1:]
    skip_postprocess = "--skip-postprocess" in raw_args
    user_args = [arg for arg in raw_args if arg != "--skip-postprocess"]
    user_args, distill_from = _pop_flag_value(user_args, "--distill-from")
    user_args, nurec_from = _pop_flag_value(user_args, "--nurec-finetune")
    # ★F95★ 명시적 teacher commit-drift 허용(이번만): 채택 run의 정확한 40자 commit을 요구한다.
    user_args, accept_teacher_commit = _pop_flag_value(user_args, "--accept-teacher-commit")
    if accept_teacher_commit is None:   # train.ps1 ExtraArgs는 `--flag=value` 형태로만 값을 붙일 수 있다
        inline = [a for a in user_args if a.startswith("--accept-teacher-commit=")]
        if inline:
            accept_teacher_commit = inline[0].split("=", 1)[1]
            user_args = [a for a in user_args if a not in inline]
    if accept_teacher_commit is not None and not distill_from:
        print("[train][error] --accept-teacher-commit 은 --distill-from 과 함께만 쓴다")
        return 1
    if accept_teacher_commit is not None:
        print(f"[train][warn] ★F95★ teacher commit drift 허용 요청: {accept_teacher_commit} "
              "(canonical config/stack 동일 조건에서만 통과 — receipt에 context_drift 기록)")
    teacher_adoption = None
    nurec_adoption = None
    distill_checkpoint_sha256 = None
    nurec_checkpoint_sha256 = None
    try:
        _require_full_preflight(require_nurec=bool(nurec_from))
        _validate_forwarded_args(user_args)
        if distill_from and _effective_max_iterations(user_args) > 1:
            require_free_space(Path(ROOT), DISTILL_MINIMUM_FREE_GIB)
        start_context = capture_training_context(ROOT, require_nurec=bool(nurec_from))
        inputs: list[dict] = []
        if distill_from:
            teacher_input = verify_training_checkpoint_receipt(
                distill_from,
                project_root=ROOT,
                expected_experiment="ur5e_binpick_teacher",
            )
            # Performance approval is deliberately separate from the immutable
            # training receipt.  Seal its hash into the child run input so a
            # copied/non-winning Teacher can never seed Distill.
            teacher_adoption = verify_adopted_checkpoint(
                distill_from, project_root=ROOT, role="teacher",
                accept_commit_drift_from=accept_teacher_commit,
            )
            if teacher_adoption.get("checkpoint_sha256") != teacher_input.get(
                "checkpoint_sha256"
            ):
                raise ContractError("Teacher adoption and training receipt SHA-256 differ")
            distill_checkpoint_sha256 = teacher_adoption["checkpoint_sha256"]
            teacher_input["adoption"] = teacher_adoption
            inputs.append(teacher_input)
        if nurec_from:
            nurec_input = verify_training_checkpoint_receipt(
                nurec_from,
                project_root=ROOT,
                expected_experiment=EXPERIMENT,
            )
            nurec_adoption = verify_adopted_checkpoint(
                nurec_from, project_root=ROOT, role="student"
            )
            if nurec_adoption.get("checkpoint_sha256") != nurec_input.get(
                "checkpoint_sha256"
            ):
                raise ContractError("NuRec adoption and training receipt SHA-256 differ")
            nurec_checkpoint_sha256 = nurec_input["checkpoint_sha256"]
            nurec_input["adoption"] = nurec_adoption
            inputs.append(nurec_input)
    except (ContractError, OSError, RuntimeError) as exc:
        print(f"[train][error] trusted training provenance precheck failed: {exc}")
        return 1
    try:
        cmd, seed_dir = _build_cmd(
            user_args,
            distill_from,
            nurec_from,
            distill_checkpoint_sha256=distill_checkpoint_sha256,
            nurec_checkpoint_sha256=nurec_checkpoint_sha256,
        )
    except ContractError as exc:
        print(f"[train][error] preset identity validation failed: {exc}")
        return 1
    if cmd is None:
        return 1
    os.makedirs(LOG_DIR, exist_ok=True)
    if nurec_from:
        mode = "NuRec 파인튠(실셀 배경, 증류 연속)"
    elif distill_from:
        mode = "distillation(teacher→student)"
    else:
        mode = "stock PPO"
    print(f"[train] Phase 2 vision 학습 모드: {mode}")
    task_id = TASK_NUREC if nurec_from else (TASK_DISTILL if distill_from else TASK_PPO)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{LOG_PREFIX}_{stamp}.log")
    started_at = utc_now()
    try:
        try:
            pre_run_context = capture_training_context(ROOT, require_nurec=bool(nurec_from))
            if pre_run_context != start_context:
                raise ContractError("training context changed before child process launch")
            if distill_from:
                if teacher_input["source"] != start_context:
                    # ★F95★ 명시적 허용 시 commit만 다른 것은 통과(verify_adopted_checkpoint가 검증).
                    if accept_teacher_commit is None or {
                        **teacher_input["source"], "project_git_commit": start_context["project_git_commit"]
                    } != start_context:
                        raise ContractError(
                            "Teacher source context differs from Distill start context"
                        )
                pre_run_teacher = verify_training_checkpoint_receipt(
                    distill_from,
                    project_root=ROOT,
                    expected_experiment="ur5e_binpick_teacher",
                )
                pre_run_teacher["adoption"] = verify_adopted_checkpoint(
                    distill_from, project_root=ROOT, role="teacher",
                    accept_commit_drift_from=accept_teacher_commit,
                )
                if pre_run_teacher != teacher_input:
                    raise ContractError(
                        "Teacher checkpoint/adoption binding changed before child process launch"
                    )
            if nurec_from:
                if nurec_input["source"] != start_context:
                    parent_source = nurec_input["source"]
                    compatible_source = {
                        **parent_source,
                        "optional_assets": start_context["optional_assets"],
                    }
                    if compatible_source != start_context:
                        raise ContractError(
                            "NuRec parent source context differs from the current start context"
                        )
                pre_run_nurec = verify_training_checkpoint_receipt(
                    nurec_from,
                    project_root=ROOT,
                    expected_experiment=EXPERIMENT,
                )
                pre_run_nurec["adoption"] = verify_adopted_checkpoint(
                    nurec_from, project_root=ROOT, role="student"
                )
                if pre_run_nurec != nurec_input:
                    raise ContractError(
                        "NuRec checkpoint/adoption binding changed before child process launch"
                    )
            # _build_cmd has now copied the sealed Teacher into its exclusive
            # warm-start directory.  Recheck at the actual launch boundary so
            # both direct and scripts/train.ps1 launches retain >=25 GiB after
            # that potentially large copy, not merely before it.
            if distill_from and _effective_max_iterations(user_args) > 1:
                require_free_space(Path(ROOT), DISTILL_MINIMUM_FREE_GIB)
            warmstart_sha256 = (
                distill_checkpoint_sha256 if distill_from else nurec_checkpoint_sha256
            )
            if seed_dir is not None:
                if warmstart_sha256 is None:
                    raise ContractError("warm-start launch lacks a verified checkpoint SHA-256")
                _verify_warmstart_seed(seed_dir, warmstart_sha256)
        except (ContractError, OSError, RuntimeError) as exc:
            print(f"[train][error] immediate pre-launch provenance gate failed: {exc}")
            return 1
        rc, run_dir = _run(cmd, log_path)
    finally:
        _remove_warmstart_seed(seed_dir)
    completed_at = utc_now()
    print(f"[train] 종료코드 {rc}")
    if rc != 0:
        print("[train][error] training failed; metrics/report postprocessing is skipped")
        return rc
    if run_dir is None:
        print("[train][error] process exited 0 without exactly one RSL_RL_RUN_DIR receipt")
        return 1
    try:
        runtime_authority = verify_runtime_authority_sentinel("training")
    except RuntimeAuthorityError as exc:
        print(f"[train][error] child runtime-authority sentinel failed: {exc}")
        return 1
    try:
        artifacts = validate_training_run(run_dir, expected_root=RUNS_ROOT)
    except (OSError, TrainingArtifactError) as exc:
        print(f"[train][error] process exited 0 but training artifacts are incomplete: {exc}")
        return 1
    print(
        "[train] TRAINING_ARTIFACTS_OK "
        f"checkpoints={len(artifacts.checkpoints)} params={len(artifacts.parameter_files)} "
        f"tensorboard={len(artifacts.tensorboard_events)}"
    )
    print(f"[store] 학습 산출물: {run_dir}")
    try:
        end_context = capture_training_context(ROOT, require_nurec=bool(nurec_from))
        if distill_from:
            end_teacher_adoption = verify_adopted_checkpoint(
                distill_from, project_root=ROOT, role="teacher",
                accept_commit_drift_from=accept_teacher_commit,
            )
            if end_teacher_adoption != teacher_adoption:
                raise ContractError(
                    "Teacher adoption binding changed while Distill was active"
                )
        if nurec_from:
            end_nurec_adoption = verify_adopted_checkpoint(
                nurec_from, project_root=ROOT, role="student"
            )
            if end_nurec_adoption != nurec_adoption:
                raise ContractError(
                    "Student adoption binding changed while NuRec training was active"
                )
        receipt = finalize_training_receipt(
            run_dir,
            project_root=ROOT,
            experiment_name=EXPERIMENT,
            task_id=task_id,
            mode="nurec" if nurec_from else ("distillation" if distill_from else "ppo"),
            started_at=started_at,
            completed_at=completed_at,
            arguments=raw_args,
            start_context=start_context,
            end_context=end_context,
            inputs=inputs,
            runtime_authority=runtime_authority,
            accept_parent_commit_drift_from=accept_teacher_commit,   # ★F95-b★ 시작 허용과 한 쌍
        )
    except ContractError as exc:
        print(f"[train][error] run provenance sealing failed: {exc}")
        return 1
    print(f"[store] trusted training receipt: {receipt}")
    if skip_postprocess:
        print("[train] --skip-postprocess — metrics/report 후처리를 모두 생략합니다.")
        return 0
    export_metrics(run_dir, LOG_DIR, os.path.basename(os.path.normpath(run_dir)))
    # ★논문급 실험 산출물 자동 생성 (Phase2/Result/<run>/) — Phase1과 동일 규칙★
    try:
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "analysis", "exp_report.py"),
             "--run", run_dir, "--experiment", EXPERIMENT],
            cwd=ROOT, check=False,
        )
    except Exception as e:
        print(f"[report][warn] 논문 산출물 생성 생략(matplotlib 등 확인): {e}")
    print("[tip] TensorBoard:  python Phase2\\VisionPick_Tensorboard.py")
    print("[tip] candidate export: python Phase2\\Export_Policy.py --candidate-only --checkpoint <model_*.pt>")
    print("[tip] final export: README Phase 2의 candidate → capture → parity 순서 참조")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
