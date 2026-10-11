"""Phase 1 — teacher 학습 래퍼 (공식 scripts/rsl_rl/train.py에 위임).

목적: Isaac-BinPick-UR5e-Teacher-v0 학습을 headless로 실행하고 콘솔 로그를
      Phase1/logs/에 tee, 학습 후 TensorBoard 스칼라를 CSV/XLSX로 내보낸다.
      실제 학습·env 재실행 가드·경로 해석은 scripts/rsl_rl/train.py(공식 복사본,
      pip 설치된 ur5e_binpick.tasks를 import해 자동 등록)가 담당한다.
      이 래퍼는 isaaclab을 import하지 않는 순수 런처다.
실행 (학습 PC, isaac51 — 사전: pip install -e source/ur5e_binpick):
  python Phase1\\BinPick_Train.py                                   # 본 학습(sim.yaml teacher_num_envs)
  python Phase1\\BinPick_Train.py --num_envs 16 --max_iterations 1 --skip-postprocess  # ★스모크
  python Phase1\\BinPick_Train.py --num_envs 2048                   # VRAM 부족 시 축소
산출:
  - logs/rsl_rl/ur5e_binpick_teacher/<타임스탬프>/  (ckpt + TB — 업스트림 train.py 기본)
  - Phase1/logs/teacher_train_<시각>.log            (콘솔 tee)
  - Phase1/logs/teacher_metrics_<run>.csv / .xlsx   (TB 스칼라 export, best-effort)
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys

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
    # The outer stdlib bootstrap holds all tracked runtime inputs against
    # write/delete before this inner interpreter reaches any task/Hydra import.
    authorize_entrypoint(
        _REPO_ROOT,
        __file__,
        purpose="teacher-training",
        sentinel_stage="training",
    )

from scripts.training_provenance import (
    capture_training_context,
    finalize_training_receipt,
    utc_now,
    verify_training_checkpoint_receipt,
)
from scripts.validate_training_run import TrainingArtifactError, validate_training_run
from deployment.contracts import ContractError

# cp949 콘솔/파이프로 리다이렉트돼도 한국어·기호(★ — ⚠️) print가 죽지 않도록.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TASK = "Isaac-BinPick-UR5e-Teacher-v0"
EXPERIMENT = "ur5e_binpick_teacher"  # BinPickTeacherRunnerCfg.experiment_name과 동일해야 함
METRIC_PREFIX = "teacher_metrics"
LOG_PREFIX = "teacher_train"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # 프로젝트 루트(UR5e_Sim2Real)
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "rsl_rl", "train.py")
LOG_DIR = os.path.join(HERE, "logs")
# 업스트림 train.py는 <cwd>/logs/rsl_rl/<exp>/<ts>/ 에 저장 (cwd=ROOT로 실행)
RUNS_ROOT = os.path.join(ROOT, "logs", "rsl_rl", EXPERIMENT)

OOM_WARN_THRESHOLD = 8192  # 4096(카메라 없음)은 8GB 정상 → 8192↑에서만 경고
_FORWARD_SWITCHES = {"--video", "--export_io_descriptors", "--resume", "--headless"}
_FORWARD_VALUES = {
    "--video_length", "--video_interval", "--run_name", "--load_run", "--checkpoint",
    "--num_envs", "--max_iterations", "--seed",
}


def _require_full_preflight() -> None:
    """Run the canonical non-static preflight even for direct wrapper calls."""
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "preflight_training.py")],
        cwd=ROOT,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONNOUSERSITE": "1"},
        check=False,
    )
    if proc.returncode != 0:
        raise ContractError(f"mandatory training preflight failed: exit={proc.returncode}")


def _default_num_envs() -> str:
    """기본 num_envs = configs/sim.yaml training.teacher_num_envs (단일 소스)."""
    try:
        import yaml

        with open(os.path.join(ROOT, "configs", "sim.yaml"), encoding="utf-8") as f:
            return str(int(yaml.safe_load(f)["training"]["teacher_num_envs"]))
    except Exception as e:
        print(f"[train][warn] configs/sim.yaml에서 teacher_num_envs 읽기 실패({e}) → 4096 사용")
        return "4096"


def _num_envs_of(args: list[str]) -> int | None:
    if "--num_envs" in args:
        try:
            return int(args[args.index("--num_envs") + 1])
        except (ValueError, IndexError):
            return None
    return None


def _iters_of(args: list[str]) -> int | None:
    """--max_iterations 값 (progression 간격 산정용 — 총 클립 ~15개 목표)."""
    if "--max_iterations" in args:
        try:
            return int(args[args.index("--max_iterations") + 1])
        except (ValueError, IndexError):
            return None
    return None


def _steps_per_iter() -> int:
    """1 iteration이 진행하는 env 스텝 수 = rsl_rl num_steps_per_env (단일 소스에서 파싱).

    커리큘럼은 `common_step_counter`(env 스텝) 기준이라 iter↔스텝 환산이 필요하다.
    """
    import re

    cfg = os.path.join(ROOT, "source", "ur5e_binpick", "ur5e_binpick", "tasks",
                       "binpick", "agents", "rsl_rl_ppo_cfg.py")
    m = re.search(r"^\s*num_steps_per_env\s*=\s*(\d+)", open(cfg, encoding="utf-8").read(), re.M)
    if not m:
        raise RuntimeError(f"num_steps_per_env를 찾지 못했습니다: {cfg}")
    return int(m.group(1))


def _curriculum_offset(args: list[str]) -> int:
    """★F44★ resume 시 커리큘럼이 되감기지 않도록 시작 스텝을 산출한다.

    문제: `env.common_step_counter`는 env를 새로 만들 때마다 0에서 시작한다. 따라서
      iter 9550 체크포인트에서 600 iter만 resume하면 pregrasp frac이 ★1.0으로 되돌아가★
      전 env가 '잡힌 채 시작'이 되고, `_real_pick_mask`가 0이라 place 계열 보상이
      ★구조적으로 정확히 0★이 된다 — 베이스라인과 전혀 다른 조건에서 측정하게 된다.
      실제로 F42·F43 검증 run 두 건이 이 함정에 빠져 무효가 됐다(goal_fine 1.87→0.0000).
    해법: 체크포인트 iter × num_steps_per_env를 오프셋으로 주입 → 커리큘럼이 이어진다.
      ★사람이 기억하는 대신 래퍼가 자동 계산한다★는 점이 재발 방지의 핵심.
    """
    import re

    if "--resume" not in args:
        return 0
    it = None
    if "--checkpoint" in args:
        try:
            m = re.search(r"model_(\d+)\.pt", args[args.index("--checkpoint") + 1])
            it = int(m.group(1)) if m else None
        except (IndexError, ValueError):
            it = None
    if it is None:
        print("[train][warn] ⚠️--resume인데 --checkpoint에서 iter를 못 읽었습니다 → 커리큘럼 "
              "오프셋 0. ★커리큘럼이 처음부터 다시 시작하므로 측정이 베이스라인과 비교 불가★")
        return 0
    return it * _steps_per_iter()


def _build_cmd(user_args: list[str]) -> list[str]:
    args = list(user_args)
    _validate_forwarded_args(args)
    if "--task" in args or any(arg.startswith("--task=") for arg in args):
        raise ContractError(f"Teacher wrapper task is fixed to {TASK}")
    if "--experiment_name" in args or any(
        arg.startswith("--experiment_name=") for arg in args
    ):
        raise ContractError(f"Teacher experiment is fixed to {EXPERIMENT}")
    for flag, val in {"--task": TASK, "--num_envs": _default_num_envs()}.items():
        if flag not in user_args:
            args += [flag, val]
    if "--headless" not in args:
        args.append("--headless")  # 학습은 항상 headless (README.md §1/§2)
    # sys.executable로 실행 → train.py의 재실행 가드가 isaac51 python으로 맞춘다.
    return [sys.executable, "-u", TRAIN_SCRIPT, *args]


def _validate_forwarded_args(args: list[str]) -> None:
    index = 0
    while index < len(args):
        token = args[index]
        if token in _FORWARD_SWITCHES:
            index += 1
            continue
        flag, separator, inline = token.partition("=")
        if flag not in _FORWARD_VALUES:
            raise ContractError(f"unsupported/Hydra Teacher argument: {token}")
        if separator:
            if not inline:
                raise ContractError(f"Teacher argument requires a value: {flag}")
            index += 1
            continue
        if index + 1 >= len(args) or args[index + 1].startswith("--"):
            raise ContractError(f"Teacher argument requires a value: {flag}")
        index += 2


def _run(cmd: list[str], log_path: str, curriculum_offset: int = 0) -> tuple[int, str | None]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    # ★F44★ resume 시 커리큘럼 스텝 이어가기 (events_dr.curriculum_step이 소비)
    if curriculum_offset > 0:
        env["BP_CURRICULUM_STEP_OFFSET"] = str(curriculum_offset)
        print(f"[train] ★커리큘럼 오프셋 {curriculum_offset} 스텝★ "
              f"(= {curriculum_offset // _steps_per_iter()} iter) — resume에서 pregrasp frac·"
              f"target radius가 되감기지 않습니다")
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


def _resume_input(args: list[str]) -> list[dict]:
    if "--resume" not in args:
        return []
    if "--load_run" not in args or "--checkpoint" not in args:
        raise ContractError(
            "canonical Teacher resume requires explicit --load_run and --checkpoint"
        )
    try:
        run_name = args[args.index("--load_run") + 1]
        checkpoint_name = args[args.index("--checkpoint") + 1]
    except IndexError as exc:
        raise ContractError("Teacher resume flags require values") from exc
    if any(char in run_name + checkpoint_name for char in "*?[]"):
        raise ContractError("Teacher resume paths must be exact, not glob/regex selectors")
    checkpoint = os.path.join(RUNS_ROOT, run_name, checkpoint_name)
    return [
        verify_training_checkpoint_receipt(
            checkpoint, project_root=ROOT, expected_experiment=EXPERIMENT
        )
    ]


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
    # 이 플래그는 래퍼 전용이다. Hydra/RSL-RL에 전달하면 알 수 없는 override로 실패한다.
    skip_postprocess = "--skip-postprocess" in raw_args
    user_args = [arg for arg in raw_args if arg != "--skip-postprocess"]
    n_envs = _num_envs_of(user_args)
    if n_envs is None:
        n_envs = int(_default_num_envs())
    if n_envs >= OOM_WARN_THRESHOLD:
        print(f"[train][warn] --num_envs={n_envs} 는 RTX 3060 Ti 8GB에서 CUDA OOM 위험이 큽니다.")
        print("              8GB PC라면 중단(Ctrl+C) 후 '--num_envs 4096' 이하로 다시 실행하세요.")
    os.makedirs(LOG_DIR, exist_ok=True)
    print("[train] Phase 1 teacher는 scratch(처음부터) 학습합니다 — 이 ckpt가 Phase 2의 스승이 됩니다.")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{LOG_PREFIX}_{stamp}.log")
    try:
        _require_full_preflight()
        command = _build_cmd(user_args)
        inputs = _resume_input(user_args)
        start_context = capture_training_context(ROOT)
    except ContractError as exc:
        print(f"[train][error] trusted training provenance precheck failed: {exc}")
        return 1
    started_at = utc_now()
    rc, run_dir = _run(command, log_path, _curriculum_offset(user_args))
    completed_at = utc_now()
    print(f"[train] 종료코드 {rc}")
    if rc != 0:
        print("[train][error] 학습 실패 — metrics/report/progression/storyboard 후처리를 실행하지 않습니다.")
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
        end_context = capture_training_context(ROOT)
        receipt = finalize_training_receipt(
            run_dir,
            project_root=ROOT,
            experiment_name=EXPERIMENT,
            task_id=TASK,
            mode="resume" if inputs else "scratch",
            started_at=started_at,
            completed_at=completed_at,
            arguments=user_args,
            start_context=start_context,
            end_context=end_context,
            inputs=inputs,
            runtime_authority=runtime_authority,
        )
    except ContractError as exc:
        print(f"[train][error] run provenance sealing failed: {exc}")
        return 1
    print(f"[store] trusted training receipt: {receipt}")
    if skip_postprocess:
        print("[train] --skip-postprocess — metrics/report/progression/storyboard 후처리를 모두 생략합니다.")
        return 0
    export_metrics(run_dir, LOG_DIR, os.path.basename(os.path.normpath(run_dir)))
    # ★논문급 실험 산출물 자동 생성 (Phase1/Result/<run>/ — 학습곡선 PDF/PNG·보상항 분해·가중치 사본·
    #   요약(txt/json)·재현정보). 모든 학습 run이 논문 바로 사용 가능한 산출물을 남긴다.★
    try:
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "analysis", "exp_report.py"), "--run", run_dir],
            cwd=ROOT, check=False,
        )
    except Exception as e:
        print(f"[report][warn] 논문 산출물 생성 생략(matplotlib 등 확인): {e}")
    # ★논문 설정/ablation 표 자동 생성 (CPU — DR·obs/action·ablation 사다리)★
    _result_run = os.path.join(ROOT, "Phase1", "Result", os.path.basename(os.path.normpath(run_dir)))
    try:
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "analysis", "paper_tables.py"),
             "--out", os.path.join(ROOT, "Phase1", "Result", "paper_tables")],
            cwd=ROOT, check=False,
        )
    except Exception as e:
        print(f"[tables][warn] 설정 표 생성 생략: {e}")
    # ★학습 진척 영상 자동 생성★ (Phase1/Result/<run>/progression/iter_*.mp4) — 체크포인트를
    #   iter 간격으로 순회하며 각 정책의 에피소드를 렌더. 학습이 카메라 없이 끝난 뒤(GPU 여유)
    #   few-env로 소급 렌더하므로 8GB에서도 안전. (사용자 요청: N-iter 간격 에피소드 영상.)
    #   ★--driver(2026-07-13)★: Kit 렌더 데드락 2회 실측 → 청크별 자식 프로세스 + 타임아웃으로
    #   행 방어(행 청크만 스킵하고 전체는 완주). 간격은 총 클립 ~15개가 되도록 iter의 1/15.
    if os.environ.get("BINPICK_NO_PROG", "").strip().lower() in ("1", "true"):
        print("[prog] BINPICK_NO_PROG=1 — 진척 영상 생략(isaacsim.sensors.rtx DLL 로드 실패 회피)")
    else:
        try:
            every = max(200, (_iters_of(user_args) or 2500) // 15)
            subprocess.run(
                [sys.executable, os.path.join(HERE, "BinPick_Progression.py"),
                 "--driver", "--run", run_dir, "--every", str(every)],
                cwd=ROOT, check=False, timeout=1800,  # ★rtx 렌더 행 방어 — progression 무한대기 차단★
            )
        except Exception as e:
            print(f"[prog][warn] 학습 진척 영상 생성 생략: {e}")
    # ★논문 롤아웃 필름스트립 자동 생성★: render_pickplace(GPU — 성공 시퀀스 스토리보드)
    #   → rollout_filmstrip(CPU — 논문 1장 합성). GPU 렌더는 Kit 데드락 대비 subprocess
    #   타임아웃(600s)으로 가드 — 행이 나도 학습 산출물은 이미 확보됨(best-effort).
    _story = os.path.join(_result_run, "storyboard")
    try:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "..", "diagnostics", "render_pickplace.py"),
             "--headless", "--checkpoint", os.path.join(run_dir, f"model_{(_iters_of(user_args) or 2500) - 1}.pt"),
             "--out", _story],
            cwd=ROOT, check=False, timeout=600,
        )
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "analysis", "rollout_filmstrip.py"),
             "--frames", _story, "--out", _result_run],
            cwd=ROOT, check=False,
        )
    except subprocess.TimeoutExpired:
        print("[film][warn] 롤아웃 렌더 타임아웃(600s) — 필름스트립 생략(학습 산출물은 확보됨)")
    except Exception as e:
        print(f"[film][warn] 필름스트립 생성 생략: {e}")
    print("[tip] TensorBoard:  python Phase1\\BinPick_Tensorboard.py")
    print("[tip] 정책 GUI 재생: python Phase1\\BinPick_Env.py --num_envs 8")
    print("[tip] 다음 단계:     Phase2 vision student (teacher ckpt로 distillation)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
