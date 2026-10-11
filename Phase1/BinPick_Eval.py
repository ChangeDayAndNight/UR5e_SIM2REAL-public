#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — isaac51 python 자기 재실행 + repo 루트.
#   ① ISAACLAB_ENV_PY→활성 conda→홈 conda 자동탐색으로 isaac51 python 재실행
#   ② repo 루트를 sys.path에 추가 → 확장/common import 정합
#   ③ stdout UTF-8 / EULA / user-site 정리
# ─────────────────────────────────────────────────────────────────────────────
"""Phase 1 — teacher 정책 ★정량 평가★ (순수 자력 파지 성공률·CSV·summary).

목적: 학습된 teacher 정책을 pre-grasp 보조 없는 태스크(Isaac-BinPick-UR5e-Teacher-Eval-v0)
      에서 다수 에피소드 재생하여 실제 pick 성능을 정량화한다. lift/goal '보상'이 아니라
      물리적 성공률(파지·유지·운반)로 환산 — 논문/PROGRESS 보고용 산출물(§5).
      정책 로딩은 scripts/rsl_rl/play.py 패턴을 그대로 재사용(§4-6 검증된 경로).

성공 정의 (base 프레임, 최근접 객체 기준 — reward와 동일 함수 재사용, §4-6 정합):
  · grasp_success : 진짜파지(_grasped_mask: 양손접촉∧상대속도≈0) 도달
  · hold_frac     : 파지 유지 스텝 / 에피소드 스텝
  · place_success : 파지∧들림 상태에서 목표 ≤ goal_radius (carry)
  · ★placed_success (full pick&place 공식)★ : ever(파지∧들림) 래치 후 목표 ≤ place_tol에
    ★정지·무접촉(릴리스)★ 상태 도달 — 보상 placed_released와 동일 정의
실행 (학습 PC, isaac51 — 사용자 실행. GUI 불필요, headless 권장):
  python Phase1\\BinPick_Eval.py --num_envs 256 --n_episodes 500 --headless
  python Phase1\\BinPick_Eval.py --checkpoint logs\\rsl_rl\\ur5e_binpick_teacher\\<run>\\model_2499.pt --headless
  python Phase1\\BinPick_Eval.py --video --num_envs 16          # env0 거동 클립도 녹화
산출: Phase1/Result/eval_<timestamp>/  → episodes.csv + summary.txt (+ videos/ if --video)

⚠️ isaaclab을 import하므로 프리앰블/AppLauncher 이후에만 import한다.
"""
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import (
    configure_utf8_stdio,
    preload_native_training_modules,
    reexec_in_training_python,
)

reexec_in_training_python(__file__)
configure_utf8_stdio()

from common.runtime_authority import (
    authorize_entrypoint,
    emit_runtime_authority_sentinel,
)

if __name__ == "__main__":
    authorize_entrypoint(
        _REPO_ROOT,
        __file__,
        purpose="evaluation",
        sentinel_stage="evaluation",
    )

preload_native_training_modules()



_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
# scripts/rsl_rl 은 play.py가 cli_args를 top-level로 import하므로 sys.path에 추가.
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    # ★2026-08-10(F54) h5py 선로드 제거★ — 실측상 ★이 import가 Isaac 센서 DLL을 깨뜨린다★.
    #   h5py 3.16.0(HDF5 2.0.0)이 Isaac 번들 hdf5.dll(1.x)보다 먼저 로드되면
    #   generic_mo_io.dll이 WinError 127로 실패하고 GUI가 access violation으로 죽는다.
    #   (구 h5py에서는 반대로 작동해 워크어라운드였다 — 근거·측정 = Failure_Check F54)
    pass
except Exception:
    pass

# ═══════════════════ AppLauncher (isaaclab import은 이 아래에서만) ═══════════════════
import argparse

from isaaclab.app import AppLauncher

import cli_args  # scripts/rsl_rl/cli_args.py — play.py와 동일 (RSL-RL 인자)  # isort: skip

_TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"

parser = argparse.ArgumentParser(description="UR5e teacher/vision 정책 정량 평가 (순수 자력 파지 성공률).")
# ★2026-07-13 태스크 인자화★ — Phase2 student(depth 관측) 정량 평가도 같은 지표·코드로:
#   --task Isaac-BinPick-UR5e-Vision-Eval-v0 (증류 student ckpt, 보조 전부 off + TiledCamera)
parser.add_argument("--task", type=str, default=_TASK,
                    help=f"평가 태스크 (기본 {_TASK} / vision: Isaac-BinPick-UR5e-Vision-Eval-v0)")
parser.add_argument("--num_envs", type=int, default=256, help="병렬 env 수(통계용 — 기본 256).")
parser.add_argument("--n_episodes", type=int, default=500, help="집계할 최소 완료 에피소드 수.")
parser.add_argument("--max_steps", type=int, default=6000, help="안전 상한(무한루프 방지).")
parser.add_argument("--goal_radius", type=float, default=0.08, help="carry 성공 반경[m] (목표까지 거리).")
parser.add_argument("--place_tol", type=float, default=0.03,
                    help="★full place 성공 반경[m]★ — 릴리스(무접촉)·정지 상태로 목표 도달(보상 placed와 동일 정의).")
parser.add_argument("--video", action="store_true", default=False, help="env0 거동 클립도 녹화.")
parser.add_argument("--video_length", type=int, default=300, help="녹화 길이[steps].")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="fabric 비활성(USD I/O).")
# ★--seed 신설(2026-07-29, F26 대응)★ — 동일 조건 단일 run의 성공률은 run 간 변동에 묻힌다.
#   ckpt 채택은 ★독립 seed run 여러 개 → run 간 mean±std★로 판정해야 한다.
#   cli_args.parse_rsl_rl_cfg가 hasattr(args_cli,"seed")를 보고 agent_cfg.seed에 반영하며,
#   아래에서 env_cfg.seed에도 같은 값을 넣어 물리·DR 샘플링까지 독립시킨다.
parser.add_argument("--seed", type=int, default=None,
                    help="평가 seed (미지정=학습 cfg 기본값 42, -1=무작위). F26: 채택 판정은 다중 seed로.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if "Vision" in args_cli.task:
    args_cli.enable_cameras = True  # TiledCamera env 기동 필수 (student 관측 = depth)
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────── 이하 sim 기동 후 ───────────────────────────
import csv
import subprocess as _sp
from datetime import datetime
from pathlib import Path, PurePosixPath

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401  (import 시 gym.register)

# ★reward와 동일한 최근접-객체/파지 게이트 로직 재사용 (프레임·정의 정합, §4-6)★
#   _grasped_mask: STAGE 3 파지 성립(양손접촉∧상대속도≈0) — z-기반 지표의 저글링 허수를 걸러
#   '진짜 파지' 지표를 병행 보고한다(STAGE 0 프로브 교훈: 이전 grasp 100%는 무파지 비행이었음).
from ur5e_binpick.tasks.binpick.binpick_env_cfg import (
    MIN_LIFT,
    _any_finger_contact,
    _grasped_mask,
    _nearest_object,
    _nearest_object_vel,
    placed_success_mask,
    PLACED_MAX_SPEED,
    PLACED_MIN_CONSECUTIVE_STEPS,
)
from ur5e_binpick.tasks.binpick.mdp.scene import (
    BELT_CENTER_X, BELT_CENTER_Y, BELT_SIZE, OBJECT_SIZE,
)
from common.config_io import configs_dir, load_yaml
from scripts.evaluation_provenance import (
    EVALUATION_RECEIPT_NAME,
    begin_evaluation,
    evaluation_artifact_inventory,
    finalize_evaluation,
    write_evaluation_receipt,
)
from scripts.checkpoint_io import load_anchored_checkpoint
from deployment.contracts import sha256_file

# ★F46★ 타겟 안착 높이 — 큐브를 실제로 내려놓았는지 판별하는 기준(단일 소스: configs/common.yaml).
#   ⚠️ 이 값은 MIN_LIFT보다 낮다 → 내려놓으면 lift(w5)가 끊긴다. '공중 정지' 국소최적의 근거.
_TARGET_Z = float(load_yaml(configs_dir() / "common.yaml")["target"]["sample_range"]["z"][0])

# ★★2026-08-11(P13) 지지면 사각형 — 타겟이 '놓을 표면이 있는 곳'인지 판정★★
#   [사고 F58] 목표거리 역커리큘럼이 타겟을 클램프 없이 물체 주변 반경으로 뿌려 ★벨트 밖★으로
#     내보냈다. 큐브가 얹힐 수 있는 면은 벨트뿐인데(작업대는 시각 전용), placed는
#     |z − z_target| < 5mm를 요구하므로 그런 타겟은 ★원천 불가★다. r=0.35에서 36.5%.
#   [장치] eval이 에피소드마다 이 판정을 기록하면, 같은 결함이 재발해도 ★성공률 하락 원인이
#     '학습 실패'인지 '불가능 표본'인지 즉시 갈린다★ (그때는 이걸 몰라 3.5시간을 태웠다).
_RELEASE_RADIUS = 0.12   # release_bias RewTerm의 radius와 동일 — 지급 창의 분모를 맞춘다
_HALF = OBJECT_SIZE / 2.0
_SUPPORT_X = (BELT_CENTER_X - BELT_SIZE[0] / 2.0 + _HALF, BELT_CENTER_X + BELT_SIZE[0] / 2.0 - _HALF)
_SUPPORT_Y = (BELT_CENTER_Y - BELT_SIZE[1] / 2.0 + _HALF, BELT_CENTER_Y + BELT_SIZE[1] / 2.0 - _HALF)


def _algorithm_class_from_training_receipt(session: dict) -> str | None:
    """학습 receipt에 봉인된 `params/agent.yaml`에서 `algorithm.class_name`을 복원한다.

    ★왜 환경변수가 아니라 receipt인가★
      `BINPICK_AUX=1`은 학습에서 알고리즘을 DistillationAux로 바꾸는데, 채택 파이프라인은
      평가 조건 위조를 막으려고 `BINPICK_` 접두 환경변수를 ★전부 삭제★한다. 그 결과 aux로
      학습한 student는 평가에서 다른 알고리즘으로 재구성되어 ckpt를 로드할 수 없었다 =
      학습은 되는데 ★채택이 원천 불가★한 카드였다.
      알고리즘 클래스는 실행 시 선택지가 아니라 ★학습된 사실★이다. 그러므로 봉인된
      학습 산출물에서 읽는다 — 소독기는 그대로 두고(위조 방지 유지), 값은 해시 사슬로
      인증한다: receipt_sha256 → run_manifest.json → artifacts.parameters → agent.yaml.

    반환이 None이면 receipt가 알고리즘을 말하지 않는 것이므로 호출측이 cfg 기본값을 쓴다.
    """
    import json

    training = session["training"]
    root = Path(session["root"])
    run_root = str(training.get("run_root", ""))
    if not run_root or PurePosixPath(run_root).is_absolute() or ".." in PurePosixPath(run_root).parts:
        raise RuntimeError("training receipt run_root is missing or unsafe")
    run_dir = root / PurePosixPath(run_root)

    manifest = run_dir / "run_manifest.json"
    if sha256_file(manifest) != training.get("receipt_sha256"):
        raise RuntimeError("run_manifest.json differs from the verified training receipt hash")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    parameters = data.get("artifacts", {}).get("parameters", {})
    expected = parameters.get("params/agent.yaml")
    if not isinstance(expected, str):
        raise RuntimeError("training receipt does not seal params/agent.yaml")

    agent_yaml = run_dir / "params" / "agent.yaml"
    if sha256_file(agent_yaml) != expected:
        raise RuntimeError("params/agent.yaml differs from its sealed receipt hash")
    trained = load_yaml(agent_yaml)
    algorithm = trained.get("algorithm") if isinstance(trained, dict) else None
    name = algorithm.get("class_name") if isinstance(algorithm, dict) else None
    return str(name) if isinstance(name, str) and name else None


def _create_evaluation_output_directory(phase_dir: str) -> str:
    """Atomically reserve one canonical eval_<timestamp> result directory."""
    result_dir = os.path.join(os.path.abspath(phase_dir), "Result")
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"canonical evaluation result store is missing: {result_dir}")
    out_dir = os.path.join(
        result_dir,
        "eval_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    # This directory is an immutable official-artifact namespace.  os.mkdir is
    # intentionally exclusive: a same-second collision or pre-created path
    # must fail closed rather than reuse and overwrite an earlier evaluation.
    os.mkdir(out_dir)
    return out_dir


def _target_infeasible(target_xy, tol: float):
    """타겟이 지지 사각형에서 tol보다 멀면 ★placed 원천 불가★ (torch bool 텐서 반환)."""
    dx = (torch.clamp(_SUPPORT_X[0] - target_xy[:, 0], min=0.0)
          + torch.clamp(target_xy[:, 0] - _SUPPORT_X[1], min=0.0))
    dy = (torch.clamp(_SUPPORT_Y[0] - target_xy[:, 1], min=0.0)
          + torch.clamp(target_xy[:, 1] - _SUPPORT_Y[1], min=0.0))
    return torch.sqrt(dx * dx + dy * dy) > tol


def _binpick_env() -> str:
    """★P13★ 이 eval이 어떤 BINPICK_* 설정에서 측정됐는지 — 분포가 다르면 비교가 무효다."""
    kv = {k: v for k, v in os.environ.items() if k.startswith("BINPICK_")}
    return ", ".join(f"{k}={v}" for k, v in sorted(kv.items())) or "(없음)"


def _git_hash() -> str:
    try:
        return _sp.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    # ── cfg / checkpoint ──
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # ★env 쪽 seed도 함께 이동★ — agent_cfg.seed만 바꾸면 정책 초기화만 달라지고 물리·DR
    #   샘플링은 동일해져서 '독립 run'이 되지 않는다(F26의 상관 문제를 그대로 재현하게 된다).
    if getattr(args_cli, "seed", None) is not None and hasattr(env_cfg, "seed"):
        env_cfg.seed = agent_cfg.seed
    log_root = os.path.abspath(os.path.join(_REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        from isaaclab.utils.assets import retrieve_file_path

        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[eval] 체크포인트: {resume_path}")

    # Adoption evaluations are provenance-bearing measurements. Fail before
    # constructing the environment unless the original checkpoint, clean
    # source/config/stack, and empty override environment match its schema-2
    # training receipt exactly.
    evaluation_session = begin_evaluation(
        project_root=_REPO_ROOT,
        checkpoint_path=resume_path,
        protocol={
            "task_id": args_cli.task,
            "seed": int(agent_cfg.seed),
            "episodes": int(args_cli.n_episodes),
            "num_envs": int(args_cli.num_envs),
            "max_steps": int(args_cli.max_steps),
            "goal_radius_m": float(args_cli.goal_radius),
            "place_tolerance_m": float(args_cli.place_tol),
            "headless": bool(args_cli.headless),
            "video": bool(args_cli.video),
            "disable_fabric": bool(args_cli.disable_fabric),
        },
    )
    emit_runtime_authority_sentinel(
        "evaluation",
        required_configs={"configs/common.yaml", "configs/dr.yaml", "configs/sim.yaml"},
    )

    # ── env + 정책 (play.py 패턴) ──
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # ★산출물 위치 = Phase<N>/Result/ (§5 experiment→phase 매핑: teacher→Phase1, vision→Phase2)★
    _phase_dir = os.path.join(_REPO_ROOT, "Phase2") if "Vision" in args_cli.task else _HERE
    out_dir = _create_evaluation_output_directory(_phase_dir)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(out_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ★러너 분기(2026-07-18 — rsl_rl 3.1.2 포팅 미결 해소)★: 증류 태스크(agent cfg
    #   class_name="DistillationRunner")의 ckpt는 DistillationRunner로 로드해야 한다 —
    #   OnPolicyRunner 네임스페이스에는 Distillation 알고리즘이 없어 eval(class_name)이
    #   NameError. DistillationRunner ⊂ OnPolicyRunner라 load/get_inference_policy 동일.
    # ★★F72 — 알고리즘은 ★학습된 사실★이다: receipt에서 복원해 cfg를 덮어쓴다★★
    #   [왜] BINPICK_AUX=1로 학습한 student(DistillationAux)는 채택 평가에서 소독기가
    #     BINPICK_ 접두를 지우는 바람에 Distillation으로 재구성되어 ckpt 로드에 실패했다
    #     = 학습은 되는데 채택이 원천 불가. 소독기는 위조 방지를 위해 그대로 두고, 값만
    #     해시 사슬로 인증해 가져온다(헬퍼 docstring 참조).
    _trained_algorithm = _algorithm_class_from_training_receipt(evaluation_session)
    if _trained_algorithm is not None:
        _cfg_algorithm = getattr(agent_cfg, "algorithm", None)
        _declared = getattr(_cfg_algorithm, "class_name", None)
        if _cfg_algorithm is not None and _declared != _trained_algorithm:
            print(
                f"[eval] 알고리즘을 학습 receipt대로 복원: {_declared} → {_trained_algorithm}"
            )
            _cfg_algorithm.class_name = _trained_algorithm
    _runner_cls = (
        DistillationRunner
        if getattr(agent_cfg, "class_name", "OnPolicyRunner") == "DistillationRunner"
        else OnPolicyRunner
    )
    runner = _runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_checkpoint = load_anchored_checkpoint(
        evaluation_session["checkpoint"],
        expected_sha256=evaluation_session["training"]["checkpoint_sha256"],
        allowed_root=Path(_REPO_ROOT) / "logs" / "rsl_rl",
        torch_module=torch,
    )
    model_state = loaded_checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise RuntimeError("verified evaluation checkpoint has no model_state_dict mapping")
    # Evaluation needs only policy weights.  Calling runner.load(path) would
    # reopen a mutable pathname after provenance verification and reintroduce a
    # swap-back race; apply the exact same-open deserialized state directly.
    runner.alg.policy.load_state_dict(model_state)
    del model_state, loaded_checkpoint
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dev = env.unwrapped.device
    n = env.unwrapped.num_envs
    # ── 에피소드별 성공 누적기 (env 병렬) ──
    ever_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    lifted_steps = torch.zeros(n, dtype=torch.long, device=dev)
    ep_steps = torch.zeros(n, dtype=torch.long, device=dev)
    best_gd = torch.full((n,), float("inf"), device=dev)  # 들린 상태에서의 최소 목표거리
    # ★진짜 파지 지표(STAGE 3)★ — 양손접촉∧상대속도≈0 (저글링 배제)
    ever_grasped = torch.zeros(n, dtype=torch.bool, device=dev)
    grasped_steps = torch.zeros(n, dtype=torch.long, device=dev)
    best_gd_true = torch.full((n,), float("inf"), device=dev)  # 파지∧들림 상태의 최소 목표거리
    # ★Full Pick&Place(2026-07-10)★ — placed = ever(파지∧들림) 래치 후, 타겟 place_tol 내에
    #   ★정지·무접촉(릴리스)★ 상태 도달. ⚠️ 보상 placed_released와 '순간 조건'(tol·정지·무접촉)은
    #   동일하나 래치 정의가 다르다(2026-07-11 검토 명확화): env _lifted_latch = 배달 래치
    #   (¬파지 10스텝 ∧ 타겟>0.10m면 해제), 여기 lift_latch = 에피소드 내 영구 래치(ever) —
    #   eval은 '한 번이라도 집어 들었고 최종적으로 타겟에 안착·릴리스됐는가'의 성공률 측정이
    #   목적이라 영구 래치가 맞다(보상측 해제 조건은 학습용 anti-hacking 장치). 측정치는
    #   보상 발화 대비 같거나 관대한 상한.
    lift_latch = torch.zeros(n, dtype=torch.bool, device=dev)   # ever (lifted ∧ grasped)
    placed_streak = torch.zeros(n, dtype=torch.long, device=dev)
    # ★★2026-08-11(P13) ★무조건부★ 진단 누적기 — 상위 조건이 실패해도 반드시 값이 나온다★★
    #   [사고] iter2000 게이트에서 릴리스 지표가 전부 nan이었다. 진단 창이 '래치∧tol' ★조건부★인데
    #     그 구간 도달 에피소드가 0/128이라 분모가 0이었다 — ★상위 조건이 막히면 하위를 아예 못 읽는★
    #     설계다. 아래 지표들은 마스크가 없거나(전 구간) 래치만 요구하므로 항상 정의된다.
    #   [읽는 순서] latch_rate → dxy_min_latch 중앙값 → descend 비율 → a_grip_max 중앙값 → placed.
    #     ★각 단계의 분모(n_steps_*)를 반드시 함께 본다★.
    d_min_all = torch.full((n,), float("inf"), device=dev)     # 전 구간 최소 3D 거리(마스크 없음)
    d_min_latch = torch.full((n,), float("inf"), device=dev)   # 래치 이후 최소 3D 거리
    dxy_min_latch = torch.full((n,), float("inf"), device=dev) # 래치 이후 최소 ★수평★ 거리
    #   ⚠️수평을 따로 두는 이유: 들린 상태에서는 큐브가 타겟보다 최소 2.5cm 위라 3D 거리에
    #     ★수직 하한이 섞여★ 순수 접근 정밀도를 가린다.
    z_min_latch = torch.full((n,), float("inf"), device=dev)   # 래치 이후 최저 큐브 z
    descend_latch = torch.zeros(n, dtype=torch.bool, device=dev)  # 래치 이후 MIN_LIFT 아래로 내렸는가
    n_steps_latch_near = torch.zeros(n, dtype=torch.long, device=dev)  # ★분모★ 래치∧릴리스반경 내 스텝
    a_grip_max_latch_near = torch.full((n,), -9.9, device=dev)  # 그 구간의 그리퍼 액션 최대
    a_grip_sum_latch_near = torch.zeros(n, device=dev)
    ever_open_cmd_latch_near = torch.zeros(n, dtype=torch.bool, device=dev)  # ★open(>0) 명령 여부★
    tgt_infeasible = torch.zeros(n, dtype=torch.bool, device=dev)  # 타겟이 지지면 밖인가
    ever_placed = torch.zeros(n, dtype=torch.bool, device=dev)
    placed_steps = torch.zeros(n, dtype=torch.long, device=dev)
    # ★carry_latch(2026-07-13 추가)★ — ever(래치 ∧ 큐브 gd≤goal_radius). 기존 place_success
    #   (순간 파지검출 ∧ 8cm '동시')는 빠른 정책에서 파지 스텝이 짧아 저평가되는 아티팩트 실측
    #   (단일 scratch: place 81.8% < placed 99.4% 역전). 래치 기반은 placed와 동일 논리 계열이라
    #   빠른 정책에서도 배달을 정확히 셈. 기존 지표는 비교 연속성 위해 유지.
    ever_carry_latch = torch.zeros(n, dtype=torch.bool, device=dev)
    # ★placed 조건별 분해 계측(2026-08-01)★ — 4조건 동시 성립이라 실패 원인이 안 보인다.
    _z = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
    ever_at_tol = _z()               # 큐브가 타겟 tol 내에 들어간 적
    ever_latch_tol = _z()            # + 배달 래치
    ever_latch_tol_settled = _z()    # + 정지
    ever_released_any = _z()         # 릴리스(무접촉) 자체가 일어난 적
    ever_latch_released = _z()       # 래치 + 릴리스
    # ★정지(settled) 병목 실체 계측(2026-08-01, F42 후속)★ — '래치∧타겟 tol' 구간에서
    #   ①큐브 최저속도 ②그때의 팔 관절속도 ③그 구간 체류 스텝수를 잰다.
    #   판별: 체류가 짧으면 ★스쳐 지나감★, 체류는 긴데 최저속도가 높으면 ★팔이 계속 움직임★,
    #   팔은 느린데 큐브만 빠르면 ★접촉 지터★(F37에서 맥락을 틀려 오판했던 바로 그 항목).
    steps_latch_tol = torch.zeros(n, dtype=torch.long, device=dev)
    min_spd_at_tol = torch.full((n,), float("inf"), device=dev)   # 큐브 속도 [m/s]
    min_qd_at_tol = torch.full((n,), float("inf"), device=dev)    # 팔 관절속도 norm [rad/s]
    # ★릴리스 병목 진단(2026-08-01, F46)★ — '타겟 위 공중 정지' 가설 검증.
    #   타겟 z(=벨트면+반변)는 MIN_LIFT보다 낮으므로, 큐브를 실제로 내려놓으면 lift(w5)가 끊긴다.
    #   정책이 tol 안이면서도 ★공중에 떠 있으면★ lift+grasp+goal을 모두 유지한 채 놓지 않는
    #   국소최적이 성립한다. 구간 내 ★큐브 최저 z★와 ★MIN_LIFT 아래로 내려간 적★을 잰다.
    min_z_at_tol = torch.full((n,), float("inf"), device=dev)     # 구간 내 큐브 최저 z [m]
    # ★파지 실패 원인 계측(2026-08-03)★ — grasp이 마지막 병목(94.7%)이 됐다. 실패가
    #   ①특정 스폰 위치에 몰리면 ★도달성·기하★ 문제, ②고르게 퍼지면 ★서두름(정책이 파지
    #   신뢰성을 속도와 맞바꿈)★ 문제로, 처방이 완전히 달라진다. 추론 대신 분포를 잰다.
    spawn_b = torch.zeros((n, 2), device=dev)                     # 에피소드 시작 큐브 xy (base)
    first_contact = torch.full((n,), -1, dtype=torch.long, device=dev)  # 첫 손가락 접촉 스텝
    ever_descended = _z()                                         # 구간 내 z < MIN_LIFT 도달
    # ★F46-b★ 그리퍼 액션 포화도 — release_bias의 a_open = clamp(action[-1],0,1)이라
    #   액션 평균이 깊게 음수로 포화하면 ★open을 표본추출조차 못 한다★(탐색 잠김).
    max_agrip_zone = torch.full((n,), -9.9, device=dev)
    sum_agrip_zone = torch.zeros(n, device=dev)
    records: list[dict] = []  # 완료 에피소드 결과

    obs = env.get_observations()  # rsl_rl 3.x(2.3.2 래퍼): TensorDict 단일 반환

    # ═════════════════════════════════════════════════════════════════════════
    # ★렌더 건전성 가드(2026-08-22 신설) — Vision 평가에서만★
    #
    #   [사고] 같은 checkpoint·같은 seed인데 ★실행 순서★에 따라 결과가 뒤집혔다:
    #     seed 43을 1번째로 돌리면 grasp 80.9% / placed 34.2%,
    #     2번째로 돌리면      grasp  5.3% / placed  0.0%  (TCP가 큐브에서 41.6cm 밖에 머묾).
    #     seed 44도 동일 — 단독 88.5%/40.4% vs 연속 2번째 7.6%/0.0%.
    #     즉 seed가 아니라 ★연속 Isaac 기동★이 원인이고, student가 빈 depth를 받아
    #     큐브를 보지 못한 것이다(저장소가 경고해 온 'RTX 검정 프레임' 계열).
    #
    #   [왜 치명적인가] 이 실패는 ★조용하다★. 평가가 정상 종료하고 0.0%를 보고하므로
    #     정책 실패와 구별되지 않는다. 채택 파이프라인은 5후보×4seed=20회를 연속
    #     기동하므로 ★첫 회만 유효★하고 나머지는 전부 이 아티팩트를 낸다.
    #     실제로 student를 placed 0%로 오판해 불필요한 재증류를 걸 뻔했다.
    #
    #   [가드] 정책 관측의 depth 블록이 ★공간 분산★을 갖는지 본다. 정상 렌더는
    #     vision_obs_probe 실측에서 공간 std 0.241이었고, 검정/균일 프레임은 0에 수렴한다.
    #     미달이면 ★즉시 죽는다★ — 조용히 0%를 내는 것보다 즉시 실패가 낫다.
    #   ★★F86 — RGB 채널도 검사한다 (2026-09-07 실측)★★
    #     [사고] 위 가드는 depth만 봤는데 student는 ★RGB-D 4채널★이다. 같은 checkpoint·
    #       같은 seed로 3회 반복한 실측에서 depth 블록은 세 번 모두 ★완전히 동일★
    #       (mean 0.9712 / 공간std 0.0744 / far 83.3%)했는데 RGB만 죽었다:
    #         run1  rgb mean 0.6529 std 0.2939 zero  0.0%  → latch 21.9% placed 3.5%
    #         run2  rgb mean 0.0007 std 0.0011 zero 70.5%  → latch  0.2% placed 0.0%  ★검정★
    #         run3  rgb mean 0.6529 std 0.2939 zero  0.0%  → latch 19.1% placed 2.3%
    #       depth 가드는 셋 다 통과시켰고, 채택 평가 21회 중 9회가 이 상태로 측정돼
    #       '정책이 큐브를 못 잡는다'로 오판됐다(grasp 0.80 ↔ 0.12가 같은 조합에서 교대).
    #     [판정] RGB 블록의 공간 std가 depth와 같은 임계(0.02) 아래면 죽은 프레임이다
    #       (정상 0.29 vs 검정 0.001 — 두 자릿수 여유).
    #   ★F94(2026-09-22)★ 가드 본체는 common/render_guard.py로 옮겨 학습(train.py)과 공유한다 —
    #     새 PC에서 타일 ≥128 + DLSS면 RGB만 죽는데 학습 경로에는 가드가 없어 증류가 조용히 오염됐다.
    if "Vision" in args_cli.task:
        _flat = obs["policy"] if hasattr(obs, "keys") and "policy" in obs.keys() else obs
        from common.config_io import build_preprocess_cfg, configs_dir, load_yaml
        from common.render_guard import check_vision_render_health
        check_vision_render_health(_flat, build_preprocess_cfg(load_yaml(configs_dir() / "common.yaml")), where="eval")
    # ═════════════════════════════════════════════════════════════════════════
    step = 0
    while len(records) < args_cli.n_episodes and step < args_cli.max_steps and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # ★step 후 상태 읽기 — done env는 이미 리셋되었으나(z낮음·미들림) OR/최소 누적에 무해★
            pos, _d = _nearest_object(env.unwrapped)          # (n,3) base
            target = env.unwrapped.command_manager.get_command("target_pos")[:, :3]  # (n,3) base
            lifted = pos[:, 2] > MIN_LIFT
            gd = torch.norm(pos - target, dim=1)
            grasped = _grasped_mask(env.unwrapped, force_n=0.1, rel_speed_mps=0.2)  # (n,) bool

            ever_lifted |= lifted
            lifted_steps += lifted.long()
            ever_grasped |= grasped
            grasped_steps += grasped.long()
            ep_steps += 1
            gd_masked = torch.where(lifted, gd, torch.full_like(gd, float("inf")))
            best_gd = torch.minimum(best_gd, gd_masked)
            gd_true = torch.where(lifted & grasped, gd, torch.full_like(gd, float("inf")))
            best_gd_true = torch.minimum(best_gd_true, gd_true)
            # ★full place: 래치(파지∧들림 달성) 후 타겟 tol 내 정지·무접촉(릴리스)★
            # done env는 step에서 이미 리셋됨 — 새 에피소드 상태(스폰 큐브가 우연히 타겟 근처)
            # 를 옛 래치로 placed 오판하지 않도록 done 마스크로 제외(레거시 OR/min과 달리
            # placed는 4조건 동시라 오염 가능).
            not_done = ~dones.to(dev).bool().flatten()
            lift_latch |= lifted & grasped & not_done
            ever_carry_latch |= lift_latch & (gd < args_cli.goal_radius) & not_done
            # ★F37★ 임계는 binpick_env_cfg.PLACED_MAX_SPEED 단일 소스 — 학습과 평가가
            #   다른 값을 쓰면 '학습에선 성공인데 평가에선 실패'가 된다(구버전은 양쪽에 0.10 하드코딩).
            obj_spd = _nearest_object_vel(env.unwrapped).norm(dim=-1)
            settled = obj_spd < PLACED_MAX_SPEED
            _contact_now = _any_finger_contact(env.unwrapped, 0.05)
            released = ~_contact_now
            if step == 0:
                spawn_b = pos[:, :2].clone()
            _newc = _contact_now & (first_contact < 0)
            first_contact = torch.where(_newc, ep_steps, first_contact)
            placed_candidate = placed_success_mask(
                env.unwrapped,
                tol=args_cli.place_tol,
                max_speed=PLACED_MAX_SPEED,
                contact_n=0.05,
                delivery_latch=lift_latch,
            ) & not_done
            placed_streak = torch.where(
                placed_candidate, placed_streak + 1, torch.zeros_like(placed_streak)
            )
            placed_now = placed_streak >= PLACED_MIN_CONSECUTIVE_STEPS
            ever_placed |= placed_now
            placed_steps += placed_now.long()
            # ★2026-08-01 조건별 분해 계측★ — placed는 4조건 동시 성립이라 실패해도
            #   ★어느 조건이 막는지★ 알 수 없다. 각 조건과 누적 조합을 따로 세어
            #   병목을 추측이 아니라 수치로 특정한다(placed 정체 진단의 핵심 도구).
            at_tol = gd < args_cli.place_tol
            ever_at_tol |= at_tol & not_done                       # 타겟 tol 내 도달
            ever_latch_tol |= lift_latch & at_tol & not_done       # + 배달 래치
            ever_latch_tol_settled |= lift_latch & at_tol & settled & not_done   # + 정지
            ever_released_any |= released & not_done               # 릴리스(무접촉) 자체
            ever_latch_released |= lift_latch & released & not_done  # 래치 + 릴리스
            # ★정지 병목 실체 계측★ — '래치∧tol' 구간에서만 큐브/팔 속도의 최저치를 남긴다.
            in_zone = lift_latch & at_tol & not_done
            steps_latch_tol += in_zone.long()
            _qd = env.unwrapped.scene["robot"].data.joint_vel[:, :6].norm(dim=-1)
            _big = torch.full_like(obj_spd, float("inf"))
            min_spd_at_tol = torch.minimum(min_spd_at_tol, torch.where(in_zone, obj_spd, _big))
            min_qd_at_tol = torch.minimum(min_qd_at_tol, torch.where(in_zone, _qd, _big))
            # ★F46★ 구간 내 큐브 높이 — '공중 정지' 국소최적 판별
            min_z_at_tol = torch.minimum(min_z_at_tol, torch.where(in_zone, pos[:, 2], _big))
            ever_descended |= in_zone & (pos[:, 2] < MIN_LIFT)
            _ag = actions[:, -1].detach()
            max_agrip_zone = torch.maximum(max_agrip_zone, torch.where(in_zone, _ag, torch.full_like(_ag, -9.9)))
            sum_agrip_zone += torch.where(in_zone, _ag, torch.zeros_like(_ag))
            # ★★P13 무조건부 계측★★ — 위 in_zone(래치∧tol) 조건부 지표와 달리 항상 값이 남는다.
            _tgt = env.unwrapped.command_manager.get_command("target_pos")[:, :3]
            _live = not_done
            d_min_all = torch.minimum(d_min_all, torch.where(_live, gd, _big))
            _after = lift_latch & _live
            d_min_latch = torch.minimum(d_min_latch, torch.where(_after, gd, _big))
            _dxy = torch.norm(_tgt[:, :2] - pos[:, :2], dim=-1)
            dxy_min_latch = torch.minimum(dxy_min_latch, torch.where(_after, _dxy, _big))
            z_min_latch = torch.minimum(z_min_latch, torch.where(_after, pos[:, 2], _big))
            descend_latch |= _after & (pos[:, 2] < MIN_LIFT)
            # 릴리스 보상이 지급되는 반경(release_bias radius=0.12)과 같은 창 — ★분모를 함께 남긴다★
            _near = _after & (gd < _RELEASE_RADIUS)
            n_steps_latch_near += _near.long()
            a_grip_max_latch_near = torch.maximum(
                a_grip_max_latch_near, torch.where(_near, _ag, torch.full_like(_ag, -9.9)))
            a_grip_sum_latch_near += torch.where(_near, _ag, torch.zeros_like(_ag))
            ever_open_cmd_latch_near |= _near & (_ag > 0.0)
            tgt_infeasible |= _target_infeasible(_tgt[:, :2], args_cli.place_tol) & _live

            done_idx = torch.nonzero(dones, as_tuple=False).flatten()
            # 한 physics step에서 여러 env가 동시에 끝날 수 있다. 요청한 표본 수를 넘겨
            # 기록하면 checkpoint마다 N이 달라져 Wilson CI와 순위 비교가 불공정해진다.
            # 남은 수만 결정적으로 취해 `--n_episodes`를 정확한 계약으로 유지한다.
            remaining = max(0, args_cli.n_episodes - len(records))
            for i in done_idx[:remaining].tolist():
                es = int(ep_steps[i].item())
                if es <= 0:
                    continue
                bg = float(best_gd[i].item())
                bgt = float(best_gd_true[i].item())
                records.append({
                    "episode": len(records),
                    # legacy(z-기반 — 저글링 포함 가능): 참고용 유지
                    "lifted_ever": int(bool(ever_lifted[i].item())),
                    "airborne_frac": round(float(lifted_steps[i].item()) / es, 4),
                    "place_z_only": int(bg <= args_cli.goal_radius),
                    # ★진짜 파지(STAGE 3 게이트 정의)★ — 이게 공식 성공률
                    "grasp_success": int(bool(ever_grasped[i].item())),
                    "hold_frac": round(float(grasped_steps[i].item()) / es, 4),
                    "place_success": int(bgt <= args_cli.goal_radius),
                    # ★carry(래치 기반) — 배달 공식 지표★: ever(래치 ∧ gd≤goal_radius)
                    "carry_latch_success": int(bool(ever_carry_latch[i].item())),
                    # ★full pick&place 공식 지표★: 릴리스(무접촉)·정지 상태로 타겟 도달
                    "placed_success": int(bool(ever_placed[i].item())),
                    "placed_frac": round(float(placed_steps[i].item()) / es, 4),
                    # ★조건별 분해★ — placed 실패 시 어디서 막혔는지 특정용
                    "c_at_tol": int(bool(ever_at_tol[i].item())),
                    "c_latch_tol": int(bool(ever_latch_tol[i].item())),
                    "c_latch_tol_settled": int(bool(ever_latch_tol_settled[i].item())),
                    "c_released": int(bool(ever_released_any[i].item())),
                    "c_latch_released": int(bool(ever_latch_released[i].item())),
                    # ★정지 병목 실체★ — 래치∧tol 구간의 체류 스텝 / 큐브·팔 최저속도
                    "zone_steps": int(steps_latch_tol[i].item()),
                    "zone_min_obj_spd": (round(float(min_spd_at_tol[i].item()), 4)
                                         if min_spd_at_tol[i].item() != float("inf") else ""),
                    "zone_min_arm_qd": (round(float(min_qd_at_tol[i].item()), 4)
                                        if min_qd_at_tol[i].item() != float("inf") else ""),
                    "zone_min_obj_z": (round(float(min_z_at_tol[i].item()), 4)
                                       if min_z_at_tol[i].item() != float("inf") else ""),
                    "zone_descended": int(bool(ever_descended[i].item())),
                    "zone_max_a_grip": round(float(max_agrip_zone[i].item()), 4),
                    "zone_mean_a_grip": (round(float(sum_agrip_zone[i].item() / max(1, int(steps_latch_tol[i].item()))), 4)
                                         if steps_latch_tol[i].item() > 0 else ""),
                    # ★파지 실패 원인 계측★ — 스폰 위치·첫 접촉 시점
                    "spawn_x": round(float(spawn_b[i, 0].item()), 4),
                    "spawn_y": round(float(spawn_b[i, 1].item()), 4),
                    "first_contact_step": int(first_contact[i].item()),
                    "min_goal_dist_m": round(bgt, 4) if bgt != float("inf") else "",
                    # ★P13 무조건부 컬럼★ — 상위 조건이 실패해도 값이 있다
                    "d_min_all": (round(float(d_min_all[i].item()), 4)
                                  if d_min_all[i].item() != float("inf") else ""),
                    "d_min_latch": (round(float(d_min_latch[i].item()), 4)
                                    if d_min_latch[i].item() != float("inf") else ""),
                    "dxy_min_latch": (round(float(dxy_min_latch[i].item()), 4)
                                      if dxy_min_latch[i].item() != float("inf") else ""),
                    "z_min_latch": (round(float(z_min_latch[i].item()), 4)
                                    if z_min_latch[i].item() != float("inf") else ""),
                    "descend_latch": int(bool(descend_latch[i].item())),
                    "n_steps_latch_near": int(n_steps_latch_near[i].item()),
                    "a_grip_max_latch_near": (round(float(a_grip_max_latch_near[i].item()), 4)
                                              if n_steps_latch_near[i].item() > 0 else ""),
                    "a_grip_mean_latch_near": (
                        round(float(a_grip_sum_latch_near[i].item() / int(n_steps_latch_near[i].item())), 4)
                        if n_steps_latch_near[i].item() > 0 else ""),
                    "open_cmd_latch_near": (int(bool(ever_open_cmd_latch_near[i].item()))
                                            if n_steps_latch_near[i].item() > 0 else ""),
                    "target_infeasible": int(bool(tgt_infeasible[i].item())),
                    "ep_steps": es,
                })
                ever_lifted[i] = False
                lifted_steps[i] = 0
                ever_grasped[i] = False
                grasped_steps[i] = 0
                ep_steps[i] = 0
                best_gd[i] = float("inf")
                best_gd_true[i] = float("inf")
                lift_latch[i] = False
                placed_streak[i] = 0
                ever_placed[i] = False
                placed_steps[i] = 0
                ever_carry_latch[i] = False
                # ★조건별 분해 누산기도 함께 클리어★ (CSV 기록 ★이후★여야 한다 —
                #   기록 전에 지우면 전부 False로 남는다)
                ever_at_tol[i] = False
                ever_latch_tol[i] = False
                ever_latch_tol_settled[i] = False
                ever_released_any[i] = False
                ever_latch_released[i] = False
                steps_latch_tol[i] = 0
                min_spd_at_tol[i] = float("inf")
                min_qd_at_tol[i] = float("inf")
                min_z_at_tol[i] = float("inf")
                ever_descended[i] = False
                spawn_b[i] = pos[i, :2]      # done 후 pos는 ★새 에피소드★ 스폰 위치
                first_contact[i] = -1
                max_agrip_zone[i] = -9.9
                sum_agrip_zone[i] = 0.0
                # ★P13 무조건부 누적기★ — 같은 규칙: CSV 기록 ★이후★에 클리어한다
                d_min_all[i] = float("inf")
                d_min_latch[i] = float("inf")
                dxy_min_latch[i] = float("inf")
                z_min_latch[i] = float("inf")
                descend_latch[i] = False
                n_steps_latch_near[i] = 0
                a_grip_max_latch_near[i] = -9.9
                a_grip_sum_latch_near[i] = 0.0
                ever_open_cmd_latch_near[i] = False
                tgt_infeasible[i] = False
        step += 1
        if args_cli.video and step >= args_cli.video_length:
            # 영상만 필요한 짧은 실행이면 여기서 마무리 (통계는 부분적)
            if args_cli.n_episodes <= n:
                break

    env.close()

    # ── 집계 + 산출물 ──
    m = len(records)
    if m == 0:
        print("[eval] ⚠️ 완료 에피소드 0 — max_steps 상향 또는 num_envs 확인.")
        return
    grasp = sum(r["grasp_success"] for r in records) / m
    place = sum(r["place_success"] for r in records) / m
    carry_latch = sum(r["carry_latch_success"] for r in records) / m
    placed = sum(r["placed_success"] for r in records) / m
    placed_frac = sum(r["placed_frac"] for r in records) / m
    hold = sum(r["hold_frac"] for r in records) / m
    gds = [r["min_goal_dist_m"] for r in records if isinstance(r["min_goal_dist_m"], float)]
    mean_gd = sum(gds) / len(gds) if gds else float("nan")
    # legacy(z-기반) — 저글링 포함 가능, 비교/회귀 검출용
    lifted_ever = sum(r["lifted_ever"] for r in records) / m
    airborne = sum(r["airborne_frac"] for r in records) / m
    place_z = sum(r["place_z_only"] for r in records) / m

    # ★placed 조건별 분해(2026-08-01) — 병목을 추측이 아니라 수치로 특정★
    _r = lambda k: sum(r[k] for r in records) / m
    c_at, c_lt = _r("c_at_tol"), _r("c_latch_tol")
    c_lts, c_rel, c_lr = _r("c_latch_tol_settled"), _r("c_released"), _r("c_latch_released")
    _pct = lambda a, b: (a / b * 100) if b > 0 else float("nan")
    _zs = [r["zone_steps"] for r in records if r["zone_steps"] > 0]
    _zo = [r["zone_min_obj_spd"] for r in records if isinstance(r["zone_min_obj_spd"], float)]
    _zq = [r["zone_min_arm_qd"] for r in records if isinstance(r["zone_min_arm_qd"], float)]
    _zz = [r["zone_min_obj_z"] for r in records if isinstance(r["zone_min_obj_z"], float)]
    _zd = [r["zone_descended"] for r in records if r["zone_steps"] > 0]
    _zg = [r["zone_mean_a_grip"] for r in records if isinstance(r["zone_mean_a_grip"], float)]
    _zgm = [r["zone_max_a_grip"] for r in records if r["zone_steps"] > 0]

    # ★★P13(2026-08-11) 무조건부 진단 집계 — 상위 조건이 실패해도 반드시 숫자가 나온다★★
    def _med(vals):
        v = sorted(x for x in vals if isinstance(x, float))
        return v[len(v) // 2] if v else float("nan")

    _n = max(1, len(records))
    _dxy = _med([r["dxy_min_latch"] for r in records])
    _dmin = _med([r["d_min_latch"] for r in records])
    _zmin = _med([r["z_min_latch"] for r in records])
    _desc = sum(r["descend_latch"] for r in records) / _n
    _near_n = [r for r in records if r["n_steps_latch_near"] > 0]
    _agmax = _med([r["a_grip_max_latch_near"] for r in records])
    _agmean = _med([r["a_grip_mean_latch_near"] for r in records])
    _opencmd = (sum(r["open_cmd_latch_near"] for r in _near_n) / len(_near_n)) if _near_n else float("nan")
    _infeas = sum(r["target_infeasible"] for r in records) / _n
    _open_rate = (sum(1 for v in _zgm if v > 0) / len(_zgm)) if _zgm else float("nan")
    _med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    breakdown = (
        f"{'-'*60}\n"
        f"★placed 조건별 분해★ (4조건 동시 성립 — 어디서 막히는지)\n"
        f"  ①파지 {grasp*100:.1f}% → ②배달래치 {carry_latch*100:.1f}% → "
        f"③타겟 {args_cli.place_tol*100:.0f}cm {c_at*100:.1f}% → ④래치∧tol {c_lt*100:.1f}%\n"
        f"  → ⑤+정지 {c_lts*100:.1f}% → ⑥placed {placed*100:.1f}%\n"
        f"  전환율: 래치∧tol→+정지 ★{_pct(c_lts, c_lt):.1f}%★ / +정지→placed {_pct(placed, c_lts):.1f}%\n"
        f"  (참고) 릴리스 발생 {c_rel*100:.1f}% / 래치∧릴리스 {c_lr*100:.1f}%\n"
        f"★정지 병목 실체★ — '래치∧tol' 구간 계측 (해당 에피소드 {len(_zs)}/{m})\n"
        f"  체류 스텝  중앙 {_med(_zs):.0f} / 최대 {max(_zs) if _zs else 0}\n"
        f"  큐브 최저속도 중앙 ★{_med(_zo):.4f}★ m/s  (정지 임계 {PLACED_MAX_SPEED})\n"
        f"  팔 관절속도  중앙 ★{_med(_zq):.4f}★ rad/s\n"
        f"  판별: 체류↓=스쳐지나감 / 체류↑·큐브속도↑·팔속도↑=팔이 안 멈춤 / "
        f"팔속도↓인데 큐브속도↑=접촉 지터\n"
        f"★릴리스 병목 실체★ — 같은 구간의 큐브 높이·그리퍼 액션\n"
        f"  큐브 최저 z 중앙 {_med(_zz):.4f} m (타겟 {_TARGET_Z:.3f} 대비 {(_med(_zz)-_TARGET_Z)*1000:+.0f}mm, "
        f"MIN_LIFT {MIN_LIFT:.3f})\n"
        f"  MIN_LIFT 아래로 내려간 에피소드 {(sum(_zd)/len(_zd)*100 if _zd else float('nan')):.1f}%"
        f"   (내려간 적 없음 = 공중 정지 국소최적)\n"
        f"  ★그리퍼 액션[-1] 중앙 {_med(_zg):.3f}★ / 구간 최대의 중앙 {_med(_zgm):.3f}"
        f"   (action<0=close, a_open=clamp(action,0,1))\n"
        f"  ★타겟에서 open(>0)을 명령한 에피소드 {(_open_rate*100):.1f}%★\n"
        f"  판별: 액션이 깊게 음수(≲−3)면 ★포화 잠김★ — release_bias의 gradient가 0이라\n"
        f"        게이팅을 아무리 고쳐도 릴리스는 학습되지 않는다(F46).\n"
    )

    csv_path = os.path.join(out_dir, "episodes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    _role = "student" if "Vision" in args_cli.task else "teacher"  # 산출물 phase 매핑과 동일 규칙
    summary = (
        f"UR5e {_role} 정량 평가 — 순수 자력 파지 (pre-grasp 보조 없음)\n"
        f"{'='*60}\n"
        f"완료 에피소드      : {m}\n"
        f"★진짜 파지 지표(STAGE 3 게이트: 양손접촉∧상대속도≈0 — 공식 성공률)★\n"
        f"grasp 성공률       : {grasp*100:.1f}%   (파지 성립 도달)\n"
        f"hold 평균 유지율   : {hold*100:.1f}%   (파지 유지 스텝 비율)\n"
        f"carry 성공률       : {carry_latch*100:.1f}%   (★래치 기반 배달 공식: ever(래치∧목표 ≤{args_cli.goal_radius*100:.0f}cm)★)\n"
        f"carry(순간검출·구) : {place*100:.1f}%   (파지검출∧들림∧≤{args_cli.goal_radius*100:.0f}cm 동시 — 빠른 정책서 저평가, 비교용)\n"
        f"★placed 성공률     : {placed*100:.1f}%   (릴리스·정지로 타겟 ≤{args_cli.place_tol*100:.0f}cm — full pick&place)★\n"
        f"placed 유지 비율   : {placed_frac*100:.1f}%   (안착 상태 스텝 비율)\n"
        f"파지시 평균 목표거리: {mean_gd*100:.1f}cm\n"
        f"[legacy z-기반 — 저글링 포함 가능(비교용)] lifted {lifted_ever*100:.1f}% / "
        f"airborne {airborne*100:.1f}% / place(z) {place_z*100:.1f}%\n"
        + breakdown +
        f"{'-'*60}\n"
        "★무조건부 진단(P13)★ — 위 '존' 지표가 nan이어도 여기는 항상 값이 있다\n"
        f"  래치 후 최소 ★수평★거리 중앙 : {_dxy:.4f} m   (3D 최소 {_dmin:.4f} m)\n"
        f"  래치 후 최저 큐브 z    중앙 : {_zmin:.4f} m   (MIN_LIFT {MIN_LIFT:.4f} / 타겟 {_TARGET_Z:.4f})\n"
        f"  ★하강 시도(MIN_LIFT 아래) 비율 : {_desc*100:.1f}%★\n"
        f"  릴리스 지급창(래치∧{_RELEASE_RADIUS}m) 도달 : {len(_near_n)}/{len(records)} 에피소드  ← ★분모★\n"
        f"  그 창에서 ★open(>0) 명령 : {_opencmd*100:.1f}%★   (그리퍼 액션 최대 중앙 {_agmax:+.4f} / 평균 중앙 {_agmean:+.4f})\n"
        f"  ★타겟이 지지면 밖(placed 원천 불가) : {_infeas*100:.1f}%★  ← 0%가 아니면 성공률 상한이 그만큼 깎인다 (F58)\n"
        f"{chr(45)*60}\n"
        f"재현정보:\n"
        f"  task           : {args_cli.task}\n"
        f"  checkpoint     : {resume_path}\n"
        f"  num_envs       : {n}\n"
        f"  seed           : {getattr(agent_cfg, 'seed', 'n/a')}\n"
        f"  MIN_LIFT[m]    : {MIN_LIFT:.4f}\n"
        f"  goal_radius[m] : {args_cli.goal_radius}\n"
        f"  git            : {_git_hash()}\n"
        f"  BINPICK_env    : {_binpick_env()}\n"
    )
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"[eval] ★산출★ {out_dir}  (episodes.csv + summary.txt)")
    # ★논문용 통계 자동 생성 (CPU — 성공률 95% CI·실패모드·결과표 md/tex)★
    try:
        completed = _sp.run(
            [sys.executable, os.path.join(_REPO_ROOT, "analysis", "eval_stats.py"),
             "--eval-dir", out_dir, "--label", _role],
            cwd=_REPO_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"eval_stats failed: exit={completed.returncode}")
    except Exception as e:  # noqa: BLE001
        print(f"[stats][error] eval 통계 생성 실패: {e}")
        raise
    receipt = finalize_evaluation(evaluation_session, evaluation_dir=out_dir)
    receipt_path = write_evaluation_receipt(
        os.path.join(out_dir, EVALUATION_RECEIPT_NAME), receipt
    )
    if evaluation_artifact_inventory(
        out_dir,
        project_root=_REPO_ROOT,
        task_id=args_cli.task,
    ) != receipt["artifacts"]:
        raise RuntimeError("evaluation artifacts changed while sealing their receipt")
    receipt_relative = receipt_path.resolve().relative_to(Path(_REPO_ROOT).resolve()).as_posix()
    receipt_sha256 = sha256_file(receipt_path)
    print(f"[eval] provenance receipt: {receipt_path}")
    print(
        f"EVALUATION_RECEIPT_OK sha256={receipt_sha256} path={receipt_relative}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
