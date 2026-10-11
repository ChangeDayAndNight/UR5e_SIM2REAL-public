#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — isaac51 python 자기 재실행 + repo 루트.
# ─────────────────────────────────────────────────────────────────────────────
"""Phase 1 — ★placed 달성가능성 프로브★ (2026-08-09 신설, 진단 전용).

목적: 20k run(`2026-08-08_00-24-58`)에서 `Episode_Reward/placed`가 20,000 iteration 내내
      ★정확히 0★이었다. §9-5.1("보상 항을 고쳐도 안 들면 gradient가 실제로 흐르는지부터
      재라")에 따라, placed 조건이 ①정책이 못 하는 것인지 ②환경에서 애초에 성립 불가인지를
      구분한다. 추론이 아니라 sim 실측으로 가른다.

방법: `placed_success_mask`의 5개 AND 항(래치·거리·정지·무접촉·실제개방)을 ★항별로 분해★해
      per-step 계측한다. 두 모드를 같은 프로토콜로 비교한다.
  · mode A (`--mode policy`)     : 정책 그대로 — '정책이 무엇을 못 하는가'
  · mode B (`--mode force_open`) : 타겟존(래치∧거리<tol) 진입 즉시 그리퍼 액션을 ★강제 개방★
                                   (+1)으로 덮어쓴다 — '놓기만 하면 placed가 성립하는가'
      ⇒ B에서도 placed가 0이면 ★환경/판정 결함★(정책 탓이 아니다).
      ⇒ B에서 placed가 높으면 ★정책의 릴리스 미학습★(보상 지형 문제).

⚠️ 진단 전용 — 학습 코드·보상·cfg를 일절 수정하지 않는다. 판정 함수는 학습과 동일한
   `binpick_env_cfg`의 단일 소스를 import해서 쓴다(F30/F37류 맥락 불일치 방지, §9-5.2).

실행 (학습 PC, isaac51 — GUI 불필요, headless):
  python Phase1\\BinPick_PlacedProbe.py --checkpoint <model_*.pt> --mode policy     --headless
  python Phase1\\BinPick_PlacedProbe.py --checkpoint <model_*.pt> --mode force_open --headless
산출: Phase1/Result/probe_<timestamp>/ → episodes.csv + summary.txt + probe.json

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
preload_native_training_modules()



_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
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

import cli_args  # scripts/rsl_rl/cli_args.py

_TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"

parser = argparse.ArgumentParser(description="placed 조건 달성가능성 프로브(진단 전용).")
parser.add_argument("--task", type=str, default=_TASK, help="평가 태스크(보조 리셋 off).")
parser.add_argument("--num_envs", type=int, default=128, help="병렬 env 수.")
parser.add_argument("--n_episodes", type=int, default=256, help="집계할 최소 완료 에피소드 수.")
parser.add_argument("--max_steps", type=int, default=4000, help="안전 상한(무한루프 방지).")
parser.add_argument("--place_tol", type=float, default=0.03, help="placed 판정 반경[m].")
parser.add_argument("--mode", type=str, default="policy",
                    choices=("policy", "force_open", "force_open_high", "force_open_smart",
                             "force_open_freeze", "force_open_retreat", "force_open_tcp", "force_open_settle", "force_open_deploy",
                             "carry_hold_deploy", "grasp_hold_deploy",
                             "free_air", "descend_sweep", "settle_floor", "scripted_place"),
                    help="policy=정책 그대로 / force_open=타겟존 진입 시 강제 개방 / "
                         "force_open_high=★공중(벨트에서 먼 높이)에서 강제 개방★ — "
                         "손가락이 열리지 않는 원인이 '벨트/큐브와의 접촉 간섭'인지 가르는 대조군 / "
                         "★scripted_place=정책을 완전히 배제하고 팔·그리퍼를 전부 스크립트로 "
                         "구동해 '완벽한 place'를 실행★ — placed가 환경에서 애초에 성립 가능한지를 "
                         "가리는 결정적 대조군(실패하면 원인은 정책이 아니라 환경/판정이다).")
# free_air: 팔 액션을 0으로 고정(홈 자세 유지 = 큐브/벨트와 무접촉)한 채 그리퍼만
#   close(20스텝) → open(나머지)으로 강제한다. 접촉이 전혀 없는 조건에서 finger 관절이
#   open 목표(q=0)에 도달하는지 = ★액추에이터 단독 능력★을 잰다.
# force_open_smart: ★xy 정렬 + 벨트 여유 높이★가 동시에 성립할 때만 개방한다.
#   force_open(무조건 타겟존 진입 시)·force_open_high(높이만 보고)는 정렬을 안 봐서
#   큐브가 목표에서 3~4cm 벗어난 곳에 떨어진다 — 그 결과의 낮은 placed를 '환경 상한'으로
#   오독하면 안 된다. 이 모드가 ★잘 조준해서 놓았을 때의 실제 상한★을 잰다.
# descend_sweep: 팔을 ★수직 하강만★ 시키며 TCP 높이와 손가락 접촉력을 기록한다.
#   그리퍼는 닫은 채(큐브 미파지) 내려가므로 f_obj≈0이고, f_net이 튀는 순간이 곧
#   ★패드가 벨트에 닿는 높이★다. 이 값이 IK 하강 바닥(TCP z 하한)의 실측 근거가 된다.
# settle_floor: 팔을 전혀 움직이지 않고(액션 0) 큐브가 스폰 낙하 후 ★스스로 안착★하게 둔 뒤,
#   '실제로 정지한 큐브'가 이 시뮬레이터에서 내는 속도 분포를 잰다. placed 정지 임계는
#   이 분포의 상한보다 위여야 한다 — 그 아래는 정지와 노이즈를 구분할 수 없기 때문이다.
parser.add_argument("--settle_steps", type=int, default=40,
                    help="settle_floor: 계측 시작 전 안착 대기 스텝 수.")
parser.add_argument("--smart_xy_tol", type=float, default=0.015,
                    help="force_open_smart: 개방 허용 xy 정렬 오차[m].")
parser.add_argument("--smart_z", type=float, default=0.045,
                    help="force_open_smart: 개방 허용 큐브 높이 상한[m] (벨트 여유 확보).")
parser.add_argument("--sp_align", type=int, default=25, help="scripted_place: 수평 정렬 스텝")
parser.add_argument("--sp_descend", type=int, default=25, help="scripted_place: 하강 스텝")
parser.add_argument("--sp_open", type=int, default=15, help="scripted_place: 개방 유지 스텝")
parser.add_argument("--sp_gain", type=float, default=40.0,
                    help="scripted_place: 위치오차[m]→정규화 액션 이득 (0.025m 오차 → 1.0 포화)")
parser.add_argument("--sp_retreat", type=float, default=0.6,
                    help="scripted_place: 후퇴(상승) 액션 크기 [0,1]")
parser.add_argument("--high_z", type=float, default=0.12,
                    help="force_open_high 모드에서 개방을 트리거할 큐브 높이[m] (벨트 상면 기준).")
parser.add_argument("--tcp_settle_mps", type=float, default=0.02,
                    help="force_open_settle: 개방 허용 TCP 속도 상한[m/s].")
parser.add_argument("--tcp_z", type=float, default=0.075,
                    help="force_open_tcp: 개방 허용 TCP 높이 상한[m] (파지중심 기준).")
parser.add_argument("--no-success-term", dest="no_success_term", action="store_true",
                    default=False,
                    help="placed_success DoneTerm 비활성 — 배포 등급 지속성(10스텝) 측정용.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="fabric 비활성(USD I/O).")
parser.add_argument("--seed", type=int, default=42, help="평가 seed(기본 42 — eval 프로토콜과 동일).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────── 이하 sim 기동 후 ───────────────────────────
import csv
import json
import subprocess as _sp
from datetime import datetime

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401  (import 시 gym.register)

# ★학습 보상과 ★같은 함수★를 쓴다 — 판정 드리프트 금지(§4-6, F30/F37 교훈)★
from ur5e_binpick.tasks.binpick.binpick_env_cfg import (
    placed_success_mask,
    MIN_LIFT,
    PLACED_GRIPPER_OPEN_TOL_M,
    PLACED_MAX_SPEED,
    PLACED_MIN_CONSECUTIVE_STEPS,
    _any_finger_contact,
    _grasped_mask,
    _gripper_open_mask,
    _lifted_latch,
    _nearest_object,
    _nearest_object_vel,
)
from ur5e_binpick.tasks.binpick.mdp.observations import tcp_pose_b as bp_obs_tcp
from ur5e_binpick.tasks.binpick.mdp.ur5e_assets import GRIPPER_JOINT_NAMES, GRIPPER_OPEN_M
from common.config_io import configs_dir, load_yaml
import common.release_supervisor as _rs

_TARGET_Z = float(load_yaml(configs_dir() / "common.yaml")["target"]["sample_range"]["z"][0])
_CONTACT_N = 0.05  # placed_success_mask 기본값과 동일
_TABLE_HEIGHT = float(load_yaml(configs_dir() / "sim.yaml")["scene"]["table_height_m"])


def _finger_forces(env):
    """(손가락↔큐브 필터드 접촉력 최대, 손가락 전체 접촉력 최대) — 각 (N,) [N].

    필터드(`force_matrix_w`)는 `filter_prim_paths_expr`에 등록된 ★큐브★와의 접촉만 담는다.
    `net_forces_w`는 상대를 가리지 않는 총 접촉력이다. 두 값의 차이가 곧 ★큐브가 아닌 것과의
    접촉★(벨트/작업대/자기 충돌)이며, F50 잔여 병목인 '손가락이 안 벌어짐'의 직접 증거가 된다.
    """
    sensors = env.scene.sensors
    zero = torch.zeros(env.num_envs, device=env.device)
    if "left_finger_contact" not in sensors or "right_finger_contact" not in sensors:
        return zero, zero
    f_obj, f_net = zero.clone(), zero.clone()
    for name in ("left_finger_contact", "right_finger_contact"):
        data = sensors[name].data
        fm = data.force_matrix_w
        if fm is not None:
            f_obj = torch.maximum(f_obj, fm.squeeze(1).norm(dim=-1).amax(dim=1))
        nf = data.net_forces_w
        if nf is not None:
            f_net = torch.maximum(f_net, nf.norm(dim=-1).amax(dim=1))
    return f_obj, f_net


def _git_hash() -> str:
    try:
        return _sp.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.seed is not None and hasattr(env_cfg, "seed"):
        env_cfg.seed = agent_cfg.seed
    # ★--no-success-term(2026-08-16 신설)★ — placed_success DoneTerm을 끈다.
    #   [왜] IsaacLab은 종료 시 ★auto-reset★하므로, 3스텝 DoneTerm이 켜져 있으면 프로브가
    #     step 후 관측하는 streak는 ★2에서 멈춘다★(3번째 스텝의 상태는 이미 리셋된 새 에피소드다).
    #     그래서 '성공했는가'는 재도 ★얼마나 오래 놓인 채 유지되는가★는 이 상태로 잴 수 없다.
    #     배포 등급 기준(STRICT_PLACED_MIN_CONSECUTIVE_STEPS=10)을 재려면 종료를 꺼야 한다.
    #   ⚠️진단 전용 — 학습 cfg는 건드리지 않는다(런타임 인스턴스만 수정).
    if args_cli.no_success_term:
        if hasattr(env_cfg.terminations, "placed_success"):
            env_cfg.terminations.placed_success = None
            print("[probe] ★placed_success DoneTerm 비활성★ — 지속성 측정 모드")
        else:
            raise RuntimeError("placed_success DoneTerm을 찾지 못했다 — 계약이 바뀌었다")
    log_root = os.path.abspath(os.path.join(_REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        from isaaclab.utils.assets import retrieve_file_path

        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[probe] 체크포인트: {resume_path}")
    print(f"[probe] 모드: {args_cli.mode}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    out_dir = os.path.join(_HERE, "Result", "probe_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = (
        DistillationRunner
        if getattr(agent_cfg, "class_name", "OnPolicyRunner") == "DistillationRunner"
        else OnPolicyRunner
    )
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    _prev_tcp = None
    _rs_cfg = _rs.load_release_supervisor_cfg()
    _carried = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=args_cli.device)
    _grip_closed_prev = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=args_cli.device)
    _z_at_close = torch.zeros(args_cli.num_envs, dtype=torch.float32, device=args_cli.device)
    _settle = torch.zeros(args_cli.num_envs, dtype=torch.long, device=args_cli.device)  # ★F92★
    _FORCE_MODE = args_cli.mode in ("force_open", "force_open_high", "force_open_smart",
                                   "force_open_freeze", "force_open_retreat", "force_open_tcp", "force_open_settle", "force_open_deploy",
                                   "carry_hold_deploy", "grasp_hold_deploy")
    dev = uenv.device
    n = uenv.num_envs
    tol = float(args_cli.place_tol)
    grip_ids, _ = uenv.scene["robot"].find_joints(list(GRIPPER_JOINT_NAMES))
    finger_body_ids, finger_body_names = uenv.scene["robot"].find_bodies(
        ["left_finger_link", "right_finger_link"]
    )
    print(f"[probe] finger bodies: {finger_body_names} ids={finger_body_ids}")

    _bz = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
    _bl = lambda: torch.zeros(n, dtype=torch.long, device=dev)
    _bf = lambda v: torch.full((n,), float(v), device=dev)

    # ── 에피소드 누적기 ──
    ep_steps = _bl()
    # ★scripted_place 전용★ — 정책이 파지·운반을 마쳐 ★래치가 성립한 뒤★부터 스크립트가
    #   인수한다(파지는 정책이 100%/97.7%로 이미 검증됐다 — 이 대조군은 place 국면만 본다).
    #   −1 = 아직 인수 전. 인수 후 매 스텝 +1 → 단계 전환에 쓰인다.
    sp_t = _bl() - 1
    ever_latch = _bz()
    ever_zone = _bz()               # 래치 ∧ 거리<tol
    zone_steps = _bl()
    min_spd_zone = _bf("inf")       # 존 내 큐브 최저 속도
    min_z_zone = _bf("inf")         # 존 내 큐브 최저 z
    min_openerr_zone = _bf("inf")   # 존 내 |q-OPEN| 최댓값(양손)의 최솟값
    max_agrip_zone = _bf(-99.0)     # 존 내 그리퍼 액션 최댓값
    ever_settled = _bz()            # 존 ∧ 속도<임계
    ever_open = _bz()               # 존 ∧ 실제 개방
    ever_nocontact = _bz()          # 존 ∧ 무접촉
    ever_open_nc = _bz()            # 존 ∧ 개방 ∧ 무접촉
    ever_cand = _bz()               # 5조건 순간 동시 성립
    cand_steps = _bl()
    streak = _bl()
    max_streak = _bl()
    ever_placed = _bz()             # streak >= PLACED_MIN_CONSECUTIVE_STEPS
    forced = _bz()                  # mode B: 강제 개방 래치
    forced_at = _bl()               # 강제 개방 시작 스텝
    # ★G-probe(2026-09-10 신설, 진단 전용) — 지배적 실패 그룹 '파지 O·운반 X'의 기전 계측★
    #   [무엇을 재나] model_23000(2048ep) 평가에서 파지 93.5% → 배달 래치 49.9%로 ★43.7%p★가
    #     이 한 구간에서 죽는다. 그 그룹의 파지 유지율은 0.14, 큐브가 공중에 뜬 스텝은 2.7%뿐.
    #     후보 기전은 둘이고 서로 다른 해법을 요구한다:
    #       (a) 정책이 폐합 뒤 ★개방을 명령★해 놓친다 — BinaryJointPositionAction은 매 스텝
    #           action[-1] 부호로 명령을 재계산하며 latch가 없다(F67). 한 스텝만 뒤집혀도 열린다.
    #           ⇒ 해법은 폐합 래치(명령 이력만 쓰므로 실기 배포 가능).
    #       (b) 애초에 ★들어올리지 않는다★ — 팔이 상승하지 않는다. ⇒ 폐합 래치는 무용.
    #   [왜 carry_hold_deploy로는 못 가르나] 그 모드의 고정은 `_carried`(폐합 상태 TCP 1cm 상승)
    #     ★이후★에만 걸린다. 이 실패 그룹은 그 상승 자체에 실패하므로 발동 조건에 닿지 않는다.
    #     실측으로도 force_open_deploy 대비 개선이 없었다(36.33% vs 37.70%, 반복 변동 내).
    #   [판별] g_open_after_close > 0 이 흔하면 (a). g_max_rise가 lift_rise_m(0.010) 미만에
    #     머무르면 (b). 정책 raw 명령과 수퍼바이저 적용 후 명령을 ★둘 다★ 기록해 구분한다.
    g_closed_ever = _bz()           # ★유효 폐합★(적용 명령의 open→close 전이)이 있었는가
    g_applied_closed_prev = _bz()   # 직전 스텝의 적용 폐합 상태(전이 검출용)
    g_z_close = _bf(float("nan"))   # 첫 폐합 시점 TCP z(상승량 기준값)
    g_open_raw = _bl()              # 첫 폐합 이후 정책 raw 개방 명령 스텝 수   → (a)
    g_open_applied = _bl()          # 첫 폐합 이후 실제 적용된 개방 명령 스텝 수 → (a)
    g_max_rise = _bf(0.0)           # 첫 폐합 이후 TCP 최대 상승량[m]           → (b)
    # ★D-probe(2026-09-13 신설, 진단 전용) — 단계별 ★raw depth★ 거리 분포★
    #   [왜] Codex 루프 진단에서 depth 무효(=far 처리) 픽셀 비율이 approach 51% → ★grasp 70%★
    #     → lift 34% → carry 38%였고, 같은 grasp 단계에서 student의 행동 오차가 teacher 신호의
    #     60%(다른 단계 6~17%)로 가장 나쁘다. 전처리는 near(0.28 m) 미만을 far로 덮으므로
    #     파지 순간 큐브가 depth에서 '없는 것'으로 보인다. 허용 레버(depth 전처리)로 무엇을
    #     바꿔야 하는지는 ★무효 픽셀의 실제 거리★가 결정한다 → 단계별 raw depth 히스토그램.
    #   [단계 정의] Codex Worker와 동일: grasp=_grasped_mask(0.1N, 0.2m/s) 이력,
    #     lift=grasp∧obj z>MIN_LIFT 이력, carry=lift∧목표 8cm 이내 이력.
    _D_EDGES = torch.tensor([0.0, 0.10, 0.15, 0.20, 0.25, 0.28, 0.40, 0.60, 1.20, 1e9], device=dev)
    _D_LABELS = ("<0.10", "0.10-0.15", "0.15-0.20", "0.20-0.25", "0.25-0.28", "0.28-0.40",
                 "0.40-0.60", "0.60-1.20", ">=1.20")
    d_hist = {k: torch.zeros(len(_D_LABELS), dtype=torch.float64, device=dev)
              for k in ("approach", "grasp", "lift", "carry")}
    d_envsteps = {k: 0 for k in d_hist}
    d_flag_grasp = _bz(); d_flag_lift = _bz(); d_flag_carry = _bz()
    steps_to_settle = _bf("nan")    # 강제 개방 후 첫 placed까지 스텝
    # ★릴리스 직후 시계열 추적★ — '놓았는데 왜 안 멈추는가'를 스텝 단위로 본다.
    #   개방 명령 시작(t=0) 후 TRACE_LEN 스텝의 큐브 속도·거리·z·손가락 개방오차 중앙값.
    TRACE_LEN = 30
    #   a_cmd  = action manager가 실제로 받은 그리퍼 액션(지연 반영 전 raw)
    #   q_tgt  = 액추에이터에 기록된 finger 관절 ★목표★ — 이 값이 0인데 q가 안 따라오면
    #            '명령은 열림인데 물리적으로 막힘' = 판정이 아니라 기구/접촉 문제다.
    #   f_obj  = 손가락↔★큐브★ 접촉력(필터드, force_matrix_w) 최대
    #   f_net  = 손가락의 ★전체★ 접촉력(net_forces_w) 최대
    #     ⇒ f_net ≫ f_obj 이면 손가락이 큐브가 아닌 것(=벨트/자기 자신)에 닿아 있다는 뜻이다.
    #   f_z    = 손가락 링크 world z — 벨트 상면(TABLE_HEIGHT)보다 낮으면 벨트에 파묻힌 것.
    trace = {k: torch.full((n, TRACE_LEN), float("nan"), device=dev)
             for k in ("spd", "dist", "z", "open_err", "a_cmd", "q_tgt",
                       "f_obj", "f_net", "f_z")}
    traces: list[dict] = []

    records: list[dict] = []
    free_air_log: list[tuple] = []
    obs = env.get_observations()
    step = 0
    while len(records) < args_cli.n_episodes and step < args_cli.max_steps and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            if args_cli.mode == "settle_floor":
                actions = torch.zeros_like(actions)   # 팔·그리퍼 전부 정지
            if args_cli.mode == "free_air":
                # 팔은 정지(상대 IK delta=0), 그리퍼만 close 20스텝 → open
                actions = torch.zeros_like(actions)
                actions[:, -1] = -1.0 if step < 20 else 1.0
            elif args_cli.mode == "scripted_place":
                # ★★정책 배제 — 팔·그리퍼를 전부 스크립트로 구동(2026-08-12)★★
                #   목적: "완벽하게 놓으면 placed가 켜지는가"를 환경에 직접 묻는다.
                #   pregrasp 보조로 큐브를 쥔 채 시작한 env를 대상으로 4단계를 강제한다:
                #     ① ALIGN   타겟 xy로 수평 이동(높이 유지) — z는 건드리지 않는다
                #     ② DESCEND 안착 높이까지 수직 하강(그리퍼는 닫은 채)
                #     ③ OPEN    그리퍼 개방(action[-1]=+1)을 open_steps 동안 유지
                #     ④ RETREAT 수직 상승으로 손을 완전히 뺀다 → 이후 정지하고 placed 관측
                #   ⚠️이 모드가 placed를 못 켜면 ★정책 탓이 아니다★ — 판정식 또는 기하가 범인이다.
                _pos_s, _ = _nearest_object(uenv)
                _tgt_s = uenv.command_manager.get_command("target_pos")[:, :3]
                _dxy = _tgt_s[:, :2] - _pos_s[:, :2]
                _dz_need = _tgt_s[:, 2] - _pos_s[:, 2]
                _scripted = sp_t >= 0                # 인수한 env만 덮어쓴다
                actions = torch.where(_scripted.unsqueeze(1), torch.zeros_like(actions), actions)
                _t = sp_t                           # 인수 후 경과 스텝(−1=인수 전)
                _s_align = args_cli.sp_align
                _s_desc = _s_align + args_cli.sp_descend
                _s_open = _s_desc + args_cli.sp_open
                # 단계별 마스크
                _m_align = _scripted & (_t < _s_align)
                _m_desc = _scripted & (_t >= _s_align) & (_t < _s_desc)
                _m_open = _scripted & (_t >= _s_desc) & (_t < _s_open)
                _m_ret = _scripted & (_t >= _s_open)
                # ① 수평 정렬 — IK delta는 [-1,1] 정규화이므로 방향만 주고 크기는 clamp
                _gain = args_cli.sp_gain
                _ax = torch.clamp(_dxy[:, 0] * _gain, -1.0, 1.0)
                _ay = torch.clamp(_dxy[:, 1] * _gain, -1.0, 1.0)
                # ⚠️else는 ★정책 액션 보존★ — zeros_like로 덮으면 인수 전 env의 팔이 멈춰
                #   정책이 큐브에 접근하지 못한다(2026-08-12 1회차 실측: latch 97.7% → ★1.95%★).
                actions[:, 0] = torch.where(_m_align, _ax, actions[:, 0])
                actions[:, 1] = torch.where(_m_align, _ay, actions[:, 1])
                # ② 하강 — 남은 z 오차만큼(음수) 내려간다
                _az = torch.clamp(_dz_need * _gain, -1.0, 1.0)
                actions[:, 2] = torch.where(_m_desc, _az, actions[:, 2])
                # ③ 개방 / ④ 후퇴(+z 상승)
                actions[:, 2] = torch.where(_m_ret, torch.ones_like(_az) * args_cli.sp_retreat,
                                            actions[:, 2])
                _grip_scripted = torch.where(_m_align | _m_desc,
                                             -torch.ones_like(_az),   # 닫은 채 유지
                                             torch.ones_like(_az))    # 개방 후 유지
                actions[:, -1] = torch.where(_scripted, _grip_scripted, actions[:, -1])
            elif args_cli.mode == "descend_sweep":
                # 팔을 수직 하강만 시킨다(상대 IK delta z<0). 그리퍼는 닫은 채 유지.
                actions = torch.zeros_like(actions)
                actions[:, 2] = -1.0
                actions[:, -1] = -1.0
            if _FORCE_MODE:
                # ★트리거를 한 번이라도 만족한 env는 그 뒤로 그리퍼를 강제 개방★
                #   (BinaryJointPositionAction 부호 계약: action[-1] > 0 → open)
                actions = actions.clone()
                # ★carry_hold_deploy(2026-08-24) — 폐합 latch 후보의 직접 실측★
                #   [무엇을 재나] force_open_deploy와 릴리스 판정이 ★완전히 같고★, 다른 것은
                #     운반 중(carry latch 성립∧릴리스 전) 그리퍼를 ★닫힌 채로 고정★하는 것뿐이다.
                #     둘을 같은 checkpoint로 돌리면 그 차이가 곧 '폐합 latch의 효과'다.
                #   [왜 필요한가] BinaryJointPositionAction은 매 스텝 action[-1] 부호로 명령을
                #     재계산하며 latch가 없다(F67에서 소스 확인). 따라서 ★한 스텝만★ 부호가
                #     뒤집혀도 그리퍼가 열려 큐브를 놓는다. teacher는 정확한 상태라 흔들리지
                #     않지만 student는 112² RGB-D 한 장으로 판단해 흔들릴 수 있다 — 실측된
                #     병목(파지 88.5% → 배달 래치 68.4%, hold 28.8% vs teacher 49.7%)과 부합한다.
                #   [배포 적합성] 판정 입력이 그리퍼 명령 이력과 TCP z뿐이라 물체 pose를 쓰지
                #     않는다 = 실기에서 그대로 실행 가능하다(F66 릴리스 primitive와 같은 계약).
                if args_cli.mode == "carry_hold_deploy":
                    actions[_carried & (~forced), -1] = -1.0
                # ★grasp_hold_deploy(2026-09-11 신설) — 폐합 ★시점★에 거는 고정★
                #   [왜 carry_hold_deploy로 부족한가] 그 모드는 `_carried`(폐합 상태 TCP
                #     1cm 상승) 이후에만 고정한다. G-probe 실측에서 유효 폐합 뒤 TCP 최대
                #     상승량 중앙이 ★1~3mm★이고 ★59.6~64.1%★가 1cm 문턱에 닿지 못한다 —
                #     지배적 실패 그룹은 발동 조건에 애초에 도달하지 않는다. 실제로
                #     force_open_deploy 대비 개선이 없었다(36.33% vs 37.70%).
                #   [무엇을 재나] 릴리스 판정은 force_open_deploy와 ★완전히 동일★하고,
                #     다른 것은 ★유효 폐합(may_close 게이트를 통과한 open→close 전이)★
                #     이후 릴리스 전까지 그리퍼를 닫힌 채 고정하는 것뿐이다. 같은 checkpoint로
                #     두 모드를 돌리면 그 차이가 곧 '폐합 시점 latch의 효과'다.
                #   [왜 필요한가] G-probe 실측: 유효 폐합 뒤 정책이 개방을 명령한 에피소드가
                #     ★74.6~78.1%★(중앙 9~11 스텝). BinaryJointPositionAction은 매 스텝
                #     부호로 명령을 재계산하며 latch가 없어(F67) 한 스텝만 뒤집혀도 놓친다.
                #   [배포 적합성] 판정 입력이 ★그리퍼 명령 이력과 TCP z★뿐이라 물체 pose를
                #     쓰지 않는다 = 실기에서 그대로 실행 가능(기존 릴리스 primitive와 같은 계약).
                if args_cli.mode == "grasp_hold_deploy":
                    actions[g_closed_ever & (~forced), -1] = -1.0
                actions[forced, -1] = 1.0
                # ★force_open_freeze(2026-08-16 신설)★ — 개방과 동시에 팔 액션을 0으로 고정한다.
                #   [왜] q=0.020/0.017/0.016 세 폐합 모두 placed(≥3스텝)가 정확히 0.00%였고
                #     개방 후 큐브 속도가 0.167~0.183 m/s로 안착 바닥(0.003~0.006)의 30~60배였다.
                #     침투 depenetration이 주원인이라면 폐합을 8mm→4mm로 줄일 때 속도가 크게
                #     떨어져야 하는데 9%만 줄었다 ⇒ ★남은 용의자는 '팔이 계속 움직인다'★이다.
                #     실제로 큐브 z가 안착고(−0.0125)에 못 내려오고 −0.005~−0.010에서 떠 있고
                #     f_obj가 18~21N으로 되돌아온다 = 손가락이 반복 재접촉.
                #   [무엇을 가르는가] 이 모드에서 큐브가 안착하면 원인은 ★릴리스 후 팔 거동★이고
                #     (해법: 릴리스 직후 정지/후퇴를 행동 계약에 넣는다), 그래도 안 되면 원인은
                #     ★파지-접촉 모델 자체★다(해법: 큐브/그리퍼 기하·솔버 재설계).
                if args_cli.mode in ("force_open_freeze", "force_open_tcp", "force_open_settle", "force_open_deploy"):
                    actions[forced, :-1] = 0.0   # 마지막 = 그리퍼 차원(개방 유지)
                # ★force_open_retreat(2026-08-16 신설)★ — 개방과 동시에 ★수직 상승★한다.
                #   [왜] freeze 모드에서 큐브 속도는 안착 바닥(0.0041)까지 떨어졌는데 placed는
                #     여전히 ≥3스텝 0.00%였고, 큐브 z가 −0.009로 ★안착고(−0.0125)보다 3.5mm 높다★.
                #     = 큐브가 벨트가 아니라 ★멈춰 선 손가락 위에 얹혀★ 있다는 뜻이고, 그래서
                #     `_any_finger_contact`가 계속 참이라 무접촉 항이 성립하지 않는다.
                #   [해법] 개방 후 팔을 ★들어올려 손가락을 큐브에서 빼낸다★ = 튜토리얼 phase 7.
                if args_cli.mode == "force_open_retreat":
                    actions[forced, :-1] = 0.0
                    actions[forced, 2] = float(args_cli.sp_retreat)   # +z 상승
            # ★G-probe★ 정책 raw 그리퍼 명령을 ★강제 개방 덮어쓰기 이후·env.step 직전★에
            #   붙잡는다. force 모드에서는 이 값이 이미 덮어써져 있으므로 (a)/(b) 판별은
            #   policy 모드 산출물로만 해석한다(모드별 오독 방지 — summary에 모드를 명기).
            _g_raw = actions[:, -1].detach().clone()
            obs, _, dones, _ = env.step(actions)

            pos, _ = _nearest_object(uenv)
            target = uenv.command_manager.get_command("target_pos")[:, :3]
            dist = torch.norm(pos - target, dim=1)
            spd = _nearest_object_vel(uenv).norm(dim=-1)
            latch = _lifted_latch(uenv)
            contact = _any_finger_contact(uenv, _CONTACT_N)
            opened = _gripper_open_mask(uenv)
            q = uenv.scene["robot"].data.joint_pos[:, grip_ids]
            open_err = (q - GRIPPER_OPEN_M).abs().amax(dim=1)
            not_done = ~dones.to(dev).bool().flatten()

            zone = latch & (dist < tol) & not_done
            # ★2026-08-09(F50) 판정 재설계 반영★: 성공 게이트는 `placed_success_mask`와 동일하게
            #   ¬접촉까지다. `opened`(관절 허용대)는 게이트에서 빠졌고 ★진단 지표로만★ 보고한다.
            # ★★2026-08-11 공유 mask 경유로 정정 — 진단이 학습보다 헐거웠다★★
            #   [사고] 여기서 AND를 ★재구현★하고 있었는데, 같은 날 `placed_success_mask`에
            #     ★표면 안착 조건(|z − z_target| < PLACED_SURFACE_TOL_M)★이 추가되면서
            #     이 줄만 갱신되지 않았다 ⇒ ★프로브가 학습·eval보다 헐거워 placed를 낙관 편향★으로
            #     보고한다. `BinPick_Eval.py`는 처음부터 공유 mask를 호출해 무사했다.
            #   [정정] 재구현을 폐기하고 ★학습 보상과 같은 함수★를 호출한다 — §9-5 2번
            #     '측정 맥락과 대상 맥락을 항목별로 대조하라'를 사람 주의력이 아니라 ★구조★로 만든다.
            #   `zone`(래치∧거리∧not_done)은 아래 항별 분해 보고에 계속 쓰이므로 그대로 둔다.
            cand = placed_success_mask(
                uenv,
                tol=tol,
                max_speed=PLACED_MAX_SPEED,
                contact_n=_CONTACT_N,
                delivery_latch=latch,
            ) & not_done
            streak = torch.where(cand, streak + 1, torch.zeros_like(streak))
            placed_now = streak >= PLACED_MIN_CONSECUTIVE_STEPS

            ep_steps += 1
            ever_latch |= latch & not_done
            ever_zone |= zone
            zone_steps += zone.long()

            # ── ★D-probe 계측(모드 무관)★ — raw depth는 노이즈·전처리 이전 센서 출력 ──
            _dd = uenv.scene["tiled_camera"].data.output["distance_to_image_plane"][..., 0]
            _dd = torch.nan_to_num(_dd, nan=1e9, posinf=1e9, neginf=0.0)
            _dg = _grasped_mask(uenv, force_n=0.1, rel_speed_mps=0.2)
            d_flag_grasp |= _dg & not_done
            d_flag_lift |= _dg & (pos[:, 2] > MIN_LIFT) & not_done
            d_flag_carry |= d_flag_lift & (dist < 0.08) & not_done
            _ds = torch.zeros(n, dtype=torch.long, device=dev)
            _ds[d_flag_grasp] = 1; _ds[d_flag_lift] = 2; _ds[d_flag_carry] = 3
            for _di, _dk in enumerate(d_hist):
                _dm = (_ds == _di) & not_done
                _dc = int(_dm.sum())
                if _dc:
                    _db = torch.bucketize(_dd[_dm].flatten(), _D_EDGES[1:-1], right=False)
                    d_hist[_dk] += torch.bincount(_db, minlength=len(_D_LABELS)).to(torch.float64)
                    d_envsteps[_dk] += _dc

            # ── ★G-probe 계측(모드 무관)★ ──
            #   폐합 부호 계약은 환경과 동일: action[-1] < 0 = close, > 0 = open.
            #   ★기준점은 반드시 '적용된 명령의 open→close 전이'에 잡는다★
            #     [사고 2026-09-10] 처음엔 '첫 raw 폐합 명령'에 잡았더니 상승량 중앙이 정확히
            #       0.0000 m·문턱 미달 100%로 나왔다. 정책은 ★스텝 0부터 그리퍼 액션이 음수★라
            #       (§7.5 실측 grip 중앙 −0.459) 기준점이 ★홈 자세 높이★로 굳고, 이후 팔은
            #       하강만 하므로 상승량이 영원히 0이 된다 — 지표가 죽어 있었다.
            #     [정정] 수퍼바이저와 ★같은 방식★으로 간다. may_close 게이트(TCP z<0.040)를
            #       통과해 실제로 적용된 폐합만 전이로 인정하고, 열렸다 다시 닫히면 기준점을
            #       ★다시★ 잡는다(F71 계약: 절대 z가 아니라 폐합 시점 대비 상대 상승량).
            _g_applied = uenv.action_manager.action[:, -1]
            _g_tcpz = bp_obs_tcp(uenv)[:, 2]
            _g_cl = (_g_applied < 0.0) & not_done
            _g_new = _g_cl & (~g_applied_closed_prev)          # open→close 전이 = 새 기준점
            g_z_close = torch.where(_g_new, _g_tcpz, g_z_close)
            g_applied_closed_prev = _g_cl.clone()
            g_closed_ever |= _g_new
            _g_after = g_closed_ever & not_done                 # 유효 폐합 이후 구간만 집계
            g_open_raw += (_g_after & (_g_raw > 0.0)).long()
            g_open_applied += (_g_after & (_g_applied > 0.0)).long()
            g_max_rise = torch.maximum(
                g_max_rise,
                torch.where(_g_after & (g_z_close == g_z_close),
                            _g_tcpz - g_z_close, torch.zeros_like(_g_tcpz)),
            )
            _inf = torch.full_like(spd, float("inf"))
            min_spd_zone = torch.minimum(min_spd_zone, torch.where(zone, spd, _inf))
            min_z_zone = torch.minimum(min_z_zone, torch.where(zone, pos[:, 2], _inf))
            min_openerr_zone = torch.minimum(min_openerr_zone, torch.where(zone, open_err, _inf))
            max_agrip_zone = torch.maximum(
                max_agrip_zone, torch.where(zone, actions[:, -1].float(), torch.full_like(spd, -99.0))
            )
            ever_settled |= zone & (spd < PLACED_MAX_SPEED)
            ever_open |= zone & opened
            ever_nocontact |= zone & (~contact)
            ever_open_nc |= zone & opened & (~contact)
            ever_cand |= cand
            cand_steps += cand.long()
            max_streak = torch.maximum(max_streak, streak)
            newly_placed = placed_now & (~ever_placed)
            steps_to_settle = torch.where(
                newly_placed & (forced_at > 0),
                (ep_steps - forced_at).float(),
                steps_to_settle,
            )
            ever_placed |= placed_now

            if args_cli.mode == "scripted_place":
                # 래치(파지∧들림)가 성립한 env부터 인수. 이후 매 스텝 증가.
                _start = latch & not_done & (sp_t < 0)
                sp_t = torch.where(_start, torch.zeros_like(sp_t), sp_t)
                sp_t = torch.where(sp_t >= 0, sp_t + 1, sp_t)

            if _FORCE_MODE:
                # force_open      : 타겟존(래치∧3cm) 진입 = 벨트 바로 위에서 개방
                # force_open_high : 래치∧큐브 z>high_z = ★공중★에서 개방(접촉 간섭 없는 대조군)
                if args_cli.mode == "force_open":
                    trigger = zone
                elif args_cli.mode == "force_open_high":
                    trigger = latch & (pos[:, 2] > args_cli.high_z) & not_done
                elif args_cli.mode in ("force_open_deploy", "carry_hold_deploy",
                                       "grasp_hold_deploy"):
                    # ★force_open_deploy(2026-08-16) — ★배포와 문자 그대로 같은 판정★★
                    #   common/release_supervisor.py를 그대로 호출한다. 학습 액션항과 실기
                    #   rollout이 쓸 바로 그 코드이므로, 여기서 나온 수치가 곧 계약의 성능이다.
                    #   ⚠️latch도 ★명령 기반★이다 — 실기 2FG7에 폭/파지 feedback이 없어서
                    #     sim에서만 접촉력으로 latch를 걸면 배포에서 다른 시점에 걸린다.
                    tcp = bp_obs_tcp(uenv)
                    if _prev_tcp is None:
                        _prev_tcp = tcp[:, :3].clone()
                    tcp_spd = (tcp[:, :3] - _prev_tcp).norm(dim=1) / float(uenv.step_dt)
                    _prev_tcp = tcp[:, :3].clone()
                    grip_closed = uenv.action_manager.action[:, -1] < 0.0
                    # ★F71★ carry latch는 ★폐합 시점 대비 상대 상승량★을 받는다(계약).
                    #   절대 z를 넘기면 닫는 순간 latch가 굳어 파지 직후 릴리스가 터진다.
                    _newly = grip_closed & (~_grip_closed_prev)
                    _z_at_close = torch.where(_newly, tcp[:, 2], _z_at_close)
                    _grip_closed_prev = grip_closed.clone()
                    _carried = _rs.carry_latch_update(
                        _carried, grip_closed, tcp[:, 2] - _z_at_close, _rs_cfg)
                    _settle = _rs.settle_counter_update(_settle, tcp_spd, _rs_cfg)  # ★F92★ 연속 정지 창
                    trigger = _rs.release_decision(_carried, tcp[:, 2], _settle, _rs_cfg) & not_done
                elif args_cli.mode == "force_open_settle":
                    # ★force_open_settle(2026-08-16) — 배포 가능 트리거 2안★
                    #   [왜 TCP-xy가 실패했나] force_open_tcp는 ‖TCP_xy − target_xy‖로 쟀는데,
                    #     파지 오프셋 때문에 TCP_xy ≠ cube_xy라 큐브 기반(63.7%) 대비 24%에 그쳤다.
                    #   [이 트리거] '큐브가 타겟 위인가'를 묻지 않고 ★정책이 배치를 마치고
                    #     멈췄는가★를 묻는다: 파지 래치 ∧ TCP가 배치 높이까지 내려옴 ∧ TCP 정지.
                    #     어디에 놓을지는 ★정책이 정한다★(정책은 target_pos를 관측하고 큐브-타겟
                    #     거리로 보상받으므로, 큐브가 타겟 위일 때 멈추도록 학습된다).
                    #   [배포 가용성] TCP pose = RTDE/FK, TCP 속도 = 연속 FK 차분 또는 관절속도.
                    #     ★큐브 pose 추정기가 필요 없다★ = 비전 없이 트리거된다.
                    tcp = bp_obs_tcp(uenv)
                    if _prev_tcp is None:
                        _prev_tcp = tcp[:, :3].clone()
                    tcp_spd = (tcp[:, :3] - _prev_tcp).norm(dim=1) / float(uenv.step_dt)
                    _prev_tcp = tcp[:, :3].clone()
                    trigger = (
                        latch
                        & (tcp[:, 2] < args_cli.tcp_z)
                        & (tcp_spd < args_cli.tcp_settle_mps)
                        & not_done
                    )
                elif args_cli.mode == "force_open_tcp":
                    # ★force_open_tcp(2026-08-16 신설) — ★배포 가능한 트리거★★
                    #   [왜] force_open_smart/freeze의 트리거는 ★큐브 pose(sim ground truth)★를
                    #     쓴다. 실기에는 그 신호가 없다(D435i RGB-D만 있고 큐브 pose 추정기는
                    #     구현·검증된 바 없다). sim에서만 되는 트리거로 63.7%를 확인해 봐야
                    #     배포로 못 넘어간다.
                    #   [해법] 큐브는 파지되어 TCP에 강체로 붙어 있으므로 ★TCP pose★로 대체한다.
                    #     TCP는 실기에서 RTDE/FK로 얻고(관측 슬롯 tcp_pose(7)), target_pos(3)는
                    #     입력으로 주어진다 ⇒ ★두 신호 모두 배포 시 가용★. 비전 추정기 불필요.
                    #   ⚠️ z 임계는 큐브 중심이 아니라 TCP(파지중심) 기준이라 값이 다르다 —
                    #     파지 오프셋만큼 위에 있으므로 smart_z에 그 차이를 더해 쓴다.
                    tcp = bp_obs_tcp(uenv)              # (N,7) base frame pos+quat
                    xy_err = torch.norm(tcp[:, :2] - target[:, :2], dim=1)
                    trigger = (
                        latch
                        & (xy_err < args_cli.smart_xy_tol)
                        & (tcp[:, 2] < args_cli.tcp_z)
                        & not_done
                    )
                else:  # force_open_smart — xy 정렬 ∧ 벨트 여유 높이에서만 개방
                    xy_err = torch.norm(pos[:, :2] - target[:, :2], dim=1)
                    trigger = (
                        latch
                        & (xy_err < args_cli.smart_xy_tol)
                        & (pos[:, 2] < args_cli.smart_z)
                        # ★F51 하강 바닥 도입 후★: 바닥(TABLE_HEIGHT+0.030)이 패드-벨트 충돌을
                        #   막아 주므로 하한은 안착 높이 근처까지 내려도 안전하다.
                        & (pos[:, 2] > _TARGET_Z)
                        & not_done
                    )
                newly_forced = trigger & (~forced)
                forced_at = torch.where(newly_forced, ep_steps, forced_at)
                forced |= trigger
                # 개방 명령 후 경과 스텝(t=0 = 첫 강제 개방 스텝)
                k = ep_steps - forced_at
                in_trace = forced & (k >= 0) & (k < TRACE_LEN)
                if bool(in_trace.any()):
                    rows = torch.nonzero(in_trace).flatten()
                    cols = k[rows]
                    trace["spd"][rows, cols] = spd[rows]
                    trace["dist"][rows, cols] = dist[rows]
                    trace["z"][rows, cols] = pos[rows, 2]
                    trace["open_err"][rows, cols] = open_err[rows]
                    trace["a_cmd"][rows, cols] = uenv.action_manager.action[rows, -1].float()
                    trace["q_tgt"][rows, cols] = (
                        uenv.scene["robot"].data.joint_pos_target[:, grip_ids].amax(dim=1)[rows]
                    )
                    f_obj, f_net = _finger_forces(uenv)
                    trace["f_obj"][rows, cols] = f_obj[rows]
                    trace["f_net"][rows, cols] = f_net[rows]
                    trace["f_z"][rows, cols] = (
                        uenv.scene["robot"].data.body_pos_w[:, finger_body_ids, 2].amin(dim=1)[rows]
                    )

            # ── 완료 에피소드 수확 ──
            done_ids = torch.nonzero(dones.flatten()).flatten()
            for i in done_ids.tolist():
                records.append({
                    "ep_steps": int(ep_steps[i]),
                    "ever_latch": int(ever_latch[i]),
                    "ever_zone": int(ever_zone[i]),
                    "zone_steps": int(zone_steps[i]),
                    "min_speed_zone": float(min_spd_zone[i]),
                    "min_z_zone": float(min_z_zone[i]),
                    "min_open_err_zone": float(min_openerr_zone[i]),
                    "max_agrip_zone": float(max_agrip_zone[i]),
                    "ever_settled": int(ever_settled[i]),
                    "ever_open": int(ever_open[i]),
                    "ever_nocontact": int(ever_nocontact[i]),
                    "ever_open_nocontact": int(ever_open_nc[i]),
                    "ever_candidate": int(ever_cand[i]),
                    "candidate_steps": int(cand_steps[i]),
                    "max_streak": int(max_streak[i]),
                    "placed": int(ever_placed[i]),
                    "forced_at": int(forced_at[i]),
                    "steps_to_placed_after_open": float(steps_to_settle[i]),
                    # ★G-probe★ — 파지 후 운반 실패 기전 판별용(위 누적기 주석 참조)
                    "g_closed_ever": int(g_closed_ever[i]),
                    "g_open_raw": int(g_open_raw[i]),
                    "g_open_applied": int(g_open_applied[i]),
                    "g_max_rise": float(g_max_rise[i]),
                })
                if _FORCE_MODE and bool(forced[i]):
                    traces.append({k: trace[k][i].tolist() for k in trace})
            if done_ids.numel() > 0:
                z = torch.zeros_like(ep_steps[done_ids])
                ep_steps[done_ids] = z
                sp_t[done_ids] = z - 1          # scripted_place 인수 상태 해제(−1)
                zone_steps[done_ids] = z
                cand_steps[done_ids] = z
                max_streak[done_ids] = z
                streak[done_ids] = z
                forced_at[done_ids] = z
                g_open_raw[done_ids] = z
                g_open_applied[done_ids] = z
                for b in (ever_latch, ever_zone, ever_settled, ever_open, ever_nocontact,
                          ever_open_nc, ever_cand, ever_placed, forced):
                    b[done_ids] = False
                # ★G-probe 누적기도 함께 해제★ — 안 하면 다음 에피소드가 '이미 폐합함'
                #   상태로 시작해 상승량 기준값(g_z_close)이 이전 에피소드 것으로 굳는다.
                g_closed_ever[done_ids] = False
                g_applied_closed_prev[done_ids] = False
                g_z_close[done_ids] = float("nan")
                g_max_rise[done_ids] = 0.0
                d_flag_grasp[done_ids] = False
                d_flag_lift[done_ids] = False
                d_flag_carry[done_ids] = False
                # ★수퍼바이저 운반 latch도 반드시 함께 해제한다★ — 안 하면 다음 에피소드가
                #   '이미 운반 중' 상태로 시작해 파지 전에 릴리스가 발화한다(계측 오염).
                _carried[done_ids] = False
                _grip_closed_prev[done_ids] = False
                _z_at_close[done_ids] = 0.0
                for f, v in ((min_spd_zone, float("inf")), (min_z_zone, float("inf")),
                             (min_openerr_zone, float("inf")), (max_agrip_zone, -99.0),
                             (steps_to_settle, float("nan"))):
                    f[done_ids] = v
                for t in trace.values():
                    t[done_ids] = float("nan")
            if args_cli.mode == "settle_floor":
                if step >= args_cli.settle_steps:
                    free_air_log.extend(
                        (step, float(v), float(z))
                        for v, z in zip(spd.tolist(), pos[:, 2].tolist())
                    )
                if step >= args_cli.settle_steps + 60:
                    break
            if args_cli.mode == "descend_sweep":
                f_obj_s, f_net_s = _finger_forces(uenv)
                tcp_z = bp_obs_tcp(uenv)[:, 2]
                free_air_log.append(
                    (step, float(tcp_z.median()),
                     float(uenv.scene["robot"].data.body_pos_w[:, finger_body_ids, 2].amin(dim=1).median()),
                     float(f_obj_s.median()), float(f_net_s.median()),
                     float((f_net_s > 5.0).float().mean()))
                )
                if step >= 60:
                    break
            if args_cli.mode == "free_air":
                q_all = uenv.scene["robot"].data.joint_pos[:, grip_ids]
                tgt = uenv.scene["robot"].data.joint_pos_target[:, grip_ids]
                free_air_log.append(
                    (step, float(uenv.action_manager.action[:, -1].median()),
                     float(tgt.amax(dim=1).median()),
                     float(q_all[:, 0].median()), float(q_all[:, 1].median()),
                     float(contact.float().mean()))
                )
                if step >= 80:
                    break
        step += 1
        if step % 200 == 0:
            print(f"[probe] step {step} · 완료 에피소드 {len(records)}")

    if args_cli.mode == "settle_floor":
        env.close()
        vs = sorted(v for _s, v, _z in free_air_log)
        zs = sorted(z for _s, _v, z in free_air_log)
        n = len(vs)
        q = lambda p: vs[min(n - 1, int(p * n))]
        text = chr(10).join([
            "UR5e — ★정지한 큐브의 속도 노이즈 바닥 실측★ (팔 무동작·자연 안착)",
            "=" * 70,
            f"표본 {n} (env × step) · 안착 대기 {args_cli.settle_steps} 스텝 후 60 스텝 수집",
            f"큐브 z 중앙 {zs[n//2]:.5f} m (안착 기대값 {_TARGET_Z:.4f})",
            "-" * 70,
            "속도 분위수 [m/s]",
            f"  p50  {q(0.50):.6f}",
            f"  p90  {q(0.90):.6f}",
            f"  p99  {q(0.99):.6f}",
            f"  p999 {q(0.999):.6f}",
            f"  max  {vs[-1]:.6f}",
            "-" * 70,
            "판정 지침: placed 정지 임계는 ★p99.9보다 위★여야 한다 —",
            "           그 아래로 잡으면 '진짜 정지'와 '솔버 노이즈'를 구분할 수 없다.",
            f"git : {_git_hash()}",
        ])
        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(text + chr(10))
        print(text)
        print(f"[probe] 산출물: {out_dir}")
        return

    if args_cli.mode == "descend_sweep":
        env.close()
        rows = ["  step | tcp_z[m] | finger_z[m] | f_obj[N] | f_net[N] | 접촉env"]
        rows += [f"  {st:4d} | {tz:8.4f} | {fz:11.4f} | {fo:8.3f} | {fn:8.3f} | {fr:6.1%}"
                 for st, tz, fz, fo, fn, fr in free_air_log]
        hit = [r for r in free_air_log if r[4] > 5.0 and r[3] < 1.0]
        first = hit[0] if hit else None
        text = chr(10).join(
            ["UR5e — ★수직 하강 스윕: 패드-벨트 접촉 높이 실측★",
             "=" * 70,
             f"벨트 상면(TABLE_HEIGHT) = {_TABLE_HEIGHT:.4f} m · 타겟 안착 z = {_TARGET_Z:.4f} m",
             "그리퍼 닫힘 유지·수직 하강만(상대 IK delta z<0). f_obj≈0인데 f_net이 튀면 = 벨트 접촉.",
             "-" * 70] + rows + ["-" * 70,
             (f"★첫 비-큐브 접촉★: step {first[0]} · TCP z = {first[1]:.4f} m · "
              f"finger_link z = {first[2]:.4f} m · f_net {first[4]:.1f} N"
              if first else "★비-큐브 접촉이 감지되지 않음 — 하강 폭/스텝 수를 늘려 재측정할 것★"),
             (f"⇒ 하강 바닥 후보 = TCP z ≥ {first[1]:.4f} m "
              f"(타겟 안착 z {_TARGET_Z:.4f} 대비 여유 {first[1]-_TARGET_Z:+.4f} m)"
              if first else ""),
             f"git             : {_git_hash()}"])
        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(text + chr(10))
        print(text)
        print(f"[probe] 산출물: {out_dir}")
        return

    if args_cli.mode == "free_air":
        env.close()
        hdr = "  step | a_cmd | q_tgt[m] | q_L[m]  | q_R[m]  | contact"
        rows = [hdr] + [
            f"  {st:4d} | {a:5.1f} | {t:8.4f} | {l:7.4f} | {r:7.4f} | {c:6.1%}"
            for st, a, t, l, r, c in free_air_log
        ]
        text = "\n".join(
            ["UR5e — ★무접촉(free-air) 그리퍼 액추에이터 테스트★",
             "=" * 62,
             f"체크포인트      : {resume_path}",
             f"판정            : open 목표 q={GRIPPER_OPEN_M} · 허용 {PLACED_GRIPPER_OPEN_TOL_M} m · "
             f"close 목표 {0.020} m",
             "팔 액션 0 고정(홈 유지) · 그리퍼만 close 20스텝 → open",
             "-" * 62] + rows + [
             "-" * 62,
             "판정 지침: 무접촉에서도 q가 open 허용대에 못 들어오면 ★액추에이터/관절 한계 설정 결함★.",
             "           무접촉에서는 들어오는데 타겟존에서만 막히면 ★접촉 간섭(벨트/큐브)★.",
             f"git             : {_git_hash()}"])
        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)
        print(f"[probe] 산출물: {out_dir}")
        return

    env.close()
    records = records[: args_cli.n_episodes]
    N = len(records)
    if N == 0:
        raise SystemExit("[probe] ⚠️ 완료 에피소드 0 — 집계 불가")

    def frac(key: str) -> float:
        return 100.0 * sum(r[key] for r in records) / N

    def med(key: str, only_zone: bool = True) -> float:
        vals = sorted(r[key] for r in records
                      if (not only_zone or r["ever_zone"]) and r[key] == r[key] and r[key] != float("inf"))
        return vals[len(vals) // 2] if vals else float("nan")

    stats = {
        "mode": args_cli.mode,
        "checkpoint": str(resume_path),
        "task": args_cli.task,
        "n_episodes": N,
        "num_envs": n,
        "seed": args_cli.seed,
        "place_tol_m": tol,
        "placed_max_speed_mps": PLACED_MAX_SPEED,
        "placed_open_tol_m": PLACED_GRIPPER_OPEN_TOL_M,
        "placed_min_steps": PLACED_MIN_CONSECUTIVE_STEPS,
        "gripper_open_target_m": GRIPPER_OPEN_M,
        "target_z_m": _TARGET_Z,
        "min_lift_m": MIN_LIFT,
        "git": _git_hash(),
        "pct": {
            "latch": frac("ever_latch"),
            "zone(latch&dist<tol)": frac("ever_zone"),
            "zone&settled": frac("ever_settled"),
            "zone&open": frac("ever_open"),
            "zone&nocontact": frac("ever_nocontact"),
            "zone&open&nocontact": frac("ever_open_nocontact"),
            # ★게이트 조건 수는 F50 이후 4개(래치·거리·정지·무접촉)★ — 개도는 게이트에서 빠졌다.
            "candidate(게이트 동시)": frac("ever_candidate"),
            f"placed({PLACED_MIN_CONSECUTIVE_STEPS}스텝 지속)": frac("placed"),
        },
        "median_in_zone": {
            "zone_steps": med("zone_steps"),
            "min_speed_mps": med("min_speed_zone"),
            "min_z_m": med("min_z_zone"),
            "min_open_err_m": med("min_open_err_zone"),
            "max_gripper_action": med("max_agrip_zone"),
            "steps_to_placed_after_forced_open": med("steps_to_placed_after_open"),
        },
    }

    # ── ★G-probe — '파지 O·운반 X' 기전 판별★ ──
    #   모분모는 ★폐합을 명령한 에피소드★다. 폐합조차 없던 에피소드를 섞으면 (a)/(b) 어느
    #   쪽도 아닌 '접근 실패'가 비율을 희석해 판별을 흐린다.
    _closed = [r for r in records if r.get("g_closed_ever")]
    if _closed:
        def _pct(pred) -> float:
            return 100.0 * sum(1 for r in _closed if pred(r)) / len(_closed)

        def _med_c(key: str) -> float:
            vals = sorted(r[key] for r in _closed if r[key] == r[key])
            return vals[len(vals) // 2] if vals else float("nan")

        _lift = float(_rs.load_release_supervisor_cfg().lift_rise_m)
        stats["g_probe"] = {
            "n_all": N,
            "n_closed": len(_closed),
            "lift_rise_m": _lift,
            "pct_open_raw": _pct(lambda r: r["g_open_raw"] > 0),
            "pct_open_applied": _pct(lambda r: r["g_open_applied"] > 0),
            "med_open_raw": _med_c("g_open_raw"),
            "med_open_applied": _med_c("g_open_applied"),
            "med_max_rise": _med_c("g_max_rise"),
            "pct_rise_below": _pct(lambda r: r["g_max_rise"] < _lift),
        }

    # ── ★D-probe 통계★ — 단계별 raw depth 분포(픽셀 비율 %) ──
    stats["d_probe"] = {"bins": list(_D_LABELS), "stages": {}}
    for _dk, _dh in d_hist.items():
        _tot = float(_dh.sum())
        _pct = [100.0 * float(v) / _tot for v in _dh] if _tot > 0 else [float("nan")] * len(_D_LABELS)
        _cum = lambda upto: sum(_pct[:upto]) if _tot > 0 else float("nan")
        stats["d_probe"]["stages"][_dk] = {
            "env_steps": d_envsteps[_dk], "pct_by_bin": _pct,
            "pct_below_0.10": _cum(1), "pct_below_0.15": _cum(2), "pct_below_0.20": _cum(3),
            "pct_below_0.25": _cum(4), "pct_below_0.28": _cum(5),
        }

    # ── 릴리스 직후 시계열(중앙값) — '놓았는데 왜 안 멈추는가' ──
    if traces:
        def _med(vals: list[float]) -> float:
            vals = sorted(v for v in vals if v == v)
            return vals[len(vals) // 2] if vals else float("nan")

        stats_trace = {
            key: [_med([tr[key][k] for tr in traces]) for k in range(len(traces[0][key]))]
            for key in traces[0]
        }
        stats["post_open_median"] = {"n_traces": len(traces), **stats_trace}

    with open(os.path.join(out_dir, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(out_dir, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    lines = [
        "UR5e teacher — placed 달성가능성 프로브 (진단 전용)",
        "=" * 62,
        f"모드            : {args_cli.mode}",
        f"체크포인트      : {resume_path}",
        f"에피소드/seed   : {N} / {args_cli.seed}",
        f"판정 상수       : tol {tol} m · 정지 {PLACED_MAX_SPEED} m/s · "
        f"개방 |q-{GRIPPER_OPEN_M}|≤{PLACED_GRIPPER_OPEN_TOL_M} m · {PLACED_MIN_CONSECUTIVE_STEPS}스텝 지속",
        "-" * 62,
        "★placed 5조건 항별 분해 (에피소드 비율)★",
    ]
    for k, v in stats["pct"].items():
        lines.append(f"  {k:<28s}: {v:6.2f}%")
    lines += ["-" * 62, "★타겟존 내 물리량 중앙값★"]
    for k, v in stats["median_in_zone"].items():
        lines.append(f"  {k:<34s}: {v:.5g}")
    if "post_open_median" in stats:
        pom = stats["post_open_median"]
        lines += ["-" * 62,
                  f"★강제 개방 후 시계열 중앙값 (n={pom['n_traces']})★  t=0이 개방 명령 첫 스텝",
                  "   t | spd[m/s] | dist[m] | z[m]   | open_err[m] | a_cmd  | q_tgt[m] "
                  "| f_obj[N] | f_net[N] | finger_z[m]"]
        for k in range(0, len(pom["spd"]), 2):
            lines.append(f"  {k:2d} | {pom['spd'][k]:8.4f} | {pom['dist'][k]:7.4f} | "
                         f"{pom['z'][k]:6.4f} | {pom['open_err'][k]:11.4f} | "
                         f"{pom['a_cmd'][k]:6.2f} | {pom['q_tgt'][k]:8.4f} | "
                         f"{pom['f_obj'][k]:8.3f} | {pom['f_net'][k]:8.3f} | {pom['f_z'][k]:.4f}")
    if "d_probe" in stats:
        dp = stats["d_probe"]
        lines += ["-" * 62,
                  "★D-probe — 단계별 raw depth 분포(픽셀 %)★  (현행 near 0.28 m 미만 = far로 덮임)",
                  "  stage     env-steps  " + " ".join(f"{b:>9s}" for b in dp["bins"])]
        for _dk, _dv in dp["stages"].items():
            lines.append(f"  {_dk:<9s} {_dv['env_steps']:>9d}  " +
                         " ".join(f"{v:9.1f}" for v in _dv["pct_by_bin"]))
        lines.append("  누적(<near 후보) :  " + "  ".join(
            f"{_dk}: <0.10 {_dv['pct_below_0.10']:.0f}% / <0.15 {_dv['pct_below_0.15']:.0f}% / "
            f"<0.20 {_dv['pct_below_0.20']:.0f}% / <0.28 {_dv['pct_below_0.28']:.0f}%"
            for _dk, _dv in dp["stages"].items()))
    if "g_probe" in stats:
        gp = stats["g_probe"]
        lines += [
            "-" * 62,
            f"★G-probe — '파지 O·운반 X' 기전 판별★ (모드={stats['mode']}, "
            f"폐합한 에피소드 {gp['n_closed']}/{gp['n_all']})",
            "  [(a) 폐합 뒤 개방 명령] 정책 raw 개방 스텝>0 인 에피소드 : "
            f"{gp['pct_open_raw']:6.2f}%   (중앙 {gp['med_open_raw']:.0f} 스텝)",
            "  [(a) 실제 적용된 개방]   적용 개방 스텝>0 인 에피소드 : "
            f"{gp['pct_open_applied']:6.2f}%   (중앙 {gp['med_open_applied']:.0f} 스텝)",
            "  [(b) 들어올리지 않음]   폐합 후 TCP 최대 상승량 중앙 : "
            f"{gp['med_max_rise']:.4f} m   (운반 래치 문턱 {gp['lift_rise_m']:.3f} m)",
            f"     └ 상승량이 문턱 미만인 에피소드 : {gp['pct_rise_below']:6.2f}%",
            "  판별: 개방 비율↑ ⇒ (a) 폐합 래치가 해법(명령 이력만 쓰므로 실기 배포 가능).",
            "        상승량 문턱 미달↑ ⇒ (b) 팔이 애초에 들지 않음 — 폐합 래치는 무용.",
        ]
    lines += [
        "-" * 62,
        "판정 지침: force_open 모드에서도 placed≈0이면 ★환경/판정 결함★(정책 탓 아님).",
        "           force_open에서 placed가 높으면 ★정책의 릴리스 미학습★(보상 지형 문제).",
        f"git             : {stats['git']}",
    ]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"[probe] 산출물: {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
