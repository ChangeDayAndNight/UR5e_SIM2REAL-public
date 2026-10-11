#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — 공식 IsaacLab rsl_rl 스크립트에 이식.
#   ① ISAACLAB_ENV_PY→활성 conda→홈 conda 자동탐색으로 isaac51 python 자기 재실행
#      (아무 python으로 실행해도 동작. 하드코딩 절대경로 없음.)
#   ② repo 루트를 sys.path에 추가 → 확장(ur5e_binpick)이 top-level `common`을 import
#      (배포 번들 §8-2 무변경 — common은 repo 루트에 그대로).
#   ③ stdout UTF-8 / EULA / user-site 정리. h5py 선로드는 하지 않는다(F54).
#   원본(공식 train.py/play.py) 본문은 이 블록 '아래'에 그대로 이어진다.
# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)  # 확장이 repo 루트의 `common`을 import (§확장)

from scripts.training_env import (
    configure_utf8_stdio,
    preload_native_training_modules,
    reexec_in_training_python,
)

reexec_in_training_python(__file__)
configure_utf8_stdio()

from common.runtime_authority import (
    activate_runtime_authority_child,
    emit_runtime_authority_sentinel,
)

# Low-level training is authority-bearing and may not be invoked directly.
# Activation precedes AppLauncher, task registration, and Hydra config loading.
activate_runtime_authority_child(
    _REPO_ROOT,
    allowed_purposes={"teacher-training", "vision-training"},
)

# This low-level entrypoint can be invoked outside the public PowerShell wrapper,
# so it must enforce the full preflight itself.  An environment flag or mutable
# receipt would only discourage accidental bypass and is not an authorization.
import subprocess as _preflight_subprocess

_preflight = _preflight_subprocess.run(
    [sys.executable, str(Path(_REPO_ROOT) / "scripts" / "preflight_training.py")],
    cwd=_REPO_ROOT,
    env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONNOUSERSITE": "1"},
    check=False,
)
if _preflight.returncode != 0:
    raise SystemExit(f"mandatory low-level training preflight failed: exit={_preflight.returncode}")

# Load Torch/TensorDict's native DLL chain before Kit.  On Windows, importing
# it for the first time after Isaac Sim 5.1 starts can terminate with 0xC0000005.
preload_native_training_modules()

# ═══════════════════ 이하 공식 IsaacLab train.py / play.py 원본 ═══════════════════
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--expected_checkpoint_sha256",
    type=str,
    default=None,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _normalize_expected_checkpoint_sha256(value: str | None) -> str | None:
    """Accept only the exact digest syntax used by the parent wrapper."""
    if value is None:
        return None
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("expected checkpoint SHA-256 must be 64 lowercase hex characters")
    return value


def _validate_checkpoint_load_contract(
    *,
    is_resume: bool,
    is_distillation: bool,
    expected_sha256: str | None,
    load_run: str | None,
    checkpoint_name: str | None,
) -> bool:
    """Fail closed for scratch, canonical resume, and wrapper warm-start modes."""
    load_requested = bool(is_resume or is_distillation)
    if expected_sha256 is not None and not load_requested:
        raise ValueError("checkpoint SHA-256 authorization is invalid for scratch training")
    if is_distillation and not is_resume:
        if expected_sha256 is None:
            raise ValueError(
                "Distillation/NuRec warm-start requires a parent-verified checkpoint SHA-256"
            )
        if (
            not isinstance(load_run, str)
            or not load_run.startswith(("_distill-warmstart-", "_nurec-warmstart-"))
            or Path(load_run).name != load_run
            or checkpoint_name != "model_0.pt"
        ):
            raise ValueError(
                "Distillation/NuRec warm-start requires an exclusive wrapper-owned seed"
            )
    return load_requested


try:
    args_cli.expected_checkpoint_sha256 = _normalize_expected_checkpoint_sha256(
        args_cli.expected_checkpoint_sha256
    )
except ValueError as exc:
    parser.error(str(exc))

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check the project-pinned RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

# 커스텀 Vision/Distillation 클래스는 3.1.2 constructor/runner 계약에 고정되어 있다.
RSL_RL_VERSION = "3.1.2"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) != version.parse(RSL_RL_VERSION):
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nRun:"
        "\n\n\tpowershell -ExecutionPolicy Bypass -File scripts/bootstrap_training.ps1\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import ur5e_binpick.tasks  # noqa: F401  (import 시 gym.register — UR5e binpick 태스크 등록)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from scripts.checkpoint_io import load_anchored_runner_checkpoint
from scripts.training_provenance import verify_training_checkpoint_receipt

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    emit_runtime_authority_sentinel(
        "training",
        required_configs={"configs/common.yaml", "configs/dr.yaml", "configs/sim.yaml"},
    )
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    is_distillation = agent_cfg.algorithm.class_name.startswith("Distillation")
    expected_checkpoint_sha256 = args_cli.expected_checkpoint_sha256
    load_requested = _validate_checkpoint_load_contract(
        is_resume=bool(agent_cfg.resume),
        is_distillation=is_distillation,
        expected_sha256=expected_checkpoint_sha256,
        load_run=args_cli.load_run,
        checkpoint_name=args_cli.checkpoint,
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    # Wrapper-owned, unambiguous receipt.  The parent parses its dedicated
    # subprocess stream instead of guessing the newest directory (race-prone).
    print(f"RSL_RL_RUN_DIR={log_dir}", flush=True)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if load_requested:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        if agent_cfg.resume:
            resume_binding = verify_training_checkpoint_receipt(
                resume_path,
                project_root=_REPO_ROOT,
                expected_experiment=agent_cfg.experiment_name,
            )
            receipt_checkpoint_sha256 = resume_binding["checkpoint_sha256"]
            if (
                expected_checkpoint_sha256 is not None
                and expected_checkpoint_sha256 != receipt_checkpoint_sha256
            ):
                raise ValueError(
                    "resume checkpoint SHA-256 differs from its trusted training receipt"
                )
            expected_checkpoint_sha256 = receipt_checkpoint_sha256
        else:
            resolved_resume = Path(resume_path).absolute().resolve()
            if (
                resolved_resume.parent.name != args_cli.load_run
                or resolved_resume.name != args_cli.checkpoint
            ):
                raise ValueError("warm-start selector resolved outside its exclusive seed")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ★F94(2026-09-22) 학습 경로 렌더 가드★ — Vision/Distill 태스크는 첫 관측의 depth·RGB 블록이
    #   공간 구조를 갖는지 확인하고 아니면 ★기동 전에★ 죽는다. 새 PC에서 타일 ≥128 + DLSS면 RGB만
    #   죽는 결함이 있고(depth 정상), 이 가드가 없던 증류 1회(20k iter)가 깨진 RGB로 조용히 학습됐다.
    if "Vision" in args_cli.task:
        from common.config_io import build_preprocess_cfg, configs_dir, load_yaml
        from common.render_guard import check_vision_render_health
        _first = env.get_observations()
        _flat = _first["policy"] if hasattr(_first, "keys") and "policy" in _first.keys() else _first
        check_vision_render_health(_flat, build_preprocess_cfg(load_yaml(configs_dir() / "common.yaml")), where="train")

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if load_requested:
        if expected_checkpoint_sha256 is None:
            raise RuntimeError("checkpoint load reached runner without a trusted SHA-256")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # RSL-RL 3.1.2 performs the actual policy/optimizer/iteration/RND state
        # application.  It receives an already-open, hash-anchored stream so
        # the mutable resume pathname is never reopened after authorization.
        load_anchored_runner_checkpoint(
            resume_path,
            expected_sha256=expected_checkpoint_sha256,
            allowed_root=log_root_path,
            runner=runner,
        )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
