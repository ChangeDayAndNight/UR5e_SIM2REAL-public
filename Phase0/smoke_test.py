"""Phase0 스모크 테스트 — Isaac Sim 헤드리스 부팅 + UR5e 로드 + 60 물리 스텝 검증.

목적: isaac51 환경에서 (1) Kit 부팅, (2) configs/common.yaml 타이밍 계약
      (physics_hz/control_hz → decimation 파생) 검증, (3) 확장 모듈
      ur5e_binpick.tasks.binpick.mdp.ur5e_assets의 UR5E_CFG 스폰,
      (4) 60 물리 스텝 무에러 확인. Franka Phase0 smoke_test 패턴 이식.
실행 (학습 PC, 프로젝트 루트 UR5e_Sim2Real에서):
    python -u Phase0\\smoke_test.py --headless
    (잘못된 python으로 실행해도 자기 재실행 가드가 env python으로 재실행)
산출: 기본 실행 성공 sentinel "UR5E_SMOKE_OK robot=ur5e_cfg joints=8".

참고: 기본 실행은 확장 import·필수 USD·로봇 스폰·관절 계약 중 하나라도 실패하면 즉시
      비정상 종료한다. `--no-robot`은 Kit 자체를 격리 진단할 때만 쓰며 별도
      `ISAAC_BOOT_SMOKE_OK` sentinel을 출력한다.
"""

# --- 자기 재실행 가드: 검증된 env python + 클린 환경 보장 ---
# 기본 python(MS Store)이나 오염된 PYTHONPATH로 실행해도, 검증된 isaac51
# python으로 PYTHONPATH 제거 + user-site 비활성화 + EULA 동의 상태로 자기 자신을
# 재실행한다.
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

import argparse

from isaaclab.app import AppLauncher

# --- 1) 인자 파서 (AppLauncher가 --headless/--device 등 표준 플래그 주입) ---
parser = argparse.ArgumentParser(description="Phase0: UR5e 헤드리스 스모크 테스트")
parser.add_argument(
    "--no-robot",
    action="store_true",
    help="UR5E_CFG 스폰을 건너뛰고 기본 프리미티브만으로 검증 (USD 문제 격리용)",
)
parser.add_argument("--steps", type=int, default=60, help="물리 스텝 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- 2) 시뮬레이터 기동 (이 줄 이전에는 torch/isaaclab/isaacsim import 금지) ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 3) 본 코드 ---
import threading

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from common.config_io import configs_dir, load_yaml
from common.timing import derive_decimation


def _spawn_ur5e():
    """UR5E_CFG를 스폰한다. 기본 smoke에서는 어떤 오류도 폴백하지 않는다.

    ⚠️ isaaclab을 import하는 확장 모듈이므로 반드시 AppLauncher 이후에 import할 것.
    """

    try:
        from isaaclab.assets import Articulation

        from ur5e_binpick.tasks.binpick.mdp.ur5e_assets import UR5E_CFG, UR5E_USD_PATH
    except ImportError as e:
        raise RuntimeError(
            "[smoke] ur5e_binpick 확장 import 실패. bootstrap_training.ps1과 "
            "editable install을 확인하세요."
        ) from e
    if not Path(UR5E_USD_PATH).is_file():
        raise FileNotFoundError(
            f"[smoke] canonical UR5e USD 없음: {UR5E_USD_PATH}. "
            "assets/asset_manifest.json과 Git checkout을 확인하세요."
        )
    try:
        return Articulation(UR5E_CFG.replace(prim_path="/World/UR5e"))
    except Exception as e:
        raise RuntimeError(f"[smoke] UR5E_CFG 스폰 실패: {UR5E_USD_PATH}") from e


def main() -> None:
    # 타이밍 계약 검증 (요소3): 여기서 ValueError가 나면 configs/common.yaml이 잘못된 것.
    common_cfg = load_yaml(configs_dir() / "common.yaml")
    physics_hz = float(common_cfg["control"]["physics_hz"])
    control_hz = float(common_cfg["control"]["control_hz"])
    decimation = derive_decimation(physics_hz, control_hz)
    print(
        f"[smoke] 타이밍 계약 OK: physics={physics_hz}Hz, control={control_hz}Hz, "
        f"decimation={decimation}"
    )

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / physics_hz, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 최소 씬: 바닥 + 돔 라이트
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = None if args_cli.no_robot else _spawn_ur5e()
    robot_mode = "primitive_only" if args_cli.no_robot else "ur5e_cfg"

    sim.reset()
    sim_dt = sim.get_physics_dt()

    num_joints = 0
    if robot is not None:
        from common.robot_spec import UR5E_ARM_JOINT_NAMES
        from ur5e_binpick.tasks.binpick.mdp.ur5e_assets import GRIPPER_JOINT_NAMES

        num_joints = robot.num_joints
        print(f"[smoke] UR5e 로드 완료: joints={num_joints} names={robot.joint_names}")
        expected_names = [*UR5E_ARM_JOINT_NAMES, *GRIPPER_JOINT_NAMES]
        if num_joints != len(expected_names) or list(robot.joint_names) != expected_names:
            raise RuntimeError(
                "[smoke] 로봇 관절 계약 불일치: "
                f"expected={expected_names}, actual={list(robot.joint_names)}"
            )

    n_steps = int(args_cli.steps)
    if n_steps <= 0:
        raise ValueError(f"[smoke] --steps는 양수여야 함: {n_steps}")
    for _ in range(n_steps):
        if robot is not None:
            # 기본 자세 유지 타깃 (중력에 무너지지 않는지 확인)
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.write_data_to_sim()
        sim.step()
        if robot is not None:
            robot.update(sim_dt)

    if robot is not None:
        for state_name in ("joint_pos", "joint_vel"):
            if not torch.isfinite(getattr(robot.data, state_name)).all():
                raise RuntimeError(
                    f"[smoke] ★스텝 후 {state_name}에 NaN/inf — 물리 발산★"
                )

    sentinel = "ISAAC_BOOT_SMOKE_OK" if args_cli.no_robot else "UR5E_SMOKE_OK"
    print(
        f"{sentinel} robot={robot_mode} joints={num_joints} steps={n_steps} "
        f"decimation={decimation} cuda={torch.cuda.is_available()} device={sim.device}",
        flush=True,
    )


main()

# Windows teardown 우회: simulation_app.close()가 이 PC에서 멈출 수 있음 —
# 데몬 스레드로 닫고 타임아웃 후 강제 종료 (Franka Gotcha 2).
closer = threading.Thread(target=simulation_app.close, daemon=True)
closer.start()
closer.join(timeout=30)
os._exit(0)
