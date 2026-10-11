"""Phase0 씬 체크 — ur5e_binpick.tasks.binpick.mdp.scene의 InteractiveScene 전체 스폰 + 스텝 검증.

목적: bin-picking 기반 씬(테이블/bin/UR5e/객체/카메라)이 GUI·headless 양쪽에서
      에러 없이 스폰·스텝되는지 확인하고, 엔티티별 prim 유효성을 리포트한다.
실행 (학습 PC, 프로젝트 루트 UR5e_Sim2Real에서):
    # GUI — [사람] 눈으로 배치 확인
    python Phase0\\scene_bin_ur5e.py --num_envs 2
    # headless — [Claude] 무에러 스텝 검증
    python -u Phase0\\scene_bin_ur5e.py --headless --num_envs 4 --steps 200
    # ⚠️ 씬에 TiledCamera 센서가 포함돼 있으면 --enable_cameras를 추가할 것
산출: stdout 엔티티별 [scene] 리포트 + 성공 sentinel "BINPICK_SCENE_OK".

의존: 확장 ur5e_binpick.tasks.binpick.mdp.scene (pip install -e source/ur5e_binpick)
      — import 실패 시 명확한 에러로 종료(exit 2).
"""

# --- 자기 재실행 가드: 검증된 env python + 클린 환경 보장 ---
# 기본 python(MS Store)이나 오염된 PYTHONPATH로 실행해도, 검증된 isaac51
# python으로 PYTHONPATH 제거 + user-site 비활성화 + EULA 동의 상태로 자기 자신을
# 재실행한다.
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()




os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# GUI/카메라 모드에서 omni.sensors의 hdf5 DLL 충돌 회피 — Kit 기동 전 h5py 선로드
# (Franka setup 메모 Gotcha 4).
try:
    # ★2026-08-10(F54) h5py 선로드 제거★ — 실측상 ★이 import가 Isaac 센서 DLL을 깨뜨린다★.
    #   h5py 3.16.0(HDF5 2.0.0)이 Isaac 번들 hdf5.dll(1.x)보다 먼저 로드되면
    #   generic_mo_io.dll이 WinError 127로 실패하고 GUI가 access violation으로 죽는다.
    #   (구 h5py에서는 반대로 작동해 워크어라운드였다 — 근거·측정 = Failure_Check F54)
    pass
except Exception as _e:
    print("[scene] h5py preimport skipped:", _e)

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# --- 1) 인자 파서 ---
parser = argparse.ArgumentParser(description="Phase0: bin-picking 씬 스폰·스텝 체크")
parser.add_argument("--num_envs", type=int, default=2, help="복제할 환경 수")
parser.add_argument(
    "--steps",
    type=int,
    default=200,
    help="0이면 창 닫을 때까지(GUI 확인용), >0이면 그 스텝 뒤 종료(headless 검증용)",
)
parser.add_argument("--env_spacing", type=float, default=3.0, help="env 간 간격 [m]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- 2) 시뮬레이터 기동 (이 줄 이전에는 torch/isaaclab/isaacsim import 금지) ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 3) 본 코드 ---
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext

from common.config_io import configs_dir, load_yaml


def _load_scene_cfg_class() -> type[InteractiveSceneCfg]:
    """ur5e_binpick.tasks.binpick.mdp.scene에서 씬 cfg 클래스를 찾는다.

    ⚠️ isaaclab을 import하는 모듈이므로 AppLauncher 이후에만 호출할 것.
    클래스명은 BinPickSceneCfg로 확정됨(README.md §9 — mdp/scene.py 실재). 폴백 탐색
    (모듈 안의 유일한 InteractiveSceneCfg 서브클래스)은 방어용으로만 유지.
    """
    import importlib

    mod = importlib.import_module("ur5e_binpick.tasks.binpick.mdp.scene")
    if hasattr(mod, "BinPickSceneCfg"):
        return mod.BinPickSceneCfg
    candidates = [
        v
        for v in vars(mod).values()
        if isinstance(v, type)
        and issubclass(v, InteractiveSceneCfg)
        and v is not InteractiveSceneCfg
    ]
    if len(candidates) == 1:
        print(f"[scene] 씬 cfg 클래스 자동 탐색: {candidates[0].__name__}")
        return candidates[0]
    raise ImportError(
        f"ur5e_binpick.tasks.binpick.mdp.scene에서 씬 cfg 클래스를 특정할 수 없음 (후보 {len(candidates)}개) "
        "— BinPickSceneCfg로 이름을 맞추거나 이 스크립트를 갱신할 것"
    )


def _report_entities(scene: InteractiveScene) -> int:
    """엔티티별 prim 유효성/초기화 상태 리포트 — invalid 개수 반환."""
    invalid = 0

    # (a) cfg에 선언된 asset들의 env_0 prim 경로 유효성
    #     (InteractiveScene 생성 시 cfg.prim_path의 {ENV_REGEX_NS}는
    #      "/World/envs/env_.*"로 치환돼 있음 → env_0으로 구체화해 검사)
    for asset_name, asset_cfg in scene.cfg.__dict__.items():
        if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
            continue
        prim_path = getattr(asset_cfg, "prim_path", None)
        if prim_path is None:
            continue
        path_env0 = prim_path.replace("env_.*", "env_0")
        ok = prim_utils.is_prim_path_valid(path_env0)
        mark = "OK " if ok else "★INVALID★"
        print(f"[scene]   prim {mark} {asset_name:<16} → {path_env0}")
        if not ok:
            invalid += 1

    # (b) 런타임 엔티티 초기화 상태 (Articulation/RigidObject/Sensor: is_initialized)
    for key in scene.keys():
        if key == "terrain":
            continue  # terrain 미사용 (keys()가 항상 포함하므로 스킵)
        entity = scene[key]
        is_init = getattr(entity, "is_initialized", None)
        n_inst = getattr(entity, "num_instances", "-")
        if is_init is False:
            invalid += 1
        print(
            f"[scene]   entity {key:<16} type={type(entity).__name__:<20} "
            f"initialized={is_init} instances={n_inst}"
        )
    return invalid


def main() -> None:
    common_cfg = load_yaml(configs_dir() / "common.yaml")
    physics_hz = float(common_cfg["control"]["physics_hz"])

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / physics_hz, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, 1.8, 1.5], target=[0.45, 0.0, 0.2])

    try:
        scene_cls = _load_scene_cfg_class()
    except ImportError as e:
        print(f"⚠️ [scene] 씬 cfg 로드 실패: {e}")
        print("⚠️ [scene] ur5e_binpick 확장 설치(pip install -e source/ur5e_binpick) 확인 후 다시 실행할 것 (exit 2)")
        raise SystemExit(2)

    scene_cfg = scene_cls(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    sim_dt = sim.get_physics_dt()
    print(f"[scene] 스폰 완료: num_envs={scene.num_envs}, entities={scene.keys()}")

    invalid = _report_entities(scene)
    if invalid > 0:
        print(f"⚠️ [scene] ★invalid 엔티티 {invalid}개 — 씬 구성 확인 필요 (exit 1)★")
        raise SystemExit(1)

    # 스텝 루프: 모든 articulation을 기본 자세 타깃으로 유지하며 물리 스텝
    count = 0
    while simulation_app.is_running():
        for art in scene.articulations.values():
            art.set_joint_position_target(art.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1
        if count % 100 == 0:
            print(f"[scene] step {count} ...")
        if args_cli.steps and count >= args_cli.steps:
            break

    # 스텝 후 발산 검사 (NaN/inf = 물리·액추에이터 cfg 문제)
    for name, art in scene.articulations.items():
        if not torch.isfinite(art.data.joint_pos).all():
            raise RuntimeError(f"[scene] ★{name} joint_pos NaN/inf — 물리 발산★")

    n_entities = len(scene.keys()) - 1  # "terrain" 제외
    print(
        f"BINPICK_SCENE_OK steps={count} num_envs={scene.num_envs} "
        f"entities={n_entities} invalid=0 device={sim.device}",
        flush=True,
    )


main()

# Windows teardown 우회: close()가 멈출 수 있어 데몬 스레드로 닫고 강제 종료.
closer = threading.Thread(target=simulation_app.close, daemon=True)
closer.start()
closer.join(timeout=30)
os._exit(0)
