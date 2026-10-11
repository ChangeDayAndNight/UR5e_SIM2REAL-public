#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — isaac51 python 자기 재실행 + 경로.
#   ① ISAACLAB_ENV_PY→활성 conda→홈 conda 자동탐색으로 isaac51 python 재실행
#   ② repo 루트를 sys.path에 추가(확장/common import 정합 — 이 스크립트는 미사용이나 규약 유지)
#   ③ stdout UTF-8 / EULA / user-site 정리
# ─────────────────────────────────────────────────────────────────────────────
"""OnRobot 2FG7 URDF → USD 변환 (isaac51, [사람] 실행).

목적: assets/onrobot_2fg7/onrobot_2fg7.urdf 를 Isaac Lab UrdfConverter로 USD로 변환.
      업스트림 convert_urdf.py CLI가 노출하지 않는 2FG7 전용 설정을 코드로 고정:
        - convert_mimic_joints_to_normal_joints=True
            → mimic(right=left) 을 2개의 독립 관절로 변환하고 대칭 명령으로 구동한다
              (Franka 2핑거 방식 — Isaac의 mimic 커플링보다 견고. BinaryJointPositionAction이
               두 관절을 함께 지정). --keep-mimic 로 원본 1-DOF(mimic 유지) 선택 가능.
        - collider_type="convex_decomposition"
            → 손가락 파지면이 오목하므로 convex_hull(기본)이면 파지가 부정확 → 분해 콜라이더.
        - joint_drive: position 제어(스티프니스/댐핑 기본값은 UR5E 결합 후 actuator cfg가 최종 결정).
실행 (학습 PC, isaac51 — 아무 python으로 실행해도 프리앰블이 env로 재실행):
  python scripts\\tools\\convert_2fg7_usd.py --headless
  python scripts\\tools\\convert_2fg7_usd.py --headless --keep-mimic     # 원본 1-DOF(mimic) 유지
산출: assets/onrobot_2fg7/onrobot_2fg7.usd

⚠️ 검증: 변환 후 GUI로 열어 두 핑거가 대칭 개폐되는지, open/close 방향을 확인할 것
   (README.md 주의: left_finger_joint 원점이 닫힘/열림 중 무엇인지 자명하지 않음).
"""
import os
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

# ═══════════════════ AppLauncher (isaaclab import은 이 아래에서만) ═══════════════════
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OnRobot 2FG7 URDF→USD 변환.")
parser.add_argument(
    "--input",
    default=os.path.join(_REPO_ROOT, "assets", "onrobot_2fg7", "onrobot_2fg7.urdf"),
    help="입력 URDF 경로 (기본: assets/onrobot_2fg7/onrobot_2fg7.urdf).",
)
parser.add_argument(
    "--output",
    default=os.path.join(_REPO_ROOT, "assets", "onrobot_2fg7", "onrobot_2fg7.usd"),
    help="출력 USD 경로.",
)
parser.add_argument(
    "--keep-mimic",
    action="store_true",
    help="mimic 관절을 유지(1-DOF). 미지정 시 독립 2관절로 변환(권장, 대칭 구동).",
)
parser.add_argument(
    "--fix-base",
    action="store_true",
    help="루트 링크를 월드에 고정. ★UR5e 결합(ur5e_2fg7) 변환 시 지정★ "
    "(로봇은 테이블 장착). 단독 그리퍼 변환 시에는 미지정(팔에 붙을 free root).",
)
parser.add_argument("--joint-stiffness", type=float, default=2000.0, help="핑거 드라이브 스티프니스[N/m].")
parser.add_argument("--joint-damping", type=float, default=100.0, help="핑거 드라이브 댐핑[N/(m/s)].")
parser.add_argument(
    "--collider-type",
    default="convex_hull",
    choices=["convex_hull", "convex_decomposition"],
    help="손가락 콜라이더. ★2026-07-08 기본 convex_hull★: decomposition은 좁고 불안정한 접촉"
    "(grasp_sweep ±0.5cm 창·팔 움직임 중 큐브 미끄러짐)을 만든 것으로 진단. hull은 평면 파지면을"
    "보존해 넓고 안정된 접촉을 기대(실 2FG7 평면 pad 정합). GUI로 개폐·파지 확인할 것.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def main() -> None:
    urdf_path = os.path.abspath(args_cli.input)
    if not check_file_path(urdf_path):
        raise ValueError(f"[convert] 입력 URDF를 찾을 수 없습니다: {urdf_path}")
    dest_path = os.path.abspath(args_cli.output)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        # 단독 그리퍼: base 미고정(팔 flange에 붙을 free root). UR5e 결합 변환은
        # --fix-base 지정(로봇은 테이블 장착 = 루트 고정).
        fix_base=args_cli.fix_base,
        merge_fixed_joints=True,
        # ★2FG7 mimic 처리★: 기본은 독립 2관절 변환(대칭 구동, 견고). --keep-mimic 시 유지.
        convert_mimic_joints_to_normal_joints=not args_cli.keep_mimic,
        # ★손가락 콜라이더 (2026-07-08: convex_hull 기본)★ — decomposition의 좁고 불안정한
        #   접촉이 파지 미끄러짐·좁은 그립창(±0.5cm)의 원인으로 진단. hull은 평면 파지면 보존.
        collider_type=args_cli.collider_type,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            ),
        ),
    )

    print("-" * 80)
    print(f"[convert] 입력 URDF : {urdf_path}")
    print(f"[convert] 출력 USD  : {dest_path}")
    print(f"[convert] mimic     : {'유지(1-DOF)' if args_cli.keep_mimic else '독립 2관절 변환'}")
    print("[convert] UrdfConverterCfg:")
    print_dict(cfg.to_dict(), nesting=0)
    print("-" * 80)

    converter = UrdfConverter(cfg)
    print(f"[convert] ★생성 완료★: {converter.usd_path}")
    print("[convert] 다음: GUI로 열어 대칭 개폐 확인 → UR5e flange 결합 → gripper_action 배선")
    print("-" * 80)

    # GUI 모드면 결과 스테이지를 띄워 육안 확인.
    settings = carb.settings.get_settings()
    if settings.get("/app/window/enabled") or settings.get("/app/livestream/enabled"):
        stage_utils.open_stage(converter.usd_path)
        app = omni.kit.app.get_app_interface()
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
