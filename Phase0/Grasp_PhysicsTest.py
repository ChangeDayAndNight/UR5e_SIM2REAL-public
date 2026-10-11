#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""OnRobot 2FG7 ★집기 물리 achievability 검증★ (isaac51, [사람] 또는 Claude 직접).

목적: RL이 집기를 못 배우는 것이 (a)물리적으로 파지 불가라서인지 (b)탐색이 어려워
      서인지 가른다. RL 없이 결정론적 스크립트로 그리퍼를 큐브 위에 두고 하강→파지
      →들기 하여, ★큐브가 그리퍼와 함께 들리는가★를 측정한다.
      성공 → 파지는 물리적으로 가능(문제는 RL 탐색 → approach-shaping/커리큘럼/장기학습).
      실패 → 파지 물리 자체가 안 됨(핑거 마찰/그립력/큐브 크기·마찰 조정 필요).
실행 (학습 PC, isaac51):
  python Phase0\\Grasp_PhysicsTest.py                       # 헤드리스 판정(빠름)
  python Phase0\\Grasp_PhysicsTest.py --video               # 거동 mp4 녹화(Phase0/logs/videos/)
  python Phase0\\Grasp_PhysicsTest.py --grasp-z 0.16 --cube-friction 1.2   # 튜닝
산출: Phase0/logs/grasp_physics_<시각>.txt (판정) + (--video 시) mp4.
전제: assets/onrobot_2fg7/onrobot_2fg7.usd (convert_2fg7_usd.py로 생성).

⚠️ isaaclab은 AppLauncher 이후에만 import.
"""
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()



_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
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

# ═══════════════════ AppLauncher ═══════════════════
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OnRobot 2FG7 집기 물리 검증")
parser.add_argument("--video", action="store_true", help="거동 mp4 녹화")
parser.add_argument("--grasp-z", type=float, default=0.16, help="파지 시 루트 높이[m]. STL: 핑거 grip면 base z 0.117~0.162 → 0.16이면 grip 중심이 큐브(0.02)")
parser.add_argument("--cube-size", type=float, default=0.04, help="큐브 한 변[m]")
parser.add_argument("--cube-friction", type=float, default=1.5, help="큐브 정/동 마찰")
parser.add_argument("--cube-mass", type=float, default=0.05, help="큐브 질량[kg]")
# ★2026-07-31 릴리스 계측(F37 진단)★ — teacher v11이 큐브를 벨트에 내려놓고도 ★손을 안 놓는다★
#   (hold_frac 0.891 / placed 6.8%). 보상 산술은 놓기가 1.9배 우월하므로 원인은 물리/판정 조건에 있다.
#   가설 3종: ①접촉력 비대칭(실측 L21.6N vs R6.0N) ②settled 임계 0.10m/s ③침투 8mm 탄성 방출.
#   이 모드는 '들어올린 뒤 ★벨트에 내려놓고 연다★'를 스크립트로 재현해 큐브 변위·속도를 잰다.
parser.add_argument("--release-test", action="store_true",
                    help="lift 후 벨트로 내려놓고 그리퍼를 연다. 큐브 변위/속도를 placed 임계와 대조.")
parser.add_argument("--place-tol", type=float, default=0.03, help="placed 위치 허용[m] (대조용)")
parser.add_argument("--settle-speed", type=float, default=0.10, help="placed 정지 임계[m/s] (대조용)")
parser.add_argument("--close-speed", type=float, default=0.05,
                    help="핑거 폐합속도 상한 [m/s] (PhysX maxJointVelocity). 학습 cfg = 0.05.")
parser.add_argument("--stiffness", type=float, default=5000.0,
                    help="핑거 위치 강성 [N/m]. 접촉력 ≈ stiffness x 편측침투. 학습 cfg는 2000.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import datetime as _dt

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRIPPER_USD = os.path.join(ROOT, "assets", "onrobot_2fg7", "onrobot_2fg7.usd")
LOG_DIR = os.path.join(HERE, "logs")

# ★현행 CAD/URDF 계약★: q=0.0 → inner 간격 74mm(열림), q=0.020 → 34mm(닫힘).
#   50mm 큐브는 닫힘(q↑) 도중 핑거가 닿아 그립.
OPEN, CLOSE = 0.0, 0.020
START_Z, LIFT_Z = 0.40, 0.55  # 루트 시작/들기 높이
# 180° about x → 그리퍼 +z(핑거)가 월드 -z(아래)를 향함
QUAT_DOWN = (0.0, 1.0, 0.0, 0.0)  # wxyz


def _gripper_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/Gripper",
        spawn=sim_utils.UsdFileCfg(
            usd_path=GRIPPER_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),  # 루트 직접 제어
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, START_Z), rot=QUAT_DOWN),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*finger.*"],
                # ★2026-07-29 학습 cfg와 정합★ — 구버전은 velocity_limit=0.45만 줬는데
                #   ImplicitActuator는 이 값을 ★무시★하고 velocity_limit_sim만 PhysX
                #   maxJointVelocity로 적용한다(actuator_pd.py L79-100). 즉 폐합속도 무제한 →
                #   강성을 올릴수록 한 substep에 큐브를 관통해 오히려 파지가 나빠졌다
                #   (스윕이 비단조로 나온 원인). 학습과 같은 0.05로 캡해야 진단이 유효하다.
                velocity_limit_sim=args_cli.close_speed, effort_limit_sim=140.0,  # 2FG7 파지력
                # ★2026-07-29 --stiffness 인자화★ — 큐브 5cm 전환 시 파지력이 병목인지
                #   가르기 위해 스윕이 필요했다. 접촉력 ≈ stiffness × 편측침투이므로
                #   (5cm 큐브 = 침투 4mm) 5000이면 20N뿐 — 실물 2FG7 상한 140N에 크게 못 미친다.
                stiffness=args_cli.stiffness, damping=100.0,
            ),
        },
    )


def _cube_cfg() -> RigidObjectCfg:
    s = args_cli.cube_size
    return RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(s, s, s),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.cube_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args_cli.cube_friction,
                dynamic_friction=args_cli.cube_friction,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, s / 2.0)),
    )


def _root_pose(z: float, device) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, z, *QUAT_DOWN]], device=device)


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    sim.set_camera_view([0.55, 0.45, 0.4], [0.0, 0.0, 0.18])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    gripper = Articulation(_gripper_cfg())
    cube = RigidObject(_cube_cfg())

    camera = None
    frames = []
    if args_cli.video:
        from isaaclab.sensors import Camera, CameraCfg

        camera = Camera(CameraCfg(
            prim_path="/World/TestCam", height=480, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
            offset=CameraCfg.OffsetCfg(pos=(0.55, 0.45, 0.4), rot=(0.0, 0.0, 0.0, 1.0)),
        ))

    sim.reset()
    dev = sim.device
    fids, fnames = gripper.find_joints([".*finger.*"], preserve_order=True)
    print(f"[grasp] 그리퍼 finger 관절: {fnames}")
    # ★지오메트리 진단★: root=START_Z일 때 각 바디 world z → 핑거끝 오프셋 파악(grasp_z 튜닝).
    #   핑거끝 world z = START_Z − 오프셋. 큐브 중심(0.02)에 맞추려면 grasp_z = 0.02 + 오프셋.
    _bz = gripper.data.body_pos_w[0]
    for nm, p in zip(gripper.body_names, _bz):
        print(f"[grasp][geom] {nm:22s} world_z={float(p[2]):.4f}")
    _tip_z = float(_bz[:, 2].min())
    print(f"[grasp][geom] 최저 바디 z={_tip_z:.4f} (root={START_Z}) → 오프셋 {START_Z - _tip_z:.4f}m; "
          f"권장 grasp_z ≈ {0.02 + (START_Z - _tip_z):.3f}")
    # ★큐브를 실제 grasp 중심(두 finger 바디 xy 중점)으로 이동★ — URDF finger origin이
    #   비대칭(+0.032/−0.054)이라 x=0이 아니다. cube를 그 xy로 옮겨 위치 어긋남을 제거.
    # ★STL 검증 grip 중심 = (0,0)★: 핑거 inner face x=±0.022 → 중심 x=0; y 중심 0.
    #   (link 프레임 midpoint −0.011이 아님 — CAD 메시가 base 프레임 기준.)
    _cx, _cy = 0.0, 0.0
    cube.write_root_pose_to_sim(
        torch.tensor([[_cx, _cy, args_cli.cube_size / 2, 1.0, 0.0, 0.0, 0.0]], device=dev)
    )
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
    print(f"[grasp] 큐브를 grip 중심(STL)으로: x={_cx:.4f} y={_cy:.4f}")
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([[0.55, 0.45, 0.4]], device=dev), torch.tensor([[0.0, 0.0, 0.18]], device=dev)
        )

    # 단계별 스텝 수 (dt=1/120)
    PH = {"settle": 40, "approach": 90, "close": 60, "lift": 110, "hold": 40}
    if args_cli.release_test:
        # 들어올린 뒤 ★다시 내려놓고 → 열고 → 관찰★ (안착 시나리오 재현)
        PH = {"settle": 40, "approach": 90, "close": 60, "lift": 80,
              "lower": 90, "release": 60, "observe": 120}
    total = sum(PH.values())
    gz = args_cli.grasp_z
    cube_z_settled = None

    def _finger(target):
        gripper.set_joint_position_target(torch.full((1, len(fids)), target, device=dev), joint_ids=fids)

    _rel_ref = [None]      # 놓기 직전 큐브 위치(기준점)
    _rel_log: list = []     # (step, phase, 변위[m], 속도[m/s], finger)
    _finger(OPEN)
    step = 0
    for phase, n in PH.items():
        for k in range(n):
            # 루트 높이 스케줄
            if phase == "approach":
                z = START_Z + (gz - START_Z) * (k + 1) / n
            elif phase == "lift":
                z = gz + (LIFT_Z - gz) * (k + 1) / n
            elif phase == "lower":
                # ★안착 높이까지 하강★ — 큐브 바닥이 지면에 닿는 루트 높이(=grasp_z)로 복귀
                z = LIFT_Z + (gz - LIFT_Z) * (k + 1) / n
            else:
                z = {"settle": START_Z, "close": gz, "hold": LIFT_Z,
                     "release": gz, "observe": gz}[phase]
            gripper.write_root_pose_to_sim(_root_pose(z, dev))
            if phase in ("close", "lift", "hold", "lower"):
                _finger(CLOSE)
            elif phase in ("release", "observe"):
                _finger(OPEN)          # ★그리퍼 개방 — 여기서부터가 계측 구간★
            # ★close까지 큐브를 grasp 중심에 핀 고정★ → 핑거가 확실히 그립을 형성.
            #   lift/hold에서 release → 마찰만으로 버티는지 = 진짜 파지 achievability 판정.
            if phase in ("settle", "approach", "close"):
                cube.write_root_pose_to_sim(
                    torch.tensor([[_cx, _cy, args_cli.cube_size / 2, 1.0, 0.0, 0.0, 0.0]], device=dev)
                )
                cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
            gripper.write_data_to_sim()
            cube.write_data_to_sim()
            sim.step()
            gripper.update(1 / 120)
            cube.update(1 / 120)
            if camera is not None:
                camera.update(1 / 120)
                rgb = camera.data.output["rgb"]
                if rgb is not None:
                    frames.append(rgb[0, ..., :3].cpu().numpy().copy())
            if phase == "settle" and k == n - 1:
                cube_z_settled = float(cube.data.root_pos_w[0, 2].item())
            if args_cli.release_test:
                if phase == "lower" and k == n - 1:      # 놓기 직전 기준점
                    _rel_ref[0] = cube.data.root_pos_w[0].clone()
                if phase in ("release", "observe"):
                    _spd = float(cube.data.root_lin_vel_w[0].norm().item())
                    _drift = float((cube.data.root_pos_w[0] - _rel_ref[0]).norm().item())                         if _rel_ref[0] is not None else float("nan")
                    _rel_log.append((step, phase, _drift, _spd,
                                     float(gripper.data.joint_pos[0, fids].mean().item())))
            if step % 40 == 0:
                cz = float(cube.data.root_pos_w[0, 2].item())
                fp = float(gripper.data.joint_pos[0, fids].mean().item())
                print(f"[grasp] step {step:3d} [{phase:8s}] root_z={z:.3f} finger={fp:.4f} cube_z={cz:.4f}")
            step += 1

    # ── 릴리스 계측 요약 ──
    if args_cli.release_test and _rel_log:
        print("[release] ★릴리스 계측★ (기준: placed tol "
              f"{args_cli.place_tol*100:.0f}cm / settled {args_cli.settle_speed:.2f} m/s)")
        print(f"[release] {'step':>5} {'phase':>8} {'변위cm':>8} {'속도m/s':>9} {'finger':>8}")
        for i, (st_, ph, dr, sp, fg) in enumerate(_rel_log):
            if i % 10 == 0 or sp > args_cli.settle_speed:
                flag = ""
                if sp > args_cli.settle_speed:
                    flag += "  ★속도초과★"
                if dr == dr and dr > args_cli.place_tol:
                    flag += "  ★위치이탈★"
                print(f"[release] {st_:5d} {ph:>8} {dr*100:8.2f} {sp:9.3f} {fg:8.4f}{flag}")
        drifts = [d for _, _, d, _, _ in _rel_log if d == d]
        speeds = [v for _, _, _, v, _ in _rel_log]
        tail = _rel_log[-40:]                       # 마지막 40스텝 = 정착 후 상태
        d_end = tail[-1][2] if tail else float("nan")
        v_max = max(speeds) if speeds else float("nan")
        v_end = max(v for _, _, _, v, _ in tail) if tail else float("nan")
        print(f"[release] ★최대 변위 {max(drifts)*100:.2f}cm / 최종 변위 {d_end*100:.2f}cm"
              f" (허용 {args_cli.place_tol*100:.0f}cm)")
        print(f"[release] ★최대 속도 {v_max:.3f} / 정착 후 최대 {v_end:.3f} m/s"
              f" (허용 {args_cli.settle_speed:.2f})")
        ok_pos = d_end == d_end and d_end <= args_cli.place_tol
        ok_spd = v_end <= args_cli.settle_speed
        print(f"[release] 판정: 위치 {'OK' if ok_pos else '★실패★'} / "
              f"정지 {'OK' if ok_spd else '★실패★'}  -> placed "
              f"{'성립' if (ok_pos and ok_spd) else '★불성립★'}")

    cube_z_final = float(cube.data.root_pos_w[0, 2].item())
    rise = cube_z_final - (cube_z_settled or args_cli.cube_size / 2)
    success = rise > 0.08  # 8cm 이상 함께 들림 → 파지 성공
    verdict = "★파지 성공(물리적으로 가능)★" if success else "파지 실패(큐브 미상승)"

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        "OnRobot 2FG7 집기 물리 검증",
        f"시각: {stamp}",
        f"큐브: {args_cli.cube_size*100:.0f}cm, 마찰 {args_cli.cube_friction}, 질량 {args_cli.cube_mass}kg",
        f"grasp_z: {gz}",
        f"큐브 z: 안착 {cube_z_settled:.4f} → 최종 {cube_z_final:.4f}  (상승 {rise*100:.1f}cm)",
        f"판정: {verdict}",
    ]
    report = "\n".join(lines)
    print("\n" + "=" * 60 + "\n" + report + "\n" + "=" * 60)
    with open(os.path.join(LOG_DIR, f"grasp_physics_{stamp}.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    if frames:
        try:
            import imageio.v3 as iio

            vdir = os.path.join(LOG_DIR, "videos")
            os.makedirs(vdir, exist_ok=True)
            out = os.path.join(vdir, f"grasp_physics_{stamp}.mp4")
            iio.imwrite(out, frames, fps=30)
            print(f"[grasp] 영상 저장({len(frames)}f): {out}")
        except Exception as e:
            print(f"[grasp][warn] 영상 저장 실패: {e}")

    return 0 if success else 2


if __name__ == "__main__":
    rc = main()
    # ★close() 행 방지(Windows Isaac 알려진 이슈)★: close가 멈춰도 20s 후 강제 종료 →
    #   프로세스가 반드시 끝나 완료 신호(종료 코드)가 온다. (무한 대기 방지)
    import os as _os
    import threading as _th

    _th.Timer(20.0, lambda: _os._exit(rc)).start()
    try:
        simulation_app.close()
    except Exception:
        pass
    _os._exit(rc)
