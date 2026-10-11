"""Bin-Picking 장면(InteractiveSceneCfg) 정의 — UR5e + 정적 컨베이어 벨트 + 산란 큐브 + TiledCamera.

목적: BinPickTeacherEnvCfg/BinPickVisionEnvCfg가 공유하는 장면 구성의 단일 소스.
      ★2026-07-07 실기 셀 반영: bin(4벽) → 정적 컨베이어 벨트(평면)★. 벨트 치수/큐브 크기(★5cm★)/
      객체 수/카메라 해상도·FOV는 configs/sim.yaml, 카메라 포즈는 configs/calib/hand_eye.yaml
      (요소4), TCP 오프셋은 configs/calib/tcp_offset.yaml.
실행: 직접 실행 없음 — binpick_env_cfg.py가 import.
산출: BinPickSceneCfg, make_tiled_camera_cfg(), make_finger_contact_sensor_cfg(),
      add_scatter_objects(), object_names(), make_nurec_background_cfg().

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import할 것.
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from common.calibration import (
    load_hand_eye,
    load_scene_registration,
    load_tcp_offset,
    mat_to_quat,
)
from common.config_io import configs_dir, load_yaml

from .ur5e_assets import UR5E_CFG

# ─────────────────────────────────────────────────────────────────────────────
# yaml 로드 (import 시 1회 — 이후 상수로 고정)
# ─────────────────────────────────────────────────────────────────────────────
_SIM_YAML = load_yaml(configs_dir() / "sim.yaml")
_COMMON_YAML = load_yaml(configs_dir() / "common.yaml")

NUM_OBJECTS: int = int(_SIM_YAML["scene"]["num_objects"])
# ★2026-07-07 실기 셀 반영: bin → 정적 컨베이어 벨트(평면 워크스페이스)★
#   ★물리 벨트 실측 치수★ [x 앞뒤, y 좌우, z 두께]. 벨트 상면 = base z=TABLE_HEIGHT.
#   ⚠️2026-07-28(3차)부터 ★시각·물리 형상 전용★ — 스폰/타겟 범위는 PICK_ZONE을 쓴다.
BELT_SIZE: tuple[float, float, float] = tuple(
    float(v) for v in _SIM_YAML["scene"]["belt_size_m"]
)
# ★픽존(2026-07-28 3차)★ — 로봇이 안전하게 닿는 구역. 큐브 스폰 분포(events_dr)와 타겟
#   샘플링이 이 구역을 쓴다. 물리 벨트(100cm)가 UR5e 도달한계(0.817m)보다 커지면서
#   '영원히 못 집는 큐브'가 생기는 것을 막기 위해 분리했다.
#   미지정이면 벨트 크기로 폴백(구 동작 보존).
PICK_ZONE: tuple[float, float] = tuple(  # type: ignore[assignment]
    float(v) for v in _SIM_YAML["scene"].get("pick_zone_m", BELT_SIZE[:2])
)
if PICK_ZONE[0] > BELT_SIZE[0] + 1e-9 or PICK_ZONE[1] > BELT_SIZE[1] + 1e-9:
    raise RuntimeError(
        f"[cfg] ★픽존이 벨트보다 큼★ pick_zone_m={PICK_ZONE} > belt_size_m={BELT_SIZE[:2]} — "
        "큐브가 벨트 밖에 스폰돼 즉시 굴러떨어진다. sim.yaml scene을 확인할 것."
    )
OBJECT_SIZE: float = float(_SIM_YAML["scene"]["object_size_m"])  # 큐브 한 변 [m] — ★실측 5cm★
#   ⚠️★주석 표류 주의(2026-08-11 정정)★: 이 파일 곳곳에 '3cm'로 적혀 있었으나 실제 값은
#     configs/sim.yaml object_size_m = 0.05다. 주석과 값이 어긋나 F50·F55에서 실제로 사고가 났다
#     (§9-5-4 '업스트림/주석과의 차이는 그 자체가 진단 단서'). 크기는 ★yaml이 단일 소스★다.
TABLE_HEIGHT: float = float(_SIM_YAML["scene"]["table_height_m"])  # 벨트 상면 z (base frame)

# 벨트(워크스페이스) 중심 xy — common.yaml target.sample_range의 중심과 일치시킨 공칭값.
# ★2026-07-27 yaml 승격(사용자 지시: 전 객체 P/R/S를 기록·수정 가능하게)★ — 구 하드코딩
#   (0.45/0.0/0.80)과 값 동일이라 학습 거동 불변. 배치 수정은 configs/sim.yaml scene에서,
#   GUI 실측은 place_workspace_gui.py가 같은 키를 갱신한다.
#   ⚠️ 벨트 중심을 옮기면 common.yaml target.sample_range·safety.yaml workspace도 함께 검토.
BELT_CENTER_X: float = float(_SIM_YAML["scene"].get("belt_center_xy_m", [0.45, 0.0])[0])
BELT_CENTER_Y: float = float(_SIM_YAML["scene"].get("belt_center_xy_m", [0.45, 0.0])[1])
# ★로봇 높이 반영(사용자 지시)★: 로봇 base·벨트 상면이 바닥에서 이만큼 위에 있다(작업 높이).
#   base frame 정책엔 무영향(모든 보상이 base 기준) — GUI 사실감 + 벨트 밖 낙하가 바닥까지
#   떨어지도록(드롭 종료) 하는 시각/물리 스캐폴딩. ⚠️ 실측 85cm 테이블 기준 잠정 0.80m.
STAND_HEIGHT: float = float(_SIM_YAML["scene"].get("stand_height_m", 0.80))
# ★벨트 yaw — 0 또는 ±180만 지원★ (2026-07-28(2차) 확장: 로봇이 벨트를 등지는 실셀 배치 지원)
#   큐브 스폰 분포(events_dr)·타겟 샘플링(common.yaml target.sample_range)이 전부 축정렬
#   (axis-aligned) base frame 사각형이다. 임의 각도로 벨트만 돌리면 큐브가 벨트 밖에 스폰되거나
#   타겟이 벨트를 벗어나는 '조용한 불일치'가 생긴다 → 거부한다.
#   ★그러나 180°는 안전하다★: z축 180° 회전은 (x,y)→(−x,−y)라 축정렬 박스를 자기 자신으로
#   사상하고(벨트는 y대칭), ★모든 거리를 보존★하므로 보상 std·도달성 계산이 그대로 유효하다.
#   실제 배치는 belt_center_xy_m을 −x로 옮기는 것으로 표현되며, yaw 180은 그 배치의 기록값이다.
BELT_YAW_DEG: float = float(_SIM_YAML["scene"].get("belt_yaw_deg", 0.0))
_YAW_OK = any(abs(BELT_YAW_DEG - a) < 1e-6 for a in (0.0, 180.0, -180.0))
if not _YAW_OK:
    raise RuntimeError(
        f"[cfg] ★벨트 yaw 미지원★ sim.yaml scene.belt_yaw_deg={BELT_YAW_DEG} — 스폰/타겟 샘플링이 "
        "축정렬이라 벨트만 회전하면 큐브·타겟이 벨트를 벗어난다. 허용값은 0 / ±180 뿐이다"
        "(180°는 축정렬 박스를 보존하는 특수각). 그 외 각도가 필요하면 events_dr 스폰 분포와 "
        "common.yaml target.sample_range를 함께 회전시킬 것."
    )

CAM_WIDTH: int = int(_SIM_YAML["camera"]["width"])
CAM_HEIGHT: int = int(_SIM_YAML["camera"]["height"])
CAM_VFOV_DEG: float = float(_SIM_YAML["camera"]["vertical_fov_deg"])
CAM_HFOV_DEG: float = float(_SIM_YAML["camera"].get("horizontal_fov_deg", 0.0))  # full 모드 전용
CAM_NEAR_CLIP: float = float(_SIM_YAML["camera"]["clipping_near_m"])
_CONTROL_HZ: float = float(_COMMON_YAML["control"]["control_hz"])

# USD 카메라 aperture 관례값 (upstream 기본 20.955) — FOV는 focal/aperture 비율로
# 결정되므로 aperture는 고정하고 focal을 역산한다.
_CAM_APERTURE = 20.955


def object_names(num_objects: int = NUM_OBJECTS) -> list[str]:
    """산란 객체의 scene 엔티티 이름 목록 — obs/dr/env cfg가 공유하는 네이밍 계약."""
    return [f"object_{i}" for i in range(num_objects)]


def _focal_from_vertical_fov(vfov_deg: float, aperture: float) -> float:
    """세로 FOV[deg] → USD focal_length. 정사각 렌더라 세로=가로 aperture.

    fov = 2·atan(aperture / (2·focal))  ⇒  focal = aperture / (2·tan(fov/2))
    """
    return aperture / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))


# ═════════════════════════════════════════════════════════════════════════════
# ★요소4: TiledCamera 포즈 = configs/calib/hand_eye.yaml ★T_flange_cam★ — 단 하나의 파일★
#   (2026-07-14 eye-in-hand 전환 — 구 T_base_cam(고정 카메라) 표기는 폐기)
#
#   sim  : 아래 make_tiled_camera_cfg()가 이 파일로 카메라를 wrist_3_link에 "부착"한다.
#   real : 배포측 역투영(T_base_flange @ T_flange_cam 합성)이 "같은 파일"을 읽는다.
#   ⇒ Calib_HandEye.py가 실측으로 이 yaml을 갱신하면 sim 재학습 판단
#     (dr.yaml camera_pose 섭동 범위 초과 여부)과 real 역투영이 동시에 갱신된다.
#   여기 값을 하드코딩하거나 다른 파일에서 읽는 순간 요소4가 무너진다 — 금지.
# ═════════════════════════════════════════════════════════════════════════════
def make_tiled_camera_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/wrist_3_link/WristCam",
) -> TiledCameraCfg:
    """hand_eye.yaml의 T_flange_cam으로 ★eye-in-hand(손목 장착)★ TiledCamera cfg 생성.

    ★2026-07-14 eye-in-hand 전환(사용자 확정 — GUI 확인 후 지시)★: 카메라 prim을
    wrist_3_link 하위에 부착 → 링크와 함께 움직인다. offset = T_flange_cam
    (플랜지 프레임 기준, 공칭 = -y측 75mm 브래킷·공구축 12.4° 수렴 — hand_eye.yaml).
    - ★2026-07-17 RGB-D 전환(B안)★: common.yaml preprocess.use_rgb=True면
      data_types에 'rgb' 추가(같은 tiled render product의 병행 annotator —
      tiled_camera.py L186-194 검증. NuRec 가우시안이 rgb에 렌더됨은 프로브 실증).
      depth·rgb가 단일 카메라 출력이라 sim은 자동 정합 — real의 rs.align 등가.
    - 해상도/물리 FOV/near-clip은 configs/sim.yaml, 유효 FOV = 물리×crop_fraction
      (RGB-D 계약의 물리 FOV = D435i ★color★ vfov 42.5° — sim.yaml 주석 참조).
    - convention="ros": 광축 +Z, up −Y = D435i optical frame 관례.
    - ⚠️ D435i 최소거리(~0.28m) 모사는 렌더가 아니라 ★공유 전처리의 최소거리
      무효화(depth_near_m 미만 → far)★가 담당 — sim 렌더 near-clip은 작게 유지
      (기하 컬링으로 '투시'되는 비물리 현상 방지, sim.yaml clipping_near_m).
    """
    T_flange_cam = load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml")
    pos = tuple(float(v) for v in T_flange_cam[:3, 3])
    quat_wxyz = tuple(float(v) for v in mat_to_quat(T_flange_cam[:3, :3]))
    # ★유효 FOV = tan-공간 크롭 등가(2026-07-17 적대 리뷰 수정)★ — real의 픽셀
    #   크롭(obs_preprocess._center_crop_square)은 f·tan(vfov/2)를 자르는 것이므로
    #   유효 화각은 2·atan(f·tan(vfov/2))다. 구 선형식(vfov×f)은 f<1에서 sim이
    #   real보다 넓게 렌더되는 잠복 불일치(현행 f=1.0이라 동치 — 재활성 대비 정정).
    _crop = float(_COMMON_YAML.get("preprocess", {}).get("crop_fraction", 1.0))
    effective_vfov = math.degrees(
        2.0 * math.atan(_crop * math.tan(math.radians(CAM_VFOV_DEG) / 2.0))
    )
    # ★RGB-D 계약(use_rgb) = data_types 단일 분기점★ — 관측(observations.policy_obs)
    #   과 같은 common.yaml 값을 읽어 카메라·관측이 함께 갈린다(계약 불일치 시
    #   policy_obs가 RuntimeError로 기동 거부).
    _use_rgb = bool(_COMMON_YAML.get("preprocess", {}).get("use_rgb", False))
    _data_types = ["distance_to_image_plane"] + (["rgb"] if _use_rgb else [])
    # ★frame_mode=full(2026-07-18 — RGB-D 1차 실패 근본수정)★: 정사각 픽셀 그리드
    #   (현행 112×112)에 ★비등방 FOV(가로 69.4° × 세로 42.5°)★를 담는다 — real의 848×480
    #   전체 프레임 → 112² 비등방 리사이즈와 각도 샘플링 등가. USD: 세로는 focal×
    #   vertical_aperture, 가로는 horizontal_aperture로 독립 지정(sensors_cfg.py L63/L72).
    _frame_mode = str(_COMMON_YAML.get("preprocess", {}).get("frame_mode", "square"))
    _focal = _focal_from_vertical_fov(effective_vfov, _CAM_APERTURE)
    if _frame_mode == "full" and CAM_HFOV_DEG > 0.0:
        _h_aperture = 2.0 * _focal * math.tan(math.radians(CAM_HFOV_DEG) / 2.0)
        _v_aperture = _CAM_APERTURE
    else:
        _h_aperture = _CAM_APERTURE
        _v_aperture = None  # 정사각 렌더 = 세로 aperture를 가로에서 파생(구 계약)
    # ★해상도 동기 가드(2026-07-20 감사)★: 렌더(sim.yaml camera) = 전처리 출력
    #   (common.yaml preprocess) 동일이 '크롭 no-op' 계약의 전제 — 어긋나면 sim 관측이
    #   재리사이즈되어 real과 각도 샘플링이 미세하게 갈린다(112² 전환 같은 계약 변경 시
    #   한쪽만 바꾸는 사고 차단). 기동 거부가 정상 동작.
    from common.config_io import build_preprocess_cfg as _bpc
    from common.config_io import configs_dir as _cdir
    from common.config_io import load_yaml as _ly

    _pre = _bpc(_ly(_cdir() / "common.yaml"))
    # ★Student v2(F89)★ full 모드의 렌더 격자는 ★FOV 종횡비★를 따른다(Kit 렌더는 등방 픽셀이라
    #   세로 FOV = 가로 FOV × height/width). 실기 848×480 전체 프레임 → 112² 비등방 리사이즈와
    #   같은 각도 샘플링 = width·tan(vfov/2)/tan(hfov/2) 높이로 렌더한 뒤 공유 전처리가 112²로 리사이즈.
    if _frame_mode == "full" and CAM_HFOV_DEG > 0.0:
        _expected_h = int(round(_pre.out_width * math.tan(math.radians(effective_vfov) / 2.0)
                                / math.tan(math.radians(CAM_HFOV_DEG) / 2.0)))
    else:
        _expected_h = _pre.out_height
    if (CAM_WIDTH, CAM_HEIGHT) != (_pre.out_width, _expected_h):
        raise RuntimeError(
            f"[cfg] ★해상도 계약 불일치★ sim.yaml camera {CAM_WIDTH}x{CAM_HEIGHT} ≠ 기대 "
            f"{_pre.out_width}x{_expected_h} (preprocess {_pre.out_width}x{_pre.out_height}, "
            f"FOV {CAM_HFOV_DEG}x{effective_vfov:.1f}°) — 두 파일을 함께 갱신할 것"
        )
    return TiledCameraCfg(
        prim_path=prim_path,
        # 렌더 주기 = 제어 주기(요소3의 control_hz) — 물리 스텝마다 렌더하면 낭비.
        update_period=1.0 / _CONTROL_HZ,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        data_types=_data_types,
        # far 초과/무효 → 0: real D435i의 "결측=0" 관례와 동일하게 만들어
        # 공유 전처리(결측→far 대체)가 sim/real 같은 입력 분포를 받게 한다.
        depth_clipping_behavior="zero",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=quat_wxyz, convention="ros"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_focal,
            horizontal_aperture=_h_aperture,
            vertical_aperture=_v_aperture,
            # Franka 교훈: 기본 near-clip(0.01)이 아닌 sim.yaml 값 사용.
            clipping_range=(CAM_NEAR_CLIP, 20.0),
        ),
    )


# ★d435.dae/d435i.usd 메시 프레임 → optical 관례 보정(2026-07-15 실증)★:
#   메시: 렌즈(글라스) 면=+y_mesh, 폭=x_mesh(±45mm), 상하=z_mesh(-25~0) — 재질
#   지오메트리 파싱으로 확정(전면 디테일 y=+10~12mm 면, 후면 슬롯 -y 면).
#   optical(z=광축,x=이미지우,y=이미지하) 기준 메시 축: x_m=-x_o, y_m=z_o, z_m=y_o
#   → 열벡터 행렬(대합: M=Mᵀ=M⁻¹). 본체 '시각' 부착에만 적용 — hand_eye.yaml
#   행렬은 순수 optical이라 TiledCamera/real 파이프라인은 무보정 사용.
D435_MESH_IN_OPTICAL = (
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
)


def make_d435i_body_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/wrist_3_link/D435iBody",
):
    """★D435i 카메라 '본체'(시각 전용) — 런타임 부착, hand_eye.yaml 단일 소스★.

    ★2026-07-15 v3 복원★: TCW1 체인 정정 때 본체를 URDF에 구웠다가, 사용자 GUI 배치
    워크플로(place_d435i_gui.py 고스트)와 충돌해 런타임 부착으로 되돌림 —
    브라켓만 URDF에 굽고, 본체는 이 함수가 T_flange_cam(hand_eye.yaml) 포즈로 부착.
    → 배치 수정 = yaml 수정만으로 본체·depth 센서가 함께 이동(재변환 불필요).
    - 자산: assets/d435i/d435i.usd (콜라이더·질량 없음 — 시각 전용, 사용자 승인).
      질량 72g은 URDF d435i_camera_link(visual 없는 링크)가 담당.
    - USD 미존재 시 None 반환 — 호출측이 부착 생략(가드).
    """
    usd_path = str(configs_dir().parent / "assets" / "d435i" / "d435i.usd")
    if not Path(usd_path).is_file():
        print(f"[scene][warn] D435i 본체 USD 없음({usd_path}) — 부착 생략 "
              f"(scripts/tools/convert_d435i_usd.py 실행 후 재기동 시 표시)")
        return None
    import numpy as _np

    T = load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml")
    pos = tuple(float(v) for v in T[:3, 3])
    # optical → 메시 프레임 보정(글라스=+y_mesh) — 시각 부착에만 적용
    R_mesh = T[:3, :3] @ _np.array(D435_MESH_IN_OPTICAL)
    quat_wxyz = tuple(float(v) for v in mat_to_quat(R_mesh))
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=quat_wxyz),
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        collision_group=-1,
    )


def make_nurec_background_cfg(
    prim_path: str = "{ENV_REGEX_NS}/NuRecBackground",
):
    """★NuRec 실셀 3DGS 배경(Workplace.usdz) — 파인튠 env 전용, 시각(rgb)만★.

    ★2026-07-17 B안(RGB-D 증류 + NuRec 배경 파인튠)★: 실셀 스캔 가우시안 볼륨을
    configs/calib/scene_registration.yaml(T_base_nurec + scale — 단일 소스, §4-5 커밋
    의무)로 로봇 base frame에 정합해 스폰한다. 프로브 실증(2026-07-17): 가우시안은
    TiledCamera 'rgb'에만 렌더되고 depth annotator에는 안 잡힘(메시만) → depth 계약
    불변, 참조 루트 xformOp 이동·스케일 반영 확인, 볼륨 1개당 VRAM +0.4GB(무공유).

    가드 4종 (전부 ★기동 거부★ — 이 함수의 유일 호출자는 파인튠 전용 env cfg라
    '부착 생략 후 계속'이 성립하지 않는다. 2026-07-17 적대 리뷰 반영):
      ① usdz 미존재 → RuntimeError — 배경 없는 run이 'NuRec 파인튠' 산출물로
         조용히 완주되는 것 차단(usdz는 gitignore 대용량 — 자산 없는 머신 함정).
      ② meta.calibrated=false → RuntimeError — 정합 전 기동 거부.
      ③ residual_m > 0.01 → RuntimeError — 허용오차 1cm 게이트.
      ④ ★항등 placeholder 감지★(T≈I, scale≈1) → RuntimeError — 도구 회귀로
         calibrated=true가 미정합 값에 찍혀도 방어(스캔 extent ±수십 m라 진짜
         정합이 항등일 수 없음).
    """
    usd_path = configs_dir().parent / "assets" / "Scene" / "Workplace.usdz"
    if not usd_path.is_file():
        raise RuntimeError(
            f"[scene] ★NuRec 파인튠 기동 거부★ — 배경 usdz 없음({usd_path}). "
            f"assets/Scene/Workplace.usdz(로컬 보존 자산)를 복원한 뒤 재시도 — "
            f"배경 없는 'NuRec 파인튠'은 무의미 run이라 조용히 진행하지 않는다."
        )
    reg_path = configs_dir() / "calib" / "scene_registration.yaml"
    meta = (load_yaml(reg_path) or {}).get("meta", {}) or {}
    if not bool(meta.get("calibrated", False)):
        raise RuntimeError(
            "[scene] ★NuRec 파인튠 기동 거부★ — scene_registration.yaml meta.calibrated=false "
            "(정합 전). 절차: scripts/tools/place_nurec_gui.py(고스트 GUI, [사람]) → "
            "register_nurec_umeyama.py(선택) → meta 기입·커밋 후 재시도."
        )
    residual = meta.get("residual_m", None)
    if residual is not None and float(residual) > 0.01:
        raise RuntimeError(
            f"[scene] ★NuRec 파인튠 기동 거부★ — 정합 잔차 {float(residual)*1000:.1f}mm > "
            f"허용 10mm(scene_registration.yaml 게이트). 재정합 후 재시도."
        )
    T, scale = load_scene_registration(reg_path)
    import numpy as _np

    if abs(scale - 1.0) < 1e-6 and _np.allclose(T, _np.eye(4), atol=1e-6):
        raise RuntimeError(
            "[scene] ★NuRec 파인튠 기동 거부★ — scene_registration.yaml이 항등 "
            "placeholder(T=I, scale=1)인데 calibrated=true — 미정합 값 의심. "
            "place_nurec_gui.py로 실제 정합 후 재시도."
        )
    pos = tuple(float(v) for v in T[:3, 3])
    quat_wxyz = tuple(float(v) for v in mat_to_quat(T[:3, :3]))
    print(f"[scene] ★NuRec 배경 부착★ {usd_path.name} pos={tuple(round(v, 4) for v in pos)} "
          f"scale={scale:.6g} (method={meta.get('method')}, date={meta.get('date')})")
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=quat_wxyz),
        # UsdFileCfg.scale 검증됨(from_files_cfg.py L36 FileCfg.scale) — Umeyama 균일
        # 스케일을 3축 동일 적용. 프로브에서 참조 루트 스케일 반영 확인.
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path), scale=(scale, scale, scale)),
        # 시각 전용(가우시안 볼륨 — 콜라이더 없음). -1 = 전역 그룹(d435i_body와 동일 관례).
        collision_group=-1,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ★실셀 작업대(CAD) 시각 자산 — 2026-08-10 (P8, 사용자 지시)★
#   물리는 절차적 박스 벨트(_make_belt)가 그대로 담당하고, 이 자산은 ★시각 전용★이다.
#   [왜 물리로 안 쓰나] 원본 메시가 22.6MB다. 4096 env에 콜라이더로 얹으면 비용이 크고,
#   평면 상판은 박스 근사가 물리적으로 동등하면서 접촉이 훨씬 안정적이다(F05 계열 교훈).
#   [배치 근거] `README.md §7` 실측:
#     · 자산 상판 상면 = 로컬 z 0.8702 → TABLE_HEIGHT에 맞추려면 z를 (TABLE_HEIGHT − 0.8702)
#     · 자산 상판 AABB 중심 = (−0.0463, −0.5272) → 작업면 중심(BELT_CENTER)에 맞추는 xy 보정
#   [실패 안전] 참조 지오메트리(sim_assets/, .gitignore)가 없는 PC에서는 조용히 None을 반환해
#     씬이 그대로 뜬다 — 학습은 물리(박스)만으로 성립하므로 이 자산은 선택적이다.
# ═════════════════════════════════════════════════════════════════════════════
# ★자산 실측(2026-08-10) — `workspace_table_only.usda`의 table_instanceable★
#   x[-0.191,+0.783] (0.974m) · y[±0.576] (1.151m) · z[0, ★0.870★] (0.870m)
#   ⚠️자산에는 테이블이 2개다: Shop_Table(0.653×★4.250★×0.827 = '기다란' 벤치)은
#     usda에서 active=false로 제외했다(사용자 지시). 남은 것이 작업대 본체다.
#   ★정합 근거★: 자산 상판 높이 0.870 m가 도면의 base 장착 높이 0.87 m와 정확히 일치한다
#     → 작업대 하면을 바닥에 놓으면 상판 상면이 로봇 base(z=0)에 오고,
#       작업면(TABLE_HEIGHT=-0.0375)은 상판보다 ★3.75 cm 아래★ = 도면 치수와 일치.
# ★★2026-08-11 작업대 시각 자산 제거 (사용자 지시)★★
#   `make_workcell_table_visual_cfg()`와 배치 상수(_TABLE_USD_*)를 삭제했다.
#   [영향 없음 근거] 이 자산은 ★rigid_props=None · collision_props=None★ 시각 전용이었고
#     물리 작업면은 절차적 박스 `_make_belt()`가 전담한다 — 학습 거동·보상·판정 무변화.
#   [자산 파일] `assets/Workspace/workspace_table_only.usda`는 ★디스크에 남겨 둔다★
#     (재사용 가능·생성 절차가 README.md §7에 기록돼 있다).
#   복원이 필요하면 git 이력(이 커밋의 부모)에서 함수와 씬 한 줄을 되살린다.

def make_nurec_belt_visual_cfg(sink_m: float = 0.003) -> AssetBaseCfg:
    """★NuRec 파인튠 전용: depth·시각 대역 벨트 플레이트(3mm 침하, 무충돌)★.

    ★2026-07-17 사용자 지시("워크스페이스 색이 실기 배포 갭 — 투명화해 3DGS 테이블과
    정합")의 계약 보존 구현★. 단순 투명화(비가시화)는 함정: depth annotator는 메시만
    보므로(가우시안 미기여 — 프로브 실증) 벨트를 지우면 depth가 바닥(-0.8m)으로 뚫려
    real D435i depth(실제 벨트 표면 감지)와 어긋난다. 해법 = 3중 분리:
      · 물리: 기존 collision 벨트(z=0)는 ★비가시화★(spawn.visible=False — PhysX 콜라이더는
        가시성과 무관)로 유지 → 큐브 안착·타겟 z 계약 불변.
      · RGB: 스캔의 실제 벨트(z=0에 정합)가 가우시안으로 렌더 — sim 플레이트가 안 가림.
      · depth: 본 함수의 무충돌 플레이트(상면 z=-sink_m)가 담당 — 스캔 벨트면 3mm 아래라
        가우시안이 위에서 덮어 RGB에는 거의 안 보이고, depth는 실기와 3mm 이내
        (dr.yaml depth_noise σ 범위)로 일치. 스캔 스플랫이 성긴 지점의 필러를 겸해
        실벨트 근사색(어두운 초록)을 입힌다.
    """
    bx, by, bz = BELT_SIZE
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BeltVisualNuRec",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(BELT_CENTER_X, BELT_CENTER_Y, TABLE_HEIGHT - sink_m - bz / 2.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(bx, by, bz),
            # collision_props 없음 = 콜라이더 미생성(시각·depth 전용) — 물리는 기존 벨트 담당
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.30, 0.18)),
        ),
    )


def make_finger_contact_sensor_cfg() -> ContactSensorCfg:
    """그리퍼 손가락 ContactSensor cfg — privileged critic 관측용. ★기본 OFF★.

    ⚠️ VRAM/성능 TODO: ContactSensor는 env당 상시 활성 센서라 teacher 4096 env에서
      메모리·성능 비용이 추가된다. 켜기 전에 소규모(num_envs 64)로 VRAM 실측 후
      teacher에 적용할지 결정할 것. (vision 128 env는 부담 작음.)
    ⚠️ TODO(확인 필요 — 그리퍼 자산 확정 대기): 손가락 prim 경로가 미정
      (_ur5e_assets.py GRIPPER TODO 블록). 확정 전에 이 cfg를 scene에 붙이면
      prim 매칭 실패로 기동이 죽는다 — binpick_env_cfg의 enable 플래그가 기본
      False인 이유. 또한 UR5E_CFG.spawn.activate_contact_sensors=True로 바꿔야 한다.
    """
    return ContactSensorCfg(
        # TODO(확인 필요): 그리퍼 USD 확정 후 실제 손가락 링크 경로로 교체.
        prim_path="{ENV_REGEX_NS}/Robot/.*finger.*",
        update_period=0.0,
        # 접촉 상대를 산란 객체로 한정 (바닥/bin 접촉 잡음 제거).
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Object_{i}" for i in range(NUM_OBJECTS)],
    )


def make_finger_contact_sensor_cfgs() -> dict[str, ContactSensorCfg]:
    """★STAGE 3(2026-07-09) 파지 게이트용 — 손가락별 ContactSensor 2개★.

    위 단일 cfg(.*finger.* 다중 매칭)와 달리 손가락당 '단일 prim'으로 분리 —
    filtered 접촉(force_matrix_w)은 센서 prim이 env당 1개일 때만 보장된다
    (vendored contact_sensor_cfg.py L47-51 검증). 링크명은 ur5e_2fg7.urdf 확정
    (left_finger_link/right_finger_link — 프로브 실행으로 경로·힘 값 검증됨 2026-07-09).
    전제: UR5E_CFG.spawn.activate_contact_sensors=True + 큐브 spawn도 동일(아래).
    """
    filt = [f"{{ENV_REGEX_NS}}/Object_{i}" for i in range(NUM_OBJECTS)]
    return {
        "left_finger_contact": ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_finger_link",
            update_period=0.0, filter_prim_paths_expr=list(filt)),
        "right_finger_contact": ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_finger_link",
            update_period=0.0, filter_prim_paths_expr=list(filt)),
    }


def make_scatter_object_cfg(index: int) -> RigidObjectCfg:
    """벨트 위 산란 큐브 1개의 RigidObjectCfg (index별 스폰 높이를 계단식으로 분리).

    ★큐브 = 실측 ★5cm★(OBJECT_SIZE, configs/sim.yaml)★. TODO: 실제 대상 부품 메시(USD)로
    교체하고, 형상 다양화가 필요하면 sim_utils.MultiAssetSpawnerCfg(검증됨: spawners/wrappers)
    로 무작위 형상 스폰을 검토할 것.
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Object_{index}",
        # 스폰 z를 index마다 띄워(큐브 한 변+여유) 초기 상호관통 방지 — 리셋 이벤트(events_dr)가
        # 매 에피소드 벨트 위로 재산란한다. 벨트 상면 = TABLE_HEIGHT(=0).
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BELT_CENTER_X,
                BELT_CENTER_Y,
                TABLE_HEIGHT + OBJECT_SIZE / 2.0 + 0.04 + (OBJECT_SIZE + 0.02) * index,
            ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(OBJECT_SIZE, OBJECT_SIZE, OBJECT_SIZE),  # ★5cm 큐브(실측)★
            # ★STAGE 3: 접촉 리포팅 활성★ — 손가락 ContactSensor의 filter 대상이므로
            # 큐브에도 contact reporter API 필요 (spawner_cfg.py L92, 프로브 검증).
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                # ★2026-07-10 5.0→1.0★: 그리퍼가 큐브에 깊이 파고들 때 PhysX가 큐브를 급격히
                #   밀어내(depenetration) 파지 대신 이탈시킴. 낮추면 깊은 파지가 완만히 정착 →
                #   유지력↑·이탈↓ (무지지 mid-air pregrasp 안정화). 표준 5.0보다 보수적.
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            # ★5cm 큐브 ≈ 125cm³★ — 경량 금속/플라스틱 부품.
            #   50g은 5cm 큐브에서 밀도 0.4 g/cm³에 해당한다. ★2026-08-12 사용자 확인값★이며,
            #   실제 대상 부품이 바뀌면 질량을 다시 측정한다. ★2026-07-10 0.03→0.05kg 상향★:
            #   30g는 finger 드라이브(~48N)에 너무 가벼워 파지 형성 중 튕겨나감(포렌식 확정).
            #   50g으로 접촉 안정성 향상.
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            # ★2026-07-10 마찰재질 추가★: 이전엔 물리재질 없음(PhysX 기본 저마찰) → 잡아도 큐브가
            #   중력으로 pad 사이를 미끄러져 낙하(렌더 확인). 파지 유지엔 높은 마찰 필수(Franka DexCube도
            #   고마찰 baked). static/dynamic 1.0(고무 pad ~ 금속/플라스틱 큐브 접촉 보수적 값).
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2 + 0.15 * index, 0.4, 0.8 - 0.15 * index)
            ),
        ),
    )


def add_scatter_objects(scene_cfg: InteractiveSceneCfg, num_objects: int = NUM_OBJECTS) -> None:
    """scene cfg 인스턴스에 object_0..N-1을 동적으로 추가.

    Franka Phase5 검증 패턴: configclass 인스턴스에 __post_init__ 단계에서 새 엔티티
    속성을 setattr하면 InteractiveScene이 정상 인식한다(wrist_cam 사례). 객체 수가
    configs/sim.yaml에서 오므로 클래스 정적 선언 대신 이 헬퍼를 쓴다.
    """
    for i, name in enumerate(object_names(num_objects)):
        setattr(scene_cfg, name, make_scatter_object_cfg(i))


def _static_box(
    name: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float] = (0.3, 0.3, 0.32),
) -> AssetBaseCfg:
    """정적 큐보이드(충돌만, 물리 바디 없음) 1개 — 벨트/스탠드 공용.

    ⚠️ prim_path는 top-level(`{ENV_REGEX_NS}/<name>`) — 공유 부모 아래 중첩하면 멀티-env
    클론이 존재하지 않는 부모 Xform을 찾다 실패한다(RuntimeError: Unable to find source
    prim path). Isaac Lab 관례대로 각 정적 에셋은 env 네임스페이스 바로 아래 독립 prim으로 둔다.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def _make_belt() -> AssetBaseCfg:
    """정적 컨베이어 벨트(평면 워크스페이스). 상면 = z=TABLE_HEIGHT(=0)에 오도록 배치.

    ★사용자 결정: 벨트는 정지(static) 상태에서 시작 — 별도 모션 없음★. 큐브가 이 위에
    산란·안착하고 로봇이 집어 공중 목표로 운반한다. bin 벽이 없어(평면) 밀어내면 벨트
    밖으로 떨어져 드롭 종료(=밀기 회피의 물리적 근거). 치수 = configs/sim.yaml belt_size_m.
    """
    bx, by, bz = BELT_SIZE
    return _static_box(
        "Belt",
        size=(bx, by, bz),
        pos=(BELT_CENTER_X, BELT_CENTER_Y, TABLE_HEIGHT - bz / 2.0),  # 상면 = TABLE_HEIGHT
        # ★실셀 근사색(2026-07-18 사용자 결정 — 실사 갭 원천 축소)★: 실셀은 '초록'
        #   컨베이어(NuRec 스캔 실측) — 구 (0.18,0.18,0.20) 고무색은 실사와 무관한
        #   외형이라 r5의 NuRec 배경 갭 4.5×(14.1→3.1%)의 원인 축. NuRec depth 대역
        #   플레이트(make_nurec_belt_visual_cfg)와 동일 값으로 두 도메인 RGB 일치.
        #   ⚠️ 정밀 보정은 Phase3 ObsParity(실기 D435i 프레임 대조)에서.
        color=(0.08, 0.30, 0.18),  # 실셀 초록 컨베이어 근사
    )


@configclass
class BinPickSceneCfg(InteractiveSceneCfg):
    """UR5e bin-picking 장면 — 정적 엔티티만 선언 (bin → 정적 컨베이어 벨트, 2026-07-07).

    정적: plane(바닥, base보다 STAND_HEIGHT 아래) · robot · ee_frame · belt(평면 워크스페이스) ·
    belt_stand/robot_stand(높이 스캐폴딩) · light. 산란 큐브(object_0..)는 add_scatter_objects()로,
    TiledCamera/ContactSensor는 env cfg __post_init__에서 조건부로 추가한다 (teacher는 카메라 없음).
    """

    # 바닥(floor) — ★로봇 높이 반영★: 로봇 base·벨트 상면(z=TABLE_HEIGHT)보다 STAND_HEIGHT
    # 아래. 벨트 밖으로 밀린 큐브가 바닥까지 떨어져 드롭 종료가 발화한다(밀기 회피 물리 근거).
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_HEIGHT - STAND_HEIGHT)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # UR5e (base = env origin — 요소4 전제)
    robot = UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ★요소4(2/2): TCP 프레임 = configs/calib/tcp_offset.yaml T_flange_tcp★
    # real의 getActualTCPPose 보정과 같은 파일 — 여기 오프셋을 임의 수정 금지.
    # ⚠️ TODO(확인 필요): UR5e USD의 링크 프림 이름(base_link / wrist_3_link 가정)을
    #   자산 확정 후 검증할 것. flange 링크가 "tool0"인 변형도 흔하다.
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                name="tcp",
                offset=OffsetCfg(
                    # tcp_offset.yaml에서 로드 (아래 __post_init__가 아닌 import 시
                    # 결정 — configclass 기본값은 클래스 생성 시 고정되므로 여기서
                    # 모듈 레벨 로드값을 사용한다).
                    pos=tuple(
                        float(v)
                        for v in load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")[:3, 3]
                    ),
                    rot=tuple(
                        float(v)
                        for v in mat_to_quat(
                            load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")[:3, :3]
                        )
                    ),
                ),
            ),
        ],
    )

    # ★정적 컨베이어 벨트(평면 워크스페이스) — bin 4벽 대체★. 상면 = z=TABLE_HEIGHT.
    belt = _make_belt()
    # ★2026-08-11 사용자 지시로 작업대 시각 자산 제거★ — `workcell_table`을 씬에서 뺐다.
    #   물리는 처음부터 위 belt(절차적 박스)가 전담했으므로 ★학습 거동에는 영향이 없다★
    #   (자산은 rigid_props=None·collision_props=None 시각 전용이었다).
    #   되살리려면 git 이력(이 커밋의 부모)에서 팩토리 함수와 이 배선 한 줄을 함께 복원한다.
    # 벨트 지지 스탠드(바닥→벨트 하면) + 로봇 지지대 — ★로봇 높이 반영★(시각/물리 스캐폴딩,
    # base frame 정책엔 무영향). 벨트/로봇이 바닥에서 STAND_HEIGHT 위에 놓인 워크스테이션 형상.
    # ⚠️ configclass/InteractiveScene은 클래스 본문의 모든 속성을 asset cfg로 순회하므로
    #   중간 스칼라 변수를 클래스 본문에 두면 안 된다(=asset 오인 에러). BELT_SIZE[2]를 인라인한다.
    belt_stand = _static_box(
        "BeltStand",
        size=(BELT_SIZE[0] * 0.7, BELT_SIZE[1] * 0.5, STAND_HEIGHT - BELT_SIZE[2]),
        pos=(
            BELT_CENTER_X,
            BELT_CENTER_Y,
            TABLE_HEIGHT - BELT_SIZE[2] - (STAND_HEIGHT - BELT_SIZE[2]) / 2.0,
        ),
        # 실셀 근사(2026-07-18): 벨트 프레임 = 은색 알루미늄(스캔 실측 근사)
        color=(0.62, 0.63, 0.66),
    )
    # ★로봇 받침대 = 로봇 base(z=0)까지 닿게★ (2026-07-16 float 수정). 로봇 base는
    #   z=0 고정(ur5e_assets pos=(0,0,0))인데, TCW1 farming 수정으로 TABLE_HEIGHT를
    #   -0.086으로 낮추자 받침대 상면(구: TABLE_HEIGHT)이 base보다 8.6cm 낮아져 로봇이
    #   떠 보였다. 받침대를 벨트가 아닌 로봇 base 높이에 디커플 — 바닥→base(z=0)로 채운다.
    #   (물리 무관 — 받침대는 정적 스캐폴딩, 로봇은 fix_base. 학습·그립-벨트 거리 불변.)
    robot_stand = _static_box(
        "RobotStand",
        # ★2026-08-11 도면 정합★ 0.22 → ★0.21★ (도면: 마운트 플레이트·지지대 210×210).
        size=(0.21, 0.21, STAND_HEIGHT - TABLE_HEIGHT),           # 바닥→z=0 (높이 0.87 = 도면 870)
        pos=(0.0, 0.0, (TABLE_HEIGHT - STAND_HEIGHT) / 2.0),      # 상면 = z=0(로봇 base)
        # 실셀 근사(2026-07-18): 로봇 테이블 = 알루미늄 프로파일(스캔 실측 근사)
        color=(0.62, 0.63, 0.66),
    )

    # 조명 — dr.yaml lighting.intensity_scale의 중심 기준 공칭값.
    # ⚠️ 광량 랜덤화는 stock EventTerm 없음(검증됨) → events_dr.randomize_light_intensity
    #   (커스텀 TODO)가 이 prim("/World/light")을 USD 레벨에서 조작한다.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
