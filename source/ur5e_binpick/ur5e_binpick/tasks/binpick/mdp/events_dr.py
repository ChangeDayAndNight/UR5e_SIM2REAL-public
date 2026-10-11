"""요소2: Domain Randomization EventTerm 배선 — configs/dr.yaml → Isaac Lab events.

목적: dr.yaml의 각 섹션을 (검증된) upstream EventTerm 또는 커스텀 항으로 매핑하는
      팩토리 모음. env cfg(__post_init__)가 여기서 만든 항을 events cfg에 붙인다.
      "실측이 dr.yaml 범위 안에 들어오는가"가 Phase3 게이트 — 범위 수정은 dr.yaml
      에서만 한다 (이 파일에 숫자 하드코딩 금지).
실행: 직접 실행 없음 — binpick_env_cfg.py가 import.
산출: make_*_event() 팩토리들, 커스텀 이벤트 함수(resample_light_intensity /
      randomize_camera_pose / resample_depth_noise_params / resample_rgb_noise_params /
      randomize_static_visual_color).

검증 현황(vendored source — 구 2.1.0 검증을 2.3.2로 재확인, envs/mdp/events.py):
  ✔ randomize_rigid_body_mass / randomize_rigid_body_material / randomize_actuator_gains
  ✔ randomize_visual_texture_material / randomize_visual_color (2.3.2 L1567, ManagerTermBase)
    — ⚠️ 둘 다 scene.replicate_physics=False 필수(2.3.2 L1611-1616 명시적 RuntimeError)
    — ⚠️ AssetBaseCfg 정적 자산(벨트/스탠드)에는 사용 불가: scene이 XformPrimView로
      저장(interactive_scene.py L798)해 `.cfg`가 없어 upstream init(L1624 asset.cfg.prim_path)
      이 AttributeError → 커스텀 randomize_static_visual_color가 담당(아래 ③).
  ✘ 광량 랜덤화 stock 없음(2.3.2 재확인) → 커스텀 resample_light_intensity(구현됨, 전역)
  ✘ 카메라 포즈 랜덤화 stock 없음 → 커스텀 randomize_camera_pose(구현됨, ①-e)

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import할 것.
"""

from __future__ import annotations

import os

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from common.config_io import configs_dir, load_yaml

from .scene import NUM_OBJECTS, OBJECT_SIZE, PICK_ZONE, object_names
from .ur5e_assets import GRIPPER_MAX_OPEN_GAP_M as _GRIPPER_OPEN_GAP

# 큐브 색 팔레트(빨강/파랑) — configs/sim.yaml 단일 소스. 없으면 빈 리스트 = 색 변경 안 함.
_OBJECT_COLOR_PALETTE = [
    tuple(float(v) for v in c)
    for c in load_yaml(configs_dir() / "sim.yaml")["scene"].get("object_color_palette", [])
]

_DR = load_yaml(configs_dir() / "dr.yaml")

# 산란 시 벨트 가장자리와의 여유 [m] (객체 반너비 + 약간) — 벽이 없으므로 여유를 넉넉히
# 잡아 스폰 직후 큐브가 벨트 밖으로 굴러떨어지지 않게 한다.
_SCATTER_MARGIN = OBJECT_SIZE / 2.0 + 0.05


# ═════════════════════════════════════════════════════════════════════════════
# ① 리셋: 장면 초기화 + 객체 산란 (DR는 아니지만 이벤트라 여기서 함께 배선)
# ═════════════════════════════════════════════════════════════════════════════
def make_reset_scene_event() -> EventTerm:
    """전체 장면을 default 상태로 리셋 (upstream reset_scene_to_default, L1096)."""
    return EventTerm(func=mdp.reset_scene_to_default, mode="reset")


def make_object_scatter_events(num_objects: int = NUM_OBJECTS) -> dict[str, EventTerm]:
    """객체별 벨트 위 균일 산란 이벤트 (upstream reset_root_state_uniform, L823).

    pose_range는 각 객체의 default root state(계단식 스폰 z — scene.py 참조)에
    '더해지는' 오프셋이다. z는 낙하 시작 높이의 소폭 랜덤. 벨트는 벽이 없으므로 xy 범위를
    ★픽존(PICK_ZONE)★ 안쪽(가장자리 여유 _SCATTER_MARGIN)으로 제한해 스폰 큐브가 굴러떨어지지
    않게 하고, 동시에 로봇 도달·안전 범위를 벗어나지 않게 한다.
    """
    # ★2026-07-28(3차) 픽존 기준★ — 구버전은 물리 벨트(BELT_SIZE)에서 파생했는데, 실측 벨트가
    #   100cm가 되자 모서리 스폰이 UR5e 도달한계 밖으로 나갔다(영원히 못 집는 큐브).
    bx, by = PICK_ZONE
    half_x = max(bx / 2.0 - _SCATTER_MARGIN, 0.01)
    half_y = max(by / 2.0 - _SCATTER_MARGIN, 0.01)
    terms: dict[str, EventTerm] = {}
    for name in object_names(num_objects):
        terms[f"scatter_{name}"] = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-half_x, half_x),
                    "y": (-half_y, half_y),
                    "z": (0.0, 0.05),
                    "yaw": (-3.1416, 3.1416),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(name),
            },
        )
    return terms


# ═════════════════════════════════════════════════════════════════════════════
# ①-b 역방향 커리큘럼: pre-grasp assist (2026-07-08)
#   문제: 2FG7 그립 창이 좁아(±0.5cm) 동적 파지 형성을 무작위 탐색으로 못 뚫음(TCP는 이미 정정).
#   해법(Franka 검증): 일부 env를 '큐브가 이미 잡힌 상태'로 리셋 → lift+goal 보상이 흘러 정책이
#   먼저 hold+운반을 학습. frac(pre-grasp 비율)을 curriculum이 lift 성공률로 감소시켜(1→0) 점차
#   실제 파지를 요구한다. 홈 파지점 = 실제 finger '콜라이더 pad 중심' 실측(리셋 이벤트에선
#   body_pos_w가 stale라 FK 대신 고정값). z=0.26이 MIN_LIFT(0.04) 위 → 잡은 채로 시작하면 lift 즉시 발화.
#   ★2026-07-10 정정★: 이전 (0.5214,0.1444,0.3679)은 finger 너클(바디원점)이라 pad보다 ~3.5cm 위였다
#   → pregrasp가 pad 밖에 큐브를 놓아 그립 실패. measure_grasp_frame.py로 pad 중심 실측값으로 교체.
#   ★2026-07-15 재실측(TCW1)★: 툴체인저 삽입(+86.5mm)으로 홈 파지점 하강 → (0.4919,0.1328,0.2617).
#   (경위: 홈 포즈를 reaching-over(-75/+80/-95)로 바꿨다가 farming 유발로 검증된 홈(-90/+90/-90)
#   복귀 — robot_spec.py 참조. 복귀 홈서 재측정한 값이 아래. ⚠️★홈 포즈를 바꾸면 이 값을 반드시
#   재측정★(measure_grasp_frame.py). TODO(견고화): env startup FK로 자동 산출해 하드코딩 제거.)
# ★2026-07-28 홈 포즈 교체에 따른 재산출★ (구: 0.4919, 0.1328, 0.2617 — 구 홈 -90/+90/-90)
#   새 홈 = robot_spec.UR5E_HOME_JOINT_POS (0/-90/+45/-32/-90/0, 카메라가 작업면을 보는 자세).
#   산출 = urdf_fk.fk(홈)['wrist_3_link'] @ tcp_offset.T_flange_tcp 의 병진.
#   ★검증★: 같은 방법을 구 홈에 적용하면 구 값과 0.0mm 일치(tests/test_urdf_fk.py 골든이
#   '홈 FK 파지중심 == 이 상수'를 강제하므로 둘이 어긋나면 테스트가 즉시 잡는다).
#   ⚠️ z=0.525로 구(0.262)보다 26cm 높다 — pregrasp 보조는 큐브를 그 높이에서 '이미 잡은' 상태로
#   시작시키므로 운반 거리가 길어진다. 조기 게이트(iter 300~600)에서 lift/carry 확인할 것.
# ★2026-07-28(2차) 배치 180°★: 홈 shoulder_pan 0→π에 따라 (x,y)→(−x,−y). z 불변.
#   FK 재산출 검증 완료(오차 0.0mm) — tests/test_home_pose_contract.py가 홈 FK와의 정합을 강제한다.
# ★2026-07-30 툴 체인 5단 정정★: tool flange(5mm)·QC-RS v3(11.25mm) 두 단이 누락돼 있었다
#   (사용자 실측 tool0→그리퍼끝 265mm). tcp_offset z 0.2262→0.2425 → 파지중심 16.26mm 이동.
# ★2026-08-30 홈 교체(0/-76/-103/-96/+93/+4)에 맞춰 재산출★ — URDF FK ∘ tcp_offset.
#   구값 (-0.2975, -0.1328, 0.5091)은 구 홈(180/-90/+45/-32/-90/0)의 파지중심이었다.
#   홈을 바꾸면 이 값이 즉시 무효가 되므로 함께 갱신한다(tests/test_urdf_fk.py가 정합 강제).
_PREGRASP_GRIP = (-0.3588, 0.1159, 0.2327)  # 홈 포즈 finger 콜라이더 pad 중점(=파지중심), env 원점 기준
from .ur5e_assets import GRIPPER_CLOSE_M as _GRIPPER_CLOSE_Q  # ★단일 소스: close=0.020(깊은 파지).
# ⚠️ 0.011(얕은 파지)은 접촉력 부족으로 파지 0% → 기각됨(ur5e_assets.py 참조). 이 값을 낮추지 말 것.


def make_pregrasp_assist_event(object_name: str = "object_0", default_frac: float = 0.75) -> EventTerm:
    """리셋 시 frac 비율의 env에서 object_name을 홈 파지점에 놓고 그리퍼를 닫아 '잡힌 상태'로 시작.

    ★scatter_* 이후·reset_all 이후에 적용되어야 함★(그래야 object_0 위치·그리퍼를 덮어쓴다) →
    env cfg __post_init__에서 마지막에 setattr. frac은 env._bp_pregrasp_frac(curriculum)이 갱신,
    없으면 default_frac.
    """
    return EventTerm(
        func=_pregrasp_reset,
        mode="reset",
        params={"object_name": object_name, "default_frac": default_frac},
    )


def _pregrasp_reset(env, env_ids, object_name: str, default_frac: float) -> None:
    import torch

    ids = (
        torch.arange(env.num_envs, device=env.device)
        if isinstance(env_ids, slice)
        else (env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids, device=env.device))
    )
    # ★place-farming 차단용 pregrasp-env 플래그 영속화(2026-07-16)★: '잡힌 채 시작한' env는
    #   place 완성 보상(placed/release_bias/place_fine)에서 제외한다(binpick_env_cfg._real_pick_mask).
    #   근거: pregrasp가 리셋 즉시 (grasped∧z>MIN_LIFT)로 배달 latch를 사실상 공짜로 얻어, 타겟이
    #   pregrasp 바로 아래에 스폰되면 '내려놓기'만으로 placed(w50)를 farm → grasp 학습 붕괴(TCW1이
    #   그립을 8.6cm 낮춰 이 지름길을 가깝게 만들어 촉발). place-side 보상을 pregrasp env에서 끊으면
    #   공짜 latch가 place 보상에 도달하는 경로가 원천 차단되고, pick/carry(grasp/lift/goal) 부트스트랩은
    #   유지된다. real-pick env(¬pregrasp)에서만 release gradient가 흘러 정당하게 place를 학습하며,
    #   frac→0에서 전 env가 real-pick이 되어 배포 조건과 일치. (BINPICK_STAGE=2의 '잡힌 채 시작 제거'
    #   통찰을 stage 경계 없이 per-env로 연속 구현.)
    if not hasattr(env, "_bp_pregrasp_env") or env._bp_pregrasp_env.shape[0] != env.num_envs:
        env._bp_pregrasp_env = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if ids.numel() == 0:
        return
    env._bp_pregrasp_env[ids] = False   # 리셋되는 env 기본값 = real-pick(place 보상 마스크 안 함)
    frac = float(getattr(env, "_bp_pregrasp_frac", default_frac))
    if frac <= 0.0:
        return
    sel = ids[torch.rand(ids.numel(), device=env.device) < frac]
    if sel.numel() == 0:
        return
    env._bp_pregrasp_env[sel] = True    # '잡힌 채 시작' env → place-side 보상 차단 대상
    # ① 큐브를 홈 파지점(world = env 원점 + 파지점)에 정지 배치
    obj = env.scene[object_name]
    grip = torch.tensor(_PREGRASP_GRIP, device=env.device, dtype=torch.float32)
    pose = torch.zeros((sel.numel(), 7), device=env.device)
    pose[:, :3] = env.scene.env_origins[sel] + grip
    pose[:, 3] = 1.0  # identity quat (wxyz)
    obj.write_root_pose_to_sim(pose, env_ids=sel)
    obj.write_root_velocity_to_sim(torch.zeros((sel.numel(), 6), device=env.device), env_ids=sel)
    # ② 그리퍼를 큐브에 ★t=0부터 물린 상태★로 세팅 (2026-07-10 근본 재설계)
    #   이전: finger를 접촉 q(gap=cube_size, 큐브면에 '닿기만')로 두고 close로 눌렀다. 문제: t=0에
    #   그립력 0 → 무지지 큐브가 즉시 자유낙하, finger 폐합(velocity_limit_sim=0.05m/s)보다 빨리
    #   그립존을 벗어남(렌더·프로브 확정). → finger를 큐브에 '살짝 파고든 clamp q'로 두어 t=0부터
    #   그립력을 형성하고(대칭이라 순 힘 0 → 큐브 유지), 큐브 velocity=0으로 낙하 시작을 막는다.
    #   clamp q: gap = cube_size − penetration. _pen=0.007(3.5mm/측) → gap = 0.030−0.007 = 0.023 →
    #   q = (0.044−gap)/2 = 0.0105 (아래 L141 공식과 일치).
    robot = env.scene["robot"]
    if not hasattr(env, "_bp_finger_ids"):
        env._bp_finger_ids, _ = robot.find_joints(["left_finger_joint", "right_finger_joint"])
    fj = env._bp_finger_ids
    _pen = 0.007  # 총 파고듦[m] (3.5mm/측) — 즉시 그립력(박스 콜라이더), 과하면 depenetration 이탈
    # ★2026-07-29 하드코딩 제거(실사고)★: 구 코드는 열림 폭을 ★0.044로 박아뒀다★.
    #   핑거팁 실측 반영으로 열림이 80mm가 되자 clamp_q가 0.0005(=간격 79mm)로 계산돼
    #   ★pregrasp가 사실상 활짝 열린 채 시작★ → 큐브가 즉시 낙하 → grasp 보상 영구 0.
    #   (파지 물리 문제로 오진하고 마찰·강성·폐합속도를 한참 뒤졌다.)
    #   간격 모델: gap(q) = OPEN − 2q  →  gap = OBJECT_SIZE − _pen 이 되는 q를 푼다.
    clamp_q = max(0.0, min(_GRIPPER_CLOSE_Q, (_GRIPPER_OPEN_GAP - OBJECT_SIZE + _pen) / 2.0))
    jpos = torch.full((sel.numel(), len(fj)), clamp_q, device=env.device)
    robot.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), joint_ids=fj, env_ids=sel)
    # 드라이브 target = close(0.020)로 눌러 그립 유지(velocity_limit_sim이 폐합속도를 캡 →
    # 큐브에 막혀 clamp 부근에서 정지, 지속 그립력). 마찰재질(binpick_env_cfg 전역)이 낙하 저항.
    robot.set_joint_position_target(
        torch.full((sel.numel(), len(fj)), _GRIPPER_CLOSE_Q, device=env.device),
        joint_ids=fj,
        env_ids=sel,
    )


def make_pregrasp_curriculum(
    start: float = 0.9,
    end: float = 0.0,
    warmup_steps: int = 12000,
    decay_steps: int = 48000,
    min_grasp_ema: float = 0.30,
    update_interval_steps: int = 2400,
    max_drop_per_update: float = 0.05,
    deadline_steps: int = 120000,
):
    """★2026-08-05(P1-b ①) 단축: warmup 20000→12000 · decay 120000→48000★
    frac 0 도달이 iter 5833 → ★2500★. 근거: 이 일정을 늘린 이유가 "실측 그리퍼로 바꾼 뒤
    파지가 안 자란다"(v5)였는데, 그 원인이 ★F29·F30(그리퍼 규격·pregrasp 하드코딩)과
    F46(그리퍼 액션 포화)★이었고 전부 해소됐다 ⇒ ★존재하지 않는 결함에 대한 보상책★이었다.
    실측: 구 일정에서도 자력 파지는 iter 5000~5500에 회복(grasp 0.655 → 1.277).
    ⚠️안전장치: 페널티 최종단에 ★자력 파지 EMA 게이트★(P1-b ④)가 걸려 있어, 파지가 덜 자라면
      강한 페널티가 지연된다 — 단축이 과했을 때의 완충.

    ★역방향 커리큘럼 스케줄★: pre-grasp 비율(_bp_pregrasp_frac)을 warmup 후 선형 감소(start→end).

    처음엔 frac=start(대부분 잡힌 상태로 시작 → hold+운반 학습), 이후 감소시켜 점차 실제 파지를
    요구한다(frac=end=0에서 순수 파지). 스텝 기반(common_step_counter) — num_steps_per_env에 무관.
    ⚠️ 성공률 적응형(Franka reverse_place)이 이상적이나, pre-grasp에선 lift가 항상 높아 신호가
    애매 → 우선 스케줄. 전체 재학습(4096×2500)서 frac=0 도달 후 파지 유지되는지가 최종 게이트.

    ★2026-07-29 커리큘럼 강화(사용자 지시)★ — 구: start0.75 · warmup3000 · decay40000
      (= iter 125부터 감소, iter 1792에 보조 소멸). 실측 그리퍼(열림 80mm)로 바꾼 뒤
      탐색 난이도가 올라(접촉까지 편측 15mm ← 구 7.3mm) 그 일정 안에 파지가 자라지 못했다
      (v5: iter50 grasp 0.0086 → iter500 0.0118, 1.4배. 구 계보는 같은 구간 17배).
      → start 1.0(전 env 보조) · warmup 20000스텝(≈833 iter) · decay 120000(≈5000 iter):
      보조를 훨씬 오래 유지해 hold+운반을 먼저 굳히고, 파지 요구는 iter 833부터 5833에 걸쳐
      완만히 올린다. 10000 iter 예산에서 마지막 ~4000 iter는 순수 자력 파지 구간이라
      '보조에 기댄 채 끝나는' 위험은 없다(_real_pick_mask가 place 보상도 차단).
    """
    from isaaclab.managers import CurriculumTermCfg

    return CurriculumTermCfg(
        func=_pregrasp_frac_schedule,
        params={
            "start": start,
            "end": end,
            "warmup": warmup_steps,
            "decay": decay_steps,
            "min_grasp_ema": min_grasp_ema,
            "update_interval": update_interval_steps,
            "max_drop": max_drop_per_update,
            "deadline": deadline_steps,
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# ★커리큘럼 스텝 오프셋 (2026-08-01, F44)★
#
# 문제: 모든 스텝 기반 커리큘럼은 `env.common_step_counter`를 읽는데, 이 카운터는
#   ★resume해도 0에서 다시 시작★한다(env가 새로 만들어지므로). 그 결과 iter 9550
#   체크포인트에서 600 iter만 resume하면 pregrasp frac이 ★1.0으로 되돌아간다★ —
#   warmup 20000스텝(833 iter)조차 못 넘기기 때문. 그러면 전 env가 pregrasp이 되고
#   `_real_pick_mask`가 0이라 place 계열 보상이 ★구조적으로 정확히 0★이 된다.
#   실제로 F42·F43 검증 run 두 건이 이 조건에서 측정돼 무효가 됐다
#   (goal_fine 1.87 → 0.0000, placed 0.0725 → 0.0000).
#
# 해법: 환경변수 `BP_CURRICULUM_STEP_OFFSET`로 시작 스텝을 주입한다.
#   Phase1/BinPick_Train.py가 `--resume` 시 체크포인트 iter × num_steps_per_env로
#   ★자동 계산해 설정★하므로, 사람이 기억할 필요가 없다(재발 방지의 핵심).
_STEP_OFFSET: int = int(os.environ.get("BP_CURRICULUM_STEP_OFFSET", "0") or 0)


def curriculum_step(env) -> int:
    """커리큘럼이 참조할 학습 진행 스텝 — resume 오프셋을 더한 값.

    ⚠️스텝 기반 커리큘럼은 ★전부★ 이 함수를 써야 한다. `common_step_counter`를 직접
    읽으면 resume에서 커리큘럼이 되감긴다(F44).
    """
    return int(getattr(env, "common_step_counter", 0)) + _STEP_OFFSET


def _pregrasp_frac_schedule(
    env,
    env_ids,
    start: float,
    end: float,
    warmup: int,
    decay: int,
    min_grasp_ema: float,
    update_interval: int,
    max_drop: float,
    deadline: int,
) -> float:
    """Performance-gated reverse curriculum with a fixed real-pick probe cohort.

    Ten percent of environments are real-pick probes from the beginning.  The
    assist fraction may follow its nominal time schedule only after those probes
    demonstrate sufficient grasp ability.  Bounded updates avoid a discontinuous
    collapse; the deadline prevents an indefinitely assisted policy.
    """
    s = curriculum_step(env)
    if s < warmup:
        env._bp_pregrasp_frac = float(start)
        env._bp_pregrasp_last_update = s
        return float(start)

    nominal = max(end, start - (start - end) * (s - warmup) / max(1, decay))
    current = float(getattr(env, "_bp_pregrasp_frac", start))
    last = int(getattr(env, "_bp_pregrasp_last_update", warmup))
    ema = float(getattr(env, "_bp_grasp_ema", 0.0))
    eligible = ema >= min_grasp_ema or s >= deadline
    # ★F81 정정 — deadline은 '자격'만 부여한다. 갱신 ★주기★까지 무시하면 안 된다★
    #   구 조건 `(s - last >= update_interval or s >= deadline)`은 deadline 이후 이 함수가
    #   호출될 때마다(리셋 배치마다) max_drop을 깎아, 자기 계약("Bounded updates avoid a
    #   discontinuous collapse")을 스스로 위반했다 — 실측: deadline 직후 0.9→0 급락.
    #   갱신 주기는 언제나 update_interval로 강제한다.
    if eligible and s - last >= update_interval:
        frac = max(nominal, current - max_drop)
        env._bp_pregrasp_last_update = s
    else:
        frac = current
    env._bp_pregrasp_frac = float(frac)
    return float(frac)


def make_target_radius_curriculum(
    start: float = 0.05, end: float = 0.35, warmup_steps: int = 3000, decay_steps: int = 100000
):  # ★2026-07-24 decay 40000→100000★: b5-v1 붕괴(radius 0.35 급도달 iter1800→grasp 0) 수정.
   #   b4가 균일 45cm서 grasp 43% = radius 0.35는 능력 내이므로 원인은 확대 '속도' → 완만화.
   #   radius 0.35 도달 iter~4300(파지 성숙 동반), 붕괴 구간 iter2000엔 radius 0.185로 안전.
    """★b5 목표거리 역커리큘럼(2026-07-23)★: 목표를 물체 근처에 두는 반경(_bp_target_radius)을
    warmup 후 선형 '증가'(start→end)시킨다 — 초기엔 짧은 운반(carry→placed 전환을 자주 성공)
    후 점차 원거리 운반으로 확대(Florensa 2017 Reverse Curriculum). TargetCurriculumPoseCommand.
    _resample_command가 _bp_target_radius를 소비. 스텝 기반(common_step_counter — pregrasp
    커리큘럼과 동형, num_steps_per_env 무관). ★farming 회피는 _real_pick_mask가 담당★."""
    from isaaclab.managers import CurriculumTermCfg

    return CurriculumTermCfg(
        func=_target_radius_schedule,
        params={"start": start, "end": end, "warmup": warmup_steps, "decay": decay_steps},
    )


def _target_radius_schedule(env, env_ids, start: float, end: float, warmup: int, decay: int) -> float:
    s = curriculum_step(env)   # ★F44★ resume 오프셋 반영 — 직접 카운터 읽기 금지
    if s < warmup:
        r = start
    else:
        r = min(end, start + (end - start) * (s - warmup) / max(1, decay))
    env._bp_target_radius = float(r)
    env._bp_target_curr_on = True  # command가 참조하는 활성 플래그(term 부착 = 게이트 on)
    return float(r)


# ═════════════════════════════════════════════════════════════════════════════
# ①-e ★동작 페널티 단계 강화 커리큘럼 (F43/F45, 2026-08-01)★
# ═════════════════════════════════════════════════════════════════════════════
def make_penalty_schedule_curriculum(term_name: str, schedule: list[tuple[int, float]]):
    """보상항 `term_name`의 가중치를 스텝 구간별로 교체한다 (약하게 시작 → 단계적 강화).

    ★근거(업스트림 직계)★: Isaac Lab v2.3.2 `manipulation/lift/lift_env_cfg.py`의
    CurriculumCfg가 `modify_reward_weight`로 action_rate·joint_vel을 −1e-4 → ★−1e-1★로
    올린다. 그 함수는 램프가 아니라 `common_step_counter > num_steps`에서의 ★교체★다
    (envs/mdp/curriculums.py). 여기서는 ★여러 단계★를 하나의 term으로 다루도록 일반화했다.

    업스트림 방식(term 하나에 CurrTerm 여러 개 부착)을 쓰지 않은 이유: 각 CurrTerm이
    매 스텝 자기 값을 덮어쓰므로 ★매니저 순회 순서에 결과가 의존★한다. 단일 term이
    '조건을 만족하는 마지막 단계'를 고르면 순서 의존이 사라진다.

    왜 단계적인가(실측): −1e-1을 이미 수렴한 정책에 갑자기 걸면 정지는 완벽히 해결되지만
    (팔 4.23 → 0.043 rad/s, 정지 전환율 7.6% → 79.5%) ★정밀 접근이 약해졌다★
    (타겟 3cm 도달 58.8% → 31.8%). 중간 단계를 둬 정밀 접근을 먼저 굳힌다.

    Args:
        term_name: RewardsCfg의 항 이름 (예: "action_rate")
        schedule: [(시작 스텝, 가중치), ...] — 스텝 오름차순. 현재 스텝 이하인 마지막 항목 적용.
                  스텝은 `curriculum_step`(= common_step_counter + resume 오프셋) 기준이라
                  ★resume에서도 되감기지 않는다★(F44).
    """
    from isaaclab.managers import CurriculumTermCfg

    assert schedule and all(
        schedule[i][0] < schedule[i + 1][0] for i in range(len(schedule) - 1)
    ), f"schedule은 스텝 오름차순이어야 합니다: {schedule}"
    return CurriculumTermCfg(
        func=_penalty_schedule, params={"term_name": term_name, "schedule": list(schedule)}
    )


def _penalty_schedule(env, env_ids, term_name: str, schedule, deadline_mult: float = 2.5) -> float:
    """스케줄 항목은 `(시작스텝, 가중치)` 또는 `(시작스텝, 가중치, 최소_파지_EMA)`.

    ★능력 게이트(2026-08-05, P1-b ④)★: 세 번째 원소가 있으면 `env._bp_grasp_ema`(자력 파지
    성공률 EMA — `grasped_gate`가 갱신)가 그 값 이상일 때만 발동한다. ★지연 전용★이며
    앞당기지 않는다. 커리큘럼을 압축했을 때 "파지 기본기가 덜 섰는데 강한 페널티가 걸려
    carry가 무너지는"(F45: 97.3 → 82.0%) 사고를 막는 완충 장치다.
    ⚠️무한 대기 방지: `시작스텝 × deadline_mult`를 넘으면 EMA와 무관하게 발동한다.
    """
    s = curriculum_step(env)
    ema = float(getattr(env, "_bp_grasp_ema", 1.0))   # 미설정이면 게이트 무효화(1.0)
    weight = None
    for entry in schedule:
        start, w = entry[0], entry[1]
        need = entry[2] if len(entry) > 2 else 0.0
        if s >= start and (ema >= need or s >= start * deadline_mult):
            weight = w
    if weight is None:
        # 첫 단계 이전 — cfg에 적힌 초기 가중치를 그대로 둔다
        return float(env.reward_manager.get_term_cfg(term_name).weight)
    cfg = env.reward_manager.get_term_cfg(term_name)
    if cfg.weight != weight:
        cfg.weight = float(weight)
        env.reward_manager.set_term_cfg(term_name, cfg)
        print(f"[curriculum] ★{term_name} 가중치 → {weight}★ (step {s}, 자력파지 EMA {ema:.3f})")
    return float(weight)


# (①-c place-side 부트스트랩(v1 already-released / v2 grasped-mid-carry)과 place_frac 커리큘럼은
#  '잡힌 채 시작' 보조 env의 place 보상 farming으로 pick+carry가 붕괴해 접근 자체를 폐기(2026-07-11).
#  현행 릴리스 학습 = BINPICK_STAGE 2단계 + release_bias smooth(binpick_env_cfg.py).
#  실패 경위 = README.md 결정 로그, 삭제된 구현 = git 히스토리 참조.)


# ═════════════════════════════════════════════════════════════════════════════
# ①-d ★관측 지연 DR (2026-07-15, 필수 판정 적용)★ — 요소2 latency
#   실기 루프(D435i 취득→전처리→추론→전송)는 지연이 반드시 존재. 정책이 무지연
#   sim에만 적합하면 배포 열화 확정 → 에피소드마다 env별 지연(dr.yaml
#   latency.obs_delay_steps)을 재샘플한다. 기본 소비처는 RGB-D만 지연하고 최신
#   proprio를 유지한다. delay_proprio=True는 구 whole-observation 대조군 전용.
#   isaaclab.utils.buffers.DelayBuffer 사용(자체 구현 금지 규칙 — API vendored 검증:
#   set_time_lag(int|Tensor, batch_ids), reset(batch_ids), compute(data)→지연본.
#   'lag>버퍼길이면 최신 반환'이라 리셋 직후 이전 에피소드 누출 없음).
#   소비처: observations.policy_obs 말단이 env._bp_obs_delay.compute() 통과.
#   ⚠️ 액션측 지연은 실측치(real.local.yaml measured_action_delay_ms) 확보 전 보류.
# ═════════════════════════════════════════════════════════════════════════════
def make_obs_delay_event() -> EventTerm:
    lo, hi = _DR["latency"]["obs_delay_steps"]
    return EventTerm(func=_resample_obs_delay, mode="reset",
                     params={"lo": int(lo), "hi": int(hi)})


def _resample_obs_delay(env, env_ids, lo: int, hi: int) -> None:
    import torch

    from isaaclab.utils.buffers import DelayBuffer

    buf = getattr(env, "_bp_obs_delay", None)
    if buf is None or buf.batch_size != env.num_envs:
        buf = DelayBuffer(hi, env.num_envs, device=env.device)
        env._bp_obs_delay = buf
    all_ids = torch.arange(env.num_envs, device=env.device)
    ids = (all_ids if env_ids is None else all_ids[env_ids] if isinstance(env_ids, slice)
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if ids.numel() == 0:
        return
    # ⚠️ dtype=torch.int 필수 — DelayBuffer._time_lags가 int32라 int64(randint 기본)면
    #   index_put dtype 불일치 RuntimeError (2026-07-15 프로브 크래시로 실측 확인).
    lags = torch.randint(lo, hi + 1, (ids.numel(),), dtype=torch.int, device=env.device)
    buf.set_time_lag(lags, batch_ids=ids.tolist())
    buf.reset(batch_ids=ids.tolist())


# ═════════════════════════════════════════════════════════════════════════════
# ①-e ★카메라 pose DR (2026-07-15, 필수 판정 적용 — 커스텀, stock 부재 검증됨)★
#   dr.yaml camera_pose(±1cm/±1°) = '허용 캘리브레이션 오차'(§4-5 재학습 판정과 연결).
#   eye-in-hand 마운트 공차는 로봇별 '고정' 오차이므로 에피소드가 아니라 ★startup에
#   env별 1회★ 섭동 — 512 env가 서로 다른 오조립을 동시에 학습(정확한 의미론).
#   구현: 각 env의 WristCam 프림 로컬 xformOp(translate/orient)에 델타 합성.
# ═════════════════════════════════════════════════════════════════════════════
def make_camera_pose_event() -> EventTerm:
    p = _DR["camera_pose"]
    return EventTerm(func=randomize_camera_pose, mode="startup",
                     params={"pos_m": float(p["pos_perturb_m"]),
                             "rot_deg": float(p["rot_perturb_deg"])})


def randomize_camera_pose(env, env_ids, pos_m: float, rot_deg: float) -> None:
    import torch
    import omni.usd
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()
    n = env.num_envs
    d_pos = (torch.rand(n, 3) * 2.0 - 1.0) * pos_m       # ±pos_m [m]
    d_rpy = (torch.rand(n, 3) * 2.0 - 1.0) * rot_deg     # ±rot_deg [deg]
    applied = 0
    for i in range(n):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/wrist_3_link/WristCam")
        if not prim.IsValid():
            continue
        t_attr = prim.GetAttribute("xformOp:translate")
        o_attr = prim.GetAttribute("xformOp:orient")
        if not t_attr or not o_attr:
            continue
        t = t_attr.Get()
        t_attr.Set(type(t)(t[0] + float(d_pos[i, 0]), t[1] + float(d_pos[i, 1]), t[2] + float(d_pos[i, 2])))
        rot = (
            Gf.Rotation(Gf.Vec3d(1, 0, 0), float(d_rpy[i, 0]))
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), float(d_rpy[i, 1]))
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(d_rpy[i, 2]))
        )
        dq = rot.GetQuat()
        q = o_attr.Get()
        q_cls = type(q)  # Gf.Quatf | Gf.Quatd — 프림 타입 유지
        new_q = Gf.Quatd(q.GetReal(), Gf.Vec3d(*q.GetImaginary())) * dq
        o_attr.Set(q_cls(new_q.GetReal(), type(q.GetImaginary())(*new_q.GetImaginary())))
        applied += 1
    print(f"[dr] ★카메라 pose DR★ {applied}/{n} env WristCam에 ±{pos_m*100:.1f}cm/±{rot_deg:.1f}° 섭동(startup 1회)",
          flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# ② physics 섹션 → 검증된 upstream 항
# ═════════════════════════════════════════════════════════════════════════════
def make_object_mass_event(object_name: str) -> EventTerm:
    """객체 질량 곱셈 스케일 (dr.yaml physics.object_mass_scale)."""
    lo, hi = _DR["physics"]["object_mass_scale"]
    return EventTerm(
        func=mdp.randomize_rigid_body_mass,
        # upstream tip: CPU 텐서 조작이라 초기화 시 1회 권장 → startup.
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(object_name),
            "mass_distribution_params": (float(lo), float(hi)),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


def make_object_material_event(object_name: str) -> EventTerm:
    """객체 마찰/반발 계수 (dr.yaml physics.object_friction / object_restitution)."""
    f_lo, f_hi = _DR["physics"]["object_friction"]
    r_lo, r_hi = _DR["physics"]["object_restitution"]
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",  # upstream 주의: 물리 머티리얼은 초기화 시 버킷 샘플(L165-173)
        params={
            "asset_cfg": SceneEntityCfg(object_name),
            "static_friction_range": (float(f_lo), float(f_hi)),
            "dynamic_friction_range": (float(f_lo), float(f_hi)),
            "restitution_range": (float(r_lo), float(r_hi)),
            "num_buckets": 64,
            "make_consistent": True,  # dynamic <= static 물리 제약 준수
        },
    )


def make_finger_material_event() -> EventTerm:
    """Randomise both 2FG7 finger-pad materials using the current SysID prior."""
    f_lo, f_hi = _DR["physics"]["finger_friction"]
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (float(f_lo), float(f_hi)),
            "dynamic_friction_range": (float(f_lo), float(f_hi)),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


def make_actuator_gains_event() -> EventTerm:
    """로봇 관절 게인 스케일 (dr.yaml physics.joint_stiffness_scale / damping_scale).

    upstream tip(L471-473): implicit actuator는 CPU 경유라 startup 권장.
    TODO: 에피소드별 재샘플이 꼭 필요해지면 mode="reset" 전환 후 스텝 시간 실측.
    """
    s_lo, s_hi = _DR["physics"]["joint_stiffness_scale"]
    d_lo, d_hi = _DR["physics"]["joint_damping_scale"]
    return EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (float(s_lo), float(s_hi)),
            "damping_distribution_params": (float(d_lo), float(d_hi)),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# ③ lighting 섹션 → 색은 upstream, 광량은 커스텀(TODO)
# ═════════════════════════════════════════════════════════════════════════════
def make_visual_color_event(object_name: str) -> EventTerm:
    """객체 시각 색상 랜덤화 (upstream randomize_visual_color — 2.3.2 L1567 ManagerTermBase).

    ⚠️ 전제(소스 검증): scene.replicate_physics=False 아니면 RuntimeError
      (2.3.2 events.py L1611-1616) — vision env cfg가 색 DR 부착 시 함께 끈다.
    ✔ mesh_name 검증 완료(2026-07-17): CuboidCfg 스폰 구조 = "{prim}/geometry/mesh"
      (vendored shapes.py L290-294 _spawn_geom_from_prim_type) — upstream 관례
      "{asset_prim_path}/{mesh_name}"(L1574-1578)과 결합해 "geometry/mesh" 확정.
    """
    return EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(object_name),
            "colors": {"r": (0.05, 0.95), "g": (0.05, 0.95), "b": (0.05, 0.95)},
            "event_name": f"color_{object_name}",
            "mesh_name": "geometry/mesh",  # ✔ 검증됨(shapes.py L290-294)
        },
    )


def randomize_static_visual_color(
    env,
    env_ids,
    entity_prim_name: str,
    colors: dict[str, tuple[float, float]],
) -> None:
    """★커스텀★ AssetBaseCfg 정적 자산(벨트/스탠드)의 PreviewSurface 색 랜덤화 (리셋, per-env).

    upstream randomize_visual_color를 못 쓰는 이유(소스 검증): AssetBaseCfg 자산은
    InteractiveScene이 XformPrimView로 저장(interactive_scene.py L798)하는데 이 뷰에는
    `.cfg`가 없어 upstream init의 `asset.cfg.prim_path`(events.py L1624)가 AttributeError.
    → randomize_camera_pose(①-e)와 같은 'env별 USD prim 직접 조작' 패턴으로 구현.

    prim 경로(소스 검증): CuboidCfg + PreviewSurfaceCfg 스폰 구조 =
      "/World/envs/env_{i}/<name>/geometry/material/Shader"
      (shapes.py L290-307: 머티리얼 = {prim}/geometry/material + visual_materials.py
       L62-93: Shader 자식 프림에 inputs:diffuseColor 속성 — camelCase 변환 확인).
    """
    from pxr import Gf

    stage = env.scene.stage  # interactive_scene.py L138 — 검증됨
    ids = (
        torch.arange(env.num_envs)
        if isinstance(env_ids, slice) or env_ids is None
        else (env_ids.cpu() if torch.is_tensor(env_ids) else torch.tensor(env_ids))
    )
    if ids.numel() == 0:
        return
    lo = torch.tensor([colors[k][0] for k in ("r", "g", "b")])
    hi = torch.tensor([colors[k][1] for k in ("r", "g", "b")])
    rgb = lo + torch.rand(ids.numel(), 3) * (hi - lo)
    for row, i in enumerate(ids.tolist()):
        prim = stage.GetPrimAtPath(
            f"/World/envs/env_{i}/{entity_prim_name}/geometry/material/Shader"
        )
        if not prim.IsValid():
            continue
        attr = prim.GetAttribute("inputs:diffuseColor")
        if not attr:
            continue
        attr.Set(Gf.Vec3f(float(rgb[row, 0]), float(rgb[row, 1]), float(rgb[row, 2])))


def randomize_visual_color_choice(
    env,
    env_ids,
    entity_prim_name: str,
    palette: list[tuple[float, float, float]],
) -> None:
    """★이산 팔레트★에서 색을 골라 per-env로 적용 (예: 빨강/파랑 큐브).

    ★사용자 지시(2026-08-11)★: "큐브를 생성할 때 빨간색, 파란색 큐브가 나올 수 있게".
    연속 균일(randomize_static_visual_color)과 달리 ★지정된 색 중 하나★를 고른다 —
    실물 부품이 특정 색으로 정해져 있을 때 그 분포를 그대로 재현하기 위함이다.

    ★upstream randomize_visual_color를 쓰지 않는 이유★:
      그 클래스는 `scene.replicate_physics=False`를 ★강제★한다(events.py L1611-1616 RuntimeError).
      4096 env에서 replication을 끄면 물리 파싱이 env마다 일어나 VRAM·기동시간이 폭증한다
      (그래서 기존 색 DR은 vision cfg의 num_envs 128에만 붙어 있다).
      여기서는 `randomize_static_visual_color`와 같은 ★USD prim 직접 조작★ 패턴을 써서
      replication을 켠 채로 색만 바꾼다.

    ★mode="startup" 권장★: 색은 학습에 영향이 없고(teacher는 카메라가 없다) per-env USD 쓰기가
      env 수만큼 드는 Python 루프이므로, 리셋마다 돌리면 4096 env에서 비용이 크다.
      startup이면 ★기동 시 1회★로 끝나고, 그 run 동안 env마다 고정된 색을 갖는다
      (= 씬에 빨강·파랑 큐브가 섞여 나온다). 에피소드마다 바꿔야 하면 mode="reset"으로 올린다.

    prim 경로 근거는 `randomize_static_visual_color` docstring과 동일
      ("/World/envs/env_{i}/<name>/geometry/material/Shader", shapes.py L290-307).
    """
    from pxr import Gf

    if not palette:
        return
    stage = env.scene.stage
    ids = (
        torch.arange(env.num_envs)
        if isinstance(env_ids, slice) or env_ids is None
        else (env_ids.cpu() if torch.is_tensor(env_ids) else torch.tensor(env_ids))
    )
    if ids.numel() == 0:
        return
    pick = torch.randint(0, len(palette), (ids.numel(),))
    applied = 0
    for row, i in enumerate(ids.tolist()):
        prim = stage.GetPrimAtPath(
            f"/World/envs/env_{i}/{entity_prim_name}/geometry/material/Shader"
        )
        if not prim.IsValid():
            continue
        attr = prim.GetAttribute("inputs:diffuseColor")
        if not attr:
            continue
        r, g, b = palette[int(pick[row])]
        attr.Set(Gf.Vec3f(float(r), float(g), float(b)))
        applied += 1
    # ★조용한 실패 방지★ — prim 경로가 어긋나면 색이 안 바뀌는데 ★오류도 안 나고 로그도 없다★.
    #   실제로 2026-08-11에 "경고가 없으니 적용됐겠지"라고 읽었다가, 팔레트의 파랑이 기존 색과
    #   거의 같아 눈으로도 구분이 안 돼 ★적용 여부를 확정하지 못했다★. ⇒ 항상 결과를 보고한다.
    counts = torch.bincount(pick, minlength=len(palette)).tolist()
    # env_0의 색을 함께 찍는다 — 스냅샷 카메라가 env_0을 담으므로 ★이미지와 로그를 대조★할 수 있다.
    _e0 = f"  env_0={tuple(round(v, 2) for v in palette[int(pick[0])])}" if ids.numel() else ""
    print(f"[dr] {entity_prim_name} 색 적용 {applied}/{ids.numel()} env — 팔레트별 {counts}{_e0}"
          + ("" if applied else "  ⚠️★0건: prim 경로 확인 필요"
             f" /World/envs/env_*/{entity_prim_name}/geometry/material/Shader★"))


def make_object_color_choice_events(
    num_objects: int = NUM_OBJECTS, mode: str = "startup"
) -> dict[str, EventTerm]:
    """큐브 색을 ★빨강/파랑 중 하나★로 (configs/sim.yaml scene.object_color_palette)."""
    palette = [tuple(float(v) for v in c) for c in _OBJECT_COLOR_PALETTE]
    return {
        f"color_choice_{name}": EventTerm(
            func=randomize_visual_color_choice,
            mode=mode,
            params={"entity_prim_name": f"Object_{i}", "palette": list(palette)},
        )
        for i, name in enumerate(object_names(num_objects))
    }


# 정적 자산 색 DR 대상: (events attr 이름, scene prim 이름) — scene.py _static_box와 일치.
_STATIC_COLOR_ENTITIES = (
    ("color_belt", "Belt"),
    ("color_belt_stand", "BeltStand"),
    ("color_robot_stand", "RobotStand"),
)


def make_static_color_events() -> dict[str, EventTerm]:
    """벨트/스탠드(정적 자산) 색 DR 항 묶음 — 이름은 _STATIC_COLOR_ENTITIES 고정.

    NuRec 파인튠 cfg가 이 '이름'으로 개별 무효화(None)한다 — 이름 변경 시 동기 필수.
    """
    colors = {"r": (0.05, 0.95), "g": (0.05, 0.95), "b": (0.05, 0.95)}
    return {
        attr: EventTerm(
            func=randomize_static_visual_color,
            mode="reset",
            params={"entity_prim_name": prim_name, "colors": dict(colors)},
        )
        for attr, prim_name in _STATIC_COLOR_ENTITIES
    }


def make_rgb_visual_color_terms(num_objects: int = NUM_OBJECTS) -> dict[str, EventTerm]:
    """★RGB-D 전환(2026-07-17 B안) 시각 색 DR 묶음★ — 큐브(stock) + 벨트/스탠드(커스텀).

    2026-07-15의 '색 DR 실효 0' 판정은 depth-only 관측 시대의 것 — rgb 관측 채택으로 반전.
    ⚠️ 큐브 항(color_object_*)은 upstream 클래스라 scene.replicate_physics=False 필수
      (events.py L1611-1616) — 부착측(BinPickVisionEnvCfg)이 함께 설정한다.
    """
    terms: dict[str, EventTerm] = {}
    if bool(_DR["lighting"].get("randomize_object_color", False)):
        for name in object_names(num_objects):
            terms[f"color_{name}"] = make_visual_color_event(name)
    terms.update(make_static_color_events())
    return terms


def make_visual_texture_event(object_name: str, texture_paths: list[str]) -> EventTerm:
    """객체 표면 텍스처 랜덤화 (upstream randomize_visual_texture_material, L1126).

    ⚠️ TODO: texture_paths에 로컬 텍스처 이미지 파일 목록이 필요하다 — 텍스처 팩
      확보 전에는 make_visual_color_event만 사용 (dr.yaml randomize_object_color).
    ⚠️ 전제: replicate_physics=False (위와 동일).
    """
    return EventTerm(
        func=mdp.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(object_name),
            "texture_paths": texture_paths,
            "event_name": f"texture_{object_name}",
            "texture_rotation": (0.0, 3.1416),
        },
    )


def resample_light_intensity(
    env,
    env_ids,
    intensity_scale_range: tuple[float, float],
    light_prim_path: str = "/World/light",
) -> None:
    """★커스텀★ 광량 랜덤화(2026-07-17 구현) — stock EventTerm 없음(2.3.2 재확인). dr.yaml lighting.

    ★전역 DR★: /World/light(DomeLight)는 모든 env가 공유하는 '전역 프림'이라 per-env
    개별화가 불가 — interval 모드(is_global_time=True)로 '전체 장면'의 광량을 주기 재샘플
    한다(event_manager.py L216-223: 전역 interval은 env_ids=None으로 호출됨 — 검증).
    env별 개별 광량이 필요해지면 라이트를 {ENV_REGEX_NS} 하위로 옮겨야 한다(렌더 비용 TODO).

    속성명 검증(2.3.2 vendored): DomeLightCfg → spawn_light가 cfg 필드를
    "inputs:<attr>"로 기록(lights.py L81-83, camel_case) → intensity = "inputs:intensity".
    베이스 광량은 하드코딩하지 않고 ★첫 호출 때 프림에서 읽어 캐시★ — 공칭값의 단일
    소스는 scene.py light cfg(intensity=3000.0) 하나로 유지된다.
    """
    stage = env.scene.stage  # interactive_scene.py L138 — 검증됨
    prim = stage.GetPrimAtPath(light_prim_path)
    if not prim.IsValid():
        if not getattr(env, "_bp_light_warned", False):
            env._bp_light_warned = True
            print(f"[dr][warn] 광량 DR: 라이트 프림 없음({light_prim_path}) — 스킵", flush=True)
        return
    attr = prim.GetAttribute("inputs:intensity")
    if not attr:
        if not getattr(env, "_bp_light_warned", False):
            env._bp_light_warned = True
            print(f"[dr][warn] 광량 DR: inputs:intensity 속성 없음({light_prim_path}) — 스킵",
                  flush=True)
        return
    base = getattr(env, "_bp_light_base_intensity", None)
    if base is None:
        base = float(attr.Get())  # 스폰 공칭값(3000 — scene.py) 캐시: 재샘플 누적 오염 방지
        env._bp_light_base_intensity = base
        print(f"[dr] ★광량 DR(전역)★ 베이스 {base:.0f} × U{tuple(intensity_scale_range)} "
              f"interval 재샘플 시작", flush=True)
    lo, hi = intensity_scale_range
    scale = float(torch.empty(1).uniform_(float(lo), float(hi)).item())
    attr.Set(base * scale)


def make_light_intensity_event(
    interval_range_s: tuple[float, float] = (4.0, 8.0)
) -> EventTerm:
    """광량 랜덤화 EventTerm (dr.yaml lighting.intensity_scale) — interval·전역.

    interval 근거: 전역 프림이라 'env별 리셋'과 동기화 자체가 불가 → 에피소드
    시간 스케일(EPISODE_LENGTH_S=8s의 0.5~1배)로 전체 장면을 주기 재샘플한다.
    cfg 필드 검증: EventTermCfg.interval_range_s(manager_term_cfg.py L269) /
    is_global_time(L280) — is_global_time=True면 단일 시계로 발화(event_manager.py L216).
    """
    lo, hi = _DR["lighting"]["intensity_scale"]
    return EventTerm(
        func=resample_light_intensity,
        mode="interval",
        interval_range_s=(float(interval_range_s[0]), float(interval_range_s[1])),
        is_global_time=True,  # ★전역 DR★ — 위 docstring
        params={"intensity_scale_range": (float(lo), float(hi))},
    )


# (④ camera_pose 섹션의 구 TODO 스텁 쌍은 2026-07-15 ①-e 구현으로 대체·삭제 —
#  ⚠️ 같은 이름의 뒤쪽 정의가 앞쪽 구현을 가리는 셰도잉 사고의 원인이었음.
#  구 스케치의 set_world_poses 방식은 eye-in-hand에선 부적합: WristCam은 링크
#  자식이라 world 포즈가 매 프레임 링크에 끌려가 리셋 섭동이 유지되지 않는다.
#  → USD 로컬 xformOp 섭동(startup 1회 = 마운트 공차 의미론)이 정답.)


# ═════════════════════════════════════════════════════════════════════════════
# ⑤ depth_noise 섹션 → 파라미터 재샘플 (노이즈 적용 자체는 observations.policy_obs)
# ═════════════════════════════════════════════════════════════════════════════
def resample_depth_noise_params(
    env,
    env_ids: torch.Tensor,
    sigma_base_m: float,
    sigma_scale_per_m2: float,
    dropout_prob_range: tuple[float, float],
) -> None:
    """Resample only reset environments, without changing ongoing episodes."""
    from common.noise_parameters import resample_episode_parameters

    resample_episode_parameters(
        env, "_binpick_depth_noise", env_ids,
        {"dropout_prob": tuple(float(v) for v in dropout_prob_range)},
    )
    if hasattr(env, "_binpick_depth_noise"):
        env._binpick_depth_noise.update(
            sigma_base_m=float(sigma_base_m), sigma_scale_per_m2=float(sigma_scale_per_m2)
        )


def make_depth_noise_resample_event() -> EventTerm:
    """depth 노이즈 파라미터 재샘플 EventTerm (dr.yaml depth_noise)."""
    dn = _DR["depth_noise"]
    return EventTerm(
        func=resample_depth_noise_params,
        mode="reset",
        params={
            "sigma_base_m": float(dn["sigma_base_m"]),
            "sigma_scale_per_m2": float(dn["sigma_scale_per_m2"]),
            "dropout_prob_range": tuple(float(v) for v in dn["dropout_prob"]),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# ⑤-b ★rgb 노이즈 재샘플 (2026-07-17 RGB-D 전환 B안)★ — depth_noise(⑤)와 동일 패턴
#   노이즈 '적용' 자체는 observations.policy_obs가 common.rgb_noise.apply_rgb_noise로
#   수행 — 여기서는 에피소드마다 파라미터(env 공통 스칼라)만 재샘플해 심는다.
# ═════════════════════════════════════════════════════════════════════════════
def resample_rgb_noise_params(
    env,
    env_ids,
    brightness_range: tuple[float, float],
    contrast_range: tuple[float, float],
    pixel_sigma_range: tuple[float, float],
) -> None:
    """Resample RGB jitter per reset environment, not per entire vector batch."""
    from common.noise_parameters import resample_episode_parameters

    resample_episode_parameters(env, "_binpick_rgb_noise", env_ids, {
        "brightness_scale": tuple(float(v) for v in brightness_range),
        "contrast_scale": tuple(float(v) for v in contrast_range),
        "pixel_sigma": tuple(float(v) for v in pixel_sigma_range),
    })


def make_rgb_noise_resample_event() -> EventTerm:
    """rgb 노이즈 파라미터 재샘플 EventTerm (dr.yaml rgb_noise)."""
    rn = _DR["rgb_noise"]
    return EventTerm(
        func=resample_rgb_noise_params,
        mode="reset",
        params={
            "brightness_range": tuple(float(v) for v in rn["brightness_scale"]),
            "contrast_range": tuple(float(v) for v in rn["contrast_scale"]),
            "pixel_sigma_range": tuple(float(v) for v in rn["pixel_sigma"]),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# 묶음 팩토리 — env cfg가 setattr로 events cfg에 부착
# ═════════════════════════════════════════════════════════════════════════════
def make_physics_dr_terms(num_objects: int = NUM_OBJECTS) -> dict[str, EventTerm]:
    """teacher/vision 공통 물리 DR 항 묶음 (dr.yaml physics)."""
    terms: dict[str, EventTerm] = {
        "actuator_gains": make_actuator_gains_event(),
        "finger_material": make_finger_material_event(),
    }
    for name in object_names(num_objects):
        terms[f"mass_{name}"] = make_object_mass_event(name)
        terms[f"material_{name}"] = make_object_material_event(name)
    return terms


def make_visual_dr_terms(num_objects: int = NUM_OBJECTS) -> dict[str, EventTerm]:
    """vision 전용 시각 DR 항 묶음 (dr.yaml lighting / camera_pose / depth_noise).

    ⚠️ 색/텍스처 항은 replicate_physics=False 필수 — env cfg가 함께 설정할 것.
    ⚠️ 2026-07-15부터 env cfg는 이 묶음을 부착하지 않는다(색/텍스처 = depth-only
      정책에 실효 0 판정). depth_noise_resample은 make_depth_noise_resample_event()
      로 개별 부착. 카메라 pose는 ①-e(구현 완료).
    ⚠️ 2026-07-17 RGB-D 전환 후 시각 DR 신규 부착은 make_rgb_visual_color_terms /
      make_rgb_noise_resample_event / make_light_intensity_event 개별 팩토리 사용 —
      이 legacy 묶음은 호환 보존용(env cfg 미부착 상태 유지).
    """
    terms: dict[str, EventTerm] = {"depth_noise_resample": make_depth_noise_resample_event()}
    if bool(_DR["lighting"].get("randomize_object_color", False)):
        for name in object_names(num_objects):
            terms[f"color_{name}"] = make_visual_color_event(name)
    return terms
