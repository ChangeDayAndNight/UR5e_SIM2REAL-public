"""UR5e ArticulationCfg — 시뮬레이션 로봇 자산 정의.

목적: UR5e 팔 + OnRobot 2FG7 그리퍼의 ArticulationCfg 단일 소스.
      upstream UR10_CFG(isaaclab_assets/robots/universal_robots.py)를 템플릿으로
      각색 — ★isaaclab_assets에는 UR5e cfg가 없음(검증됨, 2.1.0)★.
실행: 직접 실행 없음 — scene.py / binpick_env_cfg.py가 import.
산출: UR5E_CFG (ArticulationCfg), UR5E_ARM_JOINT_NAMES, UR5E_HOME_JOINT_POS.

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import할 것.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 관절 이름/홈 포즈는 순수 상수 → common.robot_spec 단일 소스에서 재수출.
# (env cfg·관측·sim/sim_robot.py 어댑터가 계속 이 이름으로 import한다. sim_robot는
#  확장에 비의존하도록 common에서 직접 import — 값 어긋남 방지 단일 소스.)
from common.config_io import configs_dir
from common.robot_spec import UR5E_ARM_JOINT_NAMES, UR5E_HOME_JOINT_POS  # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
# USD 경로 — ★UR5e + Magbot TCW1 + OnRobot 2FG7 결합 자산 (2026-07-15 체인 정정)★
#   실기 체인 = tool0 ─fixed→ TCW1 툴체인저(Ø95×86.5mm, 1.162kg — magbot.kr 공식
#   스펙 웹 검증) ─fixed→ 2FG7 ─prismatic×2→ fingers. ★TCP +86.5mm 전방 이동★
#   + D435i 브라켓·카메라 본체(시각)도 URDF에 내장(자기완결 — GUI에서 전체 보임).
#   assets/ur5e_2fg7/ur5e_tcw1_2fg7.urdf 를 convert_2fg7_usd.py --fix-base 로 변환.
#   (구 ur5e_2fg7.usd = TCW1 부재 구성 — 2026-07-15 이전 학습에만 유효, 참조 중단.)
#   ⚠️ 이 USD가 없으면(변환 미실행) 기동이 죽는다 → README.md §7 절차.
# ─────────────────────────────────────────────────────────────────────────────
UR5E_USD_PATH = str(configs_dir().parent / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.usd")

# ─────────────────────────────────────────────────────────────────────────────
# OnRobot 2FG7 그리퍼 스트로크 — ★STL 메시 검증(2026-07-06 Grasp_PhysicsTest)★
#   left_finger_joint ∈ [0, 0.020] m (right 1:1 대칭). 현행 CAD 검증 inner gap:
#     ★q=0.0 → 74mm(열림/release), q=0.020 → 34mm(닫힘/grip)★
#   (이전엔 반대로 배선 → 정책의 '닫기'가 실제로 열기라 큐브를 못 잡고 '밀기'를
#    학습. STL 정점 bound로 확정 후 정정.) ★현 큐브 = 5cm(sim.yaml object_size_m=0.05)★.
# ─────────────────────────────────────────────────────────────────────────────
# ★파지 가능 폭 상한(2026-07-29 신설)★ — q=0(완전 열림)에서의 핑거 inner face 간격.
#   출처: README.md §7 — STL 정점 bound 검증(2026-07-06).
#   ⚠️★이 값보다 큰 물체는 그리퍼가 벌어지지 않아 원천적으로 못 집는다★.
#     sim.yaml object_size_m을 키울 때 반드시 이 상한과 대조할 것
#     (tests/test_gripper_capacity.py가 자동으로 강제한다).
#   ※ 실물 2FG7 데이터시트는 외측 파지 35~73mm를 지원하지만, 그건 ★핑거팁을 바깥 위치에
#     장착했을 때★다. 현재 URDF/STL은 6~44mm를 주는 안쪽 위치를 모델링하고 있다 —
#     더 큰 물체가 필요하면 핑거팁 실장 위치 확인 → URDF 오프셋 수정 → tcp_offset 재측정
#     → 홈 FK·_PREGRASP_GRIP 재산출이 함께 따라와야 한다.
#   ★2026-07-30 CAD 도면 확정(2fg7.png)★ 폐합 안쪽 34 / 바깥 44 · 열림 안쪽 74 / 바깥 84mm →
#     핑거팁 두께 = (44−34)/2 = ★5mm★, 스트로크 = (74−34)/2 = ★20mm/편측★.
#     구 자산(STL)은 안쪽 44.6/6.6mm·두께 19.2mm — STL은 브래킷 포함 단순화 블록이라
#     실제 접촉면 위치와 달랐다. joint origin을 좌우 ★14.70mm★ 바깥으로 이동 → 74.0/34.0mm.
#     ⚠️2026-07-29에 '실측 80/40mm'을 안쪽 간격으로 오독해 17.7mm 옮겼던 것을 정정(F32).
#     ⚠️콜라이더 두께는 STL의 19.2mm를 유지한다 — CAD 실제 두께 5mm로 하면 50mm 큐브의
#       침투 8mm가 두께를 넘어 ★관통★한다. 뒷면은 아무것과도 닿지 않아 무해하다.
GRIPPER_MAX_OPEN_GAP_M: float = 0.074

# ★현행 열림 목표★: q=0.0, inner gap 74mm. 5cm 물체에 양측 합 24mm 여유가 있어
#   접근·release가 가능하다. 과거 q=0.010 부분개방 실험 설명은 현행 상수와 무관하다.
GRIPPER_OPEN_M: float = 0.0
# ★현행 URDF joint upper와 일치: close=0.020(완전폐합)★. 0.011(얕은 파지)은 접촉력 부족으로
#   성공률 0%였고, 0.020은 현행 5cm 큐브를 양측 8mm씩 물어 접촉력을 만든다.
#   ★핵심은 '깊은 목표 + 완만한 접근'★: velocity_limit_sim으로 폐합속도를 낮춰 깊은 파지가 급격한
#   depenetration 킥 없이 정착하게 한다(무지지 큐브 이탈 완화). 실 2FG7 force-close와 의미 정합.
GRIPPER_CLOSE_M: float = 0.020

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR5E_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # ★2026-07-09 팔 중력 disable (task-space IK 표준 — Franka HIGH_PD_CFG와 동일)★
            #   task-space DiffIK(ik_rel)로 전환하며 팔 링크 중력을 끈다. 이유: 실 UR servoJ/servoL은
            #   중력을 하드웨어에서 보상해 명령 자세를 유지하지만, sim implicit PD는 중력보상을 못해
            #   sag → holding 실패(상대 액션 grasp100%/hold2.9% 확정). 중력을 끄면 PD가 IK 관절목표를
            #   유지 → holding 성립 + sim2real net 거동 정합(Franka도 IK/HIGH_PD에서 disable_gravity=True
            #   사용 — franka.py L79). ⚠️ 객체(RigidObject)는 중력 유지 → 파지 물체는 그립력으로 잡음.
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        # ★2026-07-09 파지 관통 근본 수정 — 솔버 반복수 설정 (요소: gripper 물리)★
        #   과거 증상: 구 자산에서 finger를 close(0.019 target)로 눌러도 30mm 큐브를 관통(gate_probe: q→0.017,
        #   접촉력 ~0)하고 큐브가 튕겨 낙하 → grasp 게이트 상시 0. 원인 확정: UR5E_CFG에
        #   articulation_props 미설정 → 솔버 position iteration이 USD 기본(저값)이라, 강한 finger
        #   드라이브(stiffness 4000, ~48N)가 경량 큐브(30g) 접촉을 수렴 전에 관통. Franka
        #   (franka.py L33-34)는 solver_position_iteration_count=8을 명시 → 접촉이 큐브를 막아 그립 형성.
        #   우리 그립 강성이 Franka(2000)의 2배라 관통이 더 쉬움 → 16으로 상향(보수적).
        #   enabled_self_collisions=False: 큐브는 외부 rigid라 자기충돌 불필요(팔-팔 오접촉 회피).
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        # ★2026-07-09 STAGE 3: True로 전환★ — 프로브가 '들림 중 접촉 0%'(순수 저글링 해킹)를
        # 확정, 파지 게이트 보상(binpick_env_cfg._grasped_mask)이 손가락 ContactSensor를 요구.
        # (미설정 시 센서 기동이 RuntimeError — contact_sensor.py L266-268 검증.)
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 팔 = 홈 포즈, 그리퍼 = 열림(release) 시작 — 파지 전 상태.
        joint_pos={
            **dict(UR5E_HOME_JOINT_POS),
            "left_finger_joint": GRIPPER_OPEN_M,
            "right_finger_joint": GRIPPER_OPEN_M,
        },
        # ★가정: 로봇 base = env origin(identity pose)★ — hand_eye.yaml T_base_cam을
        # 카메라 배치에 그대로 쓰는 전제 조건 (scene.py 요소4 참조).
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            # ★finger 관절과 분리★ — 팔 6관절만. 이전 ".*"는 그리퍼까지 잡아 충돌하므로
            #   결합 자산에서는 명시적 팔 관절명으로 한정한다.
            joint_names_expr=list(UR5E_ARM_JOINT_NAMES),
            # ⚠️ TODO(Phase3 SysID로 보정): 아래는 UR10_CFG(800/40) 대비 상향값.
            #   ★2026-07-08 sag 완화★: RelativeJointPositionAction이 current 기준
            #   (joint_actions.py:191)이라 delta=0에서 복원력 0 → 팔이 중력에 처짐(sag).
            #   실 servoJ는 목표 유지(중력보상)하므로 sim도 유지해야 정합 + hold/운반 학습이 쉬움.
            #   sag 종말속도 ∝ 1/damping이라 ★damping을 크게★(40→150) 올려 sag를 늦춘다.
            #   stiffness도 상향(800→1500)해 delta 추종 응답 확보. (과하면 진동 — 실측서 조정.)
            # Isaac Lab 2.3.2/Isaac Sim 5.1의 ImplicitActuator는 PhysX 한계를 *_sim
            # 필드로 설정한다. legacy velocity_limit/effort_limit는 사용하지 않는다.
            velocity_limit_sim=3.1416,  # π rad/s(보수적)
            effort_limit_sim=150.0,  # TODO: UR5e 관절별 토크(150/150/150/28/28/28 Nm 추정)
            stiffness=1500.0,
            damping=150.0,
        ),
        "gripper": ImplicitActuatorCfg(
            # OnRobot 2FG7 finger 2관절 (URDF prismatic ★독립 2관절★ — F91, 2026-09-20).
            # BinaryJointPositionAction이 open/close 목표를 이 관절들에 기록한다.
            # ★★F91 사고★★: 종전 URDF의 right_finger_joint <mimic> 태그가 Isaac Sim 5.1 변환에서
            #   "독립 관절 변환" 옵션에도 불구하고 ★드라이브 없는 PhysX mimic API(gearing −1, rotX
            #   슬롯, 한계 [−0.004, 0.024])★로 남아 오른쪽 손가락이 수동 관절이 됐다. 왼쪽만 닫히고
            #   오른쪽은 큐브에 밀려 −4 mm 하한에 걸리는 편측 파지를 teacher가 학습했다(사용자 영상,
            #   v1 조립기 실측 L 12.6 / R −4.0 mm). 실물 2FG7은 양측 대칭이므로 mimic 태그를 제거하고
            #   한계 [0, 0.020]으로 재변환했다 — diagnostics/gripper_symmetry_probe.py가 대칭을 검증한다.
            joint_names_expr=["left_finger_joint", "right_finger_joint"],
            # ★2026-07-10 스냅-관통 수정★: ImplicitActuator는 velocity_limit를 무시하고
            #   velocity_limit_sim만 PhysX joint maxJointVelocity로 적용한다(actuator_pd.py L79-100).
            #   미설정 시 강성 4000·손가락 23g → 가속 ~2000m/s²로 한 substep(8ms)에 완전폐합 →
            #   접촉 솔버가 반응하기 전에 큐브를 관통·이탈(포렌식·렌더 확정). 실 2FG7 폐합속도
            #   수준(0.15m/s)으로 캡 → 손가락이 큐브면에서 막혀 그립력 형성. stiffness 4000→2000(Franka값).
            # ★velocity_limit는 설정 금지★ — ImplicitActuatorCfg는 velocity_limit_sim과 다른 값이면
            #   ValueError. velocity_limit_sim만 지정한다(PhysX joint maxJointVelocity).
            # ★★2026-08-11 0.05 → 0.15 (실물 2FG7 사양 정합 + 탐색 장벽 제거)★★
            #   [사고] 바로 위 주석이 스스로 "실 2FG7 폐합속도 수준(0.15m/s)으로 캡"이라 적어
            #     놓고 값은 ★0.05★(3배 느림)로 박혀 있었다. 이 3배가 릴리스 학습을 불가능하게 했다:
            #     20mm 개방에 0.40s = ★8 스텝 연속★ open 명령이 필요한데, 수렴한 정책의
            #     액션 분포(μ=−0.249, σ=0.100)에서 8연속 확률은 2.8e-18 —
            #     ★25k iteration 예산 전체의 기대 발생이 2.2e-8회★(사실상 0)다.
            #     세 번의 재학습 실패(F50/F52/F53 수정 후에도 placed 0.2%)가 전부 여기서 설명된다.
            #   [정정] 실물값 0.15로 되돌린다 → 개방 3 스텝, 25k 기대 발생 2e3회로 학습 가능해진다.
            #   ⚠️구 0.05의 목적(폐합 시 스냅-관통 방지)은 ★핑거팁 두께 10mm 정정(P10)★과
            #     solver_position_iteration_count=16로 이미 완화됐다 — 파지 스모크로 검증한다.
            # ★★2026-08-11 0.15 시도 → ★기각, 0.05 유지★ (통제 A/B 실측)★★
            #   [가설] 개방 속도가 릴리스 탐색을 막는다 → 실물값 0.15로 올리자.
            #   [기각 근거 ①] 통제 A/B(속도만 변경, 나머지 동일, 60iter×256env):
            #     iter59 grasp 0.0139(0.15) vs ★0.0427(0.05)★ · lift 0.0332 vs ★0.1010★
            #     ⇒ 속도를 3배 올리면 파지·들림이 ★1/3로 붕괴★한다(폐합 스냅-관통, 구 0.05의 원래 이유).
            #   [기각 근거 ② — 가설 자체가 틀렸다] '8스텝 연속 open 필요'는 ★|q|≤3mm(완전 개방)★
            #     기준이었는데 그 조건은 F50에서 성공 게이트에서 제거됐다. 현재 릴리스 판정은
            #     ¬접촉 단독이고, 큐브를 문 손가락(q=12mm)은 ★1~2mm만 열려도 접촉이 끊긴다★
            #     → 실제 장벽은 ★1~2 스텝★(25k 기대 발생 3e5~5e7회)이라 애초에 병목이 아니었다.
            #   ⇒ 릴리스 미학습의 실제 원인은 속도가 아니라 F52/F53의 보상 farming이었다.
            velocity_limit_sim=0.05,      # ★PhysX 개폐속도 상한 [m/s]★ — 폐합 안정성 우선(A/B 실측)
            effort_limit_sim=140.0,       # 2FG7 파지력 [N]
            # ★★2026-07-29 접촉력 재보정(F31) — 구 2000★★
            #   접촉력 ≈ 강성 × 편측침투. 그리퍼 실측 반영으로 폐합 간격이 6.6 → 42mm가 되면서
            #   같은 큐브라도 파고들 깊이가 급감했다:
            #     구 계보 3cm: 편측 11.7mm × 2000 = 23.4N  (반너비의 78% — 거의 파묻힘)
            #     현행   5cm: 편측  4.0mm × 2000 =  8.0N  (16%) → ★파지력 1/3★
            #   ★2026-07-30 CAD 정정 반영★ 폐합 34mm(구 42mm) → 50mm 큐브 침투 4.0 → ★8.0mm★.
            #   → 23.4N을 맞추는 강성 = 23.4/0.008 = 2925 ≈ ★3000★ (구 6000은 침투 4mm 기준).
            #   ★학습 스모크 실측(256env×50iter, 결정론적 — 2회 반복 동일)★:
            #     2000→0.0080  4000→0.0154  ★6000→0.0178★  8000→0.0103  12000→0.0113
            #     20000→0.0105  35000→0.0080   ← 명확한 봉우리, 계산값과 일치
            #   ⚠️큐브/그리퍼 기하를 바꾸면 ★이 값을 반드시 재계산★할 것
            #     (tests/test_grip_force_contract.py가 강제한다). 6cm면 2600이 맞는 값이다.
            #   ⚠️실물 2FG7은 최대 140N — sim 8N은 실물을 크게 과소 표현하던 것이므로
            #     강성 상향은 실물에 가까워지는 방향이다.
            stiffness=3000,             # 위치 강성(Franka panda_hand 값) — 그립력·안정성 균형
            damping=100.0,
        ),
    },
)
"""UR5e ArticulationCfg — UR10_CFG 템플릿 각색 (그리퍼 미포함)."""

# ═════════════════════════════════════════════════════════════════════════════
# GRIPPER 배선 — OnRobot 2FG7 (확정 2026-07-06). assets/ur5e_2fg7 결합 USD 사용.
#   BinaryJointPositionAction이 open/close 목표를 두 finger 관절에 대칭 기록한다
#   (정책 action[-1] > gripper_close_threshold → close). 실기 계약 = common/action_mapping.py.
#   잔여 TODO (sim 확인 후 refine):
#     - scene.py ee_frame: 현재 wrist_3_link 타깃 — 2FG7 grasp 중심으로
#       오프셋(tcp_offset.yaml) 재확인. reach 보상엔 지금도 충분(dense shaping).
#     - make_finger_contact_sensor_cfg prim "{ENV}/Robot/.*finger.*" 매칭 확인
#       (privileged 접촉력 활성화 시 + UR5E_CFG.spawn.activate_contact_sensors=True).
# ═════════════════════════════════════════════════════════════════════════════
GRIPPER_JOINT_NAMES: list[str] = ["left_finger_joint", "right_finger_joint"]
GRIPPER_OPEN_COMMAND: dict[str, float] = {jn: GRIPPER_OPEN_M for jn in GRIPPER_JOINT_NAMES}
GRIPPER_CLOSE_COMMAND: dict[str, float] = {jn: GRIPPER_CLOSE_M for jn in GRIPPER_JOINT_NAMES}
