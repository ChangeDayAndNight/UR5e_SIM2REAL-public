"""UR5e 로봇 스펙 상수 — 순수 모듈 (isaaclab/SDK import 없음, §4-2 순수성).

sim(env cfg·관측·`sim/sim_robot.py` 어댑터)과 real이 공유하는 **관절 순서·홈 포즈 계약**.
확장(sim)과 배포 번들(sim_robot 어댑터·real) 양쪽에서 동일하게 import되도록 common에 둔다.
stdlib list/dict만 사용.
"""

from __future__ import annotations

# UR 시리즈 표준 관절명 — 순서 = 관측/액션 계약. UR5e USD 확정 후 재검증할 것.
UR5E_ARM_JOINT_NAMES: list[str] = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ⚠️ configs/real.example.yaml robot.home_joints_rad와 값 동기 유지하고, 실기 확정값은
#    configs/real.local.yaml에도 함께 반영한다.
UR5E_HOME_JOINT_POS: dict[str, float] = {
    # ★★2026-07-28 홈 포즈 교체 — '카메라가 작업면을 보는 자세'(사용자 GUI 확정)★★
    #   근거: 구 홈(-90/+90/-90)은 손목 카메라가 벨트 far edge를 넘겨 봐서 ★큐브 스폰 분포의
    #   27.1%만 시야에 들어왔다★(렌더 실측: 벨트 중심에 놓인 큐브조차 프레임 밖 —
    #   Phase2/Result/diagnostics/home_pose_probe/). teacher는 특권 상태라 무영향이지만
    #   student(비전)는 에피소드의 73%를 '아무것도 안 보이는' 상태로 시작한다.
    #   이 프로젝트는 같은 기전으로 이미 실패한 이력이 있다 — 홈 가시율 18.3%일 때 student
    #   eval grasp 0.8%(접근 불능), FOV 확대(frame_mode=full)로 복구(configs/common.yaml 주석).
    #   → 사용자가 place_workspace_gui.py의 실시간 카메라·가시율을 보며 확정한 자세로 교체.
    #   ★새 홈 가시율 = 스폰 분포 100.0% · 타겟존 100.0%★ (링크 최저 z=0.163m로 벨트 간섭 없음)
    #   ⚠️ 홈을 바꾸면 events_dr._PREGRASP_GRIP(홈 파지중심)이 즉시 무효 → 함께 갱신했다
    #   (FK 산출: 2026-07-30 툴 체인 5단 정정 후 (−0.2975, −0.1328, 0.5091);
    #   tests/test_urdf_fk.py 골든이 둘의 정합을 강제).
    #   ⚠️ farming 주의: 과거 "reaching-over"(-75/+80/-95)는 보상 지형을 바꿔 파지 대신 placed
    #   farming을 유발했다(grasp 0.01 평탄). 이번 자세는 TCP가 타겟존(z=0.015)에서 52cm 위라
    #   같은 함정은 아니나, ★재학습 시 iter 300~600 grasp 조기 게이트로 감시할 것★.
    #   (구 검증 홈: 0 / -1.5708 / +1.5708 / -1.5708 / -1.5708 / 0 — 99% 달성 reference 자세)
    #   ★★2026-07-28(2차) 배치 180° — shoulder_pan 0 → π★★: 실셀에서 로봇이 벨트를 등지고
    #   서므로(벨트 = base frame −x, sim.yaml belt_center_xy_m[-0.45,0]) 팔 전체를 z축 180° 돌린다.
    #   ★나머지 5관절은 불변★ — 어깨 pan만 돌리면 팔 형상·손목 카메라의 작업면 상대 시야가
    #   완전히 동일하게 보존된다(가시율 100% 유지). FK 검증: 파지중심 (0.3012,0.1328,0.525) →
    #   (−0.3012,−0.1328,0.525), 180° 회전 예측과 ★오차 0.0mm★. UR5e pan 가동범위 ±360°라 여유.
    # ★★2026-08-30(3차) 실기 실측 자세로 교체 — 보호정지 대응★★
    #   [사고] sim에서 아무 문제 없던 기동이 실기 배포에서 ★보호정지★를 반복했다.
    #     명령 경로는 servoL(직교 서보)이라 컨트롤러가 내부에서 IK를 푼다 — 팔 ★형상★이
    #     sim과 다르면 같은 직교 델타가 전혀 다른 관절 운동이 되고, 특이점 근처에서
    #     관절속도가 폭주해 컨트롤러가 보호정지를 건다.
    #   [조치] 사용자가 실기에서 실제로 안정적으로 쓰는 자세를 sim의 단일 소스로 삼는다.
    #     구 자세 (180/-90/+45/-32/-90/0) → TCP (-0.2975, -0.1328, 0.5091)
    #     신 자세 (0/-76/-103/-96/+93/+4) → TCP (-0.3588, +0.1159, 0.2327)
    #     두 TCP 거리 37.7cm, y 부호 반전, z 28cm 하강 — ★완전히 다른 팔 형상★이다.
    #   ⚠️ 이 값으로 학습된 정책은 아직 없다. 기존 checkpoint는 구 홈 시야로 학습됐으므로
    #     ★재학습해야 이 자세의 시야와 정합★한다. 실기 즉시 구동에는 영향이 없다
    #     (Phase4/Run_Task.py는 홈으로 보내지 않고 현재 자세에서 시작한다).
    "shoulder_pan_joint": 0.0,           # 0°
    "shoulder_lift_joint": -1.3265,      # -76°
    "elbow_joint": -1.7977,              # -103°
    "wrist_1_joint": -1.6755,            # -96°
    "wrist_2_joint": 1.6232,             # +93°
    "wrist_3_joint": 0.0698,             # +4°
}

# ═════════════════════════════════════════════════════════════════════════════
# ★그리퍼 finger joint 물리/Teacher privileged 계약★
#   OnRobot 2FG7(결합 URDF): left_finger_joint ∈ [0, 0.020] m (right 1:1 대칭).
#   q=0.0 → inner gap 74mm(열림) / q=0.020 → 34mm(완전폐합). 5cm 큐브 파지 시
#   q가 접촉 위치에서 멈춘다. Teacher의 privileged 2D 관측과 물리 자산이 이 값을 쓴다.
#   ⚠️ 사본 동기화: sim 확장 mdp/ur5e_assets.py GRIPPER_CLOSE_M와 값 일치 유지
#   (확장이 common을 import하는 방향은 허용되나 기존 동작 파일 무수정 원칙으로 사본 유지).
#   Student/real RobotState는 이 값을 관측에 쓰지 않는다. 현재 DO 배선에는 실제 폭
#   피드백이 없어 별도 gripper_command_state 계약을 사용한다(common/gripper_command.py).
GRIPPER_JOINT_RANGE_M: float = 0.020

# ISO/UR geometry: shoulder-to-wrist reach used by the canonical Phase 3
# target-feasibility receipt (upper arm 0.425 m + forearm 0.3922 m).
UR5E_MAX_REACH_M: float = 0.8172
