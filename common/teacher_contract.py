"""Phase1 Teacher 관측·행동의 순수 Python 계약 (★Teacher v2 — 배포 가능한 관측, 2026-09-18★).

Isaac Lab을 import하지 않아 Phase2 체크포인트 사전검사와 단위테스트에서 사용할 수 있다.
`PrivilegedObsCfg`의 항 순서나 각 항의 폭이 바뀌면 이 파일과 회귀테스트가 함께 실패해야 한다.

★v2 설계 원칙★: Teacher는 ★실기에서 만들 수 있는 신호만★ 관측한다.
  구 42D는 실제 손가락 위치(2)·물체 속도(6)·접촉력(2)을 특권으로 받았고, 실기 배포는 그것을
  '쥐었다' 판정에서 유도한 상수로 채웠다(모듈형 v1). 그 근사가 학습 분포와 달라 실기 성능을
  보장할 수 없었다. v2는 sim 학습 관측 자체를 배포 인식 모듈의 출력과 같은 규칙으로 만든다:
    gripper_command_state(1)  실제 개도가 아니라 명령 상태(student와 동일 계약)
    object_pos(3)             ★인식 추정치★ — sim은 mdp/perception_model.py가 실측 오차·검출
                              실패·카메라 기하(최소거리·FOV)를 모사, 실기는 depth 역투영
    held(1)                   폐합 전이 시점에 추정치가 TCP 반경 안이면 latch(sim/real 동일 규칙)
"""
from __future__ import annotations


TEACHER_OBS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("joint_pos", 6),
    ("joint_vel", 6),
    ("gripper_command_state", 1),
    ("tcp_pose", 7),
    ("target_pos", 3),
    ("object_pos", 3),
    ("held", 1),
    ("last_action", 7),
)

TEACHER_OBS_DIM: int = sum(width for _name, width in TEACHER_OBS_LAYOUT)
TEACHER_ACTION_DIM: int = 7

# ★'쥐었다' 판정·추정치 합성 상수 — sim(perception_model)과 real(teacher_obs)이 같은 값을 쓴다★
#   held_radius: 폐합 전이 시점에 큐브 추정치가 TCP에서 이 반경 안이면 쥐었다고 latch한다.
#   held_offset: 쥐고 있는데 안 보이면 큐브 중심 = TCP + 이 오프셋(base). 5 cm 큐브를 파지중심
#                에서 물면 중심이 TCP보다 약 2.7 cm 아래(Codex 루프 teacher 표본 |obj−tcp| p50).
#   closed_threshold: gripper_command_state(0=open … 1=closed)를 '폐합 명령'으로 읽는 문턱.
HELD_RADIUS_M: float = 0.045
HELD_OFFSET_FROM_TCP_M: tuple[float, float, float] = (0.0, 0.0, -0.027)
GRIPPER_CLOSED_THRESHOLD: float = 0.5


def teacher_obs_slice(term_name: str) -> slice:
    """지정 Teacher 관측 항의 flat slice를 반환한다."""
    offset = 0
    for name, width in TEACHER_OBS_LAYOUT:
        if name == term_name:
            return slice(offset, offset + width)
        offset += width
    raise KeyError(f"Teacher 관측 항 없음: {term_name}")


TEACHER_OBJECT_POS_SLICE: slice = teacher_obs_slice("object_pos")


if TEACHER_OBS_DIM != 34:  # 계약 변경 시 조용히 다른 Teacher를 로드하지 않도록 import 단계 차단.
    raise RuntimeError(f"Teacher 관측 계약은 현행 34D여야 합니다: {TEACHER_OBS_DIM}")
