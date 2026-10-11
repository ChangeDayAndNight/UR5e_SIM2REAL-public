"""파지·릴리스 수퍼바이저 — ★sim 학습과 실기 배포가 문자 그대로 같은 판정을 쓴다★.

담당 범위는 둘이다.
  · 릴리스(F66): 언제 놓을지를 ★수퍼바이저가 실행★한다 — 정책은 어디에 놓을지만 정한다.
  · 폐합(F68):   언제 닫을지는 ★정책이 정하고★ 수퍼바이저는 ★높이만 제약★한다.
    (폐합까지 탈취한 F67은 teacher 학습을 F66의 20%로 망가뜨려 기각됐다 — grasp_allowed 참조.)

왜 이 모듈이 존재하는가 (2026-08-16, F65 이후 실측 근거)
────────────────────────────────────────────────────────────────────────────
`BinaryJointPositionAction`은 액션의 ★부호만★ 본다. 그래서 그리퍼 차원은 크기에
gradient가 없고, 어떤 형태의 지급을 붙여도 정책이 경계에서 수확한다(F49·F52·F53·
F56·F57·F64가 이 한 항의 상수를 다섯 번 고쳐 모두 실패). F65에서 지급을 아예 없애자
이번엔 액션이 깊은 음수로 표류했다(rms|a| 1.39 → 2.22, 성공률 0.05%).

프로브 실측(model_6000, 각 256 에피소드, 성공종료 비활성, ≥10스텝 = 배포 등급):

    policy(정책이 스스로 개방)                     0.0%
    타겟존 진입 시 강제 개방 (팔은 계속 이동)      11.7%
    ★강제 개방 + 팔 정지★                        63.7%   (트리거가 큐브 pose = sim 전용)
    ★TCP 정지 기반 트리거 + 팔 정지★             66.8%   (트리거가 TCP = 배포 가능)

⇒ 릴리스는 ★학습시킬 대상이 아니라 실행할 primitive★다. 이것이 Isaac Sim 공식
   pick&place 튜토리얼(phase 6 Release)과 ManiSkill3·대부분 실기 시스템의 구조다.

트리거 설계 — 왜 'TCP 정지'인가
────────────────────────────────────────────────────────────────────────────
'큐브가 타겟 위인가'를 묻는 트리거는 sim에서 63.7%로 잘 작동하지만 ★실기에서 쓸 수
없다★ — 큐브 pose 추정기가 구현·검증된 바 없다. TCP xy로 대체하면 파지 오프셋 때문에
24.2%로 무너진다(TCP_xy ≠ cube_xy).

그래서 이 수퍼바이저는 큐브 위치를 ★묻지 않는다★. 대신 ★정책이 배치를 마치고
멈췄는가★를 묻는다. 어디에 놓을지는 정책이 정한다 — 정책은 target_pos를 관측하고
큐브-타겟 거리로 보상받으므로, 큐브가 타겟 위일 때 멈추도록 학습된다. 필요한 신호는
TCP pose와 TCP 속도뿐이고 둘 다 실기에서 RTDE/FK로 얻는다.

⚠️ 이 모듈은 상태를 가진다(에피소드별 latch). 호출측이 리셋에서 `reset()`을 불러야
   한다. 상태를 env에 두지 않는 이유: 같은 객체를 실기 rollout도 쓰기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config_io import configs_dir, load_yaml

__all__ = [
    "ReleaseSupervisorCfg",
    "load_release_supervisor_cfg",
    "carry_latch_update",
    "edge_lockout_update",
    "grasp_allowed",
    "release_decision",
    "settle_counter_update",
]


@dataclass(frozen=True)
class ReleaseSupervisorCfg:
    """릴리스 판정 상수 — configs/common.yaml `release_supervisor` 단일 소스."""

    lift_rise_m: float       # ★F68-c★ 폐합 시점 대비 이만큼 상승하면 '운반 중' latch [m]
    tcp_z_max_m: float       # 개방 허용 TCP 높이 상한 [m] — 배치 높이까지 내려왔는가
    tcp_settle_mps: float    # 개방 허용 TCP 속도 상한 [m/s] — 정책이 멈췄는가
    grasp_z_max_m: float     # ★F67★ 폐합 허용 TCP 높이 상한 [m] — 파지 높이까지 내려왔는가
    tcp_settle_steps: int    # ★F92★ 연속 정지 창 수 — 이만큼 연속으로 느려야 '멈췄다'
    gripper_edge_lockout_steps: int  # ★F92★ 그리퍼 전이 뒤 반전을 무시하는 스텝 수(실기 펄스 길이)

    def __post_init__(self) -> None:
        if not (self.lift_rise_m > 0.0):
            raise ValueError("lift_rise_m은 양수여야 한다 — 폐합 뒤 실제로 올라와야 운반이다")
        if not (self.tcp_settle_mps > 0.0):
            raise ValueError("tcp_settle_mps는 양수여야 한다")
        if not (self.grasp_z_max_m > 0.0):
            raise ValueError("grasp_z_max_m은 양수여야 한다")
        if int(self.tcp_settle_steps) != self.tcp_settle_steps or self.tcp_settle_steps < 1:
            raise ValueError("tcp_settle_steps는 1 이상의 정수여야 한다")
        if (int(self.gripper_edge_lockout_steps) != self.gripper_edge_lockout_steps
                or self.gripper_edge_lockout_steps < 0):
            raise ValueError("gripper_edge_lockout_steps는 0 이상의 정수여야 한다")


def load_release_supervisor_cfg(path=None) -> ReleaseSupervisorCfg:
    """configs/common.yaml에서 판정 상수를 읽는다 (sim/real 단일 소스)."""
    cfg = load_yaml(path or (configs_dir() / "common.yaml"))["release_supervisor"]
    return ReleaseSupervisorCfg(
        lift_rise_m=float(cfg["lift_rise_m"]),
        tcp_z_max_m=float(cfg["tcp_z_max_m"]),
        tcp_settle_mps=float(cfg["tcp_settle_mps"]),
        grasp_z_max_m=float(cfg["grasp_z_max_m"]),
        tcp_settle_steps=int(cfg["tcp_settle_steps"]),
        gripper_edge_lockout_steps=int(cfg["gripper_edge_lockout_steps"]),
    )


def _select(cond, when_true, when_false):
    """텐서(sim)와 파이썬 스칼라(실기) 양쪽에서 동작하는 where."""
    if isinstance(cond, (bool, int)) or type(cond).__module__ == "numpy":
        return when_true if bool(cond) else when_false
    import torch  # sim 경로에서만 필요 — 실기 스칼라 경로는 torch 없이 동작

    return torch.where(cond, when_true, when_false)


def edge_lockout_update(closed_prev, closed_want, since_edge, cfg: ReleaseSupervisorCfg):
    """★F92(2026-09-20)★ 그리퍼 전이 잠금 — 전이 뒤 `gripper_edge_lockout_steps` 동안 반전을 무시한다.

    [왜] 실기 2FG7 DO-pulse 드라이버는 펄스(0.30 s = 6 스텝)가 살아 있는 동안 반대 핀을 요구하면
      ★e-stop★으로 처리한다(real_robot.set_gripper — 펄스 중첩은 하드웨어 거동이 미정의).
      sim에 같은 잠금이 없으면 정책이 결정 경계에서 부호를 떨어도 sim은 아무 대가가 없고,
      실기는 첫 반전에서 즉시 중단된다. 잠금을 ★공통 판정★에 넣어 정책이 잠금 아래에서
      학습하게 하면 드라이버의 e-stop 분기는 진짜 고장 경로로만 남는다.
    [의미] 반환 (closed_now, since_edge_new). 잠금 중(since_edge < lockout)이면 closed_prev를 유지하고,
      전이가 일어나면 since_edge를 0으로, 아니면 +1. 전이 스텝 다음의 lockout 스텝이 잠기므로 전이 간
      최소 간격은 lockout+1 스텝(6 → 350 ms > 펄스 0.30 s — 펄스 종료 직후 스텝도 보수적으로 잠근다).
      잠금은 ★절대적★이다 — 릴리스 개방도 예외가 아니다(실기 펄스 보호가 목적이므로).
      리셋에서 since_edge는 잠금 길이로 초기화한다(첫 전이 허용).
    """
    locked = since_edge < cfg.gripper_edge_lockout_steps
    closed_now = _select(locked, closed_prev, closed_want)
    edge = closed_now != closed_prev
    since_edge_new = _select(edge, since_edge * 0, since_edge + 1)
    return closed_now, since_edge_new


def settle_counter_update(settle_count, tcp_speed, cfg: ReleaseSupervisorCfg):
    """★F92★ 연속 정지 창 카운터 — 느린 창이면 +1, 아니면 0으로 되돌린다.

    [왜 단일 창이 아닌가] Teacher v2에서 스텝당 최대 변위는 12.5 mm이고 정지 임계는 1 mm(0.02 m/s
      × 50 ms)다. 운반 전 구간의 TCP z(p99 0.062)가 tcp_z_max(0.075) 아래라 높이는 가드가 아니고,
      속도 창 ★하나★가 릴리스를 비가역으로 굳힌다. 실기 servoL(lookahead 0.1 s)은 방향 전환에서
      창 하나가 1 mm 아래로 떨어질 수 있다 → 운반 중 낙하. 연속 N창(150 ms)을 요구하면 정책이
      실제로 멈춘 경우만 통과한다. 호출측은 이 값을 `release_decision`에 넘긴다.
    """
    return (settle_count + 1) * (tcp_speed < cfg.tcp_settle_mps)


def grasp_allowed(tcp_z, cfg: ReleaseSupervisorCfg):
    """★F68(2026-08-18)★ 이 높이에서 폐합을 ★허용★하는가 — 제약이지 탈취가 아니다.

    ★F67의 오진과 정정★
      F67은 폐합 시점 자체를 수퍼바이저가 정하게 했다(정책의 그리퍼 차원을 무시). 근거는
      "BinaryJointPositionAction은 한번 닫으면 유지되므로 조기 폐합을 되돌릴 수 없다"였는데
      ★이 전제가 사실이 아니었다★. 업스트림 소스(binary_joint_actions.py:131-142)는 매 스텝
      부호로 명령을 다시 계산한다(`binary_mask = actions < 0` → where(mask, close, open)) —
      latch가 없고 정책이 양수를 내면 즉시 다시 열린다.

      그 잘못된 전제 위에서 만든 F67은 teacher 학습을 크게 망가뜨렸다(실측 grasp):
        구간        F67   F67-c   F66
        0-250     0.028  0.031  0.797
        500-750   0.109  0.129  1.696
        1500-1750 0.370  0.338  1.530
      '내려가서 ★멈춰야★ 폐합'이라는 하드 전제조건이 탐색을 어렵게 만들어 F66의 20% 수준에
      머물렀고, 재시도 허용(F67-c)으로도 달라지지 않았다 = latch는 원인이 아니었다.

    ★실제 기전★
      student는 접근 내내 그리퍼를 닫은 채로 간다(close_bias 0.802 vs teacher 0.134).
      닫힌 그리퍼는 내경 34mm이고 큐브는 50mm라 ★감쌀 수가 없어 밀어낸다★. 그래서 근접은
      teacher보다 잘하면서(reach 0.734 vs 0.477) 실제 파지는 22%에 그친다(전환율 27배 열위).

    ⇒ 필요한 것은 폐합 ★시점의 탈취★가 아니라 ★높이 제약★이다. 언제 닫을지는 정책이
      정하되(teacher는 큐브에 닿아서 닫으므로 제약이 걸리지 않는다 = 학습성 보존),
      파지 높이 위에서는 닫을 수 없게 한다(student의 '닫은 채 접근'이 구조적으로 불가능).

    배포 가능성: TCP z 하나만 쓴다 — 실기 RTDE/FK로 충족, 큐브 pose 추정기 불필요.
    """
    return tcp_z < cfg.grasp_z_max_m


def carry_latch_update(carried, gripper_closed, tcp_rise, cfg: ReleaseSupervisorCfg):
    """운반 latch: 닫은 채 ★폐합 시점보다 `lift_rise_m` 이상 올라오면★ 참으로 굳는다.

    ★F68-c 정정 — 절대 높이에서 상대 상승량으로★
      구 구현은 `tcp_z > lift_z_m(0.100)`이었다. 실측(채택 teacher, 256 에피소드)에서
      ★들림 중 TCP z는 p50 0.0319 / p90 0.0448 / p99 0.0618★이고 0.100을 넘는 스텝은
      ★0.01%★뿐이다 — 정상 운반으로는 그 높이에 도달하지 않는다.
      그런데도 F66에서 latch가 걸린 이유는, 정책이 ★접근 초기 높은 곳★(전 구간 p95 0.178)
      에서 그리퍼를 닫았기 때문이다. 즉 이 latch는 운반이 아니라 ★조기 폐합★을 추적하고
      있었고, F68이 조기 폐합을 막자 함께 죽었다(zone&open 68.75% → 0.00%).

      상대 상승량은 절대 높이와 달리 작업면·큐브 크기·로봇 설치 높이가 바뀌어도 의미가
      유지되고, 실기에서도 FK로 그대로 얻는다(폐합 시점의 TCP z를 기억하면 된다).

    ⚠️ `tcp_rise` = 현재 TCP z − ★폐합 시점★ TCP z. 호출측이 폐합 전이에서 기준값을
       기록해야 한다(열리면 기준값도 무효화).
    """
    return carried | (gripper_closed & (tcp_rise > cfg.lift_rise_m))


def release_decision(carried, tcp_z, settle_count, cfg: ReleaseSupervisorCfg):
    """이번 스텝에 릴리스를 실행할지 — 운반 중 ∧ 배치 높이 ∧ ★연속 정지★.

    `settle_count`는 `settle_counter_update`가 만든 연속 정지 창 수다(★F92★ — 단일 창 속도가
    아니다). 반환은 ★순간 판정★이다. 한번 릴리스한 뒤 계속 열어 두는 것(latch)은 호출측
    책임이며 `released |= release_decision(...)` 형태로 쓴다 — 놓은 뒤 다시 닫으면 큐브를 건드린다.
    """
    return carried & (tcp_z < cfg.tcp_z_max_m) & (settle_count >= cfg.tcp_settle_steps)
