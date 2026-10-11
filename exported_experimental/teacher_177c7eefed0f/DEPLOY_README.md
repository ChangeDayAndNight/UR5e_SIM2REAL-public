# 모듈형 배포 v2 — 인식 모듈 + Teacher v2 정책 (★릴리스 아님★)

이 번들은 **배포 가능한 관측(34D)으로 학습한 Teacher v2**를 **큐브 인식 모듈**과 결합해 실기에서
돌리기 위한 산출물입니다. 학습 관측 자체가 실기 인식 모듈과 같은 규칙(오차·결측·카메라 기하)으로
만들어졌으므로 근사 항이 없습니다. 실기 패리티·지연 증거와 캘리브레이션 승인이 아직 없으므로,
사람이 e-stop 앞에 있는 랩 환경에서 명시적 모션 인가 아래에서만 사용하십시오.

- checkpoint: `logs/rsl_rl/ur5e_binpick_teacher/2026-09-20_22-22-51-931877/model_24200.pt` (iter 24200)
- sha256: `177c7eefed0fe3e1…`
- 관측/행동: 34D 상태 → 7D `ik_rel`
- sim 봉인 평가(seed 42~45 × 512, 인식 오차 DR·실행 제한 적용): placed {'42': 512, '43': 512, '44': 512, '45': 512} — pooled 100.00% (Wilson 95% 하한 99.81%), 채택 상태 `adopted`
- TorchScript 동치: 무작위 프로브 max|Δ| = 0.00e+00

## 관측 구성 (sim 학습 = 실기 조립 규칙 동일)
| 신호 | 출처 |
|---|---|
| 관절 위치/속도, TCP 포즈 | UR RTDE (RobotState) |
| gripper_command_state | 지연·전이 적분된 명령 상태(common/gripper_command, 0=open 1=closed) |
| 큐브 위치 | `common/cube_depth_locator.py` depth 역투영 → 작업면 위 높이 띠 (0.025, 0.15) m의 큐브 크기 덩어리 → 상면 중앙값, 쥐지 않으면 z를 작업면에 고정. 미검출·held면 TCP+(0.0, 0.0, -0.027) m, 아니면 마지막 추정치 |
| held | 폐합 전이 시점에 추정치가 TCP 0.045 m 이내면 latch, 열리거나 큐브가 멀리서 검출되면 해제 |
| last_action | 직전 정책 출력 clamp ±1, 기울임 채널 0 |

큐브가 40프레임(2.0 s) 넘게 안 보이고 쥐고 있지도 않으면 팔을 세운다(hold).
추정치가 0.1 m 넘게 튀면 2프레임 연속 확인 전에는 받지 않는다.
학습 시 인식 DR 범위(configs/dr.yaml perception): 편향 ≤1 cm, xy σ 0.3~1.5 cm, 검출률 60~100%, 최소거리 0.17~0.25 m, 실기 color FOV 밖 결측.

## 배포 PC에서 할 일
1. 이 폴더를 통째로 복사한다. `configs/calib/hand_eye.yaml`·`tcp_offset.yaml`이 실기 캘리브레이션 값인지 확인한다
   (`calibrated: false`면 `deployment/Calib_HandEye.py`, `Calib_TcpOffset.py`로 먼저 캘리브한다).
2. 작업면 높이(base z, 기본 -0.0375 m)가 실기와 맞는지 확인한다 — depth 추정기는 작업면 위
   높이 띠에서 큐브를 찾으므로 이 값이 2 cm 이상 틀리면 큐브를 놓친다. `python deployment/realsense_capture_test.py`
   로 프레임을 저장하고 `common/cube_depth_locator.py`로 검출되는지 본다(색분할은 보조 게이트일 뿐이다).
3. 추론만: `python Phase4/Run_Task_Teacher.py --policy-dir <이 폴더> --dry-run --verbose-perception`
   — 매 스텝 인식 결과(검출/보유/hold)와 행동을 출력한다. 검출률이 30% 미만이면 모션으로 가지 않는다.
4. 실제 모션(e-stop 앞에서, 인가 파일 필요): `--yes-i-am-at-the-estop --action-scale 1.0`
   학습이 배포 SafetyGuard 상한(0.25 m/s)과 같은 스케일로 이뤄졌으므로 action-scale 1.0이 학습 조건이다.
   처음에는 0.5로 시작해 거동을 확인한 뒤 1.0으로 올린다.

## 한계
- 실기에서는 인식 오차가 학습 DR 범위 안에 있어야 한다(hand-eye 잔차 ≤1 cm, 검출률 ≥60%). 캘리브가 어긋나면 위치가 통째로 밀린다.
- 현재 hand-eye 기하에서 그리퍼 바로 아래 큐브는 실기 세로 FOV 밖이다(README F88). 정책은 그 결측을 학습에서 경험했지만,
  브라켓을 공구축 쪽으로 수렴하게 재장착하고 hand-eye를 재캘리브하면 검출률이 올라간다.
