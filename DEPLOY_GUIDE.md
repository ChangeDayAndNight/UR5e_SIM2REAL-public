# UR5e 실기 배포 폴더 v2 — Teacher v2(34D) + Vision Student (2026-09-25)

이 폴더 하나를 배포 PC로 복사하면 됩니다. 저장소 코드(`repo/`)와 정책 번들(`policies/`)이 함께 들어 있습니다.
`repo/exported_experimental/` 안에도 같은 번들이 들어 있어, `repo/`에서 바로 실행할 수 있습니다.

| 항목 | 값 |
|---|---|
| 코드 커밋 | `5368499` (`COMMIT.txt` 참조) |
| Teacher 정책 | `policies/teacher_177c7eefed0f/` — 34D 상태 관측 → 7D `ik_rel`. 큐브 위치는 depth 인식 모듈이 추정 |
| Teacher sim 성공률 | placed **100.00%** (seed 42~45 × 512 = 2048 에피소드, Wilson 95% 하한 **99.81%**) |
| Student 정책 | `policies/student_8b415b9b0831/` — RGB-D 50,206D 관측 → 7D `ik_rel`. 카메라 영상만으로 동작 |
| Student sim 성공률 | placed **94.53%** (2048 에피소드, Wilson 95% 하한 **93.46%**, 최저 시드 480/512 = 93.75%) |
| 평가 조건 | 인식 오차 DR(편향 ≤1 cm, xy σ 0.3~1.5 cm, 검출률 60~100%, 최소거리 0.17~0.25 m, 실기 FOV 밖 결측) + 실행 제한(TCP 0.25 m/s·0.5 rad/s = 배포 SafetyGuard와 동일 상한) |
| 실기 성공률 | **미측정.** 실기 인식 오차가 위 DR 범위 안이면 sim 수치가 상한에 가깝고, 캘리브레이션이 어긋나면 급락합니다 |

두 정책 모두 **릴리스가 아닙니다**(`adopted: false` = 릴리스 레인 미승인). sim 봉인 평가는 통과했고
(`teacher_adoption`, Phase2 `adoption.json`), 실기 패리티·지연 증거와 캘리브레이션 승인이 없으므로
**사람이 e-stop 앞에 있는 랩 환경에서만** 사용하십시오.

## 어느 정책을 쓰는가

- **Teacher(권장 시작점)**: 큐브 위치를 depth 인식 모듈(`common/cube_depth_locator.py`)이 추정해 정책에 넣습니다.
  인식이 실패하면 화면에 바로 드러나고, 실패 원인(캘리브·작업면 높이·조명)을 분리해 고칠 수 있습니다.
- **Student**: 카메라 영상을 그대로 받습니다. 중간 단계가 없어 단순하지만, 실패 시 원인 분리가 어렵습니다.
  Teacher로 먼저 파이프라인을 확인한 뒤 Student로 넘어가는 것을 권합니다.

## 0. 준비 (배포 PC, Windows 11, Isaac 불필요)

```powershell
cd <이 폴더>\repo
conda create -n env_ur_real python=3.11.15 -y
conda activate env_ur_real
$env:PYTHONIOENCODING="utf-8"      # 한국어 콘솔(cp949)에서 출력이 깨지지 않도록
pip install --require-hashes -r deployment\requirements-win11.lock.txt
python deployment\DeployPC_Runbook.py --policy-dir exported_experimental\teacher_177c7eefed0f
```

런북이 6단계(환경 → 산출물 → 설정 → 캘리브 → 인가 → 하드웨어)를 점검하고 실패한 단계마다 할 일을 알려줍니다.
`--with-hardware`를 주면 RTDE·RealSense 수신 연결까지 확인합니다(명령은 보내지 않습니다).

## 1. 실기 설정 (사람 작업)

1. `configs\real.local.yaml` — 로봇 IP, D435i 시리얼(`camera.serial`), 848×480 컬러+depth 스트림 확인.
   `robot.tcp_command_verified`가 `false`면 어댑터가 `servoL`을 거부합니다. TCP 명령 검증(PROJECT_LOG.md §5) 후 `true`로 바꾸십시오.
2. **캘리브레이션** — `deployment\Calib_TcpOffset.py`, `deployment\Calib_HandEye.py`로 실측한 뒤
   `configs\calib\tcp_offset.yaml`·`hand_eye.yaml`의 `meta.calibrated: true`로 바꿉니다.
   값이 `false`면 모션이 거부됩니다(`--accept-uncalibrated`로만 감수 가능).
   - 목표: hand-eye 재투영 RMS ≤ 2 mm, 기지점 역투영 오차 median ≤ 1 cm. 정책은 편향 1 cm까지만 학습했습니다.
3. 작업면 높이(base z = −0.0375 m)와 큐브 한 변(50 mm)이 실기와 같은지 확인합니다. depth 인식기는 작업면 위
   2.5~15 cm 띠에서 큐브를 찾으므로 이 값이 2 cm 이상 틀리면 큐브를 놓칩니다.
4. **인식 확인(모션 전 필수)** — `python deployment\realsense_capture_test.py`로 프레임을 저장하고
   `common\cube_depth_locator.py`가 큐브를 찾는지 확인합니다.

## 2. 그리퍼 DO 펄스 실측 (첫 모션 전에 한 번, 미측정 항목)

2FG7은 폭·파지 피드백이 없어 소프트웨어가 **명령 상태**만 추적합니다(24스텝 = 1.20 s 전이 계약).
아래 세 값은 현재 **측정되지 않았습니다**. 첫 모션 전에 재어 계약과 맞는지 확인하십시오.

```powershell
cd <이 폴더>\repo
python deployment\Verify_Gripper2FG7.py --real-config configs\real.local.yaml --dry     # 배선/수신 확인
```

1. **완전 폐합까지 걸리는 시간** — DO5 펄스(0.30 s) 후 손가락이 멈출 때까지를 스톱워치나 영상으로 잽니다.
   1.20 s보다 길면 `configs\common.yaml`의 `gripper_command` 전이 계약과 어긋납니다(그 경우 알려주십시오 — 재학습 대상).
2. **반전 거동** — 펄스 중 반대 핀을 주면 어떻게 되는지. 소프트웨어는 전이 후 6스텝(0.30 s) 동안 반전을 무시하므로
   정상 운전에서는 발생하지 않습니다. 하드웨어가 반전에서 fault를 내는지만 확인하면 됩니다.
3. **Compute Box 프리셋** — 파지력(20~140 N), 속도, 외부/내부 파지, 개방 목표 폭. 50 mm 큐브를 74 mm 개방에서
   잡고 34 mm까지 닫는 구성이어야 합니다(도면 `2fg7.png` 기준, sim과 동일).

## 3. 단계적 실행 (각 단계에서 멈추고 확인)

```powershell
cd <이 폴더>\repo
conda activate env_ur_real
$env:PYTHONIOENCODING="utf-8"

# ① 추론만 (모션 없음) — 매 스텝 인식 결과(검출/보유/hold)와 행동을 출력
python Phase4\Run_Task_Teacher.py --policy-dir exported_experimental\teacher_177c7eefed0f `
    --dry-run --verbose-perception --target-pos="-0.50,0.15,-0.0125"

# ② 실제 모션 — e-stop 앞에서. 시작 자세가 홈이 아니면 --home-first 로 이동을 허용
python Phase4\Run_Task_Teacher.py --policy-dir exported_experimental\teacher_177c7eefed0f `
    --yes-i-am-at-the-estop --home-first --target-pos="-0.50,0.15,-0.0125"

# ③ Student (Teacher로 파이프라인을 확인한 뒤)
python Phase4\Run_Task.py --policy-dir exported_experimental\student_8b415b9b0831 `
    --dry-run --target-pos="-0.50,0.15,-0.0125"
```

기본값이 곧 학습 조건입니다: `--max-steps 200`(10초 에피소드), `--action-scale 1.0`.
`--action-scale`을 1.0 미만으로 낮추면 릴리스 수퍼바이저의 **절대** 정지 임계(0.02 m/s)와 어긋나
놓는 시점이 달라집니다. 관찰 목적 외에는 낮추지 마십시오.

① 단계에서 **검출률이 30% 미만이면 모션으로 가지 마십시오.** 캘리브레이션·작업면 높이·조명 문제입니다.

## 4. 동작 계약 (sim과 실기가 같은 코드로 판정)

- **폐합**: 정책이 결정하되 TCP z < 0.040 m에서만 새로 닫힙니다(높이 제약).
- **릴리스**: 운반 중 ∧ TCP z < 0.075 m ∧ **연속 3창(150 ms) 정지**일 때 수퍼바이저가 실행하고 팔을 세웁니다.
- **그리퍼 전이 잠금**: 전이 후 6스텝(0.30 s) 동안 반대 명령을 무시합니다(실기 DO 펄스 보호).
- **hold**: 큐브를 4초(80프레임) 넘게 못 보고 쥐고 있지도 않으면 팔을 세웁니다. 마지막 추정치 바로 위
  (예상된 사각지대)나 그리퍼 전이 중의 결측은 세지 않습니다.
- **정지 조건**: 작업영역 AABB 이탈, TCP z 바닥, 관절 속도 상한, 센서 신선도·스큐·제어주기 초과.

## 5. 알려진 한계

- **카메라 사각지대**: 현재 브라켓(hand-eye v7)에서 그리퍼 바로 아래 큐브는 세로 FOV 밖입니다. 최종 하강 구간은
  항상 블라인드이고, 정책은 그 결측을 학습에서 경험했습니다. 브라켓을 공구축 쪽으로 재장착하면 검출률이
  올라가지만, **재장착 시 hand-eye 재캘리브와 Teacher 재학습이 필요합니다**(가시성 기하가 학습 분포입니다).
- **파지 피드백 없음**: 폭·파지 감지 신호가 없어 "쥐었는가"는 명령 상태와 인식 추정치로만 판정합니다.
- **실기 지연 미측정**: `configs\real.local.yaml`의 `latency.measured_*`가 비어 있습니다. servoL lookahead 0.1 s가
  학습의 액션 지연 DR(0~1스텝)보다 크므로, 첫 세션에서 추종 지연을 재고 필요하면 조정하십시오.
- **두 정책 모두 릴리스 아님**: 정식 릴리스 레인(`deployment/release.py`)은 패리티 10,000샘플·지연 p99 벤치마크·
  커밋 동결을 추가로 요구합니다.

## 6. 파일 무결성

`SHA256SUMS.txt`로 복사본을 검증할 수 있습니다.

```powershell
cd <이 폴더>
Get-Content SHA256SUMS.txt | ForEach-Object { $h,$f = $_ -split '  ',2; if ((Get-FileHash $f -Algorithm SHA256).Hash -ine $h) { "MISMATCH: $f" } }
```

각 번들 폴더 안의 `experimental_manifest.json`도 `policy.pt`/`obs_spec.json` 해시를 들고 있고,
실행기가 기동할 때마다 대조합니다(전송 중 손상 시 거부).
