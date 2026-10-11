# UR5e Sim2Real Bin Picking

UR5e, OnRobot 2FG7, Intel RealSense D435i로 5 cm 큐브를 집어 정적 작업면의 가변 목표에 놓는 Isaac Lab 프로젝트다. 상태 기반 Teacher를 먼저 학습하고, RGB-D Student로 증류한 뒤, 같은 관측·행동·안전 계약을 실기 배포까지 유지한다.

> **현재 판정 — 2026-09-25:** Teacher v2(34D, 대칭 그리퍼)와 Vision Student가 모두 sim 봉인 평가를 통과했다(seed 42~45 × 512 = 2048 에피소드). 실기 성공률은 아직 측정하지 않았다. 두 정책은 랩 실험 레인(`adopted: false`)이며 정식 릴리스 레인(패리티·지연 벤치마크·커밋 동결)은 거치지 않았다.

| 정책 | 성공률(placed) | Wilson 95% 하한 | 관측 | 번들 |
|---|---|---|---|---|
| **Teacher** | **100.00%** | 99.81% | 34D 상태 + depth 인식 모듈의 큐브 추정 | `exported_experimental/teacher_177c7eefed0f/` |
| **Student** | **94.53%** | 93.46% | RGB-D 영상 50,206D | `exported_experimental/student_8b415b9b0831/` |

## 빠른 시작 — 실기 배포 (Windows 11, Isaac Sim 불필요)

```powershell
git clone https://github.com/ChangeDayAndNight/UR5e_SIM2REAL-public.git
cd UR5e_SIM2REAL-public
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`setup.ps1`이 conda 환경(`env_ur_real`, Python 3.11) → 의존성(torch-cpu·ur-rtde·pyrealsense2·opencv) →
`configs\real.local.yaml` 생성(템플릿) → 배포 점검 런북을 한 번에 실행하고 남은 현장 작업을 알려준다.
로봇·카메라가 없어도 여기까지는 돈다. 이어서:

```powershell
conda activate env_ur_real
$env:PYTHONIOENCODING="utf-8"

# 1) 현장 값: configs\real.local.yaml 의 robot.ip, camera.serial
#    캘리브레이션: deployment\Calib_TcpOffset.py, deployment\Calib_HandEye.py → configs\calib\*.yaml meta.calibrated: true

# 2) 추론만(모션 없음) — 매 스텝 인식 결과와 행동 출력. 검출률 30% 미만이면 3)으로 가지 않는다
python Phase4\Run_Task_Teacher.py --policy-dir exported_experimental\teacher_177c7eefed0f `
    --dry-run --verbose-perception --target-pos="-0.50,0.15,-0.0125"

# 3) 실제 모션 — 사람이 e-stop 앞에서
python Phase4\Run_Task_Teacher.py --policy-dir exported_experimental\teacher_177c7eefed0f `
    --yes-i-am-at-the-estop --home-first --target-pos="-0.50,0.15,-0.0125"
```

Student는 `Phase4\Run_Task.py --policy-dir exported_experimental\student_8b415b9b0831`로 같은 순서를 밟는다.
전체 배포 절차·동작 계약·한계·그리퍼 DO 펄스 실측 절차는 **[DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)** 에 있다.
한 페이지 프로젝트 개요(작업 셀·파이프라인·계약·결과·한계)는 **[docs/index.html](docs/index.html)** 이다 — GitHub Pages(`main` 브랜치 `/docs`)로 게시한다.

이 파일(README.md)은 위 빠른 시작과 함께 프로젝트의 **Markdown 원장**이다. 실행법, 계약, 진행 상태, 실패 이력, 자산 출처, 배포 게이트를 바꿀 때 코드와 이 파일을 같은 변경으로 갱신한다. 배포 절차 문서 `DEPLOY_GUIDE.md`와 번들별 `DEPLOY_README.md`만 예외로 허용한다(F97). 실험 결과는 JSON/CSV/TXT/TeX/PDF/PNG로 생성하며 새 Markdown 보고서를 만들지 않는다.

## 1. 고정 스택과 지원 범위

학습 재현의 기준은 다음 조합뿐이다. 자동으로 “최신” 버전을 설치하거나 다른 Isaac 조합으로 묵시적으로 넘어가지 않는다.

| 구분 | 고정값 |
|---|---|
| 학습 OS | Windows 11 native, WSL 비지원 |
| conda env / Python | `isaac51` / Python 3.11.15 |
| NVIDIA Isaac Sim | **5.1.0** pip distribution (`isaacsim==5.1.0.0`) |
| NVIDIA Isaac Lab | **v2.3.2** source checkout (`external/IsaacLab`) + editable install |
| RL | `rsl-rl-lib==3.1.2`, `tensordict==0.13.0` |
| PyTorch | `torch==2.7.0+cu128`, `torchvision==0.22.0+cu128` |
| 기준 GPU | RTX 3060 Ti 8 GB |
| 실기 env | Windows 11 별도 배포 PC, `env_ur_real`, Python 3.11.15, Isaac 미설치 |
| 실기 SDK | `ur-rtde==1.6.2`, `pyrealsense2==2.58.3.10794` |
| 정책 주기 | 20 Hz |
| 물리 주기 | 120 Hz; `decimation=6`은 두 주기에서 파생 |

`scripts/bootstrap_training.ps1`은 위 버전에서 벗어난 설치를 거부해야 한다. Isaac Lab의 Python package 내부 버전 문자열만 보고 판단하지 말고, checkout tag/commit이 `v2.3.2`인지 함께 확인한다. 설치 로그를 저장소에 커밋하지 않는다.

## 2. GitHub clone 후 바로 학습

### 2.1 준비

- 최신 NVIDIA Studio/Game Ready 드라이버와 CUDA 사용 가능 GPU
- Git, Miniconda/Conda, PowerShell 5.1 이상
- Isaac Sim/Lab과 캐시를 위한 충분한 여유 공간

```powershell
git clone https://github.com/ChangeDayAndNight/UR5e_SIM2REAL-public.git UR5e_Sim2Real
Set-Location UR5e_Sim2Real
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_training.ps1
```

bootstrap은 고정 이름 `isaac51`을 만들고 정확한 Isaac Sim 5.1.0/Isaac Lab v2.3.2 조합, PyTorch CUDA 조합과 프로젝트 extension을 설치한다. Isaac Lab checkout은 고정 경로 `external/IsaacLab`만 지원한다. bootstrap 마지막과 모든 `train.ps1` 실행은 생략할 수 없는 비정적 preflight를 수행한다. 저장소에 함께 커밋된 필수 USD는 생성하지 않으며, preflight에서 manifest 크기·SHA-256과 USD Stage/dependency를 검증한다. preflight가 하나라도 실패하면 학습하지 않는다. 특히 다음을 확인한다.

- Python 3.11.15, `isaacsim==5.1.0.0`, Isaac Lab checkout `v2.3.2`
- `torch==2.7.0+cu128`, CUDA 사용 가능, `rsl-rl-lib==3.1.2`, `tensordict==0.13.0`
- 후처리·회귀검사용 `pandas==2.2.3`, `matplotlib==3.10.3`, `openpyxl==3.1.5`
- `source/ur5e_binpick` import 및 Gym task 등록
- manifest의 필수 USD 7개가 모두 일반 Git 추적 상태이고 크기·SHA-256·Stage dependency가 일치
- `configs/common.yaml`의 120/20 Hz가 정수 decimation 6을 만드는지
- 112×112 RGB-D 관측 50,206차원, action 7차원 계약

`scripts/preflight_training.py`는 환경·자산·계약과 실제 Gym registry의 fail-fast 검사이며 공개 학습 진입점이 자동 실행한다. pytest는 별도로 실행하고, Phase0 scene smoke와 Teacher 1 iteration은 아래 `train.ps1 -Smoke`가 담당한다.

### 2.2 Teacher smoke와 본 학습

```powershell
# 최소 파이프라인 검증: 16 env / 1 iteration / seed 42 / 모든 postprocess 생략
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Preset Teacher -Smoke

# 현행 Teacher v2(F88) 본 실험: 명시적 override로 25k, seed 42, 16384 env (RTX 5070 Ti 16 GB 실측 — configs/sim.yaml 주석)
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Preset Teacher -NumEnvs 16384 -MaxIterations 25000 -Seed 42 -Headless

# 완료된 scratch run의 Teacher 채택 행렬(상위 5개 × seeds 42..45 × 512 episodes)
python Phase1/BinPick_PostTrain.py --run logs/rsl_rl/ur5e_binpick_teacher/<RUN> --max_iterations 25000

# Vision scratch 학습
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Preset Vision -NumEnvs 128 -MaxIterations 20000 -Seed 42 -Headless

# 채택 Teacher에서 Vision student 증류
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Preset Distill -DistillFrom <TEACHER_MODEL_PT> -NumEnvs 128 -MaxIterations 20000 -Seed 42 -Headless

# 선택 NuRec: Git에서 제외된 Workplace.usdz와 같은 계약의 sealed student가 모두 필요
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Preset NuRec -NuRecFrom <STUDENT_MODEL_PT> -NumEnvs 4 -MaxIterations 1 -Seed 42 -Headless -SkipPostprocess
```

2026-09-07 증류 정정: 병진/yaw Teacher 정답은 무계 IK 명령을 보존하고, 무시되는 기울임과 이진 그리퍼만 정규화한다. 기본 수집은 deterministic이며 `BINPICK_DISTILL_COLLECTION=sampled`로 고정 std 탐색 대조군을 선택할 수 있다. 모드는 `params/agent.yaml`에 저장된다. 영상만 지연하고 최신 proprio를 유지하며, reset 후 1회 재렌더하고 영상 노이즈 강도는 환경별로 재샘플한다. 관측 폭/체크포인트 가중치 구조는 유지되지만 학습·관측 분포가 달라지므로 구 성공률과 직접 비교하거나 기존 모델의 성능 개선으로 간주하지 않는다. 기존 Teacher 동역학, pregrasp reset, 파지/릴리스 조건, 실기 안전 제한은 유지한다. 실기 속도 제한에 맞춘 Teacher 검증과 파지 상태 추정은 별도 과제다.

지연 비교는 non-Smoke `scripts/train.ps1`의 `-ExtraArgs`로 `env.events.obs_delay.params.lo=0`, `env.events.obs_delay.params.hi=0`을 함께 전달한다. 구 전체 관측 지연 대조군은 `env.observations.policy.flat.params.delay_proprio=true`다. 한 번에 한 조건만 비교하고 실제 저장된 `params/env.yaml`을 확인한다. 재학습·재평가 없이 기존 배포 번들의 가중치/계약 파일을 덮어쓰지 않는다.

`configs/sim.yaml`의 저장 기본값은 20,000 iteration이며 위 full 명령은 현재 실험 프로토콜인 25,000으로 명시적으로 덮어쓴다. 완료된 schema-2 run receipt는 clean Git commit/config hash뿐 아니라 Windows build·CPython 3.11.15·전체 preflight 고정 distribution·Isaac import origin·CUDA/driver·GPU 이름/VRAM을 봉인한다. 이 증거가 누락된 legacy checkpoint는 export할 수 없다. 8 GB GPU에서 OOM이면 다른 하이퍼파라미터를 동시에 바꾸지 말고 `NumEnvs`만 4096→2048→1024 순으로 절반씩 낮춘다. smoke가 통과하지 않은 상태에서 장시간 학습을 시작하지 않는다.

smoke의 exit code 0만으로는 성공으로 보지 않는다. Phase 0은 exact sentinel `UR5E_SMOKE_OK robot=ur5e_cfg joints=8`을 내야 하고, Teacher/Vision 학습은 해당 실행이 발급한 exact run 경로에 비어 있지 않은 `model_*.pt`, `params/env.yaml`, `params/agent.yaml`, `events.out.tfevents.*`를 모두 남겨야 `TRAINING_ARTIFACTS_OK`로 통과한다.

`Phase1/BinPick_Train.py`, `Phase2/VisionPick_Train.py`, `scripts/rsl_rl/train.py`는 `train.ps1`이 호출하는 내부 구현이며 공개 진입점이 아니다. 내부 스크립트도 full preflight는 자체 강제하지만, 직접 실행하면 preset·smoke 계약과 배포 가능한 schema-2 run receipt 경로를 벗어나므로 지원하지 않는다. 다음 Phase0/Eval 명령은 학습 launcher가 아닌 수동 진단용이다.

```powershell
python Phase0/smoke_test.py --headless
python Phase0/scene_bin_ur5e.py --headless --num_envs 4 --steps 200
python Phase1/BinPick_Eval.py --num_envs 256 --n_episodes 512 --headless
```

최소 smoke 뒤 GPU 여유와 짧은 학습 거동을 더 보고 싶을 때만 256 env/50 iteration을 선택적 2차 smoke로 실행한다. 이것은 fresh clone 준비의 필수 조건이 아니다.

학습 로그와 checkpoint는 `logs/` 아래의 로컬 산출물이며 Git 대상이 아니다. 결과 분석기는 `Phase1/Result/` 아래에 구조화된 로컬 산출물을 다시 만들 수 있지만 그 디렉터리도 Git 대상이 아니다.

### 2.3 채택 게이트

Teacher는 다음을 모두 만족해야 Phase 2 입력이 된다.

1. 현행 42D privileged observation, 2FG7 `q∈[0,0.020] m` 개도 의미와 7D action checkpoint 계약 검증 통과
2. `Episode_Termination/placed_success` trailing-200 median 상위 5개(최종 checkpoint 고정 + separation 500, F70)를 assist/pregrasp 없이 seeds 42·43·44·45에서 각각 정확히 512 episodes 평가
3. 모든 seed에서 grasp/carry/placed 성공 수가 각각 `≥487/512`이고, 지표별 pooled 2,048 episodes의 Wilson 95% 하한도 `≥95%`
4. reward farming, 그리퍼 액션 포화, curriculum resume 되감김, NaN/발산 없음
5. F64 release payout/예산 비례 소모 계약과 관련 회귀 테스트 통과

`Phase1/Result/<RUN>/top5_evaluation/adoption.json`의 `status: adopted`와 exact checkpoint hash만 Phase 2 권한이다. 평가 재사용도 checkpoint hash·iteration·seed·512 episodes·CSV/stats/summary hash가 모두 같을 때만 허용한다. 봉인된 schema-2 `run_manifest.json`은 후처리가 수정하지 않으며, 선택 weight 사본이나 `winner.json`, 단일 seed 최고 수치, 오래된 checkpoint 이름은 권한이 아니다. 최종 `adoption.json`은 채택·거부 모두 덮어쓰지 않는다.

## 3. 태스크와 공통 계약

### 3.1 작업 셀

- 로봇: UR5e 6축, base 고정
- 툴 체인: `tool0 → Magbot TCW1 86.5 mm → bracket → OnRobot 2FG7`
- 물체: 한 변 0.05 m 큐브 1개
- 물리 작업면: 중심 `(-0.50, 0.0) m`, 크기 `0.58 × 1.25 × 0.04 m`, 상면 `z=-0.0375 m`
- pick zone: `0.45 × 0.60 m`
- 목표: base frame `x∈[-0.725,-0.275]`, `y∈[-0.300,0.300]`, `z=-0.0125 m`
- 안전 AABB: `[-0.75,-0.35,-0.0375]`부터 `[-0.21,0.35,0.60]`
- 실기/시뮬레이션 TCP 하강 바닥: `z=-0.0075 m`

`configs/sim.yaml`, `configs/common.yaml`, `configs/safety.yaml` 중 하나의 작업면/목표/안전 값을 바꿀 때 연동 값을 함께 검증한다. 기하·접촉·목표 분포 변경은 기존 checkpoint와 호환된다고 가정하지 않는다.

### 3.2 관측

Phase 1 Teacher v2(2026-09-18, F88)의 actor observation은 **34D**이며 ★실기에서 만들 수 있는 신호만★ 담는다: 관절 위치/속도(12), `gripper_command_state`(1, Student와 같은 명령 상태), TCP pose(7), target(3), **큐브 위치 추정치**(3), `held` latch(1), last action(7). 카메라를 쓰지 않는다. 큐브 위치는 참값이 아니라 `mdp/perception_model.py`가 배포 인식 모듈(`common/teacher_obs.py`)과 같은 규칙으로 만든 추정치다 — 카메라 기하(hand-eye·TCP offset·실기 color FOV·D435i 최소거리) 밖이면 결측, 검출 시 에피소드 편향+가우시안 노이즈(`configs/dr.yaml perception`), 미검출·held면 TCP+offset, 아니면 마지막 추정치 유지. `held`는 폐합 전이 시점에 추정치가 TCP 4.5 cm 안이면 latch되고 열리거나 큐브가 멀리서 검출되면 풀린다. critic은 34D에 sim 참값(물체 위치·속도·접촉력)을 더한 비대칭 관측을 받는다. 구 42D(실제 finger `q`·물체 속도·접촉력 특권)는 폐기했다.

Phase 2/실기 actor의 flat observation은 순서가 고정된다.

```text
depth 112×112
rgb 3×112×112
joint_pos 6
joint_vel 6
gripper_command_state 1
tcp_pose 7
target_pos 3
last_action 7
total = 50,206
```

Depth는 invalid→far 치환, near/far clip, resize, `[0,1]` 정규화를 거친다. RGB는 depth→color 정렬된 color optical frame을 사용하고 같은 공간 변환을 거친다. actor의 `gripper_command_state`는 실제 finger 위치나 폭이 아니라 지연 후 전달된 binary 명령의 결정론적 1D 상태다. `0=open`, `1=close`이고 20 Hz에서 목표 쪽으로 매 policy step `1/24`씩 이동해 1.20초에 전이한다. sim reset은 open으로 초기화한다. 실기에서는 target-pin HIGH SDK가 정확히 `True`를 반환한 사실을 예약 시점부터 40 ms 안에 확인한 명령만 한 step 갱신하며, 이전 300 ms pulse 때문에 edge/reversal을 확인할 수 없거나 dispatch가 늦으면 queue하지 않고 state invalid+E-stop으로 종료한다. motion startup은 unknown이고, 명시적 open pulse가 완전히 성공한 시점부터 다시 1.20초가 지난 뒤에만 open으로 확정한다. fault/E-stop은 즉시 unknown으로 되돌린다. 이미 외부 SDK에 진입한 HIGH 호출은 40 ms 뒤 물리적으로 실행될 수 있으며 소프트웨어는 반환 직후 LOW를 시도할 뿐 이를 방지하거나 안전 정지로 증명하지 않는다. real 전처리를 별도로 복사하지 않는다. `common/obs_preprocess.py`가 sim/real의 유일한 구현이며 `exported/obs_spec.json`이 순서, 크기, temporal semantics, config hash를 봉인한다.

Teacher v2는 34D이고 Student의 1D 의미와 config provenance도 바뀌었으므로 종전 42D Teacher·`gripper_pos` Student spec/checkpoint/normalizer 및 종전 Git/config 계보는 형상이 같아도 **전부 비호환**이다. 새 clean SHA에서 Teacher scratch→Student distillation 순서로만 새 계보를 만든다.

**P0 실기 motion 차단:** 2FG7 DO pulse 배선에는 실제 폭·파지 feedback이 없다. `gripper_command_state`는 이 사실을 숨기는 명령 echo가 아니라 sim/real이 함께 학습·계산하는 command-history 관측이며 실제 개도나 파지 성공 증거로 사용할 수 없다. 따라서 Student 연구·candidate parity는 진행할 수 있지만, 실제 width/grip 구별 evidence가 확보되기 전에는 final readiness와 실기 motion 승인을 계속 거부한다.

### 3.3 행동

정책 출력은 7D다.

- arm 6D: base-frame TCP position delta 3 + axis-angle delta 3
- gripper 1D: `action[-1] < 0`이면 close, 그 외 open
- translation/rotation scale: 각각 0.0125 m / 0.025 rad = 배포 SafetyGuard의 스텝당 상한(0.25 m/s·0.5 rad/s × 0.05 s). sim 액션텀도 같은 상한으로 clamp한다(F88)
- sim: Differential IK
- real: 같은 의미의 ActionMapper를 거쳐 검증된 `servoL` 경로

Teacher v1(42D privileged)은 실기에 배포할 수 없었다. **Teacher v2(34D, F88)는 배포 가능한 관측만 쓰므로 모듈형 배포 v2 레인**(`scripts/tools/export_teacher_policy.py` → `Phase4/Run_Task_Teacher.py`, 실험 레인: manifest `adopted=false`, sim 채택은 `sim_adopted`/`teacher_adoption`)으로 실기에서 돌린다. 정식 release 레인(`adopted=true`, `deployment/release.py`)은 여전히 Vision Student 전용이다. joint-absolute 경로와 task-space IK 경로를 섞지 않는다.

### 3.4 설정 소유권

| 파일 | 소유하는 계약 |
|---|---|
| `configs/common.yaml` | timing, action, preprocessing, proprio, target |
| `configs/sim.yaml` | scene, camera, training 규모 |
| `configs/dr.yaml` | physics/RGB-D/camera/latency randomization |
| `configs/safety.yaml` | workspace, z floor, velocity/force, e-stop latch |
| `configs/calib/*.yaml` | hand-eye, TCP, tool chain, scene registration |
| `configs/real.example.yaml` | fail-closed 실기 설정 템플릿 |
| `configs/real.local.yaml` | 배포 PC의 IP/serial/실측값/증거 경로; Git에서 제외 |

`decimation`은 저장하지 않고 120/20에서 계산한다. 캘리브레이션 결과 YAML은 생성물이지만 모델과 실험을 정의하므로 값이 실제로 검증된 경우 커밋한다. `meta.calibrated`나 evidence를 통과시키기 위해 추정값을 `true`로 바꾸지 않는다.

## 4. Phase 0–4 실행 계약

### Phase 0 — 환경과 scene smoke

학습 PC에서 다음 순서로 확인한다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Preset Teacher -Smoke
python Phase0/scene_bin_ur5e.py --headless --num_envs 4 --steps 200
python Phase0/scene_bin_ur5e.py --num_envs 2
```

성공 sentinel은 `UR5E_SMOKE_OK`, `BINPICK_SCENE_OK`다. GUI 단계에서는 flange/TCW1/2FG7 체인, finger collision, 카메라 방향, 작업면과 목표 구역을 육안 확인한다. RL 이전 파지 물리 진단은 `Phase0/Grasp_PhysicsTest.py`를 사용한다.

### Phase 1 — State Teacher PPO

- task: `Isaac-BinPick-UR5e-Teacher-v0`
- env: `BinPickTeacherEnvCfg`
- runner: `BinPickTeacherRunnerCfg`
- 기본 env: 4096, seed 42
- 산출: 로컬 TensorBoard, checkpoint, JSON/CSV/TXT/PDF/PNG/TeX

F64 이후 새 scratch run을 평가하기 전에는 Phase 2로 넘어가지 않는다. resume할 때 curriculum step offset을 검증하고, 이전 계약 checkpoint를 이어 붙이지 않는다.

### Phase 2 — RGB-D Vision Student

채택된 42D Teacher를 명시해 증류한다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Preset Distill `
  -DistillFrom <TEACHER_MODEL_PT> -NumEnvs 128 -MaxIterations 20000 -Seed 42 -Headless
python Phase2/VisionPick_PostTrain.py `
  --run logs/rsl_rl/ur5e_binpick_vision/<RUN> --max_iterations 20000
python Phase2/VisionPick_Env.py --num_envs 4
python Phase2/Export_Policy.py --candidate-only --checkpoint <VISION_STUDENT_MODEL_PT>
python Phase3/SimDeploy_Rehearsal.py `
  --candidate-dir logs/export_candidates/<CHECKPOINT_SHA_PREFIX> `
  --episodes 100 --max-steps 200 --seed 45
# 아래 final export는 실제 TCP/hand-eye/scene calibration 승인 계약이 모두 닫힌 뒤에만 실행한다.
# 현재 tracked calibration은 false/null이므로 여기서 멈추는 것이 정상이다.
python Phase2/Export_Policy.py `
  --candidate-dir logs/export_candidates/<CHECKPOINT_SHA_PREFIX> `
  --capture-manifest Phase3/logs/obs_dump_sim/<CANDIDATE_MANIFEST_SHA_PREFIX>/<CAPTURE_ID>/capture_manifest.json `
  --parity-obs logs/parity/<CANDIDATE_AND_CAPTURE_ID>.npy
```

Student 채택은 final checkpoint와 `Loss/behavior` trailing-200 median이 가장 낮은 후보 4개를 선택하되 후보 iteration 간격을 최소 500으로 유지한다. 동일한 5×4×512 성공/Wilson 게이트에 더해 finite·단계 불변식·target infeasible 0건을 요구한다. `Phase2/Result/<RUN>/posttrain_evaluation/adoption.json`이 선택한 원본 `model_N.pt`만 candidate export할 수 있고, Distill 시작·종료의 Teacher adoption hash 및 candidate manifest의 Student adoption hash가 달라지면 중단한다.

candidate-only 명령은 `approved:false`인 `policy.pt`, `obs_spec.json`, `candidate_manifest.json`만 ignored candidate store에 원자 게시하고 `release_manifest.json`은 만들지 않는다. candidate manifest는 clamp 비교 허용오차 `1e-12`, arm 6채널, 최소 분모 10,000, 채널별 비율 `<1%`인 rollout safety 계약도 고정한다. Phase 3는 이 exact candidate를 검증한 뒤 seed 45로 새 `<candidate-id>/<capture-id>` 전용 디렉터리에만 수집한다. 최소 100개의 서로 다른 episode가 각각 100 step 이하이고 모두 `placed_success`, safety trip 0, finite이며 실제로 sim adapter에 전달 완료된 명령 step이 총 10,000 이상이어야 한다. 1 episode에 10,000행을 묶은 fixture, 실패 episode, receive-only row, 중복 capture ID는 봉인하지 않는다. Sim adapter는 Isaac Lab 2.3.2의 physics-decimation 뒤 manager 순서로 control step당 `termination_manager.compute()`를 정확히 한 번 호출하고, reset 직후 이전 episode의 cached termination을 지운다. 각 dump는 동일 delay lineage의 mapper 원출력 `raw_arm_command`, SafetyGuard 뒤/action-term floor 전 `safety_arm_command`, 실제 적용된 `arm_command`를 함께 저장한다. 채널별 clamp count는 raw→safety와 safety→applied 변화의 합집합이므로 두 단계가 서로 상쇄·반전해 최종값이 raw와 같아져도 개입을 숨길 수 없다. 봉인기는 dump inventory와 episode별 row 수, 세 command 배열을 다시 계산해 영수증과 대조하므로 보고된 분모·count·fraction·step 수만 수동 수정할 수 없으며 정확히 1%도 실패한다. target feasibility는 호출자가 주는 boolean을 받지 않고 canonical `common.yaml`/`sim.yaml`/`safety.yaml`의 학습 범위·지지면·workspace·UR5e reach에서 다시 계산한다. `capture_manifest.json`은 expected episode/row/file 수, 각 파일 hash, aggregate hash, candidate manifest hash, 재계산된 `rollout_safety`, exact per-episode outcome receipt 및 각각의 SHA-256을 원자적으로 봉인한다. sibling/stale capture는 열거되지 않으므로 dataset에 섞이지 않는다.

Phase 3 성공 시 `scripts/build_parity_dataset.py`와 같은 검증 경로로 `logs/parity/*.npy`와 sidecar까지 만든다. 수동 재생성이 필요하면 `python scripts/build_parity_dataset.py --candidate-dir <CANDIDATE_DIR> --capture-manifest <CAPTURE_MANIFEST> --output logs/parity/<SAFE_NAME>.npy`를 사용한다. builder는 manifest-listed dump를 열린 단일 handle에서 읽고, 그 exact bytes의 size/hash를 검증한 뒤 같은 bytes로 dataset을 구성하므로 검증과 복사 사이 파일 교체(TOCTOU)는 게시되지 않는다. sidecar는 dataset hash/bytes/rows/dim, obs layout/config hash, candidate artifacts/manifest, training receipt/checkpoint/run, exact capture inventory, rollout-safety hash, seed-45 episode-outcome receipt/hash를 봉인한다. 최종 release manifest도 같은 evidence/hash를 검증하므로 candidate·capture·dataset·sidecar·raw command dump·episode receipt 중 하나라도 없거나 변경되면 실패한다.

Final export는 config SHA-256만으로 캘리브레이션을 승인하지 않는다. `tcp_offset`은 실측 4점법(`fixed_point_multi_orientation`)의 `calibrated:true`, 유한 잔차 `≤2 mm`여야 한다. `hand_eye`는 `eye_in_hand` 실측 뒤 post-correction 재검증과 DR coverage를 통과하고 median `≤10 mm`, p95 `≤20 mm`여야 한다. `scene_registration`은 합성 예시가 아닌 실측 landmark(`meta.measured:true`)의 Umeyama 결과이며 유한 잔차 `≤20 mm`여야 한다. 5단 tool chain도 모든 단이 measured이고 QC-RS 누락이 없으며 합계가 TCP offset과 일치해야 한다. 이 의미 계약은 final `Export_Policy`, `make/verify_release`, bundle build와 offline verifier에서 다시 계산하므로 파일과 manifest hash를 함께 고쳐도 우회할 수 없다. Candidate-only export와 Phase 3 parity는 이 P0가 열린 동안에도 비승인 상태로 보존할 수 있다.

`--out-dir`은 source tree 밖이나 임의 하위 디렉터리를 받을 수 없고 정확히 ignored `<repo>/exported`만 허용한다. release manifest에는 sidecar SHA-256과 레이아웃 블록별 대표성 통계도 기록된다. Distill checkpoint는 약 298 MiB 실측에 따라 500 iteration 간격(20k 기준 약 41개)이며, 1-iteration smoke를 제외한 Distill은 시작 전에 저장소 볼륨 여유 공간 25 GiB 이상을 강제한다. 112² RGB-D TiledCamera는 8 GB 기준 128 env부터 시작하고 OOM이면 128→64→32로 한 변수만 줄인다. 과거 MLP, 고정 카메라, GRU, TGRL 계보 수치는 현행 Teacher의 배포 근거가 아니다.

export는 다음을 모두 만들어야 한다.

- `exported/policy.pt`: TorchScript Vision Student
- `exported/obs_spec.json`: 50,206D/7D layout와 config hash
- `exported/release_manifest.json`: Student role, source checkpoint/training receipt/parity dataset hash, Git commit, config hashes
- 최소 10,000 observation에서 원 actor↔export 최대 절대오차 ≤ `1e-5`
- export host는 최소 10,000개 실제 observation, deployment host는 같은 단일 관측 호출을 최소 10,000회 측정해 inference p99가 manifest limit 이내

Teacher actor나 역할을 식별할 수 없는 모델은 release 생성 단계에서 거부한다.

### Phase 3 — calibration과 sim rehearsal

`Phase3/SimDeploy_Rehearsal.py`는 실제 rollout과 같은 `common/rollout.py` 경로를 sim adapter로 실행한다. 실기 calibration/검증 도구의 canonical 위치는 `deployment/`다.

```powershell
python Phase3/SimDeploy_Rehearsal.py --candidate-dir <CANDIDATE_DIR> --episodes 100 --max-steps 200 --seed 45
python -m pytest tests -q
python -m pip download --require-hashes --only-binary=:all: --platform win_amd64 --python-version 3.11 --implementation cp --abi cp311 --extra-index-url https://download.pytorch.org/whl/cpu -r deployment/requirements-win11.lock.txt -d deployment/wheelhouse
# calibration semantic gate가 모두 승인된 final release에만 허용된다.
python deployment/build_bundle.py --release-dir exported --output dist/Real_Verify
```

build 전에 위 명령으로 `deployment/wheelhouse/`에 CPython 3.11 `win_amd64` wheel 세트를 준비한다. 명령은 `deployment/wheelhouse.manifest.json`의 `method`와 정확히 같아야 하며, manifest와 파일명·크기·SHA-256이 하나라도 다르거나 wheelhouse가 비어 있으면 bundle 생성을 거부하는 것이 정상이다. builder는 clean Git HEAD가 embedded release의 `source.git_commit`과 정확히 같을 때만 실행하며 그 commit을 bundle manifest에도 봉인한다.

실기 장비가 필요한 단계는 배포 PC에서 사람이 수행한다. 손이 로봇/그리퍼 안에 없고 하드웨어 e-stop에 즉시 닿을 때만 다음 게이트로 진행한다.

### Phase 4 — fail-closed 실기

생성된 `dist/Real_Verify` 디렉터리를 hash가 보존되도록 배포 PC의 `C:\UR5e_Deploy\Real_Verify`로 복사한다. 배포 PC에는 Isaac을 설치하지 않는다. verifier는 `configs/real.local.yaml`과 `deployment/evidence/`만 명시적 mutable 예외로 허용하고 그 밖의 추가·변경 파일은 manifest 불일치로 거부한다. verifier에는 `--bundle-root`와, no-motion에는 더해 `--real-config`를 명시한다. `Deploy_DryRun.py`는 미래 수신 전용 인터페이스의 인자를 보존하지만 현 릴리스에서는 config 이전에 hard-block되고, `Deploy_Real.py`도 실행 인자를 해석하지 않는 hard-block stub이다. tracked template 자체는 readiness를 열 수 없다.

```powershell
conda create -n env_ur_real python=3.11.15 -y
conda activate env_ur_real
Set-Location C:\UR5e_Deploy\Real_Verify
Copy-Item configs/real.example.yaml configs/real.local.yaml
python -m pip install --no-index --find-links deployment/wheelhouse --require-hashes -r deployment/requirements-win11.lock.txt
python deployment/verify_deployment.py --mode offline --bundle-root .
```

offline verifier는 `pip`·`setuptools`·`wheel`을 포함한 모든 배포 Python distribution을 exact hash lock과 비교하고, lock 밖의 추가 distribution이 하나라도 있으면 실패한다. 따라서 기존 업무용 Python 환경을 재사용하지 말고 위 전용 환경을 사용한다.

no-motion receive-only 진단은 아래처럼 수행한다. RealSense는 global/system timestamp, `global_time_enabled=true`, `frames_queue_size=1` readback과 16초 warm-up을 강제하고, RTDE는 요청 뒤 controller timestamp가 전진한 동일 package만 받는다. 그러나 UR timestamp는 controller 부팅 후 경과시간이라 RealSense의 물리 acquisition clock과 절대 동기화되었다고 증명할 수 없다. 따라서 600초/10,000 cycle을 모두 수집해도 도구는 `kind=no_motion_diagnostic`, `passed=false`, `status=timestamp_sync_unverified` JSON만 남기고 exit code 2로 종료한다. 이 파일을 `evidence.no_motion`에 넣어도 readiness는 열리지 않는다. PTP/hardware timestamp calibration과 원자적 state association이 별도 검증되기 전에는 `kind=no_motion`, `passed=true` 발행이 코드에서 차단된다.

`Deploy_DryRun.py`는 명령을 전송하지 않는 경로지만, 현 release에서는 timestamp P0 때문에 config·release·SDK·socket을 읽기 전 unconditional guard에서 exit 2로 종료한다. 아래 두 번째 명령은 향후 별도 승인으로 이 guard가 교체된 뒤 사용할 인터페이스 예시다. 현재 하드웨어 통신은 첫 번째 canonical no-motion 진단만 허용한다.

```powershell
python deployment/verify_deployment.py --mode no-motion --bundle-root . --real-config configs/real.local.yaml --duration-s 600

python Phase4/Deploy_DryRun.py --real-config configs/real.local.yaml `
  --release-dir exported --target-pos=-0.50,0.0,-0.0125
```

**`motion_safety` hard P0:** 이번 저장소 범위에는 `motion_safety` 증거를 안전하게 수집하거나 승인·발행하는 collector가 없다. 따라서 현재 release는 `Phase4/Deploy_Real.py`뿐 아니라 RTDE/그리퍼/캘리브레이션 commissioning 도구의 모든 command-capable 모드를 hard-block하며 실제 모션을 승인하지 않는다. `configs/real.local.yaml`의 `evidence.motion_safety`와 `evidence.no_motion`을 비워 둔다. JSON을 손으로 만들거나 `passed: true`, 경로, SHA-256을 임의로 채우는 것은 증거가 아니며, 형식상 유효해 보이는 evidence만으로도 release-level 차단을 해제할 수 없다. receive-only no-motion **진단**만 허용된다.

`Deploy_Real.py`는 config/evidence/SDK를 읽거나 socket을 열기 **전** release-level unconditional guard에서 즉시 종료한다. command-capable commissioning 도구도 같은 순서로 RTDE control/DO interface 이전에 종료해야 한다. receive-only verifier가 수집한 diagnostic은 무동작 통신 자료일 뿐 유효 evidence나 현 release의 motion authorization이 아니며, 미래의 timestamp 동기화 설계·안전 검토 입력으로만 사용한다.

향후 독립적으로 검토·승인된 human-operated collector와 현장 시험 계획이 생기면, `motion_safety` 증거는 최소한 다음을 실측해야 한다.

- joint position limit margin과 manipulability margin이 양수인지, self/cell collision 검증을 통과했는지
- TCP acceleration·jerk와 joint torque의 실측값이 승인 한계 이내인지
- command acknowledgement failure가 0인지, stop distance/time이 승인 한계 이내인지
- 저속 시험이 **3단계 이상**이고 `max_tested_speed_scale <= 0.25`인지
- fault injection이 5건 이상이고 fault 이후 명령이 0인지

이 측정은 `schema_version`, `kind=motion_safety`, `contract_sha256`, 유효기간과 release/bundle 결합 hash를 가진 증거로 승인 발행되어야 한다. 이 사전 증거가 닫히지 않은 상태에서 정책 기반 저속 motion 자체를 증거 수집 수단으로 사용하지 않는다. 모든 P0가 별도 승인으로 닫힌 뒤의 속도 확대는 새로운 현장 위험 평가와 단계별 사람 승인을 받는 후속 절차이며, 현재 runbook의 일부가 아니다. 실제 CLI는 그때도 실행 직전 `--help`와 대조한다.

## 5. 배포 전 필수 게이트

앞 게이트가 실패하면 뒤 게이트를 실행하지 않는다.

| 순서 | 증거/검증 | 최소 통과 기준 |
|---:|---|---|
| 0 | repository/release integrity | 순수 테스트 PASS, bundle drift 0, Student-only manifest와 모든 SHA-256 일치 |
| 1 | RTDE receive-only | 실제 IP, robot state 정상, 명령 socket을 열지 않은 상태 확인 |
| 2 | 2FG7 driver | DO5/DO6 외 출력 불변, pulse cancel/비블로킹/안전 출력 증거 |
| 3 | joint/base parity | slope 오차 ≤5%, offset ≤0.01 rad, cross-axis ≤1°, base-frame round trip |
| 4 | D435i | serial 고정, RGB-depth align, monotonic timestamp, stale/disconnect 0 |
| 5 | real TCP command | controller TCP=flange zero, `servoL` 실제 signature, command↔base frame 검증 |
| 6 | TCP offset | 실측 calibration, residual ≤2 mm |
| 7 | hand-eye | 보정 후 재검증, median ≤10 mm, p95 ≤20 mm |
| 8 | scene registration | 작업면/target base-frame residual ≤20 mm |
| 9 | ObsParity | layout/단위/분포/invalid-depth/gripper semantics 일치 |
| 10 | latency | DR 범위 안, stale 0, 연속 overrun 0, overrun 비율 ≤1% |
| 11 | force feedback | 센서 제공/프레임 확인, read failure 0, 50 N trip 경로 검증 |
| 12 | workspace/e-stop | 셀 실측, home path collision 확인, latch 및 disconnect fault, stop 후 명령 0 |
| 13 | no-motion rehearsal | release/dependency/offline test 증거와 실제 관측 dump 검토 |
| 14 | `motion_safety` 사전 증거 | 승인된 human-operated 절차로 3단계 이상, 최대 scale ≤0.25, collision/limit/ack/stop/fault 측정 통과; 현재 collector·승인 발행자와 release-level motion authorization이 없어 **P0 BLOCKED** |

Audit P1 계약 두 항목도 scratch/release 전에 코드와 테스트로 닫는다.

- sim/real TCP quaternion은 모두 `wxyz`, unit norm, canonical sign `w>=0`이어야 한다. 현재 `common/calibration.py`, `real/real_robot.py`와 순수 회귀 테스트에 구현돼 있으나, 실제 ObsParity에서도 동일 표현을 확인해야 한다. 코드·테스트가 drift하면 scratch를 시작하지 않는다.
- 채택할 seed-45 held-out rollout은 실제로 sim adapter에 전달 완료된 명령을 분모로 삼아 arm 6채널별 full-pipeline safety clamp fraction을 계산한다. mapper 원출력과 SafetyGuard 출력은 하나의 12D DelayBuffer 입력으로 묶어 `action_delay_steps`가 0/1 중 무엇을 선택해도 같은 command lineage를 회수하며, count는 mapper→SafetyGuard와 SafetyGuard→actual-applied 변화의 채널별 합집합이다. 각 차이는 고정 허용오차 `1e-12` 초과일 때만 count하며 분모는 최소 10,000, 각 채널은 엄격히 `<1%`여야 한다. 하나라도 정확히 1%이거나 넘으면 같은 clamp를 training 경로에 넣고 Teacher/Student를 다시 학습한다. 문서나 manifest의 수치만 고치는 것으로 이 게이트나 배포 승인이 열리지 않는다.

추가 차단 조건:

- `configs/calib/tcp_offset.yaml`은 현재 sim chain 값이며 `meta.calibrated: false`다.
- hand-eye v7은 27-view ChArUco 결과(RMS 2.01 mm)를 담지만 병진 기계 보정 후 실기 재검증이 필요하다.
- 2FG7 폭 feedback 부재가 actor 관측 의미와 해결되지 않았다.
- `policy.pt`, `obs_spec.json`, manifest가 없거나 서로 다른 config를 hash하면 거부한다.
- Windows 20 Hz timer jitter, camera/RTDE disconnect, stale frame, force frame, 실제 e-stop stop distance가 미검증이면 거부한다.
- joint position limit, singularity/manipulability, self/cell collision, TCP acceleration·jerk·torque, command acknowledgement 안전 게이트는 이번 코드 범위에서 아직 구현·입증되지 않았다. validator는 `motion_safety` 증거를 강제하지만 신뢰 가능한 collector·승인 발행 경로는 없으므로, 이 P0 묶음과 fault evidence가 닫히기 전에는 저속 motion ladder를 포함한 어떤 autonomous motion도 승인하지 않는다.
- 향후 command SDK를 열기 전에는 RTDE control/IO 호출이 멈췄을 때도 e-stop thread가 같은 command lock에서 무기한 대기하지 않는 독립·bounded stop channel과 실제 wedge fault-injection 증거가 필요하다. 특히 이미 진입한 2FG7 HIGH SDK 호출은 timeout 뒤 늦게 물리 출력될 수 있고 반환 후 safe-low 시도는 독립 정지 채널이 아니다. 현 release는 이 물리 보장을 구현했다고 주장하지 않으며 command-capable 객체 생성 자체를 먼저 차단한다.
- 소프트웨어 stop은 물리 e-stop을 대체하지 않는다.

## 6. F01–F81 실패 원장

아래는 재시도 방지용 압축 색인이다. 삭제된 장문 보고서의 상세는 Git history에서 복구할 수 있다. “성공” 표기는 당시 계약에서의 관찰이며 현행 checkpoint의 배포 승인을 뜻하지 않는다.

| ID | 실패/판정 | 재발 방지 핵심 |
|---|---|---|
| F01 | 밀기-not-집기를 성공으로 오판 | 양 finger contact, lift latch, release/placed를 분리 계측 |
| F02 | 2 cm cube 기각 | 실제 2FG7 폐합 범위와 물체 치수를 먼저 대조 |
| F03 | TCP placeholder가 43.5 mm 편차 | TCP를 캘리브 단일 소스로 만들고 residual gate |
| F04 | RelativeJointPositionAction으로 holding 실패 | action 의미를 task 계약에 맞춰 선택 |
| F05 | 그리퍼 placement/collider/dynamics 3중 결함 | RL 전에 결정론적 grasp physics test |
| F06 | release reward farming 딜레마 | 지급 조건·지속 시간·총 예산을 함께 검증 |
| F07 | entropy 0.006→0.015 상향 실패 | 탐험 부족 가설을 단일 지표로 단정 금지 |
| F08 | home pose 변경 연속 실패 | 카메라 가시성·FK·안전 AABB를 동시에 검사 |
| F09 | TCW1 질량/충돌 반영이 학습 파괴 | 기하 86.5 mm만 반영, 시각 프록시는 무충돌 |
| F10 | distillation LR 오진과 step starvation | 샘플 수/optimizer 기여를 분리 측정 |
| F11 | pixel MLP loss 수렴, 능력 0 | 공간 encoder 없는 pixel actor 사용 금지 |
| F12 | 고정 카메라에서 가림/소형 물체 | eye-in-hand와 실제 FOV로 전환 |
| F13 | GRU 1차 hidden 오염 | recurrent epoch/sequence state 계약 검증 |
| F14 | RGB-D r1 협각 FOV, 큐브 불가시 | color FOV와 full-frame visibility 정량화 |
| F15 | RGB-D r2 색 DR 이중 부담 | VRAM/sample budget과 appearance DR 분리 |
| F16 | GRU 2차 시도도 실패 | 반복 구조는 명확한 병목 근거 없이는 재도입 금지 |
| F17 | NuRec 소수-env finetune 실패 | 4→2→1 env 경로는 정책 학습 본선으로 쓰지 않음 |
| F18 | r7 연장 수확 소진 | 단순 연장을 개선으로 오인하지 않음 |
| F19 | 112² h2 연장에서 collapse | 해상도 변경 checkpoint 계보 혼용 금지 |
| F20 | target-distance 역 curriculum collapse | actor 관측 target 분포 shift를 계약 변경으로 취급 |
| F21 | scratch TGRL bootstrap 실패 | sparse vision RL을 Teacher 없이 시작하지 않음 |
| F22 | encoder-only warm start 불충분 | random head가 sparse task를 재학습할 수 있다고 가정 금지 |
| F23 | vision PPO action std 1.0 고착 | 행동 분포를 reward와 함께 감시 |
| F24 | fixed-β TGRL이 증류 정책 degrade | distillation 성능을 PPO가 보존하는지 별도 gate |
| F25 | Teacher v2 진동성 catastrophic collapse | value/EV/action/std 조기 감시와 rollback |
| F26 | 단일 eval run 비재현 | 독립 seed와 Wilson CI로 채택 |
| F27 | b6 DrQ 효과가 confound | 한 실험에 한 변수만 변경 |
| F28 | Teacher v3 collapse 재발 | 회복 구간이 있어도 최종 안정성 검증 |
| F29 | sim gripper와 실제 2FG7 불일치 | 자산/데이터시트/실물 삼자 대조 |
| F30 | pregrasp opening 하드코딩 | 실제 finger geometry에서 파생 |
| F31 | 파지력 부족 오진 | collider/침투/강성/마찰을 RL과 분리 측정 |
| F32 | 2FG7 규격 정정과 tool-chain 2단 누락 | 전체 flange→TCP 체인을 원장화 |
| F33 | release_bias가 파지 포기를 보상 | 무파지 상태 지급 경로 차단 |
| F34 | contact gate가 release 다리를 너무 짧게 절단 | 시계열 지급 길이 계측 |
| F35 | goal_fine std 축소안 기각 | 적용 전 산술/도달성 검사 |
| F36 | lift가 final descent 예외에서 누락 | 단계별 reward gate truth table |
| F37 | placed 정지 임계가 접촉 지터보다 낮음 | 동일 장면/물체에서 noise floor 측정 |
| F38 | open-close 반복으로 grace 재수확 | 상태 latch와 bounded budget |
| F39 | 예산이 contact만으로 소진 | 실제 지급액과 소비를 결합 |
| F40 | 비대표 큐브의 지터 진단 | 측정 맥락과 학습 맥락 일치 테스트 |
| F41 | release grace가 초기 grasp를 밀어냄 | curriculum stage별 지급 활성화 |
| F42 | goal_fine velocity damping안 기각 | 병목 진단과 처방 효과를 분리 |
| F43 | arm action penalty로 정지는 개선, 접근 약화 | penalty 효과를 단계별로 평가 |
| F44 | resume이 curriculum을 되감음 | global step offset 회귀 테스트 |
| F45 | gripper까지 action penalty를 준다는 오진/정정 | arm-only penalty scope를 정적 테스트 |
| F46 | binary gripper action이 음의 포화로 잠김 | action magnitude 복원력/개방 확률 감시 |
| F47 | absorbing per-step placed reward가 grasp를 잠식 | 총 지급 예산으로 제한 |
| F48 | iteration 압축으로 정제 sample 부족 | curriculum 완료 뒤 refinement budget 확보 |
| F49 | 안 놓는 개체에는 release gradient 부재 | 상류 release 기회에도 smooth signal 제공 |
| F50 | placed 3조건 강화가 달성 불가능 | 조건별 달성률 probe 후 결합 |
| F51 | 벨트 충돌 뒤 큐브 재포획 | TCP z floor와 release 후 arm trajectory 검사 |
| F52 | release 상태항을 가산해 holding 보상 | 기존 지급과 대체/가산을 명시 |
| F53 | 지급과 예산 소모 분리로 farming 부활 | 하나의 `grant_and_consume` 연산 사용 |
| F54 | h5py preload workaround가 GUI를 죽임 | import workaround 방향과 실제 GUI smoke 대조 |
| F55 | 실제 탐색 장벽: gripper 속도 3배 오설정 | 물리 actuator 속도를 실물/자산과 대조 |
| F56 | open-attempt 임계가 음수 action을 허용 | action 부호/경계 property test |
| F57 | F56 수정이 나쁜 band를 넓히고 sigmoid 제거 | 성공 run의 복원 메커니즘을 제거하지 않음 |
| F58 | 목표 역 curriculum이 지지면 밖을 샘플 | target은 작업면·안전·도달영역 교집합 |
| F59 | P9 3조건 강화로 bootstrap 차단 | 완화→계측→강화 순서 |
| F60 | F54 검증 결과를 오귀속 | 변경 전후를 같은 process/context에서 비교 |
| F61 | 독립 카드 없이 F63 분석에만 남은 weight 20→50 가설 | 기각된 가중치 조정; 번호 공백을 숨기지 않음 |
| F62 | placed 속도 0.02→0.10만으로 해결 실패 | 임계값보다 release 후 trajectory를 관찰 |
| F63 | 릴리스 순간 팔이 0.12~0.21 m/s로 큐브를 쓸어냄 | 3-step placed DoneTerm과 terminal bonus, trajectory gate |
| F64 | F57이 sigmoid 복원력을 제거해 release 채널을 죽임 | sigmoid payout 복원 + 지급액 비례 budget 소비 + 2FG7 0.020 계약으로 scratch; 구 checkpoint 재사용 금지 |
| F65 | F64의 복원력이 정책을 지급 최대 지점(경계)으로 밀어, 비례 소비 안전장치의 전제를 스스로 깨뜨림 | `release_bias` w30 → **0**; 릴리스 채널을 payout에서 penalty로 전환하고 placed(w50)+terminal bonus(w1600)에 맡김 |
| F66 | 릴리스는 부호 전용 binary 액션이라 policy-gradient가 학습하지 못한다(F49·F52·F53·F56·F57·F64가 같은 항의 상수를 다섯 번 고쳐 전패) | 릴리스를 **실행 primitive**로 이관 — `common/release_supervisor.py`, TCP 정지 트리거 + 팔 정지. 배포등급 0.0% → **67.6%** |
| F67 | “binary 액션은 닫으면 유지된다”는 **틀린 전제**로 폐합 시점까지 탈취 → teacher 학습 파괴(grasp 0.031 vs F66 0.797) | 기각·철회. 업스트림은 매 스텝 부호로 재계산하며 latch가 없다 |
| F68 | 폐합을 **제약**으로 바꿨으나, 초판이 제약을 상태 지정으로 써서 들어올리는 도중 강제 개방; `lift_z_m` 절대높이는 정상 운반에서 도달 불가(0.01%)였다 | 높이 제약은 **open→close 전이만**, carry latch는 **폐합 이후 상대 상승량**으로. 상수 전수 실측 채택 |
| F69 | 증류 병목을 “모방 불가능한 특권 신호”로 보고 `contact_forces`를 뺐다(42D→40D). teacher는 99.95%로 멀쩡했으나 **student가 더 나빠졌다** — 커리큘럼 이유 후 동일 iter에서 grasp 보상 0.32배, Loss/behavior 1.89배 | **기각·되돌림.** 가설 반증: 특권 채널 제거가 곧 모방 가능성은 아니다. 접촉 없는 teacher는 정밀도 의존적이 되어 오차 있는 student에게 더 어렵다 |
| F81 | ★적대 리뷰(27 agents, 21건 생존)로 증류 3연속 실패의 잔존 원인 일괄 규명★ — ①teacher 관측 `last_action`이 **raw 정책 출력**이라 무시 채널 표류값(±7)이 자유 메모리로 되먹임되고, 증류(DAgger)에서는 student가 env를 구동해 teacher가 **분포 밖에서 질의**됨(타깃 오염 33~86% 실측) ②pregrasp EMA 게이트(0.30)를 MSE student는 영원히 못 열어 run 대부분을 90% 보조로 소모(평가는 보조 0) + deadline 분기가 호출마다 0.05 급락 + 리셋이 수퍼바이저 상태를 물리 손가락과 불일치로 초기화 ③실기 경로에 그리퍼 수퍼바이저 등가물 부재 → **큐브 근처도 아닌데 폐합**(실기 실증) ④`--action-scale`이 cfg_hash를 깨 정책 로드 거부 ⑤증류 grad clip 부재+12청크 sum 누적 | ①관측을 **실행 가능 정규화**(clamp ±1 + 기울임 0 — F80 타깃 변환과 문자 동일)로 ②증류 env는 `min_grasp_ema=0` + deadline은 자격만 부여 + 첫 스텝 물리 시딩 ③rollout에 **같은 함수·같은 설정**의 상태기계 이식 ④축소 cfg는 mapper 전용 ⑤`max_grad_norm=1.0`. ★판정 기준 교정: F80 클램프 후 Loss/behavior '무학습 바닥'≈0.5 — 절대값이 아니라 바닥 대비 하락+grasp 상승으로 판정★ |
| F80 | F79 후에도 grasp 하락(0.186→0.098) — teacher 액션이 무계라 **MSE를 grip(\|a\|≈12)이 85% 독식**, 병진 3채널 합 3.5%(dz 0.3%). student가 그리퍼 개폐만 배우고 '어디로'를 못 배움 | 증류 타깃 **±1 클램프**(액션 1.0=4 m/s로 이미 안전상한 16배 = 그 밖은 포화·정보 없음). 병진 기여율 3.5→52.3% |
| F79 | F78 수직파지가 액션 3·4를 env-무시 채널로 만들자 teacher의 그 채널이 gradient 없이 표류(사용 채널의 3.1배) — 증류가 그 노이즈까지 회귀시켜 Loss/behavior 0.83 | 환경이 무시하는 채널은 **타깃에서도 제외**(evaluate 오버라이드로 3·4=0). 마스크는 env와 같은 설정 키·같은 슬라이스임을 테스트로 강제 |
| F74~F78 | (배포 인프라·실기 정합) 모션 인가 scope화(F74)·비릴리스 실험 export 레인(F75)·배포 PC 전수 점검기(F76)·실기 실측 홈 포즈+FOV 정합 스폰존 축소(F77)·수직 파지 제약(F78) | 각 커밋 메시지 참조 (ec2523c·86f1942·7df7cce·c2f3ece·4137fc9) |
| F72 | `BINPICK_AUX=1`(물체위치 예측 보조손실)로 학습한 student는 **채택이 원천 불가**였다 — 채택 평가가 위조 방지로 `BINPICK_` 접두 환경변수를 전부 지워 러너가 다른 알고리즘으로 재구성해 ckpt 로드 실패. 학습은 되는데 배포에 도달 못 하는 카드 | 알고리즘은 실행 시 선택지가 아니라 **학습된 사실**이다 → 봉인된 `params/agent.yaml`에서 해시 사슬로 복원(`receipt_sha256`→manifest→artifacts→agent.yaml). 소독기는 그대로 두어 위조 방지 유지 |
| F71 | `carry_latch_update`의 계약은 “폐합 시점 대비 상승량”인데 **두 호출부 모두 절대 TCP z를 넘겼다** — `z_at_close`는 할당·리셋만 되고 기록·사용이 없는 죽은 상태. F68-c는 문서·설정·README에만 존재했고 함수 단위 테스트는 전부 통과했다 | 폐합 **전이**에서 기준 높이를 기록해 차이를 넘기도록 배선. **호출부 형태를 AST로 검사**하는 회귀 테스트 추가(의미 테스트만으로는 배선 위반을 못 잡는다) |
| F70 | teacher 선발이 **순간 `Train/mean_reward`**여서 조기종료 역상관의 정점(점화 직전 5200 부근)만 top-5로 — 후보 전원 74~81%로 채택 전멸. F68은 점화(7000)와 정점이 우연히 겹쳐 살았을 뿐 | 선발을 **게이트가 재는 지표**(placed_success trailing-200 median + 최종 고정 + separation 500)로. 봉인 계약상 커밋이 구 run 채택을 막아 teacher 재학습 |

F64의 핵심은 “닫힌 action에도 작은 smooth payout을 주는 것”과 “그 지급액만큼 정확히 budget을 소모하는 것”을 분리하지 않는 데 있다. 닫힘 지급을 0으로 만들면 binary actuator에는 개방 경계로 돌아갈 물리적 복원력이 없고, 소비를 step당 1로 하면 작은 payout이 20 step에 소진돼 F39가 재발한다.

### F65 — 릴리스 채널을 지급에서 페널티로 (2026-08-15)

F64 run(`2026-08-15_05-09-21-220238`)을 iter 9,705에서 중단하고 `model_9700.pt`를 프로브했다. 학습과 동일한 판정 함수, 각 256 에피소드다.

| 지표 | policy | force_open_smart (팔은 같은 정책, 개방만 강제) |
|---|---:|---:|
| latch (집어 듦) | 96.9% | 96.1% |
| **zone** (타겟 3 cm 이내 운반) | **87.5%** | 93.4% |
| zone & settled | 66.8% | 85.5% |
| **zone & open** | **0.00%** | 26.6% |
| **placed** | **0.39%** | **64.8%** |
| 타겟존 내 그리퍼 액션 최댓값 | **−0.026** | +1.0 |

강제 개방 후 placed까지 중앙값 **4 step**이다. 환경·판정·기하·운반은 정상이고 결함은 릴리스 채널 하나다.

**원인.** F64가 넣은 두 장치가 서로를 무력화한다. ①복원력(닫힘 구간 sigmoid 지급, 0에 가까울수록 큼)이 액션을 개방 경계로 민다. ②안전장치(소모 = 지급액 비례)는 정책이 깊은 닫힘에 머문다는 전제에서만 성립한다. ①이 성공하는 순간 ②의 전제가 깨진다. 실측 −0.026에서 지급은 0.2392/step으로 F64 docstring이 가정한 −1.7(0.0034)의 **139배**이고, 예산 20이 **84 step**에 소진된다. 검산: 이론 상한 20×30×0.05 = 30.0 vs 실측 `release_bias` 29.25 — **예산의 97.5%를 그리퍼를 한 번도 열지 않고 수확**했다. `grant_and_consume`은 “지급>0 ∧ 소모=0” 밴드를 정확히 없앴지만, **예산은 게이트가 아니라 수량**이라 닫힘 수확을 느리게 할 뿐 막지 못한다.

**일반 규칙.** 액션만의 함수인 지급은 어떤 형태든 수확 가능하다. 0으로 두면 gradient가 죽고(F57), 0보다 크게 두면 경계에서 farming한다(F49·F64). F52·F53·F56·F57·F64가 이 한 항의 상수를 다섯 번 고쳐 모두 실패했다 — 지급식 조정으로는 닫히지 않는다.

**해소.** 릴리스 채널을 payout에서 **penalty**로 전환한다. 페널티는 아래로 유계라 수확할 대상이 없다. 복원력은 이미 있는 `gripper_action` L2(w −0.05)가 담당한다(실측 rms|a| ≈ 0.57 — F46의 중앙 −9.9 표류와 무관). 릴리스 구동은 `placed`(w50) + `place_success_bonus`(w1600)이며, 종단 보너스 1600×0.05 = **80**이 잔여 hold 스트림(1.363/step, γ=0.98 할인 최대 54.6)을 모든 시점에서 압도한다. 문헌 정합: ManiSkill3 StackCube와 IsaacLab Franka Lift 모두 binary 그리퍼 액션을 shaping하지 않고 객체 중심 보상 + 종단 성공으로 릴리스를 학습시킨다.

**F63은 무죄다.** 성공 종료의 EV는 모든 시점에서 release 우위이므로 종료가 원인이 아니다.

**조기 게이트.** scratch에서 **iter 3,000에 `zone&open` > 5%**. 현재 0.00%라 즉시 갈린다. 미달이면 페널티 전환 방향이 틀린 것이므로 지급식 재도입이 아니라 재분석으로 간다.

**결과.** 페널티 전환만으로는 열리지 않았다 — F65 run은 iter 6,023에서 `placed_success` 0.05%로 평탄해 중단했고, `rms|a|`가 1.39 → 2.22로 단조 상승했다. 즉 “기존 L2 페널티가 복원력을 담당한다”는 전제가 틀렸다(F66에서 그 액션을 경계에 붙잡고 있던 것은 L2가 아니라 `release_bias`의 지급 경사였다). 릴리스는 보상 설계가 아니라 **실행 구조**의 문제였고, 그것이 F66이다.

### F66 — 릴리스를 학습 대상에서 실행 primitive로 (2026-08-16)

`BinaryJointPositionAction`은 액션의 **부호만** 본다. 그래서 그리퍼 차원은 크기에 gradient가 없고, 어떤 지급을 붙여도 정책이 경계에서 수확한다. F49·F52·F53·F56·F57·F64가 이 한 항의 상수를 다섯 번 고쳐 전부 실패했고, F65에서 지급을 없애자 이번엔 액션이 표류했다. 상수 조정으로 닫히는 문제가 아니다.

프로브 실측(`model_6000`, 각 256 에피소드, 성공종료 비활성, ≥10스텝 = 배포 등급):

| 조건 | 배포등급 | 트리거 배포 가능 |
|---|---:|---|
| 정책이 스스로 개방 | 0.0% | — |
| 강제 개방 + 팔 계속 이동 | 11.7% | — |
| 강제 개방 + **팔 정지** | 63.7% | ✗ 큐브 pose = sim 전용 |
| TCP-xy 트리거 + 팔 정지 | 24.2% | ✓ (파지 오프셋으로 열위) |
| **TCP 정지 트리거 + 팔 정지** | **67.6%** | ✓ **채택** |

두 가지가 결정적이다. ①**릴리스 후 팔을 세워야 한다** — 움직이는 팔이 자기가 놓은 큐브를 쳐낸다(11.7% → 63.7%). ②트리거는 “큐브가 타겟 위인가”가 아니라 **“정책이 배치를 마치고 멈췄는가”**를 물어야 한다. 어디에 놓을지는 정책이 정하고, 필요한 신호는 TCP pose·속도뿐이라 실기 RTDE/FK로 충족된다(큐브 pose 추정기 불필요).

구현은 `common/release_supervisor.py`가 sim/real 단일 소스이며, 액션 항이 스텝당 한 번 판정하고 그리퍼 항이 소비한다. 그리퍼 override는 **지연 뒤에** 적용한다 — 수퍼바이저는 실기 제어루프의 primitive라 정책 액션 지연 DR의 대상이 아니다.

**계측 함정(중요).** IsaacLab은 종료 시 auto-reset하므로 3-step `placed_success` DoneTerm이 켜져 있으면 프로브가 관측하는 연속 스텝은 **2에서 멈춘다**. `max_streak ≥ 3`을 물리적 실패로 읽으면 안 되고, 성공은 `max_streak ≥ 2 ∧ ep_steps < 100`으로 판정한다. 배포 등급(10스텝) 지속성은 `--no-success-term`으로 종료를 끄고 따로 잰다.

**결과.** F66 teacher는 채택 평가에서 grasp 99.90 / carry 99.85 / **placed 99.85%**(Wilson 하한 99.57%)로 후보 5개 전원 통과했다(`model_12200.pt`).

### F67 — 폐합 시점까지 탈취한 오진 (2026-08-18, 기각)

F66 teacher에서 증류한 RGB-D student가 무보조 7,058 iteration 동안 0.15~0.24%로 평탄했다(iter 12,072 중단). 원인을 순서대로 실측으로 잘랐다.

- **depth-blind → 기각**: depth 공간 std 0.241 / zero 0.0%, rgb·proprio·tcp_pose·target_pos 정상
- **큐브가 시야 밖 → 기각**: 파지 전 가시율 75.1%(중앙 77.8%), 미가시 에피소드 0.0%, 처음 보이기까지 중앙 0 스텝
- **조기 폐합 → 확정**: student는 `reach` 0.734(teacher 0.477)로 근접은 더 잘하면서 `close_bias` 0.802(teacher 0.134)로 6배 자주 닫고, `grasp`는 0.155(teacher 0.693)에 그친다. **전환율 0.19 vs 5.16 = 27배 열위.** 닫힌 그리퍼는 내경 34mm이고 큐브는 50mm라 감쌀 수 없어 밀어낸다.

여기까지는 옳았으나, 처방을 **“폐합 시점도 수퍼바이저가 정한다”**로 잡은 것이 오진이었다. 근거로 삼은 “한번 닫으면 유지되므로 조기 폐합은 되돌릴 수 없다”가 **사실이 아니다** — `binary_joint_actions.py:131-142`는 매 스텝 `actions < 0`으로 명령을 재계산하며 latch가 없다. 소스를 확인하지 않고 두 번의 학습(5.2 GPU-시간)을 걸었다.

결과는 teacher 학습 파괴였다(`grasp` 실측):

| 구간 | F67 | F67-c(재시도 허용) | F66 |
|---|---:|---:|---:|
| 0–250 | 0.028 | 0.031 | **0.797** |
| 500–750 | 0.109 | 0.129 | **1.696** |
| 1500–1750 | 0.370 | 0.338 | 1.530 |

“내려가서 **멈춰야** 폐합”이라는 하드 전제조건이 탐색을 막아 F66의 20%에 머물렀고, 단방향 latch를 재시도 가능하게 고쳐도(F67-c) 동일했다 = latch는 원인이 아니었다.

**교훈.** 업스트림 동작을 전제로 설계할 때는 소스를 먼저 읽는다. 그리고 사전검증이 **판정 불가**일 때(F67은 정책의 할 일 자체를 바꾸므로 F66 정책으로는 검증할 수 없었다) 그 사실을 인지하고 별도 근거를 세운다.

### F69 — 특권 관측에서 contact_forces 제거 (2026-08-22, 42D→40D) — **기각·되돌림 (2026-08-24)**

**가설.** 증류 병목이 *모방 불가능한 특권 신호*라고 봤다. F68 teacher에서 증류한 student는 grasp 88.5%를 내지만 ①파지→②배달 latch에서 20.1%p를 잃었고(teacher 0%p), hold 유지율도 28.8% vs 49.7%였다. 접촉력은 위치와 달리 시각 신호가 아니므로 RGB-D student가 복원할 수 없다 ⇒ 특권 관측을 student가 추정 가능한 것으로 제한하면 모방 가능해질 것이다.

**teacher 쪽은 성립했다.** 40D teacher는 42D보다 1000 iter 빨리 점화(iter 6000에 placed 0.81 vs 0.14)했고 채택 평가 **99.95%**(2047/2048)로 42D 채택본(99.76%)을 넘었다. 접촉 관측이 teacher 성능에 필수가 아님은 확실하다.

**student 쪽에서 반증됐다.** 40D teacher로 증류하자 기준선(37.9% student를 낸 42D 증류)보다 **전 지표에서 뒤처졌다.** 커리큘럼 이유 시점이 완전히 동일한(0.900→0.449→0.000) iter 6000 시점 비교:

| | 기준선(42D) | 40D | 비율 |
|---|---:|---:|---:|
| `Episode_Reward/reach` | 0.755 | 0.740 | 0.98 |
| `Episode_Reward/grasp` | 0.429 | 0.139 | **0.32** |
| `Episode_Reward/lift` | 0.647 | 0.285 | 0.44 |
| `Loss/behavior` | 0.221 | 0.418 | **1.89** |

접근은 동등한데 **파지에서 무너지고**, 행동 모방 오차 자체가 1.9배다. 기준선의 6000→19800 성장배수(1.75배)를 그대로 적용해도 최종은 기준선의 39%에 그쳐, 24시간을 더 태울 기대값이 음수라 iter 6000에서 중단했다.

**교훈.** *특권 채널을 빼면 모방 가능해진다*는 추론은 틀렸다. 접촉 피드백이 없는 teacher는 정확한 물체 위치에 의존해 **정밀도 의존적**인 정책을 학습하고, 그런 정책은 위치 추정에 오차가 있는 student에게 **더** 어렵다. 접촉을 관측하던 teacher의 정책이 오히려 오차에 관대했다. 특권 정보 설계는 “student가 복원 가능한가”만이 아니라 **“그 정보를 뺀 teacher가 어떤 성질의 정책을 학습하는가”**까지 봐야 한다.

### F72 — aux 카드가 배포에 도달할 수 없었다 (2026-08-25)

`BINPICK_AUX=1`은 학습에서 알고리즘을 `Distillation` → `DistillationAux`(물체 3D 위치 예측 보조손실 β·L_pos, DextrAH-G/RGB 근거)로 바꾼다. 그런데 채택 평가는 조건 위조를 막으려고 `sanitized_evaluation_environment`가 **`BINPICK_` 접두 환경변수를 전부 삭제**한다. 그래서 aux로 학습한 student를 평가하면 러너가 `Distillation`으로 알고리즘을 재구성해 **ckpt state_dict가 불일치**한다 = 학습은 되는데 **채택이 원천 불가**. 실제로 placed 37.9% student는 이 카드 없이 학습됐고(TensorBoard에 `Loss/aux_pos` 부재), 카드는 한 번도 배포에 도달하지 못했다.

**정정.** 알고리즘 클래스는 실행 시 선택지가 아니라 **학습된 사실**이다. 학습 run은 이미 `params/agent.yaml`에 `algorithm.class_name`을 남기고 schema-2 receipt가 그 파일을 sha256으로 봉인한다. 평가는 그 봉인된 값을 읽어 agent cfg를 덮어쓴다 — 소독기는 **그대로 두어** 위조 방지를 유지하고, 값은 해시 사슬로 인증한다: `receipt_sha256` → `run_manifest.json` → `artifacts.parameters` → `params/agent.yaml` → `class_name`. 환경변수 방식보다 **provenance가 강해진다**(평가가 "학습된 그대로"에 묶이고 주변 환경으로 바꿀 수 없다).

**교훈.** 실험 카드를 환경변수로 켜는 설계는, 그 변수를 지우는 보안 경계와 만나면 **조용히 배포 불가가 된다**. 학습 결과에 영향을 주는 스위치는 산출물에 기록되고 그 기록에서 복원되어야 한다.

### F71 — F68-c가 배선되지 않아 파지 직후 릴리스가 터졌다 (2026-08-24)

`carry_latch_update(carried, gripper_closed, tcp_rise, cfg)`의 docstring은 `tcp_rise`가 **폐합 시점 대비 상승량**이며 “호출측이 폐합 전이에서 기준값을 기록해야 한다”고 명시한다. 그런데 **존재하는 두 호출부가 모두 절대 `tcp[:, 2]`를 넘기고 있었다**(`delayed_actions.py`, `BinPick_PlacedProbe.py`). 상태 `z_at_close`는 할당(45행)과 리셋(294행)만 있고 **기록도 사용도 되지 않는 죽은 변수**였다 — 즉 F68-c는 config 주석·docstring·README·테스트 이름에만 있었고 시스템은 **구 의미(절대 높이)로 동작**했다.

**영향.** `lift_rise_m = 0.010`을 절대 높이로 읽으면, 폐합 허용 높이(`grasp_z_max_m = 0.040`) 안에서 닫는 순간 `z > 0.010`이 거의 항상 참이라 **들어올리기 전에** `carried`가 굳는다. 그러면 `release_decision(carried ∧ z<0.075 ∧ 속도<0.02)`이 **파지 직후 속도가 떨어지는 순간** 성립해 팔이 얼고 그리퍼가 열린다 — 집은 자리에 그대로 떨어뜨린다. teacher는 닫자마자 들어올려 속도가 유지되므로 대체로 가려지지만, 느리고 불안정한 student에서 터진다. 실측 병목과 정확히 일치한다: **파지 88.5% → 배달 래치 68.4%(−20.1%p), hold 유지율 28.8% vs teacher 49.7%.**

**왜 테스트가 못 잡았나.** `test_carry_latch_uses_relative_rise_not_absolute_height`는 **함수의 의미**만 검증한다. 사고는 **호출부가 계약을 어긴 것**이었고, 함수 테스트는 전부 통과하는데 시스템은 틀리게 동작했다. 그래서 값이 아니라 **배선의 형태**를 보는 테스트를 넣었다 — 소스 전수에서 `carry_latch_update` 호출을 AST로 찾아 세 번째 인자가 **뺄셈**인지 강제한다.

**교훈.** 계약이 “호출측이 X를 해야 한다”는 형태일 때, 단위 테스트는 그 계약을 지켜주지 않는다. 호출측 의무는 **호출측에서** 검사해야 한다.

### F70 — 순간 mean_reward 선발이 채택을 전멸시킴 (2026-08-23)

F69 teacher(최종 학습 placed 0.926)의 채택 게이트가 top-5로 **{4950, 5050, 5200, 5350, 5500}만** 평가했다 — 전부 점화 중간 구간(평가 placed 74~81% vs 게이트 95.1%)이고, 후기 안정 구간(9500+, 학습 0.92)은 아예 후보에 못 들었다.

**기전.** 성공 에피소드는 `placed_success` DoneTerm으로 3-step 만에 조기 종료되어 에피소드 보상 합이 깎인다. 그래서 `Train/mean_reward`는 성공률이 오르기 **직전**(막 점화하는 5200 부근, 108.8)에 정점을 찍고 성공률이 높아질수록 내려간다 — 순간값 선발은 이 역전의 최악 지점을 정확히 고른다. **F68이 무사했던 건 점화(7000)와 보상 정점이 우연히 겹쳤기 때문**이고, F69는 점화가 1000 iter 빨라지며 우연이 깨졌다.

**조치.** 선발 지표를 게이트가 실제로 재는 것으로 정렬 — `Episode_Termination/placed_success`의 trailing-200 median, student 선발과 같은 형태(최종 checkpoint 고정 + separation 500), 방향만 반대. F69 데이터 시뮬레이션으로 신규 선발은 {9999, 9350, 8850, 8300, 7800}(median 0.87~0.93)을 고름을 확인했다. 봉인 계약(`capture_training_context`가 git commit 비교)상 이 수정 커밋은 구 run의 채택을 영구히 막으므로 teacher를 재학습한다.

**교훈.** 선발 지표와 게이트 지표가 다르면, 그 간극은 언젠가 반드시 벌어진다. proxy가 목표와 역상관하는 구간(조기종료·시간 페널티류)이 있는지 먼저 본다.

### F68 — 폐합을 제약으로, carry latch를 상대 상승량으로 (2026-08-18)

시점 탈취가 아니라 **높이 제약**이 옳다. 언제 닫을지는 정책이 정하되(teacher는 큐브에 닿아서 닫으므로 제약이 걸리지 않는다 = 학습성 보존) 파지 높이 위에서는 닫을 수 없게 한다(student의 “닫은 채 접근”만 구조적으로 불가능).

여기서 두 결함이 더 나왔고, 둘 다 **상수를 실측 없이 정한 데서** 비롯됐다.

**① 제약을 상태 지정으로 구현 → 강제 개방.** 초판은 `closed_cmd = (carried|grasped) & ~released`였다. `carried`는 들어올려야 걸리는데 그 전에 높이 조건을 벗어나면 `grasped`가 false가 되어 **그리퍼가 강제로 열려 들어올리는 도중 큐브를 놓쳤다**. 임계를 0.080까지 올려 제약이 거의 걸리지 않는데도 배포등급이 모든 임계에서 0.00%였다는 점이 이를 드러냈다. 정정: 제약은 **open→close 전이에만** 건다(`may_close = grasp_allowed(tcp_z) | closed_prev`).

**② `lift_z_m` 절대 높이는 정상 운반에서 도달 불가.** 채택 teacher 256 에피소드 실측 — 들림 중 TCP z는 p50 **0.0319** / p90 0.0448 / p99 0.0618이고, 구 임계 0.100을 넘는 스텝은 **0.01%**뿐이다. 그런데도 F66에서 latch가 걸린 이유는 정책이 **접근 초기 높은 곳**(전 구간 p95 0.178)에서 그리퍼를 닫았기 때문이다 — 이 latch는 운반이 아니라 **조기 폐합을 추적**하고 있었다. 즉 **F66의 릴리스는 고치려던 그 결함에 의존해 동작하고 있었다.** F68이 조기 폐합을 막자 릴리스가 함께 죽었다(`zone&open` 68.75% → 0.00%).

정정: carry latch = 닫힌 채 **폐합 시점 대비 `lift_rise_m` 이상 상승**. 상대 상승량은 작업면·큐브 크기·설치 높이가 바뀌어도 의미가 유지되고 실기 FK로 그대로 얻는다.

**채택 상수(전부 실측 근거).**

| 상수 | 값 | 근거 |
|---|---:|---|
| `grasp_z_max_m` | 0.040 | 쓸기 4점(0.020/0.040/0.060/0.080) — latch가 F66 수준으로 회복되는 최소값 |
| `tcp_z_max_m` | 0.075 | 쓸기 3점(F66에서 확정) |
| `tcp_settle_mps` | 0.02 | 쓸기 2점 — 0.05는 57.8%로 열위 |
| `lift_rise_m` | 0.010 | 들림 중 TCP z 분포(p50 0.032, 실제 상승폭 1.5~2cm)보다 보수적 |

**검증.** F66 정책 그대로(F68 계약으로 학습한 적 없음) 배포등급 **63.67% → 67.19%**. 재학습에서는 `grasp` 1.511로 F66(1.455)을 넘고, 학습지표 **97.95%**로 F66 최종(87.52%) 대비 **+10.4%p**, `timeout` 92% → 1.72%로 붕괴했다.

**운영 교훈 — 학습을 중간에 끊지 않는다.** iter 20,000에서 프로세스를 끊은 checkpoint는 `train.ps1`이 정상 종료 시 봉인하는 schema-2 training receipt가 없어 **평가·채택·resume 세 경로에서 모두 거부**된다(`checkpoint has no trusted schema-2 training receipt`). 프로브로 잰 배포등급은 96.09%로 채택 기준을 넘었지만 그 가중치로 할 수 있는 일이 없어 20.6시간을 재실행했다. 중단이 필요하면 **`-MaxIterations`를 원하는 값으로 지정해 시작**한다.

### F88 — Teacher v2: 배포 가능한 관측 계약 + 인식 오차 DR + 실행 제한 정합 (2026-09-18)

**문제.** 채택 Teacher(42D, sim 100%)는 실기에서 만들 수 없는 신호(실제 finger `q`, 물체 속도, 접촉력)를 관측했고, 모듈형 배포 v1은 그것을 '쥐었다' 판정에서 유도한 상수로 채웠다. 큐브 위치는 sim에서 완벽했지만 실기 인식 모듈은 오차 p50 2 cm·검출 55~76%였고, 학습은 실행 제한(TCP 0.25 m/s) 없이 0.2 m/step 목표를 냈다. 세 갭 어느 것도 학습에서 경험한 적이 없어 실기 성공을 기대할 근거가 없었다. 추가로 hand-eye v7 기하를 실기 color FOV(69.4°×42.5°)에 대입하면 ★그리퍼 바로 아래 큐브는 어느 높이에서도 세로 FOV 밖★(광축에서 22~32°)이다 — sim TiledCamera가 등방 69.4°로 렌더해 가려져 있던 사실이다.

**조치(변인은 '배포 정합' 한 묶음).**
1. 관측 34D(`common/teacher_contract.py`): `gripper_command_state`·**추정** `object_pos`·`held` latch. sim `mdp/perception_model.py`와 실기 `common/teacher_obs.py`가 같은 상수·규칙(단일 소스)으로 만든다. critic만 sim 참값을 추가로 받는다.
2. 인식 DR(`configs/dr.yaml perception`): 에피소드별 편향 ≤1 cm, xy σ 0.3~1.5 cm, 검출 확률 0.6~1.0, 최소거리 0.17~0.25 m, FOV 밖 결측(hand-eye·TCP offset·sim.yaml camera FOV로 기하 계산). 평가(Teacher-Eval)도 같은 조건.
3. 실행 제한 정합: `task_pos_scale_m` 0.2→0.0125, `task_rot_scale_rad` 0.2→0.025(= safety.yaml 상한 × dt), sim 액션텀 `tcp_step_max_m`/`tcp_rot_step_max_rad` clamp(순서: 노름 clamp → z 바닥, 실기 SafetyGuard와 동일). 에피소드 5→10 s, γ 0.98→0.99, `place_success_bonus` 1600→3000(포기 리턴 불변식 유지).
4. 배포 조립기: `max_missed_frames_before_hold` 2→40(2 s) — 그리퍼 아래 결측은 정상이며 학습이 last_known 유지를 경험했다.

**하드웨어 메모.** 위 FOV 계산상 현재 브라켓(광축이 공구축에서 16.6° 발산)은 파지 구간 가시성이 없다. 카메라를 공구축 쪽으로 수렴하게 재장착(또는 90° 회전해 69.4° 축을 공구 방향에 두기)하고 hand-eye를 재캘리브하면 sim 인식 모델이 같은 파일을 읽으므로 재학습 없이 학습 조건이 실기와 함께 좋아진다. 재장착 전에 학습한 정책은 결측 구간을 이미 경험했으므로 재장착 후에도 유효하다.

**검증 계획.** smoke → 8192/16384 env 실측(VRAM·iter 시간) → 25k 학습 → 채택 게이트(seed 42~45 × 512, 각 ≥487, 인식 DR 포함) → Student 재증류(depth_near 0.20·카메라 종횡비 정합 계약으로).

### F89 — Student v2 관측 계약: 렌더 종횡비 정합 + depth_near 0.20 (2026-09-18, 준비 브랜치 `student-v2-contract`)

**문제.** ① Kit 렌더는 등방 픽셀이라 112² 렌더의 세로 FOV는 sim.yaml의 42.5°가 아니라 가로 69.4°였다(배포 v2 export 실측). 실기 848×480→112² 경로와 물체 크기가 세로 1.78배 어긋나 레터박스 번들이 필요했다. ② 배포 student 번들은 `depth_near_m` 0.28로 export됐지만 채택 checkpoint(model_35000)는 0.20 계보에서 학습·평가됐다(cfg_hash `e0b7812e0b054920` vs `30ed98d72c4136da`) — 배포 게이트는 번들 자신의 설정과 비교하므로 잡지 못했다.

**조치.** sim.yaml camera 112×112 → **112×63**(= 112·tan(21.25°)/tan(34.7°), 렌더 42.6°), 공유 전처리 full 리사이즈가 실기와 같은 각도 샘플링을 만든다. scene.py 해상도 가드를 FOV 종횡비 계약으로, preflight를 112×63·depth 0.20으로, export의 학습 화각을 렌더 기하에서 도출(레터박스 ≈ 항등). `depth_near_m` 0.28 → 0.20(근거: 루프 계보 commit 663c812). 구 student checkpoint·번들과 전부 비호환. Student 증류는 반드시 이 저장소의 `train.ps1 -Preset Distill`(sealed receipt)로 수행하고 Codex 루프 worktree 설정과 섞지 않는다.

**상태.** 순수 테스트만 통과(Isaac smoke는 Teacher v2 25k 학습·채택 완료 후 GPU가 비면 수행). 채택 Teacher(34D)로 증류하기 전에 이 브랜치를 `teacher-v2-deployable`에 합치고 Vision smoke를 먼저 돌린다.

### F90 — Teacher v2 run1 채택(99.56%) · 역할별 채택 게이트 · 배포 v2 export (2026-09-19)

**결과.** Teacher v2(F88, commit `dc21fe0`, 16384 env × 25k iter, seed 42) 봉인 평가(seed 42~45 × 512, 인식 오차 DR·실행 제한 적용, 보조 없음): `model_24400` placed 511/511/509/508, pooled **99.56%**(Wilson 하한 99.17%), grasp 2048/2048, carry 2040/2048 → `status: adopted`. 후보 5개 중 4개가 95% 게이트 통과(model_17450 97.5%, model_19600 99.3%, model_23400 96.8%). 학습 곡선: iter 5,000→5,500에서 placed 9%→81% 점화, 이후 완만히 상승(iter 20k 91%). 시각 관측 없이 인식 추정치만으로 이 성공률이 나왔으므로 배포 인식 모듈의 오차가 DR 범위(편향 ≤1 cm, 검출률 ≥60%) 안이면 모듈형 레인이 성립한다.

**역할별 채택 게이트(scripts/adoption.py `ROLE_GATES`).** Teacher는 종전 487/512·Wilson 0.95를 유지(증류 정답원의 기준), Student는 사용자 배포 기준(2026-09-19 지시: 성공률 75% 이상)인 384/512·Wilson 0.75를 쓴다. 게이트는 protocol에 봉인되어 receipt 검증이 같은 값을 요구한다.

**export.** `scripts/tools/export_teacher_policy.py`가 34D v2 의미로 README·manifest를 쓰고, run의 canonical adoption.json을 요약해 `teacher_adoption`·`adopted`에 기록한다(같은 commit 검증은 증류측 몫).

**provenance 메모.** 평가·채택·증류는 학습 commit과 같은 commit에서만 성립하므로, F89(Student 카메라 종횡비·depth_near 0.20)와 F90을 합친 commit에서 Teacher를 같은 레시피로 한 번 더 학습한 뒤 그 채택본으로 증류한다(run1 채택본은 모듈형 배포 번들에 쓴다).

### F91 — 그리퍼 편측 파지: 오른쪽 손가락이 수동 관절이었다 (2026-09-20)

**발견.** 사용자가 run1 학습 영상에서 "한쪽 손가락만 움직여 집는다"를 지적했다. USD를 직접 읽어 확정: `right_finger_joint`에 `PhysicsDriveAPI`가 없고 PhysX mimic API(gearing −1, 프리즘 관절인데 `rotX` 슬롯, 한계 [−0.004, 0.024])만 남아 있었다. 변환 설정(`convert_mimic_joints_to_normal_joints: true`)이 실제 산출물과 달랐다. 그래서 오른쪽 손가락은 드라이브 없이 큐브에 밀려 하한 −4 mm에 걸렸고(v1 조립기 실측 "쥐었을 때 L 12.6 / R −4.0 mm"가 이것이었다), teacher는 왼쪽 손가락으로 큐브를 오른쪽 수동 손가락에 밀어붙이는 파지를 학습했다. 실물 2FG7은 양 손가락이 같은 거리를 대칭으로 움직이는 2위치 그리퍼라 이 정책은 실기에서 성립하지 않는다(TCP가 큐브 중심에서 옆으로 치우친 채 닫으면 한쪽 손가락이 큐브 위에 얹힌다). **run1/run2 채택 teacher(99.56%)와 그 증류 student는 실기 배포 대상에서 제외한다.**

**조치.** URDF에서 `<mimic>` 제거, 오른쪽 한계를 좌측과 같은 [0, 0.020]으로 → `convert_2fg7_usd.py --fix-base --headless` 재변환 → 양 관절 모두 `PhysicsDriveAPI:linear`(2000/100/140 N), 한계 [0, 0.020] 확인 → `assets/asset_manifest.json` 크기·SHA 갱신. `diagnostics/gripper_symmetry_probe.py`(신규)가 빈 손 폐합/개방과 5 cm 큐브 파지·상승에서 |qL−qR| < 1 mm, 양측 접촉력, 큐브 유지를 판정한다. Teacher·Student를 처음부터 다시 학습한다(같은 레시피).

**교훈.** 변환기 옵션 이름을 믿지 말고 산출물(USD prim)을 읽어 확인할 것. 파지 정합은 성공률로 드러나지 않는다 — sim은 편측 파지로도 99.6%를 냈다. 영상·관절 궤적 검사를 채택 절차에 넣는다(F91 프로브).

### F92 — 리뷰 후속: 릴리스 연속 정지·그리퍼 전이 잠금·종단 보너스 산식·가시성 z 기준 (2026-09-20)

**배경.** F91 직후 8개 영역 읽기 전용 리뷰(리워드/종료, 배포 러너, 액션 정합, 인식 모델, Student, 그리퍼 체인, 관측 단위, 테스트·출처)에서 80건이 나왔다. run3 학습 계약에 걸리는 항목만 먼저 고치고(재학습은 어차피 필요), 배포 측 항목은 학습 중 worktree에서 처리한다(F93 예정).

**조치(학습 계약 변경 — sim/real 공통 모듈).**
- **릴리스 정지 판정 = 연속 3창**(RT-3). Teacher v2 스케일에서 정지 임계는 1 mm/창(0.02 m/s × 50 ms)인데 운반 전 구간 TCP z(p99 0.062)가 `tcp_z_max` 0.075 아래라 높이는 가드가 아니었다 — 속도 창 ★하나★가 비가역 릴리스를 굳혔다. 실기 servoL 추종 지연은 방향 전환에서 창 하나를 1 mm 아래로 떨어뜨릴 수 있다(운반 중 낙하). `common/release_supervisor.settle_counter_update`(느린 창 +1, 빠른 창 0)를 두고 `release_decision`이 `settle_count ≥ tcp_settle_steps(3)`를 본다. delayed_actions·rollout·PlacedProbe가 같은 함수를 쓴다.
- **그리퍼 전이 잠금 6스텝**(G1). 실기 DO-pulse 드라이버는 펄스 0.30 s 중 반전 명령을 e-stop으로 처리하는데 sim에는 잠금이 없어 정책이 경계에서 부호를 떨어도 대가가 없었다. `edge_lockout_update`가 전이 뒤 6스텝 동안 반전을 무시(직전 상태 유지; 전이 간 최소 간격 7스텝 = 350 ms > 0.30 s). 릴리스 개방도 예외 없음(펄스 보호가 목적). 테스트가 `real*.yaml pulse_s ≤ lockout × dt`를 묶는다.
- **종단 보너스 3000 → 4000**(RT-2). F63 산식이 둘 틀렸다: time_out은 부트스트랩되므로 호버 가치는 유한합이 아니라 r/(1−γ)이고, r_hover는 실측 평균 1.30이 아니라 타겟 위 저공 호버가 릴리스 없이 받는 가중치 합(reach 1+grasp 2+lift 5+goal 16+goal_fine 6 = 30 × dt = 1.50/step) → 150. 3000(=150)은 마진 1.03배였다. 4000(=200) = 1.33배. 테스트 산식도 같이 고쳤다.
- **가시성 판정 = 상면 중심·광축 z**(P4). 실기 locator는 상면 픽셀을 검출하고 RealSense Min-Z는 z 기준이다. 큐브 중심·노름 판정은 파지 직전에 ~2.5 cm + (노름−z)만큼 낙관적이었다.
- 문서 정정(G3): 액션 지연 DR은 ★팔 채널 전용★이다(그리퍼는 수퍼바이저가 루프 안에서 재계산). delayed_actions docstring·dr.yaml 주석.

**보류(근거).** P3 `bias_m` 상한 확대는 하지 않는다 — 관측 불가한 편향은 DR로 배울 수 없고(추정치가 유일한 큐브 신호) 손가락 여유(±12 mm)를 넘는 편향 DR은 학습 신호만 깎는다. 정답은 배포 PC에서 hand-eye 재캘리브·Calib_Verify. T-06 camera_pose DR(하드웨어 재장착 시 재학습 규칙으로 대체), AX-06 지연 DR(정지 큐브의 base-frame 추정은 프레임 지연과 무관; 이동 중 TCP 불일치는 xy_sigma 범위가 통계적으로 덮음), RT-4·T-13은 minor로 기록만.

**규칙.** 카메라 브라켓을 재장착하면(hand_eye.yaml 변경) Teacher도 재학습 대상이다 — 가시성 기하가 학습 분포다(F88의 '재학습 없이 재장착' 문구 철회).

### F93 — 리뷰 후속(배포 측): 시작 절차·hold·그리퍼 타이밍·리허설 그리퍼·러너 기본값 (2026-09-20)

**범위.** F92가 학습 계약을 고쳤다면 F93은 ★실기 실행 경로★를 고친다(학습 env·configs 무변경 → run3 teacher와 같은 config 해시; export는 F93 commit에서 해도 `config_drift=[]`).

- **시작 절차(D1/P1/OBS-1 blocker, OBS-2, D13).** `Run_Task_Teacher`가 실기 어댑터의 필수 절차인 ★명시적 개방 펄스 + 1.20 s 안정화★를 수행하지 않아 dry-run·모션 모두 0스텝에서 trip했다. 매 에피소드: 홈 자세 게이트(관절당 0.05 rad, `--home-first`로 moveJ 허용) → 명시적 개방(command state 0.0까지) → 루프. DRY-RUN(receive-only)은 command state를 개방으로 가정하는 프록시를 쓴다.
- **hold(P2/T-01/D7 blocker).** 현 브라켓에서 최종 하강+폐합 전이는 ≥32프레임 블라인드인데 임계가 40이라 정상 파지 중 팔이 영구 정지할 수 있었다. 임계 80(4 s) + 마지막 추정치 바로 위(held_radius+object_size)·그리퍼 비개방 중의 결측은 세지 않음 + 폐합 전이에서 리셋. hold 중에도 그리퍼를 재명령해 command-state 추적기가 학습처럼 전진한다. `max_blind_streak` 통계 추가.
- **그리퍼 개방 타이밍(AX-07).** 실기가 릴리스 판정 스텝에 바로 열었다(sim은 다음 스텝). `gripper_closed = rs_closed_prev`로 sim에 맞춤.
- **Phase3 리허설 그리퍼(AX-01/G2 blocker).** `SimRobot.set_gripper`가 액션 버퍼를 채우지 않아 수퍼바이저가 항상 open을 읽었다(리허설은 영영 못 닫음). 부호를 기록해 `send_task_delta`가 액션 버퍼 기록 → 팔 항 → 그리퍼 항 순으로 처리한다.
- **러너 기본값(D5/D6/AX-02/AX-03/S6/OBS-6, RT-1).** `--max-steps` 200(10 s 학습 에피소드), `--action-scale` 1.0(SafetyGuard가 이미 상한; <1.0은 릴리스 절대 임계와 어긋남 → 경고). Deploy_DryRun·Phase3 리허설 상한·`MAX_PHASE3_EPISODE_STEPS` 200.
- **가드·검증(D8/AX-04/OBS-9, OBS-3/S5, D2/D3).** 신선도·스큐·오버런 가드 전달, `--target-pos`를 학습 sample_range와 대조, 캘리브 `meta.calibrated=false`면 모션 거부(`--accept-uncalibrated`), `tcp_command_verified=false` 경고.
- **출처(T-04, T-09).** URDF를 `asset_manifest.json source_assets`로 USD와 묶고 preflight가 검사한다(F91 재발 방지). teacher export에 canonical config drift 게이트(`--allow-config-drift`, manifest `config_drift`).
- **문서·런북(D4/D9/D11/P8/T-08/T-14).** 런북 하드웨어 단계가 존재하지 않는 API(`read_state`/`camera.read`)를 부르던 것 수정, UTF-8 콘솔, DEPLOY_README v2 절차, README §3.3 Teacher v2 배포 레인 명시, 42D 잔재 문구.

**병합 규칙.** 학습·평가·채택·증류는 같은 commit(8033356)이어야 하므로 F93은 run3 증류·student 평가가 끝난 뒤 병합한다. 배포 폴더의 repo는 F93 병합 commit, 번들은 8033356 학습본(`commits_match=false`, `config_drift=[]`).

### F94 — 타일 ≥128 + DLSS에서 RGB 렌더만 죽는다: 증류 1회 오염, 학습 경로 렌더 가드 신설 (2026-09-22)

**발견.** run3 teacher(model_24200)로 증류한 student(256 env, 20k iter, 학습 중 placed_success 0.91)의 채택 평가가 3회 연속 `EVAL_RENDER_DEGENERATE`(rgb 공간std 0.0065, 검정 41% / 2회차는 전면 백색 0.91)로 거부됐다. depth는 정상(std 0.118). 세션 프로브(Vision-Eval env, 리셋 후 스텝별 통계)로 조건을 좁혔다: 타일 16·64 = 정상(rgb std 0.09~0.10), 타일 128·256 = 리셋 프레임 완전 검정 → 이후 스텝마다 검정/부분 렌더 교대(env 중앙값 std 0.01 ↔ 0.09). `--rendering_mode quality` 또는 `sim.render.antialiasing_mode="Off"`면 128 타일 정상. 즉 headless 기본 "performance"(DLSS) 경로가 이 GPU(RTX 5070 Ti, 드라이버 591.86, Isaac Sim 5.1)에서 많은 타일의 LdrColor를 깨뜨린다. F86(구 PC, "연속 기동 시 RGB만 검정")도 같은 계열로 본다.

**영향.** 증류가 256 env로 돌았으므로 student는 20k iter 내내 깨진 RGB(검정/깜빡임)를 봤다 — 학습 곡선이 좋아 보여도 실기 RGB(정상)와 분포가 다르다. ★무효★. 학습 경로에는 가드가 없어 조용히 지나갔다(평가 가드만 있었음). teacher는 카메라가 없어 무관.

**조치.** ① `BinPickVisionEnvCfg`에 `self.sim.render.antialiasing_mode = "Off"`(실기 카메라에 DLSS는 없으므로 정합 방향). ② 가드를 `common/render_guard.py`로 뽑아 `scripts/rsl_rl/train.py`(Vision/Distill 첫 관측)와 `Phase1/BinPick_Eval.py`가 공유 — 학습도 죽은 프레임이면 기동 전에 실패한다. ③ 오염된 증류 run 폐기, 같은 teacher로 재증류. ④ `tests/test_render_guard.py`.

**교훈.** 렌더 결함은 성공률·loss로 드러나지 않는다(student는 깨진 입력에 적응한다). 관측 채널별 공간 통계 가드는 평가뿐 아니라 ★학습 기동 시★에도 걸어야 하고, GPU/드라이버가 바뀌면 타일 수를 실제 학습 규모로 프로브해야 한다.

### F95 — 증류의 teacher commit-drift 명시적 허용(이번만): run3 teacher를 F94 commit에서 증류 (2026-09-22)

**배경.** `scripts/adoption.py::verify_adopted_checkpoint`는 현재 Git/config/stack 컨텍스트가 채택 run과 ★완전히★ 같아야 통과한다. F93/F94로 commit이 8033356 → e1c84a4로 바뀌자 run3 teacher(model_24200, 100%) 증류가 거부됐고, 처음엔 teacher run4를 e1c84a4에서 재학습하기 시작했다(≈23 h, run1≡run2 비트 동일 전례로 같은 결과 예상). 사용자 지시("달라진 게 없다면 규칙을 이번만 완화")로 run4를 중단하고 예외를 코드로 만들었다.

**근거.** run3 컨텍스트와 현재 컨텍스트의 차이는 `project_git_commit`뿐이다(canonical_configs 8개 해시·training_stack 동일 — 필드별 대조). F93은 실기/배포 코드, F94는 Vision cfg(DLSS off)·렌더 가드뿐이라 teacher 학습 env는 무변경이다.

**조치.** `verify_adopted_checkpoint(..., accept_commit_drift_from=<채택 run의 정확한 40자 commit>)`: 그 commit을 정확히 지목하고, commit을 제외한 컨텍스트가 동일할 때만 통과. binding에 `context_drift{adopted_commit, current_commit}`를 기록해 student 학습 receipt(inputs)에 남는다. `Phase2/VisionPick_Train.py --accept-teacher-commit <hex>`(--distill-from 필수), `scripts/train.ps1` ExtraArgs 허용 목록에 추가. 기본 동작은 그대로 fail-closed다.

**한계.** 이 예외는 "코드 변경이 teacher env에 무관하다"는 사람의 판단을 신뢰한다 — 자동 판정이 아니다. 다음부터는 teacher 채택 전에 배포/Vision 측 수정을 먼저 commit하거나, teacher env cfg 해시를 컨텍스트에 넣어 자동 판정으로 바꿀 것.

### F96 — Student v2 채택(94.53%)과 배포 폴더 v2 (2026-09-25)

**결과.** run3 teacher(model_24200, sim 100%)에서 512 env · 25k iter로 증류한 student가 봉인 평가를 통과했다: 채택 `model_24000`, seed 42~45 각 482/487/480/487 (=512), **pooled 94.53%, Wilson 95% 하한 93.46%** (게이트 384/512·0.75). 후보 5개 전부 통과했고 최고 pooled는 model_24999(94.68%)였지만 선정 규칙(최저 시드 우선)이 model_24000을 골랐다. 학습 종료 시 behavior loss 0.032, 학습 중 성공률 0.938.

**512 env의 효과.** 폐기된 256 env run과 같은 iter 시점 비교에서 0.70 → 0.83(6.7k iter). iter당 데이터 2배·처리량 +53%(3215 vs 2100 steps/s), 피크 VRAM 13.9 GB / 15.9 GB.

**배포 폴더 v2.** `Desktop\Deploy_UR5e_2026-09-25\` — `repo/`(commit 5368499 archive + 두 번들), `policies/teacher_177c7eefed0f`·`policies/student_8b415b9b0831`, `README_DEPLOY.md`(실행 절차·동작 계약·한계·그리퍼 DO 펄스 실측 절차), `configs/real.local.yaml` 시작점(F93 G7의 `dispatch_ack_timeout_s` 추가), `SHA256SUMS.txt`(406개). 런북 실행 결과 남은 실패는 현장 항목 하나뿐이다(카메라 시리얼 미기입).

**사용자 확인된 실기 사실(2026-09-20).** 2FG7 개폐 내측 폭 74/34 mm(도면 `2fg7.png`) = sim 가정과 일치, 파지 피드백 없음, 브라켓 = hand-eye v7 그대로, 작업면 z −0.0375 m·큐브 50 mm·펜던트 TCP 0·D435i 848×480 align 확인. 미측정: DO 펄스→완전 폐합 시간, 반전 거동, Compute Box 힘 프리셋 → 배포 README의 §2 절차로 첫 모션 전에 실측한다.

### F97 — GitHub 배포 저장소 패키징: 문서 세트 정책 확장 (2026-09-25)

**목적.** 배포 산출물을 GitHub에 올려 "받으면 바로 실행"할 수 있게 한다. 배포 패키지는 학습 저장소 트리에 ① 빠른 시작 `README.md`(성능·요구사항·3단계 실행·동작 계약·한계), ② 전체 절차 `DEPLOY_GUIDE.md`, ③ 원장 `PROJECT_LOG.md`(= 학습 저장소의 README.md), ④ 정책 번들 2종(`exported_experimental/`), ⑤ `setup.ps1`(conda 환경 → 의존성 → 점검 런북을 한 번에), ⑥ 배포용 `.gitignore`(학습 저장소는 `*.md`·`*.pt`·`exported_experimental/`를 전부 제외하므로 그대로 쓰면 정책 파일이 빠진다)를 더한 것이다.

**충돌과 조치.** 배포 트리를 clone해 `pytest tests`를 돌리면 3건이 깨졌다 — (a) `validate_repository`의 "Markdown은 README.md 하나" 정책, (b) 원장이 README.md가 아니어서 stale-path 면제에서 빠짐, (c) `test_training_entrypoints`가 학습 문서를 README.md에서만 찾음. 배포 패키지 쪽을 고치면 다음 패키징 때 되돌아가므로 **학습 저장소 쪽을 한 번 고쳤다**: Markdown 허용 집합을 {README.md, DEPLOY_GUIDE.md, PROJECT_LOG.md} + `exported_experimental/**/DEPLOY_README.md`로 좁게 확장하고(문서 난립 방지 의도는 유지), `PROJECT_LOG.md`를 stale 면제에 추가, 테스트는 `_training_doc()`으로 문서 위치만 흡수한다(검사 내용 불변).

**검증.** 배포 패키지를 실제로 `git clone`해 (1) `DeployPC_Runbook.py`가 정책 해시·TorchScript forward까지 통과(남은 실패는 현장 항목인 카메라 시리얼 하나), (2) 두 실행기 `--help` 정상, (3) `pytest tests -q` 979 passed를 확인했다. 패키지 위치는 `Desktop\ur5e-sim2real-deploy\` (102 MB, 367 파일)다.

### F98 — 학습 저장소 자체를 '받아서 바로 실행' 가능하게 (2026-09-25)

**배경.** F97의 별도 배포 패키지 대신, 사용자가 지정한 기존 원격(`ChangeDayAndNight/UR5e_SIM2REAL`)에 올리기로 했다. 그래서 같은 구성을 이 저장소에 직접 넣었다: sim 채택 번들 2종(`exported_experimental/teacher_177c7eefed0f`, `student_8b415b9b0831` — `.gitignore`에서 이 둘만 예외), `setup.ps1`(없으면 `configs/real.local.yaml`을 템플릿에서 생성), `DEPLOY_GUIDE.md`, README 첫머리의 현재 판정·성능표·빠른 시작. 원장 절 번호(§1~§11)는 코드 주석이 참조하므로 바꾸지 않았다.

**공개 저장소 원칙.** 현장 IP·카메라 시리얼은 저장소에 없다(`real.local.yaml`은 계속 ignore, 번들 사본의 IP는 비움). push 전 미push 이력 55커밋을 사설 IP·토큰·개인키 패턴으로 스캔해 해당 없음을 확인했다.

## 7. 자산, 생성물, 출처

### 학습 자산

| 경로 | 역할 | 소스/라이선스/정책 |
|---|---|---|
| `assets/ur5e_2fg7/` | 현행 결합 URDF와 meshes | UR5e BSD-3-Clause, 경유 URDF dataset MIT, 2FG7 MIT; 결합/2FG7 `NOTICE`와 세 `LICENSE-*.txt` 보존·배포 번들 포함 |
| `assets/onrobot_2fg7/` | 단독 2FG7 source URDF/mesh | upstream MIT; `NOTICE` 보존 |
| `assets/d435i/` | D435 visual mesh와 bracket | Intel RealSense mesh Apache-2.0; `NOTICE`와 `LICENSE-Apache-2.0.txt` 보존 |
| `assets/magbot_tcw1/` | 86.5 mm 시각 프록시 | 공개 CAD 부재로 프로젝트 primitive/proxy; 실제 질량·충돌 미반영 |
| `assets/Workspace/` | 작업대 shell USDA/USD | 참조 geometry는 대용량 로컬 자산; 재배포 권리 확인 전 공개 배포 금지 |
| `assets/Scene/Workplace.{ply,usdz,provenance.json}` | 선택 NuRec 실셀 배경 | 세 파일 모두 로컬/Git 제외; 원본 PLY, 변환 USDZ, byte-binding provenance가 함께 있어야 사용 |
| `2fg7.png` | 치수 대조용 원본 이미지 | vendor-derived 자료 여부와 재배포 권리 공개 전 검토 |
| `ur5e_workcell_drawing.html` | 프로젝트 작업 셀 치수도 | 유지되는 project-owned source asset |
| `docs/index.html` | 프로젝트 개요 페이지(GitHub Pages `/docs`) | project-owned; 수치·계약은 이 README를 따르며 README가 바뀌면 함께 갱신 |

학습에서 사용하는 현행 파일은 `assets/ur5e_2fg7/ur5e_tcw1_2fg7.urdf`에서 생성한 동명 `.usd`다. 구 `ur5e_2fg7.usd`는 사용하지 않는다.

```powershell
python scripts/tools/convert_2fg7_usd.py `
  --input assets/ur5e_2fg7/ur5e_tcw1_2fg7.urdf `
  --output assets/ur5e_2fg7/ur5e_tcw1_2fg7.usd --fix-base --headless
```

`assets/asset_manifest.json`에 고정된 UR5e composition 5개와 D435i 2개, 총 7개 USD는 fresh clone 학습에 필요한 일반 Git 추적 파일이다. 그 밖의 변환기로 재생성 가능한 USD, cache, checkpoint, logs, progression video만 Git에서 제외한다. LICENSE와 각 NOTICE는 문서 통합 대상이 아니며 반드시 그대로 둔다.

NuRec은 기본 Teacher/Vision 학습과 무관한 선택 기능이다. `Workplace.ply/.usdz`와 `Workplace.provenance.json`은 GitHub clone으로 복원되지 않으며 임의 파일로 대체하지 않는다. provenance JSON은 schema 1, canonical PLY/USDZ 상대경로, 두 파일의 SHA-256·byte 수, `capture_id`, `conversion_tool`, timezone-aware `created_at`을 정확히 기록한다. NuRec 학습은 공개 launcher의 `-Preset NuRec -NuRecFrom <sealed student model_*.pt>`만 사용한다. 이 preset은 `--require-nurec` preflight를 자동 적용해 세 파일의 존재·hash 불일치를 hard failure로 만들고, 그 provenance를 run receipt에 봉인한다. 부모 receipt의 Git·설정·Isaac stack이 현재 실행과 다르거나, 기존 NuRec 부모의 자산 provenance가 달라져도 시작/봉인을 거부한다.

### 결과 파일 정책

- canonical 입력: source/config/checkpoint hash, raw episode CSV/JSON, calibration YAML/evidence JSON
- 사람이 읽는 자동 요약: `.txt`
- 논문 표: `.tex`; 그림: `.pdf`/`.png`; 수치: `.csv`/`.json`
- `Phase*/Result/`: 로컬 생성 캐시, 전체 ignore
- `deployment/`: 실기 검증·bundle build의 canonical source; 생성 bundle은 ignored `dist/Real_Verify/`에 둠
- active training `logs/`: 로컬 운영 상태이며 cleanup/commit 대상이 아님

## 8. 2026-08-14 저장소 정리 내역과 복구

현재 tree에서 제거한 tracked 산출물은 다음과 같다.

| 분류 | 경로 | 파일/용량(삭제 전 HEAD) | 이유 |
|---|---|---:|---|
| 실험 산출물 | `Phase1/Result/**` | 795 files / 168.32 MiB | 재생성 가능, 중복 보고서 144 MD 포함 |
| 설치 로그 | `docs/setup_logs/**` | 9 / 0.17 MiB | 로컬 경로/환경 노출, 재현 입력 아님 |
| 렌더 스냅샷 | `Env_Fig/**` | 15 / 4.90 MiB | 오래된 생성 이미지 |
| legacy 배포 사본 | `Real_Verify/**` | 81 / 0.64 MiB | canonical `deployment/**`에서 hash bundle로 생성 |
| stale bundle | `Real_Verify.zip` | 1 / 0.26 MiB | source bundle과 drift 가능 |
| 소개 산출물 | `docs/UR5e_소개.html` | 1 / 1.06 MiB | embedded base64 중복 |
| 구 gripper 보고 | `gripper_report.html`, `gripper_problem.png` | 2 / 1.24 MiB | 핵심 내용은 본문/F05·F29~F32에 보존 |
| office 보고서 | `calibration/calibration_report.docx`, `.pptx` | 2 / 1.03 MiB | YAML/JSON이 canonical, 문서 생성물 |
| legacy direct-motion calibration demo | `calibration/{auto_capture_calib,center_probe,demo_camera_to_ik,demo_tour,orbit_capture}.py` | 5 / 0.034 MiB | hard-coded/직접 모션 경로 제거; `capture_calib.py`와 `deployment/`의 fail-closed 도구로 통일 |
| 분산 Markdown | root/Phase/docs/assets/Real_Verify의 구 문서 | README 제외 전부 | 고유 정보는 이 파일로 통합 |
| 과거 clone 로그 | `Phase1/logs/**` | tracked file 0 | Git 비추적; 현행 run만 보존하고 아래 기준으로 원본에서 정리 |

학습과 전체 후처리가 종료된 `2026-08-14 16:08:29 +09:00` 이후 원본 working copy의 Git 비추적/ignored 과거 산출물도 다음과 같이 삭제했다. 이 파일들은 Git 이력에 들어간 적이 없으므로 **삭제 후 Git으로 복구할 수 없다**.

| 분류 | 삭제 범위 | 실제 삭제량 | 이유 | 복구 |
|---|---|---:|---|---|
| 과거 Teacher run | `logs/rsl_rl/ur5e_binpick_teacher/`의 현행 run 제외 20개 디렉터리 | 2,353 files / 3,201,923,328 bytes | 구 설정 checkpoint/TensorBoard 중복 | 불가 |
| 과거 Phase 1 로그 | `Phase1/logs/`의 현행 학습 로그·metrics·watch/workspace 원자료 제외 | 111 files / 280,074,875 bytes | 구 run 지표와 진단 로그 | 불가 |
| 과거 Result의 ignored 잔여물 | `Phase1/Result/`의 현행 run 제외 MP4/checkpoint/진단 산출물 | 267 files / 405,439,627 bytes | tracked 결과 삭제 뒤 남는 재생성 가능 파일 | 불가 |
| legacy bundle cache/report | `Real_Verify/`의 ignored 잔여물 | 52 files / 971,399 bytes | canonical bundle로 재생성 | 불가 |
| stale ignored archive | `Real_Verify (2).zip` | 1 file / 625,191 bytes | source와 drift한 중복 bundle | 불가 |

합계는 **2,784 files / 3,889,034,420 bytes(3.622 GiB)**다. 삭제 직전에는 모든 대상이 의도한 root 하위인지 확인하고 reparse point가 0개임을 검증했다.

`2026-08-13_13-48-20` Teacher run은 사용자 요청에 따라 로컬에만 보존했다. run 506 files/692,973,876 bytes, 같은 이름의 Result 32 files/86,288,524 bytes, 현행 로그·metrics와 watch/workspace 원자료 10 files/68,820,399 bytes로 총 **548 files / 848,082,799 bytes**다. 최종 `model_24999.pt` SHA-256은 `c34b316eea555a836fb74418862a78d1564b2cd88a6f96e57892df8f9687089a`다. 이 run은 schema 1, training Git `b61bd5a7…`, WindowsApps Python 3.10 launcher, 구 물리·gripper 계약으로 만들어졌고 최종 `placed_success=0.0000`이므로 현행 성능 기준, resume, distillation, export 또는 실기 배포에 사용하지 않는다. 로컬 보존본 자체도 ignored 파일이므로 별도 백업 없이 삭제하면 복구할 수 없다.

위 명시 산출물과 legacy 배포 사본은 약 **177.65 MiB**다. 삭제 직전 clone의 loose Git object store는 **513.20 MiB**였다. working tree에서 지우는 커밋만으로 과거 blob이 `.git` history에서 사라지지는 않는다. 새 tree와 GitHub source archive는 가벼워지지만, 전체 clone history를 줄이려면 별도 합의 아래 `git filter-repo`/BFG, force-push, 모든 clone 재동기화가 필요하다. 이 정리에서는 사용자 요청에 따라 history rewrite를 하지 않는다.

**공개 저장소 개인정보 주의:** 현 working tree에는 실 IP, 실제 camera serial, 개인 사용자 절대경로, credential이 없지만, 삭제 전 commit의 `PROGRESS.md`, `Real_Verify/configs/real.yaml`, `Phase1/Result/**/pipeline.log`, `docs/setup_logs/**` blob에는 RFC1918 실기 IP, 과거 D435i serial, Windows 사용자/로컬 절대경로 문자열이 남아 있다. Git commit의 작성자 이메일도 Git 이력의 공개 metadata다. 일반 push는 그 과거 blob과 metadata를 계속 공개한다. 공개 전에는 이 history-only 노출을 수용할지 별도 승인 아래 history를 정제할지 명시적으로 결정하고, credential이 과거 이력에서 추가로 발견되면 history 정제와 별개로 즉시 폐기·재발급한다.

삭제 전 기준 commit은 `8d60d57`이다. 필요한 파일은 다음처럼 복구한다.

```powershell
git restore --source=8d60d57 -- Phase1/Result
git show 8d60d57:Failure_Check.md | Out-File -Encoding utf8 Failure_Check_legacy.txt
```

복구한 산출물을 다시 commit하지 않는다. 감사/법무/논문 증거에 필요한 결과는 식별된 경로만 별도 artifact storage에 보존한다.

## 9. 유지보수와 협업 규칙

### 코드 경계

1. `common/`은 Isaac/RTDE/RealSense를 import하지 않는 순수 계약·수학 계층이다.
2. sim과 real은 같은 preprocessing, timing, action mapping, rollout, safety 코드를 호출한다.
3. standalone Isaac 스크립트는 repo root/Python 재실행 guard와 stdout UTF-8 설정 후 `AppLauncher`를 만들고, 그 뒤에 Isaac module을 import한다.
4. 확인하지 않은 API signature를 추정 구현하지 않는다. 공식 source/doc 또는 installed signature를 확인하고 test로 고정한다.
5. scene/action/observation/camera/calibration 변경은 checkpoint 호환성 여부와 재학습 범위를 함께 기록한다.
6. 실기 장비가 필요한 작업은 사람이 수행한다. 코드 작성자가 IP, serial, 캘리브 성공, e-stop 성능을 추정하지 않는다.

### 검증

```powershell
# Isaac 없이 가능한 기본 검증
python scripts/validate_repository.py
python -m pytest tests -q

# 정적/문서 정책 검사 예
git ls-files "*.md"
rg -n "\.md" --glob "*.py" --glob "*.ps1" --glob "*.yaml" --glob "*.json"

# hash-manifested bundle 생성(기존 output을 덮어쓰지 않음)
python deployment/build_bundle.py --release-dir exported --output dist/Real_Verify
```

`git ls-files "*.md"`의 기대값은 `README.md` 하나다. 단, third-party license/notice 원문은 확장자와 관계없이 삭제하거나 통합하지 않는다. 하드웨어가 없는 CI에서 실기 gate를 통과한 것처럼 mock evidence를 만들지 않는다.

### Git/GitHub

- 비밀값, 실제 IP/serial, calibration capture, evidence 원본, logs, model checkpoint를 commit하지 않는다.
- `configs/real.local.yaml`과 `deployment/evidence/`는 로컬 전용이다.
- 배포 가능한 최종 Student artifact만 release manifest/hash와 함께 별도 release 절차로 취급한다.
- 여러 설정을 한 commit에서 동시에 바꿔 원인 추적을 어렵게 하지 않는다.
- active training process/log는 정리 작업에서 건드리지 않는다.
- 이 README가 stale해지는 변경은 미완료 변경이다.
- 원격은 둘이다(2026-10-11). `UR5e_SIM2REAL`(private)이 전체 이력의 원본이고, `UR5e_SIM2REAL-public`은 현재 tree만 담은 **이력 없는 단일 커밋 스냅샷**이다(§8의 history-only 노출 때문). 공개본 갱신은 private `main` tree를 공개본 위에 커밋 하나로 얹는다(private 이력은 옮기지 않는다). 작성자는 GitHub noreply 이메일, `2fg7.png`는 재배포 권리 확인 전까지 제외한다. GitHub Pages는 공개본의 `main` `/docs`다. 공개본에는 self-hosted runner를 등록하지 않는다(외부 PR이 워크플로를 바꿀 수 있다).

## 10. 남은 우선순위

1. bootstrap/preflight/smoke를 fresh clone에서 검증
2. F64 Teacher 25k scratch 완료 및 checkpoint 상위 후보 독립 평가
3. 새 `gripper_command_state` 계약으로 Teacher scratch 계보를 만든 뒤 Student 증류; **P0 실제 motion:** 별도 2FG7 width/grip feedback 확보
4. 채택된 새 Teacher로 112² RGB-D Student 증류와 export parity 10k 검증
5. real TCP offset, hand-eye 보정 후 검증, scene registration
6. RTDE `servoL` signature/base frame/joint parity와 force/e-stop fault injection
7. TCP quaternion canonical sign 실측 parity와 held-out 채널별 safety clamp `<1%` 확정
8. P0 joint limit/singularity/self·cell collision/TCP accel·jerk·torque/command acknowledgement 구현·fault evidence
9. **P0:** RTDE controller boot clock과 RealSense acquisition clock의 PTP/hardware timestamp calibration·atomic association 구현; 그 뒤에만 no-motion evidence issuer 활성화
10. **현재 P0:** 신뢰 가능한 `motion_safety` collector·승인 발행 절차 구현과 독립 현장 검토; 통과 뒤에만 별도 위험 평가로 속도 확대 검토

## 11. 방법론 참고문헌

- Pinto et al., Asymmetric Actor Critic for Image-Based Robot Learning: https://arxiv.org/abs/1710.06542
- Tobin et al., Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World: https://arxiv.org/abs/1703.06907
- Makoviychuk et al., Isaac Gym: https://arxiv.org/abs/2108.10470
- NVIDIA, Sim-to-Real Transfer of Robotic Control with Dynamics Randomization: https://research.nvidia.com/publication/2017-12_sim-real-transfer-robotic-control-dynamics-randomization
- Isaac Lab v2.3.2 Differential IK controller documentation: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.controllers.html
- Isaac Lab v2.3.2 differential IK tutorial: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/tutorials/05_controllers/run_diff_ik.html

Repository code license는 `LICENSE`를, vendored asset attribution은 각 `NOTICE`를 따른다.
