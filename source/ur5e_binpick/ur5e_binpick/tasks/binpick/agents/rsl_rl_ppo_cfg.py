"""binpick rsl_rl 러너 cfg — Teacher / Vision / Distill (한 모듈).

gym 등록(binpick/__init__.py)이 문자열 entry_point로 지연 로드:
  rsl_rl_ppo_cfg:BinPickTeacherRunnerCfg  → Isaac-BinPick-UR5e-Teacher-v0
  rsl_rl_ppo_cfg:BinPickVisionRunnerCfg   → Isaac-BinPick-UR5e-Vision-v0
  BinPickDistillRunnerCfg는 증류 확장 경로(VisionPick_Train --distill-from).

⚠️ isaaclab을 import하므로 AppLauncher 이후(= Hydra 지연 로드)에만 import된다.
공식 external 확장 구조: 이 모듈은 pip 설치된 ur5e_binpick 패키지의 일부이며,
common.* 는 repo 루트 top-level import(scripts/rsl_rl 헤더가 sys.path 보장).

rsl_rl cfg 클래스는 vendored IsaacLab 2.3.2(external/IsaacLab의 isaaclab_rl/rsl_rl/
rl_cfg.py·distillation_cfg.py) + rsl-rl-lib 3.1.2 소스에서 확인됨(2026-07-17 포팅).
★3.x 핵심 변경★: obs_groups 명시(러너가 관측그룹→역할 매핑을 cfg로 받음),
empirical_normalization → policy별 *_obs_normalization, 증류=별도 DistillationRunner.
하이퍼파라미터 초기값은 Franka Pick&Place에서 수렴 검증된 값 — ★튜닝은 나중★.
"""
from __future__ import annotations

import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,  # ★3.1.2: 증류는 별도 DistillationRunner(rl_cfg 분리)★
    RslRlDistillationStudentTeacherCfg,  # noqa: F401 — feedforward 카드 복귀용 보존
    RslRlDistillationStudentTeacherRecurrentCfg,  # noqa: F401 — recurrent 재시도 카드 보존
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from common.config_io import configs_dir, load_yaml

from .vision_policy import (
    ACTIVATION,
    ACTOR_HIDDEN_DIMS,
    CRITIC_HIDDEN_DIMS,
    TEACHER_ACTOR_HIDDEN_DIMS,
)

# 단일 소스 원칙: max_iterations/seed는 configs/sim.yaml training 섹션에서만 온다.
_TRAINING = load_yaml(configs_dir() / "sim.yaml")["training"]


def _require_valid_student_aux_combo(student_class_name: str) -> str:
    """★F81 — BINPICK_AUX와 student 백본의 무효 조합을 기동 시점에 거부 (리뷰 minor #3/#8)★

    DistillationAux.update()는 `policy.student.aux_pred`를 읽는데 그 속성은
    ResNetStudentNet에만 있다. BINPICK_AUX=1을 기본 CNN student와 조합하면 기동은
    성공하고 ★수 분 뒤 첫 update에서★ AttributeError로 죽는다 — 야간 무인 기동이면
    죽은 run을 다음 날 발견한다. 실패는 즉시·명시적으로 일어나야 한다.
    """
    aux_on = os.environ.get("BINPICK_AUX", "").lower() in ("1", "true")
    if aux_on and student_class_name != "VisionStudentTeacherResNet":
        raise RuntimeError(
            "[cfg] BINPICK_AUX=1은 BINPICK_STUDENT=resnet(ResNetStudentNet)에서만 유효하다 "
            f"— 현재 student={student_class_name}에는 aux_pred가 없어 첫 update에서 죽는다."
        )
    return student_class_name


@configclass
class BinPickTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1 teacher — 카메라 없는 특권 상태 PPO. scratch 학습.

    이 ckpt가 Phase 2 vision student의 distillation(rsl_rl 3.1.2
    StudentTeacher/Distillation) / warm-start 소스가 된다.
    """

    seed = int(_TRAINING["seed"])                      # 42 (configs/sim.yaml)
    # ★2026-08-07 논문 주 실험 확정(사용자 결정)★: scratch 20,000 iter · 4096 env · 24-step.
    # resume/이어학습 없이 처음부터 끝까지 단일 연속 run으로 수행한다.
    # F49 10k·24-step 최고 placed 92.38%로 95%에 미달했으므로 rollout을 48로 키우지 않고,
    # 동일한 표준 24-step PPO 업데이트 주기를 유지한 채 iteration 예산만 20k로 확장한다.
    # 총 환경 스텝 = 20,000 × 4,096 × 24 = 1,966,080,000.
    num_steps_per_env = 24
    max_iterations = int(_TRAINING["max_iterations"])  # 20000 — CLI --max_iterations로 덮어쓰기
    save_interval = 50
    experiment_name = "ur5e_binpick_teacher"           # BinPick_Train/Env/Tensorboard.py EXPERIMENT와 동일해야 함
    # ★3.1.2: 관측그룹→역할 매핑 명시(resolve_obs_groups 폴백에 의존하지 않음 — rl_cfg.py:159)★
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # ★3.1.2: empirical_normalization(deprecated) 대체 — False 고정(obs_spec/배포 계약)★
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,   # ★기본 복원(2026-07-11)★: 0.015 실험은 릴리스 exploration을 촉진
        #   (release_bias 0.13→0.56)했으나 entropy가 그리퍼만이 아닌 '팔 액션'까지 흔들어 pick+carry
        #   붕괴(iter6501 carry0.14·drop45 폭증). → 릴리스 유도는 release_bias smooth(open action 비례
        #   gradient, 팔 무관)로 전환하고 entropy는 기본값 유지.
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",   # adaptive-KL: KL 급증 시 LR 자동 감소(발산 가드)
        # ★Teacher v2(2026-09-18)★ 0.98 → 0.99. 실행 제한 정합(TCP 0.25 m/s)으로 pick→place가
        #   ~40스텝에서 ~100스텝으로 길어졌다(에피소드 10 s). 0.98의 유효 지평(50스텝)은 태스크
        #   길이의 절반이라 0.99(100스텝)로 같은 비율을 유지한다. place_success_bonus도 이에 맞춰
        #   상향(binpick_env_cfg — 포기 리턴 < 종단 보너스 불변식, tests/test_teacher_contract.py).
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BinPickVisionRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 2 본선 — vision student PPO (asymmetric actor-critic).

    actor 입력 = env의 "policy" 관측 그룹(단일 ObsTerm이 common.obs_preprocess.build_policy_obs
    호출) = 현행 RGB-D flat 50206dim. critic 입력 = "critic" 관측 그룹(특권 상태 — rsl_rl이
    extras["observations"]["critic"]를 자동 사용, on_policy_runner.py 줄 53-57).

    기본 PPO는 stock MLP ablation이고, `BINPICK_TGRL=1`이면 현행 custom
    VisionActorCritic RGB-D 경로를 사용한다. 두 클래스는 agents/vision_policy.py가
    rsl_rl 3.1.2 runner의 eval 네임스페이스에 명시적으로 등록한다.
    """

    seed = int(_TRAINING["seed"])
    num_steps_per_env = 24  # 업스트림 lift PPO 기본 준용(isaaclab_tasks lift rsl_rl_ppo_cfg.py)
    max_iterations = int(_TRAINING["max_iterations"])  # 단일 소스: configs/sim.yaml
    save_interval = 50
    experiment_name = "ur5e_binpick_vision"  # VisionPick_*.py의 EXPERIMENT와 동일해야 함
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}  # ★3.1.2 명시★
    policy = RslRlPpoActorCriticCfg(
        # ★TGRL Path 2a 게이트(BINPICK_TGRL, 2026-07-25)★: actor=ResNet camera + critic=특권
        #   + teacher(frozen) 로딩. 미설정=stock ActorCritic(현행 PPO 재현). eval도 같은 게이트 필수.
        class_name=("VisionActorCritic"
                    if os.environ.get("BINPICK_TGRL", "").strip().lower() in ("1", "true")
                    else "ActorCritic"),
        init_noise_std=1.0,
        # ⚠️ False 고정 — 켜면 depth 4096픽셀에 픽셀별 이동 통계가 걸리고, 그 통계가
        #   ckpt에 저장되어 export/배포 계약(common/policy_io.py make_obs_spec)이 깨진다.
        #   (3.1.2: empirical_normalization(deprecated) → policy별 플래그로 이동)
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # 아키텍처 상수의 단일 소스 = _vision_policy.py (Export_Policy 재구성과 정합)
        actor_hidden_dims=list(ACTOR_HIDDEN_DIMS),
        critic_hidden_dims=list(CRITIC_HIDDEN_DIMS),
        activation=ACTIVATION,
    )
    # PPO 값은 업스트림 lift 기본(rsl_rl_ppo_cfg.py — Franka 프로젝트에서 수렴 검증) 준용.
    algorithm = RslRlPpoAlgorithmCfg(
        # ★TGRL Path 2a 게이트★: VisionPPO(모방 리워드셰이핑 rewards += β·r_imit). 미설정=stock PPO.
        class_name=("VisionPPO"
                    if os.environ.get("BINPICK_TGRL", "").strip().lower() in ("1", "true")
                    else "PPO"),
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BinPickDistillationPolicyCfg(RslRlDistillationStudentTeacherCfg):
    # Serialized into params/agent.yaml, including when resuming old weights.
    collection_mode: str = os.environ.get("BINPICK_DISTILL_COLLECTION", "deterministic")


@configclass
class BinPickDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """확장(권장 경로 B) — Phase1 teacher → vision student 증류 (rsl_rl 3.1.2 DistillationRunner).

    ★3.1.2 포팅(2026-07-17)★: OnPolicyRunner의 Distillation 자체처리 폐지 → 별도
    DistillationRunner(distillation_cfg.py:106, class_name="DistillationRunner").
    업스트림 train.py가 class_name으로 러너를 분기한다(2.3.2 train.py:198-203).
    sanctioned teacher→student 경로 (근거, rsl_rl 3.1.2 소스 검증):
      - StudentTeacher.load_state_dict가 RL ckpt의 "actor.*" 키를 teacher로 자동 매핑
        (student_teacher.py 줄 168-203) — Phase1 PPO ckpt를 그대로 teacher로 로드.
      - train.py는 algorithm.class_name=="Distillation"이면 --resume 없이도 ckpt를
        로드한다(train.py 줄 148) → VisionPick_Train.py --distill-from이 teacher ckpt를
        시드하고 --load_run/--checkpoint로 지정.
      - Distillation 알고리즘은 student 파라미터만 최적화(distillation.py 줄 50).

    전제 상태 (2026-07-04 배선 완료):
      ① env "teacher" 관측 그룹: binpick_env_cfg.py VisionObservationsCfg에
         teacher=PrivilegedObsCfg로 노출됨(on_policy_runner.py 줄 58-62의
         extras["observations"]["teacher"]) — Phase1 teacher actor 관측과 같은 클래스라
         차원 자동 일치.
      ② teacher_hidden_dims = _vision_policy.TEACHER_ACTOR_HIDDEN_DIMS (단일 소스) —
         Phase1 BinPickTeacherRunnerCfg.policy.actor_hidden_dims의 사본이며,
         VisionPick_Train.py --distill-from이 teacher ckpt의 실제 가중치 형상과
         대조해 불일치 시 기동을 거부한다.
      ③ empirical_normalization: Phase1 teacher를 empirical_normalization=True로 학습한
         경우 teacher 정규화 통계가 privileged_obs_normalizer로 로드된다
         (on_policy_runner.py 줄 433-437) — Phase1도 False로 학습해 이 경로를 피할 것
         (이 모듈 BinPickTeacherRunnerCfg도 False 고정 — 현 계약 준수 상태).
    """

    seed = int(_TRAINING["seed"])
    num_steps_per_env = 24
    max_iterations = int(_TRAINING["max_iterations"])
    # Vision checkpoints are ~298 MiB each on the target stack.  Keep the
    # Teacher's 50-iteration cadence, but bound a 20k distillation run to ~41
    # checkpoints so the 25 GiB pre-run disk gate remains meaningful.
    save_interval = 500
    experiment_name = "ur5e_binpick_vision"  # 보관소 공유 — run 구분은 타임스탬프/로그로
    # ★3.1.2: 증류 러너의 관측그룹 매핑 — student='policy', teacher='teacher'★
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = BinPickDistillationPolicyCfg(
        # ★student 카드 이력(전부 512env×1500iter, eval=보조 전부 off)★
        #   MLP(고정캠): loss 6.2 수렴에도 grasp 0% — flat MLP-on-pixels 표현력 한계
        #   CNN(고정캠): loss 5.0·grasp 1.6% — 카메라 팔가림+해상도(진짜 원인)
        #   ★CNN(eye-in-hand): grasp 12.7%★ — 자력 파지 최초 발현 (현행 채택)
        #   GRU(eye-in-hand): grasp 0.6% ★실패★ — 훈련 후반 reward 붕괴(25→1).
        #     유력 기전: epochs=6(스텝 기아 수정)이 recurrent hidden state 인계와
        #     병리적 상호작용(같은 rollout 6회 재생 중 가중치 변화로 hidden 표류 →
        #     오염된 상태가 다음 수집 시작점 — rsl_rl recurrent 기본이 epochs=1인 이유
        #     로 추정). ★recurrent 재시도 카드 = epochs=1 + iter 연장★(클래스 보존).
        #   ★RGB-D(2026-07-18): CNN(full FOV·색DR off) 계보 0.8→1.9→5.0→13.2→14.1%로
        #     연장 수확 체감 + hold 유지율 0.8%(파지 순간 near-blind 부분 관측).
        #   ★GRU 재시도 2회(2026-07-18) → 폐기★: ①epochs=1(병리 수정판, LR 3e-4) —
        #     iter600 발산(72→221) ②LR 1e-4 — 발산 완화되나 수렴 실패(loss 바닥 ~12
        #     정체·후반 포락선 재상승 med 21→38; feedforward 동시점 loss 3.0).
        #     결론: 소형 Nature-CNN+BPTT+DAgger 조합은 본 예산에서 불성립 —
        #     재도전하려면 사전학습 인코더/transformer급 개편(대형 카드, 보류).
        # 현행 = ★CNN feedforward(RGB-D) 복귀★ — 최고 자산 r5 14.1%(model_5997).
        # ★② b1 카드(2026-07-20)★: 환경변수 BINPICK_STUDENT=resnet →
        #   VisionStudentTeacherResNet(ImageNet ResNet18 RGB + 소형 depth 브랜치,
        #   layer4+머리만 학습 — vision_policy.ResNetStudentNet). 게이트 방식인 이유:
        #   cfg 파일을 직접 바꾸면 진행 중/기존 CNN run의 eval(class_name으로 러너 재구성)
        #   이 깨진다 — BINPICK_STAGE와 동일한 안전 패턴. ⚠️eval도 같은 변수 필요.
        class_name=_require_valid_student_aux_combo(
            "VisionStudentTeacherResNet"
            if os.environ.get("BINPICK_STUDENT", "").lower() == "resnet"
            else "VisionStudentTeacher"
        ),
        init_noise_std=0.1,  # rsl_rl 3.1.2 StudentTeacher 기본(student_teacher.py)
        # ★3.1.2 필수★ — teacher(Phase1)를 False로 학습한 현 계약 유지
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        # CNN student에서 student_hidden_dims = 'CNN 머리(head)' hidden dims로 재해석.
        student_hidden_dims=[256, 128],
        # Phase1 teacher actor_hidden_dims의 단일 소스 사본(위 ② — ckpt 형상 검증은
        # VisionPick_Train.py --distill-from이 수행)
        teacher_hidden_dims=list(TEACHER_ACTOR_HIDDEN_DIMS),
        activation=ACTIVATION,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        # ★2026-07-13 근본 진단: 최적화 스텝 기아★ — distillation.py update()는 rollout 24스텝을
        #   gradient_length개씩 묶어 backward하므로, 기본(epochs=1, glen=15)은 iteration당 Adam 스텝이
        #   ★단 1회★(24//15, 나머지 9스텝 loss는 폐기)였다. 512env×250iter(3M 샘플)에서 behavior loss
        #   무진전(47~90 진동)의 원인 = 당시 4125차원 depth 입력을 250 gradient 스텝으로 학습 불가.
        #   (rsl_rl 기본값은 ~50차원 proprio 보행 student용.) 2026-07-10의 'LR 과대' 진단은 오진 —
        #   64env 실패도 같은 기아(1200스텝)였다. → epochs 6 × 24스텝 ÷ glen 12 = ★iteration당
        #   12스텝(12배, 나머지 폐기 없음)★. LR은 3e-4 유지(변인 1개 원칙 — 스텝 수만 변경).
        # ★feedforward 복귀(2026-07-18 GRU 폐기)★ — epochs 6·LR 3e-4는 class_name과
        #   3값 세트(recurrent로 갈 땐 epochs=1·LR 인하 필수였음 — 카드 이력 참조).
        # ★② b2 카드(2026-07-21)★: 환경변수 BINPICK_LR로 LR 오버라이드(미설정 시 3e-4).
        #   b1(ResNet18)이 실사 27.8%를 찍었으나 ckpt별 진동 극심(절차 6.4~17.1%) →
        #   진동 안정화용 LR 인하(사용자 결정: BINPICK_LR=1e-4 재증류). 게이트 방식인 이유:
        #   기존 run 재현(BINPICK_LR 미설정 = 3e-4 그대로)을 깨지 않기 위함. eval엔 무관.
        # ★② b4 aux 카드(2026-07-23)★: BINPICK_AUX=1 → DistillationAux(물체위치 예측 보조손실
        #   β·L_pos). 파지 실패 77.8%(접근 오차 15.3cm = 위치 인지 부재) 직격 — DextrAH-G/RGB 실증.
        #   ★eval도 같은 알고리즘이라야 ckpt 로드(러너가 class_name으로 알고리즘 재구성) — eval 시
        #   BINPICK_AUX=1 필수★. aux 헤드는 관측 계약 불변(출력 헤드)이라 teacher 재학습 불필요.
        # ★F87(2026-09-08)★ 기본 = DistillationChannelBalanced(채널별 타깃 스케일 정규화).
        #   근거는 vision_policy.DistillationChannelBalanced docstring의 실측 —
        #   병진 3채널이 MSE의 9.5%뿐이라 '어디로 옮길지'가 학습되지 않았다.
        #   BINPICK_DISTILL_LOSS=plain 이면 구 무가중 MSE(계보 재현·대조군용).
        #   ⚠️ eval은 receipt에 봉인된 class_name으로 알고리즘을 재구성한다(F72) —
        #      환경변수를 안 맞춰도 채택 평가가 스스로 복원한다.
        class_name=("DistillationAux"
                    if os.environ.get("BINPICK_AUX", "").lower() in ("1", "true")
                    else "Distillation"
                    if os.environ.get("BINPICK_DISTILL_LOSS", "").lower() == "plain"
                    else "DistillationChannelBalanced"),
        num_learning_epochs=6,
        learning_rate=float(os.environ.get("BINPICK_LR", "3.0e-4")),
        gradient_length=12,
        # ★F81 — 증류에 gradient clipping 도입 (리뷰 major #7)★
        #   rsl_rl Distillation은 max_grad_norm이 truthy일 때만 clip하는데 기본값이 None
        #   이라 ★클립이 전혀 없었다★. 게다가 update()는 gradient_length=12 청크의 손실을
        #   mean이 아닌 ★sum★으로 누적하므로 유효 gradient가 12배로 부풀 수 있다.
        #   teacher PPO와 같은 1.0으로 맞춘다(50206D 관측 CNN에서 폭주 방지).
        max_grad_norm=1.0,
    )
