"""binpick 태스크 — gym 등록 (공식 템플릿 __init__task 패턴).

★AppLauncher 이후 import 전제★: scripts/rsl_rl/train.py가 AppLauncher 기동 후
`import ur5e_binpick.tasks`로 이 모듈을 부른다. 등록 자체는 gymnasium만 쓰고,
env cfg / rsl_rl 러너 cfg는 **문자열 entry_point**로만 참조 → Hydra
(load_cfg_from_registry)가 main 안에서 지연 import·인스턴스화한다.

태스크 ID / cfg 클래스 고정 (README.md §9):
  Isaac-BinPick-UR5e-Teacher-v0 → BinPickTeacherEnvCfg / BinPickTeacherRunnerCfg
  Isaac-BinPick-UR5e-Vision-v0  → BinPickVisionEnvCfg  / BinPickVisionRunnerCfg
"""

import gymnasium as gym

from . import agents

# Phase 1 — teacher (특권 상태 관측, 카메라 없음)
gym.register(
    id="Isaac-BinPick-UR5e-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.binpick_env_cfg:BinPickTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BinPickTeacherRunnerCfg",
    },
)

# Phase 1 평가 — pre-grasp 보조 제거(순수 자력 파지 성공률 측정). 러너 cfg는 teacher와
# 동일(같은 ckpt 로드). BinPick_Eval.py가 이 태스크로 정량 평가(성공률·CSV·영상)를 낸다.
gym.register(
    id="Isaac-BinPick-UR5e-Teacher-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.binpick_env_cfg:BinPickTeacherEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BinPickTeacherRunnerCfg",
    },
)

# Phase 2 — vision student (비대칭 관측 + TiledCamera), stock PPO
gym.register(
    id="Isaac-BinPick-UR5e-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.binpick_env_cfg:BinPickVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BinPickVisionRunnerCfg",
    },
)

# Phase 2 확장 — teacher→student 증류(rsl_rl 3.1.2 DistillationRunner). 같은 vision env를
# 쓰되 러너 cfg만 Distill. teacher ckpt는 VisionPick_Train --distill-from이 시드한다.
gym.register(
    id="Isaac-BinPick-UR5e-Vision-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.binpick_env_cfg:BinPickVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BinPickDistillRunnerCfg",
    },
)

# Phase 2 확장 — ★NuRec 실셀 배경 파인튠(2026-07-17 B안)★. 실셀 3DGS 스캔 배경 위에서
# 증류 student를 소규모 fine-tune(rgb 배경 도메인 갭 축소). 러너 cfg = Distill 재사용 —
# VisionPick_Train --nurec-finetune <student ckpt>가 resume 시드(증류 연속, teacher 포함
# StudentTeacher ckpt 전체 로드). env cfg가 정합 게이트(scene_registration.yaml
# meta.calibrated) 미충족 시 기동을 거부한다.
gym.register(
    id="Isaac-BinPick-UR5e-Vision-NuRec-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.binpick_env_cfg:BinPickVisionNuRecEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BinPickDistillRunnerCfg",
    },
)

# Phase 2 평가 — 보조 전부 off + TiledCamera (student 정량 eval — BinPick_Eval --task로 사용).
# 러너 cfg = Distill(증류 ckpt의 StudentTeacher 모듈 로드 → get_inference_policy가 student 반환).
gym.register(
    id="Isaac-BinPick-UR5e-Vision-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.binpick_env_cfg:BinPickVisionEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BinPickDistillRunnerCfg",
    },
)

# Phase 2 진단 — ★절차-실사 갭 분해(2026-07-19)★: 절차 Eval에서 광량 DR만 off(변인 1).
# NuRec Eval과 'DR 구성 동일·배경만 다름' 대조 형성 — 배포 대표 조건 아님(공식 비교 금지).
gym.register(
    id="Isaac-BinPick-UR5e-Vision-Eval-NoLightDR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.binpick_env_cfg:BinPickVisionEvalNoLightDrEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BinPickDistillRunnerCfg",
    },
)

# Phase 2 평가 — ★TGRL(imitation gap) 정책 eval(2026-07-25)★. TGRL 정책은 VisionActorCritic
# (OnPolicyRunner + BinPickVisionRunnerCfg + BINPICK_TGRL 게이트)라, 증류 러너(StudentTeacher/
# DistillationRunner)를 쓰는 Vision-Eval-v0으로는 ckpt state_dict가 불일치해 로드 불가.
# → 전용 eval 태스크: ★env cfg는 기존 Eval/NuRec 재사용(조건 동일 = b3/b4와 공정 비교),
#   러너만 BinPickVisionRunnerCfg(OnPolicyRunner)로 교체★. ⚠️eval 실행 시 BINPICK_TGRL=1 +
#   BINPICK_TEACHER_CKPT 필수(VisionActorCritic 재구성·teacher 로딩 — §8-3 계약 집행).
gym.register(
    id="Isaac-BinPick-UR5e-Vision-TGRL-Eval-v0",  # 절차 씬(assist off)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.binpick_env_cfg:BinPickVisionEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BinPickVisionRunnerCfg",
    },
)
gym.register(
    id="Isaac-BinPick-UR5e-Vision-TGRL-NuRec-Eval-v0",  # 실사 씬(NuRec, num_envs=4 강제 — b3/b4와 동일 env)
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.binpick_env_cfg:BinPickVisionNuRecEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BinPickVisionRunnerCfg",
    },
)
