"""Phase 2 — vision 정책 아키텍처의 단일 소스 + RGB-D encoder 구현.

목적:
  (1) MLP actor/critic 차원·활성함수 상수의 ★단일 소스★ — rsl_rl_ppo_cfg.py의 러너 cfg와
      Export_Policy.py의 actor 오프라인 재구성이 같은 값을 읽는다(어긋나면 export 실패).
  (2) rsl_rl 3.1.2 ActorCritic.actor와 'state_dict 키 호환' MLP 빌더(build_mlp_actor) —
      ckpt의 actor.* 서브키를 1:1로 로드할 수 있는 nn.Sequential을 만든다.
  (3) ckpt state_dict에서 MLP 차원을 역추론(infer_mlp_dims_from_state_dict) — 순수 로직.
  (4) 현행 flat 50,206D 관측 → 112² RGB-D + proprio 분해(split_flat_obs).
  (5) VisionActorCritic/CNN·ResNet StudentTeacher와 RSL-RL 3.1.2 등록 구현.
실행: 직접 실행하지 않음 (라이브러리 모듈 — isaaclab 미의존, torch만 필요.
     rsl_rl은 custom policy 등록에 필요하며 없어도 순수 helper import는 성공한다).
산출: 없음 (정의 전용).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

# common.* 는 top-level import (런타임에 repo 루트가 sys.path에 있어야 함 —
# scripts/rsl_rl 헤더가 보장. 확장은 repo 루트의 common을 재사용, README.md §9).
from common.obs_preprocess import obs_layout
from common.teacher_contract import TEACHER_OBJECT_POS_SLICE
from common.types import PreprocessCfg

# ══════════════════════════════════════════════════════════════════════════════
# 1) 아키텍처 상수 — ★러너 cfg(rsl_rl_ppo_cfg)와 export 재구성(Export_Policy)의 단일 소스★
#    여기 값을 바꾸면 '다음 학습 run부터' 반영된다. 과거 ckpt는 Export_Policy가
#    state_dict에서 차원을 역추론하므로 차원 변경은 안전하지만, activation은
#    state_dict에 남지 않는다 — 아래 ACTIVATION 주석 참조.
# ══════════════════════════════════════════════════════════════════════════════

# 현행 flat vision obs 50206dim(112² RGB-D + proprio 30)을 그대로 받는 기본 MLP.
# 업스트림 lift PPO 기본([256,128,64], rsl_rl_ppo_cfg.py)보다 첫 층을 키움(고차원 입력).
ACTOR_HIDDEN_DIMS: list[int] = [512, 256, 128]
# critic은 특권 상태 obs(저차원)를 받지만 1차는 대칭 구성으로 시작 — 튜닝 지점.
CRITIC_HIDDEN_DIMS: list[int] = [512, 256, 128]
# ⚠️ activation은 ckpt state_dict에 흔적이 남지 않는다(가중치 형상만 저장).
#    Export_Policy의 오프라인 재구성은 이 상수를 신뢰한다 — 러너 cfg에서 이 상수를
#    쓰지 않고 손으로 바꾸는 것 금지(export된 정책이 조용히 다른 함수를 계산하게 됨).
ACTIVATION: str = "elu"

# ★F79★ 수직 파지 시 환경이 무시하는 액션 채널(기울임 3·4)을 증류 타깃에서도 제외한다.
#   configs/common.yaml action.vertical_grasp_only 단일 소스 — 환경(delayed_actions)과
#   같은 값을 읽어야 마스크가 갈라지지 않는다.
try:
    from common.config_io import configs_dir as _cfg_dir
    from common.config_io import load_yaml as _load_yaml

    _VERTICAL_GRASP_ONLY: bool = bool(
        _load_yaml(_cfg_dir() / "common.yaml")["action"].get("vertical_grasp_only", False)
    )
except Exception:  # noqa: BLE001 - 설정을 못 읽으면 마스크를 걸지 않는다(구 동작 보존)
    _VERTICAL_GRASP_ONLY = False

def distillation_teacher_actions(actions: torch.Tensor, vertical: bool) -> torch.Tensor:
    """Preserve unbounded IK commands; canonicalize only ignored/binary channels.

    Clipping the last-action observation does not clip execution. Arm targets
    above one remain distinct pose offsets. The gripper uses strict ``a < 0``.
    """
    if actions.shape[-1] != 7:
        raise ValueError("distillation requires the 7D IK + binary gripper contract")
    teacher_actions = actions.clone()
    if vertical:
        teacher_actions[..., 3:5] = 0.0
    teacher_actions[..., -1] = torch.where(actions[..., -1] < 0.0, -1.0, 1.0)
    return teacher_actions


# ★F87(2026-09-08)★ 증류 손실의 채널 정규화 상수 — 근거는 DistillationChannelBalanced
#   위의 실측 주석. 여기(모듈 수준)에 두는 이유는 rsl_rl이 없는 환경에서도 테스트가
#   ★실제 손실 함수 자체★를 검증할 수 있어야 하기 때문이다(복제본 검증 금지).
_CHANNEL_MS_FLOOR: float = 0.01
_CHANNEL_MS_DEGENERATE: float = 1e-6


def channel_balanced_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """채널별 타깃 평균제곱으로 정규화한 MSE (rsl_rl Distillation.loss_fn 계약).

    퇴화 채널(타깃이 항상 0인 무시 채널)은 가중 1.0으로 고정하고, 나머지는
    자기 스케일로 나누되 바닥값 아래로는 증폭하지 않는다.
    """
    flat_target = target.reshape(-1, target.shape[-1])
    mean_square = flat_target.pow(2).mean(dim=0).detach()
    scale = torch.where(
        mean_square < _CHANNEL_MS_DEGENERATE,
        torch.ones_like(mean_square),
        mean_square.clamp_min(_CHANNEL_MS_FLOOR),
    )
    return ((pred - target).pow(2) / scale).mean()

# Phase1 teacher actor의 hidden dims — 증류(BinPickDistillRunnerCfg.teacher_hidden_dims)가
# 이 값과 '정확히' 일치해야 Phase1 ckpt가 size mismatch 없이 teacher로 로드된다.
# ⚠️ Phase 독립 실행 규칙(README.md §9)상 Phase1 러너 cfg를 import하지 않고
#   사본 상수로 둔다 — Phase1 BinPickTeacherRunnerCfg.policy.actor_hidden_dims 변경 시
#   여기도 동기화할 것. 실제 집행은 VisionPick_Train.py --distill-from가 teacher ckpt의
#   가중치 형상(infer_mlp_dims_from_state_dict)과 대조해 불일치 시 기동을 거부한다.
TEACHER_ACTOR_HIDDEN_DIMS: list[int] = [256, 128, 64]

# rsl_rl.utils.resolve_nn_activation의 최소 미러 — 이 모듈을 rsl_rl 없이도 쓰기 위함.
# (검증: rsl_rl/modules/actor_critic.py 줄 12,36이 같은 이름 문자열을 소비)
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
}


def build_mlp_actor(
    num_obs: int,
    num_actions: int,
    hidden_dims: list[int] | None = None,
    activation: str | None = None,
) -> nn.Sequential:
    """rsl_rl ActorCritic.actor와 ★state_dict 키 호환★ MLP 생성.

    구성 순서는 rsl_rl 3.1.2 ActorCritic actor MLP의 Linear/activation 순서를 미러:
      Linear(obs,h0), act, Linear(h0,h1), act, ..., Linear(h_last, actions)
    → 서브모듈 인덱스가 동일해 ckpt의 "actor.0.weight", "actor.2.weight", ... 키를
      접두사만 벗겨 strict=True로 로드할 수 있다 (Export_Policy가 사용).
    """
    hidden_dims = list(hidden_dims if hidden_dims is not None else ACTOR_HIDDEN_DIMS)
    act_name = activation if activation is not None else ACTIVATION
    if act_name not in _ACTIVATIONS:
        raise ValueError(f"지원하지 않는 activation: {act_name!r} (지원: {sorted(_ACTIVATIONS)})")
    act_cls = _ACTIVATIONS[act_name]

    layers: list[nn.Module] = [nn.Linear(num_obs, hidden_dims[0]), act_cls()]
    for i in range(len(hidden_dims)):
        if i == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[i], num_actions))
        else:
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(act_cls())
    return nn.Sequential(*layers)


def infer_mlp_dims_from_state_dict(
    actor_sd: dict[str, torch.Tensor],
) -> tuple[int, list[int], int]:
    """actor 서브 state_dict("0.weight", "2.weight", ...)에서 (obs_dim, hidden_dims, action_dim) 역추론.

    nn.Linear.weight 형상은 (out_features, in_features) — 인덱스 오름차순으로 정렬해
    첫 층 in=obs_dim, 마지막 층 out=action_dim, 중간 out들=hidden_dims를 얻는다.
    순수 로직(런타임 의존 없음) — Export_Policy의 재구성/검증에 사용.
    """
    indexed: list[tuple[int, torch.Tensor]] = []
    for key, val in actor_sd.items():
        parts = key.split(".")
        if len(parts) == 2 and parts[1] == "weight" and parts[0].isdigit():
            indexed.append((int(parts[0]), val))
    if not indexed:
        raise ValueError(f"actor state_dict에서 Linear weight 키를 찾지 못함: {sorted(actor_sd)}")
    indexed.sort(key=lambda t: t[0])
    shapes = [tuple(w.shape) for _, w in indexed]  # [(out,in), ...]
    obs_dim = int(shapes[0][1])
    action_dim = int(shapes[-1][0])
    hidden_dims = [int(out) for out, _ in shapes[:-1]]
    # 연속성 검증: 층 i의 out == 층 i+1의 in (아니면 MLP가 아니거나 키 누락)
    for (out_i, _), (_, in_next) in zip(shapes[:-1], shapes[1:]):
        if out_i != in_next:
            raise ValueError(f"MLP 층 연결 불일치: {shapes} — ckpt 키 누락/변형 의심")
    return obs_dim, hidden_dims, action_dim


def split_flat_obs(
    flat_obs: torch.Tensor, pre_cfg: PreprocessCfg
) -> dict[str, torch.Tensor]:
    """flat [N, obs_dim] → 이름별 텐서 dict (depth는 [N,1,H,W]로 복원).

    레이아웃은 common.obs_preprocess.obs_layout(계약서 본문)을 그대로 소비 —
    슬라이스 경계를 여기서 재계산하지 않는다(이중 소스 금지).
    CNN encoder(VisionActorCritic)의 forward 첫 단계로 사용.
    """
    if flat_obs.dim() != 2:
        raise ValueError(f"flat obs는 [N, obs_dim]이어야 함: {tuple(flat_obs.shape)}")
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name, size in obs_layout(pre_cfg):
        chunk = flat_obs[:, offset : offset + size]
        if name == "depth":
            chunk = chunk.reshape(-1, 1, pre_cfg.out_height, pre_cfg.out_width)
        elif name == "rgb":  # ★RGB-D 계약(2026-07-17)★ — C-H-W 평탄화의 역
            chunk = chunk.reshape(-1, 3, pre_cfg.out_height, pre_cfg.out_width)
        out[name] = chunk
        offset += size
    if offset != flat_obs.shape[1]:
        raise ValueError(
            f"obs_dim 불일치: 레이아웃 합 {offset} vs 실제 {flat_obs.shape[1]} — "
            f"configs/common.yaml preprocess가 학습 시점과 다르다"
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2) ★CNN depth student (증류용, 2026-07-13 구현)★
#    [근거 — 문헌 검증 워크플로 2026-07-13] 이미지 입력 정책은 conv 인코더가 표준:
#    LBC(CoRL19)·DextrAH-G(CoRL24, 160×120 depth CNN)·Levine16. flat MLP-on-pixels는
#    선례 없음 — 실측으로도 MLP student는 behavior loss ~9에서 바닥(reward 미발현).
#    인코더 사양 = vision-RL 정준 Nature-CNN(Mnih et al. 2015): 8×8s4-32 → 4×4s2-64 →
#    3×3s1-64 → FC. 해상도/채널은 common.yaml 계약(현행 112² RGB-D)에서 읽고,
#    proprio(30)와 concat 후 MLP 머리를 적용한다.
# ══════════════════════════════════════════════════════════════════════════════


def _image_contract() -> tuple[int, int, int]:
    """이미지 계약의 단일 소스 = configs/common.yaml preprocess (이중 소스 금지).

    반환: (h, w, channels) — channels = 1(depth) + 3(use_rgb 시 rgb).
    ★2026-07-17 RGB-D 전환(B안)★: flat 이미지 블록 = [depth(h*w)] (+ [rgb(3*h*w)]).
    """
    from common.config_io import build_preprocess_cfg, configs_dir, load_yaml

    pre = build_preprocess_cfg(load_yaml(configs_dir() / "common.yaml"))
    return int(pre.out_height), int(pre.out_width), 1 + (3 if pre.use_rgb else 0)


# ══════════════════════════════════════════════════════════════════════════════
# ★② b5 augmentation 카드(2026-07-23) — DrQ/RAD random shift (증류 학습 시 student 입력)★
#   근거: Kostrikov 2020(DrQ, "Image Augmentation Is All You Need")·Laskin 2020(RAD) —
#   픽셀 입력 정책의 표준 정규화. 저해상도 시각 인코더가 카메라 정합 미세오차·프레이밍
#   편차에 불변인 표현을 학습 → sim2real 시 실사 카메라 위치·크롭 편차에 강건.
#   ★RGB-D 정합★: depth·rgb에 '동일 이동량'(같은 grid)을 적용해 픽셀 정렬을 깨지 않는다
#   (서로 다른 shift면 depth-rgb 어긋남 = 계약 위반). teacher는 특권 상태(이미지 없음)라
#   무관 — student 입력만 변형. 학습 시(self.training)만 적용, eval/deploy는 원본 통과
#   (관측 계약 불변 — deploy 시 augmentation 없음).
# ══════════════════════════════════════════════════════════════════════════════
def _drq_shift_pair(
    depth: torch.Tensor, rgb: torch.Tensor, pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """depth[N,1,H,W]·rgb[N,3,H,W]에 DrQ random shift를 '동일 이동량'으로 적용(H==W 전제).

    denisyarats/drq random_shift 이식 — replicate pad 후 per-sample 정수 이동을
    grid_sample로 벡터화. 두 모달이 같은 grid를 공유해 RGB-D 픽셀 정합을 보존한다.
    """
    n, _, h, w = depth.shape
    if h != w:  # DrQ grid는 정사각 전제 — 계약(112²) 위반 시 조기 실패
        raise ValueError(f"random shift는 정사각 이미지 전제 — got {h}x{w}")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(
        -1.0 + eps, 1.0 - eps, h + 2 * pad, device=depth.device, dtype=depth.dtype
    )[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)          # (H,W,2)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)                   # (N,H,W,2)
    shift = torch.randint(
        0, 2 * pad + 1, size=(n, 1, 1, 2), device=depth.device, dtype=depth.dtype
    )
    shift = shift * (2.0 / (h + 2 * pad))
    grid = base_grid + shift                                               # 두 모달 공통 이동
    d = nn.functional.grid_sample(
        nn.functional.pad(depth, (pad, pad, pad, pad), mode="replicate"),
        grid, padding_mode="zeros", align_corners=False,
    )
    r = nn.functional.grid_sample(
        nn.functional.pad(rgb, (pad, pad, pad, pad), mode="replicate"),
        grid, padding_mode="zeros", align_corners=False,
    )
    return d, r


def _aug_pad_from_env() -> int:
    """BINPICK_AUG 게이트 → DrQ shift pad(px). 미설정=0(비활성, 기존 재현 보존)."""
    if os.environ.get("BINPICK_AUG", "").strip().lower() not in ("1", "true"):
        return 0
    return int(os.environ.get("BINPICK_AUG_PAD", "4"))


class DepthStudentNet(nn.Module):
    """flat [N, obs_dim] → 행동 [N, num_actions] — Nature-CNN 이미지 encoder + proprio concat.

    forward가 flat 관측을 그대로 받으므로 rsl_rl StudentTeacher의 `self.student(obs)`
    호출부와 즉시 호환된다(교체 주입용). 레이아웃 전제: 이미지 블록(depth[, rgb])이
    flat의 앞 블록(common.obs_preprocess.obs_layout 계약 순서).
    ★RGB-D(2026-07-17)★: use_rgb 계약이면 depth(1ch)+rgb(3ch)를 [N,4,h,w]로 쌓아
    단일 CNN에 입력(early fusion — DextrAH-RGB·Nature-CNN 관례). 이름은 호환성
    유지를 위해 DepthStudentNet 그대로 둔다(러너 cfg·export가 참조).
    """

    def __init__(self, num_obs: int, num_actions: int,
                 head_hidden_dims: list[int] | None = None, activation: str = ACTIVATION):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        self.h, self.w, self.in_ch = _image_contract()
        self.img_dim = self.h * self.w * self.in_ch  # flat 이미지 블록 총폭(depth+rgb)
        self.proprio_dim = num_obs - self.img_dim
        if self.proprio_dim <= 0:
            raise ValueError(f"obs_dim({num_obs}) ≤ 이미지({self.img_dim}) — 레이아웃 계약 위반")
        # Nature-CNN (Mnih 2015) — 출력 크기는 아래 dummy forward로 계약 해상도에서 실측한다.
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, kernel_size=8, stride=4), act_cls(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), act_cls(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), act_cls(),
            nn.Flatten(),
        )
        with torch.no_grad():  # conv 출력 차원을 해상도 무관하게 실측(하드코딩 금지)
            enc_out = self.encoder(torch.zeros(1, self.in_ch, self.h, self.w)).shape[1]
        self.enc_fc = nn.Sequential(nn.Linear(enc_out, 256), act_cls())
        dims = list(head_hidden_dims) if head_hidden_dims else [256, 128]
        layers: list[nn.Module] = []
        in_dim = 256 + self.proprio_dim
        for d in dims:
            layers += [nn.Linear(in_dim, d), act_cls()]
            in_dim = d
        layers.append(nn.Linear(in_dim, num_actions))
        self.head = nn.Sequential(*layers)

    def _image_from_flat(self, obs: torch.Tensor) -> torch.Tensor:
        """flat 앞 블록 → [N,C,h,w] (depth 1ch [+ rgb 3ch] — 레이아웃 순서 그대로)."""
        hw = self.h * self.w
        depth = obs[:, :hw].reshape(-1, 1, self.h, self.w)
        if self.in_ch == 1:
            return depth
        rgb = obs[:, hw : self.img_dim].reshape(-1, 3, self.h, self.w)
        return torch.cat([depth, rgb], dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        img = self._image_from_flat(obs)
        proprio = obs[:, self.img_dim :]
        z = self.enc_fc(self.encoder(img))
        return self.head(torch.cat([z, proprio], dim=1))


class ResNetStudentNet(nn.Module):
    """★② b1 카드(2026-07-20)★ — RGB는 ImageNet 사전학습 ResNet18, depth는 소형 CNN 브랜치.

    flat [N, obs_dim] → 행동 [N, num_actions] — DepthStudentNet과 동일한 flat 계약이라
    rsl_rl StudentTeacher 교체 주입에 즉시 호환. 근거(2026-07-19 문헌 조사):
      · DextrAH-RGB(2412.01791): 동형 세팅(state teacher→DAgger vision student)에서
        pretrained ResNet18 ★finetune 0.89 vs frozen 0.71★ — 단 8GB 예산이라 부분
        finetune(conv1~layer3 frozen, layer4만 학습)으로 시작(Lossless Adaptation 절충).
      · depth 브랜치는 스크래치 소형 CNN 유지 — ImageNet 사전학습은 depth 모달리티에
        이득 없음(depth 증류 계열이 전부 소형 CNN인 이유). late fusion(feature concat).
      · rgb는 [0,1] 계약 → ImageNet mean/std 정규화를 forward 내부에서 수행(buffer).
    ⚠️ frozen 브랜치의 BatchNorm은 train()에서도 eval 고정(러닝 통계 오염 방지 —
       train() 오버라이드). ⚠️ Export_Policy의 CNN 키 감지(encoder.0.*)는 이 클래스를
       모름 — b1 성능 채택 확정 후 export 분기 추가(TODO — 그 전 export 시도는 거부됨).
    """

    def __init__(self, num_obs: int, num_actions: int,
                 head_hidden_dims: list[int] | None = None, activation: str = ACTIVATION):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        self.h, self.w, self.in_ch = _image_contract()
        if self.in_ch != 4:
            raise ValueError(
                f"ResNetStudentNet은 RGB-D(use_rgb=true) 계약 전용 — in_ch={self.in_ch}. "
                "depth-only는 DepthStudentNet을 쓸 것 (configs/common.yaml preprocess.use_rgb)."
            )
        self.img_dim = self.h * self.w * self.in_ch
        self.proprio_dim = num_obs - self.img_dim
        if self.proprio_dim <= 0:
            raise ValueError(f"obs_dim({num_obs}) ≤ 이미지({self.img_dim}) — 레이아웃 계약 위반")

        # ── RGB 백본: torchvision ResNet18(IMAGENET1K_V1), avgpool까지(fc 제거 → 512)
        from torchvision.models import ResNet18_Weights, resnet18  # 지연 import(배포 PC 무관)

        _rn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.rgb_stem = nn.Sequential(_rn.conv1, _rn.bn1, _rn.relu, _rn.maxpool,
                                      _rn.layer1, _rn.layer2, _rn.layer3)  # frozen부
        self.rgb_layer4 = _rn.layer4                                       # 학습부
        self.rgb_pool = nn.Sequential(_rn.avgpool, nn.Flatten())           # → 512
        for p in self.rgb_stem.parameters():
            p.requires_grad_(False)
        # ImageNet 정규화 상수(입력 계약 [0,1] → 표준화) — state_dict 비저장 buffer
        self.register_buffer("_rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("_rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                             persistent=False)

        # ── depth 브랜치: Nature-CNN 1ch(스크래치 — DepthStudentNet.encoder와 동일 사양)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), act_cls(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), act_cls(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), act_cls(),
            nn.Flatten(),
        )
        with torch.no_grad():
            d_out = self.depth_encoder(torch.zeros(1, 1, self.h, self.w)).shape[1]

        # ── 융합·머리: (512 rgb + depth) → 256 → [proprio concat] → head
        self.enc_fc = nn.Sequential(nn.Linear(512 + d_out, 256), act_cls())
        dims = list(head_hidden_dims) if head_hidden_dims else [256, 128]
        layers: list[nn.Module] = []
        in_dim = 256 + self.proprio_dim
        for d in dims:
            layers += [nn.Linear(in_dim, d), act_cls()]
            in_dim = d
        layers.append(nn.Linear(in_dim, num_actions))
        self.head = nn.Sequential(*layers)
        # ★② b4 aux 카드(2026-07-23)★: 물체 3D 위치 예측 보조 헤드(융합 latent 256 → 위치).
        #   DextrAH-G/RGB(우리와 동형 파이프라인) 실증 — 저해상도 시각 인코더가 큐브 3D 위치를
        #   명시적으로 부호화하도록 강제(파지 접근 오차 15.3cm의 근본 = 위치 인지 부재를 직격,
        #   가림 순간 위치 대체로 hold에도 기여). aux_pred는 forward마다 갱신되고 DistillationAux가
        #   teacher obs 물체위치(현행 슬롯 24:27)와 MSE(β·L_pos). ★관측 계약 불변(출력 헤드)★.
        from common.config_io import configs_dir as _cd
        from common.config_io import load_yaml as _ly

        _nobj = int(_ly(_cd() / "sim.yaml").get("scene", {}).get("num_objects", 1))
        self.aux_dim = 3 * _nobj
        self.aux_head = nn.Sequential(nn.Linear(256, 64), act_cls(), nn.Linear(64, self.aux_dim))
        self.aux_pred: torch.Tensor | None = None  # DistillationAux가 소비(마지막 forward)
        # ★② b5 카드★ DrQ random shift pad(BINPICK_AUG) — 학습 시 student 입력 augmentation.
        self._aug_pad = _aug_pad_from_env()
        if self._aug_pad > 0:
            print(f"[vision-policy] ★DrQ random shift 활성 pad={self._aug_pad}px "
                  f"(RGB-D 동일 이동, 증류 학습 입력 전용)★")

    def train(self, mode: bool = True):
        """frozen 스템은 항상 eval 고정 — BatchNorm 러닝 통계가 증류 배치로 오염되면
        ImageNet 사전학습 특징이 표류한다(부분 finetune 표준 처리)."""
        super().train(mode)
        self.rgb_stem.eval()
        return self

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hw = self.h * self.w
        depth = obs[:, :hw].reshape(-1, 1, self.h, self.w)
        rgb = obs[:, hw : self.img_dim].reshape(-1, 3, self.h, self.w)
        if self.training and self._aug_pad > 0:  # ★DrQ random shift (RGB-D 동일 이동)★
            depth, rgb = _drq_shift_pair(depth, rgb, self._aug_pad)
        rgb = (rgb - self._rgb_mean) / self._rgb_std
        with torch.no_grad():  # frozen부는 gradient 그래프 자체를 만들지 않음(VRAM 절약)
            f = self.rgb_stem(rgb)
        z_rgb = self.rgb_pool(self.rgb_layer4(f))
        z_depth = self.depth_encoder(depth)
        z = self.enc_fc(torch.cat([z_rgb, z_depth], dim=1))
        self.aux_pred = self.aux_head(z)  # ★물체 위치 예측(DistillationAux가 loss에 사용)★
        return self.head(torch.cat([z, obs[:, self.img_dim :]], dim=1))


try:
    from rsl_rl.modules import StudentTeacher as _RslStudentTeacher  # torch만 의존
except ImportError:  # 배포 PC(env_ur_real) 등 rsl_rl 부재 환경
    _RslStudentTeacher = None  # type: ignore[assignment,misc]

try:  # ★b4 aux 카드★ — 증류 알고리즘에 물체위치 보조 손실 추가용
    from rsl_rl.algorithms import Distillation as _RslDistillation
except ImportError:
    _RslDistillation = None  # type: ignore[assignment,misc]


if _RslStudentTeacher is not None:

    class VisionStudentTeacher(_RslStudentTeacher):  # type: ignore[valid-type,misc]
        """StudentTeacher의 student만 CNN(DepthStudentNet)으로 교체한 증류 정책.

        교체 안전성(rsl_rl 3.1.2 소스 재검증(2026-07-17 포팅)):
          · 부모의 self.student 사용처는 update_distribution/act_inference의 순수 호출
            `self.student(obs)` 뿐 — forward 시그니처 동일이라 교체 무해.
          · Distillation 옵티마이저는 alg 생성 시 self.policy.student.parameters()를
            읽는다(distillation.py) — 본 __init__이 그 전에 실행되므로 CNN 파라미터가 등록됨.
          · teacher 경로·ckpt 자동 매핑("actor.*"→teacher, load_state_dict)은 그대로 상속.
        student_hidden_dims는 CNN 머리(head)의 hidden dims로 재해석된다.
        """

        def __init__(self, obs, obs_groups, num_actions,
                     student_hidden_dims=(256, 128), teacher_hidden_dims=(256, 128, 64),
                     activation: str = ACTIVATION, init_noise_std: float = 0.1,
                     collection_mode: str = "deterministic", **kwargs):
            if collection_mode not in ("deterministic", "sampled"):
                raise ValueError("collection_mode must be deterministic or sampled")
            self.collection_mode = collection_mode
            # ★rsl_rl 3.1.2 계약(2026-07-17 포팅)★: (obs: TensorDict, obs_groups, num_actions).
            #   부모 act/act_inference가 get_student_obs로 그룹 flat concat '후' self.student를
            #   호출(student_teacher.py L126-135)하므로 CNN forward는 무변경 호환.
            super().__init__(obs, obs_groups, num_actions,
                             student_hidden_dims=list(student_hidden_dims),
                             teacher_hidden_dims=list(teacher_hidden_dims),
                             activation=activation, init_noise_std=init_noise_std, **kwargs)
            # ★부모가 만든 MLP student를 CNN으로 교체★ (입력폭 = policy 그룹 폭 합 — 부모 산식 미러)
            num_student_obs = sum(int(obs[g].shape[-1]) for g in obs_groups["policy"])
            self.student = DepthStudentNet(num_student_obs, num_actions,
                                           list(student_hidden_dims), activation)
            print(f"[vision-policy] ★student = CNN(DepthStudentNet)으로 교체★\n{self.student}")

        def act(self, obs, **kwargs):
            # The MSE objective never optimizes std. Do not inject fixed 2cm
            # command noise throughout precision-placement training by default.
            if self.collection_mode == "deterministic":
                student_obs = self.student_obs_normalizer(self.get_student_obs(obs))
                self._update_distribution(student_obs)
                return self.distribution.mean
            return super().act(obs, **kwargs)

        @property
        def action_std(self):
            std = super().action_std
            return torch.zeros_like(std) if self.collection_mode == "deterministic" else std

        def evaluate(self, obs):
            """증류 타깃(teacher 행동) — ★환경이 무시하는 채널을 0으로 정리한다★.

            ★★F79(2026-09-01) 실측 기반 수정★★
              [사고] F78 수직 파지 제약은 환경(delayed_actions)에서 액션 채널 3·4(기울임)를
                0으로 만든다. 그런데 teacher ★정책 자체★는 그 채널을 계속 출력하고, 환경이
                무시하므로 ★gradient가 걸리지 않는다★ — 값이 아무 제약 없이 표류한다.
              [실측] 채택 teacher(model_18550) 채널별 출력 크기:
                  사용 채널(0·1·2·5) 평균 2.31  vs  기울임 채널(3·4) 평균 7.19 = ★3.1배★
                즉 두 채널이 ★의미 없는 큰 노이즈★다.
              [결과] 증류는 7채널 전부에 MSE를 건다. student가 그 노이즈까지 회귀하려 해
                Loss/behavior가 0.826까지 올랐다(기준선 0.221의 3.7배) — 학습 용량이
                배울 수 없는 것에 낭비되고 placed는 기준선의 22%에 머물렀다.
              [정정] 환경이 무시하는 채널은 ★타깃에서도 무시★한다. student는 그 채널을
                0으로 배우고(무해 — 어차피 환경이 0으로 만든다) 나머지 5채널에 용량을 쓴다.

            ⚠️ 이 마스크는 환경의 vertical_grasp_only와 ★같은 인덱스★여야 한다.
               delayed_actions는 actions[:, 3:5]를 0으로 만든다(tests/test_vertical_grasp.py가 강제).
            """
            return distillation_teacher_actions(super().evaluate(obs), _VERTICAL_GRASP_ONLY)

    class VisionStudentTeacherResNet(VisionStudentTeacher):
        """★② b1 카드(2026-07-20)★ — student만 ResNetStudentNet(ImageNet ResNet18)으로 교체.

        VisionStudentTeacher와 동일한 교체 주입 패턴(부모 __init__이 만든 student를
        덮어씀 — Distillation 옵티마이저는 그 후 parameters()를 읽으므로 안전).
        frozen 스템(requires_grad=False) 파라미터가 옵티마이저에 포함돼도 Adam은
        grad=None 파라미터를 스킵하므로 무해(torch 표준 동작).
        """

        def __init__(self, obs, obs_groups, num_actions,
                     student_hidden_dims=(256, 128), teacher_hidden_dims=(256, 128, 64),
                     activation: str = ACTIVATION, init_noise_std: float = 0.1, **kwargs):
            super().__init__(obs, obs_groups, num_actions,
                             student_hidden_dims=student_hidden_dims,
                             teacher_hidden_dims=teacher_hidden_dims,
                             activation=activation, init_noise_std=init_noise_std, **kwargs)
            num_student_obs = sum(int(obs[g].shape[-1]) for g in obs_groups["policy"])
            self.student = ResNetStudentNet(num_student_obs, num_actions,
                                            list(student_hidden_dims), activation)
            _n_train = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            _n_frozen = sum(p.numel() for p in self.student.parameters() if not p.requires_grad)
            print(f"[vision-policy] ★student = ResNetStudentNet(ImageNet ResNet18, "
                  f"layer4+머리만 학습: {_n_train:,} / frozen {_n_frozen:,})으로 교체★")

    class DepthEncoder(nn.Module):
        """flat [N, obs_dim] → [N, enc_dim] — Nature-CNN(이미지) + proprio 통과 concat.

        VisionStudentTeacherRecurrent의 GRU '앞단' 인코더 (행동 머리는 없음).
        ★RGB-D(2026-07-17)★: DepthStudentNet과 동일한 이미지 계약(_image_contract).
        """

        def __init__(self, num_obs: int, activation: str = ACTIVATION):
            super().__init__()
            act_cls = _ACTIVATIONS[activation]
            self.h, self.w, self.in_ch = _image_contract()
            self.img_dim = self.h * self.w * self.in_ch
            self.proprio_dim = num_obs - self.img_dim
            if self.proprio_dim <= 0:
                raise ValueError(f"obs_dim({num_obs}) ≤ 이미지({self.img_dim}) — 레이아웃 계약 위반")
            self.encoder = nn.Sequential(
                nn.Conv2d(self.in_ch, 32, kernel_size=8, stride=4), act_cls(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), act_cls(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), act_cls(),
                nn.Flatten(),
            )
            with torch.no_grad():
                conv_out = self.encoder(torch.zeros(1, self.in_ch, self.h, self.w)).shape[1]
            self.enc_fc = nn.Sequential(nn.Linear(conv_out, 256), act_cls())
            self.out_dim = 256 + self.proprio_dim

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            hw = self.h * self.w
            depth = obs[:, :hw].reshape(-1, 1, self.h, self.w)
            if self.in_ch > 1:
                rgb = obs[:, hw : self.img_dim].reshape(-1, 3, self.h, self.w)
                img = torch.cat([depth, rgb], dim=1)
            else:
                img = depth
            proprio = obs[:, self.img_dim :]
            return torch.cat([self.enc_fc(self.encoder(img)), proprio], dim=1)


    try:
        from rsl_rl.modules import StudentTeacherRecurrent as _RslSTR
    except ImportError:
        _RslSTR = None  # type: ignore[assignment,misc]

    if _RslSTR is not None:

        class VisionStudentTeacherRecurrent(_RslSTR):  # type: ignore[valid-type,misc]
            """★CNN 인코더 + GRU 기억 student★ — 관측 결손(속도·가림) 카드 (2026-07-14).

            [근거 — 문헌 검증 워크플로] teacher 특권 관측(object_vel·접촉력)을 student가
            순간 관측으로 못 볼 때 시간 축으로 복원하는 것이 표준: Lee(SciRob20) history
            TCN student·Miki(SciRob22) GRU belief encoder·RMA(RSS21) 50-step history.
            우리 결손 = ①단일 depth 프레임에 속도 정보 없음 ②수직 하방 카메라라 파지
            순간 그리퍼가 큐브를 가림. rsl_rl 3.1.2 StudentTeacherRecurrent가 1급 지원
            (Distillation의 gradient_length = truncated BPTT).

            구조: obs → DepthEncoder(CNN+proprio) → memory_s(GRU) → student MLP 머리.
            부모는 memory_s가 raw obs를 받지만, 여기서는 ★인코딩을 memory 앞단★에 둔다
            — act/act_inference만 오버라이드(부모 구현 미러 + 인코더 1줄 선행).
            teacher 경로(비재귀)·ckpt actor.*→teacher 매핑은 그대로 상속.
            """

            def __init__(self, obs, obs_groups, num_actions,
                         student_hidden_dims=(256, 128), teacher_hidden_dims=(256, 128, 64),
                         activation: str = ACTIVATION, rnn_type: str = "gru",
                         rnn_hidden_dim: int = 256, rnn_num_layers: int = 1,
                         init_noise_std: float = 0.1, **kwargs):
                kwargs.pop("teacher_recurrent", None)  # 지원 안 함(teacher는 항상 MLP)
                from tensordict import TensorDict  # rsl_rl 3.x 동봉 의존 — 이 블록 한정

                # 인코더 출력 차원 먼저 산정(로컬 임시 인스턴스 — self 등록 전이라 허용)
                num_student_obs = sum(int(obs[g].shape[-1]) for g in obs_groups["policy"])
                _tmp = DepthEncoder(num_student_obs, activation)
                # ★부모 memory_s 입력폭 = policy 그룹 폭 합★ → 인코더 출력폭으로 대체하기 위해
                #   policy 그룹만 인코더 폭 자리표시 텐서로 바꾼 fake obs를 부모에 전달
                #   (부모는 shape[-1]만 소비 — 3.1.2 student_teacher_recurrent.py 차원 산정부).
                assert len(obs_groups["policy"]) == 1, "policy 그룹은 단일(flat) 계약"
                _pg = obs_groups["policy"][0]
                fake = {k: obs[k] for k in obs.keys()}
                fake[_pg] = torch.zeros(obs[_pg].shape[0], _tmp.out_dim, device=obs[_pg].device)
                fake_td = TensorDict(fake, batch_size=obs.batch_size)
                super().__init__(fake_td, obs_groups, num_actions,
                                 student_hidden_dims=list(student_hidden_dims),
                                 teacher_hidden_dims=list(teacher_hidden_dims),
                                 activation=activation, rnn_type=rnn_type,
                                 rnn_hidden_dim=rnn_hidden_dim, rnn_num_layers=rnn_num_layers,
                                 init_noise_std=init_noise_std, teacher_recurrent=False, **kwargs)
                self.depth_encoder = _tmp
                # ★업스트림 결함 보정(2026-07-18 — 기동 크래시로 발견)★:
                #   3.1.2 student_teacher_recurrent.py L84가 teacher_recurrent=False여도
                #   teacher MLP 입력을 rnn_hidden_dim으로 짓는다. 그러나 evaluate()
                #   (L160-167)는 비재귀 teacher에 '원시 teacher 관측'을 직접 먹인다 —
                #   ckpt actor.*(Teacher 관측 폭) 매핑도, 런타임 evaluate도 모두 불일치.
                #   → teacher를 관측 폭으로 재구축(feedforward StudentTeacher와 동일 산식).
                from rsl_rl.networks import MLP as _RslMLP

                _n_teacher = sum(int(obs[g].shape[-1]) for g in obs_groups["teacher"])
                self.teacher = _RslMLP(_n_teacher, num_actions,
                                       list(teacher_hidden_dims), activation)
                self.teacher.eval()
                print(f"[vision-policy] ★recurrent student = CNN({_tmp.out_dim}) → "
                      f"{rnn_type.upper()}({rnn_hidden_dim}) → 머리{list(student_hidden_dims)}★ "
                      f"(teacher MLP 재구축: 입력 {_n_teacher} — 업스트림 L84 보정)")

            def get_student_obs(self, obs):
                # ★인코딩을 memory 앞단에★ — 부모 act/act_inference가 이 훅 결과를
                #   normalizer(인코더 폭 기준 생성)→memory_s에 먹인다(3.1.2 L126-135 경로).
                return self.depth_encoder(super().get_student_obs(obs))

            def evaluate(self, obs):
                return distillation_teacher_actions(super().evaluate(obs), _VERTICAL_GRASP_ONLY)
            # evaluate(teacher)·reset·hidden state 관리는 부모 그대로 (teacher 비재귀)

    if _RslDistillation is not None:

        # ★★F87(2026-09-08) 증류 손실의 채널 정규화 — 병진이 학습되지 않던 근본★★
        #   [측정] student가 실제로 방문하는 상태에서 teacher 타깃의 MSE 기여율
        #     (model_24999 · 100배치 × 128env · Vision-Eval-v0):
        #         dx 0.8% · dy 1.9% · dz 6.8% · rz 55.0% · grip 35.6%
        #     ⇒ ★3cm 배달을 결정하는 병진 3채널의 합이 9.5%★뿐이다. ±1 클램프(F80)를
        #       걸어도 12.5%로 사실상 같다 — F80이 근거로 삼은 "클램프 시 병진 52%"는
        #       ★무작위 OOD 관측★에서 잰 값이라 실제 분포에서는 성립하지 않았다.
        #   [왜 이렇게 되나] 액션 공간의 스케일이 채널마다 다른 의미를 갖는다. 0.2 rad의
        #     회전은 '작은' 움직임이라 teacher가 |rz|≈0.98을 상시 출력하는 반면, 1cm 이동은
        #     dx=0.05에 불과하다(|dx|≈0.08). MSE는 제곱이라 큰 채널이 gradient를 독식한다.
        #   [처방] 각 채널의 제곱오차를 ★그 배치에서의 타깃 평균제곱★으로 나눈다 —
        #     행동 차원 표준화는 behavior cloning의 표준 처방이고, 배치 통계라 분포가
        #     이동해도 자기교정된다. 두 가지 안전장치를 둔다:
        #       ① ★퇴화 채널★(수직 파지로 타깃이 항상 정확히 0인 기울임 3·4)은 평균제곱이
        #          0이라 나누면 무한 가중이 된다 → 가중 1.0으로 고정한다. 그 채널은 0으로
        #          눌러두기만 하면 되고(F79: 방치하면 표류해 타깃을 오염시킨다) 학습할 것이 없다.
        #       ② ★바닥값 0.01★(타깃 RMS 0.1 = 병진 2cm 상당) 아래로는 증폭하지 않는다 —
        #          그 이하는 제어 정밀도가 아니라 노이즈를 좇는 구간이다.
        #   [기대 효과] 위 실측 스케일 기준 병진 기여율 9.5% → 약 57%.
        #   ⚠️ 손실 절대값이 이전 계보와 비교 불가해진다(정규화 때문). 판정은 ★추세★와
        #     grasp/carry/placed 실측으로 한다 — 브리프 §7.3의 탈동조 관찰과 같은 원칙.
        #   대조군: BINPICK_DISTILL_LOSS=plain 이면 기존 무가중 MSE로 되돌아간다.
        class DistillationChannelBalanced(_RslDistillation):  # type: ignore[valid-type,misc]
            """표준 Distillation + 채널별 타깃 스케일 정규화(위 F87 주석 참조)."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_fn = channel_balanced_mse

        class DistillationAux(_RslDistillation):  # type: ignore[valid-type,misc]
            """★b4 aux 카드★ — 표준 Distillation + 물체 3D 위치 예측 보조 손실.

            L = L_behavior(action 모방) + β·‖x̂_obj − x_obj(teacher)‖² (β=BINPICK_AUX_COEF, 기본 0.1).
            근거: DextrAH-G/RGB(우리와 동형 state-teacher→RGBD-student DAgger 파이프라인)가
            공통 채택 — student 시각 인코더가 큐브 3D 위치를 명시적으로 부호화하도록 강제해
            파지 접근 오차(15.3cm)의 근본(위치 인지 부재)을 직격. teacher obs 물체위치 슬롯 =
            현행 [24:27](joint6+vel6+gripper2+tcp7+target3 다음). 정확한 slice는
            `common.teacher_contract`에서 읽는다. ★update()는
            부모 코드 복제 + aux 항만 추가★(부모가 aux를 모르므로 오버라이드 불가피).
            aux_pred = ResNetStudentNet.forward가 매 스텝 저장(act_inference 경유).
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.aux_coef = float(os.environ.get("BINPICK_AUX_COEF", "0.1"))
                # gripper_pos 2D가 Teacher 관측에 삽입되며 구 22:25 → 현행 24:27로 이동했다.
                # 숫자를 재하드코딩하지 않고 common의 단일 계약에서 가져온다.
                self.aux_slice = TEACHER_OBJECT_POS_SLICE

            def update(self) -> dict[str, float]:
                self.num_updates += 1
                mean_behavior_loss = 0.0
                mean_aux_loss = 0.0
                loss = 0
                cnt = 0
                for _epoch in range(self.num_learning_epochs):
                    self.policy.reset(hidden_states=self.last_hidden_states)
                    self.policy.detach_hidden_states()
                    for obs, _, privileged_actions, dones in self.storage.generator():
                        actions = self.policy.act_inference(obs)
                        behavior_loss = self.loss_fn(actions, privileged_actions)
                        # ★aux: student가 예측한 물체위치 vs teacher obs 실제 물체위치★
                        aux_pred = self.policy.student.aux_pred
                        aux_target = obs["teacher"][:, self.aux_slice]
                        aux_loss = nn.functional.mse_loss(aux_pred, aux_target)
                        loss = loss + behavior_loss + self.aux_coef * aux_loss
                        mean_behavior_loss += behavior_loss.item()
                        mean_aux_loss += aux_loss.item()
                        cnt += 1
                        if cnt % self.gradient_length == 0:
                            self.optimizer.zero_grad()
                            loss.backward()
                            if self.is_multi_gpu:
                                self.reduce_parameters()
                            if self.max_grad_norm:
                                nn.utils.clip_grad_norm_(
                                    self.policy.student.parameters(), self.max_grad_norm)
                            self.optimizer.step()
                            self.policy.detach_hidden_states()
                            loss = 0
                        self.policy.reset(dones.view(-1))
                        self.policy.detach_hidden_states(dones.view(-1))
                mean_behavior_loss /= cnt
                mean_aux_loss /= cnt
                self.storage.clear()
                self.last_hidden_states = self.policy.get_hidden_states()
                self.policy.detach_hidden_states()
                return {"behavior": mean_behavior_loss, "aux_pos": mean_aux_loss}

    def register_rsl_rl_policy_classes() -> None:
        """rsl_rl 러너의 class_name eval 네임스페이스에 커스텀 정책 주입.

        rsl_rl 3.1.2 제약: runner가 `eval(class_name)`을 자기 모듈
        전역에서 수행 → 그 모듈 dict에 심으면 cfg 문자열로 지정 가능(포크 불필요).
        호출 시점: 이 모듈 import 시(아래) — Hydra가 러너 cfg를 파싱하며 본 모듈을
        import하므로 항상 러너 생성 이전이다.
        """
        # ★3.1.2: 증류는 별도 DistillationRunner가 class_name을 자기 모듈에서 eval★
        #   → 두 러너 모듈 모두에 주입(on_policy_runner는 PPO 경로용 유지).
        import rsl_rl.runners.distillation_runner as _dr
        import rsl_rl.runners.on_policy_runner as _opr

        for _m in (_opr, _dr):
            _m.VisionStudentTeacher = VisionStudentTeacher
            _m.VisionStudentTeacherResNet = VisionStudentTeacherResNet
            if _RslSTR is not None:
                _m.VisionStudentTeacherRecurrent = VisionStudentTeacherRecurrent
            if _RslDistillation is not None:
                _m.DistillationAux = DistillationAux  # ★b4 aux 알고리즘★
                _m.DistillationChannelBalanced = DistillationChannelBalanced  # ★F87★

    register_rsl_rl_policy_classes()


# ══════════════════════════════════════════════════════════════════════════════
# 3) CNN depth encoder ActorCritic — PPO 경로 확장 스캐폴드 (⚠️ 미구현)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from rsl_rl.modules import ActorCritic as _RslActorCritic  # torch만 의존 — isaaclab 아님
except ImportError:  # 배포 PC(env_ur_real) 등 rsl_rl 부재 환경 — 스캐폴드만 사용 불가
    _RslActorCritic = object  # type: ignore[assignment,misc]


class VisionActorCritic(_RslActorCritic):  # type: ignore[valid-type,misc]
    """★TGRL Path 2a(2026-07-24) — PPO + 모방 리워드셰이핑으로 증류갭 돌파★.

    actor = ResNetStudentNet(camera RGB-D), critic = 특권 상태 MLP(비대칭 AC, 부모 stock 유지),
    teacher = 특권 상태 MLP(frozen, Phase1 로딩). VisionPPO가 process_env_step에서
    r_imit = −‖a − μ_T(o^특권)‖² 를 환경보상에 β로 더한다(모방 리워드셰이핑, TGRL Path 2a).
    ★근거★: 현 Distillation은 순수 BC라 환경보상을 0% 써서(distillation.py:130) student가
    카메라로 못 보는 상태에서 스스로 학습할 신호가 없다 → placed 5.6% 고착. 환경보상 RL을
    켜서 '카메라로 도달 가능한' 정책을 학습하되 teacher를 리워드로 참고. 관측 shift가 없어
    b5 목표거리식 붕괴(F20)를 구조적으로 회피. ★관측계약(obs_layout)·decimation 불변★(알고리즘측).

    ⚠️ empirical_normalization=False 고정(RGB-D는 전처리에서 이미 [0,1] — 부모 stock 정규화 금지).
    ⚠️ 커스텀 정책 등록 = register_rsl_rl_policy_classes()의 monkey-patch(기존 패턴 재사용).
    teacher 로딩 = BINPICK_TEACHER_CKPT(Phase1 model_*.pt)의 actor.* → self.teacher.
    """

    is_recurrent = False

    def __init__(self, obs, obs_groups, num_actions,
                 actor_hidden_dims=(256, 128), critic_hidden_dims=(256, 256, 256),
                 teacher_hidden_dims=(256, 128, 64),
                 activation: str = ACTIVATION, init_noise_std: float = 0.1, **kwargs):
        # ★rsl_rl 3.1.2 ActorCritic 계약★: (obs: TensorDict, obs_groups, num_actions, ...).
        #   부모가 actor/critic을 stock MLP로 만든 뒤 아래에서 actor만 ResNet으로 교체(주입 패턴).
        super().__init__(obs, obs_groups, num_actions,
                         actor_hidden_dims=list(actor_hidden_dims),
                         critic_hidden_dims=list(critic_hidden_dims),
                         activation=activation, init_noise_std=init_noise_std, **kwargs)
        # ── actor 교체: camera RGB-D ResNet + ★TGRL warm-start(b3 증류 student 이식)★ ──
        #   근거 이력: F21(scratch=랜덤 인코더 붕괴) → 인코더 이식(BINPICK_TGRL_WARMSTART).
        #   F22(2026-07-25): encoder-only(랜덤 head)도 β0.3·0.05 공히 태스크보상 단조하락 —
        #   랜덤 head가 sparse env 보상서 파지를 학습 못하고 init(랜덤 탐색)보다 오히려 퇴화.
        #   → ★full warm-start(BINPICK_TGRL_WARMSTART_FULL)★: b3 정책 전체(인코더+head)를 이식해
        #   b3의 grasp 22%·placed 5.6%에서 '시작'하고 env RL로 refine(랜덤 head 제거 = 결정적 테스트).
        #   full 모드는 actor head를 b3와 동일 [256,128](b3 student_hidden_dims)로 맞춰 완전 로드.
        #   ⚠️★eval: full 모드 ckpt는 head [256,128]이므로 eval 시에도 BINPICK_TGRL_WARMSTART_FULL
        #   지정 필수★(actor 재구성 dims 정합 — 미지정 시 [512,256,128]로 재구성돼 state_dict 불일치).
        num_actor_obs = sum(int(obs[g].shape[-1]) for g in obs_groups["policy"])
        _ws_full = os.environ.get("BINPICK_TGRL_WARMSTART_FULL", "").strip()
        _ws = os.environ.get("BINPICK_TGRL_WARMSTART", "").strip()
        _B3_HEAD = [256, 128]  # b3 BinPickDistillRunnerCfg student_hidden_dims — full 이식 정합
        _actor_dims = _B3_HEAD if (_ws_full and os.path.isfile(_ws_full)) else list(actor_hidden_dims)
        self.actor = ResNetStudentNet(num_actor_obs, num_actions, _actor_dims, activation)
        if _ws_full and os.path.isfile(_ws_full):  # ★full: 인코더+head 전체 이식(b3에서 시작)★
            _wsd = torch.load(_ws_full, map_location="cpu", weights_only=True)
            _wsd = _wsd.get("model_state_dict", _wsd)
            _load = {k[len("student."):]: v for k, v in _wsd.items() if k.startswith("student.")}
            _miss, _unexp = self.actor.load_state_dict(_load, strict=False)
            print(f"[vision-policy] ★TGRL full warm-start★ {os.path.basename(_ws_full)} 정책 전체 "
                  f"{len(_load)-len(_unexp)}텐서 이식(인코더+head [256,128]) unexpected {len(_unexp)}·missing {len(_miss)}")
        elif _ws and os.path.isfile(_ws):  # encoder-only(F22서 불충분 판정 — 이력 보존)
            _wsd = torch.load(_ws, map_location="cpu", weights_only=True)
            _wsd = _wsd.get("model_state_dict", _wsd)
            _enc = ("rgb_stem", "rgb_layer4", "depth_encoder", "enc_fc")
            _load = {k[len("student."):]: v for k, v in _wsd.items()
                     if k.startswith("student.") and k[len("student."):].split(".")[0] in _enc}
            _miss, _unexp = self.actor.load_state_dict(_load, strict=False)
            print(f"[vision-policy] ★TGRL warm-start(encoder)★ {os.path.basename(_ws)} 인코더 "
                  f"{len(_load)-len(_unexp)}텐서 이식 (head fresh, unexpected {len(_unexp)})")
        elif _ws or _ws_full:
            print(f"[vision-policy] ⚠️TGRL warm-start ckpt 부재 — scratch로 진행")
        # ── ★init_noise_std 오버라이드(F23, 2026-07-25) — warm-start 파인튠 저탐색★ ──
        #   근거(F23): 4개 TGRL run 전부 action noise std가 1.0 고착(학습되나 감소 신호 부재 —
        #   vision value 추정 난이도·cold critic로 advantage noisy 추정). 정규화 액션서 std 1.0은
        #   큰 노이즈 → 정책이 'b3+노이즈'라 b3 skill(grasp22%) 발현 불가·정체. std가 sticky(init서
        #   거의 불변)라 ★init을 낮추면 그 값 유지★ → warm-start 파인튠은 저탐색(0.2)으로 시작해야
        #   b3 근처서 refine. 부모가 super에서 self.std를 init_noise_std로 만든 뒤 여기서 덮어씀.
        _init_std = os.environ.get("BINPICK_TGRL_INIT_STD", "").strip()
        if _init_std and hasattr(self, "std") and self.std is not None:
            with torch.no_grad():
                self.std.data.fill_(float(_init_std))
            print(f"[vision-policy] ★TGRL init_noise_std={float(_init_std)} 오버라이드★ (warm-start 파인튠 저탐색)")
        # ── teacher: 특권 상태 MLP(frozen) — 모방 리워드용 μ_T (obs_groups["critic"]=특권) ──
        self._teacher_obs_groups = list(obs_groups["critic"])
        num_teacher_obs = sum(int(obs[g].shape[-1]) for g in self._teacher_obs_groups)
        from rsl_rl.networks import MLP as _RslMLP
        self.teacher = _RslMLP(num_teacher_obs, num_actions, list(teacher_hidden_dims), activation)
        self.teacher.eval()
        for _p in self.teacher.parameters():
            _p.requires_grad = False
        # Phase1 teacher ckpt(actor.*) 로딩(경로 = BINPICK_TEACHER_CKPT)
        _tckpt = os.environ.get("BINPICK_TEACHER_CKPT", "").strip()
        if _tckpt and os.path.isfile(_tckpt):
            _sd = torch.load(_tckpt, map_location="cpu", weights_only=True)
            _sd = _sd.get("model_state_dict", _sd)
            _tsd = {k[len("actor."):]: v for k, v in _sd.items()
                    if k.startswith("actor.") and "actor_obs_normalizer" not in k}
            _miss, _unexp = self.teacher.load_state_dict(_tsd, strict=False)
            print(f"[vision-policy] ★TGRL teacher 로딩★ {os.path.basename(_tckpt)} "
                  f"(matched {len(_tsd)-len(_unexp)}·missing {len(_miss)})")
        else:
            print(f"[vision-policy] ⚠️TGRL teacher ckpt 미지정/부재(BINPICK_TEACHER_CKPT='{_tckpt}') "
                  "— 모방 리워드 무의미, 학습 전 반드시 지정할 것.")

    @torch.no_grad()
    def teacher_action_mean(self, obs) -> torch.Tensor:
        """특권 obs로 teacher 평균행동 μ_T (모방 리워드 r_imit=−‖a−μ_T‖² 계산용, VisionPPO가 호출)."""
        t_obs = torch.cat([obs[g] for g in self._teacher_obs_groups], dim=-1)
        return self.teacher(t_obs)


# ══════════════════════════════════════════════════════════════════════════════
# ★TGRL Path 2a 단계2 — VisionPPO (모방 리워드셰이핑)★
# ══════════════════════════════════════════════════════════════════════════════
try:
    from rsl_rl.algorithms import PPO as _RslPPO  # torch만 의존
except ImportError:
    _RslPPO = None  # type: ignore[assignment,misc]

if _RslPPO is not None:

    class VisionPPO(_RslPPO):  # type: ignore[valid-type,misc]
        """★TGRL Path 2a★ — PPO에 teacher 모방 리워드셰이핑을 추가한 알고리즘.

        process_env_step에서 r_imit = −‖a − μ_T(o^특권)‖² 를 환경보상에 β로 더한다
        (rewards += β·r_imit). 이렇게 하면 모방 신호가 GAE로 부트스트랩되어(value에 반영)
        placed 같은 sparse 장기보상에 강하고, student는 환경보상 RL로 '카메라로 도달 가능한'
        정책을 학습하되 teacher를 리워드로 참고한다. ★현 Distillation(순수 BC, 환경보상 0%)의
        placed 고착을 직격★. β = BINPICK_TGRL_BETA(기본 0.3, 스케일 민감 → 스윕 대상).
        나머지(롤아웃·GAE·surrogate·value·entropy·adaptive-LR)는 부모 PPO 그대로.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # ★β 기본 0.03(2026-07-25 오프라인 발견)★: r_imit=−‖a−μ_T‖²는 7D·비정규화라 초기
            #   ~−162(랜덤 정책). β 0.3이면 기여 −48.6로 환경보상(~1-3)을 압도해 순수 모방으로 붕괴.
            #   0.03이면 초기 기여 ~−5(수렴하며 →0). 스윕 대상 = BINPICK_TGRL_BETA(0.01/0.03/0.1).
            self.beta_imit = float(os.environ.get("BINPICK_TGRL_BETA", "0.03"))
            self._last_r_imit = 0.0
            print(f"[vision-policy] ★TGRL VisionPPO 활성 β_imit={self.beta_imit}★ "
                  "(모방 리워드셰이핑 r_imit=−‖a−μ_T‖²)")

        def process_env_step(self, obs, rewards, dones, extras):
            # ★모방 리워드셰이핑★: teacher 평균행동과의 제곱거리(음수)를 β로 가중해 보상에 가산.
            #   super가 rewards.clone()·intrinsic·time_out bootstrap을 처리(ppo.py:166-180).
            with torch.no_grad():
                mu_T = self.policy.teacher_action_mean(obs)          # 특권 obs → μ_T (frozen)
                r_imit = -((self.transition.actions - mu_T) ** 2).sum(dim=-1)  # −‖a−μ_T‖²
            self._last_r_imit = float(r_imit.mean())
            super().process_env_step(obs, rewards + self.beta_imit * r_imit, dones, extras)


# ══════════════════════════════════════════════════════════════════════════════
# ★TGRL Path 2a 단계3-a — VisionActorCritic·VisionPPO를 rsl_rl runner에 등록★
#   (기존 register_rsl_rl_policy_classes와 별도 — 이 클래스들이 그 함수 '뒤'에 정의되기 때문.
#    on_policy_runner.py가 class_name을 자기 모듈 전역에서 eval하므로 그 dict에 주입한다.)
# ══════════════════════════════════════════════════════════════════════════════
def _register_tgrl_classes() -> None:
    """VisionActorCritic(policy)·VisionPPO(algorithm)를 OnPolicyRunner의 eval 네임스페이스에 주입."""
    try:
        import rsl_rl.runners.on_policy_runner as _opr
    except ImportError:
        return
    _opr.VisionActorCritic = VisionActorCritic
    _opr.VisionPPO = VisionPPO


if _RslActorCritic is not object and _RslPPO is not None:
    _register_tgrl_classes()
