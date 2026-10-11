#!/usr/bin/env python
"""Phase 2 — 학습된 vision actor를 배포용으로 export (TorchScript + obs_spec.json).

목적: logs/rsl_rl/ur5e_binpick_vision/<run>/model_*.pt(최신 자동 선택)에서 actor만 꺼내
     TorchScript(exported/policy.pt)로 저장하고, 관측 계약서(exported/obs_spec.json)를
     configs/common.yaml에서 생성한다. 배포 PC의 common.policy_io.PolicyRunner가 이 두
     파일을 로드해 계약이 다르면 기동을 거부한다.
     지원 ckpt: ① MLP actor(teacher PPO "actor.*" / 구 MLP student "student.*")
     ② ★CNN student(DepthStudentNet — "student." 아래 encoder.*/enc_fc.*/head.*)★
     (RGB-D 증류 2026-07-17). GRU student(VisionStudentTeacherRecurrent)는 미지원 — 명시 거부.

★ AppLauncher 스크립트가 아니다 ★ — isaaclab/isaacsim을 import하지 않는다(순수 torch).
  아래 자기 재실행 가드는 Kit이 아니라 'torch가 있는 env python' 확보용이다.

실행 (프로젝트 루트; README의 전체 명령 참조):
  1. python Phase2\\Export_Policy.py --candidate-only --checkpoint <model_*.pt>
  2. Phase3 rehearsal로 candidate 전용 10,000-row capture와 parity sidecar 생성
  3. python Phase2\\Export_Policy.py --candidate-dir <dir> `
       --capture-manifest <capture_manifest.json> --parity-obs <sealed.npy>

candidate는 ``approved:false``이며 release_manifest를 만들지 않는다. 최종 단계만
정확히 ``exported/``에 3-file release를 원자 게시한다. ONNX는 미구현이라 명시 거부한다.

산출:
  - exported/policy.pt      # TorchScript actor (배포 PC에서 torch.jit.load만으로 추론)
  - exported/obs_spec.json  # 관측/행동 계약서 (common.policy_io.make_obs_spec)
  → 배포 번들: common/ + real/ + Phase4/ + deployment/ + configs/ + exported/
    및 명시된 license/URDF/FK 파일만 포함한다(README Phase 3/4 참조).
"""

# --- 0) torch가 있는 env python 보장 (필요 시 자기 재실행 — AppLauncher와 무관) ---
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()




try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import datetime as _dt
import glob
import io
import json
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# repo 루트(common) + 확장 소스(ur5e_binpick, 미설치 폴백)를 path에 — pip 설치 시엔 후자 불필요
for _p in (str(ROOT), str(ROOT / "source" / "ur5e_binpick")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ur5e_binpick.tasks.binpick.action_scheme import SIM_ACTION_SCHEME  # noqa: E402
from ur5e_binpick.tasks.binpick.agents.vision_policy import (  # noqa: E402
    ACTIVATION,
    DepthStudentNet,
    ResNetStudentNet,
    _image_contract,
    build_mlp_actor,
    infer_mlp_dims_from_state_dict,
)
from common.config_io import (  # noqa: E402
    build_action_cfg,
    build_preprocess_cfg,
    configs_dir,
    load_yaml,
)
from common.obs_preprocess import obs_dim  # noqa: E402
from common.policy_io import (  # noqa: E402
    OBS_SPEC_NAME,
    PolicyRunner,
    make_obs_spec,
    read_anchored_regular_file,
    save_obs_spec,
)
from deployment.contracts import (  # noqa: E402
    ContractError,
    sha256_file,
    validate_release_calibrations,
)
from scripts.adoption import verify_adopted_checkpoint  # noqa: E402
from scripts.checkpoint_io import load_anchored_checkpoint  # noqa: E402
from deployment.release import (  # noqa: E402
    CANDIDATE_MANIFEST_NAME,
    RELEASE_MANIFEST_NAME,
    candidate_receipt_binding,
    cleanup_release_staging,
    collect_training_run_provenance,
    create_release_staging,
    evaluate_sealed_parity_dataset,
    make_candidate_manifest,
    make_release_manifest,
    publish_candidate_staging,
    publish_release_staging,
    require_canonical_candidate_output,
    require_canonical_release_output,
    require_clean_git_commit,
    save_release_manifest,
    training_stack_provenance,
    verify_candidate,
    verify_release,
)

EXPERIMENT = "ur5e_binpick_vision"
# 업스트림 train.py 산출 위치: <repo>/logs/rsl_rl/<experiment>/<ts>/
STORE_ROOT = str(ROOT / "logs" / "rsl_rl" / EXPERIMENT)


def _latest_checkpoint(store_root: str) -> str | None:
    """store_root/<run>/model_*.pt 중 가장 최근(수정시각) 경로 — 순수 로직(isaaclab 불필요)."""
    cks = glob.glob(os.path.join(store_root, "*", "model_*.pt"))
    return max(cks, key=os.path.getmtime) if cks else None


class _ExportActor(nn.Module):
    """TorchScript export 래퍼 — 업스트림 exporter._TorchPolicyExporter의 비순환 경로 미러.

    (isaaclab_rl/rsl_rl/exporter.py 줄 47-101: actor 복사 + normalizer 합성 +
     forward = actor(normalizer(x)) + torch.jit.script. 여기서는 empirical_normalization을
     쓰지 않으므로 normalizer는 항상 Identity — 켠 ckpt는 아래에서 export를 거부한다.)
    """

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.normalizer = nn.Identity()
        self.actor = actor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))


def _extract_actor_state_dict(model_sd: dict) -> tuple[str, dict]:
    """rsl_rl ckpt의 model_state_dict에서 배포 대상 네트워크 서브 state_dict 추출.

    선택 규칙은 업스트림 exporter와 동일(exporter.py 줄 54-63):
      - "actor.*"   존재 → ActorCritic(PPO) run의 actor
      - "student.*" 존재 → StudentTeacher(Distillation) run의 student
    """
    if any(k.startswith("actor.") for k in model_sd):
        prefix = "actor."
    elif any(k.startswith("student.") for k in model_sd):
        prefix = "student."
    else:
        raise ValueError(
            f"ckpt에서 actor./student. 키를 찾지 못함 — rsl_rl model_*.pt가 맞는지 확인: "
            f"{sorted(model_sd)[:8]}..."
        )
    sub = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
    return prefix.rstrip("."), sub


# ══════════════════════════════════════════════════════════════════════════════
# ★CNN student(DepthStudentNet) 오프라인 재구성 (RGB-D 증류 전환 2026-07-17)★
#   VisionStudentTeacher ckpt는 "student." 아래에 encoder.*/enc_fc.*/head.* 서브키를
#   남긴다(vision_policy.DepthStudentNet) — MLP 역추론(infer_mlp_dims_from_state_dict)
#   으로는 재구성 불가라 전용 경로가 필요하다.
# ══════════════════════════════════════════════════════════════════════════════


def _looks_recurrent_student(model_sd: dict) -> bool:
    """GRU student(VisionStudentTeacherRecurrent) ckpt 흔적 감지 — 선제 거부용.

    재귀 ckpt는 memory_s.rnn.*(rsl_rl 3.1.2 Memory — student_teacher_recurrent.py 줄 69,
    networks/memory.py 줄 29)·depth_encoder.*(우리 확장의 CNN 앞단)가 "student."와 별도
    최상위 접두사로 저장된다. "student." 서브키만 보면 GRU 뒤의 평범한 MLP 머리라서
    MLP 경로로 오인 재구성될 수 있다(입력폭=rnn_hidden — 계약 불일치라는 오도 메시지로
    실패). 그래서 model_state_dict 전체에서 흔적을 보고 여기서 명시적으로 거부한다.
    """
    return any(
        k.startswith(("memory_s.", "depth_encoder.")) or ".rnn." in k for k in model_sd
    )


def _infer_cnn_head_dims_from_state_dict(
    student_sd: dict,
) -> tuple[list[int], int]:
    """student 서브 sd의 "head.{i}.weight" 형상에서 (head_hidden_dims, action_dim) 역추론.

    DepthStudentNet.head = Sequential(Linear, act, ..., Linear) — 마지막 Linear의
    out=action_dim, 그 앞 out들=head_hidden_dims (infer_mlp_dims_from_state_dict와
    같은 원리, 접두사 "head."만 다름). 순수 로직(런타임 의존 없음).
    """
    indexed: list[tuple[int, torch.Tensor]] = []
    for key, val in student_sd.items():
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "head" and parts[2] == "weight" and parts[1].isdigit():
            indexed.append((int(parts[1]), val))
    if not indexed:
        raise ValueError(
            f"CNN student sd에서 head.*.weight 키를 찾지 못함: {sorted(student_sd)[:8]}..."
        )
    indexed.sort(key=lambda t: t[0])
    shapes = [tuple(w.shape) for _, w in indexed]  # [(out,in), ...]
    for (out_i, _), (_, in_next) in zip(shapes[:-1], shapes[1:]):
        if out_i != in_next:
            raise ValueError(f"head MLP 층 연결 불일치: {shapes} — ckpt 키 누락/변형 의심")
    return [int(out) for out, _ in shapes[:-1]], int(shapes[-1][0])


def _rebuild_cnn_student(student_sd: dict, num_obs: int, num_actions: int) -> nn.Module:
    """CNN student(DepthStudentNet)를 ckpt 서브 sd에서 재구성 — 계약 가드 + strict 로드.

    num_obs/num_actions는 configs/common.yaml 계약값(obs_dim(pre_cfg)/action_dim)을
    받는다 — CNN ckpt에는 flat obs 폭이 직접 남지 않으므로(conv는 해상도 무관 형상)
    계약값으로 생성하고, 아래 가드+strict 로드가 ckpt와의 불일치를 잡는다.
    """
    # 가드 1) 이미지 채널 수: ckpt encoder.0.weight = (32, in_ch, 8, 8) ↔ 현 계약(_image_contract)
    h, w, in_ch = _image_contract()
    if "encoder.0.weight" not in student_sd:
        raise ValueError(
            f"CNN student sd에 encoder.0.weight가 없음 — DepthStudentNet ckpt가 아님: "
            f"{sorted(student_sd)[:8]}..."
        )
    ck_in_ch = int(student_sd["encoder.0.weight"].shape[1])
    if ck_in_ch != in_ch:
        raise ValueError(
            f"이미지 채널 계약 불일치: ckpt in_ch={ck_in_ch} vs 현 계약 in_ch={in_ch} "
            f"(configs/common.yaml use_rgb={'true' if in_ch == 4 else 'false'}). "
            + ("depth-only(1ch) ckpt를 RGB-D(4ch) 계약으로 export 시도 중 — "
               "이 ckpt를 쓰려면 use_rgb: false로 되돌려야 하지만, 그것은 계약 변경이므로 "
               "obs_spec/배포가 전부 갈립니다(README.md §3.2). 어느 쪽이 맞는지 확인 후 진행."
               if (ck_in_ch, in_ch) == (1, 4) else
               "학습 시점과 use_rgb 설정이 다릅니다 — configs/common.yaml을 ckpt 학습 시점과 "
               "맞추거나(계약 변경 주의) 올바른 ckpt를 지정하세요.")
        )

    # 가드 2) 행동 차원: head 마지막 Linear out ↔ 계약 action_dim
    head_dims, ck_act = _infer_cnn_head_dims_from_state_dict(student_sd)
    if ck_act != num_actions:
        raise ValueError(
            f"action_dim 불일치: ckpt head 출력={ck_act} vs 계약={num_actions} — "
            f"다른 실험의 ckpt이거나 configs/common.yaml action이 학습 시점과 다름"
        )

    # 가드 3) proprio 폭: ckpt head.0 입력 = enc_fc 출력폭 + proprio ↔ 계약 num_obs - h*w*in_ch
    enc_width = int(student_sd["enc_fc.0.weight"].shape[0])
    ck_proprio = int(student_sd["head.0.weight"].shape[1]) - enc_width
    contract_proprio = num_obs - h * w * in_ch
    if ck_proprio != contract_proprio:
        raise ValueError(
            f"proprio 폭 불일치: ckpt={ck_proprio} vs 계약={contract_proprio} "
            f"(obs_dim {num_obs} - 이미지 {h}x{w}x{in_ch}) — "
            f"configs/common.yaml preprocess/action이 학습 시점과 다르거나 다른 실험의 ckpt"
        )

    net = DepthStudentNet(num_obs, num_actions, head_dims, ACTIVATION)
    try:
        net.load_state_dict(student_sd, strict=True)
    except RuntimeError as err:  # 위 가드를 뚫는 잔여 불일치(예: 해상도 변경 → enc_fc 입력폭)
        raise ValueError(
            f"CNN student strict 로드 실패 — 해상도(out_height/out_width) 등 이미지 계약이 "
            f"학습 시점과 다른지 확인: {err}"
        ) from err
    print(f"[export] ★CNN student 재구성★: 이미지 {in_ch}ch {h}x{w} + proprio {ck_proprio} "
          f"→ head {head_dims} → {num_actions} (activation={ACTIVATION} 가정)")
    return net


def _rebuild_resnet_student(student_sd: dict, num_obs: int, num_actions: int) -> nn.Module:
    """ResNet student(ResNetStudentNet, b1~b3 카드)를 ckpt 서브 sd에서 재구성 — 계약 가드 + 로드.

    ResNetStudentNet = rgb_stem(frozen ResNet18)/rgb_layer4/depth_encoder/enc_fc/head(+aux_head).
    DepthStudentNet(encoder.*)과 달리 rgb_stem.*/depth_encoder.* 서브키를 남긴다. ★aux_head는
    비액션 경로★(DistillationAux 보조손실 전용, forward 반환은 head만) — pre-aux ckpt(b3 등)엔
    없으므로 strict=False로 로드하고 'missing이 aux_head/비영속 buffer뿐'인지 검증한다.
    """
    h, w, in_ch = _image_contract()
    if in_ch != 4:
        raise ValueError(
            f"ResNetStudentNet은 RGB-D(use_rgb=true, in_ch=4) 계약 전용 — 현 계약 in_ch={in_ch}. "
            "configs/common.yaml preprocess.use_rgb 확인."
        )
    # 가드 1) 채널: rgb_stem.0.weight=(64,3,7,7) ResNet18 conv1 / depth_encoder.0.weight=(32,1,8,8)
    if "rgb_stem.0.weight" not in student_sd:
        raise ValueError(f"ResNet student sd에 rgb_stem.0.weight 없음 — ResNetStudentNet ckpt 아님: "
                         f"{sorted(student_sd)[:8]}...")
    ck_rgb_ch = int(student_sd["rgb_stem.0.weight"].shape[1])
    ck_depth_ch = int(student_sd["depth_encoder.0.weight"].shape[1]) if "depth_encoder.0.weight" in student_sd else -1
    if ck_rgb_ch != 3 or ck_depth_ch != 1:
        raise ValueError(f"채널 계약 불일치: ckpt rgb={ck_rgb_ch}(≠3)/depth={ck_depth_ch}(≠1) — "
                         f"ResNetStudentNet(rgb 3ch+depth 1ch) ckpt 확인.")
    # 가드 2) 행동 차원: head 마지막 Linear out ↔ 계약
    head_dims, ck_act = _infer_cnn_head_dims_from_state_dict(student_sd)
    if ck_act != num_actions:
        raise ValueError(f"action_dim 불일치: ckpt head 출력={ck_act} vs 계약={num_actions}")
    # 가드 3) proprio: head.0 입력 = enc_fc 출력(고정 256) + proprio ↔ 계약
    ENC_OUT = 256  # ResNetStudentNet.enc_fc = Linear(512+d_out, 256) 고정
    ck_proprio = int(student_sd["head.0.weight"].shape[1]) - ENC_OUT
    contract_proprio = num_obs - h * w * in_ch
    if ck_proprio != contract_proprio:
        raise ValueError(f"proprio 폭 불일치: ckpt={ck_proprio} vs 계약={contract_proprio} "
                         f"(obs_dim {num_obs} - 이미지 {h}x{w}x{in_ch})")

    net = ResNetStudentNet(num_obs, num_actions, head_dims, ACTIVATION)
    miss, unexp = net.load_state_dict(student_sd, strict=False)
    # ★aux_head(pre-aux ckpt엔 없음)·비영속 buffer(_rgb_mean/_rgb_std)만 missing 허용★
    bad_miss = [k for k in miss if not k.startswith("aux_head.") and k not in ("_rgb_mean", "_rgb_std")]
    if bad_miss or unexp:
        raise ValueError(f"ResNet student 로드 불일치 — missing(비aux){bad_miss[:6]} / "
                         f"unexpected{list(unexp)[:6]}: 해상도·아키텍처가 학습 시점과 다른지 확인")
    net.aux_pred = None  # 배포 미사용(forward가 설정하나 반환 안 함)
    _n_aux = len([k for k in miss if k.startswith("aux_head.")])
    print(f"[export] ★ResNet student 재구성★: rgb 3ch ResNet18 + depth 1ch {h}x{w} + proprio "
          f"{ck_proprio} → head {head_dims} → {num_actions} (aux_head 미로드 {_n_aux}텐서=비액션, "
          f"activation={ACTIVATION})")
    return net


_ACTIVE_RELEASE_STAGING: tuple[Path, Path] | None = None


def _main_impl() -> int:
    global _ACTIVE_RELEASE_STAGING
    parser = argparse.ArgumentParser(description="UR5e BinPick vision actor export")
    parser.add_argument("--checkpoint", default=None,
                        help="model_*.pt 경로 (기본: logs/rsl_rl/ur5e_binpick_vision 최신)")
    parser.add_argument("--out-dir", default=None,
                        help="산출 폴더 (보안 계약상 정확히 <프로젝트 루트>/exported만 허용)")
    parser.add_argument("--candidate-only", action="store_true",
                        help="parity 전 비승인 candidate만 canonical ignored store에 원자 게시")
    parser.add_argument("--candidate-dir", type=Path, default=None,
                        help="최종 export가 소비할 exact candidate 디렉터리")
    parser.add_argument("--capture-manifest", type=Path, default=None,
                        help="Phase3가 봉인한 candidate-specific capture_manifest.json")
    parser.add_argument("--onnx", action="store_true", help="ONNX도 export (⚠️ TODO 미구현)")
    parser.add_argument(
        "--parity-obs",
        default=None,
        help="scripts/build_parity_dataset.py가 봉인한 logs/parity/*.npy (최소 10,000개)",
    )
    parser.add_argument(
        "--max-inference-p99-ms",
        type=float,
        default=10.0,
        help="CPU 단일 관측 추론 p99 계약 (기본/최대 10ms)",
    )
    args = parser.parse_args()
    if args.max_inference_p99_ms <= 0 or args.max_inference_p99_ms > 10.0:
        parser.error("--max-inference-p99-ms must be in (0, 10]")
    if args.onnx:
        parser.error("--onnx is not implemented and cannot enter candidate/release artifacts")
    if args.candidate_only:
        if any(value is not None for value in (
            args.out_dir, args.candidate_dir, args.capture_manifest, args.parity_obs
        )):
            parser.error(
                "--candidate-only accepts checkpoint only; output is canonical and parity-free"
            )
    elif args.candidate_dir is None or args.capture_manifest is None or args.parity_obs is None:
        parser.error(
            "final export requires --candidate-dir, --capture-manifest, and --parity-obs"
        )

    # Capture source provenance before creating or modifying any export artifact.
    # Calling the dirty-worktree check after policy/spec writes would make a clean
    # source checkout self-dirty and force every export to report ``unknown``.
    try:
        source_git_commit = require_clean_git_commit(ROOT)
        source_training_stack = training_stack_provenance(ROOT)
    except ContractError as exc:
        print(f"[export][error] release provenance failed before artifact creation: {exc}")
        return 3

    # 1) 계약 로드 — 학습과 같은 configs/common.yaml에서 (요소1 단일 소스)
    common_yaml = load_yaml(configs_dir() / "common.yaml")
    pre_cfg = build_preprocess_cfg(common_yaml)
    action_cfg = build_action_cfg(common_yaml)
    expected_obs = obs_dim(pre_cfg)
    expected_act = action_cfg.action_dim
    config_paths = [
        configs_dir() / "common.yaml",
        configs_dir() / "dr.yaml",
        configs_dir() / "safety.yaml",
        configs_dir() / "sim.yaml",
        *sorted((configs_dir() / "calib").glob("*.yaml")),
    ]
    print(f"[export] 계약: obs_dim={expected_obs}, action_dim={expected_act} (configs/common.yaml)")
    if not args.candidate_only:
        try:
            validate_release_calibrations(ROOT)
        except ContractError as exc:
            print(f"[export][blocked] final release calibration approval failed: {exc}")
            return 3

    # 2) 체크포인트 선택/로드 (rsl_rl save 형식 — on_policy_runner.py 줄 395-413)
    selected_candidate = None
    selected_candidate_binding = None
    if args.candidate_only:
        ckpt_path = args.checkpoint or _latest_checkpoint(STORE_ROOT)
    else:
        try:
            selected_candidate = verify_candidate(
                args.candidate_dir,
                project_root=ROOT,
                require_current_context=True,
            )
            selected_candidate_binding = candidate_receipt_binding(
                args.candidate_dir,
                selected_candidate,
                project_root=ROOT,
            )
        except ContractError as exc:
            print(f"[export][error] candidate verification failed: {exc}")
            return 3
        candidate_source = selected_candidate["source"]
        sealed_checkpoint = (
            ROOT / candidate_source["run_root"] / candidate_source["checkpoint_name"]
        ).resolve()
        if args.checkpoint is not None and Path(args.checkpoint).resolve() != sealed_checkpoint:
            print("[export][error] --checkpoint differs from the selected candidate receipt")
            return 3
        ckpt_path = str(sealed_checkpoint)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[export][error] 체크포인트를 찾지 못함: {ckpt_path!r}")
        print(f"                보관소: {STORE_ROOT} — 먼저 VisionPick_Train.py로 학습하세요.")
        return 1
    try:
        source_training_run = collect_training_run_provenance(
            Path(ckpt_path), Path(STORE_ROOT)
        )
        student_adoption = verify_adopted_checkpoint(
            Path(ckpt_path), project_root=ROOT, role="student"
        )
    except ContractError as exc:
        print(f"[export][error] checkpoint provenance/adoption failed: {exc}")
        return 3
    if selected_candidate is not None \
            and source_training_run != selected_candidate["source"]["training_run"]:
        print("[export][error] checkpoint receipt changed after candidate verification")
        return 3
    print(f"[export] ckpt 로드: {ckpt_path}")
    expected_checkpoint_sha256 = source_training_run["checkpoint_sha256"]
    if student_adoption.get("checkpoint_sha256") != expected_checkpoint_sha256:
        print("[export][error] adoption and training receipt checkpoint hashes differ")
        return 3
    try:
        loaded = load_anchored_checkpoint(
            ckpt_path,
            expected_sha256=expected_checkpoint_sha256,
            allowed_root=STORE_ROOT,
            torch_module=torch,
        )
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"[export][error] anchored checkpoint snapshot/load failed: {exc}")
        return 3
    model_sd = loaded["model_state_dict"]

    # 2b) empirical_normalization 가드 — ⚠️ TODO(계약서에 기록):
    #   켜고 학습한 ckpt는 obs_norm_state_dict(이동 mean/var)가 동봉된다
    #   (on_policy_runner.py 줄 407-410). 업스트림 exporter는 이를 normalizer 모듈로
    #   합성하지만(exporter.py 줄 71-75), 이 오프라인 재구성 경로에는 아직 미구현 —
    #   통계를 무시하면 배포 정책이 '조용히 다른 입력 분포'를 보게 되므로 export를 거부한다.
    #   해소 방법: (a) 러너 cfg의 empirical_normalization=False 유지(현 계약), 또는
    #   (b) 여기서 (x-mean)/sqrt(var+eps) 모듈을 재구성해 normalizer로 합성 + obs_spec에 기록.
    if "obs_norm_state_dict" in loaded:
        print("[export][error] ⚠️ 이 ckpt는 empirical_normalization=True로 학습됨 — "
              "정규화 통계 합성은 TODO(미구현)라 export를 거부합니다 (위 주석/README 참조).")
        return 2

    # 3) actor 재구성 — ⚠️ TODO(확인 필요): '정확한 재구성 절차'의 업스트림 대비 차이.
    #   업스트림 export는 '살아있는 policy 객체'를 deepcopy한다
    #   (exporter.py 줄 54-63: hasattr(policy,"actor") → copy.deepcopy(policy.actor)) —
    #   객체가 있으니 activation이 자동 보존된다. 여기는 오프라인(ckpt만) 재구성이라
    #   가중치 '형상'은 state_dict에서 역추론 가능하지만 activation은 남지 않는다 →
    #   vision_policy.ACTIVATION 상수(러너 cfg와 단일 소스)를 신뢰한다. 과거 run의
    #   cfg를 수동 변경했다면 이 가정이 깨진다 — run 폴더의 params/agent.yaml로 검증하는
    #   보강이 열려 있는 확인 사항.
    which, actor_sd = _extract_actor_state_dict(model_sd)
    if which != "student":
        print(
            f"[export][error] deployment releases are student-only; checkpoint selected {which!r}"
        )
        return 2

    # 3a) GRU student ckpt 선제 거부 — 은닉상태 시그니처(reset/시퀀싱) 계약이 필요한
    #     재귀 정책은 이 오프라인 exporter가 아직 다루지 못한다(오도 메시지 방지).
    if which == "student" and _looks_recurrent_student(model_sd):
        raise NotImplementedError(
            "[export] ⚠️ GRU student(VisionStudentTeacherRecurrent) ckpt 감지"
            "(memory_s/depth_encoder 가중치) — 재귀 정책 export는 미구현이라 거부합니다. "
            "비재귀 CNN student(VisionStudentTeacher) ckpt를 지정하거나 exporter를 확장하세요."
        )

    if which == "student" and any(k.startswith("rgb_stem.") for k in actor_sd):
        # 3b-1) ★ResNet student(ResNetStudentNet, b1~b3 — 배포후보 b3) 경로★
        print(f"[export] ckpt actor 종류={which}(ResNet — rgb_stem.* 서브키 감지)")
        actor = _rebuild_resnet_student(actor_sd, expected_obs, expected_act)
    elif which == "student" and any(k.startswith("encoder.") for k in actor_sd):
        # 3b-2) ★CNN student(DepthStudentNet) 경로★ — 계약 가드/로드는 _rebuild_cnn_student
        print(f"[export] ckpt actor 종류={which}(CNN — encoder.* 서브키 감지)")
        actor = _rebuild_cnn_student(actor_sd, expected_obs, expected_act)
    else:
        # 3c) MLP 경로 (teacher actor.* / 구 MLP student) — 기존 로직 무변경
        got_obs, hidden_dims, got_act = infer_mlp_dims_from_state_dict(actor_sd)
        print(f"[export] ckpt actor 종류={which}, 구조: {got_obs} → {hidden_dims} → {got_act} "
              f"(activation={ACTIVATION} 가정)")

        if got_obs != expected_obs or got_act != expected_act:
            print(f"[export][error] ckpt와 계약 불일치: ckpt obs/act = {got_obs}/{got_act}, "
                  f"계약 = {expected_obs}/{expected_act}")
            print("                configs/common.yaml이 학습 시점과 달라졌거나 다른 실험의 ckpt입니다. 중단.")
            return 2

        actor = build_mlp_actor(got_obs, got_act, hidden_dims, ACTIVATION)
        actor.load_state_dict(actor_sd, strict=True)  # 키 호환은 build_mlp_actor가 보장
    actor.eval()

    try:
        if args.candidate_only:
            requested_out_dir = require_canonical_candidate_output(
                ROOT, source_training_run["checkpoint_sha256"]
            )
        else:
            requested = Path(args.out_dir or (ROOT / "exported")).expanduser()
            requested_out_dir = require_canonical_release_output(requested, ROOT)
        out_dir = create_release_staging(requested_out_dir)
    except ContractError as exc:
        print(f"[export][error] safe artifact staging failed: {exc}")
        return 3
    _ACTIVE_RELEASE_STAGING = (out_dir, requested_out_dir)

    # 4) TorchScript 저장 (업스트림과 동일하게 script 우선 — exporter.py 줄 96-101)
    policy_path = out_dir / "policy.pt"
    if args.candidate_only:
        export_mod = _ExportActor(actor)
        try:
            scripted = torch.jit.script(export_mod)
        except Exception as err:
            print(f"[export][warn] ⚠️ torch.jit.script 실패 → torch.jit.trace 폴백: {err}")
            with torch.no_grad():
                scripted = torch.jit.trace(export_mod, torch.zeros(1, expected_obs))
        scripted.save(str(policy_path))
        print(f"[export] candidate TorchScript 저장: {policy_path}")
    else:
        candidate_path = Path(args.candidate_dir).expanduser().absolute()
        try:
            candidate_policy_bytes = read_anchored_regular_file(
                candidate_path / "policy.pt",
                expected_sha256=selected_candidate["files"]["policy.pt"],
                label="candidate policy.pt",
            )
            # Load the exact bytes whose hash was checked.  Loading the path
            # again would allow a swap-back race to benchmark a different
            # model from the one ultimately published.
            scripted = torch.jit.load(
                io.BytesIO(candidate_policy_bytes), map_location="cpu"
            )
            with policy_path.open("xb") as stream:
                stream.write(candidate_policy_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[export][error] anchored candidate policy load failed: {exc}")
            return 3
        scripted.eval()

    # 5) obs_spec.json — sim 학습 관측 ↔ real 배포 관측의 계약서
    # ★2026-08-06(감사 P7-1)★ sim이 ★실제로 학습한★ 액션 의미를 명시 전달한다.
    #   구버전은 이 인자를 넘기지 않아 당시 배포 계약(joint_absolute)으로
    #   폴백했고 ik_rel 정책 오역 게이트가 열렸다. 현행 sim/real은 모두 ik_rel이다.
    spec_path = out_dir / OBS_SPEC_NAME
    if args.candidate_only:
        spec = make_obs_spec(pre_cfg, action_cfg, action_scheme=SIM_ACTION_SCHEME)
        spec["empirical_normalization"] = False
        spec["source_checkpoint"] = os.path.basename(ckpt_path)
        spec["model_role"] = "student"
        spec["policy_sha256"] = sha256_file(policy_path)
        spec["exported_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        save_obs_spec(spec_path, spec)
    else:
        try:
            candidate_spec_bytes = read_anchored_regular_file(
                candidate_path / OBS_SPEC_NAME,
                expected_sha256=selected_candidate["files"]["obs_spec.json"],
                label="candidate obs_spec.json",
            )
            spec = json.loads(candidate_spec_bytes.decode("utf-8"))
            with spec_path.open("xb") as stream:
                stream.write(candidate_spec_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            print(f"[export][error] anchored candidate obs_spec load failed: {exc}")
            return 3
    print(f"[export] obs_spec 저장: {spec_path} (cfg_hash={spec['cfg_hash']})")

    runner_hashes = {}
    if selected_candidate is not None:
        runner_hashes = {
            "expected_policy_sha256": selected_candidate["files"]["policy.pt"],
            "expected_obs_spec_sha256": selected_candidate["files"]["obs_spec.json"],
        }
    runner = PolicyRunner(
        policy_path, spec_path, pre_cfg, action_cfg, device="cpu", **runner_hashes
    )
    dummy = torch.zeros(1, expected_obs)
    action = runner.act(dummy)
    if action.shape != (1, expected_act):
        print(f"[export][error] self-check 실패: action 형상 {action.shape} != (1,{expected_act})")
        return 4

    if args.candidate_only:
        try:
            manifest = make_candidate_manifest(
                policy_path=policy_path,
                obs_spec_path=spec_path,
                checkpoint_path=Path(ckpt_path),
                git_commit=source_git_commit,
                training_stack=source_training_stack,
                training_run=source_training_run,
                student_adoption=student_adoption,
                config_paths=config_paths,
                config_root=ROOT,
            )
            candidate_manifest_sha256 = save_release_manifest(
                out_dir / CANDIDATE_MANIFEST_NAME, manifest
            )
            if require_clean_git_commit(ROOT) != source_git_commit \
                    or training_stack_provenance(ROOT) != source_training_stack:
                raise ContractError("source Git/training context changed during candidate export")
            out_dir = publish_candidate_staging(
                out_dir,
                requested_out_dir,
                project_root=ROOT,
                expected_manifest_sha256=candidate_manifest_sha256,
            )
            _ACTIVE_RELEASE_STAGING = None
        except ContractError as exc:
            print(f"[export][error] candidate contract/publish failed: {exc}")
            return 3
        print(f"[export] non-approved candidate published: {out_dir}")
        print("[export] release_manifest.json was intentionally not created")
        return 0

    # Representative-data equivalence and CPU latency are release contracts,
    # not optional diagnostics.  Unmanifested files can never enter a bundle.
    try:
        parity, latency = evaluate_sealed_parity_dataset(
            args.parity_obs,
            actor,
            scripted,
            expected_dim=expected_obs,
            layout=spec["layout"],
            obs_cfg_hash=spec["cfg_hash"],
            training_run=source_training_run,
            candidate_dir=args.candidate_dir,
            capture_manifest_path=args.capture_manifest,
            project_root=ROOT,
        )
        if (
            selected_candidate_binding is None
            or parity.get("candidate_manifest_sha256")
            != selected_candidate_binding["manifest_sha256"]
        ):
            raise ContractError(
                "parity evidence differs from the initially verified candidate manifest bytes"
            )
        manifest = make_release_manifest(
            policy_path=policy_path,
            obs_spec_path=spec_path,
            checkpoint_path=Path(ckpt_path),
            actor_kind="student",
            config_paths=config_paths,
            config_root=ROOT,
            parity=parity,
            latency=latency,
            max_inference_p99_ms=args.max_inference_p99_ms,
            git_commit=source_git_commit,
            control_hz=float(common_yaml["control"]["control_hz"]),
            horizon_steps=100,
            training_stack=source_training_stack,
            training_run=source_training_run,
        )
        release_manifest_sha256 = save_release_manifest(
            out_dir / RELEASE_MANIFEST_NAME, manifest
        )
    except ContractError as exc:
        print(f"[export][error] release contract failed: {exc}")
        return 3
    print(
        f"[export] release manifest: {out_dir / RELEASE_MANIFEST_NAME} "
        f"(parity max={parity['max_abs_error']:.3g}, latency p99={latency['p99_ms']:.3f}ms)"
    )

    # 6) self-check — 위에서 candidate/final 공통 경로로 완료.
    print(f"[export] ★ self-check 통과: dummy obs → action {action.shape}, "
          f"범위 [{action.min():.3f}, {action.max():.3f}] "
          f"(무클립이 계약 — 안전 clamp는 SafetyGuard, common/action_mapping.py 참조)")

    try:
        verify_release(out_dir, config_root=ROOT, expected_obs_dim=expected_obs)
        if require_clean_git_commit(ROOT) != source_git_commit \
                or training_stack_provenance(ROOT) != source_training_stack:
            raise ContractError("source Git/training context changed during final export")
        current_candidate = verify_candidate(
            args.candidate_dir, project_root=ROOT, require_current_context=True
        )
        current_candidate_binding = candidate_receipt_binding(
            args.candidate_dir,
            current_candidate,
            project_root=ROOT,
        )
        if (
            current_candidate != selected_candidate
            or current_candidate_binding != selected_candidate_binding
        ):
            raise ContractError("candidate manifest changed during final export")
        out_dir = publish_release_staging(
            out_dir,
            requested_out_dir,
            config_root=ROOT,
            expected_manifest_sha256=release_manifest_sha256,
            expected_obs_dim=expected_obs,
        )
        _ACTIVE_RELEASE_STAGING = None
    except ContractError as exc:
        print(f"[export][error] staged release verification/publish failed: {exc}")
        return 3

    print(
        "[export] 완료. student release를 canonical builder로 패키징하세요: "
        f"python deployment/build_bundle.py --release-dir {out_dir} "
            "--output dist/Real_Verify"
    )
    return 0


def main() -> int:
    global _ACTIVE_RELEASE_STAGING
    try:
        return _main_impl()
    finally:
        active = _ACTIVE_RELEASE_STAGING
        _ACTIVE_RELEASE_STAGING = None
        if active is not None:
            cleanup_release_staging(*active)


if __name__ == "__main__":
    raise SystemExit(main())
