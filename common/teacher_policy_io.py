# ★모듈형 배포 v1 — Teacher(상태 정책) 로더 + 관측 계약서★ [배포 PC][학습 PC]
#
# 왜 별도 파일인가
#   common.policy_io.PolicyRunner는 student(RGB-D 50,206D) 계약서(obs_spec)를 configs/common.yaml
#   의 전처리 cfg와 대조해 기동을 거부한다. Teacher는 카메라 전처리와 무관한 상태 관측(v2 34D)을
#   받으므로 그 검증을 그대로 통과할 수 없고, 통과시키려고 PolicyRunner를 느슨하게 만들면
#   student 배포 게이트가 약해진다. 그래서 ★같은 파일 앵커링(해시 대조) 규칙★을 쓰되 계약서
#   종류(policy_kind="teacher_state")를 명시하는 병렬 로더를 둔다.
#
# 무엇을 검증하는가
#   · obs_spec.json이 teacher_state 종류이고 34D 레이아웃(common.teacher_contract, Teacher v2)과 일치할 것
#   · action_scheme(정책 출력의 의미)이 배포측 ActionCfg.mode와 같을 것 — 차원만 같은 오역 차단
#   · target_range(관측 슬롯 분포)가 현재 configs/common.yaml과 같을 것
#   · cfg_hash — 위 항목 + ActionCfg 전체를 해시한 값이 재계산과 같을 것
#   · policy.pt / obs_spec.json 바이트가 manifest 해시와 같을 것(전송 중 손상 차단)
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from common.policy_io import _load_target_range, _read_regular_file_once, read_anchored_regular_file
from common.teacher_contract import TEACHER_ACTION_DIM, TEACHER_OBS_DIM, TEACHER_OBS_LAYOUT
from common.types import ActionCfg

TEACHER_POLICY_KIND = "teacher_state"
# observations.finger_contact_forces(force_scale_n=25.0) — 학습 시 정규화 상수. 배포측 근사값도
# 같은 스케일([0,1])로 넣어야 하므로 계약서에 기록한다.
TEACHER_CONTACT_FORCE_SCALE_N = 25.0


def teacher_cfg_hash(action_cfg: ActionCfg, target_range: dict | None,
                     *, vertical_grasp_only: bool, contact_force_scale_n: float) -> str:
    payload = json.dumps(
        {
            "policy_kind": TEACHER_POLICY_KIND,
            "layout": [[n, s] for n, s in TEACHER_OBS_LAYOUT],
            "action": asdict(action_cfg),
            "target_range": target_range,
            "vertical_grasp_only": bool(vertical_grasp_only),
            "contact_force_scale_n": float(contact_force_scale_n),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_teacher_obs_spec(action_cfg: ActionCfg, *, action_scheme: str,
                          vertical_grasp_only: bool,
                          contact_force_scale_n: float = TEACHER_CONTACT_FORCE_SCALE_N,
                          target_range: dict | None = None) -> dict:
    """Teacher 배포 계약서. action_scheme은 ★sim 학습이 실제로 쓴 값★(action_scheme.SIM_ACTION_SCHEME)."""
    if not isinstance(action_scheme, str) or not action_scheme:
        raise ValueError("action_scheme은 비어 있지 않은 문자열이어야 한다")
    tr = target_range if target_range is not None else _load_target_range()
    return {
        "policy_kind": TEACHER_POLICY_KIND,
        "obs_dim": TEACHER_OBS_DIM,
        "action_dim": TEACHER_ACTION_DIM,
        "layout": [{"name": n, "size": s} for n, s in TEACHER_OBS_LAYOUT],
        "action_scheme": action_scheme,
        # observations.executable_last_action — last_action 슬롯은 clamp ±1 + 기울임 채널 3·4 = 0.
        "vertical_grasp_only": bool(vertical_grasp_only),
        "contact_force_scale_n": float(contact_force_scale_n),
        "target_range": tr,
        "cfg_hash": teacher_cfg_hash(action_cfg, tr, vertical_grasp_only=vertical_grasp_only,
                                     contact_force_scale_n=contact_force_scale_n),
        "empirical_normalization": None,
    }


def validate_teacher_obs_spec(spec: dict, action_cfg: ActionCfg) -> None:
    """배포측 계약과 다르면 ValueError — 기동 거부가 정답(§8-3: 우회 패치 금지)."""
    if spec.get("policy_kind") != TEACHER_POLICY_KIND:
        raise ValueError(
            f"obs_spec policy_kind={spec.get('policy_kind')!r} — teacher_state 계약서가 아니다. "
            f"student 번들은 Phase4/Run_Task.py로 실행할 것."
        )
    for key in ("action_scheme", "vertical_grasp_only", "contact_force_scale_n", "cfg_hash"):
        if key not in spec:
            raise ValueError(f"obs_spec에 {key}가 없다 — 구 계약서. export_teacher_policy.py로 재생성할 것")
    expected = make_teacher_obs_spec(
        action_cfg, action_scheme=action_cfg.mode,
        vertical_grasp_only=bool(spec["vertical_grasp_only"]),
        contact_force_scale_n=float(spec["contact_force_scale_n"]),
    )
    for key in ("obs_dim", "action_dim", "layout", "target_range", "cfg_hash"):
        if spec.get(key) != expected[key]:
            raise ValueError(
                f"teacher obs_spec 불일치 [{key}]: 학습 시 {spec.get(key)!r} vs 현재 "
                f"{expected[key]!r} — configs/common.yaml이 학습 시점과 다르다. 배포 중단."
            )
    if spec["action_scheme"] != expected["action_scheme"]:
        raise ValueError(
            f"★액션 스킴 불일치★: 학습 시 {spec['action_scheme']!r} vs 배포 "
            f"{expected['action_scheme']!r} — 정책 출력의 의미가 다르다. 배포 중단."
        )
    if spec.get("empirical_normalization") not in (None, False):
        raise ValueError("empirical_normalization이 켜진 teacher는 이 로더가 지원하지 않는다")


class TeacherPolicyRunner:
    """export된 Teacher actor(TorchScript) 로드 + 추론 — Isaac/rsl_rl 불필요.

    입력 계약: common.teacher_obs.TeacherObsBuilder가 만든 [N, 42] float32 텐서.
    출력: [N, 7] numpy — ★클립하지 않는다★(ik_rel 스케일·안전 clamp는 ActionMapper/SafetyGuard 책임).
    """

    def __init__(self, policy_path: Path | str, obs_spec_path: Path | str, action_cfg: ActionCfg,
                 device: str = "cpu", *, expected_policy_sha256: str | None = None,
                 expected_obs_spec_sha256: str | None = None):
        if expected_obs_spec_sha256 is None:
            spec_bytes, _ = _read_regular_file_once(obs_spec_path, "obs_spec.json")
        else:
            spec_bytes = read_anchored_regular_file(
                obs_spec_path, expected_sha256=expected_obs_spec_sha256, label="obs_spec.json")
        try:
            spec = json.loads(spec_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid obs_spec.json: {exc}") from exc
        if not isinstance(spec, dict):
            raise ValueError("obs_spec.json must contain a mapping")
        validate_teacher_obs_spec(spec, action_cfg)
        self.spec = spec
        self.device = device
        self.vertical_grasp_only = bool(spec["vertical_grasp_only"])
        self.contact_force_scale_n = float(spec["contact_force_scale_n"])
        policy_path = Path(policy_path)
        if expected_policy_sha256 is None:
            policy_bytes, actual = _read_regular_file_once(policy_path, "policy")
        else:
            policy_bytes = read_anchored_regular_file(
                policy_path, expected_sha256=expected_policy_sha256, label="policy")
            actual = expected_policy_sha256
        recorded = spec.get("policy_sha256")
        if recorded is not None and actual != recorded:
            raise ValueError("policy.pt hash does not match obs_spec.json")
        if policy_path.suffix != ".pt":
            raise ValueError(f"지원하지 않는 정책 형식: {policy_path.suffix}")
        self._module = torch.jit.load(io.BytesIO(policy_bytes), map_location=device)
        self._module.eval()
        print(f"[teacher-policy] 로드 완료 (torchscript): {policy_path.name}, "
              f"obs_dim={spec['obs_dim']}, action_dim={spec['action_dim']}")

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> np.ndarray:
        if obs.dim() != 2 or obs.shape[1] != self.spec["obs_dim"]:
            raise ValueError(f"obs 형상 오류: 기대 (N,{self.spec['obs_dim']}), 실제 {tuple(obs.shape)}")
        action = self._module(obs.to(self.device))
        if not isinstance(action, torch.Tensor):
            raise RuntimeError(f"policy returned {type(action).__name__}, expected Tensor")
        expected = (obs.shape[0], int(self.spec["action_dim"]))
        if tuple(action.shape) != expected:
            raise RuntimeError(f"policy output shape mismatch: expected {expected}, got {tuple(action.shape)}")
        if not torch.is_floating_point(action) or not torch.isfinite(action).all():
            raise RuntimeError("policy output must be finite floating point")
        return action.detach().cpu().numpy()
