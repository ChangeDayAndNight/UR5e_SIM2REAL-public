# 정책 export/load + obs_spec.json 계약.
# obs_spec.json = "sim 학습 관측 ↔ real 배포 관측"의 계약서. 배포 시 설정이
# 조금이라도 다르면 PolicyRunner가 기동을 거부한다 (조용한 분포 어긋남 방지).
from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from common.gripper_command import gripper_command_spec
from common.obs_preprocess import obs_dim, obs_layout
from common.types import ActionCfg, PreprocessCfg

OBS_SPEC_NAME = "obs_spec.json"


def _load_target_range() -> dict | None:
    """configs/common.yaml target.sample_range — ★관측 분포 계약★(순수: yaml만).

    ★2026-07-28 추가 근거★: 감사에서 확인된 cfg_hash 사각지대 — target_pos는 정책 관측
    슬롯인데 sample_range가 해시에 전혀 반영되지 않아, 타겟 분포만 바꾸면 배포측 검증이
    ★조용히 통과★했다(이번엔 홈 포즈가 ActionCfg에 들어 있어 우연히 걸렸을 뿐).
    """
    try:
        from common.config_io import configs_dir, load_yaml

        rng = load_yaml(configs_dir() / "common.yaml")["target"]["sample_range"]
        return {k: [float(v[0]), float(v[1])] for k, v in rng.items()}
    except Exception:  # noqa: BLE001  (배포 번들에 파일이 없으면 계약 항목만 생략)
        return None


def _cfg_hash(pre_cfg: PreprocessCfg, action_cfg: ActionCfg,
              target_range: dict | None = None) -> str:
    payload = json.dumps(
        {"preprocess": asdict(pre_cfg), "action": asdict(action_cfg),
         "target_range": target_range},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_obs_spec(pre_cfg: PreprocessCfg, action_cfg: ActionCfg,
                  action_scheme: str,
                  target_range: dict | None = None) -> dict:
    """Export_Policy.py가 정책과 함께 저장하는 계약서 생성.

    ★action_scheme★(2026-07-28): 정책 출력의 '의미'. sim 학습 액션텀과 배포 ActionMapper가
    같은 차원(7)이면서 의미가 다르면(예: sim=TCP pose delta vs 배포=관절 절대목표) 기존 검증은
    전부 통과하면서 실기에서 정책 출력이 조용히 오역된다.

    ★2026-08-06 필수 인자로 승격 (감사 P7-1)★
    구버전은 `action_scheme: str | None = None`이고 미지정 시 ★`action_cfg.mode`로 폴백★했다.
    그런데 당시 `action_cfg.mode`는 ★구 배포측 계약값★(joint_absolute)이었다.
    구 `Phase2/Export_Policy.py`가 이 인자를 넘기지 않아, 그 당시 결과:
        기록된 scheme = joint_absolute  ==  배포측 기댓값 joint_absolute  → ★검증 통과★
    즉 sim이 `ik_rel`(TCP pose delta)로 학습했는데도 게이트가 열려 ★조용한 오역★이 가능했다.
    ★막으려던 사고를 기본값이 되레 허용하고, "검증했다"는 거짓 안심까지 줬다.★
    ⇒ 기본값을 제거한다. 이제 인자를 빠뜨리면 ★export 시점에 TypeError★로 즉시 실패한다
      (배포 시점의 조용한 통과보다 훨씬 안전하다 — 실패를 앞당기는 것이 목적).

    Args:
        action_scheme: ★sim 학습이 실제로 쓴 액션 의미★. 배포 계약값(`action_cfg.mode`)이
            아니라 ★학습 액션텀★에서 온 값이어야 한다(예: "ik_rel").
    """
    tr = target_range if target_range is not None else _load_target_range()
    if not isinstance(action_scheme, str) or not action_scheme:
        raise ValueError(
            "action_scheme은 비어 있지 않은 문자열이어야 한다 — ★sim 학습이 실제로 쓴★ "
            "액션 의미를 넣을 것(배포 계약값 action_cfg.mode를 그대로 넣으면 검증이 무의미해진다)."
        )
    scheme = action_scheme
    return {
        "obs_dim": obs_dim(pre_cfg),
        "action_dim": action_cfg.action_dim,
        "layout": [{"name": n, "size": s} for n, s in obs_layout(pre_cfg)],
        # Explicit temporal/semantic contract.  Dimension and range alone
        # cannot distinguish the retired physical finger-q observation from
        # the new delayed command state, so absence is always rejected.
        "gripper_command_state": gripper_command_spec(
            transition_s=pre_cfg.gripper_command_transition_s,
            transition_steps=pre_cfg.gripper_command_transition_steps,
        ),
        # ★정책 출력의 의미★ — 차원만 같고 의미가 다른 오역을 차단(감사 C-1)
        "action_scheme": scheme,
        # ★관측 분포 계약★ — target_pos 슬롯의 샘플 범위(감사 cfg_hash 사각지대)
        "target_range": tr,
        "cfg_hash": _cfg_hash(pre_cfg, action_cfg, tr),
        # ⚠️ TODO(확인 필요 — 요소1 최대 함정): rsl_rl empirical_normalization
        #   사용 시 정규화 통계가 export된 actor에 "포함되는지" 확인하고 기록.
        #   포함 안 되면 배포 시 여기서 통계를 로드·적용해야 한다.
        "empirical_normalization": None,
    }


def save_obs_spec(path: Path | str, spec: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)


def load_obs_spec(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_regular_file_once(path: Path | str, label: str) -> tuple[bytes, str]:
    """Read and hash the exact bytes from one already-opened regular file."""
    source = Path(path)
    with source.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular file")
        payload = stream.read()
    return payload, hashlib.sha256(payload).hexdigest()


def _require_expected_sha256(actual: str, expected: str | None, label: str) -> None:
    if expected is None:
        return
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected)
        or actual != expected
    ):
        raise ValueError(f"{label} does not match its manifest SHA-256")


def read_anchored_regular_file(
    path: Path | str, *, expected_sha256: str, label: str
) -> bytes:
    """Return only bytes that match a manifest hash from one open handle.

    Callers that subsequently parse or load an artifact must use the returned
    bytes instead of reopening ``path``.  This keeps the manifest check and the
    consumed payload in the same file-handle transaction.
    """
    payload, actual = _read_regular_file_once(path, label)
    _require_expected_sha256(actual, expected_sha256, label)
    return payload


def validate_obs_spec(
    spec: dict, pre_cfg: PreprocessCfg, action_cfg: ActionCfg
) -> None:
    """배포측 cfg가 학습 시 계약과 다르면 ValueError — '기동 거부'가 정답.

    ★2026-07-28 검증 항목 확대★: action_scheme(정책 출력 의미)·target_range(관측 분포)를 추가.
    구 계약으로 만든 obs_spec(두 키 부재)은 cfg_hash가 어차피 달라 먼저 걸리므로,
    여기서는 '있으면 대조'한다(구 파일에 대한 오해 소지 없는 메시지를 위해 분리 처리).
    ⚠️ §8-3: 이 예외를 우회하는 패치 금지 — 원인(configs 변경/재학습 필요)을 맞추는 것이 정답.
    """
    # 배포측 기댓값 = configs/common.yaml의 계약. action_scheme은 여기서 '배포측 해석'을 뜻한다.
    expected = make_obs_spec(pre_cfg, action_cfg, action_scheme=action_cfg.mode)
    # ★2026-08-06(P7-1)★ action_scheme 부재 = 의미 불명 = ★배포 거부★.
    #   구버전은 `if "action_scheme" in spec`으로 ★있으면만 대조★했다. 그러면 이 키가 없는
    #   구 계약 파일이 조용히 통과한다 — 정책 출력의 의미를 모르는 채 실기에 나가는 것이
    #   바로 이 게이트가 막으려던 사고다. 없으면 거부하는 것이 정답이다(§8-3: 우회 금지).
    if "action_scheme" not in spec:
        raise ValueError(
            "★obs_spec에 action_scheme이 없다★ — 정책 출력의 '의미'가 기록되지 않은 계약서다. "
            "차원(action_dim)이 같아도 해석이 다르면 실기에서 조용히 오역된다. "
            "Export_Policy를 최신 버전으로 다시 실행해 재생성할 것. 배포 중단."
        )
    if "gripper_command_state" not in spec:
        raise ValueError(
            "obs_spec에 gripper_command_state 의미 계약이 없다 — 구 physical gripper_pos "
            "정책/spec은 현행 command-state 입력과 호환되지 않는다. scratch 재학습 필요."
        )
    for key in ("obs_dim", "action_dim", "layout", "gripper_command_state", "cfg_hash"):
        if spec.get(key) != expected[key]:
            raise ValueError(
                f"obs_spec 불일치 [{key}]: 학습 시 {spec.get(key)!r} vs "
                f"현재 {expected[key]!r} — configs/common.yaml이 학습 시점과 다르다. "
                f"배포 중단."
            )
    # ★의미 계약★ — 차원이 같아도 해석이 다르면 실기에서 조용히 오역된다(감사 C-1)
    if "action_scheme" in spec and spec["action_scheme"] != expected["action_scheme"]:
        raise ValueError(
            f"★액션 스킴 불일치★: 학습 시 {spec['action_scheme']!r} vs 배포 "
            f"{expected['action_scheme']!r} — 정책 출력의 '의미'가 다르다. "
            f"(예: TCP pose delta를 관절 절대목표로 해석 = 위험한 오역) 배포 중단."
        )
    if "target_range" in spec and spec["target_range"] != expected["target_range"]:
        raise ValueError(
            f"★타겟 분포 불일치★: 학습 시 {spec['target_range']!r} vs 배포 "
            f"{expected['target_range']!r} — target_pos는 정책 관측 슬롯이라 분포가 다르면 "
            f"정책이 학습하지 않은 목표를 받는다. 배포 중단."
        )


class PolicyRunner:
    """export된 actor 로드 + 추론 (Isaac/rsl_rl 불필요 — 배포 PC에서 동작).

    입력 계약: build_policy_obs가 만든 flat [N, obs_dim] 텐서.
    """

    def __init__(
        self,
        policy_path: Path | str,
        obs_spec_path: Path | str,
        pre_cfg: PreprocessCfg,
        action_cfg: ActionCfg,
        device: str = "cpu",
        *,
        expected_policy_sha256: str | None = None,
        expected_obs_spec_sha256: str | None = None,
    ):
        if expected_obs_spec_sha256 is None:
            spec_bytes, _ = _read_regular_file_once(obs_spec_path, "obs_spec.json")
        else:
            spec_bytes = read_anchored_regular_file(
                obs_spec_path,
                expected_sha256=expected_obs_spec_sha256,
                label="obs_spec.json",
            )
        try:
            spec = json.loads(spec_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid obs_spec.json: {exc}") from exc
        if not isinstance(spec, dict):
            raise ValueError("obs_spec.json must contain a mapping")
        validate_obs_spec(spec, pre_cfg, action_cfg)
        self.spec = spec
        self.device = device
        policy_path = Path(policy_path)
        if expected_policy_sha256 is None:
            policy_bytes, actual_policy_hash = _read_regular_file_once(
                policy_path, "policy"
            )
        else:
            policy_bytes = read_anchored_regular_file(
                policy_path,
                expected_sha256=expected_policy_sha256,
                label="policy",
            )
            actual_policy_hash = expected_policy_sha256
        expected_policy_hash = spec.get("policy_sha256")
        if expected_policy_hash is not None:
            if actual_policy_hash != expected_policy_hash:
                raise ValueError("policy.pt hash does not match obs_spec.json")
        if policy_path.suffix == ".pt":
            # TorchScript (기본 백엔드)
            self._module = torch.jit.load(io.BytesIO(policy_bytes), map_location=device)
            self._module.eval()
            self._backend = "torchscript"
        elif policy_path.suffix == ".onnx":
            # TODO(확장 지점): onnxruntime 백엔드 — 배포 PC에 torch를 깔지 않는
            # 선택지. providers/입출력 이름은 Export_Policy.py 산출과 맞출 것.
            raise NotImplementedError("onnx 백엔드는 TODO — 우선 policy.pt 사용")
        else:
            raise ValueError(f"지원하지 않는 정책 형식: {policy_path.suffix}")
        print(f"[policy] 로드 완료 ({self._backend}): {policy_path.name}, "
              f"obs_dim={spec['obs_dim']}, action_dim={spec['action_dim']}")

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> np.ndarray:
        """[N, obs_dim] → action [N, action_dim] (numpy). [-1,1] 미보장 — ★클립하지 않는다★.

        현행 ik_rel 스케일·안전 clamp는 ActionMapper/SafetyGuard 책임이다. 구
        joint_absolute 호환 매핑도 무클립 선형 계약을 별도 테스트한다."""
        if obs.dim() != 2 or obs.shape[1] != self.spec["obs_dim"]:
            raise ValueError(
                f"obs 형상 오류: 기대 (N,{self.spec['obs_dim']}), 실제 {tuple(obs.shape)}"
            )
        action = self._module(obs.to(self.device))
        if not isinstance(action, torch.Tensor):
            raise RuntimeError(f"policy returned {type(action).__name__}, expected Tensor")
        expected = (obs.shape[0], int(self.spec["action_dim"]))
        if tuple(action.shape) != expected:
            raise RuntimeError(
                f"policy output shape mismatch: expected {expected}, got {tuple(action.shape)}"
            )
        if not torch.is_floating_point(action) or not torch.isfinite(action).all():
            raise RuntimeError("policy output must be finite floating point")
        return action.detach().cpu().numpy()
