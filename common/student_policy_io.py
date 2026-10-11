# ★배포 v2 — 이력(history) student 로더★ [배포 PC]
#
# 왜 별도 러너인가
#   채택 checkpoint(model_35000)의 student는 16스텝 관측 이력을 쓴다. 학습측 구현
#   (scripts/student_loop_history.HistoryStudentNet)은 버퍼를 ★파이썬 속성★으로 들고
#   forward 안에서 갱신한다. 그대로 TorchScript로 내보내면 배포측이 에피소드 경계에서
#   버퍼를 비울 방법이 없고, 두 에피소드가 조용히 이어진다.
#   그래서 export는 ★무상태★ 모듈로 만든다: forward(obs, history) → (action, next_history).
#   버퍼는 이 러너가 들고, reset()이 에피소드 경계를 명시한다.
#
# 관측 계약은 student와 동일하다(common.policy_io.make_obs_spec / 50,206D). 이력은
# 관측이 아니라 ★정책 내부 상태★이므로 obs_spec을 바꾸지 않는다.
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch

from common.policy_io import (
    _read_regular_file_once, read_anchored_regular_file, validate_obs_spec,
)
from common.types import ActionCfg, PreprocessCfg


class StudentHistoryPolicyRunner:
    """무상태 TorchScript 이력 student를 로드하고 16스텝 버퍼를 관리한다.

    ⚠️ 에피소드마다 ``reset()``을 부를 것 — 부르지 않으면 직전 에피소드의 이력이 샌다.
    """

    def __init__(self, policy_path: Path | str, obs_spec_path: Path | str, pre_cfg: PreprocessCfg,
                 action_cfg: ActionCfg, *, history_steps: int, feature_dim: int, device: str = "cpu",
                 expected_policy_sha256: str | None = None, expected_obs_spec_sha256: str | None = None):
        if history_steps < 2 or feature_dim <= 0:
            raise ValueError("invalid history contract")
        if expected_obs_spec_sha256 is None:
            spec_bytes, _ = _read_regular_file_once(obs_spec_path, "obs_spec.json")
        else:
            spec_bytes = read_anchored_regular_file(
                obs_spec_path, expected_sha256=expected_obs_spec_sha256, label="obs_spec.json")
        spec = json.loads(spec_bytes.decode("utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("obs_spec.json must contain a mapping")
        validate_obs_spec(spec, pre_cfg, action_cfg)   # student와 같은 계약 검증을 그대로 쓴다
        self.spec = spec
        self.device = device
        self.history_steps = int(history_steps)
        self.feature_dim = int(feature_dim)
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
        self._history: torch.Tensor | None = None
        self.steps = 0
        print(f"[student-policy] 로드 완료 (torchscript, history {self.history_steps}단계): "
              f"{policy_path.name}, obs_dim={spec['obs_dim']}, action_dim={spec['action_dim']}")

    def reset(self) -> None:
        """에피소드 경계 — 이력 버퍼를 비운다(학습의 reset_history(None)과 동일: 0으로 시작)."""
        self._history = None
        self.steps = 0

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> np.ndarray:
        if obs.dim() != 2 or obs.shape[1] != self.spec["obs_dim"]:
            raise ValueError(f"obs 형상 오류: 기대 (N,{self.spec['obs_dim']}), 실제 {tuple(obs.shape)}")
        obs = obs.to(self.device)
        if self._history is None:
            self._history = torch.zeros(obs.shape[0], self.history_steps - 1, self.feature_dim,
                                        dtype=obs.dtype, device=self.device)
        if self._history.shape[0] != obs.shape[0]:
            raise ValueError("history batch changed without reset")
        out = self._module(obs, self._history)
        # ★무상태 계약★: 상태를 내부에 숨긴 모듈(단일 텐서 반환)을 여기서 거른다. 그대로 두면
        #   에피소드 경계에서 reset()이 아무 일도 하지 않고 이력이 조용히 이어진다.
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise RuntimeError(
                "policy must return (action, next_history) — 이 모듈은 "
                f"{type(out).__name__}({len(out) if isinstance(out, (tuple, list)) else 1}개)를 돌려준다. "
                "상태를 내부에 들고 있는 학습측 모듈을 그대로 내보낸 것이 아닌지 확인할 것.")
        action, history = out
        if not isinstance(action, torch.Tensor) or not isinstance(history, torch.Tensor):
            raise RuntimeError("policy must return (action, next_history) tensors")
        expected = (obs.shape[0], int(self.spec["action_dim"]))
        if tuple(action.shape) != expected:
            raise RuntimeError(f"policy output shape mismatch: expected {expected}, got {tuple(action.shape)}")
        if tuple(history.shape) != tuple(self._history.shape):
            raise RuntimeError("returned history changed shape")
        if not torch.is_floating_point(action) or not torch.isfinite(action).all():
            raise RuntimeError("policy output must be finite floating point")
        if not torch.isfinite(history).all():
            raise RuntimeError("policy history must stay finite")
        self._history = history
        self.steps += 1
        return action.detach().cpu().numpy()
