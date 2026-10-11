#!/usr/bin/env python
"""★비릴리스★ 실험용 정책 export — 실기 랩 작업(HIL) 전용. [Claude][학습 PC]

═══════════════════════════════════════════════════════════════════════════════
 이 산출물은 ★릴리스가 아니다★. 배포 승인 증거가 없다.
═══════════════════════════════════════════════════════════════════════════════

왜 별도 레인인가
  `deployment/release.py:verify_release()`가 요구하는 릴리스 계약은 다음을 ★전부★ 요구한다:
    · student 채택(시드별 ≥487/512, pooled Wilson ≥95%)
    · parity 데이터셋(sim↔export 출력 일치, 10,000 샘플)
    · 추론 지연 p99 벤치마크
    · release 커밋 == 학습 커밋
    · 설정 해시 집합 일치
  이는 "모든 증거가 갖춰진 동결 산출물"의 정의이고, 그래서 ★탐색적 실기 작업과는 양립하지
  않는다★ — 모션 인가를 추가한 순간 커밋 조건이 깨지고, 캘리브레이션을 갱신하면 설정 해시가
  깨진다. 릴리스 기계를 느슨하게 만드는 대신, ★자기가 릴리스가 아님을 선언하는★ 병렬 레인을
  둔다. `Export_Policy.py`와 `verify_release()`는 이 파일로 인해 ★한 줄도 바뀌지 않는다★ —
  실험 산출물이 릴리스로 둔갑할 경로가 존재하지 않는다.

무엇은 여전히 검증하는가 (느슨함이 아니라 ★범위 축소★다)
  · 학습 receipt(schema-2)가 실재하고 checkpoint 해시가 일치할 것 — 실재하는 run에 묶인다
  · 관측/행동 폭이 현행 계약(configs/common.yaml)과 일치할 것 — 폭이 다르면 실기에서 즉사한다
  · TorchScript 산출물이 원본 네트워크와 같은 출력을 낼 것 — 수치 동치를 직접 대조한다
  · 재구성은 릴리스 exporter의 ★같은 함수★를 재사용한다(_rebuild_cnn_student/_ExportActor)

무엇을 기록하는가
  manifest에 `"release": false`, `"adopted": false`와 함께 학습 커밋·현재 커밋을 ★둘 다★
  적고 불일치를 숨기지 않는다. 사람이 파일만 보고도 "이건 승인 안 된 물건"임을 알아야 한다.

실행:
  python scripts/tools/export_experimental_policy.py \
      --checkpoint logs/rsl_rl/ur5e_binpick_vision/<run>/model_19999.pt
산출: exported_experimental/<checkpoint_sha12>/
      policy.pt · obs_spec.json · experimental_manifest.json · configs/ · DEPLOY_README.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config_io import build_action_cfg, build_preprocess_cfg, configs_dir, load_yaml
from common.obs_preprocess import obs_dim
from common.policy_io import make_obs_spec, save_obs_spec
from deployment.contracts import ContractError, sha256_file
from scripts.training_provenance import verify_training_checkpoint_receipt
from ur5e_binpick.tasks.binpick.action_scheme import SIM_ACTION_SCHEME

OUT_ROOT = ROOT / "exported_experimental"
# 배포 PC가 읽어야 하는 설정 — 정책 계약(관측/행동)과 안전 한계, 그리고 캘리브레이션.
DEPLOY_CONFIGS = ("common.yaml", "safety.yaml", "sim.yaml", "real.local.yaml")


def _load_release_exporter():
    """릴리스 exporter의 재구성 함수를 재사용한다(중복 구현으로 갈라지지 않도록)."""
    spec = importlib.util.spec_from_file_location(
        "_release_exporter", ROOT / "Phase2" / "Export_Policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args(argv[1:])

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        print(f"[exp-export][error] checkpoint 없음: {checkpoint}")
        return 1

    # ── 1) 봉인된 학습 run에 묶인다 (완주+receipt가 없으면 여기서 거부) ──────────
    try:
        training = verify_training_checkpoint_receipt(
            checkpoint, project_root=ROOT, expected_experiment="ur5e_binpick_vision"
        )
    except ContractError as exc:
        print(f"[exp-export][error] 학습 provenance 실패: {exc}")
        print("                    완주하고 run_manifest.json이 봉인된 run만 사용할 수 있습니다.")
        return 3
    checkpoint_sha256 = training["checkpoint_sha256"]
    if sha256_file(checkpoint) != checkpoint_sha256:
        print("[exp-export][error] checkpoint 해시가 receipt와 다릅니다")
        return 3

    # ── 2) 관측/행동 계약 ────────────────────────────────────────────────────────
    common = load_yaml(configs_dir() / "common.yaml")
    pre_cfg = build_preprocess_cfg(common)
    action_cfg = build_action_cfg(common)
    expected_obs = int(obs_dim(pre_cfg))   # 릴리스 exporter와 동일한 계약 함수
    expected_act = int(action_cfg.action_dim)

    # ── 3) 릴리스 exporter의 재구성 경로를 그대로 사용 ──────────────────────────
    exporter = _load_release_exporter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    which, actor_sd = exporter._extract_actor_state_dict(payload["model_state_dict"])
    if not any(key.startswith("encoder.") for key in actor_sd):
        print(f"[exp-export][error] CNN student(encoder.*)가 아닙니다 — which={which}")
        return 2
    actor = exporter._rebuild_cnn_student(actor_sd, expected_obs, expected_act)
    actor.eval()
    export_mod = exporter._ExportActor(actor)

    # ── 4) TorchScript + ★수치 동치 대조★ ──────────────────────────────────────
    try:
        scripted = torch.jit.script(export_mod)
    except Exception as err:  # noqa: BLE001 - 업스트림과 같은 폴백
        print(f"[exp-export][warn] torch.jit.script 실패 → trace 폴백: {err}")
        with torch.no_grad():
            scripted = torch.jit.trace(export_mod, torch.zeros(1, expected_obs))
    probe = torch.rand(8, expected_obs)
    with torch.no_grad():
        reference = export_mod(probe)
        exported = scripted(probe)
    max_abs_error = float((reference - exported).abs().max())
    if not torch.isfinite(exported).all() or max_abs_error > 1e-5:
        print(f"[exp-export][error] TorchScript 출력이 원본과 다릅니다 (max|Δ|={max_abs_error:.3e})")
        return 3

    out_dir = args.out_root / checkpoint_sha256[:12]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "configs" / "calib").mkdir(parents=True)
    scripted.save(str(out_dir / "policy.pt"))
    save_obs_spec(
        out_dir / "obs_spec.json",
        make_obs_spec(pre_cfg, action_cfg, action_scheme=SIM_ACTION_SCHEME),
    )

    # ── 5) 배포 PC가 읽을 설정 사본 ─────────────────────────────────────────────
    copied = []
    for name in DEPLOY_CONFIGS:
        source = configs_dir() / name
        if source.is_file():
            shutil.copy2(source, out_dir / "configs" / name)
            copied.append(f"configs/{name}")
    for source in sorted((configs_dir() / "calib").glob("*.yaml")):
        shutil.copy2(source, out_dir / "configs" / "calib" / source.name)
        copied.append(f"configs/calib/{source.name}")

    # ── 6) '릴리스가 아님'을 선언하는 manifest ──────────────────────────────────
    training_commit = training["source"]["project_git_commit"]
    current_commit = _git_commit()
    manifest = {
        "schema_version": 1,
        "artifact_type": "experimental_policy",
        "release": False,
        "adopted": False,
        "not_for_release_reason": (
            "student adoption gate not passed; no parity dataset; no latency benchmark. "
            "Lab/HIL use only under an explicit motion authorization."
        ),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": checkpoint_sha256,
            "training_run_root": training["run_root"],
            "training_receipt_sha256": training["receipt_sha256"],
            "iteration": int(payload.get("iter", -1)),
        },
        "git": {
            "training_commit": training_commit,
            "export_commit": current_commit,
            "commits_match": training_commit == current_commit,
        },
        "contract": {"obs_dim": expected_obs, "action_dim": expected_act},
        "torchscript_equivalence_max_abs_error": max_abs_error,
        "files": {
            name: sha256_file(out_dir / name) for name in ("policy.pt", "obs_spec.json")
        },
        "configs_copied": copied,
    }
    (out_dir / "experimental_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_readme(out_dir, manifest)

    print(f"[exp-export] ★비릴리스 실험 산출물★ → {out_dir}")
    print(f"  checkpoint iter={manifest['checkpoint']['iteration']} sha={checkpoint_sha256[:12]}")
    print(f"  obs/act = {expected_obs}/{expected_act}   TorchScript 동치 max|Δ| = {max_abs_error:.2e}")
    print(f"  학습 커밋 {training_commit[:12]} / export 커밋 {current_commit[:12]}"
          f"  ({'일치' if manifest['git']['commits_match'] else '★불일치 — 실험 전용★'})")
    print(f"  설정 사본 {len(copied)}개")
    return 0


def _write_readme(out_dir: Path, manifest: dict) -> None:
    calibrated = "미확인"
    calib = out_dir / "configs" / "calib" / "hand_eye.yaml"
    if calib.is_file():
        calibrated = "true" if "calibrated: true" in calib.read_text(encoding="utf-8") else "false"
    (out_dir / "DEPLOY_README.md").write_text(
        f"""# 실험용 정책 배포 폴더 — ★릴리스 아님★

이 폴더는 **배포 승인을 받지 않은** 실험 산출물입니다. 랩 환경에서 사람이 e-stop 앞에
있는 상태에서만 사용하십시오.

- checkpoint: `{manifest['checkpoint']['path']}` (iter {manifest['checkpoint']['iteration']})
- sha256: `{manifest['checkpoint']['sha256'][:16]}…`
- 관측/행동: {manifest['contract']['obs_dim']} / {manifest['contract']['action_dim']}
- 학습 커밋 `{manifest['git']['training_commit'][:12]}` / export 커밋 `{manifest['git']['export_commit'][:12]}`
- **채택 게이트 미통과** — 성공률은 낮습니다. 목표는 "학습 정책이 실기를 구동한다"입니다.
- `configs/calib/hand_eye.yaml` 의 `calibrated`: **{calibrated}**

## 배포 PC에서 할 일

1. 이 폴더를 배포 PC로 복사합니다.
2. **캘리브레이션 값을 반영**합니다 (`configs/calib/tcp_offset.yaml`, `hand_eye.yaml`의
   `meta.calibrated: true`). 이것이 없으면 `policy` 범위 모션은 코드가 거부합니다.
3. **모션 인가 파일**을 만듭니다 — `configs/motion_authorization.yaml`
   (`configs/motion_authorization.example.yaml` 복사 후 채우기).
   - `operator`: 입회자 이름
   - `expires_at`: acknowledged_at + 72시간 이내
   - `scopes`: 필요한 최소 범위만 (`policy_dryrun` → 확인 후 `policy`)
4. 단계적으로 올립니다:
   - ① 수신 전용 연결 확인
   - ② `policy_dryrun` — 정책이 추론하되 명령은 보내지 않음
   - ③ `policy` + 낮은 speed_scale — 실제 모션

## 안전

`configs/safety.yaml` 의 workspace AABB · TCP z 바닥 · 관절속도 상한이 그대로 적용됩니다.
이탈 시 즉시 trip합니다. **작업 내내 e-stop 도달 거리 안에 사람이 있어야 합니다.**
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
