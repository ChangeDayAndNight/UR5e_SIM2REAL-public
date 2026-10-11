#!/usr/bin/env python
"""★모듈형 배포 v1★ Teacher(상태 정책) export — 인식 모듈과 짝을 이루는 비릴리스 실험 번들. [학습 PC]
═══════════════════════════════════════════════════════════════════════════════
 무엇을 만드나
   exported_experimental/teacher_<checkpoint_sha12>/
     policy.pt                 TorchScript actor  (34D 배포 가능 관측 → 7D ik_rel 액션, 정규화 없음)
     obs_spec.json             teacher_state 계약서 (common.teacher_policy_io.make_teacher_obs_spec)
     experimental_manifest.json 해시·provenance·관측 조립 상수(TeacherObsCfg)·검증 결과
     configs/                  배포 PC가 읽을 설정 사본(+calib/)
     DEPLOY_README.md
 무엇을 검증하나
   · 학습 receipt(run_manifest.json)가 실재하고 checkpoint 해시가 일치할 것(training_provenance)
   · params/agent.yaml: actor_hidden_dims·activation·정규화 off가 재구성과 일치할 것
   · TorchScript ≡ 원본 actor — 무작위 입력 + ★기록된 teacher 관측 표본★(Codex 루프의
     diagnostic_states.pt: teacher 34D 관측과 그때 teacher가 낸 행동)으로 수치 동치 대조
 왜 릴리스가 아닌가
   student 채택·parity·지연 벤치마크 계약은 시각 정책용이다. 이 번들은 ★인식 모듈(색분할 +
   depth 역투영)★에 의존하므로 실기 성공률은 인식
   오차에 좌우된다. 사람이 e-stop 앞에 있는 랩 작업에서만, 명시적 모션 인가 아래 쓴다.
실행:
  python scripts/tools/export_teacher_policy.py \\
      --checkpoint logs/rsl_rl/ur5e_binpick_teacher/2026-09-08_00-47-43-694665/model_17300.pt \\
      --reference-samples "C:/.../UR5e_Student75/outputs/student75/jobs/*_evaluate_*/diagnostic_states.pt"
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "source/ur5e_binpick") not in sys.path:
    sys.path.insert(0, str(ROOT / "source/ur5e_binpick"))

from common.config_io import build_action_cfg, configs_dir, load_yaml  # noqa: E402
from common.teacher_contract import TEACHER_ACTION_DIM, TEACHER_OBS_DIM  # noqa: E402
from common.teacher_obs import TeacherObsCfg  # noqa: E402
from common.teacher_policy_io import (  # noqa: E402
    TEACHER_CONTACT_FORCE_SCALE_N, make_teacher_obs_spec, validate_teacher_obs_spec,
)
from deployment.contracts import ContractError, sha256_file  # noqa: E402
from scripts.training_provenance import verify_training_checkpoint_receipt  # noqa: E402
from ur5e_binpick.tasks.binpick.action_scheme import SIM_ACTION_SCHEME  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "logs/rsl_rl/ur5e_binpick_teacher/2026-09-08_00-47-43-694665/model_17300.pt"
# 루프 worktree는 본 저장소의 ★형제 디렉터리★다. 사용자별 절대경로를 박지 않는다
# (다른 PC에서 못 쓰고 scripts/validate_repository.py 위생 검사에 걸린다).
LOOP_ROOT = Path(os.environ.get("STUDENT_LOOP_ROOT", str(ROOT.parent / "UR5e_Student75"))).resolve()
DEFAULT_SAMPLES = str(LOOP_ROOT / "outputs" / "student75" / "jobs" / "*_evaluate_*" / "diagnostic_states.pt")
LOOP_MANIFEST = LOOP_ROOT / "outputs" / "student75" / "manifest.json"
OUT_ROOT = ROOT / "exported_experimental"
DEPLOY_CONFIGS = ("common.yaml", "safety.yaml", "sim.yaml", "dr.yaml", "real.example.yaml")
EQUIVALENCE_TOL = 1e-4   # GPU(float32)에서 기록된 행동 vs CPU TorchScript


class _ExportActor(nn.Module):
    """Phase2/Export_Policy._ExportActor 미러 — normalizer는 항상 Identity(정규화 off만 export)."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.normalizer = nn.Identity()
        self.actor = actor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _activation(name: str) -> nn.Module:
    table = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "gelu": nn.GELU}
    if name not in table:
        raise ContractError(f"unsupported activation: {name}")
    return table[name]()


def rebuild_teacher_actor(model_sd: dict, agent_yaml: dict) -> nn.Sequential:
    policy = agent_yaml["policy"]
    if policy.get("class_name") != "ActorCritic":
        raise ContractError(f"teacher policy class is {policy.get('class_name')!r}, expected ActorCritic")
    if policy.get("actor_obs_normalization", False) or agent_yaml.get("empirical_normalization"):
        raise ContractError("observation normalization is enabled in the checkpoint; this exporter refuses it")
    hidden = [int(h) for h in policy["actor_hidden_dims"]]
    act = str(policy.get("activation", "elu"))
    layers: list[nn.Module] = []
    dims = [TEACHER_OBS_DIM, *hidden]
    for i in range(len(hidden)):
        layers += [nn.Linear(dims[i], dims[i + 1]), _activation(act)]
    layers.append(nn.Linear(dims[-1], TEACHER_ACTION_DIM))
    actor = nn.Sequential(*layers)
    actor_sd = {k[len("actor."):]: v for k, v in model_sd.items() if k.startswith("actor.")}
    missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
    if missing or unexpected:
        raise ContractError(f"actor state_dict mismatch: missing={missing} unexpected={unexpected}")
    return actor.eval()


def distillation_target(actions: torch.Tensor, vertical: bool) -> torch.Tensor:
    """vision_policy.distillation_teacher_actions 미러 — 기록된 teacher 행동은 이 변환을 거친 값이다.

    (기울임 채널 3·4 = 0(수직 파지), 그리퍼 채널 = 부호 ±1; 병진·yaw는 무클립 그대로.)
    """
    t = actions.clone()
    if vertical:
        t[..., 3:5] = 0.0
    t[..., -1] = torch.where(actions[..., -1] < 0.0, -1.0, 1.0)
    return t


def equivalence_on_samples(scripted: torch.jit.ScriptModule, pattern: str, *, vertical: bool) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        return {"files": 0, "samples": 0, "max_abs_error": None}
    worst = 0.0; count = 0; per_channel = torch.zeros(TEACHER_ACTION_DIM)
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        for stage, v in d.items():
            obs = v["teacher"].float(); ref = v["teacher_actions"].float()
            with torch.no_grad():
                out = distillation_target(scripted(obs), vertical)
            err = (out - ref).abs()
            worst = max(worst, float(err.max())); count += obs.shape[0]
            per_channel = torch.maximum(per_channel, err.max(dim=0).values)
    return {"files": len(files), "samples": count, "max_abs_error": worst,
            "comparison": "distillation_target(scripted(obs)) vs recorded teacher_actions",
            "max_abs_error_per_channel": [float(x) for x in per_channel]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-samples", default=DEFAULT_SAMPLES,
                        help="teacher 관측/행동이 기록된 diagnostic_states.pt glob (동치 검증)")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--allow-config-drift", action="store_true",
                        help="학습 receipt와 현재 canonical config 해시가 달라도 export (manifest에 drift 기록)")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()

    # 1) provenance
    try:
        training = verify_training_checkpoint_receipt(
            checkpoint, project_root=ROOT, expected_experiment="ur5e_binpick_teacher")
    except ContractError as exc:
        print(f"[teacher-export][error] 학습 provenance 실패: {exc}"); return 3
    checkpoint_sha256 = training["checkpoint_sha256"]
    if sha256_file(checkpoint) != checkpoint_sha256:
        print("[teacher-export][error] checkpoint 해시가 receipt와 다릅니다"); return 3
    # ★F93(리뷰 T-09)★ config drift 게이트 — commit이 달라도 canonical config(common/dr/safety/sim/calib)가
    #   학습 시점과 같아야 export한다. 다르면 거부(--allow-config-drift 로만 허용, manifest에 기록).
    from scripts.training_provenance import canonical_config_hashes
    trained_cfgs = dict((training.get("source") or {}).get("canonical_configs") or {})
    current_cfgs = canonical_config_hashes(ROOT)
    config_drift = sorted(k for k in set(trained_cfgs) | set(current_cfgs)
                          if trained_cfgs.get(k) != current_cfgs.get(k))
    if config_drift and not args.allow_config_drift:
        print(f"[teacher-export][error] 학습 시점과 canonical config가 다르다: {config_drift} — "
              "같은 config로 되돌리거나 --allow-config-drift 로 명시적으로 허용할 것"); return 3
    if config_drift:
        print(f"[teacher-export][warn] config drift 허용됨(--allow-config-drift): {config_drift}")

    # 2) 계약
    common = load_yaml(configs_dir() / "common.yaml")
    action_cfg = build_action_cfg(common)
    vertical = bool(common["action"].get("vertical_grasp_only", False))
    agent_yaml = load_yaml(checkpoint.parent / "params" / "agent.yaml")

    # 3) actor 재구성 + TorchScript
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = rebuild_teacher_actor(payload["model_state_dict"], agent_yaml)
    export_mod = _ExportActor(actor).eval()
    try:
        scripted = torch.jit.script(export_mod)
    except Exception as err:  # noqa: BLE001
        print(f"[teacher-export][warn] torch.jit.script 실패 → trace 폴백: {err}")
        with torch.no_grad():
            scripted = torch.jit.trace(export_mod, torch.zeros(1, TEACHER_OBS_DIM))
    probe = torch.randn(64, TEACHER_OBS_DIM)
    with torch.no_grad():
        random_err = float((export_mod(probe) - scripted(probe)).abs().max())
    if random_err > 1e-6:
        print(f"[teacher-export][error] TorchScript 출력이 원본과 다릅니다 (max|Δ|={random_err:.3e})"); return 3
    samples = equivalence_on_samples(scripted, args.reference_samples, vertical=vertical)
    if samples["samples"] and samples["max_abs_error"] > EQUIVALENCE_TOL:
        print(f"[teacher-export][error] 기록된 teacher 행동과 불일치 max|Δ|={samples['max_abs_error']:.3e} "
              f"> {EQUIVALENCE_TOL}"); return 3

    # 4) 번들
    out_dir = args.out_root / f"teacher_{checkpoint_sha256[:12]}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "configs" / "calib").mkdir(parents=True)
    scripted.save(str(out_dir / "policy.pt"))
    spec = make_teacher_obs_spec(action_cfg, action_scheme=SIM_ACTION_SCHEME, vertical_grasp_only=vertical,
                                 contact_force_scale_n=TEACHER_CONTACT_FORCE_SCALE_N)
    spec["policy_sha256"] = sha256_file(out_dir / "policy.pt")
    spec["exported_at"] = dt.datetime.now().isoformat(timespec="seconds")
    validate_teacher_obs_spec(spec, action_cfg)
    (out_dir / "obs_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    copied = []
    for name in DEPLOY_CONFIGS:
        src = configs_dir() / name
        if src.is_file():
            shutil.copy2(src, out_dir / "configs" / name); copied.append(f"configs/{name}")
    for src in sorted((configs_dir() / "calib").glob("*.yaml")):
        shutil.copy2(src, out_dir / "configs" / "calib" / src.name); copied.append(f"configs/calib/{src.name}")

    # ★Teacher v2(F90)★ 채택 근거 = 학습 run의 canonical adoption.json(Phase1/Result/<run>/top5_evaluation).
    #   여기서는 파일 내용을 요약해 기록만 한다(같은 commit 검증은 증류측 verify_adopted_checkpoint의 몫).
    teacher_adoption = None
    adopted = False
    adoption_path = ROOT / "Phase1" / "Result" / Path(training["run_root"]).name / "top5_evaluation" / "adoption.json"
    if adoption_path.is_file():
        try:
            ad = json.loads(adoption_path.read_text(encoding="utf-8"))
            sel = ad.get("selection") or {}
            cand = next((c for c in ad.get("candidates", []) if c.get("checkpoint_sha256") == checkpoint_sha256), None)
            adopted = bool(ad.get("status") == "adopted" and sel.get("checkpoint_sha256") == checkpoint_sha256)
            teacher_adoption = {
                "path": adoption_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(adoption_path),
                "status": ad.get("status"), "selected_checkpoint": sel.get("checkpoint_name"),
                "this_checkpoint_passed": None if cand is None else bool(cand.get("passed")),
                "placed_per_seed": None if cand is None else {
                    str(e["artifact"]["seed"]): int(e["metrics"]["placed"]) for e in cand.get("evaluations", [])},
                "pooled_placed": None if cand is None else cand["pooled"]["placed"]["rate"],
                "pooled_placed_wilson_lower_95": None if cand is None else cand["pooled"]["placed"]["wilson_lower_95"],
                "protocol": ad.get("protocol"),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[teacher-export][warn] adoption.json 요약 실패: {exc}")
            teacher_adoption = None
    elif LOOP_MANIFEST.is_file():
        try:
            lm = json.loads(LOOP_MANIFEST.read_text(encoding="utf-8"))
            ip = lm.get("initial_parent", {})
            if ip.get("teacher_sha256") == checkpoint_sha256:
                teacher_adoption = ip.get("teacher_adoption")
        except Exception:  # noqa: BLE001
            teacher_adoption = None

    training_commit = training["source"]["project_git_commit"]; current_commit = _git_commit()
    manifest = {
        "schema_version": 1,
        "artifact_type": "experimental_teacher_policy",
        "policy_kind": "teacher_state",
        "release": False,
        # 실험 레인 계약(tests/test_experimental_export.py): adopted는 항상 False — sim 채택 여부는
        # teacher_adoption.status / sim_adopted에 기록한다(릴리스 레인의 adopted와 구분).
        "adopted": False,
        "sim_adopted": adopted,
        "not_for_release_reason": (
            "modular deployment v2: Teacher trained on deployable 34D observations (command-state gripper, "
            "perception-estimated cube position with error/dropout DR, held latch) driven by the depth cube "
            "locator on the real robot. Sim-adopted, but no real-robot parity/latency evidence and no "
            "calibrated hand-eye/TCP approval yet. Lab/HIL use only under an explicit motion authorization."
        ),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": checkpoint_sha256,
            "training_run_root": training["run_root"],
            "training_receipt_sha256": training["receipt_sha256"],
            "iteration": int(payload.get("iter", -1)),
        },
        "teacher_adoption": teacher_adoption,
        "git": {"training_commit": training_commit, "export_commit": current_commit,
                "commits_match": training_commit == current_commit,
                "config_drift": config_drift},
        "contract": {"obs_dim": TEACHER_OBS_DIM, "action_dim": TEACHER_ACTION_DIM,
                     "action_scheme": SIM_ACTION_SCHEME, "vertical_grasp_only": vertical},
        "architecture": {"actor_hidden_dims": agent_yaml["policy"]["actor_hidden_dims"],
                         "activation": agent_yaml["policy"].get("activation", "elu")},
        "teacher_obs_cfg": asdict(TeacherObsCfg()),
        "perception": {
            "module": "common/cube_depth_locator.py + common/teacher_obs.py",
            "requires": ["depth frames aligned to colour intrinsics (rs.align depth->color)",
                         "configs/calib/hand_eye.yaml", "configs/calib/tcp_offset.yaml",
                         "table height / cube size matching TeacherObsCfg"],
        },
        "torchscript_equivalence": {"random_probe_max_abs_error": random_err, "recorded_teacher_samples": samples},
        "files": {name: sha256_file(out_dir / name) for name in ("policy.pt", "obs_spec.json")},
        "configs_copied": copied,
    }
    (out_dir / "experimental_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, manifest)
    print(f"[teacher-export] ★비릴리스 Teacher 번들★ → {out_dir}")
    print(f"  checkpoint iter={manifest['checkpoint']['iteration']} sha={checkpoint_sha256[:12]} "
          f"| 학습 커밋 {training_commit[:12]} / export 커밋 {current_commit[:12]}")
    print(f"  TorchScript 동치: 무작위 max|Δ|={random_err:.2e}, 기록 표본 {samples['samples']}개 "
          f"max|Δ|={samples['max_abs_error']}")
    print(f"  설정 사본 {len(copied)}개, teacher_adoption={'있음' if teacher_adoption else '없음'}")
    return 0


def _write_readme(out_dir: Path, manifest: dict) -> None:
    cfg = manifest["teacher_obs_cfg"]
    ta = manifest.get("teacher_adoption") or {}
    seeds = ta.get("placed_per_seed") or {}
    pooled = ta.get("pooled_placed")
    if seeds and pooled is not None:
        adopted_line = (
            f"- sim 봉인 평가(seed 42~45 × 512, 인식 오차 DR·실행 제한 적용): placed {seeds} — pooled "
            f"{pooled * 100:.2f}% (Wilson 95% 하한 {(ta.get('pooled_placed_wilson_lower_95') or 0) * 100:.2f}%), "
            f"채택 상태 `{ta.get('status')}`"
        )
    else:
        adopted_line = "- sim 봉인 평가: 기록 없음"
    hold_s = cfg["max_missed_frames_before_hold"] / 20.0
    (out_dir / "DEPLOY_README.md").write_text(f"""# 모듈형 배포 v2 — 인식 모듈 + Teacher v2 정책 (★릴리스 아님★)

이 번들은 **배포 가능한 관측(34D)으로 학습한 Teacher v2**를 **큐브 인식 모듈**과 결합해 실기에서
돌리기 위한 산출물입니다. 학습 관측 자체가 실기 인식 모듈과 같은 규칙(오차·결측·카메라 기하)으로
만들어졌으므로 근사 항이 없습니다. 실기 패리티·지연 증거와 캘리브레이션 승인이 아직 없으므로,
사람이 e-stop 앞에 있는 랩 환경에서 명시적 모션 인가 아래에서만 사용하십시오.

- checkpoint: `{manifest['checkpoint']['path']}` (iter {manifest['checkpoint']['iteration']})
- sha256: `{manifest['checkpoint']['sha256'][:16]}…`
- 관측/행동: {manifest['contract']['obs_dim']}D 상태 → {manifest['contract']['action_dim']}D `{manifest['contract']['action_scheme']}`
{adopted_line}
- TorchScript 동치: 무작위 프로브 max|Δ| = {manifest['torchscript_equivalence']['random_probe_max_abs_error']:.2e}

## 관측 구성 (sim 학습 = 실기 조립 규칙 동일)
| 신호 | 출처 |
|---|---|
| 관절 위치/속도, TCP 포즈 | UR RTDE (RobotState) |
| gripper_command_state | 지연·전이 적분된 명령 상태(common/gripper_command, 0=open 1=closed) |
| 큐브 위치 | `common/cube_depth_locator.py` depth 역투영 → 작업면 위 높이 띠 {cfg['z_band_m']} m의 큐브 크기 덩어리 → 상면 중앙값, 쥐지 않으면 z를 작업면에 고정. 미검출·held면 TCP+{cfg['held_offset_from_tcp_m']} m, 아니면 마지막 추정치 |
| held | 폐합 전이 시점에 추정치가 TCP {cfg['held_radius_m']} m 이내면 latch, 열리거나 큐브가 멀리서 검출되면 해제 |
| last_action | 직전 정책 출력 clamp ±1, 기울임 채널 0 |

큐브가 {cfg['max_missed_frames_before_hold']}프레임({hold_s:.1f} s) 넘게 안 보이고 쥐고 있지도 않으면 팔을 세운다(hold).
마지막 추정치 바로 위(예상된 블라인드 콘)와 그리퍼 전이 중의 결측은 세지 않는다(F93) — 현 브라켓에서 최종 하강은 항상 FOV 밖이다.
추정치가 {cfg['max_jump_m']} m 넘게 튀면 {cfg['jump_confirm_frames']}프레임 연속 확인 전에는 받지 않는다.
학습 시 인식 DR 범위(configs/dr.yaml perception): 편향 ≤1 cm, xy σ 0.3~1.5 cm, 검출률 60~100%, 최소거리 0.17~0.25 m, 실기 color FOV 밖 결측.

## 배포 PC에서 할 일
1. 이 폴더를 통째로 복사한다. `configs/calib/hand_eye.yaml`·`tcp_offset.yaml`이 실기 캘리브레이션 값인지 확인한다
   (`calibrated: false`면 `deployment/Calib_HandEye.py`, `Calib_TcpOffset.py`로 먼저 캘리브한다).
2. 작업면 높이(base z, 기본 {cfg['table_height_m']} m)가 실기와 맞는지 확인한다 — depth 추정기는 작업면 위
   높이 띠에서 큐브를 찾으므로 이 값이 2 cm 이상 틀리면 큐브를 놓친다. `python deployment/realsense_capture_test.py`
   로 프레임을 저장하고 `common/cube_depth_locator.py`로 검출되는지 본다(색분할은 보조 게이트일 뿐이다).
3. 추론만: `python Phase4/Run_Task_Teacher.py --policy-dir <이 폴더> --dry-run --verbose-perception`
   — 매 스텝 인식 결과(검출/보유/hold)와 행동을 출력한다. 검출률이 30% 미만이면 모션으로 가지 않는다.
   DRY-RUN은 receive-only라 그리퍼 펄스를 못 보내므로 command state를 '개방'으로 가정한다(held latch 없음).
4. 실제 모션(e-stop 앞에서): `--yes-i-am-at-the-estop` (기본 --max-steps 200 = 학습 10 s, --action-scale 1.0 = 학습 조건)
   · `--yes-i-am-at-the-estop`는 인가 파일 대신 ★현장 입회 선언★이다. 캘리브 게이트는 건너뛰지 않는다:
     hand_eye/tcp_offset `meta.calibrated=false`면 거부한다(`--accept-uncalibrated`로만 감수).
   · `configs/real.local.yaml robot.tcp_command_verified`가 false면 어댑터가 servoL을 거부한다 — TCP 명령 검증(README §5) 후 true.
   · 시작 절차(F93): 홈 자세 게이트(관절당 0.05 rad, `--home-first`로 moveJ 허용) → 명시적 개방 펄스 + 1.20 s 안정화 → 루프.
   · 그리퍼 반전은 전이 뒤 6스텝(0.30 s 펄스) 동안 무시된다(F92, 학습과 동일). 릴리스는 연속 3창(150 ms) 정지 뒤에만 성립한다.
   · `--action-scale < 1.0`은 릴리스 수퍼바이저의 절대 정지 임계(0.02 m/s)와 어긋나므로 관찰 목적 외에는 쓰지 않는다.

## 한계
- 실기에서는 인식 오차가 학습 DR 범위 안에 있어야 한다(hand-eye 잔차 ≤1 cm, 검출률 ≥60%). 캘리브가 어긋나면 위치가 통째로 밀린다.
- 현재 hand-eye 기하에서 그리퍼 바로 아래 큐브는 실기 세로 FOV 밖이다(README F88). 정책은 그 결측을 학습에서 경험했지만,
  브라켓을 공구축 쪽으로 수렴하게 재장착하고 hand-eye를 재캘리브하면 검출률이 올라간다.
""", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
