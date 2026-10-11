#!/usr/bin/env python
"""★배포 v2★ 이력 student export — 보정 적용/미적용 두 번들. [학습 PC]
═══════════════════════════════════════════════════════════════════════════════
 무엇을 만드나
   exported_experimental/student_<sha12>_fovmatch/   시야각 보정 ON
   exported_experimental/student_<sha12>_raw/        시야각 보정 OFF
     policy.pt                  TorchScript ★무상태★ 이력 actor
                                forward(obs[N,50206], history[N,15,286]) → (action[N,7], next_history)
     obs_spec.json              student 관측 계약서(common.policy_io.make_obs_spec — 50,206D)
     experimental_manifest.json  provenance·이력 계약·fov_match 설정·동치 검증
     configs/ · DEPLOY_README.md
   두 번들의 policy.pt는 ★같은 바이트★다. 다른 것은 실기 프레임을 학습 화각에 맞출지뿐이고,
   그 설정은 manifest에 박혀 실행기가 읽는다(실수로 뒤바꿔 돌릴 수 없다).
 왜 무상태인가
   학습측 HistoryStudentNet은 16스텝 버퍼를 파이썬 속성으로 들고 forward에서 갱신한다.
   그대로 내보내면 배포측이 에피소드 경계에서 버퍼를 비울 수 없다. 버퍼를 입출력으로
   드러내고 common.student_policy_io.StudentHistoryPolicyRunner가 들게 한다.
 무엇을 검증하나
   · 학습 receipt와 checkpoint 해시 — 릴리스 schema-2가 없으면 루프 자체
     receipt(result.json: checkpoint 해시 + runtime authority)를 ★필수로★ 검증한다
   · 관측/행동 폭이 현행 계약과 일치
   · ★시퀀스 동치★ — 원본 HistoryStudentNet을 T스텝 굴린 행동과 무상태 모듈 + 외부 버퍼의
     행동이 같은지 대조한다(이력이 실제로 이어지는지까지 본다)
실행:
  python scripts/tools/export_student_policy.py --checkpoint <model_35000.pt>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "source/ur5e_binpick"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.training_env import configure_utf8_stdio  # noqa: E402
configure_utf8_stdio()   # 콘솔이 cp949여도 한글·기호가 깨지거나 죽지 않게

from common.config_io import build_action_cfg, build_preprocess_cfg, configs_dir, load_yaml  # noqa: E402
from common.obs_preprocess import obs_dim  # noqa: E402
from common.policy_io import make_obs_spec, save_obs_spec  # noqa: E402
from deployment.contracts import ContractError, sha256_file  # noqa: E402
from scripts.training_provenance import verify_training_checkpoint_receipt  # noqa: E402
from ur5e_binpick.tasks.binpick.action_scheme import SIM_ACTION_SCHEME  # noqa: E402

# 루프 worktree는 본 저장소의 ★형제 디렉터리★다. 사용자별 절대경로를 박으면 다른 PC에서
# 못 쓰고 저장소 위생 검사(scripts/validate_repository.py)에도 걸린다 — 형제 경로로 유도하고
# 환경변수로 덮어쓸 수 있게 둔다.
LOOP_ROOT = Path(os.environ.get("STUDENT_LOOP_ROOT", str(ROOT.parent / "UR5e_Student75"))).resolve()
ADOPTED_JOB = "00211_train_75beabd3742cea50b2ec"   # 채택 학습 job (model_35000)
DEFAULT_CHECKPOINT = LOOP_ROOT / "outputs" / "student75" / "jobs" / ADOPTED_JOB / "model_35000.pt"
OUT_ROOT = ROOT / "exported_experimental"
DEPLOY_CONFIGS = ("common.yaml", "safety.yaml", "sim.yaml", "dr.yaml", "real.example.yaml")
SEQUENCE_TOL = 1e-5
# D435i color 848x480 공칭 화각. sim은 등방 렌더라 가로 화각이 세로에도 적용된다.
REAL_FOV_DEG = (69.4, 42.5)


class _ExportHistoryActor(nn.Module):
    """무상태 이력 actor — HistoryStudentNet.forward의 계산을 그대로 미러한다.

    학습측: window = cat(history, feature); action = head(feature) + temporal(window.flatten);
            history <- window[:, 1:]
    """

    def __init__(self, student: nn.Module):
        super().__init__()
        self.encoder = student.encoder
        self.enc_fc = student.enc_fc
        self.head = student.head
        self.temporal = student.temporal
        self.h = int(student.h)
        self.w = int(student.w)
        self.in_ch = int(student.in_ch)
        self.img_dim = int(student.img_dim)

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hw = self.h * self.w
        depth = obs[:, :hw].reshape(-1, 1, self.h, self.w)
        if self.in_ch == 1:
            img = depth
        else:
            rgb = obs[:, hw:self.img_dim].reshape(-1, 3, self.h, self.w)
            img = torch.cat([depth, rgb], dim=1)
        visual = self.enc_fc(self.encoder(img))
        feature = torch.cat((visual, obs[:, self.img_dim:]), dim=-1)
        window = torch.cat((history, feature.unsqueeze(1)), dim=1)
        action = self.head(feature) + self.temporal(window.flatten(1))
        return action, window[:, 1:]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def rebuild_history_student(model_sd: dict, agent: dict, num_obs: int, num_actions: int):
    """학습측 클래스를 그대로 써서 재구성한다(구조 재구현으로 갈라지지 않게).

    ⚠️ ``scripts`` 패키지 이름이 본 저장소와 루프 worktree 양쪽에 있으므로 sys.path 조작은
       본 저장소 것을 집는다. 파일 경로로 직접 로드해 ★루프가 실제로 학습에 쓴 클래스★를 쓴다.
    """
    import importlib.util
    source = LOOP_ROOT / "scripts" / "student_loop_history.py"
    if not source.is_file():
        raise ContractError(f"history student source not found: {source}")
    spec_mod = importlib.util.spec_from_file_location("_loop_student_loop_history", source)
    module = importlib.util.module_from_spec(spec_mod)
    sys.modules[spec_mod.name] = module
    spec_mod.loader.exec_module(module)
    HistoryStudentNet = module.HistoryStudentNet
    policy_cfg = agent["policy"]
    if policy_cfg.get("class_name") != "VisionStudentTeacherHistory":
        raise ContractError(f"expected a history student, got {policy_cfg.get('class_name')!r}")
    student = HistoryStudentNet(
        num_obs, num_actions, list(policy_cfg.get("student_hidden_dims", [256, 128])),
        policy_cfg.get("activation", "elu"), history_steps=int(policy_cfg["history_steps"]))
    sub = {k[len("student."):]: v for k, v in model_sd.items() if k.startswith("student.")}
    missing, unexpected = student.load_state_dict(sub, strict=False)
    if missing or unexpected:
        raise ContractError(f"student state_dict mismatch: missing={missing} unexpected={unexpected}")
    return student.eval()


def sequence_equivalence(student, scripted, num_obs: int, steps: int = 24) -> dict:
    """원본(내부 버퍼)과 무상태 모듈(외부 버퍼)을 같은 입력열로 대조한다."""
    torch.manual_seed(0)
    frames = [torch.rand(1, num_obs) for _ in range(steps)]
    student.reset_history()
    reference = []
    with torch.no_grad():
        for x in frames:
            reference.append(student(x).clone())
    history = torch.zeros(1, student.history_steps - 1, student.feature_dim)
    worst = 0.0
    with torch.no_grad():
        for x, ref in zip(frames, reference):
            action, history = scripted(x, history)
            worst = max(worst, float((action - ref).abs().max()))
    # 버퍼를 비우면 첫 행동이 다시 처음과 같아야 한다(에피소드 경계 계약)
    fresh = torch.zeros(1, student.history_steps - 1, student.feature_dim)
    with torch.no_grad():
        restart, _ = scripted(frames[0], fresh)
    restart_error = float((restart - reference[0]).abs().max())
    return {"steps": steps, "max_abs_error": worst, "reset_max_abs_error": restart_error}


def write_bundle(out_dir: Path, scripted, spec: dict, manifest: dict) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "configs" / "calib").mkdir(parents=True)
    scripted.save(str(out_dir / "policy.pt"))
    spec = dict(spec)
    spec["policy_sha256"] = sha256_file(out_dir / "policy.pt")
    spec["exported_at"] = dt.datetime.now().isoformat(timespec="seconds")
    save_obs_spec(out_dir / "obs_spec.json", spec)
    copied = []
    for name in DEPLOY_CONFIGS:
        source = configs_dir() / name
        if source.is_file():
            shutil.copy2(source, out_dir / "configs" / name)
            copied.append(f"configs/{name}")
    for source in sorted((configs_dir() / "calib").glob("*.yaml")):
        shutil.copy2(source, out_dir / "configs" / "calib" / source.name)
        copied.append(f"configs/calib/{source.name}")
    manifest = dict(manifest)
    manifest["configs_copied"] = copied
    manifest["files"] = {name: sha256_file(out_dir / name) for name in ("policy.pt", "obs_spec.json")}
    (out_dir / "experimental_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(out_dir, manifest)


def verify_loop_job_receipt(checkpoint: Path, checkpoint_sha256: str) -> dict:
    """루프 job의 ★자체 receipt★(result.json)로 checkpoint 출처를 검증한다.

    왜 필요한가
      릴리스 학습은 schema-2 receipt를 쓰지만 student 루프 job은 자체 스키마의
      ``result.json``을 쓴다. 그래서 ``verify_training_checkpoint_receipt``가 실패하고,
      예전에는 ★파일 sha256만★ 남았다 — "이 파일은 이 파일이다"라는 동어반복이라
      '어느 학습이 만들었는가'를 전혀 묶지 못한다.
      루프 receipt는 checkpoint별 sha256에 더해 runtime authority(git commit·runtime
      파일 해시·canonical yaml 해시)와 요청/환경 해시를 담는다. 스키마만 다를 뿐 결속력은
      릴리스 receipt와 같은 종류다. 그것을 검증해 기록한다.
    """
    receipt_path = checkpoint.parent / "result.json"
    if not receipt_path.is_file():
        raise ContractError(f"루프 receipt 없음: {receipt_path}")
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    if receipt.get("kind") != "training":
        raise ContractError(f"receipt kind가 training이 아니다: {receipt.get('kind')!r}")
    if receipt.get("valid") is not True:
        raise ContractError("receipt가 valid=false — 완주하지 못한 job의 checkpoint는 배포 금지")
    entries = receipt.get("checkpoints") or []
    match = next((c for c in entries if Path(str(c.get("path", ""))).name == checkpoint.name), None)
    if match is None:
        raise ContractError(
            f"receipt에 {checkpoint.name} 항목이 없다 (기록된 것: "
            f"{[Path(str(c.get('path',''))).name for c in entries]})")
    if str(match.get("sha256")) != checkpoint_sha256:
        raise ContractError(
            f"★checkpoint 해시 불일치★: receipt {match.get('sha256')} vs 파일 {checkpoint_sha256} — "
            "학습 뒤 파일이 바뀌었다")
    authority = receipt.get("runtime_authority") or {}
    return {
        "schema": "student_loop_result_v1",
        "job_dir": str(checkpoint.parent),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "request_sha256": receipt.get("request_sha256"),
        "environment_sha256": receipt.get("environment_sha256"),
        "next_iteration": receipt.get("next_iteration"),
        "runtime_authority": {
            "git_commit": authority.get("git_commit"),
            "manifest_sha256": authority.get("manifest_sha256"),
            "runtime_files_sha256": authority.get("runtime_files_sha256"),
            "canonical_yaml_sha256": authority.get("canonical_yaml_sha256"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()

    checkpoint_sha256 = sha256_file(checkpoint)
    try:
        training = verify_training_checkpoint_receipt(
            checkpoint, project_root=LOOP_ROOT, expected_experiment="ur5e_binpick_vision")
    except ContractError as exc:
        print(f"[student-export] 릴리스 schema-2 receipt 없음({exc}) → 루프 자체 receipt로 검증한다")
        training = None
    # 루프 job은 자체 스키마 receipt를 쓴다. 이것까지 없으면 출처를 묶을 수 없으므로 ★중단★한다.
    loop_receipt = verify_loop_job_receipt(checkpoint, checkpoint_sha256)
    print(f"[student-export] 루프 receipt 검증 OK — job {Path(loop_receipt['job_dir']).name}, "
          f"학습 커밋 {str(loop_receipt['runtime_authority']['git_commit'])[:12]}")

    common = load_yaml(configs_dir() / "common.yaml")
    pre_cfg = build_preprocess_cfg(common)
    action_cfg = build_action_cfg(common)
    expected_obs, expected_act = int(obs_dim(pre_cfg)), int(action_cfg.action_dim)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = json.loads((checkpoint.parent / "agent.json").read_text(encoding="utf-8"))
    student = rebuild_history_student(payload["model_state_dict"], agent, expected_obs, expected_act)
    export_mod = _ExportHistoryActor(student).eval()
    try:
        scripted = torch.jit.script(export_mod)
    except Exception as err:  # noqa: BLE001
        print(f"[student-export][warn] torch.jit.script 실패 → trace 폴백: {err}")
        with torch.no_grad():
            scripted = torch.jit.trace(
                export_mod, (torch.zeros(1, expected_obs),
                             torch.zeros(1, student.history_steps - 1, student.feature_dim)))
    equivalence = sequence_equivalence(student, scripted, expected_obs)
    if equivalence["max_abs_error"] > SEQUENCE_TOL or equivalence["reset_max_abs_error"] > SEQUENCE_TOL:
        print(f"[student-export][error] 시퀀스 동치 실패: {equivalence}")
        return 3

    spec = make_obs_spec(pre_cfg, action_cfg, action_scheme=SIM_ACTION_SCHEME)
    sim_camera = load_yaml(configs_dir() / "sim.yaml")["camera"]
    # Kit 렌더는 등방 픽셀(f_x=f_y) — 학습 세로 화각은 가로 화각과 렌더 격자 종횡비로 정해진다.
    #   ★Student v2(F89)★ sim.yaml camera 112×63이면 42.6°(≈ 실기 42.5°) → 레터박스는 항등에 가깝다.
    _hfov = float(sim_camera["horizontal_fov_deg"])
    _vfov_render = math.degrees(2.0 * math.atan(
        float(sim_camera["height"]) / float(sim_camera["width"]) * math.tan(math.radians(_hfov) / 2.0)))
    training_fov = (_hfov, round(_vfov_render, 2))
    scale_y = math.tan(math.radians(REAL_FOV_DEG[1]) / 2) / math.tan(math.radians(training_fov[1]) / 2)

    base = {
        "schema_version": 1,
        "artifact_type": "experimental_student_policy",
        "policy_kind": "vision_student_history",
        "release": False,
        "adopted": False,
        "not_for_release_reason": (
            "student adoption gate not passed (3 of 4 seeds cleared 384/512; seed 43 reached 359); "
            "no parity dataset, no latency benchmark. Lab/HIL use only under an explicit motion authorization."
        ),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint), "sha256": checkpoint_sha256,
            "iteration": int(payload.get("iter", -1)),
            "training_run_root": (training or {}).get("run_root"),
            "training_receipt_sha256": (training or {}).get("receipt_sha256"),
            "loop_receipt": loop_receipt,
            "sealed_final": {"seeds": {"42": 446, "43": 359, "44": 394, "45": 404},
                             "min_seed": 359, "episodes_per_seed": 512, "pooled": 0.783},
        },
        "history": {"steps": int(student.history_steps), "feature_dim": int(student.feature_dim),
                    "stateful": False,
                    "note": "forward(obs, history) -> (action, next_history); the runner owns the buffer and must reset per episode"},
        # 학습 커밋은 루프 receipt의 runtime authority가 들고 있다. 두 커밋을 ★둘 다★ 남기고
        # 불일치를 숨기지 않는다(실험 export 레인 공통 계약).
        "git": {
            "training_commit": loop_receipt["runtime_authority"]["git_commit"],
            "export_commit": _git_commit(),
            "commits_match": loop_receipt["runtime_authority"]["git_commit"] == _git_commit(),
        },
        "contract": {"obs_dim": expected_obs, "action_dim": expected_act, "action_scheme": SIM_ACTION_SCHEME},
        "torchscript_sequence_equivalence": equivalence,
        "training_fov_deg": {"horizontal": training_fov[0], "vertical": training_fov[1],
                             "source": "Kit renders with isotropic pixels (f_x=f_y): vertical FOV = "
                                       "2*atan(height/width*tan(hfov/2)) from sim.yaml camera "
                                       f"{sim_camera['width']}x{sim_camera['height']} (F89 aspect contract)"},
        "real_fov_deg": {"horizontal": REAL_FOV_DEG[0], "vertical": REAL_FOV_DEG[1]},
    }

    for suffix, fov_match in (("fovmatch", True), ("raw", False)):
        manifest = dict(base)
        manifest["fov_match"] = {
            "enabled": fov_match,
            "transform": "common.fov_match.FovMatchedCamera (letterbox before the shared preprocessing)",
            "vertical_content_fraction": round(scale_y, 4),
            "rationale": (
                "real frames are vertically magnified about "
                f"{1/scale_y:.2f}x relative to training; the letterbox restores the training geometry "
                "at the cost of padded (dataless) bands"
                if fov_match else
                "the frame reaches the policy exactly as the current contract defines; keeps the real "
                "context that the letterbox would replace with padding"),
        }
        out_dir = args.out_root / f"student_{checkpoint_sha256[:12]}_{suffix}"
        write_bundle(out_dir, scripted, spec, manifest)
        print(f"[student-export] {out_dir.name}  fov_match={fov_match}")
    print(f"  checkpoint iter={base['checkpoint']['iteration']} sha={checkpoint_sha256[:12]} "
          f"| obs/act {expected_obs}/{expected_act} | history {student.history_steps}단계")
    print(f"  시퀀스 동치 max|Δ| = {equivalence['max_abs_error']:.2e} "
          f"(reset {equivalence['reset_max_abs_error']:.2e}, {equivalence['steps']}스텝)")
    print(f"  학습 화각 {training_fov[0]}x{training_fov[1]}° vs 실기 {REAL_FOV_DEG[0]}x{REAL_FOV_DEG[1]}° "
          f"→ 세로 내용 비율 {scale_y:.3f}")
    return 0


def _write_readme(out_dir: Path, manifest: dict) -> None:
    fov = manifest["fov_match"]
    ck = manifest["checkpoint"]
    state = "적용" if fov["enabled"] else "미적용"
    (out_dir / "DEPLOY_README.md").write_text(f"""# 배포 v2 — 시각 student (시야각 보정 **{state}**) ★릴리스 아님★

end-to-end 시각 정책입니다. 112×112 RGB-D와 관절 상태를 받아 7차원 행동을 냅니다.
배포 승인 증거가 없으므로 사람이 e-stop 앞에 있는 랩 환경에서만, 명시적 모션 인가 아래 쓰십시오.

- checkpoint: iter {ck['iteration']}, sha `{ck['sha256'][:16]}…`
- sim 봉인 최종(4시드 × 512): {ck['sealed_final']['seeds']} — 최저 {ck['sealed_final']['min_seed']}, pooled {ck['sealed_final']['pooled']*100:.1f}%
  (채택 게이트는 네 시드 모두 384 이상 — **미달**, seed 43이 25 부족)
- 관측/행동: {manifest['contract']['obs_dim']} / {manifest['contract']['action_dim']} `{manifest['contract']['action_scheme']}`
- 이력: {manifest['history']['steps']}단계, 무상태 export — 러너가 버퍼를 들고 **에피소드마다 reset**
- TorchScript 시퀀스 동치: max|Δ| {manifest['torchscript_sequence_equivalence']['max_abs_error']:.2e}

## 시야각 보정: **{state}**

Isaac은 카메라를 **등방**으로 렌더합니다(f_x=f_y). 학습에서 student가 본 112² 프레임은 가로·세로
모두 {manifest['training_fov_deg']['horizontal']}°를 담습니다. 실기 D435i는 848×480에 가로
{manifest['real_fov_deg']['horizontal']}°·세로 {manifest['real_fov_deg']['vertical']}°이므로, 보정하지 않으면
같은 물체가 **세로로 약 {1/fov['vertical_content_fraction']:.2f}배 크게** 보입니다.

{"이 번들은 실기 프레임을 학습 화각 캔버스 가운데(세로 " + str(round(fov['vertical_content_fraction']*100)) + "%)에 놓고 위아래를 채웁니다. 물체의 화면 점유가 학습과 일치하는 대신, 채워진 띠에는 **데이터가 없습니다**(depth 결측·RGB 검정)." if fov['enabled'] else "이 번들은 프레임을 **그대로** 넘깁니다. 기하는 어긋나지만 실기 장면의 주변 맥락이 보존됩니다."}

두 번들은 **같은 정책**이며 이 설정만 다릅니다. 어느 쪽이 유리한지는 실기 대조로만 정해집니다.
두 폴더를 모두 복사해 번갈아 돌리고 결과를 비교하십시오.

## 배포 PC에서 할 일
1. 폴더를 통째로 복사하고 `configs/calib/*.yaml`이 실기 캘리브레이션 값인지 확인합니다.
2. 추론만: `python Phase4/Run_Task_Student.py --policy-dir <이 폴더> --dry-run`
3. 실제 모션(e-stop 앞에서): `--yes-i-am-at-the-estop` (기본 --action-scale 1.0 = 학습 조건, F93)

## 한계
- sim 성공률이 78.3%이고 채택 게이트에 미달했습니다. 실기 성공률은 그보다 낮습니다.
- 정책은 큐브를 "인식"하지 않습니다. 빈 장면에서도 무언가를 출력하므로 큐브 게이트가 필요합니다.
- 모듈형 v1(`teacher_*` 번들)은 인식이 명시적이라 실패 원인을 가르는 대조군으로 쓰십시오.
""", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
