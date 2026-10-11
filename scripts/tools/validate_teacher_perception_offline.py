#!/usr/bin/env python
"""★모듈형 배포 v1★ 인식 모듈 오프라인 기하 검증 — GPU·로봇 없이. [학습 PC]

Codex 루프가 봉인한 평가 표본(diagnostic_states.pt)에는 student가 본 전처리 RGB-D(112²)와
그 순간의 teacher 42D 관측(참 큐브 위치·TCP 포즈·접촉력)이 함께 들어 있다. 이 표본에
common.teacher_obs의 인식 경로(depth 역투영 + hand-eye/TCP 체인 + 높이 띠 큐브 추정)를
그대로 적용해 참값과 비교하면, 로봇을 켜기 전에 ★캘리브 체인·프레임 규약·추정기★의 오류를
cm 단위로 드러낼 수 있다.

sim 카메라 intrinsics: Isaac Lab TiledCamera는 f_x = f_y = W·focal/horizontal_aperture(등방)로
렌더·보고한다(vendored tiled_camera.py). sim.yaml의 vfov는 렌더에 반영되지 않으므로 fx=fy=
W/(2·tan(hfov/2))가 참 K다 — 이 값이 실측 오차를 최소화함을 확인했다(2026-09-15).
실기 D435i는 RealCamera.get_intrinsics()의 color intrinsics를 쓴다(같은 함수 경로).

실행:
  python scripts/tools/validate_teacher_perception_offline.py [--json-out <path>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.calibration import load_hand_eye, load_tcp_offset  # noqa: E402
from common.config_io import configs_dir, load_yaml  # noqa: E402
from common.teacher_contract import teacher_obs_slice  # noqa: E402
from common.teacher_obs import T_base_cam_from_tcp, TeacherObsCfg, make_depth_locator_cfg  # noqa: E402
from common.cube_depth_locator import locate_cube_from_depth  # noqa: E402
from common.types import Intrinsics  # noqa: E402

# 루프 worktree는 본 저장소의 ★형제 디렉터리★다. 사용자별 절대경로를 박지 않는다
# (다른 PC에서 못 쓰고 scripts/validate_repository.py 위생 검사에 걸린다).
LOOP_ROOT = Path(os.environ.get("STUDENT_LOOP_ROOT", str(ROOT.parent / "UR5e_Student75"))).resolve()
DEFAULT_SAMPLES = str(LOOP_ROOT / "outputs" / "student75" / "jobs" / "*_evaluate_*" / "diagnostic_states.pt")


def sim_intrinsics(width: int, height: int, hfov_deg: float) -> Intrinsics:
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Intrinsics(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


def depth_from_policy_obs(flat: torch.Tensor, side: int, near: float, far: float) -> np.ndarray:
    P = side * side
    d = flat[:, :P].reshape(-1, side, side).numpy().astype(np.float64)
    depth = near + d * (far - near)
    depth[d >= 0.999] = 0.0
    return depth.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-glob", default=DEFAULT_SAMPLES)
    ap.add_argument("--max-files", type=int, default=24)
    ap.add_argument("--side", type=int, default=112)
    ap.add_argument("--hfov-deg", type=float, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    hfov = args.hfov_deg or float(load_yaml(configs_dir() / "sim.yaml")["camera"]["horizontal_fov_deg"])
    intr = sim_intrinsics(args.side, args.side, hfov)
    cfg = TeacherObsCfg()
    ws = load_yaml(configs_dir() / "safety.yaml")["workspace"]
    loc_cfg = make_depth_locator_cfg(cfg, ws["min"], ws["max"])
    T_flange_cam = load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml")
    T_flange_tcp = load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")
    print(f"[val] intrinsics fx=fy={intr.fx:.2f} (hfov {hfov}°, {args.side}²) | band {loc_cfg.z_band_m} | "
          f"footprint {loc_cfg.footprint_min_frac}~{loc_cfg.footprint_max_frac}×size")

    files = sorted(glob.glob(args.samples_glob))[-args.max_files:]
    if not files:
        print("[val] 표본 없음"); return 2
    stats = defaultdict(lambda: {"n": 0, "visible": 0, "detected": 0, "err": [], "xy": [], "dz": [], "fp": 0,
                                 "reasons": Counter(), "held_true": 0, "held_agree": 0})
    for f in files:
        near, far = 0.20, 1.20
        result = Path(f).parent / "result.json"
        if result.is_file():
            c = json.loads(result.read_text(encoding="utf-8")).get("video_contract", {}).get("configs/common.yaml", {})
            near = float(c.get("preprocess.depth_near_m", near)); far = float(c.get("preprocess.depth_far_m", far))
        d = torch.load(f, map_location="cpu", weights_only=False)
        for stage, v in d.items():
            depth = depth_from_policy_obs(v["policy"].float(), args.side, near, far)
            teach = v["teacher"].float().numpy()
            for i in range(depth.shape[0]):
                t = teach[i]
                tcp = t[teacher_obs_slice("tcp_pose")]; obj = t[teacher_obs_slice("object_pos")]
                held_obs = t[teacher_obs_slice("held")]; grip_cmd = t[teacher_obs_slice("last_action")][-1]
                s = stats[stage]; s["n"] += 1
                T_base_cam = T_base_cam_from_tcp(tcp, T_flange_tcp, T_flange_cam)
                p_cam = np.linalg.inv(T_base_cam) @ np.array([*obj, 1.0])
                visible = p_cam[2] > near and abs(p_cam[0] / p_cam[2] * intr.fx) < intr.cx and abs(p_cam[1] / p_cam[2] * intr.fy) < intr.cy
                s["visible"] += int(visible)
                est = locate_cube_from_depth(depth[i], intr, T_base_cam, loc_cfg)
                if not est.present:
                    s["reasons"][est.reason.split(" (")[0]] += 1
                    continue
                s["detected"] += 1
                center = np.asarray(est.center_base)
                e = float(np.linalg.norm(center - obj))
                s["err"].append(e); s["xy"].append(float(np.linalg.norm(center[:2] - obj[:2]))); s["dz"].append(float(center[2] - obj[2]))
                s["fp"] += int(e > 0.10)
                held_true = bool(held_obs[0] > 0.5)   # ★Teacher v2★ 표본의 held latch(34D 계약)
                held_pred = bool(grip_cmd < 0.0) and float(np.linalg.norm(center - tcp[:3])) <= cfg.held_radius_m
                s["held_true"] += int(held_true); s["held_agree"] += int(held_pred == held_true)

    q = lambda a, p: float(np.percentile(a, p)) if a else float("nan")
    print(f"{'stage':<9}{'n':>5}{'vis%':>6}{'det%':>7}{'det/vis%':>9}{'err p50':>9}{'p90':>7}{'xy p50':>8}{'dz p50':>8}{'>10cm':>7}{'held agree':>12}")
    out = {}
    for stage in ("approach", "grasp", "lift", "carry"):
        s = stats[stage]
        if not s["n"]:
            continue
        print(f"{stage:<9}{s['n']:>5}{100*s['visible']/s['n']:>6.0f}{100*s['detected']/s['n']:>7.1f}"
              f"{100*s['detected']/max(1,s['visible']):>9.1f}{q(s['err'],50)*100:>8.1f}cm{q(s['err'],90)*100:>6.1f}"
              f"{q(s['xy'],50)*100:>8.1f}{q(s['dz'],50)*100:>+8.1f}{s['fp']:>7}{100*s['held_agree']/max(1,s['detected']):>11.1f}%")
        print("          미검출 사유:", dict(s["reasons"].most_common(3)))
        out[stage] = {"n": s["n"], "visible_rate": s["visible"] / s["n"], "detection_rate": s["detected"] / s["n"],
                      "detection_rate_given_visible": s["detected"] / max(1, s["visible"]),
                      "err_p50_m": q(s["err"], 50), "err_p90_m": q(s["err"], 90), "err_xy_p50_m": q(s["xy"], 50),
                      "dz_p50_m": q(s["dz"], 50), "false_positives_over_10cm": s["fp"],
                      "held_agreement": s["held_agree"] / max(1, s["detected"])}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"intrinsics": intr.__dict__, "teacher_obs_cfg": cfg.__dict__,
                                             "locator_cfg": loc_cfg.__dict__, "files": len(files), "stages": out},
                                            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[val] → {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
