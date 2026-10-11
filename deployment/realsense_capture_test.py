"""Phase0 — [사람][배포PC] RealSense D435i 캡처 검증 (Isaac 불필요, env_ur_real 전용).

목적: 배포 PC에서 real/real_camera.py(RealCamera)로 depth 프레임을 N장 받아
      ① raw depth[m] .npy ② common.obs_preprocess.preprocess_depth 통과본 .npy
      ③ 빠른 확인용 PNG(matplotlib 있으면)를 deployment/logs/에 저장한다.
실행 (배포 PC, 생성 bundle root에서, conda env: env_ur_real):
    python deployment\\realsense_capture_test.py --real-config configs\\real.local.yaml --frames 300
산출: deployment/logs/realsense_<시각>/ 아래
      raw_depth_###.npy / preproc_depth_###.npy / preview_###.png / meta.json
      + stdout 성공 sentinel "REALSENSE_CAPTURE_OK".

★ 이 raw dump가 deployment/Verify_ObsParity.py의 "real측 입력"이 된다 ★
  — sim측 렌더와 같은 preprocess_depth(요소1: 단일 공유 전처리)를 통과시켜
  분포(히스토그램/결측률)를 비교하므로, raw .npy를 지우지 말고 보관할 것.

⚠️ 이 스크립트는 isaaclab/isaacsim을 절대 import하지 않는다 (배포 번들 규칙).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
# Canonical source와 build_bundle.py가 만든 hash bundle은 모두 deployment/의 부모를
# root로 사용한다. 생성 bundle의 사본 드리프트는 bundle_manifest.json 검증이 차단한다.
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from common.config_io import build_preprocess_cfg, configs_dir, load_yaml
from deployment.contracts import load_real_config
from common.obs_preprocess import preprocess_depth


def _frame_stats(depth_m: np.ndarray, near: float, far: float) -> dict:
    """raw depth 품질 지표 — 결측률/범위 (real.local.yaml 필터 on/off 결정 근거 데이터)."""
    d = np.asarray(depth_m, dtype=np.float64)
    invalid = ~np.isfinite(d) | (d <= 0.0)
    valid = d[~invalid]
    in_range = (valid >= near) & (valid <= far)
    return {
        "shape": list(d.shape),
        "invalid_ratio": float(invalid.mean()),
        "valid_min_m": float(valid.min()) if valid.size else None,
        "valid_max_m": float(valid.max()) if valid.size else None,
        "valid_mean_m": float(valid.mean()) if valid.size else None,
        "in_range_ratio": float(in_range.mean()) if valid.size else 0.0,
    }


def _save_preview_png(path: Path, raw: np.ndarray, pre: np.ndarray, far: float) -> bool:
    """raw/전처리 depth를 나란히 PNG 저장 — matplotlib 없으면 조용히 스킵."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # 헤드리스 저장 전용 (창 안 띄움)
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(raw, cmap="viridis", vmin=0.0, vmax=far)
    axes[0].set_title("raw depth [m]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(pre, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("preprocessed [0,1]")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase0: D435i 캡처 검증 (배포 PC)")
    parser.add_argument("--frames", type=int, default=300, help="캡처할 프레임 수 (최소 300)")
    parser.add_argument("--real-config", required=True, type=Path,
                        help="로컬 실기 설정 YAML")
    parser.add_argument(
        "--out-dir", type=str, default=str(HERE / "logs"), help="덤프 루트 디렉터리"
    )
    args = parser.parse_args()

    common_yaml = load_yaml(configs_dir() / "common.yaml")
    real_cfg, _ = load_real_config(args.real_config)
    if args.frames < 300:
        parser.error("--frames must be >= 300 for camera evidence")
    pre_cfg = build_preprocess_cfg(common_yaml)

    try:
        from real.real_camera import RealCamera
    except ImportError as e:
        print(f"⚠️ [calib] real/real_camera.py import 실패: {e}")
        print("⚠️ [calib] 배포 번들 복사와 env_ur_real의 pyrealsense2 설치를 확인할 것 (exit 2)")
        raise SystemExit(2)

    # 생성자 계약(real/real_camera.py): camera 섹션 dict + enable_rgb.
    # enable_rgb = 관측 계약(PreprocessCfg.use_rgb) 단일 소스 — real.local.yaml 별도 스위치 금지.
    # (use_rgb=true면 depth가 rs.align으로 color에 정렬된 상태로 덤프됨 — sim측
    #  vertical_fov 42.5°(color intrinsics 계약) 렌더와 같은 조건 = ObsParity 입력으로 유효)
    camera = RealCamera(real_cfg["camera"], enable_rgb=pre_cfg.use_rgb)

    out_dir = Path(args.out_dir) / f"realsense_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[calib] 카메라 시작 (serial='{real_cfg['camera']['serial']}', "
          f"{real_cfg['camera']['width']}x{real_cfg['camera']['height']}"
          f"@{real_cfg['camera']['fps']}fps, filters={real_cfg['camera']['enable_filters']})")
    try:
        camera.start()  # 계약(interfaces.py): 워밍업 프레임 버리기는 start()의 책임
    except RuntimeError as e:
        print(f"⚠️ [calib] 카메라 시작 실패: {e}")
        print("⚠️ [calib] 점검: ① USB3 포트 연결 ② 다른 프로세스 점유 "
              "③ env_ur_real에 pyrealsense2 설치")
        raise SystemExit(1)
    try:
        intr = camera.get_intrinsics()
        print(f"[calib] intrinsics: fx={intr.fx:.2f} fy={intr.fy:.2f} "
              f"cx={intr.cx:.2f} cy={intr.cy:.2f} ({intr.width}x{intr.height})")

        all_stats = []
        png_saved = 0
        for i in range(args.frames):
            frame = camera.get_frame()  # depth_m: (H,W) float32 [m] — 결측은 그대로
            raw = np.asarray(frame.depth_m, dtype=np.float32)

            # ★요소1: sim 학습과 문자 그대로 같은 전처리 함수★ → [1,1,out_h,out_w]
            pre = preprocess_depth(raw, pre_cfg).squeeze().cpu().numpy()

            stats = _frame_stats(raw, pre_cfg.depth_near_m, pre_cfg.depth_far_m)
            stats["timestamp"] = frame.timestamp
            all_stats.append(stats)

            np.save(out_dir / f"raw_depth_{i:03d}.npy", raw)
            np.save(out_dir / f"preproc_depth_{i:03d}.npy", pre)
            if pre_cfg.use_rgb:
                if frame.rgb is None:
                    raise RuntimeError("RGB-D contract active but frame.rgb is missing")
                np.save(out_dir / f"raw_rgb_{i:03d}.npy", frame.rgb)
            if _save_preview_png(out_dir / f"preview_{i:03d}.png", raw, pre, pre_cfg.depth_far_m):
                png_saved += 1

            print(f"[calib] frame {i:03d}: invalid={stats['invalid_ratio']*100:.1f}% "
                  f"mean={stats['valid_mean_m']}m in_range={stats['in_range_ratio']*100:.1f}%")

        if png_saved == 0:
            print("⚠️ [calib] matplotlib 없음 — PNG 미리보기 스킵 (npy는 정상 저장)")

        meta = {
            "purpose": "Phase3 Verify_ObsParity real-side input (raw + preprocessed depth)",
            "camera": real_cfg["camera"],
            "intrinsics": {
                "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
                "width": intr.width, "height": intr.height,
            },
            "preprocess": {
                "depth_near_m": pre_cfg.depth_near_m,
                "depth_far_m": pre_cfg.depth_far_m,
                "out_height": pre_cfg.out_height,
                "out_width": pre_cfg.out_width,
            },
            "frames": all_stats,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        mean_invalid = float(np.mean([s["invalid_ratio"] for s in all_stats]))
        if mean_invalid > 0.3:
            print(f"⚠️ [calib] 평균 결측률 {mean_invalid*100:.1f}% > 30% — 조명/거리/노출 점검, "
                  "dr.yaml dropout_prob 범위 재검토 필요")
            raise SystemExit(1)
        print(f"REALSENSE_CAPTURE_OK frames={args.frames} dir={out_dir}", flush=True)
    finally:
        camera.stop()
        print("[calib] 카메라 종료")


if __name__ == "__main__":
    main()
