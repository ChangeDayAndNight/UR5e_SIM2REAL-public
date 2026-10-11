"""★F94(2026-09-22)★ Vision 관측 렌더 건전성 가드 — 학습·평가 공통(순수 torch, isaaclab 무의존).

[사고] 새 PC(RTX 5070 Ti, Isaac Sim 5.1 headless "performance" 모드 = DLSS)에서 TiledCamera 타일 수가
  128 이상이면 ★RGB 블록만★ 죽는다(리셋 프레임 완전 검정 41%·이후 스텝마다 검정/부분 렌더가 교대,
  depth는 정상). 64 타일 이하는 정상. `--rendering_mode quality` 또는 `antialiasing_mode="Off"`로
  사라진다(실측 diagnostics — 세션 프로브 128/256 env). 증류(256 env)는 이 상태로 20k iter를 돌아
  student가 깨진 RGB에 적응했고, 채택 평가는 BinPick_Eval의 가드가 3회 연속 EVAL_RENDER_DEGENERATE로
  거부해 드러났다. 학습 경로에는 가드가 없어 ★조용히★ 지나갔다 — 그래서 여기로 뽑아 train.py도 쓴다.

[규칙] depth·RGB 블록의 프레임 내 공간 std(env별 중앙값)가 임계(0.02) 미만이면 죽은 프레임이다.
  정상 depth 0.10~0.24 / RGB 0.09~0.29, 검정 프레임 ~0.001~0.008 — 한 자릿수 이상 여유.
  마커 문자열 `EVAL_RENDER_DEGENERATE`는 scripts/adoption_pipeline.py의 재시도 로직이 grep한다 — 바꾸지 말 것.
"""
from __future__ import annotations

import torch

RENDER_DEGENERATE_MARKER = "EVAL_RENDER_DEGENERATE"
SPATIAL_STD_MIN = 0.02


def vision_render_stats(flat_obs: torch.Tensor, pre_cfg) -> dict:
    """flat 정책 관측(N, D)에서 depth/RGB 블록의 건전성 통계를 만든다."""
    flat = flat_obs.reshape(flat_obs.shape[0], -1).float()
    dh = int(pre_cfg.out_height) * int(pre_cfg.out_width)
    depth = flat[:, :dh]
    out = {
        "depth_spatial_std": float(depth.std(dim=1).median()),
        "depth_far_frac": float((depth >= 0.999).float().mean()),
        "use_rgb": bool(getattr(pre_cfg, "use_rgb", False)),
    }
    if out["use_rgb"]:
        rgb = flat[:, dh:dh + 3 * dh]
        out.update({
            "rgb_spatial_std": float(rgb.std(dim=1).median()),
            "rgb_mean": float(rgb.mean()),
            "rgb_zero_frac": float((rgb <= 0.001).float().mean()),
        })
    return out


def check_vision_render_health(flat_obs: torch.Tensor, pre_cfg, *, where: str = "eval") -> dict:
    """죽은 프레임이면 즉시 RuntimeError(마커 포함) — 조용한 0%/오염 학습보다 즉시 실패가 낫다."""
    s = vision_render_stats(flat_obs, pre_cfg)
    print(f"[{where}][render-guard] depth 공간std={s['depth_spatial_std']:.4f} far비율={100*s['depth_far_frac']:.1f}%")
    if s["depth_spatial_std"] < SPATIAL_STD_MIN:
        raise RuntimeError(
            f"{RENDER_DEGENERATE_MARKER}: depth에 공간 구조가 없다"
            f"(공간std={s['depth_spatial_std']:.4f} < {SPATIAL_STD_MIN}, far={100*s['depth_far_frac']:.1f}%). "
            "렌더가 죽은 상태다 — 이 프로세스를 종료하고 GPU가 완전히 해제된 뒤 ★단독으로★ 재실행할 것. "
            "이 상태의 성공률 0%는 정책 성능이 아니라 계측 아티팩트다."
        )
    if s["use_rgb"]:
        print(f"[{where}][render-guard] rgb 공간std={s['rgb_spatial_std']:.4f} "
              f"평균={s['rgb_mean']:.4f} 검정비율={100*s['rgb_zero_frac']:.1f}%")
        if s["rgb_spatial_std"] < SPATIAL_STD_MIN:
            raise RuntimeError(
                f"{RENDER_DEGENERATE_MARKER}: rgb에 공간 구조가 없다"
                f"(공간std={s['rgb_spatial_std']:.4f} < {SPATIAL_STD_MIN}, 평균={s['rgb_mean']:.4f}, "
                f"검정비율={100*s['rgb_zero_frac']:.1f}%). depth는 살아 있어도 RGB만 죽는 실측 사례가 있다"
                "(F86, F94: 타일 ≥128 + DLSS). student는 4채널을 다 쓰므로 이 상태의 학습·평가는 무효다. "
                "BinPickVisionEnvCfg의 antialiasing_mode='Off'가 살아 있는지, --rendering_mode를 확인할 것."
            )
    return s
