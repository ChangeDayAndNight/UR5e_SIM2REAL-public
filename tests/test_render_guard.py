"""★F94(2026-09-22)★ Vision 렌더 가드 — 학습·평가 공통, DLSS 끔 계약."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from common.render_guard import RENDER_DEGENERATE_MARKER, check_vision_render_health, vision_render_stats

ROOT = Path(__file__).resolve().parents[1]
_PC = SimpleNamespace(out_height=4, out_width=5, use_rgb=True)


def _obs(depth_std: float, rgb_std: float, n: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    dh = _PC.out_height * _PC.out_width
    depth = 0.5 + depth_std * torch.randn(n, dh, generator=g)
    rgb = 0.5 + rgb_std * torch.randn(n, 3 * dh, generator=g)
    rest = torch.zeros(n, 7)
    return torch.cat([depth, rgb, rest], dim=1)


def test_healthy_frames_pass_and_report_stats():
    s = check_vision_render_health(_obs(0.2, 0.2), _PC, where="test")
    assert s["depth_spatial_std"] > 0.1 and s["rgb_spatial_std"] > 0.1 and s["use_rgb"]


def test_black_rgb_with_live_depth_is_rejected():
    """F86/F94 실측 형태: depth는 살아 있는데 RGB만 균일(검정) — 반드시 마커로 죽어야 한다."""
    with pytest.raises(RuntimeError, match=RENDER_DEGENERATE_MARKER):
        check_vision_render_health(_obs(0.2, 0.0), _PC, where="test")


def test_flat_depth_is_rejected():
    with pytest.raises(RuntimeError, match=RENDER_DEGENERATE_MARKER):
        check_vision_render_health(_obs(0.0, 0.2), _PC, where="test")


def test_depth_only_contract_skips_rgb():
    pc = SimpleNamespace(out_height=4, out_width=5, use_rgb=False)
    s = check_vision_render_health(_obs(0.2, 0.0), pc, where="test")
    assert "rgb_spatial_std" not in s


def test_marker_matches_adoption_pipeline_retry_grep():
    src = (ROOT / "scripts" / "adoption_pipeline.py").read_text(encoding="utf-8")
    assert f'_RENDER_DEGENERATE_MARKER = "{RENDER_DEGENERATE_MARKER}"' in src


def test_train_and_eval_both_call_the_shared_guard():
    for rel in ("scripts/rsl_rl/train.py", "Phase1/BinPick_Eval.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "from common.render_guard import check_vision_render_health" in src, rel
        assert "check_vision_render_health(" in src, rel


def test_vision_env_cfg_disables_dlss_antialiasing():
    """타일 ≥128 + DLSS = RGB 사망(F94). Vision cfg는 antialiasing_mode='Off'를 명시해야 한다."""
    src = (ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/binpick_env_cfg.py").read_text(encoding="utf-8")
    body = src.split("class BinPickVisionEnvCfg(")[1].split("\nclass ")[0]
    tree = ast.parse("class _X:\n" + "\n".join("    " + ln for ln in body.splitlines()[1:]) if False else src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value == "Off":
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "antialiasing_mode":
                    found = True
    assert found, "BinPickVisionEnvCfg: self.sim.render.antialiasing_mode = 'Off' 가 없다(F94)"
    assert 'self.sim.render.antialiasing_mode = "Off"' in body
