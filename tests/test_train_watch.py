"""★학습 감시기(scripts/tools/train_watch.py) 검증★ — sim 불필요.

왜 필요한가: 이 프로젝트에서 감시 실패는 ★두 번 모두 침묵★이었다.
  · F62 — 죽은 run의 로그를 붙잡고 무한 대기 → 게이트 미발화 → 12시간 소모
  · 2026-08-13 — 이중화 감시가 경로 문제로 기동 즉시 사망, 출력이 없어 구별 불가
감시기는 "돌려놓고 믿는" 물건이라, 그 자체가 검증되지 않으면 ★없는 것보다 나쁘다★
(있다고 믿게 만들기 때문). 그래서 ★실패 경로를 우선★ 고정한다.
"""
from __future__ import annotations

import importlib.util
import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_WATCH = _ROOT / "scripts" / "tools" / "train_watch.py"

# ⚠️감시기는 ★학습 PC 전용★ 도구다(`scripts/`는 배포 번들에 포함되지 않는다 — README 배포 계약).
#   해당 도구가 없는 환경에서는 이 파일이 깨지지 않고 skip돼야 하므로 파일 단위로 건너뛴다.
pytestmark = pytest.mark.skipif(
    not _WATCH.is_file(), reason="학습 PC 전용 도구(scripts/tools/train_watch.py) 없음 — 배포 PC"
)


@pytest.fixture(scope="module")
def watch():
    spec = importlib.util.spec_from_file_location("train_watch", _WATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log_text(iters: list[int], metric_vals: list[float], max_iter: int = 25000) -> str:
    """rsl_rl 콘솔 출력을 최소 형태로 재현한다."""
    out = []
    for it, v in zip(iters, metric_vals):
        out.append(f"                     Learning iteration {it}/{max_iter}")
        out.append(f"      Episode_Termination/placed_success: {v:.4f}")
    return "\n".join(out)


# ── 파싱 ────────────────────────────────────────────────────────────────────
def test_parses_iterations_and_metric(watch):
    text = _log_text([0, 1, 2], [0.0, 0.5, 0.25])
    assert watch.parse_iters(text) == [0, 1, 2]
    assert watch.parse_metric(text, "Episode_Termination/placed_success") == [0.0, 0.5, 0.25]


def test_metric_parser_does_not_confuse_similar_keys(watch):
    """`placed`와 `placed_success`처럼 한쪽이 다른 쪽의 접두인 키를 섞지 않는다."""
    text = "Episode_Reward/placed: 9.9\nEpisode_Termination/placed_success: 0.25\n"
    assert watch.parse_metric(text, "Episode_Termination/placed_success") == [0.25]


# ── ★실패 경로 — 침묵 금지★ ────────────────────────────────────────────────
def _run(*args: str) -> subprocess.CompletedProcess:
    """⚠️한국어 출력을 읽으므로 ★자식/부모 양쪽 모두 UTF-8로 고정★한다.

    Windows 기본 콘솔 코덱(cp949)에 맡기면 감시기가 정상인데도 테스트가 UnicodeDecodeError로
    깨진다 — 검증 도구가 검증 대상을 잘못 고발하는 전형(README 유지보수 계약).
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(_WATCH), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=60, env=env)


def test_exits_immediately_when_log_missing(tmp_path):
    """로그가 없으면 ★대기하지 않고 즉시★ 죽는다 (조용히 사는 것 금지)."""
    r = _run("--log", str(tmp_path / "nope.log"))
    assert r.returncode == 2, r.stdout + r.stderr


def test_exits_immediately_when_iteration_unparsable(tmp_path):
    """★F62 재발 방지★ — iteration이 없는 로그(죽은 run)를 고정하면 즉시 죽고 이유를 말한다."""
    dead = tmp_path / "dead.log"
    dead.write_text("Traceback (most recent call last):\nNameError: STRICT_\n", encoding="utf-8")
    r = _run("--log", str(dead))
    assert r.returncode == 3, r.stdout + r.stderr
    assert "iteration을 파싱하지 못했다" in r.stdout
    assert "F62" in r.stdout                     # 다음 조치를 사람에게 알려야 한다


def test_status_reports_dead_when_no_heartbeat(tmp_path):
    """하트비트가 없으면 ★한 번도 살지 않았다★고 판정한다(종료코드 1)."""
    log = tmp_path / "t.log"
    log.write_text(_log_text([1], [0.0]), encoding="utf-8")
    r = _run("--log", str(log), "--status")
    assert r.returncode == 1
    assert "한 번도 살지 않았다" in r.stdout


def test_status_reports_dead_when_heartbeat_is_stale(tmp_path, watch):
    """★핵심★ — 감시기가 죽으면 하트비트가 굳고, stat 한 번으로 그것이 드러난다."""
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"ts": 0.0, "ts_str": "1970-01-01 00:00:00",
                              "iter": 7, "state": "정상"}), encoding="utf-8")
    assert watch.show_status(hb, stale_s=900.0) == 1     # 오래된 하트비트 = 이상


def test_status_reports_alive_for_fresh_heartbeat(tmp_path, watch):
    import time
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"ts": time.time(), "ts_str": "now", "iter": 7,
                              "state": "정상"}), encoding="utf-8")
    assert watch.show_status(hb, stale_s=900.0) == 0


# ── 게이트 판정 ─────────────────────────────────────────────────────────────
def test_gate_verdict_and_heartbeat_written_on_live_log(tmp_path):
    """실제 로그로 1주기 돌려 게이트 판정 + 하트비트 생성을 확인한다.

    게이트가 최종 게이트면 감시기는 보고 후 스스로 종료하므로 timeout 없이 끝난다.
    """
    log = tmp_path / "run.log"
    log.write_text(_log_text(list(range(1, 11)), [0.02] * 10), encoding="utf-8")
    hb = tmp_path / "run.watch.json"
    r = _run("--log", str(log), "--gate", "5:0.01", "--heartbeat", str(hb),
             "--interval", "1", "--window", "10")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[GATE 5]" in r.stdout and "통과" in r.stdout
    data = json.loads(hb.read_text(encoding="utf-8"))
    assert data["iter"] == 10 and data["metric_avg"] == pytest.approx(0.02)
    assert data["gates_done"] == [5]


def test_gate_marks_shortfall_when_below_threshold(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(_log_text(list(range(1, 11)), [0.0002] * 10), encoding="utf-8")
    r = _run("--log", str(log), "--gate", "5:0.05", "--interval", "1", "--window", "10",
             "--heartbeat", str(tmp_path / "hb.json"))
    assert r.returncode == 0
    assert "[GATE 5]" in r.stdout and "미달" in r.stdout


def test_watcher_survives_cp949_console_when_output_is_redirected(tmp_path):
    """★실사고 고정(2026-08-13)★ 한국어·★ 출력이 인코딩 때문에 감시기를 죽이면 안 된다.

    콘솔 코덱이 cp949인 채로 출력을 파일에 리다이렉트하자 `UnicodeEncodeError`로
    ★기동 즉시 사망★했다. 침묵하지 않겠다는 이 도구의 존재 이유가 인코딩 하나로 무너진다.
    ⇒ `PYTHONIOENCODING`을 ★일부러 비우고★ 돌려, 스크립트 자체의 재설정만으로 살아남는지 본다.
    """
    log = tmp_path / "run.log"
    log.write_text(_log_text(list(range(1, 6)), [0.02] * 5), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    env["PYTHONUTF8"] = "0"                      # UTF-8 모드도 꺼서 기본 코덱을 강제
    r = subprocess.run(
        [sys.executable, str(_WATCH), "--log", str(log), "--gate", "3:0.01",
         "--interval", "1", "--window", "5", "--heartbeat", str(tmp_path / "hb.json")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=env,
    )
    assert "UnicodeEncodeError" not in (r.stderr or ""), f"인코딩으로 죽었다:\n{r.stderr}"
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[GATE 3]" in r.stdout
