"""학습 감시기 — ★침묵하지 않는 감시★ (순수 python, Isaac 불필요).

목적: 장기 학습(수 시간~수십 시간)의 iteration·지표를 감시하고, 게이트 미달·정체·
이상을 ★즉시★ 드러낸다. 실행:
    python scripts/tools/train_watch.py --log Phase1/logs/teacher_train_YYYYMMDD_HHMMSS.log \
        --gate 1000:0.005 --gate 3000:0.05 --metric Episode_Termination/placed_success
산출: 콘솔 스트림 + ★하트비트 파일★(--heartbeat, 기본 <log>.watch.json).

★왜 이 파일이 존재하는가 (2026-08-13 사용자 지시 — 실사고 2건)★
  ① F62: 감시가 시작 시점에 ★NameError로 죽은 이전 run의 로그★를 붙잡아 iteration을 영영
     읽지 못하고 무한 대기했다. 게이트가 한 번도 안 울렸고 ★12시간★을 태웠다.
  ② 2026-08-13: 이중화 감시가 Windows python의 `/tmp` 해석 차이로 ★기동 즉시 죽었다★.
     출력이 없는 것과 감시가 죽은 것을 구별할 수 없었다.
  ⇒ 두 사고의 공통 실패 양식은 ★침묵★이다. 그래서 이 감시기의 제1원칙은:

     ★"이상 없음"도 주기적으로 소리 내어 말한다 — 침묵은 언제나 고장이다.★

  사건이 있을 때만 출력하는 감시기는, 죽은 것과 조용한 것이 구별되지 않아 감시기가 아니다.

설계 (침묵 불가 4중 장치):
  1. ★하트비트 파일★ — 매 주기 갱신(mtime + iter + 상태). 감시기가 죽으면 파일이 굳는다
     ⇒ 감시자(사람/Claude)가 ★한 번의 stat★으로 생사를 판별할 수 있다. `--status`로 조회.
  2. ★주기적 생존 보고★ — 사건이 없어도 --alive-every 주기마다 "정상" 한 줄을 낸다.
  3. ★능동 경보★ — 로그 불가독 / iteration 미검출 / iteration 정체 / 로그 mtime 정지를
     각각 구분해 경보한다(원인이 다르면 조치가 다르다).
  4. ★기동 자가검증★ — 시작 즉시 로그를 읽어 iteration을 파싱해 보고, 실패하면 대기하지 않고
     ★즉시 0이 아닌 종료코드로 죽는다★(조용히 살아 있는 것보다 즉시 실패가 낫다 — 사고 ②).

★보고 전용★: 학습 프로세스를 중단시키지 않는다. 판정·중단은 사람/Claude가 한다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ★stdout UTF-8 재설정 (README.md §9 ②)★ — 이걸 빠뜨려 실제로 한 번 죽었다(2026-08-13).
#   출력을 파일로 리다이렉트하면 Windows 기본 코덱이 cp949가 되어 한국어·★·— 에서
#   UnicodeEncodeError로 ★감시기가 즉사★한다. 침묵하지 않겠다는 이 파일의 존재 이유가
#   인코딩 하나로 무너지므로, 환경변수(PYTHONIOENCODING)에 기대지 않고 여기서 고정한다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_metric(text: str, key: str) -> list[float]:
    """로그 전체에서 `key: <수>` 를 순서대로 뽑는다."""
    return [float(x) for x in re.findall(re.escape(key) + r":\s*(-?[\d.eE+]+)", text)]


def parse_iters(text: str) -> list[int]:
    return [int(m.group(1)) for m in _ITER_RE.finditer(text)]


def _emit(msg: str) -> None:
    print(f"[watch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_heartbeat(path: Path, payload: dict) -> None:
    """★죽으면 굳는 파일★ — 감시기 생사 판별의 단일 근거."""
    payload = {**payload, "ts": time.time(), "ts_str": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)                     # 원자적 교체 — 반쯤 쓰인 파일을 남기지 않는다
    except OSError as exc:                    # 하트비트 실패가 감시를 죽이면 안 된다
        _emit(f"⚠️하트비트 기록 실패: {exc}")


def show_status(heartbeat: Path, stale_s: float) -> int:
    """--status: 하트비트를 읽어 감시기 생사를 판정한다(종료코드 0=정상, 1=이상)."""
    if not heartbeat.is_file():
        print(f"✗ 하트비트 없음 ({heartbeat}) — 감시기가 ★한 번도 살지 않았다★")
        return 1
    data = json.loads(heartbeat.read_text(encoding="utf-8"))
    age = time.time() - float(data.get("ts", 0.0))
    ok = age <= stale_s
    mark = "★정상★" if ok else "✗정지"
    print(f"{mark} 감시기 — 마지막 갱신 {age:.0f}초 전 ({data.get('ts_str')})")
    print(f"  iter={data.get('iter')}/{data.get('max_iter')}  상태={data.get('state')}")
    print(f"  지표 {data.get('metric')} 최근평균={data.get('metric_avg')}")
    if data.get("gates_done"):
        print(f"  통과 게이트={data['gates_done']}")
    if not ok:
        print(f"  ⇒ {stale_s:.0f}초 넘게 갱신이 없다. 감시기가 죽었다 — ★재기동 필요★")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="학습 로그 감시 (보고 전용)")
    ap.add_argument("--log", required=True, help="감시할 학습 로그 (★실행 중인 것을 명시 고정★)")
    ap.add_argument("--metric", default="Episode_Termination/placed_success",
                    help="게이트 판정에 쓸 로그 키")
    ap.add_argument("--gate", action="append", default=[],
                    help="'iter:임계' 형식, 반복 가능 (예: --gate 3000:0.05)")
    ap.add_argument("--window", type=int, default=200, help="게이트 판정 시 평균낼 최근 표본 수")
    ap.add_argument("--interval", type=float, default=300.0, help="점검 주기[초]")
    ap.add_argument("--alive-every", type=float, default=1800.0,
                    help="사건이 없어도 생존 보고를 내는 주기[초] (침묵 금지)")
    ap.add_argument("--stall-alert-s", type=float, default=1800.0,
                    help="iteration이 이 시간 이상 안 늘면 경보[초]")
    ap.add_argument("--heartbeat", default=None, help="하트비트 파일 (기본 <log>.watch.json)")
    ap.add_argument("--status", action="store_true", help="하트비트만 조회하고 종료")
    args = ap.parse_args()

    log = Path(args.log).resolve()
    hb = Path(args.heartbeat) if args.heartbeat else log.with_suffix(log.suffix + ".watch.json")

    if args.status:
        return show_status(hb, stale_s=max(args.interval * 3, 900.0))

    gates: list[tuple[int, float]] = []
    for g in args.gate:
        it_s, thr_s = g.split(":")
        gates.append((int(it_s), float(thr_s)))
    gates.sort()

    # ── ★기동 자가검증★ — 조용히 살아 있느니 즉시 죽는다 (사고 ②) ──────────────
    if not log.is_file():
        _emit(f"✗기동 실패: 로그가 없다 — {log}")
        return 2
    text = _read(log)
    iters = parse_iters(text)
    if not iters:
        _emit(f"✗기동 실패: 로그에서 iteration을 파싱하지 못했다 — {log}")
        _emit("  ⇒ ★죽은 run의 로그를 고정했을 가능성★(F62 재발). 실행 중인 로그를 확인하라.")
        return 3
    _emit(f"감시 시작 — {log.name} (현재 iter={iters[-1]}, 게이트={gates or '없음'})")

    done: set[int] = set()
    last_iter, last_change_t, last_alive_t = -1, time.time(), 0.0

    while True:
        state = "정상"
        try:
            text = _read(log)
        except OSError as exc:
            state = "로그불가독"
            _emit(f"⚠️로그 읽기 실패: {exc}")
            _write_heartbeat(hb, {"state": state, "iter": last_iter, "log": str(log)})
            time.sleep(args.interval)
            continue

        iters = parse_iters(text)
        if not iters:
            state = "iteration미검출"
            _emit("⚠️iteration 미검출 — 로그 고정이 틀렸을 수 있다(F62 재발 양식)")
            _write_heartbeat(hb, {"state": state, "iter": last_iter, "log": str(log)})
            time.sleep(args.interval)
            continue

        it = iters[-1]
        max_iter = int(_ITER_RE.findall(text)[-1][1])
        vals = parse_metric(text, args.metric)
        win = vals[-args.window:] if vals else []
        avg = sum(win) / len(win) if win else 0.0

        now = time.time()
        if it != last_iter:
            last_iter, last_change_t = it, now
        elif now - last_change_t >= args.stall_alert_s:
            state = "정체"
            _emit(f"⚠️iter={it} 에서 {(now - last_change_t) / 60:.0f}분째 정체 — 프로세스 확인 필요")
            last_change_t = now                       # 같은 정체로 도배하지 않는다

        for g_it, thr in gates:
            if it >= g_it and g_it not in done:
                done.add(g_it)
                verdict = "★통과★" if avg >= thr else "✗미달"
                _emit(f"[GATE {g_it}] iter={it} {args.metric} 최근{len(win)}평균={avg:.5f} "
                      f"(기준 {thr}) → {verdict}")

        # ★침묵 금지★ — 사건이 없어도 살아 있다고 말한다
        if now - last_alive_t >= args.alive_every:
            last_alive_t = now
            _emit(f"생존 — iter={it}/{max_iter} {args.metric} 최근{len(win)}평균={avg:.5f} [{state}]")

        _write_heartbeat(hb, {"state": state, "iter": it, "max_iter": max_iter,
                              "metric": args.metric, "metric_avg": round(avg, 6),
                              "gates_done": sorted(done), "log": str(log)})

        if gates and it >= gates[-1][0] and gates[-1][0] in done:
            _emit("최종 게이트까지 보고 완료 — 감시 종료 (★학습은 그대로 진행★)")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
