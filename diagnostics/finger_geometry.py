#!/usr/bin/env python
"""★결정적 오프라인 기하 검증★ — 2FG7 손가락 STL 정점 bound + URDF 관절 운동학으로
   그리퍼 개폐 방향·그립 q를 시뮬레이터 없이 확정한다 (순수 python, numpy만).

배경: 구 계보 gate_probe에서 action=close→q=0.019인데 3cm 큐브가 관통·낙하했다.
      당시 주석은 "q=0→44mm, q=0.019→6mm"라 주장했으나,
      URDF 관절 축(left axis=-X, right axis=+X, 각 collision origin이 joint origin을
      상쇄)을 따라가면 q↑에서 두 손가락이 멀어지는(=열림) 것으로 보인다. STL 정점
      bound가 유일한 groud truth → 여기서 직접 파싱해 gap(q)를 계산한다.

계산(전부 onrobot_base_link 프레임, URDF rpy=0이라 순수 평행이동):
  left  vertices(base)  = (v_x - q, v_y, v_z)   # joint origin +0.032239 ⊖ collision origin
  right vertices(base)  = (v_x + q, v_y, v_z)   # joint origin -0.054361 ⊖ collision origin
  (STL은 mm → scale 0.001; collision origin이 joint origin을 정확히 상쇄함 — URDF 확인)

산출: 콘솔 표 (q, left X-range, right X-range, inner-face gap). 3cm 큐브 그립 q 역산.
실행: python diagnostics\\finger_geometry.py   (Isaac 불필요)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

# stdout이 cp949 콘솔/파일 리다이렉트여도 '≈'·'⚠️' 등 print가 죽지 않도록 (§4-1 ②).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MESH = _ROOT / "assets" / "ur5e_2fg7" / "meshes" / "onrobot_2fg7"


def load_stl_vertices(path: Path) -> np.ndarray:
    """바이너리 STL → (N,3) 정점 배열 [mm]. 80B 헤더 + uint32 count + count×(50B)."""
    data = path.read_bytes()
    n_tri = struct.unpack_from("<I", data, 80)[0]
    verts = np.empty((n_tri * 3, 3), dtype=np.float64)
    off = 84
    for i in range(n_tri):
        # 12B normal(skip) + 3×12B vertices + 2B attr
        base = off + i * 50 + 12
        for j in range(3):
            verts[i * 3 + j] = struct.unpack_from("<3f", data, base + j * 12)
    return verts


def main() -> None:
    vL = load_stl_vertices(_MESH / "left_finger_link.stl") * 0.001   # mm → m
    vR = load_stl_vertices(_MESH / "right_finger_link.stl") * 0.001

    # collision origin이 joint origin을 상쇄하므로, q=0에서 mesh 정점 = base 좌표.
    # left  base = (v_x - q, v_y, v_z),  right base = (v_x + q, v_y, v_z)
    print("[geom] ── STL 정점 bound (q=0, base frame, m) ──")
    for nm, v in (("left ", vL), ("right", vR)):
        lo, hi = v.min(0), v.max(0)
        print(f"[geom] {nm}  X[{lo[0]:+.4f},{hi[0]:+.4f}] "
              f"Y[{lo[1]:+.4f},{hi[1]:+.4f}] Z[{lo[2]:+.4f},{hi[2]:+.4f}]")

    lLx0, lLx1 = vL[:, 0].min(), vL[:, 0].max()   # left  mesh X range @q=0
    rRx0, rRx1 = vR[:, 0].min(), vR[:, 0].max()   # right mesh X range @q=0

    # 어느 손가락이 +X/-X 쪽인가? (q=0 mesh 중심)
    cL, cR = 0.5 * (lLx0 + lLx1), 0.5 * (rRx0 + rRx1)
    print(f"\n[geom] q=0 mesh 중심 X: left={cL:+.4f}  right={cR:+.4f}  "
          f"(left이 {'+X' if cL > cR else '-X'}쪽)")

    print(f"\n[geom] ── gap(q): 두 손가락 '마주보는 안쪽 면' 간격 [mm] ──")
    print(f"{'q[m]':>7} {'leftX_lo':>9} {'leftX_hi':>9} {'rightX_lo':>10} {'rightX_hi':>10} {'inner_gap':>10}")
    prev_gap = None
    for q in (0.0, 0.005, 0.007, 0.010, 0.015, 0.020):
        lo_L, hi_L = lLx0 - q, lLx1 - q          # left: -q
        lo_R, hi_R = rRx0 + q, rRx1 + q          # right: +q
        # inner gap = 두 X 구간 사이의 빈 공간 (겹치면 음수)
        if cL > cR:      # left이 +X 쪽 → left의 아래끝 − right의 위끝
            gap = lo_L - hi_R
        else:            # right이 +X 쪽
            gap = lo_R - hi_L
        arrow = ""
        if prev_gap is not None:
            arrow = "▲커짐(열림방향)" if gap > prev_gap else "▼줄음(닫힘방향)"
        prev_gap = gap
        print(f"{q:>7.3f} {lo_L:>9.4f} {hi_L:>9.4f} {lo_R:>10.4f} {hi_R:>10.4f} "
              f"{gap*1000:>9.1f}  {arrow}")

    # 3cm 큐브 그립 q 역산: inner_gap == 0.030 인 q
    # gap(q) = base_gap ∓ 2q (방향에 따라). 선형이므로 두 점으로 해.
    def gap_at(q):
        lo_L, hi_L = lLx0 - q, lLx1 - q
        lo_R, hi_R = rRx0 + q, rRx1 + q
        return (lo_L - hi_R) if cL > cR else (lo_R - hi_L)

    g0, g1 = gap_at(0.0), gap_at(0.020)
    print(f"\n[geom] gap(q=0)={g0*1000:.1f}mm  gap(q=0.020)={g1*1000:.1f}mm")
    print(f"[geom] ★개폐 방향★: q=0은 {'열림(gap大)' if g0 > g1 else '닫힘(gap小)'}, "
          f"q=0.020은 {'닫힘(gap小)' if g0 > g1 else '열림(gap大)'}")
    # 5cm 그립 q
    if abs(g1 - g0) > 1e-9:
        q_cube = 0.020 * (0.050 - g0) / (g1 - g0)
        print(f"[geom] 5cm 큐브 그립 q(gap=50mm) ≈ {q_cube:+.4f} "
              f"(관절범위 [0,0.020] {'내' if 0 <= q_cube <= 0.020 else '★밖★'})")
    print(f"\n[geom] 코드 상수: GRIPPER_OPEN_M=0.0, GRIPPER_CLOSE_M=0.020")
    print(f"[geom]    → 위 '개폐 방향'과 일치해야 정상. 불일치 시 부호 반전 버그(=관통·낙하 원인).")


if __name__ == "__main__":
    sys.exit(main())
