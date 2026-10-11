#!/usr/bin/env python
"""★USD 콜라이더 직접 검증★ — canonical UR5e/2FG7 USD collision prim을 조사.

배경: 구 계보(q=0.019 닫힘, 3cm 큐브) 프로브에서 손가락이 큐브를 관통하고
      fL 접촉력=0이었다. 가설: convex_hull 변환이 한쪽/양쪽
      손가락 링크에 콜라이더를 못 만들었거나(=관통), 콜라이더 근사/범위가 잘못됨.
      → USD를 열어 각 손가락 링크의 CollisionAPI·근사법·bound·물리재질을 직접 확인한다.

검사: 팔 전체 트리에서 UsdPhysics.CollisionAPI 보유 prim 목록 + 손가락/큐브 collision
      mesh의 approximation(convexHull/none/…)·world AABB·physics material(friction).
실행: python diagnostics\\usd_collision_inspect.py   (headless Kit, 수초)
      다른 USD를 조사할 때만 --usd <PATH>를 명시한다.
산출: 콘솔 리포트. (Isaac Sim이 pxr을 제공 — AppLauncher 불필요.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()




os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_USD = _ROOT / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.usd"

# ★Kit 기동★ — pxr DLL 경로 확보 목적(헤드리스). AppLauncher 후에만 pxr import 가능.
import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument("--usd", type=Path, default=_DEFAULT_USD)
AppLauncher.add_app_launcher_args(_parser)
_args, _ = _parser.parse_known_args()
_args.headless = True
_app_launcher = AppLauncher(_args)
_simulation_app = _app_launcher.app
_USD = _args.usd.expanduser().resolve()

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def main() -> None:
    if not _USD.is_file():
        raise SystemExit(f"[usd] USD 없음: {_USD}")
    stage = Usd.Stage.Open(str(_USD))
    print(f"[usd] 열림: {_USD.name}  (defaultPrim={stage.GetDefaultPrim().GetPath()})")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    targets = ("left_finger", "right_finger", "onrobot_base", "wrist_3")
    print("\n[usd] ── 관심 링크의 collision prim (approximation · world AABB) ──")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName()
        low = path.lower()
        if not any(t in low for t in targets):
            continue
        has_coll = prim.HasAPI(UsdPhysics.CollisionAPI)
        is_mesh = prim.IsA(UsdGeom.Mesh)
        # MeshCollisionAPI 근사법
        approx = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mc = UsdPhysics.MeshCollisionAPI(prim)
            approx = mc.GetApproximationAttr().Get()
        # collision 활성 여부
        coll_enabled = None
        if has_coll:
            coll_enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if has_coll or is_mesh or approx is not None:
            try:
                aabb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                mn, mx = aabb.GetMin(), aabb.GetMax()
                ext = (f"X[{mn[0]:+.4f},{mx[0]:+.4f}] "
                       f"Y[{mn[1]:+.4f},{mx[1]:+.4f}] Z[{mn[2]:+.4f},{mx[2]:+.4f}]")
            except Exception as e:
                ext = f"(bound 계산 실패: {e})"
            tag = "MESH" if is_mesh else prim.GetTypeName()
            print(f"[usd] {path}")
            print(f"        type={tag} collAPI={has_coll} enabled={coll_enabled} "
                  f"approx={approx}")
            print(f"        AABB {ext}")

    # 물리 재질(friction) — 손가락/큐브에 물리 재질이 바인딩됐는지
    print("\n[usd] ── PhysicsMaterial (friction) 정의 목록 ──")
    found_mat = False
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            found_mat = True
            m = UsdPhysics.MaterialAPI(prim)
            print(f"[usd] {prim.GetPath()}  "
                  f"staticFric={m.GetStaticFrictionAttr().Get()} "
                  f"dynFric={m.GetDynamicFrictionAttr().Get()} "
                  f"restitution={m.GetRestitutionAttr().Get()}")
    if not found_mat:
        print("[usd] ⚠️ PhysicsMaterial(friction) prim이 하나도 없음 — 기본 재질만 사용 중.")

    # ★결정적★: 손가락 collision MESH의 실제 points bound + world 변환 (스케일/위치 버그 확인)
    #   ★instance proxy 순회★ — 메시가 instanceable prototype 안에 있으면 일반 Traverse가 못 봄.
    print("\n[usd] ── 손가락 링크 collision MESH 실측 (points·extent·world 위치, instance-proxy) ──")
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for finger in ("left_finger_link", "right_finger_link"):
        hits = [p for p in stage.Traverse() if p.GetName() == finger]
        if not hits:
            print(f"[usd] ⚠️ '{finger}' prim 자체가 없음")
            continue
        link = hits[0]
        print(f"[usd] {link.GetPath()}  instanceable={link.IsInstanceable()}:")
        for child in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            typ = child.GetTypeName()
            apis = []
            if child.HasAPI(UsdPhysics.CollisionAPI):
                apis.append("Coll")
            if child.HasAPI(UsdPhysics.MeshCollisionAPI):
                apis.append(f"approx={UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Get()}")
            line = f"        {str(child.GetPath()).replace(str(link.GetPath()),'.')}  [{typ}]  {' '.join(apis)}"
            if child.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child)
                pts = mesh.GetPointsAttr().Get()
                npts = len(pts) if pts else 0
                line += f"  points={npts}"
                if npts:
                    import numpy as _np
                    a = _np.array([[p[0], p[1], p[2]] for p in pts])
                    lo, hi = a.min(0), a.max(0)
                    line += (f"\n            local bound X[{lo[0]:+.4f},{hi[0]:+.4f}] "
                             f"Y[{lo[1]:+.4f},{hi[1]:+.4f}] Z[{lo[2]:+.4f},{hi[2]:+.4f}]")
                    # world 변환 적용
                    m = xf_cache.GetLocalToWorldTransform(child)
                    corners = _np.array([[lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]]])
                    wc = []
                    for c in corners:
                        v = m.Transform((float(c[0]), float(c[1]), float(c[2])))
                        wc.append([v[0], v[1], v[2]])
                    wc = _np.array(wc)
                    wlo, whi = wc.min(0), wc.max(0)
                    line += (f"\n            WORLD bound X[{wlo[0]:+.4f},{whi[0]:+.4f}] "
                             f"Y[{wlo[1]:+.4f},{whi[1]:+.4f}] Z[{wlo[2]:+.4f},{whi[2]:+.4f}]")
                    # xform scale 확인
                    sc = (m.GetRow3(0).GetLength(), m.GetRow3(1).GetLength(), m.GetRow3(2).GetLength())
                    line += f"\n            xform scale≈({sc[0]:.3f},{sc[1]:.3f},{sc[2]:.3f})"
            print(line)

    # ★핵심 산출★: collision mesh 중심의 '링크 프레임' 오프셋 — pregrasp/TCP를 pad 중심으로
    #   옮기는 데 필요. finger body(link) 원점 대비 pad가 얼마나 떨어졌는가.
    print("\n[usd] ── collision mesh 중심의 링크-프레임 오프셋 (pad ↔ body 원점) ──")
    for finger in ("left_finger_link", "right_finger_link"):
        hits = [p for p in stage.Traverse() if p.GetName() == finger]
        if not hits:
            continue
        link = hits[0]
        meshes = [p for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies())
                  if p.IsA(UsdGeom.Mesh) and "collision" in str(p.GetPath()).lower()]
        if not meshes:
            continue
        mesh = meshes[0]
        pts = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
        import numpy as _np
        a = _np.array([[p[0], p[1], p[2]] for p in pts])
        centroid_local = 0.5 * (a.min(0) + a.max(0))   # 메시 로컬 bbox 중심 (mm, mesh 프레임)
        # mesh → link 상대변환
        rel = UsdGeom.XformCache(Usd.TimeCode.Default()).ComputeRelativeTransform(mesh, link)[0]
        v = rel.Transform((float(centroid_local[0]), float(centroid_local[1]), float(centroid_local[2])))
        print(f"[usd] {finger}: pad 중심(link frame) = ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}) m")

    # 충돌 필터링(그룹/pair) 확인 — finger↔cube 쌍이 제외됐는지
    print("\n[usd] ── CollisionGroup / filteredPairs (충돌 제외 확인) ──")
    grp = False
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.CollisionGroup):
            grp = True
            print(f"[usd] CollisionGroup: {prim.GetPath()}")
    if not grp:
        print("[usd] CollisionGroup 없음 (USD 레벨 필터링 없음 — 정상)")


if __name__ == "__main__":
    main()
    _simulation_app.close()
