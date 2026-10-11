"""
Research-grade hand-eye by nonlinear BUNDLE ADJUSTMENT. Instead of the closed-form
calibrateHandEye (which only uses per-view PnP poses), this jointly optimizes
  X          = T_EE_CAM  (flange -> camera, 6 dof)
  T_base_board                 (board pose in base, 6 dof, fixed target)
  [optionally K + distortion]
to minimize the reprojection error of EVERY ChArUco corner over ALL views. The final
reprojection RMS (px, and mm at the working distance) is the number to report; it also
decomposes the error:
  * RMS -> ~detection floor (~0.3px)  => geometry consistent, hand-eye is good.
  * RMS stays systematically high     => robot forward-kinematics (absolute) is the floor.

  python bundle_adjust.py <capdir> [--intr]      (--intr also refines intrinsics)
"""
import sys, json, glob, os, argparse
import numpy as np, cv2
from scipy.optimize import least_squares

SQX, SQY, SQ_M, MK_M = 11, 8, 0.015, 0.011   # ChArUco 11x8 squares, square 15mm, marker 11mm (200x150 page, DICT_4X4_50)
_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_board = cv2.aruco.CharucoBoard((SQX, SQY), SQ_M, MK_M, _dict)
_board.setLegacyPattern(True)   # board generated with the pre-4.6 (legacy) ChArUco layout
_det = cv2.aruco.CharucoDetector(_board)

def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).reshape(3); return M

def load(capdir):
    poses = json.load(open(os.path.join(capdir, "poses.json")))
    imgs = sorted(glob.glob(os.path.join(capdir, "cap_*.png")))
    objs, imps, Tbf = [], [], []
    H = W = None
    for p, ip in zip(poses, imgs):
        g = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2GRAY); H, W = g.shape
        ch, ids, _, _ = _det.detectBoard(g)
        if ids is None or len(ids) < 6:
            continue
        obj, imp = _board.matchImagePoints(ch, ids)
        objs.append(obj.reshape(-1, 3).astype(np.float64))
        imps.append(imp.reshape(-1, 2).astype(np.float64))
        Rg, _ = cv2.Rodrigues(np.array(p[3:6], float))
        Tbf.append(T(Rg, p[:3]))
    return objs, imps, Tbf, (W, H)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capdir")
    ap.add_argument("--intr", action="store_true", help="also refine intrinsics (fx,fy,cx,cy,k1,k2,p1,p2,k3)")
    ap.add_argument("--sqx", type=float, default=SQ_M, help="ACTUAL cell size along board X (7-col dir), m")
    ap.add_argument("--sqy", type=float, default=SQ_M, help="ACTUAL cell size along board Y (5-row dir), m")
    args = ap.parse_args()
    objs, imps, Tbf, sz = load(args.capdir)
    sx, sy = args.sqx / SQ_M, args.sqy / SQ_M   # anisotropic: cells are rectangular (misprinted)
    if abs(sx - 1) > 1e-9 or abs(sy - 1) > 1e-9:
        for o in objs:
            o[:, 0] *= sx; o[:, 1] *= sy         # charuco corner coords scale per-axis
        print(f"[*] anisotropic rescale: X {SQ_M*1000:.1f}->{args.sqx*1000:.1f}mm (x{sx:.4f}), "
              f"Y {SQ_M*1000:.1f}->{args.sqy*1000:.1f}mm (x{sy:.4f})")
    n = len(objs); print(f"[*] {n} views, {sum(len(o) for o in objs)} corner observations")

    # --- init: intrinsics + per-view board pose (PnP) + closed-form hand-eye ---
    rms0, K, dist, rvs, tvs = cv2.calibrateCamera(
        [o.astype(np.float32) for o in objs], [i.astype(np.float32) for i in imps], sz, None, None)
    print(f"[*] init intrinsic RMS = {rms0:.3f}px  fx={K[0,0]:.1f} cx={K[0,2]:.1f} dist={np.round(dist.ravel(),4).tolist()}")
    Rt2c = [cv2.Rodrigues(r)[0] for r in rvs]; tt2c = [t.reshape(3) for t in tvs]
    Rc2g, tc2g = cv2.calibrateHandEye([Tbf[i][:3,:3] for i in range(n)], [Tbf[i][:3,3] for i in range(n)],
                                      Rt2c, tt2c, method=cv2.CALIB_HAND_EYE_PARK)
    X0 = T(Rc2g, tc2g)                                    # flange -> camera
    j = int(np.argmax([len(o) for o in objs]))            # board pose init from richest view
    Tbb0 = Tbf[j] @ X0 @ T(Rt2c[j], tt2c[j])

    xr0 = cv2.Rodrigues(X0[:3,:3])[0].ravel(); xt0 = X0[:3,3]
    br0 = cv2.Rodrigues(Tbb0[:3,:3])[0].ravel(); bt0 = Tbb0[:3,3]
    p0 = np.concatenate([xr0, xt0, br0, bt0])
    if args.intr:
        p0 = np.concatenate([p0, [K[0,0],K[1,1],K[0,2],K[1,2]], dist.ravel()])

    def unpack(p):
        X = T(cv2.Rodrigues(p[0:3])[0], p[3:6])
        Tbb = T(cv2.Rodrigues(p[6:9])[0], p[9:12])
        if args.intr:
            fx,fy,cx,cy = p[12:16]; d = p[16:21]
            Kf = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]); df = d
        else:
            Kf, df = K, dist.ravel()
        return X, Tbb, Kf, df

    def per_view_resid(p):
        X, Tbb, Kf, df = unpack(p)
        out = []
        for i in range(n):
            Tcb = np.linalg.inv(Tbf[i] @ X) @ Tbb          # board pose in camera
            rvec, _ = cv2.Rodrigues(Tcb[:3,:3]); tvec = Tcb[:3,3]
            proj, _ = cv2.projectPoints(objs[i].astype(np.float64), rvec, tvec, Kf, df)
            out.append((proj.reshape(-1,2) - imps[i]))
        return out

    def resid(p):
        return np.concatenate([r.ravel() for r in per_view_resid(p)])

    rms_init = np.sqrt(np.mean(resid(p0)**2))
    # robust, well-scaled optimization; intrinsics fixed unless --intr
    xs = np.ones_like(p0); xs[3:6] = 0.1; xs[9:12] = 0.5   # meters vs radians scaling
    sol = least_squares(resid, p0, method="trf", loss="soft_l1", f_scale=2.0,
                        x_scale=xs, max_nfev=2000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    X, Tbb, Kf, df = unpack(sol.x)
    rms_fin = np.sqrt(np.mean(resid(sol.x)**2))
    pv = [np.sqrt(np.mean(r**2)) for r in per_view_resid(sol.x)]

    depths = [np.linalg.norm((np.linalg.inv(Tbf[i]@X)@Tbb)[:3,3]) for i in range(n)]
    depth = np.median(depths)
    fx = Kf[0,0] if args.intr else K[0,0]
    mmpp = depth / fx * 1000.0                       # mm per pixel at the working distance
    pv_mm = [p * d / fx * 1000.0 for p, d in zip(pv, depths)]
    print(f"[*] optimizer: status={sol.status} nfev={sol.nfev} cost {sol.cost:.1f}")
    print(f"[*] working distance (median) = {depth:.3f} m,  scale = {mmpp:.3f} mm/px")
    print(f"[*] OVERALL reprojection RMS = {rms_fin:.3f} px  =  {rms_fin*mmpp:.3f} mm")
    print(f"[*] per-view RMS  px:  min {min(pv):.2f}  median {np.median(pv):.2f}  mean {np.mean(pv):.2f}  max {max(pv):.2f}")
    print(f"[*] per-view RMS  mm:  min {min(pv_mm):.3f}  median {np.median(pv_mm):.3f}  mean {np.mean(pv_mm):.3f}  max {max(pv_mm):.3f}")
    if args.intr:
        print(f"[*] refined fx={Kf[0,0]:.1f} fy={Kf[1,1]:.1f} cx={Kf[0,2]:.1f} cy={Kf[1,2]:.1f} dist={np.round(df,4).tolist()}")
    np.set_printoptions(suppress=True, precision=6)
    print("\n=== T_EE_CAM (bundle-adjusted) ===\n", X)
    rows = ", ".join("[" + ", ".join(f"{v:.6f}" for v in r) + "]" for r in X)
    print(f"\nT_EE_CAM = np.array([{rows}])")
    json.dump(X.tolist(), open(os.path.join(args.capdir, "T_EE_CAM_ba.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
