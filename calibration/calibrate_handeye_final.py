"""
FINAL, reproducible eye-in-hand hand-eye calibration  (METHOD A: intrinsics calibrated
separately, then FIXED; hand-eye + board pose refined by bundle adjustment).

Pipeline
  1. Detect the ChArUco board in every capture (11x8, square 15mm, marker 11mm,
     DICT_4X4_50, LEGACY pattern).
  2. Camera intrinsics via cv2.calibrateCamera over all views  -> K, dist  (then FIXED).
  3. Bundle adjustment (scipy least_squares, trf + soft_l1) that jointly refines
        X   = T_EE_CAM      (flange -> camera, what we want)
        T_base_board        (fixed board pose in the robot base)
     to minimise the reprojection error of every corner over every view.
  4. Save EVERY result (matrices, intrinsics, board pose, per-view + overall errors)
     to results/ as .json and human-readable .txt.

Board pose convention:
  poses.json[i] = getActualTCPPose() = base->TCP (flange) = [x,y,z, rx,ry,rz] (m, axis-angle)
  T_EE_CAM (=X) maps CAMERA coords to FLANGE coords:  p_flange = X @ p_camera
  Camera pose in base:  T_base_cam = T_base_flange @ X

Run:  python calibrate_handeye_final.py captures_final
"""
import sys, os, json, glob
import numpy as np, cv2
from scipy.optimize import least_squares

# ---- board definition (measured) ----
SQX, SQY, SQ_M, MK_M = 11, 8, 0.015, 0.011
DICT = cv2.aruco.DICT_4X4_50
LEGACY = True

def make_board():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    b = cv2.aruco.CharucoBoard((SQX, SQY), SQ_M, MK_M, d)
    b.setLegacyPattern(LEGACY)
    return b, cv2.aruco.CharucoDetector(b)

def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).reshape(3); return M

def main(capdir):
    board, det = make_board()
    poses = json.load(open(os.path.join(capdir, "poses.json")))
    imgs = sorted(glob.glob(os.path.join(capdir, "cap_*.png")))
    objs, imps, Tbf, used = [], [], [], []
    H = W = None
    for k, (p, ip) in enumerate(zip(poses, imgs)):
        g = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2GRAY); H, W = g.shape
        ch, ids, _, _ = det.detectBoard(g)
        if ids is None or len(ids) < 6:
            continue
        o, i = board.matchImagePoints(ch, ids)
        objs.append(o.reshape(-1, 3).astype(np.float64))
        imps.append(i.reshape(-1, 2).astype(np.float64))
        Rg, _ = cv2.Rodrigues(np.array(p[3:6], float)); Tbf.append(T(Rg, p[:3])); used.append(k)
    n = len(objs)
    print(f"[1] detected board in {n}/{len(imgs)} views, {sum(len(o) for o in objs)} corners")

    # ---- 2) intrinsics (calibrated then FIXED) ----
    rms_intr, K, dist, rvs, tvs = cv2.calibrateCamera(
        [o.astype(np.float32) for o in objs], [i.astype(np.float32) for i in imps], (W, H), None, None)
    print(f"[2] intrinsics (FIXED): RMS={rms_intr:.3f}px fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")

    # ---- 3) bundle adjustment over X (=T_EE_CAM) and T_base_board ----
    Rt2c = [cv2.Rodrigues(r)[0] for r in rvs]; tt2c = [t.reshape(3) for t in tvs]
    Rc2g, tc2g = cv2.calibrateHandEye([Tbf[i][:3,:3] for i in range(n)], [Tbf[i][:3,3] for i in range(n)],
                                      Rt2c, tt2c, method=cv2.CALIB_HAND_EYE_PARK)
    X0 = T(Rc2g, tc2g)
    j = int(np.argmax([len(o) for o in objs]))
    Tbb0 = Tbf[j] @ X0 @ T(Rt2c[j], tt2c[j])
    p0 = np.concatenate([cv2.Rodrigues(X0[:3,:3])[0].ravel(), X0[:3,3],
                         cv2.Rodrigues(Tbb0[:3,:3])[0].ravel(), Tbb0[:3,3]])

    def unpack(p):
        return T(cv2.Rodrigues(p[0:3])[0], p[3:6]), T(cv2.Rodrigues(p[6:9])[0], p[9:12])
    def per_view(p):
        X, Tbb = unpack(p); out = []
        for i in range(n):
            Tcb = np.linalg.inv(Tbf[i] @ X) @ Tbb
            rvec, _ = cv2.Rodrigues(Tcb[:3,:3])
            proj, _ = cv2.projectPoints(objs[i], rvec, Tcb[:3,3], K, dist)
            out.append(proj.reshape(-1,2) - imps[i])
        return out
    resid = lambda p: np.concatenate([r.ravel() for r in per_view(p)])
    xs = np.ones_like(p0); xs[3:6] = 0.1; xs[9:12] = 0.5
    sol = least_squares(resid, p0, method="trf", loss="soft_l1", f_scale=2.0,
                        x_scale=xs, max_nfev=4000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    X, Tbb = unpack(sol.x)

    # ---- errors ----
    pv = [float(np.sqrt(np.mean(r**2))) for r in per_view(sol.x)]
    depths = [float(np.linalg.norm((np.linalg.inv(Tbf[i]@X)@Tbb)[:3,3])) for i in range(n)]
    depth = float(np.median(depths)); fx = K[0,0]
    rms_px = float(np.sqrt(np.mean(resid(sol.x)**2))); rms_mm = rms_px*depth/fx*1000
    pv_mm = [p*d/fx*1000 for p, d in zip(pv, depths)]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(X[:3,:3])-1)/2, -1, 1))))
    print(f"[3] BA done. camera offset (mm)={np.round(X[:3,3]*1000,2).tolist()}  rot={ang:.2f} deg")
    print(f"[*] reprojection RMS = {rms_px:.3f} px = {rms_mm:.3f} mm  (@ {depth:.3f} m)")
    print(f"[*] per-view RMS mm: min {min(pv_mm):.3f}  median {np.median(pv_mm):.3f}  "
          f"mean {np.mean(pv_mm):.3f}  max {max(pv_mm):.3f}")

    # ---- 4) save everything ----
    out = os.path.join(capdir, "results"); os.makedirs(out, exist_ok=True)
    intr = {"fx": K[0,0], "fy": K[1,1], "cx": K[0,2], "cy": K[1,2],
            "dist": dist.ravel().tolist(), "image_size": [W, H], "rms_px": rms_intr}
    results = {
        "method": "A: intrinsics calibrated then FIXED; hand-eye+board bundle-adjusted",
        "board": {"squaresX": SQX, "squaresY": SQY, "square_m": SQ_M, "marker_m": MK_M,
                  "dict": "DICT_4X4_50", "legacy": LEGACY},
        "n_views": n, "n_corners": int(sum(len(o) for o in objs)),
        "T_EE_CAM": X.tolist(), "T_base_board": Tbb.tolist(), "intrinsics": intr,
        "camera_offset_mm": (X[:3,3]*1000).tolist(), "camera_rot_deg": ang,
        "working_distance_m": depth,
        "error": {"reproj_rms_px": rms_px, "reproj_rms_mm": rms_mm,
                  "per_view_mm": {"min": min(pv_mm), "median": float(np.median(pv_mm)),
                                  "mean": float(np.mean(pv_mm)), "max": max(pv_mm)},
                  "per_view_px": {"min": min(pv), "median": float(np.median(pv)),
                                  "mean": float(np.mean(pv)), "max": max(pv)},
                  "per_view_px_list": pv},
    }
    json.dump(results, open(os.path.join(out, "calibration_results.json"), "w"), indent=2)
    json.dump(X.tolist(), open(os.path.join(out, "T_EE_CAM.json"), "w"), indent=2)
    json.dump(intr, open(os.path.join(out, "intrinsics.json"), "w"), indent=2)
    np.savetxt(os.path.join(out, "T_EE_CAM.txt"), X, fmt="%.8f",
               header="T_EE_CAM (flange<-camera 4x4); p_flange = T_EE_CAM @ p_camera")
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("=== Eye-in-hand hand-eye calibration (METHOD A) ===\n\n")
        f.write(f"Board: 11x8 ChArUco, square 15mm, marker 11mm, DICT_4X4_50, LEGACY pattern\n")
        f.write(f"Views used: {n}   corners: {results['n_corners']}\n\n")
        f.write(f"Intrinsics (FIXED): fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}\n")
        f.write(f"  dist(k1,k2,p1,p2,k3)={np.round(dist.ravel(),5).tolist()}  (calib RMS {rms_intr:.3f}px)\n\n")
        f.write("T_EE_CAM (camera pose in the flange frame):\n")
        f.write(np.array2string(X, precision=6, suppress_small=True) + "\n\n")
        f.write(f"camera offset from flange: {np.round(X[:3,3]*1000,2).tolist()} mm, rotation {ang:.2f} deg\n\n")
        f.write("T_base_board (fixed board pose in robot base):\n")
        f.write(np.array2string(Tbb, precision=6, suppress_small=True) + "\n\n")
        f.write(f"REPROJECTION ERROR:  RMS = {rms_px:.3f} px = {rms_mm:.3f} mm  (working dist {depth:.3f} m)\n")
        f.write(f"  per-view mm: min {min(pv_mm):.3f} / median {np.median(pv_mm):.3f} / "
                f"mean {np.mean(pv_mm):.3f} / max {max(pv_mm):.3f}\n")
    print(f"[4] saved all results to {out}/  (calibration_results.json, T_EE_CAM.json/.txt, intrinsics.json, summary.txt)")
    return results

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "captures_final")
