"""
Self-calibrate the camera intrinsics from the captured ChArUco views (factory RealSense
color intrinsics report zero distortion, which biases PnP when the board is off-center),
then redo hand-eye with the refined K+dist and validate. Writes refined intrinsics.json
and T_EE_CAM.json if it improves consistency.

  python refine_and_calibrate.py captures
"""
import sys, json, glob, os
import numpy as np, cv2

def main(capdir):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((11, 8), 0.015, 0.011, d)   # ChArUco 11x8, square 15mm, marker 11mm
    board.setLegacyPattern(True)   # board generated with the pre-4.6 (legacy) ChArUco layout
    det = cv2.aruco.CharucoDetector(board)
    poses = json.load(open(os.path.join(capdir, "poses.json")))
    imgs = sorted(glob.glob(os.path.join(capdir, "cap_*.png")))
    H = W = None
    objpts, imgpts, kept_poses = [], [], []
    for p, ip in zip(poses, imgs):
        img = cv2.imread(ip); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = g.shape[:2]
        ch, ids, _, _ = det.detectBoard(g)
        if ids is None or len(ids) < 6:
            continue
        obj, imp = board.matchImagePoints(ch, ids)
        objpts.append(obj.reshape(-1, 3).astype(np.float32))
        imgpts.append(imp.reshape(-1, 2).astype(np.float32))
        kept_poses.append(p)
    print(f"[*] {len(objpts)} views for intrinsic calibration")

    # 1) intrinsic calibration from the ChArUco views
    flags = 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, (W, H), None, None, flags=flags)
    print(f"[*] intrinsic RMS reprojection = {rms:.3f} px")
    print(f"    fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    print(f"    dist={np.round(dist.reshape(-1),5).tolist()}")

    # 2) hand-eye using the refined per-view board poses (target2cam = rvecs/tvecs)
    def T(R, t):
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).reshape(3); return M
    Rg2b, tg2b, Rt2c, tt2c = [], [], [], []
    for p, rv, tv in zip(kept_poses, rvecs, tvecs):
        Rgb, _ = cv2.Rodrigues(np.array(p[3:6], float))
        Rc, _ = cv2.Rodrigues(rv)
        Rg2b.append(Rgb); tg2b.append(np.array(p[:3], float))
        Rt2c.append(Rc); tt2c.append(tv.reshape(3))

    best = None
    for name, m in [("TSAI", cv2.CALIB_HAND_EYE_TSAI), ("PARK", cv2.CALIB_HAND_EYE_PARK),
                    ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD), ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
                    ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS)]:
        Rc2g, tc2g = cv2.calibrateHandEye(Rg2b, tg2b, Rt2c, tt2c, method=m)
        Tx = T(Rc2g, tc2g)
        c = np.array([(T(Rg2b[i], tg2b[i]) @ Tx @ T(Rt2c[i], tt2c[i]))[:3, 3] for i in range(len(Rg2b))])
        spread = np.linalg.norm(c.std(0)) * 1000
        mx = np.max(np.linalg.norm(c - c.mean(0), axis=1)) * 1000
        print(f"    {name:11s} spread={spread:5.2f}mm max={mx:5.2f}mm")
        if best is None or spread < best[0]:
            best = (spread, mx, name, Tx)
    spread, mx, name, Tx = best
    print(f"[*] best: {name}  spread={spread:.2f}mm  max={mx:.2f}mm")

    # 3) save refined intrinsics + T_EE_CAM
    json.dump({"fx": K[0,0], "fy": K[1,1], "cx": K[0,2], "cy": K[1,2],
               "dist": dist.reshape(-1).tolist()},
              open(os.path.join(capdir, "intrinsics.json"), "w"), indent=1)
    json.dump(Tx.tolist(), open(os.path.join(capdir, "T_EE_CAM.json"), "w"), indent=1)
    np.set_printoptions(suppress=True, precision=6)
    print("\n=== T_EE_CAM (refined) ===\n", Tx)
    rows = ", ".join("[" + ", ".join(f"{v:.6f}" for v in r) + "]" for r in Tx)
    print(f"\nT_EE_CAM = np.array([{rows}])")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "captures")
