"""
Eye-in-hand (D435i on UR5e flange) hand-eye calibration -> T_EE_CAM for reachability_filter.py.

Board: the project ChArUco calibration board.
       11x8 squares, square=15 mm, marker=11 mm, DICT_4X4_50, legacy pattern.

Inputs (produced by capture_calib.py or by hand):
  <capdir>/cap_000.png, cap_001.png, ...        color images of the FIXED board
  <capdir>/poses.json  = list of base->TCP poses [x,y,z, rx,ry,rz]  (m, axis-angle rad)
                         in the SAME order as the images (RTDE getActualTCPPose()).
  <capdir>/intrinsics.json = {"fx":..,"fy":..,"cx":..,"cy":..,"dist":[k1,k2,p1,p2,k3]}
                         from the RealSense COLOR stream (see capture_calib.py).

Run:
  python hand_eye_calibrate.py <capdir>
Output:
  R_cam2gripper, t_cam2gripper = camera pose w.r.t. the flange = T_EE_CAM,
  printed as a ready-to-paste 4x4 for reachability_filter.py (T_EE_CAM).
Need >= 8 varied poses (more is better; rotate/translate the wrist between shots).
"""
import sys, os, json, glob
import numpy as np
import cv2

SQX, SQY, SQ_M, MK_M = 11, 8, 0.015, 0.011   # ChArUco 11x8 squares, square 15mm, marker 11mm (200x150 page, DICT_4X4_50)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((SQX, SQY), SQ_M, MK_M, dictionary)
board.setLegacyPattern(True)   # board generated with the pre-4.6 (legacy) ChArUco layout
detector = cv2.aruco.CharucoDetector(board)

def board_pose_in_cam(img, K, dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return None
    obj, imgp = board.matchImagePoints(ch_corners, ch_ids)
    if obj is None or len(obj) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, dist)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)

def main(capdir):
    intr = json.load(open(os.path.join(capdir, "intrinsics.json")))
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    dist = np.array(intr.get("dist", [0, 0, 0, 0, 0]), float)
    poses = json.load(open(os.path.join(capdir, "poses.json")))
    imgs = sorted(glob.glob(os.path.join(capdir, "cap_*.png")))
    assert len(imgs) == len(poses), f"{len(imgs)} images vs {len(poses)} poses"

    R_g2b, t_g2b, R_t2c, t_t2c, used = [], [], [], [], 0
    for p, ip in zip(poses, imgs):
        img = cv2.imread(ip)
        bp = board_pose_in_cam(img, K, dist)
        if bp is None:
            print(f"  [skip] board not found: {os.path.basename(ip)}"); continue
        Rtc, ttc = bp
        Rgb, _ = cv2.Rodrigues(np.array(p[3:6], float))
        R_g2b.append(Rgb); t_g2b.append(np.array(p[:3], float))
        R_t2c.append(Rtc); t_t2c.append(ttc); used += 1
    print(f"[*] used {used}/{len(imgs)} views")
    assert used >= 6, "need >= 6 good views"

    R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                        method=cv2.CALIB_HAND_EYE_TSAI)
    T = np.eye(4); T[:3, :3] = R_c2g; T[:3, 3] = t_c2g.reshape(3)
    np.set_printoptions(suppress=True, precision=6)
    print("\n=== T_EE_CAM (camera w.r.t. flange) ===")
    print(T)
    print("\n--- paste into reachability_filter.py (replace the T_EE_CAM line) ---")
    rows = ", ".join("[" + ", ".join(f"{v:.6f}" for v in r) + "]" for r in T)
    print(f"T_EE_CAM = np.array([{rows}])")
    print("--- then set orient=True in analyze()/ik_reachable() for true 6-DOF V_reach ---")
    json.dump(T.tolist(), open(os.path.join(capdir, "T_EE_CAM.json"), "w"), indent=1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python hand_eye_calibrate.py <capture_dir>"); sys.exit(1)
    main(sys.argv[1])
