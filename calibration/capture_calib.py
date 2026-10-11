"""
Capture helper for hand-eye calibration: at each robot stop, grab a D435i COLOR frame and
the current base->TCP pose, save cap_XXX.png + append to poses.json. Also writes intrinsics.json.

Deps (install at the robot PC):  pip install pyrealsense2 ur-rtde
Run interactively: move the robot to a pose where the FIXED board is well in view, then press ENTER
to capture. Aim for >= 12 varied poses (vary approach angle + distance + in-plane rotation).

  python calibration/capture_calib.py --real-config configs/real.local.yaml --out captures
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from deployment.contracts import load_real_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real-config",
        type=Path,
        required=True,
        help="Ignored deploy-local config containing robot IP and pinned camera serial",
    )
    ap.add_argument("--out", default="captures")
    args = ap.parse_args()
    real_cfg, _ = load_real_config(args.real_config)
    robot_ip = str(real_cfg["robot"]["ip"])
    camera_serial = str(real_cfg["camera"]["serial"])
    os.makedirs(args.out, exist_ok=True)

    import pyrealsense2 as rs
    from rtde_receive import RTDEReceiveInterface

    # list connected devices so it's obvious which cameras are attached
    ctx = rs.context()
    devs = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name)
            for d in ctx.query_devices()}
    print("[*] connected RealSense devices:")
    for sn, name in devs.items():
        print(f"      {sn}  {name}{'  <-- using this' if sn == camera_serial else ''}")
    if camera_serial not in devs:
        raise SystemExit(f"[!] configured serial {camera_serial} not found. Attached: {list(devs)}")

    rtde = RTDEReceiveInterface(robot_ip)
    pipe = rs.pipeline(); cfg = rs.config()
    cfg.enable_device(camera_serial)   # pin the EEF camera (two cameras attached)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    json.dump({"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
               "dist": list(intr.coeffs)}, open(os.path.join(args.out, "intrinsics.json"), "w"), indent=1)
    print(f"[*] intrinsics saved (fx={intr.fx:.1f} cx={intr.ppx:.1f})")

    import cv2
    poses = []
    i = 0
    print("Move robot so the board is fully visible, then ENTER to capture (q + ENTER to finish).")
    try:
        while True:
            frames = pipe.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            cv2.imshow("live (focus terminal, ENTER=capture)", img); cv2.waitKey(1)
            s = input(f"[{i}] ENTER=capture / q=quit: ")
            if s.strip().lower() == "q":
                break
            tcp = rtde.getActualTCPPose()   # [x,y,z, rx,ry,rz] base->TCP (m, axis-angle)
            cv2.imwrite(os.path.join(args.out, f"cap_{i:03d}.png"), img)
            poses.append([float(v) for v in tcp])
            json.dump(poses, open(os.path.join(args.out, "poses.json"), "w"), indent=1)
            print(f"    saved cap_{i:03d}.png  tcp={np.round(tcp,3).tolist()}")
            i += 1
    finally:
        pipe.stop()
        disconnect = getattr(rtde, "disconnect", None)
        if callable(disconnect):
            disconnect()
    print(f"[*] {i} captures in {args.out}. Now: python hand_eye_calibrate.py {args.out}")

if __name__ == "__main__":
    main()
