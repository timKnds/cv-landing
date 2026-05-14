"""
Generates data/synthetic/demo.mp4 — synthetic approach-trajectory video.

Trajectory mirrors gt_pose() in benchmark/video_main.cpp:
  x =  0.4 * sin(t*2π) * (1-t)   damped lateral drift
  y =  0.2 * cos(t*3π) * (1-t)
  z =  6.0 - t*5.5                descend 6 m → 0.5 m
  yaw= 20.0 * cos(t*4π) * (1-t)

Usage:
    uv run scripts/generate_demo_video.py
    uv run scripts/generate_demo_video.py --frames 600 --fps 60
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import yaml


def load_camera_config(path: str) -> dict:
    with open(path) as f:
        content = f.read().replace("%YAML:1.0", "").replace("---", "")
    cfg = yaml.safe_load(content)
    return {
        "camera_matrix": np.array([
            [cfg["fx"], 0, cfg["cx"]],
            [0, cfg["fy"], cfg["cy"]],
            [0, 0, 1],
        ], dtype=np.float32),
        "dist_coeffs":   np.array([cfg["k1"], cfg["k2"], cfg["p1"], cfg["p2"]], dtype=np.float32),
        "marker_size_m": cfg["marker_size_m"],
    }


def generate_video_samples(
    cfg: dict,
    out_path: Path,
    n_frames: int = 300,
    fps: float = 30.0,
    image_size: tuple[int, int] = (1280, 720),
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm, dc, ms = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["marker_size_m"]
    w, h = image_size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_gray = cv2.aruco.generateImageMarker(aruco_dict, 0, 200, None, 1)
    marker_bgr  = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(marker_bgr, (0, 0), (199, 199), (0, 100, 200), 8)

    s        = ms / 2.0
    obj_pts  = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)
    src_pts  = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], dtype=np.float32)
    mask_src = np.full((200, 200), 255, dtype=np.uint8)

    gradient = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        v = int(60 + (row / h) * 80)
        gradient[row] = (v, v + 10, v + 20)

    for i in range(n_frames):
        t      = i / max(n_frames - 1, 1)
        gt_x   =  0.4 * np.sin(t * np.pi * 2) * (1 - t)
        gt_y   =  0.2 * np.cos(t * np.pi * 3) * (1 - t)
        gt_z   =  6.0 - t * 5.5
        gt_yaw = 20.0 * np.cos(t * np.pi * 4) * (1 - t)

        rvec = np.array([[0], [0], [np.radians(gt_yaw)]], dtype=np.float32)
        tvec = np.array([[gt_x], [gt_y], [gt_z]], dtype=np.float32)

        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cm, dc)
        img_pts    = img_pts.reshape(4, 2)

        M      = cv2.getPerspectiveTransform(src_pts, img_pts)
        warped = cv2.warpPerspective(marker_bgr, M, (w, h))
        mask   = cv2.warpPerspective(mask_src, M, (w, h))

        frame = gradient.copy()
        frame[mask > 0] = warped[mask > 0]

        cv2.putText(frame,
                    f"t={t:.2f}  z={gt_z:.2f} m  yaw={gt_yaw:.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

        writer.write(frame)

    writer.release()
    print(f"Generated {n_frames}-frame video @ {fps:.0f} FPS → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/camera.yaml")
    parser.add_argument("--out",    default="data/synthetic/demo.mp4")
    parser.add_argument("--frames", type=int,   default=300)
    parser.add_argument("--fps",    type=float, default=30.0)
    args = parser.parse_args()

    cfg = load_camera_config(args.config)
    generate_video_samples(cfg, Path(args.out), args.frames, args.fps)
