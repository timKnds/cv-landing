"""
Generates synthetic test fixture images with a known ArUco ID=0 marker.
Each image has controlled variation (angle, noise, lighting).
Exports ground_truth.json alongside the images.

Usage:
    uv run scripts/generate_test_fixtures.py
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import yaml

# ── Camera config ─────────────────────────────────────────────────────────────

def load_camera_config(path: str) -> dict:
    with open(path) as f:
        # Strip OpenCV YAML header if present
        content = f.read().replace("%YAML:1.0", "").replace("---", "")
    cfg = yaml.safe_load(content)
    camera_matrix = np.array([
        [cfg["fx"],       0, cfg["cx"]],
        [      0, cfg["fy"], cfg["cy"]],
        [      0,       0,        1   ]
    ], dtype=np.float32)
    dist_coeffs = np.array([cfg["k1"], cfg["k2"], cfg["p1"], cfg["p2"]], dtype=np.float32)
    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "marker_size_m": cfg["marker_size_m"],
    }


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_marker_at_pose(
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size_m: float,
    image_size: tuple[int, int] = (640, 480),
    bg_value: int = 128,
) -> np.ndarray:
    """
    Renders an ArUco ID=0 marker onto a plain background using projectPoints.
    Applies camera distortion to the final image.
    """
    w, h = image_size
    frame = np.full((h, w, 3), bg_value, dtype=np.uint8)

    # Generate marker image
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 200, None, 1)
    marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    # 3D corners of the marker (same convention as C++ pose_estimator)
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left (clockwise)
    # Camera frame: X=right, Y=down → top = negative Y, left = negative X
    s = marker_size_m / 2.0
    obj_pts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)

    # Project 3D corners to 2D
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    img_pts = img_pts.reshape(4, 2)

    # Warp marker texture onto projected quad
    src_corners = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners, img_pts)
    warped = cv2.warpPerspective(marker_img, M, (w, h))

    # Mask: only paint non-black warped pixels
    mask = cv2.warpPerspective(
        np.full((200, 200), 255, dtype=np.uint8), M, (w, h)
    )
    frame[mask > 0] = warped[mask > 0]

    # projectPoints already uses dist_coeffs → marker corners land on distorted positions.
    # No second remap needed — that would apply distortion twice and corrupt the pattern.
    return frame


def compute_ground_truth(
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
) -> dict:
    """Computes expected DetectionResult fields from known pose."""
    s = 0.0  # marker center at origin
    center_3d = np.array([[[s, s, s]]], dtype=np.float32)
    center_2d, _ = cv2.projectPoints(
        np.array([[0, 0, 0]], dtype=np.float32),
        rvec, tvec, camera_matrix, dist_coeffs
    )
    cx, cy = center_2d[0][0]
    w, h = image_size

    # Euler angles from rvec
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        yaw   = np.degrees(np.arctan2( R[1,0], R[0,0]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = np.degrees(np.arctan2( R[2,1], R[2,2]))
    else:
        yaw, pitch, roll = 0.0, np.degrees(np.arctan2(-R[2,0], sy)), \
                           np.degrees(np.arctan2(-R[1,2], R[1,1]))

    return {
        "pixel_center":      [float(cx), float(cy)],
        "normalized_offset": [float((cx - w/2) / (w/2)), float((cy - h/2) / (h/2))],
        "distance_m":        float(np.linalg.norm(tvec)),
        "yaw_deg":           float(yaw),
        "pitch_deg":         float(pitch),
        "roll_deg":          float(roll),
    }


# ── Variation generators ──────────────────────────────────────────────────────

def generate_fixtures(cfg: dict, out_dir: Path, image_size=(1280, 720)):
    out_dir.mkdir(parents=True, exist_ok=True)
    cm, dc, ms = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["marker_size_m"]
    ground_truth = []

    variations = []

    # Base distance, varying yaw
    for yaw_deg in [0, 15, 30, 45, -15, -30]:
        rvec = np.array([[0], [0], [np.radians(yaw_deg)]], dtype=np.float32)
        tvec = np.array([[0], [0], [2.0]], dtype=np.float32)
        variations.append((f"yaw_{yaw_deg:+03d}", rvec, tvec, 128, 0.0))

    # Varying distance
    for dist in [1.0, 1.5, 2.0, 3.0, 4.0]:
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.array([[0], [0], [dist]], dtype=np.float32)
        variations.append((f"dist_{dist:.1f}m", rvec, tvec, 128, 0.0))

    # Varying lateral offset
    for offset_m in [-0.3, -0.15, 0, 0.15, 0.3]:
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.array([[offset_m], [0], [2.0]], dtype=np.float32)
        variations.append((f"offset_x_{offset_m:+.2f}", rvec, tvec, 128, 0.0))

    # Noise variations
    for sigma in [5, 15, 30]:
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.array([[0], [0], [2.0]], dtype=np.float32)
        variations.append((f"noise_{sigma}", rvec, tvec, 128, sigma))

    # Brightness variations
    for brightness in [70, 100, 160, 200]:
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.array([[0], [0], [2.0]], dtype=np.float32)
        variations.append((f"bright_{brightness}", rvec, tvec, brightness, 0.0))

    for name, rvec, tvec, bg, noise_sigma in variations:
        frame = render_marker_at_pose(rvec, tvec, cm, dc, ms, image_size, bg)
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        filename = f"{name}.png"
        cv2.imwrite(str(out_dir / filename), frame)

        gt = compute_ground_truth(rvec, tvec, cm, dc, image_size)
        gt["filename"] = filename
        ground_truth.append(gt)

    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {len(variations)} fixture images → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/camera.yaml")
    parser.add_argument("--out",    default="data/synthetic/images")
    args = parser.parse_args()

    cfg = load_camera_config(args.config)
    generate_fixtures(cfg, Path(args.out))
