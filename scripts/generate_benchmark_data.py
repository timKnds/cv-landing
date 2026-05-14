"""
Generates benchmark data sets for robustness and performance testing.

Usage:
    uv run scripts/generate_benchmark_data.py --scenario all
    uv run scripts/generate_benchmark_data.py --scenario yaw
"""

import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path
import yaml


def clear_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


def load_camera_config(path: str) -> dict:
    with open(path) as f:
        content = f.read().replace("%YAML:1.0", "").replace("---", "")
    cfg = yaml.safe_load(content)
    return {
        "camera_matrix": np.array([
            [cfg["fx"], 0, cfg["cx"]],
            [0, cfg["fy"], cfg["cy"]],
            [0, 0, 1]
        ], dtype=np.float32),
        "dist_coeffs":   np.array([cfg["k1"], cfg["k2"], cfg["p1"], cfg["p2"]], dtype=np.float32),
        "marker_size_m": cfg["marker_size_m"],
    }


def render_base(cm, dc, ms, rvec, tvec, image_size=(1280, 720), bg=128) -> np.ndarray:
    w, h = image_size
    frame = np.full((h, w, 3), bg, dtype=np.uint8)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 200, None, 1)
    marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    s = ms / 2.0
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left
    obj_pts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cm, dc)
    img_pts = img_pts.reshape(4, 2)

    M = cv2.getPerspectiveTransform(
        np.array([[0,0],[199,0],[199,199],[0,199]], dtype=np.float32), img_pts)
    warped = cv2.warpPerspective(marker_img, M, (w, h))
    mask   = cv2.warpPerspective(np.full((200,200),255,np.uint8), M, (w, h))
    frame[mask > 0] = warped[mask > 0]

    # projectPoints already applies dist_coeffs — no second remap needed
    return frame


def apply_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, kernel: int) -> np.ndarray:
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# ── Scenario generators ───────────────────────────────────────────────────────

def gen_resolution(cfg, out_dir: Path, pool_size: int = 30):
    """
    Pool of native frames per resolution with slight pose jitter + sensor noise.
    Filenames: {w}x{h}_{i:03d}.png — loaded as round-robin in the C++ benchmark.
    """
    d = out_dir / "resolution"
    clear_dir(d)
    cm_base, dc, ms = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["marker_size_m"]
    base_w, base_h = 1280, 720
    rng = np.random.default_rng(42)

    for w, h in [(320, 240), (640, 480), (1280, 720), (1920, 1080)]:
        sx, sy = w / base_w, h / base_h
        cm = cm_base * np.array([[sx, 1, sx], [1, sy, sy], [1, 1, 1]], dtype=np.float32)
        for i in range(pool_size):
            # small pose jitter: ±3° rotation, ±5% translation in x/y, ±2% in z
            rvec = rng.uniform(-np.radians(3), np.radians(3), (3, 1)).astype(np.float32)
            tx   = rng.uniform(-0.05, 0.05)
            ty   = rng.uniform(-0.05, 0.05)
            tz   = 2.0 * rng.uniform(0.98, 1.02)
            tvec = np.array([[tx], [ty], [tz]], dtype=np.float32)
            frame = render_base(cm, dc, ms, rvec, tvec, (w, h))
            # sensor noise σ=5
            noise = rng.normal(0, 5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(d / f"{w}x{h}_{i:03d}.png"), frame)
    print(f"  resolution: 4 resolutions × {pool_size} frames = {4 * pool_size} total")


def gen_yaw(cfg, out_dir: Path):
    d = out_dir / "yaw"
    clear_dir(d)
    cm, dc, ms = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["marker_size_m"]
    angles = list(range(0, 85, 5))
    for deg in angles:
        rvec = np.array([[0],[0],[np.radians(deg)]], dtype=np.float32)
        tvec = np.array([[0],[0],[2.0]], dtype=np.float32)
        frame = render_base(cm, dc, ms, rvec, tvec)
        cv2.imwrite(str(d / f"{deg:03d}_deg.png"), frame)
    print(f"  yaw: {len(angles)} frames")


def gen_pitch(cfg, out_dir: Path):
    d = out_dir / "pitch"
    clear_dir(d)
    cm, dc, ms = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["marker_size_m"]
    angles = list(range(0, 85, 5)) + [86, 87, 88]
    for deg in angles:
        rvec = np.array([[np.radians(deg)],[0],[0]], dtype=np.float32)
        tvec = np.array([[0],[0],[2.0]], dtype=np.float32)
        frame = render_base(cm, dc, ms, rvec, tvec)
        cv2.imwrite(str(d / f"{deg:03d}_deg.png"), frame)
    print(f"  pitch: {len(angles)} frames")


def gen_brightness(cfg, out_dir: Path):
    d = out_dir / "brightness"
    clear_dir(d)
    cm, dc, ms = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["marker_size_m"]
    base_rvec = np.zeros((3,1), dtype=np.float32)
    base_tvec = np.array([[0],[0],[2.0]], dtype=np.float32)
    base = render_base(cm, dc, ms, base_rvec, base_tvec)
    pcts = [-95, -85, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 85, 95]
    for pct in pcts:
        factor = 1.0 + pct / 100.0
        frame = apply_brightness(base, factor)
        label = f"{pct:+04d}pct"
        cv2.imwrite(str(d / f"{label}.png"), frame)
    print(f"  brightness: {len(pcts)} frames")


def gen_noise(cfg, out_dir: Path):
    d = out_dir / "noise"
    clear_dir(d)
    cm, dc, ms = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["marker_size_m"]
    base_rvec = np.zeros((3,1), dtype=np.float32)
    base_tvec = np.array([[0],[0],[2.0]], dtype=np.float32)
    base = render_base(cm, dc, ms, base_rvec, base_tvec)
    sigmas = [0, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150]
    for sigma in sigmas:
        frame = apply_noise(base, sigma) if sigma > 0 else base.copy()
        cv2.imwrite(str(d / f"{sigma:03d}_sigma.png"), frame)
    print(f"  noise: {len(sigmas)} frames")


def gen_blur(cfg, out_dir: Path):
    d = out_dir / "blur"
    clear_dir(d)
    cm, dc, ms = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["marker_size_m"]
    base_rvec = np.zeros((3,1), dtype=np.float32)
    base_tvec = np.array([[0],[0],[2.0]], dtype=np.float32)
    base = render_base(cm, dc, ms, base_rvec, base_tvec)
    kernels = [1, 3, 5, 7, 11, 15, 21, 31, 51, 71, 101]
    for k in kernels:
        frame = apply_blur(base, k) if k > 1 else base.copy()
        cv2.imwrite(str(d / f"{k:03d}_kernel.png"), frame)
    print(f"  blur: {len(kernels)} frames")


def gen_scale(cfg, out_dir: Path):
    """Marker at increasing z-distance (simulates altitude / small apparent size)."""
    d = out_dir / "scale"
    clear_dir(d)
    cm, dc, ms = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["marker_size_m"]
    base_rvec = np.zeros((3,1), dtype=np.float32)
    # z in metres; larger z → smaller marker in frame
    # zero-pad to 5 chars (e.g. "001p0") so filenames sort correctly
    z_values = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0]
    for z in z_values:
        tvec = np.array([[0],[0],[z]], dtype=np.float32)
        frame = render_base(cm, dc, ms, base_rvec, tvec)
        label = f"{z:05.1f}".replace(".", "p")
        cv2.imwrite(str(d / f"{label}_m.png"), frame)
    print(f"  scale: {len(z_values)} frames")


SCENARIOS = {
    "resolution": gen_resolution,
    "yaw":        gen_yaw,
    "pitch":      gen_pitch,
    "brightness": gen_brightness,
    "noise":      gen_noise,
    "blur":       gen_blur,
    "scale":      gen_scale,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config/camera.yaml")
    parser.add_argument("--out",      default="data/benchmark")
    parser.add_argument("--scenario", default="all",
                        choices=list(SCENARIOS.keys()) + ["all"])
    args = parser.parse_args()

    cfg = load_camera_config(args.config)
    out = Path(args.out)

    targets = SCENARIOS.items() if args.scenario == "all" \
              else [(args.scenario, SCENARIOS[args.scenario])]

    for name, fn in targets:
        print(f"Generating {name}...")
        fn(cfg, out)

    print("Done.")
