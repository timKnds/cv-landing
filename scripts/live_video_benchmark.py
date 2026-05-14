"""
Live video benchmark: plays synthetic video at target FPS while updating plots
independently via a matplotlib timer.

Architecture:
  reader thread   — reads C++ JSON stdout → queue
  consumer thread — drains queue throttled to --fps, accumulates data
  main thread     — matplotlib event loop; timer fires draw at --display-fps

Usage:
    uv run scripts/live_video_benchmark.py [--config CONFIG] [--binary BINARY]
                                            [--frames N] [--fps FPS] [--display-fps N]
"""

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


FPS_BUDGETS   = {"30 FPS": 1000/30, "20 FPS": 1000/20, "10 FPS": 1000/10}
BUDGET_COLORS = ["#2ecc71", "#e67e22", "#e74c3c"]
GT_COLOR      = "#1a3a5c"
PRED_COLOR    = "#e67e22"
LATENCY_COLOR = "steelblue"


def load_camera_config(path: str):
    with open(path) as f:
        content = f.read().replace("%YAML:1.0", "").replace("---", "")
    cfg  = yaml.safe_load(content)
    K    = np.array([[cfg["fx"], 0, cfg["cx"]], [0, cfg["fy"], cfg["cy"]], [0, 0, 1]],
                    dtype=np.float32)
    dist = np.array([cfg["k1"], cfg["k2"], cfg["p1"], cfg["p2"]], dtype=np.float32)
    return K, dist, float(cfg["marker_size_m"])


def make_marker_bgr() -> np.ndarray:
    d    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    gray = cv2.aruco.generateImageMarker(d, 0, 200, None, 1)
    bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bgr, (0, 0), (199, 199), (0, 100, 200), 8)
    return bgr


def render_frame(marker_bgr, K, dist, marker_size_m,
                 gt_x, gt_y, gt_z, gt_yaw, size=(1280, 720)):
    w, h  = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        v = int(60 + (row / h) * 80)
        frame[row] = (v, v + 10, v + 20)
    s   = marker_size_m / 2.0
    obj = np.array([[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]], dtype=np.float32)
    rv  = np.array([[0],[0],[np.radians(gt_yaw)]], dtype=np.float32)
    tv  = np.array([[gt_x],[gt_y],[gt_z]],         dtype=np.float32)
    pts, _ = cv2.projectPoints(obj, rv, tv, K, dist)
    pts    = pts.reshape(4, 2)
    src    = np.array([[0,0],[199,0],[199,199],[0,199]], dtype=np.float32)
    M      = cv2.getPerspectiveTransform(src, pts)
    warped = cv2.warpPerspective(marker_bgr, M, (w, h))
    mask   = cv2.warpPerspective(np.full((200,200), 255, np.uint8), M, (w, h))
    frame[mask > 0] = warped[mask > 0]
    return frame


def overlay_detection(frame, d, K, dist, marker_size_m):
    if d.get("detected"):
        s    = marker_size_m / 2.0
        apts = np.array([[0,0,0],[s,0,0],[0,s,0],[0,0,-s]], dtype=np.float32)
        rv   = np.zeros((3,1), dtype=np.float32)
        tv   = np.array([[d["pred_x"]],[d["pred_y"]],[d["pred_z"]]], dtype=np.float32)
        ip, _ = cv2.projectPoints(apts, rv, tv, K, dist)
        o  = tuple(ip[0].ravel().astype(int))
        cv2.line(frame, o, tuple(ip[1].ravel().astype(int)), (0,0,220), 2)
        cv2.line(frame, o, tuple(ip[2].ravel().astype(int)), (0,220,0), 2)
        cv2.line(frame, o, tuple(ip[3].ravel().astype(int)), (220,0,0), 2)
    col = (0,220,80) if d.get("detected") else (0,60,220)
    det = "DETECTED" if d.get("detected") else "LOST"
    cv2.putText(frame, f"{det}  lat={d['latency_ms']:.2f} ms",
                (12,36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    if d.get("detected"):
        cv2.putText(frame,
            f"pred: x={d['pred_x']:.2f}  y={d['pred_y']:.2f}  z={d['pred_z']:.2f} m",
            (12,68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 1)
        cv2.putText(frame,
            f"GT:   x={d['gt_x']:.2f}  y={d['gt_y']:.2f}  z={d['gt_z']:.2f} m",
            (12,96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160,200,240), 1)


def _reader(proc, q):
    for line in proc.stdout:
        line = line.strip()
        if line:
            try:
                q.put(json.loads(line))
            except json.JSONDecodeError:
                pass
    q.put(None)


def _consumer(q, data, lock, fps, display_fps, done_event):
    """Drains queue at target fps; throttles with time.sleep."""
    interval   = 1.0 / fps
    last_frame = None

    while not done_event.is_set():
        t0 = time.monotonic()
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break

        # Actual FPS: 1 / time since last frame was processed
        actual_fps = 1.0 / (t0 - last_frame) if last_frame is not None else 0.0
        last_frame = t0

        nan = float("nan")
        with lock:
            data["times"].append(item["time_s"])
            data["lats"].append(item["latency_ms"])
            data["actual_fps"].append(actual_fps)
            if item.get("detected"):
                data["gt_dist"].append(item["gt_dist"])
                data["pd_dist"].append(item.get("pred_dist", nan))
                data["gt_x"].append(item["gt_x"])
                data["pd_x"].append(item.get("pred_x", nan))
                data["gt_yaw"].append(item["gt_yaw"])
                data["pd_yaw"].append(item.get("pred_yaw", nan))
            else:
                for k in ("gt_dist","pd_dist","gt_x","pd_x","gt_yaw","pd_yaw"):
                    data[k].append(nan)
            data["latest"] = item

        elapsed = time.monotonic() - t0
        wait    = interval - elapsed - 0.005 * (display_fps / fps)
        if wait > 0:
            time.sleep(wait)

    done_event.set()


def run_live(config_path, binary, n_frames, fps, display_fps):
    K, dist, marker_size_m = load_camera_config(config_path)
    marker_bgr = make_marker_bgr()

    binary_path = Path(binary)
    if not binary_path.exists():
        sys.exit(f"Binary not found: {binary}\n"
                 "Build:  cmake -B build && cmake --build build --target bench_video")

    proc = subprocess.Popen(
        [str(binary_path), config_path, "results/video_bench.csv",
         str(n_frames), str(fps)],
        stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1,
    )

    q          = queue.Queue()
    data       = dict(times=[], lats=[], actual_fps=[], gt_dist=[], pd_dist=[],
                      gt_x=[], pd_x=[], gt_yaw=[], pd_yaw=[], latest=None)
    lock       = threading.Lock()
    done_event = threading.Event()

    threading.Thread(target=_reader,   args=(proc, q),                    daemon=True).start()
    threading.Thread(target=_consumer, args=(q, data, lock, fps, display_fps, done_event), daemon=True).start()

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                            width_ratios=[1.1, 1])

    ax_video = fig.add_subplot(gs[:, 0])
    ax_video.axis("off")
    ax_video.set_title("Synthetic Video — C++ detection overlay", fontsize=9)

    ax_lat  = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_x    = fig.add_subplot(gs[2, 1])
    ax_yaw  = fig.add_subplot(gs[3, 1])

    blank      = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_handle = ax_video.imshow(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))

    # Latency
    ax_lat.set_title("Latency (ms, log)", fontsize=8)
    ax_lat.set_xlabel("Time (s)", fontsize=7)
    ax_lat.set_yscale("log")
    ax_lat.grid(True, alpha=0.3, which="both")
    ax_lat.tick_params(labelsize=7)
    ax_lat.axhline(1000/30, linestyle="--", linewidth=1.1,
                   color=BUDGET_COLORS[0], zorder=3)
    lat_line, = ax_lat.plot([], [], color=LATENCY_COLOR, linewidth=0.9, zorder=2)

    # Series
    series = []
    for ax, title in [(ax_dist,"Distance (m)"), (ax_x,"X offset (m)"), (ax_yaw,"Yaw (°)")]:
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        gt_l,   = ax.plot([], [], color=GT_COLOR,   linewidth=1.2, label="GT")
        pred_l, = ax.plot([], [], color=PRED_COLOR,  linewidth=1.0,
                          linestyle="--", label="Pred")
        ax.legend(fontsize=6.5)
        series.append((ax, gt_l, pred_l))

    # ── Draw callback — runs on main thread via timer ─────────────────────────
    def draw():
        with lock:
            if data["latest"] is None:
                return
            times      = list(data["times"])
            lats       = list(data["lats"])
            actual_fps = list(data["actual_fps"])
            gt_dist    = list(data["gt_dist"])
            pd_dist    = list(data["pd_dist"])
            gt_x       = list(data["gt_x"])
            pd_x       = list(data["pd_x"])
            gt_yaw     = list(data["gt_yaw"])
            pd_yaw     = list(data["pd_yaw"])
            d          = data["latest"]

        frame = render_frame(marker_bgr, K, dist, marker_size_m,
                             d["gt_x"], d["gt_y"], d["gt_z"], d["gt_yaw"])
        overlay_detection(frame, d, K, dist, marker_size_m)
        img_handle.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        lat_line.set_data(times, lats)
        ax_lat.relim()
        ax_lat.autoscale_view()

        for (ax, gt_l, pred_l), gv, pv in zip(
            series,
            [gt_dist, gt_x, gt_yaw],
            [pd_dist, pd_x, pd_yaw],
        ):
            gt_l.set_data(times, gv)
            pred_l.set_data(times, pv)
            ax.relim()
            ax.autoscale_view()

        n_det   = sum(1 for v in pd_dist if not math.isnan(v))
        cur_fps = actual_fps[-1] if actual_fps else 0.0
        fps_ok  = abs(cur_fps - fps) / fps < 0.1 if cur_fps > 0 else False
        fps_str = f"{cur_fps:.1f}" + ("" if fps_ok else f" ⚠ target {fps:.0f}")
        fig.suptitle(
            f"Live Benchmark  |  {len(times)}/{n_frames} frames  |  "
            f"detected {n_det}  |  lat={d['latency_ms']:.2f} ms  |  FPS {fps_str}",
            fontsize=9,
        )
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=int(1000 / display_fps))
    timer.add_callback(draw)
    timer.start()

    print(f"Live benchmark  |  {n_frames} frames @ {fps:.0f} FPS  |  "
          f"plot {display_fps:.0f} FPS  |  close window to stop.", flush=True)

    plt.show()  # blocks on main thread; event loop drives timer

    done_event.set()
    proc.terminate()
    proc.wait()

    out_path = "results/plot_video_bench.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/camera.yaml")
    parser.add_argument("--binary",      default="build/bench_video")
    parser.add_argument("--duration",    type=float, default=10.0, help="Video length in seconds")
    parser.add_argument("--fps",         type=float, default=30.0)
    parser.add_argument("--display-fps", type=float, default=30.0)
    args     = parser.parse_args()
    n_frames = round(args.duration * args.fps)
    run_live(args.config, args.binary, n_frames, args.fps, args.display_fps)
