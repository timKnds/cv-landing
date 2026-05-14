# Vision-Based Landing Target Detection

ArUco-based precision landing target detection for rocket hoppers.
Runs on Jetson Orin Nano. Outputs 6DoF pose to a downstream control process.

## Why ArUco?
- Training an own model (CNN etc.) requires huge sample size for each scenario (blur, brightness, noise, pitch, scale, yaw).
- For each output feature we need target value and loss function
- ArUco is availabe in C++ and insanely fast -> directly implementation with no data preperation overhead.
- The dection of markers allows us to also detect rotation around the normal vector, with a H-sign oder a circle to rotation detection is limited.

## Architecture

```
Camera → LandingDetector → PoseEstimator → DetectionResult (JSON stdout)
                                                ↓
                                        Control Process
```

- **LandingDetector** src/detector: ArUco `DICT_4X4_50`, ID=0, OpenCV `ArucoDetector` and grey-scaling on input images.
    - Choosen 4x4 as smallest tag, because we doesn't use the IDs, even 4x4 offers 50 IDs. Potentially switch these IDs to avoid distraction from other tags.
- **PoseEstimator** src/estimation: Calculates the geometric targets from the four previous calculated corners.
- **Output:** one JSON line per frame with `detected: true/false` always present

### Technical description
- Detector uses `AurucoDetector` and return coordinates of corners, if marker found
- Estimator input these four corners, `obj_pts_` estimated form marker size and the given distortion coefficients and return `tvec` and `rvec`.
- `tvec` can directly be used to calculate `distance_m`.
- `rvec` can't be interpreted effectively so it's transformed to a rotation matrix. From these matrix yaw, pitch and roll can be calculated via ZYX convention.

Test here how `rvec` is transformed to yaw, pitch and roll:
[![Rodrigues → Euler interactive demo](https://raw.githubusercontent.com/timKnds/cv-landing/main/docs/preview.png)](https://timknds.github.io/cv-landing/rvec_to_euler_interactive.html)

## Build

**Dependencies:** CMake ≥ 3.20, OpenCV ≥ 4.7 (with contrib/aruco)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Targets:**

| Target         | Description                             | Deploy?   |
|----------------|-----------------------------------------|-----------|
| `headless`     | Prod binary — JSON stdout, no display   | Jetson    |
| `bench_perf`   | Latency vs resolution benchmark         | Dev only  |
| `bench_robust` | Detection rate under degradation        | Dev only  |
| `bench_video`  | Per-frame latency + pose on video file  | Dev only  |
| `tests`        | Full test suite                         | Dev only  |


## Generate Data (Python)

```bash
uv sync
uv run scripts/generate_test_fixtures.py          # data/synthetic/images/
uv run scripts/generate_demo_video.py             # data/synthetic/demo.mp4
uv run scripts/generate_benchmark_data.py --scenario all  # data/benchmark/
```

`generate_demo_video.py` renders a 300-frame synthetic approach trajectory (z: 6 m → 0.5 m, damped lateral/yaw oscillation) matching the trajectory in `benchmark/video_main.cpp`. Used as input for `./build/headless` and the live video benchmark.

## Run

**Headless (prod template):**
```bash
./build/headless config/camera.yaml data/synthetic/demo.mp4
# Each frame → one JSON line to stdout
```

**For real deployment:** replace video path with `CameraSource` in `app/headless.cpp` (`CameraSource` not implemented yet):
```cpp
source = std::make_unique<CameraSource>("0");  // device index for USB camera
```
**TODO:** Replace `camera.yaml` with calibration output from `cv::calibrateCamera()`.

## Benchmark

Evaluating detection and estimation pipeline on **speed**, **robustness** and **correctness**.
### robustness
Execute pipeline on synthetic samples in `data/benchmark` (`./build/bench_robust`). Test different settings for parameters like blur, brightness, noise, pitch, resolution, scale and yaw. Visualize results via Python in `result/plot_robustness_<scenario>`.
### correctness
Compare predicted with real values. `scripts/live_video_benchmark` shows a live generated random video with predictions and the corresponding real values. Plot generated in `results/plot_video_bench.png``
### speed
`scripts/live_video_benchmark` also shows the latency of the complete process. Use `./build/bench_perf` to predict on different resolutions form `data/benchmark/resolution`. Plot via `scripts/plot_benchmarks.py`. Compares the performance of different resolutions on the speed relativ to the upper bound depending on the FPS. E.g. 30 FPS => 1s / 30 = 33,33 ms per frame.<br> **TODO:** currently executed on dev system, test on target system.

## Tests

```bash
./build/tests
```


## Output Format

```json
detected: 
{
    "detected":true,  # marker found?
    "marker_id":0,  # detected id, everytime zero
    "pixel_center":[640.0,360.0],  # pixels of the marker center 
    "normalized_offset":[0.0,0.0],  # offset from image center
    "tvec":[0.0,0.0,2.0],  # vector from marker to camera position
    "distance_m":2.0,  # ||tvec|| is distance_mv
    "yaw_deg":0.0,  # rotation around normal vector (°)
    "pitch_deg":0.0,  # forward/backward rotation (°)
    "roll_deg":0.0,  # left/right rotation (°)
    "reprojection_error_px":0.5,  # quality indicator
    "timestamp_us":123456  # steady_clock since system boot
}
```
`reprojection_error_px` gives the mean pixel distance between predicted and detected corners. 
```json
not detected:
{
    "detected":false, 
    "timestamp_us":123789
}
```

## Config

`config/camera.yaml`
Swap without recompile for different hardware.

### Parameters

```yaml
fx: 820.0  fy: 820.0  # Focal length 
cx: 640.0  cy: 360.0  # Image center
k1: -0.35  k2:  0.13  # Radial distortion coefficients
p1: 0.0    p2: 0.0  # Tangential distortion
marker_size_m: 0.5  # real marker size
aruco_dictionary: DICT_4X4_50  # Marker
corner_refinement: CORNER_REFINE_SUBPIX  # Suppixel accuracy
```
