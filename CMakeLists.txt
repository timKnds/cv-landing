cmake_minimum_required(VERSION 3.20)
project(cv_landing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.7 REQUIRED)

# nlohmann/json for integration test ground truth parsing (header-only)
include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
    URL_HASH SHA256=a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d)
FetchContent_MakeAvailable(nlohmann_json)

# ── Core library ─────────────────────────────────────────────────────────────
add_library(cv_landing_core STATIC
    src/config.cpp
    src/detector/landing_detector.cpp
    src/estimation/pose_estimator.cpp
    src/input/frame_source.cpp
)
target_include_directories(cv_landing_core PUBLIC src)
target_link_libraries(cv_landing_core PUBLIC ${OpenCV_LIBS})

# ── Visualizer (dev only) ────────────────────────────────────────────────────
add_library(cv_landing_visualizer STATIC
    src/output/visualizer.cpp
)
target_include_directories(cv_landing_visualizer PUBLIC src)
target_compile_definitions(cv_landing_visualizer PUBLIC BUILD_VISUALIZER)
target_link_libraries(cv_landing_visualizer PUBLIC ${OpenCV_LIBS})

# ── Production binary — headless, no display ─────────────────────────────────
add_executable(headless app/headless.cpp)
target_link_libraries(headless PRIVATE cv_landing_core)

# ── Benchmark: performance (latency vs resolution) ───────────────────────────
add_executable(bench_perf benchmark/perf_main.cpp)
target_link_libraries(bench_perf PRIVATE cv_landing_core)

# ── Benchmark: robustness (detection rate under degradation) ─────────────────
add_executable(bench_robust benchmark/robustness_main.cpp)
target_link_libraries(bench_robust PRIVATE cv_landing_core)

# ── Benchmark: video (per-frame latency + pose on video file) ────────────────
add_executable(bench_video benchmark/video_main.cpp)
target_link_libraries(bench_video PRIVATE cv_landing_core)

# ── Tests ────────────────────────────────────────────────────────────────────
add_executable(tests tests/main.cpp)
target_include_directories(tests PRIVATE ${nlohmann_json_SOURCE_DIR}/include)
target_link_libraries(tests PRIVATE cv_landing_core)

enable_testing()
add_test(NAME all_tests COMMAND tests)
