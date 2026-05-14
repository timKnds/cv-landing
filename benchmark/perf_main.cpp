#include "../src/config.hpp"
#include "../src/detector/landing_detector.hpp"
#include "../src/estimation/pose_estimator.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace fs = std::filesystem;
using PerfClock = std::chrono::high_resolution_clock;

static double elapsed_ms(PerfClock::time_point a, PerfClock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Resolution { int w, h; };

static std::vector<cv::Mat> load_pool(const std::string& dir, int w, int h) {
    std::string prefix = std::to_string(w) + "x" + std::to_string(h) + "_";
    std::vector<cv::Mat> pool;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0)
            pool.push_back(cv::imread(entry.path().string()));
    }
    std::sort(pool.begin(), pool.end(), [](const cv::Mat&, const cv::Mat&) { return false; });
    return pool;
}

int main(int argc, char* argv[]) {
    const std::string config_path  = (argc > 1) ? argv[1] : "config/camera.yaml";
    const std::string fixtures_dir = (argc > 2) ? argv[2] : "data/benchmark/resolution";
    const std::string output_csv   = (argc > 3) ? argv[3] : "results/perf_resolution.csv";
    constexpr int N       = 200;
    constexpr int WARMUP  = 20;

    CameraConfig config;
    try { config = CameraConfig::load(config_path); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }

    const std::vector<Resolution> resolutions = {
        {320, 240}, {640, 480}, {1280, 720}, {1920, 1080}
    };

    std::FILE* csv = std::fopen(output_csv.c_str(), "w");
    if (!csv) { std::cerr << "Cannot write: " << output_csv << "\n"; return 1; }
    std::fprintf(csv, "width,height,min_ms,max_ms,mean_ms,std_ms,detection_rate\n");

    for (const auto& res : resolutions) {
        auto pool = load_pool(fixtures_dir, res.w, res.h);
        if (pool.empty()) {
            std::cerr << "No pool frames for " << res.w << "x" << res.h << "\n";
            continue;
        }
        std::printf("  %dx%d  pool=%zu\n", res.w, res.h, pool.size());

        LandingDetector detector(config);
        PoseEstimator   estimator(config);
        int pool_sz = static_cast<int>(pool.size());

        // warmup — discard results
        for (int i = 0; i < WARMUP; ++i) {
            auto corners = detector.detect(pool[i % pool_sz]);
            if (corners) estimator.estimate(*corners, 0);
        }

        std::vector<double> times;
        times.reserve(N);
        int detected = 0;

        for (int i = 0; i < N; ++i) {
            const cv::Mat& frame = pool[i % pool_sz];
            auto t0 = PerfClock::now();
            auto corners = detector.detect(frame);
            if (corners) {
                ++detected;
                estimator.estimate(*corners, 0);
            }
            auto t1 = PerfClock::now();
            times.push_back(elapsed_ms(t0, t1));
        }

        double min_t  = *std::min_element(times.begin(), times.end());
        double max_t  = *std::max_element(times.begin(), times.end());
        double mean_t = std::accumulate(times.begin(), times.end(), 0.0) / N;
        double sq_sum = std::accumulate(times.begin(), times.end(), 0.0,
            [mean_t](double acc, double t) { return acc + (t - mean_t) * (t - mean_t); });
        double std_t    = std::sqrt(sq_sum / N);
        double det_rate = static_cast<double>(detected) / N;

        std::fprintf(csv, "%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                     res.w, res.h, min_t, max_t, mean_t, std_t, det_rate);
        std::printf("  %4dx%-4d  min=%.2fms  max=%.2fms  mean=%.2fms  std=%.2fms  det=%.1f%%\n",
                    res.w, res.h, min_t, max_t, mean_t, std_t, det_rate * 100.0);
    }

    std::fclose(csv);
    std::printf("Saved: %s\n", output_csv.c_str());
    return 0;
}
