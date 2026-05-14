#include "../src/config.hpp"
#include "../src/detector/landing_detector.hpp"
#include "../src/estimation/pose_estimator.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

// Each subdirectory of benchmark_dir is one scenario.
// Images within are sorted by filename (param value encoded in name).
int main(int argc, char* argv[]) {
    const std::string config_path    = (argc > 1) ? argv[1] : "config/camera.yaml";
    const std::string benchmark_dir  = (argc > 2) ? argv[2] : "data/benchmark";
    const std::string results_dir    = (argc > 3) ? argv[3] : "results";

    CameraConfig config;
    try { config = CameraConfig::load(config_path); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }

    const std::vector<std::string> scenarios = {
        "yaw", "pitch", "brightness", "noise", "blur", "scale"
    };

    for (const auto& scenario : scenarios) {
        const std::string dir = benchmark_dir + "/" + scenario;
        if (!fs::exists(dir)) {
            std::cerr << "Skip missing scenario dir: " << dir << "\n";
            continue;
        }

        std::vector<std::string> image_paths;
        for (const auto& entry : fs::directory_iterator(dir)) {
            const auto ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png")
                image_paths.push_back(entry.path().string());
        }
        std::sort(image_paths.begin(), image_paths.end());

        if (image_paths.empty()) {
            std::cerr << "No images in: " << dir << "\n";
            continue;
        }

        LandingDetector detector(config);
        PoseEstimator   estimator(config);

        struct Row {
            std::string param;
            bool detected;
        };
        std::vector<Row> rows;

        for (const auto& img_path : image_paths) {
            cv::Mat frame = cv::imread(img_path);
            if (frame.empty()) continue;

            // param value encoded in filename stem before first '_'
            std::string stem = fs::path(img_path).stem().string();
            std::string param = stem.substr(0, stem.find('_'));

            auto corners = detector.detect(frame);
            bool detected = corners.has_value();
            if (detected) estimator.estimate(*corners, 0);

            rows.push_back({param, detected});
        }

        const std::string csv_path = results_dir + "/robustness_" + scenario + ".csv";
        std::FILE* csv = std::fopen(csv_path.c_str(), "w");
        if (!csv) { std::cerr << "Cannot write: " << csv_path << "\n"; continue; }
        std::fprintf(csv, "scenario,param,detected\n");

        std::printf("\n--- %s ---\n", scenario.c_str());
        for (const auto& r : rows) {
            std::fprintf(csv, "%s,%s,%d\n",
                         scenario.c_str(), r.param.c_str(), r.detected ? 1 : 0);
            std::printf("  param=%-8s  %s\n",
                        r.param.c_str(), r.detected ? "OK" : "FAIL");
        }
        std::fclose(csv);
        std::printf("Saved: %s\n", csv_path.c_str());
    }
    return 0;
}
