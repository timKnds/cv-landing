#include "test_runner.hpp"
#include "../src/config.hpp"
#include "../src/detector/landing_detector.hpp"
#include "../src/estimation/pose_estimator.hpp"
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

// Integration test: runs the full pipeline on pre-generated fixtures
// and checks DetectionResult against ground truth exported by generate_test_fixtures.py.
//
// Ground truth file: data/synthetic/images/ground_truth.json
// Format: [{"filename": "...", "pixel_center": [cx, cy], "distance_m": d,
//            "yaw_deg": y, "normalized_offset": [dx, dy]}, ...]

TEST(integration_full_pipeline, {
    const std::string gt_path  = "data/synthetic/images/ground_truth.json";
    if (!std::filesystem::exists(gt_path)) return;  // skip until data generated

    std::ifstream ifs(gt_path);
    nlohmann::json gt = nlohmann::json::parse(ifs);

    auto config = CameraConfig::load("config/camera.yaml");
    LandingDetector detector(config);
    PoseEstimator   estimator(config);

    int total = 0; int passed = 0;
    for (const auto& entry : gt) {
        std::string img_path = "data/synthetic/images/" +
                               entry["filename"].get<std::string>();
        cv::Mat frame = cv::imread(img_path);
        if (frame.empty()) continue;
        ++total;

        auto corners = detector.detect(frame);
        if (!corners) continue;  // detection miss — counted as fail

        auto result = estimator.estimate(*corners, 0);

        float exp_cx  = entry["pixel_center"][0];
        float exp_cy  = entry["pixel_center"][1];
        float exp_dist = entry["distance_m"];
        float exp_yaw  = entry["yaw_deg"];

        bool ok = std::abs(result.pixel_center.x - exp_cx)  < 15.0f &&
                  std::abs(result.pixel_center.y - exp_cy)  < 15.0f &&
                  std::abs(result.distance_m     - exp_dist) < 0.5f  &&
                  std::abs(result.yaw_deg        - exp_yaw)  < 10.0f;

        if (ok) ++passed;
    }

    if (total > 0) {
        double rate = static_cast<double>(passed) / total;
        ASSERT_TRUE(rate >= 0.85);
    }
})
