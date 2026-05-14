#include "test_runner.hpp"
#include "test_helpers.hpp"
#include "../src/config.hpp"
#include "../src/detector/landing_detector.hpp"
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>

static CameraConfig load_test_config() {
    return CameraConfig::load("config/camera.yaml");
}

TEST(detect_marker_id0, {
    auto config  = load_test_config();
    LandingDetector det(config);
    cv::Mat frame = make_marker_frame();
    auto result = det.detect(frame);
    ASSERT_TRUE(result.has_value());
})

TEST(ignore_other_ids, {
    auto config  = load_test_config();
    LandingDetector det(config);

    // Render marker ID=1 — should not be detected
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Mat marker;
    cv::aruco::generateImageMarker(dict, 1, 200, marker, 1);
    cv::cvtColor(marker, marker, cv::COLOR_GRAY2BGR);
    marker.copyTo(frame(cv::Rect(220, 140, 200, 200)));

    auto result = det.detect(frame);
    ASSERT_TRUE(!result.has_value());
})

TEST(no_detection_on_blank, {
    auto config = load_test_config();
    LandingDetector det(config);
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    auto result = det.detect(blank);
    ASSERT_TRUE(!result.has_value());
})

TEST(detect_on_fixture_images, {
    const std::string dir = "data/synthetic/images";
    if (!std::filesystem::exists(dir)) return;  // skip if not generated yet

    auto config = load_test_config();
    LandingDetector det(config);
    int total = 0; int detected = 0;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        cv::Mat frame = cv::imread(entry.path().string());
        if (frame.empty()) continue;
        ++total;
        if (det.detect(frame).has_value()) ++detected;
    }

    if (total > 0) {
        double rate = static_cast<double>(detected) / total;
        // Expect >90% detection on clean synthetic fixtures
        ASSERT_TRUE(rate > 0.90);
    }
})
