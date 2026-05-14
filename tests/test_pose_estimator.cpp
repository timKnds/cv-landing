#include "test_runner.hpp"
#include "test_helpers.hpp"
#include "../src/config.hpp"
#include "../src/detector/landing_detector.hpp"
#include "../src/estimation/pose_estimator.hpp"
#include <cmath>

TEST(pose_detected_flag, {
    auto config = CameraConfig::load("config/camera.yaml");
    LandingDetector det(config);
    PoseEstimator   est(config);
    cv::Mat frame = make_marker_frame();
    auto corners = det.detect(frame);
    ASSERT_TRUE(corners.has_value());
    auto result = est.estimate(*corners, 0);
    ASSERT_TRUE(result.detected);
    ASSERT_TRUE(result.marker_id == 0);
})

TEST(pose_distance_positive, {
    auto config = CameraConfig::load("config/camera.yaml");
    LandingDetector det(config);
    PoseEstimator   est(config);
    cv::Mat frame = make_marker_frame();
    auto corners = det.detect(frame);
    ASSERT_TRUE(corners.has_value());
    auto result = est.estimate(*corners, 0);
    ASSERT_TRUE(result.distance_m > 0.0f);
})

TEST(pose_reprojection_error_low, {
    auto config = CameraConfig::load("config/camera.yaml");
    LandingDetector det(config);
    PoseEstimator   est(config);
    cv::Mat frame = make_marker_frame();
    auto corners = det.detect(frame);
    ASSERT_TRUE(corners.has_value());
    auto result = est.estimate(*corners, 0);
    // Synthetic image rendered without per-pixel distortion correction → some error expected
    ASSERT_TRUE(result.reprojection_error_px < 30.0f);
})

TEST(pose_center_near_image_center, {
    auto config = CameraConfig::load("config/camera.yaml");
    LandingDetector det(config);
    PoseEstimator   est(config);
    // 1280x720 matches camera config (cx=640, cy=360)
    cv::Mat frame = make_marker_frame(1280, 720, 400);
    auto corners = det.detect(frame);
    ASSERT_TRUE(corners.has_value());
    auto result = est.estimate(*corners, 0);
    ASSERT_NEAR(result.pixel_center.x, 640.0f, 20.0f);
    ASSERT_NEAR(result.pixel_center.y, 360.0f, 20.0f);
    ASSERT_NEAR(result.normalized_offset.x, 0.0f, 0.1f);
    ASSERT_NEAR(result.normalized_offset.y, 0.0f, 0.1f);
})

TEST(no_detection_result, {
    auto result = PoseEstimator::no_detection(12345ULL);
    ASSERT_TRUE(!result.detected);
    ASSERT_TRUE(result.marker_id == -1);
    ASSERT_TRUE(result.timestamp_us == 12345ULL);
})
