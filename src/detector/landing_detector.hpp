#pragma once
#include "../config.hpp"
#include <opencv2/aruco.hpp>
#include <optional>
#include <vector>

class LandingDetector {
public:
    static constexpr int TARGET_ID = 0;

    explicit LandingDetector(const CameraConfig& config);

    // Returns corners of marker ID=0 in image coordinates, empty if not found.
    std::optional<std::vector<cv::Point2f>> detect(const cv::Mat& frame) const;

private:
    cv::aruco::ArucoDetector detector_;
};
