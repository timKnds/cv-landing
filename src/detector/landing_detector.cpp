#include "landing_detector.hpp"
#include <opencv2/imgproc.hpp>

LandingDetector::LandingDetector(const CameraConfig& config)
    : detector_(
        cv::aruco::getPredefinedDictionary(config.dictionary),
        [&] {
            cv::aruco::DetectorParameters params;
            params.cornerRefinementMethod = config.corner_refinement;
            return params;
        }()) {}

std::optional<std::vector<cv::Point2f>> LandingDetector::detect(const cv::Mat& frame) const {
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    std::vector<int> ids;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    detector_.detectMarkers(gray, corners, ids, rejected);

    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == TARGET_ID)
            return corners[i];
    }
    return std::nullopt;
}
