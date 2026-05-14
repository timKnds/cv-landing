#pragma once
#ifdef BUILD_VISUALIZER
#include "../estimation/pose_estimator.hpp"
#include <opencv2/core.hpp>
#include <string>

class Visualizer {
public:
    explicit Visualizer(const std::string& window_name = "Landing Detector");

    // Draws detection overlay onto frame (in-place) and shows window.
    void show(cv::Mat& frame, const DetectionResult& result, double fps);

private:
    std::string window_name_;

    void draw_axes(cv::Mat& frame, const DetectionResult& r) const;
    void draw_info(cv::Mat& frame, const DetectionResult& r, double fps) const;
};
#endif
