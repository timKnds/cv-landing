#pragma once
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <string>

struct CameraConfig {
    cv::Mat camera_matrix;   // 3x3 float
    cv::Mat dist_coeffs;     // 1x4 float [k1, k2, p1, p2]
    float   marker_size_m;
    cv::aruco::PredefinedDictionaryType dictionary;
    cv::aruco::CornerRefineMethod corner_refinement;

    static CameraConfig load(const std::string& path);
};
