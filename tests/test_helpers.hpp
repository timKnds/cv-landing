#pragma once
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>

inline cv::Mat make_marker_frame(int w = 1280, int h = 720, int marker_px = 400) {
    cv::Mat frame(h, w, CV_8UC3, cv::Scalar(128, 128, 128));
    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Mat marker;
    cv::aruco::generateImageMarker(dict, 0, marker_px, marker, 1);
    cv::cvtColor(marker, marker, cv::COLOR_GRAY2BGR);
    int x = (w - marker_px) / 2;
    int y = (h - marker_px) / 2;
    marker.copyTo(frame(cv::Rect(x, y, marker_px, marker_px)));
    return frame;
}
