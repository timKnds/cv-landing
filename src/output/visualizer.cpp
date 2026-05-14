#ifdef BUILD_VISUALIZER
#include "visualizer.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

Visualizer::Visualizer(const std::string& window_name)
    : window_name_(window_name) {
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
}

void Visualizer::show(cv::Mat& frame, const DetectionResult& result, double fps) {
    if (result.detected)
        draw_axes(frame, result);
    draw_info(frame, result, fps);
    cv::imshow(window_name_, frame);
}

void Visualizer::draw_axes(cv::Mat& frame, const DetectionResult& r) const {
    // Draw coordinate axes at marker center
    const float axis_len = 0.1f;  // 10 cm
    std::vector<cv::Point3f> axis_pts = {
        {0,        0,        0},
        {axis_len, 0,        0},
        {0,        axis_len, 0},
        {0,        0,        axis_len}
    };
    std::vector<cv::Point2f> projected;
    cv::Mat rvec_mat(3, 1, CV_32F, const_cast<float*>(r.rvec.val));
    cv::Mat tvec_mat(3, 1, CV_32F, const_cast<float*>(r.tvec.val));

    // drawFrameAxes needs camera intrinsics — use identity as placeholder.
    // In practice demo.cpp should pass config to Visualizer for proper axes.
    cv::Mat cam = (cv::Mat_<float>(3,3) << 820,0,frame.cols/2.0f, 0,820,frame.rows/2.0f, 0,0,1);
    cv::Mat dist = cv::Mat::zeros(4, 1, CV_32F);
    cv::drawFrameAxes(frame, cam, dist, rvec_mat, tvec_mat, axis_len);

    // Marker center cross
    cv::drawMarker(frame, r.pixel_center, {0, 255, 0},
                   cv::MARKER_CROSS, 20, 2);
}

void Visualizer::draw_info(cv::Mat& frame, const DetectionResult& r, double fps) const {
    const int x = 10, dy = 22;
    int y = 30;
    const cv::Scalar white(255, 255, 255);
    const cv::Scalar green(0, 255, 0);
    const cv::Scalar red(0, 0, 255);
    const double font_scale = 0.6;

    auto put = [&](const std::string& text, const cv::Scalar& col = {255,255,255}) {
        cv::putText(frame, text, {x, y}, cv::FONT_HERSHEY_SIMPLEX, font_scale, col, 1);
        y += dy;
    };

    put(cv::format("FPS: %.1f", fps));

    if (r.detected) {
        put(cv::format("Center: (%.0f, %.0f)", r.pixel_center.x, r.pixel_center.y), green);
        put(cv::format("Offset: (%.2f, %.2f)", r.normalized_offset.x, r.normalized_offset.y), green);
        put(cv::format("Dist:  %.2f m", r.distance_m), green);
        put(cv::format("Yaw:   %.1f deg", r.yaw_deg), green);
        put(cv::format("Pitch: %.1f deg", r.pitch_deg), green);
        put(cv::format("Repr: %.2f px", r.reprojection_error_px), green);
    } else {
        put("NO TARGET", red);
    }
}
#endif
