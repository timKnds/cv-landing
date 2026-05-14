#include "pose_estimator.hpp"
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <cmath>

static uint64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

PoseEstimator::PoseEstimator(const CameraConfig& config) : config_(config) {
    const float h = config.marker_size_m / 2.0f;
    // ArUco corner order: top-left, top-right, bottom-right, bottom-left (clockwise)
    // Camera frame: X=right, Y=down → top = negative Y, left = negative X
    obj_pts_ = {
        {-h, -h, 0.0f},
        { h, -h, 0.0f},
        { h,  h, 0.0f},
        {-h,  h, 0.0f}
    };

    // Derive image size from principal point (cx, cy are roughly half of w, h)
    float cx = config.camera_matrix.at<float>(0, 2);
    float cy = config.camera_matrix.at<float>(1, 2);
    image_size_ = cv::Size(static_cast<int>(cx * 2), static_cast<int>(cy * 2));
}

DetectionResult PoseEstimator::estimate(const std::vector<cv::Point2f>& corners,
                                         uint64_t timestamp_us) const {
    cv::Mat rvec_mat, tvec_mat;
    cv::solvePnP(obj_pts_, corners,
                 config_.camera_matrix, config_.dist_coeffs,
                 rvec_mat, tvec_mat);

    cv::Vec3f tvec(tvec_mat.at<double>(0), tvec_mat.at<double>(1), tvec_mat.at<double>(2));
    cv::Vec3f rvec_v(rvec_mat.at<double>(0), rvec_mat.at<double>(1), rvec_mat.at<double>(2));

    // Pixel center = mean of four corners
    cv::Point2f center(0, 0);
    for (const auto& c : corners) center += c;
    center *= 0.25f;

    const float half_w = image_size_.width  / 2.0f;
    const float half_h = image_size_.height / 2.0f;
    cv::Point2f norm_offset((center.x - half_w) / half_w,
                            (center.y - half_h) / half_h);

    // Euler angles from rotation matrix (ZYX convention)
    cv::Mat R;
    cv::Rodrigues(rvec_mat, R);
    double sy = std::sqrt(R.at<double>(0,0)*R.at<double>(0,0) +
                          R.at<double>(1,0)*R.at<double>(1,0));
    double roll, pitch, yaw;
    if (sy > 1e-6) {
        roll  = std::atan2( R.at<double>(2,1), R.at<double>(2,2));
        pitch = std::atan2(-R.at<double>(2,0), sy);
        yaw   = std::atan2( R.at<double>(1,0), R.at<double>(0,0));
    } else {
        roll  = std::atan2(-R.at<double>(1,2), R.at<double>(1,1));
        pitch = std::atan2(-R.at<double>(2,0), sy);
        yaw   = 0.0;
    }
    constexpr double RAD2DEG = 180.0 / M_PI;

    DetectionResult result;
    result.detected              = true;
    result.pixel_center          = center;
    result.normalized_offset     = norm_offset;
    result.tvec                  = tvec;
    result.rvec                  = rvec_v;
    result.distance_m            = static_cast<float>(cv::norm(tvec_mat));
    result.yaw_deg               = static_cast<float>(yaw   * RAD2DEG);
    result.pitch_deg             = static_cast<float>(pitch * RAD2DEG);
    result.roll_deg              = static_cast<float>(roll  * RAD2DEG);
    result.reprojection_error_px = compute_reprojection_error(corners, rvec_mat, tvec_mat);
    result.marker_id             = 0;
    result.timestamp_us          = timestamp_us;
    return result;
}

DetectionResult PoseEstimator::no_detection(uint64_t timestamp_us) {
    DetectionResult result{};
    result.detected      = false;
    result.marker_id     = -1;
    result.timestamp_us  = timestamp_us;
    return result;
}

float PoseEstimator::compute_reprojection_error(const std::vector<cv::Point2f>& corners,
                                                 const cv::Mat& rvec,
                                                 const cv::Mat& tvec) const {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(obj_pts_, rvec, tvec,
                      config_.camera_matrix, config_.dist_coeffs, projected);
    float total = 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        cv::Point2f d = projected[i] - corners[i];
        total += std::sqrt(d.x*d.x + d.y*d.y);
    }
    return total / 4.0f;
}
