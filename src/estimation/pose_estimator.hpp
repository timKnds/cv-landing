#pragma once
#include "../config.hpp"
#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

struct DetectionResult {
    bool detected;

    // Image coordinates
    cv::Point2f pixel_center;
    cv::Point2f normalized_offset;  // [-1, 1] from image center

    // 3D pose in camera frame
    cv::Vec3f tvec;                 // x=right, y=down, z=along optical axis [m]
    cv::Vec3f rvec;                 // Rodrigues rotation vector
    float     distance_m;           // ||tvec|| — Euclidean distance to marker plane [m]

    // Marker orientation (derived from rvec)
    float yaw_deg;                  // rotation around marker normal [°]
    float pitch_deg;                // forward/back tilt [°]
    float roll_deg;                 // lateral tilt [°]

    // Quality
    float reprojection_error_px;    // mean corner reprojection error — low = reliable

    // Metadata
    int      marker_id;
    uint64_t timestamp_us;          // steady_clock microseconds
};

class PoseEstimator {
public:
    explicit PoseEstimator(const CameraConfig& config);

    DetectionResult estimate(const std::vector<cv::Point2f>& corners,
                             uint64_t timestamp_us) const;
    static DetectionResult no_detection(uint64_t timestamp_us);

private:
    CameraConfig               config_;
    std::vector<cv::Point3f>   obj_pts_;  // 3D marker corners in marker frame
    cv::Size                   image_size_;

    float compute_reprojection_error(const std::vector<cv::Point2f>& corners,
                                     const cv::Mat& rvec,
                                     const cv::Mat& tvec) const;
};
