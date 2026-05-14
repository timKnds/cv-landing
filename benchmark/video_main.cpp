#include "../src/config.hpp"
#include "../src/detector/landing_detector.hpp"
#include "../src/estimation/pose_estimator.hpp"
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
using PerfClock = std::chrono::high_resolution_clock;

static double elapsed_ms(PerfClock::time_point a, PerfClock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Synthetic approach trajectory — mirrors generate_demo_video.py
struct Pose { float x, y, z, yaw_deg; };
static Pose gt_pose(int i, int n) {
    const double t = static_cast<double>(i) / std::max(n - 1, 1);
    return {
        static_cast<float>(0.4 * std::sin(t * M_PI * 2) * (1 - t)),
        static_cast<float>(0.2 * std::cos(t * M_PI * 3) * (1 - t)),
        static_cast<float>(6.0 - t * 5.5),
        static_cast<float>(20.0 * std::cos(t * M_PI * 4) * (1 - t)),
    };
}

static cv::Mat render_frame(
    const cv::Mat& marker_bgr,
    const CameraConfig& cfg,
    const Pose& p,
    cv::Size img_size)
{
    cv::Mat frame(img_size, CV_8UC3);
    for (int row = 0; row < img_size.height; ++row) {
        uint8_t v = static_cast<uint8_t>(60 + (row * 80.0 / img_size.height));
        frame.row(row).setTo(cv::Scalar(v, v + 10, v + 20));
    }

    const float s = cfg.marker_size_m / 2.0f;
    std::vector<cv::Point3f> obj_pts = {
        {-s, -s, 0}, {s, -s, 0}, {s, s, 0}, {-s, s, 0}
    };
    cv::Mat rvec = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, p.yaw_deg * static_cast<float>(M_PI) / 180.0f);
    cv::Mat tvec = (cv::Mat_<float>(3, 1) << p.x, p.y, p.z);

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj_pts, rvec, tvec, cfg.camera_matrix, cfg.dist_coeffs, img_pts);

    std::vector<cv::Point2f> src = {{0, 0}, {199, 0}, {199, 199}, {0, 199}};
    cv::Mat M = cv::getPerspectiveTransform(src, img_pts);
    cv::Mat warped, mask_src(200, 200, CV_8UC1, cv::Scalar(255)), mask;
    cv::warpPerspective(marker_bgr, warped, M, img_size);
    cv::warpPerspective(mask_src, mask, M, img_size);
    warped.copyTo(frame, mask);
    return frame;
}

int main(int argc, char* argv[]) {
    const std::string config_path = (argc > 1) ? argv[1] : "config/camera.yaml";
    const std::string output_csv  = (argc > 2) ? argv[2] : "results/video_bench.csv";
    const int         n_frames    = (argc > 3) ? std::atoi(argv[3]) : 300;
    const double      fps         = (argc > 4) ? std::atof(argv[4]) : 30.0;
    constexpr int     WARMUP      = 5;

    CameraConfig config;
    try { config = CameraConfig::load(config_path); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }

    const cv::Size img_size(1280, 720);

    // Pre-render ArUco marker image
    auto aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Mat marker_gray, marker_bgr;
    cv::aruco::generateImageMarker(aruco_dict, 0, 200, marker_gray, 1);
    cv::cvtColor(marker_gray, marker_bgr, cv::COLOR_GRAY2BGR);
    cv::rectangle(marker_bgr, {0, 0}, {199, 199}, {0, 100, 200}, 8);

    LandingDetector detector(config);
    PoseEstimator   estimator(config);

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        auto frame = render_frame(marker_bgr, config, gt_pose(i, n_frames), img_size);
        auto corners = detector.detect(frame);
        if (corners) estimator.estimate(*corners, 0);
    }

    fs::create_directories(fs::path(output_csv).parent_path());
    FILE* csv = std::fopen(output_csv.c_str(), "w");
    if (!csv) { std::cerr << "Cannot write: " << output_csv << "\n"; return 1; }
    std::fprintf(csv,
        "frame,time_s,latency_ms,detected,"
        "gt_x,gt_y,gt_z,gt_dist,gt_yaw,"
        "pred_x,pred_y,pred_z,pred_dist,pred_yaw,"
        "reprojection_error_px\n");

    int detected_n = 0;
    double lat_sum = 0, lat_min = 1e9, lat_max = 0;

    for (int i = 0; i < n_frames; ++i) {
        const Pose p = gt_pose(i, n_frames);
        const float gt_dist = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        const double time_s = i / fps;

        auto frame = render_frame(marker_bgr, config, p, img_size);

        auto t0      = PerfClock::now();
        auto corners = detector.detect(frame);
        DetectionResult r = corners
            ? estimator.estimate(*corners, 0)
            : PoseEstimator::no_detection(0);
        auto t1 = PerfClock::now();

        const double lat = elapsed_ms(t0, t1);
        lat_sum += lat;
        if (lat < lat_min) lat_min = lat;
        if (lat > lat_max) lat_max = lat;

        if (r.detected) {
            ++detected_n;
            std::fprintf(csv,
                "%d,%.4f,%.4f,1,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                i, time_s, lat,
                p.x, p.y, p.z, gt_dist, p.yaw_deg,
                r.tvec[0], r.tvec[1], r.tvec[2], r.distance_m, r.yaw_deg,
                r.reprojection_error_px);
            std::printf(
                "{\"frame\":%d,\"time_s\":%.4f,\"latency_ms\":%.4f,\"detected\":true,"
                "\"gt_x\":%.4f,\"gt_y\":%.4f,\"gt_z\":%.4f,\"gt_dist\":%.4f,\"gt_yaw\":%.4f,"
                "\"pred_x\":%.4f,\"pred_y\":%.4f,\"pred_z\":%.4f,"
                "\"pred_dist\":%.4f,\"pred_yaw\":%.4f,\"reproj_err\":%.4f}\n",
                i, time_s, lat,
                p.x, p.y, p.z, gt_dist, p.yaw_deg,
                r.tvec[0], r.tvec[1], r.tvec[2], r.distance_m, r.yaw_deg,
                r.reprojection_error_px);
        } else {
            std::fprintf(csv,
                "%d,%.4f,%.4f,0,%.4f,%.4f,%.4f,%.4f,%.4f,,,,,,\n",
                i, time_s, lat,
                p.x, p.y, p.z, gt_dist, p.yaw_deg);
            std::printf(
                "{\"frame\":%d,\"time_s\":%.4f,\"latency_ms\":%.4f,\"detected\":false,"
                "\"gt_x\":%.4f,\"gt_y\":%.4f,\"gt_z\":%.4f,\"gt_dist\":%.4f,\"gt_yaw\":%.4f}\n",
                i, time_s, lat,
                p.x, p.y, p.z, gt_dist, p.yaw_deg);
        }
        std::fflush(stdout);
    }

    std::fclose(csv);
    std::fprintf(stderr, "Saved: %s\n", output_csv.c_str());
    std::fprintf(stderr, "Frames:   %d  @  %.0f FPS\n", n_frames, fps);
    std::fprintf(stderr, "Detected: %d  (%.1f%%)\n", detected_n, 100.0 * detected_n / n_frames);
    std::fprintf(stderr, "Latency:  min=%.2f ms  max=%.2f ms  mean=%.2f ms\n",
                lat_min, lat_max, lat_sum / n_frames);
    return 0;
}
