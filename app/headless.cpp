#include "../src/config.hpp"
#include "../src/detector/landing_detector.hpp"
#include "../src/estimation/pose_estimator.hpp"
#include "../src/input/frame_source.hpp"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>

// Outputs one JSON line per frame to stdout.
// Control process reads and parses the stream.
static void print_result(const DetectionResult& r) {
    if (r.detected) {
        std::printf(
            "{\"detected\":true,"
            "\"marker_id\":%d,"
            "\"pixel_center\":[%.2f,%.2f],"
            "\"normalized_offset\":[%.4f,%.4f],"
            "\"tvec\":[%.4f,%.4f,%.4f],"
            "\"distance_m\":%.4f,"
            "\"yaw_deg\":%.2f,"
            "\"pitch_deg\":%.2f,"
            "\"roll_deg\":%.2f,"
            "\"reprojection_error_px\":%.3f,"
            "\"timestamp_us\":%llu}\n",
            r.marker_id,
            r.pixel_center.x, r.pixel_center.y,
            r.normalized_offset.x, r.normalized_offset.y,
            r.tvec[0], r.tvec[1], r.tvec[2],
            r.distance_m,
            r.yaw_deg, r.pitch_deg, r.roll_deg,
            r.reprojection_error_px,
            static_cast<unsigned long long>(r.timestamp_us));
    } else {
        std::printf("{\"detected\":false,\"timestamp_us\":%llu}\n",
                    static_cast<unsigned long long>(r.timestamp_us));
    }
    std::fflush(stdout);
}

int main(int argc, char* argv[]) {
    const std::string config_path = (argc > 1) ? argv[1] : "config/camera.yaml";
    const std::string source_path = (argc > 2) ? argv[2] : "";

    CameraConfig config;
    try {
        config = CameraConfig::load(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    // In prod: replace with CameraSource("0") or appropriate device string
    std::unique_ptr<FrameSource> source;
    if (!source_path.empty()) {
        source = std::make_unique<VideoFileSource>(source_path, /*loop=*/false);
    } else {
        source = std::make_unique<CameraSource>("");
    }

    if (!source->isOpened()) {
        std::cerr << "Cannot open source. Provide a video path as second argument.\n";
        return 1;
    }

    LandingDetector detector(config);
    PoseEstimator   estimator(config);

    cv::Mat frame;
    while (source->read(frame)) {
        using clock = std::chrono::steady_clock;
        uint64_t ts = std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now().time_since_epoch()).count();

        auto corners = detector.detect(frame);
        DetectionResult result = corners
            ? estimator.estimate(*corners, ts)
            : PoseEstimator::no_detection(ts);

        print_result(result);
    }
    return 0;
}
