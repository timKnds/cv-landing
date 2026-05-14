#include "test_runner.hpp"

// Test file includes (each registers its TEST() cases at static init time)
#include "test_detector.cpp"
#include "test_pose_estimator.cpp"
#include "test_frame_source.cpp"
#include "test_integration.cpp"

int main() {
    return run_all_tests();
}
