#include "test_runner.hpp"
#include "../src/input/frame_source.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static std::string write_tmp_image(const std::string& dir, const std::string& name) {
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(50, 100, 150));
    std::string path = dir + "/" + name;
    cv::imwrite(path, img);
    return path;
}

TEST(image_folder_reads_images, {
    const std::string dir = "/tmp/cv_landing_test_imgs";
    fs::create_directories(dir);
    write_tmp_image(dir, "001.png");
    write_tmp_image(dir, "002.png");

    ImageFolderSource src(dir, /*loop=*/false);
    ASSERT_TRUE(src.isOpened());

    cv::Mat frame;
    int count = 0;
    while (src.read(frame)) {
        ASSERT_TRUE(!frame.empty());
        ++count;
    }
    ASSERT_TRUE(count == 2);
    fs::remove_all(dir);
})

TEST(image_folder_loops, {
    const std::string dir = "/tmp/cv_landing_test_loop";
    fs::create_directories(dir);
    write_tmp_image(dir, "001.png");

    ImageFolderSource src(dir, /*loop=*/true);
    cv::Mat frame;
    for (int i = 0; i < 5; ++i)
        ASSERT_TRUE(src.read(frame));
    fs::remove_all(dir);
})

TEST(image_folder_throws_on_empty_dir, {
    const std::string dir = "/tmp/cv_landing_empty";
    fs::create_directories(dir);
    bool threw = false;
    try { ImageFolderSource src(dir); }
    catch (const std::exception&) { threw = true; }
    ASSERT_TRUE(threw);
    fs::remove_all(dir);
})

TEST(video_file_source_fails_gracefully, {
    VideoFileSource src("/nonexistent/video.mp4", /*loop=*/false);
    ASSERT_TRUE(!src.isOpened());
    cv::Mat frame;
    ASSERT_TRUE(!src.read(frame));
})

TEST(camera_source_stub_returns_false, {
    CameraSource src("");
    ASSERT_TRUE(!src.isOpened());
    cv::Mat frame;
    ASSERT_TRUE(!src.read(frame));
})
