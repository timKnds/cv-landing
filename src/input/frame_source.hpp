#pragma once
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual bool read(cv::Mat& frame) = 0;
    virtual bool isOpened() const = 0;
};

// Reads images from a directory in sorted filename order. Loops by default.
class ImageFolderSource : public FrameSource {
public:
    explicit ImageFolderSource(const std::string& dir, bool loop = false);
    bool read(cv::Mat& frame) override;
    bool isOpened() const override;

private:
    std::vector<std::string> paths_;
    size_t index_ = 0;
    bool   loop_;
};

// Reads frames from a video file. Loops by default.
class VideoFileSource : public FrameSource {
public:
    explicit VideoFileSource(const std::string& path, bool loop = true);
    bool read(cv::Mat& frame) override;
    bool isOpened() const override;

private:
    cv::VideoCapture cap_;
    std::string      path_;
    bool             loop_;
};

// Production camera source. Stub: not implemented for PoC.
// Pass device index as string ("0") for USB camera.
class CameraSource : public FrameSource {
public:
    explicit CameraSource(const std::string& source);
    bool read(cv::Mat& frame) override;
    bool isOpened() const override;
};
