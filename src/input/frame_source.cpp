#include "frame_source.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <iostream>

namespace fs = std::filesystem;

// ── ImageFolderSource ────────────────────────────────────────────────────────

ImageFolderSource::ImageFolderSource(const std::string& dir, bool loop)
    : loop_(loop) {
    for (const auto& entry : fs::directory_iterator(dir)) {
        const auto& p = entry.path();
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
            paths_.push_back(p.string());
    }
    std::sort(paths_.begin(), paths_.end());
    if (paths_.empty())
        throw std::runtime_error("No images found in: " + dir);
}

bool ImageFolderSource::read(cv::Mat& frame) {
    if (index_ >= paths_.size()) {
        if (!loop_) return false;
        index_ = 0;
    }
    frame = cv::imread(paths_[index_++]);
    return !frame.empty();
}

bool ImageFolderSource::isOpened() const {
    return !paths_.empty();
}

// ── VideoFileSource ──────────────────────────────────────────────────────────

VideoFileSource::VideoFileSource(const std::string& path, bool loop)
    : cap_(path), path_(path), loop_(loop) {}

bool VideoFileSource::read(cv::Mat& frame) {
    if (!cap_.isOpened()) return false;
    if (!cap_.read(frame) || frame.empty()) {
        if (!loop_) return false;
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        return cap_.read(frame) && !frame.empty();
    }
    return true;
}

bool VideoFileSource::isOpened() const {
    return cap_.isOpened();
}

// ── CameraSource (stub) ──────────────────────────────────────────────────────

CameraSource::CameraSource(const std::string&) {
    std::cerr << "[CameraSource] Stub — not implemented for PoC.\n"
              << "  Pass device index as string (\"0\") for USB camera.\n";
}

bool CameraSource::read(cv::Mat&) { return false; }
bool CameraSource::isOpened() const { return false; }
