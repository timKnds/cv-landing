#include "config.hpp"
#include <opencv2/core/persistence.hpp>
#include <stdexcept>

static cv::aruco::PredefinedDictionaryType parse_dictionary(const std::string& s) {
    if (s == "DICT_4X4_50")  return cv::aruco::DICT_4X4_50;
    if (s == "DICT_5X5_100") return cv::aruco::DICT_5X5_100;
    if (s == "DICT_6X6_250") return cv::aruco::DICT_6X6_250;
    throw std::runtime_error("Unknown aruco_dictionary: " + s);
}

static cv::aruco::CornerRefineMethod parse_refinement(const std::string& s) {
    if (s == "CORNER_REFINE_NONE")    return cv::aruco::CORNER_REFINE_NONE;
    if (s == "CORNER_REFINE_SUBPIX")  return cv::aruco::CORNER_REFINE_SUBPIX;
    if (s == "CORNER_REFINE_CONTOUR") return cv::aruco::CORNER_REFINE_CONTOUR;
    throw std::runtime_error("Unknown corner_refinement: " + s);
}

CameraConfig CameraConfig::load(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("Cannot open camera config: " + path);

    CameraConfig cfg;

    float fx, fy, cx, cy, k1, k2, p1, p2;
    fs["fx"] >> fx; fs["fy"] >> fy;
    fs["cx"] >> cx; fs["cy"] >> cy;
    fs["k1"] >> k1; fs["k2"] >> k2;
    fs["p1"] >> p1; fs["p2"] >> p2;
    fs["marker_size_m"] >> cfg.marker_size_m;

    cfg.camera_matrix = (cv::Mat_<float>(3, 3) <<
        fx,  0, cx,
         0, fy, cy,
         0,  0,  1);

    cfg.dist_coeffs = (cv::Mat_<float>(1, 4) << k1, k2, p1, p2);

    std::string dict_str, refine_str;
    fs["aruco_dictionary"]  >> dict_str;
    fs["corner_refinement"] >> refine_str;
    cfg.dictionary        = parse_dictionary(dict_str);
    cfg.corner_refinement = parse_refinement(refine_str);

    return cfg;
}
