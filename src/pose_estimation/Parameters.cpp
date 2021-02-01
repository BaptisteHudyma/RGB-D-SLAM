#include "Parameters.hpp"
#include <opencv2/opencv.hpp>

using namespace poseEstimation;

Parameters::Parameters() {
    fx = fy = cx = cy = 0.5;
    img_width = img_height = 0.0;
    k1 = k2 = p1 = p2 = k3 = 0.f;
    baseline = 0.f;
    near_plane_distance = 0.1;
    far_plane_distance = 500.0;
    triangulation_ratio_test_threshold = 0.60;
    tracking_ratio_test_threshold = 0.80;
    descriptor_matching_threshold = 30.0;
    min_num_matches_for_tracking = 10;
    tracking_radius = 25;
    agast_threshold = 25;
    untracked_threshold = 10;
    staged_threshold = 2;
    detection_cell_size = 250;
    max_keypoints_per_cell = 150;
    triangulation_policy = etriangulation_policy_decreasing_matches;
    enable_logging = true;
    enable_visualization = false;
    viewer_camera_size = 0.6;
    viewer_point_size = 5;
}


bool Parameters::init_from_file(const char* configFileName) {
    cv::FileStorage config_file(configFileName, cv::FileStorage::READ);
    if (!config_file.isOpened()) {
        return false;
    }

    this->fx = config_file["fx"];
    this->fy = config_file["fy"];
    this->cx = config_file["cx"];
    this->cy = config_file["cy"];
    this->k1 = config_file["k1"];
    this->k2 = config_file["k2"];
    this->p1 = config_file["p1"];
    this->p2 = config_file["p2"];
    this->k3 = config_file["k3"];
    this->baseline = config_file["baseline"];
    this->img_width = config_file["img_width"];
    this->img_height = config_file["img_height"];
    this->near_plane_distance = config_file["near_plane_distance"];
    this->far_plane_distance = config_file["far_plane_distance"];
    this->triangulation_ratio_test_threshold = config_file["triangulation_ratio_test_threshold"];
    this->tracking_ratio_test_threshold = config_file["tracking_ratio_test_threshold"];
    this->min_num_matches_for_tracking = config_file["min_num_matches_for_tracking"];
    this->tracking_radius = config_file["tracking_radius"];
    this->agast_threshold = config_file["agast_threshold"];
    this->untracked_threshold = config_file["untracked_threshold"];
    this->staged_threshold = config_file["staged_threshold"];
    this->descriptor_matching_threshold = config_file["descriptor_matching_threshold"];
    this->detection_cell_size = config_file["detection_cell_size"];
    this->max_keypoints_per_cell = config_file["max_keypoints_per_cell"];
    this->enable_logging = (int)config_file["enable_logging"];
    this->enable_visualization = (int)config_file["enable_visualization"];
    this->triangulation_policy = config_file["triangulation_policy"];
    this->viewer_camera_size = config_file["viewer_camera_size"];
    this->viewer_point_size = config_file["viewer_point_size"];
    config_file.release();
    return true;
}
