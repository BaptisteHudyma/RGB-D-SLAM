#include "Parameters.hpp"
#include <opencv2/opencv.hpp>

namespace rgbd_slam {
namespace poseEstimation {

    Parameters::Parameters() {
        _fx = _fy = _cx = _cy = 0.5;
        _img_width = _img_height = 0.0;
        _k1 = _k2 = _p1 = _p2 = _k3 = 0.f;
        _baseline = 0.f;
        _near_plane_distance = 0.1;
        _far_plane_distance = 500.0;
        _triangulation_ratio_test_threshold = 0.60;
        _tracking_ratio_test_threshold = 0.80;
        _descriptor_matching_threshold = 30.0;
        _min_num_matches_for_tracking = 10;
        _tracking_radius = 25;
        _agast_threshold = 25;
        _untracked_threshold = 10;
        _staged_threshold = 2;
        _detection_cell_size = 250;
        _max_keypoints_per_cell = 150;
        _triangulation_policy = TriangulationPolicies::etriangulation_policy_decreasing_matches;
        _enable_logging = true;
        _enable_visualization = false;
        _viewer_camera_size = 0.6;
        _viewer_point_size = 5;
    }


    bool Parameters::init_from_file(const std::string& configFileName) {
        cv::FileStorage config_file(configFileName, cv::FileStorage::READ);
        if (!config_file.isOpened()) {
            return false;
        }

        _fx = config_file["fx"];
        _fy = config_file["fy"];
        _cx = config_file["cx"];
        _cy = config_file["cy"];
        _k1 = config_file["k1"];
        _k2 = config_file["k2"];
        _p1 = config_file["p1"];
        _p2 = config_file["p2"];
        _k3 = config_file["k3"];
        _baseline = config_file["baseline"];
        _img_width = (int)config_file["img_width"];
        _img_height = (int)config_file["img_height"];
        _near_plane_distance = config_file["near_plane_distance"];
        _far_plane_distance = config_file["far_plane_distance"];
        _triangulation_ratio_test_threshold = config_file["triangulation_ratio_test_threshold"];
        _tracking_ratio_test_threshold = config_file["tracking_ratio_test_threshold"];
        _min_num_matches_for_tracking = (int)config_file["min_num_matches_for_tracking"];
        _tracking_radius = (int)config_file["tracking_radius"];
        _agast_threshold = (int)config_file["agast_threshold"];
        _untracked_threshold = (int)config_file["untracked_threshold"];
        _staged_threshold = (int)config_file["staged_threshold"];
        _descriptor_matching_threshold = config_file["descriptor_matching_threshold"];
        _detection_cell_size = (int)config_file["detection_cell_size"];
        _max_keypoints_per_cell = (int)config_file["max_keypoints_per_cell"];
        _enable_logging = (int)config_file["enable_logging"];
        _enable_visualization = (int)config_file["enable_visualization"];
        _triangulation_policy = config_file["triangulation_policy"];
        _viewer_camera_size = config_file["viewer_camera_size"];
        _viewer_point_size = (int)config_file["viewer_point_size"];
        config_file.release();
        return true;
    }

}
}
