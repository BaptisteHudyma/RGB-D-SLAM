#ifndef LOAD_PARAMETERS_HPP
#define LOAD_PARAMETERS_HPP

#include <string>

namespace poseEstimation {

    class Parameters  {
        public:
            Parameters();
            bool init_from_file(const std::string& config_file_name);

            enum TriangulationPolicies
            {
                etriangulation_policy_decreasing_matches = 1,
                etriangulation_policy_always_triangulate,
                etriangulation_policy_map_size
            };

        public: //setters
            void set_fx(float fx) { _fx = fx; }
            void set_fy(float fy) { _fy = fy; }
            void set_cx(float cx) { _cx = cx; }
            void set_cy(float cy) { _cy = cy; }
            void set_width(float w) { _img_width = w; }
            void set_height(float h) { _img_height = h; }

        public: //getters
            float get_fx() const { return _fx; }
            float get_fy() const { return _fy; }
            float get_cx() const { return _cx; }
            float get_cy() const { return _cy; }
            float get_baseline() const { return _baseline; }

            long unsigned int get_height() const { return _img_height; }
            long unsigned int get_width() const { return _img_width; }

            float get_k1() const { return _k1; }
            float get_k2() const { return _k2; }
            float get_k3() const { return _k3; }
            float get_p1() const { return _p1; }
            float get_p2() const { return _p2; }

            float get_triangulation_ratio_test_threshold() const { return _triangulation_ratio_test_threshold; }
            float get_descriptor_matching_threshold() const { return _descriptor_matching_threshold; }
            float get_far_plane_distance() const { return _far_plane_distance; }
            float get_near_plane_distance() const { return _near_plane_distance; }
            float get_tracking_ratio_test_threshold() const { return _tracking_ratio_test_threshold; }

            unsigned int get_detection_cell_size() const { return _detection_cell_size; }
            unsigned int get_agast_threshold() const { return _agast_threshold; }
            long unsigned int get_max_keypoints_per_cell() const { return _max_keypoints_per_cell; }
            unsigned int get_tracking_radius() const { return _tracking_radius; }
            unsigned int get_staged_threshold() const { return _staged_threshold; }
            unsigned int get_untracked_threshold() const { return _untracked_threshold; }

            unsigned get_min_matches_for_tracking() const { return _min_num_matches_for_tracking; }

            unsigned int get_camera_size() const { return _viewer_camera_size; }
            unsigned int get_point_size() const { return _viewer_point_size; }

            unsigned int get_triangulation_policy() const { return _triangulation_policy; }


        private:
            // The following parameters must be specified.
            // Stereo is assumed to be undistorted and rectified.
            float _fx, _fy, _cx, _cy;
            float _baseline;
            long unsigned int _img_width;
            long unsigned int _img_height;
            float _k1, _k2, _p1, _p2, _k3; // distortion, only in rgbd case

            // The rest are optional.
            float _near_plane_distance, _far_plane_distance; // viewable region
            float _triangulation_ratio_test_threshold;      // cutoff for the ratio test performed when row matching for triangulating new map points
            float _tracking_ratio_test_threshold;           // cutoff for the ratio test when matching during tracking
            float _descriptor_matching_threshold;           // when matching for triangulation or tracking, if only one candidate feature is found for the traget feature, ratio test cannot be performed and thus the candidate is declared a match if the descriptors distance is smaller than this value.
            unsigned int _min_num_matches_for_tracking;              // minimum required matched for succesful tracking, otherwise tracking is considered lost
            unsigned int _tracking_radius;                           // the radius in pixels around each projected map point when looking for its detected image feature. Default 25px
            unsigned int _detection_cell_size;                       // To enhance detected features distrubution, the image is devided into cells and features detetin is attempted in each. This is the side length of each cell.
            unsigned int _max_keypoints_per_cell;                    // maximum number of features to detect in each cell.
            unsigned int _agast_threshold;                           // Threshold for AGAST corner detector used.
            unsigned int _untracked_threshold;                       // When a map point is not tracked in untracked_threshold number of frames then it is removed from the map.
            unsigned int _staged_threshold;                          // newly triangulated map points are initially put into a staging phase, if they were successfully tracked for staged_threshold number of frames then they are declared good and added to the local map to be used for pose estimation. If staged_threshold is set to zero then this feature is effectively disabled.
            bool _enable_logging;
            bool _enable_visualization;
            int _triangulation_policy;
            float _viewer_camera_size;
            unsigned int _viewer_point_size;
    };

}

#endif

