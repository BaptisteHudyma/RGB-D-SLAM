#ifndef LOAD_PARAMETERS_HPP
#define LOAD_PARAMETERS_HPP

#include <string>

namespace poseEstimation {

    class Parameters  {
        public:
            Parameters();
            bool init_from_file(const std::string& config_file_name);

            enum
            {
                etriangulation_policy_decreasing_matches = 1,
                etriangulation_policy_always_triangulate,
                etriangulation_policy_map_size
            };

        public: //setters
            void set_fx(float _fx) { fx = _fx; }
            void set_fy(float _fy) { fy = _fy; }
            void set_cx(float _cx) { cx = _cx; }
            void set_cy(float _cy) { cy = _cy; }
            void set_width(float w) { img_width = w; }
            void set_height(float h) { img_height = h; }

        public: //getters
            float get_fx() const { return fx; }
            float get_fy() const { return fy; }
            float get_cx() const { return cx; }
            float get_cy() const { return cy; }
            float get_baseline() const { return baseline; }

            long unsigned int get_height() const { return img_height; }
            long unsigned int get_width() const { return img_width; }

            float get_k1() const { return k1; }
            float get_k2() const { return k2; }
            float get_k3() const { return k3; }
            float get_p1() const { return p1; }
            float get_p2() const { return p2; }

            float get_triangulation_ratio_test_threshold() const { return triangulation_ratio_test_threshold; }
            float get_descriptor_matching_threshold() const { return descriptor_matching_threshold; }
            float get_far_plane_distance() const { return far_plane_distance; }
            float get_near_plane_distance() const { return near_plane_distance; }
            float get_tracking_ratio_test_threshold() const { return tracking_ratio_test_threshold; }

            int get_detection_cell_size() const { return detection_cell_size; }
            int get_agast_threshold() const { return agast_threshold; }
            long unsigned int get_max_keypoints_per_cell() const { return max_keypoints_per_cell; }
            int get_tracking_radius() const { return tracking_radius; }
            int get_staged_threshold() const { return staged_threshold; }
            int get_untracked_threshold() const { return untracked_threshold; }

            unsigned get_min_matches_for_tracking() const { return min_num_matches_for_tracking; }

            int get_camera_size() const { return viewer_camera_size; }
            int get_point_size() const { return viewer_point_size; }

            int get_triangulation_policy() const { return triangulation_policy; }


        private:
            // The following parameters must be specified.
            // Stereo is assumed to be undistorted and rectified.
            float fx, fy, cx, cy;
            float baseline;
            int img_width, img_height;
            float k1, k2, p1, p2, k3; // distortion, only in rgbd case

            // The rest are optional.
            float near_plane_distance, far_plane_distance; // viewable region
            float triangulation_ratio_test_threshold;      // cutoff for the ratio test performed when row matching for triangulating new map points
            float tracking_ratio_test_threshold;           // cutoff for the ratio test when matching during tracking
            float descriptor_matching_threshold;           // when matching for triangulation or tracking, if only one candidate feature is found for the traget feature, ratio test cannot be performed and thus the candidate is declared a match if the descriptors distance is smaller than this value.
            int min_num_matches_for_tracking;              // minimum required matched for succesful tracking, otherwise tracking is considered lost
            int tracking_radius;                           // the radius in pixels around each projected map point when looking for its detected image feature. Default 25px
            int detection_cell_size;                       // To enhance detected features distrubution, the image is devided into cells and features detetin is attempted in each. This is the side length of each cell.
            int max_keypoints_per_cell;                    // maximum number of features to detect in each cell.
            int agast_threshold;                           // Threshold for AGAST corner detector used.
            int untracked_threshold;                       // When a map point is not tracked in untracked_threshold number of frames then it is removed from the map.
            int staged_threshold;                          // newly triangulated map points are initially put into a staging phase, if they were successfully tracked for staged_threshold number of frames then they are declared good and added to the local map to be used for pose estimation. If staged_threshold is set to zero then this feature is effectively disabled.
            bool enable_logging;
            bool enable_visualization;
            int triangulation_policy;
            float viewer_camera_size;
            int viewer_point_size;
    };

};

#endif

