#ifndef IMAGE_FEATURES_STRUCT_HPP
#define IMAGE_FEATURES_STRUCT_HPP

#include "Pose.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace poseEstimation {

    class Image_Features_Struct
    {
        public:
            Image_Features_Struct();
            void init(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descs,
                    int tracking_radius, int hashing_cell_size, int vertical_search_radius, float triangulation_ratio_th,
                    float tracking_ratio_th, float desc_dist_th, std::vector<float> *kps_depth = nullptr);

            int find_match_index(const vector2 &pt, const cv::Mat &desc, float *d1, float *d2) const; 
            int row_match(const cv::Point2f &pt, const cv::Mat &desc) const;

            void mark_as_matched(const int idx, const bool val) { _matched_marks[idx] = val; }
            bool is_matched(const int idx) const { return _matched_marks[idx]; }
            void reset_matched_marks() { _matched_marks = std::vector<bool>(_keypoints.size(), false); }


            bool is_depth_associated() const { return !_kps_depths.empty(); }


        public: //getters
            int get_tracking_radius() const { return _tracking_radius; }
            unsigned int get_features_count() const { return static_cast<unsigned int>(_keypoints.size()); }
            float get_keypoint_depth(const int idx) const { return _kps_depths[idx]; }
            cv::Mat get_descriptor(const int index) const { return _descriptors.row(index); }
            const cv::KeyPoint& get_keypoint(const int index) const { return _keypoints[index]; }

            //setter
            void set_tracking_radius(unsigned int tracking_radius) { _tracking_radius = tracking_radius; }


        private:
            typedef std::vector<cv::KeyPoint> keypoint_vector;

            keypoint_vector _keypoints;
            cv::Mat _descriptors;
            cv::Ptr<cv::DescriptorMatcher> _matcher;
            std::vector<bool> _matched_marks;
            unsigned int _cell_size;
            int _cell_count_x;
            int _cell_count_y;
            int _cell_search_radius;
            unsigned int _tracking_radius;
            int _img_rows;
            int _img_cols;
            int _vertical_search_radius;
            float _triangulation_ratio_th;
            float _tracking_ratio_th;
            float _desc_dist_th;
            std::vector<float> _kps_depths;

            typedef std::vector<int> index_list_t;
            std::vector<std::vector<index_list_t>> _index_hashmap;

            typedef std::pair<int, int> index_pair_t;
            inline index_pair_t compute_hashed_index(const cv::Point2f &val, const float cell_size) const
            {
                return index_pair_t(floor(val.y / cell_size), floor(val.x / cell_size));
            }
    };

};

#endif
