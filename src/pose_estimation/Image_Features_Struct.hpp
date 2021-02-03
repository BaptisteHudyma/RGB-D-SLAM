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

            void mark_as_matched(const int idx, const bool val) { m_matched_marks[idx] = val; }
            bool is_matched(const int idx) const { return m_matched_marks[idx]; }
            void reset_matched_marks() { m_matched_marks = std::vector<bool>(m_keypoints.size(), false); }


            bool is_depth_associated() const { return !m_kps_depths.empty(); }


        public: //getters
            int get_tracking_radius() const { return m_tracking_radius; }
            int get_features_count() const { return (int)m_keypoints.size(); }
            float get_keypoint_depth(const int idx) const { return m_kps_depths[idx]; }
            cv::Mat get_descriptor(const int index) const { return m_descriptors.row(index); }
            const cv::KeyPoint& get_keypoint(const int index) const { return m_keypoints[index]; }

            //setter
            void set_tracking_radius(int tracking_radius) { m_tracking_radius = tracking_radius; }


        private:
            std::vector<cv::KeyPoint> m_keypoints;
            cv::Mat m_descriptors;
            cv::Ptr<cv::DescriptorMatcher> m_matcher;
            std::vector<bool> m_matched_marks;
            int m_cell_size;
            int m_cell_count_x, m_cell_count_y;
            int m_cell_search_radius;
            int m_tracking_radius;
            int m_img_rows, m_img_cols;
            int m_vertical_search_radius;
            float m_triangulation_ratio_th;
            float m_tracking_ratio_th;
            float m_desc_dist_th;
            std::vector<float> m_kps_depths;

            typedef std::vector<int> index_list_t;
            std::vector<std::vector<index_list_t>> m_index_hashmap;

            typedef std::pair<int, int> index_pair_t;
            inline index_pair_t compute_hashed_index(const cv::Point2f &val, const float cell_size) const
            {
                return index_pair_t(floor(val.y / cell_size), floor(val.x / cell_size));
            }
    };

};

#endif
