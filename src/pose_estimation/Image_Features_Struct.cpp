#include "Image_Features_Struct.hpp"

namespace poseEstimation {

    Image_Features_Struct::Image_Features_Struct() : 
        _cell_size(25),
        _cell_count_x(-1), _cell_count_y(-1), 
        _cell_search_radius(0), 
        _tracking_radius(0), 
        _img_rows(0), _img_cols(0), 
        _vertical_search_radius(2), _triangulation_ratio_th(0.6), 
        _tracking_ratio_th(0.8), _desc_dist_th(25.0) 
    {
        //empty
    }

    void Image_Features_Struct::init(const cv::Mat& in_image, std::vector<cv::KeyPoint>& in_keypoints, const cv::Mat& in_desc,
            int in_tracking_radius, int in_hashing_cell_size, int in_vertical_search_radius,
            float in_triangulation_ratio_th, float in_tracking_ratio_th, float in_desc_dist_th, std::vector<float> *kps_depth)
    {
        _cell_size = in_hashing_cell_size;
        _vertical_search_radius = in_vertical_search_radius;
        _triangulation_ratio_th = in_triangulation_ratio_th;
        _tracking_ratio_th = in_tracking_ratio_th;
        _desc_dist_th = in_desc_dist_th;
        _img_rows = in_image.rows;
        _img_cols = in_image.cols;
        _tracking_radius = in_tracking_radius;
        const float k_cell_size = static_cast<float>(_cell_size);
        _cell_count_x = std::ceil(_img_cols / k_cell_size);
        _cell_count_y = std::ceil(_img_rows / k_cell_size);
        _matcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher(cv::NORM_HAMMING, false));
        _keypoints.swap(in_keypoints);
        _descriptors = in_desc;
        _cell_search_radius = (_tracking_radius == _cell_size) ? 1 : std::ceil((float)_tracking_radius / k_cell_size);
        _index_hashmap.resize(_cell_count_y);
        for (int i = 0; i < _cell_count_y; i++) {
            _index_hashmap[i].resize(_cell_count_x);
        }
        for (keypoint_vector::size_type i = 0, count = _keypoints.size(); i < count; i++) {
            index_pair_t hashed_idx = compute_hashed_index(_keypoints[i].pt, k_cell_size);
            _index_hashmap[hashed_idx.first][hashed_idx.second].push_back(i);
        }
        reset_matched_marks();
        if (kps_depth) {
            _kps_depths = *kps_depth;
        }
    }

    /*
     *  find the best feature in the struct that matches the passed one and return its index, or -1 otherwise. (ratio is the ratio to use for the ratio test).
     *
     *
     *
     *  return: best feature index or -1
     */
    int Image_Features_Struct::find_match_index(const vector2& sl_pt, const cv::Mat& desc, float *d1, float *d2)const
    {
        const cv::Point2f pt(sl_pt.x(), sl_pt.y());
        const index_pair_t hash_idx = compute_hashed_index(pt, (float)_cell_size);
        int start_y = hash_idx.first - _cell_search_radius;
        if (start_y < 0)
            start_y = 0;
        int end_y = hash_idx.first + _cell_search_radius + 1;
        if (end_y > _cell_count_y)
            end_y = _cell_count_y;
        int start_x = hash_idx.second - _cell_search_radius;
        if (start_x < 0)
            start_x = 0;
        int end_x = hash_idx.second + _cell_search_radius + 1;
        if (end_x > _cell_count_x)
            end_x = _cell_count_x;

        const float r2 = static_cast<float>(_tracking_radius*_tracking_radius);
        cv::Mat mask(cv::Mat::zeros(1, _keypoints.size(), CV_8UC1));
        for (int i = start_y; i < end_y; i++) {
            for (int k = start_x; k < end_x; k++) {
                const index_list_t& kp_index_list = _index_hashmap[i][k];
                for (size_t kp = 0, count = kp_index_list.size(); kp < count; kp++) {
                    const int kp_idx = kp_index_list[kp];
                    if (!_matched_marks[kp_idx]) {
                        const float dx = _keypoints[kp_idx].pt.x - pt.x;
                        const float dy = _keypoints[kp_idx].pt.y - pt.y;
                        if ((dx*dx + dy * dy) < r2) {
                            mask.at<uint8_t>(0, kp_index_list[kp]) = 1;
                        }
                    }
                }
            }
        }

        std::vector< std::vector<cv::DMatch> > matches;
        _matcher->knnMatch(desc, _descriptors, matches, 2, mask);
        if (matches[0].size() > 1) {
            float d_ratio = matches[0][0].distance / matches[0][1].distance;
            if (d_ratio < _tracking_ratio_th) {
                *d1 = matches[0][0].distance;
                *d2 = matches[0][1].distance;
                return matches[0][0].trainIdx;
            }
        } else if ((matches[0].size() == 1) && (matches[0][0].distance <= _desc_dist_th)) {
            *d1 = matches[0][0].distance;
            *d2 = -1.0;
            return matches[0][0].trainIdx;
        }

        /*else*/
        return -1;
    }

    int Image_Features_Struct::row_match(const cv::Point2f& pt, const cv::Mat& desc)const
    {
        int start_y = int(pt.y) - _vertical_search_radius;
        if (start_y < 0)
            start_y = 0;
        int end_y = int(pt.y) + _vertical_search_radius;
        if (end_y > _img_rows)
            end_y = _img_rows;

        cv::Mat mask(cv::Mat::zeros(1, _keypoints.size(), CV_8UC1));
        for (keypoint_vector::size_type i = 0, count = _keypoints.size(); i < count; i++) {
            const cv::Point2f kp_pt = _keypoints[i].pt;
            if (!_matched_marks[i] && kp_pt.y >= start_y && kp_pt.y <= end_y) {
                mask.at<uint8_t>(0, i) = 1;
            }
        }

        std::vector< std::vector<cv::DMatch> > matches;
        _matcher->knnMatch(desc, _descriptors, matches, 2, mask);
        if (((matches[0].size() > 1) && (matches[0][0].distance / matches[0][1].distance) < _triangulation_ratio_th) ||
                ((matches[0].size() == 1) && (matches[0][0].distance <= _desc_dist_th))) {
            return matches[0][0].trainIdx;
        }

        /*else*/
        return -1;
    }




}


