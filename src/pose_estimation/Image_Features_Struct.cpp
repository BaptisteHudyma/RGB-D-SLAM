#include "Image_Features_Struct.hpp"

using namespace poseEstimation;


Image_Features_Struct::Image_Features_Struct() : 
    m_img_rows(0), m_img_cols(0), m_tracking_radius(0), 
    m_cell_count_x(-1), m_cell_count_y(-1), 
    m_cell_search_radius(0), m_cell_size(25),
    m_vertical_search_radius(2), m_triangulation_ratio_th(0.6), 
    m_tracking_ratio_th(0.8), m_desc_dist_th(25.0) 
{
    //empty
}

void Image_Features_Struct::init(const cv::Mat& in_image, std::vector<cv::KeyPoint>& in_keypoints, const cv::Mat& in_desc,
        int in_tracking_radius, int in_hashing_cell_size, int in_vertical_search_radius,
        float in_triangulation_ratio_th, float in_tracking_ratio_th, float in_desc_dist_th, std::vector<float> *kps_depth)
{
    m_cell_size = in_hashing_cell_size;
    m_vertical_search_radius = in_vertical_search_radius;
    m_triangulation_ratio_th = in_triangulation_ratio_th;
    m_tracking_ratio_th = in_tracking_ratio_th;
    m_desc_dist_th = in_desc_dist_th;
    m_img_rows = in_image.rows;
    m_img_cols = in_image.cols;
    m_tracking_radius = in_tracking_radius;
    const float k_cell_size = static_cast<float>(m_cell_size);
    m_cell_count_x = std::ceil(m_img_cols / k_cell_size);
    m_cell_count_y = std::ceil(m_img_rows / k_cell_size);
    m_matcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher(cv::NORM_HAMMING, false));
    m_keypoints.swap(in_keypoints);
    m_descriptors = in_desc;
    m_cell_search_radius = (m_tracking_radius == m_cell_size) ? 1 : std::ceil((float)m_tracking_radius / k_cell_size);
    m_index_hashmap.resize(m_cell_count_y);
    for (int i = 0; i < m_cell_count_y; i++) {
        m_index_hashmap[i].resize(m_cell_count_x);
    }
    for (int i = 0, count = m_keypoints.size(); i < count; i++) {
        index_pair_t hashed_idx = compute_hashed_index(m_keypoints[i].pt, k_cell_size);
        m_index_hashmap[hashed_idx.first][hashed_idx.second].push_back(i);
    }
    reset_matched_marks();
    if (kps_depth) {
        m_kps_depths = *kps_depth;
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
    const index_pair_t hash_idx = compute_hashed_index(pt, (float)m_cell_size);
    int start_y = hash_idx.first - m_cell_search_radius;
    if (start_y < 0)
        start_y = 0;
    int end_y = hash_idx.first + m_cell_search_radius + 1;
    if (end_y > m_cell_count_y)
        end_y = m_cell_count_y;
    int start_x = hash_idx.second - m_cell_search_radius;
    if (start_x < 0)
        start_x = 0;
    int end_x = hash_idx.second + m_cell_search_radius + 1;
    if (end_x > m_cell_count_x)
        end_x = m_cell_count_x;

    const float r2 = static_cast<float>(m_tracking_radius*m_tracking_radius);
    cv::Mat mask(cv::Mat::zeros(1, m_keypoints.size(), CV_8UC1));
    for (int i = start_y; i < end_y; i++) {
        for (int k = start_x; k < end_x; k++) {
            const index_list_t& kp_index_list = m_index_hashmap[i][k];
            for (size_t kp = 0, count = kp_index_list.size(); kp < count; kp++) {
                const int kp_idx = kp_index_list[kp];
                if (!m_matched_marks[kp_idx]) {
                    const float dx = m_keypoints[kp_idx].pt.x - pt.x;
                    const float dy = m_keypoints[kp_idx].pt.y - pt.y;
                    if ((dx*dx + dy * dy) < r2) {
                        mask.at<uint8_t>(0, kp_index_list[kp]) = 1;
                    }
                }
            }
        }
    }

    std::vector< std::vector<cv::DMatch> > matches;
    m_matcher->knnMatch(desc, m_descriptors, matches, 2, mask);
    if (matches[0].size() > 1) {
        float d_ratio = matches[0][0].distance / matches[0][1].distance;
        if (d_ratio < m_tracking_ratio_th) {
            *d1 = matches[0][0].distance;
            *d2 = matches[0][1].distance;
            return matches[0][0].trainIdx;
        }
    } else if ((matches[0].size() == 1) && (matches[0][0].distance <= m_desc_dist_th)) {
        *d1 = matches[0][0].distance;
        *d2 = -1.0;
        return matches[0][0].trainIdx;
    }

    /*else*/
    return -1;
}

int Image_Features_Struct::row_match(const cv::Point2f& pt, const cv::Mat& desc)const
{
    int start_y = int(pt.y) - m_vertical_search_radius;
    if (start_y < 0)
        start_y = 0;
    int end_y = int(pt.y) + m_vertical_search_radius;
    if (end_y > m_img_rows)
        end_y = m_img_rows;

    cv::Mat mask(cv::Mat::zeros(1, m_keypoints.size(), CV_8UC1));
    for (int i = 0, count = m_keypoints.size(); i < count; i++) {
        const cv::Point2f kp_pt = m_keypoints[i].pt;
        if (!m_matched_marks[i] && kp_pt.y >= start_y && kp_pt.y <= end_y) {
            mask.at<uint8_t>(0, i) = 1;
        }
    }

    std::vector< std::vector<cv::DMatch> > matches;
    m_matcher->knnMatch(desc, m_descriptors, matches, 2, mask);
    if (((matches[0].size() > 1) && (matches[0][0].distance / matches[0][1].distance) < m_triangulation_ratio_th) ||
            ((matches[0].size() == 1) && (matches[0][0].distance <= m_desc_dist_th))) {
        return matches[0][0].trainIdx;
    }

    /*else*/
    return -1;
}






