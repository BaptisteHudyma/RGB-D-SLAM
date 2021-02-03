#include "Image_Features_Handler.hpp"
#include "Constants.hpp"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <limits>
#include <thread>

using namespace poseEstimation;

// This function is from code in this answer: http://answers.opencv.org/question/93317/orb-keypoints-distribution-over-an-image/
static void _adaptive_non_maximal_suppresion(std::vector<cv::KeyPoint> &keypoints, const int num_to_keep,
        const float tx, const float ty)
{
    // Sort by response
    std::sort(keypoints.begin(), keypoints.end(),
            [&keypoints](const cv::KeyPoint &lhs, const cv::KeyPoint &rhs) {
            return lhs.response > rhs.response;
            });

    std::vector<cv::KeyPoint> anmsPts;
    anmsPts.reserve(num_to_keep);

    std::vector<float> radii;
    radii.resize(keypoints.size());
    std::vector<float> radiiSorted;
    radiiSorted.resize(keypoints.size());

    const float robustCoeff = 1.11;
    for (int i = 0, count_i = keypoints.size(); i < count_i; i++)
    {
        const float response = keypoints[i].response * robustCoeff;
        float radius = (std::numeric_limits<float>::max)();
        for (int j = 0; j < i && keypoints[j].response > response; j++)
        {
            const cv::Point2f diff_pt = keypoints[i].pt - keypoints[j].pt;
            radius = (std::min)(radius, diff_pt.x * diff_pt.x + diff_pt.y * diff_pt.y);
        }
        radius = sqrtf(radius);
        radii[i] = radius;
        radiiSorted[i] = radius;
    }

    std::sort(radiiSorted.begin(), radiiSorted.end(),
            [&radiiSorted](const float &lhs, const float &rhs) {
            return lhs > rhs;
            });

    const float decisionRadius = radiiSorted[num_to_keep];
    for (int i = 0, count = radii.size(); i < count; i++)
    {
        if (radii[i] >= decisionRadius)
        {
            keypoints[i].pt.x += tx;
            keypoints[i].pt.y += ty;
            anmsPts.push_back(keypoints[i]);
        }
    }

    anmsPts.swap(keypoints);
}

Image_Features_Handler::Image_Features_Handler(const Parameters &voParams)
    : voParams(voParams)
{
    assert(this->voParams.get_height() > 0);
    assert(this->voParams.get_width() > 0);
    assert(this->voParams.get_detection_cell_size() > 0);
    assert(this->voParams.get_max_keypoints_per_cell() > 0);
    assert(this->voParams.get_tracking_radius() > 0);
    assert(this->voParams.get_agast_threshold() > 0);

    int num_cells_y = 1 + ((this->voParams.get_height() - 1) / this->voParams.get_detection_cell_size());
    int num_cells_x = 1 + ((this->voParams.get_width() - 1) / this->voParams.get_detection_cell_size());
    int s = this->voParams.get_detection_cell_size();
    for (int i = 0; i < num_cells_y; i++)
    {
        for (int k = 0; k < num_cells_x; k++)
        {
            int sy = s;
            if ((i == num_cells_y - 1) && ((i + 1) * s > this->voParams.get_height()))
            {
                sy = this->voParams.get_height() - (i * s);
            }
            int sx = s;
            if ((k == num_cells_x - 1) && ((k + 1) * s > this->voParams.get_width()))
            {
                sx = this->voParams.get_width() - (k * s);
            }
            this->subImgsRects.push_back(cv::Rect(k * s, i * s, sx, sy));
        }
    }

    this->thData[0].detector = cv::AgastFeatureDetector::create(this->voParams.get_agast_threshold());
    this->thData[0].extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    this->thData[0].subImgsRects = this->subImgsRects;
    this->thData[0].voParams = &this->voParams;

    this->thData[1].detector = cv::AgastFeatureDetector::create(this->voParams.get_agast_threshold());
    this->thData[1].extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    this->thData[1].subImgsRects = this->subImgsRects;
    this->thData[1].voParams = &this->voParams;
}

static void perform_detect_corners(compute_features_data *p, std::vector<cv::KeyPoint> *all_keypoints)
{
    for (int r = 0; r < p->subImgsRects.size(); r++)
    {
        cv::Rect rect = p->subImgsRects[r];
        cv::Mat sub_img = p->img(rect);
        std::vector<cv::KeyPoint> keypoints;
        keypoints.reserve(p->voParams->get_max_keypoints_per_cell());
        p->detector->detect(sub_img, keypoints);
        if (keypoints.size() > p->voParams->get_max_keypoints_per_cell())
        {
            _adaptive_non_maximal_suppresion(keypoints, p->voParams->get_max_keypoints_per_cell(), (float)rect.x, (float)rect.y);
        }
        else
        {
            for (int i = 0; i < keypoints.size(); i++)
            {
                keypoints[i].pt.x += (float)rect.x;
                keypoints[i].pt.y += (float)rect.y;
            }
        }
        all_keypoints->insert(all_keypoints->end(), keypoints.begin(), keypoints.end());
    }
}

void Image_Features_Handler::perform_compute_features(compute_features_data *p)
{
    std::vector<cv::KeyPoint> all_keypoints;
    all_keypoints.reserve(p->subImgsRects.size() * p->voParams->get_max_keypoints_per_cell());
    perform_detect_corners(p, &all_keypoints);
    if (all_keypoints.size() < LVT_CORNERS_LOW_TH)
    {
        all_keypoints.clear();
        int original_agast_th = p->detector->getThreshold();
        int lowered_agast_th = (double)original_agast_th * 0.5 + 0.5;
        p->detector->setThreshold(lowered_agast_th);
        perform_detect_corners(p, &all_keypoints);
        p->detector->setThreshold(original_agast_th);
    }

    cv::Mat desc;
    p->extractor->compute(p->img, all_keypoints, desc);
    p->features_struct->init(p->img, all_keypoints, desc, p->voParams->get_tracking_radius(), LVT_HASHING_CELL_SIZE,
            LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, p->voParams->get_triangulation_ratio_test_threshold(),
            p->voParams->get_tracking_ratio_test_threshold(), p->voParams->get_descriptor_matching_threshold());
}

void Image_Features_Handler::perform_compute_descriptors_only(compute_features_data *p)
{
    cv::Mat desc;
    const std::vector<cv::Point2f> &ext_kp = *(p->ext_kp);
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(ext_kp.size());
    for (int i = 0, count = ext_kp.size(); i < count; i++)
    {
        cv::KeyPoint kp;
        kp.pt = ext_kp[i];
        keypoints.push_back(kp);
    }
    p->extractor->compute(p->img, keypoints, desc);
    p->features_struct->init(p->img, keypoints, desc, p->voParams->get_tracking_radius(), LVT_HASHING_CELL_SIZE,
            LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, p->voParams->get_triangulation_ratio_test_threshold(),
            p->voParams->get_tracking_ratio_test_threshold(), p->voParams->get_descriptor_matching_threshold());
}


/*
 *  Compute 3D features from gray and depth image
 *
 * in img_gray
 * in img_depth
 * out out_struct
 */
void Image_Features_Handler::compute_features(const cv::Mat& img_gray, const cv::Mat& in_img_depth, Image_Features_Struct& out_struct)
{
    // detect corners in the image as normal
    this->thData[0].img = img_gray;
    compute_features_data *p = &this->thData[0];
    std::vector<cv::KeyPoint> all_keypoints;
    all_keypoints.reserve(p->subImgsRects.size() * p->voParams->get_max_keypoints_per_cell());
    perform_detect_corners(p, &all_keypoints);
    if (all_keypoints.size() < LVT_CORNERS_LOW_TH)
    {
        all_keypoints.clear();
        int original_agast_th = p->detector->getThreshold();
        int lowered_agast_th = (double)original_agast_th * 0.5 + 0.5;
        p->detector->setThreshold(lowered_agast_th);
        perform_detect_corners(p, &all_keypoints);
        p->detector->setThreshold(original_agast_th);
    }

    // compute descriptors
    cv::Mat desc;
    p->extractor->compute(p->img, all_keypoints, desc);

    // retain corners with valid depth values
    std::vector<float> kps_depths;
    std::vector<cv::KeyPoint> filtered_kps;
    cv::Mat filtered_desc;
    kps_depths.reserve(all_keypoints.size());
    filtered_kps.reserve(all_keypoints.size());
    for (int i = 0; i < all_keypoints.size(); i++)
    {
        const cv::KeyPoint &kp = all_keypoints[i];
        const float d = in_img_depth.at<float>(kp.pt.y, kp.pt.x);
        if (d >= this->voParams.get_near_plane_distance() and d <= this->voParams.get_far_plane_distance())
        {
            kps_depths.push_back(d);
            filtered_kps.push_back(kp);
            filtered_desc.push_back(desc.row(i).clone());
        }
    }

    // Undistort keypoints if the img is distorted
    if (fabs(this->voParams.get_k1()) > 1e-5)
    {
        cv::Mat kps_mat(filtered_kps.size(), 2, CV_32F);
        for (int i = 0; i < filtered_kps.size(); i++)
        {
            kps_mat.at<float>(i, 0) = filtered_kps[i].pt.x;
            kps_mat.at<float>(i, 1) = filtered_kps[i].pt.y;
        }
        kps_mat = kps_mat.reshape(2);
        cv::Matx33f intrinsics_mtrx(this->voParams.get_fx(), 0.0, this->voParams.get_cx(),
                0.0, this->voParams.get_fy(), this->voParams.get_cy(),
                0.0, 0.0, 1.0);
        std::vector<float> dist;
        dist.push_back(this->voParams.get_k1());
        dist.push_back(this->voParams.get_k2());
        dist.push_back(this->voParams.get_p1());
        dist.push_back(this->voParams.get_p2());
        dist.push_back(this->voParams.get_k3());
        cv::undistortPoints(kps_mat, kps_mat, cv::Mat(intrinsics_mtrx), cv::Mat(dist), cv::Mat(), intrinsics_mtrx);
        kps_mat = kps_mat.reshape(1);
        for (int i = 0; i < filtered_kps.size(); i++)
        {
            cv::KeyPoint &kp = filtered_kps[i];
            kp.pt.x = kps_mat.at<float>(i, 0);
            kp.pt.y = kps_mat.at<float>(i, 1);
        }
    }

    // initialize output structs
    out_struct.init(img_gray, filtered_kps, filtered_desc, this->voParams.get_tracking_radius(), LVT_HASHING_CELL_SIZE,
            LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, this->voParams.get_triangulation_ratio_test_threshold(),
            this->voParams.get_tracking_ratio_test_threshold(), this->voParams.get_descriptor_matching_threshold(), &kps_depths);
}


/*
 *
 *
 * in features_left
 * in features_right
 * out out_matches
 */
void Image_Features_Handler::row_match(Image_Features_Struct& features_left, Image_Features_Struct& features_right,
        std::vector<cv::DMatch>& out_matches)
{
    for (int i = 0, count = features_left.get_features_count(); i < count; i++)
    {
        if (features_left.is_matched(i))
        { // if the feature in the left camera image is matched from tracking then ignore it
            continue;
        }
        cv::Mat desc = features_left.get_descriptor(i);
        const int match_idx = features_right.row_match(features_left.get_keypoint(i).pt, desc);
        if (match_idx != -1)
        {
            cv::DMatch m;
            m.queryIdx = i;
            m.trainIdx = match_idx;
            out_matches.push_back(m);
            features_left.mark_as_matched(i, true);
            features_right.mark_as_matched(match_idx, true);
        }
    }
}

