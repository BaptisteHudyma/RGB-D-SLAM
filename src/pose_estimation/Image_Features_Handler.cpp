#include "Image_Features_Handler.hpp"
#include "Constants.hpp"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <limits>
#include <thread>

namespace rgbd_slam {
namespace poseEstimation {

    /**
     * \brief This function is from code in this answer: http://answers.opencv.org/question/93317/orb-keypoints-distribution-over-an-image/
     */
    static void _adaptive_non_maximal_suppresion(keypoint_vector &keypoints, const int num_to_keep,
            const float tx, const float ty)
    {
        // Sort by response
        std::sort(keypoints.begin(), keypoints.end(),
                [&keypoints](const cv::KeyPoint &lhs, const cv::KeyPoint &rhs) {
                return lhs.response > rhs.response;
                });

        keypoint_vector anmsPts;
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
        for (unsigned int i = 0, count = radii.size(); i < count; i++)
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

    /**
     * \brief Perform a feature detection on structure p
     *
     * \param[in] p Structure containing an image, feature extractors, divisions...
     * \param[out] allKeapoints The features found in p
     */
    static void perform_detect_corners(compute_features_data *p, keypoint_vector *allKeypoints)
    {
        for (rect_vector::size_type r = 0; r < p->_subImgsRects.size(); r++)
        {
            cv::Rect rect = p->_subImgsRects[r];
            cv::Mat sub_img = p->_img(rect);
            keypoint_vector keypoints;
            keypoints.reserve(p->_voParams->get_max_keypoints_per_cell());
            p->_detector->detect(sub_img, keypoints);
            if (keypoints.size() > p->_voParams->get_max_keypoints_per_cell())
            {
                _adaptive_non_maximal_suppresion(keypoints, p->_voParams->get_max_keypoints_per_cell(), (float)rect.x, (float)rect.y);
            }
            else
            {
                for (keypoint_vector::size_type i = 0; i < keypoints.size(); i++)
                {
                    keypoints[i].pt.x += (float)rect.x;
                    keypoints[i].pt.y += (float)rect.y;
                }
            }
            allKeypoints->insert(allKeypoints->end(), keypoints.begin(), keypoints.end());
        }
    }


    Image_Features_Handler::Image_Features_Handler(const Parameters &voParams)
        : _voParams(voParams)
    {
        assert(_voParams.get_height() > 0);
        assert(_voParams.get_width() > 0);
        assert(_voParams.get_detection_cell_size() > 0);
        assert(_voParams.get_max_keypoints_per_cell() > 0);
        assert(_voParams.get_tracking_radius() > 0);
        assert(_voParams.get_agast_threshold() > 0);

        unsigned int num_cells_y = 1 + ((_voParams.get_height() - 1) / _voParams.get_detection_cell_size());
        unsigned int num_cells_x = 1 + ((_voParams.get_width() - 1) / _voParams.get_detection_cell_size());
        int s = _voParams.get_detection_cell_size();
        for (unsigned int i = 0; i < num_cells_y; i++)
        {
            for (unsigned int k = 0; k < num_cells_x; k++)
            {
                int sy = s;
                if ((i == num_cells_y - 1) && ((i + 1) * s > _voParams.get_height()))
                {
                    sy = _voParams.get_height() - (i * s);
                }
                int sx = s;
                if ((k == num_cells_x - 1) && ((k + 1) * s > _voParams.get_width()))
                {
                    sx = _voParams.get_width() - (k * s);
                }
                _subImgsRects.push_back(cv::Rect(k * s, i * s, sx, sy));
            }
        }

        _thData[0]._detector = cv::AgastFeatureDetector::create(_voParams.get_agast_threshold());
        _thData[0]._extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        _thData[0]._subImgsRects = _subImgsRects;
        _thData[0]._voParams = &_voParams;

        _thData[1]._detector = cv::AgastFeatureDetector::create(_voParams.get_agast_threshold());
        _thData[1]._extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        _thData[1]._subImgsRects = _subImgsRects;
        _thData[1]._voParams = &_voParams;
    }

    void Image_Features_Handler::perform_compute_features(compute_features_data *p)
    {
        keypoint_vector all_keypoints;
        all_keypoints.reserve(p->_subImgsRects.size() * p->_voParams->get_max_keypoints_per_cell());
        perform_detect_corners(p, &all_keypoints);
        if (all_keypoints.size() < LVT_CORNERS_LOW_TH)
        {
            all_keypoints.clear();
            int original_agast_th = p->_detector->getThreshold();
            int lowered_agast_th = (double)original_agast_th * 0.5 + 0.5;
            p->_detector->setThreshold(lowered_agast_th);
            perform_detect_corners(p, &all_keypoints);
            p->_detector->setThreshold(original_agast_th);
        }

        cv::Mat desc;
        p->_extractor->compute(p->_img, all_keypoints, desc);
        p->_features_struct->init(p->_img, all_keypoints, desc, p->_voParams->get_tracking_radius(), LVT_HASHING_CELL_SIZE,
                LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, p->_voParams->get_triangulation_ratio_test_threshold(),
                p->_voParams->get_tracking_ratio_test_threshold(), p->_voParams->get_descriptor_matching_threshold());
    }

    void Image_Features_Handler::perform_compute_descriptors_only(compute_features_data *p)
    {
        cv::Mat desc;
        const point_vector* ext_kp = p->_ext_kp;
        keypoint_vector keypoints;
        keypoints.reserve(ext_kp->size());
        for (unsigned int i = 0, count = ext_kp->size(); i < count; i++)
        {
            cv::KeyPoint kp;
            kp.pt = ext_kp->at(i);
            keypoints.push_back(kp);
        }
        p->_extractor->compute(p->_img, keypoints, desc);
        p->_features_struct->init(p->_img, keypoints, desc, p->_voParams->get_tracking_radius(), LVT_HASHING_CELL_SIZE,
                LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, p->_voParams->get_triangulation_ratio_test_threshold(),
                p->_voParams->get_tracking_ratio_test_threshold(), p->_voParams->get_descriptor_matching_threshold());
    }


    void Image_Features_Handler::compute_features(const cv::Mat& img_gray, const cv::Mat& in_img_depth, Image_Features_Struct& out_struct)
    {
        // detect corners in the image as normal
        _thData[0]._img = img_gray;
        compute_features_data *p = &_thData[0];
        keypoint_vector all_keypoints;
        all_keypoints.reserve(p->_subImgsRects.size() * p->_voParams->get_max_keypoints_per_cell());
        perform_detect_corners(p, &all_keypoints);
        if (all_keypoints.size() < LVT_CORNERS_LOW_TH)
        {
            all_keypoints.clear();
            int original_agast_th = p->_detector->getThreshold();
            int lowered_agast_th = (double)original_agast_th * 0.5 + 0.5;
            p->_detector->setThreshold(lowered_agast_th);
            perform_detect_corners(p, &all_keypoints);
            p->_detector->setThreshold(original_agast_th);
        }

        // compute descriptors
        cv::Mat desc;
        p->_extractor->compute(p->_img, all_keypoints, desc);

        // retain corners with valid depth values
        std::vector<float> kps_depths;
        keypoint_vector filtered_kps;
        cv::Mat filtered_desc;
        kps_depths.reserve(all_keypoints.size());
        filtered_kps.reserve(all_keypoints.size());
        for (keypoint_vector::size_type i = 0; i < all_keypoints.size(); i++)
        {
            const cv::KeyPoint &kp = all_keypoints[i];
            const float d = in_img_depth.at<float>(kp.pt.y, kp.pt.x);
            if (d >= _voParams.get_near_plane_distance() and d <= _voParams.get_far_plane_distance())
            {
                kps_depths.push_back(d);
                filtered_kps.push_back(kp);
                filtered_desc.push_back(desc.row(i).clone());
            }
        }

        // Undistort keypoints if the img is distorted
        if (fabs(_voParams.get_k1()) > 1e-5)
        {
            cv::Mat kps_mat(filtered_kps.size(), 2, CV_32F);
            for (keypoint_vector::size_type i = 0; i < filtered_kps.size(); i++)
            {
                kps_mat.at<float>(i, 0) = filtered_kps[i].pt.x;
                kps_mat.at<float>(i, 1) = filtered_kps[i].pt.y;
            }
            kps_mat = kps_mat.reshape(2);
            cv::Matx33f intrinsics_mtrx(_voParams.get_fx(), 0.0, _voParams.get_cx(),
                    0.0, _voParams.get_fy(), _voParams.get_cy(),
                    0.0, 0.0, 1.0);
            std::vector<float> dist;
            dist.push_back(_voParams.get_k1());
            dist.push_back(_voParams.get_k2());
            dist.push_back(_voParams.get_p1());
            dist.push_back(_voParams.get_p2());
            dist.push_back(_voParams.get_k3());
            cv::undistortPoints(kps_mat, kps_mat, cv::Mat(intrinsics_mtrx), cv::Mat(dist), cv::Mat(), intrinsics_mtrx);
            kps_mat = kps_mat.reshape(1);
            for (keypoint_vector::size_type i = 0; i < filtered_kps.size(); i++)
            {
                cv::KeyPoint &kp = filtered_kps[i];
                kp.pt.x = kps_mat.at<float>(i, 0);
                kp.pt.y = kps_mat.at<float>(i, 1);
            }
        }

        // initialize output structs
        out_struct.init(img_gray, filtered_kps, filtered_desc, _voParams.get_tracking_radius(), LVT_HASHING_CELL_SIZE,
                LVT_ROW_MATCHING_VERTICAL_SEARCH_RADIUS, _voParams.get_triangulation_ratio_test_threshold(),
                _voParams.get_tracking_ratio_test_threshold(), _voParams.get_descriptor_matching_threshold(), &kps_depths);
    }


    void Image_Features_Handler::row_match(Image_Features_Struct& features_left, Image_Features_Struct& features_right,
            std::vector<cv::DMatch>& out_matches)
    {
        //match each feature in left image with feature in righ image
        for (unsigned int i = 0, count = features_left.get_features_count(); i < count; i++)
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

}
}
