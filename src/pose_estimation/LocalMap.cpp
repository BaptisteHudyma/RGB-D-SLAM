#include "LocalMap.hpp"

#include "Image_Features_Handler.hpp"
#include "Image_Features_Struct.hpp"
#include "Constants.hpp"
#include "PoseUtils.hpp"

#include <cmath>
#include <Eigen/SVD>


namespace poseEstimation {

    static float min_x, max_x, min_y, max_y;

    /**
     * \brief Check if the point pt is visible from observer at position w2c 
     *
     * \param[in] pt Point to check 
     * \param[in] w2c Position of the observer
     * \param[in] params Parameter container
     * \param[out] outProjectedPt pt projected to world coordinates
     *
     * \return True if point pt is visible from w2c
     */
    static inline bool is_point_visible(const vector3& pt, const matrix34& w2c, const Parameters& params, vector2& outProjectedPt)
    {
        vector4 pt_h;
        pt_h << pt, 1.0;
        vector3 pt_cam = w2c * pt_h;
        if (pt_cam.z() < params.get_near_plane_distance() or pt_cam.z() > params.get_far_plane_distance())
        {
            return false;
        }
        const double inv_z = 1.0 / pt_cam.z();
        const double u = params.get_fx() * pt_cam.x() * inv_z + params.get_cx();
        const double v = params.get_fy() * pt_cam.y() * inv_z + params.get_cy();
        if (u < min_x or u > max_x or v < min_y or v > max_y) {
            return false;
        }
        outProjectedPt << u, v;
        return true;
    }

    Local_Map::Local_Map(const Parameters& voParams, Image_Features_Handler* featuresHandler) 
        : _voParams(voParams), _featuresHandler(featuresHandler)
    {
        // compute image bounds
        if (fabs(_voParams.get_k1()) < 1e-5)
        {
            min_x = 0.0;
            max_x = _voParams.get_width();
            min_y = 0.0;
            max_y = _voParams.get_height();
        }
        else
        {
            cv::Mat kps_mat(4, 2, CV_32F);
            kps_mat.at<float>(0, 0) = 0.0;
            kps_mat.at<float>(0, 1) = 0.0;
            kps_mat.at<float>(1, 0) = _voParams.get_width();
            kps_mat.at<float>(1, 1) = 0.0;
            kps_mat.at<float>(2, 0) = 0.0;
            kps_mat.at<float>(2, 1) = _voParams.get_height();
            kps_mat.at<float>(3, 0) = _voParams.get_width();
            kps_mat.at<float>(3, 1) = _voParams.get_height();
            cv::Matx33f intrinsics_mtrx(_voParams.get_fx(), 0.0, _voParams.get_cx(),
                    0.0, _voParams.get_fy(), _voParams.get_cy(),
                    0.0, 0.0, 1.0);
            std::vector<float> dist;
            dist.push_back(_voParams.get_k1());
            dist.push_back(_voParams.get_k2());
            dist.push_back(_voParams.get_p1());
            dist.push_back(_voParams.get_p2());
            dist.push_back(_voParams.get_k3());
            kps_mat = kps_mat.reshape(2);
            cv::undistortPoints(kps_mat, kps_mat, cv::Mat(intrinsics_mtrx), cv::Mat(dist), cv::Mat(), intrinsics_mtrx);
            kps_mat = kps_mat.reshape(1);
            min_x = std::min(kps_mat.at<float>(0, 0), kps_mat.at<float>(2, 0));
            max_x = std::max(kps_mat.at<float>(1, 0), kps_mat.at<float>(3, 0));
            min_y = std::min(kps_mat.at<float>(0, 1), kps_mat.at<float>(1, 1));
            max_y = std::max(kps_mat.at<float>(2, 1), kps_mat.at<float>(3, 1));
        }
    }

    Local_Map::~Local_Map()
    {
        reset();
    }

    void Local_Map::reset()
    {
        _mapPoints.clear();
        _stagedPoints.clear();
    }


    unsigned int Local_Map::find_matches(const Pose &camPose, Image_Features_Struct& features,
            vector3_array& outMapPoints, std::vector<int>& outMatchesLeft)
    {
        const matrix34 cml = Pose_Utils::compute_world_to_camera_transform(camPose);
        unsigned int matches_count = 0;
        std::vector<int> matches(_mapPoints.size(), -2); // mark each map point with its matching index from features, -2 if not visible
        vector2_array projections(_mapPoints.size());

        for (mapPointArray::size_type i = 0; i < _mapPoints.size(); ++i)
        {
            vector2 proj_pt_left;
            if (!is_point_visible(_mapPoints[i].position, cml, _voParams, proj_pt_left))
            {
                _mapPoints[i].counter += 1;
                matches[i] = -2;
                continue;
            }
            projections[i] = proj_pt_left;
            float d1, d2;
            int match_idx_left = features.find_match_index(proj_pt_left, _mapPoints[i].descriptor, &d1, &d2);
            matches[i] = match_idx_left;
            if (match_idx_left != -1)
            {
                matches_count += 1;
                features.mark_as_matched(match_idx_left, true);
            }
        }

        if (matches_count < LVT_N_MATCHES_TH) 
        { //search at 2x the tracking radius
            matches_count = 0;
            features.reset_matched_marks();
            int original_tracking_radius = features.get_tracking_radius();
            features.set_tracking_radius(2 * original_tracking_radius);
            for (mapPointArray::size_type i = 0; i < _mapPoints.size(); ++i)
            {
                if (matches[i] == -2)
                {
                    continue;
                }
                float d1, d2;
                int match_idx_left = features.find_match_index(projections[i], _mapPoints[i].descriptor, &d1, &d2);
                matches[i] = match_idx_left;
                if (match_idx_left != -1)
                {
                    matches_count += 1;
                    features.mark_as_matched(match_idx_left, true);
                }
            }
            features.set_tracking_radius(original_tracking_radius);
        }

        for (mapPointArray::size_type i = 0; i < matches.size(); ++i)
        {
            _mapPoints[i].match_idx = matches[i];
            if (matches[i] == -2) {
                continue;
            }

            if (matches[i] == -1) {
                _mapPoints[i].counter += 1;
                continue;
            }

            _mapPoints[i].age += 1;
            outMapPoints.push_back(_mapPoints[i].position);
            outMatchesLeft.push_back(matches[i]);
        }

        return matches_count;
    }


    void Local_Map::triangulate_rgbd(const Pose &camPose, Image_Features_Struct& features, mapPointArray& outPoints)
    {
        const float inv_fx = 1.0f / _voParams.get_fx();
        const float inv_fy = 1.0f / _voParams.get_fy();
        matrix34 cam_to_world_mtrx;
        cam_to_world_mtrx << camPose.get_orientation_matrix(), camPose.get_position();
        for (unsigned int i = 0, count = features.get_features_count(); i < count; ++i)
        {
            const cv::Point2f pt = features.get_keypoint(i).pt;
            const float u = pt.x;
            const float v = pt.y;
            const float z = features.get_keypoint_depth(i);
            const float x = (u - _voParams.get_cx()) * z * inv_fx;
            const float y = (v - _voParams.get_cy()) * z * inv_fy;
            vector4 pt_h;
            pt_h << x, y, z, 1.0;
            vector3 pt_w = cam_to_world_mtrx * pt_h;

            mapPoint mp;
            mp.position = pt_w;
            mp.descriptor = features.get_descriptor(i);
            mp.counter = 0;
            mp.age = 0;
            outPoints.push_back(mp);
        }
    }


    /*
     *  Triangulate feature points with no depth
     *
     * in camPose Position of the camera
     * in features Image feature points
     * in featuresRight Image features of other image
     * out outpoints Vector of map point
     */
    /*void Local_Map::triangulate(const Pose &camPose, Image_Features_Struct& features, Image_Features_Struct& featuresRight, mapPointArray& outPoints)  {
      std::vector<cv::DMatch> matches;
      _featuresHandler->row_match(features, featuresRight, matches);
      if (matches.empty())
      {   //No row matches for triangulation found.
      assert(0);
      return;
      }

      const Pose camPose_right = Pose_Utils::compute_right_camera_pose(camPose, _voParams.get_baseline());
      const matrix34 cml = Pose_Utils::compute_world_to_camera_transform(camPose);
      const matrix34 cmr = Pose_Utils::compute_world_to_camera_transform(camPose_right);
      const double cx = _voParams.get_cx(), cy = _voParams.get_cy(); // , f_inv = 1.0 / _voParams.get_focal_length();
      const double inv_fx = 1.0 / _voParams.get_fx(), inv_fy = 1.0 / _voParams.get_fy();
      outPoints.reserve(matches.size());

      for (size_t i = 0, count = matches.size(); i < count; ++i)
      {
      const cv::Point2f u1 = features.get_keypoint(matches[i].queryIdx).pt;
      const cv::Point2f u2 = featuresRight.get_keypoint(matches[i].trainIdx).pt;
      double u1_x = (u1.x - cx) * inv_fx;
      double u1_y = (u1.y - cy) * inv_fy;
      double u2_x = (u2.x - cx) * inv_fx;
      double u2_y = (u2.y - cy) * inv_fy;

    // Linear-LS triangulation
    Eigen::Matrix<double, 4, 4> A;
    A << u1_x * cml.row(2) - cml.row(0),
    u1_y * cml.row(2) - cml.row(1),
    u2_x * cmr.row(2) - cmr.row(0),
    u2_y * cmr.row(2) - cmr.row(1);

    vector3 world_pt = A.leftCols<3>().jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-A.col(3));
    assert(std::isfinite(world_pt.x()) and std::isfinite(world_pt.y()) and std::isfinite(world_pt.z()));

    // check if the point is in viewable region by camera
    vector2 proj_pt_l, proj_pt_r;
    if (not is_point_visible(world_pt, cml, _voParams, proj_pt_l) or 
    not is_point_visible(world_pt, cmr, _voParams, proj_pt_r)) {
    continue;
    }

    {
    // check reprojection error
    double err_x = proj_pt_l.x() - u1.x;
    double err_y = proj_pt_l.y() - u1.y;
    if ((err_x * err_x + err_y * err_y) > LVT_REPROJECTION_TH2) {
    continue;
    }
    }
    {
    double err_x = proj_pt_r.x() - u2.x;
    double err_y = proj_pt_r.y() - u2.y;
    if ((err_x * err_x + err_y * err_y) > LVT_REPROJECTION_TH2) {
    continue;
    }
    }

    // create map point
    mapPoint mp;
    mp.position = world_pt;
    mp.descriptor = features.get_descriptor(matches[i].queryIdx);
    mp.counter = 0;
    mp.age = 0;
    outPoints.push_back(mp);
    }
    }*/

    void Local_Map::update_with_new_triangulation(const Pose &camPose, Image_Features_Struct& features, bool dont_stage)
    {
        assert(features.is_depth_associated());

        //make sure we handle depth images
        mapPointArray new_triangulations;
        triangulate_rgbd(camPose, features, new_triangulations);

        assert(not new_triangulations.empty());    //nothing was triangulated

        if (dont_stage or _voParams.get_staged_threshold() == 0 or get_map_size() < LVT_N_MAP_POINTS)
        {
            _mapPoints.insert(_mapPoints.end(), new_triangulations.begin(), new_triangulations.end());
        }
        else
        {
            _stagedPoints.insert(_stagedPoints.end(), new_triangulations.begin(), new_triangulations.end());
        }
    }


    void Local_Map::update_staged_map_points(const Pose &camPose, Image_Features_Struct& features)
    {
        const matrix34 cml = Pose_Utils::compute_world_to_camera_transform(camPose);

        unsigned int n_erased_points = 0;
        unsigned int n_upgraded_points = 0;
        std::set<mapPoint *> points_to_be_deleted;
        for (int i = 0, count = _stagedPoints.size(); i < count; ++i)
        {
            mapPoint *mp = &(_stagedPoints[i]);
            const vector3 world_pt = mp->position;
            vector2 proj_pt_left;
            float d1, d2;
            int match_idx_left = -1;

            if (not is_point_visible(world_pt, cml, _voParams, proj_pt_left) or
                    (match_idx_left = features.find_match_index(proj_pt_left, mp->descriptor, &d1, &d2)) == -1) {
                points_to_be_deleted.insert(mp);
                n_erased_points += 1;
                continue;
            }
            features.mark_as_matched(match_idx_left, true);
            mp->counter += 1;

            if (mp->counter == _voParams.get_staged_threshold() or 
                    get_map_size() < LVT_N_MAP_POINTS) {
                _mapPoints.push_back(_stagedPoints[i]);
                points_to_be_deleted.insert(mp);
                n_upgraded_points += 1;
            }
        }

        _stagedPoints.erase(std::remove_if(_stagedPoints.begin(), _stagedPoints.end(),
                    [&points_to_be_deleted](mapPoint &pt) { return points_to_be_deleted.count(&pt) != 0; }),
                _stagedPoints.end());
    }


    void Local_Map::clean_untracked_points(Image_Features_Struct& features)
    {
        const unsigned int th = _voParams.get_untracked_threshold();
        mapPointArray cleaned_map_points;
        cleaned_map_points.reserve(_mapPoints.size());
        for (mapPointArray::size_type i = 0; i < _mapPoints.size(); ++i)
        {
            if (_mapPoints[i].counter >= th)
            {
                if (_mapPoints[i].match_idx >= 0)
                {
                    features.mark_as_matched(_mapPoints[i].match_idx, false);
                }
            }
            else
            {
                cleaned_map_points.push_back(_mapPoints[i]);
            }
        }
        cleaned_map_points.swap(_mapPoints);
    }



}
