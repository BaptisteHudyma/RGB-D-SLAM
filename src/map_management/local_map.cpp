#include "local_map.hpp"
#include "../outputs/logger.hpp"
#include "../parameters.hpp"
#include "../tracking/triangulation.hpp"
#include "../utils/camera_transformation.hpp"
#include "../utils/coordinates.hpp"
#include "../utils/covariances.hpp"
#include "../utils/random.hpp"
#include "map_point.hpp"
#include "map_primitive.hpp"
#include "matches_containers.hpp"
#include "pose.hpp"
#include "shape_primitives.hpp"
#include "types.hpp"
#include <iostream>

namespace rgbd_slam {
namespace map_management {

Local_Map::Local_Map()
{
    // Check constants
    assert(features::keypoints::INVALID_MAP_POINT_ID == INVALID_POINT_UNIQ_ID);

    _mapWriter = new outputs::XYZ_Map_Writer("out");

    // For testing purposes, one can deactivate those maps
    //_localPointMap.deactivate();
    //_localPlaneMap.deactivate();
}

Local_Map::~Local_Map() { delete _mapWriter; }

const features::keypoints::KeypointsWithIdStruct Local_Map::get_tracked_keypoints_features(
        const utils::Pose& lastPose) const
{
    const size_t numberOfNewKeypoints = _localPointMap.get_local_map_size() + _localPointMap.get_staged_map_size();

    const WorldToCameraMatrix& worldToCamera =
            utils::compute_world_to_camera_transform(lastPose.get_orientation_quaternion(), lastPose.get_position());

    // initialize output structure
    features::keypoints::KeypointsWithIdStruct keypointsWithIds;

    // TODO: check the efficiency gain of those reserve calls
    keypointsWithIds.reserve(numberOfNewKeypoints);

    const static uint refreshFrequency = Parameters::get_keypoint_refresh_frequency() * 2;
    _localPointMap.get_tracked_features(worldToCamera, keypointsWithIds, refreshFrequency);
    return keypointsWithIds;
}

matches_containers::matchContainer Local_Map::find_feature_matches(
        const utils::Pose& currentPose,
        const features::keypoints::Keypoint_Handler& detectedKeypointsObject,
        const features::primitives::plane_container& detectedPlanes)
{
    const WorldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(
            currentPose.get_orientation_quaternion(), currentPose.get_position());

    matches_containers::matchContainer matchSets;

    // find point matches
    static const size_t minimumPointsForOptimization = Parameters::get_minimum_point_count_for_optimization();
    _localPointMap.get_matches(detectedKeypointsObject, worldToCamera, false, matchSets._points);
    if (matchSets._points.size() < minimumPointsForOptimization or
        matchSets._points.size() < std::min(detectedKeypointsObject.size(), _localPointMap.get_local_map_size()) / 2)
    {
        // if the process as not enough matches, retry matches with a greater margin
        _localPointMap.get_matches(detectedKeypointsObject, worldToCamera, true, matchSets._points);
    }

    for (const features::primitives::Plane& p: detectedPlanes)
    {
        assert(not p.get_shape_mask().empty());
    }

    // find plane matches
    _localPlaneMap.get_matches(detectedPlanes, worldToCamera, false, matchSets._planes);
    return matchSets;
}

void Local_Map::update(const utils::Pose& optimizedPose,
                       const features::keypoints::Keypoint_Handler& keypointObject,
                       const features::primitives::plane_container& detectedPlanes,
                       const matches_containers::match_point_container& outlierMatchedPoints,
                       const matches_containers::match_plane_container& outlierMatchedPlanes)
{
    // TODO find a better way to display trajectory than just a new map point
    // _mapWriter->add_point(optimizedPose.get_position());

    // Unmatch detected outliers
    mark_outliers_as_unmatched(outlierMatchedPoints);
    mark_outliers_as_unmatched(outlierMatchedPlanes);

    const matrix33& poseCovariance = optimizedPose.get_position_variance();
    const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform(
            optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

    // update all local maps
    _localPointMap.update_map(cameraToWorld, poseCovariance, keypointObject);
    _localPlaneMap.update_map(cameraToWorld, poseCovariance, detectedPlanes);

    const bool addAllFeatures = false; // only add unmatched features
    add_features_to_map(poseCovariance, cameraToWorld, keypointObject, detectedPlanes, addAllFeatures);

    // add local map points to global map
    update_local_to_global();
}

void Local_Map::add_features_to_map(const matrix33& poseCovariance,
                                    const CameraToWorldMatrix& cameraToWorld,
                                    const features::keypoints::Keypoint_Handler& keypointObject,
                                    const features::primitives::plane_container& detectedPlanes,
                                    const bool addAllFeatures)
{
    _localPointMap.add_features_to_staged_map(poseCovariance, cameraToWorld, keypointObject, addAllFeatures);

    // Add unmatched poins to the staged map, to unsure tracking of new features
    _localPlaneMap.add_features_to_staged_map(poseCovariance, cameraToWorld, detectedPlanes, addAllFeatures);
}

void Local_Map::update_local_to_global()
{
    // TODO when we have a global map
}

void Local_Map::update_no_pose()
{
    // add local map points
    _localPointMap.update_with_no_tracking();

    // add planes to local map
    _localPlaneMap.update_with_no_tracking();
}

void Local_Map::reset()
{
    _localPointMap.reset();
    _localPlaneMap.reset();
}

/**
 * PROTECTED
 */

void Local_Map::draw_image_head_band(cv::Mat& debugImage) const
{
    assert(not debugImage.empty());

    const cv::Size& debugImageSize = debugImage.size();
    const uint imageWidth = debugImageSize.width;

    // 20 pixels
    const uint bandSize = 20;
    const uint placeInBand = bandSize * 0.75;

    std::stringstream textPoints;
    textPoints << "Points:" << _localPointMap.get_staged_map_size() << "|" << _localPointMap.get_local_map_size();
    const int plointLabelPosition = static_cast<int>(imageWidth * 0.15);
    cv::putText(debugImage,
                textPoints.str(),
                cv::Point(plointLabelPosition, placeInBand),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255, 1));

    std::stringstream text1;
    const double planeOffset = 0.35;
    text1 << "Planes:" << _localPlaneMap.get_staged_map_size() << "|" << _localPlaneMap.get_local_map_size();
    const int planeLabelPosition = static_cast<int>(imageWidth * planeOffset);
    cv::putText(debugImage,
                text1.str(),
                cv::Point(planeLabelPosition, placeInBand),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255, 1));

    std::stringstream text2;
    const double cylinderOffset = 0.70;
    text2 << "Cylinders:";
    const int cylinderLabelPosition = static_cast<int>(imageWidth * cylinderOffset);
    cv::putText(debugImage,
                text2.str(),
                cv::Point(cylinderLabelPosition, placeInBand),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255, 1));
}

void Local_Map::get_debug_image(const utils::Pose& camPose,
                                const bool shouldDisplayStaged,
                                const bool shouldDisplayPlaneMasks,
                                cv::Mat& debugImage) const
{
    draw_image_head_band(debugImage);

    const WorldToCameraMatrix& worldToCamMatrix =
            utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());
    // draw all map features
    _localPointMap.draw_on_image(worldToCamMatrix, debugImage, shouldDisplayStaged);
    if (shouldDisplayPlaneMasks)
        _localPlaneMap.draw_on_image(worldToCamMatrix, debugImage, shouldDisplayStaged);
}

void Local_Map::mark_outliers_as_unmatched(const matches_containers::match_point_container& outlierMatchedPoints)
{
    // Mark outliers as unmatched
    for (const matches_containers::PointMatch& match: outlierMatchedPoints)
    {
        const bool isOutlierRemoved = _localPointMap.mark_feature_with_id_as_unmatched(match._idInMap);
        // If no points were found, this is bad. A match marked as outliers must be in the local map or staged points
        if (not isOutlierRemoved)
        {
            outputs::log_error("Could not find the target point with id " + std::to_string(match._idInMap));
        }
    }
}

void Local_Map::mark_outliers_as_unmatched(const matches_containers::match_plane_container& outlierMatchedPlanes)
{
    // Mark outliers as unmatched
    for (const matches_containers::PlaneMatch& match: outlierMatchedPlanes)
    {
        const bool isOutlierRemoved = _localPlaneMap.mark_feature_with_id_as_unmatched(match._idInMap);
        // If no plane were found, this is bad. A match marked as outliers must be in the local map or staged planes
        if (not isOutlierRemoved)
        {
            outputs::log_error("Could not find the target point with id " + std::to_string(match._idInMap));
        }
    }
}

} // namespace map_management
} // namespace rgbd_slam
