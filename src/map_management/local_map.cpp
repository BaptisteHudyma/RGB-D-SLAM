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
#include <memory>

namespace rgbd_slam::map_management {

Local_Map::Local_Map()
{
    // Check constants
    assert(features::keypoints::INVALID_MAP_POINT_ID == INVALID_POINT_UNIQ_ID);

    _mapWriter = std::make_unique<outputs::OBJ_Map_Writer>("out");

    // For testing purposes, one can deactivate those maps
    //_localPoint2DMap.deactivate();
    //_localPointMap.deactivate();
    //_localPlaneMap.deactivate();
}

Local_Map::~Local_Map()
{
    _localPoint2DMap.destroy(_mapWriter);
    _localPointMap.destroy(_mapWriter);
    _localPlaneMap.destroy(_mapWriter);
}

features::keypoints::KeypointsWithIdStruct Local_Map::get_tracked_keypoints_features(
        const utils::Pose& lastPose) const noexcept
{
    const size_t numberOfNewKeypoints = _localPoint2DMap.size() + _localPointMap.size();

    const WorldToCameraMatrix& worldToCamera =
            utils::compute_world_to_camera_transform(lastPose.get_orientation_quaternion(), lastPose.get_position());

    // initialize output structure
    features::keypoints::KeypointsWithIdStruct keypointsWithIds;

    // TODO: check the efficiency gain of those reserve calls
    keypointsWithIds.reserve(numberOfNewKeypoints);

    constexpr uint refreshFrequency = parameters::detection::keypointRefreshFrequency * 2;
    _localPoint2DMap.get_tracked_features(worldToCamera, keypointsWithIds, refreshFrequency);
    _localPointMap.get_tracked_features(worldToCamera, keypointsWithIds, refreshFrequency);
    return keypointsWithIds;
}

matches_containers::matchContainer Local_Map::find_feature_matches(
        const utils::Pose& currentPose, const DetectedFeatureContainer& detectedFeatures) noexcept
{
    // store the id given to the function
    _detectedFeatureId = detectedFeatures.id;

    // get transformation matrix from estimated pose
    const WorldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(
            currentPose.get_orientation_quaternion(), currentPose.get_position());

    matches_containers::matchContainer matchSets;

    // find point matches
    _localPoint2DMap.get_matches(detectedFeatures.keypointObject,
                                 worldToCamera,
                                 false,
                                 parameters::optimization::minimumPointForOptimization,
                                 matchSets._points2D);
    if (matchSets._points.size() < parameters::optimization::minimumPointForOptimization or
        matchSets._points.size() <
                std::min(detectedFeatures.keypointObject.size(), _localPoint2DMap.get_local_map_size()) / 2)
    {
        // if the process as not enough matches, retry matches with a greater margin
        _localPoint2DMap.get_matches(detectedFeatures.keypointObject,
                                     worldToCamera,
                                     true,
                                     parameters::optimization::minimumPointForOptimization,
                                     matchSets._points2D);
    }

    // find point matches
    _localPointMap.get_matches(detectedFeatures.keypointObject,
                               worldToCamera,
                               false,
                               parameters::optimization::minimumPointForOptimization,
                               matchSets._points);
    if (matchSets._points.size() < parameters::optimization::minimumPointForOptimization or
        matchSets._points.size() <
                std::min(detectedFeatures.keypointObject.size(), _localPointMap.get_local_map_size()) / 2)
    {
        // if the process as not enough matches, retry matches with a greater margin
        _localPointMap.get_matches(detectedFeatures.keypointObject,
                                   worldToCamera,
                                   true,
                                   parameters::optimization::minimumPointForOptimization,
                                   matchSets._points);
    }

    // find plane matches
    _localPlaneMap.get_matches(detectedFeatures.detectedPlanes,
                               worldToCamera,
                               false,
                               parameters::optimization::minimumPlanesForOptimization,
                               matchSets._planes);

    return matchSets;
}

void Local_Map::update(const utils::Pose& optimizedPose,
                       const DetectedFeatureContainer& detectedFeatures,
                       const matches_containers::match_point_container& outlierMatchedPoints,
                       const matches_containers::match_plane_container& outlierMatchedPlanes) noexcept
{
    assert(_detectedFeatureId == detectedFeatures.id);

    // Unmatch detected outliers
    mark_outliers_as_unmatched(outlierMatchedPoints);
    mark_outliers_as_unmatched(outlierMatchedPlanes);

    const matrix33& poseCovariance = optimizedPose.get_position_variance();
    const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform(
            optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

    // update all local maps
    _localPoint2DMap.update_map(cameraToWorld, poseCovariance, detectedFeatures.keypointObject, _mapWriter);
    _localPointMap.update_map(cameraToWorld, poseCovariance, detectedFeatures.keypointObject, _mapWriter);
    _localPlaneMap.update_map(cameraToWorld, poseCovariance, detectedFeatures.detectedPlanes, _mapWriter);

    const bool addAllFeatures = false; // only add unmatched features
    add_features_to_map(poseCovariance, cameraToWorld, detectedFeatures, addAllFeatures);

    // add local map points to global map
    update_local_to_global();
}

void Local_Map::add_features_to_map(const matrix33& poseCovariance,
                                    const CameraToWorldMatrix& cameraToWorld,
                                    const DetectedFeatureContainer& detectedFeatures,
                                    const bool addAllFeatures) noexcept
{
    assert(_detectedFeatureId == detectedFeatures.id);

    _localPoint2DMap.add_features_to_staged_map(
            poseCovariance, cameraToWorld, detectedFeatures.keypointObject, addAllFeatures);

    _localPointMap.add_features_to_staged_map(
            poseCovariance, cameraToWorld, detectedFeatures.keypointObject, addAllFeatures);

    // Add unmatched poins to the staged map, to unsure tracking of new features
    _localPlaneMap.add_features_to_staged_map(
            poseCovariance, cameraToWorld, detectedFeatures.detectedPlanes, addAllFeatures);
}

void Local_Map::update_local_to_global() noexcept
{
    // TODO when we have a global map
}

void Local_Map::update_no_pose() noexcept
{
    _localPoint2DMap.update_with_no_tracking(_mapWriter);

    // add local map points
    _localPointMap.update_with_no_tracking(_mapWriter);

    // add planes to local map
    _localPlaneMap.update_with_no_tracking(_mapWriter);
}

void Local_Map::reset() noexcept
{
    _localPoint2DMap.reset();
    _localPointMap.reset();
    _localPlaneMap.reset();
}

/**
 * PROTECTED
 */

void Local_Map::draw_image_head_band(cv::Mat& debugImage) const noexcept
{
    assert(not debugImage.empty());

    const cv::Size& debugImageSize = debugImage.size();
    const uint imageWidth = debugImageSize.width;

    // 20 pixels
    const uint bandSize = 20;
    const int placeInBand = static_cast<int>(std::floor(bandSize * 0.75));

    std::stringstream textPoints;
    textPoints << "Points:" << std::format("{: >3}", _localPointMap.get_staged_map_size()) << "|"
               << std::format("{: >3}", _localPointMap.get_local_map_size());
    const int plointLabelPosition = static_cast<int>(imageWidth * 0.15);
    cv::putText(debugImage,
                textPoints.str(),
                cv::Point(plointLabelPosition, placeInBand),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255, 1));

    std::stringstream text1;
    const double planeOffset = 0.35;
    text1 << "Planes:" << std::format("{: >2}", _localPlaneMap.get_staged_map_size()) << "|"
          << std::format("{: >2}", _localPlaneMap.get_local_map_size());
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
                                cv::Mat& debugImage) const noexcept
{
    draw_image_head_band(debugImage);

    const WorldToCameraMatrix& worldToCamMatrix =
            utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());
    // draw all map features
    _localPoint2DMap.draw_on_image(worldToCamMatrix, debugImage, shouldDisplayStaged);
    _localPointMap.draw_on_image(worldToCamMatrix, debugImage, shouldDisplayStaged);
    if (shouldDisplayPlaneMasks)
        _localPlaneMap.draw_on_image(worldToCamMatrix, debugImage, shouldDisplayStaged);
}

void Local_Map::mark_outliers_as_unmatched(
        const matches_containers::match_point_container& outlierMatchedPoints) noexcept
{
    // Mark outliers as unmatched
    for (const matches_containers::PointMatch& match: outlierMatchedPoints)
    {
        const bool isOutlierRemoved = _localPointMap.mark_feature_with_id_as_unmatched(match._idInMap);
        // If no points were found, this is bad. A match marked as outliers must be in the local map or staged points
        if (not isOutlierRemoved)
        {
            outputs::log_error(std::format("Could not find the target point with id {}", match._idInMap));
        }
    }
}

void Local_Map::mark_outliers_as_unmatched(
        const matches_containers::match_plane_container& outlierMatchedPlanes) noexcept
{
    // Mark outliers as unmatched
    for (const matches_containers::PlaneMatch& match: outlierMatchedPlanes)
    {
        const bool isOutlierRemoved = _localPlaneMap.mark_feature_with_id_as_unmatched(match._idInMap);
        // If no plane were found, this is bad. A match marked as outliers must be in the local map or staged planes
        if (not isOutlierRemoved)
        {
            outputs::log_error(std::format("Could not find the target point with id {}", match._idInMap));
        }
    }
}

} // namespace rgbd_slam::map_management
