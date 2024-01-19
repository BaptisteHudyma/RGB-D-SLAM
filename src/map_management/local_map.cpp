#include "local_map.hpp"

#include "camera_transformation.hpp"
#include "outputs/logger.hpp"
#include "parameters.hpp"

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
    // small opti: 2D points should never be tracked
    const size_t numberOfNewKeypoints = _localPointMap.size(); // + _localPoint2DMap.size()

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

matches_containers::match_container Local_Map::find_feature_matches(
        const utils::Pose& currentPose, const DetectedFeatureContainer& detectedFeatures) noexcept
{
    // store the id given to the function
    _detectedFeatureId = detectedFeatures.id;

    // get transformation matrix from estimated pose
    const WorldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(
            currentPose.get_orientation_quaternion(), currentPose.get_position());

    matches_containers::match_container matchSets;

    // find point matches
    const double find2dPointMatchesStartTime = static_cast<double>(cv::getTickCount());
    const bool enough2dPointsForOpti =
            _localPoint2DMap.get_matches(detectedFeatures.keypointObject,
                                         worldToCamera,
                                         false,
                                         parameters::optimization::minimumPointForOptimization,
                                         matchSets);
    if (not enough2dPointsForOpti)
    {
        // if the process as not enough matches, retry matches with a greater margin
        _localPoint2DMap.get_matches(detectedFeatures.keypointObject,
                                     worldToCamera,
                                     true,
                                     parameters::optimization::minimumPointForOptimization,
                                     matchSets);
    }
    find2DPointMatchDuration +=
            (static_cast<double>(cv::getTickCount()) - find2dPointMatchesStartTime) / cv::getTickFrequency();

    // find point matches
    const double findPointMatchesStartTime = static_cast<double>(cv::getTickCount());
    const bool enoughPointsForOpti = _localPointMap.get_matches(detectedFeatures.keypointObject,
                                                                worldToCamera,
                                                                false,
                                                                parameters::optimization::minimumPointForOptimization,
                                                                matchSets);
    if (not enoughPointsForOpti)
    {
        // if the process as not enough matches, retry matches with a greater margin
        _localPointMap.get_matches(detectedFeatures.keypointObject,
                                   worldToCamera,
                                   true,
                                   parameters::optimization::minimumPointForOptimization,
                                   matchSets);
    }
    findPointMatchDuration +=
            (static_cast<double>(cv::getTickCount()) - findPointMatchesStartTime) / cv::getTickFrequency();

    // find plane matches
    const double findPlaneMatchesStartTime = static_cast<double>(cv::getTickCount());
    const bool enoughPlanesForOpti = _localPlaneMap.get_matches(detectedFeatures.detectedPlanes,
                                                                worldToCamera,
                                                                false,
                                                                parameters::optimization::minimumPlanesForOptimization,
                                                                matchSets);
    if (not enoughPlanesForOpti)
    {
        // if the process as not enough matches, retry matches with a greater margin
        _localPlaneMap.get_matches(detectedFeatures.detectedPlanes,
                                   worldToCamera,
                                   true,
                                   parameters::optimization::minimumPlanesForOptimization,
                                   matchSets);
    }
    findPlaneMatchDuration +=
            (static_cast<double>(cv::getTickCount()) - findPlaneMatchesStartTime) / cv::getTickFrequency();

    return matchSets;
}

void Local_Map::update(const utils::Pose& optimizedPose,
                       const DetectedFeatureContainer& detectedFeatures,
                       const matches_containers::match_container& outlierMatched)
{
    const double updateMapStartTime = static_cast<double>(cv::getTickCount());
    assert(_detectedFeatureId == detectedFeatures.id);

    const matrix33& poseCovariance = optimizedPose.get_position_variance();
    if (not utils::is_covariance_valid(poseCovariance))
        throw std::invalid_argument("update: The given pose covariance is invalid, map wont be update");

    // Unmatch detected outliers
    mark_outliers_as_unmatched(outlierMatched);

    const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform(
            optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

    // update all local maps
    _localPoint2DMap.update_map(cameraToWorld, poseCovariance, detectedFeatures.keypointObject, _mapWriter);
    _localPointMap.update_map(cameraToWorld, poseCovariance, detectedFeatures.keypointObject, _mapWriter);
    _localPlaneMap.update_map(cameraToWorld, poseCovariance, detectedFeatures.detectedPlanes, _mapWriter);

    // try to triangulate the new features
    const std::vector<UpgradedPoint2DType>& newFeatures = _localPoint2DMap.get_upgraded_features(cameraToWorld);
    for (const auto& upgraded: newFeatures)
    {
        _localPointMap.add_local_map_point(
                LocalMapPoint(upgraded._coordinates, upgraded._covariance, upgraded._descriptor, upgraded._matchIndex));
    }

    const bool addAllFeatures = false; // only add unmatched features
    add_features_to_map(poseCovariance, cameraToWorld, detectedFeatures, addAllFeatures);

    // add local map points to global map
    update_local_to_global();

    mapUpdateDuration += (static_cast<double>(cv::getTickCount()) - updateMapStartTime) / cv::getTickFrequency();
}

void Local_Map::add_features_to_map(const matrix33& poseCovariance,
                                    const CameraToWorldMatrix& cameraToWorld,
                                    const DetectedFeatureContainer& detectedFeatures,
                                    const bool addAllFeatures)
{
    const double addfeaturesStartTime = static_cast<double>(cv::getTickCount());

    if (not utils::is_covariance_valid(poseCovariance))
        throw std::invalid_argument("update: The given pose covariance is invalid, map wont be update");

    assert(_detectedFeatureId == detectedFeatures.id);

    _localPoint2DMap.add_features_to_staged_map(
            poseCovariance, cameraToWorld, detectedFeatures.keypointObject, addAllFeatures);

    _localPointMap.add_features_to_staged_map(
            poseCovariance, cameraToWorld, detectedFeatures.keypointObject, addAllFeatures);

    // Add unmatched poins to the staged map, to unsure tracking of new features
    _localPlaneMap.add_features_to_staged_map(
            poseCovariance, cameraToWorld, detectedFeatures.detectedPlanes, addAllFeatures);

    mapAddFeaturesDuration += (static_cast<double>(cv::getTickCount()) - addfeaturesStartTime) / cv::getTickFrequency();
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

    const double point2DOffset = 0.15;
    std::stringstream textPoints2D;
    textPoints2D << "P2D:" << std::format("{: >3}", _localPoint2DMap.get_staged_map_size()) << ":"
                 << std::format("{: >3}", _localPoint2DMap.get_local_map_size())
                 << " | Points:" << std::format("{: >3}", _localPointMap.get_staged_map_size()) << ":"
                 << std::format("{: >3}", _localPointMap.get_local_map_size())
                 << " | Planes:" << std::format("{: >2}", _localPlaneMap.get_staged_map_size()) << ":"
                 << std::format("{: >2}", _localPlaneMap.get_local_map_size()) << " | Cylinders:" << 0 << ":" << 0;
    const int ploint2DLabelPosition = static_cast<int>(imageWidth * point2DOffset);
    cv::putText(debugImage,
                textPoints2D.str(),
                cv::Point(ploint2DLabelPosition, placeInBand),
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

void Local_Map::show_statistics(const double meanFrameTreatmentDuration,
                                const uint frameCount,
                                const bool shouldDisplayDetails) const noexcept
{
    static auto get_percent_of_elapsed_time = [](double treatmentTime, double totalTimeElapsed) {
        if (totalTimeElapsed <= 0)
            return 0.0;
        return (treatmentTime / totalTimeElapsed) * 100.0;
    };

    if (frameCount > 0)
    {
        const double meanMapUpdateDuration = mapUpdateDuration / static_cast<double>(frameCount);
        outputs::log(std::format("\tMean map update time is {:.4f} seconds ({:.2f}%)",
                                 meanMapUpdateDuration,
                                 get_percent_of_elapsed_time(meanMapUpdateDuration, meanFrameTreatmentDuration)));

        const double meanMapAddFeaturesDuration = mapAddFeaturesDuration / static_cast<double>(frameCount);
        outputs::log(std::format("\tMean map add features time is {:.4f} seconds ({:.2f}%)",
                                 meanMapAddFeaturesDuration,
                                 get_percent_of_elapsed_time(meanMapAddFeaturesDuration, meanFrameTreatmentDuration)));

        const double meanFindMatchDuration =
                (find2DPointMatchDuration + findPointMatchDuration + findPlaneMatchDuration) /
                static_cast<double>(frameCount);
        outputs::log(std::format("\tMean find match time is {:.4f} seconds ({:.2f}%)",
                                 meanFindMatchDuration,
                                 get_percent_of_elapsed_time(meanFindMatchDuration, meanFrameTreatmentDuration)));

        if (shouldDisplayDetails)
        {
            const double meanFind2DPointMatchDuration = find2DPointMatchDuration / static_cast<double>(frameCount);
            outputs::log(std::format("\t\tMean find 2D point match time is {:.4f} seconds ({:.2f}%)",
                                     meanFind2DPointMatchDuration,
                                     get_percent_of_elapsed_time(meanFind2DPointMatchDuration, meanFindMatchDuration)));

            const double meanFindPointMatchDuration = findPointMatchDuration / static_cast<double>(frameCount);
            outputs::log(std::format("\t\tMean find point match time is {:.4f} seconds ({:.2f}%)",
                                     meanFindPointMatchDuration,
                                     get_percent_of_elapsed_time(meanFindPointMatchDuration, meanFindMatchDuration)));

            const double meanFindPlaneMatchDuration = findPlaneMatchDuration / static_cast<double>(frameCount);
            outputs::log(std::format("\t\tMean find plane match time is {:.4f} seconds ({:.2f}%)",
                                     meanFindPlaneMatchDuration,
                                     get_percent_of_elapsed_time(meanFindPlaneMatchDuration, meanFindMatchDuration)));
        }
    }
}

void Local_Map::mark_outliers_as_unmatched(const matches_containers::match_container& outlierMatched) noexcept
{
    // Mark outliers as unmatched
    for (const auto& match: outlierMatched)
    {
        switch (match->get_feature_type())
        {
            case FeatureType::Point:
                {
                    const bool isOutlierRemoved = _localPointMap.mark_feature_with_id_as_unmatched(match->_idInMap);
                    // If no points were found, this is bad. A match marked as outliers must be in the local map or
                    // staged points
                    if (not isOutlierRemoved)
                    {
                        outputs::log_error(std::format("Could not find the target point with id {}", match->_idInMap));
                    }
                    break;
                }
            case FeatureType::Point2d:
                {
                    const bool isOutlierRemoved = _localPoint2DMap.mark_feature_with_id_as_unmatched(match->_idInMap);
                    // If no points were found, this is bad. A match marked as outliers must be in the local map or
                    // staged points
                    if (not isOutlierRemoved)
                    {
                        outputs::log_error(
                                std::format("Could not find the target point2d with id {}", match->_idInMap));
                    }
                    break;
                }
            case FeatureType::Plane:
                {
                    const bool isOutlierRemoved = _localPlaneMap.mark_feature_with_id_as_unmatched(match->_idInMap);
                    // If no plane were found, this is bad. A match marked as outliers must be in the local map or
                    // staged plane
                    if (not isOutlierRemoved)
                    {
                        outputs::log_error(std::format("Could not find the target plane with id {}", match->_idInMap));
                    }
                    break;
                }
            default:
                {
                    outputs::log_error(std::format("The feature type {} is not handled in local map",
                                                   (int)match->get_feature_type()));
                    break;
                }
        }
    }
}

} // namespace rgbd_slam::map_management
