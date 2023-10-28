#include "map_point.hpp"
#include "coordinates.hpp"
#include "logger.hpp"
#include "parameters.hpp"
#include <exception>
#include <stdexcept>

namespace rgbd_slam::map_management {

/**
 * Point
 */

Point::Point(const utils::WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor) :
    _coordinates(coordinates),
    _descriptor(descriptor),
    _covariance(covariance)
{
    build_kalman_filter();

    assert(not _descriptor.empty() and _descriptor.cols > 0);
    assert(not _coordinates.hasNaN());
};

double Point::track(const utils::WorldCoordinate& newDetectionCoordinates,
                    const matrix33& newDetectionCovariance) noexcept
{
    assert(_kalmanFilter != nullptr);
    if (not utils::is_covariance_valid(newDetectionCovariance))
    {
        outputs::log_error("newDetectionCovariance: the covariance in invalid");
        return -1;
    }
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance: the covariance in invalid");
        return -1;
    }

    try
    {
        const auto& [newCoordinates, newCovariance] = _kalmanFilter->get_new_state(
                _coordinates, _covariance, newDetectionCoordinates, newDetectionCovariance);

        const double score = (_coordinates - newCoordinates).norm();

        _coordinates << newCoordinates;
        _covariance << newCovariance;
        assert(not _coordinates.hasNaN());
        return score;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exeption: " + std::string(ex.what()));
        return -1;
    }
}

void Point::build_kalman_filter() noexcept
{
    if (_kalmanFilter == nullptr)
    {
        const matrix33 systemDynamics = matrix33::Identity(); // points are not supposed to move, so no dynamics
        const matrix33 outputMatrix = matrix33::Identity();   // we need all positions

        const double parametersProcessNoise = 0; // TODO set in parameters
        const matrix33 processNoiseCovariance =
                matrix33::Identity() * parametersProcessNoise; // Process noise covariance

        _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter<3, 3>>(
                systemDynamics, outputMatrix, processNoiseCovariance);
    }
}

/**
 * MapPoint
 */

int MapPoint::find_match(const DetectedKeypointsObject& detectedFeatures,
                         const WorldToCameraMatrix& worldToCamera,
                         const vectorb& isDetectedFeatureMatched,
                         std::list<PointMatchType>& matches,
                         const bool shouldAddToMatches,
                         const bool useAdvancedSearch) const noexcept
{
    constexpr double searchSpaceRadius = parameters::matching::matchSearchRadius_px;
    constexpr double advancedSearchSpaceRadius = parameters::matching::matchSearchRadius_px * 2;
    const double searchRadius = useAdvancedSearch ? advancedSearchSpaceRadius : searchSpaceRadius;

    // try to match with tracking
    const int invalidfeatureIndex = features::keypoints::INVALID_MATCH_INDEX;
    int matchIndex = detectedFeatures.get_tracking_match_index(_id, isDetectedFeatureMatched);
    if (matchIndex == invalidfeatureIndex)
    {
        // No match: try to find match in a window around the point
        utils::ScreenCoordinate2D projectedMapPoint;
        const bool isScreenCoordinatesValid = _coordinates.to_screen_coordinates(worldToCamera, projectedMapPoint);
        if (isScreenCoordinatesValid)
        {
            matchIndex = detectedFeatures.get_match_index(
                    projectedMapPoint, _descriptor, isDetectedFeatureMatched, searchRadius);
        }
    }

    if (matchIndex == invalidfeatureIndex)
    {
        // unmatched point
        return UNMATCHED_FEATURE_INDEX;
    }

    assert(matchIndex >= 0);
    assert(static_cast<Eigen::Index>(matchIndex) < isDetectedFeatureMatched.size());
    if (isDetectedFeatureMatched[matchIndex])
    {
        // point was already matched
        outputs::log_error("The requested point unique index is already matched");
    }

    if (shouldAddToMatches)
    {
        matches.emplace_back(
                detectedFeatures.get_keypoint(matchIndex).get_2D(), _coordinates, _covariance.diagonal(), _id);
    }
    return matchIndex;
}

bool MapPoint::add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                              TrackedPointsObject& trackedFeatures,
                              const uint dropChance) const noexcept
{
    const bool shouldNotDropPoint = (dropChance == 0) or (utils::Random::get_random_uint(dropChance) != 0);

    assert(not _coordinates.hasNaN());
    if (shouldNotDropPoint)
    {
        utils::ScreenCoordinate2D screenCoordinates;
        if (_coordinates.to_screen_coordinates(worldToCamera, screenCoordinates))
        {
            // use previously known screen coordinates
            trackedFeatures.add(_id, screenCoordinates.x(), screenCoordinates.y());

            return true;
        }
    }
    // point was not added
    return false;
}

void MapPoint::draw(const WorldToCameraMatrix& worldToCamMatrix,
                    cv::Mat& debugImage,
                    const cv::Scalar& color) const noexcept
{
    utils::ScreenCoordinate screenPoint;
    const bool isCoordinatesValid = _coordinates.to_screen_coordinates(worldToCamMatrix, screenPoint);

    // do not display points behind the camera
    if (isCoordinatesValid and screenPoint.z() > 0)
    {
        cv::circle(debugImage,
                   cv::Point(static_cast<int>(screenPoint.x()), static_cast<int>(screenPoint.y())),
                   3,
                   color,
                   -1);
    }
}

bool MapPoint::is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept
{
    if (utils::ScreenCoordinate projectedScreenCoordinates;
        _coordinates.to_screen_coordinates(worldToCamMatrix, projectedScreenCoordinates))
    {
        return projectedScreenCoordinates.is_in_screen_boundaries();
    }
    return false;
}

void MapPoint::write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept
{
    if (mapWriter != nullptr)
    {
        mapWriter->add_point(_coordinates);
    }
    else
    {
        outputs::log_error("mapWriter is null");
    }
}

bool MapPoint::update_with_match(const DetectedPointType& matchedFeature,
                                 const matrix33& poseCovariance,
                                 const CameraToWorldMatrix& cameraToWorld) noexcept
{
    if (_matchIndex < 0)
    {
        outputs::log_error("Tries to call the function update_with_match with no associated match");
        return false;
    }

    if (const utils::ScreenCoordinate& matchedScreenPoint = matchedFeature._coordinates;
        utils::is_depth_valid(matchedScreenPoint.z()))
    {
        // transform screen point to world point
        const utils::WorldCoordinate& worldPointCoordinates = matchedScreenPoint.to_world_coordinates(cameraToWorld);
        // get a measure of the estimated variance of the new world point
        const matrix33& worldCovariance =
                utils::get_world_point_covariance(matchedScreenPoint, cameraToWorld, poseCovariance);
        // update this map point errors & position
        const double mergeScore = track(worldPointCoordinates, worldCovariance);
        if (mergeScore < 0)
            return false;

        // If a new descriptor is available, update it
        if (const cv::Mat& descriptor = matchedFeature._descriptor; not descriptor.empty())
            _descriptor = descriptor;

        return true;
    }
    else
    {
        // TODO: Point is 2D, handle separatly
    }
    return false;
}

void MapPoint::update_no_match() noexcept
{
    // do nothing
}

/**
 * StagedMapPoint
 */

StagedMapPoint::StagedMapPoint(const matrix33& poseCovariance,
                               const CameraToWorldMatrix& cameraToWorld,
                               const DetectedPointType& detectedFeature) :
    MapPoint(detectedFeature._coordinates.to_world_coordinates(cameraToWorld),
             utils::get_world_point_covariance(detectedFeature._coordinates, cameraToWorld, poseCovariance),
             detectedFeature._descriptor)
{
}

bool StagedMapPoint::should_remove_from_staged() const noexcept { return get_confidence() <= 0; }

bool StagedMapPoint::should_add_to_local_map() const noexcept
{
    constexpr double minimumConfidenceForLocalMap = parameters::mapping::pointMinimumConfidenceForMap;
    return (get_confidence() > minimumConfidenceForLocalMap);
}

double StagedMapPoint::get_confidence() const noexcept
{
    constexpr double oneOverStagedPointconfidence =
            1.0 / static_cast<double>(parameters::mapping::pointStagedAgeConfidence);
    const double confidence = static_cast<double>(_successivMatchedCount) * oneOverStagedPointconfidence;
    return std::clamp(confidence, -1.0, 1.0);
}

/**
 * LocalMapPoint
 */

LocalMapPoint::LocalMapPoint(const StagedMapPoint& stagedPoint) :
    MapPoint(stagedPoint._coordinates, stagedPoint._covariance, stagedPoint._descriptor, stagedPoint._id)
{
    // new map point, new color
    set_color();

    _matchIndex = stagedPoint._matchIndex;
    _successivMatchedCount = stagedPoint._successivMatchedCount;
}

bool LocalMapPoint::is_lost() const noexcept
{
    return (_failedTrackingCount > parameters::mapping::pointUnmatchedCountToLoose);
}

} // namespace rgbd_slam::map_management