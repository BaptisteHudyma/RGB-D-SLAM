#include "map_point2d.hpp"
#include "camera_transformation.hpp"
#include "coordinates.hpp"
#include "logger.hpp"
#include "parameters.hpp"
#include "triangulation.hpp"
#include <exception>

namespace rgbd_slam::map_management {

/**
 * Point
 */

Point2D::Point2D(const utils::ScreenCoordinate2D& coordinates,
                 const ScreenCoordinate2DCovariance& covariance,
                 const cv::Mat& descriptor) :
    _coordinates(coordinates),
    _descriptor(descriptor),
    _covariance(covariance)
{
    build_kalman_filter();

    assert(not _descriptor.empty() and _descriptor.cols > 0);
    assert(not _coordinates.hasNaN());
};

double Point2D::track(const utils::ScreenCoordinate2D& newDetectionCoordinates,
                      const matrix22& newDetectionCovariance) noexcept
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

void Point2D::build_kalman_filter() noexcept
{
    if (_kalmanFilter == nullptr)
    {
        const matrix22 systemDynamics = matrix22::Identity(); // points are not supposed to move, so no dynamics
        const matrix22 outputMatrix = matrix22::Identity();   // we need all positions

        const double parametersProcessNoise = 0; // TODO set in parameters
        const matrix22 processNoiseCovariance =
                matrix22::Identity() * parametersProcessNoise; // Process noise covariance

        _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter<2, 2>>(
                systemDynamics, outputMatrix, processNoiseCovariance);
    }
}

/**
 * MapPoint
 */

int MapPoint2D::find_match(const DetectedKeypointsObject& detectedFeatures,
                           const WorldToCameraMatrix& worldToCamera,
                           const vectorb& isDetectedFeatureMatched,
                           std::list<PointMatch2DType>& matches,
                           const bool shouldAddToMatches,
                           const bool useAdvancedSearch) const noexcept
{
    // no need for 2D points : already in screen space
    (void)worldToCamera;

    constexpr double searchSpaceRadius = parameters::matching::matchSearchRadius_px;
    constexpr double advancedSearchSpaceRadius = parameters::matching::matchSearchRadius_px * 2;
    const double searchRadius = useAdvancedSearch ? advancedSearchSpaceRadius : searchSpaceRadius;

    // try to match with tracking
    const int invalidfeatureIndex = features::keypoints::INVALID_MATCH_INDEX;
    int matchIndex = detectedFeatures.get_tracking_match_index(_id, isDetectedFeatureMatched);
    if (matchIndex == invalidfeatureIndex)
    {
        // No match: try to find match in a window around the point
        matchIndex =
                detectedFeatures.get_match_index(_coordinates, _descriptor, isDetectedFeatureMatched, searchRadius);
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

bool MapPoint2D::add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                                TrackedPointsObject& trackedFeatures,
                                const uint dropChance) const noexcept
{
    (void)worldToCamera;
    const bool shouldNotDropPoint = (dropChance == 0) or (utils::Random::get_random_uint(dropChance) != 0);

    assert(not _coordinates.hasNaN());
    if (shouldNotDropPoint)
    {
        // use previously known screen coordinates
        trackedFeatures.add(_id, _coordinates.x(), _coordinates.y());

        return true;
    }
    // point was not added
    return false;
}

void MapPoint2D::draw(const WorldToCameraMatrix& worldToCamMatrix,
                      cv::Mat& debugImage,
                      const cv::Scalar& color) const noexcept
{
    (void)worldToCamMatrix;

    // small blue circle around it
    cv::circle(debugImage,
               cv::Point(static_cast<int>(_coordinates.x()), static_cast<int>(_coordinates.y())),
               5,
               cv::Scalar(255, 0, 0),
               -1);
    cv::circle(debugImage,
               cv::Point(static_cast<int>(_coordinates.x()), static_cast<int>(_coordinates.y())),
               3,
               color,
               -1);
}

bool MapPoint2D::is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept
{
    (void)worldToCamMatrix;
    return _coordinates.is_in_screen_boundaries();
}

void MapPoint2D::write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept { (void)mapWriter; }

bool MapPoint2D::compute_upgraded(UpgradedPoint2DType& upgradedFeature) const noexcept
{
    // invalid depth: try to triangulate
    if (not utils::is_depth_valid(_lastMatchCoordinates.z()))
    {
        utils::WorldCoordinate triangulatedPoint;
        if (tracking::Triangulation::triangulate(_firstWorldToCam,
                                                 _lastMatchWorldToCamera,
                                                 _coordinates,
                                                 _lastMatchCoordinates.get_2D(),
                                                 triangulatedPoint))
        {
            upgradedFeature = triangulatedPoint;
            return true;
        }
        else
        {
            // not possible to triangulate, wait a bit
            return false;
        }
    }
    else
    {
        // depth is valid: compute world position
        upgradedFeature = _lastMatchCoordinates.to_world_coordinates(
                utils::compute_camera_to_world_transform(_lastMatchWorldToCamera));
        return true;
    }
    return false;
}

bool MapPoint2D::update_with_match(const DetectedPoint2DType& matchedFeature,
                                   const matrix33& poseCovariance,
                                   const CameraToWorldMatrix& cameraToWorld) noexcept
{
    (void)poseCovariance;
    if (_matchIndex < 0)
    {
        outputs::log_error("Tries to call the function update_with_match with no associated match");
        return false;
    }

    _lastMatchCoordinates = matchedFeature._coordinates;
    _lastMatchWorldToCamera = utils::compute_world_to_camera_transform(cameraToWorld);
    return true;
}

void MapPoint2D::update_no_match() noexcept
{
    // do nothing
}

/**
 * StagedMapPoint
 */

StagedMapPoint2D::StagedMapPoint2D(const matrix33& poseCovariance,
                                   const CameraToWorldMatrix& cameraToWorld,
                                   const DetectedPoint2DType& detectedFeature) :
    MapPoint2D(detectedFeature._coordinates.get_2D(),
               ScreenCoordinate2DCovariance(matrix22::Identity()),
               detectedFeature._descriptor)
{
    (void)poseCovariance;
    _firstWorldToCam = utils::compute_world_to_camera_transform(cameraToWorld);
}

bool StagedMapPoint2D::should_remove_from_staged() const noexcept { return get_confidence() <= 0; }

bool StagedMapPoint2D::should_add_to_local_map() const noexcept
{
    constexpr double minimumConfidenceForLocalMap = parameters::mapping::pointMinimumConfidenceForMap;
    return (get_confidence() > minimumConfidenceForLocalMap);
}

double StagedMapPoint2D::get_confidence() const noexcept
{
    constexpr double oneOverStagedPointconfidence =
            1.0 / static_cast<double>(parameters::mapping::pointStagedAgeConfidence);
    const double confidence = static_cast<double>(_successivMatchedCount) * oneOverStagedPointconfidence;
    return std::clamp(confidence, -1.0, 1.0);
}

/**
 * LocalMapPoint
 */

LocalMapPoint2D::LocalMapPoint2D(const StagedMapPoint2D& stagedPoint) :
    MapPoint2D(stagedPoint._coordinates, stagedPoint._covariance, stagedPoint._descriptor, stagedPoint._id)
{
    _firstWorldToCam = stagedPoint._firstWorldToCam;

    // new map point, new color
    set_color();

    _matchIndex = stagedPoint._matchIndex;
    _successivMatchedCount = stagedPoint._successivMatchedCount;
}

bool LocalMapPoint2D::is_lost() const noexcept
{
    return (_failedTrackingCount > parameters::mapping::pointUnmatchedCountToLoose);
}

} // namespace rgbd_slam::map_management