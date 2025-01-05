#include "map_point.hpp"

#include "coordinates/point_coordinates.hpp"
#include "covariances.hpp"
#include "logger.hpp"
#include "matches_containers.hpp"
#include "parameters.hpp"
#include "inverse_depth_with_tracking.hpp"
#include <memory>

namespace rgbd_slam::map_management {

/**
 * PointOptimizationFeature
 */

PointOptimizationFeature::PointOptimizationFeature(const ScreenCoordinate2D& matchedPoint,
                                                   const WorldCoordinate& mapPoint,
                                                   const matrix33& mapPointCovariance,
                                                   const size_t mapFeatureId,
                                                   const size_t detectedFeatureId) :
    matches_containers::IOptimizationFeature(mapFeatureId, detectedFeatureId),
    _matchedPoint(matchedPoint),
    _mapPoint(mapPoint),
    _mapPointCovariance(mapPointCovariance),
    _mapPointStandardDev(mapPointCovariance.diagonal().cwiseSqrt()) {};

static constexpr uint pointFeatureSize = 2;
size_t PointOptimizationFeature::get_feature_part_count() const noexcept { return pointFeatureSize; }

double PointOptimizationFeature::get_score() const noexcept
{
    static constexpr double optiScore = 1.0 / parameters::optimization::minimumPointForOptimization;
    return optiScore;
}

vectorxd PointOptimizationFeature::get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    // Compute retroprojected distance
    const vector2& distance = _mapPoint.get_signed_distance_2D_px(_matchedPoint, worldToCamera);
    return distance;
}

matrixd PointOptimizationFeature::get_distance_covariance(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    return utils::get_screen_2d_point_covariance(
                   _mapPoint, WorldCoordinateCovariance {_mapPointCovariance}, worldToCamera)
            .selfadjointView<Eigen::Lower>();
}

bool PointOptimizationFeature::is_inlier(const WorldToCameraMatrix& worldToCamera) const
{
    const double distance = _mapPoint.get_distance_px(_matchedPoint, worldToCamera);
    return distance <= 5;
}

double PointOptimizationFeature::get_alpha_reduction() const noexcept { return 1.0; }

FeatureType PointOptimizationFeature::get_feature_type() const noexcept { return FeatureType::Point; }

matrixd PointOptimizationFeature::get_world_covariance() const noexcept { return _mapPointCovariance; }

matches_containers::feat_ptr PointOptimizationFeature::get_variated_object() const noexcept
{
    // make random variation
    WorldCoordinate variatedCoordinates = _mapPoint;
    variatedCoordinates += utils::Random::get_normal_doubles<3>().cwiseProduct(_mapPointStandardDev);

    return std::make_shared<PointOptimizationFeature>(
            _matchedPoint, variatedCoordinates, _mapPointCovariance, _idInMap, _detectedFeatureId);
}

/**
 * MapPoint
 */

matchIndexSet MapPoint::find_match(const DetectedKeypointsObject& detectedFeatures,
                                   const WorldToCameraMatrix& worldToCamera,
                                   matches_containers::match_container& matches,
                                   const bool shouldAddToMatches,
                                   const bool useAdvancedSearch) const noexcept
{
    constexpr double searchSpaceRadius = parameters::matching::matchSearchRadius_px;
    constexpr double advancedSearchSpaceRadius = parameters::matching::matchSearchRadius_px * 2;
    const double searchRadius = useAdvancedSearch ? advancedSearchSpaceRadius : searchSpaceRadius;

    // try to match with tracking
    matchIndexSet matchIndexRes;
    int matchIndex = detectedFeatures.get_tracking_match_index(_id);
    if (matchIndex != features::keypoints::INVALID_MATCH_INDEX)
    {
        matchIndexRes.emplace(matchIndex);
    }
    else
    {
        // No match: try to find match in a window around the point
        ScreenCoordinate2D projectedMapPoint;
        const bool isScreenCoordinatesValid = _coordinates.to_screen_coordinates(worldToCamera, projectedMapPoint);
        if (isScreenCoordinatesValid)
        {
            matchIndexRes = detectedFeatures.get_match_index(projectedMapPoint, _descriptor, searchRadius);
        }
    }

    if (shouldAddToMatches)
    {
        for (const auto i: matchIndexRes)
        {
            matches.push_back(std::make_shared<PointOptimizationFeature>(
                    detectedFeatures.get_keypoint(i).get_2D(), _coordinates, _covariance, _id, i));
        }
    }
    return matchIndexRes;
}

bool MapPoint::add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                              TrackedPointsObject& trackedFeatures,
                              const uint dropChance) const noexcept
{
    std::ignore = worldToCamera;

    if (_lastMatch.has_value())
    {
        const bool shouldNotDropPoint = (dropChance == 0) or (utils::Random::get_random_uint(dropChance) != 0);

        if (shouldNotDropPoint)
        {
            // use previously known screen coordinates
            trackedFeatures.add(_id, _lastMatch->x(), _lastMatch->y());
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
    ScreenCoordinate screenPoint;
    const bool isCoordinatesValid = _coordinates.to_screen_coordinates(worldToCamMatrix, screenPoint);

    // do not display points behind the camera
    if (isCoordinatesValid and screenPoint.z() > 0 and screenPoint.is_in_screen_boundaries())
    {
        cv::circle(debugImage,
                   cv::Point(static_cast<int>(screenPoint.x()), static_cast<int>(screenPoint.y())),
                   3,
                   color,
                   -1);

        if (not is_matched())
        {
            // small red dot in the center
            cv::circle(debugImage,
                       cv::Point(static_cast<int>(screenPoint.x()), static_cast<int>(screenPoint.y())),
                       1,
                       cv::Scalar(0, 0, 255),
                       -1);
        }
    }
}

bool MapPoint::is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept
{
    if (ScreenCoordinate projectedScreenCoordinates;
        _coordinates.to_screen_coordinates(worldToCamMatrix, projectedScreenCoordinates))
    {
        return projectedScreenCoordinates.is_in_screen_boundaries();
    }
    return false;
}

void MapPoint::write_to_file(outputs::IMap_Writer* mapWriter) const noexcept
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
    if (_matchIndex.empty())
    {
        outputs::log_error("Tries to call the function update_with_match with no associated match");
        return false;
    }

    // set the last match
    _lastMatch = std::optional<ScreenCoordinate2D>(matchedFeature._coordinates.get_2D());

    const ScreenCoordinate& matchedScreenPoint = matchedFeature._coordinates;
    if (is_depth_valid(matchedScreenPoint.z()))
    {
        // transform screen point to world point
        const WorldCoordinate& worldPointCoordinates = matchedScreenPoint.to_world_coordinates(cameraToWorld);
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
        // Point is 2D, compute projection
        tracking::PointInverseDepth observation(
                ScreenCoordinate2D(matchedScreenPoint.head<2>()), cameraToWorld, poseCovariance);
        Eigen::Matrix<double, 3, 6> inverseToWorldJacobian;
        const auto projectedObservation = observation._coordinates.to_world_coordinates(inverseToWorldJacobian);
        const auto observationCovariance = tracking::PointInverseDepth::compute_cartesian_covariance(
                observation._covariance, inverseToWorldJacobian);

        // track the high uncertainty point
        const double mergeScore = track(projectedObservation, observationCovariance);
        if (mergeScore < 0)
            return false;

        // If a new descriptor is available, update it
        if (const cv::Mat& descriptor = matchedFeature._descriptor; not descriptor.empty())
            _descriptor = descriptor;

        return true;
    }
    return false;
}

bool MapPoint::merge(const MapPoint& other) noexcept
{
    try
    {
        const double mergeScore = track(other._coordinates, other._covariance);
        return mergeScore >= 0;
    }
    catch (...)
    {
        return false;
    }
}

void MapPoint::update_no_match() noexcept { _lastMatch.reset(); }

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
    // set the last match
    _lastMatch = std::optional<ScreenCoordinate2D>(detectedFeature._coordinates.get_2D());
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

LocalMapPoint::LocalMapPoint(const WorldCoordinate& coordinates,
                             const WorldCoordinateCovariance& covariance,
                             const cv::Mat& descriptor,
                             const matchIndexSet& matchIndex) :
    MapPoint(coordinates, covariance, descriptor)
{
    // new map point, new color
    set_color();

    _matchIndex = matchIndex;
    _successivMatchedCount = 1;
}

bool LocalMapPoint::is_lost() const noexcept
{
    return (_failedTrackingCount > parameters::mapping::pointUnmatchedCountToLoose);
}

} // namespace rgbd_slam::map_management