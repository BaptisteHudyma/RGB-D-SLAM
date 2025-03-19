#include "map_point.hpp"

#include "camera_transformation.hpp"
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
                                                   const vector3& mapPointStandardDev,
                                                   const size_t mapFeatureId,
                                                   const size_t detectedFeatureId) :
    matches_containers::IOptimizationFeature(mapFeatureId, detectedFeatureId),
    _matchedPoint(matchedPoint),
    _mapPoint(mapPoint),
    _mapPointStandardDev(mapPointStandardDev) {};

size_t PointOptimizationFeature::get_feature_part_count() const noexcept { return 2; }

double PointOptimizationFeature::get_score() const noexcept
{
    static constexpr double optiScore = 1.0 / parameters::optimization::minimumPointForOptimization;
    return optiScore;
}

bool PointOptimizationFeature::is_inlier(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    const double distance = _mapPoint.get_distance_px(_matchedPoint, worldToCamera);
    return distance <= parameters::optimization::ransac::maximumRetroprojectionErrorForPointInliers_px;
}

vectorxd PointOptimizationFeature::get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    // Compute retroprojected distance
    const auto& distance = _mapPoint.get_signed_distance_2D_px(_matchedPoint, worldToCamera);
    return distance;
}

double PointOptimizationFeature::get_alpha_reduction() const noexcept { return 1.0; }

matches_containers::feat_ptr PointOptimizationFeature::compute_random_variation() const noexcept
{
    // make random variation
    WorldCoordinate variatedCoordinates = _mapPoint;
    variatedCoordinates += utils::Random::get_normal_doubles<3>().cwiseProduct(_mapPointStandardDev);

    return std::make_shared<PointOptimizationFeature>(
            _matchedPoint, variatedCoordinates, _mapPointStandardDev, _idInMap, _detectedFeatureId);
}

bool PointOptimizationFeature::is_valid() const noexcept
{
    return (not _matchedPoint.hasNaN()) and (not _mapPoint.hasNaN()) and (not _mapPointStandardDev.hasNaN()) and
           (_mapPointStandardDev.array() >= 0).all();
}

FeatureType PointOptimizationFeature::get_feature_type() const noexcept { return FeatureType::Point; }

/**
 * MapPoint
 */

matchIndexSet MapPoint::find_matches(const DetectedKeypointsObject& detectedFeatures,
                                     const WorldToCameraMatrix& worldToCamera,
                                     const vectorb& isDetectedFeatureMatched,
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
            // TODO: add multiple match support
            matchIndexRes = detectedFeatures.get_match_indexes(
                    projectedMapPoint, _descriptor, isDetectedFeatureMatched, searchRadius);
        }
    }

    if (shouldAddToMatches)
    {
        for (const auto i: matchIndexRes)
        {
            matches.push_back(std::make_shared<PointOptimizationFeature>(detectedFeatures.get_keypoint(i).get_2D(),
                                                                         _coordinates,
                                                                         _covariance.diagonal().cwiseSqrt(),
                                                                         _id,
                                                                         i));
        }
    }
    return matchIndexRes;
}

bool MapPoint::add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                              TrackedPointsObject& trackedFeatures,
                              const uint dropChance) const noexcept
{
    const bool shouldNotDropPoint = (dropChance == 0) or (utils::Random::get_random_uint(dropChance) != 0);

    assert(not _coordinates.hasNaN());
    if (shouldNotDropPoint)
    {
        ScreenCoordinate2D screenCoordinates;
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

void MapPoint::write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept
{
    if (mapWriter != nullptr)
    {
        // only write in confident points
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
    if (_matchIndexes.empty())
    {
        outputs::log_error("Tries to call the function update_with_match with no associated match");
        return false;
    }

    const ScreenCoordinate& matchedScreenPoint = matchedFeature._coordinates;
    WorldToCameraMatrix w2c = utils::compute_world_to_camera_transform(cameraToWorld);
    if (is_depth_valid(matchedScreenPoint.z()))
    {
        // depth is valid, merge using the 3D model (more precise)
        if (not track_3d(matchedScreenPoint, w2c))
            return false;
    }
    // depth is invalid, merge using the 2D model (slightly faster)
    else if (not track_2d(matchedScreenPoint.get_2D(), w2c))
        return false;

    // If a new descriptor is available, update it
    if (const cv::Mat& descriptor = matchedFeature._descriptor; not descriptor.empty())
        _descriptor = descriptor;

    return true;
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

    _matchIndexes = stagedPoint._matchIndexes;
    _successivMatchedCount = stagedPoint._successivMatchedCount;
}

LocalMapPoint::LocalMapPoint(const WorldCoordinate& coordinates,
                             const WorldCoordinateCovariance& covariance,
                             const cv::Mat& descriptor,
                             const matchIndexSet& matchIndexes) :
    MapPoint(coordinates, covariance, descriptor)
{
    // new map point, new color
    set_color();

    _matchIndexes = matchIndexes;
    _successivMatchedCount = 1;
}

bool LocalMapPoint::is_lost() const noexcept
{
    return (_failedTrackingCount > parameters::mapping::pointUnmatchedCountToLoose);
}

} // namespace rgbd_slam::map_management