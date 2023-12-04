#include "map_point2d.hpp"

#include "coordinates/point_coordinates.hpp"
#include "logger.hpp"
#include "parameters.hpp"
#include "types.hpp"

namespace rgbd_slam::map_management {

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
    constexpr double searchSpaceRadius = parameters::matching::matchSearchRadius_px;
    constexpr double advancedSearchSpaceRadius = parameters::matching::matchSearchRadius_px * 2;
    const double searchRadius = useAdvancedSearch ? advancedSearchSpaceRadius : searchSpaceRadius;

    // try to match with tracking
    const int invalidfeatureIndex = features::keypoints::INVALID_MATCH_INDEX;
    int matchIndex = detectedFeatures.get_tracking_match_index(_id, isDetectedFeatureMatched);
    if (matchIndex == invalidfeatureIndex)
    {
        // No match: try to find match in a window around the point
        utils::ScreenCoordinate2D screenCoordinates;
        if (_coordinates.to_screen_coordinates(worldToCamera, screenCoordinates))
        {
            matchIndex = detectedFeatures.get_match_index(
                    screenCoordinates, _descriptor, isDetectedFeatureMatched, searchRadius);
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
        matches.emplace_back(detectedFeatures.get_keypoint(matchIndex).get_2D(), _coordinates, _covariance, _id);
    }
    return matchIndex;
}

bool MapPoint2D::add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                                TrackedPointsObject& trackedFeatures,
                                const uint dropChance) const noexcept
{
    // TODO: should those points be tracked ? I think not
    const bool shouldNotDropPoint = (dropChance == 0) or (utils::Random::get_random_uint(dropChance) != 0);

    if (shouldNotDropPoint)
    {
        utils::ScreenCoordinate2D screenCoordinates;
        if (_coordinates.to_screen_coordinates(worldToCamera, screenCoordinates))
        {
            trackedFeatures.add(_id, screenCoordinates.x(), screenCoordinates.y());
            return true;
        }
    }
    // point was not added
    return false;
}

void MapPoint2D::draw(const WorldToCameraMatrix& worldToCamMatrix,
                      cv::Mat& debugImage,
                      const cv::Scalar& color) const noexcept
{
    utils::ScreenCoordinate2D screenCoordinates;
    if (!_coordinates.to_screen_coordinates(worldToCamMatrix, screenCoordinates))
        return;

    // small blue circle around it
    cv::circle(debugImage,
               cv::Point(static_cast<int>(screenCoordinates.x()), static_cast<int>(screenCoordinates.y())),
               5,
               cv::Scalar(255, 0, 0),
               -1);
    cv::circle(debugImage,
               cv::Point(static_cast<int>(screenCoordinates.x()), static_cast<int>(screenCoordinates.y())),
               3,
               color,
               -1);
}

bool MapPoint2D::is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept
{
    (void)worldToCamMatrix;
    // screen point are always visible (by definition)
    return true;
}

void MapPoint2D::write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept { (void)mapWriter; }

bool MapPoint2D::compute_upgraded(const CameraToWorldMatrix& cameraToWorld,
                                  UpgradedPoint2DType& upgradedFeature) const noexcept
{
    Eigen::Matrix<double, 3, 6> jacobian;
    const utils::WorldCoordinate& cartesian = _coordinates.to_world_coordinates(jacobian);

    const vector3 hc(cartesian - cameraToWorld.translation());
    const double cosAlpha = static_cast<double>(_coordinates.get_bearing_vector().transpose() * hc) / hc.norm();
    const double thetad_meters =
            (sqrt(_covariance.diagonal()(PointInverseDepth::inverseDepthIndex)) / SQR(_coordinates._inverseDepth_mm)) /
            1000.0;
    const double d1_meters = hc.norm() / 1000.0;
    const double upgradeThreshold = 4.0 * thetad_meters / d1_meters * abs(cosAlpha);

    if (false and upgradeThreshold < 0.1) // 10% linearity index
    {
        try
        {
            upgradedFeature._covariance = compute_cartesian_covariance(_covariance, jacobian);
            upgradedFeature._coordinates = cartesian;
            upgradedFeature._descriptor = _descriptor;
            upgradedFeature._matchIndex = _matchIndex;
            return true;
        }
        catch (const std::exception& ex)
        {
            outputs::log_error("Catch exeption: " + std::string(ex.what()));
            return false;
        }
    }

    return false;
}

bool MapPoint2D::update_with_match(const DetectedPoint2DType& matchedFeature,
                                   const matrix33& poseCovariance,
                                   const CameraToWorldMatrix& cameraToWorld) noexcept
{
    if (_matchIndex < 0)
    {
        outputs::log_error("Tries to call the function update_with_match with no associated match");
        return false;
    }

    if (utils::is_depth_valid(matchedFeature._coordinates.z()))
    {
        // TODO: reactivate the following
        return false;

        // use the real observation, it will most likely overide the covariance inside the inverse depth point
        return track(matchedFeature._coordinates, cameraToWorld, poseCovariance, matchedFeature._descriptor);
    }
    // use a 2D observation, that will be merged with the current one
    return track(matchedFeature._coordinates.get_2D(), cameraToWorld, poseCovariance, matchedFeature._descriptor);
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
    MapPoint2D(detectedFeature._coordinates.get_2D(), cameraToWorld, poseCovariance, detectedFeature._descriptor)
{
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

LocalMapPoint2D::LocalMapPoint2D(const StagedMapPoint2D& stagedPoint) : MapPoint2D(stagedPoint, stagedPoint._id)
{
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