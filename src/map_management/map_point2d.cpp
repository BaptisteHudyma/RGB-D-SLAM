#include "map_point2d.hpp"
#include "camera_transformation.hpp"
#include "covariances.hpp"
#include "logger.hpp"
#include "parameters.hpp"
#include "triangulation.hpp"
#include "types.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <exception>

namespace rgbd_slam::map_management {

PointInverseDepth::PointInverseDepth(const utils::ScreenCoordinate2D& observation,
                                     const CameraToWorldMatrix& c2w,
                                     const matrix33& stateCovariance) :
    _coordinates(observation, c2w)
{
    _covariance.block<3, 3>(0, 0) = stateCovariance;
    _covariance(3, 3) = 0.1;       // theta angle covariance
    _covariance(4, 4) = 0.1;       // phi angle covariance
    _covariance(5, 5) = 0.1 * 0.1; // depth covariance

    assert(utils::is_covariance_valid(_covariance));
}

PointInverseDepth::PointInverseDepth(const utils::CameraCoordinate& cameraCoordinates,
                                     const CameraCoordinateCovariance& cameraCovariance,
                                     const CameraToWorldMatrix& c2w) :
    _coordinates(cameraCoordinates, c2w)
{
    // TODO: covariance projection

    assert(utils::is_covariance_valid(_covariance));
}

bool PointInverseDepth::add_observation(const utils::ScreenCoordinate2D& screenObservation,
                                        const CameraToWorldMatrix& c2w,
                                        const matrix33& stateCovariance)
{
    // get observation in world space
    const PointInverseDepth newObservation(screenObservation, c2w, stateCovariance);

    // project to this point coordinates to perform the fusion
    const WorldToCameraMatrix& w2c = utils::compute_world_to_camera_transform(c2w);
    const PointInverseDepth observationInSameSpace(newObservation._coordinates.get_camera_coordinates(w2c),
                                                   newObservation.get_camera_coordinate_variance(w2c),
                                                   _coordinates._c2w);

    assert(_kalmanFilter != nullptr);
    const auto& [newState, newCovariance] =
            _kalmanFilter->get_new_state(_coordinates.get_vector_state(),
                                         _covariance,
                                         observationInSameSpace._coordinates.get_vector_state(),
                                         observationInSameSpace._covariance);

    _coordinates.from_vector_state(newState);
    _covariance = newCovariance;
    assert(utils::is_covariance_valid(_covariance));

    return true;
}

CameraCoordinateCovariance PointInverseDepth::get_camera_coordinate_variance(
        const WorldToCameraMatrix& w2c) const noexcept
{
    const WorldCoordinateCovariance& projectedCovariance = get_cartesian_covariance();

    CameraCoordinateCovariance c;
    // TODO: jacobian of the transform
    return c;
}

ScreenCoordinateCovariance PointInverseDepth::get_screen_coordinate_variance(
        const WorldToCameraMatrix& w2c) const noexcept
{
    return utils::get_screen_point_covariance(_coordinates.get_camera_coordinates(w2c),
                                              get_camera_coordinate_variance(w2c));
}

WorldCoordinateCovariance PointInverseDepth::get_cartesian_covariance() const noexcept
{
    // jacobian of the get_cartesian() operation
    using matrix36 = Eigen::Matrix<double, 3, 6>;
    matrix36 jacobian(matrix36::Zero());
    jacobian.block<3, 3>(0, 0) = matrix33::Identity();

    // compute sin and cos in advance (opti)
    const double cosTheta = cos(_coordinates._theta_rad);
    const double sinTheta = sin(_coordinates._theta_rad);
    const double cosPhi = cos(_coordinates._phi_rad);
    const double sinPhi = sin(_coordinates._phi_rad);

    const double depth = 1.0 / _coordinates._inverseDepth_mm;
    const double theta1 = cosPhi * sinPhi;
    const double theta2 = cosPhi * cosTheta;
    jacobian.block<3, 3>(0, 3) = matrix33({{theta2 * depth, -sinTheta * sinPhi * depth, -theta1},
                                           {-cosTheta * depth, 0, sinTheta},
                                           {-theta1 * depth, -cosTheta * sinPhi * depth, -theta2}});

    WorldCoordinateCovariance worldCovariance(jacobian * _covariance * jacobian.transpose());
    assert(utils::is_covariance_valid(worldCovariance));
    return worldCovariance;
}

void PointInverseDepth::build_kalman_filter() noexcept
{
    if (_kalmanFilter == nullptr)
    {
        const matrix66 systemDynamics = matrix66::Identity(); // points are not supposed to move, so no dynamics
        const matrix66 outputMatrix = matrix66::Identity();   // we need all positions

        const double parametersProcessNoise = 0; // TODO set in parameters
        const matrix66 processNoiseCovariance =
                matrix66::Identity() * parametersProcessNoise; // Process noise covariance

        _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter<6, 6>>(
                systemDynamics, outputMatrix, processNoiseCovariance);
    }
}

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
    assert(not _descriptor.empty() and _descriptor.cols > 0);
};

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
        if (_isLastMatchCoordinatesSet)
            matchIndex = detectedFeatures.get_match_index(
                    _lastMatchCoordinates.get_2D(), _descriptor, isDetectedFeatureMatched, searchRadius);
        else
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
        if (_isLastMatchCoordinatesSet)
            trackedFeatures.add(_id, _lastMatchCoordinates.x(), _lastMatchCoordinates.y());
        else
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

    utils::ScreenCoordinate2D temp;
    if (_isLastMatchCoordinatesSet)
    {
        temp.x() = _lastMatchCoordinates.x();
        temp.y() = _lastMatchCoordinates.y();
    }
    else
    {
        temp.x() = _coordinates.x();
        temp.y() = _coordinates.y();
    }

    // small blue circle around it
    cv::circle(debugImage,
               cv::Point(static_cast<int>(temp.x()), static_cast<int>(temp.y())),
               5,
               cv::Scalar(255, 0, 0),
               -1);
    cv::circle(debugImage, cv::Point(static_cast<int>(temp.x()), static_cast<int>(temp.y())), 3, color, -1);
}

bool MapPoint2D::is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept
{
    (void)worldToCamMatrix;
    // screen point are always visible (by definition)
    return true;
}

void MapPoint2D::write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept { (void)mapWriter; }

bool MapPoint2D::compute_upgraded(const matrix33& poseCovariance, UpgradedPoint2DType& upgradedFeature) const noexcept
{
    if (not _isLastMatchCoordinatesSet)
        return false;

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
            // compute the new point covariance
            const auto& cameraPoint = triangulatedPoint.to_camera_coordinates(_lastMatchWorldToCamera);
            // point is too far ! invalid triangulation
            if (cameraPoint.z() > 5000)
                return false;

            ScreenCoordinateCovariance screenPointCovariance(ScreenCoordinateCovariance::Zero());
            screenPointCovariance.block<2, 2>(0, 0) = _covariance;
            screenPointCovariance(2, 2) = 100 * 100; // big covariance for depth
            const CameraCoordinateCovariance& projectedCovariance =
                    utils::get_camera_point_covariance(cameraPoint, screenPointCovariance);

            // set the new feature parameters
            upgradedFeature._coordinates = triangulatedPoint;
            upgradedFeature._covariance << utils::get_world_point_covariance(
                    projectedCovariance,
                    utils::compute_camera_to_world_transform(_lastMatchWorldToCamera),
                    poseCovariance);
            upgradedFeature._descriptor = _descriptor;
            upgradedFeature._matchIndex = _matchIndex;
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
        const auto& cameraToWorld = utils::compute_camera_to_world_transform(_lastMatchWorldToCamera);
        // depth is valid: compute world position
        upgradedFeature._coordinates = _lastMatchCoordinates.to_world_coordinates(cameraToWorld);
        upgradedFeature._covariance << utils::get_world_point_covariance(
                _lastMatchCoordinates, cameraToWorld, poseCovariance);
        upgradedFeature._descriptor = _descriptor;
        upgradedFeature._matchIndex = _matchIndex;
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
    if (not matchedFeature._descriptor.empty())
        _descriptor = matchedFeature._descriptor;
    _isLastMatchCoordinatesSet = true;
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
    _lastMatchWorldToCamera = _firstWorldToCam;
    _lastMatchCoordinates = detectedFeature._coordinates;
    _isLastMatchCoordinatesSet = true;
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