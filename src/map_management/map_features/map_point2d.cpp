#include "map_point2d.hpp"

#include "coordinates/inverse_depth_coordinates.hpp"
#include "coordinates/point_coordinates.hpp"
#include "line.hpp"
#include "logger.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <unsupported/Eigen/src/NumericalDiff/NumericalDiff.h>

namespace rgbd_slam::map_management {

/**
 * Point2dOptimizationFeature
 */

Point2dOptimizationFeature::Point2dOptimizationFeature(
        const ScreenCoordinate2D& matchedPoint,
        const InverseDepthWorldPoint& mapPoint,
        const tracking::PointInverseDepth::Covariance& mapPointCovariance,
        const size_t mapFeatureId) :
    matches_containers::IOptimizationFeature(mapFeatureId),
    _matchedPoint(matchedPoint),
    _mapPoint(mapPoint),
    _mapPointCovariance(mapPointCovariance),
    _mapPointStandardDev(mapPointCovariance.diagonal().cwiseSqrt()) {};

size_t Point2dOptimizationFeature::get_feature_part_count() const noexcept { return 2; }

double Point2dOptimizationFeature::get_score() const noexcept
{
    static constexpr double optiScore = 1.0 / parameters::optimization::minimumPoint2dForOptimization;
    return optiScore;
}

vectorxd Point2dOptimizationFeature::get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    const vector2& distance = _mapPoint.compute_signed_screen_distance(
            _matchedPoint, _mapPointCovariance.get_inverse_depth_variance(), worldToCamera);
    return distance;
}

matrixd Point2dOptimizationFeature::get_distance_covariance(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    const auto& jac = get_distance_jacobian(worldToCamera);
    return (jac * _mapPointCovariance.selfadjointView<Eigen::Lower>() * jac.transpose())
            .selfadjointView<Eigen::Lower>();
}

matrixd Point2dOptimizationFeature::get_distance_jacobian(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    // TODO: use a real jacobian instead of a computed one
    struct Func
    {
        Func(const WorldToCameraMatrix& worldToCamera, const double iDepthStd, const ScreenCoordinate2D& matchedPoint) :
            _worldToCamera(worldToCamera),
            _iDepthStd(iDepthStd),
            _matchedPoint(matchedPoint)
        {
        }

        enum
        {
            InputsAtCompileTime = 6,
            ValuesAtCompileTime = 2
        };

        uint values() const { return 2; }
        uint inputs() const { return 6; }

        // typedefs for the original functor
        using Scalar = double;
        using InputType = Eigen::Matrix<Scalar, InputsAtCompileTime, 1>;
        using ValueType = Eigen::Matrix<Scalar, ValuesAtCompileTime, 1>;
        using JacobianType = Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>;

        int operator()(const InputType& inputs, ValueType& outputs) const
        {
            outputs = InverseDepthWorldPoint(WorldCoordinate(inputs.head<3>()), inputs(3), inputs(4), inputs(5))
                              .compute_signed_screen_distance(_matchedPoint, _iDepthStd, _worldToCamera);
            return 0;
        }

        const WorldToCameraMatrix _worldToCamera;
        const double _iDepthStd;
        const ScreenCoordinate2D _matchedPoint;
    };

    Func distance(worldToCamera, _mapPointStandardDev(InverseDepthWorldPoint::inverseDepthIndex), _matchedPoint);
    Eigen::NumericalDiff<Func, Eigen::Central> distanceFunctor(distance);

    Func::JacobianType jacobian;
    distanceFunctor.df(_mapPoint.get_vector(), jacobian);
    return jacobian;
}

double Point2dOptimizationFeature::get_alpha_reduction() const noexcept { return 0.3; }

FeatureType Point2dOptimizationFeature::get_feature_type() const noexcept { return FeatureType::Point2d; }

matrixd Point2dOptimizationFeature::get_world_covariance() const noexcept { return _mapPointCovariance; }

/**
 * MapPoint
 */

int MapPoint2D::find_match(const DetectedKeypointsObject& detectedFeatures,
                           const WorldToCameraMatrix& worldToCamera,
                           const vectorb& isDetectedFeatureMatched,
                           matches_containers::match_container& matches,
                           const bool shouldAddToMatches,
                           const bool useAdvancedSearch) const noexcept
{
    assert(not _descriptor.empty());
    constexpr double searchSpaceRadius = parameters::matching::matchSearchRadius_px;
    constexpr double advancedSearchSpaceRadius = parameters::matching::matchSearchRadius_px * 2;
    const double searchRadius = useAdvancedSearch ? advancedSearchSpaceRadius : searchSpaceRadius;

    // try to match with tracking
    int matchIndex = detectedFeatures.get_tracking_match_index(_id, isDetectedFeatureMatched);
    if (matchIndex == features::keypoints::INVALID_MATCH_INDEX)
    {
        // No match: try to find match in a window around the point
        ScreenCoordinate2D screenCoordinates;
        if (_coordinates.to_world_coordinates().to_screen_coordinates(worldToCamera, screenCoordinates))
        {
            // TODO use a real match to 2D function, this one will fail for 2D points
            matchIndex = detectedFeatures.get_match_index(
                    screenCoordinates, _descriptor, isDetectedFeatureMatched, searchRadius);
        }
    }

    if (matchIndex == features::keypoints::INVALID_MATCH_INDEX)
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
        matches.push_back(std::make_shared<Point2dOptimizationFeature>(
                detectedFeatures.get_keypoint(matchIndex).get_2D(), _coordinates, _covariance, _id));
    }
    return matchIndex;
}

bool MapPoint2D::add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                                TrackedPointsObject& trackedFeatures,
                                const uint dropChance) const noexcept
{
    std::ignore = worldToCamera;
    std::ignore = trackedFeatures;
    std::ignore = dropChance;
    // do not track inverse depth points, it gives incorrect triangulation
    return false;
}

void MapPoint2D::draw(const WorldToCameraMatrix& worldToCamMatrix,
                      cv::Mat& debugImage,
                      const cv::Scalar& color) const noexcept
{
    utils::Segment<2> lineInScreen;
    if (not to_screen_coordinates(worldToCamMatrix, lineInScreen))
        return;

    // clamp the line to screen space (prevent problems when drawing)
    utils::Segment<2> screenCoordinates;
    if (not utils::clamp_to_screen(lineInScreen, screenCoordinates))
    {
        return;
    }

    const ScreenCoordinate2D startPoint(screenCoordinates.get_start_point());
    const ScreenCoordinate2D endPoint(screenCoordinates.get_end_point());

    // prevent a display out of the screen (visual bug)
    if (startPoint.is_in_screen_boundaries() and endPoint.is_in_screen_boundaries())
    {
        const cv::Point p1(static_cast<int>(startPoint.x()), static_cast<int>(startPoint.y()));
        const cv::Point p2(static_cast<int>(endPoint.x()), static_cast<int>(endPoint.y()));

        // if it's matched, display blue around it, else display red
        cv::line(debugImage, p1, p2, is_matched() ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), 5);

        cv::line(debugImage, p1, p2, color, 3);
    }
    else
    {
        outputs::log_error("Cannot draw a line out of screen boundaries");
    }
}

bool MapPoint2D::is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept
{
    // Those points should laways be visible but we never know
    utils::Segment<2> screenSegment;
    if (_coordinates.to_screen_coordinates(worldToCamMatrix, _covariance.get_inverse_depth_variance(), screenSegment))
    {
        // clamp to screen: if this fails, point is not visible
        utils::Segment<2> screenCoordinates;
        return utils::clamp_to_screen(screenSegment, screenCoordinates);
    }
    return false;
}

void MapPoint2D::write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept
{
    const double inverseDepthStandardDev = sqrt(_covariance.get_inverse_depth_variance());

    // TODO find a way to add 2D points without breaking stuff (those line can be kilometers long)
    // Maybe reduce the furthest estimation to a maximum ?
    if (false)
    {
        // convert to line
        std::vector<vector3> pointsOnLine;
        pointsOnLine.emplace_back(_coordinates.get_closest_estimation(inverseDepthStandardDev));
        // pointsOnLine.emplace_back(_coordinates.to_world_coordinates());
        pointsOnLine.emplace_back(_coordinates.get_furthest_estimation(inverseDepthStandardDev));

        mapWriter->add_line(pointsOnLine);
    }
}

bool MapPoint2D::compute_upgraded(const CameraToWorldMatrix& cameraToWorld,
                                  UpgradedFeature_ptr& upgradedFeature) const noexcept
{
    try
    {
        if (compute_linearity_score(cameraToWorld) < 0.1) // linearity index (percentage) (TODO: add to parameters)
        {
            Eigen::Matrix<double, 3, 6> jacobian;
            const auto& worldCoords = _coordinates.to_world_coordinates(jacobian);

            upgradedFeature = std::make_shared<UpgradedPoint2D>(
                    worldCoords, compute_cartesian_covariance(_covariance, jacobian), _descriptor, _matchIndex);
            return true;
        }
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Caught exeption while upgrading feature" + std::string(ex.what()));
        return false;
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

    if (is_depth_valid(matchedFeature._coordinates.z()))
    {
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