#include "map_point2d.hpp"

#include "coordinates/point_coordinates.hpp"
#include "covariances.hpp"
#include "outputs/logger.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include "utils/line.hpp"

#include <exception>

namespace rgbd_slam::map_management {

/**
 * Point2dOptimizationFeature
 */

Point2dOptimizationFeature::Point2dOptimizationFeature(const ScreenCoordinate2D& matchedPoint,
                                                       const InverseDepthWorldPoint& mapPoint,
                                                       const vector6& mapPointStandardDev,
                                                       const size_t mapFeatureId,
                                                       const size_t detectedFeatureId) :
    matches_containers::IOptimizationFeature(mapFeatureId, detectedFeatureId),
    _matchedPoint(matchedPoint),
    _mapPoint(mapPoint),
    _mapPointStandardDev(mapPointStandardDev) {};

size_t Point2dOptimizationFeature::get_feature_part_count() const noexcept { return 2; }

double Point2dOptimizationFeature::get_score() const noexcept
{
    static constexpr double optiScore = 1.0 / parameters::optimization::minimumPoint2dForOptimization;
    return optiScore;
}

bool Point2dOptimizationFeature::is_inlier(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    return (get_distance(worldToCamera).array() <=
            parameters::optimization::ransac::maximumRetroprojectionErrorForPoint2DInliers_px)
            .all();
}

vectorxd Point2dOptimizationFeature::get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    const vector2& distance = _mapPoint.compute_signed_screen_distance(
            _matchedPoint, _mapPointStandardDev(InverseDepthWorldPoint::inverseDepthIndex), worldToCamera);
    return distance;
}

double Point2dOptimizationFeature::get_alpha_reduction() const noexcept { return 0.3; }

matches_containers::feat_ptr Point2dOptimizationFeature::compute_random_variation() const noexcept
{
    WorldCoordinate variatedObservationPoint = _mapPoint.get_first_observation();
    // TODO: variate the observation point
    // variatedObservationPoint += utils::Random::get_normal_doubles<3>().cwiseProduct(_mapPointStandardDev.head<3>());
    const double variatedInverseDepth =
            _mapPoint.get_inverse_depth(); // do not variate the depth, the uncertainty is too great anyway
    const double variatedTheta =
            std::clamp(_mapPoint.get_theta() + utils::Random::get_normal_double() *
                                                       _mapPointStandardDev(InverseDepthWorldPoint::thetaIndex),
                       0.0,
                       M_PI);
    const double variatedPhi =
            std::clamp(_mapPoint.get_phi() + utils::Random::get_normal_double() *
                                                     _mapPointStandardDev(InverseDepthWorldPoint::phiIndex),
                       -M_PI,
                       M_PI);

    return std::make_shared<Point2dOptimizationFeature>(
            _matchedPoint,
            InverseDepthWorldPoint(variatedObservationPoint, variatedInverseDepth, variatedTheta, variatedPhi),
            _mapPointStandardDev,
            _idInMap,
            _detectedFeatureId);
}

bool Point2dOptimizationFeature::is_valid() const noexcept
{
    return (not _matchedPoint.hasNaN()) and (not _mapPoint.get_bearing_vector().hasNaN()) and
           (not _mapPointStandardDev.hasNaN()) and (_mapPointStandardDev.array() >= 0).all();
}

FeatureType Point2dOptimizationFeature::get_feature_type() const noexcept { return FeatureType::Point2d; }

/**
 * MapPoint
 */

matchIndexSet MapPoint2D::find_matches(const DetectedKeypointsObject& detectedFeatures,
                                       const WorldToCameraMatrix& worldToCamera,
                                       const matrix66& worldToCameraCovariance,
                                       const vectorb& isDetectedFeatureMatched,
                                       matches_containers::match_container& matches,
                                       const bool shouldAddToMatches,
                                       const bool useAdvancedSearch) const noexcept
{
    std::ignore = worldToCameraCovariance;

    matchIndexSet matchIndexRes;

    assert(not _descriptor.empty());

    // try to match with tracking
    int matchIndex = detectedFeatures.get_tracking_match_index(_id, isDetectedFeatureMatched);
    if (matchIndex == features::keypoints::INVALID_MATCH_INDEX)
#if 0
    // use line matching
    {
        // No match: try to find match in a window around the point
        utils::Segment<2> seg;
        if (not _coordinates.to_screen_coordinates(worldToCamera, _covariance.get_inverse_depth_variance(), seg))
        {
            return matchIndexRes;
        }
        utils::Segment<2> projection;
        if (not utils::clamp_to_screen(seg, projection))
        {
            return matchIndexRes;
        }
        matchIndexRes = detectedFeatures.get_match_index(
                projection, _descriptor, isDetectedFeatureMatched, useAdvancedSearch ? 20.0 : 10.0);
    }
#else
    // use center point projection matching
    {
        // No match: try to find match in a window around the point
        const ScreenCoordinate2D& projectedMapPoint = _coordinates.get_projected_screen_estimation(worldToCamera);

        double searchSpaceRadius = parameters::matching::matchSearchRadius_px;
        try
        {
            Eigen::Matrix<double, 3, 6> jacobian;
            const auto& worldCoords = _coordinates.to_world_coordinates(jacobian);
            const WorldCoordinateCovariance& cartCov = compute_cartesian_covariance(_covariance, jacobian);

            ScreenCoordinate2D sc;
            matrix23 toScreenJacobian;
            if (worldCoords.to_screen_coordinates(worldToCamera, sc, toScreenJacobian))
            {
                /// TODO: alert ?
            }
            // get covariance of the point in 2d
            const vector2& screenPointCovariance =
                    utils::propagate_covariance(cartCov, toScreenJacobian).diagonal().head<2>();

            searchSpaceRadius = sqrt(std::max(screenPointCovariance.x(), screenPointCovariance.y()));
        }
        catch (const std::exception& e)
        {
            /// TODO: debug this case: it happens when the inverse depth goes negative
            // outputs::log_warning(std::format("Could not compute 2D point screen covariance : %s", e.what()));
        }

        const double searchRadius = useAdvancedSearch ? searchSpaceRadius * 3.0 : searchSpaceRadius * 2.0;

        matchIndexRes = detectedFeatures.get_match_indexes(
                projectedMapPoint, _descriptor, isDetectedFeatureMatched, searchRadius);
    }
#endif

    if (matchIndex == features::keypoints::INVALID_MATCH_INDEX)
    {
        // unmatched point
        return matchIndexRes;
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
        for (const auto i: matchIndexRes)
        {
            matches.push_back(std::make_shared<Point2dOptimizationFeature>(detectedFeatures.get_keypoint(i).get_2D(),
                                                                           _coordinates,
                                                                           _covariance.diagonal().cwiseSqrt(),
                                                                           _id,
                                                                           i));
        }
    }
    return matchIndexRes;
}

bool MapPoint2D::add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                                TrackedPointsObject& trackedFeatures,
                                const uint dropChance) const noexcept
{
    std::ignore = worldToCamera;
    std::ignore = trackedFeatures;
    std::ignore = dropChance;
    // DO NOT TRACK inverse depth points, this produce weird match results and wrong triangulation
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

    if (is_matched())
    {
        const ScreenCoordinate2D startPoint(screenCoordinates.get_start_point());
        const ScreenCoordinate2D endPoint(screenCoordinates.get_end_point());

        // prevent a display out of the screen (visual bug)
        if (startPoint.is_in_screen_boundaries() and endPoint.is_in_screen_boundaries())
        {
            const cv::Point p1(static_cast<int>(startPoint.x()), static_cast<int>(startPoint.y()));
            const cv::Point p2(static_cast<int>(endPoint.x()), static_cast<int>(endPoint.y()));

            // if it's matched, display blue around it, else display red
            cv::line(debugImage, p1, p2, is_matched() ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), 3);
            cv::line(debugImage, p1, p2, color, 1);
        }
        else
        {
            outputs::log_error("Cannot draw a line out of screen boundaries");
        }
    }

    // do not display points behind the camera
    const auto centerProj = _coordinates.get_projected_screen_estimation(worldToCamMatrix);
    if (centerProj.is_in_screen_boundaries())
    {
        cv::circle(debugImage,
                   cv::Point(static_cast<int>(centerProj.x()), static_cast<int>(centerProj.y())),
                   2,
                   is_matched() ? color : cv::Scalar(255, 255, 255),
                   -1);

        const bool shouldDisplayErrorEllipse = true;
        if (shouldDisplayErrorEllipse)
        {
            Eigen::Matrix<double, 3, 6> jacobian;
            const auto& worldCoords = _coordinates.to_world_coordinates(jacobian);
            const WorldCoordinateCovariance& cartCov = compute_cartesian_covariance(_covariance, jacobian);

            ScreenCoordinate2D sc;
            matrix23 toScreenJacobian;
            if (worldCoords.to_screen_coordinates(worldToCamMatrix, sc, toScreenJacobian))
            {
                /// TODO: alert ?
            }
            // get covariance of the point in 2d
            const auto& screenPointCovariance = utils::propagate_covariance(cartCov, toScreenJacobian);

            cv::ellipse(debugImage,
                        utils::get_rotated_rect_screen_covariance(centerProj, screenPointCovariance.block<2, 2>(0, 0)),
                        color,
                        1);
        }
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
    // TODO find a way to add 2D points without breaking stuff (those line can be kilometers long)
    // Maybe reduce the furthest estimation to a maximum ?
}

bool MapPoint2D::compute_upgraded(const CameraToWorldMatrix& cameraToWorld,
                                  UpgradedFeature_ptr& upgradedFeature) const noexcept
{
    try
    {
        if (
                // sanity check
                _coordinates.get_inverse_depth() > 0.0 and
                // linearity index (percentage) (TODO: add to parameters)
                compute_linearity_score(cameraToWorld) < 0.1)
        {
            Eigen::Matrix<double, 3, 6> jacobian;
            const auto& worldCoords = _coordinates.to_world_coordinates(jacobian);

            const WorldCoordinateCovariance& cartCov = compute_cartesian_covariance(_covariance, jacobian);
            if ((cartCov.diagonal().array() > SQR(1.0)).any())
                return false;

            upgradedFeature = std::make_shared<UpgradedPoint2D>(worldCoords, cartCov, _descriptor, _matchIndexes);
            return true;
        }
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Caught exception while upgrading feature" + std::string(ex.what()));
        return false;
    }
    return false;
}

bool MapPoint2D::update_with_match(const DetectedPoint2DType& matchedFeature,
                                   const matrix66& poseCovariance,
                                   const CameraToWorldMatrix& cameraToWorld) noexcept
{
    if (_matchIndexes.empty())
    {
        outputs::log_error("Tries to call the function update_with_match with no associated match");
        return false;
    }

    const auto& matchCoordinates = matchedFeature._coordinates;

#ifndef BLOCK_2D_TO_3D_MERGE_2DPOINTS
    // use the real observation, it will most likely override the covariance inside the inverse depth point
    if (is_depth_valid(matchCoordinates.z()) and
        track_3D(matchCoordinates, matchCoordinates.get_covariance(), cameraToWorld, poseCovariance))
    {
        // success ! passthrough
    }
    // else: try 2D fusion
    else
#endif
            if (not track_2D(matchCoordinates.get_2D(),
                             matchCoordinates.get_2D().get_covariance(),
                             cameraToWorld,
                             poseCovariance))
    {
        return false;
    }

    if (not matchedFeature._descriptor.empty())
        _descriptor = matchedFeature._descriptor;

    latestMatchedFeature = matchCoordinates.get_2D();
    return true;
}

bool MapPoint2D::merge(const MapPoint2D& other) noexcept
{
    // TODO: merge
    std::ignore = other;
    return false;
}

void MapPoint2D::update_no_match() noexcept
{
    // do nothing
    latestMatchedFeature.reset();
}

/**
 * StagedMapPoint
 */

StagedMapPoint2D::StagedMapPoint2D(const matrix66& poseCovariance,
                                   const CameraToWorldMatrix& cameraToWorld,
                                   const DetectedPoint2DType& detectedFeature) :
    MapPoint2D(detectedFeature._coordinates.get_2D(), cameraToWorld, poseCovariance, detectedFeature._descriptor)
{
    // staged point start with tracking
    latestMatchedFeature = detectedFeature._coordinates.get_2D();
}

bool StagedMapPoint2D::should_remove_from_staged() const noexcept { return get_confidence() <= 0; }

bool StagedMapPoint2D::should_add_to_local_map() const noexcept
{
    constexpr double minimumConfidenceForLocalMap = parameters::mapping::pointMinimumConfidenceForMap;
    // TODO: check if this point should convert to linear
    return (get_confidence() > minimumConfidenceForLocalMap);
}

double StagedMapPoint2D::get_confidence() const noexcept
{
    constexpr double oneOverStagedPointconfidence =
            1.0 / static_cast<double>(parameters::mapping::point2dStagedAgeConfidence);
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

    latestMatchedFeature = stagedPoint.latestMatchedFeature;

    _matchIndexes = stagedPoint._matchIndexes;
    _successivMatchedCount = stagedPoint._successivMatchedCount;
}

bool LocalMapPoint2D::is_lost() const noexcept
{
    return (_failedTrackingCount > parameters::mapping::point2dUnmatchedCountToLoose);
}

} // namespace rgbd_slam::map_management
