#ifndef RGBDSLAM_MAPMANAGEMENT_MAPPOINT2D_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPPOINT2D_HPP

#include "coordinates/point_coordinates.hpp"
#include "parameters.hpp"
#include "feature_map.hpp"
#include "features/keypoints/keypoint_handler.hpp"
#include "tracking/inverse_depth_with_tracking.hpp"
#include "matches_containers.hpp"

namespace rgbd_slam::map_management {

struct UpgradedPoint2D
{
    WorldCoordinate _coordinates;
    WorldCoordinateCovariance _covariance;
    cv::Mat _descriptor;
    int _matchIndex;
};

/**
 *  \brief The OptimizationFeature for a 2d point
 */
struct Point2dOptimizationFeature : public matches_containers::IOptimizationFeature
{
    Point2dOptimizationFeature(const ScreenCoordinate2D& matchedPoint,
                               const InverseDepthWorldPoint& mapPoint,
                               const tracking::PointInverseDepth::Covariance& mapPointVariance,
                               const size_t mapFeatureId) :
        matches_containers::IOptimizationFeature(mapFeatureId),
        _matchedPoint(matchedPoint),
        _mapPoint(mapPoint),
        _mapPointVariance(mapPointVariance) {};

    size_t get_feature_part_count() const noexcept override { return 2; }

    double get_score() const noexcept override
    {
        static constexpr double optiScore = 1.0 / parameters::optimization::minimumPoint2dForOptimization;
        return optiScore;
    }

    vectorxd get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept override
    {
        const vector2& distance = _mapPoint.compute_signed_screen_distance(
                _matchedPoint, _mapPointVariance.get_inverse_depth_variance(), worldToCamera);
        return distance;
    }

    double get_max_retroprojection_error() const noexcept override
    {
        return parameters::optimization::ransac::maximumRetroprojectionErrorForPoint2DInliers_px;
    }

    double get_alpha_reduction() const noexcept override { return 0.3; }

    matches_containers::IOptimizationFeature* compute_random_variation() const noexcept override
    {
        WorldCoordinate variatedObservationPoint = _mapPoint.get_first_observation();
        // TODO: variate the observation point
        // variatedObservationPoint += utils::Random::get_normal_doubles<3>().cwiseProduct(
        //        _mapPointVariance.get_first_pose_covariance().diagonal().cwiseSqrt());
        const double variatedInverseDepth =
                _mapPoint.get_inverse_depth(); // do not variate the depth, the uncertainty is too great anyway
        const double variatedTheta =
                std::clamp(_mapPoint.get_theta() +
                                   utils::Random::get_normal_double() * sqrt(_mapPointVariance.get_theta_variance()),
                           0.0,
                           M_PI);
        const double variatedPhi = std::clamp(_mapPoint.get_phi() + utils::Random::get_normal_double() *
                                                                            sqrt(_mapPointVariance.get_phi_variance()),
                                              -M_PI,
                                              M_PI);

        return new Point2dOptimizationFeature(
                _matchedPoint,
                InverseDepthWorldPoint(variatedObservationPoint, variatedInverseDepth, variatedTheta, variatedPhi),
                _mapPointVariance,
                _idInMap);
    }

    FeatureType get_feature_type() const noexcept override { return FeatureType::Point2d; }

  protected:
    const ScreenCoordinate2D _matchedPoint;
    const InverseDepthWorldPoint _mapPoint;
    const tracking::PointInverseDepth::Covariance _mapPointVariance;
};

using DetectedKeypointsObject = features::keypoints::Keypoint_Handler;
using DetectedPoint2DType = features::keypoints::DetectedKeyPoint;
using TrackedPointsObject = features::keypoints::KeypointsWithIdStruct;
using UpgradedPoint2DType = UpgradedPoint2D; // 2D points can be upgraded to 3D

/**
 * \brief Classic 2D point feature in map, all 2D map points should inherit this.
 * Here, the 2d points are presented in inverse depth form
 */
class MapPoint2D :
    public tracking::PointInverseDepth,
    public IMapFeature<DetectedKeypointsObject, DetectedPoint2DType, TrackedPointsObject, UpgradedPoint2DType>
{
  public:
    MapPoint2D(const ScreenCoordinate2D& coordinates,
               const CameraToWorldMatrix& c2w,
               const matrix33& stateCovariance,
               const cv::Mat& descriptor) :
        PointInverseDepth(coordinates, c2w, stateCovariance, descriptor),
        IMapFeature<DetectedKeypointsObject, DetectedPoint2DType, TrackedPointsObject, UpgradedPoint2DType>()
    {
        assert(_id > 0);
        assert(not _descriptor.empty());
    }

    MapPoint2D(const tracking::PointInverseDepth& coordinates, const size_t id) :
        tracking::PointInverseDepth(coordinates),
        IMapFeature<DetectedKeypointsObject, DetectedPoint2DType, TrackedPointsObject, UpgradedPoint2DType>(id)
    {
        assert(_id > 0);
        assert(not _descriptor.empty());
    }

    virtual ~MapPoint2D() = default;

    [[nodiscard]] int find_match(const DetectedKeypointsObject& detectedFeatures,
                                 const WorldToCameraMatrix& worldToCamera,
                                 const vectorb& isDetectedFeatureMatched,
                                 matches_containers::match_container& matches,
                                 const bool shouldAddToMatches = true,
                                 const bool useAdvancedSearch = false) const noexcept override;

    [[nodiscard]] bool add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                                      TrackedPointsObject& trackedFeatures,
                                      const uint dropChance = 1000) const noexcept override;

    void draw(const WorldToCameraMatrix& worldToCamMatrix,
              cv::Mat& debugImage,
              const cv::Scalar& color) const noexcept override;

    [[nodiscard]] bool is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept override;

    void write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept override;

    [[nodiscard]] bool compute_upgraded(const CameraToWorldMatrix& cameraToWorld,
                                        UpgradedPoint2DType& upgradeFeature) const noexcept override;

    [[nodiscard]] bool is_moving() const noexcept override { return tracking::PointInverseDepth::is_moving(); }

  protected:
    [[nodiscard]] bool update_with_match(const DetectedPoint2DType& matchedFeature,
                                         const matrix33& poseCovariance,
                                         const CameraToWorldMatrix& cameraToWorld) noexcept override;

    void update_no_match() noexcept override;
};

/**
 * \brief Candidate for a map point
 */
class StagedMapPoint2D : public MapPoint2D, public IStagedMapFeature<DetectedPoint2DType>
{
  public:
    StagedMapPoint2D(const matrix33& poseCovariance,
                     const CameraToWorldMatrix& cameraToWorld,
                     const DetectedPoint2DType& detectedFeature);

    [[nodiscard]] bool should_remove_from_staged() const noexcept override;

    [[nodiscard]] bool should_add_to_local_map() const noexcept override;

    [[nodiscard]] static bool can_add_to_map(const DetectedPoint2DType& detectedPoint) noexcept
    {
        return not detectedPoint._descriptor.empty() and not is_depth_valid(detectedPoint._coordinates.z());
    }

  protected:
    double get_confidence() const noexcept;
};

/**
 * \brief A map point structure, containing all the necessary informations to identify a map point in local map
 */
class LocalMapPoint2D : public MapPoint2D, public ILocalMapFeature<StagedMapPoint2D>
{
  public:
    explicit LocalMapPoint2D(const StagedMapPoint2D& stagedPoint);

    [[nodiscard]] bool is_lost() const noexcept override;
};

using localPoint2DMap = Feature_Map<LocalMapPoint2D,
                                    StagedMapPoint2D,
                                    DetectedKeypointsObject,
                                    DetectedPoint2DType,
                                    TrackedPointsObject,
                                    UpgradedPoint2DType>;

} // namespace rgbd_slam::map_management

#endif
