#ifndef RGBDSLAM_MAPMANAGEMENT_MAPPOINT2D_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPPOINT2D_HPP

#include "coordinates/point_coordinates.hpp"
#include "feature_map.hpp"
#include "features/keypoints/keypoint_handler.hpp"
#include "tracking/inverse_depth_with_tracking.hpp"
#include "matches_containers.hpp"

namespace rgbd_slam::map_management {

/**
 *  \brief The OptimizationFeature for a 2d point
 */
struct Point2dOptimizationFeature : public matches_containers::IOptimizationFeature
{
    Point2dOptimizationFeature(const ScreenCoordinate2D& matchedPoint,
                               const InverseDepthWorldPoint& mapPoint,
                               const vector6& mapPointStandardDev,
                               const size_t mapFeatureId,
                               const size_t detectedFeatureId);

    size_t get_feature_part_count() const noexcept override;

    double get_score() const noexcept override;

    bool is_inlier(const WorldToCameraMatrix& worldToCamera) const noexcept override;

    vectorxd get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept override;

    double get_alpha_reduction() const noexcept override;

    matches_containers::feat_ptr compute_random_variation() const noexcept override;

    bool is_valid() const noexcept override;

    FeatureType get_feature_type() const noexcept override;

  protected:
    const ScreenCoordinate2D _matchedPoint;
    const InverseDepthWorldPoint _mapPoint;
    const vector6 _mapPointStandardDev;
};

using DetectedKeypointsObject = features::keypoints::Keypoint_Handler;
using DetectedPoint2DType = features::keypoints::DetectedKeyPoint;
using TrackedPointsObject = features::keypoints::KeypointsWithIdStruct;

/**
 * \brief Classic 2D point feature in map, all 2D map points should inherit this.
 * Here, the 2d points are presented in inverse depth form
 */
class MapPoint2D :
    public tracking::PointInverseDepth,
    public IMapFeature<DetectedKeypointsObject, DetectedPoint2DType, TrackedPointsObject>
{
  public:
    MapPoint2D(const ScreenCoordinate2D& coordinates,
               const CameraToWorldMatrix& c2w,
               const matrix33& stateCovariance,
               const cv::Mat& descriptor) :
        PointInverseDepth(coordinates, c2w, stateCovariance, descriptor),
        IMapFeature<DetectedKeypointsObject, DetectedPoint2DType, TrackedPointsObject>()
    {
        assert(_id > 0);
        assert(not _descriptor.empty());
    }

    MapPoint2D(const tracking::PointInverseDepth& coordinates, const size_t id) :
        tracking::PointInverseDepth(coordinates),
        IMapFeature<DetectedKeypointsObject, DetectedPoint2DType, TrackedPointsObject>(id)
    {
        assert(_id > 0);
        assert(not _descriptor.empty());
    }

    ~MapPoint2D() override = default;

    [[nodiscard]] matchIndexSet find_matches(const DetectedKeypointsObject& detectedFeatures,
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
                                        UpgradedFeature_ptr& upgradeFeature) const noexcept override;

    [[nodiscard]] bool is_moving() const noexcept override { return tracking::PointInverseDepth::is_moving(); }

    [[nodiscard]] bool update_with_match(const DetectedPoint2DType& matchedFeature,
                                         const matrix33& poseCovariance,
                                         const CameraToWorldMatrix& cameraToWorld) noexcept override;

  protected:
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

class localPoint2DMap :
    public Feature_Map<LocalMapPoint2D,
                       StagedMapPoint2D,
                       DetectedKeypointsObject,
                       DetectedPoint2DType,
                       TrackedPointsObject>
{
  public:
    FeatureType get_feature_type() const override { return FeatureType::Point2d; }

    std::string get_display_name() const override { return "P2D"; }

    DetectedKeypointsObject get_detected_feature(const DetectedFeatureContainer& features) const override
    {
        return features.keypointObject;
    }

    std::shared_ptr<TrackedPointsObject> get_tracked_features_container(
            const TrackedFeaturesContainer& tracked) const override
    {
        return tracked.trackedPoints;
    }

    size_t minimum_features_for_opti() const override
    {
        return parameters::optimization::minimumPoint2dForOptimization;
    }

  protected:
    void add_upgraded_to_local_map(const UpgradedFeature_ptr upgradedfeature) override
    {
        if (upgradedfeature->get_type() == get_feature_type())
        {
            outputs::log_error("Upgraded 2D points is not supported");
        }
        else
        {
            outputs::log_error("Cannot add this feature to the 2D point map");
        }
    }
};

} // namespace rgbd_slam::map_management

#endif
