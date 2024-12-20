#ifndef RGBDSLAM_MAPMANAGEMENT_MAPPOINT_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPPOINT_HPP

#include "feature_map.hpp"
#include "features/keypoints/keypoint_handler.hpp"
#include "tracking/point_with_tracking.hpp"
#include "matches_containers.hpp"

namespace rgbd_slam::map_management {

const size_t INVALID_POINT_UNIQ_ID = 0; // This id indicates an invalid unique id for a map point

/**
 *  \brief The OptimizationFeature for a 3d point
 */
struct PointOptimizationFeature : public matches_containers::IOptimizationFeature
{
    PointOptimizationFeature(const ScreenCoordinate2D& matchedPoint,
                             const WorldCoordinate& mapPoint,
                             const matrix33& mapPointCovariance,
                             const size_t mapFeatureId);

    size_t get_feature_part_count() const noexcept override;

    double get_score() const noexcept override;

    vectorxd get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept override;
    matrixd get_distance_covariance(const WorldToCameraMatrix& worldToCamera) const noexcept override;

    double get_alpha_reduction() const noexcept override;

    FeatureType get_feature_type() const noexcept override;

    matrixd get_world_covariance() const noexcept override;

    matches_containers::feat_ptr get_variated_object() const noexcept override;

  protected:
    const ScreenCoordinate2D _matchedPoint;
    const WorldCoordinate _mapPoint;

    const matrix33 _mapPointCovariance;
    const vector3 _mapPointStandardDev; // opti: sqrt of the diagonal of _mapPointCovariance
};

using DetectedKeypointsObject = features::keypoints::Keypoint_Handler;
using DetectedPointType = features::keypoints::DetectedKeyPoint;
using TrackedPointsObject = features::keypoints::KeypointsWithIdStruct;

/**
 * \brief Classic point feature in map, all map points should inherit this
 */
class MapPoint :
    public tracking::Point,
    public IMapFeature<DetectedKeypointsObject, DetectedPointType, TrackedPointsObject>
{
  public:
    MapPoint(const WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor) :
        tracking::Point(coordinates, covariance, descriptor),
        IMapFeature<DetectedKeypointsObject, DetectedPointType, TrackedPointsObject>()
    {
        assert(_id > 0);
    }

    MapPoint(const WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor,
             const size_t id) :
        tracking::Point(coordinates, covariance, descriptor),
        IMapFeature<DetectedKeypointsObject, DetectedPointType, TrackedPointsObject>(id)
    {
        assert(_id > 0);
    }

    ~MapPoint() override = default;

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
                                        UpgradedFeature_ptr& upgradeFeature) const noexcept override
    {
        std::ignore = cameraToWorld;
        std::ignore = upgradeFeature;
        return false;
    }

    [[nodiscard]] bool is_moving() const noexcept override { return tracking::Point::is_moving(); }

  protected:
    [[nodiscard]] bool update_with_match(const DetectedPointType& matchedFeature,
                                         const matrix33& poseCovariance,
                                         const CameraToWorldMatrix& cameraToWorld) noexcept override;

    void update_no_match() noexcept override;

    // used for tracking of 2d points
    std::optional<ScreenCoordinate2D> _lastMatch;
};

/**
 * \brief Candidate for a map point
 */
class StagedMapPoint : public MapPoint, public IStagedMapFeature<DetectedPointType>
{
  public:
    StagedMapPoint(const matrix33& poseCovariance,
                   const CameraToWorldMatrix& cameraToWorld,
                   const DetectedPointType& detectedFeature);

    [[nodiscard]] bool should_remove_from_staged() const noexcept override;

    [[nodiscard]] bool should_add_to_local_map() const noexcept override;

    [[nodiscard]] static bool can_add_to_map(const DetectedPointType& detectedPoint) noexcept
    {
        return not detectedPoint._descriptor.empty() and is_depth_valid(detectedPoint._coordinates.z());
    }

  protected:
    double get_confidence() const noexcept;
};

/**
 * \brief A map point structure, containing all the necessary informations to identify a map point in local map
 */
class LocalMapPoint : public MapPoint, public ILocalMapFeature<StagedMapPoint>
{
  public:
    explicit LocalMapPoint(const StagedMapPoint& stagedPoint);

    // constructor for upgraded features
    LocalMapPoint(const WorldCoordinate& coordinates,
                  const WorldCoordinateCovariance& covariance,
                  const cv::Mat& descriptor,
                  const int matchIndex);

    [[nodiscard]] bool is_lost() const noexcept override;
};

class localPointMap :
    public Feature_Map<LocalMapPoint, StagedMapPoint, DetectedKeypointsObject, DetectedPointType, TrackedPointsObject>
{
  public:
    FeatureType get_feature_type() const override { return FeatureType::Point; }

    std::string get_display_name() const override { return "Points"; }

    DetectedKeypointsObject get_detected_feature(const DetectedFeatureContainer& features) const override
    {
        return features.keypointObject;
    }

    std::shared_ptr<TrackedPointsObject> get_tracked_features_container(
            const TrackedFeaturesContainer& tracked) const override
    {
        return tracked.trackedPoints;
    }

    size_t minimum_features_for_opti() const override { return parameters::optimization::minimumPointForOptimization; }

  protected:
    void add_upgraded_to_local_map(const UpgradedFeature_ptr upgradedfeature) override
    {
        if (upgradedfeature->get_type() == get_feature_type())
        {
            const auto upgraded = dynamic_cast<UpgradedPoint2D&>(*upgradedfeature);

            add_to_local_map(LocalMapPoint(
                    upgraded._coordinates, upgraded._covariance, upgraded._descriptor, upgraded._matchIndex));
        }
        else
        {
            outputs::log_error("Cannot add this feature to the point map");
        }
    }
};

} // namespace rgbd_slam::map_management

#endif
