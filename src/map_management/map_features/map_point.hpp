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
                             const vector3& mapPointStandardDev,
                             const size_t mapFeatureId);

    size_t get_feature_part_count() const noexcept override;

    double get_score() const noexcept override;

    vectorxd get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept override;

    double get_max_retroprojection_error() const noexcept override;

    double get_alpha_reduction() const noexcept override;

    matches_containers::feat_ptr compute_random_variation() const noexcept override;

    FeatureType get_feature_type() const noexcept override;

  protected:
    const ScreenCoordinate2D _matchedPoint;
    const WorldCoordinate _mapPoint;
    const vector3 _mapPointStandardDev;
};

using DetectedKeypointsObject = features::keypoints::Keypoint_Handler;
using DetectedPointType = features::keypoints::DetectedKeyPoint;
using TrackedPointsObject = features::keypoints::KeypointsWithIdStruct;
using UpgradedPointType = void*; // no upgrade for 3D points

/**
 * \brief Classic point feature in map, all map points should inherit this
 */
class MapPoint :
    public tracking::Point,
    public IMapFeature<DetectedKeypointsObject, DetectedPointType, TrackedPointsObject, UpgradedPointType>
{
  public:
    MapPoint(const WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor) :
        tracking::Point(coordinates, covariance, descriptor),
        IMapFeature<DetectedKeypointsObject, DetectedPointType, TrackedPointsObject, UpgradedPointType>()
    {
        assert(_id > 0);
    }

    MapPoint(const WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor,
             const size_t id) :
        tracking::Point(coordinates, covariance, descriptor),
        IMapFeature<DetectedKeypointsObject, DetectedPointType, TrackedPointsObject, UpgradedPointType>(id)
    {
        assert(_id > 0);
    }

    virtual ~MapPoint() = default;

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
                                        UpgradedPointType& upgradeFeature) const noexcept override
    {
        (void)cameraToWorld;
        (void)upgradeFeature;
        return false;
    }

    [[nodiscard]] bool is_moving() const noexcept override { return tracking::Point::is_moving(); }

  protected:
    [[nodiscard]] bool update_with_match(const DetectedPointType& matchedFeature,
                                         const matrix33& poseCovariance,
                                         const CameraToWorldMatrix& cameraToWorld) noexcept override;

    void update_no_match() noexcept override;
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

class localPointMap : public Feature_Map<LocalMapPoint,
                                  StagedMapPoint,
                                  DetectedKeypointsObject,
                                  DetectedPointType,
                                  TrackedPointsObject,
                                  UpgradedPointType>
{
  public:

    FeatureType get_feature_type() const override {
      return FeatureType::Point;
    }

    std::string get_display_name() const override
    {
      return "Points";
    }

    DetectedKeypointsObject get_detected_feature(const DetectedFeatureContainer& features) const
    {
      return features.keypointObject;
    }
};

} // namespace rgbd_slam::map_management

#endif
