#ifndef RGBDSLAM_MAPMANAGEMENT_MAPPOINT_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPPOINT_HPP

#include "feature_map.hpp"
#include "features/keypoints/keypoint_handler.hpp"
#include "tracking/point_with_tracking.hpp"
#include "matches_containers.hpp"

namespace rgbd_slam::map_management {

const size_t INVALID_POINT_UNIQ_ID = 0; // This id indicates an invalid unique id for a map point

using DetectedKeypointsObject = features::keypoints::Keypoint_Handler;
using DetectedPointType = features::keypoints::DetectedKeyPoint;
using PointMatchType = matches_containers::PointMatch;
using TrackedPointsObject = features::keypoints::KeypointsWithIdStruct;
using UpgradedPointType = void*; // no upgrade for 3D points

class MapPoint :
    public tracking::Point,
    public IMapFeature<DetectedKeypointsObject,
                       DetectedPointType,
                       PointMatchType,
                       TrackedPointsObject,
                       UpgradedPointType>
{
  public:
    MapPoint(const WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor) :
        tracking::Point(coordinates, covariance, descriptor),
        IMapFeature<DetectedKeypointsObject,
                    DetectedPointType,
                    PointMatchType,
                    TrackedPointsObject,
                    UpgradedPointType>()
    {
        assert(_id > 0);
    }

    MapPoint(const WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor,
             const size_t id) :
        tracking::Point(coordinates, covariance, descriptor),
        IMapFeature<DetectedKeypointsObject, DetectedPointType, PointMatchType, TrackedPointsObject, UpgradedPointType>(
                id)
    {
        assert(_id > 0);
    }

    virtual ~MapPoint() = default;

    [[nodiscard]] int find_match(const DetectedKeypointsObject& detectedFeatures,
                                 const WorldToCameraMatrix& worldToCamera,
                                 const vectorb& isDetectedFeatureMatched,
                                 std::list<PointMatchType>& matches,
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

using localPointMap = Feature_Map<LocalMapPoint,
                                  StagedMapPoint,
                                  DetectedKeypointsObject,
                                  DetectedPointType,
                                  PointMatchType,
                                  TrackedPointsObject,
                                  UpgradedPointType>;

} // namespace rgbd_slam::map_management

#endif
