#ifndef RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP

#include "feature_map.hpp"
#include "features/primitives/shape_primitives.hpp"
#include "tracking/plane_with_tracking.hpp"
#include "matches_containers.hpp"

namespace rgbd_slam::map_management {

using DetectedPlaneType = features::primitives::Plane;
using DetectedPlaneObject = features::primitives::plane_container;
using PlaneMatchType = matches_containers::PlaneMatch;
using TrackedPlaneObject = void*; // TODO implement
using UpgradedPlaneType = void*;  // no upgrades for planes

/**
 * \brief Classic plane feature in map, all map plane should inherit this.
 * A plane is defined in hessian form (normal vector and distance to the origin).
 * Each plane also have a boundary polygon.
 */
class MapPlane :
    public tracking::Plane,
    public IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject, UpgradedPlaneType>
{
  public:
    MapPlane() :
        IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject, UpgradedPlaneType>()
    {
        assert(_id > 0);
    }

    explicit MapPlane(const size_t id) :
        IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject, UpgradedPlaneType>(id)
    {
        assert(_id > 0);
    }

    virtual ~MapPlane() = default;

    [[nodiscard]] int find_match(const DetectedPlaneObject& detectedFeatures,
                                 const WorldToCameraMatrix& worldToCamera,
                                 const vectorb& isDetectedFeatureMatched,
                                 std::list<PlaneMatchType>& matches,
                                 const bool shouldAddToMatches = true,
                                 const bool useAdvancedSearch = false) const noexcept override;

    [[nodiscard]] bool add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                                      TrackedPlaneObject& trackedFeatures,
                                      const uint dropChance = 1000) const noexcept override;

    void draw(const WorldToCameraMatrix& worldToCamMatrix,
              cv::Mat& debugImage,
              const cv::Scalar& color) const noexcept override;

    [[nodiscard]] bool is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept override;

    void write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept override;

    [[nodiscard]] bool compute_upgraded(const CameraToWorldMatrix& cameraToWorld,
                                        UpgradedPlaneType& upgradeFeature) const noexcept override
    {
        (void)cameraToWorld;
        (void)upgradeFeature;
        return false;
    }

    [[nodiscard]] bool is_moving() const noexcept override
    {
        // TODO
        return false;
    }

  protected:
    [[nodiscard]] bool update_with_match(const DetectedPlaneType& matchedFeature,
                                         const matrix33& poseCovariance,
                                         const CameraToWorldMatrix& cameraToWorld) noexcept override;

    void update_no_match() noexcept override;
};

/**
 * \brief Candidate for a map plane
 */
class StagedMapPlane : public MapPlane, public IStagedMapFeature<DetectedPlaneType>
{
  public:
    StagedMapPlane(const matrix33& poseCovariance,
                   const CameraToWorldMatrix& cameraToWorld,
                   const DetectedPlaneType& detectedFeature);

    [[nodiscard]] bool should_remove_from_staged() const noexcept override;

    [[nodiscard]] bool should_add_to_local_map() const noexcept override;

    [[nodiscard]] static bool can_add_to_map(const DetectedPlaneType& detectedPlane) noexcept
    {
        (void)detectedPlane;
        return true;
    }
};

/**
 * \brief A map plane structure, containing all the necessary informations to identify a map plane in local map
 */
class LocalMapPlane : public MapPlane, public ILocalMapFeature<StagedMapPlane>
{
  public:
    explicit LocalMapPlane(const StagedMapPlane& stagedPlane);

    [[nodiscard]] bool is_lost() const noexcept override;
};

using localPlaneMap = Feature_Map<LocalMapPlane,
                                  StagedMapPlane,
                                  DetectedPlaneObject,
                                  DetectedPlaneType,
                                  PlaneMatchType,
                                  TrackedPlaneObject,
                                  UpgradedPlaneType>;

} // namespace rgbd_slam::map_management

#endif
