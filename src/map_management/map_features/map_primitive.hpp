#ifndef RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP

#include "feature_map.hpp"
#include "features/primitives/shape_primitives.hpp"
#include "tracking/plane_with_tracking.hpp"
#include "matches_containers.hpp"

namespace rgbd_slam::map_management {

/**
 *  \brief The OptimizationFeature for a plane
 */
struct PlaneOptimizationFeature : public matches_containers::IOptimizationFeature
{
    PlaneOptimizationFeature(const PlaneCameraCoordinates& matchedPlane,
                             const PlaneWorldCoordinates& mapPlane,
                             const matrix44& mapPlaneCovariance,
                             const size_t mapFeatureId);

    ~PlaneOptimizationFeature() override = default;

    size_t get_feature_part_count() const noexcept override;

    double get_score() const noexcept override;

    vectorxd get_distance(const WorldToCameraMatrix& worldToCamera) const noexcept override;

    matrixd get_distance_jacobian(const WorldToCameraMatrix& worldToCamera) const noexcept;
    matrixd get_distance_covariance(const WorldToCameraMatrix& worldToCamera) const noexcept override;

    double get_alpha_reduction() const noexcept override;

    FeatureType get_feature_type() const noexcept override;

    matrixd get_world_covariance() const noexcept override;

  protected:
    const PlaneCameraCoordinates _matchedPlane;
    const PlaneWorldCoordinates _mapPlane;

    const matrix44 _mapPlaneCovariance;
    const vector4 _mapPlaneStandardDev;
};

using DetectedPlaneType = features::primitives::Plane;
using DetectedPlaneObject = features::primitives::plane_container;
using TrackedPlaneObject = void*; // TODO implement

/**
 * \brief Classic plane feature in map, all map plane should inherit this.
 * A plane is defined in hessian form (normal vector and distance to the origin).
 * Each plane also have a boundary polygon.
 */
class MapPlane : public tracking::Plane, public IMapFeature<DetectedPlaneObject, DetectedPlaneType, TrackedPlaneObject>
{
  public:
    MapPlane() : IMapFeature<DetectedPlaneObject, DetectedPlaneType, TrackedPlaneObject>() { assert(_id > 0); }

    explicit MapPlane(const size_t id) : IMapFeature<DetectedPlaneObject, DetectedPlaneType, TrackedPlaneObject>(id)
    {
        assert(_id > 0);
    }

    ~MapPlane() override = default;

    [[nodiscard]] int find_match(const DetectedPlaneObject& detectedFeatures,
                                 const WorldToCameraMatrix& worldToCamera,
                                 const vectorb& isDetectedFeatureMatched,
                                 matches_containers::match_container& matches,
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
                                        UpgradedFeature_ptr& upgradeFeature) const noexcept override
    {
        std::ignore = cameraToWorld;
        std::ignore = upgradeFeature;
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

class localPlaneMap :
    public Feature_Map<LocalMapPlane, StagedMapPlane, DetectedPlaneObject, DetectedPlaneType, TrackedPlaneObject>
{
  public:
    FeatureType get_feature_type() const override { return FeatureType::Plane; }

    std::string get_display_name() const override { return "Planes"; }

    DetectedPlaneObject get_detected_feature(const DetectedFeatureContainer& features) const override
    {
        return features.detectedPlanes;
    }

    std::shared_ptr<TrackedPlaneObject> get_tracked_features_container(
            const TrackedFeaturesContainer& tracked) const override
    {
        std::ignore = tracked;
        // no tracking for planes
        return nullptr;
    }

    size_t minimum_features_for_opti() const override { return parameters::optimization::minimumPlanesForOptimization; }

  protected:
    void add_upgraded_to_local_map(const UpgradedFeature_ptr upgradedfeature) override
    {
        if (upgradedfeature->get_type() == get_feature_type())
        {
            outputs::log_error("Upgraded planes is not supported");
        }
        else
        {
            outputs::log_error("Cannot add this feature to the plane map");
        }
    }
};

} // namespace rgbd_slam::map_management

#endif
