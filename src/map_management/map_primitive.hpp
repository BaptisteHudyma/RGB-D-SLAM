#ifndef RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP

#include "../features/primitives/shape_primitives.hpp"
#include "../parameters.hpp"
#include "../tracking/kalman_filter.hpp"
#include "../utils/camera_transformation.hpp"
#include "../utils/coordinates.hpp"
#include "../matches_containers.hpp"
#include "../utils/random.hpp"
#include "covariances.hpp"
#include "distance_utils.hpp"
#include "feature_map.hpp"
#include "polygon.hpp"
#include "types.hpp"
#include <bits/ranges_algo.h>
#include <opencv2/opencv.hpp>

namespace rgbd_slam::map_management {

using DetectedPlaneType = features::primitives::Plane;
using DetectedPlaneObject = features::primitives::plane_container;
using PlaneMatchType = matches_containers::PlaneMatch;
using TrackedPlaneObject = void*; // TODO implement

class Plane
{
  public:
    Plane();

    [[nodiscard]] utils::PlaneWorldCoordinates get_parametrization() const noexcept { return _parametrization; }
    [[nodiscard]] matrix44 get_covariance() const noexcept { return _covariance; };
    [[nodiscard]] utils::WorldPolygon get_boundary_polygon() const noexcept { return _boundaryPolygon; };

    /**
     * \brief Update this plane coordinates using a new detection
     * \param[in] cameraToWorld The matrix to go from camera to world space
     * \param[in] matchedFeature The feature matched to this world feature
     * \param[in] newDetectionParameters The detected plane parameters, projected to world
     * \param[in] newDetectionCovariance The covariance of the newly detected feature; in world coordinates
     * \return The update score (distance between old and new parametrization)
     */
    double track(const CameraToWorldMatrix& cameraToWorld,
                 const DetectedPlaneType& matchedFeature,
                 const utils::PlaneWorldCoordinates& newDetectionParameters,
                 const matrix44& newDetectionCovariance);

    utils::PlaneWorldCoordinates _parametrization; // parametrization of this plane in world space
    matrix44 _covariance;                          // covariance of this plane in world space
    utils::WorldPolygon _boundaryPolygon;          // polygon describing the boundary of the plane, in plane space

  private:
    /**
     * \brief Update the current boundary polygon with the one from the detected plane
     * \param[in] cameraToWorld The matrix to convert from caera to world space
     * \param[in] detectedPolygon The boundary polygon of the matched feature, to project to this plane space
     */
    bool update_boundary_polygon(const CameraToWorldMatrix& cameraToWorld,
                                 const utils::CameraPolygon& detectedPolygon) noexcept;

    /**
     * \brief Build the parameter kalman filter
     */
    static void build_kalman_filter() noexcept;

    // shared kalman filter, between all planes
    inline static std::unique_ptr<tracking::SharedKalmanFilter<4, 4>> _kalmanFilter = nullptr;
};

class MapPlane :
    public Plane,
    public IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>
{
  public:
    MapPlane() : IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>()
    {
        assert(_id > 0);
    }

    explicit MapPlane(const size_t id) :
        IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>(id)
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

  protected:
    [[nodiscard]] bool update_with_match(const DetectedPlaneType& matchedFeature,
                                         const matrix33& poseCovariance,
                                         const CameraToWorldMatrix& cameraToWorld) noexcept override;

    void update_no_match() noexcept override;
};

class StagedMapPlane : public MapPlane, public IStagedMapFeature<DetectedPlaneType>
{
  public:
    StagedMapPlane(const matrix33& poseCovariance,
                   const CameraToWorldMatrix& cameraToWorld,
                   const DetectedPlaneType& detectedFeature);

    [[nodiscard]] bool should_remove_from_staged() const noexcept override;

    [[nodiscard]] bool should_add_to_local_map() const noexcept override;
};

class LocalMapPlane : public MapPlane, public ILocalMapFeature<StagedMapPlane>
{
  public:
    explicit LocalMapPlane(const StagedMapPlane& stagedPlane);

    [[nodiscard]] bool is_lost() const noexcept override;
};

} // namespace rgbd_slam::map_management

#endif
