#ifndef RGBDSLAM_TRACKING_PLANE_WITH_TRACKING_HPP
#define RGBDSLAM_TRACKING_PLANE_WITH_TRACKING_HPP

#include "coordinates/plane_coordinates.hpp"
#include "coordinates/polygon_coordinates.hpp"
#include "features/primitives/shape_primitives.hpp"
#include "tracking/kalman_filter.hpp"

namespace rgbd_slam::tracking {

/**
 * \brief Defines a plane, with tracking capabilities.
 * The normal vector and distance to the origin will be tracked, and boundary polygon uses a simple merge operation
 */
class Plane
{
  public:
    Plane();

    [[nodiscard]] PlaneWorldCoordinates get_parametrization() const noexcept { return _parametrization; }
    [[nodiscard]] matrix44 get_covariance() const noexcept { return _covariance; };
    [[nodiscard]] WorldPolygon get_boundary_polygon() const noexcept { return _boundaryPolygon; };

    /**
     * \brief Update this plane coordinates using a new detection
     * \param[in] cameraToWorld The matrix to go from camera to world space
     * \param[in] matchedFeature The feature matched to this world feature
     * \param[in] newDetectionParameters The detected plane parameters, projected to world
     * \param[in] newDetectionCovariance The covariance of the newly detected feature; in world coordinates
     * \return The update score (distance between old and new parametrization)
     */
    double track(const CameraToWorldMatrix& cameraToWorld,
                 const features::primitives::Plane& matchedFeature,
                 const PlaneWorldCoordinates& newDetectionParameters,
                 const matrix44& newDetectionCovariance);

    double track(const Plane& other);

    PlaneWorldCoordinates _parametrization; // parametrization of this plane in world space
    matrix44 _covariance;                   // covariance of this plane in world space
    WorldPolygon _boundaryPolygon;          // polygon describing the boundary of the plane, in plane space

  private:
    /**
     * \brief Update the current boundary polygon with the one from the detected plane
     * \param[in] cameraToWorld The matrix to convert from caera to world space
     * \param[in] detectedPolygon The boundary polygon of the matched feature, to project to this plane space
     */
    bool update_boundary_polygon(const CameraToWorldMatrix& cameraToWorld,
                                 const CameraPolygon& detectedPolygon) noexcept;

    /**
     * \brief Build the parameter kalman filter
     */
    static void build_kalman_filter() noexcept;

    // shared kalman filter, between all planes
    inline static std::unique_ptr<tracking::SharedKalmanFilter<4, 4>> _kalmanFilter = nullptr;
};

} // namespace rgbd_slam::tracking

#endif