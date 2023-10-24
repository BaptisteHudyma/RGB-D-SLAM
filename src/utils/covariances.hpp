#ifndef RGBDSLAM_UTILS_COVARIANCES_HPP
#define RGBDSLAM_UTILS_COVARIANCES_HPP

#include "coordinates.hpp"
#include "logger.hpp"
#include "matches_containers.hpp"
#include "pose.hpp"
#include "types.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <bits/ranges_algo.h>

namespace rgbd_slam::utils {

template<int N> [[nodiscard]] bool is_covariance_valid(const Eigen::Matrix<double, N, N>& covariance) noexcept
{
    // covariance should be symetrical
    if (!covariance.isApprox(covariance.transpose()))
    {
        outputs::log_warning("Covariance is not symetrical");
        return false;
    }

    // check that this covariance is positive semi definite
    Eigen::SelfAdjointEigenSolver<const Eigen::Matrix<double, N, N>> solver(covariance);
    // any value < 0 is an error
    const bool isPositiveSemiDefinite = not std::ranges::any_of(solver.eigenvalues(), [](double value) {
        return value < 0;
    });

    if (not isPositiveSemiDefinite)
    {
        outputs::log_warning("Covariance is not positive semi definite");
    }
    return isPositiveSemiDefinite;
}

/**
 * \brief Return the expected depth quantization at this depth value.
 * \param[in] depht The measured depth value, in millimeters
 * \return The smallest possible measure in millimeters (caped at 0.5 mm)
 */
[[nodiscard]] double get_depth_quantization(const double depht) noexcept;

/**
 * \brief Compute a screen point covariance from a given point
 *
 * \param[in] point The coordinates of this 3D point (world space)
 * \param[in] pointCovariance The covariance associated with this point (world space)
 */
[[nodiscard]] ScreenCoordinateCovariance get_screen_point_covariance(
        const WorldCoordinate& point, const WorldCoordinateCovariance& pointCovariance) noexcept;

/**
 * \brief Compute a screen point covariance from a given point
 *
 * \param[in] point The coordinates of this 3D point (camera space)
 * \param[in] pointCovariance The covariance associated with this point (camera space)
 */
[[nodiscard]] ScreenCoordinateCovariance get_screen_point_covariance(
        const CameraCoordinate& point, const CameraCoordinateCovariance& pointCovariance) noexcept;

/**
 * \brief Compute the covariance of a world point
 */
[[nodiscard]] WorldCoordinateCovariance get_world_point_covariance(
        const CameraCoordinateCovariance& cameraPointCovariance,
        const CameraToWorldMatrix& cameraToWorld,
        const matrix33& poseCovariance) noexcept;
/**
 * \brief Compute covariance of a screen point in world state
 */
[[nodiscard]] WorldCoordinateCovariance get_world_point_covariance(const ScreenCoordinate& screenPoint,
                                                                   const CameraToWorldMatrix& cameraToWorld,
                                                                   const matrix33& poseCovariance) noexcept;

/**
 * \brief Compute the associated Gaussian error of a screen point when it will be transformed to camera point. This
 * function will internaly compute the covariance of the screen point.
 * \param[in] screenPoint The 2D point in screen coordinates
 * \return the covariance of the 3D camera point
 */
[[nodiscard]] CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint) noexcept;

/**
 * \brief Compute the associated Gaussian error of a screen point when it will be transformed to camera point
 * \param[in] screenPoint The 2D point in screen coordinates
 * \param[in] screenPointCovariance The covariance matrix associated with a point in screen space
 * \return the covariance of the 3D camera point
 */
[[nodiscard]] CameraCoordinateCovariance get_camera_point_covariance(
        const ScreenCoordinate& screenPoint, const ScreenCoordinateCovariance& screenPointCovariance) noexcept;

/**
 * \brief Compute the covariance of a plane using it's point cloud covariance matrix
 * \param[in] planeParameters The plane parameters to compute covariance for
 * \param[in] pointCloudCovariance The covariance of the point cloud that this plane was fitted from
 * \return the plane parameter covariance
 */
[[nodiscard]] matrix44 compute_plane_covariance(const PlaneCoordinates& planeParameters,
                                                const matrix33& pointCloudCovariance);
[[nodiscard]] matrix33 compute_reduced_plane_point_cloud_covariance(const PlaneCoordinates& planeParameters,
                                                                    const matrix44& planeCloudCovariance);

/**
 * \brief Compute the covariance of the world plane
 * \param[in] planeCoordinates The coordinates of the camera plane to compute the covariance of
 * \param[in] cameraToWorldMatrix Matrix to convert from camera to world points
 * \param[in] planeCameraToWorldMatrix Matrix to convert from camera to world planes
 * \param[in] planeCovariance The covariance of the pkance in camera space
 * \param[in] worldPoseCovariance The covariance of the observer pose
 * \return The covariance of the plane parameters in world space
 */
[[nodiscard]] matrix44 get_world_plane_covariance(const PlaneCameraCoordinates& planeCoordinates,
                                                  const CameraToWorldMatrix& cameraToWorldMatrix,
                                                  const PlaneCameraToWorldMatrix& planeCameraToWorldMatrix,
                                                  const matrix44& planeCovariance,
                                                  const matrix33& worldPoseCovariance);

} // namespace rgbd_slam::utils

#endif
