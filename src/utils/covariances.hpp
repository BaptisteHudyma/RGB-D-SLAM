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

template<int N> bool is_covariance_valid(const Eigen::Matrix<double, N, N>& covariance)
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
        outputs::log_warning("Covariance is not positive semi definte");
    }
    return isPositiveSemiDefinite;
}

/**
 * \brief Return the expected depth quantization at this depth value.
 * \param[in] depht The measured depth value, in millimeters
 * \return The smallest possible measure in millimeters (caped at 0.5 mm)
 */
double get_depth_quantization(const double depht);

/**
 * \brief Compute a screen point covariance from a given point
 *
 * \param[in] point The coordinates of this 3D point (world space)
 * \param[in] pointCovariance The covariance associated with this point (world space)
 */
ScreenCoordinateCovariance get_screen_point_covariance(const WorldCoordinate& point,
                                                       const WorldCoordinateCovariance& pointCovariance);

/**
 * \brief Compute a screen point covariance from a given point
 *
 * \param[in] point The coordinates of this 3D point (camera space)
 * \param[in] pointCovariance The covariance associated with this point (camera space)
 */
ScreenCoordinateCovariance get_screen_point_covariance(const CameraCoordinate& point,
                                                       const CameraCoordinateCovariance& pointCovariance);

/**
 * \brief Compute the covariance of a world point
 */
WorldCoordinateCovariance get_world_point_covariance(const CameraCoordinateCovariance& cameraPointCovariance,
                                                     const CameraToWorldMatrix& cameraToWorld,
                                                     const matrix33& poseCovariance);
/**
 * \brief Compute covariance of a screen point in world state
 */
WorldCoordinateCovariance get_world_point_covariance(const ScreenCoordinate& screenPoint,
                                                     const CameraToWorldMatrix& cameraToWorld,
                                                     const matrix33& poseCovariance);

/**
 * \brief Compute the associated Gaussian error of a screen point when it will be transformed to camera point. This
 * function will internaly compute the covariance of the screen point.
 * \param[in] screenPoint The 2D point in screen coordinates
 * \return the covariance of the 3D camera point
 */
CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint);

/**
 * \brief Compute the associated Gaussian error of a screen point when it will be transformed to camera point
 * \param[in] screenPoint The 2D point in screen coordinates
 * \param[in] screenPointCovariance The covariance matrix associated with a point in screen space
 * \return the covariance of the 3D camera point
 */
CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint,
                                                       const ScreenCoordinateCovariance& screenPointCovariance);

/**
 * \brief Compute the covariance of a plane using it's point cloud covariance matrix
 * \param[in] planeParameters The plane parameters to compute covariance for
 * \param[in] pointCloudCovariance The covariance of the point cloud that this plane was fitted from
 * \return the plane parameter covariance
 */
matrix44 compute_plane_covariance(const vector4& planeParameters, const matrix33& pointCloudCovariance);

matrix44 get_world_plane_covariance(const PlaneCameraToWorldMatrix& cameraToWorldMatrix,
                                    const matrix44& planeCovariance,
                                    const matrix33& worldPoseCovariance);

} // namespace rgbd_slam::utils

#endif
