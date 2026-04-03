#ifndef RGBDSLAM_UTILS_COVARIANCES_HPP
#define RGBDSLAM_UTILS_COVARIANCES_HPP

#include "coordinates/plane_coordinates.hpp"
#include "coordinates/point_coordinates.hpp"
#include "logger.hpp"
#include "types.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <bits/ranges_algo.h>
#include <opencv2/core/types.hpp>

namespace rgbd_slam::utils {

template<int N>
[[nodiscard]] bool is_covariance_valid(const Eigen::Matrix<double, N, N>& covariance, std::string& reason) noexcept
{
    // no invalid values
    if (covariance.hasNaN() or not covariance.allFinite())
    {
        reason = "invalid values";
        return false;
    }

    // special case: 1 dimention covariance
    if constexpr (N == 1)
    {
        return covariance(0, 0) >= 0;
    }

    // covariance should be symmetrical
    if (!covariance.isApprox(covariance.transpose()))
    {
        reason = "not symmetrical";
        return false;
    }

    // check the diagonal, negative elements here may indicates that wrong jacobians have been used
    if ((covariance.diagonal().array() < 0).any())
    {
        reason = "diagonal have negative components";
        return false;
    }

    // check that this covariance is positive semi definite
    const auto ldlt = covariance.template selfadjointView<Eigen::Lower>().ldlt();
    if (ldlt.info() == Eigen::NumericalIssue || !ldlt.isPositive())
    {
        reason = "not positive semi definite";
        return false;
    }
    return true;
}

template<int N> [[nodiscard]] bool is_covariance_valid(const Eigen::Matrix<double, N, N>& covariance) noexcept
{
    std::string reason;
    return is_covariance_valid(covariance, reason);
}

/**
 * \brief First order covariance propragation, with numerical approximation on diagonal
 */
template<int N, int M> Eigen::Matrix<double, M, M> propagate_covariance(const Eigen::Matrix<double, N, N>& inCovariance,
                                                                        const Eigen::Matrix<double, M, N>& jacobian,
                                                                        const double epsilon = 0.0)
{
    Eigen::Matrix<double, M, M> res =
            (jacobian * inCovariance.template selfadjointView<Eigen::Lower>() * jacobian.transpose())
                    .template selfadjointView<Eigen::Lower>();
    res.diagonal() += vectorxd::Constant(res.rows(), epsilon);
    return res;
}

/**
 * \brief Return the expected depth quantization at this depth value.
 * \param[in] depht The measured depth value, in meters
 * \return The smallest possible measure in meters
 */
[[nodiscard]] double get_depth_quantization(const double depth_m) noexcept;

/**
 * \brief Compute the covariance of a plane using it's point cloud covariance matrix
 * \param[in] planeParameters The plane parameters to compute covariance for
 * \param[in] pointCloudCovariance The covariance of the point cloud that this plane was fitted from
 * \return the plane parameter covariance
 */
[[nodiscard]] matrix44 compute_plane_covariance(const PlaneCoordinates& planeParameters,
                                                const matrix33& pointCloudCovariance);

/**
 * \brief Compute the covariance of a plane as the covariance of the equivalent point cloud
 * \param[in] planeParameters The plane parameters to compute covariance for
 * \param[in] planeCloudCovariance The covariance of the plane parameters
 * \return the equivalent point cloud parameter covariance
 */
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
                                                  const matrix66& worldPoseCovariance);

/**
 * \brief jacobian of the pose covariance to a screen projection, relative to a given point
 */
Eigen::Matrix<double, 3, 6> world_transform_of_point_jacobian(const WorldCoordinate& point,
                                                              const WorldToCameraMatrix& w2c);

/**
 * \brief jacobian of the pose covariance to a 2D screen projection, relative to a given point
 */
Eigen::Matrix<double, 2, 6> world_transform_of_2d_point_jacobian(const WorldCoordinate& point,
                                                                 const WorldToCameraMatrix& w2c);

/**
 * \brief Get the jacobian to transform a quaternion covariance to a euler form
 */
Eigen::Matrix<double, 3, 4> get_quaternion_to_euler_jacobian(const Eigen::Quaterniond& quat);

/**
 * \brief Return a rotated rect opencv strucure corresponding to the corresponding covariance
 */
cv::RotatedRect get_rotated_rect_screen_covariance(const vector2& center, const matrix22& screenCovariance);

} // namespace rgbd_slam::utils

#endif
