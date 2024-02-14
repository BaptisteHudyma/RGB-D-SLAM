#ifndef RGBDSLAM_UTILS_ANGLE_UTILS_HPP
#define RGBDSLAM_UTILS_ANGLE_UTILS_HPP

#include "../types.hpp"

namespace rgbd_slam::utils {

/**
 * \brief Compute the jacobian of the transformation of a quaternion to euler angles
 */
matrix34 get_quaternion_to_euler_jacobian(const quaternion& quat);

/**
 * \brief Compute the jacobian of the transformation of a pose matrix composed of a position and a quaternion to a
 * pose matrix composed of a position and euler angles
 */
Eigen::Matrix<double, 6, 7> get_position_quaternion_to_position_euler_jacobian(const quaternion& quat);

/**
 * \brief Compute a quaternion from the given euler angles, in radians
 */
[[nodiscard]] quaternion get_quaternion_from_euler_angles(const EulerAngles& eulerAngles) noexcept;

/**
 * \brief Compute euler angles from a given quaternion
 */
[[nodiscard]] EulerAngles get_euler_angles_from_quaternion(const quaternion& quat) noexcept;

/**
 * \brief Compute a rotation matrix from a euler angle container
 */
[[nodiscard]] matrix33 get_rotation_matrix_from_euler_angles(const EulerAngles& eulerAngles) noexcept;

} // namespace rgbd_slam::utils

#endif
