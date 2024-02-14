#include "angle_utils.hpp"
#include <unsupported/Eigen/EulerAngles>

namespace rgbd_slam::utils {

matrix34 get_quaternion_to_euler_jacobian(const quaternion& quat)
{
    ///
    /// Ref: Development of a Real-Time Attitude System Using a Quaternion Parameterization and
    /// Non-Dedicated GPS Receivers
    ///
    /// by: John B. Schleppe
    matrix34 jac;

    const double theta1 = 1.0 / (SQR(quat.z() + quat.y()) + SQR(quat.w() + quat.x()));
    const double theta2 = 1.0 / (SQR(quat.z() - quat.y()) + SQR(quat.w() - quat.x()));
    const double theta3 = 1.0 / sqrt(1.0 - 4.0 * SQR(quat.y() * quat.z() + quat.x() * quat.w()));

    jac(0, 0) = -(quat.z() + quat.y()) * theta1 + (quat.z() - quat.y()) * theta2;
    jac(0, 1) = (quat.w() + quat.x()) * theta1 - (quat.w() - quat.x()) * theta2;
    jac(0, 2) = (quat.w() + quat.x()) * theta1 + (quat.w() - quat.x()) * theta2;
    jac(0, 3) = -(quat.z() + quat.y()) * theta1 - (quat.z() - quat.y()) * theta2;

    jac(1, 0) = 2.0 * quat.w() * theta3;
    jac(1, 1) = 2.0 * quat.z() * theta3;
    jac(1, 2) = 2.0 * quat.y() * theta3;
    jac(1, 3) = 2.0 * quat.x() * theta3;

    jac(2, 0) = -(quat.z() + quat.y()) * theta1 - (quat.z() - quat.y()) * theta2;
    jac(2, 1) = (quat.w() + quat.x()) * theta1 + (quat.w() - quat.x()) * theta2;
    jac(2, 2) = (quat.w() + quat.x()) * theta1 - (quat.w() - quat.x()) * theta2;
    jac(2, 3) = -(quat.z() + quat.y()) * theta1 + (quat.z() - quat.y()) * theta2;

    return jac;
}

Eigen::Matrix<double, 6, 7> get_position_quaternion_to_position_euler_jacobian(const quaternion& quat)
{
    Eigen::Matrix<double, 6, 7> jac;
    jac.setZero();

    jac.block<3, 3>(0, 0) = matrix33::Identity();
    jac.block<3, 4>(3, 3) = get_quaternion_to_euler_jacobian(quat);
    return jac;
}

quaternion get_quaternion_from_euler_angles(const EulerAngles& eulerAngles) noexcept
{
    return quaternion(Eigen::AngleAxisd(eulerAngles.roll, vector3::UnitX()) *
                      Eigen::AngleAxisd(eulerAngles.pitch, vector3::UnitY()) *
                      Eigen::AngleAxisd(eulerAngles.yaw, vector3::UnitZ()));
}

EulerAngles get_euler_angles_from_quaternion(const quaternion& quat) noexcept
{
    const auto eulerXYZ = Eigen::EulerAngles<double, Eigen::EulerSystemXYZ>(quat.toRotationMatrix()).angles();
    return EulerAngles(eulerXYZ.z(), eulerXYZ.y(), eulerXYZ.x());
}

matrix33 get_rotation_matrix_from_euler_angles(const EulerAngles& eulerAngles) noexcept
{
    return get_quaternion_from_euler_angles(eulerAngles).toRotationMatrix();
}

} // namespace rgbd_slam::utils
