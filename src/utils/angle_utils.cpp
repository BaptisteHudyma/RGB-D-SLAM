#include "angle_utils.hpp"
#include <unsupported/Eigen/EulerAngles>

namespace rgbd_slam::utils {

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
