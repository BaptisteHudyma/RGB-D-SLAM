#include "pose.hpp"
#include "angle_utils.hpp"
#include "types.hpp"

namespace rgbd_slam::utils {

/**
 * PoseBase
 */
PoseBase::PoseBase()
{
    _position.setZero();
    _rotation.setIdentity();
}

PoseBase::PoseBase(const vector3& position, const quaternion& rotation) { set_parameters(position, rotation); }

void PoseBase::set_parameters(const vector3& position, const quaternion& rotation) noexcept
{
    _rotation = rotation.normalized();
    _position = position;
}

void PoseBase::update(const vector3& position, const quaternion& rotation) noexcept
{
    _rotation *= rotation;
    _position += position;
}

void PoseBase::display(std::ostream& os) const noexcept
{
    const EulerAngles displayAngles = get_euler_angles_from_quaternion(_rotation);
    os << "position: (" << _position.transpose() << ") millimeters | rotation: (" << displayAngles.yaw / EulerToRadian
       << ", " << displayAngles.pitch / EulerToRadian << ", " << displayAngles.roll / EulerToRadian << ") degrees";
}

std::ostream& operator<<(std::ostream& os, const PoseBase& pose)
{
    pose.display(os);
    return os;
}

vector3 PoseBase::get_rotation_lie_error() const
{
    // compute error in lie space
    const vector3 rotPart(_rotation.x(), _rotation.y(), _rotation.z());
    vector3 lieSpaceError;
    const double norm = rotPart.norm();
    if (norm < 1e-6)
    {
        lieSpaceError = 2.0 * rotPart;
    }
    else
    {
        lieSpaceError = (2.0 * atan2(norm, _rotation.w())) * (rotPart / norm);
    }
    return lieSpaceError;
}

vector6 PoseBase::get_error_vector() const noexcept
{
    const vector3& lieSpaceError = get_rotation_lie_error();
    return vector6 {
            _position.x(), _position.y(), _position.z(), lieSpaceError.x(), lieSpaceError.y(), lieSpaceError.z()};
}

double PoseBase::get_position_error(const PoseBase& pose) const noexcept
{
    return (pose.get_position() - _position).norm();
}

double PoseBase::get_rotation_error(const PoseBase& pose) const noexcept
{
    const double distanceRadian = _rotation.angularDistance(pose.get_rotation_quaternion());
    return distanceRadian / EulerToRadian;
}

/**
 * Pose
 */

Pose::Pose() : PoseBase(), _poseVariance(matrix66::Zero()) {}

Pose::Pose(const vector3& position, const quaternion& rotation) :
    PoseBase(position, rotation),
    _poseVariance(matrix66::Zero())
{
}

Pose::Pose(const vector3& position, const quaternion& rotation, const matrix66& poseVariance) :
    PoseBase(position, rotation),
    _poseVariance(poseVariance)
{
}

void Pose::display(std::ostream& os) const noexcept
{
    PoseBase::display(os);
    os << std::endl << "position standard dev (meters/degrees) : " << std::endl;
    os << "x\ty\tz\t|\troll\tpitch\tyaw" << std::endl;
    vector6 poseStd = _poseVariance.diagonal().cwiseSqrt();
    os << poseStd.head<3>().transpose() << "\t|\t" << poseStd.tail<3>().transpose() * 180.0 / M_PI << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Pose& pose)
{
    pose.display(os);
    return os;
}

} // namespace rgbd_slam::utils
