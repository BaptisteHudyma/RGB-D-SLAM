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
    _orientation.setIdentity();
}

PoseBase::PoseBase(const vector3& position, const quaternion& orientation) { set_parameters(position, orientation); }

void PoseBase::set_parameters(const vector3& position, const quaternion& orientation)
{
    _orientation = orientation;
    _position = position;
}

void PoseBase::update(const vector3& position, const quaternion& orientation)
{
    _orientation *= orientation;
    _position += position;
}

void PoseBase::display(std::ostream& os) const
{
    const EulerAngles displayAngles = get_euler_angles_from_quaternion(_orientation);
    os << "position: (" << _position.transpose() << ") millimeters | rotation: (" << displayAngles.yaw / EulerToRadian
       << ", " << displayAngles.pitch / EulerToRadian << ", " << displayAngles.roll / EulerToRadian << ") degrees";
}

std::ostream& operator<<(std::ostream& os, const PoseBase& pose)
{
    pose.display(os);
    return os;
}

double PoseBase::get_position_error(const PoseBase& pose) const { return (pose.get_position() - _position).norm(); }

double PoseBase::get_rotation_error(const PoseBase& pose) const
{
    const double distanceRadian = _orientation.angularDistance(pose.get_orientation_quaternion());
    return distanceRadian / EulerToRadian;
}

/**
 * Pose
 */

Pose::Pose() : PoseBase(), _poseVariance(matrix66::Zero()) {}

Pose::Pose(const vector3& position, const quaternion& orientation) :
    PoseBase(position, orientation),
    _poseVariance(matrix66::Zero())
{
}

Pose::Pose(const vector3& position, const quaternion& orientation, const matrix66& poseVariance) :
    PoseBase(position, orientation),
    _poseVariance(poseVariance)
{
}

void Pose::display(std::ostream& os) const
{
    PoseBase::display(os);
    os << std::endl << "position standard dev (meters/degrees) : " << std::endl;
    os << "x\ty\tz\t|\troll\tpitch\tyaw" << std::endl;
    vector6 poseStd = _poseVariance.diagonal().cwiseSqrt();
    os << poseStd.head(3).transpose() / 1000.0 << "\t|\t" << poseStd.tail(3).transpose() * 180.0 / M_PI << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Pose& pose)
{
    pose.display(os);
    return os;
}

} // namespace rgbd_slam::utils
