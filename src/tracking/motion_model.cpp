#include "motion_model.hpp"

namespace rgbd_slam {
namespace tracking {

Motion_Model::Motion_Model() { this->reset(); }

void Motion_Model::reset()
{
    _lastQ.setIdentity();
    _angularVelocity.setIdentity();

    _lastPosition.setZero();
    _linearVelocity.setZero();
}

void Motion_Model::reset(const vector3& lastPosition, const quaternion& lastRotation)
{
    _lastQ = lastRotation;
    _angularVelocity.setIdentity();

    _lastPosition = lastPosition;
    _linearVelocity.setZero();
}

const quaternion Motion_Model::get_rotational_velocity(const quaternion& lastRotation,
                                                       const quaternion& lastVelocity,
                                                       const quaternion& currentRotation) const
{
    const quaternion angVelDiff = currentRotation * lastRotation.inverse();
    quaternion newAngVel = angVelDiff.slerp(0.5, lastVelocity);
    newAngVel.normalize();
    return newAngVel;
}

const vector3 Motion_Model::get_position_velocity(const vector3& lastPosition,
                                                  const vector3& lastVelocity,
                                                  const vector3& currentPosition) const
{
    const vector3 newLinVelocity = currentPosition - lastPosition;
    // smooth velocity over time (decaying model)
    return (newLinVelocity + lastVelocity) * 0.5;
}

const utils::Pose Motion_Model::predict_next_pose(const utils::Pose& currentPose) const
{
    const quaternion& currentRotation = currentPose.get_orientation_quaternion();
    const vector3& currentPosition = currentPose.get_position();

    // compute next linear velocity
    const vector3& newLinVelocity = get_position_velocity(_lastPosition, _linearVelocity, currentPosition);
    // compute new angular velocity
    const quaternion& newVelocity = get_rotational_velocity(_lastQ, _angularVelocity, currentRotation);

    // Integrate to compute predictions
    const vector3 integralPos = currentPosition + newLinVelocity;
    quaternion integralQ = currentRotation * newVelocity;
    integralQ.normalize();

    // TODO: predict variance instead of copying ?
    return utils::Pose(integralPos, integralQ, currentPose.get_pose_variance());
}

void Motion_Model::update_model(const utils::Pose& pose)
{
    const quaternion& currentRotation = pose.get_orientation_quaternion();
    const vector3& currentPosition = pose.get_position();

    // compute next linear velocities
    _angularVelocity = get_rotational_velocity(_lastQ, _angularVelocity, currentRotation);
    _linearVelocity = get_position_velocity(_lastPosition, _linearVelocity, currentPosition);

    // update state
    _lastQ = currentRotation;
    _lastPosition = currentPosition;
}

} // namespace tracking
} // namespace rgbd_slam
