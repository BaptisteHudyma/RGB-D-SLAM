#include "motion_model.hpp"

namespace rgbd_slam::tracking {

Motion_Model::Motion_Model() { this->reset(); }

void Motion_Model::reset() noexcept
{
    _lastQ.setIdentity();
    _angularVelocity.setIdentity();

    _lastPosition.setZero();
    _linearVelocity.setZero();
}

void Motion_Model::reset(const vector3& lastPosition, const quaternion& lastRotation) noexcept
{
    _lastQ = lastRotation;
    _angularVelocity.setIdentity();

    _lastPosition = lastPosition;
    _linearVelocity.setZero();
}

quaternion Motion_Model::get_rotational_velocity(const quaternion& lastRotation,
                                                 const quaternion& lastVelocity,
                                                 const quaternion& currentRotation) const noexcept
{
    const quaternion angVelDiff = currentRotation * lastRotation.inverse();
    // smooth velocity over time (decaying model)
    quaternion newAngVel = angVelDiff.slerp(0.5, lastVelocity);
    return newAngVel.normalized();
}

vector3 Motion_Model::get_position_velocity(const vector3& lastPosition,
                                            const vector3& lastVelocity,
                                            const vector3& currentPosition) const noexcept
{
    const vector3 newLinVelocity = (currentPosition - lastPosition) / 1000.0;
    // smooth velocity over time (decaying model)
    return (newLinVelocity + lastVelocity) * 0.5;
}

utils::Pose Motion_Model::predict_next_pose(const utils::Pose& currentPose,
                                            const bool shouldIncreaseVariance) const noexcept
{
    const quaternion& currentRotation = currentPose.get_orientation_quaternion();
    const vector3& currentPosition = currentPose.get_position();

    // compute next linear velocity
    const vector3& newLinVelocity = get_position_velocity(_lastPosition, _linearVelocity, currentPosition);
    // compute new angular velocity
    const quaternion& newVelocity = get_rotational_velocity(_lastQ, _angularVelocity, currentRotation);

    // Integrate to compute predictions
    const vector3 integralPos = currentPosition + newLinVelocity;
    const quaternion integralQ = (currentRotation * newVelocity).normalized();

    matrix66 poseError = matrix66::Zero();
    if (shouldIncreaseVariance)
    {
        // add some variance to the new covariance (mm, radians)
        const vector6& stdToAdd = vector6(10, 10, 10, 0.1, 0.1, 0.1);
        poseError.diagonal() = stdToAdd.cwiseProduct(stdToAdd);
    }
    return utils::Pose(integralPos, integralQ, currentPose.get_pose_variance() + poseError);
}

void Motion_Model::update_model(const utils::Pose& pose) noexcept
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

} // namespace rgbd_slam::tracking
