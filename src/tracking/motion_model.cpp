#include "motion_model.hpp"

namespace rgbd_slam::tracking {

Motion_Model::Motion_Model() { this->reset(); }

void Motion_Model::reset() noexcept
{
    _lastQ.setIdentity();
    _angularVelocity.setIdentity();

    _lastPosition.setZero();
    _linearVelocity.setZero();

    _isLastPositionSet = false;
}

void Motion_Model::reset(const vector3& lastPosition, const quaternion& lastRotation) noexcept
{
    _lastQ = lastRotation;
    _angularVelocity = quaternion::Identity();

    _lastPosition = lastPosition;
    _linearVelocity.setZero();

    _isLastPositionSet = true;
}

utils::Pose Motion_Model::predict_next_pose(const utils::Pose& currentPose, const bool shouldIncreaseVariance) noexcept
{
    const vector3& currentPosition = currentPose.get_position();
    const quaternion& currentRotation = currentPose.get_orientation_quaternion();

    // last not set
    if (!_isLastPositionSet)
    {
        _lastQ = currentRotation;
        _lastPosition = currentPosition;
        _isLastPositionSet = true;
        return utils::Pose(currentPosition, currentRotation, currentPose.get_pose_variance());
    }

    // compute next linear velocity
    vector3 newLinVelocity = (currentPosition - _lastPosition);
    newLinVelocity = (newLinVelocity + _linearVelocity) * 0.5;

    // compute new angular velocity
    const quaternion angVelDiff = currentRotation * _lastQ.inverse();
    // smooth velocity over time (decaying model)
    const quaternion& newAngularVelocity = angVelDiff.slerp(0.5, _angularVelocity).normalized();

    // update state
    _lastQ = currentRotation;
    _angularVelocity = newAngularVelocity;
    _lastPosition = currentPosition;
    _linearVelocity = newLinVelocity;

    matrix66 poseError = matrix66::Zero();
    if (shouldIncreaseVariance)
    {
        // add some variance to the new covariance (mm, radians)
        const vector6& stdToAdd = vector6(10, 10, 10, 0.1, 0.1, 0.1);
        poseError.diagonal() = stdToAdd.cwiseProduct(stdToAdd);
    }

    // Integrate to compute predictions
    const vector3 integralPos = _lastPosition + _linearVelocity;
    const quaternion& integralQ = (currentRotation * newAngularVelocity).normalized();

    return utils::Pose(integralPos, integralQ, currentPose.get_pose_variance() + poseError);
}

} // namespace rgbd_slam::tracking
