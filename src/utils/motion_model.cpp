#include "motion_model.hpp"

namespace rgbd_slam {
namespace utils {


    Motion_Model::Motion_Model() {
        this->reset();
    }


    void Motion_Model::reset() {
        _lastQ.setIdentity();
        _angularVelocity.setIdentity();

        _lastPosition.setIdentity();
        _linearVelocity.setIdentity();
    }



    const Pose Motion_Model::predict_next_pose(const Pose& currentPose) const {
        //compute next linear velocity
        vector3 newLinVelocity = currentPose.get_position() - _lastPosition;
        newLinVelocity = (newLinVelocity + _linearVelocity) * 0.5;

        //compute new angular velocity
        const quaternion currentQ = currentPose.get_orientation_quaternion();
        const quaternion angVelDiff = currentQ * _lastQ.inverse();
        quaternion newAngVel = angVelDiff.slerp(0.5, _angularVelocity);
        newAngVel.normalize();

        //Integrate to compute predictions
        const vector3 integralPos = currentPose.get_position() + newLinVelocity;
        quaternion integralQ = currentQ * newAngVel;
        integralQ.normalize();

        return Pose(integralPos, integralQ);
    }


    void Motion_Model::update_model(const Pose& pose) {
        //compute next linear velocity
        vector3 newLinVelocity = pose.get_position() - _lastPosition;
        newLinVelocity = (newLinVelocity + _linearVelocity) * 0.5;

        //compute new angular velocity
        const quaternion currentQ = pose.get_orientation_quaternion();
        const quaternion angVelDiff = currentQ * _lastQ.inverse();
        quaternion newAngVel = angVelDiff.slerp(0.5, _angularVelocity);
        newAngVel.normalize();

        //update state
        _lastQ = currentQ;
        _angularVelocity = newAngVel;
        _lastPosition = pose.get_position();
        _linearVelocity = newLinVelocity;
    }


}
}
