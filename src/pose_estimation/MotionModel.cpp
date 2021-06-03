#include "MotionModel.hpp"

namespace poseEstimation {


    Motion_Model::Motion_Model() {
        this->reset();
    }


    void Motion_Model::reset() {
        this->lastQ.setIdentity();
        this->angularVelocity.setIdentity();

        this->lastPosition.setIdentity();
        this->linearVelocity.setIdentity();
    }



    Pose Motion_Model::predict_next_pose(const Pose& currentPose) {
        //compute next linear velocity
        vector3 newLinVelocity = currentPose.get_position() - this->lastPosition;
        newLinVelocity = (newLinVelocity + this->linearVelocity) * 0.5;

        //compute new angular velocity
        quaternion currentQ = currentPose.get_orientation_quaternion();
        quaternion angVelDiff = currentQ * this->lastQ.inverse();
        quaternion newAngVel = angVelDiff.slerp(0.5, this->angularVelocity);
        newAngVel.normalize();

        //update state
        this->lastQ = currentQ;
        this->angularVelocity = newAngVel;
        this->lastPosition = currentPose.get_position();
        this->linearVelocity = newLinVelocity;

        //Integrate to compute predictions
        vector3 integralPos = this->lastPosition + this->linearVelocity;
        quaternion integralQ = currentQ * newAngVel;
        integralQ.normalize();

        return Pose(integralPos, integralQ);
    }

}
