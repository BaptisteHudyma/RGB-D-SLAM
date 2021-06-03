#include "Pose.hpp"

namespace poseEstimation {

    Pose::Pose() {
        _position.setZero();
        _orientation.setIdentity();
    }

    Pose::Pose(const vector3 &position, const quaternion &orientation) {
        set_parameters(position, orientation);
    }

    void Pose::set_parameters(const vector3 &position, const quaternion &orientation) {
        _orientation = orientation;
        _position = position;
    }


}
