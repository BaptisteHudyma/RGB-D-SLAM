#include "pose.hpp"

#include "utils.hpp"

namespace rgbd_slam {
namespace utils {

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

    void Pose::update(const vector3& position, const quaternion& orientation) {
        _orientation *= orientation;
        _position += position;
    }


    void Pose::display(std::ostream& os) const {
        const EulerAngles displayAngles = get_euler_angles_from_quaternion(_orientation);
        os << _position.transpose() << " | " << displayAngles.yaw / EulerToRadian << ", " << displayAngles.pitch / EulerToRadian << ", " << displayAngles.roll / EulerToRadian << std::endl;
    }


    std::ostream& operator<<(std::ostream& os, const Pose& pose) {
        pose.display(os);
        return os;
    }

}
}
