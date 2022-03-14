#ifndef RGBDSLAM_UTILS_ANGLE_UTILS_HPP
#define RGBDSLAM_UTILS_ANGLE_UTILS_HPP

#include "types.hpp"

namespace rgbd_slam {
    namespace utils {

        /**
         * \brief Compute a quaternion from the given euler angles, in radians
         */
        const quaternion get_quaternion_from_euler_angles(const EulerAngles& eulerAngles);

        /**
         * \brief Compute euler angles from a given quaternion
         */
        const EulerAngles get_euler_angles_from_quaternion(const quaternion& quat);

        /**
         * \brief Compute a rotation matrix from a euler angle container
         */
        const matrix33 get_rotation_matrix_from_euler_angles(const EulerAngles& eulerAngles);

    }   // utils
}       // rgbd_slam


#endif
