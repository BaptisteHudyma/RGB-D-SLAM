#include "angle_utils.hpp"

namespace rgbd_slam {
    namespace utils {
        
        const quaternion get_quaternion_from_euler_angles(const EulerAngles& eulerAngles)
        {
            const double cy = cos(eulerAngles.yaw * 0.5);
            const double sy = sin(eulerAngles.yaw * 0.5);
            const double cp = cos(eulerAngles.pitch * 0.5);
            const double sp = sin(eulerAngles.pitch * 0.5);
            const double cr = cos(eulerAngles.roll * 0.5);
            const double sr = sin(eulerAngles.roll * 0.5);

            quaternion quat(
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy
                    );
            return quat;
        }

        const EulerAngles get_euler_angles_from_quaternion(const quaternion& quat)
        {
            EulerAngles eulerAngles;
            eulerAngles.yaw = std::atan2(
                    2 * (quat.w() * quat.x() + quat.y() * quat.z()),
                    1 - 2 * (quat.x() * quat.x() + quat.y() * quat.y())
                    );

            const double sinp = 2 * (quat.w() * quat.y() - quat.z() * quat.x());
            if (std::abs(sinp) >= 1)
                eulerAngles.pitch = std::copysign(M_PI / 2, sinp);
            else
                eulerAngles.pitch = std::asin(sinp);

            eulerAngles.roll = std::atan2(
                    2 * (quat.w() * quat.z() + quat.x() * quat.y()),
                    1 - 2 * (quat.y() * quat.y() + quat.z() * quat.z())
                    );

            return eulerAngles;
        }

        const matrix33 get_rotation_matrix_from_euler_angles(const EulerAngles& eulerAngles)
        {
            const double ch = cos(eulerAngles.roll);
            const double sh = sin(eulerAngles.roll);
            const double ca = cos(eulerAngles.pitch);
            const double sa = sin(eulerAngles.pitch);
            const double cb = cos(eulerAngles.yaw);
            const double sb = sin(eulerAngles.yaw);

            matrix33 rotationMatrix {
                {ch*ca,  sh*sb - ch*sa*cb,  ch*sa*sb + sh*cb},
                {sa,     ca*cb,             -ca*sb},
                {-sh*ca, sh*sa*cb + ch*sb,  -sh*sa*sb + ch*cb}
            };
            return rotationMatrix;
        }

    }   // utils
}       // rgbd_slam
