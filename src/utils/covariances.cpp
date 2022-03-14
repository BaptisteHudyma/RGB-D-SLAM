#include "covariances.hpp"

#include "parameters.hpp"

namespace rgbd_slam {
    namespace utils {
        
        const matrix33 get_world_point_covariance(const vector2& screenPoint, const double depth, const matrix33& screenPointCovariance)
        {
            const double cameraFX = Parameters::get_camera_1_focal_x();
            const double cameraFY = Parameters::get_camera_1_focal_y();
            const double cameraCX = Parameters::get_camera_1_center_x();
            const double cameraCY = Parameters::get_camera_1_center_y();

            // Jacobian of the screen to world function. Use absolutes to prevent negative variances
            const matrix33 jacobian {
                {depth / cameraFX, 0.0,              abs(screenPoint.x() - cameraCX) / cameraFX },
                    {0.0,              depth / cameraFY, abs(screenPoint.y() - cameraCY) / cameraFY },
                    {0.0,              0.0,              1}
            };
            const matrix33& worldPointCovariance = jacobian * screenPointCovariance * jacobian.transpose();
            return worldPointCovariance;
        }

    }   // utils
}       // rgbd_slam
