#include "covariances.hpp"

#include "parameters.hpp"
#include "camera_transformation.hpp"

namespace rgbd_slam {
    namespace utils {
        
        const matrix33 get_screen_point_covariance(const vector2& screenCoordinates, const double depth) 
        {
            // Quadratic error model (uses depth as meters)
            const double depthMeters = depth / 1000.0;
            // If depth is less than the min distance, covariance is set to a high value
            const double depthVariance = std::max(0.0001, utils::is_depth_valid(depth) ? (-0.58 + 0.74 * depthMeters + 2.73 * pow(depthMeters, 2.0)) : 1000.0);
            // a zero variance will break the kalman gain
            assert(depthVariance > 0);

            // TODO xy variance should also depend on the placement of the pixel in x and y
            const double xyVariance = pow(0.1, 2.0);

            matrix33 screenPointCovariance {
                {xyVariance, 0,          0},
                    {0,          xyVariance, 0},
                    {0,          0,          depthVariance * depthVariance},
            };
            return screenPointCovariance;
        }


        
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
            const matrix33& worldPointCovariance = (jacobian.transpose() * jacobian).inverse() * screenPointCovariance;
            return worldPointCovariance;
        }

    }   // utils
}       // rgbd_slam
