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


        const vector3 compute_pose_variance(const utils::Pose& pose, const matches_containers::match_point_container& matchedPoints)
        {
            assert(not matchedPoints.empty());

            const matrix44& transformationMatrix = utils::compute_camera_to_world_transform(pose.get_orientation_quaternion(), pose.get_position());

            vector3 sumOfErrors = vector3::Zero();
            vector3 sumOfSquaredErrors = vector3::Zero();
            size_t numberOf3Dpoints = 0; 

            // For each pair of points
            for (const matches_containers::Match& match : matchedPoints)
            {
                // We only evaluate 3D points because 2D points cannot evaluate position
                if (match._screenPoint.z() <= 0)
                    continue;

                // Convert to world coordinates
                const vector3& matchedPoint3d = utils::screen_to_world_coordinates(match._screenPoint.x(), match._screenPoint.y(), match._screenPoint.z(), transformationMatrix);

                // absolute of (world map Point - new world point)
                const vector3& matchError = (match._worldPoint - matchedPoint3d).cwiseAbs();
                sumOfErrors += matchError;
                sumOfSquaredErrors += matchError.cwiseAbs2();
                ++numberOf3Dpoints;
            }

            assert(numberOf3Dpoints > 0);
            assert(sumOfErrors.x() >= 0 and sumOfErrors.y() >= 0 and sumOfErrors.z() >= 0);
            assert(sumOfSquaredErrors.x() >= 0 and sumOfSquaredErrors.y() >= 0 and sumOfSquaredErrors.z() >= 0);

            const double numberOfMatchesInverse = 1.0 / static_cast<double>(numberOf3Dpoints);
            const vector3& mean = sumOfErrors * numberOfMatchesInverse; 
            const vector3& variance = (sumOfSquaredErrors * numberOfMatchesInverse) - mean.cwiseAbs2();

            return variance;
        }


        const matrix33 compute_pose_covariance(const utils::Pose& pose)
        {
            const vector3& poseVariance = pose.get_position_variance();

            // TODO improve covariance computation
            const matrix33 poseCovariance {
                {poseVariance.x(), 0, 0},
                {0, poseVariance.y(), 0},
                {0, 0, poseVariance.z()}
            };

            return poseCovariance;
        }


    }   // utils
}       // rgbd_slam
