#include "covariances.hpp"

#include "camera_transformation.hpp"
#include "../parameters.hpp"
#include "types.hpp"

namespace rgbd_slam {
    namespace utils {

        const matrix33 get_screen_point_covariance(const screenCoordinates& screenCoordinates) 
        {
            // Quadratic error model (uses depth as meters)
            const double depthMeters = screenCoordinates.z() / 1000.0;
            // If depth is less than the min distance, covariance is set to a high value
            const double depthVariance = std::max(1.0, utils::is_depth_valid(screenCoordinates.z()) ? (0.74 * depthMeters + 2.73 * pow(depthMeters, 2.0)) : 1000.0);
            // a zero variance will break the kalman gain
            assert(depthVariance > 0);
            // TODO xy variance should also depend on the placement of the pixel in x and y
            const double xyVariance = pow(0.1, 2.0);

            matrix33 screenPointCovariance {
                {xyVariance, 0.0,        0.0},
                {0.0,        xyVariance, 0.0},
                {0.0,        0.0,        pow(depthVariance, 2.0)},
            };
            return screenPointCovariance;
        }

        const matrix33 get_screen_point_covariance(const cameraCoordinates& cameraPoint, const matrix33& worldPointCovariance)
        {
            const double cameraFX = Parameters::get_camera_1_focal_x();
            const double cameraFY = Parameters::get_camera_1_focal_y();

            // Jacobian of the world to screen function. Use absolutes to prevent negative variances
            const matrix33 jacobian {
                {cameraFX/cameraPoint.z(), 0.0,                      -cameraFX * cameraPoint.x() / pow(cameraPoint.z(), 2.0)},
                {0.0,                      cameraFY/cameraPoint.z(), -cameraFY * cameraPoint.y() / pow(cameraPoint.z(), 2.0)},
                {0.0,                      0.0,                      1.0}
            };
            matrix33 screenPointCovariance = jacobian * worldPointCovariance * jacobian.transpose();
            return screenPointCovariance;
        }


        const matrix33 get_world_point_covariance(const screenCoordinates& screenPoint)
        {
            return get_world_point_covariance(screenPoint, get_screen_point_covariance(screenPoint));
        }

        const matrix33 get_world_point_covariance(const screenCoordinates& screenPoint, const matrix33& screenPointCovariance)
        {
            const double cameraFX = Parameters::get_camera_1_focal_x();
            const double cameraFY = Parameters::get_camera_1_focal_y();
            const double cameraCX = Parameters::get_camera_1_center_x();
            const double cameraCY = Parameters::get_camera_1_center_y();

            // Jacobian of the screen to world function. Use absolutes to prevent negative variances
            const matrix33 jacobian {
                {screenPoint.z() / cameraFX, 0.0,                        abs(screenPoint.x() - cameraCX) / cameraFX },
                {0.0,                        screenPoint.z() / cameraFY, abs(screenPoint.y() - cameraCY) / cameraFY },
                {0.0,                        0.0,                        1.0}
            };
            
            matrix33 worldPointCovariance = jacobian * screenPointCovariance * jacobian.transpose();
            return worldPointCovariance;
        }


        bool compute_pose_variance(const utils::Pose& pose, const matches_containers::match_point_container& matchedPoints, vector3& poseVariance)
        {
            assert(not matchedPoints.empty());

            const cameraToWorldMatrix& transformationMatrix = utils::compute_camera_to_world_transform(pose.get_orientation_quaternion(), pose.get_position());

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
                const worldCoordinates& matchedPoint3d = utils::screen_to_world_coordinates(match._screenPoint, transformationMatrix);

                // absolute of (world map Point - new world point)
                const vector3& matchError = (match._worldPoint - matchedPoint3d).cwiseAbs();
                sumOfErrors += matchError;
                sumOfSquaredErrors += matchError.cwiseAbs2();
                ++numberOf3Dpoints;
            }

            if (numberOf3Dpoints > 0)
            {
                assert(sumOfErrors.x() >= 0 and sumOfErrors.y() >= 0 and sumOfErrors.z() >= 0);
                assert(sumOfSquaredErrors.x() >= 0 and sumOfSquaredErrors.y() >= 0 and sumOfSquaredErrors.z() >= 0);

                const double numberOfMatchesInverse = 1.0 / static_cast<double>(numberOf3Dpoints);
                const vector3& mean = sumOfErrors * numberOfMatchesInverse; 

                poseVariance = (sumOfSquaredErrors * numberOfMatchesInverse) - mean.cwiseAbs2();

                return true;
            }
            // No 3D points to estimate the variance
            return false;
        }


        const matrix33 compute_pose_covariance(const utils::Pose& pose)
        {
            const vector3& poseVariance = pose.get_position_variance();

            // TODO improve covariance computation
            matrix33 poseCovariance {
                {poseVariance.x(), 0.0,              0.0},
                {0.0,              poseVariance.y(), 0.0},
                {0.0,              0.0,              poseVariance.z()}
            };
            return poseCovariance;
        }


    }   // utils
}       // rgbd_slam
