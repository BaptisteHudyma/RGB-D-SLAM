#include "covariances.hpp"

#include "camera_transformation.hpp"
#include "../parameters.hpp"
#include "types.hpp"

namespace rgbd_slam {
    namespace utils {

        const screenCoordinateCovariance get_screen_point_covariance(const ScreenCoordinate& ScreenCoordinate) 
        {
            // quantization of depth measurments, depending on distance (uses depth as meters)
            const double depthMeters = ScreenCoordinate.z() / 1000.0;
            // If depth is less than the min distance, covariance is set to a high value
            // Source: 2013: "3D with kinect"
            const double depthQuantization = std::max(0.53, utils::is_depth_valid(ScreenCoordinate.z()) ? (-0.53 + 0.74 * depthMeters + 2.73 * pow(depthMeters, 2.0)) : 1000.0);
            // a zero variance will break the kalman gain
            assert(depthQuantization > 0);
            // TODO xy variance should also depend on the placement of the pixel in x and y
            const double xyVariance = pow(0.1, 2.0);

            screenCoordinateCovariance screenPointCovariance;
            screenPointCovariance.base() = matrix33({
                {xyVariance, 0.0,        0.0},
                {0.0,        xyVariance, 0.0},
                {0.0,        0.0,        pow(depthQuantization, 2.0)}
            });
            return screenPointCovariance;
        }

        const screenCoordinateCovariance get_screen_point_covariance(const vector3& point, const matrix33& pointCovariance)
        {
            const double cameraFX = Parameters::get_camera_1_focal_x();
            const double cameraFY = Parameters::get_camera_1_focal_y();

            // Jacobian of the world to screen function. Use absolutes to prevent negative variances
            const matrix33 jacobian {
                {cameraFX/point.z(), 0.0,                      -cameraFX * point.x() / pow(point.z(), 2.0)},
                {0.0,                      cameraFY/point.z(), -cameraFY * point.y() / pow(point.z(), 2.0)},
                {0.0,                      0.0,                      1.0}
            };
            screenCoordinateCovariance screenPointCovariance;
            screenPointCovariance.base() = (jacobian * pointCovariance * jacobian.transpose());
            return screenPointCovariance;
        }

        const cameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint)
        {
            return get_camera_point_covariance(screenPoint, get_screen_point_covariance(screenPoint));
        }

        const cameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint, const screenCoordinateCovariance& screenPointCovariance)
        {
            const double cameraFX = Parameters::get_camera_1_focal_x();
            const double cameraFY = Parameters::get_camera_1_focal_y();
            const double cameraCX = Parameters::get_camera_1_center_x();
            const double cameraCY = Parameters::get_camera_1_center_y();

            // Jacobian of the screen to camera function. Use absolutes to prevent negative variances
            const matrix33 jacobian {
                {screenPoint.z() / cameraFX, 0.0,                        abs(screenPoint.x() - cameraCX) / cameraFX },
                {0.0,                        screenPoint.z() / cameraFY, abs(screenPoint.y() - cameraCY) / cameraFY },
                {0.0,                        0.0,                        1.0}
            };
            
            cameraCoordinateCovariance cameraPointCovariance;
            cameraPointCovariance.base() = jacobian * screenPointCovariance * jacobian.transpose();
            return cameraPointCovariance;
        }


        bool compute_pose_variance(const utils::Pose& pose, const matches_containers::match_point_container& matchedPoints, vector3& poseVariance)
        {
            assert(not matchedPoints.empty());

            const cameraToWorldMatrix& transformationMatrix = utils::compute_camera_to_world_transform(pose.get_orientation_quaternion(), pose.get_position());

            vector3 sumOfErrors = vector3::Zero();
            vector3 sumOfSquaredErrors = vector3::Zero();
            size_t numberOf3Dpoints = 0; 

            // For each pair of points
            for (const matches_containers::PointMatch& match : matchedPoints)
            {
                // We only evaluate 3D points because 2D points cannot evaluate position
                if (match._screenFeature.z() <= 0)
                    continue;

                // Convert to world coordinates
                const WorldCoordinate& matchedPoint3d = (match._screenFeature).to_world_coordinates(transformationMatrix);

                // absolute of (world map Point - new world point)
                const vector3& matchError = (match._worldFeature - matchedPoint3d).cwiseAbs();
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
