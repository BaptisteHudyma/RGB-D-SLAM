#include "triangulation.hpp"

#include "camera_transformation.hpp"
#include "../parameters.hpp"

namespace rgbd_slam {
    namespace utils {

        utils::Pose Triangulation::get_supposed_pose(const utils::Pose& pose, const double baselinePoseSupposition)
        {
            const vector3 pt(baselinePoseSupposition, 0, 0);
            const vector3 newPosition = (pose.get_orientation_matrix() * pt) + pose.get_position();
            return utils::Pose(newPosition, pose.get_orientation_quaternion());
        }

        bool Triangulation::is_retroprojection_valid(const vector3& worldPoint, const vector2& screenPoint, const matrix44& worldToCameraMatrix, const double& maximumRetroprojectionError)
        {
            vector2 projectedScreenPoint;
            const bool isRetroprojectionValid = utils::compute_world_to_screen_coordinates(worldPoint, worldToCameraMatrix, projectedScreenPoint);
            if (not isRetroprojectionValid)
            {
                return false;
            }
            const double retroprojectionError = (screenPoint - projectedScreenPoint).norm();

            // true if retroprojection error is small enough
            return (retroprojectionError > maximumRetroprojectionError);
        }

        bool Triangulation::triangulate(const matrix44& currentWorldToCameraMatrix, const matrix44& newWorldToCameraMatrix, const vector2& point2Da, const vector2& point2Db, vector3& triangulatedPoint) 
        {
            const double cameraFX = Parameters::get_camera_1_focal_x();
            const double cameraFY = Parameters::get_camera_1_focal_y();
            const double cameraCX = Parameters::get_camera_1_center_x();
            const double cameraCY = Parameters::get_camera_1_center_y();
            const double maximumRetroprojectionError = Parameters::get_maximum_retroprojection_error();

            // project x and y coordinates
            const double pointAx = (point2Da.x() - cameraCX) / cameraFX;
            const double pointAy = (point2Da.y() - cameraCY) / cameraFY;
            const double pointBx = (point2Db.x() - cameraCX) / cameraFX;
            const double pointBy = (point2Db.y() - cameraCY) / cameraFY;

            // Linear-LS triangulation
            Eigen::Matrix<double, 4, 4> triangulationMatrix;
            triangulationMatrix << pointAx * currentWorldToCameraMatrix.row(2) - currentWorldToCameraMatrix.row(0),
                                pointAy * currentWorldToCameraMatrix.row(2) - currentWorldToCameraMatrix.row(1),
                                pointBx * newWorldToCameraMatrix.row(2) - newWorldToCameraMatrix.row(0),
                                pointBy * newWorldToCameraMatrix.row(2) - newWorldToCameraMatrix.row(1);

            // singular value decomposition
            const vector3& worldPoint = triangulationMatrix.leftCols<3>().jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-triangulationMatrix.col(3));
            if (std::isfinite(worldPoint.x()) and std::isfinite(worldPoint.y()) and std::isfinite(worldPoint.z()))
            {
                // We have a good triangulation ! Maybe not good enough but still usable
                triangulatedPoint = worldPoint;

                // Check retroprojection of point in frame A
                const bool isRetroprojectionPointAValid = Triangulation::is_retroprojection_valid(worldPoint, point2Da, currentWorldToCameraMatrix, maximumRetroprojectionError);
                if (not isRetroprojectionPointAValid)
                    return false;

                // Check retroprojection of point in frame B
                const bool isRetroprojectionPointBValid = Triangulation::is_retroprojection_valid(worldPoint, point2Db, newWorldToCameraMatrix, maximumRetroprojectionError);
                if (not isRetroprojectionPointBValid)
                    return false;

                // retroprojection is good enough
                return true;
            }
            // invalid triangulation 
            return false;
        }

    }
}
