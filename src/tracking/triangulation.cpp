#include "triangulation.hpp"

#include "../types.hpp"
#include "../utils/camera_transformation.hpp"
#include "../parameters.hpp"

namespace rgbd_slam {
    namespace tracking {

        utils::Pose Triangulation::get_supposed_pose(const utils::Pose& pose, const double baselinePoseSupposition)
        {
            const vector3 pt(baselinePoseSupposition, 0, 0);
            const vector3 newPosition = (pose.get_orientation_matrix() * pt) + pose.get_position();
            return utils::Pose(newPosition, pose.get_orientation_quaternion());
        }

        bool Triangulation::is_retroprojection_valid(const utils::worldCoordinates& worldPoint, const utils::screenCoordinates& screenPoint, const worldToCameraMatrix& worldToCamera, const double& maximumRetroprojectionError)
        {
            utils::screenCoordinates projectedScreenPoint;
            const bool isRetroprojectionValid = worldPoint.to_screen_coordinates(worldToCamera, projectedScreenPoint);
            if (not isRetroprojectionValid)
            {
                return false;
            }
            const double retroprojectionError = (screenPoint - projectedScreenPoint).norm();

            // true if retroprojection error is small enough
            return (retroprojectionError > maximumRetroprojectionError);
        }

        bool Triangulation::triangulate(const worldToCameraMatrix& currentWorldToCamera, const worldToCameraMatrix& newWorldToCamera, const utils::screenCoordinates& point2Da, const utils::screenCoordinates& point2Db, utils::worldCoordinates& triangulatedPoint) 
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
            triangulationMatrix << pointAx * currentWorldToCamera.row(2) - currentWorldToCamera.row(0),
                                pointAy * currentWorldToCamera.row(2) - currentWorldToCamera.row(1),
                                pointBx * newWorldToCamera.row(2) - newWorldToCamera.row(0),
                                pointBy * newWorldToCamera.row(2) - newWorldToCamera.row(1);

            // singular value decomposition
            const utils::worldCoordinates worldPoint (
                triangulationMatrix.leftCols<3>().jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-triangulationMatrix.col(3))
            );

            if (std::isfinite(worldPoint.x()) and std::isfinite(worldPoint.y()) and std::isfinite(worldPoint.z()))
            {
                // We have a good triangulation ! Maybe not good enough but still usable
                triangulatedPoint = worldPoint;

                // Check retroprojection of point in frame A
                const bool isRetroprojectionPointAValid = Triangulation::is_retroprojection_valid(worldPoint, point2Da, currentWorldToCamera, maximumRetroprojectionError);
                if (not isRetroprojectionPointAValid)
                    return false;

                // Check retroprojection of point in frame B
                const bool isRetroprojectionPointBValid = Triangulation::is_retroprojection_valid(worldPoint, point2Db, newWorldToCamera, maximumRetroprojectionError);
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
