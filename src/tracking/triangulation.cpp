#include "triangulation.hpp"

#include "../types.hpp"
#include "../utils/camera_transformation.hpp"
#include "../parameters.hpp"
#include "coordinates.hpp"

namespace rgbd_slam {
    namespace tracking {

        utils::Pose Triangulation::get_supposed_pose(const utils::Pose& pose, const double baselinePoseSupposition)
        {
            const vector3 pt(baselinePoseSupposition, 0, 0);
            const vector3 newPosition = (pose.get_orientation_matrix() * pt) + pose.get_position();
            return utils::Pose(newPosition, pose.get_orientation_quaternion());
        }

        bool Triangulation::is_retroprojection_valid(const utils::WorldCoordinate& worldPoint, const utils::ScreenCoordinate2D& screenPoint, const worldToCameraMatrix& worldToCamera, const double& maximumRetroprojectionError)
        {
            utils::ScreenCoordinate2D projectedScreenPoint;
            const bool isRetroprojectionValid = worldPoint.to_screen_coordinates(worldToCamera, projectedScreenPoint);
            if (not isRetroprojectionValid)
            {
                return false;
            }
            const double retroprojectionError = (screenPoint - projectedScreenPoint).norm();

            // true if retroprojection error is small enough
            return (retroprojectionError > maximumRetroprojectionError);
        }

        bool Triangulation::triangulate(const worldToCameraMatrix& currentWorldToCamera, const worldToCameraMatrix& newWorldToCamera, const utils::ScreenCoordinate2D& point2Da, const utils::ScreenCoordinate2D& point2Db, utils::WorldCoordinate& triangulatedPoint) 
        {
            const double maximumRetroprojectionError = Parameters::get_maximum_retroprojection_error();

            // project x and y coordinates
            const utils::CameraCoordinate2D& pointA = point2Da.to_camera_coordinates();
            const utils::CameraCoordinate2D& pointB = point2Db.to_camera_coordinates();

            // Linear-LS triangulation
            Eigen::Matrix<double, 4, 4> triangulationMatrix;
            triangulationMatrix << pointA.x() * currentWorldToCamera.row(2) - currentWorldToCamera.row(0),
                                   pointA.y() * currentWorldToCamera.row(2) - currentWorldToCamera.row(1),
                                   pointB.x() * newWorldToCamera.row(2) - newWorldToCamera.row(0),
                                   pointB.y() * newWorldToCamera.row(2) - newWorldToCamera.row(1);

            // singular value decomposition
            const utils::WorldCoordinate worldPoint (
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
