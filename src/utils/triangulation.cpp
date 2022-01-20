#include "triangulation.hpp"

#include "utils.hpp"
#include "parameters.hpp"

namespace rgbd_slam {
    namespace utils {

        utils::Pose Triangulate::get_supposed_pose(const utils::Pose& pose, const double baselinePoseSupposition)
        {
            const vector3 pt(baselinePoseSupposition, 0, 0);
            const vector3 newPosition = (pose.get_orientation_matrix() * pt) + pose.get_position();
            return utils::Pose(newPosition, pose.get_orientation_quaternion());
        }

        bool Triangulate::triangulate(const utils::Pose& pose, const vector2& point2Da, const vector2& point2Db, vector3& triangulatedPoint) 
        {
            // TODO true baseline
            const double baseline = 0;
            return Triangulate::triangulate(pose, Triangulate::get_supposed_pose(pose, baseline), point2Da, point2Db, triangulatedPoint);
        }

        bool Triangulate::is_retroprojection_valid(const vector3& worldPoint, const vector2& screenPoint, const matrix34& worldToCameraMatrix, const double& maximumRetroprojectionError)
        {
            vector2 projectedScreenPoint;
            const bool isRetroprojectionValid = utils::world_to_screen_coordinates(worldPoint, worldToCameraMatrix, projectedScreenPoint);
            if (not isRetroprojectionValid)
            {
                return false;
            }
            const double retroprojectionError = (screenPoint - projectedScreenPoint).norm();

            // true if retroprojection error is small enough
            return (retroprojectionError > maximumRetroprojectionError);
        }

        bool Triangulate::triangulate(const utils::Pose& currentPose, const utils::Pose& newPose, const vector2& point2Da, const vector2& point2Db, vector3& triangulatedPoint) 
        {
            const double cameraFX = Parameters::get_camera_1_focal_x();
            const double cameraFY = Parameters::get_camera_1_focal_y();
            const double cameraCX = Parameters::get_camera_1_center_x();
            const double cameraCY = Parameters::get_camera_1_center_y();
            const double maximumRetroprojectionError = Parameters::get_maximum_retroprojection_error();

            const matrix34& currentWorldToCameraMatrix = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());
            const matrix34& newWorldToCameraMatrix = utils::compute_world_to_camera_transform(newPose.get_orientation_quaternion(), newPose.get_position());

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

            const vector3& worldPoint = triangulationMatrix.leftCols<3>().jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-triangulationMatrix.col(3));
            if (not std::isfinite(worldPoint.x()) or 
                    not std::isfinite(worldPoint.y()) or 
                    not std::isfinite(worldPoint.z()))
            {
                // point triangulation failed
                return false;
            }

            // Check retroprojection of point in frame A
            const bool isRetroprojectionPointAValid = Triangulate::is_retroprojection_valid(worldPoint, point2Da, currentWorldToCameraMatrix, maximumRetroprojectionError);
            if (not isRetroprojectionPointAValid)
                return false;

            // Check retroprojection of point in frame B
            const bool isRetroprojectionPointBValid = Triangulate::is_retroprojection_valid(worldPoint, point2Db, newWorldToCameraMatrix, maximumRetroprojectionError);
            if (not isRetroprojectionPointBValid)
                return false;


            triangulatedPoint = worldPoint;
            return true;
        }

    }
}
