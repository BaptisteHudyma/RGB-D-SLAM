#include "utils.hpp"

#include "parameters.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace rgbd_slam {
    namespace utils {

        void log(std::string_view message, const std::source_location& location)
        {
            std::cout << "[INF] "
                << std::filesystem::path(location.file_name()).filename().string() << "("
                << location.line() << ":" << location.column() <<  ") "
                //<< location.function_name() << " | "
                << message << std::endl;
        }
        void log_error(std::string_view message, const std::source_location& location)
        {
            std::cerr << "[ERR] "
                << std::filesystem::path(location.file_name()).filename().string() << "("
                << location.line() << ":" << location.column() <<  ") "
                //<< location.function_name() << " | "
                << message << std::endl;
        }

        const vector3 screen_to_world_coordinates(const double screenX, const double screenY, const double measuredZ, const matrix44& screenToWorldMatrix) 
        {
            const double x = (screenX - Parameters::get_camera_1_center_x()) * measuredZ / Parameters::get_camera_1_focal_x();
            const double y = (screenY - Parameters::get_camera_1_center_y()) * measuredZ / Parameters::get_camera_1_focal_y();

            vector4 worldPoint;
            worldPoint << x, y, measuredZ, 1.0;
            return screen_to_world_coordinates(worldPoint, screenToWorldMatrix).head<3>();
        }

        const vector4 screen_to_world_coordinates(const vector4& vector4d, const matrix44& screenToWorldMatrix)
        {
            return screenToWorldMatrix * vector4d;
        }

        bool world_to_screen_coordinates(const vector3& position3D, const matrix44& worldToScreenMatrix, vector2& screenCoordinates)
        {
            vector4 ptH;
            ptH << position3D, 1.0;
            const vector4& point4d = world_to_screen_coordinates(ptH, worldToScreenMatrix);
            assert(point4d[3] != 0);
            const vector3& point3D = point4d.head<3>() / point4d[3]; 

            if (point3D.z() <= 0) {
                return false;
            }

            const double inverseDepth  = 1.0 / point3D.z();
            const double screenX = Parameters::get_camera_1_focal_x() * point3D.x() * inverseDepth + Parameters::get_camera_1_center_x();
            const double screenY = Parameters::get_camera_1_focal_y() * point3D.y() * inverseDepth + Parameters::get_camera_1_center_y();

            screenCoordinates = vector2(screenX, screenY);
            return true;
        }

        const vector4 world_to_screen_coordinates(const vector4& vector4, const matrix44& worldToScreenMatrix)
        {
            return worldToScreenMatrix * vector4;
        }

        const matrix44 compute_camera_to_world_transform(const quaternion& rotation, const vector3& position)
        {
            matrix44 screenToWorldMatrix;
            screenToWorldMatrix << rotation.toRotationMatrix(), position,  0, 0, 0, 1;
            return screenToWorldMatrix;
        }

        const matrix44 compute_world_to_camera_transform(const quaternion& rotation, const vector3& position)
        {
            return compute_world_to_camera_transform(compute_camera_to_world_transform(rotation, position));
        }

        const matrix44 compute_world_to_camera_transform(const matrix44& screenToWorldMatrix)
        {
            return screenToWorldMatrix.inverse();
        }

        const matrix33 get_world_point_covariance(const vector2& screenPoint, const double depth, const matrix33& screenPointCovariance)
        {
            const double cameraFX = Parameters::get_camera_1_focal_x();
            const double cameraFY = Parameters::get_camera_1_focal_y();
            const double cameraCX = Parameters::get_camera_1_center_x();
            const double cameraCY = Parameters::get_camera_1_center_y();

            assert(cameraFX > 0);
            assert(cameraFY > 0);
            assert(cameraCX > 0);
            assert(cameraCY > 0);

            // Jacobian of the screen to world function. Use absolutes to prevent negative variances
            const matrix33 jacobian {
                {depth / cameraFX, 0.0,              abs(screenPoint.x() - cameraCX) / cameraFX },
                    {0.0,              depth / cameraFY, abs(screenPoint.y() - cameraCY) / cameraFY },
                    {0.0,              0.0,              1}
            };
            const matrix33& worldPointCovariance = jacobian * screenPointCovariance * jacobian.transpose();
            return worldPointCovariance;
        }


        const quaternion get_quaternion_from_euler_angles(const EulerAngles& eulerAngles)
        {
            const double cy = cos(eulerAngles.yaw * 0.5);
            const double sy = sin(eulerAngles.yaw * 0.5);
            const double cp = cos(eulerAngles.pitch * 0.5);
            const double sp = sin(eulerAngles.pitch * 0.5);
            const double cr = cos(eulerAngles.roll * 0.5);
            const double sr = sin(eulerAngles.roll * 0.5);

            const quaternion quat(
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


    }
}
