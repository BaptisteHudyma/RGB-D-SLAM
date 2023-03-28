#ifndef RGBDSLAM_TYPES_HPP
#define RGBDSLAM_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <vector>

namespace rgbd_slam {
/*
 *        Declare the most common types used in this program
 */

const double EulerToRadian = M_PI / 180.0;

using Matrixb = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
using matrixf = Eigen::MatrixXf;
using matrixd = Eigen::MatrixXd;
using vector2 = Eigen::Vector2d;
using vectorxd = Eigen::VectorXd;
using vectorb = Eigen::Vector<bool, Eigen::Dynamic>;
using vector3 = Eigen::Matrix<double, 3, 1>;
using vector4 = Eigen::Vector4d;
using matrix22 = Eigen::Matrix2d;
using matrix33 = Eigen::Matrix3d;
using matrix34 = Eigen::Matrix<double, 3, 4>;
using matrix43 = Eigen::Matrix<double, 4, 3>;
using matrix44 = Eigen::Matrix4d;
using quaternion = Eigen::Quaternion<double>;

using vector6 = Eigen::Matrix<double, 6, 1>;
using matrix66 = Eigen::Matrix<double, 6, 6>;

struct ScreenCoordinateCovariance : public matrix33
{
};
struct CameraCoordinateCovariance : public matrix33
{
};
struct WorldCoordinateCovariance : public matrix33
{
};

// define new classes to not mix types
struct WorldToCameraMatrix : public matrix44
{
};
struct CameraToWorldMatrix : public matrix44
{
};
struct PlaneWorldToCameraMatrix : public matrix44
{
};
struct PlaneCameraToWorldMatrix : public matrix44
{
};

struct EulerAngles
{
    double yaw;
    double pitch;
    double roll;

    EulerAngles() : yaw(0.0), pitch(0.0), roll(0.0) {};

    EulerAngles(const double y, const double p, const double r) : yaw(y), pitch(p), roll(r) {};
};

using vector3_vector = std::vector<vector3, Eigen::aligned_allocator<vector3>>;
} // namespace rgbd_slam

#endif
