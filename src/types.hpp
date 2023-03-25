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

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> Matrixb;
typedef Eigen::MatrixXf matrixf;
typedef Eigen::MatrixXd matrixd;
typedef Eigen::Vector2d vector2;
typedef Eigen::VectorXd vectorxd;
typedef Eigen::Vector<bool, Eigen::Dynamic> vectorb;
typedef Eigen::Matrix<double, 3, 1> vector3;
typedef Eigen::Vector4d vector4;
typedef Eigen::Matrix2d matrix22;
typedef Eigen::Matrix3d matrix33;
typedef Eigen::Matrix<double, 3, 4> matrix34;
typedef Eigen::Matrix<double, 4, 3> matrix43;
typedef Eigen::Matrix4d matrix44;
typedef Eigen::Quaternion<double> quaternion;

typedef Eigen::Matrix<double, 6, 1> vector6;
typedef Eigen::Matrix<double, 6, 6> matrix66;

struct screenCoordinateCovariance : public matrix33
{
};
struct cameraCoordinateCovariance : public matrix33
{
};
struct worldCoordinateCovariance : public matrix33
{
};

// define new classes to not mix types
struct worldToCameraMatrix : public matrix44
{
};
struct cameraToWorldMatrix : public matrix44
{
};
struct planeWorldToCameraMatrix : public matrix44
{
};
struct planeCameraToWorldMatrix : public matrix44
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

typedef std::vector<vector3, Eigen::aligned_allocator<vector3>> vector3_vector;
} // namespace rgbd_slam

#endif
