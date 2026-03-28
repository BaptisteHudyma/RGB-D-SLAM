#include "covariances.hpp"

#include "../parameters.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <cmath>
#include <stdexcept>

#include <boost/math/distributions/chi_squared.hpp>

namespace rgbd_slam::utils {

double get_depth_quantization(const double depth_m) noexcept
{
    // minimum depth diparity at z is the quadratic function  a + b z + c z^2
    const static double depthSigmaError = parameters::depthSigmaError;
    constexpr double depthSigmaMultiplier = parameters::depthSigmaMultiplier;
    constexpr double depthSigmaMargin = parameters::depthSigmaMargin;
    const double quantization_mm = depthSigmaMargin + depthSigmaMultiplier * depth_m + depthSigmaError * SQR(depth_m);
    return std::max(0.5, quantization_mm) / 1000.0;
}

matrix23 get_camera_to_screen2d_jacobian(const CameraCoordinate& point)
{
    const static vector2 cameraF = Parameters::get_camera_1_focal();
    // Jacobian of the camera to screen function
    matrix23 jacobian {{cameraF.x() / point.z(), 0.0, -cameraF.x() * point.x() / SQR(point.z())},
                       {0.0, cameraF.y() / point.z(), -cameraF.y() * point.y() / SQR(point.z())}};
    return jacobian;
}

matrix33 get_camera_to_screen_jacobian(const CameraCoordinate& point)
{
    const matrix23& camToScreenJac = get_camera_to_screen2d_jacobian(point);
    matrix33 jacobian;
    // camera to screen u & v
    jacobian.block<2, 3>(0, 0) = camToScreenJac;
    // camera to screen z
    jacobian.block<1, 3>(2, 0) = vector3::UnitZ().transpose();
    return jacobian;
}

ScreenCoordinateCovariance get_screen_point_covariance(const CameraCoordinate& point,
                                                       const matrix33& pointCovariance) noexcept
{
    // Jacobian of the camera to screen function
    const matrix33& jacobian = get_camera_to_screen_jacobian(point);

    ScreenCoordinateCovariance screenPointCovariance;
    screenPointCovariance << utils::propagate_covariance(pointCovariance, jacobian);
    return screenPointCovariance;
}

ScreenCoordinateCovariance get_screen_point_covariance(const WorldCoordinate& point,
                                                       const WorldCoordinateCovariance& pointCovariance,
                                                       const WorldToCameraMatrix& worldToCamera) noexcept
{
    return get_screen_point_covariance(point.to_camera_coordinates(worldToCamera),
                                       get_camera_point_covariance(pointCovariance, worldToCamera, matrix66::Zero()));
}

ScreenCoordinateCovariance get_screen_point_covariance(const CameraCoordinate& point,
                                                       const CameraCoordinateCovariance& pointCovariance) noexcept
{
    return get_screen_point_covariance(point.base(), pointCovariance.base());
}

CameraCoordinateCovariance get_camera_point_covariance(const WorldCoordinateCovariance& worldPointCovariance,
                                                       const WorldToCameraMatrix& worldToCamera,
                                                       const matrix66& poseCovariance) noexcept
{
    const matrix33& rotation = worldToCamera.rotation();

    CameraCoordinateCovariance cov;
    // TODO: correct covariance with pose
    cov << propagate_covariance(worldPointCovariance, rotation) + poseCovariance.block<3, 3>(0, 0);
    return cov;
}

WorldCoordinateCovariance get_world_point_covariance(const CameraCoordinateCovariance& cameraPointCovariance,
                                                     const CameraToWorldMatrix& cameraToWorld,
                                                     const matrix66& poseCovariance) noexcept
{
    const matrix33& rotation = cameraToWorld.rotation();

    WorldCoordinateCovariance cov;
    // TODO: correct covariance with pose
    cov << propagate_covariance(cameraPointCovariance, rotation) + poseCovariance.block<3, 3>(0, 0);
    return cov;
}

WorldCoordinateCovariance get_world_point_covariance(const ScreenCoordinate& screenPoint,
                                                     const CameraToWorldMatrix& cameraToWorld,
                                                     const matrix66& poseCovariance) noexcept
{
    return get_world_point_covariance(utils::get_camera_point_covariance(screenPoint), cameraToWorld, poseCovariance);
}

CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint) noexcept
{
    return get_camera_point_covariance(screenPoint, screenPoint.get_covariance());
}

CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint,
                                                       const ScreenCoordinateCovariance& screenPointCovariance) noexcept
{
    static const vector2 cameraF = Parameters::get_camera_1_focal();
    static const vector2 cameraC = Parameters::get_camera_1_center();

    // Jacobian of the screen to camera function
    const matrix33 jacobian {{screenPoint.z() / cameraF.x(), 0.0, (screenPoint.x() - cameraC.x()) / cameraF.x()},
                             {0.0, screenPoint.z() / cameraF.y(), (screenPoint.y() - cameraC.y()) / cameraF.y()},
                             {0.0, 0.0, 1.0}};

    return CameraCoordinateCovariance {propagate_covariance(screenPointCovariance, jacobian)};
}

matrix44 compute_plane_covariance(const PlaneCoordinates& planeParameters, const matrix33& pointCloudCovariance)
{
    if (not is_covariance_valid(pointCloudCovariance))
    {
        throw std::invalid_argument(
                "compute_plane_covariance: the argument pointCloudCovariance is an invalid covariance matrix");
    }

    const vector3& normal = planeParameters.get_normal();
    const double d = planeParameters.get_d();
    if (utils::double_equal(d, 0.0))
    {
        throw std::invalid_argument("compute_plane_covariance: The d of planeParameters should not be 0");
    }
    if (not utils::double_equal(normal.norm(), 1.0))
    {
        throw std::invalid_argument("compute_plane_covariance: The normal of planeParameters should have a norm of 1");
    }

    // reduce the parametrization
    const vector3 parameters = normal * d;
    const double a = parameters.x();
    const double b = parameters.y();
    const double c = parameters.z();

    const double aSquared = SQR(a);
    const double bSquared = SQR(b);
    const double cSquared = SQR(c);

    // common divider of all partial derivatives
    const double divider = pow(aSquared + bSquared + cSquared, 3.0 / 2.0);
    const double common = 1.0 / sqrt(aSquared + bSquared + cSquared);

    // compute the jacobian of the 3 parameter plane to 4 parameters plane transformation -> (vect, 1) / norm(vect)
    // with vect = normal * d
    const matrix43 jacobian({
            {common - aSquared / divider, -(a * b) / divider, -(a * c) / divider},
            {-(a * b) / divider, common - bSquared / divider, -(b * c) / divider},
            {-(a * c) / divider, -(b * c) / divider, common - cSquared / divider},
            {-a / divider, -b / divider, -c / divider},
    });

    matrix44 planeParameterCovariance = propagate_covariance(pointCloudCovariance, jacobian);
    // add a little bit of variance on the diagonal to counter floatting points errors
    planeParameterCovariance.diagonal()(0) += SQR(0.01); // meters
    planeParameterCovariance.diagonal()(1) += SQR(0.3);
    planeParameterCovariance.diagonal()(2) += SQR(0.3);
    planeParameterCovariance.diagonal()(3) += SQR(0.3);

    std::string failureReason;
    if (not is_covariance_valid(planeParameterCovariance, failureReason))
    {
        throw std::logic_error(
                "compute_plane_covariance: planeParameterCovariance is an invalid covariance matrix after process: " +
                failureReason);
    }
    return planeParameterCovariance;
}

matrix33 compute_reduced_plane_point_cloud_covariance(const PlaneCoordinates& planeParameters,
                                                      const matrix44& planeCloudCovariance)
{
    if (not is_covariance_valid(planeCloudCovariance))
    {
        throw std::invalid_argument(
                "compute_reduced_plane_point_cloud_covariance: planeCloudCovariance is an invalid covariance matrix");
    }

    const vector3& normal = planeParameters.get_normal();
    const double d = planeParameters.get_d();
    if (utils::double_equal(d, 0.0))
    {
        throw std::invalid_argument(
                "compute_reduced_plane_point_cloud_covariance: compute_plane_covariance: The d of planeParameters "
                "should not be 0");
    }
    if (not utils::double_equal(normal.norm(), 1.0))
    {
        throw std::invalid_argument(
                "compute_reduced_plane_point_cloud_covariance: compute_plane_covariance: The normal of planeParameters "
                "should have a norm of 1");
    }

    // compute the jacobian of the 4 parameter plane to 3 parameters plane transformation -> vect = normal * d
    const matrix34 jacobian({
            {d, 0, 0, normal.x()},
            {0, d, 0, normal.y()},
            {0, 0, d, normal.z()},
    });

    const matrix33& pointCloudCovariance =
            // add a little bit of variance on the diagonal to counter floatting points errors
            propagate_covariance(planeCloudCovariance, jacobian, 0.01);
    if (not is_covariance_valid(pointCloudCovariance))
    {
        throw std::logic_error(
                "compute_reduced_plane_point_cloud_covariance: pointCloudCovariance is an invalid covariance matrix "
                "after process");
    }
    return pointCloudCovariance;
}

matrix44 get_world_plane_covariance(const PlaneCameraCoordinates& planeCoordinates,
                                    const CameraToWorldMatrix& cameraToWorldMatrix,
                                    const PlaneCameraToWorldMatrix& planeCameraToWorldMatrix,
                                    const matrix44& planeCovariance,
                                    const matrix66& worldPoseCovariance)
{
    if (not is_covariance_valid(planeCovariance))
    {
        throw std::invalid_argument("get_world_plane_covariance: planeCovariance is an invalid covariance matrix");
    }

    // transform to point form
    const matrix33& pointCloudCovariance =
            compute_reduced_plane_point_cloud_covariance(planeCoordinates, planeCovariance);

    // covert covariance to world
    const matrix33& rotation = cameraToWorldMatrix.rotation();
    const matrix33& pointCloudWorlCovariance =
            // TODO: resolve true world covariance
            propagate_covariance(pointCloudCovariance, rotation) + worldPoseCovariance.block<3, 3>(0, 0);

    std::string failureReason;
    if (not is_covariance_valid(pointCloudWorlCovariance, failureReason))
    {
        throw std::logic_error(
                "get_world_plane_covariance: pointCloudWorlCovariance is an invalid covariance matrix after process: " +
                failureReason);
    }
    // convert back to plane hessian form
    return compute_plane_covariance(planeCoordinates.to_world_coordinates(planeCameraToWorldMatrix),
                                    pointCloudWorlCovariance);
}

matrix33 get_world_to_camera_pose_jacobian(const WorldCoordinate& p, const WorldToCameraMatrix& w2c)
{
    matrix33 hatMat;
    // clang-format off
    hatMat <<
            0.0, -p.z(), p.y(),
            p.z(), 0.0, -p.x(),
            -p.y(), p.x(), 0.0;
    // clang-format on

    const matrix33& Pcdr = -w2c.rotation() * hatMat;

    return Pcdr;
}

Eigen::Matrix<double, 3, 6> world_transform_of_point_jacobian(const WorldCoordinate& point,
                                                              const WorldToCameraMatrix& w2c)
{
    const matrix33& toScreenJacobian = utils::get_camera_to_screen_jacobian(point.to_camera_coordinates(w2c));

    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.block<3, 3>(0, 0) = toScreenJacobian * matrix33::Identity();
    jacobian.block<3, 3>(0, 3) = toScreenJacobian * get_world_to_camera_pose_jacobian(point, w2c);
    return jacobian;
}

Eigen::Matrix<double, 2, 6> world_transform_of_2d_point_jacobian(const WorldCoordinate& point,
                                                                 const WorldToCameraMatrix& w2c)
{
    const matrix23& toScreenJacobian = utils::get_camera_to_screen2d_jacobian(point.to_camera_coordinates(w2c));

    Eigen::Matrix<double, 2, 6> jacobian;
    jacobian.block<2, 3>(0, 0) = toScreenJacobian * matrix33::Identity();
    jacobian.block<2, 3>(0, 3) = toScreenJacobian * get_world_to_camera_pose_jacobian(point, w2c);
    return jacobian;
}

Eigen::Matrix<double, 3, 4> get_quaternion_to_euler_jacobian(const Eigen::Quaterniond& quat)
{
    const double q1 = quat.w();
    const double q2 = quat.x();
    const double q3 = quat.y();
    const double q4 = quat.z();

    const double denomA = SQR(q3 + q2) + SQR(q4 + q1);
    const double denomB = SQR(q3 - q2) + SQR(q4 - q1);

    const double denomC = sqrt(1.0 - 4.0 * SQR(q2 * q3 + q1 * q4));

    Eigen::Matrix<double, 3, 4> jac;
    // clang-format off
    jac << 
    -(q3+q2)/denomA + (q3-q2)/denomB, (q4+q1)/denomA - (q4-q1)/denomB, (q4+q1)/denomA + (q4-q1)/denomB, -(q3+q2)/denomA - (q3-q2)/denomB,
    2.0 * q4 / denomC, 2.0 * q3 / denomC, 2.0 * q2 / denomC, 2.0 * q1 / denomC, 
    -(q3+q2)/denomA - (q3-q2)/denomB, (q4+q1)/denomA + (q4-q1)/denomB, (q4+q1)/denomA - (q4-q1)/denomB, -(q3+q2)/denomA + (q3-q2)/denomB;
    // clang-format on
    return jac;
}

cv::RotatedRect get_rotated_rect_screen_covariance(const vector2& center, const matrix22& screenCovariance)
{
    // 95% inclusion range;
    static const double chiTest = sqrt(boost::math::quantile(boost::math::chi_squared(2.0), 0.95));

    Eigen::SelfAdjointEigenSolver<matrix22> eigenSolver(screenCovariance);
    // ascending order
    const vector2& eigenValues = eigenSolver.eigenvalues();
    const matrix22& eigenVector = eigenSolver.eigenvectors();

    double angle = atan2(eigenVector.col(1).y(), eigenVector.col(1).x());
    if (angle < 0.0)
        angle += 2.0 * M_PI;
    angle *= 180.0 / M_PI;

    return cv::RotatedRect(cv::Point(static_cast<int>(center.x()), static_cast<int>(center.y())),
                           cv::Size2d(chiTest * sqrt(eigenValues(1)), chiTest * sqrt(eigenValues(0))),
                           -angle);
}

} // namespace rgbd_slam::utils
