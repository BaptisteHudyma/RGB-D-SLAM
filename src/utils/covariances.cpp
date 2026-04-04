#include "covariances.hpp"

#include "coordinates/point_coordinates.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include "utils/distance_utils.hpp"

#include <Eigen/src/Core/util/Constants.h>

#include <boost/math/distributions/chi_squared.hpp>
#include <cmath>
#include <stdexcept>

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
    matrix33 worldToScreenJacobian;
    ScreenCoordinate sc;
    std::ignore = point.to_screen_coordinates(w2c, sc, worldToScreenJacobian);

    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.block<3, 3>(0, 0) = worldToScreenJacobian * matrix33::Identity();
    jacobian.block<3, 3>(0, 3) = worldToScreenJacobian;
    return jacobian;
}

Eigen::Matrix<double, 2, 6> world_transform_of_2d_point_jacobian(const WorldCoordinate& point,
                                                                 const WorldToCameraMatrix& w2c)
{
    matrix23 worldToScreenJacobian;
    ScreenCoordinate2D sc;
    std::ignore = point.to_screen_coordinates(w2c, sc, worldToScreenJacobian);

    Eigen::Matrix<double, 2, 6> jacobian;
    jacobian.block<2, 3>(0, 0) = worldToScreenJacobian * matrix33::Identity();
    jacobian.block<2, 3>(0, 3) = worldToScreenJacobian;
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
