#ifndef RGBDSLAM_TYPES_HPP
#define RGBDSLAM_TYPES_HPP

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Macros.h>
#include <Eigen/src/Core/util/XprHelper.h>
#include <Eigen/src/QR/CompleteOrthogonalDecomposition.h>
#include <vector>

namespace rgbd_slam {
/*
 *        Declare the most common types used in this program
 */

constexpr double EulerToRadian = M_PI / 180.0;

using Matrixb = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
using matrixf = Eigen::MatrixXf;
using matrixd = Eigen::MatrixXd;
using vector2 = Eigen::Vector2d;
using vectorxd = Eigen::VectorXd;
using vectorb = Eigen::Vector<bool, Eigen::Dynamic>;
using vector3 = Eigen::Matrix<double, 3, 1>;
using vector4 = Eigen::Vector4d;

using matrix22 = Eigen::Matrix2d;
using matrix23 = Eigen::Matrix<double, 2, 3>;
using matrix24 = Eigen::Matrix<double, 2, 4>;

using matrix32 = Eigen::Matrix<double, 3, 2>;
using matrix33 = Eigen::Matrix3d;
using matrix34 = Eigen::Matrix<double, 3, 4>;

using matrix42 = Eigen::Matrix<double, 4, 2>;
using matrix43 = Eigen::Matrix<double, 4, 3>;
using matrix44 = Eigen::Matrix4d;

using quaternion = Eigen::Quaternion<double>;

using vector6 = Eigen::Matrix<double, 6, 1>;
using vector7 = Eigen::Matrix<double, 7, 1>;

using matrix66 = Eigen::Matrix<double, 6, 6>;
using matrix77 = Eigen::Matrix<double, 7, 7>;

struct ScreenCoordinate2dCovariance : public matrix22
{
};
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
struct TransitionMatrix : public matrix44
{
    using matrix44::matrix44;
    [[nodiscard]] matrix33 rotation() const noexcept { return this->block<3, 3>(0, 0); };
    [[nodiscard]] vector3 translation() const noexcept { return this->col(3).head<3>(); };

    matrix33 essencial_matrix() const
    {
        const vector3& t = translation();
        // skew /cross product matrix
        matrix33 t_hat;
        t_hat << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;

        return t_hat * rotation();
    }
};

struct WorldToCameraMatrix : public TransitionMatrix
{
};
struct CameraToWorldMatrix : public TransitionMatrix
{
};
struct PlaneWorldToCameraMatrix : public TransitionMatrix
{
};
struct PlaneCameraToWorldMatrix : public TransitionMatrix
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

// define an optimized squared
template<class T> T constexpr inline SQR(const T x) { return x * x; }

using vector3_vector = std::vector<vector3, Eigen::aligned_allocator<vector3>>;

} // namespace rgbd_slam

// alow to round eigen matrix when needed
template<typename scalar> struct threshold_op
{
    scalar threshold;
    threshold_op(const scalar& value) : threshold(value) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const scalar operator()(const scalar& a) const
    {
        return threshold < std::abs(a) ? a : scalar(0);
    }
    template<typename packet> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const packet packetOp(const packet& a) const
    {
        using namespace Eigen::internal;
        return pand(pcmp_lt(pset1<packet>(threshold), pabs(a)), a);
    }
};

/**
 * \brief Moore Penrose pseudo inverse
 */
template<typename MatType> auto pseudoInverse(const MatType& a)
{
    return a.completeOrthogonalDecomposition().pseudoInverse();
}

namespace Eigen {
namespace internal {

template<typename scalar> struct functor_traits<threshold_op<scalar>>
{
    enum
    {
        Cost = 3 * NumTraits<scalar>::AddCost,
        PacketAccess = packet_traits<scalar>::HasAbs
    };
};

/// round the given eigen matrix to zero
#define ROUND_MAT(mat) mat.unaryExpr(threshold_op<double>(1e-10))

} // namespace internal
} // namespace Eigen

#endif
