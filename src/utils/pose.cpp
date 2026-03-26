#include "pose.hpp"
#include "angle_utils.hpp"
#include "covariances.hpp"
#include "logger.hpp"
#include "types.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>

#include "extended_kalman_filter.hpp"

namespace rgbd_slam::utils {

/**
 * \brief Compute the quaternion multiplication matrix
 */
matrix44 get_quaternion_multiplication_matrix(const quaternion& vec)
{
    matrix44 skew;
    // clang-format off
    skew <<
    vec.w(), -vec.x(), -vec.y(), -vec.z(),  // w
    vec.x(),  vec.w(),  vec.z(), -vec.y(),  // x
    vec.y(), -vec.z(),  vec.w(),  vec.x(),  // y
    vec.z(),  vec.y(), -vec.x(),  vec.w();  // z
    // clang-format on
    return skew;
}

quaternion get_quaternion_from_speed(const vector3& vec) { return quaternion(0.0, vec.x(), vec.y(), vec.z()); }

vector4 get_rotation_vector(const quaternion& quat) noexcept
{
    // real quaternion order in vector (order x y z w)
    const vector4& r_xyzw = quat.coeffs();
    return vector4(r_xyzw(3), r_xyzw(0), r_xyzw(1), r_xyzw(2));
}

quaternion get_rotation_from_vector(const vector4& vec) noexcept
{
    return quaternion(vec(0), vec(1), vec(2), vec(3)).normalized();
}

template<int N = 13, int M = 7, int NE = N, int ME = M> class PoseTrackingEstimator :
    public tracking::StateEstimator<N, M, NE, ME>
{
  public:
    virtual ~PoseTrackingEstimator() = default;

    Eigen::Vector<double, N> f(const Eigen::Vector<double, N>& state) const noexcept override
    {
        return PoseBaseWithSpeed(state).predict(_deltaTime).get_vector();
    }

    Eigen::Matrix<double, NE, NE> f_jacobian(const Eigen::Vector<double, N>& state) const noexcept override
    {
        Eigen::Matrix<double, NE, NE> jacobian;
        PoseBaseWithSpeed(state).predict(_deltaTime, jacobian);
        return jacobian;
    }

    Eigen::Vector<double, M> h(const Eigen::Vector<double, N>& state) const noexcept override
    {
        // measurment of position and rotation
        // WILL NOT BE USED
        return state.template head<M>();
    }

    Eigen::Matrix<double, ME, NE> h_jacobian(const Eigen::Vector<double, N>& state) const noexcept override
    {
        // measurment of position and rotation
        // WILL NOT BE USED

        std::ignore = state;

        Eigen::Matrix<double, ME, NE> H;
        H.setZero();
        // position update
        H.template block<PoseBase::positionSize, PoseBase::positionSize>(PoseBase::positionIndex,
                                                                         PoseBase::positionIndex)
                .diagonal()
                .setConstant(1.0);
        // rotation update
        H.template block<PoseBase::rotationSize, PoseBase::rotationSize>(PoseBase::rotationIndex,
                                                                         PoseBase::rotationIndex)
                .diagonal()
                .setConstant(1.0);
        return H;
    }

    PoseTrackingEstimator(const Pose& pose,
                          const PoseBase& measurment,
                          const Eigen::Matrix<double, ME, ME>& measurmentCovariance,
                          const double deltaTime) :

        tracking::StateEstimator<N, M, NE, ME>(
                pose.get_vector(), pose.get_covar(), measurment.get_vector(), measurmentCovariance),
        _deltaTime(deltaTime)
    {
    }

  private:
    double _deltaTime;
};

/**
 * PoseBase
 */
PoseBase::PoseBase()
{
    _position.setZero();
    _rotation.setIdentity();
}

PoseBase::PoseBase(const vector3& position, const quaternion& rotation) { set_parameters(position, rotation); }

PoseBase::PoseBase(const vector7& vector) { PoseBase::set_from_vector(vector); }

void PoseBase::set_parameters(const vector3& position, const quaternion& rotation) noexcept
{
    _rotation = rotation.normalized();
    _position = position;
}

void PoseBase::display(std::ostream& os) const noexcept
{
    const EulerAngles displayAngles = get_euler_angles_from_quaternion(_rotation);
    os << "position: (" << _position.transpose() << ") millimeters | rotation: (" << displayAngles.yaw / EulerToRadian
       << ", " << displayAngles.pitch / EulerToRadian << ", " << displayAngles.roll / EulerToRadian << ") degrees";
}

std::ostream& operator<<(std::ostream& os, const PoseBase& pose)
{
    pose.display(os);
    return os;
}

vector7 PoseBase::get_vector() const noexcept
{
    vector7 t;
    t.segment<positionSize>(positionIndex) = _position;
    t.segment<rotationSize>(rotationIndex) = get_rotation_vector(_rotation);
    return t;
}

void PoseBase::set_from_vector(const vector7& vec)
{
    const vector4& r_wxyz = vec.segment<rotationSize>(rotationIndex);

    set_parameters(vec.segment<positionSize>(positionIndex), get_rotation_from_vector(r_wxyz));
}

double PoseBase::get_position_error(const PoseBase& pose) const noexcept
{
    return (pose.get_position() - _position).norm();
}

double PoseBase::get_rotation_error(const PoseBase& pose) const noexcept
{
    const double distanceRadian = _rotation.angularDistance(pose.get_rotation_quaternion());
    return distanceRadian / EulerToRadian;
}

/*
 * PoseBaseWithSpeed
 */

PoseBaseWithSpeed::PoseBaseWithSpeed() : PoseBase(), _rotationSpeed(vector3::Zero()), _positionSpeed(vector3::Zero()) {}

PoseBaseWithSpeed::PoseBaseWithSpeed(const vector3& position, const quaternion& rotation) :
    PoseBase(position, rotation),
    _rotationSpeed(vector3::Zero()),
    _positionSpeed(vector3::Zero())
{
}

PoseBaseWithSpeed::PoseBaseWithSpeed(const Eigen::Vector<double, 13>& vector)
{
    PoseBaseWithSpeed::set_from_vector(vector);
}

// 1 over update delay
static constexpr double linearDecayRate = 0.9;
static constexpr double angularDecayRate = 0.9;

PoseBaseWithSpeed PoseBaseWithSpeed::predict(const double deltaTime) const
{
    const vector3& p = get_position();
    const vector4& q = get_rotation_vector(get_rotation_quaternion());

    const double linearDecay = linearDecayRate;
    const double angularDecay = angularDecayRate;

    // get the predicted state
    const vector3 predictedPosition = p + _positionSpeed * deltaTime;

    const vector4 qDot = 0.5 * get_quaternion_multiplication_matrix(get_quaternion_from_speed(_rotationSpeed)) * q;
    const vector4 qres = q + qDot * deltaTime;
    const quaternion& predictedRotation = get_rotation_from_vector(qres);

    PoseBaseWithSpeed result(predictedPosition, predictedRotation);
    result._positionSpeed = _positionSpeed * linearDecay;
    result._rotationSpeed = _rotationSpeed * angularDecay;
    return result;
}

PoseBaseWithSpeed PoseBaseWithSpeed::predict(const double deltaTime, Eigen::Matrix<double, 13, 13>& jacobian) const
{
    const double linearDecay = linearDecayRate;
    const double angularDecay = angularDecayRate;

    // rotation derivation
    matrix44 dSpeed = get_quaternion_multiplication_matrix(get_quaternion_from_speed(_rotationSpeed * deltaTime));
    dSpeed.diagonal().setConstant(1.0);

    // rotation speed
    const quaternion& q = get_rotation_quaternion();
    Eigen::Matrix<double, 4, 3> speedDerivative;
    // clang-format off
    speedDerivative <<
    -q.x(), -q.y(), -q.z(), // w
     q.w(), -q.z(),  q.y(), // x
     q.z(),  q.w(), -q.x(), // y
    -q.y(),  q.x(),  q.w(); // z
    // clang-format on
    speedDerivative *= deltaTime / 2.0;

    jacobian.setIdentity();

    // position update is linear
    jacobian.block<Pose::linearErrorSize, Pose::linearErrorSize>(Pose::linearErrorIndex, Pose::linearSpeedErrorIndex)
            .diagonal()
            .setConstant(deltaTime);

    // decay speed
    jacobian.block<Pose::rotationErrorSize, Pose::rotationErrorSize>(Pose::linearSpeedErrorIndex,
                                                                     Pose::linearSpeedErrorIndex)
            .diagonal() *= linearDecay;

    // rotation update (small update, so linear)
    jacobian.block<Pose::rotationErrorSize, Pose::rotationErrorSize>(Pose::rotationErrorIndex,
                                                                     Pose::rotationErrorIndex) = dSpeed;
    jacobian.block<Pose::rotationErrorSize, Pose::rotationSpeedSize>(Pose::rotationErrorIndex,
                                                                     Pose::angularSpeedErrorIndex) = speedDerivative;

    // decay speed
    jacobian.block<Pose::angularSpeedErrorSize, Pose::angularSpeedErrorSize>(Pose::angularSpeedErrorIndex,
                                                                             Pose::angularSpeedErrorIndex)
            .diagonal() *= angularDecay;

    return predict(deltaTime);
}

/**
 * Pose
 */

void Pose::build_pose_tracker() const
{
    const double deltaTime = 1.0 / 30.0;

    _poseKalman = std::make_unique<tracking::ExtendedKalmanFilter<13, 7>>(get_Q(deltaTime));
}

Pose::PoseVarianceT Pose::get_Q(double deltaTime) const
{
    static constexpr double sigmaA = 1.0; // m/s²
    static constexpr double sigmaB = 2.0; // rad/s²

    const double dt2 = deltaTime * deltaTime;
    const double dt3 = dt2 * deltaTime;

    const double q11_lin = SQR(sigmaA) * dt3 / 3.0;
    const double q12_lin = SQR(sigmaA) * dt2 / 2.0;
    const double q22_lin = SQR(sigmaA) * deltaTime;
    const double q11_ang = SQR(sigmaB) * dt3 / 3.0;
    const double q22_ang = SQR(sigmaB) * deltaTime;

    PoseVarianceT Q = PoseVarianceT::Zero();
    // position noise
    Q.block<Pose::linearErrorSize, Pose::linearErrorSize>(Pose::linearErrorIndex, Pose::linearErrorIndex)
            .diagonal()
            .setConstant(q11_lin);
    Q.block<Pose::linearErrorSize, Pose::linearSpeedErrorSize>(Pose::linearErrorIndex, Pose::linearSpeedErrorIndex)
            .diagonal()
            .setConstant(q12_lin);
    Q.block<Pose::linearSpeedErrorSize, Pose::linearSpeedErrorSize>(Pose::linearSpeedErrorIndex,
                                                                    Pose::linearSpeedErrorIndex)
            .diagonal()
            .setConstant(q22_lin);
    // rotation noise
    Q.block<Pose::rotationErrorSize, Pose::rotationErrorSize>(Pose::rotationErrorIndex, Pose::rotationErrorIndex)
            .diagonal()
            .setConstant(q11_ang);

    /*
    Q.block<Pose::angularSpeedErrorSize, Pose::angularSpeedErrorSize>(Pose::angularSpeedErrorIndex,
                                                                  Pose::rotationErrorIndex)
        .diagonal()
        .setConstant(q12_ang);
    */

    Q.block<Pose::angularSpeedErrorSize, Pose::angularSpeedErrorSize>(Pose::angularSpeedErrorIndex,
                                                                      Pose::angularSpeedErrorIndex)
            .diagonal()
            .setConstant(q22_ang);

    // make symetric
    Q = Q.selfadjointView<Eigen::Lower>();
    return Q;
}

Pose::Pose() : PoseBaseWithSpeed(), _latestUpdateTime_s(0.0), _poseVariance(PoseVarianceT::Zero())
{
    _poseVariance.block<6, 6>(linearSpeedErrorIndex, linearSpeedErrorIndex).diagonal().setConstant(1e2);

    if (_poseKalman == nullptr)
    {
        build_pose_tracker();
    }
}

Pose::Pose(const vector3& position, const quaternion& rotation) :
    PoseBaseWithSpeed(position, rotation),
    _latestUpdateTime_s(0.0),
    _poseVariance(PoseVarianceT::Zero())
{
    _poseVariance.block<6, 6>(linearSpeedErrorIndex, linearSpeedErrorIndex).diagonal().setConstant(1e2);

    if (_poseKalman == nullptr)
    {
        build_pose_tracker();
    }
}

Pose::Pose(const vector3& position, const quaternion& rotation, const PoseVarianceT& poseVariance) :
    PoseBaseWithSpeed(position, rotation),
    _latestUpdateTime_s(0.0),
    _poseVariance(poseVariance)
{
    if (_poseKalman == nullptr)
    {
        build_pose_tracker();
    }
}

Pose::Pose(const Eigen::Vector<double, 13>& vector) : _latestUpdateTime_s(0.0), _poseVariance(PoseVarianceT::Zero())
{
    PoseBaseWithSpeed::set_from_vector(vector);
}

void Pose::display(std::ostream& os) const noexcept
{
    PoseBase::display(os);
    os << std::endl << "position standard dev (meters/degrees) : " << std::endl;
    os << "x\ty\tz\t|\troll\tpitch\tyaw" << std::endl;
    vector6 poseStd = get_pose_variance().diagonal().cwiseSqrt();
    os << poseStd.head<3>().transpose() << "\t|\t" << poseStd.tail<3>().transpose() * 180.0 / M_PI << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Pose& pose)
{
    pose.display(os);
    return os;
}

Pose Pose::predict(const double measurmentTime_s) const noexcept
{
    // invalid timestamp
    if (_latestUpdateTime_s < 0)
    {
        outputs::log(std::format("First prediction call"));

        // init
        Pose result = *this;
        result._latestUpdateTime_s = measurmentTime_s;
        return result;
    }

    // same time, return same prediction
    if (abs(measurmentTime_s - _latestUpdateTime_s) < 0.001)
    {
        Pose result = *this;
        result._latestUpdateTime_s = measurmentTime_s;
        return result;
    }
    if (measurmentTime_s < _latestUpdateTime_s)
    {
        outputs::log_error(std::format(
                "time must advance in the future: {:.4f} -> {:.4f}", _latestUpdateTime_s, measurmentTime_s));
        Pose result = *this;
        result._latestUpdateTime_s = measurmentTime_s;
        return result;
    }
    const double deltaT = measurmentTime_s - _latestUpdateTime_s;

    // use prediction
    matrix77 c = matrix77::Zero();
    PoseTrackingEstimator estimator(*this, PoseBase(), c, deltaT);
    const auto& [predictedPoseVector, predictedCovariance] = _poseKalman->predict_state(&estimator, get_Q(deltaT));

    Pose result(predictedPoseVector);
    result._poseVariance = predictedCovariance;
    result._latestUpdateTime_s = measurmentTime_s;
    return result;
}

matrix66 Pose::get_pose_variance() const noexcept
{
    const auto& quat = get_rotation_quaternion();

    // transform to position x euler with the jacobian:
    Eigen::Matrix<double, 6, positionSize + rotationSize> toPoseJac;
    toPoseJac.setZero();
    toPoseJac.block<positionSize, positionSize>(positionIndex, positionIndex).diagonal().setConstant(1.0);
    toPoseJac.block<3, rotationSize>(rotationIndex, rotationIndex) = utils::get_quaternion_to_euler_jacobian(quat);

    return utils::propagate_covariance(
            _poseVariance.block<positionSize + rotationSize, positionSize + rotationSize>(0, 0).eval(), toPoseJac);
}

matrix77 Pose::get_pose_variance_quaternion() const noexcept
{
    return _poseVariance.block<positionSize + rotationSize, positionSize + rotationSize>(0, 0);
}

bool Pose::update_with_new_pose(const PoseBase& measurment,
                                const matrix77& measurmentCovariance,
                                const double measurmentTime_s)
{
    if (_latestUpdateTime_s <= 0)
    {
        // init
        _latestUpdateTime_s = measurmentTime_s;
        return true;
    }
    if (measurmentTime_s <= _latestUpdateTime_s)
    {
        outputs::log_error(std::format(
                "time must advance in the future: {:.4f} -> {:.4f}", _latestUpdateTime_s, measurmentTime_s));
        return false;
    }
    const double deltaT = measurmentTime_s - _latestUpdateTime_s;

    try
    {
        PoseTrackingEstimator estimator(*this, measurment, measurmentCovariance, deltaT);

        const auto& [newState, newCovariance] = _poseKalman->get_new_state(&estimator);

        if (not utils::is_covariance_valid(newCovariance))
        {
            outputs::log_error("Pose covariance invalid after merge");
            return false;
        }

        set_from_vector(newState);
        _poseVariance = newCovariance;
        _latestUpdateTime_s = measurmentTime_s;
        return true;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exception: " + std::string(ex.what()));
    }
    return false;
}

} // namespace rgbd_slam::utils
