#ifndef RGBDSLAM_UTILS_EXTENDED_KALMAN_FILTER_HPP
#define RGBDSLAM_UTILS_EXTENDED_KALMAN_FILTER_HPP

#include "covariances.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <stdexcept>

namespace rgbd_slam::tracking {

template<int N, int M, int NE, int ME> class StateEstimator
{
  public:
    Eigen::Vector<double, N> state() const noexcept { return _feature; }
    Eigen::Matrix<double, NE, NE> state_covariance() const noexcept { return _featureCovariance; }

    Eigen::Vector<double, M> measurment() const noexcept { return _measurment; }
    Eigen::Matrix<double, ME, ME> measurment_covariance() const noexcept { return _measurmentCovariance; }

    /**
     * Compute the new state estimate from the current state (default is no dynamics)
     */
    virtual Eigen::Vector<double, N> f(const Eigen::Vector<double, N>& state) const noexcept
    {
        // no dynamics, just return the same state
        return state;
    }

    Eigen::Matrix<double, NE, NE> f_jacobian(const Eigen::Vector<double, N>& state) const noexcept
    {
        std::ignore = state;
        // no dynamic jacobian
        return Eigen::Matrix<double, NE, NE>::Identity();
    }

    /**
     * Compute the measurment equation from the given state
     */
    virtual Eigen::Vector<double, M> h(const Eigen::Vector<double, N>& state) const noexcept = 0;

    virtual Eigen::Matrix<double, ME, NE> h_jacobian(const Eigen::Vector<double, N>& state) const noexcept = 0;

    /**
     * \brief Special innovation covariance computation
     */
    virtual inline Eigen::Matrix<double, ME, ME> h_innovation(
            const Eigen::Vector<double, N>& state,
            const Eigen::Matrix<double, NE, NE>& estimateErrorCovariance,
            const Eigen::Matrix<double, ME, NE>& hJacobian) const noexcept
    {
        std::ignore = state;
        return utils::propagate_covariance(estimateErrorCovariance, hJacobian);
    }

    virtual inline Eigen::Vector<double, N> get_new_state(
            const Eigen::Vector<double, N>& predictedState,
            const Eigen::Matrix<double, NE, ME>& kalmanGain) const noexcept
    {
        static_assert(M == ME, "measurment error is not the same as measurment size, you must overload this function");
        static_assert(N == NE, "state error is not the same as state size, you must overload this function");

        return predictedState + kalmanGain * (measurment() - h(predictedState));
    }

    StateEstimator(const Eigen::Vector<double, N>& feature,
                   const Eigen::Matrix<double, NE, NE>& featureCovariance,
                   const Eigen::Vector<double, M>& measurment,
                   const Eigen::Matrix<double, ME, ME>& measurmentCovariance) :
        _feature(feature),
        _featureCovariance(featureCovariance),
        _measurment(measurment),
        _measurmentCovariance(measurmentCovariance)
    {
    }

  private:
    const Eigen::Vector<double, N> _feature;
    const Eigen::Matrix<double, NE, NE> _featureCovariance;
    const Eigen::Vector<double, M> _measurment;
    const Eigen::Matrix<double, ME, ME> _measurmentCovariance;
};

/**
 * \brief Implement a Kalman filter that can be shared by multiple systems, if they share the same dimentions.
 * N is number of state variables
 * M is number of outputs
 * NE is the error size of the state
 * ME is the error size of the measurment
 */
template<int N, int M, int NE = N, int ME = M> class ExtendedKalmanFilter
{
  public:
    /**
     * \brief Create a Kalman filter with the specified matrices.
     * \param[in] processNoiseCovariance Process noise covariance
     */
    ExtendedKalmanFilter(const Eigen::Matrix<double, NE, NE>& processNoiseCovariance) :
        _processNoiseCovariance(processNoiseCovariance)
    {
    }

    [[nodiscard]] std::pair<Eigen::Vector<double, N>, Eigen::Matrix<double, NE, NE>> get_new_state(
            StateEstimator<N, M, NE, ME>* estimator)
    {
        const auto& stateNoiseCovariance = estimator->template state_covariance();
        const auto& measurementNoiseCovariance = estimator->template measurment_covariance();
        const auto& state = estimator->template state();

        // check parameters
        if (not utils::is_covariance_valid(stateNoiseCovariance))
        {
            throw std::invalid_argument(
                    "ExtendedKalmanFilter::get_new_state: stateNoiseCovariance is an invalid covariance matrix");
        }
        if (not utils::is_covariance_valid(measurementNoiseCovariance))
        {
            throw std::invalid_argument(
                    "ExtendedKalmanFilter::get_new_state: measurementNoiseCovariance is an invalid covariance matrix");
        }

        // Get new raw estimate
        const Eigen::Vector<double, N>& newStateEstimate = estimator->template f(state);
        const Eigen::Matrix<double, NE, NE>& estimateErrorCovariance =
                utils::propagate_covariance(stateNoiseCovariance, estimator->template f_jacobian(state)) +
                _processNoiseCovariance;

        if (not utils::is_covariance_valid(estimateErrorCovariance))
        {
            throw std::logic_error("ExtendedKalmanFilter::get_new_state: produced an invalid estimateErrorCovariance");
        }

        const auto& hJacobian = estimator->template h_jacobian(newStateEstimate);

        // compute inovation covariance
        const Eigen::Matrix<double, ME, ME>& inovation =
                estimator->template h_innovation(newStateEstimate, estimateErrorCovariance, hJacobian) +
                measurementNoiseCovariance;

        const Eigen::Matrix<double, ME, ME>& inovationInverted = pseudoInverse(inovation);

        // compute Kalman gain
        const Eigen::Matrix<double, NE, ME>& kalmanGain =
                (estimateErrorCovariance.template selfadjointView<Eigen::Lower>()) * hJacobian.transpose() *
                inovationInverted;

        // get new state
        const Eigen::Vector<double, N>& newState = estimator->template get_new_state(newStateEstimate, kalmanGain);

        // standard covariance update
        // Eigen::Matrix<double, N, N> newCovariance = (_identity - kalmanGain * hJacobian) * estimateErrorCovariance;
        // force symmetry
        // newCovariance = ((newCovariance + newCovariance.transpose()) / 2.0).eval();

        // Alternative "Joseph stabilized" version, better with numerical accuracies (and symmetry)
        Eigen::Matrix<double, NE, NE> newCovariance =
                utils::propagate_covariance(
                        estimateErrorCovariance,
                        (Eigen::Matrix<double, NE, NE>::Identity() - kalmanGain * hJacobian).eval()) +
                utils::propagate_covariance(measurementNoiseCovariance, kalmanGain);

        std::string err;
        if (not utils::is_covariance_valid(newCovariance, err))
        {
            throw std::logic_error("ExtendedKalmanFilter::get_new_state: produced an invalid covariance (" + err + ")");
        }
        // return the covariance and state estimation
        return std::make_pair(newState, newCovariance);
    }

    // Matrices for computation
    const Eigen::Matrix<double, NE, NE> _processNoiseCovariance;
};

} // namespace rgbd_slam::tracking

#endif
