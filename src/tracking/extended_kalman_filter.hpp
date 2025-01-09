#ifndef RGBDSLAM_UTILS_EXTENDED_KALMAN_FILTER_HPP
#define RGBDSLAM_UTILS_EXTENDED_KALMAN_FILTER_HPP

#include "covariances.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <stdexcept>

namespace rgbd_slam::tracking {

template<int N, int M> class StateEstimator
{
  public:
    Eigen::Vector<double, N> state() const noexcept { return _feature; }
    Eigen::Matrix<double, N, N> state_covariance() const noexcept { return _featureCovariance; }

    Eigen::Vector<double, M> measurment() const noexcept { return _measurment; }
    Eigen::Matrix<double, M, M> measurment_covariance() const noexcept { return _measurmentCovariance; }

    /**
     * Compute the new state estimate from the current state (default is no dynamics)
     */
    virtual Eigen::Vector<double, N> f(const Eigen::Vector<double, N>& state) const noexcept
    {
        // no dynamics, just return the same state
        return state;
    }

    Eigen::Matrix<double, N, N> f_jacobian(const Eigen::Vector<double, N>& state) const noexcept
    {
        std::ignore = state;
        // no dynamic jacobian
        return Eigen::Matrix<double, N, N>::Identity();
    }

    /**
     * Compute the measurment equation from the given state
     */
    virtual Eigen::Vector<double, M> h(const Eigen::Vector<double, N>& state) const noexcept = 0;

    virtual Eigen::Matrix<double, M, N> h_jacobian(const Eigen::Vector<double, N>& state) const noexcept = 0;

    StateEstimator(const Eigen::Vector<double, N>& feature,
                   const Eigen::Matrix<double, N, N>& featureCovariance,
                   const Eigen::Vector<double, M>& measurment,
                   const Eigen::Matrix<double, M, M>& measurmentCovariance) :
        _feature(feature),
        _featureCovariance(featureCovariance),
        _measurment(measurment),
        _measurmentCovariance(measurmentCovariance)
    {
    }

  private:
    const Eigen::Vector<double, N> _feature;
    const Eigen::Matrix<double, N, N> _featureCovariance;
    const Eigen::Vector<double, M> _measurment;
    const Eigen::Matrix<double, M, M> _measurmentCovariance;
};

/**
 * \brief Implement a Kalman filter that can be shared by multiple systems, if they share the same dimentions.
 * N is number of state variables
 * M is number of outputs
 * P number of inputs
 */
template<int N, int M> class ExtendedKalmanFilter
{
  public:
    /**
     * \brief Create a Kalman filter with the specified matrices.
     * \param[in] processNoiseCovariance Process noise covariance
     */
    ExtendedKalmanFilter(const Eigen::Matrix<double, N, N>& processNoiseCovariance) :
        _processNoiseCovariance(processNoiseCovariance),
        _identity(Eigen::Matrix<double, N, N>::Identity())
    {
    }

    [[nodiscard]] std::pair<Eigen::Vector<double, N>, Eigen::Matrix<double, N, N>> get_new_state(
            StateEstimator<N, M>* estimator)
    {
        const auto& stateNoiseCovariance = estimator->template state_covariance();
        const auto& measurementNoiseCovariance = estimator->template measurment_covariance();
        const auto& state = estimator->template state();

        // check parameters
        if (not utils::is_covariance_valid(stateNoiseCovariance))
        {
            throw std::invalid_argument(
                    "SharedKalmanFilter::get_new_state: stateNoiseCovariance is an invalid covariance matrix");
        }
        if (not utils::is_covariance_valid(measurementNoiseCovariance))
        {
            throw std::invalid_argument(
                    "SharedKalmanFilter::get_new_state: measurementNoiseCovariance is an invalid covariance matrix");
        }

        // Get new raw estimate
        const Eigen::Vector<double, N>& newStateEstimate = estimator->template f(state);
        const Eigen::Matrix<double, N, N>& estimateErrorCovariance =
                utils::propagate_covariance(stateNoiseCovariance, estimator->template f_jacobian(state), 0.0) +
                _processNoiseCovariance;

        if (not utils::is_covariance_valid(estimateErrorCovariance))
        {
            throw std::logic_error("SharedKalmanFilter::get_new_state: produced an invalid estimateErrorCovariance");
        }

        const auto& hJacobian = estimator->template h_jacobian(newStateEstimate);

        // compute inovation covariance
        const Eigen::Matrix<double, M, M>& inovationCovariance =
                utils::propagate_covariance(estimateErrorCovariance, hJacobian, 0.0) + measurementNoiseCovariance;

        // cannot inverse the inovation covariance matrix, no gain to compute (gain too small)
        if (utils::double_equal(inovationCovariance.determinant(), 0))
        {
            // do not update state or covariance
            if (not utils::is_covariance_valid(estimateErrorCovariance))
            {
                throw std::logic_error(
                        "SharedKalmanFilter::get_new_state: produced an invalid covariance estimateErrorCovariance");
            }
            return std::make_pair(newStateEstimate, estimateErrorCovariance);
        }

        // compute Kalman gain
        const Eigen::Matrix<double, N, M>& kalmanGain =
                (estimateErrorCovariance.template selfadjointView<Eigen::Lower>()) * hJacobian.transpose() *
                inovationCovariance.inverse();

        const Eigen::Vector<double, N>& newState =
                newStateEstimate +
                kalmanGain * (estimator->template measurment() - estimator->template h(newStateEstimate));

        Eigen::Matrix<double, N, N> newCovariance = (_identity - kalmanGain * hJacobian) *
                                                    (estimateErrorCovariance.template selfadjointView<Eigen::Lower>());
        // force symetrie for covariance
        newCovariance = newCovariance.template selfadjointView<Eigen::Lower>();

        if (not utils::is_covariance_valid(newCovariance))
        {
            throw std::logic_error("SharedKalmanFilter::get_new_state: produced an invalid covariance");
        }
        // return the covariance and state estimation
        return std::make_pair(newState, newCovariance);
    }

    // Matrices for computation
    const Eigen::Matrix<double, N, N> _processNoiseCovariance;

    // stateDimension-size identity
    const Eigen::Matrix<double, N, N> _identity;
};

} // namespace rgbd_slam::tracking

#endif