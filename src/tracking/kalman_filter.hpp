#ifndef RGBDSLAM_UTILS_KALMAN_FILTER_HPP
#define RGBDSLAM_UTILS_KALMAN_FILTER_HPP

#include "covariances.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <stdexcept>

namespace rgbd_slam::tracking {

/**
 * \brief Implement a Kalman filter that can be shared by multiple systems, if they share the same dimentions.
 * N is number of state variables
 * M is number of outputs
 * P number of inputs
 */
template<int N, int M> class SharedKalmanFilter
{
  public:
    /**
     * \brief Create a Kalman filter with the specified matrices.
     * \param[in] systemDynamics System dynamics matrix
     * \param[in] outputMatrix Output matrix
     * \param[in] processNoiseCovariance Process noise covariance
     */
    SharedKalmanFilter(const Eigen::Matrix<double, N, N>& systemDynamics,
                       const Eigen::Matrix<double, M, N>& outputMatrix,
                       const Eigen::Matrix<double, N, N>& processNoiseCovariance) :
        _systemDynamics(systemDynamics),
        _outputMatrix(outputMatrix),
        _processNoiseCovariance(processNoiseCovariance),
        _identity(Eigen::Matrix<double, N, N>::Identity())
    {
    }

    /**
     * \brief Update the estimated state based on measured values. The time step is assumed to remain constant.
     * \param[in] currentState The current system state
     * \param[in] stateNoiseCovariance The current state covariance
     * \param[in] newMeasurement new measurement
     * \param[in] measurementNoiseCovariance Measurement noise covariance
     *
     * \return A pair of the new state and covariance matrix
     */
    [[nodiscard]] std::pair<Eigen::Vector<double, N>, Eigen::Matrix<double, N, N>> get_new_state(
            const Eigen::Vector<double, N>& currentState,
            const Eigen::Matrix<double, N, N>& stateNoiseCovariance,
            const Eigen::Vector<double, M>& newMeasurement,
            const Eigen::Matrix<double, M, M>& measurementNoiseCovariance)
    {
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
        const Eigen::Vector<double, N>& newStateEstimate = _systemDynamics * currentState;
        const Eigen::Matrix<double, N, N>& estimateErrorCovariance =
                utils::propagate_covariance(stateNoiseCovariance, _systemDynamics) + _processNoiseCovariance;

        // compute inovation covariance
        const Eigen::Matrix<double, M, M>& inovation =
                utils::propagate_covariance(estimateErrorCovariance, _outputMatrix) + measurementNoiseCovariance;
        // cannot inverse the inovation covariance matrix: use pseudoinverse.
        // it is slower but mathematicaly stable
        Eigen::Matrix<double, M, M> inovationInverted;
        if (utils::double_equal(inovation.determinant(), 0))
            inovationInverted = pseudoInverse(inovation);
        else
            inovationInverted = inovation.inverse();

        // compute Kalman gain
        const Eigen::Matrix<double, N, M>& kalmanGain =
                (estimateErrorCovariance.template selfadjointView<Eigen::Lower>()) * _outputMatrix.transpose() *
                inovationInverted;

        Eigen::Vector<double, N> newState =
                newStateEstimate + kalmanGain * (newMeasurement - _outputMatrix * newStateEstimate);

        // standard covariance update
        // newCovariance = (_identity - kalmanGain * _outputMatrix) * estimateErrorCovariance
        // force symetrie
        // newCovariance = ((newCovariance + newCovariance.transpose()) / 2.0).eval();

        // Alternative "Joseph stabilized" version, better with numerical accuracies (and symetrie)
        Eigen::Matrix<double, N, N> newCovariance =
                utils::propagate_covariance(estimateErrorCovariance, (_identity - kalmanGain * _outputMatrix).eval()) +
                utils::propagate_covariance(measurementNoiseCovariance, kalmanGain);

        std::string res;
        if (not utils::is_covariance_valid(newCovariance, res))
        {
            throw std::logic_error("SharedKalmanFilter::get_new_state: produced an invalid covariance: " + res);
        }
        // return the covariance and state estimation
        return std::make_pair(newState, newCovariance);
    }

    // Matrices for computation
    Eigen::Matrix<double, N, N> _systemDynamics;
    const Eigen::Matrix<double, M, N> _outputMatrix;
    const Eigen::Matrix<double, N, N> _processNoiseCovariance;

    // stateDimension-size identity
    const Eigen::Matrix<double, N, N> _identity;
};

/**
 * \brief Implement a Kalman filter, to track a single system
 */
template<int N, int M> class KalmanFilter : public SharedKalmanFilter<N, M>
{
  public:
    /**
     * \brief Create a Kalman filter with the specified matrices.
     * \param[in] systemDynamics System dynamics matrix
     * \param[in] outputMatrix Output matrix
     * \param[in] processNoiseCovariance Process noise covariance
     */
    KalmanFilter(const Eigen::Matrix<double, N, N>& systemDynamics,
                 const Eigen::Matrix<double, M, N>& outputMatrix,
                 const Eigen::Matrix<double, N, N>& processNoiseCovariance) :
        SharedKalmanFilter<N, M>(systemDynamics, outputMatrix, processNoiseCovariance),
        _isInitialized(false),
        _stateEstimate(systemDynamics.rows())
    {
    }

    /**
     * \brief Initialize the filter with a guess for initial states
     * \param[in] firstEstimateErrorCovariance Estimate error covariance for the first guess
     * \param[in] x0 The original state estimate
     */
    void init(const Eigen::Matrix<double, N, N>& firstEstimateErrorCovariance,
              const Eigen::Vector<double, N>& x0) noexcept
    {
        _estimateErrorCovariance = firstEstimateErrorCovariance;
        _stateEstimate = x0;
        _isInitialized = true;
    }

    /**
     * \brief Update the estimated state based on measured values. The time step is assumed to remain constant.
     * \param[in] newMeasurement new measurement
     * \param[in] measurementNoiseCovariance Measurement noise covariance
     */
    void update(const Eigen::Vector<double, M>& newMeasurement,
                const Eigen::Matrix<double, M, M>& measurementNoiseCovariance)
    {
        if (not is_initialized())
        {
            throw std::logic_error("KalmanFilter::update: called before initialization");
        }

        // update the covariance and state estimation
        const std::pair<Eigen::Vector<double, N>, Eigen::Matrix<double, N, N>>& res = this->get_new_state(
                _stateEstimate, _estimateErrorCovariance, newMeasurement, measurementNoiseCovariance);
        _stateEstimate = res.first;
        _estimateErrorCovariance = res.second;
    }

    /**
     * \brief Update the estimated state based on measured values, using the given time step and dynamics matrix.
     * \param[in] newMeasurement new measurement
     * \param[in] measurementNoiseCovariance Measurement noise covariance
     * \param[in] systemDynamics systemDynamics new system dynamics matrix
     */
    void update(const Eigen::Vector<double, N>& newMeasurement,
                const Eigen::Matrix<double, N, N>& measurementNoiseCovariance,
                const Eigen::Matrix<double, N, N>& systemDynamics)
    {
        this->_systemDynamics = systemDynamics;
        this->update(newMeasurement, measurementNoiseCovariance);
    }

    [[nodiscard]] Eigen::Vector<double, N> get_state() const noexcept { return _stateEstimate; };
    [[nodiscard]] Eigen::Matrix<double, N, N> get_state_covariance() const noexcept
    {
        return _estimateErrorCovariance;
    };

    [[nodiscard]] bool is_initialized() const noexcept { return _isInitialized; }

  private:
    // Is the filter isInitialized?
    bool _isInitialized;

    // Estimated state
    Eigen::Vector<double, N> _stateEstimate;
    Eigen::Matrix<double, N, N> _estimateErrorCovariance;
};

} // namespace rgbd_slam::tracking

#endif