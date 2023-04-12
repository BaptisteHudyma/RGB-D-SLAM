#ifndef RGBDSLAM_UTILS_KALMAN_FILTER_HPP
#define RGBDSLAM_UTILS_KALMAN_FILTER_HPP

#include "../../types.hpp"
#include "covariances.hpp"
#include "distance_utils.hpp"
#include <Eigen/src/Core/Matrix.h>

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
    std::pair<Eigen::Vector<double, N>, Eigen::Matrix<double, N, N>> get_new_state(
            const Eigen::Vector<double, N>& currentState,
            const Eigen::Matrix<double, N, N>& stateNoiseCovariance,
            const Eigen::Vector<double, M>& newMeasurement,
            const Eigen::Matrix<double, M, M>& measurementNoiseCovariance)
    {
        // check parameters
        assert(utils::is_covariance_valid(stateNoiseCovariance));
        assert(utils::is_covariance_valid(measurementNoiseCovariance));

        // Get new raw estimate
        const Eigen::Vector<double, N>& newStateEstimate = _systemDynamics * currentState;
        const Eigen::Matrix<double, N, N>& estimateErrorCovariance =
                _systemDynamics * stateNoiseCovariance * _systemDynamics.transpose() + _processNoiseCovariance;

        // compute inovation covariance
        const Eigen::Matrix<double, M, M>& inovationCovariance =
                _outputMatrix * estimateErrorCovariance * _outputMatrix.transpose() + measurementNoiseCovariance;

        // cannot inverse the inovation covariance matrix, no gain to compute (gain too small)
        if (utils::double_equal(inovationCovariance.determinant(), 0))
        {
            // do not update state or covariance
            assert(utils::is_covariance_valid(estimateErrorCovariance));
            return std::make_pair(newStateEstimate, estimateErrorCovariance);
        }

        // compute Kalman gain
        const Eigen::Matrix<double, N, M>& kalmanGain =
                (estimateErrorCovariance.template selfadjointView<Eigen::Lower>()) * _outputMatrix.transpose() *
                inovationCovariance.inverse();

        const Eigen::Vector<double, N>& newState =
                newStateEstimate + kalmanGain * (newMeasurement - _outputMatrix * newStateEstimate);

        Eigen::Matrix<double, N, N> newCovariance = (_identity - kalmanGain * _outputMatrix) *
                                                    (estimateErrorCovariance.template selfadjointView<Eigen::Lower>());

        /*  // Alternative non Joseph form
        const Eigen::Matrix<double, N, N>& temp = _identity - kalmanGain * _outputMatrix;
        const Eigen::Matrix<double, N, N>& newCovariance = temp * estimateErrorCovariance * temp.transpose() +
                                       kalmanGain * measurementNoiseCovariance * kalmanGain.transpose();
        */

        assert(utils::is_covariance_valid(newCovariance));

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
    void init(const Eigen::Matrix<double, N, N>& firstEstimateErrorCovariance, const Eigen::Vector<double, N>& x0)
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
        assert(is_initialized());

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

    Eigen::Vector<double, N> get_state() const { return _stateEstimate; };
    Eigen::Matrix<double, N, N> get_state_covariance() const { return _estimateErrorCovariance; };

    bool is_initialized() const { return _isInitialized; }

  private:
    // Is the filter isInitialized?
    bool _isInitialized;

    // Estimated state
    Eigen::Vector<double, N> _stateEstimate;
    Eigen::Matrix<double, N, N> _estimateErrorCovariance;
};

} // namespace rgbd_slam::tracking

#endif