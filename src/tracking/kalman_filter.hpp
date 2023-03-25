#ifndef RGBDSLAM_UTILS_KALMAN_FILTER_HPP
#define RGBDSLAM_UTILS_KALMAN_FILTER_HPP

#include "../../types.hpp"

namespace rgbd_slam {
namespace tracking {

/**
 * \brief Implement a Kalman filter that can be shared by multiple systems, if they share the same dimentions
 */
class SharedKalmanFilter
{
  public:
    /**
     * \brief Create a Kalman filter with the specified matrices.
     * \param[in] systemDynamics System dynamics matrix
     * \param[in] outputMatrix Output matrix
     * \param[in] processNoiseCovariance Process noise covariance
     */
    SharedKalmanFilter(const matrixd& systemDynamics,
                       const matrixd& outputMatrix,
                       const matrixd& processNoiseCovariance);

    /**
     * \brief Update the estimated state based on measured values. The time step is assumed to remain constant.
     * \param[in] currentState The current system state
     * \param[in] stateNoiseCovariance The current state covariance
     * \param[in] newMeasurement new measurement
     * \param[in] measurementNoiseCovariance Measurement noise covariance
     *
     * \return A pair of the new state and covariance matrix
     */
    std::pair<vectorxd, matrixd> get_new_state(const vectorxd& currentState,
                                               const matrixd& stateNoiseCovariance,
                                               const vectorxd& newMeasurement,
                                               const matrixd& measurementNoiseCovariance);

  protected:
    // Matrices for computation
    matrixd _systemDynamics;
    const matrixd _outputMatrix;
    const matrixd _processNoiseCovariance;

    // stateDimension-size identity
    matrixd _identity;
};

/**
 * \brief Implement a Kalman filter, to track a single system
 */
class KalmanFilter : public SharedKalmanFilter
{
  public:
    /**
     * \brief Create a Kalman filter with the specified matrices.
     * \param[in] systemDynamics System dynamics matrix
     * \param[in] outputMatrix Output matrix
     * \param[in] processNoiseCovariance Process noise covariance
     */
    KalmanFilter(const matrixd& systemDynamics, const matrixd& outputMatrix, const matrixd& processNoiseCovariance);

    /**
     * \brief Initialize the filter with a guess for initial states
     * \param[in] firstEstimateErrorCovariance Estimate error covariance for the first guess
     * \param[in] x0 The original state estimate
     */
    void init(const matrixd& firstEstimateErrorCovariance, const vectorxd& x0);

    /**
     * \brief Update the estimated state based on measured values. The time step is assumed to remain constant.
     * \param[in] newMeasurement new measurement
     * \param[in] measurementNoiseCovariance Measurement noise covariance
     */
    void update(const vectorxd& newMeasurement, const matrixd& measurementNoiseCovariance);

    /**
     * \brief Update the estimated state based on measured values, using the given time step and dynamics matrix.
     * \param[in] newMeasurement new measurement
     * \param[in] measurementNoiseCovariance Measurement noise covariance
     * \param[in] systemDynamics systemDynamics new system dynamics matrix
     */
    void update(const vectorxd& newMeasurement,
                const matrixd& measurementNoiseCovariance,
                const matrixd& systemDynamics);

    vectorxd get_state() const { return _stateEstimate; };
    matrixd get_state_covariance() const { return _estimateErrorCovariance; };

    bool is_initialized() const { return _isInitialized; }

  private:
    // Is the filter isInitialized?
    bool _isInitialized;

    // Estimated state
    vectorxd _stateEstimate;
    matrixd _estimateErrorCovariance;
};
} // namespace tracking
} // namespace rgbd_slam

#endif