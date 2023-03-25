#include "kalman_filter.hpp"

namespace rgbd_slam {
namespace tracking {

SharedKalmanFilter::SharedKalmanFilter(const matrixd& systemDynamics,
                                       const matrixd& outputMatrix,
                                       const matrixd& processNoiseCovariance) :
    _systemDynamics(systemDynamics),
    _outputMatrix(outputMatrix),
    _processNoiseCovariance(processNoiseCovariance),
    _identity(systemDynamics.rows(), systemDynamics.rows())
{
    _identity.setIdentity();
}

std::pair<vectorxd, matrixd> SharedKalmanFilter::get_new_state(const vectorxd& currentState,
                                                               const matrixd& stateNoiseCovariance,
                                                               const vectorxd& newMeasurement,
                                                               const matrixd& measurementNoiseCovariance)
{
    // Get new raw estimate
    const vectorxd& newStateEstimate = _systemDynamics * currentState;
    const matrixd& estimateErrorCovariance =
            _systemDynamics * stateNoiseCovariance * _systemDynamics.transpose() + _processNoiseCovariance;

    // compute Kalman gain
    const matrixd& kalmanGain =
            estimateErrorCovariance * _outputMatrix.transpose() *
            (_outputMatrix * estimateErrorCovariance * _outputMatrix.transpose() + measurementNoiseCovariance)
                    .inverse();

    // return the covariance and state estimation
    return std::make_pair(newStateEstimate + kalmanGain * (newMeasurement - _outputMatrix * newStateEstimate),
                          (_identity - kalmanGain * _outputMatrix) * estimateErrorCovariance);
}

KalmanFilter::KalmanFilter(const matrixd& systemDynamics,
                           const matrixd& outputMatrix,
                           const matrixd& processNoiseCovariance) :
    SharedKalmanFilter(systemDynamics, outputMatrix, processNoiseCovariance),
    _isInitialized(false),
    _stateEstimate(systemDynamics.rows())
{
}

void KalmanFilter::init(const matrixd& firstEstimateErrorCovariance, const vectorxd& x0)
{
    _estimateErrorCovariance = firstEstimateErrorCovariance;
    _stateEstimate = x0;
    _isInitialized = true;
}

void KalmanFilter::update(const vectorxd& newMeasurement, const matrixd& measurementNoiseCovariance)
{
    assert(is_initialized());

    const std::pair<vectorxd, matrixd>& res =
            get_new_state(_stateEstimate, _estimateErrorCovariance, newMeasurement, measurementNoiseCovariance);

    // update the covariance and state estimation
    _estimateErrorCovariance = res.second;
    _stateEstimate = res.first;
}

void KalmanFilter::update(const vectorxd& newMeasurement,
                          const matrixd& measurementNoiseCovariance,
                          const matrixd& systemDynamics)
{
    _systemDynamics = systemDynamics;
    update(newMeasurement, measurementNoiseCovariance);
}

} // namespace tracking
} // namespace rgbd_slam