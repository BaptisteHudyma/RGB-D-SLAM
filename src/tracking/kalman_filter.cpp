#include "kalman_filter.hpp"
#include "covariances.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <tuple>

namespace rgbd_slam::tracking {

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
    // check parameters
    assert(utils::is_covariance_valid(stateNoiseCovariance));
    assert(utils::is_covariance_valid(measurementNoiseCovariance));

    // Get new raw estimate
    const vectorxd& newStateEstimate = _systemDynamics * currentState;
    const matrixd& estimateErrorCovariance =
            _systemDynamics * stateNoiseCovariance * _systemDynamics.transpose() + _processNoiseCovariance;

    // compute inovation covariance
    const matrixd inovationCovariance =
            _outputMatrix * estimateErrorCovariance * _outputMatrix.transpose() + measurementNoiseCovariance;

    // cannot inverse the inovation covariance matrix, no gain to compute (gain too small)
    if (utils::double_equal(inovationCovariance.determinant(), 0))
    {
        // do not update state or covariance
        assert(utils::is_covariance_valid(estimateErrorCovariance));
        return std::make_pair(newStateEstimate, estimateErrorCovariance);
    }

    // compute Kalman gain
    const matrixd& kalmanGain = estimateErrorCovariance * _outputMatrix.transpose() * inovationCovariance.inverse();

    const vectorxd& newState = newStateEstimate + kalmanGain * (newMeasurement - _outputMatrix * newStateEstimate);

    const matrixd& covUpdateMatrix = (_identity - kalmanGain * _outputMatrix);
    const matrixd& newCovariance = (_identity - kalmanGain * _outputMatrix) * estimateErrorCovariance;

    /*  // Alternative non Joseph form
    const matrixd& temp = _identity - kalmanGain * _outputMatrix;
    const matrixd& newCovariance = temp * estimateErrorCovariance * temp.transpose() +
                                   kalmanGain * measurementNoiseCovariance * kalmanGain.transpose();
    */

    assert(newCovariance.cols() == newCovariance.rows());
    assert(utils::is_covariance_valid(newCovariance));

    // return the covariance and state estimation
    return std::make_pair(newState, newCovariance);
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

    // update the covariance and state estimation
    const std::pair<vectorxd, matrixd>& res =
            get_new_state(_stateEstimate, _estimateErrorCovariance, newMeasurement, measurementNoiseCovariance);
    _stateEstimate = res.first;
    _estimateErrorCovariance = res.second;
}

void KalmanFilter::update(const vectorxd& newMeasurement,
                          const matrixd& measurementNoiseCovariance,
                          const matrixd& systemDynamics)
{
    _systemDynamics = systemDynamics;
    update(newMeasurement, measurementNoiseCovariance);
}

} // namespace rgbd_slam::tracking