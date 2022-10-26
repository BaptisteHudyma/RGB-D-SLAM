#include "kalman_filter.hpp"

namespace rgbd_slam {
    namespace tracking {

        SharedKalmanFilter::SharedKalmanFilter(const Eigen::MatrixXd& systemDynamics,
                                    const Eigen::MatrixXd& outputMatrix,
                                    const Eigen::MatrixXd& processNoiseCovariance):
                _systemDynamics(systemDynamics), _outputMatrix(outputMatrix), 
                _processNoiseCovariance(processNoiseCovariance),
                _identity(systemDynamics.rows(), systemDynamics.rows())
        {
            _identity.setIdentity();
        }

        std::pair<Eigen::VectorXd, Eigen::MatrixXd> SharedKalmanFilter::get_new_state(const Eigen::VectorXd& currentState, const Eigen::MatrixXd& stateNoiseCovariance, const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance) 
        {
            // Get new raw estimate
            const Eigen::VectorXd& newStateEstimate = _systemDynamics * currentState;
            const Eigen::MatrixXd& estimateErrorCovariance = _systemDynamics * stateNoiseCovariance * _systemDynamics.transpose() + _processNoiseCovariance;
            
            // compute Kalman gain
            const Eigen::MatrixXd& kalmanGain = estimateErrorCovariance * _outputMatrix.transpose() * (_outputMatrix * estimateErrorCovariance * _outputMatrix.transpose() + measurementNoiseCovariance).inverse();

            // return the covariance and state estimation
            return std::make_pair(
                newStateEstimate + kalmanGain * (newMeasurement - _outputMatrix * newStateEstimate), 
                (_identity - kalmanGain * _outputMatrix) * estimateErrorCovariance
            );
        }


        KalmanFilter::KalmanFilter(const Eigen::MatrixXd& systemDynamics,
                                    const Eigen::MatrixXd& outputMatrix,
                                    const Eigen::MatrixXd& processNoiseCovariance):
                SharedKalmanFilter(systemDynamics, outputMatrix,processNoiseCovariance),
                _isInitialized(false), _stateEstimate(systemDynamics.rows())
        {}

        void KalmanFilter::init(const Eigen::MatrixXd& firstEstimateErrorCovariance, const Eigen::VectorXd& x0)
        {
            _estimateErrorCovariance = firstEstimateErrorCovariance;
            _stateEstimate = x0;
            _isInitialized = true;
        }

        void KalmanFilter::update(const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance) 
        {
            assert(is_initialized());

            const std::pair<Eigen::VectorXd, Eigen::MatrixXd>& res = get_new_state(_stateEstimate, _estimateErrorCovariance, newMeasurement,measurementNoiseCovariance);

            // update the covariance and state estimation
            _estimateErrorCovariance = res.second;
            _stateEstimate = res.first;
        }

        void KalmanFilter::update(const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance, const Eigen::MatrixXd& systemDynamics)
        {

            _systemDynamics = systemDynamics;
            update(newMeasurement, measurementNoiseCovariance);
        }
 
    }
}