#include <iostream>
#include <stdexcept>

#include "kalman_filter.hpp"

namespace rgbd_slam {
    namespace tracking {

        SharedKalmanFilter::SharedKalmanFilter(const Eigen::MatrixXd& systemDynamics,
                                    const Eigen::MatrixXd& outputMatrix,
                                    const Eigen::MatrixXd& processNoiseCovariance):
                systemDynamics(systemDynamics), outputMatrix(outputMatrix), 
                processNoiseCovariance(processNoiseCovariance),
                I(systemDynamics.rows(), systemDynamics.rows())
        {
            I.setIdentity();
        }

        std::pair<Eigen::VectorXd, Eigen::MatrixXd> SharedKalmanFilter::get_new_state(const Eigen::VectorXd& currentState, const Eigen::MatrixXd& currentMeasurementNoiseCovariance, const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance) 
        {
            // Get new raw estimate
            const Eigen::VectorXd& newStateEstimate = systemDynamics * currentState;
            const Eigen::MatrixXd& estimateErrorCovariance = systemDynamics * currentMeasurementNoiseCovariance * systemDynamics.transpose() + processNoiseCovariance;
            
            // compute Kalman gain
            const Eigen::MatrixXd kalmanGain = estimateErrorCovariance * outputMatrix.transpose() * (outputMatrix * estimateErrorCovariance * outputMatrix.transpose() + measurementNoiseCovariance).inverse();

            // return the covariance and state estimation
            return std::make_pair(
                newStateEstimate + kalmanGain * (newMeasurement - outputMatrix * newStateEstimate), 
                (I - kalmanGain * outputMatrix) * estimateErrorCovariance
            );
        }


        KalmanFilter::KalmanFilter(const Eigen::MatrixXd& systemDynamics,
                                    const Eigen::MatrixXd& outputMatrix,
                                    const Eigen::MatrixXd& processNoiseCovariance):
                SharedKalmanFilter(systemDynamics, outputMatrix,processNoiseCovariance),
                isInitialized(false), stateEstimate(systemDynamics.rows())
        {}

        void KalmanFilter::init(const Eigen::MatrixXd& firstEstimateErrorCovariance, const Eigen::VectorXd& x0)
        {
            estimateErrorCovariance = firstEstimateErrorCovariance;
            stateEstimate = x0;
            isInitialized = true;
        }

        void KalmanFilter::update(const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance) 
        {
            assert(is_initialized());

            const std::pair<Eigen::VectorXd, Eigen::MatrixXd>& res = get_new_state(stateEstimate, estimateErrorCovariance, newMeasurement,measurementNoiseCovariance);

            // update the covariance and state estimation
            estimateErrorCovariance = res.second;
            stateEstimate = res.first;
        }

        void KalmanFilter::update(const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance, const Eigen::MatrixXd& systemDynamics)
        {

            this->systemDynamics = systemDynamics;
            update(newMeasurement, measurementNoiseCovariance);
        }
 
    }
}