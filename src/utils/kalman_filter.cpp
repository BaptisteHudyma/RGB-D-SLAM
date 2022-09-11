#include <iostream>
#include <stdexcept>

#include "kalman_filter.hpp"

namespace rgbd_slam {
    namespace utils {

        KalmanFilter::KalmanFilter(const Eigen::MatrixXd& systemDynamics,
                                    const Eigen::MatrixXd& outputMatrix,
                                    const Eigen::MatrixXd& processNoiseCovariance):
                systemDynamics(systemDynamics), outputMatrix(outputMatrix), 
                processNoiseCovariance(processNoiseCovariance),
                measurementDimension(outputMatrix.rows()), stateDimension(systemDynamics.rows()),
                isInitialized(false),
                I(stateDimension, stateDimension), stateEstimate(stateDimension)
        {
            I.setIdentity();
        }

        void KalmanFilter::init(const Eigen::MatrixXd& firstEstimateErrorCovariance, const Eigen::VectorXd& x0) {
            estimateErrorCovariance = firstEstimateErrorCovariance;
            stateEstimate = x0;
            isInitialized = true;
        }

        void KalmanFilter::update(const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance) {

            if(not is_initialized())
                throw std::runtime_error("Filter is not isInitialized!");

            // Get new raw estimate
            Eigen::VectorXd newStateEstimate = systemDynamics * stateEstimate;
            estimateErrorCovariance = systemDynamics * estimateErrorCovariance * systemDynamics.transpose() + processNoiseCovariance;
            
            // compute Kalman gain
            const Eigen::MatrixXd kalmanGain = estimateErrorCovariance * outputMatrix.transpose() * (outputMatrix * estimateErrorCovariance * outputMatrix.transpose() + measurementNoiseCovariance).inverse();
            
            // update the estimate with covariance
            newStateEstimate += kalmanGain * (newMeasurement - outputMatrix * newStateEstimate);

            // update the covariance and state estimation
            estimateErrorCovariance = (I - kalmanGain * outputMatrix) * estimateErrorCovariance;
            stateEstimate = newStateEstimate;
        }

        void KalmanFilter::update(const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance, const Eigen::MatrixXd& systemDynamics) {

            this->systemDynamics = systemDynamics;
            update(newMeasurement, measurementNoiseCovariance);
        }
 
    }
}