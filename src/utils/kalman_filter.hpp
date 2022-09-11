#ifndef RGBDSLAM_UTILS_KALMAN_FILTER_HPP
#define RGBDSLAM_UTILS_KALMAN_FILTER_HPP


#include <Eigen/Dense>

namespace rgbd_slam {
    namespace utils {

        class KalmanFilter {

        public:

            /**
            * \brief Create a Kalman filter with the specified matrices.
            * \param[in] systemDynamics System dynamics matrix
            * \param[in] outputMatrix Output matrix
            * \param[in] processNoiseCovariance Process noise covariance
            * \param[in] measurementNoiseCovariance Measurement noise covariance
            */
            KalmanFilter(
                    const Eigen::MatrixXd& systemDynamics,
                    const Eigen::MatrixXd& outputMatrix,
                    const Eigen::MatrixXd& processNoiseCovariance,
                    const Eigen::MatrixXd& measurementNoiseCovariance
            );

            /**
            * \brief Initialize the filter with a guess for initial states
            * \param[in] firstEstimateErrorCovariance Estimate error covariance for the first guess
            * \param[in] x0 The original state estimate
            */
            void init(const Eigen::MatrixXd& firstEstimateErrorCovariance, const Eigen::VectorXd& x0 );

            /**
            * \brief Update the estimated state based on measured values. The time step is assumed to remain constant.
            * \param[in] newMeasurement new measurement
            */
            void update(const Eigen::VectorXd& newMeasurement);

            /**
            * \brief Update the estimated state based on measured values, using the given time step and dynamics matrix.
            * \param[in] newMeasurement new measurement
            * \param[in] systemDynamics systemDynamics new system dynamics matrix 
            */
            void update(const Eigen::VectorXd& y, const Eigen::MatrixXd& systemDynamics);


            Eigen::VectorXd get_state() const {
                return stateEstimate;
            };

            bool is_initialized() const {
                return isInitialized;
            }

        private:

            // Matrices for computation
            Eigen::MatrixXd systemDynamics;
            const Eigen::MatrixXd outputMatrix;
            const Eigen::MatrixXd processNoiseCovariance;
            const Eigen::MatrixXd measurementNoiseCovariance;

            // System dimensions
            const size_t measurementDimension;
            const size_t stateDimension;

            // Is the filter isInitialized?
            bool isInitialized;

            // stateDimension-size identity
            Eigen::MatrixXd I;

            // Estimated state
            Eigen::VectorXd stateEstimate;
            Eigen::MatrixXd estimateErrorCovariance;
        };
    }
}

#endif