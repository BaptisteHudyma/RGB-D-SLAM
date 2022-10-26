#ifndef RGBDSLAM_UTILS_KALMAN_FILTER_HPP
#define RGBDSLAM_UTILS_KALMAN_FILTER_HPP


#include <Eigen/Dense>

namespace rgbd_slam {
    namespace tracking {

        /**
         * \brief Implement a Kalman filter that can be shared by multiple systems, if they share the same dimentions
         */
        class SharedKalmanFilter {
        public:

            /**
            * \brief Create a Kalman filter with the specified matrices.
            * \param[in] systemDynamics System dynamics matrix
            * \param[in] outputMatrix Output matrix
            * \param[in] processNoiseCovariance Process noise covariance
            */
            SharedKalmanFilter(
                const Eigen::MatrixXd& systemDynamics,
                const Eigen::MatrixXd& outputMatrix,
                const Eigen::MatrixXd& processNoiseCovariance
            );

            /**
            * \brief Update the estimated state based on measured values. The time step is assumed to remain constant.
            * \param[in] currentState The current system state
            * \param[in] stateNoiseCovariance The current state covariance
            * \param[in] newMeasurement new measurement
            * \param[in] measurementNoiseCovariance Measurement noise covariance
            *
            * \return A pair of the new state and covariance matrix
            */
            std::pair<Eigen::VectorXd, Eigen::MatrixXd> get_new_state(const Eigen::VectorXd& currentState, const Eigen::MatrixXd& stateNoiseCovariance, const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance);

        protected:

            // Matrices for computation
            Eigen::MatrixXd _systemDynamics;
            const Eigen::MatrixXd _outputMatrix;
            const Eigen::MatrixXd _processNoiseCovariance;

            // stateDimension-size identity
            Eigen::MatrixXd _identity;
        };


        /**
         * \brief Implement a Kalman filter, to track a single system
         */
        class KalmanFilter : public SharedKalmanFilter {
        public:
            /**
            * \brief Create a Kalman filter with the specified matrices.
            * \param[in] systemDynamics System dynamics matrix
            * \param[in] outputMatrix Output matrix
            * \param[in] processNoiseCovariance Process noise covariance
            */
            KalmanFilter(
                const Eigen::MatrixXd& systemDynamics,
                const Eigen::MatrixXd& outputMatrix,
                const Eigen::MatrixXd& processNoiseCovariance
            );

            /**
            * \brief Initialize the filter with a guess for initial states
            * \param[in] firstEstimateErrorCovariance Estimate error covariance for the first guess
            * \param[in] x0 The original state estimate
            */
            void init(const Eigen::MatrixXd& firstEstimateErrorCovariance, const Eigen::VectorXd& x0);

            /**
            * \brief Update the estimated state based on measured values. The time step is assumed to remain constant.
            * \param[in] newMeasurement new measurement
            * \param[in] measurementNoiseCovariance Measurement noise covariance
            */
            void update(const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance);

            /**
            * \brief Update the estimated state based on measured values, using the given time step and dynamics matrix.
            * \param[in] newMeasurement new measurement
            * \param[in] measurementNoiseCovariance Measurement noise covariance
            * \param[in] systemDynamics systemDynamics new system dynamics matrix 
            */
            void update(const Eigen::VectorXd& newMeasurement, const Eigen::MatrixXd& measurementNoiseCovariance, const Eigen::MatrixXd& systemDynamics);


            Eigen::VectorXd get_state() const {
                return _stateEstimate;
            };
            Eigen::MatrixXd get_state_covariance() const {
                return _estimateErrorCovariance;
            };

            bool is_initialized() const {
                return _isInitialized;
            }

        private:
            // Is the filter isInitialized?
            bool _isInitialized;

            // Estimated state
            Eigen::VectorXd _stateEstimate;
            Eigen::MatrixXd _estimateErrorCovariance;
        };
    }
}

#endif