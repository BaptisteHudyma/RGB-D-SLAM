#include <gtest/gtest.h>

#include "utils/kalman_filter.hpp"
#include <Eigen/src/Core/Matrix.h>

#include <list>
#include <random>

namespace rgbd_slam::utils {

    TEST(KalmanFilteringTests, BuildingHeightGuess)
    {
        // try to find the height of a building with noisy measurements 

        const std::list<double> measurments ({
            48.54, 47.11, 55.01, 55.15, 49.89,
            40.85, 46.72, 50.05, 51.27, 49.95});

        // estimated height of the building
        const double originalState = 60.0;      // meters
        const double originalStateUncertainty = 15;  // meters

        const double measurementError = 5;

        const int stateDimension = 1; // Number of states
        const int measurementDimension = 1; // Number of measurements

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension);
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension);
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension);
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension);

        // no dynamics
        systemDynamics << 1;
        // one data, one output
        outputMatrix << 1;
        // measurement noise covariance
        measurementNoiseCovariance << measurementError * measurementError;
        processNoiseCovariance << 0;

        // Only speed and position are related
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance
        estimateErrorCovariance << originalStateUncertainty * originalStateUncertainty;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);
        
        Eigen::VectorXd x0(stateDimension);
        x0 << originalState;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : measurments) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(50, kf.get_state().x(), 0.5);
    }

    TEST(KalmanFilteringTests, TemperatureInTank)
    {
        // try to find the temperature of a water tank
        const std::list<double> measurments = {
            49.95, 49.967, 50.1, 50.106, 49.992,
            49.819, 49.933, 50.007, 50.023, 49.99
        };

        // estimated original parameters (degrees celsius)
        const double originalState = 10.0;
        const double originalStateUncertainty = 100; //high

        const double measurementError = 0.1;    //celsius

        const int stateDimension = 1; // Number of states
        const int measurementDimension = 1; // Number of measurements

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension);
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension);
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension);
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension);

        // no dynamics
        systemDynamics << 1;
        // one data, one output
        outputMatrix << 1;
        // measurement noise covariance
        measurementNoiseCovariance << measurementError * measurementError;
        processNoiseCovariance << 0;

        // Only speed and position are related
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance
        estimateErrorCovariance << originalStateUncertainty * originalStateUncertainty;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);
        
        Eigen::VectorXd x0(stateDimension);
        x0 << originalState;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : measurments) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(50, kf.get_state().x(), 0.05);
    }

    TEST(KalmanFilteringTests, TemperatureInHeatingTank)
    {
        // try to find the temperature of a water tank
        const std::list<double> measurments = {
            50.45, 50.967, 51.6, 52.106, 52.492,
            52.819, 53.433, 54.007, 54.523, 54.99 
        };

        // estimated original parameters (degrees celsius)
        const double originalState = 10.0;
        const double originalStateUncertainty = 100; //high

        const double measurementError = 0.1;    //celsius

        const int stateDimension = 1; // Number of states
        const int measurementDimension = 1; // Number of measurements

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension);
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension);
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension);
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension);

        // no dynamics
        systemDynamics << 1;
        // one data, one output
        outputMatrix << 1;
        // measurement noise covariance
        measurementNoiseCovariance << measurementError * measurementError;
        processNoiseCovariance << 0.15;

        // Only speed and position are related
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance
        estimateErrorCovariance << originalStateUncertainty * originalStateUncertainty;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);
        
        Eigen::VectorXd x0(stateDimension);
        x0 << originalState;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : measurments) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(measurments.back(), kf.get_state().x(), 0.05);
    }

    TEST(KalmanFilteringTests, VehiculeLocationEstimation)
    {
        const int stateDimension = 6; // Number of states (position and speed)
        const int measurementDimension = 2; // Number of measurements (observed position)

        const double dt = 1;                //seconds
        const double accelerationStd = 0.2; //m/s²
        const double measurementError = 3;  // meters

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension);
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension);
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension);
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension);

        // no dynamics
        systemDynamics << 
            1, dt, 0.5*dt*dt, 0, 0,  0,
            0, 1,  dt,        0, 0,  0,
            0, 0,  1,         0, 0,  0,
            0, 0,  0,         1, dt, 0.5*dt*dt,
            0, 0,  0,         0, 1,  dt,
            0, 0,  0,         0, 0,  1;

        // We get only positions
        outputMatrix << 
            1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0;
        // measurement noise covariance
        measurementNoiseCovariance << 
            measurementError * measurementError, 0,
            0,                                   measurementError * measurementError;
        processNoiseCovariance << 
            pow(dt, 4.0) / 4.0, pow(dt, 3.0) / 2.0, pow(dt, 2.0) / 2.0, 0, 0, 0,
            pow(dt, 3.0) / 2.0, dt*dt, dt, 0, 0, 0,
            pow(dt, 2.0) / 2.0, dt, 1, 0, 0, 0,
            0, 0, 0, pow(dt, 4.0) / 4.0, pow(dt, 3.0) / 2.0, pow(dt, 2.0) / 2.0,
            0, 0, 0, pow(dt, 3.0) / 2.0, dt*dt, dt,
            0, 0, 0, pow(dt, 2.0) / 2.0, dt, 1;
        processNoiseCovariance *= accelerationStd * accelerationStd;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);
        
        Eigen::VectorXd x0(stateDimension);
        Eigen::MatrixXd x0ErrorCovariance(stateDimension, stateDimension); // Estimate error covariance
        x0 << 0, 0, 0, 0, 0 ,0;
        x0ErrorCovariance << 
            500, 0,   0,   0,   0,   0,
            0,   500, 0,   0,   0,   0,
            0,   0,   500, 0,   0,   0,
            0,   0,   0,   500, 0,   0,
            0,   0,   0,   0,   500, 0,
            0,   0,   0,   0,   0,   500;
        kf.init(x0ErrorCovariance, x0);

        const std::list<std::pair<double, double>> measurements = {
            {-393.66, 300.4},
            {-375.93, 301.78},
            {-351.04, 295.1},
            {-328.96, 305.19},
            {-299.35, 301.06},
            {-273.36, 302.05},
            {-245.89, 300},
            {-222.58, 303.57},
            {-198.03, 296.33},
            {-174.17, 297.65},
            {-146.32, 297.41},
            {-123.72, 299.61},
            {-103.47, 299.6},
            {-78.23, 302.39},
            {-52.63, 295.04},
            {-23.34, 300.09},
            {25.96, 294.72},
            {49.72, 298.61},
            {76.94, 294.64},
            {95.38, 284.88},
            {119.83, 272.82},
            {144.01, 264.93},
            {161.84, 251.46},
            {180.56, 241.27},
            {201.42, 222.98},
            {222.62, 203.73},
            {239.4, 184.1},
            {252.51, 166.12},
            {266.26, 138.71},
            {271.75, 119.71},
            {277.4, 100.41},
            {294.12, 79.76},
            {301.23, 50.62},
            {291.8, 32.99},
            {299.89, 2.14}
        };

        // Feed measurements into filter, output estimated states
        for(const std::pair<double, double>& measurement : measurements) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement.first, measurement.second;

            // update the model
            kf.update(y);
        }
        
        // estimate state position; 1.5 meters or so
        EXPECT_NEAR(measurements.back().first,  kf.get_state()[0], 1.5);
        EXPECT_NEAR(measurements.back().second, kf.get_state()[3], 1.5);
    }

    /**
    * RUN FREE FALL TESTS
    */

    const double gravityConst = -9.81; 

    std::list<double> get_1D_free_fall(const double initialPosition, const double measurementNoise, const size_t numberOfMeasurements, const double dt)
    {
        // set random
        std::random_device randomDevice;
        std::mt19937 randomEngine(randomDevice());
        std::uniform_real_distribution<double> errorDistribution(-measurementNoise, measurementNoise);

        std::list<double> trajectory;
        trajectory.insert(trajectory.cend(), initialPosition);

        double lastPosition = initialPosition;
        for(size_t i = 0; i < numberOfMeasurements; ++i)
        {
            const double newPosition = lastPosition + 0.5 * gravityConst * dt * dt; 

            trajectory.insert(trajectory.cend(), newPosition + errorDistribution(randomEngine));

            lastPosition = newPosition;
        }

        return trajectory;
    }


    TEST(KalmanFilteringTests, 1DProjectileMotionNoNoiseNoVariance) 
    {
        const int stateDimension = 3; // Number of states
        const int measurementDimension = 1; // Number of measurements

        const double dt = 1.0/30; // Time step
        const size_t numberOfMeasurements = 100;
        const double initialPosition = 0.0;
        const double measurementNoise = 0.0;
        const double expectedNoise = 0.0;

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension); // System dynamics matrix
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension); // Output matrix
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension); // Measurement noise covariance
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance

        // Discrete LTI projectile motion, measuring position only
        systemDynamics << 
            1,  dt, 0, 
            0,  1,  dt,
            0,  0,  1;
        // Position, speed, acceleration
        outputMatrix << 1, 0, 0;

        // Reasonable covariance matrices
        measurementNoiseCovariance << expectedNoise * expectedNoise;
        // Only speed and position are related
        processNoiseCovariance << 
            .05, .05, .0,
            .05, .05, .0,
            .0,  .0,  .0;
        estimateErrorCovariance << 
            .1, .1,    .1,
            .1, 10000, 10,
            .1, 10,    100;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);

        std::list<double> trajectoryPoints = get_1D_free_fall(initialPosition, measurementNoise, numberOfMeasurements, dt);

        // Best guess of initial states
        Eigen::VectorXd x0(stateDimension);
        x0 << 0, 0, gravityConst;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : trajectoryPoints) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(trajectoryPoints.back(), kf.get_state().x(), 0.001);
    }

    TEST(KalmanFilteringTests, 1DProjectileMotionNoNoiseSmallVariance) 
    {
        const int stateDimension = 3; // Number of states
        const int measurementDimension = 1; // Number of measurements

        const double dt = 1.0/30; // Time step
        const size_t numberOfMeasurements = 100;
        const double initialPosition = 0.0;
        const double measurementNoise = 0.0;
        const double expectedNoise = 0.001;

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension); // System dynamics matrix
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension); // Output matrix
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension); // Measurement noise covariance
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance

        // Discrete LTI projectile motion, measuring position only
        systemDynamics << 
            1,  dt, 0, 
            0,  1,  dt,
            0,  0,  1;
        // Position, speed, acceleration
        outputMatrix << 1, 0, 0;

        // Reasonable covariance matrices
        measurementNoiseCovariance << expectedNoise * expectedNoise;
        // Only speed and position are related
        processNoiseCovariance << 
            .05, .05, .0,
            .05, .05, .0,
            .0,  .0,  .0;
        estimateErrorCovariance << 
            .1, .1,    .1,
            .1, 10000, 10,
            .1, 10,    100;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);

        std::list<double> trajectoryPoints = get_1D_free_fall(initialPosition, measurementNoise, numberOfMeasurements, dt);

        // Best guess of initial states
        Eigen::VectorXd x0(stateDimension);
        x0 << 0, 0, gravityConst;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : trajectoryPoints) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(trajectoryPoints.back(), kf.get_state().x(), 0.001);
    }

    TEST(KalmanFilteringTests, 1DProjectileMotionNoNoiseMediumVariance) 
    {
        const int stateDimension = 3; // Number of states
        const int measurementDimension = 1; // Number of measurements

        const double dt = 1.0/30; // Time step
        const size_t numberOfMeasurements = 100;
        const double initialPosition = 0.0;
        const double measurementNoise = 0.0;
        const double expectedNoise = 0.01;

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension); // System dynamics matrix
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension); // Output matrix
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension); // Measurement noise covariance
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance

        // Discrete LTI projectile motion, measuring position only
        systemDynamics << 
            1,  dt, 0, 
            0,  1,  dt,
            0,  0,  1;
        // Position, speed, acceleration
        outputMatrix << 1, 0, 0;

        // Reasonable covariance matrices
        measurementNoiseCovariance << expectedNoise * expectedNoise;
        // Only speed and position are related
        processNoiseCovariance << 
            .05, .05, .0,
            .05, .05, .0,
            .0,  .0,  .0;
        estimateErrorCovariance << 
            .1, .1,    .1,
            .1, 10000, 10,
            .1, 10,    100;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);

        std::list<double> trajectoryPoints = get_1D_free_fall(initialPosition, measurementNoise, numberOfMeasurements, dt);

        // Best guess of initial states
        Eigen::VectorXd x0(stateDimension);
        x0 << 0, 0, gravityConst;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : trajectoryPoints) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(trajectoryPoints.back(), kf.get_state().x(), 0.001);
    }

    TEST(KalmanFilteringTests, 1DProjectileMotionNoNoiseHighVariance) 
    {
        const int stateDimension = 3; // Number of states
        const int measurementDimension = 1; // Number of measurements

        const double dt = 1.0/30; // Time step
        const size_t numberOfMeasurements = 100;
        const double initialPosition = 0.0;
        const double measurementNoise = 0.0;
        const double expectedNoise = 0.1;

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension); // System dynamics matrix
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension); // Output matrix
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension); // Measurement noise covariance
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance

        // Discrete LTI projectile motion, measuring position only
        systemDynamics << 
            1,  dt, 0, 
            0,  1,  dt,
            0,  0,  1;
        // Position, speed, acceleration
        outputMatrix << 1, 0, 0;

        // Reasonable covariance matrices
        measurementNoiseCovariance << expectedNoise * expectedNoise;
        // Only speed and position are related
        processNoiseCovariance << 
            .05, .05, .0,
            .05, .05, .0,
            .0,  .0,  .0;
        estimateErrorCovariance << 
            .1, .1,    .1,
            .1, 10000, 10,
            .1, 10,    100;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);

        std::list<double> trajectoryPoints = get_1D_free_fall(initialPosition, measurementNoise, numberOfMeasurements, dt);

        // Best guess of initial states
        Eigen::VectorXd x0(stateDimension);
        x0 << 0, 0, gravityConst;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : trajectoryPoints) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(trajectoryPoints.back(), kf.get_state().x(), 0.001);
    }

    TEST(KalmanFilteringTests, 1DProjectileMotionSmallNoiseSmallVariance) 
    {
        const int stateDimension = 3; // Number of states
        const int measurementDimension = 1; // Number of measurements

        const double dt = 1.0/30; // Time step
        const size_t numberOfMeasurements = 100;
        const double initialPosition = 0.0;
        const double measurementNoise = 0.001;
        const double expectedNoise = 0.001;

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension); // System dynamics matrix
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension); // Output matrix
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension); // Measurement noise covariance
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance

        // Discrete LTI projectile motion, measuring position only
        systemDynamics << 
            1,  dt, 0, 
            0,  1,  dt,
            0,  0,  1;
        // Position, speed, acceleration
        outputMatrix << 1, 0, 0;

        // Reasonable covariance matrices
        measurementNoiseCovariance << expectedNoise * expectedNoise;
        // Only speed and position are related
        processNoiseCovariance << 
            .05, .05, .0,
            .05, .05, .0,
            .0,  .0,  .0;
        estimateErrorCovariance << 
            .1, .1,    .1,
            .1, 10000, 10,
            .1, 10,    100;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);

        std::list<double> trajectoryPoints = get_1D_free_fall(initialPosition, measurementNoise, numberOfMeasurements, dt);

        // Best guess of initial states
        Eigen::VectorXd x0(stateDimension);
        x0 << 0, 0, gravityConst;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : trajectoryPoints) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(trajectoryPoints.back(), kf.get_state().x(), 0.001);
    }

    TEST(KalmanFilteringTests, 1DProjectileMotionMediumNoiseMediumVariance) 
    {
        const int stateDimension = 3; // Number of states
        const int measurementDimension = 1; // Number of measurements

        const double dt = 1.0/30; // Time step
        const size_t numberOfMeasurements = 100;
        const double initialPosition = 0.0;
        const double measurementNoise = 0.01;
        const double expectedNoise = 0.01;

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension); // System dynamics matrix
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension); // Output matrix
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension); // Measurement noise covariance
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance

        // Discrete LTI projectile motion, measuring position only
        systemDynamics << 
            1,  dt, 0, 
            0,  1,  dt,
            0,  0,  1;
        // Position, speed, acceleration
        outputMatrix << 1, 0, 0;

        // Reasonable covariance matrices
        measurementNoiseCovariance << expectedNoise * expectedNoise;
        // Only speed and position are related
        processNoiseCovariance << 
            .05, .05, .0,
            .05, .05, .0,
            .0,  .0,  .0;
        estimateErrorCovariance << 
            .1, .1,    .1,
            .1, 10000, 10,
            .1, 10,    100;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);

        std::list<double> trajectoryPoints = get_1D_free_fall(initialPosition, measurementNoise, numberOfMeasurements, dt);

        // Best guess of initial states
        Eigen::VectorXd x0(stateDimension);
        x0 << 0, 0, gravityConst;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : trajectoryPoints) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(trajectoryPoints.back(), kf.get_state().x(), 0.001);
    }

    TEST(KalmanFilteringTests, 1DProjectileMotionHighNoiseHighVariance) 
    {
        const int stateDimension = 3; // Number of states
        const int measurementDimension = 1; // Number of measurements

        const double dt = 1.0/30; // Time step
        const size_t numberOfMeasurements = 1500;
        const double initialPosition = 0.0;
        const double measurementNoise = 0.1;
        const double expectedNoise = 0.1;

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension); // System dynamics matrix
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension); // Output matrix
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension); // Measurement noise covariance
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance

        // Discrete LTI projectile motion, measuring position only
        systemDynamics << 
            1,  dt, 0, 
            0,  1,  dt,
            0,  0,  1;
        // Position, speed, acceleration
        outputMatrix << 1, 0, 0;

        // Reasonable covariance matrices
        measurementNoiseCovariance << expectedNoise * expectedNoise;
        // Only speed and position are related
        processNoiseCovariance << 
            .05, .05, .0,
            .05, .05, .0,
            .0,  .0,  .0;
        estimateErrorCovariance << 
            .1, .1,    .1,
            .1, 10000, 10,
            .1, 10,    100;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);

        std::list<double> trajectoryPoints = get_1D_free_fall(initialPosition, measurementNoise, numberOfMeasurements, dt);

        // Best guess of initial states
        Eigen::VectorXd x0(stateDimension);
        x0 << 0, 0, gravityConst;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : trajectoryPoints) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }

        // estimate state position
        EXPECT_NEAR(trajectoryPoints.back(), kf.get_state().x(), 0.05);
    }

    TEST(KalmanFilteringTests, 1DProjectileParabolaWithNoise) 
    {
        const int stateDimension = 3; // Number of states
        const int measurementDimension = 1; // Number of measurements

        const double dt = 1.0/30; // Time step

        Eigen::MatrixXd systemDynamics(stateDimension, stateDimension); // System dynamics matrix
        Eigen::MatrixXd outputMatrix(measurementDimension, stateDimension); // Output matrix
        Eigen::MatrixXd processNoiseCovariance(stateDimension, stateDimension); // Process noise covariance
        Eigen::MatrixXd measurementNoiseCovariance(measurementDimension, measurementDimension); // Measurement noise covariance
        Eigen::MatrixXd estimateErrorCovariance(stateDimension, stateDimension); // Estimate error covariance

        // Discrete LTI projectile motion, measuring position only
        systemDynamics << 
            1,  dt, 0, 
            0,  1,  dt, 
            0,  0,  1;
        // Position, speed, gravity
        outputMatrix << 1, 0, 0;

        // Reasonable covariance matrices
        // Only speed and position are related
        processNoiseCovariance << 
            .05, .05, .0,
            .05, .05, .0,
            .0,  .0,  .0;
        measurementNoiseCovariance << 5;
        estimateErrorCovariance << 
            .1, .1,    .1,
            .1, 10000, 10,
            .1, 10,    100;

        // Construct the filter
        KalmanFilter kf(systemDynamics, outputMatrix, processNoiseCovariance, measurementNoiseCovariance);

        // List of noisy position measurements (y)
        const std::vector<double> measurements = {
            1.04202710058, 1.10726790452, 1.2913511148, 1.48485250951, 1.72825901034,
            1.74216489744, 2.11672039768, 2.14529225112, 2.16029641405, 2.21269371128,
            2.57709350237, 2.6682215744, 2.51641839428, 2.76034056782, 2.88131780617,
            2.88373786518, 2.9448468727, 2.82866600131, 3.0006601946, 3.12920591669,
            2.858361783, 2.83808170354, 2.68975330958, 2.66533185589, 2.81613499531,
            2.81003612051, 2.88321849354, 2.69789264832, 2.4342229249, 2.23464791825,
            2.30278776224, 2.02069770395, 1.94393985809, 1.82498398739, 1.52526230354,
            1.86967808173, 1.18073207847, 1.10729605087, 0.916168349913, 0.678547664519,
            0.562381751596, 0.355468474885, -0.155607486619, -0.287198661013, -0.602973173813
        };

        // Best guess of initial states
        Eigen::VectorXd x0(stateDimension);
        x0 << measurements[0], 0, -9.81;
        kf.init(estimateErrorCovariance, x0);

        // Feed measurements into filter, output estimated states
        for(const double measurement : measurements) {
            Eigen::VectorXd y(measurementDimension);
            y << measurement;

            // update the model
            kf.update(y);
        }
        
        // estimate end state position
        EXPECT_NEAR(measurements.back(), kf.get_state().x(), 0.15);
    }
}