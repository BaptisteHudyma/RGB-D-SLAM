#include "point_with_tracking.hpp"

#include "coordinates/point_coordinates.hpp"
#include "logger.hpp"
#include "types.hpp"
#include "utils/covariances.hpp"
#include <stdexcept>

namespace rgbd_slam::tracking {

/**
 * Define the estimator for the inverse depth fuse/tracker
 */
template<int N = 3, int M = 2> class Point2dEstimator : public StateEstimator<N, M>
{
  public:
    Eigen::Vector<double, M> h(const Eigen::Vector<double, N>& state) const noexcept override
    {
        ScreenCoordinate2D sc;
        if (!WorldCoordinate(state).to_screen_coordinates(_w2c, sc))
        {
            outputs::log_error("screen projection failed");
        }
        return sc;
    }

    Eigen::Matrix<double, M, N> h_jacobian(const Eigen::Vector<double, N>& state) const noexcept override
    {
        return WorldCoordinate(state).to_screen2d_coordinates_jacobian(_w2c);
    }

    Point2dEstimator(const Eigen::Vector<double, N>& feature,
                     const Eigen::Matrix<double, N, N>& featureCovariance,
                     const Eigen::Vector<double, M>& measurment,
                     const Eigen::Matrix<double, M, M>& measurmentCovariance,
                     const WorldToCameraMatrix& w2c) :

        StateEstimator<N, M>(feature, featureCovariance, measurment, measurmentCovariance),
        _w2c(w2c)
    {
    }

  private:
    const WorldToCameraMatrix _w2c;
};

template<int N = 3, int M = 3> class Point3dEstimator : public StateEstimator<N, M>
{
  public:
    Eigen::Vector<double, M> h(const Eigen::Vector<double, N>& state) const noexcept override
    {
        ScreenCoordinate sc;
        if (!WorldCoordinate(state).to_screen_coordinates(_w2c, sc))
        {
            outputs::log_error("screen projection failed");
        }
        return sc;
    }

    Eigen::Matrix<double, M, N> h_jacobian(const Eigen::Vector<double, N>& state) const noexcept override
    {
        return WorldCoordinate(state).to_screen_coordinates_jacobian(_w2c);
    }

    Point3dEstimator(const Eigen::Vector<double, N>& feature,
                     const Eigen::Matrix<double, N, N>& featureCovariance,
                     const Eigen::Vector<double, M>& measurment,
                     const Eigen::Matrix<double, M, M>& measurmentCovariance,
                     const WorldToCameraMatrix& w2c) :

        StateEstimator<N, M>(feature, featureCovariance, measurment, measurmentCovariance),
        _w2c(w2c)
    {
    }

  private:
    const WorldToCameraMatrix _w2c;
};

/**
 * Point
 */

Point::Point(const WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor) :
    _coordinates(coordinates),
    _descriptor(descriptor),
    _covariance(covariance)
{
    build_kalman_filter();

    if (_descriptor.empty() or _descriptor.cols <= 0)
        throw std::invalid_argument("Point constructor: descriptor is empty");
    if (_coordinates.hasNaN())
        throw std::invalid_argument("Point constructor: point coordinates contains NaN");
    if (not utils::is_covariance_valid(_covariance))
        throw std::invalid_argument("Point constructor: covariance in invalid");
};

double Point::track(const WorldCoordinate& newDetectionCoordinates, const matrix33& newDetectionCovariance) noexcept
{
    assert(_kalmanFilter != nullptr);
    if (not utils::is_covariance_valid(newDetectionCovariance))
    {
        outputs::log_error("newDetectionCovariance: the covariance is invalid");
        return -1;
    }
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance : the covariance is invalid");
        exit(-1);
    }

    try
    {
        const auto& [newCoordinates, newCovariance] = _kalmanFilter->get_new_state(
                _coordinates, _covariance, newDetectionCoordinates, newDetectionCovariance);

        /*
        // moved above the uncertainty of this point
        _isMoving = ((_coordinates - newDetectionCoordinates).array() >
                        newDetectionCovariance.diagonal().cwiseSqrt().array())
                            .any();
        */

        const double score = (_coordinates - newCoordinates).norm();

        _coordinates << newCoordinates;
        _covariance << newCovariance;
        return score;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exeption: " + std::string(ex.what()));
        return -1;
    }
}

bool Point::track_3d(const ScreenCoordinate& newDetection, const WorldToCameraMatrix& w2c) noexcept
{
    assert(_kalmanFuse3d != nullptr);
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance : the covariance is invalid");
        exit(-1);
    }

    try
    {
        const matrix33& screenPointCovariance = newDetection.get_covariance();

        Point3dEstimator estimator(_coordinates, _covariance, newDetection, screenPointCovariance, w2c);

        const auto& [newState, newCovariance] = _kalmanFuse3d->get_new_state(&estimator);

        if (not utils::is_covariance_valid(newCovariance))
        {
            outputs::log_error("Inverse depth point covariance is invalid after merge");
            return false;
        }

        _coordinates << newState;
        _covariance << newCovariance;
        return true;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exeption: " + std::string(ex.what()));
    }
    return false;
}

bool Point::track_2d(const ScreenCoordinate2D& newDetection, const WorldToCameraMatrix& w2c) noexcept
{
    assert(_kalmanFuse2d != nullptr);
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance : the covariance is invalid");
        exit(-1);
    }

    try
    {
        const matrix22& screenPointCovariance = newDetection.get_covariance();
        Point2dEstimator estimator(_coordinates, _covariance, newDetection, screenPointCovariance, w2c);

        const auto& [newState, newCovariance] = _kalmanFuse2d->get_new_state(&estimator);

        if (not utils::is_covariance_valid(newCovariance))
        {
            outputs::log_error("Inverse depth point covariance is invalid after merge");
            return false;
        }

        _coordinates = newState;
        _covariance = WorldCoordinateCovariance {newCovariance};

        return true;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exeption: " + std::string(ex.what()));
    }
    return false;
}

void Point::build_kalman_filter() noexcept
{
    const matrix33 systemDynamics = matrix33::Identity(); // points are not supposed to move, so no dynamics
    const matrix33 outputMatrix = matrix33::Identity();   // we need all positions

    const double parametersProcessNoise = SQR(0.01);                                       // TODO set in parameters
    const matrix33 processNoiseCovariance = matrix33::Identity() * parametersProcessNoise; // Process noise covariance

    _kalmanFilter =
            std::make_unique<tracking::SharedKalmanFilter<3, 3>>(systemDynamics, outputMatrix, processNoiseCovariance);

    _kalmanFuse3d = std::make_unique<tracking::ExtendedKalmanFilter<3, 3>>(processNoiseCovariance);

    _kalmanFuse2d = std::make_unique<tracking::ExtendedKalmanFilter<3, 2>>(matrix33::Identity() * 1e-6);
}

} // namespace rgbd_slam::tracking