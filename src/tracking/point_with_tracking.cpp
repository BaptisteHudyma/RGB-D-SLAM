#include "point_with_tracking.hpp"

#include "coordinates/point_coordinates.hpp"
#include "outputs/logger.hpp"
#include "types.hpp"
#include "utils/covariances.hpp"

#include <stdexcept>

namespace rgbd_slam::tracking {

/**
 * Define the estimator for the inverse depth fuse/tracker
 */
template<int N = 3, int M = 2, int NE = N, int ME = M> class Point2dEstimator : public StateEstimator<N, M, NE, ME>
{
  public:
    virtual ~Point2dEstimator() = default;

    std::pair<Eigen::Vector<double, M>, Eigen::Matrix<double, ME, NE>> h(
            const Eigen::Vector<double, N>& state) const noexcept override
    {
        ScreenCoordinate2D sc;
        matrix23 toScreenJacobian;
        if (!WorldCoordinate(state).to_screen_coordinates(_w2c, sc, toScreenJacobian))
        {
            outputs::log_error("screen projection failed");
        }
        return {sc, toScreenJacobian};
    }

    inline Eigen::Matrix<double, ME, ME> h_innovation(
            const Eigen::Vector<double, N>& state,
            const Eigen::Matrix<double, NE, NE>& estimateErrorCovariance,
            const Eigen::Matrix<double, ME, NE>& hJacobian) const noexcept override
    {
        const Eigen::Matrix<double, 2, 6>& hPoseJacobian =
                utils::world_transform_of_2d_point_jacobian(WorldCoordinate(state), _w2c);

        const matrix22& cov = utils::propagate_covariance(estimateErrorCovariance, hJacobian) +
                              utils::propagate_covariance(_poseCovariance, hPoseJacobian);
        return cov;
    }

    Point2dEstimator(const Eigen::Vector<double, N>& feature,
                     const Eigen::Matrix<double, NE, NE>& featureCovariance,
                     const Eigen::Vector<double, M>& measurment,
                     const Eigen::Matrix<double, ME, ME>& measurmentCovariance,
                     const WorldToCameraMatrix& w2c,
                     const matrix66& poseCovariance) :

        StateEstimator<N, M, NE, ME>(feature, featureCovariance, measurment, measurmentCovariance),
        _w2c(w2c),
        _poseCovariance(poseCovariance)
    {
    }

  private:
    const WorldToCameraMatrix _w2c;
    // pose covariance is represented a pose matrix and the rotation part as a delta theta rotation error
    const matrix66 _poseCovariance;
};

template<int N = 3, int M = 3, int NE = N, int ME = M> class Point3dEstimator : public StateEstimator<N, M, NE, ME>
{
  public:
    virtual ~Point3dEstimator() = default;

    std::pair<Eigen::Vector<double, M>, Eigen::Matrix<double, ME, NE>> h(
            const Eigen::Vector<double, N>& state) const noexcept override
    {
        matrix33 toScreenJacobian;
        ScreenCoordinate sc;
        if (!WorldCoordinate(state).to_screen_coordinates(_w2c, sc, toScreenJacobian))
        {
            outputs::log_error("screen projection failed");
        }
        return {sc, toScreenJacobian};
    }

    inline Eigen::Matrix<double, ME, ME> h_innovation(
            const Eigen::Vector<double, N>& state,
            const Eigen::Matrix<double, NE, NE>& estimateErrorCovariance,
            const Eigen::Matrix<double, ME, NE>& hJacobian) const noexcept override
    {
        const Eigen::Matrix<double, 3, 6>& hPoseJacobian =
                utils::world_transform_of_point_jacobian(WorldCoordinate(state), _w2c);
        const matrix33& cov = utils::propagate_covariance(estimateErrorCovariance, hJacobian) +
                              utils::propagate_covariance(_poseCovariance, hPoseJacobian);
        return cov;
    }

    Point3dEstimator(const Eigen::Vector<double, N>& feature,
                     const Eigen::Matrix<double, NE, NE>& featureCovariance,
                     const Eigen::Vector<double, M>& measurment,
                     const Eigen::Matrix<double, ME, ME>& measurmentCovariance,
                     const WorldToCameraMatrix& w2c,
                     const matrix66& poseCovariance) :

        StateEstimator<N, M, NE, ME>(feature, featureCovariance, measurment, measurmentCovariance),
        _w2c(w2c),
        _poseCovariance(poseCovariance)
    {
    }

  private:
    const WorldToCameraMatrix _w2c;
    const matrix66 _poseCovariance;
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
    if (_kalmanFuse3d == nullptr or _kalmanFuse2d == nullptr)
    {
        build_kalman_filter();
    }

    if (_descriptor.empty() or _descriptor.cols <= 0)
        throw std::invalid_argument("Point constructor: descriptor is empty");
    if (_coordinates.hasNaN())
        throw std::invalid_argument("Point constructor: point coordinates contains NaN");
    if (not utils::is_covariance_valid(_covariance))
        throw std::invalid_argument("Point constructor: covariance in invalid");
};

bool Point::track_3d(const ScreenCoordinate& newDetection,
                     const WorldToCameraMatrix& w2c,
                     const matrix66& poseCovariance) noexcept
{
    assert(_kalmanFuse3d != nullptr);
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance : the covariance is invalid");
        exit(-1);
    }

    try
    {
        Point3dEstimator estimator(
                _coordinates, _covariance, newDetection, newDetection.get_covariance(), w2c, poseCovariance);

        const auto& [newState, newCovariance] = _kalmanFuse3d->get_new_state(&estimator);

        if (not utils::is_covariance_valid(newCovariance))
        {
            outputs::log_error("New depth point covariance is invalid after merge");
            return false;
        }

        _coordinates << newState;
        _covariance << newCovariance;
        return true;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exception: " + std::string(ex.what()));
    }
    return false;
}

bool Point::track_2d(const ScreenCoordinate2D& newDetection,
                     const WorldToCameraMatrix& w2c,
                     const matrix66& poseCovariance) noexcept
{
    assert(_kalmanFuse2d != nullptr);
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance : the covariance is invalid");
        exit(-1);
    }

    try
    {
        Point2dEstimator estimator(
                _coordinates, _covariance, newDetection, newDetection.get_covariance(), w2c, poseCovariance);

        const auto& [newState, newCovariance] = _kalmanFuse2d->get_new_state(&estimator);

        if (not utils::is_covariance_valid(newCovariance))
        {
            outputs::log_error("New depth point covariance is invalid after merge");
            return false;
        }

        _coordinates = newState;
        _covariance = WorldCoordinateCovariance {newCovariance};

        return true;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exception: " + std::string(ex.what()));
    }
    return false;
}

void Point::build_kalman_filter() noexcept
{
    _kalmanFuse3d = std::make_unique<tracking::ExtendedKalmanFilter<3, 3>>(matrix33::Zero());
    _kalmanFuse2d = std::make_unique<tracking::ExtendedKalmanFilter<3, 2>>(matrix33::Zero());
}

} // namespace rgbd_slam::tracking
