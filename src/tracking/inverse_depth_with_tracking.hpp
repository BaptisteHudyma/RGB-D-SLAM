
#ifndef RGBDSLAM_TRACKING_INVERSE_DEPTH_WITH_TRACKING_HPP
#define RGBDSLAM_TRACKING_INVERSE_DEPTH_WITH_TRACKING_HPP

#include "types.hpp"
#include "coordinates/inverse_depth_coordinates.hpp"
#include "kalman_filter.hpp"
#include "../utils/line.hpp"

#include <opencv2/opencv.hpp>

#include "extended_kalman_filter.hpp"

namespace rgbd_slam::tracking {

/**
 * Define the estimator for the inverse depth fuse/tracker
 */
template<int N = 6, int M = 2> class InverseDepthEstimator : public StateEstimator<N, M>
{
  public:
    Eigen::Vector<double, N> state() const noexcept override { return _feature; }
    Eigen::Matrix<double, N, N> state_covariance() const noexcept override { return _featureCovariance; }

    Eigen::Vector<double, M> measurment() const noexcept override { return _measurment; }
    Eigen::Matrix<double, M, M> measurment_covariance() const noexcept override { return _measurmentCovariance; }

    Eigen::Vector<double, N> f(const Eigen::Vector<double, N>& state) const noexcept override
    {
        // no dynamics, just return the same state
        return state;
    }

    Eigen::Matrix<double, N, N> f_jacobian(const Eigen::Vector<double, N>& state) const noexcept override
    {
        std::ignore = state;
        // no dynamic jacobian
        return Eigen::Matrix<double, N, N>::Identity();
    }

    Eigen::Vector<double, M> h(const Eigen::Vector<double, N>& state) const noexcept override
    {
        return InverseDepthWorldPoint(state).get_projected_screen_estimation(_w2c);
    }

    Eigen::Matrix<double, M, N> h_jacobian(const Eigen::Vector<double, N>& state) const noexcept override
    {
        return InverseDepthWorldPoint(state).get_projected_screen_estimation_jacobian(_w2c, 0.0);
    }

    InverseDepthEstimator(const Eigen::Vector<double, N>& feature,
                          const Eigen::Matrix<double, N, N>& featureCovariance,
                          const Eigen::Vector<double, M>& measurment,
                          const Eigen::Matrix<double, M, M>& measurmentCovariance,
                          const WorldToCameraMatrix& w2c) :
        _feature(feature),
        _featureCovariance(featureCovariance),
        _measurment(measurment),
        _measurmentCovariance(measurmentCovariance),
        _w2c(w2c)
    {
    }

  private:
    const Eigen::Vector<double, N> _feature;
    const Eigen::Matrix<double, N, N> _featureCovariance;
    const Eigen::Vector<double, M> _measurment;
    const Eigen::Matrix<double, M, M> _measurmentCovariance;
    const WorldToCameraMatrix _w2c;
};

/**
 * \brief Defines a 2D point, with tracking capabilities
 */
struct PointInverseDepth
{
    static constexpr uint firstPoseIndex = InverseDepthWorldPoint::firstPoseIndex;
    static constexpr uint inverseDepthIndex = InverseDepthWorldPoint::inverseDepthIndex;
    static constexpr uint thetaIndex = InverseDepthWorldPoint::thetaIndex;
    static constexpr uint phiIndex = InverseDepthWorldPoint::phiIndex;

    struct Covariance : public matrix66
    {
        using matrix66::matrix66;
        matrix33 get_first_pose_covariance() const { return block<3, 3>(firstPoseIndex, firstPoseIndex); }
        double get_inverse_depth_variance() const { return diagonal()(inverseDepthIndex); };
        double get_theta_variance() const { return diagonal()(thetaIndex); };
        double get_phi_variance() const { return diagonal()(phiIndex); };
    };

    InverseDepthWorldPoint _coordinates;
    Covariance _covariance;
    cv::Mat _descriptor;

    PointInverseDepth(const ScreenCoordinate2D& observation,
                      const CameraToWorldMatrix& c2w,
                      const matrix33& stateCovariance,
                      const cv::Mat& descriptor = cv::Mat());

    PointInverseDepth(const PointInverseDepth& other);

    [[nodiscard]] matrix33 get_covariance_of_observed_pose() const noexcept { return _covariance.block<3, 3>(0, 0); }

    /**
     * \brief Add an new measurment to the tracking
     * \param[in] observation The new observation
     * \param[in] c2w The cam to world matrix
     * \param[in] stateCovariance The covariance of the observer position
     * \param[in] descriptor The descriptor of this point
     * \return True if the tracking succeeded, false if something is wrong
     */
    [[nodiscard]] bool track(const ScreenCoordinate2D& observation,
                             const CameraToWorldMatrix& c2w,
                             const matrix33& stateCovariance,
                             const cv::Mat& descriptor) noexcept;
    /**
     * \brief Add an new measurment to the tracking
     * \param[in] observation The new observation
     * \param[in] c2w The cam to world matrix
     * \param[in] stateCovariance The covariance of the observer position
     * \param[in] descriptor The descriptor of this point
     * \return True if the tracking succeeded, false if something is wrong
     */
    [[nodiscard]] bool track(const ScreenCoordinate& observation,
                             const CameraToWorldMatrix& c2w,
                             const matrix33& stateCovariance,
                             const cv::Mat& descriptor);

    /**
     * \brief Compute a line that represent the potential position of the inverse depth point, taking into account the
     * uncertainty
     * \param[in] w2c The world to camera matrix
     * \param[out] screenSegment the projection of this point as a segment
     * \return true if screenSegment is valid
     */
    [[nodiscard]] bool to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                             utils::Segment<2>& screenSegment) const noexcept;

    /**
     * \brief Compute the covariance of the cartesian projection of this inverse depth, in camera space
     */
    [[nodiscard]] CameraCoordinateCovariance get_camera_coordinate_variance(const WorldToCameraMatrix& w2c) const;

    /**
     * \brief Compute the covariance of the cartesian projection of this inverse depth, in screen space
     */
    [[nodiscard]] ScreenCoordinateCovariance get_screen_coordinate_variance(const WorldToCameraMatrix& w2c) const;

    /**
     * \brief Compute th covariance of the cartesian projection of this inverse depth
     */
    [[nodiscard]] static WorldCoordinateCovariance compute_cartesian_covariance(
            const InverseDepthWorldPoint& coordinates, const matrix66& covariance);

    [[nodiscard]] static WorldCoordinateCovariance compute_cartesian_covariance(
            const matrix66& covariance, const Eigen::Matrix<double, 3, 6>& jacobian);

    /**
     * \brief Get the inverse depth covariance from the world point covariance
     */
    [[nodiscard]] static Covariance compute_inverse_depth_covariance(const WorldCoordinateCovariance& pointCovariance,
                                                                     const matrix33& firstPoseCovariance,
                                                                     const Eigen::Matrix<double, 6, 3>& jacobian);

    /**
     * \brief compute a linearity score, indicating if this point can be converted to a cartesian coordinate with true
     * gaussian covariance
     */
    [[nodiscard]] double compute_linearity_score(const CameraToWorldMatrix& cameraToWorld) const noexcept;

    [[nodiscard]] bool is_moving() const noexcept { return _isMoving; }

  protected:
    /**
     * \brief Build the caracteristics of the kalman filter
     */
    static void build_kalman_filter() noexcept;

    // shared kalman filter, between all points
    inline static std::unique_ptr<tracking::SharedKalmanFilter<3, 3>> _kalmanFilter = nullptr;
    inline static std::unique_ptr<tracking::ExtendedKalmanFilter<6, 2>> _extendedKalmanFilter = nullptr;

  private:
    bool _isMoving = false;
};

} // namespace rgbd_slam::tracking

#endif