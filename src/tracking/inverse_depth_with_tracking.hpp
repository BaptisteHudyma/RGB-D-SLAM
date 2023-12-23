
#ifndef RGBDSLAM_TRACKING_INVERSE_DEPTH_WITH_TRACKING_HPP
#define RGBDSLAM_TRACKING_INVERSE_DEPTH_WITH_TRACKING_HPP

#include "types.hpp"
#include "coordinates/inverse_depth_coordinates.hpp"
#include "kalman_filter.hpp"
#include "../utils/line.hpp"

#include <opencv2/opencv.hpp>

namespace rgbd_slam::tracking {

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
    };

    InverseDepthWorldPoint _coordinates;
    Covariance _covariance;
    cv::Mat _descriptor;

    PointInverseDepth(const ScreenCoordinate2D& observation,
                      const CameraToWorldMatrix& c2w,
                      const matrix33& stateCovariance);
    PointInverseDepth(const ScreenCoordinate2D& observation,
                      const CameraToWorldMatrix& c2w,
                      const matrix33& stateCovariance,
                      const cv::Mat& descriptor);

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
                             const cv::Mat& descriptor);
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

  protected:
    /**
     * \brief update the value of this point using an observation in cartesian space
     */
    [[nodiscard]] bool update_with_cartesian(const WorldCoordinate& point,
                                             const WorldCoordinateCovariance& covariance,
                                             const cv::Mat& descriptor);

    /**
     * \brief Build the caracteristics of the kalman filter
     */
    static void build_kalman_filter() noexcept;

    // shared kalman filter, between all points
    inline static std::unique_ptr<tracking::SharedKalmanFilter<3, 3>> _kalmanFilter = nullptr;
};

} // namespace rgbd_slam::tracking

#endif