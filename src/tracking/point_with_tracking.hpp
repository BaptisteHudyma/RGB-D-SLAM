#ifndef RGBDSLAM_TRACKING_POINT_WITH_TRACKING_HPP
#define RGBDSLAM_TRACKING_POINT_WITH_TRACKING_HPP

#include "utils/coordinates/point_coordinates.hpp"
#include "kalman_filter.hpp"
#include <opencv2/opencv.hpp>

namespace rgbd_slam::tracking {

struct Point
{
    // world coordinates
    utils::WorldCoordinate _coordinates;
    // 3D descriptor (ORB)
    cv::Mat _descriptor;
    // position covariance
    WorldCoordinateCovariance _covariance;

    Point(const utils::WorldCoordinate& coordinates,
          const WorldCoordinateCovariance& covariance,
          const cv::Mat& descriptor);

    /**
     * \brief update this point coordinates using a new detection
     * \param[in] newDetectionCoordinates The newly detected point
     * \param[in] newDetectionCovariance The newly detected point covariance
     * \return The distance between the updated position ans the previous one, -1 if an error occured
     */
    double track(const utils::WorldCoordinate& newDetectionCoordinates,
                 const matrix33& newDetectionCovariance) noexcept;

  private:
    /**
     * \brief Build the caracteristics of the kalman filter
     */
    static void build_kalman_filter() noexcept;

    // shared kalman filter, between all points
    inline static std::unique_ptr<tracking::SharedKalmanFilter<3, 3>> _kalmanFilter = nullptr;
};

struct PointInverseDepth
{
    static constexpr uint firstPoseIndex = utils::InverseDepthWorldPoint::firstPoseIndex;
    static constexpr uint inverseDepthIndex = utils::InverseDepthWorldPoint::inverseDepthIndex;
    static constexpr uint thetaIndex = utils::InverseDepthWorldPoint::thetaIndex;
    static constexpr uint phiIndex = utils::InverseDepthWorldPoint::phiIndex;

    struct Covariance : public matrix66
    {
        matrix33 get_first_pose_covariance() const { return block<3, 3>(firstPoseIndex, firstPoseIndex); }
    };

    utils::InverseDepthWorldPoint _coordinates;
    Covariance _covariance;
    cv::Mat _descriptor;

    PointInverseDepth(const utils::ScreenCoordinate2D& observation,
                      const CameraToWorldMatrix& c2w,
                      const matrix33& stateCovariance,
                      const cv::Mat& descriptor);

    PointInverseDepth(const PointInverseDepth& other);

    matrix33 get_covariance_of_observed_pose() const noexcept { return _covariance.block<3, 3>(0, 0); }

    /**
     * \brief Add an new measurment to the tracking
     * \param[in] observation The new observation
     * \param[in] c2w The cam to world matrix
     * \param[in] stateCovariance The covariance of the observer position
     * \return True if the tracking succeeded, false if something is wrong
     */
    [[nodiscard]] bool track(const utils::ScreenCoordinate2D& observation,
                             const CameraToWorldMatrix& c2w,
                             const matrix33& stateCovariance,
                             const cv::Mat& descriptor);
    [[nodiscard]] bool track(const utils::ScreenCoordinate& observation,
                             const CameraToWorldMatrix& c2w,
                             const matrix33& stateCovariance,
                             const cv::Mat& descriptor);

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
            const utils::InverseDepthWorldPoint& coordinates, const matrix66& covariance);

    [[nodiscard]] static WorldCoordinateCovariance compute_cartesian_covariance(
            const matrix66& covariance, const Eigen::Matrix<double, 3, 6>& jacobian);

    /**
     * \brief Get the inverse depth covariance from the world point covariance
     */
    [[nodiscard]] static Covariance compute_inverse_depth_covariance(const WorldCoordinateCovariance& pointCovariance,
                                                                     const matrix33& firstPoseCovariance,
                                                                     const Eigen::Matrix<double, 6, 6>& jacobian);

    /**
     * \brief compute a linearity score, indicating if this point can be converted to a cartesian coordinate with true
     * gaussian covariance
     */
    [[nodiscard]] double compute_linearity_score(const CameraToWorldMatrix& cameraToWorld) const noexcept;

  protected:
    /**
     * \brief update the value of this point using an observation in cartesian space
     */
    bool update_with_cartesian(const utils::WorldCoordinate& point,
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