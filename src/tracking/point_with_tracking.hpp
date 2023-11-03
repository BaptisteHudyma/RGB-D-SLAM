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
    utils::InverseDepthWorldPoint _coordinates;
    matrix66 _covariance = matrix66::Zero();
    cv::Mat _descriptor;

    PointInverseDepth(const utils::ScreenCoordinate2D& observation,
                      const CameraToWorldMatrix& c2w,
                      const matrix33& stateCovariance,
                      const cv::Mat& descriptor);

    PointInverseDepth(const PointInverseDepth& other);

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

    /**
     * \brief Compute th covariance of the cartesian projection of this inverse depth
     */
    [[nodiscard]] WorldCoordinateCovariance get_cartesian_covariance() const noexcept;

    /**
     * \brief Compute the covariance of the cartesian projection of this inverse depth, in camera space
     */
    [[nodiscard]] CameraCoordinateCovariance get_camera_coordinate_variance(
            const WorldToCameraMatrix& w2c, const matrix33& stateCovariance) const noexcept;

    /**
     * \brief Compute the covariance of the cartesian projection of this inverse depth, in screen space
     */
    [[nodiscard]] ScreenCoordinateCovariance get_screen_coordinate_variance(
            const WorldToCameraMatrix& w2c, const matrix33& stateCovariance) const noexcept;

    /**
     * \brief Get the inverse depth covariance from the world point covariance
     */
    [[nodiscard]] matrix66 get_inverse_depth_covariance(const utils::WorldCoordinate& point,
                                                        const WorldCoordinateCovariance& pointCovariance,
                                                        const matrix33& posevariance) const noexcept;

  protected:
    /**
     * \brief Construct an inverse depth point from a camera point (intend to be used for new observations)
     */
    PointInverseDepth(const utils::CameraCoordinate& cameraCoordinates,
                      const CameraCoordinateCovariance& cameraCovariance,
                      const CameraToWorldMatrix& c2w,
                      const matrix33& posevariance);

    /**
     * \brief Build the caracteristics of the kalman filter
     */
    static void build_kalman_filter() noexcept;

    // shared kalman filter, between all points
    inline static std::unique_ptr<tracking::SharedKalmanFilter<6, 6>> _kalmanFilter = nullptr;
};

} // namespace rgbd_slam::tracking

#endif