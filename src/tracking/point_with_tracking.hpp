#ifndef RGBDSLAM_TRACKING_POINT_WITH_TRACKING_HPP
#define RGBDSLAM_TRACKING_POINT_WITH_TRACKING_HPP

#include "types.hpp"
#include "coordinates/point_coordinates.hpp"
#include "kalman_filter.hpp"
#include "extended_kalman_filter.hpp"

#include <opencv2/opencv.hpp>

namespace rgbd_slam::tracking {

/**
 * \brief Defines a point, with tracking capabilities
 */
struct Point
{
    // world coordinates
    WorldCoordinate _coordinates;
    // 3D descriptor (ORB)
    cv::Mat _descriptor;
    // position covariance
    WorldCoordinateCovariance _covariance;

    Point(const WorldCoordinate& coordinates, const WorldCoordinateCovariance& covariance, const cv::Mat& descriptor);

    /**
     * \brief update this point coordinates using another one
     * \param[in] otherCoordinates The point to update with
     * \param[in] otherCovariance The covariance of the point to update with
     * \return The distance between the updated position ans the previous one, -1 if an error occured
     */
    double track(const WorldCoordinate& otherCoordinates, const matrix33& otherCovariance) noexcept;

    /**
     * \brief track this point coordinates using a new detection, with no depth infos
     * \param[in] newDetection The new 3D detection
     * \param[in] w2c World to camera matrix
     * \return True if this track operation succeded
     */
    bool track_3d(const ScreenCoordinate& newDetection, const WorldToCameraMatrix& w2c) noexcept;

    /**
     * \brief track this point coordinates using a new detection, with no depth infos
     * \param[in] newDetection The new 2D detection
     * \param[in] w2c World to camera matrix
     * \return True if this track operation succeded
     */
    bool track_2d(const ScreenCoordinate2D& newDetection, const WorldToCameraMatrix& w2c) noexcept;

    [[nodiscard]] bool is_moving() const noexcept { return _isMoving; }

  private:
    /**
     * \brief Build the caracteristics of the kalman filter
     */
    static void build_kalman_filter() noexcept;

    // shared kalman filter, between all points
    inline static std::unique_ptr<tracking::SharedKalmanFilter<3, 3>> _kalmanFilter = nullptr;
    inline static std::unique_ptr<tracking::ExtendedKalmanFilter<3, 3>> _kalmanFuse3d = nullptr;
    inline static std::unique_ptr<tracking::ExtendedKalmanFilter<3, 2>> _kalmanFuse2d = nullptr;

    bool _isMoving = false;
};

} // namespace rgbd_slam::tracking

#endif