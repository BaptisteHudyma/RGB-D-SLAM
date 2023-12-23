#ifndef RGBDSLAM_TRACKING_POINT_WITH_TRACKING_HPP
#define RGBDSLAM_TRACKING_POINT_WITH_TRACKING_HPP

#include "types.hpp"
#include "coordinates/point_coordinates.hpp"
#include "kalman_filter.hpp"

#include <opencv2/opencv.hpp>

namespace rgbd_slam::tracking {

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
     * \brief update this point coordinates using a new detection
     * \param[in] newDetectionCoordinates The newly detected point
     * \param[in] newDetectionCovariance The newly detected point covariance
     * \return The distance between the updated position ans the previous one, -1 if an error occured
     */
    double track(const WorldCoordinate& newDetectionCoordinates, const matrix33& newDetectionCovariance) noexcept;

  private:
    /**
     * \brief Build the caracteristics of the kalman filter
     */
    static void build_kalman_filter() noexcept;

    // shared kalman filter, between all points
    inline static std::unique_ptr<tracking::SharedKalmanFilter<3, 3>> _kalmanFilter = nullptr;
};

} // namespace rgbd_slam::tracking

#endif