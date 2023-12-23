#ifndef RGBDSLAM_COORD_POLYGON_HPP
#define RGBDSLAM_COORD_POLYGON_HPP

#include "../types.hpp"
#include "coordinates/point_coordinates.hpp"
#include <boost/geometry/geometry.hpp>
#include <opencv2/core/mat.hpp>

#include "utils/polygon.hpp"

namespace rgbd_slam {

class WorldPolygon;
class CameraPolygon;

/**
 * \brief Describe a polygon in camera coordinates
 */
class CameraPolygon : public utils::Polygon
{
  public:
    using Polygon::Polygon;
    CameraPolygon(const Polygon& other) : Polygon(other) {};

    /**
     * \brief display the polygon in screen space on the given image
     * \param[in] color Color to draw this polygon with
     * \param[in, out] debugImage Image on which the polygon will be displayed
     */
    void display(const cv::Scalar& color, cv::Mat& debugImage) const noexcept;

    /**
     * \brief Project this polygon to the world space
     * \param[in] cameraToWorld Matrix to go from camera to world space
     * \return The polygon in world coordinates
     */
    [[nodiscard]] WorldPolygon to_world_space(const CameraToWorldMatrix& cameraToWorld) const;

    /**
     * \brief Compute the boundary points in screen coordinates
     */
    [[nodiscard]] std::vector<ScreenCoordinate> get_screen_points() const;

    /**
     * \brief Project this polygon to screen space
     */
    [[nodiscard]] polygon to_screen_space() const;

    /**
     * \brief Check that this polygon is visible in screen space
     * \return true if the polygon is visible from the camera 1
     */
    [[nodiscard]] bool is_visible_in_screen_space() const;
};

/**
 * \brief Describe a polygon in world coordinates
 */
class WorldPolygon : public utils::Polygon
{
  public:
    using Polygon::Polygon;
    WorldPolygon(const Polygon& other) : Polygon(other) {};

    /**
     * \brief project this polygon to camera space
     * \param[in] worldToCamera A matrix to convert from world to camera view
     / \return This polygon in camera space
     */
    [[nodiscard]] CameraPolygon to_camera_space(const WorldToCameraMatrix& worldToCamera) const;

    /**
     * \brief Merge the other polygon into this one
     * \param[in] other The other polygon to merge into this one
     */
    void merge(const WorldPolygon& other);

    /**
     * \brief display the polygon in screen space on the given image
     * \param[in] worldToCamera A matrix tp convert from world to camera space
     * \param[in] color Color to draw this polygon with
     * \param[in, out] debugImage Image on which the polygon will be displayed
     */
    void display(const WorldToCameraMatrix& worldToCamera, const cv::Scalar& color, cv::Mat& debugImage) const noexcept;
};

} // namespace rgbd_slam
#endif