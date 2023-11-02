#ifndef RGBDSLAM_UTILS_POLYGON_UTILS_HPP
#define RGBDSLAM_UTILS_POLYGON_UTILS_HPP

#include "../types.hpp"
#include "coordinates/point_coordinates.hpp"
#include <boost/geometry/geometry.hpp>
#include <opencv2/core/mat.hpp>

namespace rgbd_slam::utils {

class WorldPolygon;
class CameraPolygon;

/**
 * \brief Describe a polygon by it's points in the polygon space
 */
class Polygon
{
  public:
    using point_2d = boost::geometry::model::d2::point_xy<double>;
    using polygon = boost::geometry::model::polygon<point_2d>;
    using multi_polygon = boost::geometry::model::multi_polygon<polygon>;
    using box_2d = boost::geometry::model::box<point_2d>;

    Polygon() = default;
    /**
     * \brief Build constructor
     * \param[in] points The points to put in the polygon. They will be projected to the polygon space
     * \param[in] normal The normal of the plane that contains those points
     * \param[in] center The center of the plane that contains those points
     */
    Polygon(const std::vector<vector3>& points, const vector3& normal, const vector3& center);

    /**
     * \brief transform constructor
     * \param[in] otherPolygon The polygon to project to a new space
     * \param[in] normal The normal of the plane that contains those points
     * \param[in] center The center of the plane that contains those points
     */
    Polygon(const Polygon& otherPolygon, const vector3& normal, const vector3& center);

    /**
     * \brief copy constructor for the project functions
     */
    Polygon(const std::vector<point_2d>& boundaryPoints,
            const vector3& xAxis,
            const vector3& yAxis,
            const vector3& center);

    /**
     * \brief Compute the convex hull for a set of points
     * \param[in] points The points to compute a convex hull for
     * \return The corresponding polygon. Can be empty if the process failed
     */
    [[nodiscard]] static polygon compute_convex_hull(const std::vector<vector2>& points) noexcept;

    /**
     * \brief Compute the concave hull for a set of points
     * \param[in] points The points to compute a concave hull for
     * \return The corresponding polygon. Can be empty if the process failed
     */
    [[nodiscard]] static polygon compute_concave_hull(const std::vector<vector2>& points) noexcept;

    /**
     * \brief Return true if this point is in the polygon boundaries
     * \param[in] point The point to test, in the polygon space
     * \return true if the point is inside the polygon
     */
    [[nodiscard]] bool contains(const vector2& point) const noexcept;

    /**
     * \brief Return true if this polygon is valid
     */
    [[nodiscard]] bool is_valid() const noexcept { return boost::geometry::is_valid(_polygon); }
    /**
     * \brief Return true if this polygon is valid
     * \param[out] reason The failure reason
     */
    [[nodiscard]] bool is_valid(std::string& reason) const noexcept
    {
        return boost::geometry::is_valid(_polygon, reason);
    }

    /**
     * \brief get number of points in boundary
     */
    [[nodiscard]] size_t boundary_length() const noexcept
    {
        size_t boundarylength = _polygon.outer().size();
        return boundarylength == 0 ? 0 : boundarylength - 1;
    };

    /**
     * \brief Compute this polygon area
     * \return the area of the polygon, or 0.0 if not set
     */
    [[nodiscard]] double area() const noexcept;
    [[nodiscard]] double get_area() const noexcept { return _area; };

    /**
     * \brief merge the other polygon into this one using the union of both polygons
     */
    void merge_union(const Polygon& other);

    /**
     * \brief project this polygon on the next polygon plane.
     * This projection may be reduced to a line or a point.
     * \param[in] nextNormal The normal to project to
     * \param[in] nextCenter The center to project to
     * \return A polygon projected to the new space
     */
    [[nodiscard]] Polygon project(const vector3& nextNormal, const vector3& nextCenter) const;

    /**
     * \brief project this polygon on the next polygon plane.
     * This projection may be reduced to a line or a point.
     * \param[in] nextXAxis The x axis to project to
     * \param[in] nextYAxis The y axis to project to
     * \param[in] nextCenter The center to project to
     * \return A polygon projected to the new space
     */
    [[nodiscard]] Polygon project(const vector3& nextXAxis, const vector3& nextYAxis, const vector3& nextCenter) const;

    /**
     * \brief Transform a polygon to a new space, keeping the same polygon shape, area and shuch.
     * \param[in] nextNormal The normal of polygon in the target space
     * \param[in] nextCenter The center of the polygon in th target space
     * \return A new polygon in the target space
     */
    [[nodiscard]] Polygon transform(const vector3& nextNormal, const vector3& nextCenter) const;

    /**
     * \brief Transform a polygon to a new space, keeping the same polygon shape, area and shuch.
     * \param[in] nextXAxis The x axis of polygon in the target space
     * \param[in] nextYAxis The y axis of polygon in the target space
     * \param[in] nextCenter The center of the polygon in th target space
     * \return A new polygon in the target space
     */
    [[nodiscard]] Polygon transform(const vector3& nextXAxis,
                                    const vector3& nextYAxis,
                                    const vector3& nextCenter) const;

    /**
     * \brief Compute the inter/over of the two polygons
     * \return Inter of Union of the two polygons, or 0 if they do not overlap
     */
    [[nodiscard]] double inter_over_union(const Polygon& other) const;

    /**
     * \brief compute the area of the intersection of two polygons
     * \return the inter polygon area, or 0 if they do not overlap
     */
    [[nodiscard]] double inter_area(const Polygon& other) const;

    /**
     * \brief compute the area of the union of two polygons
     * \return the inter polygon area, or 0 if they do not overlap
     */
    [[nodiscard]] double union_area(const Polygon& other) const;

    /**
     * \brief Compute the union of this polygon and another one.
     * \return The union if it exists, or an empty polygon
     */
    [[nodiscard]] polygon union_one(const Polygon& other) const;

    /**
     * \brief Compute the inter of this polygon and another one.
     * \return The inter if it exists, or an empty polygon
     */
    [[nodiscard]] polygon inter_one(const Polygon& other) const;

    /**
     * \brief Simplify the boundary of the current polygon
     * \param[in] distanceThreshold max lateral distance between points to simplify (mm)
     */
    void simplify(const double distanceThreshold = 10) noexcept;

    /**
     * \brief compute and return the polygon boundary, in the unprojected space
     */
    [[nodiscard]] std::vector<vector3> get_unprojected_boundary() const;

    [[nodiscard]] vector3 get_center() const noexcept { return _center; };
    [[nodiscard]] vector3 get_x_axis() const noexcept { return _xAxis; };
    [[nodiscard]] vector3 get_y_axis() const noexcept { return _yAxis; };
    [[nodiscard]] vector3 get_normal() const noexcept { return _xAxis.cross(_yAxis); };

  protected:
    polygon _polygon;

    vector3 _center;
    vector3 _xAxis;
    vector3 _yAxis;

    double _area; // this polygon area: computation savings

    /**
     * \brief Transform boundary points to the new space
     * \param[in] transformationMatrix The matrix to transform between the spaces
     * \param[in] nextXAxis The x axis of polygon in the target space
     * \param[in] nextYAxis The y axis of polygon in the target space
     * \param[in] nextCenter The center of the polygon in th target space
     * \return The boundary in the target space
     */
    [[nodiscard]] std::vector<point_2d> transform_boundary(const matrix44& transformationMatrix,
                                                           const vector3& nextXAxis,
                                                           const vector3& nextYAxis,
                                                           const vector3& nextCenter) const;
};

/**
 * \brief Describe a polygon in camera coordinates
 */
class CameraPolygon : public Polygon
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
class WorldPolygon : public Polygon
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

} // namespace rgbd_slam::utils

#endif