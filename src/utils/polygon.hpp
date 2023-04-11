#ifndef RGBDSLAM_UTILS_POLYGON_UTILS_HPP
#define RGBDSLAM_UTILS_POLYGON_UTILS_HPP

#include "../types.hpp"
#include "coordinates.hpp"
#include <boost/geometry/geometry.hpp>
#include <opencv2/core/mat.hpp>

namespace rgbd_slam::utils {

class WorldPolygon;
class CameraPolygon;

/**
 * \brief Compute the two vectors that span the plane
 * \return a pair of vector u and v, normal to the plane normal
 */
std::pair<vector3, vector3> get_plane_coordinate_system(const vector3& normal);

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
    Polygon(const std::vector<vector2>& points, const vector3& center, const vector3& xAxis, const vector3& yAxis);
    Polygon(const std::vector<point_2d>& boundaryPoints,
            const vector3& center,
            const vector3& xAxis,
            const vector3& yAxis);

    /**
     * \brief Compute the convex hull for a set of points
     * \param[in] points The points to compute a convex hull for
     * \return The ordered points defining a convex hull of points
     */
    static std::vector<point_2d> compute_convex_hull(const std::vector<vector2>& points);

    /**
     * \brief Return true if this point is in the polygon boundaries
     */
    bool contains(const vector2& point) const;

    /**
     * \brief Return the boundary as an ordered vector
     */
    std::vector<vector2> get_boundary() const;

    /**
     * \brief Compute the box envelop of this polygon
     */
    std::vector<vector2> get_envelop() const;

    /**
     * \brief Compute this polygon area
     */
    double area() const;

    /**
     * \brief merge the other polygon into this one
     */
    void merge(const Polygon& other);

    Polygon project(const vector3& nextCenter, const vector3& nextXAxis, const vector3& nextYAxis) const;

    /**
     * \brief Compute the inter/over of the two polygons
     */
    double inter_over_union(const Polygon& other) const;

    /**
     * \brief compute the area of the intersection of two polygons
     */
    double inter_area(const Polygon& other) const;

    /**
     * \brief Compute the union of this polygon and another one.
     * \return The union if it exists, or an empty polygon
     */
    polygon union_one(const Polygon& other) const;

    /**
     * \brief Compute the inter of this polygon and another one.
     * \return The inter if it exists, or an empty polygon
     */
    polygon inter_one(const Polygon& other) const;

  protected:
    /**
     * \brief Simplify the boundary of the current polygon
     * \param[in] distanceThreshold max lateral distance between points to simplify (mm)
     */
    void simplify(const double distanceThreshold = 50);

    polygon _polygon;

    vector3 _center;
    vector3 _xAxis;
    vector3 _yAxis;
};

class CameraPolygon : public Polygon
{
  public:
    CameraPolygon() = default;
    CameraPolygon(const std::vector<vector2>& points,
                  const vector3& center,
                  const vector3& xAxis,
                  const vector3& yAxis) :
        Polygon(points, center, xAxis, yAxis) {};
    CameraPolygon(const std::vector<point_2d>& boundaryPoints,
                  const vector3& center,
                  const vector3& xAxis,
                  const vector3& yAxis) :
        Polygon(boundaryPoints, center, xAxis, yAxis) {};

    /**
     * \brief display the polygon in screen space on the given image
     * \param[in] color Color to draw this polygon with
     * \param[out] debugImage Image on which the polygon will be displayed
     */
    void display(const cv::Scalar& color, cv::Mat& debugImage) const;

    WorldPolygon project(const CameraToWorldMatrix& c2w) const;

    std::vector<ScreenCoordinate> get_screen_points() const;
};

class WorldPolygon : public Polygon
{
  public:
    WorldPolygon() = default;
    WorldPolygon(const std::vector<vector2>& points,
                 const vector3& center,
                 const vector3& xAxis,
                 const vector3& yAxis) :
        Polygon(points, center, xAxis, yAxis) {};
    WorldPolygon(const std::vector<point_2d>& boundaryPoints,
                 const vector3& center,
                 const vector3& xAxis,
                 const vector3& yAxis) :
        Polygon(boundaryPoints, center, xAxis, yAxis) {};

    CameraPolygon project(const WorldToCameraMatrix& w2c) const;

    void merge(const WorldPolygon& other);
};

} // namespace rgbd_slam::utils

#endif