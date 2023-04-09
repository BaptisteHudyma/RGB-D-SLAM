#ifndef RGBDSLAM_UTILS_POLYGON_UTILS_HPP
#define RGBDSLAM_UTILS_POLYGON_UTILS_HPP

#include "../types.hpp"
#include <boost/geometry/geometry.hpp>
#include <opencv2/core/mat.hpp>

namespace rgbd_slam::utils {

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
    Polygon(const std::vector<vector2>& points);
    Polygon(const std::vector<point_2d>& boundaryPoints);

    /**
     * \brief Compute the convex hull for a set of points
     * \param[in] points The points to compute a convex hull for
     * \return The ordered points defining a convex hull of points
     */
    static Polygon compute_convex_hull(const std::vector<vector2>& points);

    /**
     * \brief Return true if this point is in the polygon boundaries
     */
    bool contains(const vector2& point) const;

    /**
     * merge the other polygon into this one
     */
    void merge(const Polygon& other);

    /**
     * \brief Project a polygon to another space
     * \param[in] currentCenter Center of this polygon
     * \param[in] currentUVec Vector to project the points in the polygon
     * \param[in] currentVVec Vector to project the points in the polygon
     * \param[in] nextCenter Next center of this polygon
     * \param[in] nextUVec Next vector to project the points in the polygon
     * \param[in] nextVVec Next vector to project the points in the polygon
     * \return a new polygon, with it's projection in the new space
     */
    Polygon project(const vector3& currentCenter,
                    const vector3& currentUVec,
                    const vector3& currentVVec,
                    const vector3& nextCenter,
                    const vector3& nextUVec,
                    const vector3& nextVVec) const;

    /**
     * \brief display the polygon in screen space on the given image
     * \param[in] center Center of this polygon, in world space
     * \param[in] uVec Vector to project the points out of the polygon space
     * \param[in] vVec Vector to project the points out of the polygon space
     * \param[in] color Color to draw this polygon with
     * \param[out] debugImage Image on which the polygon will be displayed
     */
    void display(const vector3& center,
                 const vector3& uVec,
                 const vector3& vVec,
                 const cv::Scalar& color,
                 cv::Mat& debugImage) const;

  protected:
    /**
     * \brief Simplify the boundary of the current polygon
     * \param[in] distanceThreshold max lateral distance between points to simplify (mm)
     */
    void simplify(const double distanceThreshold = 50);

  private:
    polygon _polygon;
};

} // namespace rgbd_slam::utils

#endif