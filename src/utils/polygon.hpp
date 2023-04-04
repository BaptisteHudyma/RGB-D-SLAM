#ifndef RGBDSLAM_UTILS_POLYGON_UTILS_HPP
#define RGBDSLAM_UTILS_POLYGON_UTILS_HPP

#include "../types.hpp"

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
    Polygon() = default;
    Polygon(const std::vector<vector2>& points, const vector2& lowerLeftBoundary, const vector2& upperRightBoundary);

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

    size_t get_number_of_points() const { return _boundaryPoints.size(); };
    std::vector<vector2> get_boundary_points() const { return _boundaryPoints; };

  private:
    std::vector<vector2> _boundaryPoints;

    // represent the two sides of the rectangle englobing this polygon
    vector2 _lowerLeftBoundary = vector2::Zero();
    vector2 _upperRightBoundary = vector2::Zero();
};

} // namespace rgbd_slam::utils

#endif