/**
 * The Moreira-Santos algorithm
 * Implemented by acraig5075 on GitHub
 */

#ifndef CONCAVE_POLYGON_FITTING_HPP
#define CONCAVE_POLYGON_FITTING_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace polygon {

struct Point
{
    double x = 0.0;
    double y = 0.0;
    std::uint64_t id = 0;

    Point() = default;

    Point(double x, double y) : x(x), y(y) {}
};

struct PointValue
{
    Point point;
    double distance = 0.0;
    double angle = 0.0;
};

using PointVector = std::vector<Point>;
using PointValueVector = std::vector<PointValue>;
using LineSegment = std::pair<Point, Point>;

/**
 * \brief Compute the concave hull of a set of points
 * \param[in] dataset
 * \param[in] k Nearest neigtbors to consider (>= 1)
 * \param[in] iterate If set to true, will perform some iterqtions to find the best fitting polygon
 */
PointVector ConcaveHull(PointVector& dataset, size_t k = 4, const bool iterate = false);

} // namespace polygon

#endif