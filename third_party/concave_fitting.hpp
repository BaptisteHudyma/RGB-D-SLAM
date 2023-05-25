/**
 * The Moreira-Santos algorithm
 * From: "Concave hull: A k-nearest neighbours approach for the computation of the region occupied by a set of points."
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
 * \param[in] dataset The points to fit a concave hull to. Wont be modified, but an internal copy will be created
 * \param[out] hull The ordered boundary of the concave polygon
 * \param[in] maxIterations Maximum iterations of the algorithm
 * \return True if a correct boundary was computed
 */
bool compute_concave_hull(const PointVector& dataset, PointVector& hull, const uint8_t maxIterations = 1);
/**
 * \brief Compute the concave hull of a set of points
 * \param[in, out] dataset The points to fit a concave hull to. Will be modified
 * \param[out] hull The ordered boundary of the concave polygon
 * \param[in] maxIterations Maximum iterations of the algorithm
 * \return True if a correct boundary was computed
 */
bool compute_concave_hull(PointVector& dataset, PointVector& hull, const uint8_t maxIterations = 1);

} // namespace polygon

#endif