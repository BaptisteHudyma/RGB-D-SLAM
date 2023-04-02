#include "polygon.hpp"

namespace rgbd_slam::utils {

std::vector<vector2> get_best_fitting_polygon(const std::vector<vector2>& points)
{
    const uint pointCount = points.size();

    std::vector<uint> link(pointCount, 0); // link points indexes [point index] -> next point index

    // first half of points
    for (uint i = 0; i < pointCount - 1; ++i)
    {
        const vector2& pointA = points[i];
        double closestDist1 = 1e10;
        double closestDist2 = 1e10;
        uint closestPoint1 = 0;
        uint closestPoint2 = 0;
        // second half of points
        for (uint j = i + 1; j < pointCount; j++)
        {
            const vector2& pointB = points[j];
            const double dist = (pointB - pointA).lpNorm<1>();
            if (dist < closestDist1)
            {
                closestDist2 = closestDist1;
                closestPoint2 = closestPoint1;

                closestPoint1 = j;
                closestDist1 = dist;
            }
            else if (dist < closestDist2)
            {
                closestPoint2 = j;
                closestDist2 = dist;
            }
        }

        // found the two closest points, link them
        if (closestDist1 < 1e5)
            link[i] = closestPoint1;
        if (closestDist2 < 1e5)
            link[closestPoint2] = i;
    }

    std::vector<vector2> orderedPoints;
    orderedPoints.reserve(pointCount);
    uint highestIndex = 0;
    for (uint i = 0, currentIndex = 0; i < pointCount; ++i)
    {
        orderedPoints.emplace_back(points[currentIndex]);
        currentIndex = link[currentIndex];

        // looped arround the polygon: maybe some points were not used
        if (currentIndex == 0 or currentIndex == highestIndex)
        {
            /*outputs::log_warning("Polygon fit used only " + std::to_string(orderedPoints.size()) +
                                 " on the available " + std::to_string(pointCount));*/
            break;
        }
        highestIndex = std::max(highestIndex, currentIndex);
    }

    return orderedPoints;
}

} // namespace rgbd_slam::utils
