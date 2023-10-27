#include "concave_fitting.hpp"

#include <cfloat>
#include <iterator>
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cassert>
#include <unordered_map>
#include <cstdint>

#include <flann.hpp>

#if defined USE_OPENMP
#    if !defined _OPENMP
#        pragma message("You've chosen to want OpenMP usage but have not made it a compilation option. Compile with /openmp")
#    endif
#endif

namespace polygon {

using std::uint64_t;

static const size_t stride = 24; // size in bytes of x, y, id

// Algorithm-specific
auto NearestNeighboursFlann(flann::Index<flann::L2<double>>& index, const Point& p, size_t k) -> PointValueVector;
auto ConcaveHull(PointVector& dataset, size_t k, bool iterate) -> PointVector;
auto ConcaveHull(PointVector& dataset, size_t k, PointVector& hull) -> bool;
auto SortByAngle(PointValueVector& values, const Point& p, double prevAngle) -> PointVector;
auto AddPoint(PointVector& points, const Point& p) -> void;

// General maths
auto PointsEqual(const Point& a, const Point& b) -> bool;
auto Angle(const Point& a, const Point& b) -> double;
auto NormaliseAngle(double radians) -> double;
auto PointInPolygon(const Point& p, const PointVector& list) -> bool;
auto Intersects(const LineSegment& a, const LineSegment& b) -> bool;

// Point list utilities
auto FindMinYPoint(const PointVector& points) -> Point;
auto RemoveDuplicates(PointVector& points) -> void;
auto IdentifyPoints(PointVector& points) -> void;
auto RemoveHull(PointVector& points, const PointVector& hull) -> PointVector::iterator;
auto MultiplePointInPolygon(PointVector::iterator begin, PointVector::iterator end, const PointVector& hull) -> bool;

// Unit tests
auto TestAngle() -> void;
auto TestIntersects() -> void;
auto TestSplit() -> void;

// Unit tests
void TestAngle();
void TestIntersects();
void TestSplit();

bool compute_concave_hull(const PointVector& points, PointVector& hull, const uint8_t maxIterations)
{
    PointVector dataset = points;
    RemoveDuplicates(dataset);

    return compute_concave_hull(dataset, hull, maxIterations);
}

bool compute_concave_hull(PointVector& points, PointVector& hull, const uint8_t maxIterations)
{
    // use prime numbers because why not (success of fitting happens at a mean of 3.7 neighbors)
    static const std::vector<uint> possibleNeigborsValues = {3,  5,  7,  11, 13, 17, 21, 23, 29, 31, 37, 41,
                                                             43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    assert(points.size() >= 3);

    const uint8_t trueMaxIteration = std::min(maxIterations, (uint8_t)possibleNeigborsValues.size());

    uint nearestNeigbors = possibleNeigborsValues[0];
    for (uint8_t iteration = 0; iteration < trueMaxIteration; ++iteration)
    {
        hull.clear();
        if (ConcaveHull(points, nearestNeigbors, hull))
            return true;

        nearestNeigbors = possibleNeigborsValues[iteration];
        if (nearestNeigbors > points.size())
            break;
    }
    return false;
}

// The main algorithm from the Moreira-Santos paper.
auto ConcaveHull(PointVector& pointList, size_t k, PointVector& hull) -> bool
{
    hull.clear();

    if (pointList.size() < 3)
    {
        return true;
    }
    if (pointList.size() == 3)
    {
        hull = pointList;
        return true;
    }

    // construct a randomized kd-tree index using 4 kd-trees
    // 2 columns, but stride = 24 bytes in width (x, y, ignoring id)
    flann::Matrix<double> matrix(&(pointList.front().x), pointList.size(), 2, stride);
    flann::Index<flann::L2<double>> flannIndex(matrix, flann::KDTreeIndexParams(4));
    flannIndex.buildIndex();

    // Initialise hull with the min-y point
    Point firstPoint = FindMinYPoint(pointList);
    AddPoint(hull, firstPoint);

    // Until the hull is of size > 3 we want to ignore the first point from nearest neighbour searches
    Point currentPoint = firstPoint;
    flannIndex.removePoint(firstPoint.id);

    double prevAngle = 0.0;
    int step = 1;

    // Iterate until we reach the start, or until there's no points left to process
    while ((!PointsEqual(currentPoint, firstPoint) || step == 1) && hull.size() != pointList.size())
    {
        if (step == 4)
        {
            // Put back the first point into the dataset and into the flann index
            firstPoint.id = pointList.size();
            flann::Matrix<double> firstPointMatrix(&firstPoint.x, 1, 2, stride);
            flannIndex.addPoints(firstPointMatrix);
        }

        PointValueVector kNearestNeighbours = NearestNeighboursFlann(flannIndex, currentPoint, k);
        PointVector cPoints = SortByAngle(kNearestNeighbours, currentPoint, prevAngle);

        bool its = true;
        size_t i = 0;

        while (its && i < cPoints.size())
        {
            i++;
            size_t lastPoint = 0;
            if (PointsEqual(cPoints[i - 1], firstPoint))
            {
                lastPoint = 1;
            }

            size_t j = 2;
            its = false;

            while (!its && j < hull.size() - lastPoint)
            {
                auto line1 = std::make_pair(hull[step - 1], cPoints[i - 1]);
                auto line2 = std::make_pair(hull[step - j - 1], hull[step - j]);
                its = Intersects(line1, line2);
                j++;
            }
        }

        if (its)
        {
            return false;
        }
        currentPoint = cPoints[i - 1];

        AddPoint(hull, currentPoint);

        prevAngle = Angle(hull[step], hull[step - 1]);

        flannIndex.removePoint(currentPoint.id);

        step++;
    }

    // The original points less the points belonging to the hull need to be fully enclosed by the hull in order to
    // return true.
    PointVector dataset = pointList;
    auto newEnd = RemoveHull(dataset, hull);
    const bool isAllEnclosed = MultiplePointInPolygon(begin(dataset), newEnd, hull);
    return isAllEnclosed;
}

// Compare a and b for equality
auto Equal(double a, double b) -> bool { return fabs(a - b) <= DBL_EPSILON; }

// Compare value to zero
auto Zero(double a) -> bool { return fabs(a) <= DBL_EPSILON; }

// Compare for a < b
auto LessThan(double a, double b) -> bool { return a < (b - DBL_EPSILON); }

// Compare for a <= b
auto LessThanOrEqual(double a, double b) -> bool { return a <= (b + DBL_EPSILON); }

// Compare for a > b
auto GreaterThan(double a, double b) -> bool { return a > (b + DBL_EPSILON); }

// Compare whether two points have the same x and y
auto PointsEqual(const Point& a, const Point& b) -> bool { return Equal(a.x, b.x) && Equal(a.y, b.y); }

// Remove duplicates in a list of points
auto RemoveDuplicates(PointVector& points) -> void
{
    sort(begin(points), end(points), [](const Point& a, const Point& b) {
        if (Equal(a.x, b.x))
            return LessThan(a.y, b.y);
        else
            return LessThan(a.x, b.x);
    });

    auto newEnd = unique(begin(points), end(points), [](const Point& a, const Point& b) {
        return PointsEqual(a, b);
    });

    points.erase(newEnd, end(points));
}

// Uniquely id the points for binary searching
auto IdentifyPoints(PointVector& points) -> void
{
    uint64_t id = 0;

    for (auto itr = begin(points); itr != end(points); ++itr, ++id)
    {
        itr->id = id;
    }
}

// Find the point having the smallest y-value
auto FindMinYPoint(const PointVector& points) -> Point
{
    assert(!points.empty());

    auto itr = min_element(begin(points), end(points), [](const Point& a, const Point& b) {
        if (Equal(a.y, b.y))
            return GreaterThan(a.x, b.x);
        else
            return LessThan(a.y, b.y);
    });

    return *itr;
}

// Lookup by ID and remove a point from a list of points
auto RemovePoint(PointVector& list, const Point& p) -> void
{
    auto itr = std::lower_bound(begin(list), end(list), p, [](const Point& a, const Point& b) {
        return a.id < b.id;
    });

    assert(itr != end(list) && itr->id == p.id);

    if (itr != end(list))
        list.erase(itr);
}

// Add a point to a list of points
auto AddPoint(PointVector& points, const Point& p) -> void { points.push_back(p); }

// Return the k-nearest points in a list of points from the given point p (uses Flann library).
auto NearestNeighboursFlann(flann::Index<flann::L2<double>>& index, const Point& p, size_t k) -> PointValueVector
{
    std::vector<int> vIndices(k);
    std::vector<double> vDists(k);
    double test[] = {p.x, p.y};

    flann::Matrix<double> query(test, 1, 2);
    flann::Matrix<int> mIndices(vIndices.data(), 1, static_cast<int>(vIndices.size()));
    flann::Matrix<double> mDists(vDists.data(), 1, static_cast<int>(vDists.size()));

    int count_ = index.knnSearch(query, mIndices, mDists, k, flann::SearchParams(128));
    size_t count = static_cast<size_t>(count_);

    PointValueVector result(count);

    for (size_t i = 0; i < count; ++i)
    {
        int id = vIndices[i];
        const double* point = index.getPoint(id);
        result[i].point.x = point[0];
        result[i].point.y = point[1];
        result[i].point.id = id;
        result[i].distance = vDists[i];
    }

    return result;
}

// Returns a list of points sorted in descending order of clockwise angle
auto SortByAngle(PointValueVector& values, const Point& from, double prevAngle) -> PointVector
{
    for_each(begin(values), end(values), [from, prevAngle](PointValue& to) {
        to.angle = NormaliseAngle(Angle(from, to.point) - prevAngle);
    });

    sort(begin(values), end(values), [](const PointValue& a, const PointValue& b) {
        return GreaterThan(a.angle, b.angle);
    });

    PointVector angled(values.size());

    transform(begin(values), end(values), begin(angled), [](const PointValue& pv) {
        return pv.point;
    });

    return angled;
}

// Get the angle in radians measured clockwise from +'ve x-axis
auto Angle(const Point& a, const Point& b) -> double
{
    double angle = -atan2(b.y - a.y, b.x - a.x);

    return NormaliseAngle(angle);
}

// Return angle in range: 0 <= angle < 2PI
auto NormaliseAngle(double radians) -> double
{
    if (radians < 0.0)
        return radians + M_PI + M_PI;
    else
        return radians;
}

// Return the new logical end after removing points from dataset having ids belonging to hull
auto RemoveHull(PointVector& points, const PointVector& hull) -> PointVector::iterator
{
    std::vector<uint64_t> ids(hull.size());

    transform(begin(hull), end(hull), begin(ids), [](const Point& p) {
        return p.id;
    });

    sort(begin(ids), end(ids));

    return remove_if(begin(points), end(points), [&ids](const Point& p) {
        return binary_search(begin(ids), end(ids), p.id);
    });
}

// Uses OpenMP to determine whether a condition exists in the specified range of elements.
// https://msdn.microsoft.com/en-us/library/ff521445.aspx
template<class InIt, class Predicate> bool omp_parallel_any_of(InIt first, InIt last, const Predicate& pr)
{
    typedef typename std::iterator_traits<InIt>::value_type item_type;

    // A flag that indicates that the condition exists.
    bool found = false;

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(last - first); ++i)
    {
        if (!found)
        {
            item_type& cur = *(first + i);

            // If the element satisfies the condition, set the flag to cancel the operation.
            if (pr(cur))
            {
                found = true;
            }
        }
    }

    return found;
}

// Check whether all points in a begin/end range are inside hull.
auto MultiplePointInPolygon(PointVector::iterator begin, PointVector::iterator end, const PointVector& hull) -> bool
{
    if (begin == end)
    {
        // if begin == end, all points should be in hull
        return true;
    }
    auto test = [&hull](const Point& p) {
        return !PointInPolygon(p, hull);
    };

    bool anyOutside = true;

#if defined USE_OPENMP

    anyOutside = omp_parallel_any_of(begin, end, test); // multi-threaded

#else

    anyOutside = std::any_of(begin, end, test); // single-threaded

#endif

    return !anyOutside;
}

// Point-in-polygon test
auto PointInPolygon(const Point& p, const PointVector& list) -> bool
{
    if (list.size() <= 2)
        return false;

    const double x = p.x;
    const double y = p.y;

    int inout = 0;
    auto v0 = list.begin();
    auto v1 = v0 + 1;

    while (v1 != list.end())
    {
        if (((LessThanOrEqual(v0->y, y) and LessThan(y, v1->y)) or
             (LessThanOrEqual(v1->y, y) and LessThan(y, v0->y))) and
            !Zero(v1->y - v0->y) and LessThan(x, v0->x + ((v1->x - v0->x) * (y - v0->y) / (v1->y - v0->y))))
        {
            inout++;
        }
        v0 = v1;
        v1++;
    }

    if (inout == 0)
        return true;         // no iteration, point is good
    else if (inout % 2 == 0) // par: point is outside of polygon
        return false;

    // point is in the polygon
    return true;
}

// Test whether two line segments intersect each other
auto Intersects(const LineSegment& a, const LineSegment& b) -> bool
{
    // https://www.topcoder.com/community/data-science/data-science-tutorials/geometry-concepts-line-intersection-and-its-applications/

    const double& ax1 = a.first.x;
    const double& ay1 = a.first.y;
    const double& ax2 = a.second.x;
    const double& ay2 = a.second.y;
    const double& bx1 = b.first.x;
    const double& by1 = b.first.y;
    const double& bx2 = b.second.x;
    const double& by2 = b.second.y;

    double a1 = ay2 - ay1;
    double b1 = ax1 - ax2;
    double c1 = a1 * ax1 + b1 * ay1;
    double a2 = by2 - by1;
    double b2 = bx1 - bx2;
    double c2 = a2 * bx1 + b2 * by1;
    double det = a1 * b2 - a2 * b1;

    if (Zero(det))
    {
        return false;
    }
    else
    {
        double x = (b2 * c1 - b1 * c2) / det;
        double y = (a1 * c2 - a2 * c1) / det;

        bool on_both = true;
        on_both = on_both && LessThanOrEqual(std::min(ax1, ax2), x) && LessThanOrEqual(x, std::max(ax1, ax2));
        on_both = on_both && LessThanOrEqual(std::min(ay1, ay2), y) && LessThanOrEqual(y, std::max(ay1, ay2));
        on_both = on_both && LessThanOrEqual(std::min(bx1, bx2), x) && LessThanOrEqual(x, std::max(bx1, bx2));
        on_both = on_both && LessThanOrEqual(std::min(by1, by2), y) && LessThanOrEqual(y, std::max(by1, by2));
        return on_both;
    }
}

// Unit test of Angle() function
auto TestAngle() -> void
{
    auto ToDegrees = [](double radians) {
        return radians * 180.0 / M_PI;
    };

    auto Test = [&](const Point& p, double expected) {
        double actual = ToDegrees(Angle({0.0, 0.0}, p));
        assert(Equal(actual, expected));
    };

    double value = ToDegrees(atan(3.0 / 4.0));

    Test({5.0, 0.0}, 0.0);
    Test({4.0, 3.0}, 360.0 - value);
    Test({3.0, 4.0}, 270.0 + value);
    Test({0.0, 5.0}, 270.0);
    Test({-3.0, 4.0}, 270.0 - value);
    Test({-4.0, 3.0}, 180.0 + value);
    Test({-5.0, 0.0}, 180.0);
    Test({-4.0, -3.0}, 180.0 - value);
    Test({-3.0, -4.0}, 90.0 + value);
    Test({0.0, -5.0}, 90.0);
    Test({3.0, -4.0}, 90.0 - value);
    Test({4.0, -3.0}, 0.0 + value);
}

// Unit test the Intersects() function
auto TestIntersects() -> void
{
    using std::make_pair;

    std::unordered_map<char, Point> values;
    values['A'] = {0.0, 0.0};
    values['B'] = {-1.5, 3.0};
    values['C'] = {2.0, 2.0};
    values['D'] = {-2.0, 1.0};
    values['E'] = {-2.5, 5.0};
    values['F'] = {-1.5, 7.0};
    values['G'] = {1.0, 9.0};
    values['H'] = {-4.0, 7.0};
    values['I'] = {3.0, 10.0};
    values['J'] = {2.0, 11.0};
    values['K'] = {-1.0, 11.0};
    values['L'] = {-3.0, 11.0};
    values['M'] = {-5.0, 9.5};
    values['N'] = {-6.0, 7.5};
    values['O'] = {-6.0, 4.0};
    values['P'] = {-5.0, 2.0};

    auto Test = [&values](const char a1, const char a2, const char b1, const char b2, bool expected) {
        assert(Intersects(make_pair(values[a1], values[a2]), make_pair(values[b1], values[b2])) == expected);
        assert(Intersects(make_pair(values[a2], values[a1]), make_pair(values[b1], values[b2])) == expected);
        assert(Intersects(make_pair(values[a1], values[a2]), make_pair(values[b2], values[b1])) == expected);
        assert(Intersects(make_pair(values[a2], values[a1]), make_pair(values[b2], values[b1])) == expected);
    };

    Test('B', 'D', 'A', 'C', false);
    Test('A', 'B', 'C', 'D', true);
    Test('L', 'K', 'H', 'F', false);
    Test('E', 'C', 'F', 'B', true);
    Test('P', 'C', 'E', 'B', false);
    Test('P', 'C', 'A', 'B', true);
    Test('O', 'E', 'C', 'F', false);
    Test('L', 'C', 'M', 'N', false);
    Test('L', 'C', 'N', 'B', false);
    Test('L', 'C', 'M', 'K', true);
    Test('L', 'C', 'G', 'I', false);
    Test('L', 'C', 'I', 'E', true);
    Test('M', 'O', 'N', 'F', true);
}

} // namespace polygon