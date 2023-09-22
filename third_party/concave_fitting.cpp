#include "concave_fitting.hpp"

#include <cfloat>
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
#        pragma message( \
                "You've chosen to want OpenMP usage but have not made it a compilation option. Compile with /openmp")
#    endif
#endif

namespace polygon {

using std::uint64_t;

static const size_t stride = 24; // size in bytes of x, y, id

// Floating point comparisons
bool Equal(double a, double b);
bool Zero(double a);
bool LessThan(double a, double b);
bool LessThanOrEqual(double a, double b);
bool GreaterThan(double a, double b);

// Algorithm-specific
PointValueVector NearestNeighboursFlann(flann::Index<flann::L2<double>>& index, const Point& p, size_t k);
PointVector ConcaveHull(PointVector& dataset, size_t k, bool iterate);
bool ConcaveHull(PointVector& dataset, size_t k, PointVector& hull);
PointVector SortByAngle(PointValueVector& values, const Point& p, double prevAngle);
void AddPoint(PointVector& points, const Point& p);

// General maths
bool PointsEqual(const Point& a, const Point& b);
double Angle(const Point& a, const Point& b);
double NormaliseAngle(double radians);
bool PointInPolygon(const Point& p, const PointVector& list);
bool Intersects(const LineSegment& a, const LineSegment& b);

// Point list utilities
Point FindMinYPoint(const PointVector& points);
void RemoveDuplicates(PointVector& points);
void IdentifyPoints(PointVector& points);
PointVector::iterator RemoveHull(PointVector& points, const PointVector& hull);
bool MultiplePointInPolygon(PointVector::const_iterator begin,
                            PointVector::const_iterator end,
                            const PointVector& hull);

// Unit tests
void TestAngle();
void TestIntersects();
void TestSplit();

PointVector ConcaveHull(PointVector& dataset, const size_t k, const uint8_t maxIterations)
{
    assert(k > 0);
    assert(dataset.size() >= 3);

    size_t nearestNeigbors = k;
    uint8_t iteration = 1;
    while (nearestNeigbors < dataset.size())
    {
        PointVector hull;
        if (ConcaveHull(dataset, nearestNeigbors, hull) || iteration >= maxIterations)
        {
            return hull;
        }
        ++nearestNeigbors;
        ++iteration;
    }

    return {};
}

/**
 * \brief The main algorithm from the Moreira-Santos paper.
 * \param[in, out] pointList the set of points to check
 * \param[in] k the number of nearest neigbors to consider
 * \param[out] hull The concave hull found by the algorithm
 */
bool ConcaveHull(PointVector& pointList, size_t k, PointVector& hull)
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
            size_t lastPoint = 0;
            if (PointsEqual(cPoints[i], firstPoint))
                lastPoint = 1;

            size_t j = 2;
            its = false;

            while (!its && j < hull.size() - lastPoint)
            {
                const LineSegment& line1 = std::make_pair(hull[step - 1], cPoints[i]);
                const LineSegment& line2 = std::make_pair(hull[step - j - 1], hull[step - j]);
                its = Intersects(line1, line2);
                j++;
            }

            if (its)
                i++;
        }

        if (its)
            return false;

        currentPoint = cPoints[i];

        AddPoint(hull, currentPoint);

        prevAngle = Angle(hull[step], hull[step - 1]);

        flannIndex.removePoint(currentPoint.id);

        step++;
    }

    // The original points less the points belonging to the hull need to be fully enclosed by the hull in order to
    // return true.
    PointVector dataset = pointList;
    const PointVector::iterator newEnd = RemoveHull(dataset, hull);
    const bool allEnclosed = MultiplePointInPolygon(dataset.begin(), newEnd, hull);

    return allEnclosed;
}

// Compare a and b for equality
bool Equal(double a, double b) { return fabs(a - b) <= DBL_EPSILON; }

// Compare value to zero
bool Zero(double a) { return fabs(a) <= DBL_EPSILON; }

// Compare for a < b
bool LessThan(double a, double b) { return a < (b - DBL_EPSILON); }

// Compare for a <= b
bool LessThanOrEqual(double a, double b) { return a <= (b + DBL_EPSILON); }

// Compare for a > b
bool GreaterThan(double a, double b) { return a > (b + DBL_EPSILON); }

// Compare whether two points have the same x and y
bool PointsEqual(const Point& a, const Point& b) { return Equal(a.x, b.x) && Equal(a.y, b.y); }

/**
 * \brief Remove duplicates in a list of point
 * \param[in, out] points
 */
void RemoveDuplicates(PointVector& points)
{
    std::ranges::sort(points, [](const Point& a, const Point& b) {
        if (Equal(a.x, b.x))
            return LessThan(a.y, b.y);
        else
            return LessThan(a.x, b.x);
    });

    const PointVector::const_iterator newEnd = std::ranges::unique(points, [](const Point& a, const Point& b) {
                                                   return PointsEqual(a, b);
                                               }).end();

    points.erase(newEnd, points.end());
}

/**
 * \brief Uniquely id the points for binary searching
 * \param[in, out] points
 */
void IdentifyPoints(PointVector& points)
{
    uint64_t id = 0;

    for (PointVector::iterator itr = points.begin(); itr != points.end(); ++itr, ++id)
    {
        itr->id = id;
    }
}

/**
 * \brief Find the point having the smallest y-value
 * \param[in] points
 * \return The point with the smallest y
 */
Point FindMinYPoint(const PointVector& points)
{
    assert(!points.empty());

    const PointVector::const_iterator itr = std::ranges::min_element(points, [](const Point& a, const Point& b) {
        if (Equal(a.y, b.y))
            return GreaterThan(a.x, b.x);
        else
            return LessThan(a.y, b.y);
    });

    return *itr;
}

/**
 * \brief Lookup by ID and remove a point from a list of points
 * \param[in, out] list
 * \param[in] p The point to remove
 */
void RemovePoint(PointVector& list, const Point& p)
{
    const PointVector::const_iterator itr = std::ranges::lower_bound(list, p, [](const Point& a, const Point& b) {
        return a.id < b.id;
    });

    assert(itr != list.end() && itr->id == p.id);

    if (itr != list.end())
        list.erase(itr);
}

/**
 * \brief Add a point to a list of points
 * \param[in, out] points
 * \param[in] p The point to add
 */
void AddPoint(PointVector& points, const Point& p) { points.emplace_back(p); }

/**
 * \brief Return the k-nearest points in a list of points from the given point p (uses Flann library).
 * \param[in, out] index
 * \param[in] p The point to check
 * \param[in] k The number of neigbors to check
 * \return The k nearest neigbors of the point
 */
PointValueVector NearestNeighboursFlann(flann::Index<flann::L2<double>>& index, const Point& p, const size_t k)
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

/**
 * \brief Returns a list of points sorted in descending order of clockwise angle
 * \param[in, out] values
 * \param[in] from
 * \param[in] prevAngle
 * \return The points sorted by angles
 */
PointVector SortByAngle(PointValueVector& values, const Point& from, const double prevAngle)
{
    std::ranges::for_each(values, [from, prevAngle](PointValue& to) {
        to.angle = NormaliseAngle(Angle(from, to.point) - prevAngle);
    });

    std::ranges::sort(values, [](const PointValue& a, const PointValue& b) {
        return GreaterThan(a.angle, b.angle);
    });

    PointVector angled(values.size());
    std::ranges::transform(values, angled.begin(), [](const PointValue& pv) {
        return pv.point;
    });

    return angled;
}

/**
 * \brief Get the angle in radians measured clockwise from +'ve x-axis
 * \param[in] a
 * \param[in] b
 * \return The angle between the two points
 */
double Angle(const Point& a, const Point& b)
{
    double angle = -atan2(b.y - a.y, b.x - a.x);

    return NormaliseAngle(angle);
}

/**
 * \brief Return angle in range: 0 <= angle < 2PI
 * \param[in] radians The angle to constraint
 * \return The new angle
 */
double NormaliseAngle(const double radians)
{
    if (radians < 0.0)
        return radians + M_PI + M_PI;
    else
        return radians;
}

/**
 * \brief Return the new logical end after removing points from dataset having ids belonging to hull
 * \param[in, out] points Points to check
 * \param[in] hull The hull to check
 * \return The new end of the points vector
 */
PointVector::iterator RemoveHull(PointVector& points, const PointVector& hull)
{
    std::vector<uint64_t> ids(hull.size());

    std::ranges::transform(hull, ids.begin(), [](const Point& p) {
        return p.id;
    });

    std::ranges::sort(ids);

    // return the new end
    return std::ranges::remove_if(points,
                                  [&ids](const Point& p) {
                                      return std::ranges::binary_search(ids, p.id);
                                  })
            .end();
}

/**
 * \brief Uses OpenMP to determine whether a condition exists in the specified range of elements.
 * https://msdn.microsoft.com/en-us/library/ff521445.aspx
 * \param[in] first first iterator
 * \param[in] last last iterator
 * \param[in] pr the predicate
 * \return true if aany if the element respect the predicate
 */
template<class InIt, class Predicate> bool omp_parallel_any_of(InIt first, InIt last, const Predicate& pr)
{
    using item_type = typename std::iterator_traits<InIt>::value_type;

    for (int i = 0; i < static_cast<int>(last - first); ++i)
    {
        item_type& cur = *(first + i);

        // If the element satisfies the condition, set the flag to cancel the operation.
        if (pr(cur))
            return true;
    }

    // not found
    return false;
}

/**
 * \brief Check whether all points in a begin/end range are inside hull.
 * \param[in] begin The iterator to the begin of the points to test
 * \param[in] end The iterator to the end of the points to test
 * \param[in] hull The polygon boundary
 * \return true if all the points are in the polygon
 */
bool MultiplePointInPolygon(PointVector::const_iterator begin, PointVector::const_iterator end, const PointVector& hull)
{
    if (begin == end)
        return false;

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

/**
 * \brief Check that a point is in a polygon
 * \param[in] p The point to test
 * \param[in] list The polygon boundary
 * \return true if the point is in the polygon
 */
bool PointInPolygon(const Point& p, const PointVector& list)
{
    if (list.size() <= 2)
        return false;

    const double& x = p.x;
    const double& y = p.y;

    int inout = 0;
    PointVector::const_iterator v0 = list.begin();
    PointVector::const_iterator v1 = v0 + 1;

    while (v1 != list.end())
    {
        if ((LessThanOrEqual(v0->y, y) && LessThan(y, v1->y)) || (LessThanOrEqual(v1->y, y) && LessThan(y, v0->y)))
        {
            if (!Zero(v1->y - v0->y))
            {
                double tdbl1 = (y - v0->y) / (v1->y - v0->y);
                double tdbl2 = v1->x - v0->x;

                if (LessThan(x, v0->x + (tdbl2 * tdbl1)))
                    inout++;
            }
        }

        v0 = v1;
        ++v1;
    }

    if (inout == 0)
        return false;
    else if (inout % 2 == 0)
        return false;
    else
        return true;
}

/**
 * \brief Test whether two line segments intersect each other
 * \param[in] a the first line segment
 * \param[in] b the second line segment
 * \return true if the line segment intersects
 */
bool Intersects(const LineSegment& a, const LineSegment& b)
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

/**
 * \brief Unit test of Angle() function
 */
void TestAngle()
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

/**
 * \brief Unit test the Intersects() function
 */
void TestIntersects()
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