#include "line.hpp"
#include "coordinates/point_coordinates.hpp"
#include "logger.hpp"
#include "polygon.hpp"

#include <boost/geometry/geometries/infinite_line.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xyz.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/algorithms/intersection.hpp>
#include <format>

namespace rgbd_slam::utils {

[[nodiscard]] bool clamp_to_screen(const Segment<2>& line, Segment<2>& out) noexcept
{
    typedef boost::geometry::model::d2::point_xy<double> point_t;
    typedef boost::geometry::model::linestring<point_t> line_t;

    line_t lineBoost({point_t(line.get_start_point().x(), line.get_start_point().y()),
                      point_t(line.get_end_point().x(), line.get_end_point().y())});

    const auto& boundar = get_static_screen_boundary_polygon();
    std::vector<point_t> output;
    if (boost::geometry::intersection(lineBoost, boundar, output))
    {
        switch (output.size())
        {
            case 0: // no intersection with boundary
                {
                    const ScreenCoordinate2D startPoint(line.get_start_point());
                    const ScreenCoordinate2D endPoint(line.get_end_point());
                    // no points in boundary, quit
                    if (not startPoint.is_in_screen_boundaries() or not endPoint.is_in_screen_boundaries())
                    {
                        return false;
                    }
                    // all points in boundary, continue
                    out.set_points(line.get_start_point(), line.get_end_point());
                    break;
                }
            case 1:
                {
                    // find the point in the screen boundary
                    const ScreenCoordinate2D startPoint(output[0].x(), output[0].y());
                    const ScreenCoordinate2D endPoint(output[0].x(), output[0].y());
                    if (startPoint.is_in_screen_boundaries())
                    {
                        out.set_points(startPoint, vector2(output[0].x(), output[0].y()));
                    }
                    else
                    {
                        out.set_points(vector2(output[0].x(), output[0].y()), endPoint);
                    }
                    break;
                }
            case 2:
                {
                    out.set_points(vector2(output[0].x(), output[0].y()), vector2(output[1].x(), output[1].y()));
                    break;
                }
                // should never have other cases
            default:
                {
                    outputs::log_error(std::format("unhandled case for value {}", output.size()));
                    return false;
                }
        }
        return true;
    }
    return false;
}

bool intersects(const ILine<2>& firstLine, const ILine<2>& secondLine, Eigen::Vector<double, 2>& point) noexcept
{
    constexpr double projectionDistance = 100.0;
    Segment<2> seg1(firstLine.get_start_point(),
                    firstLine.get_start_point() + (projectionDistance * firstLine.compute_normal()));
    Segment<2> seg2(secondLine.get_start_point(),
                    secondLine.get_start_point() + (projectionDistance * secondLine.compute_normal()));
    return intersects(seg1, seg2, point);
}

bool intersects(const ILine<3>& firstLine, const ILine<3>& secondLine, Eigen::Vector<double, 3>& point) noexcept
{
    constexpr double projectionDistance = 100.0;
    Segment<3> seg1(firstLine.get_start_point(),
                    firstLine.get_start_point() + (projectionDistance * firstLine.compute_normal()));
    Segment<3> seg2(secondLine.get_start_point(),
                    secondLine.get_start_point() + (projectionDistance * secondLine.compute_normal()));
    return intersects(seg1, seg2, point);
}

bool intersects(const Segment<2>& firstLine, const Segment<2>& secondLine, Eigen::Vector<double, 2>& point) noexcept
{
    typedef boost::geometry::model::d2::point_xy<double> point_t;
    typedef boost::geometry::model::linestring<point_t> line_t;

    line_t seg1({point_t(firstLine.get_start_point().x(), firstLine.get_start_point().y()),
                 point_t(firstLine.get_end_point().x(), firstLine.get_end_point().y())});
    line_t seg2({point_t(secondLine.get_start_point().x(), secondLine.get_start_point().y()),
                 point_t(secondLine.get_end_point().x(), secondLine.get_end_point().y())});

    std::vector<point_t> output;
    if (boost::geometry::intersection(seg1, seg2, output))
    {
        point.x() = output[0].x();
        point.y() = output[0].y();
        return true;
    }
    return false;
}

bool intersects(const Segment<3>& firstLine, const Segment<3>& secondLine, Eigen::Vector<double, 3>& point) noexcept
{
    typedef boost::geometry::model::d3::point_xyz<double> point_t;
    typedef boost::geometry::model::linestring<point_t> line_t;

    line_t seg1(
            {point_t(firstLine.get_start_point().x(), firstLine.get_start_point().y(), firstLine.get_start_point().z()),
             point_t(firstLine.get_end_point().x(), firstLine.get_end_point().y(), firstLine.get_end_point().z())});
    line_t seg2(
            {point_t(secondLine.get_start_point().x(),
                     secondLine.get_start_point().y(),
                     secondLine.get_start_point().z()),
             point_t(secondLine.get_end_point().x(), secondLine.get_end_point().y(), secondLine.get_end_point().z())});

    std::vector<point_t> output;
    if (boost::geometry::intersection(seg1, seg2, output))
    {
        point.x() = output[0].x();
        point.y() = output[0].y();
        point.z() = output[0].z();
        return true;
    }
    return false;
}

// instanciation to check compile failures
void test_segments()
{
    // 1D
    Segment<1> test1d;
    Eigen::Vector<double, 1> point1d;

    std::ignore = test1d.distance(Eigen::Vector<double, 1>(0));
    // std::ignore = intersects(test1d, Segment<1>(), point1d);

    // 2d
    Segment<2> test2d;
    Eigen::Vector<double, 2> point2d;

    std::ignore = test2d.distance(Eigen::Vector<double, 2>(0));
    std::ignore = intersects(test2d, Segment<2>(), point2d);

    // 3d
    Segment<3> test3d;
    Eigen::Vector<double, 3> point3d;

    std::ignore = test3d.distance(Eigen::Vector<double, 3>(0));
    std::ignore = intersects(test3d, Segment<3>(), point3d);
}

void test_lines()
{
    // 1D
    Line<1> test1d;
    Eigen::Vector<double, 1> point1d;

    std::ignore = test1d.distance(Eigen::Vector<double, 1>(0));
    // std::ignore = intersects(test1d, Line<1>(), point1d);

    // 2d
    Line<2> test2d;
    Eigen::Vector<double, 2> point2d;

    std::ignore = test2d.distance(Eigen::Vector<double, 2>(0));
    std::ignore = intersects(test2d, Line<2>(), point2d);

    // 3d
    Line<3> test3d;
    Eigen::Vector<double, 3> point3d;

    std::ignore = test3d.distance(Eigen::Vector<double, 3>(0));
    std::ignore = intersects(test3d, Line<3>(), point3d);
}

} // namespace rgbd_slam::utils