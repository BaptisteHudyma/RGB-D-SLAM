#include "line.hpp"

namespace rgbd_slam::utils {

// instanciation to check compile failures
void test_segments()
{
    // 1D
    Segment<1> test1d;
    Eigen::Vector<double, 1> point1d;

    std::ignore = test1d.distance(Eigen::Vector<double, 1>(0));
    std::ignore = test1d.intersects(Segment<1>(), point1d);

    // 2d
    Segment<2> test2d;
    Eigen::Vector<double, 2> point2d;

    std::ignore = test2d.distance(Eigen::Vector<double, 2>(0));
    std::ignore = test2d.intersects(Segment<2>(), point2d);

    // 3d
    Segment<3> test3d;
    Eigen::Vector<double, 3> point3d;

    std::ignore = test3d.distance(Eigen::Vector<double, 3>(0));
    std::ignore = test3d.intersects(Segment<3>(), point3d);
}

void test_lines()
{
    // 1D
    Line<1> test1d;
    Eigen::Vector<double, 1> point1d;

    std::ignore = test1d.distance(Eigen::Vector<double, 1>(0));
    std::ignore = test1d.intersects(Line<1>(), point1d);

    // 2d
    Line<2> test2d;
    Eigen::Vector<double, 2> point2d;

    std::ignore = test2d.distance(Eigen::Vector<double, 2>(0));
    std::ignore = test2d.intersects(Line<2>(), point2d);

    // 3d
    Line<3> test3d;
    Eigen::Vector<double, 3> point3d;

    std::ignore = test3d.distance(Eigen::Vector<double, 3>(0));
    std::ignore = test3d.intersects(Line<3>(), point3d);
}

} // namespace rgbd_slam::utils