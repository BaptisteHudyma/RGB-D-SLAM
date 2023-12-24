#ifndef RGBDSLAM_UTILS_DISTANCE_UTILS_HPP
#define RGBDSLAM_UTILS_DISTANCE_UTILS_HPP

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <limits>

namespace rgbd_slam::utils {

/**
 * \brief compute a distance bewteen two angles in radian
 * \param[in] angleA
 * \param[in] angleB
 */
[[nodiscard]] double angle_distance(const double angleA, const double angleB) noexcept;

/**
 * \brief Check an equality between two doubles, with an epsilon: TODO move this to a more sensible location
 * \param[in] a
 * \param[in] b
 * \param[in] epsilon
 * \return true if the two values are equal at +- epsilon
 */
[[nodiscard]] bool double_equal(const double a,
                                const double b,
                                const double epsilon = std::numeric_limits<double>::epsilon()) noexcept;

/**
 * \brief compute the distance between two lines.
 * This is the distance between the two points closest to each others on each line
 * \param[out] closestL1toL2 The closest point on L1 to L2
 * \param[out] closestL2toL1 The closest point on L2 to L1
 * \return a signed distance, in the same unit as the points
 */
template<int Dim> [[nodiscard]] bool compute_closest_points(const Eigen::Vector<double, Dim>& line1point,
                                                            const Eigen::Vector<double, Dim>& line1normal,
                                                            const Eigen::Vector<double, Dim>& line2point,
                                                            const Eigen::Vector<double, Dim>& line2normal,
                                                            Eigen::Vector<double, Dim>& closestL1toL2,
                                                            Eigen::Vector<double, Dim>& closestL2toL1) noexcept
{
    // we want to find a point that the two lines pass by, define by the origin of the line and the normal :
    // P = p1 + t1 * d1
    // P = p2 + t2 * d2

    const Eigen::Vector<double, Dim> n = line1normal.cross(line2normal);
    // normals are parallels, distance is the distance between points
    if (n.isApproxToConstant(0.0))
    {
        // parallel line distance
        return false;
    }

    const Eigen::Vector<double, Dim>& n1 = line1normal.cross(n);
    const Eigen::Vector<double, Dim>& n2 = line2normal.cross(n);

    const double t1 = (line2point - line1point).dot(n2) / line1normal.dot(n2);
    const double t2 = (line1point - line2point).dot(n1) / line2normal.dot(n1);

    closestL1toL2 = line1point + t1 * line1normal; // closest point to line 2 on line 1
    closestL2toL1 = line2point + t2 * line2normal; // closest point to line 1 on line 2
    return true;
}

/**
 * \brief compute the distance between two lines.
 * This is the distance between the two points closest to each others on each line
 * \return a signed distance, in the same unit as the points
 */
template<int Dim>
[[nodiscard]] Eigen::Vector<double, Dim> signed_line_distance(const Eigen::Vector<double, Dim>& line1point,
                                                              const Eigen::Vector<double, Dim>& line1normal,
                                                              const Eigen::Vector<double, Dim>& line2point,
                                                              const Eigen::Vector<double, Dim>& line2normal) noexcept
{
    Eigen::Vector<double, Dim> closestL1toL2;
    Eigen::Vector<double, Dim> closestL2toL1;
    if (compute_closest_points<Dim>(line1point, line1normal, line2point, line2normal, closestL1toL2, closestL2toL1))
    {
        return closestL1toL2 - closestL2toL1;
    }

    return line1normal.cross(line1point - line2point);
}

} // namespace rgbd_slam::utils

#endif
