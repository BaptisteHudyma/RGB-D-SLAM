#ifndef BASIC_CHANGES_HPP
#define BASIC_CHANGES_HPP

#include "types.hpp"

namespace rgbd_slam::utils {

struct Cartesian;
struct Spherical;

struct Cartesian
{
    double x;
    double y;
    double z;

    Cartesian(const vector3& vec) : x(vec.x()), y(vec.y()), z(vec.z()) {};
    Cartesian(const double x, const double y, const double z) : x(x), y(y), z(z) {};

    vector3 vec() const { return vector3(x, y, z); }

    /**
     * \brief Transform a given coordinate from shperical to cartesian space.
     * \param[in] coord
     */
    static Cartesian from(Spherical coord);

    /**
     * \brief Transform a given coordinate from shperical to cartesian space.
     * \param[in] coord
     * \param[out] jacobian the jacobian of this transformation
     */
    static Cartesian from(Spherical coord, matrix33& jacobian);
};

struct Spherical
{
    double p;
    double theta;
    double phi;

    Spherical(const double radius, const double theta, const double phi) : p(radius), theta(theta), phi(phi) {}
    Spherical(const vector3& vec) : Spherical(vec.x(), vec.y(), vec.z()) {};

    vector3 vec() const { return vector3(p, theta, phi); }

    /**
     * \brief Transform a given coordinate from cartesian to spherical space.
     */
    static Spherical from(Cartesian coord);

    /**
     * \brief Transform a given coordinate from cartesian to spherical space.
     * \param[in] coord
     * \param[out] jacobian the jacobian of this transformation
     */
    static Spherical from(Cartesian coord, matrix33& jacobian);
};

} // namespace rgbd_slam::utils

#endif