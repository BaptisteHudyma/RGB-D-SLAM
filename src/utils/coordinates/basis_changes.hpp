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

    vector3 vec() const { return vector3(x, y, z); }
    void from_vec(const vector3& vec)
    {
        x = vec.x();
        y = vec.y();
        z = vec.z();
    }

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
    double theta;
    double phi;
    double p;

    vector3 vec() const { return vector3(theta, phi, p); }

    void from_vec(const vector3& vec)
    {
        theta = vec.x();
        phi = vec.y();
        p = vec.z();
    }

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