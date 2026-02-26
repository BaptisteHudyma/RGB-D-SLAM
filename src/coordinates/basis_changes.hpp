#ifndef RGBDSLAM_BASIC_CHANGES_HPP
#define RGBDSLAM_BASIC_CHANGES_HPP

#include "types.hpp"

namespace rgbd_slam {

struct Cartesian;
struct Spherical;

/**
 * \brief Contain a cartesian point, with methods to convert between representations
 * ALL IS DEFINED IN A ROBOTIC COORDINATE SYSTEM : X forward, Y left, Z up
 */
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
    static Cartesian from(const Spherical& coord);

    /**
     * \brief Transform a given coordinate from shperical to cartesian space.
     * \param[in] coord
     * \param[out] jacobian the jacobian of this transformation
     */
    static Cartesian from(const Spherical& coord, matrix33& jacobian);
};

/**
 * \brief Contain a spherical point, with methods to convert between representations
 */
struct Spherical
{
    double p;           // radius
    double polar_rad;   // polar angle
    double azimuth_rad; // azimuth angle

    static constexpr uint RadiusIndex = 0;
    static constexpr uint PolarIndex = 1;
    static constexpr uint AzimuthIndex = 2;

    Spherical(const double radius, const double polar, const double azimuth) :
        p(radius),
        polar_rad(polar),
        azimuth_rad(azimuth)
    {
    }
    Spherical(const vector3& vec) : Spherical(vec.x(), vec.y(), vec.z()) {};

    vector3 vec() const
    {
        vector3 vec;
        vec(RadiusIndex) = p;
        vec(PolarIndex) = polar_rad;
        vec(AzimuthIndex) = azimuth_rad;
        return vec;
    }

    /**
     * \brief Transform a given coordinate from cartesian to spherical space.
     */
    static Spherical from(const Cartesian& coord);

    /**
     * \brief Transform a given coordinate from cartesian to spherical space.
     * \param[in] coord
     * \param[out] jacobian the jacobian of this transformation
     */
    static Spherical from(const Cartesian& coord, matrix33& jacobian);
};

} // namespace rgbd_slam

#endif
