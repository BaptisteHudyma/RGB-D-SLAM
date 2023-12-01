#include "basis_changes.hpp"

namespace rgbd_slam::utils {

Cartesian Cartesian::from(Spherical coord)
{
    const double sinTheta = sin(coord.theta);

    Cartesian res;
    res.x = coord.p * sinTheta * cos(coord.phi);
    res.y = coord.p * sinTheta * sin(coord.phi);
    res.z = coord.p * cos(coord.theta);
    return res;
}

Cartesian Cartesian::from(Spherical coord, matrix33& jacobian)
{
    const double sinTheta = sin(coord.theta);
    const double cosTheta = cos(coord.theta);
    const double sinPhi = sin(coord.phi);
    const double cosPhi = cos(coord.phi);
    const double d = coord.p;

    jacobian = matrix33({{sinTheta * cosPhi, d * cosTheta * sinPhi, -d * sinTheta * sinPhi},
                         {sinTheta * sinPhi, d * cosTheta * sinPhi, d * sinTheta * cosPhi},
                         {cosTheta, -d * sinTheta, 0}});

    return from(coord);
}

Spherical Spherical::from(Cartesian coord)
{
    const double p = coord.vec().norm();
    const double xx = coord.x * coord.x;
    const double yy = coord.y * coord.y;

    Spherical res;
    res.p = p;
    res.theta = atan2(sqrt(xx + yy), coord.z);
    res.phi = atan2(coord.y, coord.x);
    return res;
}

Spherical Spherical::from(Cartesian coord, matrix33& jacobian)
{
    const double p = coord.vec().norm();
    const double pp = p * p;
    const double xx = coord.x * coord.x;
    const double yy = coord.y * coord.y;

    const double Sqr = sqrt(xx + yy);

    jacobian = matrix33({{coord.x / p, coord.y / p, coord.z / p},
                         {coord.x * coord.z / (pp * Sqr), coord.y * coord.z / (pp * Sqr), -Sqr / pp},
                         {-coord.y / (xx + yy), coord.x / (xx + yy), 0}});
    return from(coord);
}

} // namespace rgbd_slam::utils