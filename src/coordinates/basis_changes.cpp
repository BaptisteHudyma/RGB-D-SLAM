#include "basis_changes.hpp"

namespace rgbd_slam {

Cartesian Cartesian::from(const Spherical& coord)
{
    const double sinTheta = sin(coord.polar_rad);
    return Cartesian(coord.p * sinTheta * cos(coord.azimuth_rad),
                     coord.p * sinTheta * sin(coord.azimuth_rad),
                     coord.p * cos(coord.polar_rad));
}

Cartesian Cartesian::from(const Spherical& coord, matrix33& jacobian)
{
    const double sinTheta = sin(coord.polar_rad);
    const double cosTheta = cos(coord.polar_rad);
    const double sinPhi = sin(coord.azimuth_rad);
    const double cosPhi = cos(coord.azimuth_rad);
    const double d = coord.p;

    // X   Y  Z -> radius
    jacobian.col(Spherical::RadiusIndex) = vector3 {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
    // X   Y  Z -> polar
    jacobian.col(Spherical::PolarIndex) = vector3 {d * cosTheta * cosPhi, d * cosTheta * sinPhi, -d * sinTheta};
    // X   Y  Z -> azimuth
    jacobian.col(Spherical::AzimuthIndex) = vector3 {-d * sinTheta * sinPhi, d * sinTheta * cosPhi, 0};

    return from(coord);
}

Spherical Spherical::from(const Cartesian& coord)
{
    return Spherical(coord.vec().norm(), atan2(sqrt(SQR(coord.x) + SQR(coord.y)), coord.z), atan2(coord.y, coord.x));
}

Spherical Spherical::from(const Cartesian& coord, matrix33& jacobian)
{
    const double x = coord.x;
    const double y = coord.y;
    const double z = coord.z;
    const double xx = SQR(x);
    const double yy = SQR(y);
    const double zz = SQR(z);

    const double radiusSqr = xx + yy + zz;
    const double theta2 = xx + yy;
    const double radius = sqrt(radiusSqr);
    const double SqrtTheta2 = sqrt(theta2);
    const double inverseTheta1Theta2 = 1.0 / (SqrtTheta2 * radiusSqr);

    // radius -> X Y Z
    jacobian.row(Spherical::RadiusIndex) = vector3 {x / radius, y / radius, z / radius};
    // polar -> X Y Z
    jacobian.row(Spherical::PolarIndex) =
            vector3 {x * z * inverseTheta1Theta2, y * z * inverseTheta1Theta2, -theta2 * inverseTheta1Theta2};
    // azimuth -> X Y Z
    jacobian.row(Spherical::AzimuthIndex) = vector3 {-y / theta2, x / theta2, 0};

    return from(coord);
}

} // namespace rgbd_slam
