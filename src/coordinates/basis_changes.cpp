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

    const double theta1 = sinPhi * sinTheta;
    const double theta2 = cosPhi * sinTheta;

    //
    jacobian = matrix33({
            //
            // radius              polar            azimuth
            {theta2, d * cosTheta * cosPhi, -d * theta1}, // x
            {theta1, d * cosTheta * sinPhi, d * theta2},  // y
            {cosTheta, -d * sinTheta, 0}                  // z
    });

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

    const double theta1 = xx + yy + zz;
    const double theta2 = xx + yy;
    const double SqrtTheta1 = sqrt(theta1);
    const double SqrtTheta2 = sqrt(theta2);
    const double inverseTheta1Theta2 = 1.0 / (SqrtTheta2 * theta1);

    jacobian = matrix33({{x / SqrtTheta1, y / SqrtTheta1, z / SqrtTheta1},
                         {x * z * inverseTheta1Theta2, y * z * inverseTheta1Theta2, -SqrtTheta2 / theta1},
                         {-y / theta2, x / theta2, 0}});
    return from(coord);
}

} // namespace rgbd_slam
