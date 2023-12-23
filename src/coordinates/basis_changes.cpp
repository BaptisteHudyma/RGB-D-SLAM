#include "basis_changes.hpp"

namespace rgbd_slam {

Cartesian Cartesian::from(Spherical coord)
{
    const double sinTheta = sin(coord.theta);
    return Cartesian(
            coord.p * sinTheta * cos(coord.phi), coord.p * sinTheta * sin(coord.phi), coord.p * cos(coord.theta));
}

Cartesian Cartesian::from(Spherical coord, matrix33& jacobian)
{
    const double sinTheta = sin(coord.theta);
    const double cosTheta = cos(coord.theta);
    const double sinPhi = sin(coord.phi);
    const double cosPhi = cos(coord.phi);
    const double d = coord.p;

    const double theta1 = sinPhi * sinTheta;
    const double theta2 = cosPhi * sinTheta;

    jacobian = matrix33({{theta2, d * cosTheta * cosPhi, -d * theta1},
                         {theta1, d * cosTheta * sinPhi, d * theta2},
                         {cosTheta, -d * sinTheta, 0}});

    return from(coord);
}

Spherical Spherical::from(Cartesian coord)
{
    return Spherical(coord.vec().norm(), atan2(sqrt(SQR(coord.x) + SQR(coord.y)), coord.z), atan2(coord.y, coord.x));
}

Spherical Spherical::from(Cartesian coord, matrix33& jacobian)
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