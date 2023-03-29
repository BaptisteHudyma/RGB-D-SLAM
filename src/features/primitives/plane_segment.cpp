#include "plane_segment.hpp"
#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "Eigen/Eigenvalues"
#include "coordinates.hpp"
#include "covariances.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <cmath>
#include <iostream>

namespace rgbd_slam::features::primitives {

Plane_Segment::Plane_Segment()
{
    assert(_isStaticSet);

    clear_plane_parameters();
    _isPlanar = false;
}

Plane_Segment::Plane_Segment(const Plane_Segment& seg) :
    _pointCount(seg._pointCount),
    _score(seg._score),
    _MSE(seg._MSE),
    _isPlanar(seg._isPlanar),
    _centroid(seg._centroid),
    _normal(seg._normal.normalized()),
    _d(seg._d),
    _Sx(seg._Sx),
    _Sy(seg._Sy),
    _Sz(seg._Sz),
    _Sxs(seg._Sxs),
    _Sys(seg._Sys),
    _Szs(seg._Szs),
    _Sxy(seg._Sxy),
    _Syz(seg._Syz),
    _Szx(seg._Szx)
{
    assert(_isStaticSet);
}

/**
 * \brief Runs a check on a depth value, and increment a discontinuity counter if needed
 * \param[in] pixelDepth The depth value to check
 * \param[in,out] lastPixelDepth Last depht value to pass the continuity test
 * \return False if a continuities is detected
 */
bool is_continuous(const float pixelDepth, float& lastPixelDepth)
{
    // ignore empty depth values
    if (pixelDepth > 0)
    {
        // check for suddent jumps in the depth values, that are superior to the expected quantization for this
        // depth
        if (abs(pixelDepth - lastPixelDepth) <= pow(2 * utils::get_depth_quantization(pixelDepth), 2.0))
        {
            // no suddent jump
            lastPixelDepth = pixelDepth;
            return true;
        }
        return false;
    }
    return true;
}

bool Plane_Segment::is_cell_vertical_continuous(const matrixf& depthMatrix) const
{
    const uint startValue = _cellWidth / 2;
    const uint endValue = _ptsPerCellCount - startValue;

    float lastPixelDepth = std::max(depthMatrix(startValue),
                                    depthMatrix(startValue + _cellWidth)); /* handles missing pixels on the borders*/
    if (lastPixelDepth <= 0)
        return false;

    // Scan vertically through the middle
    for (uint i = startValue + _cellWidth; i < endValue; i += _cellWidth)
    {
        const float pixelDepth = depthMatrix(i);
        if (not is_continuous(pixelDepth, lastPixelDepth))
            return false;
    }
    // continuous
    return true;
}

bool Plane_Segment::is_cell_horizontal_continuous(const matrixf& depthMatrix) const
{
    const uint startValue = static_cast<uint>(_cellWidth * (_cellHeight / 2.0));
    const uint endValue = startValue + _cellWidth;

    float lastPixelDepth =
            std::max(depthMatrix(startValue), depthMatrix(startValue + 1)); /* handles missing pixels on the borders*/
    if (lastPixelDepth <= 0)
        return false;

    // Scan horizontally through the middle
    for (uint i = startValue + 1; i < endValue; ++i)
    {
        const float pixelDepth = depthMatrix(i);
        if (not is_continuous(pixelDepth, lastPixelDepth))
            return false;
    }
    // continuous
    return true;
}

void Plane_Segment::init_plane_segment(const matrixf& depthCloudArray, const uint cellId)
{
    clear_plane_parameters();

    const uint offset = cellId * _ptsPerCellCount;

    // get z of depth points
    const matrixf& zMatrix = depthCloudArray.block(offset, 2, _ptsPerCellCount, 1);

    // Check number of missing depth points
    _pointCount = static_cast<uint>((zMatrix.array() > 0).count());
    if (_pointCount < _minZeroPointCount)
    {
        return;
    }

    // Check for discontinuities using cross search
    // Search discontinuities only in a vertical line passing through the center, than an horizontal line passing
    // through the center.
    if (not is_cell_horizontal_continuous(zMatrix) and is_cell_vertical_continuous(zMatrix))
    {
        // this segment is not continuous...
        return;
    }

    // get points x and y coords
    const matrixf& xMatrix = depthCloudArray.block(offset, 0, _ptsPerCellCount, 1);
    const matrixf& yMatrix = depthCloudArray.block(offset, 1, _ptsPerCellCount, 1);

    // set PCA components
    _Sx = xMatrix.sum();
    _Sy = yMatrix.sum();
    _Sz = zMatrix.sum();
    _Sxs = (xMatrix.array() * xMatrix.array()).sum();
    _Sys = (yMatrix.array() * yMatrix.array()).sum();
    _Szs = (zMatrix.array() * zMatrix.array()).sum();
    _Sxy = (xMatrix.array() * yMatrix.array()).sum();
    _Szx = (xMatrix.array() * zMatrix.array()).sum();
    _Syz = (yMatrix.array() * zMatrix.array()).sum();

    assert(_Sz > _pointCount);
    assert(_Sxs > 0);
    assert(_Sys > 0);
    assert(_Szs > 0);

    _isPlanar = true;
    // fit a plane to those points
    fit_plane();
    // plane variance should be less than depth quantization, plus a tolerance factor
    _isPlanar = _isPlanar and _MSE <= pow(2 * utils::get_depth_quantization(_centroid.z()), 2.0);
}

void Plane_Segment::expand_segment(const Plane_Segment& planeSegment)
{
    _Sx += planeSegment._Sx;
    _Sy += planeSegment._Sy;
    _Sz += planeSegment._Sz;

    _Sxs += planeSegment._Sxs;
    _Sys += planeSegment._Sys;
    _Szs += planeSegment._Szs;

    _Sxy += planeSegment._Sxy;
    _Syz += planeSegment._Syz;
    _Szx += planeSegment._Szx;

    assert(_Sz > 0);
    assert(_Sxs > 0);
    assert(_Sys > 0);
    assert(_Szs > 0);

    _pointCount += planeSegment._pointCount;
}

matrix33 Plane_Segment::get_point_cloud_covariance() const
{
    assert(_pointCount > 0);
    const double oneOverCount = 1.0 / static_cast<double>(_pointCount);

    // diagonal
    // TODO: why do I have a zero sometimes ? floatting point error accumulation ?
    const double xxCovariance = std::max(0.0, _Sxs - _Sx * _Sx * oneOverCount);
    const double yyCovariance = std::max(0.0, _Sys - _Sy * _Sy * oneOverCount);
    const double zzCovariance = std::max(0.0, _Szs - _Sz * _Sz * oneOverCount);
    assert(xxCovariance >= 0);
    assert(yyCovariance >= 0);
    assert(zzCovariance >= 0);

    // bottom/top half
    const double xyCovariance = _Sxy - _Sx * _Sy * oneOverCount;
    const double xzCovariance = _Szx - _Sx * _Sz * oneOverCount;
    const double yzCovariance = _Syz - _Sy * _Sz * oneOverCount;

    // Expressing covariance as E[PP^t] + E[P]*E[P^T]
    matrix33 covariance({{xxCovariance, xyCovariance, xzCovariance},
                         {xyCovariance, yyCovariance, yzCovariance},
                         {xzCovariance, yzCovariance, zzCovariance}});
    return covariance;
}

void Plane_Segment::fit_plane()
{
    assert(_pointCount > 0);
    const double oneOverCount = 1.0 / static_cast<double>(_pointCount);

    // get the centroid of the plane
    _centroid = vector3(_Sx, _Sy, _Sz) * oneOverCount;

    const matrix33 pointCloudCov = get_point_cloud_covariance();
    // special case:
    if (utils::double_equal(pointCloudCov.determinant(), 0))
    {
        _isPlanar = false;
        return;
    }
    // no need to fill the upper part, the adjoint solver does not need it
    Eigen::SelfAdjointEigenSolver<matrix33> eigenSolver(pointCloudCov);
    // eigen values are the point variance along the eigen vectors (sorted by ascending order)
    const vector3& eigenValues = eigenSolver.eigenvalues();
    // best eigen vector is the most reliable direction for this plane normal
    const vector3& eigenVector = eigenSolver.eigenvectors().col(0);

    // some values have floatting points errors, renormalize
    _normal = eigenVector.normalized();
    _d = -_normal.dot(_centroid.base());

    // point normal toward the camera
    if (_d <= 0)
    {
        _normal = -_normal;
        _d = -_d;
    }

    assert(utils::double_equal(_normal.norm(), 1.0));

    // variance of points in our plane divided by number of points in the plane
    _MSE = eigenValues(0) * oneOverCount;
    // second best variance divided by variance of this plane patch
    _score = eigenValues(1) / std::max(eigenValues(0), 1e-6);
    // const double curvature = eigenValues(0) / eigenValues.sum();

    // failure case: covariance matrix is hill formed
    if (_MSE < 0 or _score < 0)
    {
        // TODO; find out why
        // outputs::log_warning("Plane patch covariance matrix is ill formed, rejecting it");
        _isPlanar = false;
        return;
    }

    // set segment as planar
    _isPlanar = true;
}

/*
 * Sets all the plane parameters to zero
 */
void Plane_Segment::clear_plane_parameters()
{
    _isPlanar = false;

    _pointCount = 0;
    _score = 0;
    _MSE = 0;

    _centroid.setZero();
    _normal.setZero();
    _d = 0;

    // Clear saved plane parameters
    _Sx = 0;
    _Sy = 0;
    _Sz = 0;
    _Sxs = 0;
    _Sys = 0;
    _Szs = 0;
    _Sxy = 0;
    _Syz = 0;
    _Szx = 0;
}

double Plane_Segment::get_cos_angle(const Plane_Segment& p) const
{
    assert(_isPlanar);
    return (_normal.dot(p._normal));
}
double Plane_Segment::get_point_distance(const vector3& point) const { return pow(_normal.dot(point) + _d, 2.0); }
bool Plane_Segment::can_be_merged(const Plane_Segment& p, const double maxMatchDistance) const
{
    const static double maximumMergeAngle = cos(Parameters::get_maximum_plane_merge_angle() * M_PI / 180.0);
    return get_cos_angle(p) > maximumMergeAngle and get_point_distance(p.get_centroid().base()) < maxMatchDistance;
}

} // namespace rgbd_slam::features::primitives
