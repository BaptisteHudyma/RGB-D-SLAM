#include "plane_segment.hpp"
#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "Eigen/Eigenvalues"
#include "covariances.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <cmath>

namespace rgbd_slam::features::primitives {

Plane_Segment::Plane_Segment()
{
    assert(_isStaticSet);

    clear_plane_parameters();
}

Plane_Segment::Plane_Segment(const Plane_Segment& seg) :
    _pointCount(seg._pointCount),
    _score(seg._score),
    _MSE(seg._MSE),
    _isPlanar(seg._isPlanar),
    _centroid(seg._centroid),
    _parametrization(seg._parametrization),
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
bool is_continuous(const float pixelDepth, float& lastPixelDepth) noexcept
{
    // ignore empty depth values
    if (pixelDepth > 0)
    {
        // check for suddent jumps in the depth values, that are superior to the expected quantization for this depth
        // TODO: this 4.0 x is weird
        if (abs(pixelDepth - lastPixelDepth) <= 4.0 * utils::get_depth_quantization(pixelDepth))
        {
            // no suddent jump
            lastPixelDepth = pixelDepth;
            return true;
        }
        return false;
    }
    return true;
}

bool Plane_Segment::is_cell_vertical_continuous(const matrixf& depthMatrix) const noexcept
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
        if (not is_continuous(depthMatrix(i), lastPixelDepth))
            return false;
    }
    // continuous
    return true;
}

bool Plane_Segment::is_cell_horizontal_continuous(const matrixf& depthMatrix) const noexcept
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
        if (not is_continuous(depthMatrix(i), lastPixelDepth))
            return false;
    }
    // continuous
    return true;
}

void Plane_Segment::init_plane_segment(const matrixf& depthCloudArray, const uint cellId) noexcept
{
    clear_plane_parameters();

    const uint offset = cellId * _ptsPerCellCount;

    // get z of depth points
    const matrixf& zMatrix = depthCloudArray.block(offset, 2, _ptsPerCellCount, 1);

    // Check for discontinuities using cross search
    // Search discontinuities only in a vertical line passing through the center, than an horizontal line passing
    // through the center.
    if (not is_cell_horizontal_continuous(zMatrix) or not is_cell_vertical_continuous(zMatrix))
    {
        // this segment is not continuous
        return;
    }
    // remove cells with too much empty values
    if ((zMatrix.array() > 0).count() < _ptsPerCellCount / 2)
    {
        return;
    }

    // get points x and y coords
    const matrixf& xMatrix = depthCloudArray.block(offset, 0, _ptsPerCellCount, 1);
    const matrixf& yMatrix = depthCloudArray.block(offset, 1, _ptsPerCellCount, 1);

    _pointCount = 0;
    const uint zSize = zMatrix.size();
    for (uint i = 0; i < zSize; ++i)
    {
        const float z = zMatrix(i);
        if (z > 0)
        {
            ++_pointCount;

            const float x = xMatrix(i);
            const float y = yMatrix(i);

            // set PCA components
            _Sx += x;
            _Sy += y;
            _Sz += z;
            _Sxs += x * x;
            _Sys += y * y;
            _Szs += z * z;
            _Sxy += x * y;
            _Szx += x * z;
            _Syz += y * z;
        }
    }

    // Check number of missing depth points
    if (_pointCount < _minZeroPointCount)
    {
        return;
    }

    assert(_Sxs > 0);
    assert(_Sys > 0);
    assert(_Szs > 0);

    // fit a plane to those points
    fit_plane();
    // plane variance should be less than depth quantization, plus a tolerance factor
    _isPlanar = _MSE <= pow(utils::get_depth_quantization(_centroid.z()), 2.0);
}

void Plane_Segment::expand_segment(const Plane_Segment& planeSegment) noexcept
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

matrix33 Plane_Segment::get_point_cloud_covariance() const noexcept
{
    const matrix33 pointCloudHessian({{_Sxs, _Sxy, _Szx}, {_Sxy, _Sys, _Syz}, {_Szx, _Syz, _Szs}});

    // 0 determinant cannot be inverted
    assert(not utils::double_equal(pointCloudHessian.determinant(), 0.0));

    const matrix33& covariance = pointCloudHessian.inverse();
    assert(utils::is_covariance_valid(covariance));
    return covariance;
}

matrix33 Plane_Segment::get_point_cloud_Huygen_covariance() const noexcept
{
    assert(_pointCount > 0);
    const double oneOverCount = 1.0 / static_cast<double>(_pointCount);

    // diagonal
    // The diagonal should always be >= 0 (Cauchy Schwarz)
    // Here it's not always the case because of floating point error accumulation
    const double xxCovariance = std::max(0.0, _Sxs - _Sx * _Sx * oneOverCount);
    const double yyCovariance = std::max(0.0, _Sys - _Sy * _Sy * oneOverCount);
    const double zzCovariance = std::max(0.0, _Szs - _Sz * _Sz * oneOverCount);
    assert(xxCovariance >= 0);
    assert(yyCovariance >= 0);
    assert(zzCovariance >= 0);

    // bottom/top half. As above, this too will have floating point error accumulation but nothing we can do about it
    const double xyCovariance = _Sxy - _Sx * _Sy * oneOverCount;
    const double xzCovariance = _Szx - _Sx * _Sz * oneOverCount;
    const double yzCovariance = _Syz - _Sy * _Sz * oneOverCount;

    // Expressing covariance as E[PP^t] + E[P]*E[P^T]: König-Huygen formula
    matrix33 covariance({{xxCovariance, xyCovariance, xzCovariance},
                         {xyCovariance, yyCovariance, yzCovariance},
                         {xzCovariance, yzCovariance, zzCovariance}});
    return covariance;
}

void Plane_Segment::fit_plane() noexcept
{
    _isPlanar = false;

    assert(_pointCount > 0);
    const double oneOverCount = 1.0 / static_cast<double>(_pointCount);

    // get the centroid of the plane
    _centroid << vector3(_Sx, _Sy, _Sz) * oneOverCount;

    const matrix33 pointCloudCov = get_point_cloud_Huygen_covariance();
    // special case: degenerate covariance
    // if (not utils::is_covariance_valid(pointCloudCov))
    if (utils::double_equal(pointCloudCov.determinant(), 0))
    {
        return;
    }

    // no need to fill the upper part, the adjoint solver does not need it
    Eigen::SelfAdjointEigenSolver<matrix33> eigenSolver(pointCloudCov);
    // eigen values are the point variance along the eigen vectors (sorted by ascending order)
    const vector3& eigenValues = eigenSolver.eigenvalues().cwiseAbs();
    // best eigen vector is the most reliable direction for this plane normal
    const vector3& eigenVector = eigenSolver.eigenvectors().col(0);

    /** Alternative plane parameter computation
    const matrix33& pcc = get_point_cloud_covariance();
    const vector3 e(-_Sx, -_Sy, -_Sz);
    const vector3 params = (pcc * e).transpose();
    const vector3 normal = params / params.norm();
    const double d = 1.0 / params.norm();
    */

    // some values have floating points errors, renormalize
    const vector3& normal = eigenVector.normalized();
    const double d = -normal.dot(_centroid);

    // point normal toward the camera
    if (d <= 0)
        _parametrization = utils::PlaneCoordinates(-normal, -d);
    else
        _parametrization = utils::PlaneCoordinates(normal, d);
    assert(utils::double_equal(normal.norm(), 1.0));

    // variance of points in our plane divided by number of points in the plane
    _MSE = eigenValues(0) * oneOverCount;
    // second best variance divided by variance of this plane patch
    _score = eigenValues(1) / std::max(eigenValues(0), 1e-6);
    // const double curvature = eigenValues(0) / eigenValues.sum();

    // set segment as planar
    _isPlanar = true;
}

/*
 * Sets all the plane parameters to zero
 */
void Plane_Segment::clear_plane_parameters() noexcept
{
    _isPlanar = false;

    _pointCount = 0;
    _score = 0;
    _MSE = std::numeric_limits<double>::max();

    _centroid.setZero();
    _parametrization = utils::PlaneCoordinates();

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

double Plane_Segment::get_cos_angle(const Plane_Segment& p) const noexcept
{
    assert(_isPlanar);
    return _parametrization.get_cos_angle(p._parametrization);
}
double Plane_Segment::get_point_distance(const vector3& point) const noexcept
{
    return _parametrization.get_point_distance(point);
}
double Plane_Segment::get_point_distance_squared(const vector3& point) const noexcept
{
    return _parametrization.get_point_distance_squared(point);
}

bool Plane_Segment::can_be_merged(const Plane_Segment& p, const double maxMatchDistance) const noexcept
{
    constexpr double maximumMergeAngle = cos(parameters::detection::maximumPlaneAngleForMerge_d * M_PI / 180.0);
    return get_cos_angle(p) > maximumMergeAngle and get_point_distance(p.get_centroid()) < maxMatchDistance;
}

} // namespace rgbd_slam::features::primitives
