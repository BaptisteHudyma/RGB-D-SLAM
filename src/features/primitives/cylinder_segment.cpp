#include "cylinder_segment.hpp"
#include "../../parameters.hpp"
#include "../../utils/random.hpp"

namespace rgbd_slam::features::primitives {

Cylinder_Segment::Cylinder_Segment(const Cylinder_Segment& seg, const uint subRegionId) :
    _cellActivatedCount(0),
    _segmentCount(0)
{
    // copy stored data
    _radius.push_back(seg._radius[subRegionId]);
    _centers.push_back(seg._centers[subRegionId]);
    //_pointsAxis1 = seg.pointsAxis1;
    //_pointsAxis2 = seg.pointsAxis2;
    //_normalsAxis1Axis2 = seg._normalsAxis1Axis2;
    _axis = seg._axis;
}

/*
 *  Copy constructor
 */
Cylinder_Segment::Cylinder_Segment(const Cylinder_Segment& seg) :
    _axis(seg._axis),
    _centers(seg._centers),
    _radius(seg._radius),
    _cellActivatedCount(0),
    _segmentCount(0)
{
    //_pointsAxis1 = seg.pointsAxis1;
    //_pointsAxis2 = seg.pointsAxis2;
    //_normalsAxis1Axis2 = seg._normalsAxis1Axis2;
}

Cylinder_Segment::Cylinder_Segment(const std::vector<Plane_Segment>& planeGrid,
                                   const vectorb& isActivatedMask,
                                   const uint cellActivatedCount) :
    _cellActivatedCount(cellActivatedCount),
    _segmentCount(0)
{
    const size_t samplesCount = isActivatedMask.size();
    assert(samplesCount == planeGrid.size());
    assert(_cellActivatedCount <= isActivatedMask.size());

    _local2globalMap.assign(_cellActivatedCount, 0);

    matrixd planeNormals(3, 2 * _cellActivatedCount);
    matrixd planeCentroids(3, _cellActivatedCount);

    // Init. normals and centroids
    uint j = 0;
    for (uint i = 0; i < samplesCount; ++i)
    {
        if (isActivatedMask[i])
        {
            assert(i < planeGrid.size());
            assert(j < _cellActivatedCount);
            assert(j < _local2globalMap.size());

            const vector3& planeNormal = planeGrid[i].get_normal();
            const vector3& planeCentroid = planeGrid[i].get_centroid();
            planeNormals(0, j) = planeNormal.x();
            planeNormals(1, j) = planeNormal.y();
            planeNormals(2, j) = planeNormal.z();
            planeCentroids(0, j) = planeCentroid.x();
            planeCentroids(1, j) = planeCentroid.y();
            planeCentroids(2, j) = planeCentroid.z();

            _local2globalMap[j] = i;
            ++j;
        }
    }

    // Concatenate [Normals -Normals]
    for (uint i = 0; i < samplesCount; ++i)
    {
        if (isActivatedMask[i])
        {
            assert(j < 2 * _cellActivatedCount);
            const vector3& planeNormal = planeGrid[i].get_normal();
            planeNormals(0, j) = -planeNormal.x();
            planeNormals(1, j) = -planeNormal.y();
            planeNormals(2, j) = -planeNormal.z();
            ++j;
        }
    }

    // Compute covariance
    const matrixd cov = (planeNormals * planeNormals.adjoint()) / static_cast<double>(planeNormals.cols() - 1);

    // PCA using QR decomposition for symmetric matrices
    Eigen::SelfAdjointEigenSolver<matrix33> eigenSolver(cov);
    const vector3& eigenValues = eigenSolver.eigenvalues();
    // biggest over smallest eigen value
    const double score = eigenValues(2) / eigenValues(0);

    // Checkpoint 1
    constexpr float minimumCyinderScore = parameters::detection::cylinderRansacMinimumScore;
    if (score < minimumCyinderScore)
    {
        return;
    }

    const vector3& cylinderAxis = eigenSolver.eigenvectors().col(0);
    _axis = cylinderAxis;

    matrixd NCpy = planeNormals.block(0, 0, 3, _cellActivatedCount);
    planeNormals = NCpy; /* This avoids memory issues */
    matrixd projectedCentroids(3, _cellActivatedCount);

    // Projection to plane: P' = P-theta*<P.theta>
    const matrixd& centroidsDotTheta = cylinderAxis.transpose() * planeCentroids;
    projectedCentroids.row(0) = planeCentroids.row(0) - centroidsDotTheta * cylinderAxis(0);
    projectedCentroids.row(1) = planeCentroids.row(1) - centroidsDotTheta * cylinderAxis(1);
    projectedCentroids.row(2) = planeCentroids.row(2) - centroidsDotTheta * cylinderAxis(2);
    const matrixd& normalsDotTheta = cylinderAxis.transpose() * planeNormals;
    planeNormals.row(0) -= normalsDotTheta * cylinderAxis(0);
    planeNormals.row(1) -= normalsDotTheta * cylinderAxis(1);
    planeNormals.row(2) -= normalsDotTheta * cylinderAxis(2);

    // Normalize projected normals
    const matrixd& normalsNorm = planeNormals.colwise().norm();
    planeNormals.row(0) = planeNormals.row(0).array() / normalsNorm.array();
    planeNormals.row(1) = planeNormals.row(1).array() / normalsNorm.array();
    planeNormals.row(2) = planeNormals.row(2).array() / normalsNorm.array();

    // Ransac params
    constexpr float pSuccess =
            parameters::detection::cylinderRansacProbabilityOfSuccess; // probability of selecting only inliers in a
                                                                       // ransac iteration
    constexpr float w = parameters::detection::cylinderRansacInlierProportions; // inliers / all elements
    const static uint maximumIterations = static_cast<uint>(logf(1.0f - pSuccess) / logf(1.0f - powf(w, 3.0f)));

    uint planeSegmentsLeft = _cellActivatedCount;
    Matrixb idsLeftMask(1, _cellActivatedCount);

    std::vector<uint> idsLeft;
    for (uint i = 0; i < _cellActivatedCount; i++)
    {
        idsLeft.push_back(i);
        idsLeftMask(i) = true;
    }
    // Sequential RANSAC main loop
    const size_t minimumCellActivated = static_cast<uint>(parameters::detection::minimumCellActivatedProportion *
                                                          static_cast<double>(samplesCount));
    while (planeSegmentsLeft > minimumCellActivated and planeSegmentsLeft > 0.1 * _cellActivatedCount)
    {
        Matrixb isInlierFinal(true, _cellActivatedCount);
        // RANSAC loop
        const size_t maxInliersCount = run_ransac_loop(
                maximumIterations, idsLeft, planeNormals, projectedCentroids, idsLeftMask, isInlierFinal);

        // Checkpoint 2
        if (maxInliersCount < 6)
            break;

        // Remove cells from list of remaining cells AND compute LLS solution using all inliers
        double b = 0;
        vector3 sumOfNormals = vector3::Zero();
        vector3 sumOfCenters = vector3::Zero();
        idsLeft.clear();
        for (uint i = 0; i < _cellActivatedCount; i++)
        {
            if (isInlierFinal(i))
            {
                // Remove cell from remaining cells
                idsLeftMask(i) = false;
                planeSegmentsLeft--;

                // compute LLS solution using all inliers
                sumOfNormals += planeNormals.col(i);
                sumOfCenters += projectedCentroids.col(i);
                b += (planeNormals.col(i).array() * projectedCentroids.col(i).array()).sum();
            }
            else if (idsLeftMask(i))
            {
                idsLeft.push_back(i);
            }
        }

        const double oneOverMaxInliersCountSquared = 1.0 / static_cast<double>(maxInliersCount * maxInliersCount);
        const double a = 1 - sumOfNormals.squaredNorm() * oneOverMaxInliersCountSquared;
        b /= static_cast<double>(maxInliersCount);
        b -= sumOfNormals.dot(sumOfCenters) * oneOverMaxInliersCountSquared;
        double radius = b / a;
        const matrixd center = (sumOfCenters - radius * sumOfNormals) / maxInliersCount;

        // Rectify radius if concave
        if (radius < 0)
            radius = -radius;

        // Add cylinder
        _segmentCount += 1;
        _radius.push_back(radius);
        _centers.push_back(center);
        _inliers.push_back(isInlierFinal);

        // Save points on axis
        const vector3& P1d = center;
        const vector3& P2d = center + cylinderAxis;
        const double P1P2d = (P2d - P1d).norm();

        // Compute mean squared error
        double mse = 0;
        for (uint i = 0; i < _cellActivatedCount; i++)
        {
            if (isInlierFinal(i))
            {
                const vector3 P3 = planeCentroids.block<3, 1>(0, i);
                // Compute point to line distance
                // Use point-to-line distances (same as cylinder distances)
                // distance = SQR((P3 - center).norm() - radius);
                const double distance = SQR(((P2d - P1d).cross(P3 - P2d)).norm() / P1P2d - radius);
                mse += distance;
            }
        }
        mse /= static_cast<double>(maxInliersCount);
        _MSE.push_back(mse);

        // Save points on axis, useful for computing distances afterwards
        _pointsAxis1.push_back(P1d);
        _pointsAxis2.push_back(P2d);
        _normalsAxis1Axis2.push_back(P1P2d);
    }
}

size_t Cylinder_Segment::run_ransac_loop(const uint maximumIterations,
                                         const std::vector<uint>& idsLeft,
                                         const matrixd& planeNormals,
                                         const matrixd& projectedCentroids,
                                         const Matrixb& idsLeftMask,
                                         Matrixb& isInlierFinal) const noexcept
{
    assert(maximumIterations > 0);
    // not enough ids left
    if (idsLeft.size() < 3)
        return 0;

    const uint planeIdsLeft = static_cast<uint>(idsLeft.size());
    const uint inliersAcceptedCount = static_cast<uint>(std::floor(0.9 * planeIdsLeft));

    constexpr float maximumSqrtDistance = parameters::detection::cylinderRansacSqrtMaxDistance;
    // Score of the maximum inliers configuration
    double minHypothesisDist = maximumSqrtDistance * static_cast<float>(planeIdsLeft);
    // Indexes of the inliers of the best configuration
    std::vector<uint> finalInlierIndexes;

    // Run ransac loop
    for (uint iteration = 0; iteration < maximumIterations; ++iteration)
    {
        // Random triplet
        const uint id1 = idsLeft[utils::Random::get_random_uint(planeIdsLeft)];
        const uint id2 = idsLeft[utils::Random::get_random_uint(planeIdsLeft)];
        const uint id3 = idsLeft[utils::Random::get_random_uint(planeIdsLeft)];
        // normals of random planes
        const vector3& normal1 = planeNormals.col(id1);
        const vector3& normal2 = planeNormals.col(id2);
        const vector3& normal3 = planeNormals.col(id3);
        // centers of random planes
        const vector3& centroid1 = projectedCentroids.col(id1);
        const vector3& centroid2 = projectedCentroids.col(id2);
        const vector3& centroid3 = projectedCentroids.col(id3);

        // Sum of normals/centroids
        const vector3 sumOfNormals = (normal1 + normal2 + normal3);
        const vector3 sumOfCenters = (centroid1 + centroid2 + centroid3);

        // LLS solution for triplets
        const double a = 1.0 - sumOfNormals.squaredNorm() / 9.0;
        const double b = ((normal1.array() * centroid1.array()) + (normal2.array() * centroid2.array()) +
                          (normal3.array() * centroid3.array()))
                                         .sum() /
                                 3.0 -
                         (sumOfNormals.dot(sumOfCenters) / 9.0);
        // compute cylinder center and radius
        const double radius = b / a;
        const double oneOverRadiusSquared = 1.0 / (radius * radius);
        const vector3 center = (sumOfCenters - radius * sumOfNormals) / 3.0;

        // MSAC truncated distance
        std::vector<uint> inlierIndexes;
        double dist = 0.0;
        for (uint i = 0; i < _cellActivatedCount; ++i)
        {
            if (not idsLeftMask(i))
            {
                continue;
            }
            // Normal dist
            const double distance =
                    ((projectedCentroids.col(i) - radius * planeNormals.col(i)) - center.col(0)).squaredNorm() *
                    oneOverRadiusSquared;
            if (distance < maximumSqrtDistance)
            {
                dist += distance;
                inlierIndexes.push_back(i);
            }
            else
            {
                dist += maximumSqrtDistance;
            }
        }

        if (dist < minHypothesisDist)
        {
            // Keep parameters of the best transformation
            minHypothesisDist = dist;
            finalInlierIndexes.swap(inlierIndexes);

            // early stop
            if (inlierIndexes.size() > inliersAcceptedCount)
                break;
        }
    }

    // Compute the final inliers set
    isInlierFinal.setConstant(false);
    for (const uint inlierIndex: finalInlierIndexes)
        isInlierFinal(inlierIndex) = true;

    return finalInlierIndexes.size();
}

double Cylinder_Segment::get_distance(const vector3& point) const noexcept
{
    double minDist = this->get_distance(point, 0);
    const size_t pointAxisSize = _pointsAxis1.size();
    for (vector3_vector::size_type i = 1; i < pointAxisSize; ++i)
    {
        const double nd = this->get_distance(point, i);
        if (minDist > nd)
            minDist = nd;
    }
    return minDist;
}

double Cylinder_Segment::get_distance(const vector3& point, const size_t segmentId) const noexcept
{
    const vector3 pointAxis2to1 = _pointsAxis2[segmentId] - _pointsAxis1[segmentId];
    const vector3 pointAxisTo2 = point - _pointsAxis2[segmentId];
    const double pointAxisNorm = (pointAxis2to1.cross(pointAxisTo2)).norm();
    return pointAxisNorm / _normalsAxis1Axis2[segmentId] - _radius[segmentId];
}

/*
 *  Getters
 */
uint Cylinder_Segment::get_segment_count() const noexcept { return _segmentCount; }

double Cylinder_Segment::get_MSE_at(const uint index) const noexcept
{
    assert(index < _MSE.size());
    return _MSE[index];
}

bool Cylinder_Segment::is_inlier_at(const uint indexA, const uint indexB) const noexcept
{
    assert(indexA < _inliers.size() and indexB < _inliers[indexA].size());
    return _inliers[indexA](indexB);
}

uint Cylinder_Segment::get_local_to_global_mapping(const uint index) const noexcept
{
    assert(index < _local2globalMap.size());
    return _local2globalMap[index];
}

const vector3& Cylinder_Segment::get_axis1_point(const uint index) const noexcept
{
    assert(index < _pointsAxis1.size());
    return _pointsAxis1[index];
}

const vector3& Cylinder_Segment::get_axis2_point(const uint index) const noexcept
{
    assert(index < _pointsAxis2.size());
    return _pointsAxis2[index];
}

double Cylinder_Segment::get_axis_normal(const uint index) const noexcept
{
    assert(index < _normalsAxis1Axis2.size());
    return _normalsAxis1Axis2[index];
}

double Cylinder_Segment::get_radius(const uint index) const noexcept
{
    assert(index < _radius.size());
    return _radius[index];
}

double Cylinder_Segment::get_normal_similarity(const Cylinder_Segment& other) const noexcept
{
    return std::abs(_axis.dot(other._axis));
}

vector3 Cylinder_Segment::get_normal() const noexcept { return _axis; }

Cylinder_Segment::~Cylinder_Segment() { _cellActivatedCount = 0; }

} // namespace rgbd_slam::features::primitives
