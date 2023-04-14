#include "primitive_detection.hpp"
#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "coordinates.hpp"
#include "cylinder_segment.hpp"
#include "distance_utils.hpp"
#include "plane_segment.hpp"
#include "shape_primitives.hpp"
#include "types.hpp"
#include <Eigen/src/Core/Array.h>
#include <limits>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>

namespace rgbd_slam::features::primitives {

Primitive_Detection::Primitive_Detection(const uint width, const uint height, const uint blocSize) :
    _histogram(blocSize),
    _width(width),
    _height(height),
    _pointsPerCellCount(blocSize * blocSize),
    _horizontalCellsCount(_width / blocSize),
    _verticalCellsCount(_height / blocSize),
    _totalCellCount(_verticalCellsCount * _horizontalCellsCount)
{
    assert(blocSize > 0);
    assert(width > 0);
    assert(height > 0);

    // Init variables
    _isUnassignedMask = vectorb::Zero(_totalCellCount);
    _cellDistanceTols.assign(_totalCellCount, 0.0f);

    _gridPlaneSegmentMap =
            cv::Mat_<int>(static_cast<int>(_verticalCellsCount), static_cast<int>(_horizontalCellsCount), 0);
    _gridCylinderSegMap =
            cv::Mat_<int>(static_cast<int>(_verticalCellsCount), static_cast<int>(_horizontalCellsCount), 0);

    _mask = cv::Mat(static_cast<int>(_verticalCellsCount), static_cast<int>(_horizontalCellsCount), CV_8U);
    _maskEroded = cv::Mat(static_cast<int>(_verticalCellsCount), static_cast<int>(_horizontalCellsCount), CV_8U);
    _maskDilated = cv::Mat(static_cast<int>(_verticalCellsCount), static_cast<int>(_horizontalCellsCount), CV_8U);
    _maskBoundary = cv::Mat(static_cast<int>(_verticalCellsCount), static_cast<int>(_horizontalCellsCount), CV_8U);

    _maskCrossKernel = cv::Mat::ones(3, 3, CV_8U);
    _maskCrossKernel.at<uchar>(0, 0) = 0;
    _maskCrossKernel.at<uchar>(2, 2) = 0;
    _maskCrossKernel.at<uchar>(0, 2) = 0;
    _maskCrossKernel.at<uchar>(2, 0) = 0;

    _maskSquareKernel = cv::Mat::ones(3, 3, CV_8U);

    // set before anything related to planar cells
    Plane_Segment::set_static_members(blocSize, _pointsPerCellCount);

    // array of unique_ptr<Plane_Segment>
    const Plane_Segment defaultPlaneSegment = Plane_Segment();
    _planeGrid.reserve(_totalCellCount);
    for (uint i = 0; i < _totalCellCount; ++i)
    {
        // fill with empty nodes
        _planeGrid.push_back(defaultPlaneSegment);
    }

    // perf measurments
    resetTime = 0;
    initTime = 0;
    growTime = 0;
    mergeTime = 0;
    refineTime = 0;
}

void Primitive_Detection::find_primitives(const matrixf& depthMatrix,
                                          plane_container& planeContainer,
                                          cylinder_container& primitiveContainer)
{
    // reset used data structures
    reset_data();

    int64 t1 = cv::getTickCount();
    // init planar grid
    init_planar_cell_fitting(depthMatrix);
    double td = static_cast<double>(cv::getTickCount() - t1) / cv::getTickFrequency();
    resetTime += td;

    // init and fill histogram
    t1 = cv::getTickCount();
    const uint remainingPlanarCells = init_histogram();
    td = static_cast<double>(cv::getTickCount() - t1) / cv::getTickFrequency();
    initTime += td;

    t1 = cv::getTickCount();
    const intpair_vector& cylinder2regionMap = grow_planes_and_cylinders(remainingPlanarCells);
    td = static_cast<double>(cv::getTickCount() - t1) / cv::getTickFrequency();
    growTime += td;

    // merge sparse planes
    t1 = cv::getTickCount();
    const uint_vector& planeMergeLabels = merge_planes();
    td = static_cast<double>(cv::getTickCount() - t1) / cv::getTickFrequency();
    initTime += td;
    mergeTime += td;

    t1 = cv::getTickCount();
    // fill the final planes vector
    add_planes_to_primitives(planeMergeLabels, depthMatrix, planeContainer);
    td = static_cast<double>(cv::getTickCount() - t1) / cv::getTickFrequency();
    initTime += td;
    refineTime += td;

    t1 = cv::getTickCount();
    // refine cylinders boundaries and fill the final cylinders vector
    add_cylinders_to_primitives(cylinder2regionMap, primitiveContainer);
    td = static_cast<double>(cv::getTickCount() - t1) / cv::getTickFrequency();
    initTime += td;
    refineTime += td;
}

void Primitive_Detection::reset_data()
{
    _histogram.reset();

    // planeGrid SHOULD NOT be cleared
    _planeSegments.clear();
    _cylinderSegments.clear();

    _gridPlaneSegmentMap = 0;
    _gridCylinderSegMap = 0;

    // reset stacked distances
    // activation map do not need to be cleared
    _isUnassignedMask = vectorb::Zero(_isUnassignedMask.size());
    std::fill_n(_cellDistanceTols.begin(), _cellDistanceTols.size(), 0.0f);

    // mat masks do not need to be cleared
    // kernels should not be cleared
}

void Primitive_Detection::init_planar_cell_fitting(const matrixf& depthCloudArray)
{
    const static float sinAngleForMerge =
            sinf(static_cast<float>(Parameters::get_maximum_plane_merge_angle() * M_PI / 180.0));

    // for each planeGrid cell
    const size_t planeGridSize = _planeGrid.size();
    for (size_t stackedCellId = 0; stackedCellId < planeGridSize; ++stackedCellId)
    {
        // init the plane patch
        Plane_Segment& planeSegment = _planeGrid[stackedCellId];
        planeSegment.init_plane_segment(depthCloudArray, static_cast<uint>(stackedCellId));
        // if this plane patch is planar, compute the diagonal distance
        if (planeSegment.is_planar())
        {
            const uint offset = static_cast<uint>(std::floor(stackedCellId * _pointsPerCellCount));
            // cell diameter, in millimeters
            const float cellDiameter = (
                                               // right down corner (x, y, z)
                                               depthCloudArray.block(offset + _pointsPerCellCount - 1, 0, 1, 3) -
                                               // left up corner (x, y, z)
                                               depthCloudArray.block(offset, 0, 1, 3))
                                               .norm();
            // merge distance threshold (from "2021 - Real Time Plane Detection with Consistency from Point Cloud
            // Sequences") use the plane diameter as a merge threshold, with a small error (1.5)
            _cellDistanceTols[stackedCellId] =
                    1.5f * cellDiameter * sinAngleForMerge * sqrtf(static_cast<float>(planeSegment.get_point_count()));
        }
    }
#if 0
// use this to debug the initial is_planar function
                // Resize with no interpolation
                _mask = cv::Scalar(0);
                for(uint row = 0, activationIndex = 0; row < _verticalCellsCount; ++row) 
                {
                    for(uint col = 0; col < _horizontalCellsCount; ++col, ++activationIndex)
                        _mask.at<uchar>(row, col) = _planeGrid[activationIndex].is_planar();
                }
                cv::Mat planeMask;
                cv::resize(_mask * 255, planeMask, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
                cv::imshow("is_depth_continuous", planeMask);
#endif
}

uint Primitive_Detection::init_histogram()
{
    uint remainingPlanarCells = 0;
    matrixd histBins(_totalCellCount, 2);

    const size_t planeGridSize = _planeGrid.size();
    for (uint cellId = 0; cellId < planeGridSize; ++cellId)
    {
        const Plane_Segment& planePatch = _planeGrid[cellId];
        if (planePatch.is_planar())
        {
            const vector3& planeNormal = planePatch.get_normal();
            const double nx = planeNormal.x();
            const double ny = planeNormal.y();
            histBins(cellId, 0) = acos(-planeNormal.z());
            histBins(cellId, 1) = atan2(nx, ny);

            assert(not std::isnan(histBins(cellId, 0)));
            assert(not std::isnan(histBins(cellId, 1)));

            ++remainingPlanarCells;
            _isUnassignedMask[cellId] = true;
        }
    }
    _histogram.init_histogram(histBins, _isUnassignedMask);
    return remainingPlanarCells;
}

Primitive_Detection::intpair_vector Primitive_Detection::grow_planes_and_cylinders(const uint remainingPlanarCells)
{
    intpair_vector cylinder2regionMap;

    uint untriedPlanarCellsCount = remainingPlanarCells;
    // find seed planes and make them grow
    while (untriedPlanarCellsCount > 0)
    {
        // get seed candidates
        const std::vector<uint>& seedCandidates = _histogram.get_points_from_most_frequent_bin();
        if (const static uint planeSeedCount =
                    static_cast<uint>(Parameters::get_minimum_plane_seed_proportion() * _totalCellCount);
            seedCandidates.size() < planeSeedCount)
        {
            break;
        }

        // select seed cell with min MSE
        uint seedId = 0; // should not necessarily stay to 0 after the loop
        double minMSE = std::numeric_limits<double>::max();
        for (const uint seedCandidate: seedCandidates)
        {
            const double candidateMSE = _planeGrid[seedCandidate].get_MSE();
            if (candidateMSE >= minMSE)
                continue;

            seedId = seedCandidate;
            minMSE = candidateMSE;
            if (minMSE <= 0)
                break;
        }
        if (minMSE >= std::numeric_limits<double>::max())
        {
            // seedId is invalid
            outputs::log_warning("Could not find a single plane segment: invalid seed");
            break;
        }

        // try to grow the selected plane at seedId
        grow_plane_segment_at_seed(seedId, untriedPlanarCellsCount, cylinder2regionMap);
    }
    return cylinder2regionMap;
}

void Primitive_Detection::grow_plane_segment_at_seed(const uint seedId,
                                                     uint& untriedPlanarCellsCount,
                                                     intpair_vector& cylinder2regionMap)
{
    assert(seedId < _planeGrid.size());
    const Plane_Segment& planeToGrow = _planeGrid[seedId];
    if (not planeToGrow.is_planar())
    {
        // cannot grow a non planar patch
        return;
    }

    // copy plane segment in new object, to try to grow it in a non destructive way
    Plane_Segment newPlaneSegment(planeToGrow);

    // Seed cell growing
    const uint y = seedId / _horizontalCellsCount;
    const uint x = seedId % _horizontalCellsCount;

    // activationMap set to false (will have bits at true when a plane segment will be merged to this one)
    vectorb isActivatedMap = vectorb::Zero(_totalCellCount);
    const size_t activationMapSize = isActivatedMap.size();
    // grow plane region, fill isActivatedMap
    region_growing(x, y, newPlaneSegment, isActivatedMap);

    assert(activationMapSize == static_cast<size_t>(_isUnassignedMask.size()));
    assert(activationMapSize == _planeGrid.size());

    // merge activated cells & remove them from histogram
    uint cellActivatedCount = 0;
    bool isPlaneFitable = false;
    for (uint planeSegmentIndex = 0; planeSegmentIndex < activationMapSize; ++planeSegmentIndex)
    {
        if (isActivatedMap[planeSegmentIndex])
        {
            const Plane_Segment& planeSegment = _planeGrid[planeSegmentIndex];
            if (planeSegment.is_planar())
            {
                newPlaneSegment.expand_segment(planeSegment);
                ++cellActivatedCount;
                _histogram.remove_point(planeSegmentIndex);
                _isUnassignedMask[planeSegmentIndex] = false;

                assert(untriedPlanarCellsCount > 0);
                --untriedPlanarCellsCount;
                isPlaneFitable = true;
            }
        }
    }

    if (const static uint minimumCellActivated =
                static_cast<uint>(Parameters::get_minimum_cell_activated_proportion() * _totalCellCount);
        not isPlaneFitable or cellActivatedCount < minimumCellActivated)
    {
        _histogram.remove_point(seedId);
        return;
    }

    // fit plane to merged data
    newPlaneSegment.fit_plane();

    // set it as plane or cylinder
    // TODO: why 100 ? seems random
    if (newPlaneSegment.get_score() > 100)
    {
        add_plane_segment_to_features(newPlaneSegment, isActivatedMap);
    }
    // TODO: why 5 ? seems random
    else if (cellActivatedCount > 5)
    {
        cylinder_fitting(cellActivatedCount, isActivatedMap, cylinder2regionMap);
    }
}

void Primitive_Detection::add_plane_segment_to_features(const Plane_Segment& newPlaneSegment,
                                                        const vectorb& isActivatedMap)
{
    const size_t activationMapSize = isActivatedMap.size();

    // its certainly a plane or we ignore cylinder detection
    _planeSegments.push_back(newPlaneSegment);
    const int currentPlaneCount = static_cast<int>(_planeSegments.size());
    // mark cells that belong to this plane with a new id
    for (int row = 0, activationIndex = 0; row < static_cast<int>(_verticalCellsCount); ++row)
    {
        int* rowPtr = _gridPlaneSegmentMap.ptr<int>(row);
        for (int col = 0; col < static_cast<int>(_horizontalCellsCount); ++col, ++activationIndex)
        {
            assert(activationIndex < static_cast<int>(activationMapSize));

            if (isActivatedMap[activationIndex])
                rowPtr[col] = currentPlaneCount;
        }
    }
}

bool Primitive_Detection::find_plane_segment_in_cylinder(const Cylinder_Segment& cylinderSegment,
                                                         const uint cellActivatedCount,
                                                         const uint segId,
                                                         Plane_Segment& newMergedPlane)
{
    bool isPlaneSegmentFitable = false;
    for (uint col = 0; col < cellActivatedCount; ++col)
    {
        if (cylinderSegment.is_inlier_at(segId, col))
        {
            const uint localMapIndex = cylinderSegment.get_local_to_global_mapping(col);
            assert(localMapIndex < _planeGrid.size());

            if (const Plane_Segment& planeSegment = _planeGrid[localMapIndex]; planeSegment.is_planar())
            {
                newMergedPlane.expand_segment(planeSegment);
                isPlaneSegmentFitable = true;
            }
        }
    }
    return isPlaneSegmentFitable;
}

void Primitive_Detection::add_cylinder_to_features(const Cylinder_Segment& cylinderSegment,
                                                   const uint cellActivatedCount,
                                                   const uint segId,
                                                   const Plane_Segment& newMergedPlane,
                                                   intpair_vector& cylinder2regionMap)
{
    // Model selection based on MSE
    if (newMergedPlane.get_MSE() < cylinderSegment.get_MSE_at(segId))
    {
        // MSE of the plane is less than MSE of the cylinder + this plane, so keep this one as a plane
        _planeSegments.push_back(newMergedPlane);
        const int currentPlaneCount = static_cast<int>(_planeSegments.size());
        for (uint col = 0; col < cellActivatedCount; ++col)
        {
            if (cylinderSegment.is_inlier_at(segId, col))
            {
                const int cellId = static_cast<int>(cylinderSegment.get_local_to_global_mapping(col));
                _gridPlaneSegmentMap.at<int>(cellId / static_cast<int>(_horizontalCellsCount),
                                             cellId % static_cast<int>(_horizontalCellsCount)) = currentPlaneCount;
            }
        }
    }
    else
    {
        // Set a new cylinder
        assert(_cylinderSegments.size() > 0);
        cylinder2regionMap.push_back(std::make_pair(_cylinderSegments.size() - 1, segId));
        const int cylinderCount = static_cast<int>(cylinder2regionMap.size());

        for (uint col = 0; col < cellActivatedCount; ++col)
        {
            if (cylinderSegment.is_inlier_at(segId, col))
            {
                const int cellId = static_cast<int>(cylinderSegment.get_local_to_global_mapping(col));
                _gridCylinderSegMap.at<int>(cellId / static_cast<int>(_horizontalCellsCount),
                                            cellId % static_cast<int>(_horizontalCellsCount)) = cylinderCount;
            }
        }
    }
}

void Primitive_Detection::cylinder_fitting(const uint cellActivatedCount,
                                           const vectorb& isActivatedMap,
                                           intpair_vector& cylinder2regionMap)
{
    // try cylinder fitting on the activated planes
    const Cylinder_Segment& cylinderSegment = Cylinder_Segment(_planeGrid, isActivatedMap, cellActivatedCount);
    // TODO: emplace back
    _cylinderSegments.push_back(cylinderSegment);

    // Fit planes to subsegments
    const uint cylinderSegmentCount = cylinderSegment.get_segment_count();
    for (uint segId = 0; segId < cylinderSegmentCount; ++segId)
    {
        Plane_Segment newMergedPlane;
        // No continuous planes, pass
        if (not find_plane_segment_in_cylinder(cylinderSegment, cellActivatedCount, segId, newMergedPlane))
            continue;

        newMergedPlane.fit_plane();

        add_cylinder_to_features(cylinderSegment, cellActivatedCount, segId, newMergedPlane, cylinder2regionMap);
    }
}

Primitive_Detection::uint_vector Primitive_Detection::merge_planes()
{
    const uint planeCount = static_cast<uint>(_planeSegments.size());

    Matrixb isPlanesConnectedMatrix = get_connected_components_matrix(_gridPlaneSegmentMap, planeCount);
    assert(isPlanesConnectedMatrix.rows() == isPlanesConnectedMatrix.cols());

    uint_vector planeMergeLabels;
    planeMergeLabels.reserve(planeCount);
    for (uint planeIndex = 0; planeIndex < planeCount; ++planeIndex)
        // We use planes indexes as ids
        planeMergeLabels.push_back(planeIndex);

    const uint isPlanesConnectedMatrixRows = static_cast<uint>(isPlanesConnectedMatrix.rows());
    const uint isPlanesConnectedMatrixCols = static_cast<uint>(isPlanesConnectedMatrix.cols());
    for (uint row = 0; row < isPlanesConnectedMatrixRows; ++row)
    {
        bool wasPlaneExpanded = false;
        const uint planeId = planeMergeLabels[row];
        Plane_Segment& planeToExpand = _planeSegments[planeId];
        if (not planeToExpand.is_planar())
            continue;

        for (uint col = row + 1; col < isPlanesConnectedMatrixCols; ++col)
        {
            if (not isPlanesConnectedMatrix(row, col))
                continue;

            const Plane_Segment& mergePlane = _planeSegments[col];
            if (not mergePlane.is_planar())
                continue;

            // normals are close enough, distance is small enough
            if (planeToExpand.can_be_merged(mergePlane, _cellDistanceTols[col]))
            {
                // merge plane segments
                planeToExpand.expand_segment(mergePlane);
                planeMergeLabels[col] = planeId;
                wasPlaneExpanded = true;
            }
            else
            {
                isPlanesConnectedMatrix(row, col) = false;
                isPlanesConnectedMatrix(col, row) = false;
            }
        }
        if (wasPlaneExpanded) // plane was merged with other planes
            planeToExpand.fit_plane();
    }

    return planeMergeLabels;
}

void Primitive_Detection::add_planes_to_primitives(const uint_vector& planeMergeLabels,
                                                   const matrixf& depthMatrix,
                                                   plane_container& planeContainer)
{
    const uint planeCount = static_cast<uint>(_planeSegments.size());
    planeContainer.clear();
    planeContainer.reserve(planeCount);

    // refine the coarse planes boundaries to smoother versions
    for (uint planeIndex = 0; planeIndex < planeCount; ++planeIndex)
    {
        // index of this plane merge index
        const uint planeMergeLabel = planeMergeLabels[planeIndex];
        if (planeIndex != planeMergeLabel)
            continue; // plane should be merged by another plane index

        const Plane_Segment& planeSegment = _planeSegments[planeIndex];
        if (not planeSegment.is_planar())
            continue; // not planar segment: TODO: remove ?

        _mask = cv::Scalar(0);
        // add all merged planes to the mask
        for (uint j = planeIndex; j < planeCount; ++j)
        {
            if (planeMergeLabels[j] == planeMergeLabel)
                _mask.setTo(1, _gridPlaneSegmentMap == (j + 1));
        }

        // erode considering the border as an obstacle
        cv::erode(_mask, _maskEroded, _maskCrossKernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        double min;
        double max;
        cv::minMaxLoc(_maskEroded, &min, &max);

        if (max <= 0 or min >= max) // completely eroded: irrelevant plane
            continue;

        // dilate to get boundaries
        cv::dilate(_mask, _maskDilated, _maskSquareKernel);
        _maskBoundary = _maskDilated - _maskEroded;

        // add new plane to final shapes
        planeContainer.emplace_back(planeSegment,
                                    compute_plane_segment_boundary(planeSegment, depthMatrix, _maskBoundary));
    }
}

utils::CameraPolygon Primitive_Detection::compute_plane_segment_boundary(const Plane_Segment& planeSegment,
                                                                         const matrixf& depthMatrix,
                                                                         const cv::Mat& boundaryMask) const
{
    const vector3& normal = planeSegment.get_normal();
    // TODO: To reactivate with the TODO in update_boundary_polygon is resolved
    // const utils::CameraCoordinate& center = -(planeSegment.get_normal() * planeSegment.get_plane_d());
    const utils::CameraCoordinate& center = planeSegment.get_centroid();

    std::vector<vector3> boundaryPoints;
    // Cell refinement
    for (uint cellRow = 0, stackedCellId = 0; cellRow < _verticalCellsCount; ++cellRow)
    {
        const uchar* boundary = boundaryMask.ptr<uchar>(static_cast<int>(cellRow));
        for (uint cellColum = 0; cellColum < _horizontalCellsCount; ++cellColum, ++stackedCellId)
        {
            // not on plane boundary
            if (boundary[cellColum] <= 0)
                continue;

            // get points from this plane patch
            const uint offset = stackedCellId * _pointsPerCellCount;
            const Eigen::ArrayXf& xMatrix = depthMatrix.block(offset, 0, _pointsPerCellCount, 1).array();
            const Eigen::ArrayXf& yMatrix = depthMatrix.block(offset, 1, _pointsPerCellCount, 1).array();
            const Eigen::ArrayXf& zMatrix = depthMatrix.block(offset, 2, _pointsPerCellCount, 1).array();
            assert(xMatrix.size() == yMatrix.size() and xMatrix.size() == zMatrix.size());

            const std::vector<vector3>& definingPoints = find_defining_points(planeSegment, xMatrix, yMatrix, zMatrix);
            // add to boundary points
            boundaryPoints.insert(boundaryPoints.cend(), definingPoints.cbegin(), definingPoints.cend());
        }
    }

    // construct a polygon from those points
    return utils::CameraPolygon(boundaryPoints, normal, center);
}

std::vector<vector3> Primitive_Detection::find_defining_points(const Plane_Segment& planeSegment,
                                                               const Eigen::ArrayXf& xMatrix,
                                                               const Eigen::ArrayXf& yMatrix,
                                                               const Eigen::ArrayXf& zMatrix) const
{
    // TODO: set in parameters
    const double maxBoundaryDistance = 9 * planeSegment.get_MSE();
    const vector3& center = planeSegment.get_centroid();

    std::vector<vector3> definingPoints;

    // get point farthest to plane centroid and out of plane centroid
    double farthestDist = 0;
    vector3 farthestPoint = vector3::Zero();
    for (uint i = 0; i < xMatrix.size(); i++)
    {
        const vector3 point(xMatrix(i), yMatrix(i), zMatrix(i));
        if (point.z() <= 0) // ignore invalid depth
            continue;

        // if distance of this point to the plane < threshold, this point is contained in the plane
        if (planeSegment.get_point_distance(point) < maxBoundaryDistance)
        {
            const double dist = (point - center).lpNorm<2>();
            if (dist > farthestDist)
            {
                farthestDist = dist;
                farthestPoint = point;
            }
        }
    }
    if (farthestDist > 0 and not farthestPoint.isZero())
        definingPoints.push_back(farthestPoint);

    return definingPoints;
}

void Primitive_Detection::add_cylinders_to_primitives(const intpair_vector& cylinderToRegionMap,
                                                      cylinder_container& cylinderContainer)
{
    const size_t numberOfCylinder = cylinderToRegionMap.size();
    cylinderContainer.clear();
    cylinderContainer.reserve(numberOfCylinder);

    for (uint cylinderIndex = 0; cylinderIndex < numberOfCylinder; ++cylinderIndex)
    {
        // Build mask
        _mask = cv::Scalar(0);
        _mask.setTo(1, _gridCylinderSegMap == (cylinderIndex + 1));

        // Opening
        cv::dilate(_mask, _mask, _maskCrossKernel);
        cv::erode(_mask, _mask, _maskCrossKernel);
        cv::erode(_mask, _maskEroded, _maskCrossKernel);
        double min;
        double max;
        cv::minMaxLoc(_maskEroded, &min, &max);

        if (max <= 0 or min >= max) // completely eroded: irrelevant cylinder
            continue;

        const uint regId = cylinderToRegionMap[cylinderIndex].first;

        // add new cylinder to final shapes
        cylinderContainer.emplace_back(_cylinderSegments[regId]);
    }
}

Matrixb Primitive_Detection::get_connected_components_matrix(const cv::Mat& segmentMap,
                                                             const size_t numberOfPlanes) const
{
    assert(segmentMap.rows > 0);
    assert(segmentMap.cols > 0);

    Matrixb isPlanesConnectedMatrix =
            Matrixb::Constant(static_cast<int>(numberOfPlanes), static_cast<int>(numberOfPlanes), false);
    if (numberOfPlanes == 0)
        return isPlanesConnectedMatrix;

    const int rows2scanCount = segmentMap.rows - 1;
    const int cols2scanCount = segmentMap.cols - 1;
    for (int row = 0; row < rows2scanCount; ++row)
    {
        const int* rowPtr = segmentMap.ptr<int>(row);
        const int* rowBelowPtr = segmentMap.ptr<int>(row + 1);
        for (int col = 0; col < cols2scanCount; ++col)
        {
            // value of the pixel at this coordinates. Represents a plane segment
            const int planeId = rowPtr[col];
            if (planeId <= 0)
                continue;

            const int nextPlaneId = rowPtr[col + 1];
            const int belowPlaneId = rowBelowPtr[col];
            if (nextPlaneId > 0 and planeId != nextPlaneId)
            {
                isPlanesConnectedMatrix(planeId - 1, nextPlaneId - 1) = true;
                isPlanesConnectedMatrix(nextPlaneId - 1, planeId - 1) = true;
            }
            if (belowPlaneId > 0 and planeId != belowPlaneId)
            {
                isPlanesConnectedMatrix(planeId - 1, belowPlaneId - 1) = true;
                isPlanesConnectedMatrix(belowPlaneId - 1, planeId - 1) = true;
            }
        }
    }

    return isPlanesConnectedMatrix;
}

void Primitive_Detection::region_growing(const uint x,
                                         const uint y,
                                         const Plane_Segment& planeToExpand,
                                         vectorb& isActivatedMap)
{
    assert(isActivatedMap.size() == _isUnassignedMask.size());
    assert(_horizontalCellsCount > 0);

    const int index = static_cast<int>(x + _horizontalCellsCount * y);
    if (static_cast<size_t>(index) >= _totalCellCount)
        return;

    assert(index < isActivatedMap.size());
    assert(index < _isUnassignedMask.size());
    if ((not _isUnassignedMask[index]) or isActivatedMap[index])
        // pixel is not part of a component or already labelled, or not a plane (_isUnassignedMask is always false
        // for non planar patches)
        return;

    assert(static_cast<size_t>(index) < _planeGrid.size());
    const Plane_Segment& planePatch = _planeGrid[index];
    if (planeToExpand.can_be_merged(planePatch, _cellDistanceTols[index]))
    {
        // mark this plane as merged
        isActivatedMap[index] = true;

        // Now label the 4 neighbours:
        if (x > 0)
            region_growing(x - 1, y, planePatch, isActivatedMap); // left  pixel
        if (x < _width - 1)
            region_growing(x + 1, y, planePatch, isActivatedMap); // right pixel
        if (y > 0)
            region_growing(x, y - 1, planePatch, isActivatedMap); // upper pixel
        if (y < _height - 1)
            region_growing(x, y + 1, planePatch, isActivatedMap); // lower pixel
    }
    // else: do not merge this plane segment
}

} // namespace rgbd_slam::features::primitives
