#include "primitive_detection.hpp"
#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "coordinates.hpp"
#include "covariances.hpp"
#include "cylinder_segment.hpp"
#include "distance_utils.hpp"
#include "plane_segment.hpp"
#include "random.hpp"
#include "shape_primitives.hpp"
#include "types.hpp"
#include <Eigen/src/Core/Array.h>
#include <algorithm>
#include <atomic>
#include <bits/ranges_algo.h>
#include <cstddef>
#include <limits>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace rgbd_slam::features::primitives {

Primitive_Detection::Primitive_Detection(const uint width, const uint height, const uint blocSize) :
    _histogram(blocSize),
    _pointsPerCellCount(blocSize * blocSize),
    _horizontalCellsCount(width / blocSize),
    _verticalCellsCount(height / blocSize),
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

    _mask = cv::Mat_<uchar>(static_cast<int>(_verticalCellsCount), static_cast<int>(_horizontalCellsCount));

    _maskCrossKernel = cv::Mat_<uchar>::ones(3, 3);
    _maskCrossKernel.at<uchar>(0, 0) = 0;
    _maskCrossKernel.at<uchar>(2, 2) = 0;
    _maskCrossKernel.at<uchar>(0, 2) = 0;
    _maskCrossKernel.at<uchar>(2, 0) = 0;

    _maskSquareKernel = cv::Mat_<uchar>::ones(3, 3);

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
                                          const cv::Mat_<float>& depthImage,
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
    add_planes_to_primitives(planeMergeLabels, depthImage, planeContainer);
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
    _isUnassignedMask = vectorb::Constant(_isUnassignedMask.size(), false);

    // mat masks do not need to be cleared
    // kernels should not be cleared
}

void Primitive_Detection::init_planar_cell_fitting(const matrixf& depthCloudArray)
{
    const static float sinAngleForMerge =
            sinf(static_cast<float>(Parameters::get_maximum_plane_merge_angle() * M_PI / 180.0));
    const static float planeMergeDistanceThreshold = Parameters::get_maximum_plane_merge_distance();

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
                                               depthCloudArray.block<1, 3>(offset + _pointsPerCellCount - 1, 0) -
                                               // left up corner (x, y, z)
                                               depthCloudArray.block<1, 3>(offset, 0))
                                               .norm();
            // merge distance threshold (from "2021 - Real Time Plane Detection with Consistency from Point Cloud
            // Sequences") use the plane diameter as a merge threshold, with a small error (1.5)
            _cellDistanceTols[stackedCellId] = std::min(
                    planeMergeDistanceThreshold,
                    cellDiameter * sinAngleForMerge * sqrtf(static_cast<float>(planeSegment.get_point_count())));
        }
        else
        {
            _cellDistanceTols[stackedCellId] = 0;
        }
    }
#if 0
    // use this to debug the initial is_planar function
    // Resize with no interpolation
    _mask = 0;
    for (uint row = 0, activationIndex = 0; row < _verticalCellsCount; ++row)
    {
        for (uint col = 0; col < _horizontalCellsCount; ++col, ++activationIndex)
        {
            _mask.at<uchar>(row, col) = _planeGrid[activationIndex].is_planar() * 255;
        }
    }
    cv::Mat_<uchar> planeMask;
    cv::resize(_mask, planeMask, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
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
        const static uint planeSeedCount =
                static_cast<uint>(Parameters::get_minimum_plane_seed_proportion() * _totalCellCount);
        if (seedCandidates.size() < planeSeedCount)
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
    vectorb isActivatedMap = vectorb::Constant(_totalCellCount, false);
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

    const static uint minimumCellActivated =
            static_cast<uint>(Parameters::get_minimum_cell_activated_proportion() * _totalCellCount);
    if (not isPlaneFitable or cellActivatedCount < minimumCellActivated)
    {
        _histogram.remove_point(seedId);
        return;
    }

    // fit plane to merged data
    newPlaneSegment.fit_plane();
    if (not newPlaneSegment.is_planar())
    {
        outputs::log("Plane segment is not planar after merge");
        return;
    }

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

            const Plane_Segment& planeSegment = _planeGrid[localMapIndex];
            if (planeSegment.is_planar())
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
        if (not newMergedPlane.is_planar())
            outputs::log("Plane segment is not planar after merge");

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
    {
        // We use planes indexes as ids
        planeMergeLabels.emplace_back(planeIndex);
    }

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
            const static float planeMergeDistanceThreshold = Parameters::get_maximum_plane_merge_distance();
            if (planeToExpand.can_be_merged(mergePlane, planeMergeDistanceThreshold))
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
        {
            planeToExpand.fit_plane();
            if (not planeToExpand.is_planar())
                outputs::log("Plane segment is not planar after merge");
        }
    }

    return planeMergeLabels;
}

void Primitive_Detection::add_planes_to_primitives(const uint_vector& planeMergeLabels,
                                                   const cv::Mat_<float>& depthImage,
                                                   plane_container& planeContainer)
{
    const uint planeCount = static_cast<uint>(_planeSegments.size());
    planeContainer.clear();
    planeContainer.reserve(planeCount);

#ifdef DEBUG_DETECTED_POLYGONS
    cv::Mat debugImage(depthImage.size(), CV_8UC3, cv::Scalar(255, 255, 255));
#endif

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

        _mask = 0;
        // add all merged planes to the mask
        for (uint j = planeIndex; j < planeCount; ++j)
        {
            if (planeMergeLabels[j] == planeMergeLabel)
                _mask.setTo(1, _gridPlaneSegmentMap == (j + 1));
        }
#ifdef DEBUG_DETECTED_POLYGONS
        const uint pixelPerCellSide = static_cast<uint>(sqrtf(static_cast<float>(_pointsPerCellCount)));
        cv::Scalar color(utils::Random::get_random_uint(255),
                         utils::Random::get_random_uint(255),
                         utils::Random::get_random_uint(255));
        _mask.forEach([&debugImage, &color, &pixelPerCellSide](const uchar value, const int position[]) {
            if (value > 0)
                cv::rectangle(debugImage,
                              cv::Point(position[1] * pixelPerCellSide, position[0] * pixelPerCellSide),
                              cv::Point((position[1] + 1) * pixelPerCellSide, (position[0] + 1) * pixelPerCellSide),
                              color,
                              -1);
        });
#endif
        // get the ordered boundary points
        const std::vector<vector3>& orderedBoundary = compute_plane_segment_boundary(planeSegment, depthImage, _mask);
        if (orderedBoundary.size() < 3)
        {
            outputs::log_warning("Could not find a correct boundary polygon, rejecting plane segment");
            continue; // ignore this polygon
        }

        const utils::CameraPolygon polygon(orderedBoundary, planeSegment.get_normal(), planeSegment.get_center());
        std::string debug;
        if (polygon.is_valid(debug) and polygon.boundary_length() >= 3)
        {
            // add new plane to final shapes
            planeContainer.emplace_back(planeSegment, polygon);
#ifdef DEBUG_DETECTED_POLYGONS
            /*polygon.display(cv::Scalar(utils::Random::get_random_uint(255),
                                       utils::Random::get_random_uint(255),
                                       utils::Random::get_random_uint(255)),
                            debugImage);*/
#endif
        }
        else
        {
            std::cout << "Polyfit error: " << debug << std::endl;
        }
    }

#ifdef DEBUG_DETECTED_POLYGONS
    cv::imshow("temp", debugImage);
#endif
}

std::vector<vector3> Primitive_Detection::compute_plane_segment_boundary(const Plane_Segment& planeSegment,
                                                                         const cv::Mat_<float>& depthImage,
                                                                         const cv::Mat_<uchar>& mask) const
{
    // erode considering the border as an obstacle
    cv::Mat_<uchar> maskEroded(mask.size());
    cv::erode(mask, maskEroded, _maskCrossKernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

    // dilate to get boundaries
    cv::Mat_<uchar> maskBoundary(mask.size());
    cv::dilate(mask, maskBoundary, _maskSquareKernel);
    maskBoundary = maskBoundary - maskEroded;

    const double maxBoundaryDistance = 3 * sqrt(planeSegment.get_MSE());
    auto is_point_in_plane = [&planeSegment, &maxBoundaryDistance](const utils::ScreenCoordinate& point) {
        // Check that the point is inside the screen
        return (point.z() > 0 and point.x() >= 0 and point.y() >= 0) and
               // if distance of this point to the plane < threshold, this point is contained in the plane
               planeSegment.get_point_distance(point.to_camera_coordinates()) < maxBoundaryDistance;
    };
    const uint pixelPerCellSide = static_cast<uint>(sqrtf(static_cast<float>(_pointsPerCellCount)));

    std::vector<vector3> boundaryPoints;
    std::mutex mut;

    // Cell refinement
    maskBoundary.forEach([this, &depthImage, &boundaryPoints, &mut, &is_point_in_plane, pixelPerCellSide](
                                 const uchar value, const int position[]) {
        // cell is not in boundary, pass
        if (value <= 0)
            return;

        const uint cellColum = position[1];
        const uint cellRow = position[0];

        // get the pixels in image space (+-1 for neigbors)
        const int xStart = static_cast<int>(cellColum * pixelPerCellSide);
        const int yStart = static_cast<int>(cellRow * pixelPerCellSide);
        const int xEnd = static_cast<int>((cellColum + 1) * pixelPerCellSide);
        const int yEnd = static_cast<int>((cellRow + 1) * pixelPerCellSide);

        // get the defining points of this cell area
        const std::vector<vector3>& definingPoints =
                find_defining_points(depthImage, xStart, yStart, xEnd, yEnd, is_point_in_plane);

        // add boundary points to the total of boundaries
        std::scoped_lock<std::mutex> lock(mut);
        boundaryPoints.insert(boundaryPoints.cend(), definingPoints.cbegin(), definingPoints.cend());
    });

    return boundaryPoints;
}

std::vector<vector3> Primitive_Detection::find_defining_points(const cv::Mat_<float>& depthImage,
                                                               const int xStart,
                                                               const int yStart,
                                                               const int xEnd,
                                                               const int yEnd,
                                                               auto is_point_in_plane) const
{
    std::vector<vector3> definingPoints;

    const cv::Rect cellRoiRect(cv::Point(xStart, yStart), cv::Point(xEnd, yEnd));
    const cv::Mat_<float>& cellRoi = depthImage(cellRoiRect);

    std::mutex mut;
    // iterate over all pixels of this cell
    cellRoi.forEach([&depthImage, &is_point_in_plane, &definingPoints, &mut, xStart, yStart](const float value,
                                                                                             const int position[]) {
        const int x = xStart + position[1];
        const int y = yStart + position[0];

        const utils::ScreenCoordinate point(x, y, value);
        if (is_point_in_plane(point))
        {
            constexpr uint numberOfNeigbor = 3;                           // neigboring points (3x3 here)
            constexpr uint centerCoordinates = (numberOfNeigbor - 1) / 2; // coordinates of the center in the neigbors
            static_assert(numberOfNeigbor % 2 == 1);

            // the neigbors will contain the center point ! needs to be culled out
            cv::Mat_<float> neigtbors(numberOfNeigbor, numberOfNeigbor);
            // getRectSubPix can get values out of the image
            cv::getRectSubPix(depthImage, neigtbors.size(), cv::Point(x, y), neigtbors);
            assert(not neigtbors.empty());

            // check number of neigbors in the plane
            std::atomic<uint> inPlaneNeigborsCount = 0;
            neigtbors.forEach([&inPlaneNeigborsCount, &is_point_in_plane, x, y](const float neigtborsValue,
                                                                                const int neightborPosition[]) {
                // Do not add the center cell in neigbor treatment
                if (neightborPosition[1] != centerCoordinates && neightborPosition[0] != centerCoordinates &&
                    is_point_in_plane(utils::ScreenCoordinate(
                            neightborPosition[1] + x - 1, neightborPosition[0] + y - 1, neigtborsValue)))
                {
                    inPlaneNeigborsCount += 1;
                }
            });

            // Check that most of the neigtbors are not in the plane (edge)
            // and there is at least some neigbors (not a noise value)
            constexpr uint minExistingNeigborCount = 2; // number of valid neigtbors to accept
            constexpr uint maxEmptyNeigborCount =
                    (numberOfNeigbor * numberOfNeigbor - 1) - 0; // number of empty neigbors to accept
            if (inPlaneNeigborsCount >= minExistingNeigborCount and inPlaneNeigborsCount < maxEmptyNeigborCount)
            {
                // mutex for definingPoints
                std::scoped_lock<std::mutex> lock(mut);

                const vector3& candidate = point.to_camera_coordinates();
                // do not add this point if it's too close to a point already in this cell
                const bool isFarEnough = std::ranges::none_of(definingPoints, [&candidate](const vector3& bPoint) {
                    static constexpr double minSetDistance = 1000.0;
                    return (bPoint - candidate).lpNorm<1>() < minSetDistance;
                });

                // this point is far enough from the others and can be added to the boundary
                if (isFarEnough)
                {
                    definingPoints.emplace_back(candidate);
                }
            }
        }
    });

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
        _mask = 0;
        _mask.setTo(1, _gridCylinderSegMap == (cylinderIndex + 1));

        // Opening
        cv::Mat_<uchar> maskEroded;
        cv::dilate(_mask, _mask, _maskCrossKernel);
        cv::erode(_mask, _mask, _maskCrossKernel);
        cv::erode(_mask, maskEroded, _maskCrossKernel);
        double min;
        double max;
        cv::minMaxLoc(maskEroded, &min, &max);

        if (max <= 0 or min >= max) // completely eroded: irrelevant cylinder
            continue;

        const uint regId = cylinderToRegionMap[cylinderIndex].first;

        // add new cylinder to final shapes
        cylinderContainer.emplace_back(_cylinderSegments[regId]);
    }
}

Matrixb Primitive_Detection::get_connected_components_matrix(const cv::Mat_<int>& segmentMap,
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
    {
        outputs::log_error("Reached onvalid index while parsingf neigthbors");
        return;
    }

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
        if (x < _horizontalCellsCount - 1)
            region_growing(x + 1, y, planePatch, isActivatedMap); // right pixel
        if (y > 0)
            region_growing(x, y - 1, planePatch, isActivatedMap); // upper pixel
        if (y < _verticalCellsCount - 1)
            region_growing(x, y + 1, planePatch, isActivatedMap); // lower pixel
    }
    // else: do not merge this plane segment
}

} // namespace rgbd_slam::features::primitives
