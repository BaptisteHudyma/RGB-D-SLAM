#include "primitive_detection.hpp"

#include <limits>
#include <opencv2/highgui.hpp>

#include "cylinder_segment.hpp"
#include "plane_segment.hpp"
#include "shape_primitives.hpp"

#include "../../parameters.hpp"
#include "../../outputs/logger.hpp"
#include "types.hpp"

namespace rgbd_slam {
    namespace features {
        namespace primitives {

            Primitive_Detection::Primitive_Detection(const uint width, const uint height, const uint blocSize)
                :  
                    _histogram(blocSize), 
                    _width(width), _height(height),  
                    _pointsPerCellCount(blocSize * blocSize),
                    _cellWidth(blocSize), _cellHeight(blocSize),
                    _horizontalCellsCount(_width / _cellWidth), _verticalCellsCount(_height / _cellHeight),
                    _totalCellCount(_verticalCellsCount * _horizontalCellsCount)
            {
                //Init variables
                _isUnassignedMask = vectorb::Zero(_totalCellCount);
                _cellDistanceTols.assign(_totalCellCount, 0.0f);

                _gridPlaneSegmentMap = cv::Mat_<int>(_verticalCellsCount, _horizontalCellsCount, 0);
                _gridCylinderSegMap  = cv::Mat_<int>(_verticalCellsCount, _horizontalCellsCount, 0);

                _mask = cv::Mat(_verticalCellsCount, _horizontalCellsCount, CV_8U);
                _maskEroded = cv::Mat(_verticalCellsCount, _horizontalCellsCount, CV_8U);

                _maskCrossKernel = cv::Mat::ones(3, 3, CV_8U);
                _maskCrossKernel.at<uchar>(0,0) = 0;
                _maskCrossKernel.at<uchar>(2,2) = 0;
                _maskCrossKernel.at<uchar>(0,2) = 0;
                _maskCrossKernel.at<uchar>(2,0) = 0;

                //array of unique_ptr<Plane_Segment>
                const Plane_Segment defaultPlaneSegment = Plane_Segment(_cellWidth, _pointsPerCellCount);
                _planeGrid.reserve(_totalCellCount);
                for(uint i = 0; i < _totalCellCount; ++i) 
                {
                    //fill with empty nodes
                    _planeGrid.push_back(defaultPlaneSegment);
                }

                //perf measurments
                resetTime = 0;
                initTime = 0;
                growTime = 0;
                mergeTime = 0;
                refineTime = 0;
            }

            void Primitive_Detection::find_primitives(const matrixf& depthMatrix, plane_container& planeContainer, cylinder_container& primitiveContainer) 
            {
                //reset used data structures
                reset_data();

                int64 t1 = cv::getTickCount();
                //init planar grid
                init_planar_cell_fitting(depthMatrix);
                double td = (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());
                resetTime += td;

                //init and fill histogram
                t1 = cv::getTickCount();
                const uint remainingPlanarCells = init_histogram();
                td = (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());
                initTime += td;

                t1 = cv::getTickCount();
                const intpair_vector& cylinder2regionMap = grow_planes_and_cylinders(remainingPlanarCells);
                td = (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());
                growTime += td;

                //merge sparse planes
                t1 = cv::getTickCount();
                const uint_vector& planeMergeLabels = merge_planes();
                td = (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());
                initTime += td;
                mergeTime += td;

                t1 = cv::getTickCount();
                //fill the final planes vector
                add_planes_to_primitives(planeMergeLabels, planeContainer);
                td = (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());
                initTime += td;
                refineTime += td;

                t1 = cv::getTickCount();
                //refine cylinders boundaries and fill the final cylinders vector
                add_cylinders_to_primitives(cylinder2regionMap, primitiveContainer); 
                td = (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());
                initTime += td;
                refineTime += td;
            }

            void Primitive_Detection::reset_data() 
            {
                _histogram.reset();

                //planeGrid SHOULD NOT be cleared
                _planeSegments.clear();
                _cylinderSegments.clear();

                _gridPlaneSegmentMap = 0;
                _gridCylinderSegMap = 0;

                //reset stacked distances
                //activation map do not need to be cleared
                _isUnassignedMask = vectorb::Zero(_isUnassignedMask.size());
                std::fill_n(_cellDistanceTols.begin(), _cellDistanceTols.size(), 0.0f);

                //mat masks do not need to be cleared
                //kernels should not be cleared
            }

            void Primitive_Detection::init_planar_cell_fitting(const matrixf& depthCloudArray) 
            {
                const static float maximumCosAngle = Parameters::get_maximum_plane_match_angle();
                const static float sinCosAngleForMerge = sqrtf(1.0f - powf(maximumCosAngle, 2.0f));

                //for each planeGrid cell
                const size_t planeGridSize = _planeGrid.size();
                for(size_t stackedCellId = 0; stackedCellId < planeGridSize; ++stackedCellId) {
                    //init the plane patch
                    Plane_Segment& planeSegment = _planeGrid[stackedCellId];
                    planeSegment.init_plane_segment(depthCloudArray, stackedCellId);
                    // if this plane patch is planar, compute the diagonal distance
                    if (planeSegment.is_planar()) {
                        const uint offset = stackedCellId * _pointsPerCellCount;
                        // cell diameter, in millimeters
                        const float cellDiameter = (
                                    // right down corner (x, y, z)
                                    depthCloudArray.block(offset + _pointsPerCellCount - 1, 0, 1, 3) - 
                                    // left up corner (x, y, z)
                                    depthCloudArray.block(offset, 0, 1, 3)
                                    ).norm();
                        //array of sqrt(depth) metrics: neighbors merging threshold
                        const static float maxMergeDistance = Parameters::get_maximum_merge_distance();
                        _cellDistanceTols[stackedCellId] = powf(std::clamp(cellDiameter * sinCosAngleForMerge, 20.0f, maxMergeDistance), 2.0f);
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
                for(uint cellId = 0; cellId < planeGridSize; ++cellId) 
                { 
                    const Plane_Segment& planePatch = _planeGrid[cellId];
                    if(planePatch.is_planar())
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
                //find seed planes and make them grow
                while(untriedPlanarCellsCount > 0) 
                {
                    //get seed candidates
                    const std::vector<uint>& seedCandidates = _histogram.get_points_from_most_frequent_bin();
                    const static uint planeSeedCount = Parameters::get_minimum_plane_seed_count();
                    if (seedCandidates.size() < planeSeedCount)
                        break;

                    //select seed cell with min MSE
                    uint seedId = 0;    //should not necessarily stay to 0 after the loop
                    double minMSE = std::numeric_limits<double>::max();
                    for(const uint seedCandidate : seedCandidates)
                    {
                        const double candidateMSE = _planeGrid[seedCandidate].get_MSE();
                        if(candidateMSE < minMSE) 
                        {
                            seedId = seedCandidate;
                            minMSE = candidateMSE;
                            if(minMSE <= 0)
                                break;
                        }
                    }
                    if (minMSE >= std::numeric_limits<double>::max())
                    {
                        // seedId is invalid
                        outputs::log_error("Could not find a single plane segment");
                        break;
                    }

                    // try to grow the selected plane at seedId
                    grow_plane_segment_at_seed(seedId, untriedPlanarCellsCount, cylinder2regionMap);
                }
                return cylinder2regionMap;
            }


            void Primitive_Detection::grow_plane_segment_at_seed(const uint seedId, uint& untriedPlanarCellsCount, intpair_vector& cylinder2regionMap)
            {
                assert(seedId < _planeGrid.size());
                const Plane_Segment& planeToGrow = _planeGrid[seedId];
                if (not planeToGrow.is_planar())
                {
                    // cannot grow a non planar patch
                    return;
                }

                //copy plane segment in new object, to try to grow it in a non destructive way
                Plane_Segment newPlaneSegment(planeToGrow);

                //Seed cell growing
                const uint y = static_cast<uint>(seedId / _horizontalCellsCount);
                const uint x = static_cast<uint>(seedId % _horizontalCellsCount);

                //activationMap set to false (will have bits at true when a plane segment will be merged to this one)
                vectorb isActivatedMap = vectorb::Zero(_totalCellCount);
                const size_t activationMapSize = isActivatedMap.size();
                //grow plane region, fill isActivatedMap
                region_growing(x, y, newPlaneSegment, isActivatedMap);

                assert(activationMapSize == static_cast<size_t>(_isUnassignedMask.size()));
                assert(activationMapSize == _planeGrid.size());

                //merge activated cells & remove them from histogram
                uint cellActivatedCount = 0;
                bool isPlaneFitable = false;
                for(uint planeSegmentIndex = 0; planeSegmentIndex < activationMapSize; ++planeSegmentIndex) 
                {
                    if(isActivatedMap[planeSegmentIndex]) 
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

                const static uint minimumCellActivated = Parameters::get_minimum_cell_activated();
                if(not isPlaneFitable or cellActivatedCount < minimumCellActivated) 
                {
                    _histogram.remove_point(seedId);
                    return;
                }

                //fit plane to merged data
                newPlaneSegment.fit_plane();

                // TODO: why 100 ? seems random
                if(newPlaneSegment.get_score() > 100) 
                {
                    //its certainly a plane or we ignore cylinder detection
                    _planeSegments.push_back(newPlaneSegment);
                    const size_t currentPlaneCount = _planeSegments.size();
                    //mark cells that belong to this plane with a new id
                    for(uint row = 0, activationIndex = 0; row < _verticalCellsCount; ++row) 
                    {
                        int* rowPtr = _gridPlaneSegmentMap.ptr<int>(row);
                        for(uint col = 0; col < _horizontalCellsCount; ++col, ++activationIndex)
                        {
                            assert(activationIndex < activationMapSize);

                            if(isActivatedMap[activationIndex])
                                rowPtr[col] = currentPlaneCount;
                        }
                    }
                }
                // TODO: why 5 ? seems random
                else if(cellActivatedCount > 5) 
                {
                    cylinder_fitting(cellActivatedCount, isActivatedMap, cylinder2regionMap);
                }
            }

            void Primitive_Detection::cylinder_fitting(const uint cellActivatedCount, const vectorb& isActivatedMap, intpair_vector& cylinder2regionMap)
            {
                //try cylinder fitting on the activated planes
                const Cylinder_Segment& cylinderSegment = Cylinder_Segment(_planeGrid, isActivatedMap, cellActivatedCount);
                _cylinderSegments.push_back(cylinderSegment);


                // Fit planes to subsegments
                for(uint segId = 0; segId < cylinderSegment.get_segment_count(); ++segId)
                {
                    bool isPlaneSegmentFitable = false;
                    Plane_Segment newMergedPlane(_cellWidth, _pointsPerCellCount);
                    for(uint col = 0; col < cellActivatedCount; ++col)
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

                    // No continuous planes, pass
                    if (not isPlaneSegmentFitable)
                        continue;

                    newMergedPlane.fit_plane();
                    // Model selection based on MSE
                    if(newMergedPlane.get_MSE() < cylinderSegment.get_MSE_at(segId))
                    {
                        //MSE of the plane is less than MSE of the cylinder + this plane, so keep this one as a plane
                        _planeSegments.push_back(newMergedPlane);
                        const uint currentPlaneCount = _planeSegments.size();
                        for(uint col = 0; col < cellActivatedCount; ++col)
                        {
                            if (cylinderSegment.is_inlier_at(segId, col))
                            {
                                const uint cellId = cylinderSegment.get_local_to_global_mapping(col);
                                _gridPlaneSegmentMap.at<int>(cellId / _horizontalCellsCount, cellId % _horizontalCellsCount) = currentPlaneCount;
                            }
                        }
                    }
                    else 
                    {
                        // Set a new cylinder
                        assert(_cylinderSegments.size() > 0);
                        cylinder2regionMap.push_back(std::make_pair(_cylinderSegments.size() - 1, segId));
                        const size_t cylinderCount = cylinder2regionMap.size();

                        for(uint col = 0; col < cellActivatedCount; ++col)
                        {
                            if (cylinderSegment.is_inlier_at(segId, col))
                            {
                                const uint cellId = cylinderSegment.get_local_to_global_mapping(col);
                                _gridCylinderSegMap.at<int>(cellId / _horizontalCellsCount, cellId % _horizontalCellsCount) = cylinderCount;
                            }
                        }
                    }
                }
            }

            Primitive_Detection::uint_vector Primitive_Detection::merge_planes() 
            {
                const uint planeCount = _planeSegments.size();

                Matrixb isPlanesConnectedMatrix = get_connected_components_matrix(_gridPlaneSegmentMap, planeCount);
                assert(isPlanesConnectedMatrix.rows() == isPlanesConnectedMatrix.cols());

                uint_vector planeMergeLabels;
                planeMergeLabels.reserve(planeCount);
                for(uint planeIndex = 0; planeIndex < planeCount; ++planeIndex)
                    // We use planes indexes as ids
                    planeMergeLabels.push_back(planeIndex);

                const uint isPlanesConnectedMatrixRows = isPlanesConnectedMatrix.rows();
                const uint isPlanesConnectedMatrixCols = isPlanesConnectedMatrix.cols();
                for(uint row = 0; row < isPlanesConnectedMatrixRows; ++row) 
                {
                    bool wasPlaneExpanded = false;
                    const uint planeId = planeMergeLabels[row];
                    Plane_Segment& planeToExpand = _planeSegments[planeId];
                    if (not planeToExpand.is_planar())
                        continue;

                    for(uint col = row + 1; col < isPlanesConnectedMatrixCols; ++col) 
                    {
                        if(isPlanesConnectedMatrix(row, col)) 
                        {
                            const Plane_Segment& mergePlane = _planeSegments[col];
                            if (not mergePlane.is_planar())
                                continue;

                            // normals are close enough, distance is small enough
                            const static float maxMergeDistance = Parameters::get_maximum_merge_distance();
                            if(planeToExpand.can_be_merged(mergePlane, maxMergeDistance))
                            {
                                //merge plane segments
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
                    }
                    if(wasPlaneExpanded)    //plane was merged with other planes
                        planeToExpand.fit_plane();
                }

                return planeMergeLabels;
            }

            void Primitive_Detection::add_planes_to_primitives(const uint_vector& planeMergeLabels, plane_container& planeContainer) 
            {
                const uint planeCount = _planeSegments.size();
                planeContainer.clear();
                planeContainer.reserve(planeCount);

                //refine the coarse planes boundaries to smoother versions
                for(uint planeIndex = 0; planeIndex < planeCount; ++planeIndex) 
                {
                    if (planeIndex != planeMergeLabels[planeIndex])
                        continue;
                    if (not _planeSegments[planeIndex].is_planar())
                        continue;

                    _mask = cv::Scalar(0);
                    for(uint j = planeIndex; j < planeCount; ++j) 
                    {
                        if(planeMergeLabels[j] == planeMergeLabels[planeIndex])
                            _mask.setTo(1, _gridPlaneSegmentMap == (j + 1));
                    }
                    // Opening
                    cv::dilate(_mask, _mask, _maskCrossKernel);
                    cv::erode(_mask, _mask, _maskCrossKernel);
                    cv::erode(_mask, _maskEroded, _maskCrossKernel);
                    double min, max;
                    cv::minMaxLoc(_maskEroded, &min, &max);

                    if(max <= 0 or min >= max)    //completely eroded: irrelevant plane
                        continue;

                    //add new plane to final shapes
                    planeContainer.emplace_back(Plane(_planeSegments[planeIndex], _mask));
                }
            }

            void Primitive_Detection::add_cylinders_to_primitives(const intpair_vector& cylinderToRegionMap, cylinder_container& cylinderContainer) 
            {
                const size_t numberOfCylinder = cylinderToRegionMap.size();
                cylinderContainer.clear();
                cylinderContainer.reserve(numberOfCylinder);

                for(uint cylinderIndex = 0; cylinderIndex < numberOfCylinder; ++cylinderIndex)
                {
                    // Build mask
                    _mask = cv::Scalar(0);
                    _mask.setTo(1, _gridCylinderSegMap == (cylinderIndex + 1));

                    // Opening
                    cv::dilate(_mask, _mask, _maskCrossKernel);
                    cv::erode(_mask, _mask, _maskCrossKernel);
                    cv::erode(_mask, _maskEroded, _maskCrossKernel);
                    double min, max;
                    cv::minMaxLoc(_maskEroded, &min, &max);

                    if(max <= 0 or min >= max)    //completely eroded: irrelevant cylinder 
                        continue;

                    const uint regId = cylinderToRegionMap[cylinderIndex].first;

                    //add new cylinder to final shapes
                    cylinderContainer.emplace_back(Cylinder(_cylinderSegments[regId], _mask));
                }
            }

            Matrixb Primitive_Detection::get_connected_components_matrix(const cv::Mat& segmentMap, const size_t numberOfPlanes) const 
            {
                assert(segmentMap.rows > 0);
                assert(segmentMap.cols > 0);

                Matrixb isPlanesConnectedMatrix = Matrixb::Constant(numberOfPlanes, numberOfPlanes, false);
                if (numberOfPlanes == 0)
                    return isPlanesConnectedMatrix;

                const uint rows2scanCount = segmentMap.rows - 1;
                const uint cols2scanCount = segmentMap.cols - 1;
                for(uint row = 0; row < rows2scanCount; ++row) 
                {
                    const int* rowPtr = segmentMap.ptr<int>(row);
                    const int* rowBelowPtr = segmentMap.ptr<int>(row + 1);
                    for(uint col = 0; col < cols2scanCount; ++col) 
                    {
                        //value of the pixel at this coordinates. Represents a plane segment
                        const int planeId = rowPtr[col];
                        if(planeId > 0) 
                        {
                            const int nextPlaneId = rowPtr[col + 1];
                            const int belowPlaneId = rowBelowPtr[col];
                            if(nextPlaneId > 0 and planeId != nextPlaneId) 
                            {
                                isPlanesConnectedMatrix(planeId - 1, nextPlaneId - 1) = true;
                                isPlanesConnectedMatrix(nextPlaneId - 1, planeId - 1) = true;
                            }
                            if(belowPlaneId > 0 and planeId != belowPlaneId) 
                            {
                                isPlanesConnectedMatrix(planeId - 1, belowPlaneId - 1) = true;
                                isPlanesConnectedMatrix(belowPlaneId - 1, planeId - 1) = true;
                            }
                        }
                    }
                }

                return isPlanesConnectedMatrix;
            }


            void Primitive_Detection::region_growing(const uint x, const uint y, const Plane_Segment& planeToExpand, vectorb& isActivatedMap) 
            {
                assert(isActivatedMap.size() == _isUnassignedMask.size());
                assert(_horizontalCellsCount > 0);

                const size_t index = x + _horizontalCellsCount * y;
                if (index >= _totalCellCount)
                    return;

                assert(index < static_cast<size_t>(isActivatedMap.size()));
                assert(index < static_cast<size_t>(_isUnassignedMask.size()));
                if ((not _isUnassignedMask[index]) or isActivatedMap[index]) 
                    //pixel is not part of a component or already labelled, or not a plane (_isUnassignedMask is always false for non planar patches)
                    return;

                assert(index < _planeGrid.size());
                const Plane_Segment& planePatch = _planeGrid[index];
                if (planeToExpand.can_be_merged(planePatch, _cellDistanceTols[index]))
                {
                    // mark this plane as merged
                    isActivatedMap[index] = true;

                    // Now label the 4 neighbours:
                    if (x > 0)
                        region_growing(x - 1, y, planePatch, isActivatedMap);   // left  pixel
                    if (x < _width - 1)  
                        region_growing(x + 1, y, planePatch, isActivatedMap);  // right pixel
                    if (y > 0)        
                        region_growing(x, y - 1, planePatch, isActivatedMap);   // upper pixel 
                    if (y < _height - 1) 
                        region_growing(x, y + 1, planePatch, isActivatedMap);   // lower pixel
                }
                //else: do not merge this plane segment
            }



            Primitive_Detection::~Primitive_Detection() 
            {
            }

        }
    }
}

