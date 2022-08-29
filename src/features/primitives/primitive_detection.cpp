#include "primitive_detection.hpp"

#include <limits>

#include "cylinder_segment.hpp"
#include "parameters.hpp"
#include "logger.hpp"
#include "plane_segment.hpp"
#include "shape_primitives.hpp"

//index offset of a cylinder to a plane: used for masks display purposes
const uint CYLINDER_CODE_OFFSET = 50;


//lib simplification

namespace rgbd_slam {
    namespace features {
        namespace primitives {

            Primitive_Detection::Primitive_Detection(const uint width, const uint height, const uint blocSize, const float minCosAngleForMerge, const float maxMergeDistance)
                :  
                    _histogram(blocSize), 
                    _width(width), _height(height),  
                    _pointsPerCellCount(blocSize * blocSize), 
                    _minCosAngleForMerge(minCosAngleForMerge), _maxMergeDist(maxMergeDistance),
                    _cellWidth(blocSize), _cellHeight(blocSize),
                    _horizontalCellsCount(_width / _cellWidth), _verticalCellsCount(_height / _cellHeight),
                    _totalCellCount(_verticalCellsCount * _horizontalCellsCount)
            {
                //Init variables
                _isActivatedMap.assign(_totalCellCount, false);
                _isUnassignedMask.assign(_totalCellCount, false);
                _cellDistanceTols.assign(_totalCellCount, 0.0f);

                _gridPlaneSegmentMap = cv::Mat_<int>(_verticalCellsCount, _horizontalCellsCount, 0);
                _gridCylinderSegMap = cv::Mat_<int>(_verticalCellsCount, _horizontalCellsCount, 0);

                _mask = cv::Mat(_verticalCellsCount, _horizontalCellsCount, CV_8U);
                _maskEroded = cv::Mat(_verticalCellsCount, _horizontalCellsCount, CV_8U);

                _maskCrossKernel = cv::Mat::ones(3, 3, CV_8U);
                _maskCrossKernel.at<uchar>(0,0) = 0;
                _maskCrossKernel.at<uchar>(2,2) = 0;
                _maskCrossKernel.at<uchar>(0,2) = 0;
                _maskCrossKernel.at<uchar>(2,0) = 0;

                //array of unique_ptr<Plane_Segment>
                _planeGrid.reserve(_totalCellCount);
                for(uint i = 0; i < _totalCellCount; ++i) 
                {
                    //fill with empty nodes
                    _planeGrid.push_back(std::make_unique<Plane_Segment>(_cellWidth, _pointsPerCellCount));
                }

                //perf measurments
                resetTime = 0;
                initTime = 0;
                growTime = 0;
                mergeTime = 0;
                refineTime = 0;
            }

            void Primitive_Detection::find_primitives(const Eigen::MatrixXf& depthMatrix, primitive_container& primitiveSegments) 
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
                add_planes_to_primitives(planeMergeLabels, primitiveSegments);
                td = (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());
                initTime += td;
                refineTime += td;

                t1 = cv::getTickCount();
                //refine cylinders boundaries and fill the final cylinders vector
                add_cylinders_to_primitives(cylinder2regionMap, primitiveSegments); 
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
                std::fill_n(_isUnassignedMask.begin(), _isUnassignedMask.size(), false);
                std::fill_n(_cellDistanceTols.begin(), _cellDistanceTols.size(), 0.0f);

                //mat masks do not need to be cleared
                //kernels should not be cleared
            }

            void Primitive_Detection::init_planar_cell_fitting(const Eigen::MatrixXf& depthCloudArray) 
            {
                const float sinCosAngleForMerge = sqrtf(1.0f - powf(_minCosAngleForMerge, 2.0f));

                //for each planeGrid cell
                const size_t planeGridSize = _planeGrid.size();
                for(size_t stackedCellId = 0; stackedCellId < planeGridSize; ++stackedCellId) {
                    //init the plane grid cell
                    _planeGrid[stackedCellId]->init_plane_segment(depthCloudArray, stackedCellId);

                    if (_planeGrid[stackedCellId]->is_planar()) {
                        const uint cellDiameter = static_cast<uint>((
                                    depthCloudArray.block(stackedCellId * _pointsPerCellCount + _pointsPerCellCount - 1, 0, 1, 3) - 
                                    depthCloudArray.block(stackedCellId * _pointsPerCellCount, 0, 1, 3)
                                    ).norm());

                        //array of depth metrics: neighbors merging threshold
                        _cellDistanceTols[stackedCellId] = powf(std::clamp(cellDiameter * sinCosAngleForMerge, 20.0f, _maxMergeDist), 2.0f);
                    }
                }
            }

            uint Primitive_Detection::init_histogram() 
            {
                uint remainingPlanarCells = 0;
                Eigen::MatrixXd histBins(_totalCellCount, 2);

                const size_t planeGridSize = _planeGrid.size();
                for(uint cellId = 0; cellId < planeGridSize; ++cellId) 
                { 
                    if(_planeGrid[cellId]->is_planar()) 
                    {
                        const vector3& planeNormal = _planeGrid[cellId]->get_normal();
                        const double nx = planeNormal.x();
                        const double ny = planeNormal.y();

                        const double projNormal = 1.0 / sqrt(nx * nx + ny * ny);
                        histBins(cellId, 0) = acos( -planeNormal.z());
                        histBins(cellId, 1) = atan2(nx * projNormal, ny * projNormal);
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

                uint cylinderCount = 0;
                uint unaffectedPlanarCells = remainingPlanarCells;
                //find seed planes and make them grow
                while(unaffectedPlanarCells > 0) 
                {
                    //get seed candidates
                    const std::vector<uint>& seedCandidates = _histogram.get_points_from_most_frequent_bin();
                    if (seedCandidates.size() < Parameters::get_minimum_plane_seed_count())
                        break;

                    //select seed cell with min MSE
                    uint seedId = 0;    //should not necessarily stay to 0 after the loop
                    double minMSE = std::numeric_limits<double>::max();
                    for(const uint seedCandidate : seedCandidates)
                    {
                        if(_planeGrid[seedCandidate]->get_MSE() < minMSE) 
                        {
                            seedId = seedCandidate;
                            minMSE = _planeGrid[seedCandidate]->get_MSE();
                            if(minMSE <= 0)
                                break;
                        }
                    }
                    if (minMSE >= std::numeric_limits<double>::max())
                    {
                        utils::log_error("Could not find a single plane segment");
                        break;
                    }


                    //copy plane segment in new object
                    Plane_Segment newPlaneSegment(*_planeGrid[seedId]);
                    if (not newPlaneSegment.is_planar())
                        continue;

                    //Seed cell growing
                    const uint y = static_cast<uint>(seedId / _horizontalCellsCount);
                    const uint x = static_cast<uint>(seedId % _horizontalCellsCount);

                    //activationMap set to false
                    const size_t activationMapSize = _isActivatedMap.size();
                    std::fill_n(_isActivatedMap.begin(), activationMapSize, false);
                    //grow plane region, fill _isActivatedMap
                    region_growing(x, y, newPlaneSegment.get_normal(), newPlaneSegment.get_plane_d());

                    assert(activationMapSize == _isUnassignedMask.size());
                    assert(activationMapSize == _planeGrid.size());

                    //merge activated cells & remove them from histogram
                    uint cellActivatedCount = 0;
                    bool isPlaneFitable = false;
                    for(uint planeSegmentIndex = 0; planeSegmentIndex < activationMapSize; ++planeSegmentIndex) 
                    {
                        if(_isActivatedMap[planeSegmentIndex]) 
                        {
                            const Plane_Segment& planeSegment = *(_planeGrid[planeSegmentIndex]);
                            if (planeSegment.is_planar())
                            {
                                newPlaneSegment.expand_segment(planeSegment);
                                ++cellActivatedCount;
                                _histogram.remove_point(planeSegmentIndex);
                                _isUnassignedMask[planeSegmentIndex] = false;

                                assert(unaffectedPlanarCells > 0);
                                --unaffectedPlanarCells;
                                isPlaneFitable = true;
                            }
                        }
                    }

                    if(not isPlaneFitable or cellActivatedCount < Parameters::get_minimum_cell_activated()) 
                    {
                        _histogram.remove_point(seedId);
                        continue;
                    }

                    //fit plane to merged data
                    newPlaneSegment.fit_plane();

                    if(newPlaneSegment.get_score() > 100) 
                    {
                        //its certainly a plane or we ignore cylinder detection
                        _planeSegments.push_back(std::make_unique<Plane_Segment>(newPlaneSegment));
                        const size_t currentPlaneCount = _planeSegments.size();
                        //mark cells
                        for(uint row = 0, activationIndex = 0; row < _verticalCellsCount; ++row) 
                        {
                            int* rowPtr = _gridPlaneSegmentMap.ptr<int>(row);
                            for(uint col = 0; col < _horizontalCellsCount; ++col, ++activationIndex)
                            {
                                assert(activationIndex < activationMapSize);

                                if(_isActivatedMap[activationIndex])
                                    rowPtr[col] = currentPlaneCount;
                            }
                        }

                    }
                    else if(cellActivatedCount > 5) 
                    {
                        //cylinder fitting
                        // It is an extrusion
                        _cylinderSegments.push_back(std::make_unique<Cylinder_Segment>(_planeGrid, _isActivatedMap, cellActivatedCount));
                        const Cylinder_Segment& cylinderSegment = *(_cylinderSegments.back());

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

                                    const Plane_Segment& planeSegment = *(_planeGrid[localMapIndex]);
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
                                //MSE of the plane is less than MSE of the cylinder + this plane 
                                _planeSegments.push_back(std::make_unique<Plane_Segment>(newMergedPlane));
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
                                assert(_cylinderSegments.size() > 0);

                                ++cylinderCount;
                                cylinder2regionMap.push_back(std::make_pair(_cylinderSegments.size() - 1, segId));
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
                }//\while

                return cylinder2regionMap;
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
                    const Plane_Segment& testPlane = *(_planeSegments[planeId]);
                    if (not testPlane.is_planar())
                        continue;

                    const vector3& testPlaneNormal = testPlane.get_normal();

                    for(uint col = row + 1; col < isPlanesConnectedMatrixCols; ++col) 
                    {
                        if(isPlanesConnectedMatrix(row, col)) 
                        {
                            const Plane_Segment& mergePlane = *(_planeSegments[col]);
                            if (not mergePlane.is_planar())
                                continue;

                            const vector3& mergePlaneNormal = mergePlane.get_normal();
                            const double cosAngle = testPlaneNormal.dot(mergePlaneNormal);

                            const vector3& mergePlaneMean = mergePlane.get_mean();
                            const double distance = pow(
                                    testPlaneNormal.dot(mergePlaneMean) + testPlane.get_plane_d(),
                                    2);

                            if(cosAngle > _minCosAngleForMerge and distance < _maxMergeDist) 
                            {
                                //merge plane segments
                                _planeSegments[planeId]->expand_segment(mergePlane);
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
                        _planeSegments[planeId]->fit_plane();
                }

                return planeMergeLabels;
            }

            void Primitive_Detection::add_planes_to_primitives(const uint_vector& planeMergeLabels, primitive_container& primitiveSegments) 
            {
                //refine the coarse planes boundaries to smoother versions
                const uint planeCount = _planeSegments.size();
                uchar planeIdAllocator = 0;
                for(uint planeIndex = 0; planeIndex < planeCount; ++planeIndex) 
                {
                    if (planeIndex != planeMergeLabels[planeIndex])
                        continue;
                    if (not _planeSegments[planeIndex]->is_planar())
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

                    // new plane ID
                    const uchar planeId = ++planeIdAllocator;
                    assert(planeId < CYLINDER_CODE_OFFSET);

                    //add new plane to final shapes
                    primitiveSegments.emplace(planeId, std::make_unique<Plane>(*(_planeSegments[planeIndex]), planeId, _mask));
                }
            }

            void Primitive_Detection::add_cylinders_to_primitives(const intpair_vector& cylinderToRegionMap, primitive_container& primitiveSegments) 
            {
                uchar cylinderIdAllocator = CYLINDER_CODE_OFFSET;
                for(uint cylinderIndex = 0; cylinderIndex < cylinderToRegionMap.size(); ++cylinderIndex)
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

                    // Affect a new cylinder id
                    const uchar cylinderId = ++cylinderIdAllocator;

                    const uint regId = cylinderToRegionMap[cylinderIndex].first;

                    //add new cylinder to final shapes
                    primitiveSegments.emplace(cylinderId, std::make_unique<Cylinder>(*(_cylinderSegments[regId]), cylinderId, _mask));
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
                    const int *rowPtr = segmentMap.ptr<int>(row);
                    const int *rowBelowPtr = segmentMap.ptr<int>(row + 1);
                    for(uint col = 0; col < cols2scanCount; ++col) 
                    {
                        const int pixelValue = rowPtr[col];
                        if(pixelValue > 0) 
                        {
                            if(rowPtr[col + 1] > 0 and pixelValue != rowPtr[col + 1]) 
                            {
                                isPlanesConnectedMatrix(pixelValue - 1, rowPtr[col + 1] - 1) = true;
                                isPlanesConnectedMatrix(rowPtr[col + 1] - 1, pixelValue - 1) = true;
                            }
                            if(rowBelowPtr[col] > 0 and pixelValue != rowBelowPtr[col]) 
                            {
                                isPlanesConnectedMatrix(pixelValue - 1, rowBelowPtr[col] - 1) = true;
                                isPlanesConnectedMatrix(rowBelowPtr[col] - 1, pixelValue - 1) = true;
                            }
                        }
                    }
                }

                return isPlanesConnectedMatrix;
            }


            void Primitive_Detection::region_growing(const uint x, const uint y, const vector3& seedPlaneNormal, const double seedPlaneD) 
            {
                assert(_isActivatedMap.size() == _isUnassignedMask.size());
                assert(_horizontalCellsCount > 0);
                assert(seedPlaneD >= 0);

                const size_t index = x + _horizontalCellsCount * y;
                if (index >= _totalCellCount)
                    return;

                assert(index < _isActivatedMap.size());
                assert(index < _isUnassignedMask.size());
                if ((not _isUnassignedMask[index]) or _isActivatedMap[index]) 
                    //pixel is not part of a component or already labelled
                    return;

                assert(index < _planeGrid.size()); 

                const vector3& secPlaneNormal = _planeGrid[index]->get_normal();
                const vector3& secPlaneMean = _planeGrid[index]->get_mean();
                const double secPlaneD = _planeGrid[index]->get_plane_d();

                if (
                        //_planeGrid[index]->is_depth_discontinuous(secPlaneMean) or 
                        seedPlaneNormal.dot(secPlaneNormal) < _minCosAngleForMerge or
                        pow(seedPlaneNormal.dot(secPlaneMean) + seedPlaneD, 2.0) > _cellDistanceTols[index]
                   )//angle between planes < threshold or dist between planes > threshold
                    return;

                _isActivatedMap[index] = true;

                // Now label the 4 neighbours:
                if (x > 0)
                    region_growing(x - 1, y, secPlaneNormal, secPlaneD);   // left  pixel
                if (x < _width - 1)  
                    region_growing(x + 1, y, secPlaneNormal, secPlaneD);  // right pixel
                if (y > 0)        
                    region_growing(x, y - 1, secPlaneNormal, secPlaneD);   // upper pixel 
                if (y < _height - 1) 
                    region_growing(x, y + 1, secPlaneNormal, secPlaneD);   // lower pixel
            }



            Primitive_Detection::~Primitive_Detection() 
            {
            }

        }
    }
}

