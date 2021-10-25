#include <limits>
#include "PrimitiveDetection.hpp"
#include "parameters.hpp"

//index offset of a cylinder to a plane: used for masks display purposes
const unsigned int CYLINDER_CODE_OFFSET = 50;


//lib simplification

namespace rgbd_slam {
namespace features {
namespace primitives {

        Primitive_Detection::Primitive_Detection(const unsigned int height, const unsigned int width, const unsigned int blocSize, const float minCosAngleForMerge, const float maxMergeDistance, const bool useCylinderDetection)
            :  
                _histogram(blocSize), 
                _width(width), _height(height),  _blocSize(blocSize), 
                _pointsPerCellCount(_blocSize * _blocSize), 
                _minCosAngleForMerge(minCosAngleForMerge), _maxMergeDist(maxMergeDistance),
                _useCylinderDetection(useCylinderDetection),
                _cellWidth(blocSize), _cellHeight(blocSize),
                _horizontalCellsCount(_width / _cellWidth), _verticalCellsCount(_height / _cellHeight),
                _totalCellCount(_verticalCellsCount * _horizontalCellsCount)

        {
            //Init variables
            _activationMap = new bool[_totalCellCount];
            _unassignedMask = new bool[_totalCellCount];
            _distancesStacked = new float[_width * _height];
            _segMapStacked = new unsigned char[_width * _height];
            _cellDistanceTols = new float[_totalCellCount];

            _gridPlaneSegmentMap = cv::Mat_<int>(_verticalCellsCount, _horizontalCellsCount, 0);
            _gridPlaneSegMapEroded = cv::Mat_<uchar>(_verticalCellsCount, _horizontalCellsCount, uchar(0));
            _gridCylinderSegMap = cv::Mat_<int>(_verticalCellsCount, _horizontalCellsCount, 0);
            _gridCylinderSegMapEroded = cv::Mat_<uchar>(_verticalCellsCount, _horizontalCellsCount, uchar(0));

            _distancesCellStacked = Eigen::ArrayXf::Zero(_pointsPerCellCount, 1);

            _mask = cv::Mat(_verticalCellsCount, _horizontalCellsCount, CV_8U);
            _maskEroded = cv::Mat(_verticalCellsCount, _horizontalCellsCount, CV_8U);
            _maskDilated = cv::Mat(_verticalCellsCount, _horizontalCellsCount, CV_8U);
            _maskDiff = cv::Mat(_verticalCellsCount, _horizontalCellsCount, CV_8U);

            //init kernels
            _maskSquareKernel = cv::Mat::ones(3, 3, CV_8U);
            _maskCrossKernel = cv::Mat::ones(3, 3, CV_8U);

            _maskCrossKernel.at<uchar>(0,0) = 0;
            _maskCrossKernel.at<uchar>(2,2) = 0;
            _maskCrossKernel.at<uchar>(0,2) = 0;
            _maskCrossKernel.at<uchar>(2,0) = 0;

            //array of unique_ptr<Plane_Segment>
            _planeGrid = new plane_segment_unique_ptr[_totalCellCount];
            for(unsigned int i = 0; i < _totalCellCount; ++i) {
                //fill with empty nodes
                _planeGrid[i] = std::make_unique<Plane_Segment>(_cellWidth, _pointsPerCellCount);
            }

            //perf measurments
            resetTime = 0;
            initTime = 0;
            growTime = 0;
            mergeTime = 0;
            refineTime = 0;
            setMaskTime = 0;
        }

        void Primitive_Detection::apply_masks(const cv::Mat& inputImage, const std::vector<cv::Vec3b>& colors, const cv::Mat& maskImage, const primitive_container& primitiveSegments, cv::Mat& labeledImage, const std::map<int, int>& associatedIds, const double timeElapsed) {
            //apply masks on image
            for(unsigned int r = 0; r < _height; ++r){
                const cv::Vec3b* rgbPtr = inputImage.ptr<cv::Vec3b>(r);
                cv::Vec3b* outPtr = labeledImage.ptr<cv::Vec3b>(r);

                for(unsigned int c = 0; c < _width; ++c){
                    const int index = static_cast<int>(maskImage.at<uchar>(r, c)) - 1;   //get index of plane/cylinder at [r, c]

                    if(index < 0) {
                        // No primitive detected
                        outPtr[c] = rgbPtr[c];
                    }
                    else if(associatedIds.contains(index)) {    //shape associated with last frame shape
                        //there is a mask to display 
                        const unsigned int primitiveIndex = static_cast<unsigned int>(associatedIds.at(index));
                        if (colors.size() <= primitiveIndex) {
                            std::cerr << "Id of primitive is greater than available colors" << std::endl;
                        }
                        else
                        {
                            outPtr[c] = colors[primitiveIndex] * 0.5 + rgbPtr[c] * 0.5;
                        }
                    }
                    else {
                        //shape associated with nothing
                        if (colors.size() <= static_cast<unsigned int>(index)) {
                            std::cerr << "Id of primitive is greater than available colors" << std::endl;
                        }
                        else
                        {
                            outPtr[c] = colors[index] * 0.2 + rgbPtr[c] * 0.8;
                        }
                    }
                }
            }

            // Show frame rate and labels
            cv::rectangle(labeledImage, cv::Point(0,0), cv::Point(_width, 20), cv::Scalar(0,0,0), -1);
            if(timeElapsed > 0) {
                std::stringstream fps;
                fps << static_cast<int>((1 / timeElapsed + 0.5)) << " fps";
                cv::putText(labeledImage, fps.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));
            }

            //show plane labels
            if (primitiveSegments.size() > 0){
                std::stringstream text1;
                text1 << "Planes:";
                double pos = _width * 0.25;
                cv::putText(labeledImage, text1.str(), cv::Point(pos, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));


                std::stringstream text2;
                text2 << "Cylinders:";
                pos = _width * 0.60;
                cv::putText(labeledImage, text2.str(), cv::Point(pos, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));


                unsigned int count = 0;
                for(const std::unique_ptr<Primitive>& prim : primitiveSegments){
                    unsigned int id = prim->get_id();

                    if(associatedIds.contains(id))
                        id = associatedIds.at(id);

                    if(id >= CYLINDER_CODE_OFFSET)
                        pos = _width * 0.60;
                    else
                        pos = _width * 0.25;

                    cv::rectangle(labeledImage,  cv::Point(pos + 80 + 15 * count, 6),
                            cv::Point(pos + 90 + 15 * count, 16), 
                            cv::Scalar(
                                colors[id][0],
                                colors[id][1],
                                colors[id][2]),
                            -1);
                    ++count;
                }
            }
        }



        /*
         * Find the planes in the organized depth matrix using region growing
         * Segout will contain a 2D representation of the planes
         */
        void Primitive_Detection::find_primitives(const Eigen::MatrixXf& depthMatrix, primitive_container& primitiveSegments, cv::Mat& segOut) {

            //reset used data structures
            reset_data();

            double t1 = cv::getTickCount();
            //init planar grid
            init_planar_cell_fitting(depthMatrix);
            double td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
            resetTime += td;

            //init and fill histogram
            t1 = cv::getTickCount();
            int remainingPlanarCells = init_histogram();
            td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
            initTime += td;


            t1 = cv::getTickCount();
            intpair_vector cylinder2regionMap;
            grow_planes_and_cylinders(remainingPlanarCells, cylinder2regionMap);
            td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
            growTime += td;

            //merge sparse planes
            uint_vector planeMergeLabels;
            merge_planes(planeMergeLabels);

            t1 = cv::getTickCount();
            //refine planes boundaries and fill the final planes vector
            refine_plane_boundaries(depthMatrix, planeMergeLabels, primitiveSegments);
            td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
            mergeTime += td;

            if(_useCylinderDetection) {
                t1 = cv::getTickCount();
                //refine cylinders boundaries and fill the final cylinders vector
                refine_cylinder_boundaries(depthMatrix, cylinder2regionMap, primitiveSegments); 
                td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
                refineTime += td;
            }

            t1 = cv::getTickCount();
            //set mask image
            set_masked_display(segOut);
            td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
            setMaskTime += td;
        }

        void Primitive_Detection::reset_data() {
            _histogram.reset();

            //planeGrid SHOULD NOT be cleared
            _planeSegments.clear();
            _cylinderSegments.clear();

            _gridPlaneSegmentMap = 0;
            _gridPlaneSegMapEroded = 0;
            _gridCylinderSegMap = 0;
            _gridCylinderSegMapEroded = 0;

            _distancesCellStacked.setZero();

            //reset stacked distances
            //activation map do not need to be cleared
            std::fill_n(_unassignedMask, _totalCellCount, false);
            std::fill_n(_distancesStacked, _height * _width, std::numeric_limits<float>::max());
            std::fill_n(_segMapStacked, _height * _width, 0);
            std::fill_n(_cellDistanceTols, _totalCellCount, 0.0);

            //mat masks do not need to be cleared
            //kernels should not be cleared
        }

        void Primitive_Detection::init_planar_cell_fitting(const Eigen::MatrixXf& depthCloudArray) {
            float sinCosAngleForMerge = sqrt(1 - pow(_minCosAngleForMerge, 2));

            //for each planeGrid cell
            for(unsigned int stackedCellId = 0; stackedCellId < _totalCellCount; ++stackedCellId) {
                //init the plane grid cell
                _planeGrid[stackedCellId]->init_plane_segment(depthCloudArray, stackedCellId);

                if (_planeGrid[stackedCellId]->is_planar()) {
                    int cellDiameter = (
                            depthCloudArray.block(stackedCellId * _pointsPerCellCount + _pointsPerCellCount - 1, 0, 1, 3) - 
                            depthCloudArray.block(stackedCellId * _pointsPerCellCount, 0, 1, 3)
                            ).norm(); 

                    //array of depth metrics: neighbors merging threshold
                    _cellDistanceTols[stackedCellId] = pow(std::min(std::max(cellDiameter * sinCosAngleForMerge, 20.0f), _maxMergeDist), 2);
                }
            }
        }

        int Primitive_Detection::init_histogram() {
            int remainingPlanarCells = 0;

            Eigen::MatrixXd histBins(_totalCellCount, 2);
            for(unsigned int cellId = 0; cellId < _totalCellCount; ++cellId) {  
                if(_planeGrid[cellId]->is_planar()) {
                    const Eigen::Vector3d& planeNormal = _planeGrid[cellId]->get_normal();
                    const double nx = planeNormal[0];
                    const double ny = planeNormal[1];

                    double projNormal = 1 / sqrt(nx * nx + ny * ny);
                    histBins(cellId, 0) = acos( -planeNormal[2] );  //acos(normal.z)
                    histBins(cellId, 1) = atan2(nx * projNormal, ny * projNormal);
                    remainingPlanarCells += 1;
                    _unassignedMask[cellId] = true; 
                }
            }
            _histogram.init_histogram(histBins, _unassignedMask);
            return remainingPlanarCells;
        }

        void Primitive_Detection::grow_planes_and_cylinders(unsigned int remainingPlanarCells, intpair_vector& cylinder2regionMap) {
            unsigned int cylinderCount = 0;
            std::vector<unsigned int> seedCandidates;
            //find seed planes and make them grow
            while(remainingPlanarCells > 0) {
                //get seed candidates
                seedCandidates.clear();
                _histogram.get_points_from_most_frequent_bin(seedCandidates);

                if (seedCandidates.size() < Parameters::get_minimum_plane_seed_count())
                    break;

                //select seed cell with min MSE
                unsigned int seedId = 0;    //should not necessarily stay to 0 after the loop
                float minMSE = INT_MAX;
                for(unsigned int i = 0; i < seedCandidates.size(); ++i) {
                    unsigned int seedCandidate = seedCandidates[i];
                    if(_planeGrid[seedCandidate]->get_MSE() < minMSE) {
                        seedId = static_cast<int>(seedCandidate);
                        minMSE = _planeGrid[seedCandidate]->get_MSE();
                        //if(minMSE <= 0)
                        //    break;
                    }
                }

                //copy plane segment in new object
                Plane_Segment newPlaneSegment(*_planeGrid[seedId]);

                //Seed cell growing
                unsigned int y = static_cast<unsigned int>(seedId / _horizontalCellsCount);
                unsigned int x = static_cast<unsigned int>(seedId % _horizontalCellsCount);

                //activationMap set to false
                std::fill_n(_activationMap, _totalCellCount, false);
                //grow plane region
                region_growing(x, y, newPlaneSegment.get_normal(), newPlaneSegment.get_plane_d());


                //merge activated cells & remove them from histogram
                unsigned int cellActivatedCount = 0;
                for(unsigned int i = 0; i < _totalCellCount; ++i) {
                    if(_activationMap[i]) {
                        newPlaneSegment.expand_segment(_planeGrid[i]);
                        cellActivatedCount += 1;
                        _histogram.remove_point(i);
                        _unassignedMask[i] = false;
                        remainingPlanarCells -= 1;
                    }
                }

                if(cellActivatedCount < Parameters::get_minimum_cell_activated()) {
                    _histogram.remove_point(seedId);
                    continue;
                }

                //fit plane to merged data
                newPlaneSegment.fit_plane();

                if(not _useCylinderDetection or newPlaneSegment.get_score() > 100) {
                    //its certainly a plane or we ignore cylinder detection
                    _planeSegments.push_back(std::make_unique<Plane_Segment>(newPlaneSegment));
                    int currentPlaneCount = _planeSegments.size();
                    //mark cells
                    int i = 0;
                    for(unsigned int r = 0; r < _verticalCellsCount; ++r) {
                        int* row = _gridPlaneSegmentMap.ptr<int>(r);
                        for(unsigned int c = 0; c < _horizontalCellsCount; ++c) {
                            if(_activationMap[i])
                                row[c] = currentPlaneCount;
                            i += 1;
                        }
                    }

                }
                else if(_useCylinderDetection and cellActivatedCount > 5) {
                    //cylinder fitting
                    // It is an extrusion
                    _cylinderSegments.push_back(std::make_unique<Cylinder_Segment>(_planeGrid,_totalCellCount, _activationMap, cellActivatedCount));
                    const cylinder_segment_unique_ptr& cy = _cylinderSegments.back();

                    // Fit planes to subsegments
                    for(unsigned int segId = 0; segId < cy->get_segment_count(); ++segId){
                        newPlaneSegment.clear_plane_parameters();
                        for(unsigned int c = 0; c < cellActivatedCount; ++c){
                            if (cy->is_inlier_at(segId, c)){
                                int localMap = cy->get_local_to_global_mapping(c);
                                newPlaneSegment.expand_segment(_planeGrid[localMap]);
                            }
                        }
                        newPlaneSegment.fit_plane();
                        // Model selection based on MSE
                        if(newPlaneSegment.get_MSE() < cy->get_MSE_at(segId)){
                            //MSE of the plane is less than MSE of the cylinder + this plane 
                            _planeSegments.push_back(std::make_unique<Plane_Segment>(newPlaneSegment));
                            int currentPlaneCount = _planeSegments.size();
                            for(unsigned int c = 0; c < cellActivatedCount; ++c){
                                if (cy->is_inlier_at(segId, c)){
                                    int cellId = cy->get_local_to_global_mapping(c);
                                    _gridPlaneSegmentMap.at<int>(cellId / _horizontalCellsCount, cellId % _horizontalCellsCount) = currentPlaneCount;
                                }
                            }
                        }
                        else {
                            cylinderCount += 1;
                            cylinder2regionMap.push_back(std::make_pair(_cylinderSegments.size() - 1, segId));
                            for(unsigned int c = 0; c < cellActivatedCount; ++c){
                                if (cy->is_inlier_at(segId, c)){
                                    int cellId = cy->get_local_to_global_mapping(c);
                                    _gridCylinderSegMap.at<int>(cellId / _horizontalCellsCount, cellId % _horizontalCellsCount) = cylinderCount;
                                }
                            }
                        }
                    }
                }
            }//\while
        }

        void Primitive_Detection::merge_planes(uint_vector& planeMergeLabels) {
            const unsigned int planeCount = _planeSegments.size();

            Eigen::MatrixXd planesAssocMat = Eigen::MatrixXd::Zero(planeCount, planeCount);
            get_connected_components(_gridPlaneSegmentMap, planesAssocMat);

            for(unsigned int i = 0; i < planeCount; ++i)
                planeMergeLabels.push_back(i);

            for(unsigned int r = 0; r < planesAssocMat.rows(); ++r) {
                unsigned int planeId = planeMergeLabels[r];
                bool planeWasExpanded = false;
                const plane_segment_unique_ptr& testPlane = _planeSegments[planeId];
                const Eigen::Vector3d& testPlaneNormal = testPlane->get_normal();

                for(unsigned int c = r+1; c < planesAssocMat.cols(); ++c) {
                    if(planesAssocMat(r, c)) {
                        const plane_segment_unique_ptr& mergePlane = _planeSegments[c];
                        const Eigen::Vector3d& mergePlaneNormal = mergePlane->get_normal();
                        double cosAngle = testPlaneNormal.dot(mergePlaneNormal);

                        const Eigen::Vector3d& mergePlaneMean = mergePlane->get_mean();
                        double distance = pow(
                                testPlaneNormal.dot(mergePlaneMean) + testPlane->get_plane_d()
                                , 2);

                        if(cosAngle > _minCosAngleForMerge and distance < _maxMergeDist) {
                            //merge plane segments
                            _planeSegments[planeId]->expand_segment(mergePlane);
                            planeMergeLabels[c] = planeId;
                            planeWasExpanded = true;
                        }
                        else {
                            planesAssocMat(r, c) = false;
                        }
                    }
                }
                if(planeWasExpanded)    //plane was merged with other planes
                    _planeSegments[planeId]->fit_plane();
            }
        }

        void Primitive_Detection::refine_plane_boundaries(const Eigen::MatrixXf& depthCloudArray, uint_vector& planeMergeLabels, primitive_container& primitiveSegments) {
            //refine the coarse planes boundaries to smoother versions
            unsigned int planeCount = _planeSegments.size();
            for(unsigned int i = 0; i < planeCount; ++i) {
                if (i != planeMergeLabels[i])
                    continue;

                _mask = cv::Scalar(0);
                for(unsigned int j = i; j < planeCount; ++j) {
                    if(planeMergeLabels[j] == planeMergeLabels[i])
                        _mask.setTo(1, _gridPlaneSegmentMap == j + 1);
                }

                cv::erode(_mask, _maskEroded, _maskCrossKernel);
                double min, max;
                cv::minMaxLoc(_maskEroded, &min, &max);

                if(max == 0)    //completely eroded
                    continue;

                cv::dilate(_mask, _maskDilated, _maskSquareKernel);
                _maskDiff = _maskDilated - _maskEroded;

                //add new plane to final shapes
                primitiveSegments.push_back(std::move(std::make_unique<Plane>(_planeSegments[i], planeMergeLabels[i], _maskDilated)));


                uchar planeNr = (unsigned char)primitiveSegments.size();
                const Eigen::Vector3d& planeNormal = _planeSegments[i]->get_normal();
                float nx = planeNormal[0];
                float ny = planeNormal[1];
                float nz = planeNormal[2];
                float d = _planeSegments[i]->get_plane_d();
                //TODO: better distance metric
                float maxDist = 9 * _planeSegments[i]->get_MSE();

                _gridPlaneSegMapEroded.setTo(planeNr, _maskEroded > 0);

                //cell refinement
                for(unsigned int cellR = 0, stackedCellId = 0; cellR < _verticalCellsCount; ++cellR) {
                    unsigned char* rowPtr = _maskDiff.ptr<uchar>(cellR);

                    for(unsigned int cellC = 0; cellC < _horizontalCellsCount; ++cellC, ++stackedCellId) {
                        unsigned int offset = stackedCellId * _pointsPerCellCount;
                        unsigned int nextOffset = offset + _pointsPerCellCount;

                        if(rowPtr[cellC] > 0) {
                            //compute distance block
                            _distancesCellStacked = 
                                depthCloudArray.block(offset, 0, _pointsPerCellCount, 1).array() * nx +
                                depthCloudArray.block(offset, 1, _pointsPerCellCount, 1).array() * ny +
                                depthCloudArray.block(offset, 2, _pointsPerCellCount, 1).array() * nz +
                                d;

                            //Assign pixel
                            for(unsigned int pt = offset, j = 0; pt < nextOffset; ++j, ++pt) {
                                float dist = pow(_distancesCellStacked(j), 2);
                                if(dist < maxDist and dist < _distancesStacked[pt]) {
                                    _distancesStacked[pt] = dist;
                                    _segMapStacked[pt] = planeNr;
                                }
                            }
                        }
                    }
                }
            }
        }

        void Primitive_Detection::refine_cylinder_boundaries(const Eigen::MatrixXf& depthCloudArray, intpair_vector& cylinderToRegionMap, primitive_container& primitiveSegments) {
            if(not _useCylinderDetection)
                return; //no cylinder detections

            int cylinderFinalCount = 0;
            for(unsigned int i = 0; i < cylinderToRegionMap.size(); i++){
                // Build mask
                _mask = cv::Scalar(0);
                _mask.setTo(1, _gridCylinderSegMap == (i + 1));

                // Erode to obtain borders
                cv::erode(_mask, _maskEroded, _maskCrossKernel);
                double min, max;
                cv::minMaxLoc(_maskEroded, &min, &max);

                // If completely eroded ignore cylinder
                if (max == 0)
                    continue;

                cylinderFinalCount += 1;

                // Dilate to obtain borders
                cv::dilate(_mask, _maskDilated, _maskSquareKernel);
                _maskDiff = _maskDilated - _maskEroded;

                _gridCylinderSegMapEroded.setTo((unsigned char)CYLINDER_CODE_OFFSET + cylinderFinalCount, _maskEroded > 0);

                int regId = cylinderToRegionMap[i].first;
                int subRegId = cylinderToRegionMap[i].second;
                const cylinder_segment_unique_ptr& cylinderSegRef = _cylinderSegments[regId];

                //add new cylinder to final shapes
                primitiveSegments.push_back(std::move(std::make_unique<Cylinder>(cylinderSegRef, CYLINDER_CODE_OFFSET + cylinderFinalCount -  1, _maskDilated)));


                // Get variables needed for point-surface distance computation
                const Eigen::Vector3d P2 = cylinderSegRef->get_axis2_point(subRegId);
                const Eigen::Vector3d P1P2 = P2 - cylinderSegRef->get_axis1_point(subRegId);
                double P1P2Normal = cylinderSegRef->get_axis_normal(subRegId);
                double radius = cylinderSegRef->get_radius(subRegId);
                double maxDist = 9 * cylinderSegRef->get_MSE_at(subRegId);

                // Cell refinement
                for(unsigned int cellR = 0, stackedCellId = 0; cellR < _verticalCellsCount; cellR += 1){
                    uchar* rowPtr = _maskDiff.ptr<uchar>(cellR);
                    for(unsigned int cellC = 0; cellC < _horizontalCellsCount; cellC++, stackedCellId++) {
                        unsigned int offset = stackedCellId * _pointsPerCellCount;
                        unsigned int nextOffset = offset + _pointsPerCellCount;
                        if(rowPtr[cellC] > 0){
                            // Update cells
                            for(unsigned int pt = offset, j = 0; pt < nextOffset; pt++, j++) {
                                Eigen::Vector3d point = depthCloudArray.row(pt).cast<double>();
                                if(point(2) > 0){
                                    double dist = pow(P1P2.cross(point - P2).norm() / P1P2Normal - radius, 2);
                                    if(dist < maxDist and dist < _distancesStacked[pt]){ 
                                        _distancesStacked[pt] = dist;
                                        _segMapStacked[pt] = CYLINDER_CODE_OFFSET + cylinderFinalCount;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        void Primitive_Detection::set_masked_display(cv::Mat& segOut) {
            //copy and rearranging
            // Copy inlier list to matrix form
            for(unsigned int cellR = 0; cellR < _verticalCellsCount; cellR += 1){
                uchar* gridPlaneErodedRowPtr = _gridPlaneSegMapEroded.ptr<uchar>(cellR);
                uchar* gridCylinderErodedRowPtr = _gridCylinderSegMapEroded.ptr<uchar>(cellR);
                unsigned int rOffset = cellR * _cellHeight;
                unsigned int rLimit = rOffset + _cellHeight;

                for(unsigned int cellC = 0; cellC < _horizontalCellsCount; cellC += 1){
                    unsigned int cOffset = cellC * _cellWidth;

                    if (gridPlaneErodedRowPtr[cellC] > 0){
                        // Set rectangle equal to assigned cell
                        segOut(cv::Rect(cOffset, rOffset, _cellWidth, _cellHeight)).setTo(gridPlaneErodedRowPtr[cellC]);
                    } 
                    else if(gridCylinderErodedRowPtr[cellC] > 0) {
                        // Set rectangle equal to assigned cell
                        segOut(cv::Rect(cOffset, rOffset, _cellWidth, _cellHeight)).setTo(gridCylinderErodedRowPtr[cellC]);
                    }
                    else {
                        unsigned int cLimit = cOffset + _cellWidth;
                        // Set cell pixels one by one
                        uchar* stackPtr = &_segMapStacked[_pointsPerCellCount * cellR * _horizontalCellsCount + _pointsPerCellCount * cellC];
                        for(unsigned int r = rOffset, i = 0; r < rLimit; r++){
                            uchar* rowPtr = segOut.ptr<uchar>(r);
                            for(unsigned int c = cOffset; c < cLimit; c++, i++){
                                uchar id = stackPtr[i];
                                if(id > 0) {
                                    rowPtr[c] = id;
                                }
                            }
                        }
                    }
                }
            }
        }


        void Primitive_Detection::get_connected_components(const cv::Mat& segmentMap, Eigen::MatrixXd& planesAssociationMatrix) {
            unsigned int rows2scanCount = segmentMap.rows - 1;
            unsigned int cols2scanCount = segmentMap.cols - 1;

            for(unsigned int r = 0; r < rows2scanCount; r += 1) {
                const int *row = segmentMap.ptr<int>(r);
                const int *rowBelow = segmentMap.ptr<int>(r + 1);
                for(unsigned int c = 0; c < cols2scanCount; c += 1) {
                    int pixelValue = row[c];
                    if(pixelValue > 0) {
                        if(row[c + 1] > 0 and pixelValue != row[c + 1]) {
                            planesAssociationMatrix(pixelValue - 1, row[c + 1] - 1) = true;
                            planesAssociationMatrix(row[c + 1] - 1, pixelValue - 1) = true;
                        }
                        if(rowBelow[c] > 0 and pixelValue != rowBelow[c]) {
                            planesAssociationMatrix(pixelValue - 1, rowBelow[c] - 1) = true;
                            planesAssociationMatrix(rowBelow[c] - 1, pixelValue - 1) = true;
                        }
                    }
                }
            }
        }


        void Primitive_Detection::region_growing(const unsigned short x, const unsigned short y, const Eigen::Vector3d& seedPlaneNormal, const double seedPlaneD) {
            unsigned int index = x + _horizontalCellsCount * y;
            if (index >= _totalCellCount or 
                    not _unassignedMask[index] or _activationMap[index]) {
                //pixel is not part of a component or already labelled
                return;
            }

            const Eigen::Vector3d& secPlaneNormal = _planeGrid[index]->get_normal();
            const Eigen::Vector3d& secPlaneMean = _planeGrid[index]->get_mean();
            const double& secPlaneD = _planeGrid[index]->get_plane_d();

            if (
                    //planeGrid[index].is_depth_discontinuous() 
                    seedPlaneNormal.dot(secPlaneNormal) < _minCosAngleForMerge
                    or pow(seedPlaneNormal.dot(secPlaneMean) + seedPlaneD, 2) > _cellDistanceTols[index]
               )//angle between planes < threshold or dist between planes > threshold
                return;
            _activationMap[index] = true;

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



        Primitive_Detection::~Primitive_Detection() {
            delete []_unassignedMask;
            delete []_activationMap;
            delete []_segMapStacked;
            delete []_distancesStacked;
            delete []_cellDistanceTols;

            delete []_planeGrid;
        }

}
}
}

