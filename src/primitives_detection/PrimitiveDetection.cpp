#include <limits>
#include "PrimitiveDetection.hpp"
#include "Parameters.hpp"


//lib simplification
using namespace primitiveDetection;
using namespace Eigen;
using namespace std;

/*
 * Default constructor
 * Given the depth image height and width and an organized matrix of 3D points, sets the plane detection algorithm
 * blocSize is the height and width of a plane grid cell
 * minCosAngleForMerge 
 * maxMergeDistance
 */
Primitive_Detection::Primitive_Detection(const unsigned int height, const unsigned int width, const unsigned int blocSize, const float minCosAngleForMerge, const float maxMergeDistance, const bool useCylinderDetection)
    :  histogram(20), 
    width(width), height(height),  blocSize(blocSize), 
    pointsPerCellCount(blocSize * blocSize), 
    minCosAngleForMerge(minCosAngleForMerge), maxMergeDist(maxMergeDistance),
    useCylinderDetection(useCylinderDetection),
    cellWidth(blocSize), cellHeight(blocSize),
    horizontalCellsCount(width / cellWidth), verticalCellsCount(height / cellHeight),
    totalCellCount(verticalCellsCount * horizontalCellsCount)

{
    //Init variables
    this->activationMap = new bool[this->totalCellCount];
    this->unassignedMask = new bool[this->totalCellCount];
    this->distancesStacked = new float[this->width * this->height];
    this->segMapStacked = new unsigned char[this->width * this->height];
    this->cellDistanceTols = new float[this->totalCellCount];

    this->gridPlaneSegmentMap = cv::Mat_<int>(this->verticalCellsCount, this->horizontalCellsCount, 0);
    this->gridPlaneSegMapEroded = cv::Mat_<uchar>(verticalCellsCount, horizontalCellsCount, uchar(0));
    this->gridCylinderSegMap = cv::Mat_<int>(this->verticalCellsCount, this->horizontalCellsCount, 0);
    this->gridCylinderSegMapEroded = cv::Mat_<uchar>(verticalCellsCount, horizontalCellsCount, uchar(0));

    this->distancesCellStacked = ArrayXf::Zero(this->pointsPerCellCount, 1);

    this->mask = cv::Mat(verticalCellsCount, horizontalCellsCount, CV_8U);
    this->maskEroded = cv::Mat(verticalCellsCount, horizontalCellsCount, CV_8U);
    this->maskDilated = cv::Mat(verticalCellsCount, horizontalCellsCount, CV_8U);
    this->maskDiff = cv::Mat(verticalCellsCount, horizontalCellsCount, CV_8U);

    //init kernels
    this->maskSquareKernel = cv::Mat::ones(3, 3, CV_8U);
    this->maskCrossKernel = cv::Mat::ones(3, 3, CV_8U);

    this->maskCrossKernel.at<uchar>(0,0) = 0;
    this->maskCrossKernel.at<uchar>(2,2) = 0;
    this->maskCrossKernel.at<uchar>(0,2) = 0;
    this->maskCrossKernel.at<uchar>(2,0) = 0;

    //array of unique_ptr<Plane_Segment>
    this->planeGrid = new unique_ptr<Plane_Segment>[this->totalCellCount];
    for(int i = 0; i < this->totalCellCount; i += 1) {
        //fill with empty nodes
        this->planeGrid[i] = make_unique<Plane_Segment>(this->cellWidth, this->pointsPerCellCount);
    }

    //perf measurments
    resetTime = 0;
    initTime = 0;
    growTime = 0;
    mergeTime = 0;
    refineTime = 0;
    setMaskTime = 0;
}


/*
 * in inputImage input RGB image on which to put the planes and cylinder masks
 * in maskImage Output image of find_plane_regions
 * out labeledImage Output RGB image, similar to input but with planes and cylinder masks
 */
void Primitive_Detection::apply_masks(const cv::Mat& inputImage, const std::vector<cv::Vec3b>& colors, const cv::Mat& maskImage, const std::vector<Plane_Segment>& planeParams, const std::vector<Cylinder_Segment>& cylinderParams, cv::Mat& labeledImage, const double timeElapsed) {
    //apply masks on image
    for(int r = 0; r < this->height; r++){
        const cv::Vec3b* rgbPtr = inputImage.ptr<cv::Vec3b>(r);
        cv::Vec3b* outPtr = labeledImage.ptr<cv::Vec3b>(r);
        for(int c = 0; c < this->width; c++){
            const int index = maskImage.at<uchar>(r, c);   //get index of plane/cylinder at [r, c]
            //int index = maskImage(r, c);   //get index of plane/cylinder at [r, c]
            if(index <= 0) {
                outPtr[c] = rgbPtr[c] / 2;
            }
            else {
                //there is a mask to display 
                outPtr[c] = colors[index - 1] / 2 + rgbPtr[c] / 2;
            }
        }
    }

    // Show frame rate and labels
    cv::rectangle(labeledImage, cv::Point(0,0), cv::Point(this->width, 20), cv::Scalar(0,0,0), -1);
    if(timeElapsed > 0) {
        std::stringstream fps;
        fps << (int)(1 / timeElapsed + 0.5) << " fps";
        cv::putText(labeledImage, fps.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));
    }

    //show plane labels
    if (planeParams.size() > 0){
        std::stringstream text;
        text << "Planes:";
        double pos = this->width * 0.25;
        cv::putText(labeledImage, text.str(), cv::Point(pos, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));
        for(unsigned int j = 0; j < planeParams.size(); j += 1){
            cv::rectangle(labeledImage,  cv::Point(pos + 80 + 15 * j, 6),
                    cv::Point(pos + 90 + 15 * j, 16), 
                    cv::Scalar(
                        colors[j][0],
                        colors[j][1],
                        colors[j][2]),
                    -1);
        }
    }
    //show cylinder labels
    if (cylinderParams.size() > 0){
        std::stringstream text;
        text << "Cylinders:";
        double pos = this->width * 0.60;
        cv::putText(labeledImage, text.str(), cv::Point(pos, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));
        for(unsigned int j = 0; j < cylinderParams.size(); j += 1){
            cv::rectangle(labeledImage,  cv::Point(pos + 80 + 15 * j, 6),
                    cv::Point(pos + 90 + 15 * j, 16), 
                    cv::Scalar(
                        colors[CYLINDER_CODE_OFFSET + j][0],
                        colors[CYLINDER_CODE_OFFSET + j][1],
                        colors[CYLINDER_CODE_OFFSET + j][2]),
                    -1);
        }
    }
}



/*
 * Find the planes in the organized depth matrix using region growing
 * Segout will contain a 2D representation of the planes
 */
void Primitive_Detection::find_primitives(const Eigen::MatrixXf& depthMatrix, std::vector<Plane_Segment>& planeSegmentsFinal, std::vector<Cylinder_Segment>& cylinderSegmentsFinal, cv::Mat& segOut) {

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
    std::vector<std::pair<int,int>> cylinder2regionMap;
    grow_planes_and_cylinders(cylinder2regionMap, remainingPlanarCells);
    td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
    growTime += td;

    //merge sparse planes
    vector<unsigned int> planeMergeLabels;
    merge_planes(planeMergeLabels);

    t1 = cv::getTickCount();
    //refine planes boundaries and fill the final planes vector
    refine_plane_boundaries(depthMatrix, planeMergeLabels, planeSegmentsFinal);
    td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
    mergeTime += td;

    t1 = cv::getTickCount();
    //refine cylinders boundaries and fill the final cylinders vector
    refine_cylinder_boundaries(depthMatrix, cylinder2regionMap, cylinderSegmentsFinal); 
    td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
    refineTime += td;

    t1 = cv::getTickCount();
    //set mask image
    set_masked_display(segOut);
    td = (cv::getTickCount() - t1) / (double)cv::getTickFrequency();
    setMaskTime += td;
}

/*
 *  Reset the stored data in prevision of another analysis 
 *
 */
void Primitive_Detection::reset_data() {
    this->histogram.reset();

    //planeGrid SHOULD NOT be cleared
    this->planeSegments.clear();
    this->cylinderSegments.clear();

    this->gridPlaneSegmentMap = 0;
    this->gridPlaneSegMapEroded = 0;
    this->gridCylinderSegMap = 0;
    this->gridCylinderSegMapEroded = 0;

    this->distancesCellStacked.setZero();

    //reset stacked distances
    //activation map do not need to be cleared
    std::fill_n(this->unassignedMask, this->totalCellCount, false);
    std::fill_n(this->distancesStacked, this->height * this->width, std::numeric_limits<float>::max());
    std::fill_n(this->segMapStacked, this->height * this->width, 0);
    std::fill_n(this->cellDistanceTols, this->totalCellCount, 0.0);

    //mat masks do not need to be cleared
    //kernels should not be cleared
}

/*
 *  Init planeGrid and cellDistanceTols
 *
 */
void Primitive_Detection::init_planar_cell_fitting(const MatrixXf& depthCloudArray) {
    float sinCosAngleForMerge = sqrt(1 - pow(this->minCosAngleForMerge, 2));

    //for each planeGrid cell
    for(int stackedCellId = 0; stackedCellId < this->totalCellCount; stackedCellId += 1) {
        //init the plane grid cell
        this->planeGrid[stackedCellId]->init_plane_segment(depthCloudArray, stackedCellId);

        if (this->planeGrid[stackedCellId]->is_planar()) {
            int cellDiameter = (
                    depthCloudArray.block(stackedCellId * this->pointsPerCellCount + this->pointsPerCellCount - 1, 0, 1, 3) - 
                    depthCloudArray.block(stackedCellId * this->pointsPerCellCount, 0, 1, 3)
                    ).norm(); 

            //array of depth metrics: neighbors merging threshold
            this->cellDistanceTols[stackedCellId] = pow(min(max(cellDiameter * sinCosAngleForMerge, 20.0f), this->maxMergeDist), 2);
        }
    }
}

/*
 *  Initialize and fill the histogram
 *
 * returns the number of initial planar surfaces
 */
int Primitive_Detection::init_histogram() {
    int remainingPlanarCells = 0;

    MatrixXd histBins(this->totalCellCount, 2);
    for(int cellId = 0; cellId < this->totalCellCount; cellId += 1) {  
        if(this->planeGrid[cellId]->is_planar()) {
            const Vector3d& planeNormal = this->planeGrid[cellId]->get_normal();
            const double nx = planeNormal[0];
            const double ny = planeNormal[1];

            double projNormal = 1 / sqrt(nx * nx + ny * ny);
            histBins(cellId, 0) = acos( -planeNormal[2] );  //acos(normal.z)
            histBins(cellId, 1) = atan2(nx * projNormal, ny * projNormal);
            remainingPlanarCells += 1;
            this->unassignedMask[cellId] = true; 
        }
    }
    histogram.init_histogram(histBins, this->unassignedMask);
    return remainingPlanarCells;
}

/*
 *  grow planes and find cylinders from those planes
 */
void Primitive_Detection::grow_planes_and_cylinders(std::vector<std::pair<int,int>>& cylinder2regionMap, int remainingPlanarCells) {
    int cylinderCount = 0;
    //find seed planes and make them grow
    while(remainingPlanarCells > 0) {
        //get seed candidates
        vector<int> seedCandidates;
        this->histogram.get_points_from_most_frequent_bin(seedCandidates);

        if (seedCandidates.size() < MIN_SEED_COUNT)
            break;

        //select seed cell with min MSE
        int seedId = -1;
        float minMSE = INT_MAX;
        for(unsigned int i = 0; i < seedCandidates.size(); i++) {
            int seedCandidate = seedCandidates[i];
            if(this->planeGrid[seedCandidate]->get_MSE() < minMSE) {
                seedId = seedCandidate;
                minMSE = this->planeGrid[seedCandidate]->get_MSE();
                //if(minMSE <= 0)
                //    break;
            }
        }
        if (seedId < 0) {
            //error
            cerr << "Error : no min MSE in graph" << std::endl;
            exit(-1);
        }

        //copy plane segment in new object
        Plane_Segment newPlaneSegment(*this->planeGrid[seedId]);

        //Seed cell growing
        int y = seedId / this->horizontalCellsCount;
        int x = seedId % this->horizontalCellsCount;

        //activationMap set to false
        std::fill_n(this->activationMap, this->totalCellCount, false);
        //grow plane region
        region_growing(x, y, newPlaneSegment.get_normal(), newPlaneSegment.get_plane_d());


        //merge activated cells & remove them from histogram
        unsigned int cellActivatedCount = 0;
        for(int i = 0; i < this->totalCellCount; i+=1) {
            if(this->activationMap[i]) {
                newPlaneSegment.expand_segment(this->planeGrid[i]);
                cellActivatedCount += 1;
                this->histogram.remove_point(i);
                this->unassignedMask[i] = false;
                remainingPlanarCells -= 1;
            }
        }

        if(cellActivatedCount < MIN_CELL_ACTIVATED) {
            this->histogram.remove_point(seedId);
            continue;
        }

        //fit plane to merged data
        newPlaneSegment.fit_plane();

        if(not this->useCylinderDetection or newPlaneSegment.get_score() > 100) {
            //its certainly a plane or we ignore cylinder detection
            this->planeSegments.push_back(make_unique<Plane_Segment>(newPlaneSegment));
            int currentPlaneCount = this->planeSegments.size();
            //mark cells
            int i = 0;
            for(int r = 0; r < this->verticalCellsCount; r += 1) {
                int* row = this->gridPlaneSegmentMap.ptr<int>(r);
                for(int c = 0; c < this->horizontalCellsCount; c += 1) {
                    if(this->activationMap[i])
                        row[c] = currentPlaneCount;
                    i += 1;
                }
            }

        }
        else if(this->useCylinderDetection and cellActivatedCount > 5) {
            //cylinder fitting
            // It is an extrusion
            this->cylinderSegments.push_back(make_unique<Cylinder_Segment>(this->planeGrid,this->totalCellCount, this->activationMap, cellActivatedCount));
            const unique_ptr<Cylinder_Segment>& cy = this->cylinderSegments.back();

            // Fit planes to subsegments
            for(int segId = 0; segId < cy->get_segment_count(); segId++){
                newPlaneSegment.clear_plane_parameters();
                for(unsigned int c = 0; c < cellActivatedCount; c++){
                    if (cy->get_inlier_at(segId, c)){
                        int localMap = cy->get_local_to_global_mapping(c);
                        newPlaneSegment.expand_segment(this->planeGrid[localMap]);
                    }
                }
                newPlaneSegment.fit_plane();
                // Model selection based on MSE
                if(newPlaneSegment.get_MSE() < cy->get_MSE_at(segId)){
                    this->planeSegments.push_back(make_unique<Plane_Segment>(newPlaneSegment));
                    int currentPlaneCount = this->planeSegments.size();
                    for(unsigned int c = 0; c < cellActivatedCount; c++){
                        if (cy->get_inlier_at(segId, c)){
                            int cellId = cy->get_local_to_global_mapping(c);
                            this->gridPlaneSegmentMap.at<int>(cellId / this->horizontalCellsCount, cellId % this->horizontalCellsCount) = currentPlaneCount;
                        }
                    }
                }else{
                    cylinderCount += 1;
                    cylinder2regionMap.push_back(make_pair(cylinderSegments.size() - 1, segId));
                    for(unsigned int c = 0; c < cellActivatedCount; c++){
                        if (cy->get_inlier_at(segId, c)){
                            int cellId = cy->get_local_to_global_mapping(c);
                            this->gridCylinderSegMap.at<int>(cellId / this->horizontalCellsCount, cellId % this->horizontalCellsCount) = cylinderCount;
                        }
                    }
                }
            }
        }
    }//\while
}

/*
 *  Merge close planes by comparing normals and MSE
 */
void Primitive_Detection::merge_planes(vector<unsigned int>& planeMergeLabels) {
    const unsigned int planeCount = this->planeSegments.size();

    MatrixXd planesAssocMat = MatrixXd::Zero(planeCount, planeCount);
    get_connected_components(this->gridPlaneSegmentMap, planesAssocMat);

    for(unsigned int i = 0; i < planeCount; i += 1)
        planeMergeLabels.push_back(i);

    for(unsigned int r = 0; r < planesAssocMat.rows(); r += 1) {
        unsigned int planeId = planeMergeLabels[r];
        bool planeWasExpanded = false;
        const unique_ptr<Plane_Segment>& testPlane = this->planeSegments[planeId];
        const Vector3d& testPlaneNormal = testPlane->get_normal();

        for(unsigned int c = r+1; c < planesAssocMat.cols(); c += 1) {
            if(planesAssocMat(r, c)) {
                const unique_ptr<Plane_Segment>& mergePlane = this->planeSegments[c];
                const Vector3d& mergePlaneNormal = mergePlane->get_normal();
                double cosAngle = testPlaneNormal.dot(mergePlaneNormal);

                const Vector3d& mergePlaneMean = mergePlane->get_mean();
                double distance = pow(
                        testPlaneNormal.dot(mergePlaneMean) + testPlane->get_plane_d()
                        , 2);

                if(cosAngle > this->minCosAngleForMerge and distance < this->maxMergeDist) {
                    //merge plane segments
                    this->planeSegments[planeId]->expand_segment(mergePlane);
                    planeMergeLabels[c] = planeId;
                    planeWasExpanded = true;
                }
                else {
                    planesAssocMat(r, c) = false;
                }
            }
        }
        if(planeWasExpanded)    //plane was merged with other planes
            planeSegments[planeId]->fit_plane();
    }
}

/*
 *  Refine the final plane edges in mask images
 */
void Primitive_Detection::refine_plane_boundaries(const MatrixXf& depthCloudArray, vector<unsigned int>& planeMergeLabels, vector<Plane_Segment>& planeSegmentsFinal) {
    //refine the coarse planes boundaries to smoother versions
    unsigned int planeCount = this->planeSegments.size();
    for(unsigned int i = 0; i < planeCount; i += 1) {
        if (i != planeMergeLabels[i])
            continue;

        this->mask = cv::Scalar(0);
        for(unsigned int j = i; j < planeCount; j += 1) {
            if(planeMergeLabels[j] == planeMergeLabels[i])
                this->mask.setTo(1, this->gridPlaneSegmentMap == j + 1);
        }

        cv::erode(this->mask, this->maskEroded, this->maskCrossKernel);
        double min, max;
        cv::minMaxLoc(this->maskEroded, &min, &max);

        if(max == 0)    //completely eroded
            continue;

        //copy in new object
        planeSegmentsFinal.push_back(*this->planeSegments[i]);

        cv::dilate(this->mask, this->maskDilated, this->maskSquareKernel);
        this->maskDiff = this->maskDilated - this->maskEroded;

        uchar planeNr = (unsigned char)planeSegmentsFinal.size();
        const Vector3d& planeNormal = this->planeSegments[i]->get_normal();
        float nx = planeNormal[0];
        float ny = planeNormal[1];
        float nz = planeNormal[2];
        float d = this->planeSegments[i]->get_plane_d();
        //TODO: better distance metric
        float maxDist = 9 * this->planeSegments[i]->get_MSE();

        this->gridPlaneSegMapEroded.setTo(planeNr, this->maskEroded > 0);

        //cell refinement
        for (int cellR = 0, stackedCellId = 0; cellR < this->verticalCellsCount; cellR += 1) {
            unsigned char* rowPtr = this->maskDiff.ptr<uchar>(cellR);

            for(int cellC = 0; cellC < this->horizontalCellsCount; cellC++, stackedCellId++) {
                int offset = stackedCellId * this->pointsPerCellCount;
                int nextOffset = offset + this->pointsPerCellCount;

                if(rowPtr[cellC] > 0) {
                    //compute distance block
                    distancesCellStacked = 
                        depthCloudArray.block(offset, 0, this->pointsPerCellCount, 1).array() * nx +
                        depthCloudArray.block(offset, 1, this->pointsPerCellCount, 1).array() * ny +
                        depthCloudArray.block(offset, 2, this->pointsPerCellCount, 1).array() * nz +
                        d;

                    //Assign pixel
                    for(int pt = offset, j = 0; pt < nextOffset; j += 1, pt += 1) {
                        float dist = pow(distancesCellStacked(j), 2);
                        if(dist < maxDist and dist < this->distancesStacked[pt]) {
                            this->distancesStacked[pt] = dist;
                            this->segMapStacked[pt] = planeNr;
                        }
                    }
                }
            }
        }
    }
}

/*
 *  Refine the cylinder edges in mask images
 */
void Primitive_Detection::refine_cylinder_boundaries(const MatrixXf& depthCloudArray, std::vector<pair<int, int>>& cylinderToRegionMap, std::vector<Cylinder_Segment>& cylinderSegmentsFinal) {
    if(not this->useCylinderDetection)
        return; //no cylinder detections

    int cylinderCount = cylinderToRegionMap.size();

    int cylinderFinalCount = 0;
    for(int i = 0; i < cylinderCount; i++){
        // Build mask
        this->mask = cv::Scalar(0);
        this->mask.setTo(1, this->gridCylinderSegMap == (i + 1));

        // Erode to obtain borders
        cv::erode(this->mask, this->maskEroded, this->maskCrossKernel);
        double min, max;
        cv::minMaxLoc(this->maskEroded, &min, &max);

        // If completely eroded ignore cylinder
        if (max == 0)
            continue;

        cylinderFinalCount += 1;

        // Dilate to obtain borders
        cv::dilate(this->mask, this->maskDilated, this->maskSquareKernel);
        this->maskDiff = this->maskDilated - this->maskEroded;

        this->gridCylinderSegMapEroded.setTo((unsigned char)CYLINDER_CODE_OFFSET + cylinderFinalCount, this->maskEroded > 0);

        int regId = cylinderToRegionMap[i].first;
        int subRegId = cylinderToRegionMap[i].second;
        const unique_ptr<Cylinder_Segment>& cylinderSegRef = this->cylinderSegments[regId];

        cylinderSegmentsFinal.push_back(Cylinder_Segment(*cylinderSegRef, subRegId));


        // Get variables needed for point-surface distance computation
        const Eigen::Vector3d P2 = cylinderSegRef->get_axis2_point(subRegId);
        const Eigen::Vector3d P1P2 = P2 - cylinderSegRef->get_axis1_point(subRegId);
        double P1P2Normal = cylinderSegRef->get_axis_normal(subRegId);
        double radius = cylinderSegRef->get_radius(subRegId);
        double maxDist = 9 * cylinderSegRef->get_MSE_at(subRegId);

        // Cell refinement
        for (int cellR = 0, stackedCellId = 0; cellR < this->verticalCellsCount; cellR += 1){
            uchar* rowPtr = this->maskDiff.ptr<uchar>(cellR);
            for (int cellC = 0; cellC < this->horizontalCellsCount; cellC++, stackedCellId++) {
                int offset = stackedCellId * this->pointsPerCellCount;
                int nextOffset = offset + this->pointsPerCellCount;
                if(rowPtr[cellC] > 0){
                    // Update cells
                    for(int pt = offset, j = 0; pt < nextOffset; pt++, j++) {
                        Eigen::Vector3d point = depthCloudArray.row(pt).cast<double>();
                        if(point(2) > 0){
                            double dist = pow(P1P2.cross(point - P2).norm() / P1P2Normal - radius, 2);
                            if(dist < maxDist and dist < this->distancesStacked[pt]){ 
                                this->distancesStacked[pt] = dist;
                                this->segMapStacked[pt] = CYLINDER_CODE_OFFSET + cylinderFinalCount;
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 *  Set output image pixel value with the index of the detected shape
 */
void Primitive_Detection::set_masked_display(cv::Mat& segOut) {
    //copy and rearranging
    // Copy inlier list to matrix form
    for (int cellR = 0; cellR < this->verticalCellsCount; cellR += 1){
        uchar* gridPlaneErodedRowPtr = this->gridPlaneSegMapEroded.ptr<uchar>(cellR);
        uchar* gridCylinderErodedRowPtr = this->gridCylinderSegMapEroded.ptr<uchar>(cellR);
        int rOffset = cellR * this->cellHeight;
        int rLimit = rOffset + this->cellHeight;

        for (int cellC = 0; cellC < this->horizontalCellsCount; cellC += 1){
            int cOffset = cellC * this->cellWidth;

            if (gridPlaneErodedRowPtr[cellC] > 0){
                // Set rectangle equal to assigned cell
                segOut(cv::Rect(cOffset, rOffset, this->cellWidth, this->cellHeight)).setTo(gridPlaneErodedRowPtr[cellC]);
            } 
            else if(gridCylinderErodedRowPtr[cellC] > 0) {
                // Set rectangle equal to assigned cell
                segOut(cv::Rect(cOffset, rOffset, this->cellWidth, this->cellHeight)).setTo(gridCylinderErodedRowPtr[cellC]);
            }
            else {
                int cLimit = cOffset + this->cellWidth;
                // Set cell pixels one by one
                uchar* stackPtr = &this->segMapStacked[this->pointsPerCellCount * cellR * this->horizontalCellsCount + this->pointsPerCellCount * cellC];
                for(int r = rOffset, i = 0; r < rLimit; r++){
                    uchar* rowPtr = segOut.ptr<uchar>(r);
                    for(int c = cOffset; c < cLimit; c++, i++){
                        if(stackPtr[i] > 0){
                            rowPtr[c] = stackPtr[i];
                        }
                    }
                }
            }
        }
    }
}




/*
 *  Fill an association matrix that links connected plane components
 */
void Primitive_Detection::get_connected_components(const cv::Mat& segmentMap, Eigen::MatrixXd& planesAssociationMatrix) {
    int rows2scanCount = segmentMap.rows - 1;
    int cols2scanCount = segmentMap.cols - 1;

    for(int r = 0; r < rows2scanCount; r += 1) {
        const int *row = segmentMap.ptr<int>(r);
        const int *rowBelow = segmentMap.ptr<int>(r + 1);
        for(int c = 0; c < cols2scanCount; c += 1) {
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


/*
 *  Recursively Grow a plane seed and merge it with it's neighbors
 */
void Primitive_Detection::region_growing(const unsigned short x, const unsigned short y, const Vector3d& seedPlaneNormal, const double seedPlaneD) {
    int index = x + horizontalCellsCount * y;
    if (index >= horizontalCellsCount * verticalCellsCount or 
            not this->unassignedMask[index] or this->activationMap[index]) {
        //pixel is not part of a component or already labelled
        return;
    }

    const Vector3d& secPlaneNormal = planeGrid[index]->get_normal();
    const Vector3d& secPlaneMean = planeGrid[index]->get_mean();
    const double& secPlaneD = planeGrid[index]->get_plane_d();

    if (
            //planeGrid[index].is_depth_discontinuous() 
            seedPlaneNormal.dot(secPlaneNormal) < this->minCosAngleForMerge
            or pow(seedPlaneNormal.dot(secPlaneMean) + seedPlaneD, 2) > this->cellDistanceTols[index]
       )//angle between planes < threshold or dist between planes > threshold
        return;
    activationMap[index] = true;

    // Now label the 4 neighbours:
    if (x > 0)
        region_growing(x - 1, y, secPlaneNormal, secPlaneD);   // left  pixel
    if (x < this->width - 1)  
        region_growing(x + 1, y, secPlaneNormal, secPlaneD);  // right pixel
    if (y > 0)        
        region_growing(x, y - 1, secPlaneNormal, secPlaneD);   // upper pixel 
    if (y < this->height - 1) 
        region_growing(x, y + 1, secPlaneNormal, secPlaneD);   // lower pixel
}



Primitive_Detection::~Primitive_Detection() {
    delete []this->unassignedMask;
    delete []this->activationMap;
    delete []this->segMapStacked;
    delete []this->distancesStacked;
    delete []this->cellDistanceTols;

    delete []this->planeGrid;
}





