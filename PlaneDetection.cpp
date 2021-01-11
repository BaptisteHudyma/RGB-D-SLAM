#include <limits>
#include "PlaneDetection.hpp"


//lib simplification
using namespace planeDetection;
using namespace Eigen;
using namespace std;

/*
 * Default constructor
 * Given the depth image height and width and an organized matrix of 3D points, sets the plane detection algorithm
 * blocSize is the height and width of a plane grid cell
 * minCosAngleForMerge 
 * maxMergeDistance
 */
Plane_Detection::Plane_Detection(unsigned int height, unsigned int width, unsigned int blocSize, float minCosAngleForMerge, float maxMergeDistance, bool useCylinderDetection)
    :  histogram(20), width(width), height(height),  blocSize(blocSize), pointsPerCellCount(blocSize * blocSize), minCosAngleForMerge(minCosAngleForMerge), maxMergeDist(maxMergeDistance)
{
    //Init variables
    this->cellWidth = this->blocSize;
    this->cellHeight = this->blocSize;

    this->horizontalCellsCount = this->width / this->cellWidth;
    this->verticalCellsCount = this->height / this->cellHeight;
    this->totalCellCount = this->verticalCellsCount * this->horizontalCellsCount;

    this->activationMap = new bool[this->totalCellCount];
    this->unassignedMask = new bool[this->totalCellCount];
    this->distancesStacked = new float[this->width * this->height];
    this->segMapStacked = new unsigned char[this->width * this->height];

    //this->cylinder_detection = cylinder_detection;
    this->useCylinderDetection = useCylinderDetection;

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
    this->maskSquareKernel = cv::Mat::ones(3,3,CV_8U);
    this->maskCrossKernel = cv::Mat::ones(3,3,CV_8U);

    this->maskCrossKernel.at<uchar>(0,0) = 0;
    this->maskCrossKernel.at<uchar>(2,2) = 0;
    this->maskCrossKernel.at<uchar>(0,2) = 0;
    this->maskCrossKernel.at<uchar>(2,0) = 0;

    for(int i = 0; i < this->totalCellCount; i += 1) {
        //fill with null nodes
        this->planeGrid.push_back(make_unique<Plane_Segment>(this->cellWidth, this->pointsPerCellCount));
    }

}

/*
 *  Reset the stored data in prevision of another analysis 
 *
 */
void Plane_Detection::reset_data() {
    this->distancesCellStacked.setZero();
    this->gridPlaneSegmentMap = 0;
    this->gridPlaneSegMapEroded = 0;
    this->gridCylinderSegMap = 0;
    this->gridCylinderSegMapEroded = 0;

    //reset stacked distances
    std::fill_n(this->segMapStacked, this->height * this->width, 0);
    std::fill_n(this->distancesStacked, this->height * this->width, std::numeric_limits<float>::max());

    this->planeSegments.clear();
    this->cylinderSegments.clear();
}

/*
 * Find the planes in the organized depth matrix using region growing
 * Segout will contain a 2D representation of the planes
 */
void Plane_Detection::find_plane_regions(Eigen::MatrixXf& depthMatrix, std::vector<Plane_Segment>& planeSegmentsFinal, std::vector<Cylinder_Segment>& cylinderSegmentsFinal, cv::Mat& segOut) {
    //reset used data structures
    reset_data();

    //init planar grid
    vector<float> cellDistanceTols(this->totalCellCount, 0); 
    init_planar_cell_fitting(depthMatrix, cellDistanceTols);

    //init and fill histogram
    int remainingPlanarCells = init_histogram();
    int cylinderCount = 0;
    std::vector<std::pair<int,int>> cylinder2regionMap;

    //find seed planes and make them grow
    while(remainingPlanarCells > 0) {
        //get seed candidates
        vector<int> seedCandidates;
        this->histogram.get_points_from_most_frequent_bin(seedCandidates);

        if (seedCandidates.size() < 5)
            break;

        //select seed cell with min MSE
        int seedId = -1;
        float minMSE = INT_MAX;
        for(unsigned int i = 0; i < seedCandidates.size(); i+=1) {
            int seedCandidate = seedCandidates[i];
            if(this->planeGrid[seedCandidate]->get_MSE() < minMSE) {
                seedId = seedCandidate;
                minMSE = this->planeGrid[seedCandidate]->get_MSE();
                //minMSE = this->planeGrid[i]->get_MSE();
            }
        }
        if (seedId < 0) {
            //error
            cerr << "Error : no min MSE in graph" << std::endl;
            exit(-1);
        }

        //copy plane segment in new object
        Plane_Segment newPlaneSegment = *this->planeGrid[seedId];

        //Seed cell growing
        int y = seedId / this->horizontalCellsCount;
        int x = seedId % this->horizontalCellsCount;

        std::fill_n(this->activationMap, this->totalCellCount, false);
        region_growing(cellDistanceTols, x, y, newPlaneSegment.get_normal(), newPlaneSegment.get_plane_d());

        //merge activated cells & remove them from histogram an list of remaining cells
        int cellActivatedCount = 0;
        for(int i = 0; i < this->totalCellCount; i+=1) {
            if(this->activationMap[i]) {
                newPlaneSegment.expand_segment(this->planeGrid[i]);
                cellActivatedCount += 1;
                this->histogram.remove_point(i);
                this->unassignedMask[i] = false;
                remainingPlanarCells -= 1;
            }
        }

        if(cellActivatedCount < 4)
            continue;

        //fit plane to merged data
        newPlaneSegment.fit_plane();

        if(not this->useCylinderDetection or newPlaneSegment.get_score() > 100) {
            //its certainly a plane or we ignore cylinder detection
            this->planeSegments.push_back(newPlaneSegment);
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
            Cylinder_Segment cy(this->planeGrid, this->activationMap, cellActivatedCount);
            this->cylinderSegments.push_back(cy);
            // Fit planes to subsegments
            for(int segId = 0; segId < cy.get_segment_count(); segId++){
                newPlaneSegment.clear_plane_parameters();
                for(int c = 0; c < cellActivatedCount; c++){
                    if (cy.get_inlier_at(segId, c)){
                        newPlaneSegment.expand_segment(this->planeGrid[cy.get_local_to_global_mapping(c)]);
                    }
                }
                newPlaneSegment.fit_plane();
                // Model selection based on MSE
                if(newPlaneSegment.get_MSE() < cy.get_MSE_at(segId)){
                    this->planeSegments.push_back(newPlaneSegment);
                    int currentPlaneCount = this->planeSegments.size();
                    for(int c = 0; c < cellActivatedCount; c++){
                        if (cy.get_inlier_at(segId, c)){
                            int cellId = cy.get_local_to_global_mapping(c);
                            this->gridPlaneSegmentMap.at<int>(cellId / this->horizontalCellsCount, cellId % this->horizontalCellsCount) = currentPlaneCount;
                        }
                    }
                    cy.set_cylindrical_mask(segId, false);
                }else{
                    cylinderCount += 1;
                    cylinder2regionMap.push_back(make_pair(cylinderSegments.size() - 1, segId));
                    for(int c = 0; c < cellActivatedCount; c++){
                        if (cy.get_inlier_at(segId, c)){
                            int cellId = cy.get_local_to_global_mapping(c);
                            this->gridCylinderSegMap.at<int>(cellId / this->horizontalCellsCount, cellId % this->horizontalCellsCount) = cylinderCount;
                        }
                    }
                    cy.set_cylindrical_mask(segId, true);
                }
            }

        }
    }//\while

    //merge sparse planes
    vector<unsigned int> planeMergeLabels;
    merge_planes(planeMergeLabels);

    //refine planes boundaries
    refine_plane_boundaries(depthMatrix, planeMergeLabels, planeSegmentsFinal);

    //refine cylinders boundaries
    //cylinder2regionMap;
    refine_cylinder_boundaries(depthMatrix, cylinder2regionMap, cylinderCount, cylinderSegmentsFinal); 

    //copy and rearranging
    // Copy inlier list to matrix form
    for (int cellR = 0; cellR < this->verticalCellsCount; cellR += 1){
        uchar* rowPtr = segOut.ptr<uchar>(cellR);
        uchar* gridPlaneErodedRowPtr = this->gridPlaneSegMapEroded.ptr<uchar>(cellR);
        uchar* gridCylinderErodedRowPtr = this->gridCylinderSegMapEroded.ptr<uchar>(cellR);
        int rOffset = cellR * this->cellHeight;
        int rLimit = rOffset + this->cellHeight;
        for (int cellC = 0; cellC < this->horizontalCellsCount; cellC += 1){
            int cOffset = cellC * this->cellWidth;

            if (gridPlaneErodedRowPtr[cellC] > 0){
                // Set rectangle equal to assigned cell
                segOut(cv::Rect(cOffset, rOffset, this->cellWidth, this->cellHeight)).setTo(gridPlaneErodedRowPtr[cellC]);
            }else{
                if(gridCylinderErodedRowPtr[cellC] > 0){
                    // Set rectangle equal to assigned cell
                    segOut(cv::Rect(cOffset, rOffset, this->cellWidth, this->cellHeight)).setTo(gridCylinderErodedRowPtr[cellC]);
                }else{
                    int cLimit = cOffset + this->cellWidth;
                    // Set cell pixels one by one
                    uchar* stackPtr = &this->segMapStacked[this->pointsPerCellCount * cellR * this->horizontalCellsCount + this->pointsPerCellCount * cellC];
                    for(int r = rOffset; r < rLimit; r++){
                        rowPtr = segOut.ptr<uchar>(r);
                        for(int c = cOffset; c < cLimit; c++){
                            if(*stackPtr > 0){
                                rowPtr[c] = *stackPtr;
                            }
                            stackPtr++;
                        }
                    }
                }
            }
        }
    }

    for(int i = 0; i < cylinderCount; i += 1) {
        int regId = cylinder2regionMap[i].first;
        if(regId > -1) {
            int subRegId = cylinder2regionMap[i].second;
            cylinderSegmentsFinal.push_back(Cylinder_Segment(this->cylinderSegments[regId], subRegId));
        }
    }
}

/*
 *  Recursively Grow a plane seed and merge it with it's neighbors
 */
void Plane_Detection::region_growing(vector<float>& cellDistTols, const unsigned short x, const unsigned short y, const Vector3d& seedPlaneNormal, const double seedPlaneD) {
    int index = x + horizontalCellsCount * y;
    if (index > horizontalCellsCount * verticalCellsCount or not this->unassignedMask[index] or this->activationMap[index]) {
        //pixel is not part of a component or already labelled
        return;
    }

    const Vector3d& secPlaneNormal = planeGrid[index]->get_normal();
    const Vector3d& secPlaneMean = planeGrid[index]->get_mean();
    const double& secPlaneD = planeGrid[index]->get_plane_d();

    if (
            //planeGrid[index].is_depth_discontinuous() 
            seedPlaneNormal.dot(secPlaneNormal) < this->minCosAngleForMerge
            or pow(seedPlaneNormal.dot(secPlaneMean) + seedPlaneD, 2) > cellDistTols[index]
       )//max_merge_dist
        return;
    activationMap[index] = true;

    // Now label the 4 neighbours:
    if (x > 0)
        region_growing(cellDistTols, x - 1, y, secPlaneNormal, secPlaneD);   // left  pixel
    if (x < this->width - 1)  
        region_growing(cellDistTols, x + 1, y, secPlaneNormal, secPlaneD);  // right pixel
    if (y > 0)        
        region_growing(cellDistTols, x, y - 1, secPlaneNormal, secPlaneD);   // upper pixel 
    if (y < this->height - 1) 
        region_growing(cellDistTols, x, y + 1, secPlaneNormal, secPlaneD);   // lower pixel
}

/*
 *  Merge close planes by comparing normals and MSE
 */
void Plane_Detection::merge_planes(vector<unsigned int>& planeMergeLabels) {
    const unsigned int planeCount = this->planeSegments.size();

    MatrixXd planesAssocMat = MatrixXd::Zero(planeCount, planeCount);
    get_connected_components(this->gridPlaneSegmentMap, planesAssocMat);

    for(unsigned int i = 0; i < planeCount; i += 1)
        planeMergeLabels.push_back(i);

    for(unsigned int r = 0; r < planesAssocMat.rows(); r += 1) {
        unsigned int planeId = planeMergeLabels[r];
        bool planeWasExpanded = false;
        const Plane_Segment& testPlane = this->planeSegments[planeId];
        const Vector3d& testPlaneNormal = testPlane.get_normal();

        for(unsigned int c = r+1; c < planesAssocMat.cols(); c += 1) {
            if(planesAssocMat(r, c)) {
                const Plane_Segment& mergePlane = this->planeSegments[c];
                const Vector3d& mergePlaneNormal = mergePlane.get_normal();
                double cosAngle = testPlaneNormal.dot(mergePlaneNormal);

                const Vector3d& mergePlaneMean = mergePlane.get_mean();
                //double distance = pow(
                //        testPlaneNormal.dot(mergePlaneMean) + testPlane.get_plane_d()
                //        , 2);
                double distance = pow(
                        this->planeSegments[r].get_normal()[0] * mergePlaneMean[0] 
                        + testPlaneNormal[1] * mergePlaneMean[1]  
                        + testPlaneNormal[2] * mergePlaneMean[2] 
                        + testPlane.get_plane_d(), 2);

                if(cosAngle > this->minCosAngleForMerge and distance < this->maxMergeDist) {
                    this->planeSegments[planeId].expand_segment(mergePlane);
                    planeMergeLabels[c] = planeId;
                    planeWasExpanded = true;
                }
                else {
                    planesAssocMat(r, c) = false;
                }
            }
        }
        if(planeWasExpanded)    //plane was merged with other planes
            planeSegments[planeId].fit_plane();
    }
}

/*
 *  Refine the final plane edges in mask images
 */
void Plane_Detection::refine_plane_boundaries(MatrixXf& depthCloudArray, vector<unsigned int>& planeMergeLabels, vector<Plane_Segment>& planeSegmentsFinal) {
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

        planeSegmentsFinal.push_back(this->planeSegments[i]);

        cv::dilate(this->mask, this->maskDilated, this->maskSquareKernel);
        this->maskDiff = this->maskDilated - this->maskEroded;

        int stackedCellId = 0;
        uchar planeNr = (unsigned char)planeSegmentsFinal.size();
        const Vector3d& planeNormal = this->planeSegments[i].get_normal();
        float nx = planeNormal[0];
        float ny = planeNormal[1];
        float nz = planeNormal[2];
        float d = this->planeSegments[i].get_plane_d();

        this->gridPlaneSegMapEroded.setTo(planeNr, this->maskEroded > 0);

        //cell refinement
        for (int cellR = 0; cellR < this->verticalCellsCount; cellR += 1) {
            unsigned char* rowPtr = this->maskDiff.ptr<uchar>(cellR);

            for(int cellC = 0; cellC < this->horizontalCellsCount; cellC += 1) {
                int offset = stackedCellId * this->pointsPerCellCount;
                int nextOffset = offset + this->pointsPerCellCount;

                if(rowPtr[cellC] > 0) {
                    //TODO: better distance metric
                    float maxDist = 9 * this->planeSegments[i].get_MSE();
                    //compute distance block
                    distancesCellStacked = 
                        depthCloudArray.block(offset, 0, this->pointsPerCellCount, 1).array() * nx +
                        depthCloudArray.block(offset, 1, this->pointsPerCellCount, 1).array() * ny +
                        depthCloudArray.block(offset, 2, this->pointsPerCellCount, 1).array() * nz +
                        d;

                    //Assign pixel
                    int j = 0;
                    for(int pt = offset; pt < nextOffset; j += 1, pt += 1) {
                        float dist = pow(distancesCellStacked(j), 2);
                        if(dist < maxDist and dist < this->distancesStacked[pt]) {
                            this->distancesStacked[pt] = dist;
                            this->segMapStacked[pt] = planeNr;
                        }
                    }
                }
                stackedCellId += 1;
            }
        }
    }
}


void Plane_Detection::refine_cylinder_boundaries(MatrixXf& depthCloudArray, std::vector<pair<int, int>>& cylinderToRegionMap, int cylinderCount, std::vector<Cylinder_Segment>& cylinderSegmentsFinal) {
    if(not this->useCylinderDetection)
        return; //no cylinder detections

    Eigen::Vector3d point;
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
        if (max == 0){
            continue;
        }

        cylinderFinalCount += 1;

        // Dilate to obtain borders
        cv::dilate(this->mask, this->maskDilated, this->maskSquareKernel);
        this->maskDiff = this->maskDilated - this->maskEroded;

        int stackedCellId = 0;

        this->gridCylinderSegMapEroded.setTo((unsigned char)CYLINDER_CODE_OFFSET + cylinderFinalCount, this->maskEroded > 0);

        int regId = cylinderToRegionMap[i].first;
        int subRegId = cylinderToRegionMap[i].second;
        const Cylinder_Segment& tempCopy = cylinderSegments[regId];


        // Get variables needed for point-surface distance computation
        const Eigen::Vector3d P2 = tempCopy.get_axis2_point(subRegId);
        const Eigen::Vector3d P1P2 = P2 - tempCopy.get_axis1_point(subRegId);
        double P1P2Normal = tempCopy.get_axis_normal(subRegId);
        double radius = tempCopy.get_radius(subRegId);

        // Cell refinement
        for (int cellR = 0; cellR < this->verticalCellsCount; cellR += 1){
            uchar* rowPtr = this->maskDiff.ptr<uchar>(cellR);
            for (int cellC = 0; cellC < this->horizontalCellsCount; cellC += 1){
                int offset = stackedCellId * this->pointsPerCellCount;
                int nextOffset = offset + this->pointsPerCellCount;
                if(rowPtr[cellC] > 0){
                    double maxDist = 9 * tempCopy.get_MSE_at(subRegId);
                    // Update cells
                    int j = 0;
                    for(int pt = offset; pt < nextOffset; j++,pt++) {
                        point(2) = depthCloudArray(pt, 2);
                        if(point(2) > 0){
                            point(0) = depthCloudArray(pt,0);
                            point(1) = depthCloudArray(pt,1);
                            double dist = P1P2.cross(point - P2).norm() / P1P2Normal - radius;
                            dist *= dist;
                            if(dist < maxDist and dist < this->distancesStacked[pt]){ 
                                this->distancesStacked[pt] = dist;
                                this->segMapStacked[pt] = CYLINDER_CODE_OFFSET + cylinderFinalCount;
                            }
                        }
                    }
                }
                stackedCellId += 1;
            }
        }
    }

}




/*
 *  Fill an association matrix that links connected plane components
 */
void Plane_Detection::get_connected_components(cv::Mat& segmentMap, Eigen::MatrixXd& planesAssociationMatrix) {
    int rows2scanCount = segmentMap.rows - 1;
    int cols2scanCount = segmentMap.cols - 1;

    for(int r = 0; r < rows2scanCount; r += 1) {
        int *row = segmentMap.ptr<int>(r);
        int *rowBelow = segmentMap.ptr<int>(r + 1);
        for(int c = 0; c < cols2scanCount; c += 1) {
            int pixelValue = row[c];
            if(pixelValue > 0) {
                if(row[c + 1] > 0 and pixelValue != row[c + 1]) {
                    planesAssociationMatrix(pixelValue - 1, row[c + 1] - 1) = true;
                    //  planesAssociationMatrix(row[c + 1] - 1, pixelValue - 1) = true;
                }
                if(rowBelow[c] > 0 and pixelValue != rowBelow[c]) {
                    planesAssociationMatrix(pixelValue - 1, rowBelow[c] - 1) = true;
                    //  planesAssociationMatrix(rowBelow[c] - 1, pixelValue - 1) = true;
                }
            }
        }
    }
    for(int r=0; r < planesAssociationMatrix.rows(); r++){
        for(int c = r+1; c < planesAssociationMatrix.cols(); c++){
            planesAssociationMatrix(r,c) = planesAssociationMatrix(r,c) or planesAssociationMatrix(c,r);
        }
    }
}





/*
 *  Initialize and fill the histogram
 *
 * returns the number of initial planar surfaces
 */
int Plane_Detection::init_histogram() {
    int remainingPlanarCells = 0;

    MatrixXd histBins(this->totalCellCount, 2);
    for(int cellId = 0; cellId < this->totalCellCount; cellId += 1) {  
        this->unassignedMask[cellId] = false; 
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
 *  Init planeGrid and cellDistanceTols
 *
 */
void Plane_Detection::init_planar_cell_fitting(MatrixXf& depthCloudArray, vector<float>& cellDistanceTols) {
    float sinCosAngleForMerge = sqrt(1 - pow(this->minCosAngleForMerge, 2));

    //fro each planeGrid cell
    for(int stackedCellId = 0; stackedCellId < this->totalCellCount; stackedCellId += 1) {
        //init the cell
        this->planeGrid[stackedCellId]->init_plane_segment(depthCloudArray, stackedCellId);

        if (this->planeGrid[stackedCellId]->is_planar()) {
            int cellDiameter = (
                    depthCloudArray.block(stackedCellId * this->pointsPerCellCount + this->pointsPerCellCount - 1, 0, 1, 3) - 
                    depthCloudArray.block(stackedCellId * this->pointsPerCellCount, 0, 1, 3)
                    ).norm(); 
            cellDistanceTols[stackedCellId] = pow(min(max(cellDiameter * sinCosAngleForMerge, 20.0f), this->maxMergeDist), 2);
        }
    }
}


Plane_Detection::~Plane_Detection() {
    delete []this->unassignedMask;
    delete []this->activationMap;
    delete []this->segMapStacked;
    delete []this->distancesStacked;

    this->planeGrid.clear();
    this->planeSegments.clear();

    this->gridPlaneSegmentMap.release();
    this->gridPlaneSegMapEroded.release();
    this->gridCylinderSegMap.release();
    this->gridCylinderSegMapEroded.release();

    this->mask.release();
    this->maskEroded.release();
    this->maskDilated.release();
    this->maskDiff.release();

    this->maskSquareKernel.release();
    this->maskCrossKernel.release();
}





