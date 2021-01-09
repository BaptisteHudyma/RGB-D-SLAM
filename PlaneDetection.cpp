#include <iostream>
#include "PlaneDetection.hpp"


//lib simplification
using namespace planeDetection;
using namespace Eigen;
using namespace std;

//default constructor
Plane_Detection::Plane_Detection(unsigned int width, unsigned int height, unsigned int blocSize, float minCosAngleForMerge, float maxMergeDistance)
    :  histogram(20), width(width), height(height),  blocSize(blocSize), pointsPerCellCount(blocSize * blocSize), minCosAngleForMerge(minCosAngleForMerge), maxMergeDist(maxMergeDistance)
{
    //Init variables
    this->cellWidth = blocSize;
    this->cellHeight = blocSize;

    this->horizontalCellsCount = this->width / cellWidth;
    this->verticalCellsCount = this->height / cellHeight;

    this->totalCellCount = this->verticalCellsCount * this->horizontalCellsCount;
    this->activationMap = new bool[this->totalCellCount];
    this->unassignedMask = new bool[this->totalCellCount];
    this->distancesStacked = new float[width * height];
    this->segMapStacked = new unsigned char[width * height];

    distancesCellStacked = ArrayXf::Zero(this->pointsPerCellCount, 1);

    //init kernels
    cv::Mat maskSquareKernel = cv::Mat::ones(3,3,CV_8U);

    cv::Mat maskCrossKernel = cv::Mat::ones(3,3,CV_8U);
    maskCrossKernel.at<uchar>(0,0) = 0;
    maskCrossKernel.at<uchar>(2,2) = 0;
    maskCrossKernel.at<uchar>(0,2) = 0;
    maskCrossKernel.at<uchar>(2,0) = 0;


    for(int i = 0; i < 20; i += 1) {
        //fill with null nodes
        this->planeGrid.push_back(nullptr);
    }

}

void Plane_Detection::find_plane_regions(Eigen::MatrixXf& depthMatrix) {
    //find planes in given depth matrix

    //reset stacked distances
    std::memset(this->distancesStacked, 100, this->height * this->width * sizeof(float)); /* = to really high float*/
    std::memset(this->segMapStacked, (uchar)0, this->height * this->width * sizeof(unsigned char));

    //init planar grid
    vector<float> cellDistanceTols = init_planar_cell_fitting(depthMatrix);

    //fill histogram
    init_histogram(this->planeGrid);

    int remainingPlanarCells = 10;
    while(remainingPlanarCells > 0) {
        //get seed candidates
        vector<int> seedCandidates = this->histogram.get_points_from_most_frequent_bin();

        if (seedCandidates.size() < 5)
            break;

        //select seed cell with min MSE
        int seedId = -1;
        float minMSE = INT_MAX;
        for(unsigned int i = 0; i < seedCandidates.size(); i+=1) {
            int seedCandidate = seedCandidates[i];
            if(this->planeGrid[seedCandidate]->get_MSE() < minMSE) {
                seedId = seedCandidate;
                minMSE = this->planeGrid[i]->get_MSE();
            }
        }
        if (seedId < 0) {
            //error
            cout << "Error : no min MSE in grap" << std::endl;
            exit(-1);
        }

        //copy plane segment in new object
        Plane_Segment newPlaneSegment(*planeGrid[seedId]);

        //Seed cell growing
        int y = seedId / this->horizontalCellsCount;
        int x = seedId % this->horizontalCellsCount;
        Vector3d seedPlaneNormal = newPlaneSegment.get_normal();
        double seedPlaneD = newPlaneSegment.get_plane_d();

        memset(activationMap, false, sizeof(bool) * totalCellCount);
        region_growing(cellDistanceTols, x, y, seedPlaneNormal, seedPlaneD);

        //merge activated cells & remove them from histogram an list of remaining cells
        int cellActivatedCount = 0;
        for(int i = 0; i < totalCellCount; i+=1) {
            if(activationMap[i]) {
                newPlaneSegment.expand_segment(planeGrid[i]);
                cellActivatedCount += 1;
                histogram.remove_point(i);
                unassignedMask[i] = false;
                remainingPlanarCells -= 1;
            }
        }

        if(cellActivatedCount < 4)
            continue;

        //fit plane to merged data
        newPlaneSegment.fit_plane();

        if(newPlaneSegment.get_score() < 100) {
            //its certainly a plane
            this->planeSegments.push_back(newPlaneSegment);
            int currentPlaneCount = this->planeSegments.size();
            //mark cells
            int i = 0;
            int* row;
            for(int r = 0; r < this->verticalCellsCount; r += 1) {
                row = gridPlaneSegmentMap.ptr<int>(r);
                for(int c = 0; c < this->horizontalCellsCount; c += 1) {
                    if(activationMap[i])
                        row[c] = currentPlaneCount;
                    i += 1;
                }
            }
        }
        else {
            //cylinder fitting
        }

    }//\while

    //merge sparse planes
    vector<unsigned int> planeMergeLabels = merge_planes();
    //refine planes boundaries
    refine_plane_boundaries(depthMatrix, planeMergeLabels);

}


void Plane_Detection::region_growing(vector<float>& cellDistTols, const unsigned short x, const unsigned short y, const Vector3d seedPlaneNormal, const double seedPlaneD) {
    //grows a plane region in depth image graph


    int index = x + horizontalCellsCount * y;
    if (not unassignedMask[index] or activationMap[index]) {
        //pixel is not part of a component or already labelled
        return;
    }

    const Vector3d& secPlaneNormal = planeGrid[index]->get_normal();
    const Vector3d& secPlaneMean = planeGrid[index]->get_mean();
    double secPlaneD = planeGrid[index]->get_plane_d();

    if (seedPlaneNormal[0] * secPlaneNormal[0] + seedPlaneNormal[1] * secPlaneNormal[1] + seedPlaneNormal[2] * secPlaneNormal[2] < this->minCosAngleForMerge or 
            pow(seedPlaneNormal[0] * secPlaneMean[0] + seedPlaneNormal[1] * secPlaneMean[1] + seedPlaneNormal[2] * secPlaneMean[2] + seedPlaneD, 2) > cellDistTols[index])//max_merge_dist
        return;
    activationMap[index] = true;

    // Now label the 4 neighbours:
    if (x > 0)        
        region_growing(cellDistTols, x - 1, y, secPlaneNormal, secPlaneD);   // left  pixel
    if (x < width - 1)  
        region_growing(cellDistTols, x + 1, y, secPlaneNormal, secPlaneD);  // right pixel
    if (y > 0)        
        region_growing(cellDistTols, x, y - 1, secPlaneNormal, secPlaneD);   // upper pixel 
    if (y < height - 1) 
        region_growing(cellDistTols, x, y + 1, secPlaneNormal, secPlaneD);   // lower pixel
}


void Plane_Detection::init_histogram(vector<unique_ptr<Plane_Segment>>& grid) {
    MatrixXd histBins(this->totalCellCount, 2);
    vector<bool> planarFlags(this->totalCellCount, false);

    int remainingPlanarCells = 0;
    double nx, ny, nz;
    for(int cellId = 0; cellId < this->totalCellCount; cellId += 1) {  
        if(grid[cellId]->is_planar()) {
            const Vector3d& planeNormal = grid[cellId]->get_normal();
            nx = planeNormal[0];
            ny = planeNormal[1];
            nz = planeNormal[2];

            double projNormal = sqrt(nx * nx + ny * ny);
            histBins(cellId, 0) = acos(-nz);
            histBins(cellId, 1) = atan2(nx / projNormal, ny / projNormal);
            planarFlags[cellId] = grid[cellId]->get_score();
            remainingPlanarCells += 1;
        }
    }
    histogram.init_histogram(histBins, planarFlags);
}

vector<float> Plane_Detection::init_planar_cell_fitting(MatrixXf& depthCloudArray) {
    //init planeGrid and cellDistanceTols
    //return cellDistanceTols

    vector<float> cellDistanceTols(this->totalCellCount, 0);
    int stackedCellId = 0;
    float cellDiameter;
    float sinCosAngleForMerge = sqrt(1 - pow(this->minCosAngleForMerge, 2));

    for(int cellR = 0; cellR < this->verticalCellsCount; cellR += 1) {
        for(int cellC = 0; cellC < this->horizontalCellsCount; cellC += 1) {
            planeGrid[stackedCellId] = unique_ptr<Plane_Segment>(new Plane_Segment(depthCloudArray, stackedCellId, this->pointsPerCellCount, this->cellWidth));

            if (planeGrid[stackedCellId]->is_planar()) {
                cellDiameter = (
                        depthCloudArray.block(stackedCellId * this->pointsPerCellCount + this->pointsPerCellCount - 1, 0, 1, 3) - 
                        depthCloudArray.block(stackedCellId * this->pointsPerCellCount + this->pointsPerCellCount, 0, 1, 3)
                        ).norm(); 
                cellDistanceTols[stackedCellId] = pow(min(max(cellDiameter * sinCosAngleForMerge, 20.0f), this->maxMergeDist), 2);
            }
            stackedCellId += 1;
        }
    }
    return cellDistanceTols;
}


vector<unsigned int> Plane_Detection::merge_planes() {
    //merges planes from planeGrid
    unsigned int planeCount = this->planeSegments.size();
    MatrixXd planesAssocMat = MatrixXd::Zero(planeCount, planeCount);

    get_connected_components(gridPlaneSegmentMap, planesAssocMat);

    vector<unsigned int> planeMergeLabels;
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
                double cosAngle = 
                    testPlaneNormal[0] * mergePlaneNormal[0] +
                    testPlaneNormal[1] * mergePlaneNormal[1] +
                    testPlaneNormal[2] * mergePlaneNormal[2];

                const Vector3d& mergePlaneMean = mergePlane.get_mean();
                double distance = pow(
                        this->planeSegments[r].get_normal()[0] * mergePlaneMean[0] + 
                        testPlaneNormal[1] * mergePlaneMean[1] +
                        testPlaneNormal[2] * mergePlaneMean[2] +
                        testPlane.get_plane_d()
                        , 2);

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
    return planeMergeLabels;
}

vector<Plane_Segment> Plane_Detection::refine_plane_boundaries(MatrixXf& depthCloudArray, vector<unsigned int>& planeMergeLabels) {
    //refine the coarse planes boundaries to smoother versions

    vector<Plane_Segment> planeSegmentsFinal;

    unsigned int planeCount = planeMergeLabels.size();
    for(unsigned int i = 0; i < planeCount; i += 1) {
        if (i != planeMergeLabels[i])
            continue;

        mask = cv::Scalar(0);
        for(unsigned int j = i; j < planeCount; j += 1) {
            if(planeMergeLabels[j] == planeMergeLabels[i]) {
                mask.setTo(1, gridPlaneSegmentMap == j + 1);
            }
        }

        cv::erode(mask, maskEroded, maskCrossKernel);
        double min, max;
        cv::minMaxLoc(maskEroded, &min, &max);

        if(max == 0)    //completely eroded
            continue;

        planeSegmentsFinal.push_back(this->planeSegments[i]);

        cv::dilate(mask, maskDilated, maskSquareKernel);
        cv::Mat maskDiff = maskDilated - maskEroded;

        int stackedCellId = 0;
        //uchar planeCount = (unsigned char)planeSegmentsFinal.size();
        unsigned int planeCount = planeSegmentsFinal.size();
        const Vector3d& planeNormal = planeSegments[i].get_normal();
        float nx = planeNormal[0];
        float ny = planeNormal[1];
        float nz = planeNormal[2];
        float d = planeSegments[i].get_plane_d();

        gridPlaneSegMapEroded.setTo(planeCount, maskEroded > 0);

        //cell refinement
        for (int cellR = 0; cellR < this->verticalCellsCount; cellR += 1) {
            unsigned char* rowPtr = maskDiff.ptr<uchar>(cellR);

            for(int cellC = 0; cellC < this->horizontalCellsCount; cellC += 1) {
                int offset = stackedCellId * this->pointsPerCellCount;
                int nextOffset = offset + this->pointsPerCellCount;

                if(rowPtr[cellC] > 0) {
                    float maxDist = 9 * this->planeSegments[i].get_MSE();
                    //compute distance block
                    distancesCellStacked = 
                        depthCloudArray.block(offset, 0, this->pointsPerCellCount, 1).array() * nx +
                        depthCloudArray.block(offset, 1, this->pointsPerCellCount, 1).array() * ny +
                        depthCloudArray.block(offset, 2, this->pointsPerCellCount, 1).array() * nz +
                        d;

                    //Assign pixel
                    for(int pt = offset; pt < nextOffset; pt += 1) {
                        float dist = pow(distancesCellStacked(0), 2);
                        if(dist < maxDist and dist < distancesStacked[pt]) {
                            distancesStacked[pt] = dist;
                            this->segMapStacked[pt] = planeCount;
                        }
                    }
                }
                stackedCellId += 1;
            }
        }
    }
    return planeSegmentsFinal;
}


void Plane_Detection::get_connected_components(cv::Mat& segmentMap, Eigen::MatrixXd& planesAssociationMatrix) {
    int rows2scanCount = segmentMap.rows - 1;
    int cols2scanCount = segmentMap.cols - 1;

    for(int r = 0; r < rows2scanCount; r += 1) {
        int *row = segmentMap.ptr<int>(r);
        int *rowBelow = segmentMap.ptr<int>(r + 1);
        for(int c = 0; c < cols2scanCount; c += 1) {
            int pixelValue = row[c];
            if(pixelValue > 0) {
                if(row[c + 1] > 0 and pixelValue != row[c + 1])
                    planesAssociationMatrix(pixelValue - 1, row[c + 1] - 1) = true;
                if(rowBelow[c] > 0 and pixelValue != rowBelow[c])
                    planesAssociationMatrix(pixelValue - 1, rowBelow[c] - 1) = true;
            }
        }
    }
    for(int r = 0; r < planesAssociationMatrix.rows(); r += 1) {
        for(int c = r + 1; c < planesAssociationMatrix.cols(); c += 1) {
            planesAssociationMatrix(r, c) = planesAssociationMatrix(r, c) or planesAssociationMatrix(c, r);
        }
    }
}





Plane_Detection::~Plane_Detection() {
    delete []this->unassignedMask;
    delete []this->activationMap;
    delete []this->segMapStacked;
    delete []this->distancesStacked;
}





