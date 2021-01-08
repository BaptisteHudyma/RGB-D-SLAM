#include <iostream>
#include "PlaneDetection.hpp"


//lib simplification
using namespace planeDetection;
using namespace Eigen;
using namespace std;

//default constructor
Plane_Detection::Plane_Detection(unsigned int width, unsigned int height, unsigned int blocSize = 20)
    :  histogram(20), width(width), height(height),  blocSize(blocSize) 
{
    //Init variables
    this->cellWidth = blocSize;
    this->cellHeight = blocSize;

    this->horizontalCellsCount = this->width / cellWidth;
    this->verticalCellsCount = this->height / cellHeight;

    this->totalCellCount = this->verticalCellsCount * this->horizontalCellsCount;
    this->activationMap = new bool[this->totalCellCount];
    this->unassignedMask = new bool[this->totalCellCount];



    for(int i = 0; i < 20; i+=1) {
        //fill with null nodes
        this->planeGrid.push_back(nullptr);
    }

    //initialize histogram



}

void Plane_Detection::grow_plane_regions() {

    vector<float> cellDistanceTols(this->totalCellCount, 0);

    int remainingPlanarCells = 10;
    while(remainingPlanarCells > 0) {
        //get seed candidates
        vector<int>& seedCandidates = this->histogram.get_points_from_most_frequent_bin();

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
                newPlaneSegment.expandSegment(planeGrid[i]);
                cellActivatedCount += 1;
                histogram.remove_point(i);
                unassignedMask[i] = false;
                remainingPlanarCells -= 1;
            }
        }

        if(cellActivatedCount < 4)
            continue;

        //fit plane to merged data
        newPlaneSegment.fitPlane();

        if(newPlaneSegment.get_score() < 100) {
            //its certainly a plane
            planeSegments.push_back(newPlaneSegment);
            int currentPlaneCount = planeSegments.size();
            //mark cells
            int i = 0;
            int* rows;
            for(int r = 0; r < this->verticalCellsCount; r += 1) {
                row = gridPlaneSegmentMap.ptr<int>(r);
                for(int c = 0; c < this->horizontalCellsCount; c += 1) {
                    if(activationMap[i])
                        row[c] = currentPlaneCount;
                    i += 1;
                }
            }
            else {
                //cylinder fitting
            }
        }//\while
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
        double secPlaneD = Grid[index]->d;

        if (seedPlaneNormal[0] * secPlaneNormal[0] + seedPlaneNormal[1] * secPlaneNormal[1] + seedPlaneNormal[2] * secPlaneNormal[2] < min_cos_angle_4_merge or 
                pow(seedPlaneNormal[0] * secPlaneMean[0] + seedPlaneNormal[1] * secPlaneMean[1] + seedPlaneNormal[2] * secPlaneMean[2] + seedPlaneD, 2) > cellDistTols[index])//max_merge_dist
            return;
        activationMap[index] = true;

        // Now label the 4 neighbours:
        if (x > 0)        
            region_growing(cellDistTols, x - 1, y, secPlaneNormal, secPlaneD);   // left  pixel
        if (x < width-1)  
            region_growing(cellDistTols, x + 1, y, secPlaneNormal, secPlaneD);  // right pixel
        if (y > 0)        
            region_growing(cellDistTols, x, y - 1, secPlaneNormal, secPlaneD);   // upper pixel 
        if (y < height-1) 
            region_growing(cellDistTols, x, y + 1, secPlaneNormal, secPlaneD);   // lower pixel
    }



    Plane_Detection::~Plane_Detection() {
        delete []this->unassignedMask;
        delete []this->activationMap;
    }




