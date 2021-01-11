#ifndef PLANE_DETECTION_H
#define PLANE_DETECTION_H

#include <math.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "PlaneSegment.hpp"
#include "CylinderSegment.hpp"
#include "Histogram.hpp"

const unsigned int CYLINDER_CODE_OFFSET = 50;

namespace planeDetection {

    class Plane_Detection {
        //check for planes in an organized depth points matrix 
        public:
            Plane_Detection(unsigned int width, unsigned int height, unsigned int blocSize = 20, float minCosAngeForMerge = 0.9659, float maxMergeDist = 50, bool useCylinderDetection = false);

            void find_plane_regions(Eigen::MatrixXf& depthMatrix, std::vector<Plane_Segment>& planeSegmentsFinal, std::vector<Cylinder_Segment>& cylinderSegmentsFinal, cv::Mat& segOut);  //detect planes in depth image

            ~Plane_Detection();

        protected:
            void reset_data();

            void init_planar_cell_fitting(Eigen::MatrixXf& depthCloudArray, std::vector<float>& cellDistTols);
            int init_histogram();
            void region_growing(std::vector<float>& cellDistTols, const unsigned short x, const unsigned short y, const Eigen::Vector3d& seedPlaneNormal, const double seedPlaneD);

            void merge_planes(std::vector<unsigned int>& planeMergeLabels);

            void refine_plane_boundaries(Eigen::MatrixXf& depthCloudArray, std::vector<unsigned int>& planeMergeLabels, std::vector<Plane_Segment>& planeSegmentsFinal);
            void refine_cylinder_boundaries(std::vector<std::pair<int, int>>& cylinderToRegionMap, int cylinderCount);
            void refine_cylinder_boundaries(Eigen::MatrixXf& depthCloudArray, std::vector<std::pair<int, int>>& cylinderToRegionMap, int cylinderCount, std::vector<Cylinder_Segment>& cylinderSegmentsFinal); 

            void get_connected_components(cv::Mat& segmentMap, Eigen::MatrixXd& planesAssociationMatrix);


        private:
            Histogram histogram;

            std::vector<std::unique_ptr<Plane_Segment>> planeGrid;
            std::vector<Plane_Segment> planeSegments;
            std::vector<Cylinder_Segment> cylinderSegments;

            cv::Mat_<int> gridPlaneSegmentMap;
            cv::Mat_<uchar> gridPlaneSegMapEroded;
            cv::Mat_<int> gridCylinderSegMap;
            cv::Mat_<uchar> gridCylinderSegMapEroded;

            Eigen::ArrayXf distancesCellStacked;

            const int width;
            const int height;
            const int blocSize;
            const int pointsPerCellCount;
            const float minCosAngleForMerge;
            const float maxMergeDist;

            bool useCylinderDetection;

            int cellWidth;
            int cellHeight;

            int totalCellCount;
            int horizontalCellsCount;
            int verticalCellsCount;

            //arrays
            bool* activationMap;
            bool* unassignedMask;
            float* distancesStacked;
            unsigned char* segMapStacked;

            //mat
            cv::Mat mask;
            cv::Mat maskEroded;
            cv::Mat maskDilated;
            cv::Mat maskDiff;

            //kernels
            cv::Mat maskSquareKernel;
            cv::Mat maskCrossKernel;
    };

}

#endif
