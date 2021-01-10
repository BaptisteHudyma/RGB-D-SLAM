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
#include "Histogram.hpp"

namespace planeDetection {

    class Plane_Detection {
        //check for planes in an organized depth points matrix 
        public:
            Plane_Detection(unsigned int width, unsigned int height, unsigned int blocSize = 20, float minCosAngeForMerge = 0.0, float maxMergeDist = 0.0);

            void find_plane_regions(Eigen::MatrixXf& depthMatrix, cv::Mat& segOut);  //detect planes in depth image

            ~Plane_Detection();

        protected:
            void reset_data();

            void init_planar_cell_fitting(Eigen::MatrixXf& depthCloudArray, std::vector<float>& cellDistTols);
            void init_histogram(std::vector<std::unique_ptr<Plane_Segment>>& grid, int& remainingPlanarCells);
            void region_growing(std::vector<float>& cellDistTols, const unsigned short x, const unsigned short y, const Eigen::Vector3d seedPlaneNormal, const double seedPlaneD);

            void merge_planes(std::vector<unsigned int>& planeMergeLabels);
            void refine_plane_boundaries(Eigen::MatrixXf& depthCloudArray, std::vector<unsigned int>& planeMergeLabels, std::vector<Plane_Segment>& planeSegmentsFinal);
            void get_connected_components(cv::Mat& segmentMap, Eigen::MatrixXd& planesAssociationMatrix);


        private:
            std::vector<std::unique_ptr<Plane_Segment>> planeGrid;
            std::vector<Plane_Segment> planeSegments;

            cv::Mat_<int> gridPlaneSegMap;
            cv::Mat_<uchar> gridPlaneSegMapEroded;
            cv::Mat_<int> gridCylinderSegMap;
            cv::Mat_<uchar> gridCylinderSegMapEroded;

            Eigen::ArrayXf distancesCellStacked;

            Histogram histogram;

            const int width;
            const int height;
            const int blocSize;
            const int pointsPerCellCount;
            const float minCosAngleForMerge;
            const float maxMergeDist;

            int cellWidth;
            int cellHeight;

            int totalCellCount;
            int horizontalCellsCount;
            int verticalCellsCount;

            //arrays
            bool* activationMap;
            bool* unassignedMask;
            float* distancesStacked;
            unsigned char * segMapStacked;

            //mat
            cv::Mat_<int> gridPlaneSegmentMap;
            cv::Mat mask;
            cv::Mat maskEroded;
            cv::Mat maskSquareEroded;
            cv::Mat maskDilated;
            cv::Mat maskDiff;

            //kernels
            cv::Mat maskSquareKernel;
            cv::Mat maskCrossKernel;
    };

}

#endif
