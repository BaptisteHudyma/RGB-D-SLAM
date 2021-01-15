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


namespace planeDetection {

    class Plane_Detection {
        //check for planes in an organized depth points matrix 
        public:
            Plane_Detection(unsigned int width, unsigned int height, unsigned int blocSize = 20, float minCosAngeForMerge = 0.9659, float maxMergeDist = 50, bool useCylinderDetection = false);

            void find_plane_regions(Eigen::MatrixXf& depthMatrix, std::vector<Plane_Segment>& planeSegmentsFinal, std::vector<Cylinder_Segment>& cylinderSegmentsFinal, cv::Mat& segOut);  //detect planes in depth image
            
            void apply_masks(cv::Mat& inputImage, std::vector<cv::Vec3b>& colors, cv::Mat& maskImage, std::vector<Plane_Segment>& planeParams, std::vector<Cylinder_Segment>& cylinderParams, cv::Mat& labeledImage, double elapsedTime=0);
            ~Plane_Detection();

            //perf measurments
            double resetTime;
            double initTime;
            double growTime;
            double mergeTime;
            double refineTime;
            double setMaskTime;

        protected:
            void reset_data();

            void init_planar_cell_fitting(Eigen::MatrixXf& depthCloudArray);
            int init_histogram();

            void grow_planes_and_cylinders(std::vector<std::pair<int, int>>& cylinderToRegionMap, int remainingPlanarCells);
            void merge_planes(std::vector<unsigned int>& planeMergeLabels);
            void refine_plane_boundaries(Eigen::MatrixXf& depthCloudArray, std::vector<unsigned int>& planeMergeLabels, std::vector<Plane_Segment>& planeSegmentsFinal);
            void refine_cylinder_boundaries(Eigen::MatrixXf& depthCloudArray, std::vector<std::pair<int, int>>& cylinderToRegionMap, std::vector<Cylinder_Segment>& cylinderSegmentsFinal); 
            void set_masked_display(cv::Mat& segOut); 

            void region_growing(const unsigned short x, const unsigned short y, const Eigen::Vector3d& seedPlaneNormal, const double seedPlaneD);
            void get_connected_components(cv::Mat& segmentMap, Eigen::MatrixXd& planesAssociationMatrix);

        private:
            Histogram histogram;

            const int width;
            const int height;
            const int blocSize;
            const int pointsPerCellCount;
            const float minCosAngleForMerge;
            const float maxMergeDist;

            const bool useCylinderDetection;

            const int cellWidth;
            const int cellHeight;

            const int horizontalCellsCount;
            const int verticalCellsCount;
            const int totalCellCount;

            std::unique_ptr<Plane_Segment> *planeGrid;
            std::vector<std::unique_ptr<Plane_Segment>> planeSegments;
            std::vector<std::unique_ptr<Cylinder_Segment>> cylinderSegments;

            cv::Mat_<int> gridPlaneSegmentMap;
            cv::Mat_<uchar> gridPlaneSegMapEroded;
            cv::Mat_<int> gridCylinderSegMap;
            cv::Mat_<uchar> gridCylinderSegMapEroded;

            Eigen::ArrayXf distancesCellStacked;

            //arrays
            bool* activationMap;
            bool* unassignedMask;
            float* distancesStacked;
            unsigned char* segMapStacked;
            float* cellDistanceTols;

            //mat
            cv::Mat mask;
            cv::Mat maskEroded;
            cv::Mat maskDilated;
            cv::Mat maskDiff;

            //kernels
            cv::Mat maskSquareKernel;
            cv::Mat maskCrossKernel;

        private:
            //prevent backend copy
            Plane_Detection(const Plane_Detection&);
            Plane_Detection& operator=(const Plane_Detection&);
    };

}

#endif
