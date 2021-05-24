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


namespace primitiveDetection {

    class Primitive_Detection {
        //check for planes in an organized depth points matrix 
        protected:
            //typdefs
            typedef std::vector<std::pair<int, int>> intpair_vector;
            typedef std::unique_ptr<Plane_Segment> plane_segment_unique_ptr;
            typedef std::unique_ptr<Cylinder_Segment> cylinder_segment_unique_ptr;
            typedef std::vector<plane_segment_unique_ptr> planes_ptr_vector; 
            typedef std::vector<cylinder_segment_unique_ptr> cylinders_ptr_vector; 
            typedef std::vector<Plane_Segment> planes_vector; 
            typedef std::vector<Cylinder_Segment> cylinders_vector; 
            typedef std::vector<unsigned int> uint_vector;

        public:
            Primitive_Detection(const unsigned int width, const unsigned int height, const unsigned int blocSize = 20, const float minCosAngeForMerge = 0.9659, const float maxMergeDist = 50, const bool useCylinderDetection = false);

            void find_primitives(const Eigen::MatrixXf& depthMatrix, planes_vector& planeSegmentsFinal, cylinders_vector& cylinderSegmentsFinal, cv::Mat& segOut);  //detect 3D primitives in depth image
            
            void apply_masks(const cv::Mat& inputImage, const std::vector<cv::Vec3b>& colors, const cv::Mat& maskImage, const planes_vector& planeParams, const cylinders_vector& cylinderParams, cv::Mat& labeledImage, const double elapsedTime=0);
            ~Primitive_Detection();

            //perf measurments
            double resetTime;
            double initTime;
            double growTime;
            double mergeTime;
            double refineTime;
            double setMaskTime;

        protected:
            void reset_data();

            void init_planar_cell_fitting(const Eigen::MatrixXf& depthCloudArray);
            int init_histogram();

            void grow_planes_and_cylinders(intpair_vector& cylinderToRegionMap, int remainingPlanarCells);
            void merge_planes(uint_vector& planeMergeLabels);
            void refine_plane_boundaries(const Eigen::MatrixXf& depthCloudArray, uint_vector& planeMergeLabels, planes_vector& planeSegmentsFinal);
            void refine_cylinder_boundaries(const Eigen::MatrixXf& depthCloudArray, intpair_vector& cylinderToRegionMap, cylinders_vector& cylinderSegmentsFinal); 
            void set_masked_display(cv::Mat& segOut); 

            void region_growing(const unsigned short x, const unsigned short y, const Eigen::Vector3d& seedPlaneNormal, const double seedPlaneD);
            void get_connected_components(const cv::Mat& segmentMap, Eigen::MatrixXd& planesAssociationMatrix);

        private:
            Histogram _histogram;

            const int _width;
            const int _height;
            const int _blocSize;
            const int _pointsPerCellCount;
            const float _minCosAngleForMerge;
            const float _maxMergeDist;

            const bool _useCylinderDetection;

            const int _cellWidth;
            const int _cellHeight;

            const int _horizontalCellsCount;
            const int _verticalCellsCount;
            const int _totalCellCount;

            plane_segment_unique_ptr *_planeGrid;
            planes_ptr_vector _planeSegments;
            cylinders_ptr_vector _cylinderSegments;

            cv::Mat_<int> _gridPlaneSegmentMap;
            cv::Mat_<uchar> _gridPlaneSegMapEroded;
            cv::Mat_<int> _gridCylinderSegMap;
            cv::Mat_<uchar> _gridCylinderSegMapEroded;

            Eigen::ArrayXf _distancesCellStacked;

            //arrays
            bool* _activationMap;
            bool* _unassignedMask;
            float* _distancesStacked;
            unsigned char* _segMapStacked;
            float* _cellDistanceTols;

            //mat
            cv::Mat _mask;
            cv::Mat _maskEroded;
            cv::Mat _maskDilated;
            cv::Mat _maskDiff;

            //kernels
            cv::Mat _maskSquareKernel;
            cv::Mat _maskCrossKernel;

        private:
            //prevent backend copy
            Primitive_Detection(const Primitive_Detection&);
            Primitive_Detection& operator=(const Primitive_Detection&);
    };

}

#endif
