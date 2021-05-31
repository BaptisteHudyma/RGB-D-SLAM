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
#include "ShapePrimitives.hpp"


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
            typedef std::list<std::unique_ptr<IPrimitive>> primitive_container; 
            typedef std::vector<unsigned int> uint_vector;

        public:
            Primitive_Detection(unsigned int width, unsigned int height, unsigned int blocSize = 20, float minCosAngeForMerge = 0.9659, float maxMergeDist = 50, bool useCylinderDetection = false);

            void find_primitives(const Eigen::MatrixXf& depthMatrix, primitive_container& primitiveSegments, cv::Mat& segOut);  //detect 3D primitives in depth image



            /**
             * \brief Apply each plane and cylinder mask on the maskImage, with colors corresponding to the ids of colors vector
             *
             * \param[in] inputImage input RGB image on which to put the planes and cylinder masks
             * \param[in] colors A vector of colors, that must remain consistant at each call
             * \param[in] maskImage An image where each pixel is the index of a plane or cylinder
             * \param[in] primitiveSegments A container of the planes and cylinders detected in inputImage
             * \param[in/out] labeledImage The final result: an image with colored masks applied for each plane and cylinder, as well as a bar displaying informations on the top. Must be passed as an empty image of the same dimensions as labeledImage.
             * \param[in] timeElapsed the time elapsed since last frame. Used to display fps
             * \param[in] mastched A map associating each plane/cylinder index to the ids of last frame version of those planes/cylinders
             */
            void apply_masks(const cv::Mat& inputImage, const std::vector<cv::Vec3b>& colors, const cv::Mat& maskImage, const primitive_container& primitiveSegments, cv::Mat& labeledImage, const std::map<int, int>& associatedIds, double elapsedTime=0);


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

            void grow_planes_and_cylinders(intpair_vector& cylinderToRegionMap, unsigned int remainingPlanarCells);
            void merge_planes(uint_vector& planeMergeLabels);
            void refine_plane_boundaries(const Eigen::MatrixXf& depthCloudArray, uint_vector& planeMergeLabels, primitive_container& primitiveSegments);
            void refine_cylinder_boundaries(const Eigen::MatrixXf& depthCloudArray, intpair_vector& cylinderToRegionMap, primitive_container& primitiveSegments); 
            void set_masked_display(cv::Mat& segOut); 

            void region_growing(unsigned short x, unsigned short y, const Eigen::Vector3d& seedPlaneNormal, double seedPlaneD);
            void get_connected_components(const cv::Mat& segmentMap, Eigen::MatrixXd& planesAssociationMatrix);

        private:
            Histogram _histogram;

            const unsigned int _width;
            const unsigned int _height;
            const unsigned int _blocSize;
            const unsigned int _pointsPerCellCount;
            const float _minCosAngleForMerge;
            const float _maxMergeDist;

            const bool _useCylinderDetection;

            const unsigned int _cellWidth;
            const unsigned int _cellHeight;

            const unsigned int _horizontalCellsCount;
            const unsigned int _verticalCellsCount;
            const unsigned int _totalCellCount;

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
