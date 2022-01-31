#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVEDETECTION_HPP 
#define RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVEDETECTION_HPP

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
#include "types.hpp"


namespace rgbd_slam {
namespace features {
namespace primitives {

    typedef std::list<primitive_uniq_ptr> primitive_container; 

    /**
      * \brief Main extraction class. Extracts shape primitives from an organized cloud of points
      */
    class Primitive_Detection {
        //check for planes in an organized depth points matrix 
        protected:
            //typdefs
            typedef std::vector<std::pair<int, int>> intpair_vector;
            typedef std::unique_ptr<Plane_Segment> plane_segment_unique_ptr;
            typedef std::unique_ptr<Cylinder_Segment> cylinder_segment_unique_ptr;
            typedef std::vector<plane_segment_unique_ptr> planes_ptr_vector; 
            typedef std::vector<cylinder_segment_unique_ptr> cylinders_ptr_vector; 
            typedef std::vector<uint> uint_vector;

        public:

            /**
              * \param[in] width The fixed depth image width
              * \param[in] height The fixed depth image height
              * \param[in] blocSize Size of an image division, in pixels.
              * \param[in] minCosAngeForMerge Minimum cosinus of the angle of two planes to merge those planes
              * \param[in] maxMergeDist Maximum distance between the center of two planes to merge those planes
              * \param[in] useCylinderDetection Transform some planes in cylinders, when they show an obvious cylinder shape
              */
            Primitive_Detection(const uint width, const uint height, const uint blocSize = 20, const float minCosAngeForMerge = 0.9659, const float maxMergeDist = 50, const bool useCylinderDetection = false);

            /**
              * \brief Main compute function: computes the primitives in the depth imahe
              *
              * \param[in] depthMatrix Organized cloud of points, constructed from depth map
              * \param[out] primitiveSegments Container of detected segments in depth image
              * \param[out] segOut Output image, where each pixel is associated with a shape ID
              */
            void find_primitives(const Eigen::MatrixXf& depthMatrix, primitive_container& primitiveSegments, cv::Mat& segOut);  //detect 3D primitives in depth image



            /**
             * \brief Apply each plane and cylinder mask on the maskImage, with colors corresponding to the ids of colors vector
             *
             * \param[in] inputImage input RGB image on which to put the planes and cylinder masks
             * \param[in] colors A vector of colors, that must remain consistant at each call
             * \param[in] maskImage An image where each pixel is the index of a plane or cylinder
             * \param[in] primitiveSegments A container of the planes and cylinders detected in inputImage
             * \param[in, out] labeledImage The final result: an image with colored masks applied for each plane and cylinder, as well as a bar displaying informations on the top. Must be passed as an empty image of the same dimensions as labeledImage.
             * \param[in] elapsedTime The time elapsed since last frame. Used to display fps
             * \param[in] associatedIds A map associating each plane/cylinder index to the ids of last frame version of those planes/cylinders
             */
            void apply_masks(const cv::Mat& inputImage, const std::vector<cv::Vec3b>& colors, const cv::Mat& maskImage, const primitive_container& primitiveSegments, cv::Mat& labeledImage, const std::unordered_map<int, uint>& associatedIds, const double elapsedTime=0);


            ~Primitive_Detection();

            //perf measurments
            double resetTime;
            double initTime;
            double growTime;
            double mergeTime;
            double refineTime;
            double setMaskTime;

        protected:

            /**
             * \brief Reset the stored data in prevision of another analysis 
             */
            void reset_data();

            /**
             * \brief Init planeGrid and cellDistanceTols
             *
             * \param[in] depthCloudArray Organized point cloud extracted from depth images
             */
            void init_planar_cell_fitting(const Eigen::MatrixXf& depthCloudArray);

            /**
             * \brief Initialize and fill the histogram bins
             *
             * \return  Number of initial planar surfaces
             */
            int init_histogram();

            /**
             * \brief grow planes and find cylinders from those planes
             *
             * \param[out] cylinderToRegionMap Associate a cylinder ID with all the planes IDs that composes it
             * \param[in] remainingPlanarCells Unmatched plane count 
             */
            void grow_planes_and_cylinders(uint remainingPlanarCells, intpair_vector& cylinderToRegionMap);

            /**
             * \brief Merge close planes by comparing normals and MSE
             *
             * \param[out] planeMergeLabels Container of merged indexes: associates plane index to other plane index
             */
            void merge_planes(uint_vector& planeMergeLabels);

            /**
             * \brief Refine the final plane edges in mask images
             *
             * \param[in] depthCloudArray Organized cloud point
             * \param[out] planeMergeLabels Container associating plane ID to global plane IDs
             * \param[in, out] primitiveSegments Container of shapes detected in this depth image
             */
            void refine_plane_boundaries(const Eigen::MatrixXf& depthCloudArray, uint_vector& planeMergeLabels, primitive_container& primitiveSegments);

            /**
             * \brief Refine the cylinder edges in mask images
             *
             * \param[in] depthCloudArray Organized cloud point
             * \param[out] cylinderToRegionMap Associate a cylinder ID with all the planes IDs that composes it
             * \param[in, out] primitiveSegments Container of shapes detected in this depth image
             */
            void refine_cylinder_boundaries(const Eigen::MatrixXf& depthCloudArray, intpair_vector& cylinderToRegionMap, primitive_container& primitiveSegments); 

            /**
             * \brief Set output image pixel value with the index of the detected shape
             *
             * \param[out] segOut Image segmented by shapes ids: Associates an image coordinate to a shape ID
             */
            void set_masked_display(cv::Mat& segOut); 

            /**
             * \brief Recursively Grow a plane seed and merge it with it's neighbors
             *
             * \param[in] x Start X coordinates
             * \param[in] y Start Y coordinates
             * \param[in] seedPlaneNormal Normal of the plane to grow from (Components A, B, C of the standard plane equation)
             * \param[in] seedPlaneD D component of the plane to grow from
             */
            void region_growing(const unsigned short x, const unsigned short y, const vector3& seedPlaneNormal, const double seedPlaneD);

            /**
             * \brief Fill an association matrix that links connected plane components
             *
             * \param[in] segmentMap
             * \param[out] planesAssociationMatrix
             */
            void get_connected_components(const cv::Mat& segmentMap, Matrixb& planesAssociationMatrix);

        private:
            Histogram _histogram;

            const uint _width;
            const uint _height;
            const uint _blocSize;
            const uint _pointsPerCellCount;
            const float _minCosAngleForMerge;
            const float _maxMergeDist;

            const bool _useCylinderDetection;

            const uint _cellWidth;
            const uint _cellHeight;

            const uint _horizontalCellsCount;
            const uint _verticalCellsCount;
            const uint _totalCellCount;

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
}
}

#endif
