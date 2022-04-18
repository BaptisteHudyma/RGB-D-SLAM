#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVEDETECTION_HPP 
#define RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVEDETECTION_HPP

#include <algorithm>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include "plane_segment.hpp"
#include "cylinder_segment.hpp"
#include "histogram.hpp"
#include "shape_primitives.hpp"
#include "types.hpp"


namespace rgbd_slam {
    namespace features {
        namespace primitives {

            typedef std::map<uchar, primitive_uniq_ptr> primitive_container; 

            /**
             * \brief Main extraction class. Extracts shape primitives from an organized cloud of points
             */
            class Primitive_Detection {
                //check for planes in an organized depth points matrix 
                protected:
                    //typdefs
                    typedef std::vector<std::pair<int, int>> intpair_vector;
                    typedef std::shared_ptr<Plane_Segment> plane_segment_unique_ptr;
                    typedef std::shared_ptr<Cylinder_Segment> cylinder_segment_unique_ptr;
                    typedef std::vector<plane_segment_unique_ptr> planes_ptr_vector; 
                    typedef std::vector<cylinder_segment_unique_ptr> cylinders_ptr_vector; 
                    typedef std::vector<uint> uint_vector;

                public:

                    /**
                     * \param[in] width The fixed depth image width
                     * \param[in] height The fixed depth image height
                     * \param[in] blocSize Size of an image division, in pixels.
                     * \param[in] minCosAngleForMerge Minimum cosinus of the angle of two planes to merge those planes
                     * \param[in] maxMergeDistance Maximum distance between the center of two planes to merge those planes
                     */
                    Primitive_Detection(const uint width, const uint height, const uint blocSize = 20, const float minCosAngleForMerge = 0.9659f, const float maxMergeDistance = 50.0f);

                    /**
                     * \brief Main compute function: computes the primitives in the depth imahe
                     *
                     * \param[in] depthMatrix Organized cloud of points, constructed from depth map
                     * \param[out] primitiveSegments Container of detected segments in depth image
                     */
                    void find_primitives(const Eigen::MatrixXf& depthMatrix, primitive_container& primitiveSegments);

                    ~Primitive_Detection();

                    //perf measurments
                    double resetTime;
                    double initTime;
                    double growTime;
                    double mergeTime;
                    double refineTime;

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
                    uint init_histogram();

                    /**
                     * \brief grow planes and find cylinders from those planes
                     *
                     * \param[in] remainingPlanarCells Unmatched plane count 
                     *
                     * \return A container that associates a cylinder ID with all the planes IDs that composes it
                     */
                    intpair_vector grow_planes_and_cylinders(const uint remainingPlanarCells);

                    /**
                     * \brief Merge close planes by comparing normals and MSE
                     *
                     * \return Container of merged indexes: associates plane index to other plane index
                     */
                    uint_vector merge_planes();

                    /**
                     * \brief Add final plane to primitives, compute a mask for display 
                     *
                     * \param[in] planeMergeLabels Container associating plane ID to global plane IDs
                     * \param[in, out] primitiveSegments Container of shapes detected in this depth image
                     */
                    void add_planes_to_primitives(const uint_vector& planeMergeLabels, primitive_container& primitiveSegments);

                    /**
                     * \brief Add final cylinders to primitives, compute a mask for display
                     *
                     * \param[in] cylinderToRegionMap Associate a cylinder ID with all the planes IDs that composes it
                     * \param[in, out] primitiveSegments Container of shapes detected in this depth image
                     */
                    void add_cylinders_to_primitives(const intpair_vector& cylinderToRegionMap, primitive_container& primitiveSegments); 

                    /**
                     * \brief Recursively Grow a plane seed and merge it with it's neighbors
                     *
                     * \param[in] x Start X coordinates
                     * \param[in] y Start Y coordinates
                     * \param[in] seedPlaneNormal Normal of the plane to grow from (Components A, B, C of the standard plane equation)
                     * \param[in] seedPlaneD D component of the plane to grow from
                     */
                    void region_growing(const uint x, const uint y, const vector3& seedPlaneNormal, const double seedPlaneD);

                    /**
                     * \brief Fill an association matrix that links connected plane components
                     *
                     * \param[in] segmentMap
                     * \param[in] numberOfPlanes Number of plane segments in segment map
                     *
                     * \return A symmetrical boolean matrix, indicating if a plane segment is connected to another plane segment
                     */
                    Matrixb get_connected_components_matrix(const cv::Mat& segmentMap, const size_t numberOfPlanes) const;



                private:
                    Histogram _histogram;

                    const uint _width;
                    const uint _height;
                    const uint _pointsPerCellCount;
                    const float _minCosAngleForMerge;
                    const float _maxMergeDist;

                    const uint _cellWidth;
                    const uint _cellHeight;

                    const uint _horizontalCellsCount;
                    const uint _verticalCellsCount;
                    const uint _totalCellCount;

                    std::vector<plane_segment_unique_ptr> _planeGrid;
                    planes_ptr_vector _planeSegments;
                    cylinders_ptr_vector _cylinderSegments;

                    cv::Mat_<int> _gridPlaneSegmentMap;
                    cv::Mat_<int> _gridCylinderSegMap;

                    //arrays
                    std::vector<bool> _isActivatedMap;
                    std::vector<bool> _isUnassignedMask;
                    std::vector<float> _cellDistanceTols;

                    // primitive cell mask
                    cv::Mat _mask;
                    cv::Mat _maskEroded;
                    // kernel
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
