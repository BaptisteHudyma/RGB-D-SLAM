#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVEDETECTION_HPP 
#define RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVEDETECTION_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "plane_segment.hpp"
#include "cylinder_segment.hpp"
#include "histogram.hpp"
#include "shape_primitives.hpp"

#include "../../types.hpp"


namespace rgbd_slam {
    namespace features {
        namespace primitives {

            /**
             * \brief Main extraction class.
             * Extracts shape primitives from an organized cloud of points
             */
            class Primitive_Detection {
                //check for planes in an organized depth points matrix 
                protected:
                    //typdefs
                    typedef std::vector<std::pair<int, int>> intpair_vector;
                    typedef std::vector<Plane_Segment> plane_segments_container; 
                    typedef std::vector<Cylinder_Segment> cylinder_segments_container; 
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
                     * \param[out] planeContainer Container of detected planes in depth image
                     * \param[out] primitiveContainer Container of detected cylinders in depth image
                     */
                    void find_primitives(const matrixf& depthMatrix, plane_container& planeContainer, cylinder_container& primitiveContainer);

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
                    void init_planar_cell_fitting(const matrixf& depthCloudArray);

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
                     * \brief When given a plan seed, try to make it grow with it's neighboring cells. Try to fit a cylinder to those merged planes
                     * \param[in] seedId The id of the plane to try to grow
                     * \param[in, out] untriedPlanarCellsCount Count of planar cells that have not been tested for merge yet
                     * \param[in, out] cylinder2regionMap A container that associates a cylinder ID with all the planes IDs that composes it
                     */
                    void grow_plane_segment_at_seed(const uint seedId, uint& untriedPlanarCellsCount, intpair_vector& cylinder2regionMap);

                    /**
                     *
                     * \param[in] cellActivatedCount Number of activated planar cells
                     * \param[in] isActivatedMap A vector associating for each planar patch a flag indicating if it was merged this iteration
                     * \param[in, out] cylinder2regionMap A container that associates a cylinder ID with all the planes IDs that composes it
                     */
                    void cylinder_fitting(const uint cellActivatedCount, const vectorb& isActivatedMap, intpair_vector& cylinder2regionMap);

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
                     * \param[out] planeContainer Container of planes detected in this depth image
                     */
                    void add_planes_to_primitives(const uint_vector& planeMergeLabels, plane_container& planeContainer);

                    /**
                     * \brief Add final cylinders to primitives, compute a mask for display
                     *
                     * \param[in] cylinderToRegionMap Associate a cylinder ID with all the planes IDs that composes it
                     * \param[out] cylinderContainer Container of cylinders detected in this depth image
                     */
                    void add_cylinders_to_primitives(const intpair_vector& cylinderToRegionMap, cylinder_container& cylinderContainer); 

                    /**
                     * \brief Recursively Grow a plane seed and merge it with it's neighbors
                     *
                     * \param[in] x Start X coordinates
                     * \param[in] y Start Y coordinates
                     * \param[in] planeToExpand The plane to grow
                     * \param[in, out] isActivatedMap map of flags, indicating which plane segment were merged
                     */
                    void region_growing(const uint x, const uint y, const Plane_Segment& planeToExpand, vectorb& isActivatedMap);

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

                    plane_segments_container _planeGrid;
                    plane_segments_container _planeSegments;
                    cylinder_segments_container _cylinderSegments;

                    // a grid representing a scaled down depth image, where each pixel represents a potential planar region.
                    // Each pixel as the value of the plane it belongs to, this value is the index of the plane (> 0).
                    // A value of 0 means no plane in this cell
                    cv::Mat_<int> _gridPlaneSegmentMap;
                    // Same, for the cylinders
                    cv::Mat_<int> _gridCylinderSegMap;

                    //arrays
                    vectorb _isUnassignedMask;
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
