#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVEDETECTION_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVEDETECTION_HPP

#include "../../types.hpp"
#include "cylinder_segment.hpp"
#include "histogram.hpp"
#include "plane_segment.hpp"
#include "shape_primitives.hpp"
#include "polygon.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace rgbd_slam::features::primitives {

/**
 * \brief Main extraction class.
 * Extracts shape primitives from an organized cloud of points
 */
class Primitive_Detection
{
    // check for planes in an organized depth points matrix
  protected:
    // typdefs
    using intpair_vector = std::vector<std::pair<int, int>>;
    using plane_segments_container = std::vector<Plane_Segment>;
    using cylinder_segments_container = std::vector<Cylinder_Segment>;
    using uint_vector = std::vector<uint>;

  public:
    /**
     * \param[in] width The fixed depth image width
     * \param[in] height The fixed depth image height
     */
    Primitive_Detection(const uint width, const uint height);

    /**
     * \brief Main compute function: computes the primitives in the depth imahe
     * \param[in] depthMatrix Organized cloud of points, constructed from depth map
     * \param[in] depthImage The depth map used to construct depthMatrix
     * \param[out] planeContainer Container of detected planes in depth image
     * \param[out] primitiveContainer Container of detected cylinders in depth image
     */
    void find_primitives(const matrixf& depthMatrix,
                         const cv::Mat_<float>& depthImage,
                         plane_container& planeContainer,
                         cylinder_container& primitiveContainer) noexcept;

    // perf measurments
    double resetTime;
    double initTime;
    double growTime;
    double mergeTime;
    double refineTime;

  protected:
    /**
     * \brief Reset the stored data in prevision of another analysis
     */
    void reset_data() noexcept;

    /**
     * \brief Init planeGrid and cellDistanceTols
     * \param[in] depthCloudArray Organized point cloud extracted from depth images
     */
    void init_planar_cell_fitting(const matrixf& depthCloudArray) noexcept;

    /**
     * \brief Initialize and fill the histogram bins
     * \return  Number of initial planar surfaces
     */
    [[nodiscard]] uint init_histogram() noexcept;

    /**
     * \brief grow planes and find cylinders from those planes
     * \param[in] remainingPlanarCells Unmatched plane count
     * \return A container that associates a cylinder ID with all the planes IDs that composes it
     */
    [[nodiscard]] intpair_vector grow_planes_and_cylinders(const uint remainingPlanarCells) noexcept;

    /**
     * \brief When given a plan seed, try to make it grow with it's neighboring cells. Try to fit a cylinder to those
     * merged planes
     * \param[in] seedId The id of the plane to try to grow
     * \param[in, out] untriedPlanarCellsCount Count of planar cells that have not been tested for merge yet
     * \param[in, out] cylinder2regionMap A container that associates a cylinder ID with all the planes IDs that
     * composes it
     */
    void grow_plane_segment_at_seed(const uint seedId,
                                    uint& untriedPlanarCellsCount,
                                    intpair_vector& cylinder2regionMap) noexcept;

    /**
     * \brief Add the given plane segment to the tracked features
     * \param[in] newPlaneSegment The plane segment to add
     * \param[in] isActivatedMap container of the patches to add to this plane. A value at true means that the
     * associated patch index is part of this plane
     */
    void add_plane_segment_to_features(const Plane_Segment& newPlaneSegment, const vectorb& isActivatedMap) noexcept;

    /**
     * \brief Compute the plane hull in plane coordinates
     * \param[in] planeSegment The plane segment to compute a boundary for
     * \param[in] depthImage The depth image used to create depthMatrix
     * \param[in] mask The mask of this plane segment in image space
     * \return The boundary point of the polygon
     */
    [[nodiscard]] std::vector<vector3> compute_plane_segment_boundary(const Plane_Segment& planeSegment,
                                                                      const cv::Mat_<float>& depthImage,
                                                                      const cv::Mat_<uchar>& mask) const noexcept;

    /**
     * \brief Try to fit a plane to a cylinder
     * \param[in] cylinderSegment The cylinder segment to fit
     * \param[in] cellActivatedCount Number of activated planar cells
     * \param[in] segId Id of the part of the cylinfer we try to fit
     * \param[out] newMergedPlane A plane segment that is part of the cylinder. Not define if this function returns
     * false
     * \return True if a plane fitting was found
     */
    [[nodiscard]] bool find_plane_segment_in_cylinder(const Cylinder_Segment& cylinderSegment,
                                                      const uint cellActivatedCount,
                                                      const uint segId,
                                                      Plane_Segment& newMergedPlane) noexcept;

    /**
     * \param[in] cylinderSegment The cylinder segment to fit
     * \param[in] cellActivatedCount Number of activated planar cells
     * \param[in] segId Id of the part of the cylinfer we try to fit
     * \param[in] newMergedPlane A plane segment that is part of the cylinder
     * \param[in, out] cylinder2regionMap A container that associates a cylinder ID with all the planes IDs
     * that composes it
     */
    void add_cylinder_to_features(const Cylinder_Segment& cylinderSegment,
                                  const uint cellActivatedCount,
                                  const uint segId,
                                  const Plane_Segment& newMergedPlane,
                                  intpair_vector& cylinder2regionMap) noexcept;

    /**
     * \brief
     * \param[in] cellActivatedCount Number of activated planar cells
     * \param[in] isActivatedMap A vector associating for each planar patch a flag indicating if it was merged this
     * iteration
     * \param[in, out] cylinder2regionMap A container that associates a cylinder ID with all the planes IDs
     * that composes it
     */
    void cylinder_fitting(const uint cellActivatedCount,
                          const vectorb& isActivatedMap,
                          intpair_vector& cylinder2regionMap) noexcept;

    /**
     * \brief Merge close planes by comparing normals and MSE
     *
     * \return Container of merged indexes: associates plane index to other plane index
     */
    [[nodiscard]] uint_vector merge_planes() noexcept;

    /**
     * \brief Add final plane to primitives, compute a mask for display
     *
     * \param[in] planeMergeLabels Container associating plane ID to global plane IDs
     * \param[in] depthImage The depth image used to construct the depthImage
     * \param[out] planeContainer Container of planes detected in this depth image
     */
    void add_planes_to_primitives(const uint_vector& planeMergeLabels,
                                  const cv::Mat_<float>& depthImage,
                                  plane_container& planeContainer) noexcept;

    /**
     * \brief Add final cylinders to primitives, compute a mask for display
     *
     * \param[in] cylinderToRegionMap Associate a cylinder ID with all the planes IDs that composes it
     * \param[out] cylinderContainer Container of cylinders detected in this depth image
     */
    void add_cylinders_to_primitives(const intpair_vector& cylinderToRegionMap,
                                     cylinder_container& cylinderContainer) noexcept;

    /**
     * \brief Recursively Grow a plane seed and merge it with it's neighbors
     *
     * \param[in] x Start X coordinates
     * \param[in] y Start Y coordinates
     * \param[in] planeToExpand The plane to grow
     * \param[in, out] isActivatedMap map of flags, indicating which plane segment were merged
     */
    void region_growing(const uint x,
                        const uint y,
                        const Plane_Segment& planeToExpand,
                        vectorb& isActivatedMap) noexcept;

    /**
     * \brief Fill an association matrix that links connected plane components
     *
     * \param[in] segmentMap
     * \param[in] numberOfPlanes Number of plane segments in segment map
     *
     * \return A symmetrical boolean matrix, indicating if a plane segment is connected to another plane segment
     */
    [[nodiscard]] Matrixb get_connected_components_matrix(const cv::Mat_<int>& segmentMap,
                                                          const size_t numberOfPlanes) const noexcept;

  private:
    Histogram<parameters::detection::depthMapPatchSize_px> _histogram;

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

    // arrays
    vectorb _isUnassignedMask;
    std::vector<float> _cellDistanceTols;

    // primitive cell mask (preallocated)
    cv::Mat_<uchar> _mask;
    cv::Mat_<uchar> _maskEroded;
    cv::Mat_<uchar> _maskDilated;
    // kernel
    cv::Mat_<uchar> _maskCrossKernel;
    cv::Mat_<uchar> _maskSquareKernel;

    // prevent backend copy
    Primitive_Detection(const Primitive_Detection&);
    Primitive_Detection& operator=(const Primitive_Detection&);
};

} // namespace rgbd_slam::features::primitives

#endif
