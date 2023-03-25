#ifndef RGBDSLAM_FEATURES_PRIMITIVES_CYLINDERSEGMENT_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_CYLINDERSEGMENT_HPP

#include "../../types.hpp"
#include "plane_segment.hpp"
#include <vector>

namespace rgbd_slam {
namespace features {
namespace primitives {

/**
 * \brief Stored a cylinder segment.
 * Computes the parameters (radius, normal of the main axis, eigen values) with a RANSAC fitting
 */
class Cylinder_Segment
{
  public:
    /**
     * \brief Main constructor: fits a cylinder using the plane segments in planeGrid, using RANSAC
     * \param[in] planeGrid The plane segment container
     * \param[in] isActivatedMask An array of size planeCount, referencing activated plane segments
     * \param[in] cellActivatedCount
     */
    Cylinder_Segment(const std::vector<Plane_Segment>& planeGrid,
                     const vectorb& isActivatedMask,
                     const uint cellActivatedCount);

    /**
     * \brief Copy constructor
     * \param[in] seg Cylinder_Segment to copy
     * \param[in] subRegionId Cylinder element ID to copy
     */
    Cylinder_Segment(const Cylinder_Segment& seg, const uint subRegionId);

    /**
     * \brief Copy constructor
     * \param[in] seg Cylinder_Segment to copy
     */
    Cylinder_Segment(const Cylinder_Segment& seg);

    /**
     * \brief Compute the point to cylinder surface distance. This distance is an approximation, our cylinder being
     * defined as a sum of plane segments and points on it's main axis
     * \param[in] point The point to compute distance to
     * \return The signed distance between the point and cylinder surface
     */
    double get_distance(const vector3& point) const;

    ~Cylinder_Segment();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  public:
    /**
     * \brief
     * \return The number of plane segments fitted in this cylinder surface
     */
    uint get_segment_count() const;

    /**
     * \brief
     * \param[in] index The index of the cylinder part to search
     * \return the Mean Sqared Error of the fitting process
     */
    double get_MSE_at(const uint index) const;

    /**
     * \brief
     */
    bool is_inlier_at(const uint indexA, const uint indexB) const;

    /**
     * \brief
     * \param[in] index The index of the cylinder part to search
     */
    uint get_local_to_global_mapping(const uint index) const;

    /**
     * \brief
     * \param[in] index The index of the cylinder part to search
     */
    const vector3& get_axis1_point(const uint index) const;

    /**
     * \param[in] index The index of the cylinder part to search
     */
    const vector3& get_axis2_point(const uint index) const;

    /**
     * \brief Return the normal of a portion of this cylinder segement
     *
     * \param[in] index The index of the portion to return, between 0 and _normalsAxis1Axis2.size()
     */
    double get_axis_normal(const uint index) const;

    /**
     * \brief Return the radius of this cylinder segment
     * \return The radius of the cylinder segment, in frame units
     */
    double get_radius(const uint index) const;

    /**
     * \brief Return the absolute result of the dot product of the two normals
     * \param[in] other The cyclinder segment to compare normal with
     * \return A double between 0 and 1, 0 when the normals are orthogonal, 1 il they are parallels.
     */
    double get_normal_similarity(const Cylinder_Segment& other);

    const vector3 get_normal() const;

  protected:
    /**
     * \brief Run a RANSAC pose optimization with 6 points
     * \param[in] maximumIterations The maximum RANSAC loops that this function will run
     * \param[in] idsLeft Ids of the planes left to fit. Contains all the ids, and mark as false the already fitted
     * segments
     * \param[in] planeNormals Normals of the planes to fit
     * \param[in] projectedCentroids
     * \param[in] idsLeftMask container of the same size as idsLeft, indicating which ids are inliers
     * \param[in] IFinal
     * \brief Return the number of inliers of this cylinder fitting
     */
    size_t run_ransac_loop(const uint maximumIterations,
                           const std::vector<uint>& idsLeft,
                           const matrixd& planeNormals,
                           const matrixd& projectedCentroids,
                           const Matrixb& idsLeftMask,
                           Matrixb& IFinal);

    double get_distance(const vector3& point, const size_t segmentId) const;

  private:
    vector3 _axis;

    std::vector<matrixd> _centers;
    vector3_vector _pointsAxis1;
    vector3_vector _pointsAxis2;
    std::vector<double> _normalsAxis1Axis2;
    std::vector<Matrixb> _inliers;

    std::vector<double> _MSE;
    std::vector<double> _radius;

    uint _cellActivatedCount;
    uint _segmentCount;
    std::vector<uint> _local2globalMap;

  private:
    // prevent dangerous and inefficient back end copy
    Cylinder_Segment& operator=(const Cylinder_Segment& seg); // copy operator
};

} // namespace primitives
} // namespace features
} // namespace rgbd_slam

#endif
