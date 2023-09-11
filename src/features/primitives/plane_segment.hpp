#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PLANESEGMENT_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_PLANESEGMENT_HPP

#include <cmath>

#include "../../parameters.hpp"
#include "../../types.hpp"
#include "coordinates.hpp"

namespace rgbd_slam::features::primitives {

/**
 * \brief Node class representing a depth graph point.
 * Used to find planes in a depth image.
 * Mainly inspired from CAPE program
 */
class Plane_Segment
{
  public:
    Plane_Segment();
    /**
     * \brief Copy construtor
     */
    Plane_Segment(const Plane_Segment& seg);

    static void set_static_members(const uint cellWidth, const uint pointPerCellCount)
    {
        assert(pointPerCellCount > 0);
        assert(cellWidth > 0);

        _ptsPerCellCount = pointPerCellCount;
        _minZeroPointCount = static_cast<uint>(
                std::floor(static_cast<float>(_ptsPerCellCount) * Parameters::get_minimum_zero_depth_proportion()));
        _cellWidth = cellWidth;
        _cellHeight = _ptsPerCellCount / _cellWidth;

        _isStaticSet = true;
    }

    void init_plane_segment(const matrixf& depthCloudArray, const uint cellId);

    /**
     * \brief Merge the PCA saved values in prevision of a plane fitting. This function do not make any new plane
     * calculations
     *
     * \param[in] planeSegment Another plane segment
     */
    void expand_segment(const Plane_Segment& planeSegment);

    /**
     * \brief Compute the dot product of two plane normals, It is thecos of the angle between those normals
     * \param[in] p The other plane segment to check the angle to
     * \return A number between -1 and 1
     */
    double get_cos_angle(const Plane_Segment& p) const;

    /**
     * \brief Return the distance from a plane to the given point
     */
    double get_point_distance(const vector3& point) const;
    double get_point_distance_squared(const vector3& point) const;

    /**
     * \brief Check if this plane segment and another onsatisfy the mertge conditions
     * \param[in] p The other plane segment to check merge conditions with
     * \param[in] maxMatchDistance Maximum distance after which two planes wont be merged
     * \return True if the planes could be merged. It is based on the normal angles and center distances
     */
    bool can_be_merged(const Plane_Segment& p, const double maxMatchDistance) const;

    /**
     * \brief Fit a plane to the contained points using PCA
     */
    void fit_plane();

    /**
     * \brief Clears this segment parameters to restart analysis
     */
    void clear_plane_parameters(); // clear node plane parameters

    /**
     * \brief Compute the covariance of the points in the plane
     * \return the filled covariance matrix of the point cloud
     */
    matrix33 get_point_cloud_covariance() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double get_MSE() const { return _MSE; };
    vector3 get_normal() const { return _parametrization.get_normal(); };
    utils::CameraCoordinate get_centroid() const { return _centroid; };
    utils::CameraCoordinate get_center() const { return _parametrization.get_center(); };
    double get_plane_d() const { return _parametrization.get_d(); };
    vector4 get_parametrization() const { return _parametrization.get_parametrization(); };
    bool is_planar() const { return _isPlanar; };
    double get_score() const { return _score; };
    uint get_point_count() const { return _pointCount; };

  protected:
    /**
     * \brief Check that the point cloud for this plane patch is verticaly continuous
     * \param[in] depthMatrix The depth representation of the image
     */
    bool is_cell_vertical_continuous(const matrixf& depthMatrix) const;

    /**
     * \brief Check that the point cloud for this plane patch is horizontaly continuous
     * \param[in] depthMatrix The depth representation of the image
     */
    bool is_cell_horizontal_continuous(const matrixf& depthMatrix) const;

    /**
     * \brief return the covariance of this point cloud computed from König-Huygen formula
     */
    matrix33 get_point_cloud_Huygen_covariance() const;

  private:
    static inline uint _ptsPerCellCount;   // max nb of points per initial cell
    static inline uint _minZeroPointCount; // min acceptable zero points in a node
    static inline uint _cellWidth;
    static inline uint _cellHeight;
    static inline bool _isStaticSet = false;

    uint _pointCount = 0;                             // point count
    double _score = 0.0;                              // plane fitting score
    double _MSE = std::numeric_limits<double>::max(); // plane fitting mean square error
    bool _isPlanar = false; // true if node represent a correct node, false: ignore node while mapping

    utils::CameraCoordinate _centroid; // mean point of all points in node
    utils::PlaneCoordinates _parametrization;

    // PCA stored coeffs: efficient calculations of point cloud characteristics
    double _Sx = 0.0;  // sum of x
    double _Sy = 0.0;  // sum of y
    double _Sz = 0.0;  // sum of z
    double _Sxs = 0.0; // sum of x squared
    double _Sys = 0.0; // sum of y squared
    double _Szs = 0.0; // sum of z squared
    double _Sxy = 0.0; // sum of x*y
    double _Syz = 0.0; // sum of y*z
    double _Szx = 0.0; // sum of z*x

    // prevent backend copy
    Plane_Segment& operator=(const Plane_Segment& seg);
};

} // namespace rgbd_slam::features::primitives

#endif
