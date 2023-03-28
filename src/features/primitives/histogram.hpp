#ifndef RGBDSLAM_FEATURES_PRIMITIVES_HISTOGRAM_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_HISTOGRAM_HPP

#include "../../types.hpp"
#include <vector>

namespace rgbd_slam::features::primitives {

/**
 * \brief Basic 2D Histogram class, handling an histogram of N x N
 */
class Histogram
{
  public:
    /**
     * \param[in] binPerCoordCount Size of a bin, in pixels
     */
    Histogram(const uint binPerCoordCount);

    /**
     * \brief Initialise the histogram
     *
     * \param[in] points Points to put directly in the histogram bins
     * \param[in] isUnasignedMask Array of size points.rows(), indicating which point is in a planar segment
     */
    void init_histogram(const matrixd& points, const vectorb& isUnasignedMask);

    /**
     * \brief Return the points in the bin containing the most points
     *
     * \return Container storing the points in the biggest bin
     */
    std::vector<uint> get_points_from_most_frequent_bin() const;

    /**
     * \brief Remove all points from a bin
     */
    void remove_point(const uint pointId);

    /**
     * \brief Empty bins and clear content
     */
    void reset();

  private:
    std::vector<uint> _H;
    std::vector<int> _B;

    const uint _binPerCoordCount;
    uint _pointCount;

    const double _minX;
    const double _minY;
    const double _maxXminX;
    const double _maxYminY;

    // prevent backend copy
    Histogram(const Histogram&);
    Histogram& operator=(const Histogram&);
};

} // namespace rgbd_slam::features::primitives

#endif
