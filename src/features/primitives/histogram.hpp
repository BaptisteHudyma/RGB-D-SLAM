#ifndef RGBDSLAM_FEATURES_PRIMITIVES_HISTOGRAM_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_HISTOGRAM_HPP

#include "../../types.hpp"
#include "../../outputs/logger.hpp"
#include <format>
#include <vector>

namespace rgbd_slam::features::primitives {

// limits
constexpr double minX = 0;
constexpr double minY = -M_PI;
constexpr double maxXminX = M_PI - minX;
constexpr double maxYminY = M_PI - minY;

/**
 * \brief Basic 2D Histogram class, handling an histogram of N x N
 */
template<size_t Size> class Histogram
{
  public:
    Histogram()
    {
        _histogram.fill(0);
        reset();
    }

    /**
     * \brief Initialise the histogram
     *
     * \param[in] points Points to put directly in the histogram bins
     * \param[in] isUnasignedMask Array of size points.rows(), indicating which point is in a planar segment
     */
    void init_histogram(const matrixd& points, const vectorb& isUnasignedMask) noexcept
    {
        //_reset();
        _pointCount = static_cast<uint>(points.rows());
        _bins.assign(_pointCount, -1);

        assert(_pointCount == isUnasignedMask.size());
        assert(_bins.size() == static_cast<size_t>(isUnasignedMask.size()));

        for (uint i = 0; i < _pointCount; ++i)
        {
            if (isUnasignedMask[i])
            {
                const int xQ = static_cast<int>(floor((Size - 1) * (points(i, 0) - minX) / maxXminX));
                // dealing with degeneracy
                int yQ = 0;
                if (xQ > 0)
                    yQ = static_cast<int>(floor((Size - 1) * (points(i, 1) - minY) / maxYminY));

                const uint bin = yQ * Size + xQ;
                assert(i < _bins.size());
                _bins[i] = static_cast<int>(bin);

                assert(bin < _histogram.size());
                _histogram[bin] += 1;
            }
        }
    }

    /**
     * \brief Return the points in the bin containing the most points
     *
     * \return Container storing the points in the biggest bin
     */
    [[nodiscard]] std::vector<uint> get_points_from_most_frequent_bin() const noexcept
    {
        int mostFrequentBin = -1;
        uint maxOccurencesCount = 0;

        for (uint i = 0; i < _histogram.size(); ++i)
        {
            // get most frequent bin index
            if (_histogram[i] > maxOccurencesCount)
            {
                mostFrequentBin = static_cast<int>(i);
                maxOccurencesCount = _histogram[i];
            }
        }

        std::vector<uint> pointsIds;
        if (mostFrequentBin >= 0)
        {
            pointsIds.reserve(_pointCount);
            // most frequent bin is not empty
            for (uint i = 0; i < _pointCount; ++i)
            {
                if (_bins[i] == mostFrequentBin)
                {
                    pointsIds.push_back(i);
                }
            }
        }
        return pointsIds;
    }

    /**
     * \brief Remove all points from a bin
     */
    void remove_point(const uint pointId) noexcept
    {
        if (pointId >= _bins.size())
        {
            outputs::log_error(std::format("Histogram: remove_point called on invalid ID {}", pointId));
            exit(-1);
        }
        if (_histogram[_bins[pointId]] != 0)
            _histogram[_bins[pointId]] -= 1;
        _bins[pointId] = 1;
    }

    /**
     * \brief Empty bins and clear content
     */
    void reset() noexcept
    {
        _histogram.fill(0);
        _bins.clear();
    }

  private:
    std::array<uint, Size * Size> _histogram;
    std::vector<int> _bins;

    uint _pointCount;

    // prevent backend copy
    Histogram(const Histogram&);
    Histogram& operator=(const Histogram&);
};

} // namespace rgbd_slam::features::primitives

#endif
