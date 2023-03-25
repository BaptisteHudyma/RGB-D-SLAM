#include "histogram.hpp"
#include "../../outputs/logger.hpp"

namespace rgbd_slam {
namespace features {
namespace primitives {

Histogram::Histogram(const uint binPerCoordCount) :
    _binPerCoordCount(binPerCoordCount),
    // set limits
    _minX(0),
    _minY(-M_PI),
    _maxXminX(M_PI - _minX),
    _maxYminY(M_PI - _minY)
{
    _H.assign(_binPerCoordCount * _binPerCoordCount, 0);
    reset();
}

void Histogram::reset()
{
    std::fill_n(_H.begin(), _H.size(), 0);
    _B.clear();
}

void Histogram::init_histogram(const matrixd& points, const vectorb& isUnasignedMask)
{
    //_reset();
    _pointCount = points.rows();
    _B.assign(_pointCount, -1);

    assert(_pointCount == isUnasignedMask.size());
    assert(_B.size() == static_cast<size_t>(isUnasignedMask.size()));

    for (uint i = 0; i < _pointCount; ++i)
    {
        if (isUnasignedMask[i])
        {
            const int xQ = (_binPerCoordCount - 1) * (points(i, 0) - _minX) / _maxXminX;
            // dealing with degeneracy
            int yQ = 0;
            if (xQ > 0)
                yQ = (_binPerCoordCount - 1) * (points(i, 1) - _minY) / _maxYminY;

            const uint bin = yQ * _binPerCoordCount + xQ;
            assert(i < _B.size());
            _B[i] = bin;

            assert(bin < _H.size());
            _H[bin] += 1;
        }
    }
}

std::vector<uint> Histogram::get_points_from_most_frequent_bin() const
{
    int mostFrequentBin = -1;
    uint maxOccurencesCount = 0;

    const size_t binSize = _H.size();
    for (uint i = 0; i < binSize; ++i)
    {
        // get most frequent bin index
        if (_H[i] > maxOccurencesCount)
        {
            mostFrequentBin = i;
            maxOccurencesCount = _H[i];
        }
    }

    std::vector<uint> pointsIds;
    if (mostFrequentBin >= 0)
    {
        pointsIds.reserve(_pointCount);
        // most frequent bin is not empty
        for (uint i = 0; i < _pointCount; ++i)
        {
            if (_B[i] == mostFrequentBin)
            {
                pointsIds.push_back(i);
            }
        }
    }
    return pointsIds;
}

void Histogram::remove_point(const uint pointId)
{
    if (pointId > _B.size())
    {
        outputs::log_error("Histogram: remove_point called on invalid ID");
        exit(-1);
    }
    if (_H[_B[pointId]] != 0)
        _H[_B[pointId]] -= 1;
    _B[pointId] = 1;
}

Histogram::~Histogram()
{
}

} // namespace primitives
} // namespace features
} // namespace rgbd_slam
