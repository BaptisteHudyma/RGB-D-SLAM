#include "Histogram.hpp"
#include "utils.hpp"

namespace rgbd_slam {
namespace features {
namespace primitives {

    Histogram::Histogram(const uint binPerCoordCount) : 
        _binPerCoordCount(binPerCoordCount),
        _binCount(_binPerCoordCount * _binPerCoordCount),
        //set limits
        _minX(0), _minY(-M_PI),
        _maxXminX(M_PI - _minX), _maxYminY(M_PI - _minY)
    {
        _H = new uint[_binCount];
        reset();
    }

    void Histogram::reset() {
        std::fill_n(_H, _binCount, 0);
        _B.clear();
    }

    void Histogram::init_histogram(const Eigen::MatrixXd& points, const bool* flags) {
        //_reset();
        _pointCount = points.rows();
        _B.assign(_pointCount, -1);

        for(uint i = 0; i < _pointCount; i += 1) {
            if(flags[i]) {
                int xQ = (_binPerCoordCount - 1) * (points(i, 0) - _minX) / _maxXminX;
                //dealing with degeneracy
                int yQ = 0;
                if(xQ > 0)
                    yQ = (_binPerCoordCount - 1) * (points(i, 1) - _minY) / _maxYminY;

                int bin = yQ * _binPerCoordCount + xQ;
                _B[i] = bin;
                _H[bin] += 1;
            }
        }
    }

    void Histogram::get_points_from_most_frequent_bin(std::vector<uint>& pointsIds ) const {
        int mostFrequentBin = -1;
        uint maxOccurencesCount = 0;
        for(uint i = 0; i < _binCount; i += 1) {
            //get most frequent bin index
            if(_H[i] > maxOccurencesCount) {
                mostFrequentBin = i;
                maxOccurencesCount = _H[i];
            }
        }

        if(mostFrequentBin >= 0) {
            //most frequent bin is not empty
            for(uint i = 0; i < _pointCount; i += 1) {
                if(_B[i] == mostFrequentBin) {
                    pointsIds.push_back(i);
                }
            }
        }
    }

    void Histogram::remove_point(const uint pointId) {
        if(pointId > _B.size()) {
            utils::log_error("Histogram: remove_point called on invalid ID");
            exit(-1);
        }
        if(_H[_B[pointId]] != 0)
            _H[_B[pointId]] -= 1;
        _B[pointId] = 1;
    }


    Histogram::~Histogram() {
        delete []_H;
    }

}
}
}
