#include "Histogram.hpp"

#include <iostream>

namespace primitiveDetection {

    Histogram::Histogram(unsigned int binPerCoordCount) : 
        _binPerCoordCount(binPerCoordCount),
        _binCount(_binPerCoordCount * _binPerCoordCount),
        //set limits
        _minX(0), _minY(-M_PI),
        _maxXminX(M_PI - _minX), _maxYminY(M_PI - _minY)
    {
        _H = new unsigned int[_binCount];
        reset();
    }

    void Histogram::reset() {
        std::fill_n(_H, _binCount, 0);
        _B.clear();
    }

    void Histogram::init_histogram(Eigen::MatrixXd& points, bool* flags) {
        //_reset();
        _pointCount = points.rows();
        _B.assign(_pointCount, -1);

        for(unsigned int i = 0; i < _pointCount; i += 1) {
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

    void Histogram::get_points_from_most_frequent_bin(std::vector<unsigned int>& pointIds ) {
        int mostFrequentBin = -1;
        unsigned int maxOccurencesCount = 0;
        for(unsigned int i = 0; i < _binCount; i += 1) {
            //get most frequent bin index
            if(_H[i] > maxOccurencesCount) {
                mostFrequentBin = i;
                maxOccurencesCount = _H[i];
            }
        }

        if(maxOccurencesCount != 0) {
            //most frequent bin is not empty
            for(unsigned int i = 0; i < _pointCount; i += 1) {
                if(_B[i] == mostFrequentBin) {
                    pointIds.push_back(i);
                }
            }
        }
    }

    void Histogram::remove_point(unsigned int pointId) {
        if(pointId > _B.size()) {
            std::cerr << "Histogram: remove_point called on invalid ID" << std::endl;
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
