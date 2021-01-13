#include "Histogram.hpp"

using namespace planeDetection;

Histogram::Histogram(int binPerCoordCount) 
    : binPerCoordCount(binPerCoordCount),
    binCount(this->binPerCoordCount * this->binPerCoordCount),
    //set limits
    minX(0), minY(-M_PI),
    maxXminX(M_PI - minX), maxYminY(M_PI - minY)
{
    this->H = new int[this->binCount];
    reset();
}

void Histogram::reset() {
    std::fill_n(this->H, this->binCount, 0);
    this->B.clear();
}

void Histogram::init_histogram(Eigen::MatrixXd& points, bool* flags) {
    //this->reset();
    this->pointCount = points.rows();
    this->B.assign(this->pointCount, -1);

    for(int i = 0; i < this->pointCount; i += 1) {
        if(flags[i]) {
            int xQ = (this->binPerCoordCount - 1) * (points(i, 0) - this->minX) / this->maxXminX;
            //dealing with degeneracy
            int yQ = 0;
            if(xQ > 0)
                yQ = (this->binPerCoordCount - 1) * (points(i, 1) - this->minY) / this->maxYminY;

            int bin = yQ * this->binPerCoordCount + xQ;
            B[i] = bin;
            H[bin] += 1;
        }
    }
}

void Histogram::get_points_from_most_frequent_bin(std::vector<int>& pointIds ) {
    int mostFrequentBin = -1;
    int maxOccurencesCount = 0;
    for(int i = 0; i < this->binCount; i += 1) {
        //get most frequent bin index
        if(H[i] > maxOccurencesCount) {
            mostFrequentBin = i;
            maxOccurencesCount = H[i];
        }
    }

    if(maxOccurencesCount > 0) {
        //most frequent bin is not empty
        for(int i = 0; i < this->pointCount; i += 1) {
            if(B[i] == mostFrequentBin) {
                pointIds.push_back(i);
            }
        }
    }
}

void Histogram::remove_point(int pointId) {
    H[B[pointId]] -= 1;
    B[pointId] = 1;
}


Histogram::~Histogram() {
    delete []this->H;
}
