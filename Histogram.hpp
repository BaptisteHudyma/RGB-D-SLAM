#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#define DMAX std::numeric_limits<float>::max()
#define DMIN std::numeric_limits<float>::min()

namespace planeDetection {

    class Histogram {
        public:
            Histogram(int h);

            std::vector<int>& get_points_from_most_frequent_bin();

            std::vector<int> H;
            std::vector<int> B;

            int binPerCoordsCount;
            int binCount;
            int pointCount;
    };

}

#endif
