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
            Histogram(int binPerCoordCount);

            void init_histogram(Eigen::MatrixXd& points, bool* flags);
            std::vector<int> get_points_from_most_frequent_bin();
	        void remove_point(int pointId);

            void reset();

            ~Histogram();
        
        protected:
            

        private:
            std::vector<int> H;
            std::vector<int> B;

            const int binPerCoordCount;
            int binCount;
            int pointCount;
    };

}

#endif
