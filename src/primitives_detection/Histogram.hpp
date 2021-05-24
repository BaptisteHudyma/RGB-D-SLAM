#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <Eigen/Dense>
#include <vector>

namespace primitiveDetection {

    class Histogram {
        public:
            Histogram(int binPerCoordCount);

            void init_histogram(Eigen::MatrixXd& points, bool* flags);
            void get_points_from_most_frequent_bin(std::vector<int>&);
	        void remove_point(int pointId);
            void reset();

            ~Histogram();
        
        protected:

        private:
            int* _H;
            std::vector<int> _B;

            const int _binPerCoordCount;
            const int _binCount;
            unsigned int _pointCount;

            const double _minX;
            const double _minY;
            const double _maxXminX;
            const double _maxYminY;

        private:
            //prevent backend copy
            Histogram(const Histogram&);
            Histogram& operator=(const Histogram&);
    };

}

#endif
