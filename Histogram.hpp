#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <Eigen/Dense>
#include <vector>

namespace planeDetection {

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
            std::vector<int> H;
            std::vector<int> B;

            const int binPerCoordCount;
            int binCount;
            int pointCount;

        private:
            //prevent backend copy
            Histogram(const Histogram&);
            Histogram& operator=(const Histogram&);
    };

}

#endif
