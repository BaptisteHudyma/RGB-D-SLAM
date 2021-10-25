#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <Eigen/Dense>
#include <vector>

namespace rgbd_slam {
namespace features {
namespace primitives {

    /**
      * \brief Basic 2D Histogram class, handling an histogram of N x N
      */
    class Histogram {
        public:
            /**
              * \param[in] binPerCoordCount Size of a bin, in pixels
              */
            Histogram(const unsigned int binPerCoordCount);

            /**
              * \brief Initialise the histogram 
              *
              * \param[in] points Points to put directly in the histogram bins
              * \param[in] flags Array of size points.rows(), indicating which point is in a planar segment
              */
            void init_histogram(const Eigen::MatrixXd& points, const bool* flags);

            /**
              * \brief Return the points in the bin containing the most points
              *
              * \param[out] pointsIDs Container storing the points in the biggest bin
              */
            void get_points_from_most_frequent_bin(std::vector<unsigned int>& pointsIDs);

            /**
              * \brief Remove all points from a bin
              */
	        void remove_point(const unsigned int pointId);

            /**
              * \brief Empty bins and clear content
              */
            void reset();

            ~Histogram();
        
        protected:

        private:
            unsigned int* _H;
            std::vector<int> _B;

            const unsigned int _binPerCoordCount;
            const unsigned int _binCount;
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
}
}

#endif
