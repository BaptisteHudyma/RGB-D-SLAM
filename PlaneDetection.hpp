#ifndef PLANE_DETECTION_H
#define PLANE_DETECTION_H

#include <math.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "PlaneSegment.hpp"
#include "Histogram.hpp"

namespace planeDetection {

    class Plane_Detection {
        public:
            Plane_Detection(unsigned int width, unsigned int height, unsigned int blocSize);
            ~Plane_Detection();

            void grow_plane_regions();  //detect planes in depth image

        protected:
            void region_growing(std::vector<float>& cellDistTols, const unsigned short x, const unsigned short y, const Eigen::Vector3d seedPlaneNormal, const double seedPlaneD);


        private:
            std::vector<std::unique_ptr<Plane_Segment>> planeGrid;
            Histogram histogram;

            const int width;
            const int height;
            const int blocSize;

            int cellWidth;
            int cellHeight;

            int totalCellCount;
            int horizontalCellsCount;
            int verticalCellsCount;

            //arrays
            bool* activationMap;
            bool* unassignedMask;

            //mat
            cv::Mat<int> gridPlaneSegmentMap;
    };

}

#endif
