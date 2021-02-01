#ifndef PNP_SOLVER_HPP
#define PNP_SOLVER_HPP

#include "Pose.hpp"
#include "Image_Features_Struct.hpp"


namespace g2o
{
    class SparseOptimizer;
}

namespace poseEstimation {

    class PNP_Solver {
        public:
            PNP_Solver(double fx, double fy, double cx, double cy, double baseline);
            ~PNP_Solver();

            Pose compute_pose(const Pose& camPose, Image_Features_Struct& features, const vector3_array& matchedPoints, const std::vector<int>& matchOutliers);

        private:
            double fx;
            double fy;
            double cx;
            double cy;
            double baseline;

            g2o::SparseOptimizer *optimizer;
    };

}

#endif
