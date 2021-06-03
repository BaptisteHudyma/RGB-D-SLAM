#ifndef PNP_SOLVER_HPP
#define PNP_SOLVER_HPP

#include "Pose.hpp"
#include "Image_Features_Struct.hpp"
#include <g2o/core/sparse_optimizer.h>


namespace poseEstimation {

    /**
     * \brief Compute a pose estimation
     */
    class PNP_Solver {
        public:
            PNP_Solver(double fx, double fy, double cx, double cy, double baseline);
            ~PNP_Solver();

            /**
             * \brief Refine a pose estimation using the matched points in this frame and local map
             *
             * \param[in] camPose Current camera pose
             * \param[in] features Features detected in frame
             * \param[in] matchedPoints Macthed features with previous frame
             * \param[in] matchLeft Container associating this frame feature ids to map ids
             *
             * \return The pose estimated from the previous pose using matched features
             */
            Pose compute_pose(const Pose& camPose, Image_Features_Struct& features, const vector3_array& matchedPoints, const std::vector<int>& matchLeft);

        private:
            double _fx;
            double _fy;
            double _cx;
            double _cy;
            double _baseline;

            g2o::SparseOptimizer _optimizer;
    };

}

#endif
