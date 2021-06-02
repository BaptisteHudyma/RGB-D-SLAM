#ifndef MOTION_MODEL_HPP
#define MOTION_MODEL_HPP

#include <Eigen/Dense>
#include "Pose.hpp"

namespace poseEstimation {

    /**
     * \brief Dead reckoning class: guess next pose using a motion model
     */
    class Motion_Model {

        public:
            Motion_Model();
            void reset();

            /**
             * \brief Predicts next pose with motion model (dead reckoning)
             *
             * \param[in] currentPose Last frame pose
             */
            Pose predict_next_pose(const Pose& currentPose);

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        protected:

        private:
                quaternion lastQ;
                quaternion angularVelocity;
                vector3 lastPosition;
                vector3 linearVelocity;
    };

}

#endif
