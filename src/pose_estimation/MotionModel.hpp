#ifndef MOTION_MODEL_HPP
#define MOTION_MODEL_HPP

#include "types.hpp"
#include "Pose.hpp"

#include <Eigen/Dense>

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

            /**
             * \brief Update the motion model using the refined pose
             *
             * \param[in] pose Refined pose estimation
             */
            void update_model(const Pose& pose);

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        protected:

        private:
                quaternion _lastQ;
                quaternion _angularVelocity;
                vector3 _lastPosition;
                vector3 _linearVelocity;
    };

}

#endif
