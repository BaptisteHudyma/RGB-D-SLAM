#ifndef RGBDSLAM_UTILS_MOTIONMODEL_HPP
#define RGBDSLAM_UTILS_MOTIONMODEL_HPP

#include "types.hpp"
#include "pose.hpp"

namespace rgbd_slam {
namespace utils {

    /**
     * \brief Dead reckoning class: guess next pose using a motion model
     */
    class Motion_Model {

        public:
            Motion_Model();
            void reset();
            void reset(const vector3& lastPosition, const quaternion& lastRotation);

            /**
             * \brief Predicts next pose with motion model (dead reckoning)
             *
             * \param[in] currentPose Last frame pose
             */
            const Pose predict_next_pose(const Pose& currentPose) const;

            /**
             * \brief Update the motion model using the refined pose
             *
             * \param[in] pose Refined pose estimation
             */
            void update_model(const Pose& pose);

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        protected:

        private:
                // Last known rotation quaternion estimated by the motion model (set by update_model)
                quaternion _lastQ;
                // Last computed angular velocity
                quaternion _angularVelocity;
                // Last known translation estimated by the motion model (set by update_model)
                vector3 _lastPosition;
                // Last computed linear velocity
                vector3 _linearVelocity;
    };

}
}

#endif
