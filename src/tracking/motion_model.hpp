#ifndef RGBDSLAM_UTILS_MOTIONMODEL_HPP
#define RGBDSLAM_UTILS_MOTIONMODEL_HPP

#include "../types.hpp"
#include "../utils/pose.hpp"

namespace rgbd_slam::tracking {

/**
 * \brief Dead reckoning class: guess next pose using a motion model
 */
class Motion_Model
{
  public:
    Motion_Model();
    void reset() noexcept;
    void reset(const vector3& lastPosition, const quaternion& lastRotation) noexcept;

    /**
     * \brief Predicts next pose with motion model (dead reckoning)
     * \param[in] currentPose Last frame pose
     * \param[in] shouldIncreaseVariance If true, an uncertainty will be added to variance of the predicted pose
     */
    [[nodiscard]] utils::Pose predict_next_pose(const utils::Pose& currentPose,
                                                const bool shouldIncreaseVariance = true) const noexcept;

    /**
     * \brief Update the motion model using the refined pose
     *
     * \param[in] pose Refined pose estimation
     */
    void update_model(const utils::Pose& pose) noexcept;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    vector3 get_position_velocity() const noexcept { return _linearVelocity; };
    quaternion get_angular_velocity() const noexcept { return _angularVelocity; };

  protected:
    [[nodiscard]] quaternion get_rotational_velocity(const quaternion& lastRotation,
                                                     const quaternion& lastVelocity,
                                                     const quaternion& currentRotation) const noexcept;
    [[nodiscard]] vector3 get_position_velocity(const vector3& lastPosition,
                                                const vector3& lastVelocity,
                                                const vector3& currentPosition) const noexcept;

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

} // namespace rgbd_slam::tracking

#endif
